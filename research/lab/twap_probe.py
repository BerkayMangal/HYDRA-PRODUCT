"""One-shot native public TWAP slice audit; no parent forecasts or orders."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from collectors.hyperliquid_research import PublicHyperliquid
from services.wallet_monitor import frozen_cohorts
from .paths import ROOT, write_json

LIMITS = (
    "At most 2000 recent slices per account; beginning of history and market-wide coverage unknown. "
    "Past fills returned at the first request are historical observations, not newly executed events. "
    "Parent schedule, remaining size and cancellation state are not available from these fills. "
    "Account counts are not independent owners; fills across accounts are not summed into market volume. "
    "This probe cannot establish alpha or enable orders."
)


def btc_slices(body, observed_ms):
    if not isinstance(body, list):
        raise ValueError("Expected a public TWAP slice list")
    found = {}
    for item in body:
        fill = item['fill']
        if fill['coin'] != 'BTC':
            continue
        event = int(fill['time'])
        px, size = float(fill['px']), float(fill['sz'])
        if (not all(math.isfinite(value) and value > 0 for value in (px, size))
                or not 0 < event <= observed_ms or fill['side'] not in ('A', 'B')):
            raise ValueError("Valid historical public slice required")
        key = str(fill['tid'])
        found.setdefault(key, {
            'trade_id': key, 'twap_id': item['twapId'],
            'event_ms': event, 'observed_ms': observed_ms,
            'order_side': 'buy' if fill['side'] == 'B' else 'sell',
            'price': px, 'base_size': size,
            'observation_phase': 'history_at_first_observation',
        })
    return list(found.values())


def run(out, client_factory=PublicHyperliquid, cohorts=None):
    out = Path(out)
    # Each trial has a new destination; never replace an earlier receipt record.
    out.mkdir(parents=True, exist_ok=False)
    cohorts = frozen_cohorts() if cohorts is None else cohorts
    from .hyperliquid_shadow import validate_cohort
    for cohort in cohorts.values():
        validate_cohort(cohort, 240)
    wallets = sorted({wallet for cohort in cohorts.values() for wallet in cohort['wallets']})
    if not 1 <= len(wallets) <= 24:
        raise ValueError("Bounded frozen public cohort required")
    started = datetime.now(timezone.utc).isoformat()

    def read(wallet):
        client = client_factory()
        try:
            body = client.query('userTwapSliceFills', user=wallet)
            observed = int(time.time() * 1000)
            raw = json.dumps({'wallet': wallet, 'observed_ms': observed, 'response': body},
                             sort_keys=True, allow_nan=False).encode()
            archive = gzip.compress(raw, mtime=0)
            filename = wallet + '.json.gz'
            (out / filename).write_bytes(archive)
            rows = btc_slices(body, observed)
            write_json(out / (wallet + '-btc.json'), rows)
            return {'wallet': wallet, 'status': 'observed', 'observed_ms': observed,
                    'response_rows': len(body), 'possibly_truncated': len(body) >= 2000,
                    'unique_btc_slices': len(rows), 'archive': filename,
                    'archive_sha256': hashlib.sha256(archive).hexdigest(),
                    'earliest_btc_event_ms': min((r['event_ms'] for r in rows), default=None),
                    'latest_btc_event_ms': max((r['event_ms'] for r in rows), default=None)}
        except Exception as exc:
            return {'wallet': wallet, 'status': 'unavailable', 'error_type': type(exc).__name__}
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        accounts = list(pool.map(read, wallets))
    result = {'started_at': started, 'completed_at': datetime.now(timezone.utc).isoformat(),
              'query': 'userTwapSliceFills', 'requested_accounts': len(wallets),
              'successful_accounts': sum(row['status'] == 'observed' for row in accounts),
              'accounts_with_btc_slices': sum(row.get('unique_btc_slices', 0) > 0 for row in accounts),
              'possibly_truncated_accounts': sum(row.get('possibly_truncated', False) for row in accounts),
              'orders_enabled': False, 'alpha_validated': False, 'limits': LIMITS,
              'cohorts': {name: {'sha256': hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest(),
                                'accounts': len(cohort['wallets'])} for name, cohort in cohorts.items()},
              'accounts': accounts}
    write_json(out / 'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / 'twap-probes' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result = run(out)
    print(json.dumps({key: value for key, value in result.items() if key != 'accounts'}, indent=2))


if __name__ == '__main__':
    main()
