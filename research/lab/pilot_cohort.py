"""Freeze at most eight current active BTC accounts before the data pilot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from collectors.hyperliquid_research import PublicHyperliquid
from services.wallet_monitor import frozen_cohorts
from research.forward_ledger import slices
from .paths import write_json

SPEC = {'version': 'pilot_activity_cohort_v1', 'maximum_candidates': 32,
        'maximum_selected': 8, 'lookback_seconds': 120, 'concurrency': 2,
        'candidates': 'Existing frozen public cohorts plus current BTC recentTrades participants; first 32 lexicographic addresses',
        'selection': 'Observed BTC fill count in fixed source window descending, then address; no PnL fields used',
        'caps': 'Capped histories are activity lower bounds, not certified total counts', 'orders_enabled': False}


def select(observations):
    eligible = [row for row in observations if row.get('btc_fill_count', 0) > 0 and row.get('error') is None]
    return [row['wallet'] for row in sorted(eligible, key=lambda row: (-row['btc_fill_count'], row['wallet']))[:SPEC['maximum_selected']]]


def run(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    end = int(time.time() * 1000); start = end - SPEC['lookback_seconds'] * 1000
    declared = {'created_ms': end, 'specification': SPEC,
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    with (out / 'discovery-spec.json').open('x') as stream:
        json.dump(declared, stream, indent=2)
    client = PublicHyperliquid()
    try:
        trades = client.query('recentTrades', coin='BTC')
        observed = int(time.time() * 1000)
    finally:
        client.close()
    raw = json.dumps({'observed_ms': observed, 'body': trades}, sort_keys=True).encode()
    (out / 'recent-trades.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    if not isinstance(trades, list): raise ValueError('Expected public BTC trades')
    addresses = {wallet.lower() for row in trades for wallet in row.get('users', [])
                 if isinstance(wallet, str) and re.fullmatch('0x[0-9a-fA-F]{40}', wallet)}
    addresses |= {wallet.lower() for group in frozen_cohorts().values() for wallet in group['wallets']}
    candidates = sorted(addresses)[:SPEC['maximum_candidates']]

    def inspect(wallet):
        client = PublicHyperliquid(); sample = {'wallet': wallet}
        try:
            body = client.query('userFillsByTime', user=wallet, startTime=start, endTime=end, aggregateByTime=False)
            receipt = int(time.time() * 1000)
            normalized = slices([{'fill': row, 'twapId': row.get('twapId') or 'not-a-twap'} for row in body], receipt)
            count = sum(row['coin'] == 'BTC' and start < row['event_ms'] <= end for row in normalized)
            sample.update(observed_ms=receipt, body=body, btc_fill_count=count, capped=len(body) >= 2000)
        except Exception as exc:
            sample.update(observed_ms=int(time.time() * 1000), error=type(exc).__name__)
        finally:
            client.close()
        return sample

    observations, archives = [], {}
    with ThreadPoolExecutor(max_workers=SPEC['concurrency']) as pool:
        for sample in pool.map(inspect, candidates):
            filename = sample['wallet'] + '.json.gz'
            packed = gzip.compress(json.dumps(sample, sort_keys=True, allow_nan=True).encode(), mtime=0)
            (out / filename).write_bytes(packed)
            archives[filename] = hashlib.sha256(packed).hexdigest()
            observations.append({key: value for key, value in sample.items() if key != 'body'})
    wallets = select(observations)
    write_json(out / 'discovery-summary.json', {'specification': SPEC, 'observations': observations, 'archives': archives, 'selected': wallets})
    if len(wallets) < 2:
        raise ValueError('Fewer than two currently active BTC accounts; preserve discovery without launching pilot')
    cohort = {'wallets': sorted(wallets), 'frozen_at_ms': int(time.time() * 1000),
              'scope': 'Current active BTC public-account sample, not independent owners or market-wide coverage',
              'selection_specification': SPEC, 'discovery_sha256': hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest(),
              'orders_enabled': False}
    write_json(out / 'cohort.json', cohort)
    print(json.dumps({'selected_accounts': len(wallets), 'inspected': len(observations), 'orders_enabled': False}))
    return cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)


if __name__ == '__main__': main()
