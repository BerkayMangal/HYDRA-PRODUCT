"""Predeclared activity discovery; sampled public trades, no PnL selection."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from collectors.hyperliquid_research import PublicHyperliquid
from research.forward_ledger import digest, slices
from services.wallet_monitor import frozen_cohorts
from .paths import write_json

SPECIFICATION = {
    'version': 'active_btc_twap_discovery_v1',
    'design': 'Source discovery only, not a performance test or alpha claim. All first observations are historical bootstrap.',
    'candidate_source': 'BTC public recentTrades participants; existing 22 frozen accounts excluded',
    'trade_snapshots': 30, 'snapshot_interval_seconds': 2,
    'candidate_order': 'Lexicographic unique valid public address after sampling',
    'maximum_accounts_probed': 96, 'maximum_concurrent_reads': 2,
    'activity_window_seconds': 900, 'maximum_last_slice_age_seconds': 120,
    'minimum_absolute_signed_btc_notional_usd': 100000,
    'maximum_selected_accounts': 12,
    'selection_order': 'Lexicographic eligible address; not ranked by return or PnL',
    'source_gate': 'Valid native recent slice history; capped history must reach common activity-window start; no future or conflicting rows',
    'window_reference': 'Local timestamp immediately before any account probes; later events excluded',
    'maximum_runtime_seconds': 600,
    'orders_enabled': False, 'alpha_validated': False,
    'limits': 'Recent-trade participants are a bounded activity sample, not market-wide discovery. Size is not predictive performance. Parent remaining size and independent ownership are unknown. Negative discovery does not prove no large TWAP exists.',
}


def trade_participants(body, observed_ms):
    if not isinstance(body, list) or len(body) > 1000:
        raise ValueError('Bounded public trade list required')
    found = set()
    for row in body:
        event = row['time']
        if isinstance(event, bool) or not isinstance(event, int) or not 0 < event <= observed_ms or row['coin'] != 'BTC':
            raise ValueError('Invalid or future BTC trade')
        if not isinstance(row.get('users'), list):
            raise ValueError('Public trade participants unavailable')
        for wallet in row['users']:
            if isinstance(wallet, str) and re.fullmatch('0x[0-9a-fA-F]{40}', wallet) and int(wallet[2:], 16) != 0:
                found.add(wallet.lower())
    return found


def screen_account(sample, reference_ms):
    start = reference_ms - SPECIFICATION['activity_window_seconds'] * 1000
    base = {'wallet': sample['wallet'], 'selected_for_future_study': False,
            'observation_phase': 'history_at_first_observation'}
    try:
        if sample.get('error'):
            raise ValueError(sample['error'])
        rows = slices(sample['body'], sample['observed_ms'])
        if sample['started_ms'] < reference_ms or sample['observed_ms'] < sample['started_ms']:
            raise ValueError('Invalid probe receipt order')
        oldest = min((row['event_ms'] for row in rows), default=None)
        capped = len(sample['body']) >= 2000
        if capped and (oldest is None or oldest > start):
            return {**base, 'status': 'unknown', 'reason': 'capped_history_does_not_reach_window', 'capped': True}
        window = [row for row in rows if row['coin'] == 'BTC' and start < row['event_ms'] <= reference_ms]
        net = sum((1 if row['side'] == 'B' else -1) * row['price'] * row['size'] for row in window)
        gross = sum(row['price'] * row['size'] for row in window)
        if not all(math.isfinite(value) for value in (net, gross)):
            raise ValueError('Invalid activity arithmetic')
        latest = max((row['event_ms'] for row in window), default=None)
        fresh = latest is not None and reference_ms - latest <= SPECIFICATION['maximum_last_slice_age_seconds'] * 1000
        eligible = fresh and abs(net) >= SPECIFICATION['minimum_absolute_signed_btc_notional_usd']
        return {**base, 'status': 'observed', 'capped': capped, 'oldest_returned_event_ms': oldest,
                'window_btc_slices': len(window), 'gross_notional_usd': gross, 'signed_notional_usd': net,
                'last_window_slice_ms': latest, 'recent_slice': fresh,
                'economic_activity_eligible': eligible,
                'reason': 'eligible_activity_for_future_study' if eligible else 'below_activity_floor' if fresh else 'no_recent_btc_slice'}
    except (KeyError, TypeError, ValueError) as exc:
        return {**base, 'status': 'unknown', 'reason': type(exc).__name__ + ': ' + str(exc)[:150]}


def select_accounts(accounts):
    return sorted(row['wallet'] for row in accounts if row.get('economic_activity_eligible'))[:SPECIFICATION['maximum_selected_accounts']]


def run(out, *, client_factory=PublicHyperliquid, clock=time.time, wait=time.sleep):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    excluded = sorted({wallet.lower() for group in frozen_cohorts().values() for wallet in group['wallets']})
    started = clock()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5)
    declaration = {'declared_at_ms': int(started * 1000), 'specification': SPECIFICATION,
                   'excluded_accounts': excluded, 'implementation_sha256': source_sha,
                   'source_commit': revision.stdout.strip() if revision.returncode == 0 else None}
    with (out / 'discovery-spec.json').open('x') as stream:
        stream.write(json.dumps(declaration, sort_keys=True, indent=2) + '\n')
    candidates, archives, errors = set(), {}, []

    def archive(filename, body):
        raw = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=True).encode()
        packed = gzip.compress(raw, mtime=0)
        (out / filename).write_bytes(packed)
        archives[filename] = {'sha256': hashlib.sha256(packed).hexdigest(), 'decoded_sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(packed)}

    client = client_factory()
    try:
        for index in range(SPECIFICATION['trade_snapshots']):
            began = clock()
            sample = {'started_ms': int(began * 1000)}
            try:
                sample['body'] = client.query('recentTrades', coin='BTC')
                sample['observed_ms'] = int(clock() * 1000)
                candidates.update(trade_participants(sample['body'], sample['observed_ms']))
            except Exception as exc:
                sample.update(observed_ms=int(clock() * 1000), error=type(exc).__name__ + ': ' + str(exc)[:150])
                errors.append({'trade_snapshot': index, 'error': sample['error']})
            archive(f'trades-{index:02d}.json.gz', sample)
            if clock() - started >= SPECIFICATION['maximum_runtime_seconds']:
                break
            if index + 1 < SPECIFICATION['trade_snapshots']:
                wait(max(0, SPECIFICATION['snapshot_interval_seconds'] - (clock() - began)))
    finally:
        client.close()
    planned = sorted(candidates - set(excluded))[:SPECIFICATION['maximum_accounts_probed']]
    reference = int(clock() * 1000)

    def probe(wallet):
        began = int(clock() * 1000)
        sample = {'wallet': wallet, 'started_ms': began, 'body': None}
        worker = client_factory()
        try:
            if clock() - started >= SPECIFICATION['maximum_runtime_seconds']:
                raise RuntimeError('Discovery budget exhausted')
            sample['body'] = worker.query('userTwapSliceFills', user=wallet)
            sample['observed_ms'] = int(clock() * 1000)
        except Exception as exc:
            sample.update(observed_ms=int(clock() * 1000), error=type(exc).__name__)
        finally:
            worker.close()
        return sample

    accounts = []
    with ThreadPoolExecutor(max_workers=SPECIFICATION['maximum_concurrent_reads']) as pool:
        for sample in pool.map(probe, planned):
            archive(sample['wallet'] + '.json.gz', sample)
            accounts.append(screen_account(sample, reference))
    selected = select_accounts(accounts)
    for row in accounts:
        row['selected_for_future_study'] = row['wallet'] in selected
    result = {'schema_version': 1, 'status': 'source_discovery_only', 'orders_enabled': False, 'alpha_validated': False,
              'declared_at': datetime.fromtimestamp(started, timezone.utc).isoformat(),
              'completed_at': datetime.fromtimestamp(clock(), timezone.utc).isoformat(),
              'declaration_sha256': digest(declaration), 'window_end_ms': reference,
              'window_start_ms': reference - SPECIFICATION['activity_window_seconds'] * 1000,
              'distinct_sampled_participants': len(candidates), 'existing_accounts_excluded': len(set(excluded) & candidates),
              'accounts_probed': len(accounts), 'successful_accounts': sum(row['status'] == 'observed' for row in accounts),
              'economic_activity_eligible': sum(row.get('economic_activity_eligible', False) for row in accounts),
              'selected_accounts': selected, 'accounts': accounts, 'errors': errors, 'archives': archives,
              'limits': SPECIFICATION['limits']}
    write_json(out / 'summary.json', result)
    if selected:
        write_json(out / 'candidate-cohort.json', {'wallets': selected, 'frozen_at_ms': int(clock() * 1000),
                    'scope': 'Newly discovered current BTC TWAP activity; bootstrap only, no forward performance evidence',
                    'discovery_declaration_sha256': digest(declaration), 'orders_enabled': False})
    return result


def audit_discovery(directory):
    directory = Path(directory)
    declaration = json.loads((directory / 'discovery-spec.json').read_text())
    result = json.loads((directory / 'summary.json').read_text())
    if declaration['specification'] != SPECIFICATION or result['declaration_sha256'] != digest(declaration):
        raise ValueError('Discovery definition differs')
    candidates, accounts = set(), []
    for filename, metadata in result['archives'].items():
        if Path(filename).name != filename:
            raise ValueError('Archive must be inside the discovery directory')
        packed = (directory / filename).read_bytes()
        raw = gzip.decompress(packed)
        if (len(packed) != metadata['bytes'] or hashlib.sha256(packed).hexdigest() != metadata['sha256']
                or hashlib.sha256(raw).hexdigest() != metadata['decoded_sha256']):
            raise ValueError('Discovery archive digest differs')
        sample = json.loads(raw)
        if not declaration['declared_at_ms'] <= sample['started_ms'] <= sample['observed_ms']:
            raise ValueError('Invalid local observation order')
        if filename.startswith('trades-'):
            if not sample.get('error'):
                candidates.update(trade_participants(sample['body'], sample['observed_ms']))
        else:
            if filename != sample['wallet'] + '.json.gz':
                raise ValueError('Account archive identity differs')
            accounts.append(screen_account(sample, result['window_end_ms']))
    planned = sorted(candidates - set(declaration['excluded_accounts']))[:SPECIFICATION['maximum_accounts_probed']]
    if sorted(row['wallet'] for row in accounts) != planned:
        raise ValueError('Probed accounts differ from predeclared participant selection')
    selected = select_accounts(accounts)
    for row in accounts:
        row['selected_for_future_study'] = row['wallet'] in selected
    if sorted(accounts, key=lambda row: row['wallet']) != result['accounts'] or selected != result['selected_accounts']:
        raise ValueError('Recorded screening differs from archived observations')
    if result['orders_enabled'] is not False or result['alpha_validated'] is not False:
        raise ValueError('Source discovery cannot authorize orders or alpha')
    return {'archives_verified': len(result['archives']), 'definition_matches': True,
            'screening_reproduced': True, 'sampled_participants': len(candidates),
            'accounts_probed': len(accounts), 'selected_accounts': len(selected),
            'source_coverage_certified_complete': False, 'alpha_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    body = audit_discovery(args.out) if args.audit else run(args.out)
    print(json.dumps({key: value for key, value in body.items() if key not in ('archives', 'accounts')}, indent=2))


if __name__ == '__main__':
    main()
