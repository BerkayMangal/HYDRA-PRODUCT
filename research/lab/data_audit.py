"""Read-only integrity diagnostics for the existing public-data journals."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .paths import ROOT, write_json


def audit_journal(path, kind):
    path = Path(path).resolve()
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        check = [row[0] for row in db.execute('PRAGMA quick_check')]
        raw_table = 'snapshots' if kind == 'okx' else 'raw_snapshots'
        checked = mismatches = 0
        for payload, expected in db.execute(f'SELECT payload,sha256 FROM {raw_table}'):
            checked += 1
            mismatches += hashlib.sha256(payload.encode()).hexdigest() != expected
        result = {'sqlite_quick_check': check, 'raw_payloads_checked': checked,
                  'raw_digest_mismatches': mismatches}
        if kind == 'okx':
            result.update(
                trade_rows=dict(db.execute('SELECT instrument,COUNT(*) FROM trades GROUP BY instrument')),
                invalid_trade_times=db.execute('SELECT COUNT(*) FROM trades WHERE event_ms IS NULL OR observed_ms IS NULL OR event_ms<=0 OR observed_ms<event_ms').fetchone()[0],
                invalid_trade_arithmetic=db.execute('''SELECT COUNT(*) FROM trades WHERE
                  price IS NULL OR btc_size IS NULL OR quote_size IS NULL OR raw_size IS NULL
                  OR price<=0 OR btc_size<=0 OR quote_size<=0 OR raw_size<=0
                  OR ABS(price)>1.7976931348623157e308 OR ABS(btc_size)>1.7976931348623157e308
                  OR ABS(quote_size)>1.7976931348623157e308 OR ABS(raw_size)>1.7976931348623157e308
                  OR ABS(quote_size-price*btc_size)>MAX(0.00000001,ABS(quote_size)*0.0000000001)''').fetchone()[0],
                base_conversion_ratios=list(db.execute('SELECT instrument,ROUND(btc_size/raw_size,10),COUNT(*) FROM trades GROUP BY instrument,ROUND(btc_size/raw_size,10)')),
                collecting_cycle_counts=dict(db.execute('SELECT status,COUNT(*) FROM health GROUP BY status')),
            )
            invalid = result['invalid_trade_times'] + result['invalid_trade_arithmetic']
        else:
            windows = {}
            for payload, in db.execute('SELECT payload FROM features'):
                body = json.loads(payload)
                scheduler = body.get('collection_protocol', {}).get('scheduler', 'legacy')
                counts = windows.setdefault(scheduler, {'snapshots': 0, 'valid_inventory_windows': 0})
                counts['snapshots'] += 1
                counts['valid_inventory_windows'] += body.get('sampled_inventory_change') is not None
            result.update(
                fills=db.execute('SELECT COUNT(*) FROM fills').fetchone()[0],
                invalid_fill_times=db.execute('SELECT COUNT(*) FROM fills WHERE event_ms IS NULL OR first_observed_ms IS NULL OR event_ms<=0 OR first_observed_ms<event_ms').fetchone()[0],
                inventory_windows=windows,
                capped_fill_cycles=sum(bool(json.loads(payload).get('fills_may_be_truncated')) for payload, in db.execute('SELECT payload FROM health')),
            )
            invalid = result['invalid_fill_times']
        result['integrity_checks_passed'] = check == ['ok'] and mismatches == 0 and invalid == 0 and checked > 0
        result['coverage_certified_complete'] = False
        return result
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--okx', type=Path, default=ROOT / 'data/shadow.sqlite3')
    parser.add_argument('--primary', type=Path, default=ROOT / 'hyperliquid/observations.sqlite3')
    parser.add_argument('--additional', type=Path, default=Path(__file__).resolve().parents[2] / 'data/research-wide/hyperliquid/observations.sqlite3')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out and args.out.resolve() in {getattr(args, name).resolve() for name in ('okx', 'primary', 'additional')}:
        parser.error('--out must not replace a source journal')
    report = {'audited_at': datetime.now(timezone.utc).isoformat(),
              'method': 'Read-only transaction: quick_check, every decoded payload SHA256, temporal and OKX arithmetic diagnostics',
              'limits': 'Integrity diagnostics do not certify upstream truth, full source coverage, independent accounts or alpha.',
              'journals': {}}
    for name in ('okx', 'primary', 'additional'):
        try:
            report['journals'][name] = audit_journal(getattr(args, name), name)
        except (OSError, ValueError, sqlite3.Error) as exc:
            report['journals'][name] = {'integrity_checks_passed': False, 'error': type(exc).__name__ + ': ' + str(exc)[:180]}
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not all(body['integrity_checks_passed'] for body in report['journals'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
