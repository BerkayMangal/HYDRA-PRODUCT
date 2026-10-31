"""Bounded native TWAP forward collection and append-only paper signal ledger."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time

import requests

from collectors.hyperliquid_research import PublicHyperliquid
from services.wallet_monitor import frozen_cohorts
from research.forward_ledger import SignalLedger
from .paths import ROOT, write_json
from .runtime import next_poll_deadline, validate_budget


def read_account(wallet, stop):
    started = int(time.time() * 1000)
    client = PublicHyperliquid()
    try:
        if stop.is_set():
            raise RuntimeError('Collector stopping')
        body = client.query('userTwapSliceFills', user=wallet)
        return {'wallet': wallet, 'started_ms': started,
                'observed_ms': int(time.time() * 1000), 'body': body}
    except Exception as exc:
        return {'wallet': wallet, 'started_ms': started,
                'observed_ms': int(time.time() * 1000), 'body': None,
                'error': type(exc).__name__}
    finally:
        client.close()


def audit(directory):
    """Read-only verification; this does not create or settle a missing trial."""
    import sqlite3, hashlib, gzip, math
    path = Path(directory) / 'ledger.sqlite3'
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        integrity = db.execute('PRAGMA quick_check').fetchall()
        checked = bad = 0
        for table in ['requests', 'quotes']:
            for packed, expected in db.execute(f'SELECT raw_gzip,sha256 FROM {table}'):
                checked += 1
                bad += hashlib.sha256(gzip.decompress(packed)).hexdigest() != expected
        protocol = json.loads((Path(directory) / 'protocol.json').read_text())
        from research.forward_ledger import digest
        stored = db.execute('SELECT protocol_hash FROM metadata WHERE id=1').fetchone()[0]
        spec = protocol['specification']
        entry_errors = exit_errors = model_errors = reference_errors = 0
        def matches_quote(quote):
            row = db.execute('SELECT payload FROM quotes WHERE id=? AND valid=1', (quote['quote_id'],)).fetchone()
            return row is not None and json.loads(row[0]) == {k: v for k, v in quote.items() if k != 'quote_id'}
        entries = {}
        for decision_id, payload, decision in db.execute('SELECT e.decision_id,e.payload,d.payload FROM entries e JOIN decisions d ON d.id=e.decision_id'):
            entry, decision = json.loads(payload), json.loads(decision)
            q, due = entry['quote'], decision['entry_due_ms']
            entries[decision_id] = entry
            entry_errors += not (due <= q['event_ms'] <= q['observed_ms'] <= due + spec['max_reference_delay_seconds'] * 1000)
            reference_errors += not matches_quote(q)
        for decision_id, payload in db.execute("SELECT decision_id,payload FROM outcomes WHERE status='completed'"):
            body = json.loads(payload)
            entry, q = entries[decision_id], body['exit']
            due = entry['exit_due_ms']
            exit_errors += not (due <= q['event_ms'] <= q['observed_ms'] <= due + spec['max_reference_delay_seconds'] * 1000)
            reference_errors += not matches_quote(q)
            expected = entry['quantity'] * q['bid'] * (1-spec['slippage_per_side_bps']/10000) * (1-spec['fee_per_side_bps']/10000) / spec['notional_usdt'] - 1
            for variant in ('strategy', 'control'):
                if body[variant] is not None:
                    model_errors += not math.isclose(body[variant]['modeled_net_return'], expected, abs_tol=1e-12)
        return {'sqlite_quick_check': integrity, 'raw_records_checked': checked,
                'raw_digest_mismatches': bad, 'protocol_hash_matches': stored == digest(protocol),
                'future_fills': db.execute('SELECT COUNT(*) FROM fills WHERE event_ms>first_observed_ms').fetchone()[0],
                'retroactive_entries': db.execute("SELECT COUNT(*) FROM entries e JOIN decisions d ON d.id=e.decision_id WHERE e.observed_ms<d.observed_ms").fetchone()[0],
                'entry_reference_violations': entry_errors, 'exit_reference_violations': exit_errors,
                'modeled_return_mismatches': model_errors, 'quote_reference_mismatches': reference_errors}
    finally:
        db.close()


def run(directory, until, once=False, report_mirror=None):
    import fcntl
    directory = Path(directory)
    # Keep the lock outside the trial so the protocol is its first file.
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(directory) + '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = SignalLedger(directory, frozen_cohorts())
    if report_mirror and Path(report_mirror).exists():
        try:
            prior = json.loads(Path(report_mirror).read_text())
            if prior.get('status') != 'prospective_research_only':
                raise ValueError('Mirror is not a forward report')
        except (OSError, ValueError, AttributeError):
            ledger.close()
            lock.close()
            raise ValueError('Refusing to overwrite an existing non-report mirror') from None
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    ledger.run_event({'state': 'started', 'pid': os.getpid(), 'until': until.isoformat() if until else None,
                      'orders_enabled': False})
    session = requests.Session()
    pool = ThreadPoolExecutor(max_workers=2)
    batch_futures = None
    next_batch = next_quote = next_report = time.monotonic()
    batch_started = None
    state = 'collecting'
    error = None

    def publish():
        report = ledger.report(runtime_state=state)
        report['worker'] = {'pid': os.getpid(), 'until': until.isoformat() if until else None, 'error': error}
        write_json(directory / 'summary.json', report)
        if report_mirror:
            write_json(report_mirror, report)
        return report

    try:
        while not stop.is_set() and (until is None or datetime.now(timezone.utc) < until):
            if shutil.disk_usage(directory).free < 1024 ** 3:
                raise RuntimeError('Less than 1 GiB free; forward collection stopped')
            tick = time.monotonic()
            if batch_futures is None and tick >= next_batch:
                batch_started = tick
                batch_futures = [pool.submit(read_account, wallet, stop) for wallet in ledger.wallets]
            if tick >= next_quote:
                began = time.monotonic()
                try:
                    response = session.get('https://www.okx.com/api/v5/market/ticker',
                                           params={'instId': 'BTC-USDT'}, timeout=10)
                    response.raise_for_status()
                    body = response.json()
                except Exception as exc:
                    body = {'error': type(exc).__name__}
                now = int(time.time() * 1000)
                ledger.quote(body, now)
                ledger.advance(now)
                next_quote = next_poll_deadline(began, time.monotonic(), ledger.spec['quote_poll_seconds'])
            if batch_futures is not None and all(future.done() for future in batch_futures):
                samples = [future.result() for future in batch_futures]
                batch_id = ledger.batch(samples, int(time.time() * 1000))
                now = int(time.time() * 1000)
                ledger.advance(now)
                ledger.decide(batch_id, int(time.time() * 1000))
                next_batch = next_poll_deadline(batch_started, time.monotonic(), ledger.spec['twap_poll_seconds'])
                batch_futures = None
                summary = publish()
                print(json.dumps({'observed_at': datetime.now(timezone.utc).isoformat(),
                                  'batch_id': batch_id, 'accounts': len(samples),
                                  'successful_accounts': summary['last_batch']['successful_accounts'],
                                  'source_errors': summary['last_batch']['errors'],
                                  'orders_enabled': False}), flush=True)
                if once:
                    break
            if time.monotonic() >= next_report:
                publish()
                next_report = time.monotonic() + 30
            stop.wait(.25)
    except Exception as exc:
        ledger.db.rollback()
        state, error = 'failed', type(exc).__name__ + ': ' + str(exc)[:200]
        ledger.run_event({'state': state, 'error': error, 'orders_enabled': False})
        raise
    finally:
        stop.set()
        for future in batch_futures or []:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        session.close()
        if state != 'failed':
            state = 'stopped'
        ledger.run_event({'state': state, 'orders_enabled': False})
        publish()
        ledger.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--until')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--report-mirror', type=Path)
    args = parser.parse_args()
    if args.audit:
        if args.out is None:
            parser.error('--audit requires an existing --out')
        result = audit(args.out)
        print(json.dumps(result, indent=2))
        if (result['sqlite_quick_check'] != [('ok',)] or not result['protocol_hash_matches']
                or any(result[key] for key in ['raw_digest_mismatches', 'future_fills', 'retroactive_entries',
                                              'entry_reference_violations', 'exit_reference_violations', 'modeled_return_mismatches', 'quote_reference_mismatches'])):
            raise SystemExit(1)
        return
    until = validate_budget(args.once, args.until)
    directory = args.out or ROOT / 'twap-forward' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run(directory, until, args.once, args.report_mirror)


if __name__ == '__main__':
    main()
