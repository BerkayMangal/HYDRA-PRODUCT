"""Bounded async public data pilot with REST metadata sentinel and spot flow."""

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import ssl
import time

import requests
from websockets.asyncio.client import connect

from collectors.hyperliquid_research import PublicHyperliquid
from research.liquidation_pilot import PilotJournal, SPEC
from .paths import ROOT, write_json
from .runtime import next_poll_deadline, validate_budget


def read_wallet(wallet):
    client = PublicHyperliquid(); start = int(time.time() * 1000)
    try:
        body = client.query('userFillsByTime', user=wallet,
                            startTime=start - SPEC['rest_lookback_seconds'] * 1000,
                            endTime=start, aggregateByTime=False)
        return {'body': body, 'transport_ms': int(time.time() * 1000), 'request_started_ms': start}
    except Exception as exc:
        return {'body': {'error': type(exc).__name__, 'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}, 'transport_ms': int(time.time() * 1000)}
    finally:
        client.close()


def read_okx(kind):
    path = 'trades' if kind == 'spot' else 'ticker'
    params = {'instId': 'BTC-USDT'}
    if kind == 'spot': params['limit'] = '500'
    try:
        response = requests.get('https://www.okx.com/api/v5/market/' + path, params=params, timeout=8)
        response.raise_for_status()
        return {'body': response.json(), 'transport_ms': int(time.time() * 1000)}
    except Exception as exc:
        return {'body': {'error': type(exc).__name__, 'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}, 'transport_ms': int(time.time() * 1000)}


async def collect_hyperliquid(ledger, stop, until):
    attempt = 0
    while not stop.is_set() and datetime.now(timezone.utc) < until:
        connection = 'hl-' + str(time.time_ns()); first = set(); admitted = set()
        ledger.feed_event('hyperliquid', connection, 'connecting')
        try:
            async with connect('wss://api.hyperliquid.xyz/ws', open_timeout=10, close_timeout=3,
                               ssl=ssl.create_default_context(), max_size=2*1024*1024) as ws:
                ledger.feed_event('hyperliquid', connection, 'connected')
                for wallet in ledger.wallets:
                    await ws.send(json.dumps({'method': 'subscribe', 'subscription': {'type': 'userFills', 'user': wallet, 'aggregateByTime': False}}))
                await ws.send(json.dumps({'method': 'subscribe', 'subscription': {'type': 'trades', 'coin': 'BTC'}}))
                last_ping = time.monotonic(); attempt = 0
                while not stop.is_set() and datetime.now(timezone.utc) < until:
                    if time.monotonic() - last_ping > 40:
                        await ws.send(json.dumps({'method': 'ping'})); last_ping = time.monotonic()
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 2)
                    except asyncio.TimeoutError:
                        continue
                    observed = int(time.time() * 1000)
                    try:
                        body = json.loads(raw)
                    except (ValueError, TypeError):
                        ledger.ingest('transport_error', {'invalid_ws_text': str(raw)}, observed, connection=connection)
                        continue
                    if not isinstance(body, dict):
                        ledger.ingest('transport_error', {'invalid_ws_envelope': body}, observed, connection=connection)
                        continue
                    channel = body.get('channel')
                    if channel == 'userFills':
                        data = body.get('data', {})
                        if not isinstance(data, dict):
                            ledger.ingest('transport_error', {'invalid_user_envelope': body}, observed, connection=connection)
                            continue
                        wallet = str(data.get('user', '')).lower()
                        bootstrap = wallet not in first or data.get('isSnapshot') is True
                        result = ledger.ingest('hl_ws', data.get('fills'), observed, connection=connection,
                                               wallet=wallet, bootstrap=bootstrap, envelope=body)
                        if result['valid']: first.add(wallet)
                    elif channel == 'trades':
                        result = ledger.ingest('hl_market', body.get('data'), observed, connection=connection, bootstrap='market' not in first, envelope=body)
                        if result['valid']: first.add('market')
                    else:
                        ledger.ingest('control', body, observed, connection=connection)
                        if channel == 'subscriptionResponse':
                            subscription = body.get('data', {}).get('subscription', {})
                            if subscription.get('type') == 'userFills': admitted.add(subscription.get('user', '').lower())
                            ledger.feed_event('hyperliquid', connection, 'subscribed', {'admitted_wallets': sorted(admitted), 'expected_wallets': len(ledger.wallets)})
                        elif channel == 'error':
                            raise RuntimeError('Hyperliquid subscription rejected: ' + str(body.get('data'))[:100])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ledger.feed_event('hyperliquid', connection, 'disconnected', {'error': type(exc).__name__})
            attempt += 1
            try: await asyncio.wait_for(stop.wait(), min(60, 2 ** min(attempt, 5)))
            except asyncio.TimeoutError: pass
        finally:
            ledger.feed_event('hyperliquid', connection, 'closed')


async def collect_sentinel(ledger, stop, until):
    first = set(); semaphore = asyncio.Semaphore(SPEC['rest_concurrency'])
    async def one(wallet):
        async with semaphore:
            if stop.is_set() or datetime.now(timezone.utc) >= until: return
            sample = await asyncio.to_thread(read_wallet, wallet)
            observed = int(time.time() * 1000)
            result = ledger.ingest('hl_rest', sample['body'], observed, wallet=wallet,
                                   bootstrap=wallet not in first, transport_ms=sample['transport_ms'], envelope=sample)
            if result['valid']: first.add(wallet)
    while not stop.is_set() and datetime.now(timezone.utc) < until:
        began = time.monotonic()
        await asyncio.gather(*(one(wallet) for wallet in ledger.wallets))
        delay = max(0, next_poll_deadline(began, time.monotonic(), SPEC['rest_sentinel_seconds']) - time.monotonic())
        try: await asyncio.wait_for(stop.wait(), min(delay, max(0, until.timestamp()-time.time())))
        except asyncio.TimeoutError: pass


async def collect_okx(ledger, stop, until, kind):
    first = True
    while not stop.is_set() and datetime.now(timezone.utc) < until:
        began = time.monotonic()
        sample = await asyncio.to_thread(read_okx, kind)
        result = ledger.ingest(kind, sample['body'], int(time.time() * 1000), bootstrap=first, transport_ms=sample['transport_ms'])
        if result['valid']: first = False
        delay = max(0, next_poll_deadline(began, time.monotonic(), SPEC[kind + '_poll_seconds']) - time.monotonic())
        try: await asyncio.wait_for(stop.wait(), min(delay, max(0, until.timestamp()-time.time())))
        except asyncio.TimeoutError: pass


async def run_async(directory, cohort, until, mirror):
    ledger = PilotJournal(directory, cohort); stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM): loop.add_signal_handler(sig, stop.set)
    ledger.run_event({'state': 'started', 'pid': os.getpid(), 'until': until.isoformat(), 'orders_enabled': False})
    tasks = [asyncio.create_task(collect_hyperliquid(ledger, stop, until)),
             asyncio.create_task(collect_sentinel(ledger, stop, until)),
             asyncio.create_task(collect_okx(ledger, stop, until, 'spot')),
             asyncio.create_task(collect_okx(ledger, stop, until, 'quote'))]
    state, error = 'collecting', None
    def publish():
        body = ledger.report(state, until.isoformat(), error)
        write_json(Path(directory) / 'summary.json', body)
        if mirror: write_json(mirror, body)
        return body
    try:
        next_report = 0
        while not stop.is_set() and datetime.now(timezone.utc) < until:
            free = shutil.disk_usage(directory).free
            size = sum(path.stat().st_size for path in Path(directory).glob('pilot.sqlite3*'))
            if free < SPEC['minimum_free_bytes'] or size > SPEC['maximum_journal_bytes']:
                raise RuntimeError('Pilot disk budget reached')
            for task in tasks:
                if task.done():
                    task.result()
                    if datetime.now(timezone.utc) < until: raise RuntimeError('A collector stopped before its budget')
            if time.monotonic() >= next_report:
                body = publish(); next_report = time.monotonic() + 30
                print(json.dumps({'generated_at': body['generated_at'], 'live_btc_ws_fills': body['live_ws_unique_btc_fills'],
                                  'rejected_frames': body['rejected_frames'], 'orders_enabled': False}), flush=True)
            try: await asyncio.wait_for(stop.wait(), 1)
            except asyncio.TimeoutError: pass
    except Exception as exc:
        ledger.db.rollback(); state, error = 'failed', type(exc).__name__ + ': ' + str(exc)[:180]
        raise
    finally:
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        if state != 'failed': state = 'stopped'
        ledger.run_event({'state': state, 'error': error, 'orders_enabled': False})
        publish(); ledger.close()


def audit(directory):
    path = Path(directory); db = sqlite3.connect((path / 'pilot.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        checks = [row[0] for row in db.execute('PRAGMA quick_check')]
        count = bad = 0
        for packed, sha in db.execute('SELECT raw_gzip,sha256 FROM frames'):
            count += 1; bad += hashlib.sha256(gzip.decompress(packed)).hexdigest() != sha
        protocol = json.loads((path / 'protocol.json').read_text())
        from research.liquidation_pilot import digest
        return {'quick_check': checks, 'raw_frames_checked': count, 'raw_digest_mismatches': bad,
                'protocol_hash_matches': db.execute('SELECT hash FROM metadata').fetchone()[0] == digest(protocol),
                'future_fills': db.execute('SELECT COUNT(*) FROM fills WHERE event_ms>observed_ms').fetchone()[0],
                'metadata_before_fill_receipt': db.execute('''SELECT COUNT(*) FROM annotations a JOIN seen s ON s.wallet=a.wallet AND s.coin=a.coin AND s.tid=a.tid WHERE a.observed_ms<s.first_observed_ms''').fetchone()[0],
                'orders_enabled': False}
    finally: db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--cohort', type=Path)
    parser.add_argument('--until'); parser.add_argument('--report-mirror', type=Path); parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        result = audit(args.out); print(json.dumps(result, indent=2))
        if (result['quick_check'] != ['ok'] or not result['protocol_hash_matches'] or
                any(result[k] for k in ['raw_digest_mismatches', 'future_fills', 'metadata_before_fill_receipt'])): raise SystemExit(1)
        return
    until = validate_budget(False, args.until)
    if args.cohort is None: parser.error('--cohort is required for a new/continued pilot')
    if args.report_mirror:
        protected = [args.out / 'protocol.json', args.out / 'pilot.sqlite3', args.cohort]
        if args.report_mirror.resolve() in {path.resolve() for path in protected}:
            parser.error('--report-mirror must not replace a protocol, journal or cohort')
        if args.report_mirror.exists():
            prior = json.loads(args.report_mirror.read_text())
            if prior.get('status') != 'liquidation_data_pilot':
                parser.error('--report-mirror refuses to overwrite an unrelated file')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(args.out)+'.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    try: asyncio.run(run_async(args.out, json.loads(args.cohort.read_text()), until, args.report_mirror))
    finally: lock.close()


if __name__ == '__main__': main()
