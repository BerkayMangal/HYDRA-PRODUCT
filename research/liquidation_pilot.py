"""Append-only public data-capability pilot. Never creates trading decisions."""

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time


SPEC = {
    'version': 'liquidation_data_pilot_v1', 'phase': 'data_capability_only',
    'maximum_wallets': 8, 'maximum_hyperliquid_connections': 1,
    'wallet_selection': 'Observed recent BTC fill count, then address; no PnL ranking',
    'spot_instrument': 'OKX BTC-USDT spot', 'spot_poll_seconds': 5,
    'quote_poll_seconds': 5, 'rest_sentinel_seconds': 60,
    'rest_concurrency': 2, 'rest_lookback_seconds': 120,
    'max_live_latency_seconds_for_review': 10, 'max_quote_age_seconds': 5,
    'max_spot_poll_gap_seconds': 20, 'max_spot_trade_age_seconds': 15,
    'minimum_free_bytes': 1073741824, 'maximum_journal_bytes': 4294967296,
    'first_receipt': 'Immutable collector ingress on the main event loop; HTTP transport receipt stored separately.',
    'snapshot_policy': 'Every first user frame per connection and every isSnapshot frame is bootstrap. First REST sample per account is bootstrap. Bootstrap IDs never become new signals.',
    'metadata_policy': 'Explicit liquidation knowledge has its own first receipt. Later REST enrichment never borrows an earlier WS fill receipt.',
    'coverage': 'Sampled accounts and connection-observed public BTC trades. No venue-wide liquidation coverage or independent ownership claim.',
    'spot_coverage': 'Inferred only from actual previously recorded trade-ID overlap, uninterrupted bounded polling, and fresh valid trades; not venue-certified completeness.',
    'limit_policy': 'One connection, at most eight user subscriptions and one public BTC subscription. Admission/errors measured. Current published provider contract was unreachable; contract review remains a readiness blocker.',
    'orders_enabled': False, 'strategy_enabled': False, 'alpha_validated': False,
}


def encode(body):
    return json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(body):
    return hashlib.sha256(encode(body).encode()).hexdigest()


def finite(value, positive=False):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a market number')
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError('Invalid market number')
    return result


def timestamp(value, observed):
    result = finite(value, True)
    if result != int(result) or result > observed:
        raise ValueError('Invalid or future event time')
    return int(result)


def native_fill(body, observed):
    coin, tid = body['coin'], body['tid']
    if not isinstance(coin, str) or isinstance(tid, bool) or not isinstance(tid, (str, int)) or not str(tid):
        raise ValueError('Fill identity required')
    row = {'coin': coin, 'trade_id': str(tid), 'event_ms': timestamp(body['time'], observed),
           'price': finite(body['px'], True), 'size': finite(body['sz'], True), 'side': body['side']}
    if row['side'] not in ('A', 'B') or not math.isfinite(row['price'] * row['size']):
        raise ValueError('Invalid fill arithmetic or side')
    return row


def explicit_liquidation(body, wallet, row):
    info = body.get('liquidation')
    if info is None:
        return None
    if not isinstance(info, dict):
        raise ValueError('Invalid liquidation metadata')
    import re
    victim = info.get('liquidatedUser')
    if not isinstance(victim, str):
        raise ValueError('Explicit liquidation victim required')
    victim = victim.lower()
    if not re.fullmatch('0x[0-9a-f]{40}', victim):
        raise ValueError('Explicit liquidation victim required')
    victim_side = row['side'] if victim == wallet else ('A' if row['side'] == 'B' else 'B')
    return {'victim': victim, 'liquidated_position_side': 'long' if victim_side == 'A' else 'short',
            'method': info.get('method', 'unknown'), 'notional_usd': row['price'] * row['size']}


class PilotJournal:
    def __init__(self, directory, cohort, *, clock=time.time):
        import platform
        import re
        import subprocess
        self.directory, self.clock = Path(directory), clock
        self.directory.mkdir(parents=True, exist_ok=True)
        self.wallets = sorted(cohort['wallets'])
        if not 1 <= len(self.wallets) <= SPEC['maximum_wallets'] or len(set(self.wallets)) != len(self.wallets):
            raise ValueError('One to eight unique frozen wallets required')
        if any(not re.fullmatch('0x[0-9a-f]{40}', wallet) for wallet in self.wallets):
            raise ValueError('Valid public addresses required')
        repo = Path(__file__).resolve().parents[1]
        files = ['research/liquidation_pilot.py', 'research/lab/liquidation_pilot.py',
                 'collectors/hyperliquid_research.py']
        hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files if (repo / name).is_file()}
        expected = {'specification': SPEC, 'cohort': cohort, 'implementation_sha256': hashes}
        path = self.directory / 'protocol.json'
        if path.exists():
            self.protocol = json.loads(path.read_text())
            if any(self.protocol.get(key) != value for key, value in expected.items()):
                raise ValueError('Existing pilot definition differs; preserve it and start a new pilot')
        else:
            if any(self.directory.iterdir()):
                raise ValueError('Refusing to initialize over non-pilot data')
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True, capture_output=True, timeout=5)
            self.protocol = {**expected, 'created_ms': int(clock() * 1000),
                             'source_commit': revision.stdout.strip(), 'python': platform.python_version()}
            with path.open('x') as stream:
                stream.write(encode(self.protocol) + '\n')
        self.started_ms = self.protocol['created_ms']
        self.protocol_hash = digest(self.protocol)
        self.db = sqlite3.connect(self.directory / 'pilot.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA recursive_triggers=ON')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS metadata(id INTEGER PRIMARY KEY CHECK(id=1),hash TEXT,payload TEXT);
          CREATE TABLE IF NOT EXISTS frames(id INTEGER PRIMARY KEY,source TEXT,connection TEXT,wallet TEXT,observed_ms INTEGER,transport_ms INTEGER,bootstrap INTEGER,valid INTEGER,error TEXT,raw_gzip BLOB,sha256 TEXT,returned_rows INTEGER,capped INTEGER);
          CREATE INDEX IF NOT EXISTS frames_source ON frames(source,observed_ms);
          CREATE TABLE IF NOT EXISTS seen(wallet TEXT,coin TEXT,tid TEXT,first_observed_ms INTEGER,phase TEXT,PRIMARY KEY(wallet,coin,tid));
          CREATE TABLE IF NOT EXISTS fills(source TEXT,wallet TEXT,coin TEXT,tid TEXT,event_ms INTEGER,observed_ms INTEGER,phase TEXT,semantic_sha TEXT,payload TEXT,frame_id INTEGER,PRIMARY KEY(source,wallet,coin,tid));
          CREATE INDEX IF NOT EXISTS fills_time ON fills(source,event_ms,observed_ms);
          CREATE TABLE IF NOT EXISTS annotations(source TEXT,wallet TEXT,coin TEXT,tid TEXT,event_ms INTEGER,observed_ms INTEGER,phase TEXT,payload TEXT,frame_id INTEGER,PRIMARY KEY(source,wallet,coin,tid));
          CREATE INDEX IF NOT EXISTS annotations_identity ON annotations(coin,tid,observed_ms);
          CREATE TABLE IF NOT EXISTS feed_events(id INTEGER PRIMARY KEY,feed TEXT,connection TEXT,observed_ms INTEGER,state TEXT,payload TEXT);
          CREATE TABLE IF NOT EXISTS quotes(id INTEGER PRIMARY KEY,observed_ms INTEGER,event_ms INTEGER,bid REAL,ask REAL,bid_size REAL,ask_size REAL,frame_id INTEGER);
          CREATE TABLE IF NOT EXISTS spot_polls(id INTEGER PRIMARY KEY,observed_ms INTEGER,overlap INTEGER,frame_id INTEGER);
          CREATE TABLE IF NOT EXISTS run_events(id INTEGER PRIMARY KEY,observed_ms INTEGER,payload TEXT);
        ''')
        prior = self.db.execute('SELECT hash FROM metadata WHERE id=1').fetchone()
        if prior and prior[0] != self.protocol_hash:
            self.db.close()
            raise ValueError('Pilot journal definition mismatch')
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES(1,?,?)', (self.protocol_hash, encode(self.protocol)))
        for table in ['metadata', 'frames', 'seen', 'fills', 'annotations', 'feed_events', 'quotes', 'spot_polls', 'run_events']:
            for op in ('UPDATE', 'DELETE'):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{op.lower()} BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT,'append-only pilot'); END")
        self.db.commit()

    def close(self):
        self.db.close()

    def run_event(self, body):
        self.db.execute('INSERT INTO run_events(observed_ms,payload) VALUES(?,?)', (int(self.clock() * 1000), encode(body)))
        self.db.commit()

    def feed_event(self, feed, connection, state, body=None, observed_ms=None):
        self.db.execute('INSERT INTO feed_events(feed,connection,observed_ms,state,payload) VALUES(?,?,?,?,?)',
                        (feed, connection, observed_ms or int(self.clock() * 1000), state, encode(body or {})))
        self.db.commit()

    def ingest(self, source, body, observed_ms, *, connection='', wallet=None, bootstrap=False, transport_ms=None, envelope=None):
        """Validate the full frame before accepting any normalized rows; preserve raw failures."""
        transport_ms = observed_ms if transport_ms is None else transport_ms
        raw = json.dumps(body if envelope is None else envelope, sort_keys=True, separators=(',', ':'), allow_nan=True).encode()
        error, rows, annotations, quote = None, [], [], None
        overlap = False
        try:
            if not self.started_ms <= transport_ms <= observed_ms:
                raise ValueError('Invalid receipt chronology')
            if source in ('hl_ws', 'hl_rest'):
                if wallet not in self.wallets or not isinstance(body, list) or len(body) > 2000:
                    raise ValueError('Expected bounded public wallet fills')
                unique = {}
                for item in body:
                    row = native_fill(item, transport_ms)
                    if row['coin'] != 'BTC':
                        continue
                    key = row['trade_id']
                    if key in unique and unique[key] != row:
                        raise ValueError('Conflicting fill within source frame')
                    unique[key] = row
                    rows.append(row)
                    annotations.append(explicit_liquidation(item, wallet, row))
            elif source == 'hl_market':
                if not isinstance(body, list) or len(body) > 5000:
                    raise ValueError('Bounded public BTC trades required')
                rows = [native_fill(row, transport_ms) for row in body]
                if any(row['coin'] != 'BTC' for row in rows):
                    raise ValueError('Wrong public market')
                wallet = ''
            elif source == 'spot':
                if body.get('code') != '0' or not isinstance(body.get('data'), list) or not 1 <= len(body['data']) <= 500:
                    raise ValueError('Expected OKX BTC spot trade sample')
                wallet = ''
                for item in body['data']:
                    if item.get('instId', 'BTC-USDT') != 'BTC-USDT' or item['side'] not in ('buy', 'sell'):
                        raise ValueError('Wrong spot instrument/side')
                    rows.append(native_fill({'coin': 'BTC', 'tid': item['tradeId'], 'time': item['ts'],
                                'px': item['px'], 'sz': item['sz'], 'side': 'B' if item['side'] == 'buy' else 'A'}, transport_ms))
                overlap = any(self.db.execute("SELECT 1 FROM fills WHERE source='spot' AND wallet='' AND coin='BTC' AND tid=?", (row['trade_id'],)).fetchone() for row in rows)
            elif source == 'quote':
                item = body['data'][0]
                if body.get('code') != '0' or item['instId'] != 'BTC-USDT':
                    raise ValueError('Wrong quote instrument')
                event = timestamp(item['ts'], transport_ms)
                bid, ask, bs, az = [finite(item[k], True) for k in ('bidPx', 'askPx', 'bidSz', 'askSz')]
                if bid >= ask or observed_ms - event > SPEC['max_quote_age_seconds'] * 1000:
                    raise ValueError('Stale/crossed quote')
                quote = (observed_ms, event, bid, ask, bs, az)
            elif source not in ('control', 'transport_error'):
                raise ValueError('Unknown read-only source')
            if source == 'transport_error':
                raise ValueError('Source transport failed')
            for row in rows:
                old = self.db.execute('SELECT semantic_sha FROM fills WHERE source=? AND wallet=? AND coin=? AND tid=?',
                                      (source, wallet or '', row['coin'], row['trade_id'])).fetchone()
                if old and old[0] != digest(row):
                    raise ValueError('Previously observed fill changed')
            if source in ('hl_ws', 'hl_rest'):
                frame_metadata = {}
                for row, annotation in zip(rows, annotations):
                    if annotation is None:
                        continue
                    prior_frame = frame_metadata.get((row['coin'], row['trade_id']))
                    if prior_frame and any(prior_frame[key] != annotation[key] for key in ('victim', 'liquidated_position_side', 'notional_usd')):
                        raise ValueError('Conflicting liquidation metadata within frame')
                    frame_metadata[(row['coin'], row['trade_id'])] = annotation
                    old = self.db.execute('SELECT payload FROM annotations WHERE coin=? AND tid=? LIMIT 1', (row['coin'], row['trade_id'])).fetchone()
                    if old:
                        prior = json.loads(old[0])
                        if any(prior[key] != annotation[key] for key in ('victim', 'liquidated_position_side', 'notional_usd')):
                            raise ValueError('Conflicting explicit liquidation metadata')
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)[:160]
        cur = self.db.execute('INSERT INTO frames(source,connection,wallet,observed_ms,transport_ms,bootstrap,valid,error,raw_gzip,sha256,returned_rows,capped) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                              (source, connection, wallet, observed_ms, transport_ms, int(bootstrap), int(error is None), error,
                               gzip.compress(raw, mtime=0), hashlib.sha256(raw).hexdigest(), len(body) if isinstance(body, list) else None,
                               int(source == 'hl_rest' and isinstance(body, list) and len(body) >= 2000)))
        frame_id = cur.lastrowid
        if error is None:
            for index, row in enumerate(rows):
                phase = 'bootstrap' if bootstrap else 'pretrial' if row['event_ms'] < self.started_ms else 'live'
                seen_wallet = wallet or source
                self.db.execute('INSERT OR IGNORE INTO seen VALUES(?,?,?,?,?)',
                                (seen_wallet, row['coin'], row['trade_id'], observed_ms, phase))
                first = self.db.execute('SELECT phase FROM seen WHERE wallet=? AND coin=? AND tid=?', (seen_wallet, row['coin'], row['trade_id'])).fetchone()[0]
                phase = 'bootstrap' if bootstrap else first
                self.db.execute('INSERT OR IGNORE INTO fills VALUES(?,?,?,?,?,?,?,?,?,?)',
                                (source, wallet or '', row['coin'], row['trade_id'], row['event_ms'], observed_ms, phase, digest(row), encode(row), frame_id))
                if source in ('hl_ws', 'hl_rest') and annotations[index] is not None:
                    self.db.execute('INSERT OR IGNORE INTO annotations VALUES(?,?,?,?,?,?,?,?,?)',
                                    (source, wallet, row['coin'], row['trade_id'], row['event_ms'], observed_ms, phase, encode(annotations[index]), frame_id))
            if source == 'spot':
                self.db.execute('INSERT INTO spot_polls(observed_ms,overlap,frame_id) VALUES(?,?,?)', (observed_ms, int(overlap), frame_id))
            if quote:
                self.db.execute('INSERT INTO quotes(observed_ms,event_ms,bid,ask,bid_size,ask_size,frame_id) VALUES(?,?,?,?,?,?,?)', (*quote, frame_id))
        self.db.commit()
        return {'valid': error is None, 'error': error, 'rows': len(rows) if error is None else 0, 'frame_id': frame_id}

    def flow(self, now, minutes):
        start = now - minutes * 60000
        rows = self.db.execute('SELECT observed_ms,overlap FROM spot_polls WHERE observed_ms BETWEEN ? AND ? ORDER BY observed_ms',
                               (start - SPEC['max_spot_poll_gap_seconds'] * 1000, now)).fetchall()
        before = [row for row in rows if row['observed_ms'] <= start]
        reason = None
        if not before or not rows or now - rows[-1]['observed_ms'] > SPEC['max_spot_poll_gap_seconds'] * 1000:
            reason = 'warmup_or_stale'
        else:
            rows = [before[-1]] + [row for row in rows if row['observed_ms'] > start]
            if any(not row['overlap'] for row in rows):
                reason = 'trade_id_overlap_unknown'
            if any(b['observed_ms'] - a['observed_ms'] > SPEC['max_spot_poll_gap_seconds'] * 1000 for a, b in zip(rows, rows[1:])):
                reason = 'spot_poll_gap'
            if self.db.execute("SELECT 1 FROM frames WHERE source='spot' AND valid=0 AND observed_ms BETWEEN ? AND ?", (start, now)).fetchone():
                reason = 'source_error'
        totals, newest = Counter(), 0
        for record in self.db.execute("SELECT payload FROM fills WHERE source='spot' AND event_ms>? AND event_ms<=? AND observed_ms<=?", (start, now, now)):
            row = json.loads(record[0]); totals[row['side']] += row['price'] * row['size']; newest = max(newest, row['event_ms'])
        total = totals['A'] + totals['B']
        if not total or not math.isfinite(total) or now - newest > SPEC['max_spot_trade_age_seconds'] * 1000:
            reason = reason or 'no_fresh_valid_spot_trades'
        return {'usable': reason is None, 'reason': reason, 'minutes': minutes,
                'imbalance': (totals['B'] - totals['A']) / total if reason is None else None}

    def report(self, state='collecting', until=None, error=None):
        now = int(self.clock() * 1000)
        live_total = self.db.execute("SELECT COUNT(*) FROM fills WHERE source='hl_ws' AND phase='live' AND event_ms>=?", (self.started_ms,)).fetchone()[0]
        latencies = sorted(row[0] / 1000 for row in self.db.execute("SELECT observed_ms-event_ms FROM fills WHERE source='hl_ws' AND phase='live' AND event_ms>=? ORDER BY observed_ms DESC LIMIT 10000", (self.started_ms,)))
        per_source = dict(self.db.execute('SELECT source,COUNT(*) FROM frames GROUP BY source'))
        last = dict(self.db.execute('SELECT source,MAX(observed_ms) FROM frames WHERE valid=1 GROUP BY source'))
        phases = {name: dict(self.db.execute('SELECT phase,COUNT(*) FROM fills WHERE source=? GROUP BY phase', (name,))) for name in ('hl_ws', 'hl_rest', 'spot', 'hl_market')}
        matches = self.db.execute('''SELECT COUNT(*) FROM annotations r JOIN fills w
             ON w.source='hl_ws' AND w.wallet=r.wallet AND w.coin=r.coin AND w.tid=r.tid AND w.phase='live'
             WHERE r.source='hl_rest' AND r.phase='live' ''').fetchone()[0]
        missing_flags = self.db.execute('''SELECT COUNT(*) FROM annotations r JOIN fills w
             ON w.source='hl_ws' AND w.wallet=r.wallet AND w.coin=r.coin AND w.tid=r.tid AND w.phase='live'
             LEFT JOIN annotations a ON a.source='hl_ws' AND a.wallet=r.wallet AND a.coin=r.coin AND a.tid=r.tid
                  AND a.phase='live'
             WHERE r.source='hl_rest' AND r.phase='live' AND a.tid IS NULL''').fetchone()[0]
        explicit = dict(self.db.execute("SELECT source,COUNT(*) FROM annotations WHERE phase='live' GROUP BY source"))
        metadata_latency = {}
        for source in ('hl_ws', 'hl_rest'):
            values = sorted(row[0] / 1000 for row in self.db.execute('SELECT observed_ms-event_ms FROM annotations WHERE source=? AND phase=\'live\' ORDER BY observed_ms DESC LIMIT 10000', (source,)))
            metadata_latency[source] = {'sample_count': len(values), 'p95': values[math.ceil(.95 * len(values))-1] if values else None}
        failures = self.db.execute('SELECT COUNT(*) FROM frames WHERE valid=0').fetchone()[0]
        feeds = {}
        for feed in ('hyperliquid',):
            row = self.db.execute('SELECT connection,state,observed_ms,payload FROM feed_events WHERE feed=? ORDER BY id DESC LIMIT 1', (feed,)).fetchone()
            feeds[feed] = dict(row) if row else {'state': 'not_started'}
        quote = self.db.execute('SELECT * FROM quotes ORDER BY id DESC LIMIT 1').fetchone()
        quote_age = (now - quote['event_ms']) / 1000 if quote else None
        flow = {str(m): self.flow(now, m) for m in (5, 15)}
        p95 = latencies[min(len(latencies)-1, math.ceil(.95 * len(latencies))-1)] if latencies else None
        blockers = ['published_provider_limits_not_freshly_verified', 'liquidation_coverage_is_sampled']
        if len(latencies) < 100: blockers.append('insufficient_live_latency_sample')
        if p95 is None or p95 > SPEC['max_live_latency_seconds_for_review']: blockers.append('live_latency_not_passed')
        if matches < 3: blockers.append('insufficient_matched_liquidation_metadata_cases')
        meta_p95 = metadata_latency['hl_ws']['p95']
        if meta_p95 is None or meta_p95 > SPEC['max_live_latency_seconds_for_review']: blockers.append('explicit_liquidation_information_latency_not_passed')
        if missing_flags: blockers.append('ws_liquidation_metadata_missing_in_matched_cases')
        if not all(row['usable'] for row in flow.values()): blockers.append('spot_flow_window_not_ready')
        if quote_age is None or quote_age > SPEC['max_quote_age_seconds']: blockers.append('quote_not_fresh')
        return {'schema_version': 1, 'status': 'liquidation_data_pilot', 'runtime_state': state,
                'generated_at': datetime.fromtimestamp(now / 1000, timezone.utc).isoformat(),
                'started_at': datetime.fromtimestamp(self.started_ms / 1000, timezone.utc).isoformat(),
                'protocol_sha256': self.protocol_hash, 'wallet_count': len(self.wallets),
                'orders_enabled': False, 'strategy_enabled': False, 'alpha_validated': False,
                'worker': {'pid': __import__('os').getpid(), 'until': until, 'error': error},
                'frames': per_source, 'last_valid_ingress_ms': last, 'rejected_frames': failures,
                'capped_rest_responses': self.db.execute("SELECT COUNT(*) FROM frames WHERE source='hl_rest' AND capped=1").fetchone()[0],
                'fill_phases': phases, 'live_ws_unique_btc_fills': live_total,
                'live_latency_seconds': {'sample': 'Latest at most 10000 eligible unique BTC fills', 'sample_count': len(latencies), 'p50': latencies[len(latencies)//2] if latencies else None, 'p95': p95},
                'explicit_live_metadata_by_source': explicit, 'rest_explicit_with_live_ws_fill': matches,
                'explicit_information_latency_seconds': metadata_latency,
                'matched_rest_liquidations_without_ws_flag': missing_flags,
                'feeds': feeds, 'quote_age_seconds': quote_age, 'spot_flow': flow,
                'readiness_blockers': blockers, 'source_review_ready': False,
                'limits': SPEC['coverage'] + ' ' + SPEC['limit_policy']}
