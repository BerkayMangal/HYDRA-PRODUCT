"""Append-only, causal TWAP event study. Public data, no signing or orders."""

from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import time
from types import MappingProxyType


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a market number')
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError('Finite market value required') from None
    if not math.isfinite(value):
        raise ValueError('Finite market value required')
    return value


def slices(body, observed_ms):
    if not isinstance(body, list) or len(body) > 2000:
        raise ValueError('Bounded recent slice list required')
    found = {}
    for item in body:
        fill = item['fill']
        coin, tid, twap = fill['coin'], fill['tid'], item['twapId']
        event, price, size = number(fill['time']), number(fill['px']), number(fill['sz'])
        if (not isinstance(coin, str) or not coin or isinstance(tid, bool)
                or not isinstance(tid, (str, int)) or not str(tid)
                or isinstance(twap, bool) or not isinstance(twap, (str, int))
                or not str(twap) or event != int(event) or not 0 < event <= observed_ms
                or price <= 0 or size <= 0 or fill['side'] not in ('A', 'B')):
            raise ValueError('Invalid or future slice')
        row = {'coin': coin, 'trade_id': str(tid), 'twap_id': str(twap),
               'event_ms': int(event), 'side': fill['side'], 'price': price, 'size': size}
        key = (coin, str(tid))
        if key in found and found[key] != row:
            raise ValueError('Conflicting duplicate within a source response')
        found[key] = row
    return list(found.values())


class SignalLedger:
    def __init__(self, directory, cohorts, *, spec=None, clock=time.time):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.spec = MappingProxyType(json.loads(encoded(spec or json.loads((Path(__file__).parent / 'twap_forward_spec.json').read_text()))))
        cohorts = json.loads(encoded(cohorts))
        if self.spec.get('orders_enabled') is not False:
            raise ValueError('Read-only research specification required')
        for key in ['notional_usdt', 'feature_window_seconds', 'hold_seconds', 'decision_bucket_seconds',
                    'twap_poll_seconds', 'quote_poll_seconds', 'max_account_poll_gap_seconds',
                    'max_batch_span_seconds', 'max_quote_age_seconds', 'max_reference_delay_seconds']:
            if number(self.spec[key]) <= 0:
                raise ValueError('Positive research budgets and horizons required')
        for key in ['fee_per_side_bps', 'slippage_per_side_bps']:
            if not 0 <= number(self.spec[key]) <= 100:
                raise ValueError('Assumed costs must be between 0 and 100 bp per side')
        self.wallets = sorted({wallet.lower() for group in cohorts.values() for wallet in group['wallets']})
        if not 1 <= len(self.wallets) <= 24:
            raise ValueError('Bounded frozen cohort required')
        from research.lab.hyperliquid_shadow import validate_cohort
        for cohort in cohorts.values():
            validate_cohort(cohort, 240)
        implementations = {}
        for relative in ['research/forward_ledger.py', 'research/lab/twap_forward.py']:
            source = Path(__file__).resolve().parents[1] / relative
            if source.is_file():
                implementations[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        expected = {'schema_version': 1, 'specification': dict(self.spec),
                    'cohorts': cohorts, 'implementation_sha256': implementations}
        path = self.directory / 'protocol.json'
        if path.exists():
            self.protocol = json.loads(path.read_text())
            if any(self.protocol.get(key) != value for key, value in expected.items()):
                raise ValueError('Existing trial definition differs; use a new trial directory')
        else:
            if any(self.directory.iterdir()):
                raise ValueError('Refusing to initialize over existing non-trial data')
            try:
                revision = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                          cwd=Path(__file__).resolve().parents[1],
                                          capture_output=True, text=True, timeout=5)
                source_commit = revision.stdout.strip() if revision.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                source_commit = None
            self.protocol = {**expected, 'created_ms': int(clock() * 1000),
                             'created_source_commit': source_commit}
            with path.open('x') as stream:
                stream.write(encoded(self.protocol) + '\n')
        self.protocol_hash = digest(self.protocol)
        self.started_ms = self.protocol['created_ms']
        self.db = sqlite3.connect(self.directory / 'ledger.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA recursive_triggers=ON')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS metadata(id INTEGER PRIMARY KEY CHECK(id=1),protocol_hash TEXT,payload TEXT);
          CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY,wallet TEXT,started_ms INTEGER,observed_ms INTEGER,ok INTEGER,oldest_ms INTEGER,capped INTEGER,error TEXT,raw_gzip BLOB,sha256 TEXT);
          CREATE INDEX IF NOT EXISTS requests_wallet ON requests(wallet,id);
          CREATE TABLE IF NOT EXISTS fills(wallet TEXT,coin TEXT,trade_id TEXT,event_ms INTEGER,first_observed_ms INTEGER,eligibility TEXT,semantic_sha TEXT,payload TEXT,PRIMARY KEY(wallet,coin,trade_id));
          CREATE INDEX IF NOT EXISTS fills_time ON fills(event_ms,first_observed_ms);
          CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY,observed_ms INTEGER,feature_end_ms INTEGER,ok INTEGER,payload TEXT);
          CREATE TABLE IF NOT EXISTS quotes(id INTEGER PRIMARY KEY,observed_ms INTEGER,event_ms INTEGER,valid INTEGER,reason TEXT,payload TEXT,raw_gzip BLOB,sha256 TEXT);
          CREATE INDEX IF NOT EXISTS quote_time ON quotes(observed_ms,event_ms);
          CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY,bucket INTEGER UNIQUE,observed_ms INTEGER,payload TEXT);
          CREATE TABLE IF NOT EXISTS entries(decision_id INTEGER PRIMARY KEY REFERENCES decisions(id),observed_ms INTEGER,payload TEXT);
          CREATE TABLE IF NOT EXISTS outcomes(decision_id INTEGER PRIMARY KEY REFERENCES decisions(id),observed_ms INTEGER,status TEXT,payload TEXT);
          CREATE TABLE IF NOT EXISTS run_events(id INTEGER PRIMARY KEY,observed_ms INTEGER,payload TEXT);
        ''')
        existing = self.db.execute('SELECT protocol_hash FROM metadata WHERE id=1').fetchone()
        if existing and existing[0] != self.protocol_hash:
            self.db.close()
            raise ValueError('Journal protocol hash differs')
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES(1,?,?)', (self.protocol_hash, encoded(self.protocol)))
        for table in ['metadata', 'requests', 'fills', 'batches', 'quotes', 'decisions', 'entries', 'outcomes', 'run_events']:
            for operation in ('UPDATE', 'DELETE'):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'append-only trial journal'); END")
        self.db.commit()

    def close(self):
        self.db.close()

    def run_event(self, body, observed_ms=None):
        self.db.execute('INSERT INTO run_events(observed_ms,payload) VALUES(?,?)',
                        (observed_ms or int(self.clock() * 1000), encoded(body)))
        self.db.commit()

    @staticmethod
    def raw(body):
        # Preserve even malformed/nonfinite decoded source responses for audit.
        raw = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=True).encode()
        return gzip.compress(raw, mtime=0), hashlib.sha256(raw).hexdigest()

    def quote(self, body, observed_ms):
        packed, sha = self.raw(body)
        valid, reason, row = True, None, {}
        try:
            source = body['data'][0]
            if body.get('code') != '0' or source['instId'] != 'BTC-USDT':
                raise ValueError('Wrong quote source/instrument')
            event = number(source['ts'])
            bid, ask = number(source['bidPx']), number(source['askPx'])
            bid_size, ask_size = number(source['bidSz']), number(source['askSz'])
            if (event != int(event) or not 0 <= observed_ms - event <= self.spec['max_quote_age_seconds'] * 1000
                    or not 0 < bid < ask or bid_size <= 0 or ask_size <= 0):
                raise ValueError('Stale, future, crossed or empty quote')
            row = {'event_ms': int(event), 'observed_ms': observed_ms, 'bid': bid, 'ask': ask,
                   'bid_size': bid_size, 'ask_size': ask_size, 'mid': bid + (ask - bid) / 2}
        except (KeyError, IndexError, TypeError, ValueError):
            valid, reason = False, 'invalid_or_stale_quote'
        self.db.execute('INSERT INTO quotes(observed_ms,event_ms,valid,reason,payload,raw_gzip,sha256) VALUES(?,?,?,?,?,?,?)',
                        (observed_ms, row.get('event_ms'), int(valid), reason, encoded(row), packed, sha))
        self.db.commit()
        return valid

    def request(self, sample):
        wallet, start, observed = sample['wallet'], sample['started_ms'], sample['observed_ms']
        body = sample.get('body')
        packed, sha = self.raw(sample)
        previous = self.db.execute('SELECT * FROM requests WHERE wallet=? AND ok=1 ORDER BY id DESC LIMIT 1', (wallet,)).fetchone()
        error, rows = sample.get('error'), []
        try:
            if error or wallet not in self.wallets or not self.started_ms <= start <= observed:
                raise ValueError('Failed request or invalid local receipt time')
            rows = slices(body, observed)
            if previous:
                old_body = json.loads(gzip.decompress(previous['raw_gzip']))['body']
                old_keys = {(r['coin'], r['trade_id']) for r in slices(old_body, previous['observed_ms'])}
                keys = {(r['coin'], r['trade_id']) for r in rows}
                if old_keys and not old_keys.intersection(keys):
                    raise ValueError('Recent history has no actual trade-ID overlap')
            for row in rows:
                if row['coin'] != 'BTC':
                    continue
                old = self.db.execute('SELECT semantic_sha FROM fills WHERE wallet=? AND coin=? AND trade_id=?',
                                      (wallet, row['coin'], row['trade_id'])).fetchone()
                if old and old[0] != digest(row):
                    raise ValueError('Previously observed fill changed')
        except (KeyError, TypeError, ValueError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)[:120]
        ok = error is None
        self.db.execute('INSERT INTO requests(wallet,started_ms,observed_ms,ok,oldest_ms,capped,error,raw_gzip,sha256) VALUES(?,?,?,?,?,?,?,?,?)',
                        (wallet, start, observed, int(ok), min((r['event_ms'] for r in rows), default=None),
                         int(isinstance(body, list) and len(body) >= 2000), error, packed, sha))
        if ok:
            for row in rows:
                if row['coin'] != 'BTC':
                    continue
                eligibility = ('bootstrap' if previous is None else 'pretrial' if row['event_ms'] < self.started_ms
                               else 'late' if observed - row['event_ms'] > self.spec['max_new_slice_receipt_delay_seconds'] * 1000
                               else 'eligible')
                self.db.execute('INSERT OR IGNORE INTO fills VALUES(?,?,?,?,?,?,?,?)',
                                (wallet, 'BTC', row['trade_id'], row['event_ms'], observed, eligibility, digest(row), encoded(row)))
        return {'wallet': wallet, 'ok': ok, 'error': error,
                'capped': isinstance(body, list) and len(body) >= 2000,
                'oldest_ms': min((r['event_ms'] for r in rows), default=None)}

    def batch(self, samples, observed_ms):
        if sorted(s['wallet'] for s in samples) != self.wallets:
            raise ValueError('Every frozen account must have exactly one response or error')
        rows = [self.request(sample) for sample in samples]
        end = min(sample['started_ms'] for sample in samples)
        errors = [row['error'] for row in rows if not row['ok']]
        if any(sample['observed_ms'] > observed_ms for sample in samples):
            errors.append('receipt_after_batch_time')
        if observed_ms - end > self.spec['max_batch_span_seconds'] * 1000:
            errors.append('source_batch_too_old')
        payload = {'observed_ms': observed_ms, 'feature_end_ms': end,
                   'successful_accounts': sum(row['ok'] for row in rows),
                   'requested_accounts': len(self.wallets), 'capped_accounts': sum(row['capped'] for row in rows),
                   'requests': rows, 'errors': errors}
        cursor = self.db.execute('INSERT INTO batches(observed_ms,feature_end_ms,ok,payload) VALUES(?,?,?,?)',
                                 (observed_ms, end, int(not errors), encoded(payload)))
        self.db.commit()
        return cursor.lastrowid

    def source_features(self, batch_id, now):
        batch = self.db.execute('SELECT * FROM batches WHERE id=?', (batch_id,)).fetchone()
        body = json.loads(batch['payload'])
        end = batch['feature_end_ms']
        start = end - self.spec['feature_window_seconds'] * 1000
        gap = self.spec['max_account_poll_gap_seconds'] * 1000
        history = self.db.execute('SELECT * FROM batches WHERE feature_end_ms BETWEEN ? AND ? AND observed_ms<=? ORDER BY feature_end_ms,id',
                                  (start - gap, end, now)).fetchall()
        before = [row for row in history if row['feature_end_ms'] <= start]
        reasons = []
        if not before:
            reasons.append('coverage_warmup')
        else:
            history = [before[-1]] + [row for row in history if row['feature_end_ms'] > start]
            if any(not row['ok'] for row in history):
                reasons.append('source_error_in_window')
            if any(b['feature_end_ms'] - a['feature_end_ms'] > gap for a, b in zip(history, history[1:])):
                reasons.append('source_poll_gap')
        if not batch['ok'] or now - end > self.spec['max_batch_span_seconds'] * 1000:
            reasons.append('current_source_unavailable')
        if any(row['capped'] and (row['oldest_ms'] is None or row['oldest_ms'] > start) for row in body['requests']):
            reasons.append('capped_history_does_not_reach_window_start')
        fills = self.db.execute("SELECT wallet,eligibility,payload FROM fills WHERE event_ms>? AND event_ms<=? AND first_observed_ms<=?", (start, end, now)).fetchall()
        if any(row['eligibility'] == 'late' for row in fills):
            reasons.append('late_slice_in_window')
        eligible = [(row['wallet'], json.loads(row['payload'])) for row in fills if row['eligibility'] == 'eligible']
        net = sum(row['size'] if row['side'] == 'B' else -row['size'] for _, row in eligible)
        if not math.isfinite(net):
            reasons.append('invalid_flow_arithmetic')
        keys = sorted((wallet, row['coin'], row['trade_id']) for wallet, row in eligible) if not reasons else []
        return {'window_start_ms': start, 'window_end_ms': end, 'net_btc_base': net if not reasons else None,
                'eligible_slices': len(eligible), 'active_accounts': len({wallet for wallet, _ in eligible}),
                'source_lag_seconds': (now - end) / 1000, 'reasons': reasons,
                'input_fill_keys': keys, 'input_fill_keys_sha256': digest(keys)}

    def decide(self, batch_id, now):
        if now < self.started_ms or abs(self.clock() * 1000 - now) > 1000:
            raise ValueError('Decisions must be recorded at the actual current clock')
        batch_time = self.db.execute('SELECT observed_ms FROM batches WHERE id=?', (batch_id,)).fetchone()[0]
        last_time = self.db.execute('SELECT MAX(observed_ms) FROM decisions').fetchone()[0]
        if batch_time > now or (last_time is not None and now < last_time):
            raise ValueError('Future batches and retroactive decisions are forbidden')
        bucket = now // (self.spec['decision_bucket_seconds'] * 1000)
        if self.db.execute('SELECT 1 FROM decisions WHERE bucket=?', (bucket,)).fetchone():
            return None
        features = self.source_features(batch_id, now)
        reasons = list(features['reasons'])
        current = self.db.execute('SELECT id,payload FROM quotes WHERE valid=1 AND observed_ms<=? AND observed_ms>=? ORDER BY observed_ms DESC,id DESC LIMIT 1',
                                  (now, now - self.spec['max_reference_delay_seconds'] * 1000)).fetchone()
        past_due = now - self.spec['feature_window_seconds'] * 1000
        past = self.reference(past_due, now)
        current_quote = {**json.loads(current['payload']), 'quote_id': current['id']} if current else None
        if current is None or past is None:
            reasons.append('momentum_reference_unavailable')
        if self.db.execute('SELECT 1 FROM decisions d LEFT JOIN outcomes o ON d.id=o.decision_id WHERE o.decision_id IS NULL').fetchone():
            reasons.append('previous_paired_event_pending')
        payload = {'observed_ms': now, 'bucket': bucket, 'batch_id': batch_id,
                   'features': features, 'status': 'abstain' if reasons else 'decision',
                   'reasons': reasons, 'strategy_long': None if reasons else features['net_btc_base'] > 0,
                   'control_long': None if reasons else current_quote['mid'] > past['mid'],
                   'momentum_references': {'current': current_quote, 'past': past},
                   'entry_due_ms': now + self.spec['entry_latency_seconds'] * 1000,
                   'protocol_sha256': self.protocol_hash, 'orders_enabled': False}
        cursor = self.db.execute('INSERT INTO decisions(bucket,observed_ms,payload) VALUES(?,?,?)', (bucket, now, encoded(payload)))
        decision_id = cursor.lastrowid
        if reasons or not (payload['strategy_long'] or payload['control_long']):
            self.outcome(decision_id, now, 'abstain' if reasons else 'flat', {'strategy': None, 'control': None})
        self.db.commit()
        return decision_id

    def reference(self, due, as_of):
        row = self.db.execute('SELECT id,payload FROM quotes WHERE valid=1 AND observed_ms>=? AND event_ms>=? AND observed_ms<=? AND observed_ms<=? ORDER BY observed_ms,id LIMIT 1',
                               (due, due, due + self.spec['max_reference_delay_seconds'] * 1000, as_of)).fetchone()
        return {**json.loads(row['payload']), 'quote_id': row['id']} if row else None

    def outcome(self, decision_id, now, status, body):
        self.db.execute('INSERT INTO outcomes VALUES(?,?,?,?)', (decision_id, now, status, encoded(body)))

    def advance(self, now):
        pending = self.db.execute('SELECT d.id,d.payload,e.payload AS entry FROM decisions d LEFT JOIN entries e ON e.decision_id=d.id LEFT JOIN outcomes o ON o.decision_id=d.id WHERE o.decision_id IS NULL ORDER BY d.id').fetchall()
        for row in pending:
            decision = json.loads(row['payload'])
            fee = self.spec['fee_per_side_bps'] / 10000
            slip = self.spec['slippage_per_side_bps'] / 10000
            if row['entry'] is None:
                quote = self.reference(decision['entry_due_ms'], now)
                if quote:
                    quantity = self.spec['notional_usdt'] / (quote['ask'] * (1 + slip) * (1 + fee))
                    if not math.isfinite(quantity) or quantity <= 0:
                        self.outcome(row['id'], now, 'arithmetic_unknown', {'strategy': None, 'control': None})
                        continue
                    if quote['ask_size'] < quantity:
                        self.outcome(row['id'], now, 'entry_depth_unknown', {'strategy': None, 'control': None})
                        continue
                    entry = {'quote': quote, 'quantity': quantity,
                             'exit_due_ms': quote['observed_ms'] + self.spec['hold_seconds'] * 1000}
                    self.db.execute('INSERT INTO entries VALUES(?,?,?)', (row['id'], quote['observed_ms'], encoded(entry)))
                elif now > decision['entry_due_ms'] + self.spec['max_reference_delay_seconds'] * 1000:
                    self.outcome(row['id'], now, 'entry_missing', {'strategy': None, 'control': None})
                    continue
                else:
                    continue
            else:
                entry = json.loads(row['entry'])
            exit_quote = self.reference(entry['exit_due_ms'], now)
            if exit_quote:
                if exit_quote['bid_size'] < entry['quantity']:
                    self.outcome(row['id'], now, 'exit_depth_unknown', {'strategy': None, 'control': None})
                    continue
                original = entry['quote']
                net = (entry['quantity'] * exit_quote['bid'] * (1 - slip) * (1 - fee)
                       / self.spec['notional_usdt'] - 1)
                observed_mids = [json.loads(r[0])['mid'] / original['mid'] - 1 for r in self.db.execute('SELECT payload FROM quotes WHERE valid=1 AND observed_ms BETWEEN ? AND ?', (original['observed_ms'], exit_quote['observed_ms']))]
                if not math.isfinite(net) or any(not math.isfinite(value) for value in observed_mids):
                    self.outcome(row['id'], now, 'arithmetic_unknown', {'strategy': None, 'control': None})
                    continue
                trade = {'modeled_net_return': net, 'modeled_pnl_usdt': net * self.spec['notional_usdt'],
                         'gross_quote_move': exit_quote['bid'] / original['ask'] - 1,
                         'sampled_mfe': max(observed_mids, default=0),
                         'sampled_mae': min(observed_mids, default=0)}
                self.outcome(row['id'], now, 'completed', {
                    'entry': original, 'exit': exit_quote,
                    'strategy': trade if decision['strategy_long'] else None,
                    'control': trade if decision['control_long'] else None})
            elif now > entry['exit_due_ms'] + self.spec['max_reference_delay_seconds'] * 1000:
                self.outcome(row['id'], now, 'exit_missing', {'strategy': None, 'control': None})
        self.db.commit()

    def report(self, now=None, runtime_state='collecting'):
        now = int(self.clock() * 1000) if now is None else now
        decision_rows = self.db.execute('SELECT id,payload FROM decisions ORDER BY id').fetchall()
        decisions = [json.loads(row['payload']) for row in decision_rows]
        outcomes = [(r['status'], json.loads(r['payload'])) for r in self.db.execute('SELECT status,payload FROM outcomes ORDER BY decision_id')]
        last = self.db.execute('SELECT id,observed_ms,payload FROM batches ORDER BY id DESC LIMIT 1').fetchone()
        last_quote = self.db.execute('SELECT MAX(observed_ms) FROM quotes WHERE valid=1').fetchone()[0]
        completed = [body for status, body in outcomes if status == 'completed']
        unknown = sum(status not in ('completed', 'flat', 'abstain') for status, _ in outcomes)
        stats = {}
        for variant in ('strategy', 'control'):
            trades = [body[variant] for body in completed if body[variant] is not None]
            stats[variant] = {'completed_trades': len(trades),
                              'mean_modeled_net_return': sum(t['modeled_net_return'] for t in trades) / len(trades) if trades else None,
                              'sum_modeled_pnl_usdt': sum(t['modeled_pnl_usdt'] for t in trades) if trades else None}
        days = (now - self.started_ms) / 86400000
        paired_differences = [(body['strategy']['modeled_net_return'] if body['strategy'] else 0)
                              - (body['control']['modeled_net_return'] if body['control'] else 0)
                              for body in completed]
        recent = []
        for record in decision_rows[-12:]:
            decision = json.loads(record['payload'])
            outcome = self.db.execute('SELECT status,payload FROM outcomes WHERE decision_id=?', (record['id'],)).fetchone()
            body = json.loads(outcome['payload']) if outcome else {}
            recent.append({**decision, 'decision_id': record['id'],
                           'features': {k: v for k, v in decision['features'].items() if k != 'input_fill_keys'},
                           'outcome_status': outcome['status'] if outcome else 'pending',
                           'strategy_modeled_net_return': body.get('strategy', {}).get('modeled_net_return') if body.get('strategy') else None,
                           'control_modeled_net_return': body.get('control', {}).get('modeled_net_return') if body.get('control') else None})
        from collections import Counter
        abstention_reasons = Counter(reason for decision in decisions for reason in decision['reasons'])
        return {'schema_version': 1, 'status': 'prospective_research_only', 'orders_enabled': False,
                'alpha_validated': False, 'protocol_sha256': self.protocol_hash,
                'created_source_commit': self.protocol.get('created_source_commit'),
                'started_at': datetime.fromtimestamp(self.started_ms / 1000, timezone.utc).isoformat(),
                'generated_at': datetime.fromtimestamp(now / 1000, timezone.utc).isoformat(),
                'runtime_state': runtime_state, 'last_valid_quote_ms': last_quote,
                'last_batch': json.loads(last['payload']) if last else None,
                'last_batch_features': {k: v for k, v in self.source_features(last['id'], last['observed_ms']).items() if k != 'input_fill_keys'} if last else None,
                'account_count': len(self.wallets), 'decisions': len(decisions),
                'abstentions': sum(d['status'] == 'abstain' for d in decisions),
                'valid_flat_decisions': sum(d['status'] == 'decision' and not(d['strategy_long'] or d['control_long']) for d in decisions),
                'pending_events': len(decisions) - len(outcomes), 'unknown_active_outcomes': unknown,
                'fill_eligibility_counts': dict(self.db.execute('SELECT eligibility,COUNT(*) FROM fills GROUP BY eligibility')),
                'request_errors': self.db.execute('SELECT COUNT(*) FROM requests WHERE ok=0').fetchone()[0],
                'capped_responses': self.db.execute('SELECT COUNT(*) FROM requests WHERE capped=1').fetchone()[0],
                'invalid_quotes': self.db.execute('SELECT COUNT(*) FROM quotes WHERE valid=0').fetchone()[0],
                'results': stats, 'recent_decisions': recent,
                'abstention_reason_counts': dict(abstention_reasons),
                'paired_completed_events': len(paired_differences),
                'paired_mean_net_difference': sum(paired_differences) / len(paired_differences) if paired_differences else None,
                'assumptions': {k: self.spec[k] for k in ['notional_usdt', 'fee_per_side_bps', 'slippage_per_side_bps', 'entry_latency_seconds', 'hold_seconds']},
                'review_sample_thresholds_met': days >= self.spec['minimum_elapsed_days_for_review'] and stats['strategy']['completed_trades'] >= self.spec['minimum_completed_strategy_trades_for_review'] and unknown == 0,
                'limits': self.spec['limits'] + ' ' + self.spec['accounting']}
