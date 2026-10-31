"""Recorded TWAP economics and concentration; no new decisions or settlement."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from .lab.forward_report import RecordedLedger


def concentration(fills):
    accounts, parents = {}, set()
    for wallet, row in fills:
        stats = accounts.setdefault(wallet, {'wallet': wallet, 'slices': 0, 'gross_base': 0,
                                             'net_base': 0, 'gross_notional_usd': 0,
                                             'signed_notional_usd': 0})
        sign = 1 if row['side'] == 'B' else -1
        stats['slices'] += 1
        stats['gross_base'] += row['size']
        stats['net_base'] += sign * row['size']
        stats['gross_notional_usd'] += row['price'] * row['size']
        stats['signed_notional_usd'] += sign * row['price'] * row['size']
        if not all(math.isfinite(stats[key]) for key in ('gross_base', 'net_base', 'gross_notional_usd', 'signed_notional_usd')):
            raise ValueError('Nonfinite activity arithmetic')
        parents.add((wallet, row['twap_id']))
    rows = sorted(accounts.values(), key=lambda item: (-item['gross_notional_usd'], item['wallet']))
    gross = sum(row['gross_notional_usd'] for row in rows)
    if not math.isfinite(gross):
        raise ValueError('Nonfinite activity arithmetic')
    shares = [row['gross_notional_usd'] / gross for row in rows] if gross else []
    net_base = sum(row['net_base'] for row in rows)
    return {'active_accounts': len(rows), 'observed_parent_ids': len(parents),
            'observed_slices': len(fills), 'account_side_gross_notional_usd': gross,
            'account_side_signed_notional_usd': sum(row['signed_notional_usd'] for row in rows),
            'net_base': net_base, 'largest_account_share': max(shares) if shares else None,
            'effective_account_count': 1 / sum(value * value for value in shares) if shares else None,
            'net_base_without_largest_account': net_base - rows[0]['net_base'] if rows else None,
            'accounts': rows,
            'limits': 'Observed sampled account-side activity, not venue volume or independent owners. Parent IDs reveal no remaining quantity or schedule.'}


def trade_economics(entry, exit_quote, recorded, spec):
    fee, slip = spec['fee_per_side_bps'] / 10000, spec['slippage_per_side_bps'] / 10000
    ratio = exit_quote['bid'] / entry['ask']
    gross_mid = exit_quote['mid'] / entry['mid'] - 1
    gross_quote = ratio - 1
    net = ratio * (1 - slip) * (1 - fee) / ((1 + slip) * (1 + fee)) - 1
    if not math.isclose(net, recorded['modeled_net_return'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Recorded modeled return fails independent cost arithmetic')
    after_slip = ratio * (1 - slip) / (1 + slip)
    return {'gross_mid_return_bps': gross_mid * 10000,
            'gross_bid_ask_return_bps': gross_quote * 10000,
            'spread_drag_bps': (gross_mid - gross_quote) * 10000,
            'assumed_fee_slippage_drag_bps': (gross_quote - net) * 10000,
            'modeled_net_return_bps': net * 10000,
            'max_break_even_fee_per_side_bps_at_assumed_slippage':
                ((after_slip - 1) / (after_slip + 1) * 10000) if after_slip >= 1 else None,
            'costs_exceed_gross_quote_move': gross_quote < gross_quote - net}


def diagnose_trial(directory):
    ledger = RecordedLedger(directory)
    try:
        summary = ledger.report(now=ledger.as_of_ms, runtime_state='recorded_snapshot')
        features = summary['last_batch_features']
        fills = []
        if features:
            for row in ledger.db.execute("SELECT wallet,payload FROM fills WHERE eligibility='eligible' AND event_ms>? AND event_ms<=? AND first_observed_ms<=?",
                                         (features['window_start_ms'], features['window_end_ms'], summary['last_batch']['observed_ms'])):
                fills.append((row['wallet'], json.loads(row['payload'])))
        activity = concentration(fills)
        activity.update(window_start_ms=features['window_start_ms'] if features else None,
                        window_end_ms=features['window_end_ms'] if features else None,
                        source_window_usable=features is not None and not features['reasons'],
                        source_reasons=features['reasons'] if features else ['no_source_batch'])
        by_variant = {name: {'completed': [], 'valid_flat_decisions': 0, 'pending_active': 0,
                             'unknown_active': 0, 'abstentions': 0} for name in ('strategy', 'control')}
        events = []
        for record in ledger.db.execute('''SELECT d.id,d.payload,e.payload AS entry,
                  o.status AS outcome_status,o.payload AS outcome FROM decisions d
                  LEFT JOIN entries e ON e.decision_id=d.id LEFT JOIN outcomes o ON o.decision_id=d.id ORDER BY d.id'''):
            decision = json.loads(record['payload'])
            outcome = json.loads(record['outcome']) if record['outcome'] else None
            item = {'decision_id': record['id'], 'observed_ms': decision['observed_ms'],
                    'strategy_long': decision['strategy_long'], 'control_long': decision['control_long'],
                    'outcome_status': record['outcome_status'] or 'pending', 'economics': {}}
            for variant, stats in by_variant.items():
                active = decision[variant + '_long']
                if active is None:
                    stats['abstentions'] += 1
                elif not active:
                    stats['valid_flat_decisions'] += 1
                elif outcome is None:
                    stats['pending_active'] += 1
                elif record['outcome_status'] != 'completed':
                    stats['unknown_active'] += 1
                else:
                    metrics = trade_economics(json.loads(record['entry'])['quote'], outcome['exit'], outcome[variant], ledger.spec)
                    stats['completed'].append(metrics)
                    item['economics'][variant] = metrics
            events.append(item)
        results = {}
        fields = ('gross_mid_return_bps', 'gross_bid_ask_return_bps', 'spread_drag_bps',
                  'assumed_fee_slippage_drag_bps', 'modeled_net_return_bps')
        for variant, stats in by_variant.items():
            trades = stats.pop('completed')
            results[variant] = {**stats, 'completed_trades': len(trades),
                                'mean': {field: sum(row[field] for row in trades) / len(trades) if trades else None for field in fields}}
        fee, slip = ledger.spec['fee_per_side_bps'] / 10000, ledger.spec['slippage_per_side_bps'] / 10000
        return {'schema_version': 1, 'status': 'recorded_forward_diagnostics',
                'recorded_through': summary['generated_at'], 'started_at': summary['started_at'],
                'protocol_sha256': ledger.protocol_hash, 'orders_enabled': False, 'alpha_validated': False,
                'analysis_design': 'Post-start descriptive diagnostics; no new trading rule, parameter search or alpha validation.',
                'signal_activity': activity, 'results': results, 'recent_events': events[-12:],
                'required_bid_over_entry_ask_gain_bps_at_assumed_costs':
                    (((1 + slip) * (1 + fee) / ((1 - slip) * (1 - fee))) - 1) * 10000,
                'assumptions': summary['assumptions'], 'unknown_active_outcomes': summary['unknown_active_outcomes'],
                'paired_completed_events': summary['paired_completed_events'],
                'paired_mean_net_difference_bps': summary['paired_mean_net_difference'] * 10000 if summary['paired_mean_net_difference'] is not None else None,
                'limits': 'Fixed-notional paper events, not portfolio NAV or actual fills. Tiny dependent samples do not establish alpha. Flat controls are not completed trades. Gross/net decomposition is descriptive, not a newly tested strategy.'}
    finally:
        ledger.close()


def load_diagnostics(clock=time.time):
    configured = os.environ.get('HYDRA_FORWARD_TRIAL')
    absent = {'schema_version': 1, 'status': 'unavailable' if configured else 'not_configured',
              'orders_enabled': False, 'alpha_validated': False}
    if not configured:
        return absent
    try:
        body = diagnose_trial(Path(configured))
        age = clock() - datetime.fromisoformat(body['recorded_through']).astimezone(timezone.utc).timestamp()
        body.update(record_age_seconds=age, is_stale=not 0 <= age <= 90)
        return body
    except (OSError, ValueError, TypeError, KeyError, IndexError, sqlite3.Error):
        return absent
