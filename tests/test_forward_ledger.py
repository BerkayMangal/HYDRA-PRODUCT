"""Synthetic clock/book fixtures test causality; these are not strategy results."""

from copy import deepcopy
import json
import sqlite3

import pytest

from research.forward_ledger import SignalLedger, slices
from research.lab.twap_forward import audit

WALLET = '0x' + '1' * 40
COHORTS = {'fixed': {'wallets': [WALLET], 'scope': 'synthetic test', 'frozen_at_ms': 1}}


def quote(when, bid=99, ask=101, size=100):
    return {'code': '0', 'data': [{'instId': 'BTC-USDT', 'ts': str(when),
            'bidPx': str(bid), 'askPx': str(ask), 'bidSz': str(size), 'askSz': str(size)}]}


def fill(tid=1, event=990000, side='B', price=80000):
    return {'twapId': 9, 'fill': {'coin': 'BTC', 'tid': tid, 'time': event,
            'side': side, 'px': str(price), 'sz': '1'}}


def batch(ledger, when, body=None, error=None):
    sample = {'wallet': WALLET, 'started_ms': when, 'observed_ms': when + 1000,
              'body': body if body is not None else [fill()]}
    if error:
        sample['error'] = error
    return ledger.batch([sample], when + 1000)


def make_ledger(tmp_path):
    clock = [1000.0]
    ledger = SignalLedger(tmp_path / 'trial', COHORTS, clock=lambda: clock[0])
    return ledger, clock


def ready_trial(tmp_path):
    ledger, clock = make_ledger(tmp_path)
    for when in range(1000000, 1961001, 10000):
        ledger.quote(quote(when), when)
    for i, when in enumerate([1000000, 1240000, 1480000, 1720000, 1960000]):
        body = [fill()] + ([fill(2, 1230000)] if i else [])
        batch_id = batch(ledger, when, body)
    clock[0] = 1961
    ledger.quote(quote(1961000), 1961000)
    return ledger, clock, batch_id


def test_bootstrap_duplicates_never_become_new_flow_or_refresh_first_receipt(tmp_path):
    ledger, clock = make_ledger(tmp_path)
    batch(ledger, 1000000)
    first = ledger.db.execute('SELECT * FROM fills').fetchone()
    batch(ledger, 1240000, [fill(), fill(2, 1230000), fill(2, 1230000)])
    again = ledger.db.execute('SELECT * FROM fills WHERE trade_id="1"').fetchone()
    assert again['first_observed_ms'] == first['first_observed_ms'] == 1001000
    assert again['eligibility'] == 'bootstrap'
    assert ledger.db.execute('SELECT COUNT(*) FROM fills').fetchone()[0] == 2
    assert ledger.db.execute('SELECT eligibility FROM fills WHERE trade_id="2"').fetchone()[0] == 'eligible'
    ledger.close()


def test_changed_old_fill_is_unknown_and_raw_response_remains_auditable(tmp_path):
    ledger, _ = make_ledger(tmp_path)
    batch(ledger, 1000000)
    original = ledger.db.execute('SELECT payload FROM fills').fetchone()[0]
    batch(ledger, 1240000, [fill(price=81000)])
    assert ledger.db.execute('SELECT ok FROM requests ORDER BY id DESC LIMIT 1').fetchone()[0] == 0
    assert ledger.db.execute('SELECT payload FROM fills').fetchone()[0] == original
    assert audit(ledger.directory)['raw_digest_mismatches'] == 0
    ledger.close()


def test_missing_history_is_not_flat_and_abstention_consumes_the_bucket(tmp_path):
    ledger, clock = make_ledger(tmp_path)
    batch_id = batch(ledger, 1000000)
    clock[0] = 1001
    first = ledger.decide(batch_id, 1001000)
    body = json.loads(ledger.db.execute('SELECT payload FROM decisions').fetchone()[0])
    assert body['status'] == 'abstain'
    assert body['strategy_long'] is None
    assert body['features']['net_btc_base'] is None
    batch_id = batch(ledger, 1240000)
    clock[0] = 1241
    assert ledger.decide(batch_id, 1241000) is None
    assert first is not None
    assert ledger.report()['results']['strategy']['mean_modeled_net_return'] is None
    ledger.close()


def test_entry_waits_for_observed_future_book_and_models_both_sides_of_costs(tmp_path):
    ledger, clock, batch_id = ready_trial(tmp_path)
    decision = ledger.decide(batch_id, 1961000)
    ledger.advance(1961000)
    assert ledger.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] == 0
    ledger.quote(quote(1963000, 100, 102), 1963000)
    ledger.advance(1963000)
    # This future fixture is already in storage; as-of entry/exit lookup must still hide it.
    ledger.quote(quote(2864000, 104, 106), 2864000)
    ledger.advance(2000000)
    assert ledger.db.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0] == 0
    clock[0] = 2864
    ledger.advance(2864000)
    result = json.loads(ledger.db.execute('SELECT payload FROM outcomes WHERE decision_id=?', (decision,)).fetchone()[0])
    expected = 104 * (1 - .0004) * (1 - .0008) / (102 * (1 + .0004) * (1 + .0008)) - 1
    assert result['strategy']['modeled_net_return'] == pytest.approx(expected)
    assert result['control'] is None
    assert ledger.report()['paired_mean_net_difference'] == pytest.approx(expected)
    assert ledger.report()['alpha_validated'] is False
    verification = audit(ledger.directory)
    assert verification['entry_reference_violations'] == 0
    assert verification['exit_reference_violations'] == 0
    assert verification['quote_reference_mismatches'] == 0
    assert verification['modeled_return_mismatches'] == 0
    ledger.close()


@pytest.mark.parametrize('phase', ['entry', 'exit'])
def test_missing_reference_is_never_replaced_by_zero_or_a_late_quote(tmp_path, phase):
    ledger, clock, batch_id = ready_trial(tmp_path)
    ledger.decide(batch_id, 1961000)
    if phase == 'exit':
        ledger.quote(quote(1963000, 100, 102), 1963000)
        ledger.advance(1963000)
        after_deadline = 2900000
    else:
        after_deadline = 2000000
    ledger.advance(after_deadline)
    original = ledger.db.execute('SELECT status,payload FROM outcomes').fetchone()
    assert original['status'] == phase + '_missing'
    ledger.quote(quote(after_deadline + 1000), after_deadline + 1000)
    ledger.advance(after_deadline + 1000)
    assert ledger.db.execute('SELECT payload FROM outcomes').fetchone()[0] == original['payload']
    assert ledger.report()['unknown_active_outcomes'] == 1
    assert ledger.report()['results']['strategy']['mean_modeled_net_return'] is None
    ledger.close()


def test_insufficient_displayed_depth_is_unknown_not_an_assumed_fill(tmp_path):
    ledger, clock, batch_id = ready_trial(tmp_path)
    ledger.decide(batch_id, 1961000)
    ledger.quote(quote(1963000, size=.0001), 1963000)
    ledger.advance(1963000)
    assert ledger.db.execute('SELECT status FROM outcomes').fetchone()[0] == 'entry_depth_unknown'
    assert ledger.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] == 0
    ledger.close()


def test_source_gap_blocks_a_decision_even_with_positive_observed_flow(tmp_path):
    ledger, clock, _ = ready_trial(tmp_path)
    later = 2500000
    batch_id = batch(ledger, later, [fill(), fill(2, 1230000), fill(3, 2490000)])
    clock[0] = 2501
    decision = ledger.decide(batch_id, 2501000)
    body = json.loads(ledger.db.execute('SELECT payload FROM decisions WHERE id=?', (decision,)).fetchone()[0])
    assert 'source_poll_gap' in body['reasons']
    assert body['strategy_long'] is None
    ledger.close()


def test_append_only_protocol_and_records_survive_restart(tmp_path):
    ledger, clock = make_ledger(tmp_path)
    batch(ledger, 1000000)
    definition = (ledger.directory / 'protocol.json').read_bytes()
    for operation in ['UPDATE fills SET first_observed_ms=5', 'DELETE FROM fills',
                      "INSERT OR REPLACE INTO fills SELECT wallet,coin,trade_id,event_ms,5,eligibility,semantic_sha,payload FROM fills"]:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            ledger.db.execute(operation)
        ledger.db.rollback()
    spec = dict(ledger.spec)
    with pytest.raises(TypeError):
        ledger.spec['fee_per_side_bps'] = 9
    ledger.close()
    resumed = SignalLedger(tmp_path / 'trial', COHORTS, clock=lambda: clock[0])
    assert resumed.db.execute('SELECT COUNT(*) FROM fills').fetchone()[0] == 1
    resumed.close()
    spec['fee_per_side_bps'] = 9
    with pytest.raises(ValueError, match='definition differs'):
        SignalLedger(tmp_path / 'trial', COHORTS, spec=spec, clock=lambda: clock[0])
    assert (tmp_path / 'trial/protocol.json').read_bytes() == definition


def test_capped_recent_history_must_reach_the_feature_window(tmp_path):
    ledger, clock, _ = ready_trial(tmp_path)
    body = [fill(100 + i, 1950000) for i in range(2000)]
    batch_id = batch(ledger, 2200000, body)
    features = ledger.source_features(batch_id, 2201000)
    assert 'capped_history_does_not_reach_window_start' in features['reasons']
    assert features['net_btc_base'] is None
    ledger.close()


def test_late_new_fills_cannot_be_used_as_timely_flow(tmp_path):
    ledger, clock, _ = ready_trial(tmp_path)
    batch_id = batch(ledger, 2200000, [fill(), fill(2, 1230000), fill(3, 1800000)])
    features = ledger.source_features(batch_id, 2201000)
    assert 'late_slice_in_window' in features['reasons']
    assert features['net_btc_base'] is None
    ledger.close()


def test_future_slices_quotes_and_decision_backfill_are_rejected(tmp_path):
    ledger, clock = make_ledger(tmp_path)
    with pytest.raises(ValueError):
        slices([fill(event=1002000)], 1001000)
    assert not ledger.quote(quote(1002000), 1001000)
    assert not ledger.quote(quote(900000), 1001000)
    batch_id = batch(ledger, 1000000)
    with pytest.raises(ValueError, match='actual current clock'):
        ledger.decide(batch_id, 900000)
    ledger.close()


def test_absent_audit_never_creates_a_fake_journal(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        audit(tmp_path / 'missing')
    assert not (tmp_path / 'missing/ledger.sqlite3').exists()
