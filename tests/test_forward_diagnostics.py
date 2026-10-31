import pytest

from research.forward_diagnostics import concentration, diagnose_trial, load_diagnostics, trade_economics
from tests.test_forward_ledger import ready_trial, quote


def test_economics_decomposes_spread_and_assumed_costs_independently():
    entry = {'bid': 99, 'ask': 101, 'mid': 100}
    exit_quote = {'bid': 101, 'ask': 103, 'mid': 102}
    spec = {'fee_per_side_bps': 8, 'slippage_per_side_bps': 4}
    net = (1-.0004)*(1-.0008)/((1+.0004)*(1+.0008))-1
    result = trade_economics(entry, exit_quote, {'modeled_net_return': net}, spec)
    assert result['gross_mid_return_bps'] == pytest.approx(200)
    assert result['gross_bid_ask_return_bps'] == pytest.approx(0)
    assert result['spread_drag_bps'] == pytest.approx(200)
    assert result['assumed_fee_slippage_drag_bps'] == pytest.approx(-net*10000)
    assert result['max_break_even_fee_per_side_bps_at_assumed_slippage'] is None
    with pytest.raises(ValueError, match='independent cost'):
        trade_economics(entry, exit_quote, {'modeled_net_return': .5}, spec)


def test_activity_exposes_concentration_and_parent_ids_without_owner_claim():
    fills = [('a', {'twap_id': '1', 'side': 'B', 'price': 100, 'size': 9}),
             ('b', {'twap_id': '1', 'side': 'A', 'price': 100, 'size': 1})]
    result = concentration(fills)
    assert result['active_accounts'] == result['observed_parent_ids'] == 2
    assert result['largest_account_share'] == .9
    assert result['effective_account_count'] == pytest.approx(1 / (.9**2 + .1**2))
    assert result['net_base_without_largest_account'] == -1
    assert concentration(fills[:1])['net_base_without_largest_account'] == 0
    assert concentration([])['effective_account_count'] is None
    fills[0][1]['price'] = 1e308
    with pytest.raises(ValueError, match='Nonfinite'):
        concentration(fills)


def test_recorded_diagnostics_keep_flat_control_separate_and_do_not_settle(tmp_path):
    ledger, clock, batch_id = ready_trial(tmp_path)
    ledger.decide(batch_id, 1961000)
    before = ledger.db.total_changes
    pending = diagnose_trial(ledger.directory)
    assert pending['results']['strategy']['pending_active'] == 1
    assert pending['results']['control']['valid_flat_decisions'] == 1
    assert pending['results']['control']['mean']['modeled_net_return_bps'] is None
    assert ledger.db.total_changes == before
    ledger.quote(quote(1963000,100,102),1963000); ledger.advance(1963000)
    ledger.quote(quote(2864000,104,106),2864000); ledger.advance(2864000)
    result = diagnose_trial(ledger.directory)
    assert result['results']['strategy']['completed_trades'] == 1
    assert result['results']['control']['completed_trades'] == 0
    assert result['results']['control']['valid_flat_decisions'] == 1
    assert result['results']['strategy']['mean']['gross_bid_ask_return_bps'] == pytest.approx((104/102-1)*10000)
    assert result['recent_events'][-1]['economics']['strategy']['max_break_even_fee_per_side_bps_at_assumed_slippage'] > 0
    assert result['alpha_validated'] is False
    ledger.close()


def test_diagnostics_missing_or_invalid_trial_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv('HYDRA_FORWARD_TRIAL',raising=False)
    assert load_diagnostics()['status'] == 'not_configured'
    monkeypatch.setenv('HYDRA_FORWARD_TRIAL',str(tmp_path/'absent'))
    assert load_diagnostics()['status'] == 'unavailable'
    assert not (tmp_path/'absent').exists()
