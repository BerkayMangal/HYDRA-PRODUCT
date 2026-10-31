import pytest

from research.funding_feasibility import HOUR_MS, funding_screen


def history(rate='0.0000125'):
    return [{'coin':'BTC','time':index*HOUR_MS,'fundingRate':rate} for index in range(1,169)]


def test_hourly_funding_annualization_and_four_leg_cost_hurdle():
    result=funding_screen(history(),'BTC',1,168*HOUR_MS+1,168*HOUR_MS+2)
    assert result['observed_hourly_payments']==168
    assert result['simple_annualized_observed_funding_on_perp_notional']==pytest.approx(.1095)
    assert result['full_window_short_received_funding_bps']==pytest.approx(21)
    assert result['last_24h_short_received_funding_bps']==pytest.approx(3)
    assert result['assumed_entry_exit_cost_bps_before_unknown_costs']==41
    assert result['full_window_funding_less_assumed_cost_bps']==pytest.approx(-20)
    assert result['screen']=='funding_did_not_cover_assumed_costs'


def test_missing_hour_remains_unknown_and_negative_funding_is_paid_by_short():
    result=funding_screen(history()[1:],'BTC',1,168*HOUR_MS+1,168*HOUR_MS+2)
    assert result['full_window_short_received_funding_bps'] is None
    assert result['full_window_funding_less_assumed_cost_bps'] is None
    negative=funding_screen(history('-.0000125'),'BTC',1,168*HOUR_MS+1,168*HOUR_MS+2)
    assert negative['full_window_short_received_funding_bps']==pytest.approx(-21)
    assert negative['negative_payment_hours']==168


def test_duplicate_payment_not_double_counted_and_conflicts_rejected():
    body=history()
    assert funding_screen(body+[body[0]],'BTC',1,168*HOUR_MS+1,168*HOUR_MS+2)['observed_hourly_payments']==168
    with pytest.raises(ValueError,match='Conflicting'):
        funding_screen(body+[dict(body[0],fundingRate='.05')],'BTC',1,168*HOUR_MS+1,168*HOUR_MS+2)


def test_future_nonfinite_or_wrong_asset_cannot_pass_and_positive_screen_is_not_alpha():
    for change in [{'time':169*HOUR_MS},{'fundingRate':'nan'},{'coin':'ETH'}]:
        with pytest.raises(ValueError):
            funding_screen([dict(history()[0],**change)],'BTC',1,168*HOUR_MS+1,168*HOUR_MS+2)
    result=funding_screen(history('.0001'),'BTC',1,168*HOUR_MS+1,168*HOUR_MS+2)
    assert result['screen']=='requires_executable_basis_and_cost_verification'
    assert result['alpha_validated'] is result['orders_enabled'] is False
