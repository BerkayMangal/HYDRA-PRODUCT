"""Costs cannot turn missing references into returns or a winning strategy."""

from copy import deepcopy
import json

import pytest
from fastapi.testclient import TestClient

from dashboard import app
from research.cost_scenarios import cost_scenario
from research.dashboard_summary import load_summary


def report():
    return {
        'schema_version': 1,
        'generated_at': '2026-10-30T05:00:00+00:00',
        'status': 'research_only_alpha_unproven',
        'orders_enabled': False,
        'collection_state': 'recorded_snapshot',
        'missing_inputs': [],
        'hyperliquid': {'requested_wallets': 6},
        'receipt_replay': {'inventory': {'all_inventory_directions': {
            '15': {'observed_groups': 4, 'nonoverlapping_intervals': 2,
                   'mean_signed_gross_price_move': .003},
            '60': {'observed_groups': 0, 'nonoverlapping_intervals': 0,
                   'mean_signed_gross_price_move': None},
            '240': {'observed_groups': 2, 'nonoverlapping_intervals': 1,
                    'mean_signed_gross_price_move': -.001},
        }, 'rows': [{'raw_wallet_details': 'not_exported'}]}},
    }


def setup_report(tmp_path, monkeypatch, body=None):
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(body or report()))
    monkeypatch.setenv('HYDRA_RESEARCH_REPORT', str(path))
    return path


def test_cost_units_charge_both_sides_and_funding_only_once(tmp_path, monkeypatch):
    setup_report(tmp_path, monkeypatch)
    result = cost_scenario(load_summary(), 8, 4, 2)
    assert result['assumptions']['round_trip_hurdle_bps'] == 26
    assert result['rows'][0]['mean_signed_gross_move_bps'] == pytest.approx(30)
    assert result['rows'][0]['mean_move_less_assumed_cost_bps'] == pytest.approx(4)
    assert result['rows'][2]['mean_move_less_assumed_cost_bps'] == pytest.approx(-36)
    assert result['alpha_validated'] is False
    assert result['orders_enabled'] is False


def test_zero_observations_remain_unknown_after_costs(tmp_path, monkeypatch):
    setup_report(tmp_path, monkeypatch)
    result = cost_scenario(load_summary())
    row = result['rows'][1]
    assert row['mean_move_less_assumed_cost_bps'] is None
    assert row['status'] == 'unknown'


def test_cost_scenarios_do_not_mutate_report_or_share_user_inputs(tmp_path, monkeypatch):
    setup_report(tmp_path, monkeypatch)
    body = load_summary()
    before = deepcopy(body)
    cheap = cost_scenario(body, 1, 1, 0)
    expensive = cost_scenario(body, 10, 5, 0)
    assert cheap['rows'][0]['mean_move_less_assumed_cost_bps'] == pytest.approx(26)
    assert expensive['rows'][0]['mean_move_less_assumed_cost_bps'] == pytest.approx(0)
    assert body == before
    assert 'raw_wallet_details' not in json.dumps(body)


def test_unavailable_report_does_not_use_a_silent_cost_fallback():
    result = cost_scenario({'status': 'unavailable', 'orders_enabled': False})
    assert result['status'] == 'unavailable'
    assert result['rows'] == []


def test_invalid_costs_are_rejected():
    for value in [-1, 101, float('nan'), float('inf'), True, '8']:
        with pytest.raises(ValueError):
            cost_scenario({}, fee_per_side_bps=value)


def test_cost_endpoint_is_per_request_and_does_not_enable_orders(tmp_path, monkeypatch):
    setup_report(tmp_path, monkeypatch)
    with TestClient(app) as client:
        first = client.get('/api/research/costs', params={'fee_per_side_bps': 1})
        default = client.get('/api/research/costs')
    assert first.status_code == 200
    assert first.json()['assumptions']['round_trip_hurdle_bps'] == 10
    assert default.json()['assumptions']['round_trip_hurdle_bps'] == 24
    assert default.headers['cache-control'] == 'no-store'
    assert default.json()['alpha_validated'] is False


def test_invalid_public_query_cannot_produce_a_cost_result(tmp_path, monkeypatch):
    setup_report(tmp_path, monkeypatch)
    with TestClient(app) as client:
        for value in ['-1', '101', 'nan', 'inf']:
            assert client.get('/api/research/costs', params={'fee_per_side_bps': value}).status_code == 422


def test_nonfinite_recorded_movement_is_unavailable_not_zero(tmp_path, monkeypatch):
    body = report()
    body['receipt_replay']['inventory']['all_inventory_directions']['15']['mean_signed_gross_price_move'] = float('nan')
    setup_report(tmp_path, monkeypatch, body)
    response = TestClient(app).get('/api/research/costs')
    assert response.json()['status'] == 'unavailable'
    assert response.json()['rows'] == []


def test_recorded_native_twap_evidence_is_dated_and_not_a_live_program(tmp_path, monkeypatch):
    setup_report(tmp_path, monkeypatch)
    probe = TestClient(app).get('/api/research').json()['native_twap_probe']
    assert probe['status'] == 'recorded_native_probe'
    assert probe['successful_accounts'] == probe['requested_accounts'] == 22
    assert probe['accounts_with_btc_slices'] == 2
    assert probe['possibly_truncated_accounts'] == 2
    assert probe['completed_at'].startswith('2026-10-30T07:13:')
    assert probe['alpha_validated'] is False
    assert 'accounts' not in probe
