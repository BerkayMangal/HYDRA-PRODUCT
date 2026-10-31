from datetime import datetime, timezone
import json
from unittest.mock import Mock

from fastapi.testclient import TestClient

import dashboard
from collectors.hyperliquid_research import PublicHyperliquid
from research.forward_summary import load_forward_summary


def report(now=1000):
    return {'schema_version': 1, 'status': 'prospective_research_only', 'orders_enabled': False,
            'alpha_validated': False, 'protocol_sha256': 'a' * 64,
            'generated_at': datetime.fromtimestamp(now, timezone.utc).isoformat(),
            'runtime_state': 'collecting', 'last_valid_quote_ms': now * 1000 - 1000,
            'last_batch': {'observed_ms': now * 1000 - 10000, 'requested_accounts': 22,
                           'successful_accounts': 22, 'errors': []},
            'account_count': 22,
            'results': {name: {'completed_trades': 0, 'mean_modeled_net_return': None}
                        for name in ['strategy', 'control']}}


def save(tmp_path, monkeypatch, body):
    path = tmp_path / 'forward.json'
    path.write_text(json.dumps(body))
    monkeypatch.setenv('HYDRA_FORWARD_REPORT', str(path))
    return path


def test_report_health_preserves_recorded_time_and_unknown_returns(tmp_path, monkeypatch):
    body = report()
    path = save(tmp_path, monkeypatch, body)
    original = path.read_bytes()
    result = load_forward_summary(clock=lambda: 1000)
    assert result['source_current_healthy'] is True
    assert result['generated_at'] == body['generated_at']
    assert result['results']['strategy']['mean_modeled_net_return'] is None
    assert path.read_bytes() == original


def test_stale_stopped_and_partial_data_cannot_look_live(tmp_path, monkeypatch):
    body = report()
    save(tmp_path, monkeypatch, body)
    assert load_forward_summary(clock=lambda: 1100)['source_current_healthy'] is False
    body['runtime_state'] = 'stopped'
    save(tmp_path, monkeypatch, body)
    assert load_forward_summary(clock=lambda: 1000)['source_current_healthy'] is False
    body['runtime_state'] = 'collecting'
    body['last_batch']['successful_accounts'] = 21
    save(tmp_path, monkeypatch, body)
    assert load_forward_summary(clock=lambda: 1000)['source_current_healthy'] is False


def test_invalid_report_does_not_fall_back_to_a_valid_trial(tmp_path, monkeypatch):
    path = tmp_path / 'missing.json'
    monkeypatch.setenv('HYDRA_FORWARD_REPORT', str(path))
    assert load_forward_summary()['status'] == 'unavailable'
    for key, value in [('orders_enabled', True), ('alpha_validated', True),
                       ('generated_at', '2026-10-31T10:00:00'), ('protocol_sha256', 'bad')]:
        body = report()
        body[key] = value
        save(tmp_path, monkeypatch, body)
        assert load_forward_summary()['status'] == 'unavailable'
    body = report()
    body['results']['strategy']['mean_modeled_net_return'] = float('nan')
    save(tmp_path, monkeypatch, body)
    assert load_forward_summary()['status'] == 'unavailable'


def test_no_completed_trades_cannot_advertise_zero_return(tmp_path, monkeypatch):
    body = report()
    body['results']['strategy']['mean_modeled_net_return'] = 0
    save(tmp_path, monkeypatch, body)
    assert load_forward_summary()['status'] == 'unavailable'


def test_forward_api_reads_only_the_report_without_starting_collection(tmp_path, monkeypatch):
    save(tmp_path, monkeypatch, report(datetime.now(timezone.utc).timestamp()))
    query = Mock(side_effect=AssertionError('Page polling must not query upstream'))
    monkeypatch.setattr(PublicHyperliquid, 'query', query)
    with TestClient(dashboard.app) as client:
        result = client.get('/api/research/forward')
    assert result.status_code == 200
    assert result.headers['cache-control'] == 'no-store'
    assert result.json()['alpha_validated'] is False
    query.assert_not_called()


def test_offline_report_uses_recorded_time_and_never_writes_a_trade(tmp_path):
    from research.forward_ledger import SignalLedger
    from research.lab.forward_report import recorded_report
    wallet = '0x' + '1' * 40
    ledger = SignalLedger(tmp_path / 'trial', {'fixed': {'wallets': [wallet]}}, clock=lambda: 1000)
    ledger.run_event({'state': 'stopped'}, observed_ms=1001000)
    ledger.close()
    before = (tmp_path / 'trial/protocol.json').read_bytes()
    first = recorded_report(tmp_path / 'trial')
    second = recorded_report(tmp_path / 'trial')
    assert first == second
    assert first['runtime_state'] == 'recorded_snapshot'
    assert first['generated_at'] == datetime.fromtimestamp(1001, timezone.utc).isoformat()
    assert first['decisions'] == 0
    assert first['results']['strategy']['mean_modeled_net_return'] is None
    assert (tmp_path / 'trial/protocol.json').read_bytes() == before
