from unittest.mock import Mock

import pytest

from collectors.hyperliquid_research import PublicHyperliquid
from research.lab.twap_probe import btc_slices, run

WALLET = '0x' + '1' * 40


def fill():
    return {'twapId': 9, 'fill': {'coin': 'BTC', 'time': 1000, 'tid': 2,
            'side': 'B', 'px': '80000', 'sz': '0.1'}}


def test_official_twap_query_is_public_but_order_endpoint_stays_forbidden():
    session = Mock()
    session.post.return_value.json.return_value = []
    client = PublicHyperliquid(session)
    assert client.query('userTwapSliceFills', user=WALLET) == []
    assert session.post.call_args.kwargs['json']['type'] == 'userTwapSliceFills'
    with pytest.raises(ValueError):
        client.query('twapOrder', user=WALLET)


def test_old_twap_fills_cannot_be_claimed_as_new_orders_or_remaining_flow():
    rows = btc_slices([fill(), fill()], 5000)
    assert len(rows) == 1
    assert rows[0]['observed_ms'] == 5000
    assert rows[0]['observation_phase'] == 'history_at_first_observation'
    assert 'remaining_size' not in rows[0]
    with pytest.raises(ValueError):
        btc_slices([fill()], 999)


def test_probe_distinguishes_empty_history_from_api_failure_and_preserves_receipts(tmp_path):
    client = Mock()
    client.query.return_value = []
    cohorts = {'test': {'wallets': [WALLET]}}
    result = run(tmp_path / 'empty', lambda: client, cohorts)
    assert result['successful_accounts'] == 1
    assert result['accounts_with_btc_slices'] == 0
    client.query.side_effect = RuntimeError('down')
    failed = run(tmp_path / 'failed', lambda: client, cohorts)
    assert failed['successful_accounts'] == 0
    assert failed['accounts'][0]['status'] == 'unavailable'
    assert failed['orders_enabled'] is False
    with pytest.raises(FileExistsError):
        run(tmp_path / 'empty', lambda: client, cohorts)
