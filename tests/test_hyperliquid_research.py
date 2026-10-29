from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from collectors.hyperliquid_research import PublicHyperliquid
from collectors.venom_collector import VenomCollector
from research.hyperliquid_features import market_contexts, sampled_positions, realized_liquidation_fills, signed_inventory_change


WALLET = "0x" + "1" * 40
COUNTERPARTY = "0x" + "2" * 40
CONTEXT = [{"universe": [{"name": "BTC"}]}, [{"markPx": "80000", "openInterest": "40000",
             "funding": "0.0000125", "dayNtlVlm": "1000000000"}]]


def state(size="1", liquidation="79200", event=100000):
    return {"wallet": WALLET, "observed_ms": 100001, "state": {"time": event, "assetPositions": [
        {"position": {"coin": "BTC", "szi": size, "entryPx": "80000", "liquidationPx": liquidation,
                      "positionValue": "80000", "leverage": {"type": "cross", "value": 10}}}]}}


def fill(side="A", victim=WALLET, tid=1):
    return {"coin": "BTC", "tid": tid, "time": 100000, "side": side, "px": "80000", "sz": "1",
            "dir": "Close Long", "liquidation": {"liquidatedUser": victim, "method": "market"}}


def test_hyperliquid_units_and_hourly_apr():
    btc = market_contexts(CONTEXT)["BTC"]
    assert btc["open_interest_usd"] == 3_200_000_000.
    assert btc["open_interest_base"] == 40000.
    assert btc["funding_simple_apr_pct"] == pytest.approx(10.95)


def test_venom_includes_btc_in_dollar_open_interest_filter():
    collector = VenomCollector({})
    response = Mock(status_code=200)
    response.json.return_value = deepcopy(CONTEXT)
    with patch("requests.post", return_value=response):
        result = collector._fetch_hyperliquid()["venom_hyperliquid"]
    assert result[0]["symbol"] == "BTC"
    assert result[0]["oi_m"] == 3200.
    assert result[0]["funding_annualized"] == pytest.approx(10.9, abs=.1)


def test_venom_health_does_not_hide_connectivity_failure():
    import requests
    with patch("requests.post", side_effect=requests.ConnectionError("unavailable")):
        assert VenomCollector({}).health_check() is False


def test_metadata_context_alignment_is_required():
    with pytest.raises(ValueError):market_contexts([{"universe": []}, CONTEXT[1]])


def test_nonfinite_market_values_are_rejected():
    data = deepcopy(CONTEXT)
    data[1][0]["openInterest"] = "nan"
    with pytest.raises(ValueError):market_contexts(data)


def test_long_liquidation_band_is_not_an_actual_liquidation():
    result = sampled_positions([state()], market_contexts(CONTEXT), 100010)
    assert result["sampled_assets"]["BTC"]["sampled_long_inventory_within_0.010"] == 80000.
    assert result["positions"][0]["liquidation_level_status"] == "estimated"
    assert realized_liquidation_fills([(WALLET, [], 100010)]) == []


def test_missing_liquidation_level_is_unknown_not_safe():
    result = sampled_positions([state(liquidation=None)], market_contexts(CONTEXT), 100010)
    assert result["sampled_assets"]["BTC"]["missing_liquidation_levels"] == 1
    assert result["sampled_assets"]["BTC"]["sampled_long_inventory_within_0.010"] == 0.


def test_wrong_side_liquidation_level_is_excluded():
    result = sampled_positions([state(liquidation="81000")], market_contexts(CONTEXT), 100010)
    assert result["positions"][0]["liquidation_distance_fraction"] is None


def test_stale_or_future_observation_cannot_be_current_wallet_data():
    assert sampled_positions([state()], market_contexts(CONTEXT), 300000)["successful_wallets"] == 0
    assert sampled_positions([state()], market_contexts(CONTEXT), 99999)["successful_wallets"] == 0


def test_failed_wallet_is_not_interpreted_as_flat():
    before = sampled_positions([state()], market_contexts(CONTEXT), 100010)
    missing = sampled_positions([{"wallet": WALLET, "observed_ms": 100010, "state": None}], market_contexts(CONTEXT), 100020)
    assert signed_inventory_change(before, missing) is None


def test_price_change_is_not_wallet_buying():
    before = sampled_positions([state()], market_contexts(CONTEXT), 100010)
    higher = deepcopy(CONTEXT)
    higher[1][0]["markPx"] = "90000"
    after = sampled_positions([state()], market_contexts(higher), 100020)
    assert signed_inventory_change(before, after)["BTC"] == 0.


def test_successful_empty_account_is_a_real_position_close():
    before = sampled_positions([state()], market_contexts(CONTEXT), 100010)
    closed = state()
    closed["state"]["assetPositions"] = []
    after = sampled_positions([closed], market_contexts(CONTEXT), 100020)
    assert signed_inventory_change(before, after)["BTC"] == -1.


def test_different_wallet_cohort_cannot_be_compared():
    before = sampled_positions([state()], market_contexts(CONTEXT), 100010)
    changed = state()
    changed["wallet"] = COUNTERPARTY
    after = sampled_positions([changed], market_contexts(CONTEXT), 100020)
    assert signed_inventory_change(before, after) is None


def test_liquidation_counterparty_is_not_the_victim_direction():
    victim_event = realized_liquidation_fills([(WALLET, [fill(side="A")], 100010)])[0]
    counter_event = realized_liquidation_fills([(COUNTERPARTY, [fill(side="B")], 100010)])[0]
    assert victim_event["liquidated_position_side"] == counter_event["liquidated_position_side"] == "long"


def test_two_sides_of_same_liquidation_trade_are_counted_once():
    events = realized_liquidation_fills([(WALLET, [fill(side="A")], 100010),
                                       (COUNTERPARTY, [fill(side="B")], 100020)])
    assert len(events) == 1
    assert events[0]["fill_notional_usd"] == 80000.
    assert events[0]["observed_ms"] == 100010


def test_generic_close_without_liquidation_evidence_is_not_forced_flow():
    row = fill()
    row.pop("liquidation")
    assert realized_liquidation_fills([(WALLET, [row], 100010)]) == []


def test_only_public_queries_are_available():
    client = PublicHyperliquid(session=Mock())
    for kind in ["order", "withdraw3", "approveAgent"]:
        with pytest.raises(ValueError):client.query(kind)
    client.session.post.assert_not_called()
    with pytest.raises(ValueError):client.query("metaAndAssetCtxs", type="order")


def test_unobserved_future_fill_is_rejected():
    with pytest.raises(ValueError):realized_liquidation_fills([(WALLET, [fill()], 99999)])


def test_concentration_is_reported_separately_from_account_count():
    first = state(size="9")
    second = state(size="1")
    second["wallet"] = COUNTERPARTY
    result = sampled_positions([first, second], market_contexts(CONTEXT), 100010)
    btc = result["sampled_assets"]["BTC"]
    assert btc["sampled_accounts_with_position"] == 2
    assert btc["largest_account_gross_share"] == pytest.approx(.9)
    assert btc["effective_account_count_by_gross"] < 1.3
