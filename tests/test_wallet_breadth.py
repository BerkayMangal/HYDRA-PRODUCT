"""Sampling and concentration controls must not manufacture independent evidence."""

from copy import deepcopy
from unittest.mock import patch
import json
import sqlite3

import pytest

from research.wallet_breadth import btc_wallet_change
from research.lab.discover_cohort import discover, select_accounts, stratum
from research.lab.hyperliquid_shadow import validate_cohort
from research.lab.breadth_report import analyze
from research.lab.event_replay import PriceTape
from research.lab.report import generate
from research.hyperliquid_features import SPECIFICATION_SHA256


def wallet(index):
    return "0x" + f"{index:040x}"


def feature(sizes):
    return {
        "wallets": [wallet(i) for i in range(len(sizes))],
        "unavailable_wallets": [],
        "positions": [
            {
                "coin": "BTC",
                "wallet": wallet(i),
                "signed_base_size": size,
                "notional_usd": abs(size) * 80000,
            }
            for i, size in enumerate(sizes)
            if size
        ],
    }


def observation(index, size, pnl="0"):
    return {
        "wallet": wallet(index),
        "observed_ms": 100001,
        "state": {
            "time": 100000,
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": str(size), "unrealizedPnl": pnl}}
            ],
        },
    }


def test_one_large_buyer_is_not_a_broad_account_consensus():
    result = btc_wallet_change(feature([1, 1, 1]), feature([11, 0.5, 0.5]))
    assert result["sampled_net_base_change"] == 9
    assert result["equal_account_direction_balance"] == pytest.approx(-1 / 3)
    assert result["aggregate_direction_survives_every_single_account_removal"] is False
    assert result["net_without_largest_current_btc_account"] == -1


def test_consistent_breadth_can_survive_every_single_account_removal():
    result = btc_wallet_change(feature([1, 1, 1]), feature([2, 2, 2]))
    assert result["accounts_buying"] == 3
    assert result["aggregate_direction_survives_every_single_account_removal"] is True


def test_price_appreciation_is_not_extra_wallet_buying():
    before = feature([1, 2])
    after = deepcopy(before)
    for row in after["positions"]:
        row["notional_usd"] *= 2
    result = btc_wallet_change(before, after)
    assert result["sampled_net_base_change"] == 0
    assert result["accounts_unchanged"] == 2
    assert result["largest_absolute_change_share"] is None


def test_missing_wallet_or_changed_cohort_abstains():
    before = feature([1, 1])
    after = feature([2, 1])
    after["unavailable_wallets"] = [wallet(0)]
    assert btc_wallet_change(before, after) is None
    assert btc_wallet_change(before, feature([2])) is None


def test_successfully_flat_account_is_a_position_close():
    result = btc_wallet_change(feature([1, 1]), feature([0, 1]))
    assert result["accounts_selling"] == 1
    assert result["sampled_net_base_change"] == -1


def test_single_account_is_not_robust_to_its_own_removal():
    result = btc_wallet_change(feature([1]), feature([2]))
    assert result["aggregate_direction_survives_every_single_account_removal"] is False


def test_nonfinite_account_size_is_not_valid_breadth():
    after = feature([1])
    after["positions"][0]["signed_base_size"] = float("nan")
    with pytest.raises(ValueError):
        btc_wallet_change(feature([1]), after)


def test_discovery_ignores_pnl_and_retains_lexicographic_stratum_caps():
    original = [observation(i, 0.5, pnl=str(i)) for i in range(12)]
    reordered = deepcopy(original[::-1])
    for item in reordered:
        item["state"]["assetPositions"][0]["position"]["unrealizedPnl"] = "999999999"
    first, buckets = select_accounts(original, 80000.0, set())
    second, _ = select_accounts(reordered, 80000.0, set())
    assert first == second == [wallet(i) for i in range(6)]
    assert len(buckets[0]) == 6


def test_stratum_boundaries_and_short_notional_use_correct_units():
    assert stratum(9999.0) is None
    assert stratum(10000.0) == 0
    assert stratum(100000.0) == 1
    assert stratum(1000000.0) == 2
    selected, _ = select_accounts([observation(1, -2)], 80000.0, set())
    assert selected == [wallet(1)]


def test_original_accounts_and_stale_state_cannot_sneak_into_new_sample():
    stale = observation(2, 2)
    stale["observed_ms"] = 300001
    selected, _ = select_accounts([observation(1, 2), stale], 80000.0, {wallet(1)})
    assert selected == []


def test_declaration_is_exclusive_before_any_new_discovery_calls(tmp_path):
    (tmp_path / "discovery-spec.json").write_text("already declared")
    with patch("research.lab.discover_cohort.PublicHyperliquid") as client:
        with pytest.raises(FileExistsError):
            discover(tmp_path, tmp_path / "no-original.json")
    client.assert_not_called()


def test_wider_cohort_requires_a_slower_public_api_budget():
    cohort = {"wallets": [wallet(i) for i in range(16)]}
    with pytest.raises(ValueError, match="240"):
        validate_cohort(cohort, 120)
    validate_cohort(cohort, 240)


def test_duplicate_addresses_cannot_inflate_account_count():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cohort({"wallets": [wallet(1), wallet(1)]}, 240)


def test_breadth_replay_pairs_recorded_prior_snapshot_and_abstains_zero_breadth():
    before, after = feature([1, 1]), feature([2, 0.5])
    before.update(
        sampled_inventory_change=None, inventory_change_actual_window_minutes=None
    )
    after.update(
        sampled_inventory_change={"BTC": 0.5},
        inventory_change_actual_window_minutes=15.0,
    )
    report = analyze([(1000, before), (901000, after)], PriceTape([]))
    assert report["usable_quarter_hours"] == 1
    assert report["variants"]["aggregate"]["directional_observations"] == 1
    assert report["variants"]["equal_account_breadth"]["directional_observations"] == 0
    assert report["variants"]["robust_aggregate"]["directional_observations"] == 0


def journal(path, wallets):
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE health(observed_ms INTEGER,payload TEXT); CREATE TABLE features(observed_ms INTEGER,spec_hash TEXT,payload TEXT); CREATE TABLE raw_snapshots(id INTEGER); CREATE TABLE fills(coin TEXT); CREATE TABLE liquidations(coin TEXT);"
        )
        payload = {
            "wallets": wallets,
            "unavailable_wallets": [],
            "requested_wallets": len(wallets),
            "successful_wallets": len(wallets),
            "positions": [],
            "sampled_assets": {},
            "scope": "test frozen sample",
        }
        db.execute(
            "INSERT INTO features VALUES(?,?,?)",
            (1000, SPECIFICATION_SHA256, json.dumps(payload)),
        )


def test_reports_deduplicate_overlapping_accounts_without_summing_liquidation_samples(
    tmp_path,
):
    primary, extra = tmp_path / "primary.sqlite3", tmp_path / "extra.sqlite3"
    journal(primary, [wallet(1), wallet(2)])
    journal(extra, [wallet(2), wallet(3)])
    report = generate(
        tmp_path / "missing-okx",
        primary,
        tmp_path / "results",
        tmp_path / "report",
        [extra, extra],
    )
    assert report["wallet_sample_coverage"]["distinct_public_accounts"] == 3
    assert report["wallet_sample_coverage"]["cohort_count"] == 2
    assert report["wallet_sample_coverage"]["market_wide"] is False
    assert len(report["additional_wallet_samples"]) == 1


def test_corrupt_additional_journal_does_not_hide_primary_sample_or_look_successful(
    tmp_path,
):
    primary, extra = tmp_path / "primary.sqlite3", tmp_path / "bad.sqlite3"
    journal(primary, [wallet(1)])
    extra.write_text("not a database")
    report = generate(
        tmp_path / "missing-okx",
        primary,
        tmp_path / "results",
        tmp_path / "report",
        [extra],
    )
    assert report["hyperliquid"]["requested_wallets"] == 1
    assert report["additional_wallet_errors"]
    assert report["wallet_sample_coverage"]["cohort_count"] == 1


def test_flat_total_inventory_has_no_fragile_direction_to_count():
    before, after = feature([1, 1]), feature([2, 0])
    before.update(
        sampled_inventory_change=None, inventory_change_actual_window_minutes=None
    )
    after.update(
        sampled_inventory_change={"BTC": 0.0},
        inventory_change_actual_window_minutes=15.0,
    )
    report = analyze([(1000, before), (901000, after)], PriceTape([]))
    assert report["aggregate_direction_fragile_quarter_hours"] == 0
    assert report["variants"]["aggregate"]["directional_observations"] == 0
