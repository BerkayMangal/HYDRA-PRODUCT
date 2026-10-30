"""Deployment-facing wallet snapshots must stay separate from trading signals."""

from copy import deepcopy
import json
import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import dashboard
from services.wallet_monitor import WalletMonitor, frozen_cohorts

WALLETS = ["0x" + f"{i:040x}" for i in (1, 2)]
METADATA = [
    {"universe": [{"name": "BTC"}]},
    [
        {
            "markPx": "80000",
            "openInterest": "40000",
            "funding": "0.0000125",
            "dayNtlVlm": "1000000000",
        }
    ],
]


class Client:
    def __init__(self, states, fail_metadata=False):
        self.states, self.fail_metadata = states, fail_metadata

    def query(self, kind, **kwargs):
        if kind == "metaAndAssetCtxs":
            if self.fail_metadata:
                raise RuntimeError("source unavailable")
            return deepcopy(METADATA)
        assert kind == "clearinghouseState"
        value = self.states[kwargs["user"]]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


def setup_monitor(tmp_path, sizes=(1.0, -2.0)):
    clock = [1000.0]
    states = {}
    for wallet, size in zip(WALLETS, sizes):
        states[wallet] = {
            "time": 1000000,
            "assetPositions": (
                [
                    {
                        "position": {
                            "coin": "BTC",
                            "szi": str(size),
                            "entryPx": "80000",
                            "liquidationPx": "70000" if size > 0 else "90000",
                            "leverage": {"type": "cross"},
                        }
                    }
                ]
                if size
                else []
            ),
        }
    client = Client(states)
    cohorts = {
        "original": {
            "wallets": WALLETS,
            "scope": "test frozen sample",
            "frozen_at_ms": 1,
        }
    }
    monitor = WalletMonitor(
        cohorts,
        client_factory=lambda: client,
        database=tmp_path / "wallet.sqlite3",
        clock=lambda: clock[0],
    )
    return monitor, client, clock


def test_frozen_production_membership_is_22_unique_accounts():
    cohorts = frozen_cohorts()
    assert len(cohorts["original"]["wallets"]) == 6
    assert len(cohorts["additional"]["wallets"]) == 16
    assert (
        len({wallet for cohort in cohorts.values() for wallet in cohort["wallets"]})
        == 22
    )


def test_live_wallet_snapshot_uses_source_units_and_journals_raw_data(tmp_path):
    monitor, _, _ = setup_monitor(tmp_path)
    result = monitor.cycle()
    assert result["successful_accounts"] == 2
    assert result["status"] == "observed"
    assert result["orders_enabled"] is False
    assert result["journal_written"] is True
    assert result["btc_positions"][0]["notional_usd"] == 160000.0
    assert result["cold_start"] is True
    with sqlite3.connect(monitor.database) as db:
        raw = json.loads(
            db.execute("SELECT raw_json FROM wallet_observations").fetchone()[0]
        )
    assert len(raw["states"]) == 2


def test_successfully_flat_wallet_is_not_a_failed_read(tmp_path):
    monitor, _, _ = setup_monitor(tmp_path, sizes=(0.0, 0.0))
    result = monitor.cycle()
    assert result["successful_accounts"] == 2
    assert result["btc_positions"] == []
    assert result["errors"] == {}


def test_failed_wallet_stays_unknown_and_blocks_breadth(tmp_path):
    monitor, client, _ = setup_monitor(tmp_path)
    client.states[WALLETS[0]] = RuntimeError("down")
    result = monitor.cycle()
    assert result["status"] == "partial"
    assert result["successful_accounts"] == 1
    assert WALLETS[0] in result["errors"]
    assert result["cohorts"]["original"]["btc_inventory_change"] is None


def test_metadata_failure_does_not_republish_fresh_positions(tmp_path):
    monitor, client, _ = setup_monitor(tmp_path)
    monitor.cycle()
    client.fail_metadata = True
    result = monitor.cycle()
    assert result["status"] == "unavailable"
    assert result["btc_positions"] == []
    assert result["is_stale"] is True


def test_wallet_cache_expiry_does_not_refresh_its_observation_time(tmp_path):
    monitor, _, clock = setup_monitor(tmp_path)
    monitor.cycle()
    clock[0] += 361
    result = monitor.get_snapshot()
    assert result["status"] == "stale"
    assert result["observed_ms"] == 1000000


def test_inventory_change_only_appears_after_matching_history(tmp_path):
    monitor, client, clock = setup_monitor(tmp_path)
    monitor.cycle()
    clock[0] += 900
    for state in client.states.values():
        state["time"] = 1900000
    client.states[WALLETS[0]]["assetPositions"][0]["position"]["szi"] = "2"
    result = monitor.cycle()
    change = result["cohorts"]["original"]["btc_inventory_change"]
    assert change["sampled_net_base_change"] == 1.0
    assert result["cold_start"] is False
    assert result["cohorts"]["original"]["change_window_minutes"] == 15.0


def test_snapshot_cannot_be_mutated_by_an_api_reader(tmp_path):
    monitor, _, _ = setup_monitor(tmp_path)
    first = monitor.cycle()
    first["btc_positions"].clear()
    assert len(monitor.get_snapshot()["btc_positions"]) == 2


def test_journal_failure_is_visible_without_faking_complete_health(tmp_path):
    monitor, _, _ = setup_monitor(tmp_path)
    monitor.database = tmp_path
    result = monitor.cycle()
    assert result["status"] == "partial"
    assert result["journal_written"] is False
    assert "journal" in result["errors"]


def test_public_dashboard_has_release_wallets_and_reviewed_sources(
    tmp_path, monkeypatch
):
    monitor, _, _ = setup_monitor(tmp_path)
    monitor.cycle()
    monkeypatch.setattr(dashboard, "_wallet_monitor", monitor)
    with TestClient(dashboard.app) as client:
        version = client.get("/api/version").json()
        assert version["release_id"] == "hydra-research-20261030.1"
        assert version["orders_enabled"] is False
        wallets = client.get("/api/wallets").json()
        assert wallets["successful_accounts"] == 2
        tools = client.get("/api/research/tools").json()
        assert len(tools["entries"]) == 19
        assert tools["service_trials_verified"] is False


def test_research_fill_window_covers_the_slower_poll_interval(tmp_path, monkeypatch):
    from research.lab import hyperliquid_shadow as worker

    monkeypatch.setattr(worker, "ROOT", tmp_path)
    db = worker.connect()
    calls = []

    class FillClient:
        def query(self, kind, **kwargs):
            if kind == "metaAndAssetCtxs":
                return METADATA
            if kind == "clearinghouseState":
                return {"time": 1000000, "assetPositions": []}
            calls.append(kwargs)
            return []

    cohort = {"wallets": [WALLETS[0]], "scope": "test", "frozen_at_ms": 0}
    with patch("research.lab.hyperliquid_shadow.time.time", return_value=1000.0), patch(
        "research.lab.hyperliquid_shadow.time.sleep"
    ):
        result = worker.cycle(db, FillClient(), cohort, lookback_seconds=360)
    assert calls[0]["endTime"] - calls[0]["startTime"] == 360000
    assert result["fills_requested_lookback_seconds"] == 360
    assert db.execute("SELECT COUNT(*) FROM fill_requests").fetchone()[0] == 1
    db.close()


def test_unrequested_fill_interval_is_exposed_as_a_source_gap(tmp_path, monkeypatch):
    from research.lab import hyperliquid_shadow as worker

    monkeypatch.setattr(worker, "ROOT", tmp_path)
    db = worker.connect()
    db.execute(
        "INSERT INTO fill_requests VALUES(?,?,?,?,?,?)",
        (WALLETS[0], 700000, 520000, 700000, 0, 0),
    )

    class FillClient:
        def query(self, kind, **kwargs):
            if kind == "metaAndAssetCtxs":
                return METADATA
            if kind == "clearinghouseState":
                return {"time": 1000000, "assetPositions": []}
            return []

    with patch("research.lab.hyperliquid_shadow.time.time", return_value=1000.0), patch(
        "research.lab.hyperliquid_shadow.time.sleep"
    ):
        result = worker.cycle(
            db,
            FillClient(),
            {"wallets": [WALLETS[0]], "scope": "test", "frozen_at_ms": 0},
        )
    assert "fill_window_gap:" + WALLETS[0] in result["errors"]
    db.close()
