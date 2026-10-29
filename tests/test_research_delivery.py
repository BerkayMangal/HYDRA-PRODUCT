"""Regression checks for causal joins, incomplete evidence and bounded workers."""

from datetime import datetime, timedelta, timezone
import json
import sqlite3
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from collectors.macro_collector import MacroCollector
from collectors.okx_collector import OKXCollector
from dashboard import app
from features.unified_frame import UnifiedFrameBuilder
from research.dashboard_summary import load_summary
from research.lab.event_replay import PriceTape, liquidation_replay, outcomes, summarize
from research.lab.report import generate
from research.lab.runtime import validate_budget
from research.lab.restore import restore
from signals.event_calendar import EventCalendar


def test_stop_budget_must_be_explicit_future_and_bounded():
    assert validate_budget(True, None) is None
    now = datetime.now(timezone.utc)
    assert validate_budget(False, (now + timedelta(hours=1)).isoformat()) > now
    for value in (
        None,
        "2020-01-01T00:00:00+00:00",
        "2099-01-01T00:00:00+00:00",
        "2099-01-01T00:00:00",
    ):
        with pytest.raises(ValueError):
            validate_budget(False, value)


def test_price_entry_requires_both_receipt_and_source_after_decision():
    tape = PriceTape([(1010, 990, 100.0), (1020, 1015, 102.0), (1030, 1040, 110.0)])
    assert tape.at_or_after(1000)["price"] == 102.0
    assert tape.at_or_after(1025) is None
    assert tape.at_or_after(0, max_delay_ms=500) is None


def test_missing_future_prices_cannot_be_manufactured_from_an_old_tick():
    tape = PriceTape([(1000, 1000, 100.0), (901000, 901000, 101.0)])
    result = outcomes(tape, 1000, 1)
    assert result["horizons"]["15"]["signed_gross_price_move"] == pytest.approx(0.01)
    assert result["horizons"]["60"]["status"] == "missing_reference"


def test_liquidations_group_on_receipt_not_hindsight_source_time():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE liquidations(coin TEXT,event_ms INTEGER,first_observed_ms INTEGER,payload TEXT)"
    )
    for event, receipt, victim in [
        (61000, 121000, "a"),
        (62000, 125000, "b"),
        (63000, 181000, "a"),
        (240000, 220000, "bad"),
    ]:
        payload = {
            "event_ms": event,
            "victim": victim,
            "liquidated_position_side": "long",
            "fill_notional_usd": 100.0,
        }
        db.execute(
            "INSERT INTO liquidations VALUES('BTC',?,?,?)",
            (event, receipt, json.dumps(payload)),
        )
    with patch(
        "research.lab.event_replay.spot_window",
        return_value={"usable": True, "imbalance": 0.1},
    ) as flow:
        result = liquidation_replay(db, None, PriceTape([]), 60000)
    assert result["receipt_minute_side_groups"] == 2
    assert result["rows"][0]["decision_ms"] == 180000
    assert result["rows"][0]["fills_known_at_decision"] == 2
    assert result["rows"][1]["decision_ms"] == 240000
    assert result["rejected_precohort_or_future_events"] == 1
    assert flow.call_args_list[0].args[1] == 180000


def test_overlapping_price_horizons_are_not_counted_as_separate_intervals():
    tape = PriceTape(
        [(t, t, 100.0) for t in (1000, 61000, 901000, 961000, 3601000, 3661000)]
    )
    rows = [outcomes(tape, start, 1) for start in (1000, 61000)]
    result = summarize(rows)
    assert result["15"]["observed_groups"] == 2
    assert result["15"]["nonoverlapping_intervals"] == 1


def test_missing_journals_remain_unknown_and_reporter_does_not_create_them(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    report = generate(missing, missing, tmp_path / "results", tmp_path / "report")
    assert set(report["missing_inputs"]) == {
        "okx_journal",
        "hyperliquid_journal",
        "historical_search",
    }
    assert not missing.exists()
    assert report["orders_enabled"] is False


def test_corrupt_or_absent_report_does_not_advertise_valid_research(
    tmp_path, monkeypatch
):
    report = tmp_path / "summary.json"
    monkeypatch.setenv("HYDRA_RESEARCH_REPORT", str(report))
    assert load_summary()["status"] == "unavailable"
    report.write_text('{"schema_version":1,"orders_enabled":true}')
    assert load_summary()["status"] == "unavailable"
    report.write_text("{")
    response = TestClient(app).get("/api/research")
    assert response.status_code == 200
    assert response.json()["collection_state"] == "unknown"
    assert response.headers["cache-control"] == "no-store"


def test_yield_changes_are_percentage_points_and_prices_are_percent():
    assert MacroCollector._change("us10y", 4.1, 4.0) == pytest.approx(0.1)
    assert MacroCollector._change("qqq", 410.0, 400.0) == pytest.approx(2.5)


def test_adjacent_macro_polls_do_not_create_daily_changes():
    builder = UnifiedFrameBuilder()
    builder._pipeline.ingest_macro(
        {"ts": 1000, "qqq_current": 400.0, "us10y_current": 4.0}
    )
    builder._pipeline.ingest_macro(
        {"ts": 1060, "qqq_current": 410.0, "us10y_current": 4.1}
    )
    result = builder._compute_macro_regime_features(
        {"qqq_current": 410.0, "us10y_current": 4.1}
    )
    assert "qqq_ret_1d" not in result
    assert "us10y_change_1d" not in result


def test_daily_macro_change_uses_clock_time_instead_of_24_exchange_bars():
    collector = MacroCollector({})
    collector.symbols, collector.em_currencies = {"us10y": "^TNX"}, {}
    index = pd.DatetimeIndex(
        [
            f"2026-10-{day} {hour}:00:00+00:00"
            for day in (26, 27, 28)
            for hour in range(10, 17)
        ]
    )
    history = pd.DataFrame(
        {"Close": [4 + i * 0.01 for i in range(len(index))]}, index=index
    )
    ticker = Mock()
    ticker.history.return_value = history
    with patch("collectors.macro_collector.yf.Ticker", return_value=ticker):
        result = collector._fetch_market_data()
    assert result["us10y_change_24h"] == pytest.approx(0.07)
    assert collector._cache["us10y_current"][1] == index[-1].timestamp()


def test_fred_yield_fallback_has_same_units_and_does_not_refresh_source_date():
    collector = MacroCollector({})
    collector._fred_key = "test-unused"
    response = Mock(ok=True)
    response.json.return_value = {
        "observations": [
            {"date": "2026-10-28", "value": "4.1"},
            {"date": "2026-10-27", "value": "4.0"},
        ]
    }
    with patch("requests.get", return_value=response):
        result = collector._fetch_fred_fallback(["us10y"])
    assert result["us10y_change_24h"] == pytest.approx(0.1)
    assert (
        collector._cache["us10y_current"][1]
        == datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp()
    )


def test_bounded_trade_sample_cannot_impersonate_time_window_cvd():
    collector = OKXCollector({})
    collector.exchange.market = Mock(return_value={"contractSize": 0.01})
    collector.exchange.fetch_trades = Mock(
        return_value=[
            {"amount": 5.0, "side": "buy", "timestamp": 1000},
            {"amount": 2.0, "side": "sell", "timestamp": 2000},
        ]
    )
    result = collector._fetch_cvd_from_trades()
    assert result["perp_trade_sample_net_btc"] == pytest.approx(0.03)
    assert result["perp_trade_sample_complete_window"] is False
    assert not {"cvd_perp", "cvd_5m_delta", "cvd_buy_ratio"}.intersection(result)


def test_restore_preserves_a_different_existing_dataset_before_any_writes(tmp_path):
    target = tmp_path / "research"
    (target / "data").mkdir(parents=True)
    original = target / "data/btc-usdt-1h.csv"
    original.write_text("user-owned different dataset\n")
    with pytest.raises(FileExistsError):
        restore(target)
    assert original.read_text() == "user-owned different dataset\n"
    assert not (target / "results").exists()


def test_daily_rate_observations_cannot_be_promoted_to_confirmed_fomc_dates():
    calendar = EventCalendar(fred_api_key="test-unused")
    with patch("requests.get") as request:
        with pytest.raises(ValueError, match="daily rate series"):
            calendar._fetch_fred()
    request.assert_not_called()
    assert calendar._events
    assert all(event.confirmed is False for event in calendar._events)


def test_guessed_macro_calendar_does_not_publish_decision_grade_countdowns():
    result = MacroCollector({})._fetch_calendar_proximity()
    assert result["macro_calendar_confirmed"] is False
    assert "fomc_hours_until" not in result
