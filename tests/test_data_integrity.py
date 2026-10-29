"""Regression checks for observed data corruption and unsafe freshness paths."""
from copy import deepcopy
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from collectors.base import BaseCollector
from collectors.etf_collector import ETFFlowCollector, parse_flows
from collectors.macro_collector import MacroCollector
from collectors.okx_collector import OKXCollector
from collectors.unified import UnifiedDataStore
from dashboard import DashboardState
from features.pipeline import FeaturePipeline
from features.quality import FeatureStatus, FreshnessTracker
from features.registry import ALL_CONTRACTS
from features.unified_frame import UnifiedFrameBuilder


class FailedCollector(BaseCollector):
    def fetch(self):
        return {}

    def health_check(self):
        return False


def test_okx_honors_environment_proxy_with_tls_verification():
    exchange = OKXCollector({}).exchange
    assert exchange.session.trust_env is True
    assert exchange.session.verify is True


def test_okx_price_freshness_uses_exchange_event_not_receipt_time():
    c = OKXCollector({})
    times = c.observation_times({"last_price": 100., "last_price_event_ms": 1000000}, 2000.)
    assert times["last_price"] == 1000.
    assert c.observation_times({"last_price": 100.}, 2000.)["last_price"] is None


def test_liquidation_family_window_and_contract_notional():
    collector = OKXCollector({})
    collector.exchange.market = MagicMock(return_value={"contractSize": .01})
    details = [{"side": "sell", "sz": "2", "bkPx": "100", "ts": "100000000"},
               {"side": "buy", "sz": "3", "bkPx": "100", "ts": "100000000"},
               {"side": "sell", "sz": "9999", "bkPx": "100", "ts": "1"}]
    response = {"data": [{"instId": "BTC-USDT-SWAP", "details": details},
                         {"instId": "BTC-USDT-OTHER", "details": details}]}
    with patch.object(collector, "_api_get", return_value=response) as request, patch("time.time", return_value=100000):
        result = collector._fetch_liquidations()
    assert request.call_args.kwargs["params"]["instFamily"] == "BTC-USDT"
    assert result["liq_total"] == 5.
    assert result["liq_long_vol"] == 2.
    assert result["liq_sample_only"] is True


def test_failure_does_not_republish_cached_values():
    c = FailedCollector("test", {})
    c.max_retries = 1
    c.last_data = {"last_price": 123.}
    c.last_fetch_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    c.field_timestamps = {"last_price": 42.}
    assert c.safe_fetch() == {}
    assert c.consecutive_errors == 1
    assert c.field_timestamps == {"last_price": 42.}
    assert c.last_fetch_time.year == 2025


def test_macro_cache_preserves_original_observation_time():
    c = MacroCollector({})
    c._cache["vix_current"] = (20., 42.)
    assert c.observation_times({"vix_current": 20.}, 1000.)["vix_current"] == 42.


def test_normalization_never_changes_shared_market_facts():
    raw = {"timestamp": 1760000400000, "cvd_perp": 20., "total_stablecoin_mcap": 303_000_000_000.}
    original = deepcopy(raw)
    store = UnifiedDataStore({})
    with patch("time.time", return_value=10000):
        store.update(raw, {k: 8000. for k in raw})
        store.update(raw, {k: 8000. for k in raw})
    assert raw == original


def test_repeated_observation_is_not_another_normalization_sample():
    store = UnifiedDataStore({})
    with patch("time.time", return_value=10000):
        for _ in range(25):
            store.update({"cvd_perp": 5.}, {"cvd_perp": 9999.})
    assert len(store._observations["cvd_perp"]) == 1
    assert len(store.history) == 1


def test_expired_value_cannot_return_as_a_zscore():
    store = UnifiedDataStore({})
    with patch("time.time", return_value=10000):
        for i in range(25):
            store.update({"cvd_perp": float(i)}, {"cvd_perp": 9900. + i})
    with patch("time.time", return_value=20000):
        result = store.update({"cvd_perp": 24.}, {"cvd_perp": 9924.})
    assert pd.isna(result["cvd_perp"])
    assert "cvd_perp_zscore" not in result


def test_zscore_retains_source_time():
    with patch("time.time", return_value=10000):
        result = UnifiedDataStore({}).update({"cvd_perp": 5.}, {"cvd_perp": 9900.})
    assert result.attrs["source_timestamps"]["cvd_perp_zscore"] == 9900.


def test_frame_rebuild_does_not_rejuvenate_source_or_bypass_expiry():
    features = pd.Series({"cvd_perp_zscore": 2.})
    features.attrs["source_timestamps"] = {"cvd_perp_zscore": 1000.}
    builder = UnifiedFrameBuilder()
    with patch("time.time", return_value=1001):
        builder.build(features, {})
    with patch("time.time", return_value=10000):
        frame = builder.build(features, {})
    report = frame.quality.feature_reports["cvd_perp_zscore"]
    assert report.status == FeatureStatus.MISSING
    assert report.age_seconds == 9000.
    assert "cvd_perp_zscore" not in frame.to_legacy_series()


def test_numeric_value_without_provenance_is_missing():
    frame = UnifiedFrameBuilder().build(pd.Series({"cvd_perp_zscore": 2.}), {})
    assert frame.quality.feature_reports["cvd_perp_zscore"].status == FeatureStatus.MISSING


def test_critical_price_failure_forces_red_even_with_other_data():
    builder = UnifiedFrameBuilder()
    with patch("time.time", return_value=10000):
        values = pd.Series({name: contract.neutral_value for name, contract in ALL_CONTRACTS.items()})
        values.attrs["source_timestamps"] = {name: 10000. for name in values.index}
        frame = builder.build(values, {})
    assert frame.quality.circuit_breaker_level == "RED"
    assert "btc_price" in frame.quality.abstain_reason


def test_zero_and_future_source_times_are_not_replaced_with_now():
    tracker = FreshnessTracker()
    with patch("time.time", return_value=100):
        tracker.mark_updated("cvd_perp_zscore", 0.)
        assert tracker.get_age("cvd_perp_zscore") == 100.
        tracker.mark_updated("cvd_perp_zscore", 1000.)
        assert tracker.classify(ALL_CONTRACTS["cvd_perp_zscore"], 1.).status == FeatureStatus.MISSING


def candle(ts, volume=10.):
    return {"timestamp": ts, "open": 100., "high": 102., "low": 99., "close": 101., "volume": volume}


def test_duplicate_five_minute_candles_do_not_multiply_volume():
    pipe = FeaturePipeline()
    start = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
    for i in range(12):
        for _ in range(3):
            pipe.ingest_candle(candle(start + i * 300000))
    assert len(pipe._ring) == 1
    assert pipe._ring[-1]["volume"] == 120.


def test_incomplete_hour_does_not_emit_a_valid_bar():
    pipe = FeaturePipeline()
    start = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
    for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        pipe.ingest_candle(candle(start + i * 300000))
    assert not pipe._ring


def test_candle_without_event_timestamp_is_rejected():
    pipe = FeaturePipeline()
    assert not pipe.ingest_candle({"close": 100., "volume": 10.})
    assert not pipe._accumulator._pending


def test_authoritative_hourly_bars_are_deduplicated_and_open_bar_excluded():
    pipe = FeaturePipeline()
    start = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
    closed = [start, 100., 102., 99., 101., 120.]
    with patch("time.time", return_value=start / 1000 + 3700):
        assert pipe.ingest_candle({"hourly_candles": [closed, closed, [start + 3600000, 100., 102., 99., 101., 20.]]})
        assert not pipe.ingest_candle({"hourly_candles": [closed]})
    assert len(pipe._ring) == 1
    assert pipe._ring[-1]["volume"] == 120.


def test_hourly_gap_does_not_create_returns_across_missing_hours():
    pipe = FeaturePipeline()
    start = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
    pipe.ingest_candle({"hourly_candles": [[start, 100., 102., 99., 101., 120.]]})
    pipe.ingest_candle({"hourly_candles": [[start + 7200000, 100., 102., 99., 101., 120.]]})
    assert len(pipe._ring) == 1
    assert pipe.transform_live() is None


def test_invalid_warmup_candle_is_not_accepted():
    pipe = FeaturePipeline()
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": [102., 90., 102.], "low": 99., "close": 101., "volume": 10.}, index=index)
    pipe.warm_up(frame)
    assert len(pipe._ring) == 1
    assert pipe._ring[-1]["ts"] == index[-1].timestamp()


ETF = '''<table><tr><th>Date</th><th>A</th><th>B</th><th>Total</th></tr>
<tr><td>19 Oct 2026</td><td>1</td><td>2</td><td>3</td></tr>
<tr><td>23 Oct 2026</td><td>(10.0)</td><td>20.0</td><td>10.0</td></tr>
<tr><td>26 Oct 2026</td><td><span>1,100.0</span></td><td>−10.0</td><td>1,090.0</td></tr>
<tr><td>27 Oct 2026</td><td>-</td><td>-</td><td>0.0</td></tr>
<tr><td>Total</td><td>9</td><td>9</td><td>18</td></tr>
<tr><td>Maximum</td><td>999</td><td>999</td><td>1998</td></tr></table>'''


def test_etf_uses_dated_total_and_calendar_window_not_last_cells():
    result = parse_flows(ETF, today=date(2026, 10, 27))
    assert result["etf_flow_date"] == "2026-10-26"
    assert result["etf_pending_date"] == "2026-10-27"
    assert result["etf_net_flow_daily"] == 1_090_000_000.
    assert result["etf_net_flow_7d"] == 1_100_000_000.
    assert result["etf_flow_days_in_7d"] == 2


def test_etf_pending_table_is_not_zero_flow():
    html = '<table><tr><th>Date</th><th>A</th><th>Total</th></tr><tr><td>27 Oct 2026</td><td>-</td><td>0</td></tr></table>'
    assert parse_flows(html, today=date(2026, 10, 27)) == {}


def test_etf_vendor_multiline_header_with_unlabelled_date():
    html = ETF.replace('<th>Date</th>', '<th></th>')
    result = parse_flows(html, today=date(2026, 10, 27))
    assert result["etf_flow_date"] == "2026-10-26"
    assert result["etf_net_flow_daily"] == 1_090_000_000.


def test_etf_real_zero_is_valid_and_old_or_unreconciled_rows_fail():
    html = '<table><tr><th>Date</th><th>A</th><th>Total</th></tr><tr><td>27 Oct 2026</td><td>0</td><td>0</td></tr></table>'
    assert parse_flows(html, today=date(2026, 10, 27))["etf_net_flow_daily"] == 0.
    assert parse_flows(html, today=date(2026, 11, 10)) == {}
    with pytest.raises(ValueError):
        parse_flows(html.replace('<td>0</td><td>0</td>', '<td>10</td><td>20</td>'), today=date(2026, 10, 27))


def test_etf_repoll_does_not_rejuvenate_same_published_flow():
    collector = ETFFlowCollector({})
    with patch.object(collector, "_scrape_farside", return_value=parse_flows(ETF, today=date(2026, 10, 27))):
        first = collector.safe_fetch()
        ts = collector.field_timestamps["etf_net_flow_daily"]
        collector.safe_fetch()
    assert collector.field_timestamps["etf_net_flow_daily"] == ts
    assert first["etf_net_flow_daily"] == 1_090_000_000.


def test_dashboard_accepts_real_float_engine_output():
    state = DashboardState()
    state.update_signal({"engines": {"microstructure": 5.41, "flow": -17.04, "macro": -1.52}})
    assert state.signal_history[-1]["micro"] == 5.41


def test_abstention_clears_previous_dashboard_and_hybrid_signal():
    import main
    state = DashboardState()
    state.current_signal = {"direction": "LONG", "state": "BULLISH"}
    state.hybrid_signal = {"state": "BULLISH"}
    with patch.object(main, "dashboard_state", state):
        bot = main.HydraBot.__new__(main.HydraBot)
        bot._publish_abstention("stale_price")
    assert state.current_signal["state"] == "NO_SIGNAL"
    assert state.hybrid_signal["state"] == "NO_SIGNAL"
    assert state.current_signal["score"] == 0


def test_orchestrator_failure_retains_age_and_never_overwrites_exchange_ratio():
    from main import HydraBot
    collector = FailedCollector("OKX", {})
    collector.max_retries = 1
    collector.fetch = MagicMock(return_value={"last_price": 100., "last_price_event_ms": 1000000, "nested": {"value": 1}})
    bot = HydraBot.__new__(HydraBot)
    bot.collectors = {"okx": collector}
    bot.use_coinglass = False
    bot._get_intervals = lambda: {"fast": 1, "medium": 1000000, "slow": 1000000, "very_slow": 1000000}
    bot.last_poll = {"fast": 0, "medium": 1000, "slow": 1000, "very_slow": 1000}
    bot._accumulated_data = {"total_stablecoin_mcap": 300e9, "total_mcap_trillion": 3., "stablecoin_exchange_ratio": .5}
    bot._accumulated_timestamps = {"total_stablecoin_mcap": 900., "total_mcap_trillion": 900.}
    with patch("time.time", return_value=1000):
        first = bot.collect_all()
    source_time = bot._accumulated_timestamps["last_price"]
    first["nested"]["value"] = 100
    assert bot._accumulated_data["nested"]["value"] == 1
    assert first["stablecoin_exchange_ratio"] == .5
    assert first["stablecoin_to_total_market_cap_ratio"] == .1
    collector.fetch.return_value = {}
    with patch("time.time", return_value=1100):
        second = bot.collect_all()
    assert second["last_price"] == 100.
    assert bot._accumulated_timestamps["last_price"] == source_time
    collector.fetch.return_value = {"last_price": 999., "last_price_event_ms": 999000}
    with patch("time.time", return_value=1200):
        older = bot.collect_all()
    assert older["last_price"] == 100.
    assert older["last_price_event_ms"] == 1000000


def test_quality_scaling_precedes_actionable_state_classification():
    from engines.output import SubSignal, SignalGrade, build_engine_output
    from signals.layer1_decision import Layer1DecisionEngine, DecisionState
    engines = [build_engine_output(name, [SubSignal("decision", 100., 100., SignalGrade.DECISION, 1., True)], 1.)
               for name in ("microstructure", "flow", "macro")]
    engine = Layer1DecisionEngine()
    assert engine.decide(*engines).state == DecisionState.BULLISH
    degraded = engine.decide(*engines, score_multiplier=.1)
    assert degraded.composite_score == 10.
    assert degraded.state == DecisionState.NEUTRAL
    assert not engine.decide(*engines, score_multiplier=.5).state.is_actionable
