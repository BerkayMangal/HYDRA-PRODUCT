"""
features/registry.py
─────────────────────
HYDRA Feature Contract Registry (Phase 2, Step 5)

Every feature in the system has a contract here. Features without
contracts are rejected by the quality gate.

STRUCTURE
---------
  PIPELINE_CONTRACTS  — Features computed by FeaturePipeline (OHLCV ring buffer)
  COLLECTOR_CONTRACTS — Features from collectors, z-scored by UnifiedDataStore
  MACRO_CONTRACTS     — Macro regime features (fixed in Phase 2)
  ALL_CONTRACTS       — Union of all, keyed by name

MAINTENANCE
-----------
  Adding a new feature? Add its contract here FIRST. The quality system
  will reject any feature not in this registry.
"""

from __future__ import annotations

from typing import Dict

from features.quality import FeatureContract, FeatureOrigin, SourceTier

# ---------------------------------------------------------------------------
# Pipeline features (from FeaturePipeline ring buffer, used by ML)
# ---------------------------------------------------------------------------

_PIPELINE: list[FeatureContract] = [
    # ── Trend / EMA ──────────────────────────────────────────────────────
    FeatureContract("above_ema50",  "1 if close > EMA(50,1H)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.5, True, 50),
    FeatureContract("above_ema200", "1 if close > EMA(200,1H)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.5, True, 200),
    FeatureContract("ema_stack",    "1 if EMA(8)>EMA(21)>EMA(50)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 50),
    FeatureContract("dist_ema50",   "(close-EMA50)/EMA50*100", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 50),
    FeatureContract("dist_ema200",  "(close-EMA200)/EMA200*100", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 200),
    FeatureContract("ema50_slope",  "pct_change(EMA50, 24 bars)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 74),
    # ── Drawdown ─────────────────────────────────────────────────────────
    FeatureContract("drawdown_from_ath", "(close-rolling_max)/rolling_max*100", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 1),
    FeatureContract("dd_speed", "drawdown.diff(24)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 25),
    # ── Volatility ───────────────────────────────────────────────────────
    FeatureContract("atr_pct", "ATR(14)/close*100", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 1.5, True, 14),
    FeatureContract("vol_regime", "rolling_std(ret,24)/rolling_std(ret,168)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 1.0, True, 168),
    FeatureContract("bb_w", "BB width: 4*std(20)/mean(20)*100", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 2.0, True, 20),
    # ── Volume ───────────────────────────────────────────────────────────
    FeatureContract("rvol", "volume/rolling_mean(volume,24)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 1.0, True, 24),
    FeatureContract("vol_trend", "rolling_mean(vol,24)/rolling_mean(vol,168)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 1.0, True, 168),
    FeatureContract("cvd_ratio", "signed_vol(6h)/abs_vol(24h)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 24),
    # ── Oscillators ──────────────────────────────────────────────────────
    FeatureContract("rsi", "RSI(14) Wilder-smoothed", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 50.0, True, 14),
    FeatureContract("adx", "ADX(14) Wilder-smoothed", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 25.0, True, 28),
    FeatureContract("vwap_dist", "(close-VWAP_24h)/VWAP_24h*100", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 24),
    # ── Returns ──────────────────────────────────────────────────────────
    FeatureContract("ret_4",  "close.pct_change(4) — 4H return", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 4),
    FeatureContract("ret_24", "close.pct_change(24) — 24H return", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 24),
    FeatureContract("ret_48", "close.pct_change(48) — 48H return", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 48),
    # ── Funding ──────────────────────────────────────────────────────────
    FeatureContract("fund_z", "funding z-score (72h lookback)", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 72),
    FeatureContract("fund_extreme_long", "1 if fund_z > 2.0", FeatureOrigin.OHLCV, SourceTier.A, 3600, 7200, 0.0, True, 72),
    # ── Calendar ─────────────────────────────────────────────────────────
    FeatureContract("hour", "UTC hour (0-23)", FeatureOrigin.CALENDAR, SourceTier.A, 300, 600, 12.0, False, 0),
    FeatureContract("is_us", "1 if US session", FeatureOrigin.CALENDAR, SourceTier.A, 300, 600, 0.0, False, 0),
    FeatureContract("day", "weekday 0=Mon..6=Sun", FeatureOrigin.CALENDAR, SourceTier.A, 300, 600, 3.0, False, 0),
    FeatureContract("is_weekend", "1 if day>=5", FeatureOrigin.CALENDAR, SourceTier.A, 300, 600, 0.0, False, 0),
]

# ---------------------------------------------------------------------------
# Macro regime features (FIXED: no more z-scores from forward-filled daily data)
# ---------------------------------------------------------------------------

_MACRO: list[FeatureContract] = [
    # Level features — raw values, NOT z-scored (daily data cannot produce valid hourly z-scores)
    FeatureContract("vix_level", "VIX current value", FeatureOrigin.MACRO, SourceTier.C, 86400, 172800, 20.0, True, 0),
    FeatureContract("vix_regime", "0/1/2 — low/medium/high vol", FeatureOrigin.MACRO, SourceTier.C, 86400, 172800, 1.0, True, 0),
    FeatureContract("vix_spike", "1 if VIX > 30", FeatureOrigin.MACRO, SourceTier.C, 86400, 172800, 0.0, True, 0),
    FeatureContract("qqq_ret_1d", "QQQ daily return (%)", FeatureOrigin.MACRO, SourceTier.C, 86400, 172800, 0.0, True, 0),
    FeatureContract("qqq_momentum_5d", "QQQ 5-day return (%)", FeatureOrigin.MACRO, SourceTier.C, 86400, 259200, 0.0, True, 0),
    FeatureContract("us10y_level", "US10Y yield", FeatureOrigin.MACRO, SourceTier.C, 86400, 172800, 4.0, True, 0),
    FeatureContract("us10y_change_1d", "US10Y 1d change (pp)", FeatureOrigin.MACRO, SourceTier.C, 86400, 172800, 0.0, True, 0),
    FeatureContract("gold_btc_ratio_z", "BTC/Gold ratio z-score (multi-day)", FeatureOrigin.MACRO, SourceTier.C, 86400, 259200, 0.0, True, 0),
    FeatureContract("fg_val", "Fear & Greed index (0-100)", FeatureOrigin.MACRO, SourceTier.C, 86400, 172800, 50.0, False, 0),
    FeatureContract("fg_regime", "-1/0/+1: fear/neutral/greed", FeatureOrigin.MACRO, SourceTier.C, 86400, 172800, 0.0, False, 0),
]

# ---------------------------------------------------------------------------
# Collector features (z-scored by UnifiedDataStore, used by Layer 1 engines)
# ---------------------------------------------------------------------------

_COLLECTOR: list[FeatureContract] = [
    # ── Microstructure (5-min refresh, 10-min TTL) ───────────────────────
    FeatureContract("cvd_spot_zscore", "CVD spot z-score", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 900, 0.0, True),
    FeatureContract("cvd_perp_zscore", "CVD perp z-score", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 900, 0.0, True),
    FeatureContract("cvd_5m_delta_zscore", "CVD 5m delta z-score", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 900, 0.0, True),
    FeatureContract("close_pct_5m", "5-min price change %", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 900, 0.0, True),
    FeatureContract("oi_change_pct_zscore", "OI change % z-score", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 900, 0.0, True),
    FeatureContract("funding_rate_zscore", "Funding rate z-score", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 1800, 0.0, True),
    FeatureContract("basis_spread_pct_zscore", "Basis spread z-score", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 900, 0.0, True),
    FeatureContract("ob_imbalance_raw_zscore", "Orderbook imbalance z-score", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 900, 0.0, True),
    FeatureContract("liq_imbalance_zscore", "Liquidation imbalance z-score", FeatureOrigin.COLLECTOR, SourceTier.B, 300, 1800, 0.0, True),
    FeatureContract("liq_total_zscore", "Total liquidation z-score", FeatureOrigin.COLLECTOR, SourceTier.B, 300, 1800, 0.0, True),
    FeatureContract("ls_ratio_zscore", "Long/Short ratio z-score", FeatureOrigin.COLLECTOR, SourceTier.A, 300, 1800, 0.0, True),

    # ── Flow (15-min refresh, 30-min TTL) ────────────────────────────────
    FeatureContract("exchange_netflow_btc_zscore", "Exchange netflow z-score", FeatureOrigin.COLLECTOR, SourceTier.B, 900, 3600, 0.0, True),
    FeatureContract("etf_net_flow_7d_zscore", "ETF 7d flow z-score", FeatureOrigin.COLLECTOR, SourceTier.B, 900, 7200, 0.0, True),
    FeatureContract("etf_net_flow_daily_zscore", "ETF daily flow z-score", FeatureOrigin.COLLECTOR, SourceTier.B, 900, 7200, 0.0, True),
    FeatureContract("fear_greed_value", "Fear & Greed index (0-100)", FeatureOrigin.COLLECTOR, SourceTier.C, 3600, 86400, 50.0, False),
    FeatureContract("stablecoin_exchange_ratio_zscore", "Stablecoin ratio z-score", FeatureOrigin.COLLECTOR, SourceTier.B, 900, 3600, 0.0, True),
    FeatureContract("total_mcap_change_24h", "Total mcap 24h change %", FeatureOrigin.COLLECTOR, SourceTier.B, 900, 3600, 0.0, True),

    # ── Macro (collector-level, used by MacroEngine) ─────────────────────
    FeatureContract("dxy_vs_sma20", "DXY vs 20h SMA %", FeatureOrigin.COLLECTOR, SourceTier.C, 3600, 86400, 0.0, True),
    FeatureContract("dxy_change_24h", "DXY 24h change %", FeatureOrigin.COLLECTOR, SourceTier.C, 3600, 86400, 0.0, True),
    FeatureContract("us10y_change_24h", "US10Y 24h change (pp)", FeatureOrigin.COLLECTOR, SourceTier.C, 3600, 86400, 0.0, True),
    FeatureContract("vix_current", "VIX current level", FeatureOrigin.COLLECTOR, SourceTier.C, 3600, 86400, 20.0, True),
    FeatureContract("vix_change_24h", "VIX 24h change %", FeatureOrigin.COLLECTOR, SourceTier.C, 3600, 86400, 0.0, True),
    FeatureContract("btc_spx_correlation", "BTC-SPX rolling corr", FeatureOrigin.COLLECTOR, SourceTier.C, 3600, 86400, 0.0, True),
    FeatureContract("spx_vs_sma20", "SPX vs 20h SMA %", FeatureOrigin.COLLECTOR, SourceTier.C, 3600, 86400, 0.0, True),
    FeatureContract("btc_dominance", "BTC market dominance %", FeatureOrigin.COLLECTOR, SourceTier.B, 900, 7200, 55.0, False),

    # ── Event flags ──────────────────────────────────────────────────────
    FeatureContract("fomc_just_passed", "FOMC just passed flag", FeatureOrigin.CALENDAR, SourceTier.A, 300, 7200, 0.0, True),
    FeatureContract("cpi_just_passed", "CPI just passed flag", FeatureOrigin.CALENDAR, SourceTier.A, 300, 7200, 0.0, True),
    FeatureContract("fomc_is_imminent", "FOMC imminent flag", FeatureOrigin.CALENDAR, SourceTier.A, 300, 7200, 0.0, True),
    FeatureContract("cpi_is_imminent", "CPI imminent flag", FeatureOrigin.CALENDAR, SourceTier.A, 300, 7200, 0.0, True),
    FeatureContract("event_dampen_factor", "Pre-event dampen 0.1-1.0", FeatureOrigin.CALENDAR, SourceTier.A, 300, 7200, 1.0, True),

    # ── Session / time ───────────────────────────────────────────────────
    FeatureContract("session_label", "Trading session name", FeatureOrigin.CALENDAR, SourceTier.A, 300, 600, 0.0, False),
]

# ---------------------------------------------------------------------------
# Prediction market features (dynamic keys — registered as a group)
# ---------------------------------------------------------------------------

# Prediction market features have dynamic names (poly_fed_cut_prob, kalshi_recession_prob, etc.)
# They are identified by prefix, not exact name. The quality system uses a special rule for these.
PREDICTION_MARKET_PREFIXES = ("poly_", "kalshi_")

# Default contract for any prediction market feature
PREDICTION_MARKET_DEFAULT_CONTRACT = FeatureContract(
    name="__prediction_market__",
    description="Dynamic prediction market probability",
    origin=FeatureOrigin.PREDICTION,
    source_tier=SourceTier.C,
    refresh_seconds=900,
    max_staleness_seconds=3600,
    neutral_value=0.5,
    decision_eligible=False,
)


# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------

ALL_CONTRACTS: Dict[str, FeatureContract] = {}
for _contract in _PIPELINE + _MACRO + _COLLECTOR:
    if _contract.name in ALL_CONTRACTS:
        raise ValueError(f"Duplicate feature contract: {_contract.name}")
    ALL_CONTRACTS[_contract.name] = _contract

PIPELINE_FEATURE_NAMES = [c.name for c in _PIPELINE]
MACRO_FEATURE_NAMES = [c.name for c in _MACRO]
COLLECTOR_FEATURE_NAMES = [c.name for c in _COLLECTOR]


def get_contract(feature_name: str) -> FeatureContract:
    """
    Look up the contract for a feature by name.

    Falls back to prediction market default for poly_/kalshi_ prefixes.
    Raises KeyError for truly unknown features.
    """
    contract = ALL_CONTRACTS.get(feature_name)
    if contract is not None:
        return contract

    # Dynamic prediction market features
    for prefix in PREDICTION_MARKET_PREFIXES:
        if feature_name.startswith(prefix):
            return FeatureContract(
                name=feature_name,
                description=f"Prediction market: {feature_name}",
                origin=FeatureOrigin.PREDICTION,
                source_tier=SourceTier.C,
                refresh_seconds=900,
                max_staleness_seconds=3600,
                neutral_value=0.5,
                decision_eligible=False,
            )

    raise KeyError(f"No contract registered for feature: {feature_name}")
