"""
features/pipeline.py
─────────────────────
HYDRA Unified Feature Pipeline

PROBLEM THIS SOLVES (P0-3 from AUDIT.md)
-----------------------------------------
The original system computed ML features two different ways:

  Training:  180 lines of Pandas arithmetic on a full OHLCV DataFrame
  Live:      A dict lookup with silent defaults and linear extrapolations
             (ret_8 = ret_4 * 2, ret_48 = ret_24 * 1.5, gold_btc_z = 0, ...)

The model was trained on one feature distribution and received a different
one at inference time — a fundamental correctness error.

SOLUTION
--------
This module provides ONE class — FeaturePipeline — with ONE implementation
of every feature computation. Both training and live inference call the same
underlying logic:

  Training:
    pipeline = FeaturePipeline()
    pipeline.warm_up(historical_df)          # seeds the ring buffer
    features_df = pipeline.transform_batch(historical_df)

  Live (each 5-min OKX cycle):
    pipeline.ingest_candle(candle_dict)      # feeds the ring buffer
    pipeline.ingest_macro(macro_dict)        # feeds the macro buffer
    result = pipeline.transform_live()       # same math as transform_batch

RING BUFFER DESIGN
------------------
The pipeline maintains an internal deque of 1H OHLCV bars (max 500 bars ≈ 21
days). Each call to ingest_candle() appends the latest 5-min candle to an
accumulator; when a complete 1H bar closes, it is pushed to the ring buffer.
All technical indicators are computed from this buffer, so:

  ret_4   = actual 4H return (not ret_24h / 6)
  ret_48  = actual 48H return (not ret_24h * 1.5)
  ema_50  = actual EMA(50) over 1H bars
  drawdown_from_ath = rolling max over 1H bars

FEATURE REGISTRY
----------------
Every feature is documented in FEATURE_REGISTRY with:
  - name
  - description
  - source ("ohlcv" | "funding" | "macro" | "calendar")
  - can_compute_live (True if ring buffer is sufficient, False if external)
  - min_bars (minimum 1H bars required for a valid value)

DATA QUALITY REPORT
-------------------
transform_live() returns a FeatureVector dataclass that includes:
  - values: Dict[str, float]  — feature values
  - quality: FeatureQuality   — per-feature classification
  - quality_score: float      — 0.0–1.0 (fraction of real features)
  - can_predict: bool         — False if quality_score < threshold

After Step 3, the ML engine will refuse to generate a signal when
can_predict is False.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Quality classification enum
# ---------------------------------------------------------------------------

class FeatureSource(str, Enum):
    REAL     = "real"       # Computed from ring buffer with sufficient history
    EXTERNAL = "external"   # Sourced from an external feed (macro, funding)
    PROXY    = "proxy"      # Approximation used due to insufficient data
    DEFAULT  = "default"    # Hardcoded fallback (worst quality)


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    source_type: str          # "ohlcv" | "funding" | "macro" | "calendar"
    min_bars: int             # minimum 1H bars before value is valid
    can_compute_live: bool    # True = ring buffer is enough


FEATURE_REGISTRY: Dict[str, FeatureSpec] = {
    spec.name: spec for spec in [
        # ── Trend / EMA ──────────────────────────────────────────────────
        FeatureSpec("above_ema50",    "1 if close > EMA(50,1H), else 0",         "ohlcv", 50,  True),
        FeatureSpec("above_ema200",   "1 if close > EMA(200,1H), else 0",        "ohlcv", 200, True),
        FeatureSpec("ema_stack",      "1 if EMA(8)>EMA(21)>EMA(50)",             "ohlcv", 50,  True),
        FeatureSpec("dist_ema50",     "(close-EMA50)/EMA50*100",                 "ohlcv", 50,  True),
        FeatureSpec("dist_ema200",    "(close-EMA200)/EMA200*100",               "ohlcv", 200, True),
        FeatureSpec("ema50_slope",    "pct_change(EMA50, 24 bars)",              "ohlcv", 74,  True),
        # ── Drawdown ─────────────────────────────────────────────────────
        FeatureSpec("drawdown_from_ath", "(close-rolling_max)/rolling_max*100", "ohlcv", 1,   True),
        FeatureSpec("dd_speed",          "drawdown_from_ath.diff(24)",           "ohlcv", 25,  True),
        # ── Volatility ───────────────────────────────────────────────────
        FeatureSpec("atr_pct",     "ATR(14)/close*100",                          "ohlcv", 14,  True),
        FeatureSpec("vol_regime",  "rolling_std(ret,24)/rolling_std(ret,168)",   "ohlcv", 168, True),
        FeatureSpec("bb_w",        "BB width: 4*std(20)/mean(20)*100",           "ohlcv", 20,  True),
        # ── Volume ───────────────────────────────────────────────────────
        FeatureSpec("rvol",       "volume/rolling_mean(volume,24)",              "ohlcv", 24,  True),
        FeatureSpec("vol_trend",  "rolling_mean(vol,24)/rolling_mean(vol,168)",  "ohlcv", 168, True),
        FeatureSpec("cvd_ratio",  "signed_vol(6h)/abs_vol(24h)",                 "ohlcv", 24,  True),
        # ── Oscillators ──────────────────────────────────────────────────
        FeatureSpec("rsi",       "RSI(14) Wilder-smoothed",                      "ohlcv", 14,  True),
        FeatureSpec("adx",       "ADX(14) Wilder-smoothed",                      "ohlcv", 28,  True),
        FeatureSpec("vwap_dist", "(close-VWAP_24h)/VWAP_24h*100",               "ohlcv", 24,  True),
        # ── Returns (all from ring buffer — NO proxies) ───────────────────
        FeatureSpec("ret_4",   "close.pct_change(4)  — actual 4H return",       "ohlcv", 4,   True),
        FeatureSpec("ret_24",  "close.pct_change(24) — actual 24H return",      "ohlcv", 24,  True),
        FeatureSpec("ret_48",  "close.pct_change(48) — actual 48H return",      "ohlcv", 48,  True),
        # ── Funding ──────────────────────────────────────────────────────
        FeatureSpec("fund_z",            "funding z-score (72h lookback)",       "funding", 72, True),
        FeatureSpec("fund_extreme_long", "1 if fund_z > 2.0",                   "funding", 72, True),
        # ── Macro (PHASE 2 FIX: regime features, NOT z-scores) ──────────
        # RATIONALE: Macro data arrives daily from yfinance. Broadcasting
        # one daily value to 500 hourly rows then computing rolling z-scores
        # produces rolling_std=0 → z-score=NaN → default 0. These features
        # were DEAD in production. The fix: treat macro as what it is —
        # low-frequency regime data. Levels and regimes, not z-scores.
        FeatureSpec("vix_level",       "VIX current value",                     "macro", 0,   False),
        FeatureSpec("vix_spike",       "1 if VIX > 30",                         "macro", 0,   False),
        FeatureSpec("vix_regime",      "0/1/2 — low/medium/high vol regime",    "macro", 0,   False),
        FeatureSpec("qqq_ret_1d",      "QQQ daily return (%)",                  "macro", 0,   False),
        FeatureSpec("qqq_momentum_5d", "QQQ 5-day return (%)",                  "macro", 0,   False),
        FeatureSpec("us10y_level",     "US10Y yield level",                     "macro", 0,   False),
        FeatureSpec("us10y_change_1d", "US10Y 1d change (pp)",                  "macro", 0,   False),
        FeatureSpec("gold_btc_ratio_z","BTC/Gold ratio z (multi-day buffer)",   "macro", 0,   False),
        FeatureSpec("fg_val",          "Fear & Greed index (0-100)",            "macro", 0,   False),
        FeatureSpec("fg_regime",       "-1/0/+1: fear/neutral/greed",           "macro", 0,   False),
        # ── Calendar ─────────────────────────────────────────────────────
        FeatureSpec("hour",       "UTC hour (0–23)",                             "calendar", 0, True),
        FeatureSpec("is_us",      "1 if 13<=hour<21 (US session)",              "calendar", 0, True),
        FeatureSpec("day",        "weekday: 0=Mon … 6=Sun",                     "calendar", 0, True),
        FeatureSpec("is_weekend", "1 if day >= 5",                              "calendar", 0, True),
    ]
}

# Ordered feature list (training and live must iterate in the same order)
FEATURE_NAMES: List[str] = list(FEATURE_REGISTRY.keys())

# Minimum quality score to allow a prediction
MIN_QUALITY_SCORE: float = 0.70  # 70% real or external features required


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureQuality:
    """Per-feature quality classification for one prediction row."""
    real:     List[str] = field(default_factory=list)   # computed from buffer
    external: List[str] = field(default_factory=list)   # sourced from feed
    proxy:    List[str] = field(default_factory=list)   # approximated
    default:  List[str] = field(default_factory=list)   # hardcoded fallback
    missing:  List[str] = field(default_factory=list)   # not in output at all

    @property
    def score(self) -> float:
        """Fraction of features that are real or external."""
        total = len(FEATURE_NAMES)
        high_quality = len(self.real) + len(self.external)
        return high_quality / total if total > 0 else 0.0

    @property
    def can_predict(self) -> bool:
        return self.score >= MIN_QUALITY_SCORE

    def summary(self) -> str:
        return (
            f"quality={self.score:.0%} "
            f"[real={len(self.real)} ext={len(self.external)} "
            f"proxy={len(self.proxy)} default={len(self.default)} "
            f"missing={len(self.missing)}]"
        )


@dataclass
class FeatureVector:
    """Output of transform_live() — a single prediction-ready feature row."""
    values:        Dict[str, float]
    quality:       FeatureQuality
    timestamp_utc: datetime
    bar_count:     int    # number of 1H bars in ring buffer
    macro_bars:    int    # number of macro observations in buffer

    @property
    def quality_score(self) -> float:
        return self.quality.score

    @property
    def can_predict(self) -> bool:
        return self.quality.can_predict

    def to_array(self, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Return feature values as a 1D float32 array in canonical order."""
        cols = feature_cols or FEATURE_NAMES
        return np.array([self.values.get(c, 0.0) for c in cols], dtype=np.float32)


# ---------------------------------------------------------------------------
# Internal OHLCV bar accumulator (5-min → 1H)
# ---------------------------------------------------------------------------

@dataclass
class _Bar:
    """One 5-min candle."""
    ts_utc: datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float
    funding: Optional[float] = None   # attached at ingestion time if available


class _HourlyAccumulator:
    """
    Collects 5-min candles and emits completed 1H bars.

    A bar is considered complete when its hour slot changes, i.e., when
    the first candle of a new hour arrives.
    """

    def __init__(self) -> None:
        self._pending: List[_Bar] = []
        self._current_hour: Optional[int] = None

    def push(self, bar: _Bar) -> Optional[Dict[str, float]]:
        """
        Push a 5-min bar. Returns a completed 1H OHLCV dict when a
        new hour boundary is crossed, else None.
        """
        hour_slot = bar.ts_utc.replace(minute=0, second=0, microsecond=0)
        if self._current_hour is None:
            self._current_hour = hour_slot
            self._pending.append(bar)
            return None

        if hour_slot != self._current_hour:
            # Hour boundary crossed — emit completed bar
            completed = self._emit()
            self._current_hour = hour_slot
            self._pending = [bar]
            return completed

        self._pending.append(bar)
        return None

    def _emit(self) -> Optional[Dict[str, float]]:
        if not self._pending:
            return None
        bars = self._pending
        fundings = [b.funding for b in bars if b.funding is not None]
        return {
            "ts":      self._current_hour.timestamp(),
            "open":    bars[0].open,
            "high":    max(b.high for b in bars),
            "low":     min(b.low  for b in bars),
            "close":   bars[-1].close,
            "volume":  sum(b.volume for b in bars),
            "funding": fundings[-1] if fundings else None,
        }


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Single canonical feature pipeline for both training and live inference.

    Usage — training
    ----------------
    >>> pipe = FeaturePipeline()
    >>> feature_df = pipe.transform_batch(historical_ohlcv_df)

    Usage — live (called once per 5-min OKX cycle)
    -----------------------------------------------
    >>> pipe = FeaturePipeline()
    >>> pipe.warm_up(historical_ohlcv_df)   # seed ring buffer at startup
    >>> ...
    >>> pipe.ingest_candle(okx_raw_dict, funding_rate=0.0001)
    >>> pipe.ingest_macro(macro_dict)
    >>> result: FeatureVector = pipe.transform_live()
    >>> if result.can_predict:
    ...     model.predict(result.to_array())
    """

    # Ring buffer capacity: 500 1H bars ≈ 21 days (enough for vol_regime 168h window)
    RING_CAPACITY: int = 500
    # Macro buffer capacity: 500 daily-ish observations (forward-filled to hourly)
    MACRO_CAPACITY: int = 500

    def __init__(self) -> None:
        self._ring: Deque[Dict[str, float]] = collections.deque(maxlen=self.RING_CAPACITY)
        self._macro: Deque[Dict[str, float]] = collections.deque(maxlen=self.MACRO_CAPACITY)
        self._accumulator = _HourlyAccumulator()
        self._warmup_done: bool = False

    # ------------------------------------------------------------------
    # Warm-up (called at startup with historical data)
    # ------------------------------------------------------------------

    def warm_up(self, df: pd.DataFrame) -> None:
        """
        Seed the ring buffer with historical 1H OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            1H OHLCV DataFrame with columns: open, high, low, close, volume.
            Optional column: funding (float).
            Index: datetime (UTC-aware or naive).
        """
        if df.empty:
            logger.warning("[Pipeline] warm_up called with empty DataFrame")
            return

        df = df.sort_index()
        # Normalise index to UTC-naive for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        for ts, row in df.iterrows():
            entry: Dict[str, float] = {
                "ts":     pd.Timestamp(ts).timestamp(),
                "open":   float(row.get("open", row.get("close", 0))),
                "high":   float(row.get("high", row.get("close", 0))),
                "low":    float(row.get("low",  row.get("close", 0))),
                "close":  float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "funding": float(row["funding"]) if "funding" in row and pd.notna(row["funding"]) else None,
            }
            self._ring.append(entry)

        self._warmup_done = True
        logger.info(
            "[Pipeline] Warm-up complete: {} 1H bars in ring buffer",
            len(self._ring),
        )

    # ------------------------------------------------------------------
    # Live ingestion
    # ------------------------------------------------------------------

    def ingest_candle(
        self,
        okx_data: Dict[str, Any],
        funding_rate: Optional[float] = None,
    ) -> bool:
        """
        Ingest one 5-min OHLCV snapshot from the OKX collector.

        Returns True when a new 1H bar is emitted to the ring buffer.

        Parameters
        ----------
        okx_data : dict
            Keys expected: open, high, low, close, volume, timestamp (ms)
        funding_rate : float, optional
            Current OKX perpetual funding rate.
        """
        try:
            ts_raw = okx_data.get("timestamp")
            if ts_raw is None:
                ts_utc = datetime.now(timezone.utc)
            elif isinstance(ts_raw, datetime):
                # ccxt or other collector already returned a datetime
                ts_utc = ts_raw if ts_raw.tzinfo else ts_raw.replace(tzinfo=timezone.utc)
            elif isinstance(ts_raw, (int, float)):
                # Millisecond epoch timestamp (standard ccxt format)
                ts_ms = float(ts_raw)
                if ts_ms > 1e12:  # milliseconds
                    ts_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                else:  # seconds
                    ts_utc = datetime.fromtimestamp(ts_ms, tz=timezone.utc)
            elif isinstance(ts_raw, str):
                # ISO format string fallback
                try:
                    ts_utc = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except ValueError:
                    ts_utc = datetime.fromtimestamp(float(ts_raw) / 1000, tz=timezone.utc)
            else:
                ts_utc = datetime.now(timezone.utc)

            bar = _Bar(
                ts_utc  = ts_utc,
                open    = float(okx_data.get("open",   okx_data.get("close", 0))),
                high    = float(okx_data.get("high",   okx_data.get("close", 0))),
                low     = float(okx_data.get("low",    okx_data.get("close", 0))),
                close   = float(okx_data.get("close",  okx_data.get("last_price", 0))),
                volume  = float(okx_data.get("volume", 0)),
                funding = funding_rate,
            )

            completed = self._accumulator.push(bar)
            if completed:
                self._ring.append(completed)
                logger.debug(
                    "[Pipeline] New 1H bar emitted. Ring: {} bars",
                    len(self._ring),
                )
                return True
            return False

        except Exception as exc:
            logger.error("[Pipeline] ingest_candle error: {}", exc)
            return False

    def ingest_macro(self, macro_data: Dict[str, Any]) -> None:
        """
        Ingest one macro observation from the MacroCollector.

        Parameters
        ----------
        macro_data : dict
            Expected keys: vix_current, qqq_current, us10y_current,
            gold_current, fear_greed_value, ts (unix epoch, optional).
        """
        try:
            ts = float(macro_data.get("ts", time.time()))
            entry: Dict[str, float] = {
                "ts": ts,
                "vix":        float(macro_data.get("vix_current",    0) or 0),
                "qqq":        float(macro_data.get("qqq_current",    0) or 0),
                "us10y":      float(macro_data.get("us10y_current",  0) or 0),
                "gold":       float(macro_data.get("gold_current",   0) or 0),
                "fear_greed": float(macro_data.get("fear_greed_value", 50) or 50),
            }
            # Only add if at least one macro value is non-zero
            if any(v > 0 for k, v in entry.items() if k != "ts"):
                self._macro.append(entry)
        except Exception as exc:
            logger.warning("[Pipeline] ingest_macro error: {}", exc)

    # ------------------------------------------------------------------
    # Core computation (shared between batch and live)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all technical feature computations to a 1H OHLCV DataFrame.

        This is THE canonical implementation. Both transform_batch() and
        transform_live() ultimately call this method on a DataFrame
        constructed from the ring buffer.

        Parameters
        ----------
        df : pd.DataFrame
            Columns: open, high, low, close, volume
            Optional: funding, vix, qqq, us10y, gold, fear_greed
            Index: datetime (any timezone or naive — treated as ordered)

        Returns
        -------
        pd.DataFrame
            Input columns + all FEATURE_NAMES columns appended.
        """
        f = df.copy()
        close  = f["close"]
        high   = f["high"]
        low    = f["low"]
        volume = f["volume"]
        open_  = f["open"]

        # ── EMA trend ────────────────────────────────────────────────────
        for span in (8, 21, 50, 200):
            f[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

        f["above_ema50"]  = (close > f["ema_50"]).astype(np.int8)
        f["above_ema200"] = (close > f["ema_200"]).astype(np.int8)
        f["ema_stack"]    = (
            (f["ema_8"] > f["ema_21"]) & (f["ema_21"] > f["ema_50"])
        ).astype(np.int8)
        f["dist_ema50"]   = (close - f["ema_50"])  / f["ema_50"].replace(0, np.nan)  * 100
        f["dist_ema200"]  = (close - f["ema_200"]) / f["ema_200"].replace(0, np.nan) * 100
        f["ema50_slope"]  = f["ema_50"].pct_change(24) * 100

        # ── Drawdown ─────────────────────────────────────────────────────
        f["rolling_ath"]       = close.cummax()
        f["drawdown_from_ath"] = (
            (close - f["rolling_ath"]) / f["rolling_ath"].replace(0, np.nan) * 100
        )
        f["dd_speed"] = f["drawdown_from_ath"].diff(24)

        # ── True Range & ATR ─────────────────────────────────────────────
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        f["atr_pct"] = (
            tr.ewm(alpha=1 / 14, adjust=False).mean()
            / close.replace(0, np.nan) * 100
        )

        # ── Volatility regime ─────────────────────────────────────────────
        ret_1h = close.pct_change(1)
        vol_24  = ret_1h.rolling(24).std()
        vol_168 = ret_1h.rolling(168).std().replace(0, np.nan)
        f["vol_regime"] = vol_24 / vol_168

        # ── Bollinger Band width ──────────────────────────────────────────
        roll20_std  = close.rolling(20).std()
        roll20_mean = close.rolling(20).mean().replace(0, np.nan)
        f["bb_w"] = roll20_std / roll20_mean * 100 * 4

        # ── Volume features ───────────────────────────────────────────────
        vol_ma24  = volume.rolling(24).mean().replace(0, np.nan)
        vol_ma168 = volume.rolling(168).mean().replace(0, np.nan)
        f["rvol"]      = volume / vol_ma24
        f["vol_trend"] = vol_ma24 / vol_ma168

        signed_vol = np.where(close >= open_, volume, -volume)
        cvd_6    = pd.Series(signed_vol, index=f.index).rolling(6).sum()
        abs_vol24 = pd.Series(np.abs(signed_vol), index=f.index).rolling(24).sum()
        f["cvd_ratio"] = cvd_6 / abs_vol24.replace(0, np.nan)

        # ── RSI (Wilder's smoothing) ───────────────────────────────────────
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
        loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
        f["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # ── ADX (Wilder's smoothing) ───────────────────────────────────────
        high_diff = high.diff()
        low_diff  = low.diff()
        plus_dm   = high_diff.where((high_diff > -low_diff) & (high_diff > 0), 0.0)
        minus_dm  = (-low_diff).where((-low_diff > high_diff) & (-low_diff > 0), 0.0)
        atr_ew    = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
        di_plus   = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_ew
        di_minus  = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_ew
        di_sum    = (di_plus + di_minus).replace(0, np.nan)
        dx        = 100 * (di_plus - di_minus).abs() / di_sum
        f["adx"]  = dx.ewm(alpha=1 / 14, adjust=False).mean()

        # ── Rolling VWAP (24h) ────────────────────────────────────────────
        rpv           = (close * volume).rolling(24).sum()
        rv            = volume.rolling(24).sum().replace(0, np.nan)
        vwap          = rpv / rv
        f["vwap_dist"] = (close - vwap) / vwap.replace(0, np.nan) * 100

        # ── Returns (from ring buffer — NO proxies) ───────────────────────
        f["ret_4"]   = close.pct_change(4)
        f["ret_24"]  = close.pct_change(24)
        f["ret_48"]  = close.pct_change(48)

        # ── Funding ───────────────────────────────────────────────────────
        if "funding" in f.columns and f["funding"].notna().sum() > 10:
            fund       = f["funding"].ffill()
            fund_mean  = fund.rolling(72).mean()
            fund_std   = fund.rolling(72).std().replace(0, np.nan)
            f["fund_z"]            = (fund - fund_mean) / fund_std
            f["fund_extreme_long"] = (f["fund_z"] > 2.0).astype(np.int8)
        else:
            f["fund_z"]            = np.nan
            f["fund_extreme_long"] = np.nan

        # ── Macro (PHASE 2 FIX: regime features, NOT z-scores) ─────────
        # Daily data forward-filled to hourly rows produces rolling_std=0.
        # These features use LEVELS and REGIMES which are statistically valid
        # for low-frequency data.
        if "vix" in f.columns and f["vix"].notna().any():
            vix_f = f["vix"].ffill()
            f["vix_level"]  = vix_f
            f["vix_regime"] = np.where(vix_f > 30, 2, np.where(vix_f > 22, 1, 0))
            f["vix_spike"]  = (vix_f > 30).astype(np.int8)
        else:
            f["vix_level"]  = np.nan
            f["vix_regime"] = np.nan
            f["vix_spike"]  = np.nan

        if "qqq" in f.columns and f["qqq"].notna().any():
            qqq_f = f["qqq"].ffill()
            # Daily return: use pct_change(24) only if we have >1 distinct value
            n_distinct = qqq_f.nunique()
            if n_distinct >= 2:
                f["qqq_ret_1d"]      = qqq_f.pct_change(24) * 100
                f["qqq_momentum_5d"] = qqq_f.pct_change(120) * 100
            else:
                # Only one value — cannot compute valid change
                f["qqq_ret_1d"]      = np.nan
                f["qqq_momentum_5d"] = np.nan
        else:
            f["qqq_ret_1d"]      = np.nan
            f["qqq_momentum_5d"] = np.nan

        if "us10y" in f.columns and f["us10y"].notna().any():
            us10y_f = f["us10y"].ffill()
            f["us10y_level"] = us10y_f
            n_distinct = us10y_f.nunique()
            if n_distinct >= 2:
                f["us10y_change_1d"] = us10y_f.diff(24)  # absolute change in pp
            else:
                f["us10y_change_1d"] = np.nan
        else:
            f["us10y_level"]     = np.nan
            f["us10y_change_1d"] = np.nan

        if "gold" in f.columns and not close.empty and f["gold"].notna().any():
            gold_f = f["gold"].ffill().replace(0, np.nan)
            ratio = close / gold_f
            # Only compute z-score if we have multiple distinct gold values
            n_distinct = gold_f.nunique()
            if n_distinct >= 3:
                ratio_mean = ratio.rolling(168, min_periods=3).mean()
                ratio_std  = ratio.rolling(168, min_periods=3).std().replace(0, np.nan)
                f["gold_btc_ratio_z"] = (ratio - ratio_mean) / ratio_std
            else:
                f["gold_btc_ratio_z"] = np.nan
        else:
            f["gold_btc_ratio_z"] = np.nan

        if "fear_greed" in f.columns and f["fear_greed"].notna().any():
            fg = f["fear_greed"].ffill()
            f["fg_val"]    = fg
            f["fg_regime"] = np.where(fg <= 25, -1, np.where(fg >= 75, 1, 0))
        else:
            f["fg_val"]    = np.nan
            f["fg_regime"] = np.nan

        # ── Calendar ──────────────────────────────────────────────────────
        f["hour"]       = f.index.hour if hasattr(f.index, "hour") else 0
        f["is_us"]      = ((f["hour"] >= 13) & (f["hour"] < 21)).astype(np.int8)
        f["day"]        = f.index.dayofweek if hasattr(f.index, "dayofweek") else 0
        f["is_weekend"] = (f["day"] >= 5).astype(np.int8)

        return f

    # ------------------------------------------------------------------
    # Training path: transform_batch()
    # ------------------------------------------------------------------

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for a historical OHLCV DataFrame.

        Used for model training and offline backtesting.

        Parameters
        ----------
        df : pd.DataFrame
            1H OHLCV. Index: UTC datetime. Columns: open/high/low/close/volume.
            Optional: funding, vix, qqq, us10y, gold, fear_greed.

        Returns
        -------
        pd.DataFrame
            Original columns + all FEATURE_NAMES columns.
            Rows with insufficient history are present but have NaN feature values.
        """
        if df.empty:
            logger.warning("[Pipeline] transform_batch: empty DataFrame")
            return df

        df = df.sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        result = self._compute_features_from_df(df)
        available = [c for c in FEATURE_NAMES if c in result.columns]
        missing   = [c for c in FEATURE_NAMES if c not in result.columns]
        if missing:
            logger.warning(
                "[Pipeline] transform_batch: {} features absent: {}",
                len(missing), missing,
            )
        logger.info(
            "[Pipeline] transform_batch: {} rows, {}/{} features",
            len(result), len(available), len(FEATURE_NAMES),
        )
        return result

    # ------------------------------------------------------------------
    # Live path: transform_live()
    # ------------------------------------------------------------------

    def transform_live(self) -> Optional[FeatureVector]:
        """
        Compute a feature vector from the current ring buffer state.

        Used for live ML inference. Called after ingest_candle() and
        ingest_macro() have been updated for the current cycle.

        Returns None if the ring buffer has fewer than 5 bars (insufficient
        for any meaningful computation).
        """
        if len(self._ring) < 5:
            logger.warning(
                "[Pipeline] transform_live: only {} bars in buffer, need ≥5",
                len(self._ring),
            )
            return None

        # Build DataFrame from ring buffer
        df = self._ring_to_df()

        # Attach macro data
        if self._macro:
            df = self._attach_macro_to_df(df)

        # Compute features (same function as training)
        f = self._compute_features_from_df(df)

        # Extract last row
        last = f.iloc[-1]
        now  = datetime.now(timezone.utc)

        values: Dict[str, float] = {}
        quality = FeatureQuality()
        n_bars  = len(self._ring)

        for name, spec in FEATURE_REGISTRY.items():
            val = last.get(name, np.nan)

            if pd.isna(val) or np.isinf(val):
                # Feature is NaN — classify and fill
                if spec.min_bars > n_bars:
                    # Not enough history yet — proxy or default
                    if spec.source_type == "ohlcv":
                        quality.proxy.append(name)
                        values[name] = self._fallback(name)
                    else:
                        quality.default.append(name)
                        values[name] = 0.0
                else:
                    # History available but data missing (e.g. funding not ingested)
                    quality.default.append(name)
                    values[name] = 0.0
                    logger.debug(
                        "[Pipeline] Feature '{}' is NaN despite {} bars", name, n_bars
                    )
            else:
                # Value is present — classify quality
                if spec.source_type in ("ohlcv", "calendar"):
                    quality.real.append(name)
                elif spec.source_type in ("funding", "macro"):
                    quality.external.append(name)
                values[name] = float(val)

        return FeatureVector(
            values        = values,
            quality       = quality,
            timestamp_utc = now,
            bar_count     = n_bars,
            macro_bars    = len(self._macro),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ring_to_df(self) -> pd.DataFrame:
        """Convert ring buffer to a DataFrame with a proper DatetimeIndex."""
        rows = list(self._ring)
        df   = pd.DataFrame(rows)
        df["dt"] = pd.to_datetime(df["ts"], unit="s")
        df = df.set_index("dt").drop(columns=["ts"])
        return df

    def _attach_macro_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach macro observations to the OHLCV DataFrame.

        For live inference: macro data is intraday-sparse (daily or less
        frequent), so we broadcast the most recent non-null value across
        the entire ring buffer. This is the same effective result as
        forward-filling daily data to hourly bars.

        For historical batch use (where macro is already in df columns from
        the original historical fetch), this method is a no-op.
        """
        if not self._macro:
            return df

        # Take the most recent macro observation for each field
        # (self._macro is ordered oldest-first; take last non-null value)
        for col in ("vix", "qqq", "us10y", "gold", "fear_greed"):
            if col in df.columns:
                continue   # Already present from batch/warm-up data
            # Find last non-null value in macro buffer
            val = None
            for obs in reversed(list(self._macro)):
                candidate = obs.get(col)
                if candidate is not None and candidate != 0:
                    val = candidate
                    break
            if val is not None:
                df[col] = val   # broadcast single value to all rows (daily ffill equivalent)

        return df

    @staticmethod
    def _fallback(feature_name: str) -> float:
        """
        Conservative fallback values for features with insufficient history.

        These are chosen to be neutral (not extreme) so they don't inject
        a spurious signal. Wherever possible, 0.0 is correct for z-scored
        features. For ratio features, 1.0 is neutral.
        """
        neutral_ratios = {"rvol", "vol_trend", "vol_regime"}
        neutral_50     = {"rsi"}
        if feature_name in neutral_ratios:
            return 1.0
        if feature_name in neutral_50:
            return 50.0
        return 0.0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return pipeline status for logging / dashboard."""
        return {
            "ring_buffer_bars":   len(self._ring),
            "macro_buffer_obs":   len(self._macro),
            "warmup_done":        self._warmup_done,
            "ring_capacity":      self.RING_CAPACITY,
            "total_features":     len(FEATURE_NAMES),
            "min_quality_score":  MIN_QUALITY_SCORE,
        }
