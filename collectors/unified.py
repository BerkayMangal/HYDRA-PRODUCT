"""
collectors/unified.py — UnifiedDataStore v4
═══════════════════════════════════════════
FIXES vs v3:

  FIX 1 — LOCF with Exponential Decay (replaces hard NaN on TTL expiry)
  ─────────────────────────────────────────────────────────────────────
  v3 set stale features to np.nan immediately on TTL expiry.
  Result: Fear=nan/100, ETF nanM in pulse logs, ML model fed with NaN rows.

  Quantitative rationale: fear & greed index is a *slow-moving* signal.
  A value from 40 minutes ago is still informative; setting it to NaN
  destroys information that was perfectly valid minutes earlier. The
  correct approach is exponential decay toward a neutral prior:

      decayed = last_value + (neutral_prior - last_value) * (1 - exp(-k*excess_age))

  where excess_age = (current_time - last_ts) - ttl, and k controls
  the decay rate (k=1 means value decays to 63% of the gap from neutral
  in one TTL-length of extra age).

  After 2× TTL the value is ~63% decayed toward neutral.
  After 4× TTL the value is ~86% decayed — effectively neutral.
  Hard NaN is only applied after 6× TTL (value indistinguishable from prior).

  This preserves the statistical direction of old data while reducing
  its magnitude — mathematically equivalent to a Bayesian update that
  shrinks toward a diffuse prior as evidence becomes stale.

  FIX 2 — Extended Feature Group Registry
  ────────────────────────────────────────
  Prediction market features (poly_*, kalshi_*) were not in any feature
  group → defaulted to microstructure TTL (600s). They are collected at
  the 'medium' interval (900s). They expired BEFORE the next collection
  cycle, guaranteed. Added a 'prediction_market' group with TTL aligned
  to the medium polling interval.

  FIX 3 — Per-feature Neutral Priors for Decay
  ─────────────────────────────────────────────
  Fear & Greed decays toward 50 (neutral). Funding rate decays toward 0.
  OI change decays toward 0. This prevents a stale extreme value from
  permanently anchoring the model's view of the world.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger


# Neutral priors for LOCF decay — what to decay toward when data goes stale.
# Key insight: decay to the "uninformative" value, not zero.
_NEUTRAL_PRIORS: Dict[str, float] = {
    "fear_greed_value":          50.0,   # 0–100, neutral = 50
    "btc_dominance":             55.0,   # typical baseline
    "ls_ratio":                   1.0,   # long = short = neutral
    "funding_rate":               0.0,
    "oi_change_pct":              0.0,
    "ob_imbalance_raw":           0.0,
    "cvd_buy_ratio":              0.5,
    "vix_current":               20.0,   # historical average
    "dxy_current":              104.0,   # approximate mean
    "stablecoin_exchange_ratio":  0.0,
    "stablecoin_ratio_delta":     0.0,
    "alt_season_index":           0.0,
    "etf_net_flow_daily":         0.0,
    "etf_net_flow_7d":            0.0,
    "exchange_netflow_btc":       0.0,
    "exchange_netflow_7d":        0.0,
    "miner_outflow_btc":          0.0,
    "defi_tvl_change_24h":        0.0,
    "total_mcap_change_24h":      0.0,
}

# After this many TTL multiples past expiry, apply hard NaN.
_HARD_NAN_MULTIPLIER: float = 6.0
# Decay rate constant: at 1× excess TTL the value has decayed ~63% toward neutral.
_DECAY_K: float = 1.0


class UnifiedDataStore:
    """
    Central feature store for HYDRA v4.

    Core changes from v3:
      - LOCF with exponential decay replaces hard NaN on TTL expiry.
      - Prediction market features have their own group + TTL.
      - Decay is logged at DEBUG level to avoid log spam.
    """

    MIN_ROWS_FOR_ZSCORE: int = 20
    TTL_MULTIPLIER: float = 2.0

    _CRITICAL_FEATURES: tuple = (
        "close",
        "volume",
        "funding_rate",
        "cvd_perp",
        "ob_imbalance_raw",
    )

    def __init__(self, config: Dict) -> None:
        self.config  = config
        self.history: pd.DataFrame = pd.DataFrame()

        self.feature_groups: Dict[str, list] = {
            "microstructure": [
                "oi_change", "oi_change_pct", "funding_rate", "basis_spread_pct",
                "cvd_perp", "cvd_5m_delta", "cvd_buy_ratio", "cvd_spot",
                "ob_imbalance_raw", "bid_wall_pct", "ask_wall_pct",
                "ls_ratio", "ls_ratio_delta", "liq_long_vol", "liq_short_vol",
                "liq_total", "liq_imbalance", "spread_bps", "close_pct_5m",
                "volume", "volume_change_pct",
            ],
            "flow": [
                "exchange_netflow_btc", "exchange_netflow_7d",
                "miner_outflow_btc", "etf_net_flow_daily", "etf_net_flow_7d",
                "stablecoin_exchange_ratio", "stablecoin_ratio_delta",
                "fear_greed_value", "total_mcap_change_24h",
            ],
            "macro": [
                "dxy_current", "dxy_change_24h", "dxy_vs_sma20",
                "us10y_current", "us10y_change_24h", "us10y_vs_sma20",
                "spx_current", "spx_change_24h", "spx_vs_sma20",
                "vix_current", "vix_change_24h",
                "btc_spx_correlation",
                "btc_dominance", "total_mcap_trillion",
            ],
            # FIX 2: prediction markets were unknown → microstructure TTL (600s)
            # but collected at medium interval (900s) → always stale.
            # Now they have their own group with TTL = 2 × 900s = 1800s.
            "prediction_market": [
                # Polymarket
                "poly_fed_cut_prob", "poly_btc_100k_prob", "poly_btc_ath_prob",
                "poly_recession_prob", "poly_crypto_ban_prob",
                # Kalshi
                "kalshi_fed_rate_cut_prob", "kalshi_inflation_above_prob",
                # Prediction markets collector aggregates
                "fed_cut_probability", "recession_probability",
                "pred_market_fed_spread", "pred_market_avg_spread",
            ],
        }

        # Build reverse lookup
        self._feature_to_group: Dict[str, str] = {
            feat: group
            for group, feats in self.feature_groups.items()
            for feat in feats
        }

        # Also add prefix-based group lookup for dynamically-named features
        # e.g. poly_will_china_invade_taiwan_before_2027_prob
        self._group_prefixes: Dict[str, str] = {
            "poly_": "prediction_market",
            "kalshi_": "prediction_market",
            "pred_market_": "prediction_market",
        }

        norm_config = config.get("normalization", {})
        self.norm_windows: Dict[str, int] = {
            "microstructure":    norm_config.get("microstructure", {}).get("window_hours", 24),
            "flow":              norm_config.get("flow",           {}).get("window_days",  7) * 24,
            "macro":             norm_config.get("macro",          {}).get("window_days", 30) * 24,
            "prediction_market": 7 * 24,  # 7-day window for prediction market signals
        }

        coll_config    = config.get("collectors", {})
        poll_micro     = int(coll_config.get("microstructure", {}).get("polling_interval_sec", 300))
        poll_flow      = int(coll_config.get("flow",           {}).get("polling_interval_sec", 3600))
        poll_macro     = int(coll_config.get("macro",          {}).get("polling_interval_sec", 3600))
        poll_medium    = 900   # medium interval (polymarket, kalshi)
        poll_very_slow = 14400

        self.feature_ttls: Dict[str, float] = {
            "microstructure":    poll_micro     * self.TTL_MULTIPLIER,
            "flow":              poll_flow      * self.TTL_MULTIPLIER,
            "macro":             poll_macro     * self.TTL_MULTIPLIER,
            "prediction_market": poll_medium    * self.TTL_MULTIPLIER,  # FIX 2
        }

        self._very_slow_features: Set[str] = {
            "exchange_netflow_btc", "exchange_netflow_7d", "exchange_netflow_trend",
            "miner_outflow_btc", "miner_outflow_7d_avg",
            "stablecoin_exchange_ratio", "stablecoin_ratio_delta",
            "etf_net_flow_daily", "etf_net_flow_7d",
        }
        self._very_slow_ttl: float = poll_very_slow * self.TTL_MULTIPLIER

        self.bars_per_hour: int = 12

        logger.info(
            "[DataStore] v4 | TTL micro={}s flow={}s macro={}s pred_mkt={}s very_slow={}s | LOCF decay enabled",
            self.feature_ttls["microstructure"],
            self.feature_ttls["flow"],
            self.feature_ttls["macro"],
            self.feature_ttls["prediction_market"],
            self._very_slow_ttl,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        collector_data: Dict[str, Any],
        feature_timestamps: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        now    = datetime.now(timezone.utc)
        now_ts = time.time()

        collector_data["timestamp"]    = now
        collector_data["hour_utc"]     = now.hour
        collector_data["day_of_week"]  = now.weekday()
        collector_data["session_label"] = self._get_session(now.hour)

        # ── LOCF with Exponential Decay (replaces hard NaN) ──────────────
        if feature_timestamps:
            decayed_count = 0
            nan_count     = 0
            stale_names   = []

            for feat, last_ts in feature_timestamps.items():
                age_seconds = now_ts - last_ts
                ttl         = self._get_feature_ttl(feat)

                if age_seconds <= ttl:
                    continue  # fresh — no action

                excess_age  = age_seconds - ttl

                # Hard NaN: feature is extremely stale (> 6× TTL)
                if excess_age > (_HARD_NAN_MULTIPLIER - 1) * ttl:
                    collector_data[feat] = np.nan
                    nan_count  += 1
                    stale_names.append(f"{feat}({age_seconds/3600:.1f}h→NaN)")
                    continue

                # Soft decay: apply LOCF with exponential decay toward neutral
                current_val = collector_data.get(feat)
                if current_val is None:
                    continue
                try:
                    current_float = float(current_val)
                    if math.isnan(current_float):
                        continue
                except (TypeError, ValueError):
                    continue

                neutral    = _NEUTRAL_PRIORS.get(feat, 0.0)
                # decay factor: 0 = no decay (fresh), 1 = fully at neutral
                decay      = 1.0 - math.exp(-_DECAY_K * (excess_age / ttl))
                decayed_val = current_float + (neutral - current_float) * decay

                collector_data[feat] = decayed_val
                decayed_count += 1
                logger.debug(
                    "[DataStore] LOCF decay: {} {:.3g}→{:.3g} (age={:.0f}s, ttl={:.0f}s, decay={:.1%})",
                    feat, current_float, decayed_val, age_seconds, ttl, decay
                )

            if nan_count > 0:
                logger.warning(
                    "[DataStore] {} features expired (>{}×TTL) → hard NaN: {}",
                    nan_count, _HARD_NAN_MULTIPLIER,
                    ", ".join(stale_names[:10]) + ("..." if nan_count > 10 else ""),
                )
            if decayed_count > 0:
                logger.debug("[DataStore] {} features LOCF-decayed toward neutral", decayed_count)

        # ── Append to history ────────────────────────────────────────────
        row      = pd.Series(collector_data)
        row.name = now

        self.history = pd.concat(
            [self.history, pd.DataFrame([row])],
            ignore_index=False,
        )

        max_rows = 30 * 24 * self.bars_per_hour
        if len(self.history) > max_rows:
            self.history = self.history.iloc[-max_rows:]

        normalized = self._normalize_latest()
        combined   = pd.concat([row, normalized])

        n_rows = len(self.history)
        completeness = self._check_completeness(combined, feature_timestamps, now_ts)
        combined["feature_completeness"] = completeness
        combined["data_maturity"]        = min(n_rows / self.MIN_ROWS_FOR_ZSCORE, 1.0)

        logger.debug(
            "[DataStore] Rows:{} | Features:{} | Completeness:{:.0%} | Maturity:{:.0%}",
            n_rows, len(combined), completeness, combined["data_maturity"],
        )

        return combined

    # ── TTL helpers ─────────────────────────────────────────────────────────

    def _get_feature_ttl(self, feature: str) -> float:
        if feature in self._very_slow_features:
            return self._very_slow_ttl

        # Exact group lookup
        group = self._feature_to_group.get(feature)
        if group:
            return self.feature_ttls[group]

        # FIX 2: prefix-based group lookup for dynamic poly_/kalshi_ features
        for prefix, grp in self._group_prefixes.items():
            if feature.startswith(prefix):
                return self.feature_ttls[grp]

        # Unknown feature: conservative microstructure TTL
        return self.feature_ttls["microstructure"]

    # ── Normalization ───────────────────────────────────────────────────────

    def _normalize_latest(self) -> pd.Series:
        normalized: Dict[str, float] = {}
        n_rows = len(self.history)

        if n_rows < 2:
            return self._cold_start_scaling()

        for group_name, features in self.feature_groups.items():
            use_zscore = n_rows >= self.MIN_ROWS_FOR_ZSCORE

            if use_zscore:
                window_hours = self.norm_windows.get(group_name, 24)
                window_bars  = window_hours * self.bars_per_hour
                window       = self.history.tail(min(window_bars, n_rows))
            else:
                window = self.history

            for feat in features:
                if feat not in window.columns:
                    continue

                col = pd.to_numeric(window[feat], errors="coerce").dropna()
                if len(col) < 1:
                    continue

                current_val = col.iloc[-1]

                if use_zscore and len(col) >= 5:
                    mean = col.mean()
                    std  = col.std()
                    if std > 0:
                        normalized[f"{feat}_zscore"] = float((current_val - mean) / std)
                    else:
                        normalized[f"{feat}_zscore"] = 0.0
                    normalized[f"{feat}_pctrank"] = float(
                        (col < current_val).sum() / len(col)
                    )
                else:
                    scaled = self._scale_raw(feat, current_val)
                    normalized[f"{feat}_zscore"] = scaled
                    normalized[f"{feat}_pctrank"] = 0.5

        return pd.Series(normalized)

    def _cold_start_scaling(self) -> pd.Series:
        normalized: Dict[str, float] = {}
        if self.history.empty:
            return pd.Series(normalized)

        row = self.history.iloc[-1]

        for group_name, features in self.feature_groups.items():
            for feat in features:
                if feat in row and pd.notna(row[feat]):
                    val    = float(row[feat])
                    scaled = self._scale_raw(feat, val)
                    normalized[f"{feat}_zscore"] = scaled
                    normalized[f"{feat}_pctrank"] = 0.5

        return pd.Series(normalized)

    def _scale_raw(self, feature: str, value: float) -> float:
        if feature == "funding_rate":
            return float(np.clip(value / 0.015, -3, 3))
        if feature == "oi_change_pct":
            return float(np.clip(value / 1.5, -3, 3))
        if feature == "ob_imbalance_raw":
            return float(np.clip(value / 0.25, -3, 3))
        if feature == "cvd_buy_ratio":
            return float(np.clip((value - 0.5) / 0.08, -3, 3))
        if feature == "ls_ratio":
            return float(np.clip((value - 1.0) / 0.15, -3, 3))
        if feature == "ls_ratio_delta":
            return float(np.clip(value / 0.05, -3, 3))
        if feature == "close_pct_5m":
            return float(np.clip(value / 0.3, -3, 3))
        if feature == "volume_change_pct":
            return float(np.clip(value / 50.0, -3, 3))
        if feature == "spread_bps":
            return float(np.clip((value - 2.0) / 2.0, -3, 3))
        if feature == "cvd_perp":
            if value == 0:
                return 0.0
            return float(np.clip(np.sign(value) * min(abs(value) / 50.0, 3), -3, 3))
        if feature == "cvd_5m_delta":
            if value == 0:
                return 0.0
            return float(np.clip(np.sign(value) * min(abs(value) / 30.0, 3), -3, 3))
        if feature == "etf_net_flow_daily":
            return float(np.clip(value / 500_000_000.0, -3, 3))
        if feature == "etf_net_flow_7d":
            return float(np.clip(value / 2_000_000_000.0, -3, 3))
        if feature in ("exchange_netflow_btc", "exchange_netflow_trend"):
            return float(np.clip(-value / 3_000.0, -3, 3))
        if feature == "exchange_netflow_7d":
            return float(np.clip(-value / 15_000.0, -3, 3))
        if feature == "miner_outflow_btc":
            return float(np.clip(value / 1_000.0, -3, 3))
        if "_change_" in feature or "_vs_sma" in feature:
            return float(np.clip(value / 1.0, -3, 3))
        if feature == "vix_current":
            return float(np.clip((value - 20.0) / 5.0, -3, 3))
        if feature == "fear_greed_value":
            return float(np.clip((value - 50.0) / 15.0, -3, 3))
        if feature == "total_mcap_change_24h":
            return float(np.clip(value / 2.0, -3, 3))
        if feature == "btc_dominance":
            return float(np.clip((value - 50.0) / 5.0, -3, 3))
        if feature == "stablecoin_ratio_delta":
            return float(np.clip(value / 0.005, -3, 3))
        if value == 0:
            return 0.0
        return float(np.clip(value, -3.0, 3.0))

    # ── Data quality ────────────────────────────────────────────────────────

    def _check_completeness(
        self,
        row: pd.Series,
        feature_timestamps: Optional[Dict[str, float]],
        now_ts: float,
    ) -> float:
        present = 0
        for feat in self._CRITICAL_FEATURES:
            val = row.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if feature_timestamps and feat in feature_timestamps:
                age = now_ts - feature_timestamps[feat]
                ttl = self._get_feature_ttl(feat)
                if age > _HARD_NAN_MULTIPLIER * ttl:
                    continue
            present += 1
        return present / len(self._CRITICAL_FEATURES)

    # ── Session label ────────────────────────────────────────────────────────

    @staticmethod
    def _get_session(hour_utc: int) -> str:
        if 0 <= hour_utc < 8:
            return "tokyo"
        elif 8 <= hour_utc < 14:
            return "london"
        elif 14 <= hour_utc < 21:
            return "new_york"
        else:
            return "off_hours"

    # ── Read helpers ─────────────────────────────────────────────────────────

    def get_latest(self) -> Optional[pd.Series]:
        if self.history.empty:
            return None
        return self.history.iloc[-1]

    def get_window(self, hours: int) -> pd.DataFrame:
        bars = hours * self.bars_per_hour
        return self.history.tail(min(bars, len(self.history)))

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_snapshot(self, path: str = "./data/history.parquet") -> None:
        try:
            self.history.to_parquet(path)
            logger.info("[DataStore] Saved {} rows to {}", len(self.history), path)
        except Exception as e_parquet:
            logger.warning("[DataStore] Parquet save failed ({}), trying CSV", e_parquet)
            try:
                csv_path = path.replace(".parquet", ".csv")
                self.history.to_csv(csv_path)
                logger.info("[DataStore] Saved {} rows to {} (CSV)", len(self.history), csv_path)
            except Exception as e_csv:
                logger.error("[DataStore] Snapshot save failed: {}", e_csv)

    def load_snapshot(self, path: str = "./data/history.parquet") -> None:
        try:
            self.history = pd.read_parquet(path)
            logger.info("[DataStore] Loaded {} rows from {}", len(self.history), path)
            return
        except FileNotFoundError:
            pass
        except Exception as e_parquet:
            logger.warning("[DataStore] Parquet load failed: {}", e_parquet)

        csv_path = path.replace(".parquet", ".csv")
        try:
            self.history = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            logger.info("[DataStore] Loaded {} rows from {} (CSV)", len(self.history), csv_path)
        except FileNotFoundError:
            logger.warning("[DataStore] No snapshot at {} or {} — starting fresh", path, csv_path)
        except Exception as e_csv:
            logger.error("[DataStore] Snapshot load failed: {}", e_csv)

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def status(self) -> Dict:
        n = len(self.history)
        return {
            "rows":         n,
            "columns":      int(len(self.history.columns)) if not self.history.empty else 0,
            "maturity":     f"{min(n / self.MIN_ROWS_FOR_ZSCORE, 1.0):.0%}",
            "completeness": float(
                self._check_completeness(
                    self.history.iloc[-1],
                    feature_timestamps=None,
                    now_ts=time.time(),
                )
            ) if not self.history.empty else 0.0,
        }
