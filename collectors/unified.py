"""Immutable observations, source-time TTLs and per-feature normalization.

Raw market facts are never decayed. Repeated reads of one observation do not
add statistical samples, and expired values cannot reappear through z-scores.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger


class UnifiedDataStore:
    """Central feature store with independent source-time histories."""

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
            "[DataStore] TTL micro={}s flow={}s macro={}s pred_mkt={}s very_slow={}s | immutable observations",
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
        now = datetime.now(timezone.utc)
        now_ts = time.time()
        # Never modify the shared raw observation, including its candle timestamp.
        collector_data = dict(collector_data)
        collector_data["computed_at"] = now
        collector_data["hour_utc"] = now.hour
        collector_data["day_of_week"] = now.weekday()
        collector_data["session_label"] = self._get_session(now.hour)
        source_times = dict(feature_timestamps or {})
        if feature_timestamps is None:
            source_times = {key: now_ts for key in collector_data}

        # Quality affects availability, never the numeric value of a market fact.
        observations = getattr(self, "_observations", {})
        any_new = False
        for feature in self._feature_to_group:
            value = collector_data.get(feature)
            ts = source_times.get(feature)
            try:
                valid = ts is not None and -5 <= now_ts - ts <= self._get_feature_ttl(feature) and np.isfinite(float(value))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                if feature in collector_data:
                    collector_data[feature] = np.nan
                continue
            history = observations.setdefault(feature, [])
            if not history or ts > history[-1][0]:
                history.append((ts, float(value)))
                any_new = True
            # An identical source time is not another statistical observation.
            hours = self.norm_windows.get(self._feature_to_group[feature], 24)
            observations[feature] = [(t, v) for t, v in history if t >= now_ts - hours * 3600]
        self._observations = observations
        self._latest_source_times = source_times

        # ── Append to history ────────────────────────────────────────────
        row      = pd.Series(collector_data)
        row.name = now

        if any_new or self.history.empty:
            self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=False)

        max_rows = 30 * 24 * self.bars_per_hour
        if len(self.history) > max_rows:
            self.history = self.history.iloc[-max_rows:]

        normalized = self._normalize_latest(row)
        combined   = pd.concat([row, normalized])

        n_rows = len(self.history)
        completeness = self._check_completeness(combined, feature_timestamps, now_ts)
        combined["feature_completeness"] = completeness
        combined["data_maturity"]        = min(n_rows / self.MIN_ROWS_FOR_ZSCORE, 1.0)

        logger.debug(
            "[DataStore] Rows:{} | Features:{} | Completeness:{:.0%} | Maturity:{:.0%}",
            n_rows, len(combined), completeness, combined["data_maturity"],
        )

        normalized_times = {name: source_times.get(name.removesuffix("_zscore").removesuffix("_pctrank"))
                            for name in normalized.index}
        combined.attrs["source_timestamps"] = {**source_times, **normalized_times}
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

    def _normalize_latest(self, row: pd.Series) -> pd.Series:
        normalized = {}
        for feature, history in self._observations.items():
            current = row.get(feature)
            try:
                if not np.isfinite(float(current)):
                    continue
            except (TypeError, ValueError):
                continue
            # Do not resurrect the last valid value when the latest one expired.
            values = np.array([value for _, value in history], dtype=float)
            if not len(values):
                continue
            if len(values) >= self.MIN_ROWS_FOR_ZSCORE:
                std = values.std(ddof=1)
                scaled = float((current - values.mean()) / std) if std > 0 else 0.0
                rank = float((values < current).mean())
            else:
                scaled, rank = self._scale_raw(feature, float(current)), 0.5
            normalized[feature + "_zscore"] = scaled
            normalized[feature + "_pctrank"] = rank
        return pd.Series(normalized, dtype=float)

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
            if feature_timestamps is not None:
                ts = feature_timestamps.get(feat)
                if ts is None or not -5 <= now_ts - ts <= self._get_feature_ttl(feat):
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
                    feature_timestamps=getattr(self, "_latest_source_times", {}),
                    now_ts=time.time(),
                )
            ) if not self.history.empty else 0.0,
        }
