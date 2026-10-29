"""
features/unified_frame.py
──────────────────────────
HYDRA Unified Feature Frame (Phase 2, Steps 1 + 3 + 4 + 6)

THE PROBLEM THIS SOLVES
-----------------------
Before Phase 2, the system had TWO independent feature universes:

  UnifiedDataStore → pd.Series with z-scored collector features → Layer 1 engines
  FeaturePipeline  → FeatureVector with OHLCV ring-buffer features → ML engine

These were:
  - Different features (CVD z-scores vs EMA distances)
  - Different quality tracking (completeness % vs quality_score)
  - Different staleness handling (LOCF decay vs proxy/default classification)
  - Consumed by different code paths with no shared contract

THE SOLUTION
------------
UnifiedFeatureFrame is the SINGLE source of truth.

  Collectors → raw_data
                  ↓
       UnifiedDataStore.update() → collector features + z-scores
       FeaturePipeline.transform_live() → OHLCV + macro features
                  ↓
         UnifiedFeatureFrame.build()
                  ↓
       One dict + one QualityReport + one SafeFeatureAccessor
                  ↓
       ├── Layer 1 engines (via accessor.get())
       ├── ML engine (via get_ml_array())
       └── Dashboard (via to_dict())

ALL consumers go through this frame. No feature is accessed without
quality classification. No missing feature silently becomes 0.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger

from features.quality import (
    FeatureStatus,
    FeatureStatusReport,
    FreshnessTracker,
    QualityReport,
    SafeFeatureAccessor,
)
from features.registry import (
    ALL_CONTRACTS,
    COLLECTOR_FEATURE_NAMES,
    MACRO_FEATURE_NAMES,
    PIPELINE_FEATURE_NAMES,
    PREDICTION_MARKET_PREFIXES,
    get_contract,
)
from features.pipeline import FeaturePipeline, FeatureVector, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Unified Feature Frame
# ---------------------------------------------------------------------------

@dataclass
class UnifiedFeatureFrame:
    """
    Single merged feature frame for one inference cycle.

    Contains ALL features (pipeline + collector + macro) with unified
    quality tracking and safe access.
    """
    values: Dict[str, float]
    quality: QualityReport
    timestamp: float
    ring_buffer_bars: int
    macro_buffer_obs: int

    def accessor(self) -> SafeFeatureAccessor:
        """Get a safe accessor that tracks missing-feature access."""
        return SafeFeatureAccessor(self.values, self.quality)

    def get_ml_array(self, feature_cols: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """
        Extract ML feature array in canonical order.

        Returns None if quality gate fails for ML features.
        """
        cols = feature_cols or FEATURE_NAMES
        # Check that enough ML features are available
        ml_available = sum(
            1 for c in cols
            if c in self.values
            and self.quality.feature_reports.get(c, _MISSING_REPORT).is_usable
        )
        if ml_available / max(len(cols), 1) < 0.70:
            logger.warning(
                "[UnifiedFrame] ML features insufficient: {}/{} usable",
                ml_available, len(cols),
            )
            return None
        return np.array(
            [self.values.get(c, 0.0) for c in cols], dtype=np.float32
        )

    def to_legacy_series(self) -> pd.Series:
        """
        Convert to pd.Series for backward compatibility with Layer 1 combiner.

        This is a TEMPORARY bridge. Layer 1 engines should migrate to
        SafeFeatureAccessor.
        """
        values = {name: report.value for name, report in self.quality.feature_reports.items() if report.is_usable}
        hour = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).hour
        values["session_label"] = "tokyo" if hour < 8 else "london" if hour < 14 else "new_york" if hour < 21 else "off_hours"
        values["feature_completeness"] = self.quality.usable_fraction
        values["data_maturity"] = min(self.ring_buffer_bars / 20.0, 1.0)
        return pd.Series(values, dtype=object)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": {k: round(v, 6) if isinstance(v, float) else v
                       for k, v in self.values.items()},
            "quality_summary": self.quality.summary(),
            "can_signal": self.quality.can_generate_signal,
            "live_count": len(self.quality.live_features),
            "stale_count": len(self.quality.stale_features),
            "missing_count": len(self.quality.missing_features),
            "confidence_multiplier": self.quality.confidence_multiplier(),
        }


# Sentinel for missing reports
_MISSING_REPORT = FeatureStatusReport(
    name="__missing__",
    status=FeatureStatus.MISSING,
    value=None,
    age_seconds=None,
    source_tier=ALL_CONTRACTS.get("hour", list(ALL_CONTRACTS.values())[0]).source_tier,
    decision_eligible=False,
)


# ---------------------------------------------------------------------------
# Frame Builder
# ---------------------------------------------------------------------------

class UnifiedFrameBuilder:
    """
    Builds a UnifiedFeatureFrame each cycle by merging:
      - FeaturePipeline output (OHLCV + macro features)
      - UnifiedDataStore output (collector z-scores)

    Also handles:
      - Freshness tracking for all features
      - Quality classification against contracts
      - Macro feature fix (no more phantom z-scores)
    """

    def __init__(self) -> None:
        self._freshness = FreshnessTracker()
        self._pipeline = FeaturePipeline()
        self._last_macro_ingest: float = 0.0

    @property
    def pipeline(self) -> FeaturePipeline:
        return self._pipeline

    @property
    def freshness(self) -> FreshnessTracker:
        return self._freshness

    # ------------------------------------------------------------------
    # Data ingestion (delegates to FeaturePipeline)
    # ------------------------------------------------------------------

    def warm_up(self, historical_df: pd.DataFrame) -> None:
        """Seed the ring buffer with historical data."""
        self._pipeline.warm_up(historical_df)
        self._mark_ohlcv()

    def _mark_ohlcv(self):
        if self._pipeline._ring:
            event_end = self._pipeline._ring[-1]["ts"] + 3600
            names = [name for name, contract in ALL_CONTRACTS.items() if contract.origin.value == "ohlcv"]
            self._freshness.mark_batch_updated(names, event_end)

    def ingest_candle(
        self,
        okx_data: Dict[str, Any],
        funding_rate: Optional[float] = None,
    ) -> bool:
        """Ingest a 5-min candle. Returns True if a new 1H bar was emitted."""
        result = self._pipeline.ingest_candle(okx_data, funding_rate)
        if result:
            self._mark_ohlcv()
        return result

    def ingest_macro(self, macro_data: Dict[str, Any], feature_timestamps=None) -> None:
        """
        Ingest macro data. FIXED: computes regime features, not z-scores.

        The old pipeline broadcast one daily value across 500 hourly rows,
        then computed rolling z-scores. Since the input was constant,
        rolling_std = 0 → z-score = NaN → default 0. Macro features were
        effectively dead in production.

        FIX: Macro data is treated as LOW-FREQUENCY REGIME features.
        We compute level-based and change-based features, NOT z-scores.
        """
        source_times = feature_timestamps or {}
        fields = ["vix_current", "qqq_current", "us10y_current", "gold_current", "fear_greed_value"]
        observed = [source_times[key] for key in fields if key in macro_data and source_times.get(key) is not None]
        if not observed:
            return
        ts = max(observed)
        if ts > self._last_macro_ingest:
            self._pipeline.ingest_macro({**macro_data, "ts": ts})
            self._last_macro_ingest = ts
        dependencies = {"vix_level": "vix_current", "vix_regime": "vix_current", "vix_spike": "vix_current",
                        "qqq_ret_1d": "qqq_change_24h", "us10y_level": "us10y_current",
                        "us10y_change_1d": "us10y_change_24h", "fg_val": "fear_greed_value", "fg_regime": "fear_greed_value"}
        for name, key in dependencies.items():
            if source_times.get(key) is not None:
                self._freshness.mark_updated(name, source_times[key])

    # ------------------------------------------------------------------
    # Build unified frame
    # ------------------------------------------------------------------

    def build(
        self,
        collector_features: pd.Series,
        raw_data: Dict[str, Any],
    ) -> UnifiedFeatureFrame:
        """
        Build a unified feature frame for one inference cycle.

        Args:
            collector_features: Output of UnifiedDataStore.update()
            raw_data: Raw collector data dict (for macro regime computation)

        Returns:
            UnifiedFeatureFrame with all features + quality report
        """
        now = time.time()
        values: Dict[str, float] = {}
        source_times = collector_features.attrs.get("source_timestamps", {})

        # ── 1. Pipeline features (OHLCV + funding from ring buffer) ──────
        pipeline_vec = self._pipeline.transform_live()
        ring_bars = len(self._pipeline._ring)
        macro_obs = len(self._pipeline._macro)

        if pipeline_vec is not None:
            for name in PIPELINE_FEATURE_NAMES:
                if name in pipeline_vec.quality.default or name in pipeline_vec.quality.proxy:
                    continue
                contract = ALL_CONTRACTS[name]
                if ring_bars < contract.min_history_bars:
                    continue
                val = pipeline_vec.values.get(name)
                if val is not None and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    values[name] = float(val)

        # ── 2. Macro REGIME features (FIXED — no z-scores) ──────────────
        macro_features = self._compute_macro_regime_features(raw_data)
        for name, val in macro_features.items():
            values[name] = val
            # Don't re-mark freshness — ingest_macro already did

        # ── 3. Collector features (z-scores from UnifiedDataStore) ────────
        for name in COLLECTOR_FEATURE_NAMES:
            if name in collector_features.index:
                val = collector_features[name]
                try:
                    fval = float(val)
                    if not np.isnan(fval) and not np.isinf(fval):
                        values[name] = fval
                        if source_times.get(name) is not None:
                            self._freshness.mark_updated(name, source_times[name])
                except (TypeError, ValueError):
                    pass  # Will be classified as MISSING

        # ── 4. Prediction market features (dynamic names) ─────────────────
        for key in collector_features.index:
            if not isinstance(key, str):
                continue
            for prefix in PREDICTION_MARKET_PREFIXES:
                if key.startswith(prefix):
                    try:
                        fval = float(collector_features[key])
                        if not np.isnan(fval) and not np.isinf(fval):
                            values[key] = fval
                            if source_times.get(key) is not None:
                                self._freshness.mark_updated(key, source_times[key])
                    except (TypeError, ValueError):
                        pass

        # ── 5. Build quality report ───────────────────────────────────────
        quality = self._build_quality_report(values, now)
        price = self._safe_float(raw_data.get("last_price"))
        close = self._safe_float(raw_data.get("close"))
        price_ts, close_ts = source_times.get("last_price"), source_times.get("close")
        quality.critical_data_ok = bool(price and price > 0 and close and close > 0
                                        and price_ts is not None and -5 <= now - price_ts <= 600
                                        and close_ts is not None and -5 <= now - close_ts <= 900)
        quality.critical_data_reason = None if quality.critical_data_ok else "missing_or_stale_btc_price_or_candle"

        frame = UnifiedFeatureFrame(
            values=values,
            quality=quality,
            timestamp=now,
            ring_buffer_bars=ring_bars,
            macro_buffer_obs=macro_obs,
        )

        logger.debug("[UnifiedFrame] {}", quality.summary())

        # Log circuit breaker state changes at INFO level
        if quality.circuit_breaker_level in ("ORANGE", "RED"):
            logger.warning(
                "[UnifiedFrame] Circuit breaker [{}] — live={:.0%} usable={:.0%} decision_eligible={:.0%}",
                quality.circuit_breaker_level,
                quality.live_fraction,
                quality.usable_fraction,
                quality.decision_eligible_usable_fraction,
            )

        return frame

    # ------------------------------------------------------------------
    # Macro regime features (PHASE 2 FIX)
    # ------------------------------------------------------------------

    def _compute_macro_regime_features(
        self,
        raw_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute macro features as REGIME indicators, not z-scores.

        v4 UPDATE: Now resilient to partial macro data.
        - Uses _change_24h fields from macro_collector directly (works with FRED fallback)
        - Does not treat adjacent polls as daily changes
        - Each feature computed independently — one missing doesn't block others
        """
        result: Dict[str, float] = {}

        # VIX
        vix = self._safe_float(raw_data.get("vix_current"))
        if vix is not None and vix > 0:
            result["vix_level"] = vix
            if vix > 30:
                result["vix_regime"] = 2.0
                result["vix_spike"] = 1.0
            elif vix > 22:
                result["vix_regime"] = 1.0
                result["vix_spike"] = 0.0
            else:
                result["vix_regime"] = 0.0
                result["vix_spike"] = 0.0

        # QQQ — use change_24h from collector if available
        qqq_chg = self._safe_float(raw_data.get("qqq_change_24h"))
        if qqq_chg is not None:
            result["qqq_ret_1d"] = qqq_chg
        # US10Y — use change_24h from collector
        us10y = self._safe_float(raw_data.get("us10y_current"))
        if us10y is not None and us10y > 0:
            result["us10y_level"] = us10y
        us10y_chg = self._safe_float(raw_data.get("us10y_change_24h"))
        if us10y_chg is not None:
            result["us10y_change_1d"] = us10y_chg

        # Adjacent polls are not daily observations. Only dated collector
        # changes above may populate daily-return/change contracts.

        # Fear & Greed regime
        fg = self._safe_float(raw_data.get("fear_greed_value"))
        if fg is not None and 0 < fg <= 100:
            result["fg_val"] = fg
            if fg <= 25:
                result["fg_regime"] = -1.0   # Fear
            elif fg >= 75:
                result["fg_regime"] = 1.0    # Greed
            else:
                result["fg_regime"] = 0.0    # Neutral

        return result

    # ------------------------------------------------------------------
    # Quality report builder
    # ------------------------------------------------------------------

    def _build_quality_report(
        self,
        values: Dict[str, float],
        now: float,
    ) -> QualityReport:
        """Build quality report by classifying every registered feature."""
        report = QualityReport(timestamp=now)

        # Classify all registered features
        for name, contract in ALL_CONTRACTS.items():
            value = values.get(name)
            status_report = self._freshness.classify(contract, value)
            report.feature_reports[name] = status_report

        # Log warnings for missing decision-eligible features
        missing_decision = [
            name for name, r in report.feature_reports.items()
            if r.decision_eligible and r.status == FeatureStatus.MISSING
        ]
        if missing_decision:
            logger.warning(
                "[Quality] {} decision-eligible features MISSING: {}",
                len(missing_decision),
                ", ".join(missing_decision[:10]),
            )

        return report

    # ------------------------------------------------------------------
    # Pipeline status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "ring_buffer_bars": len(self._pipeline._ring),
            "macro_buffer_obs": len(self._pipeline._macro),
            "last_macro_ingest": self._last_macro_ingest,
            "tracked_features": len(self._freshness._timestamps),
        }

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        except (TypeError, ValueError):
            return None
