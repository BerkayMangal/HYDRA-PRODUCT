"""
features/quality.py
────────────────────
HYDRA Feature Quality & Contract System (Phase 2)

PURPOSE
-------
Eliminates the two most dangerous failure modes identified in the audit:
  1. Silent degradation: features defaulting to 0 without anyone knowing
  2. Stale data: features past their useful life being treated as live

Every feature has a CONTRACT that defines:
  - where it comes from (source tier)
  - how often it should refresh
  - what to do when it's missing or stale
  - whether it's suitable for decisions

Every inference cycle produces a QUALITY REPORT that:
  - classifies each feature as LIVE / STALE / MISSING
  - computes aggregate quality scores
  - gates signal generation (system can say "I don't know")

DESIGN PRINCIPLES
-----------------
  - No feature silently becomes 0
  - No NaN silently becomes a default
  - Missing is MISSING, not "neutral"
  - Stale is degraded, not fresh
  - The system must be able to abstain
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceTier(str, Enum):
    """Data source reliability tier (from audit trust classification)."""
    A = "A"    # Decision-grade: OKX price, FeaturePipeline OHLCV
    B = "B"    # Useful but indirect: CoinGlass, CoinGecko
    C = "C"    # Dashboard/narrative: yfinance macro, Fear&Greed, DeFi TVL
    D = "D"    # Misleading/fragile: should not enter decision path


class FeatureStatus(str, Enum):
    """Runtime status of a single feature in one inference cycle."""
    LIVE    = "live"      # Fresh data, within TTL
    STALE   = "stale"     # Past TTL but within hard-expiry; value decayed
    MISSING = "missing"   # No data available or hard-expired


class FeatureOrigin(str, Enum):
    """Where a feature's value was computed."""
    OHLCV      = "ohlcv"        # Ring buffer (1H bars)
    COLLECTOR  = "collector"    # Raw collector data, z-scored
    MACRO      = "macro"        # Macro regime features
    CALENDAR   = "calendar"     # Time-of-day, day-of-week
    DERIVED    = "derived"      # Computed from other features
    PREDICTION = "prediction"   # Prediction markets


# ---------------------------------------------------------------------------
# Feature Contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureContract:
    """
    Immutable contract defining the expected behavior of a single feature.

    Every feature in the system MUST have a contract. Features without
    contracts are rejected at the quality gate.
    """
    name: str
    description: str
    origin: FeatureOrigin
    source_tier: SourceTier
    refresh_seconds: float          # Expected update frequency
    max_staleness_seconds: float    # Hard expiry: after this, feature is MISSING
    neutral_value: float            # What "no information" looks like (NOT a default!)
    decision_eligible: bool         # Can this feature participate in signal generation?
    min_history_bars: int = 0       # Minimum 1H bars needed for valid computation

    @property
    def ttl(self) -> float:
        """Alias for max_staleness_seconds."""
        return self.max_staleness_seconds


# ---------------------------------------------------------------------------
# Feature Status Report (per-feature)
# ---------------------------------------------------------------------------

@dataclass
class FeatureStatusReport:
    """Runtime status of a single feature."""
    name: str
    status: FeatureStatus
    value: Optional[float]         # None if MISSING
    age_seconds: Optional[float]   # None if unknown
    source_tier: SourceTier
    decision_eligible: bool
    degradation_reason: Optional[str] = None

    @property
    def is_usable(self) -> bool:
        """Feature has a value (LIVE or STALE)."""
        return self.status != FeatureStatus.MISSING and self.value is not None

    @property
    def is_live(self) -> bool:
        return self.status == FeatureStatus.LIVE


# ---------------------------------------------------------------------------
# Aggregate Quality Report (per-inference-cycle)
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """
    Aggregate quality assessment for one inference cycle.

    CIRCUIT BREAKER (v4):
      GREEN  — All systems nominal. Full confidence signals.
      YELLOW — Some degradation (macro down, some collectors stale).
               Signals generated with reduced confidence.
      ORANGE — Significant degradation (only microstructure reliable).
               Signals generated with heavily reduced confidence.
      RED    — Critical failure (OKX down or <25% features).
               System abstains entirely.

    Previously this was binary (pass/fail with high thresholds).
    Result: quality gate *never* passed on Railway because macro
    features were chronically unavailable. The system was permanently
    mute despite having perfectly good microstructure data.
    """
    timestamp: float
    feature_reports: Dict[str, FeatureStatusReport] = field(default_factory=dict)

    # GREEN thresholds (ideal — all data flowing)
    GREEN_LIVE_FRACTION: float = 0.50
    GREEN_USABLE_FRACTION: float = 0.70
    GREEN_DECISION_ELIGIBLE: float = 0.60

    # YELLOW thresholds (degraded — some sources down)
    YELLOW_LIVE_FRACTION: float = 0.30
    YELLOW_USABLE_FRACTION: float = 0.45
    YELLOW_DECISION_ELIGIBLE: float = 0.40

    # ORANGE thresholds (micro-only — most external sources down)
    ORANGE_LIVE_FRACTION: float = 0.15
    ORANGE_USABLE_FRACTION: float = 0.25
    ORANGE_DECISION_ELIGIBLE: float = 0.20

    # Below ORANGE → RED (abstain)

    @property
    def total_features(self) -> int:
        return len(self.feature_reports)

    @property
    def live_features(self) -> List[str]:
        return [n for n, r in self.feature_reports.items() if r.status == FeatureStatus.LIVE]

    @property
    def stale_features(self) -> List[str]:
        return [n for n, r in self.feature_reports.items() if r.status == FeatureStatus.STALE]

    @property
    def missing_features(self) -> List[str]:
        return [n for n, r in self.feature_reports.items() if r.status == FeatureStatus.MISSING]

    @property
    def live_fraction(self) -> float:
        if self.total_features == 0:
            return 0.0
        return len(self.live_features) / self.total_features

    @property
    def usable_fraction(self) -> float:
        if self.total_features == 0:
            return 0.0
        usable = sum(1 for r in self.feature_reports.values() if r.is_usable)
        return usable / self.total_features

    @property
    def decision_eligible_usable_fraction(self) -> float:
        eligible = [r for r in self.feature_reports.values() if r.decision_eligible]
        if not eligible:
            return 0.0
        usable = sum(1 for r in eligible if r.is_usable)
        return usable / len(eligible)

    @property
    def tier_a_usable_fraction(self) -> float:
        """Fraction of Tier-A (decision-grade) features that are usable."""
        tier_a = [r for r in self.feature_reports.values()
                  if r.source_tier == SourceTier.A]
        if not tier_a:
            return 0.0
        return sum(1 for r in tier_a if r.is_usable) / len(tier_a)

    @property
    def circuit_breaker_level(self) -> str:
        """
        Determine circuit breaker level.

        Returns: 'GREEN', 'YELLOW', 'ORANGE', or 'RED'
        """
        lf = self.live_fraction
        uf = self.usable_fraction
        de = self.decision_eligible_usable_fraction

        if (lf >= self.GREEN_LIVE_FRACTION
                and uf >= self.GREEN_USABLE_FRACTION
                and de >= self.GREEN_DECISION_ELIGIBLE):
            return "GREEN"

        if (lf >= self.YELLOW_LIVE_FRACTION
                and uf >= self.YELLOW_USABLE_FRACTION
                and de >= self.YELLOW_DECISION_ELIGIBLE):
            return "YELLOW"

        if (lf >= self.ORANGE_LIVE_FRACTION
                and uf >= self.ORANGE_USABLE_FRACTION
                and de >= self.ORANGE_DECISION_ELIGIBLE):
            return "ORANGE"

        return "RED"

    @property
    def can_generate_signal(self) -> bool:
        """Master gate: should the system produce a signal this cycle?

        v4: True for GREEN, YELLOW, and ORANGE. Only RED causes abstain.
        """
        return self.circuit_breaker_level != "RED"

    @property
    def abstain_reason(self) -> Optional[str]:
        level = self.circuit_breaker_level
        if level != "RED":
            return None
        reasons = []
        if self.live_fraction < self.ORANGE_LIVE_FRACTION:
            reasons.append(
                f"live={self.live_fraction:.0%}<{self.ORANGE_LIVE_FRACTION:.0%}"
            )
        if self.usable_fraction < self.ORANGE_USABLE_FRACTION:
            reasons.append(
                f"usable={self.usable_fraction:.0%}<{self.ORANGE_USABLE_FRACTION:.0%}"
            )
        if self.decision_eligible_usable_fraction < self.ORANGE_DECISION_ELIGIBLE:
            reasons.append(
                f"decision_eligible_usable={self.decision_eligible_usable_fraction:.0%}"
                f"<{self.ORANGE_DECISION_ELIGIBLE:.0%}"
            )
        return "; ".join(reasons)

    def summary(self) -> str:
        level = self.circuit_breaker_level
        return (
            f"quality[{level}]: live={len(self.live_features)}/{self.total_features} "
            f"stale={len(self.stale_features)} missing={len(self.missing_features)} "
            f"| can_signal={self.can_generate_signal}"
        )

    def confidence_multiplier(self) -> float:
        """
        Score multiplier based on data quality AND circuit breaker level.

        GREEN  → 0.85–1.0
        YELLOW → 0.55–0.85
        ORANGE → 0.30–0.55
        RED    → 0 (should never be called, system abstains)
        """
        level = self.circuit_breaker_level
        base = self.usable_fraction
        stale_penalty = len(self.stale_features) * 0.3 / max(self.total_features, 1)
        raw = max(0.0, min(1.0, base - stale_penalty))

        if level == "GREEN":
            return max(0.85, raw)
        elif level == "YELLOW":
            return max(0.55, min(0.85, raw))
        elif level == "ORANGE":
            return max(0.30, min(0.55, raw))
        else:
            return 0.0


# ---------------------------------------------------------------------------
# Feature Freshness Tracker
# ---------------------------------------------------------------------------

class FreshnessTracker:
    """
    Tracks last-update timestamps for every feature.

    Used by the quality system to classify features as LIVE / STALE / MISSING.
    """

    def __init__(self) -> None:
        self._timestamps: Dict[str, float] = {}

    def mark_updated(self, feature_name: str, ts: Optional[float] = None) -> None:
        self._timestamps[feature_name] = ts or time.time()

    def mark_batch_updated(self, feature_names: List[str], ts: Optional[float] = None) -> None:
        t = ts or time.time()
        for name in feature_names:
            self._timestamps[name] = t

    def get_age(self, feature_name: str) -> Optional[float]:
        """Age in seconds, or None if never seen."""
        ts = self._timestamps.get(feature_name)
        if ts is None:
            return None
        return time.time() - ts

    def classify(
        self,
        contract: FeatureContract,
        value: Optional[float],
    ) -> FeatureStatusReport:
        """
        Classify a feature's runtime status against its contract.

        Returns a FeatureStatusReport with status, value, and degradation reason.
        """
        age = self.get_age(contract.name)

        # Never seen this feature
        if age is None or value is None:
            return FeatureStatusReport(
                name=contract.name,
                status=FeatureStatus.MISSING,
                value=None,
                age_seconds=age,
                source_tier=contract.source_tier,
                decision_eligible=contract.decision_eligible,
                degradation_reason="never_received" if age is None else "value_is_none",
            )

        # Check for NaN/inf
        try:
            fval = float(value)
            if np.isnan(fval) or np.isinf(fval):
                return FeatureStatusReport(
                    name=contract.name,
                    status=FeatureStatus.MISSING,
                    value=None,
                    age_seconds=age,
                    source_tier=contract.source_tier,
                    decision_eligible=contract.decision_eligible,
                    degradation_reason=f"value_is_{'nan' if np.isnan(fval) else 'inf'}",
                )
        except (TypeError, ValueError):
            return FeatureStatusReport(
                name=contract.name,
                status=FeatureStatus.MISSING,
                value=None,
                age_seconds=age,
                source_tier=contract.source_tier,
                decision_eligible=contract.decision_eligible,
                degradation_reason=f"unparseable_value: {type(value).__name__}",
            )

        # Fresh
        if age <= contract.max_staleness_seconds:
            return FeatureStatusReport(
                name=contract.name,
                status=FeatureStatus.LIVE,
                value=fval,
                age_seconds=age,
                source_tier=contract.source_tier,
                decision_eligible=contract.decision_eligible,
            )

        # Stale but not expired (within 3× TTL)
        hard_expiry = contract.max_staleness_seconds * 3.0
        if age <= hard_expiry:
            return FeatureStatusReport(
                name=contract.name,
                status=FeatureStatus.STALE,
                value=fval,
                age_seconds=age,
                source_tier=contract.source_tier,
                decision_eligible=contract.decision_eligible,
                degradation_reason=f"stale: age={age:.0f}s > ttl={contract.max_staleness_seconds:.0f}s",
            )

        # Hard expired
        return FeatureStatusReport(
            name=contract.name,
            status=FeatureStatus.MISSING,
            value=None,
            age_seconds=age,
            source_tier=contract.source_tier,
            decision_eligible=contract.decision_eligible,
            degradation_reason=f"expired: age={age:.0f}s > 3×ttl={hard_expiry:.0f}s",
        )


# ---------------------------------------------------------------------------
# Safe Feature Access (replaces features.get(x, 0))
# ---------------------------------------------------------------------------

class SafeFeatureAccessor:
    """
    Wraps a feature dict + quality report to provide safe access.

    Replaces the dangerous pattern:
        value = features.get("some_feature", 0)  # silent default!

    With:
        value = accessor.get("some_feature")      # returns None if missing
        value = accessor.require("some_feature")   # raises if missing
    """

    def __init__(
        self,
        values: Dict[str, float],
        quality: QualityReport,
    ) -> None:
        self._values = values
        self._quality = quality
        self._accessed: Set[str] = set()
        self._missing_accessed: Set[str] = set()

    def get(self, name: str) -> Optional[float]:
        """
        Get feature value. Returns None if missing/expired — NEVER a silent default.
        """
        self._accessed.add(name)
        report = self._quality.feature_reports.get(name)
        if report is None or not report.is_usable:
            self._missing_accessed.add(name)
            return None
        return report.value

    def get_or(self, name: str, fallback: float) -> Tuple[float, bool]:
        """
        Get feature value with explicit fallback.

        Returns (value, is_real) — caller MUST use is_real to adjust confidence.
        """
        val = self.get(name)
        if val is None:
            return fallback, False
        return val, True

    def require(self, name: str) -> float:
        """Get feature value. Raises ValueError if missing."""
        val = self.get(name)
        if val is None:
            raise ValueError(f"Required feature '{name}' is missing or expired")
        return val

    @property
    def missing_accessed(self) -> Set[str]:
        """Features that were accessed but were missing."""
        return self._missing_accessed.copy()

    @property
    def access_completeness(self) -> float:
        """Fraction of accessed features that were available."""
        if not self._accessed:
            return 1.0
        available = len(self._accessed) - len(self._missing_accessed)
        return available / len(self._accessed)
