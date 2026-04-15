"""
engines/output.py
──────────────────
HYDRA Structured Engine Output (Phase 3, Steps 1-3)

Every Layer 1 engine returns an EngineOutput instead of a loose dict.
This enforces:
  - Explicit confidence derived from data quality + signal coherence
  - Classification of every sub-signal as DECISION / SECONDARY / CONTEXT
  - Per-engine suppression with explicit reason
  - No more vague scores with hidden assumptions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Signal grade classification
# ---------------------------------------------------------------------------

class SignalGrade(str, Enum):
    """
    How much influence a sub-signal may have on the final score.

    DECISION:  Can materially move the directional score.
               Must come from Tier A/B data with real-time freshness.
    SECONDARY: May adjust confidence or add minor score nudge (capped).
               Useful but indirect evidence.
    CONTEXT:   Dashboard/narrative only. MUST NOT affect score.
               Includes LLM output, Fear & Greed, prediction markets.
    """
    DECISION  = "decision"
    SECONDARY = "secondary"
    CONTEXT   = "context"


# ---------------------------------------------------------------------------
# Sub-signal output
# ---------------------------------------------------------------------------

@dataclass
class SubSignal:
    """One sub-signal within an engine's computation."""
    name: str
    raw_value: float              # The computed value before capping
    capped_value: float           # After np.clip to [-100, 100]
    grade: SignalGrade
    weight: float                 # Engine-internal weight for this signal
    is_real: bool                 # True if computed from live data (not default/missing)
    note: str = ""                # Optional explanation

    @property
    def weighted_contribution(self) -> float:
        """Score contribution = capped_value × weight, but only if real."""
        if not self.is_real:
            return 0.0
        return self.capped_value * self.weight

    @property
    def is_active(self) -> bool:
        return self.is_real and abs(self.capped_value) > 5.0


# ---------------------------------------------------------------------------
# Engine confidence
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    """Engine-level confidence — derived, not guessed."""
    HIGH   = "HIGH"     # ≥70% decision-grade signals real, coherent direction
    MEDIUM = "MEDIUM"   # ≥50% decision-grade signals real, mostly coherent
    LOW    = "LOW"      # <50% decision-grade signals real, or contradictory
    NONE   = "NONE"     # Engine suppressed


def _compute_confidence(
    signals: List[SubSignal],
    suppressed: bool,
) -> ConfidenceLevel:
    """
    Derive confidence from signal quality and coherence.

    NOT a vague number. Derived from:
      1. What fraction of decision-grade signals are real (have live data)
      2. Whether active signals agree on direction (coherence)
    """
    if suppressed:
        return ConfidenceLevel.NONE

    decision_signals = [s for s in signals if s.grade == SignalGrade.DECISION]
    if not decision_signals:
        return ConfidenceLevel.NONE

    # 1. Decision-grade completeness
    real_decision = sum(1 for s in decision_signals if s.is_real)
    decision_completeness = real_decision / len(decision_signals)

    # 2. Directional coherence among active decision-grade signals
    active_decision = [s for s in decision_signals if s.is_active]
    if len(active_decision) >= 2:
        signs = {np.sign(s.capped_value) for s in active_decision}
        coherent = len(signs) == 1 and 0.0 not in signs
    elif len(active_decision) == 1:
        coherent = True
    else:
        coherent = False  # No active signals

    # Classification
    if decision_completeness >= 0.70 and coherent:
        return ConfidenceLevel.HIGH
    elif decision_completeness >= 0.50 and coherent:
        return ConfidenceLevel.MEDIUM
    elif decision_completeness >= 0.50:
        return ConfidenceLevel.MEDIUM  # data ok but mixed signals
    else:
        return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Main engine output
# ---------------------------------------------------------------------------

@dataclass
class EngineOutput:
    """
    Structured output from a Layer 1 engine.

    Replaces the old loose dict with enforced schema.
    """
    engine_name: str
    raw_score: float                         # Before any capping
    score: float                             # Clipped to [-100, 100]
    direction: str                           # LONG / SHORT / NEUTRAL
    confidence: ConfidenceLevel
    access_completeness: float               # From FeatureAccessMixin
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    signals: List[SubSignal] = field(default_factory=list)
    regime: str = ""                         # MacroEngine only
    event_outcome: str = ""                  # MacroEngine only
    event_dampen_factor: float = 1.0         # MacroEngine only

    @property
    def decision_grade_completeness(self) -> float:
        """Fraction of decision-grade signals with real data."""
        dec = [s for s in self.signals if s.grade == SignalGrade.DECISION]
        if not dec:
            return 0.0
        return sum(1 for s in dec if s.is_real) / len(dec)

    @property
    def active_signal_count(self) -> int:
        return sum(1 for s in self.signals if s.is_active)

    @property
    def signal_summary(self) -> Dict[str, float]:
        return {s.name: round(s.capped_value, 2) for s in self.signals}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_name,
            "score": round(self.score, 2),
            "direction": self.direction,
            "confidence": self.confidence.value,
            "access_completeness": round(self.access_completeness, 2),
            "decision_grade_completeness": round(self.decision_grade_completeness, 2),
            "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
            "signals": self.signal_summary,
            "active_signals": self.active_signal_count,
            "regime": self.regime,
            "event_outcome": self.event_outcome,
        }

    # Legacy compat: engines dict access
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


# ---------------------------------------------------------------------------
# Builder helper for engines
# ---------------------------------------------------------------------------

def build_engine_output(
    engine_name: str,
    signals: List[SubSignal],
    access_completeness: float,
    *,
    decision_score_threshold: float = 15.0,
    agreement_boost: float = 1.3,
    # Suppression criteria
    min_decision_completeness: float = 0.30,
    # MacroEngine extras
    regime: str = "",
    event_outcome: str = "",
    event_dampen_factor: float = 1.0,
) -> EngineOutput:
    """
    Standard builder: computes score, direction, confidence, suppression.

    Suppression rules (Step 3):
      - If <30% of decision-grade signals are real → suppress
      - If engine has zero active signals → suppress (nothing to say)
    """
    # Compute raw score: only decision + secondary contribute to score
    decision_contribution = sum(
        s.weighted_contribution for s in signals
        if s.grade in (SignalGrade.DECISION, SignalGrade.SECONDARY)
    )
    # v9 FIX: Secondary cap raised 30%→60%. At 30% the Flow engine (1 DECISION
    # signal) was capped so hard it could never produce a meaningful score.
    decision_only = sum(
        s.weighted_contribution for s in signals
        if s.grade == SignalGrade.DECISION
    )
    secondary_only = sum(
        s.weighted_contribution for s in signals
        if s.grade == SignalGrade.SECONDARY
    )
    if abs(decision_only) > 0:
        secondary_cap = abs(decision_only) * 0.60
        if abs(secondary_only) > secondary_cap:
            secondary_only = np.sign(secondary_only) * secondary_cap
    raw_score = decision_only + secondary_only

    # Agreement boost: ≥3 strong signals same direction
    strong = [s for s in signals if s.is_active and abs(s.capped_value) > 25
              and s.grade == SignalGrade.DECISION]
    if len(strong) >= 3:
        signs = {np.sign(s.capped_value) for s in strong}
        if len(signs) == 1:
            raw_score *= agreement_boost

    score = float(np.clip(raw_score, -100, 100))

    # Suppression check
    dec_signals = [s for s in signals if s.grade == SignalGrade.DECISION]
    dec_real = sum(1 for s in dec_signals if s.is_real)
    dec_completeness = dec_real / max(len(dec_signals), 1)

    active_count = sum(1 for s in signals if s.is_active)

    suppressed = False
    suppression_reason = None

    if dec_completeness < min_decision_completeness:
        suppressed = True
        suppression_reason = (
            f"decision_grade_completeness={dec_completeness:.0%}"
            f"<{min_decision_completeness:.0%}"
        )
        score = 0.0
    elif active_count == 0:
        suppressed = True
        suppression_reason = "no_active_signals"
        score = 0.0

    # Direction
    if suppressed:
        direction = "NEUTRAL"
    elif abs(score) >= decision_score_threshold:
        direction = "LONG" if score > 0 else "SHORT"
    else:
        direction = "NEUTRAL"

    confidence = _compute_confidence(signals, suppressed)

    return EngineOutput(
        engine_name=engine_name,
        raw_score=round(raw_score, 2),
        score=round(score, 2),
        direction=direction,
        confidence=confidence,
        access_completeness=round(access_completeness, 2),
        suppressed=suppressed,
        suppression_reason=suppression_reason,
        signals=signals,
        regime=regime,
        event_outcome=event_outcome,
        event_dampen_factor=event_dampen_factor,
    )
