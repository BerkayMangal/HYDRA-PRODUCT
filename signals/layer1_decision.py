"""
signals/layer1_decision.py
───────────────────────────
HYDRA Layer 1 Decision Engine (Phase 3, Steps 4 + 5 + 7)

Replaces the loose combiner + separate decision layer with a single,
disciplined decision engine.

WHAT CHANGED FROM PHASE 2
--------------------------
1. Engines return EngineOutput (structured, with confidence + grades)
2. Combiner uses confidence-weighted scoring, not flat weight × score
3. Conflict detection: engines disagreeing → score penalty + low confidence
4. NO_SIGNAL vs NEUTRAL distinction:
     NO_SIGNAL = epistemic abstention (insufficient evidence)
     NEUTRAL   = valid view that market is balanced
5. Decision explanation payload for every cycle (observability)
6. Narrative contamination explicitly blocked (no Pulse/LLM influence)

DECISION STATE MACHINE
----------------------
  Input: 3 engine outputs + quality report
    │
    ├── ALL engines suppressed? → NO_SIGNAL (insufficient evidence)
    ├── Quality gate fails?     → NO_SIGNAL (data too degraded)
    ├── Pre-event blackout?     → NO_SIGNAL (event uncertainty)
    ├── High conflict?          → NO_SIGNAL (engines contradict)
    │
    ├── Score below threshold?  → NEUTRAL (market is balanced)
    ├── Score above threshold + low confidence?  → WEAK_BULLISH / WEAK_BEARISH
    └── Score above threshold + med/high conf?   → BULLISH / BEARISH
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from engines.output import EngineOutput, ConfidenceLevel, SignalGrade


# ---------------------------------------------------------------------------
# Decision states
# ---------------------------------------------------------------------------

class DecisionState(str, Enum):
    """
    Final decision states — NO_SIGNAL and NEUTRAL are distinct.

    NO_SIGNAL:     System cannot form a view. Do not act.
    NEUTRAL:       System has a view: market is balanced. Stay flat intentionally.
    WEAK_BULLISH:  Directional lean, low confidence. Informational.
    WEAK_BEARISH:  Directional lean, low confidence. Informational.
    BULLISH:       Directional signal, medium+ confidence.
    BEARISH:       Directional signal, medium+ confidence.
    """
    NO_SIGNAL    = "NO_SIGNAL"
    NEUTRAL      = "NEUTRAL"
    WEAK_BULLISH = "WEAK_BULLISH"
    WEAK_BEARISH = "WEAK_BEARISH"
    BULLISH      = "BULLISH"
    BEARISH      = "BEARISH"

    @property
    def is_directional(self) -> bool:
        return self in (DecisionState.WEAK_BULLISH, DecisionState.WEAK_BEARISH,
                        DecisionState.BULLISH, DecisionState.BEARISH)

    @property
    def is_actionable(self) -> bool:
        """Only BULLISH/BEARISH are strong enough for delivery."""
        return self in (DecisionState.BULLISH, DecisionState.BEARISH)


# ---------------------------------------------------------------------------
# Decision explanation (Step 7 — observability)
# ---------------------------------------------------------------------------

@dataclass
class DecisionExplanation:
    """
    Structured explanation for every decision cycle.
    Suitable for logs, dashboard, postmortem review.
    """
    timestamp_utc: str
    state: DecisionState
    composite_score: float
    confidence: str                   # "HIGH" / "MEDIUM" / "LOW" / "NONE"
    consensus_strength: float         # 0.0–1.0: how much engines agree
    conflict_score: float             # 0.0–1.0: how much they disagree

    # Engine breakdown
    engine_scores: Dict[str, float]
    engine_confidences: Dict[str, str]
    engine_suppressions: Dict[str, Optional[str]]
    engine_weights_used: Dict[str, float]

    # Quality
    quality_gate_passed: bool
    quality_gate_reason: str
    data_completeness: float
    decision_grade_evidence: float    # Fraction of decision-grade signals that are real

    # Suppression
    suppression_reasons: List[str]

    # Context (for dashboard, NOT for decisions)
    regime: str
    event_outcome: str
    session: str

    def to_dict(self) -> Dict[str, Any]:
        # Map state to legacy direction
        if self.state in (DecisionState.BULLISH, DecisionState.WEAK_BULLISH):
            direction = "LONG"
        elif self.state in (DecisionState.BEARISH, DecisionState.WEAK_BEARISH):
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        return {
            "timestamp_utc": self.timestamp_utc,
            "state": self.state.value,
            "composite_score": round(self.composite_score, 2),
            "confidence": self.confidence,
            "consensus_strength": round(self.consensus_strength, 2),
            "conflict_score": round(self.conflict_score, 2),
            "engine_scores": {k: round(v, 2) for k, v in self.engine_scores.items()},
            "engine_confidences": self.engine_confidences,
            "engine_suppressions": self.engine_suppressions,
            "engine_weights_used": {k: round(v, 3) for k, v in self.engine_weights_used.items()},
            "quality_gate_passed": self.quality_gate_passed,
            "quality_gate_reason": self.quality_gate_reason,
            "data_completeness": round(self.data_completeness, 2),
            "decision_grade_evidence": round(self.decision_grade_evidence, 2),
            "suppression_reasons": self.suppression_reasons,
            "regime": self.regime,
            "event_outcome": self.event_outcome,
            "session": self.session,
            # Dashboard backward-compat fields
            "direction": direction,
            "suppressed": len(self.suppression_reasons) > 0,
            "suppression_reason": "; ".join(self.suppression_reasons) if self.suppression_reasons else None,
            "conviction": self.confidence,
            "score": round(self.composite_score, 2),
        }


# ---------------------------------------------------------------------------
# Layer 1 Decision Engine
# ---------------------------------------------------------------------------

# Confidence → weight multiplier
_CONFIDENCE_WEIGHT = {
    ConfidenceLevel.HIGH:   1.0,
    ConfidenceLevel.MEDIUM: 0.7,
    ConfidenceLevel.LOW:    0.5,    # v9 FIX: was 0.3 — killed low-data engines
    ConfidenceLevel.NONE:   0.0,    # Suppressed engines contribute nothing
}

# Base engine weights (priors — multiply by confidence weight)
_BASE_WEIGHTS = {
    "microstructure": 0.45,
    "flow":           0.30,
    "macro":          0.25,
}

# Score thresholds — v9 FIX: recalibrated to match actual scoring algebra.
# Previously 25/40 which was unreachable: typical strong-signal composite ≈ 15-25.
_DIRECTIONAL_THRESHOLD = 12.0     # Below this: NEUTRAL
_STRONG_SIGNAL_THRESHOLD = 22.0   # Above this with med+ confidence: BULLISH/BEARISH

# Conflict threshold
_HIGH_CONFLICT_THRESHOLD = 0.60   # Above this: suppress to NO_SIGNAL


class Layer1DecisionEngine:
    """
    Combines engine outputs into a final decision with full observability.

    Phase 3 replacement for Layer1BiasCombiner + HybridDecisionLayer.
    """

    def __init__(
        self,
        signal_threshold: float = _DIRECTIONAL_THRESHOLD,
        strong_threshold: float = _STRONG_SIGNAL_THRESHOLD,
        conflict_threshold: float = _HIGH_CONFLICT_THRESHOLD,
    ) -> None:
        self.signal_threshold = signal_threshold
        self.strong_threshold = strong_threshold
        self.conflict_threshold = conflict_threshold
        self._history: List[DecisionExplanation] = []

    def decide(
        self,
        micro: EngineOutput,
        flow: EngineOutput,
        macro: EngineOutput,
        *,
        quality_gate_passed: bool = True,
        quality_gate_reason: str = "ok",
        data_completeness: float = 1.0,
        session: str = "unknown",
    ) -> DecisionExplanation:
        """
        Main entry: combine three engine outputs into a final decision.
        """
        now = datetime.now(timezone.utc).isoformat()
        engines = {"microstructure": micro, "flow": flow, "macro": macro}
        suppression_reasons: List[str] = []

        # ── Gate 1: Quality ──────────────────────────────────────────
        if not quality_gate_passed:
            suppression_reasons.append(f"quality_gate: {quality_gate_reason}")

        # ── Gate 2: Pre-event blackout ───────────────────────────────
        if macro.event_outcome == "pre_event_blackout":
            suppression_reasons.append("pre_event_blackout")

        # ── Gate 3: All engines suppressed ───────────────────────────
        active_engines = {n: e for n, e in engines.items() if not e.suppressed}
        if not active_engines:
            suppression_reasons.append("all_engines_suppressed")

        # ── Compute confidence-weighted score ────────────────────────
        weighted_score = 0.0
        total_weight = 0.0
        weights_used: Dict[str, float] = {}

        for name, eng in engines.items():
            base_w = _BASE_WEIGHTS.get(name, 0.0)
            conf_w = _CONFIDENCE_WEIGHT.get(eng.confidence, 0.0)
            effective_w = base_w * conf_w

            weights_used[name] = effective_w
            weighted_score += eng.score * effective_w
            total_weight += effective_w

        if total_weight > 0:
            # Normalize so weights sum to ~1 (prevents score deflation
            # when one engine is suppressed)
            composite = weighted_score / total_weight
        else:
            composite = 0.0

        # Apply event dampening
        composite *= macro.event_dampen_factor

        composite = float(np.clip(composite, -100, 100))

        # ── Conflict detection ───────────────────────────────────────
        consensus_strength, conflict_score = self._compute_conflict(engines)

        if conflict_score > self.conflict_threshold and not suppression_reasons:
            suppression_reasons.append(
                f"engine_conflict={conflict_score:.2f}>{self.conflict_threshold}"
            )

        # ── Decision-grade evidence ──────────────────────────────────
        all_decision_sigs = []
        for eng in engines.values():
            all_decision_sigs.extend(
                s for s in eng.signals if s.grade == SignalGrade.DECISION
            )
        dec_evidence = (
            sum(1 for s in all_decision_sigs if s.is_real) /
            max(len(all_decision_sigs), 1)
        )
        if dec_evidence < 0.25 and not suppression_reasons:
            suppression_reasons.append(
                f"insufficient_decision_evidence={dec_evidence:.0%}"
            )

        # ── Final state ──────────────────────────────────────────────
        if suppression_reasons:
            state = DecisionState.NO_SIGNAL
            composite = 0.0
            confidence_str = "NONE"
        else:
            # Aggregate confidence from active engines
            confidences = [e.confidence for e in active_engines.values()]
            if ConfidenceLevel.HIGH in confidences:
                agg_conf = ConfidenceLevel.HIGH
            elif ConfidenceLevel.MEDIUM in confidences:
                agg_conf = ConfidenceLevel.MEDIUM
            else:
                agg_conf = ConfidenceLevel.LOW
            confidence_str = agg_conf.value

            abs_score = abs(composite)
            if abs_score < self.signal_threshold:
                state = DecisionState.NEUTRAL
            elif abs_score < self.strong_threshold or agg_conf == ConfidenceLevel.LOW:
                state = (DecisionState.WEAK_BULLISH if composite > 0
                         else DecisionState.WEAK_BEARISH)
            else:
                state = (DecisionState.BULLISH if composite > 0
                         else DecisionState.BEARISH)

        # ── Build explanation ────────────────────────────────────────
        explanation = DecisionExplanation(
            timestamp_utc=now,
            state=state,
            composite_score=composite,
            confidence=confidence_str,
            consensus_strength=consensus_strength,
            conflict_score=conflict_score,
            engine_scores={n: e.score for n, e in engines.items()},
            engine_confidences={n: e.confidence.value for n, e in engines.items()},
            engine_suppressions={n: e.suppression_reason for n, e in engines.items()},
            engine_weights_used=weights_used,
            quality_gate_passed=quality_gate_passed,
            quality_gate_reason=quality_gate_reason,
            data_completeness=data_completeness,
            decision_grade_evidence=dec_evidence,
            suppression_reasons=suppression_reasons,
            regime=macro.regime,
            event_outcome=macro.event_outcome,
            session=session,
        )

        self._history.append(explanation)
        if len(self._history) > 500:
            self._history = self._history[-500:]

        logger.info(
            "[Decision] {} | score={:+.1f} | conf={} | consensus={:.2f} | "
            "conflict={:.2f} | {}",
            state.value, composite, confidence_str,
            consensus_strength, conflict_score,
            "; ".join(suppression_reasons) if suppression_reasons else "ok",
        )

        return explanation

    # ------------------------------------------------------------------
    # Conflict / consensus
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_conflict(
        engines: Dict[str, EngineOutput],
    ) -> tuple[float, float]:
        """
        Compute consensus strength and conflict score.

        consensus_strength: 1.0 = all active engines agree on direction
        conflict_score:     1.0 = engines point in opposite directions with high confidence
        """
        active = [(n, e) for n, e in engines.items()
                  if not e.suppressed and abs(e.score) > 10]

        if len(active) < 2:
            return 0.5, 0.0  # Not enough engines for meaningful conflict

        scores = [e.score for _, e in active]
        signs = [np.sign(s) for s in scores]

        # Consensus: fraction of engines agreeing with the majority direction
        if signs:
            majority = max(set(signs), key=signs.count)
            agree = sum(1 for s in signs if s == majority)
            consensus = agree / len(signs)
        else:
            consensus = 0.5

        # Conflict: weighted by score magnitude
        # Two engines at +50 and -50 = high conflict
        # Two engines at +50 and -5 = low conflict
        if len(scores) >= 2:
            pos_weight = sum(abs(s) for s in scores if s > 0)
            neg_weight = sum(abs(s) for s in scores if s < 0)
            total = pos_weight + neg_weight
            if total > 0:
                conflict = 2 * min(pos_weight, neg_weight) / total
            else:
                conflict = 0.0
        else:
            conflict = 0.0

        return float(consensus), float(conflict)

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def to_legacy_signal(self, explanation: DecisionExplanation) -> Dict[str, Any]:
        """Convert to the old signal dict format for backward compat."""
        d = explanation.to_dict()
        # Map new states to old direction/confidence
        state = explanation.state
        if state in (DecisionState.BULLISH, DecisionState.WEAK_BULLISH):
            direction = "LONG"
        elif state in (DecisionState.BEARISH, DecisionState.WEAK_BEARISH):
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        return {
            "score": d["composite_score"],
            "direction": direction,
            "confidence": d["confidence"],
            "state": d["state"],
            "agreement": ("ALIGNED" if d["consensus_strength"] > 0.7
                         else "CONFLICT" if d["conflict_score"] > 0.4
                         else "MIXED"),
            "suppressed_reason": "; ".join(d["suppression_reasons"]) or None,
            "engines": d["engine_scores"],
            "regime": d["regime"],
            "event_outcome": d["event_outcome"],
            "session": d["session"],
            "feature_completeness": d["data_completeness"],
            "data_maturity": d["data_completeness"],
            "timestamp": d["timestamp_utc"],
            "decision_explanation": d,
        }

    def get_stats(self, last_n: int = 100) -> Dict[str, Any]:
        recent = self._history[-last_n:]
        if not recent:
            return {"n": 0}
        states = [e.state.value for e in recent]
        return {
            "n": len(recent),
            "no_signal_pct": states.count("NO_SIGNAL") / len(states),
            "neutral_pct": states.count("NEUTRAL") / len(states),
            "bullish_pct": (states.count("BULLISH") + states.count("WEAK_BULLISH")) / len(states),
            "bearish_pct": (states.count("BEARISH") + states.count("WEAK_BEARISH")) / len(states),
            "avg_conflict": np.mean([e.conflict_score for e in recent]),
            "avg_consensus": np.mean([e.consensus_strength for e in recent]),
        }
