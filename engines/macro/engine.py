"""
Macro Engine v5 — Phase 3.
Structured output, signal grading, engine-level suppression.

Signal grades:
  DECISION:  VIX regime, rates regime (verifiable, market-traded)
  SECONDARY: DXY regime, SPX correlation, BTC dominance
  CONTEXT:   Prediction markets (thin, lagging), event calendar
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from engines.feature_access import FeatureAccessMixin
from engines.output import (
    EngineOutput, SubSignal, SignalGrade, build_engine_output,
)

_DXY_REACTION_THRESHOLD_PCT = 0.20
_RATES_REACTION_THRESHOLD_PCT = 0.05
_EVENT_MAX_SIGNAL = 40.0
_PRED_HIGH_CONVICTION = 0.65
_PRED_MID_CONVICTION = 0.50
_PRED_COUNTER_LEVEL = 0.30
_VIX_LOW = 15.0; _VIX_ELEVATED = 25.0; _VIX_CRISIS = 35.0
_BTC_DOM_HIGH = 60.0; _BTC_DOM_MID = 55.0; _BTC_DOM_ALT = 45.0


class MacroEngine(FeatureAccessMixin):

    def __init__(self, config: Dict) -> None:
        self.config = config

    def compute(self, features: pd.Series) -> EngineOutput:
        self._reset_access()
        sigs = []

        # 1. DXY REGIME — SECONDARY (yfinance, 15min delay)
        # v9 FIX: multipliers raised (was 20/25 → now 35/40)
        dxy_trend, t_ok = self._feat(features, "dxy_vs_sma20")
        dxy_change, c_ok = self._feat(features, "dxy_change_24h")
        val = 0.0
        if t_ok and abs(dxy_trend) > 0.3:
            val += -np.sign(dxy_trend) * min(abs(dxy_trend) * 35, 80)
        if c_ok and abs(dxy_change) > 0.3:
            val += -np.sign(dxy_change) * min(abs(dxy_change) * 40, 60)
        sigs.append(SubSignal("dxy_regime", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.18, t_ok or c_ok))

        # 2. RATES REGIME — DECISION (traded instrument, verifiable)
        us10y_change, ok = self._feat(features, "us10y_change_24h")
        val = 0.0
        if ok:
            if us10y_change > 0.20: val = -60
            elif us10y_change > 0.10: val = -30
            elif us10y_change < -0.20: val = 60
            elif us10y_change < -0.10: val = 30
        sigs.append(SubSignal("rates_regime", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.15, ok))

        # 3. SPX CORRELATION — SECONDARY
        # v9 FIX: multiplier raised (12 → 20)
        corr, corr_ok = self._feat(features, "btc_spx_correlation")
        spx_trend, trend_ok = self._feat(features, "spx_vs_sma20")
        val = 0.0
        is_real = corr_ok and trend_ok
        if is_real and abs(corr) > 0.4:
            val = np.sign(corr) * spx_trend * 20 * min(abs(corr), 1.0)
        sigs.append(SubSignal("spx_correlation", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.12, is_real))

        # 4. PREDICTION MARKETS — CONTEXT (thin, lagging, crowd-wisdom)
        pred_val = self._compute_prediction_markets(features)
        sigs.append(SubSignal("prediction_markets", pred_val, float(np.clip(pred_val, -100, 100)),
                              SignalGrade.CONTEXT, 0.0, abs(pred_val) > 0,
                              note="context-only: does not affect score"))

        # 5. VOLATILITY REGIME — DECISION (VIX is real-time, traded)
        # v9 FIX: multipliers raised ~1.5x
        vix_val, vix_ok = self._feat(features, "vix_current", neutral=20.0)
        vix_change, vc_ok = self._feat(features, "vix_change_24h")
        regime = "normal"
        val = 0.0
        if vix_ok and vix_val > 0:
            if vix_val > _VIX_CRISIS: regime = "crisis"; val = -50
            elif vix_val > _VIX_ELEVATED: regime = "high_vol"; val = -20
            elif vix_val < _VIX_LOW: regime = "low_vol"; val = +20
        if vc_ok:
            if vix_change > 15: val -= 40
            elif vix_change > 5: val -= 15
            elif vix_change < -10: val += 25
            elif vix_change < -5: val += 12
        sigs.append(SubSignal("volatility_regime", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.25, vix_ok))

        # 6. EVENT CALENDAR — CONTEXT (not a trading signal per se)
        event_signal, event_outcome = self._compute_event_calendar(features)
        sigs.append(SubSignal("event_calendar", event_signal, float(np.clip(event_signal, -100, 100)),
                              SignalGrade.CONTEXT, 0.0, abs(event_signal) > 0,
                              note="context-only: events handled by blackout, not by score"))

        # 7. BTC DOMINANCE — SECONDARY (slow, context-adjacent)
        # v9 FIX: values raised
        btc_dom, ok = self._feat(features, "btc_dominance")
        val = 0.0
        if ok and btc_dom > 0:
            if btc_dom > _BTC_DOM_HIGH: val = 25
            elif btc_dom > _BTC_DOM_MID: val = 10
            elif btc_dom < _BTC_DOM_ALT: val = -20
        sigs.append(SubSignal("btc_dominance", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.10, ok))

        # Event dampening
        raw_dampen = features.get("event_dampen_factor", 1.0)
        try:
            _d = float(raw_dampen)
            event_dampen = float(np.clip(_d, 0.1, 1.0)) if not np.isnan(_d) else 1.0
        except (TypeError, ValueError):
            event_dampen = 1.0

        self._log_missing("MacroEngine")

        return build_engine_output(
            "macro", sigs, self._access_completeness(),
            decision_score_threshold=10.0,
            regime=regime,
            event_outcome=event_outcome,
            event_dampen_factor=event_dampen,
        )

    def _compute_prediction_markets(self, features: pd.Series) -> float:
        scores = []
        for key in features.index:
            if not isinstance(key, str): continue
            kl = key.lower()
            if not key.startswith(("poly_", "kalshi_")) or not kl.endswith("_prob"): continue
            val = features.get(key, None)
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0): continue
            c = self._map_prob(val, kl)
            if c is not None: scores.append(c)
        if not scores: return 0.0
        return float(np.clip(sum(scores) / len(scores) * 60, -100, 100))

    def _map_prob(self, prob: float, kl: str) -> Optional[float]:
        if any(kw in kl for kw in ["fed", "rate_cut", "interest"]):
            if prob > _PRED_HIGH_CONVICTION: return +1.0
            elif prob > _PRED_MID_CONVICTION: return +0.4
            elif prob < _PRED_COUNTER_LEVEL: return -0.6
            return 0.0
        if "recession" in kl:
            if prob > _PRED_HIGH_CONVICTION: return -0.8
            elif prob > _PRED_MID_CONVICTION: return -0.4
            return 0.0
        if ("bitcoin" in kl or "btc" in kl) and "above" in kl:
            if prob > 0.70: return +0.5
            elif prob < 0.25: return -0.5
            return 0.0
        return None

    def _compute_event_calendar(self, features: pd.Series) -> Tuple[float, str]:
        def _flag(k):
            v = features.get(k, 0)
            try:
                f = float(v)
                return (not np.isnan(f)) and f > 0.5
            except: return False

        if _flag("fomc_is_imminent") or _flag("cpi_is_imminent"):
            return 0.0, "pre_event_blackout"
        if not _flag("fomc_just_passed") and not _flag("cpi_just_passed"):
            return 0.0, "no_event"

        dxy_c, d_ok = self._feat(features, "dxy_change_24h")
        us_c, u_ok = self._feat(features, "us10y_change_24h")
        if not d_ok: dxy_c = 0.0
        if not u_ok: us_c = 0.0
        dv = (1 if dxy_c > _DXY_REACTION_THRESHOLD_PCT else
              -1 if dxy_c < -_DXY_REACTION_THRESHOLD_PCT else 0)
        rv = (1 if us_c > _RATES_REACTION_THRESHOLD_PCT else
              -1 if us_c < -_RATES_REACTION_THRESHOLD_PCT else 0)
        h = dv + rv
        lbl = "fomc" if _flag("fomc_just_passed") else "cpi"
        if h >= 2: return float(-_EVENT_MAX_SIGNAL), f"{lbl}_hawkish"
        elif h <= -2: return float(+_EVENT_MAX_SIGNAL), f"{lbl}_dovish"
        elif h == 1: return float(-min((abs(dxy_c)+abs(us_c))*15, 20)), f"{lbl}_mild_hawk"
        elif h == -1: return float(+min((abs(dxy_c)+abs(us_c))*15, 20)), f"{lbl}_mild_dove"
        return 0.0, f"{lbl}_ambiguous"
