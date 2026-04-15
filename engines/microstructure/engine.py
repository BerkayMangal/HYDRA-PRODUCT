"""
Microstructure Engine v3 — Phase 3.
Structured output, signal grading, engine-level suppression.
"""
import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger

from engines.feature_access import FeatureAccessMixin
from engines.output import (
    EngineOutput, SubSignal, SignalGrade, build_engine_output,
)

# Critical decision-grade features: if ALL missing, engine suppresses.
_CRITICAL_FEATURES = {"cvd_spot_zscore", "oi_change_pct_zscore", "ob_imbalance_raw_zscore"}


class MicrostructureEngine(FeatureAccessMixin):

    def __init__(self, config: Dict):
        self.config = config

    def compute(self, features: pd.Series) -> EngineOutput:
        self._reset_access()
        sigs = []

        # 1. CVD DIVERGENCE — DECISION grade
        # v9 FIX: multipliers raised ~2x (was 25/30 → now 45/55)
        cvd_spot_z, spot_ok = self._feat(features, 'cvd_spot_zscore')
        cvd_perp_z, perp_ok = self._feat(features, 'cvd_perp_zscore')
        is_real = spot_ok or perp_ok
        if is_real:
            if np.sign(cvd_spot_z) == np.sign(cvd_perp_z):
                val = (cvd_spot_z + cvd_perp_z) / 2 * 45
            else:
                val = cvd_spot_z * 55
        else:
            val = 0
        sigs.append(SubSignal("cvd_divergence", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.20, is_real))

        # 2. CVD 5m MOMENTUM — DECISION
        cvd_5m_z, cvd5_ok = self._feat(features, 'cvd_5m_delta_zscore')
        price_5m, p5_ok = self._feat(features, 'close_pct_5m')
        val = 0
        if cvd5_ok:
            if p5_ok and price_5m > 0.1 and cvd_5m_z < -1:
                val = cvd_5m_z * 35
            elif p5_ok and price_5m < -0.1 and cvd_5m_z > 1:
                val = cvd_5m_z * 35
            else:
                val = cvd_5m_z * 25
        sigs.append(SubSignal("cvd_5m_momentum", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.10, cvd5_ok))

        # 3. OI TRAP — DECISION
        oi_z, oi_ok = self._feat(features, 'oi_change_pct_zscore')
        val = 0
        if oi_ok and abs(oi_z) > 1.5:
            if p5_ok and abs(price_5m) < 0.1:
                d = np.sign(cvd_5m_z) if cvd5_ok and abs(cvd_5m_z) > 0.5 else 0
                val = d * oi_z * 35
            elif p5_ok:
                val = np.sign(price_5m) * abs(oi_z) * 30
        elif oi_ok and oi_z < -1.5 and p5_ok:
            val = -np.sign(price_5m) * abs(oi_z) * 20
        sigs.append(SubSignal("oi_trap_signal", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.15, oi_ok))

        # 4. TREND COMPOSITE — DECISION (v9 NEW: uses computed TA features)
        # EMA stack + RSI + VWAP distance → clear trend signal
        ema_stack, ema_ok = self._feat(features, 'ema_stack')
        rsi, rsi_ok = self._feat(features, 'rsi', neutral=50.0)
        vwap_dist, vwap_ok = self._feat(features, 'vwap_dist')
        dist_ema50, ema50_ok = self._feat(features, 'dist_ema50')
        trend_real = sum([ema_ok, rsi_ok, vwap_ok, ema50_ok]) >= 2
        val = 0
        if trend_real:
            # EMA stack: +1 = bullish alignment, 0 = not aligned
            if ema_ok:
                val += 25 if ema_stack > 0.5 else -15
            # RSI: >60 bullish momentum, <40 bearish momentum
            if rsi_ok:
                if rsi > 65: val += 20
                elif rsi > 55: val += 10
                elif rsi < 35: val -= 20
                elif rsi < 45: val -= 10
            # VWAP distance: above = bullish, below = bearish
            if vwap_ok:
                val += np.clip(vwap_dist * 8, -20, 20)
            # EMA50 distance: confirms trend strength
            if ema50_ok:
                val += np.clip(dist_ema50 * 5, -15, 15)
        sigs.append(SubSignal("trend_composite", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.15, trend_real))

        # 5. FUNDING PRESSURE — SECONDARY (contrarian, not direct)
        fr_z, fr_ok = self._feat(features, 'funding_rate_zscore')
        val = -fr_z * 30 if fr_ok else 0
        sigs.append(SubSignal("funding_pressure", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.10, fr_ok))

        # 6. BASIS DEMAND — SECONDARY
        basis_z, ok = self._feat(features, 'basis_spread_pct_zscore')
        val = basis_z * 25 if ok else 0
        sigs.append(SubSignal("basis_demand", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.08, ok))

        # 7. ORDERBOOK IMBALANCE — DECISION
        ob_z, ok = self._feat(features, 'ob_imbalance_raw_zscore')
        val = ob_z * 45 if ok else 0
        sigs.append(SubSignal("ob_imbalance", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.12, ok))

        # 8. LIQUIDATION CASCADE — SECONDARY
        liq_imb_z, limb_ok = self._feat(features, 'liq_imbalance_zscore')
        liq_total_z, ltot_ok = self._feat(features, 'liq_total_zscore')
        mult = 35 if (ltot_ok and abs(liq_total_z) > 1.5) else 15
        val = -liq_imb_z * mult if limb_ok else 0
        sigs.append(SubSignal("liq_cascade_risk", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.05, limb_ok))

        # 9. L/S RATIO — SECONDARY (contrarian)
        ls_z, ok = self._feat(features, 'ls_ratio_zscore')
        val = -ls_z * 25 if ok else 0
        sigs.append(SubSignal("ls_contrarian", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.05, ok))

        self._log_missing("MicroEngine")

        return build_engine_output(
            "microstructure", sigs, self._access_completeness(),
            decision_score_threshold=15.0,
        )
