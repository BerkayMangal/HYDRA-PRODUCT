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
        cvd_spot_z, spot_ok = self._feat(features, 'cvd_spot_zscore')
        cvd_perp_z, perp_ok = self._feat(features, 'cvd_perp_zscore')
        is_real = spot_ok or perp_ok
        if is_real:
            if np.sign(cvd_spot_z) == np.sign(cvd_perp_z):
                val = (cvd_spot_z + cvd_perp_z) / 2 * 25
            else:
                val = cvd_spot_z * 30
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
                val = cvd_5m_z * 20
            elif p5_ok and price_5m < -0.1 and cvd_5m_z > 1:
                val = cvd_5m_z * 20
            else:
                val = cvd_5m_z * 15
        sigs.append(SubSignal("cvd_5m_momentum", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.10, cvd5_ok))

        # 3. OI TRAP — DECISION
        oi_z, oi_ok = self._feat(features, 'oi_change_pct_zscore')
        val = 0
        if oi_ok and abs(oi_z) > 1.5:
            if p5_ok and abs(price_5m) < 0.1:
                d = np.sign(cvd_5m_z) if cvd5_ok and abs(cvd_5m_z) > 0.5 else 0
                val = d * oi_z * 20
            elif p5_ok:
                val = np.sign(price_5m) * abs(oi_z) * 15
        elif oi_ok and oi_z < -1.5 and p5_ok:
            val = -np.sign(price_5m) * abs(oi_z) * 10
        sigs.append(SubSignal("oi_trap_signal", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.18, oi_ok))

        # 4. FUNDING PRESSURE — SECONDARY (contrarian, not direct)
        fr_z, fr_ok = self._feat(features, 'funding_rate_zscore')
        val = -fr_z * 20 if fr_ok else 0
        sigs.append(SubSignal("funding_pressure", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.12, fr_ok))

        # 5. BASIS DEMAND — SECONDARY
        basis_z, ok = self._feat(features, 'basis_spread_pct_zscore')
        val = basis_z * 15 if ok else 0
        sigs.append(SubSignal("basis_demand", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.08, ok))

        # 6. ORDERBOOK IMBALANCE — DECISION
        ob_z, ok = self._feat(features, 'ob_imbalance_raw_zscore')
        val = ob_z * 25 if ok else 0
        sigs.append(SubSignal("ob_imbalance", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.15, ok))

        # 7. LIQUIDATION CASCADE — SECONDARY
        liq_imb_z, limb_ok = self._feat(features, 'liq_imbalance_zscore')
        liq_total_z, ltot_ok = self._feat(features, 'liq_total_zscore')
        mult = 25 if (ltot_ok and abs(liq_total_z) > 1.5) else 10
        val = -liq_imb_z * mult if limb_ok else 0
        sigs.append(SubSignal("liq_cascade_risk", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.08, limb_ok))

        # 8. L/S RATIO — SECONDARY (contrarian)
        ls_z, ok = self._feat(features, 'ls_ratio_zscore')
        val = -ls_z * 15 if ok else 0
        sigs.append(SubSignal("ls_contrarian", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.09, ok))

        self._log_missing("MicroEngine")

        return build_engine_output(
            "microstructure", sigs, self._access_completeness(),
            decision_score_threshold=15.0,
        )
