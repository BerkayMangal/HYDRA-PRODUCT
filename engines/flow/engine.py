"""
Flow Engine v3 — Phase 3.
Structured output, signal grading, engine-level suppression.

Signal grades:
  DECISION:  ETF flow (institutional, verifiable)
  SECONDARY: Exchange netflow, stablecoin power, market sentiment
  CONTEXT:   Fear & Greed (daily, contrarian folk heuristic — cannot drive score)
"""
import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger

from engines.feature_access import FeatureAccessMixin
from engines.output import (
    EngineOutput, SubSignal, SignalGrade, build_engine_output,
)


class FlowEngine(FeatureAccessMixin):

    def __init__(self, config: Dict):
        self.config = config

    def compute(self, features: pd.Series) -> EngineOutput:
        self._reset_access()
        sigs = []

        # 1. EXCHANGE NETFLOW — SECONDARY (paid tier often missing)
        netflow_z, ok = self._feat(features, 'exchange_netflow_btc_zscore')
        val = -netflow_z * 40 if ok else 0
        sigs.append(SubSignal("exchange_flow", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.18, ok))

        # 2. ETF NET FLOW — DECISION (institutional, verifiable)
        # v9 FIX: multipliers raised (was 20/15 → now 35/25)
        etf_7d_z, ok7 = self._feat(features, 'etf_net_flow_7d_zscore')
        etf_daily_z, okd = self._feat(features, 'etf_net_flow_daily_zscore')
        val = 0.0
        is_real = ok7 or okd
        if ok7:
            val = etf_7d_z * 35
        if okd and abs(etf_daily_z) > 1.5:
            val += np.sign(etf_daily_z) * 25
        sigs.append(SubSignal("etf_flow", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.28, is_real))

        # 3. FEAR & GREED — CONTEXT (daily, contrarian folk heuristic)
        # Phase 3: demoted from decision to context-only.
        # Cannot materially affect score — informational for dashboard.
        fg_value, fg_ok = self._feat(features, 'fear_greed_value', neutral=50.0)
        fg_signal = 0
        if fg_ok and 0 < fg_value <= 100:
            if fg_value <= 20:
                fg_signal = 60 + (20 - fg_value) * 2
            elif fg_value <= 35:
                fg_signal = 20 + (35 - fg_value) / 15 * 40
            elif fg_value <= 65:
                fg_signal = 0
            elif fg_value <= 80:
                fg_signal = -20 - (fg_value - 65) / 15 * 40
            else:
                fg_signal = -60 - (fg_value - 80) * 2
        sigs.append(SubSignal("fear_greed", fg_signal, float(np.clip(fg_signal, -100, 100)),
                              SignalGrade.CONTEXT, 0.0, fg_ok,
                              note="context-only: does not affect score"))

        # 4. STABLECOIN POWER — SECONDARY
        stable_z, ok = self._feat(features, 'stablecoin_exchange_ratio_zscore')
        val = stable_z * 25 if ok else 0
        sigs.append(SubSignal("stablecoin_power", val, float(np.clip(val, -100, 100)),
                              SignalGrade.SECONDARY, 0.14, ok))

        # 5. MARKET SENTIMENT — DECISION (v9 FIX: promoted from SECONDARY)
        # Total crypto mcap change is verifiable on-chain data, not opinion.
        mcap_change, ok = self._feat(features, 'total_mcap_change_24h')
        val = 0
        if ok:
            if mcap_change > 3: val = 60
            elif mcap_change > 1: val = 30
            elif mcap_change < -3: val = -60
            elif mcap_change < -1: val = -30
        sigs.append(SubSignal("market_sentiment", val, float(np.clip(val, -100, 100)),
                              SignalGrade.DECISION, 0.22, ok))

        self._log_missing("FlowEngine")

        return build_engine_output(
            "flow", sigs, self._access_completeness(),
            decision_score_threshold=10.0,
        )
