"""
tests/test_integration.py
──────────────────────────
End-to-end integration test for the full HYDRA signal pipeline.

Simulates one complete cycle: synthetic data → engines → decision → output.
Verifies the entire chain works together without import errors or type mismatches.

Run:
    python -m pytest tests/test_integration.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def _synthetic_features() -> pd.Series:
    """Build a realistic feature Series as the unified frame would produce."""
    return pd.Series({
        # Microstructure (collector z-scores)
        "cvd_spot_zscore": 1.2,
        "cvd_perp_zscore": 0.8,
        "cvd_5m_delta_zscore": 0.6,
        "close_pct_5m": 0.15,
        "oi_change_pct_zscore": 1.8,
        "funding_rate_zscore": -0.3,
        "basis_spread_pct_zscore": 0.2,
        "ob_imbalance_raw_zscore": 0.9,
        "liq_imbalance_zscore": -0.1,
        "liq_total_zscore": 0.05,
        "ls_ratio_zscore": 0.3,
        # Flow
        "exchange_netflow_btc_zscore": -0.5,
        "etf_net_flow_7d_zscore": 1.5,
        "etf_net_flow_daily_zscore": 0.8,
        "fear_greed_value": 35.0,
        "stablecoin_exchange_ratio_zscore": 0.2,
        "total_mcap_change_24h": 1.5,
        # Macro
        "dxy_vs_sma20": -0.5,
        "dxy_change_24h": -0.4,
        "us10y_change_24h": -0.12,
        "vix_current": 18.0,
        "vix_change_24h": -3.0,
        "btc_spx_correlation": 0.55,
        "spx_vs_sma20": 1.2,
        "btc_dominance": 58.0,
        # Events
        "fomc_just_passed": 0,
        "cpi_just_passed": 0,
        "fomc_is_imminent": 0,
        "cpi_is_imminent": 0,
        "event_dampen_factor": 1.0,
        # Meta
        "session_label": "new_york",
        "feature_completeness": 0.85,
        "data_maturity": 1.0,
    })


class TestFullPipeline(unittest.TestCase):
    """End-to-end: features → engines → decision → output."""

    def test_full_bullish_cycle(self):
        from engines.microstructure.engine import MicrostructureEngine
        from engines.flow.engine import FlowEngine
        from engines.macro.engine import MacroEngine
        from signals.layer1_decision import Layer1DecisionEngine, DecisionState

        features = _synthetic_features()

        micro = MicrostructureEngine({}).compute(features)
        flow = FlowEngine({}).compute(features)
        macro = MacroEngine({}).compute(features)

        engine = Layer1DecisionEngine()
        result = engine.decide(micro, flow, macro)

        # With these bullish features, we should get a directional signal
        self.assertIsNotNone(result)
        self.assertIn(result.state, (
            DecisionState.BULLISH, DecisionState.WEAK_BULLISH,
            DecisionState.NEUTRAL,  # acceptable if scores cancel
        ))

        # Explanation should be complete
        d = result.to_dict()
        self.assertIn("composite_score", d)
        self.assertIn("engine_scores", d)
        self.assertIn("suppression_reasons", d)
        self.assertEqual(len(d["engine_scores"]), 3)

    def test_full_no_data_cycle(self):
        """Empty features → all engines suppress → NO_SIGNAL."""
        from engines.microstructure.engine import MicrostructureEngine
        from engines.flow.engine import FlowEngine
        from engines.macro.engine import MacroEngine
        from signals.layer1_decision import Layer1DecisionEngine, DecisionState

        empty = pd.Series(dtype=float)

        micro = MicrostructureEngine({}).compute(empty)
        flow = FlowEngine({}).compute(empty)
        macro = MacroEngine({}).compute(empty)

        # All engines should be suppressed
        self.assertTrue(micro.suppressed)
        self.assertTrue(flow.suppressed)
        self.assertTrue(macro.suppressed)

        result = Layer1DecisionEngine().decide(micro, flow, macro)
        self.assertEqual(result.state, DecisionState.NO_SIGNAL)
        self.assertIn("all_engines_suppressed", result.suppression_reasons)

    def test_legacy_signal_compat(self):
        """Decision engine produces a legacy dict with required keys."""
        from engines.microstructure.engine import MicrostructureEngine
        from engines.flow.engine import FlowEngine
        from engines.macro.engine import MacroEngine
        from signals.layer1_decision import Layer1DecisionEngine

        features = _synthetic_features()
        micro = MicrostructureEngine({}).compute(features)
        flow = FlowEngine({}).compute(features)
        macro = MacroEngine({}).compute(features)

        engine = Layer1DecisionEngine()
        result = engine.decide(micro, flow, macro)
        legacy = engine.to_legacy_signal(result)

        required = {"score", "direction", "confidence", "timestamp",
                    "engines", "regime", "session", "state"}
        self.assertTrue(required.issubset(legacy.keys()),
                        f"Missing: {required - legacy.keys()}")
        self.assertIn(legacy["direction"], ("LONG", "SHORT", "NEUTRAL"))

    def test_context_signals_do_not_inflate_score(self):
        """Fear & Greed at extreme values should NOT move the score."""
        from engines.flow.engine import FlowEngine
        from engines.output import SignalGrade

        # Extreme Fear = 10 (historically strong contrarian signal)
        features = pd.Series({
            "fear_greed_value": 10.0,
            # No other features → only context signal active
        })
        result = FlowEngine({}).compute(features)

        # Fear & Greed should be context-only
        fg = next((s for s in result.signals if s.name == "fear_greed"), None)
        self.assertIsNotNone(fg)
        self.assertEqual(fg.grade, SignalGrade.CONTEXT)
        self.assertGreater(abs(fg.capped_value), 50,  # strong contrarian reading
                           "F&G=10 should produce a large signal value")

        # But the engine score should be ~0 because context has weight=0
        self.assertTrue(result.suppressed,
                        "Engine should suppress: only context signal, no decision data")

    def test_engine_output_type_consistency(self):
        """All engines return EngineOutput, not dict."""
        from engines.microstructure.engine import MicrostructureEngine
        from engines.flow.engine import FlowEngine
        from engines.macro.engine import MacroEngine
        from engines.output import EngineOutput

        features = _synthetic_features()
        for EngineCls in (MicrostructureEngine, FlowEngine, MacroEngine):
            result = EngineCls({}).compute(features)
            self.assertIsInstance(result, EngineOutput,
                                 f"{EngineCls.__name__} should return EngineOutput")

    def test_quality_gate_blocks_when_low(self):
        """Quality gate failure → NO_SIGNAL."""
        from engines.microstructure.engine import MicrostructureEngine
        from engines.flow.engine import FlowEngine
        from engines.macro.engine import MacroEngine
        from signals.layer1_decision import Layer1DecisionEngine, DecisionState

        features = _synthetic_features()
        micro = MicrostructureEngine({}).compute(features)
        flow = FlowEngine({}).compute(features)
        macro = MacroEngine({}).compute(features)

        result = Layer1DecisionEngine().decide(
            micro, flow, macro,
            quality_gate_passed=False,
            quality_gate_reason="live=25%<50%",
        )
        self.assertEqual(result.state, DecisionState.NO_SIGNAL)


class TestCostAlignment(unittest.TestCase):
    """Paper-trade and backtest cost models are aligned."""

    def test_paper_trade_includes_slippage(self):
        """Paper-trade cost per side should include slippage, not just fees."""
        from ml.signal_engine import _COST_PER_SIDE
        # Should be (0.001 + 0.0004) / 2 = 0.0007
        self.assertAlmostEqual(_COST_PER_SIDE, 0.0007, places=5)
        # Must be > fee-only (0.0005)
        self.assertGreater(_COST_PER_SIDE, 0.0005)

    def test_walk_forward_cost_matches(self):
        from ml.research.walk_forward_v2 import COST
        from ml.signal_engine import _COST_PER_SIDE
        wf_per_side = COST.total_cost_per_trade / 2
        self.assertAlmostEqual(_COST_PER_SIDE, wf_per_side, places=5,
                               msg="Paper-trade and walk-forward must use same cost/side")


if __name__ == "__main__":
    unittest.main()
