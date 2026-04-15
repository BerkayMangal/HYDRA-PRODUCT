"""
tests/test_feature_pipeline.py
───────────────────────────────
Unit tests for features/pipeline.py

Run:
    python -m pytest tests/test_feature_pipeline.py -v
    python tests/test_feature_pipeline.py  # standalone
"""

import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.pipeline import (
    FeaturePipeline,
    FeatureQuality,
    FeatureVector,
    FEATURE_NAMES,
    FEATURE_REGISTRY,
    MIN_QUALITY_SCORE,
)


def _make_df(n: int = 250, seed: int = 42, include_macro: bool = True) -> pd.DataFrame:
    """Synthetic 1H OHLCV + macro DataFrame for testing."""
    np.random.seed(seed)
    dates  = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    closes = 80_000 + np.cumsum(np.random.randn(n) * 300)
    df = pd.DataFrame(
        {
            "open":   closes + np.random.randn(n) * 50,
            "high":   closes + np.abs(np.random.randn(n) * 100),
            "low":    closes - np.abs(np.random.randn(n) * 100),
            "close":  closes,
            "volume": np.abs(np.random.randn(n) * 500 + 3_000),
            "funding": np.random.randn(n) * 0.0001,
        },
        index=dates,
    )
    if include_macro:
        df["vix"]        = np.abs(np.random.randn(n) * 4 + 20)
        df["qqq"]        = 480 + np.cumsum(np.random.randn(n) * 1.5)
        df["us10y"]      = 4.3 + np.random.randn(n) * 0.08
        df["gold"]       = 2_600 + np.cumsum(np.random.randn(n) * 7)
        df["fear_greed"] = np.clip(50 + np.random.randn(n) * 15, 0, 100)
    return df


class TestFeatureRegistry(unittest.TestCase):
    """FEATURE_REGISTRY integrity checks."""

    def test_registry_has_all_feature_names(self):
        for name in FEATURE_NAMES:
            self.assertIn(name, FEATURE_REGISTRY, f"'{name}' in FEATURE_NAMES but not FEATURE_REGISTRY")

    def test_registry_spec_names_match_keys(self):
        for key, spec in FEATURE_REGISTRY.items():
            self.assertEqual(key, spec.name)

    def test_ret8_removed(self):
        """ret_8 = ret_4 * 2 was a proxy and must not exist."""
        self.assertNotIn("ret_8", FEATURE_NAMES)
        self.assertNotIn("ret_8", FEATURE_REGISTRY)

    def test_min_bars_non_negative(self):
        for name, spec in FEATURE_REGISTRY.items():
            self.assertGreaterEqual(spec.min_bars, 0, f"{name}.min_bars < 0")

    def test_source_types_valid(self):
        valid = {"ohlcv", "funding", "macro", "calendar"}
        for name, spec in FEATURE_REGISTRY.items():
            self.assertIn(spec.source_type, valid, f"{name} has unknown source_type '{spec.source_type}'")


class TestTransformBatch(unittest.TestCase):
    """transform_batch() correctness."""

    def setUp(self):
        self.df   = _make_df(n=250, include_macro=True)
        self.pipe = FeaturePipeline()
        self.result = self.pipe.transform_batch(self.df)

    def test_all_features_present_with_full_macro(self):
        missing = [f for f in FEATURE_NAMES if f not in self.result.columns]
        self.assertEqual(missing, [], f"Missing features: {missing}")

    def test_last_row_no_nan_with_sufficient_history(self):
        last = self.result.iloc[-1]
        nan_feats = [f for f in FEATURE_NAMES if f in self.result.columns and pd.isna(last[f])]
        self.assertEqual(nan_feats, [], f"NaN in last row: {nan_feats}")

    def test_ret4_is_real_4h_return(self):
        """ret_4 must equal pct_change(4) on the close series, not a proxy."""
        expected = self.df["close"].pct_change(4).iloc[-1]
        actual   = self.result["ret_4"].iloc[-1]
        self.assertAlmostEqual(actual, expected, places=8, msg="ret_4 is not close.pct_change(4)")

    def test_ret48_is_real_48h_return(self):
        expected = self.df["close"].pct_change(48).iloc[-1]
        actual   = self.result["ret_48"].iloc[-1]
        self.assertAlmostEqual(actual, expected, places=8, msg="ret_48 is not close.pct_change(48)")

    def test_gold_btc_z_is_not_hardcoded_zero(self):
        """gold_btc_ratio_z must be computed from gold price, not hardcoded 0."""
        if "gold_btc_ratio_z" not in self.result.columns:
            self.skipTest("gold_btc_ratio_z not in output (may need >3 distinct gold values)")
        z_vals = self.result["gold_btc_ratio_z"].dropna()
        if len(z_vals) > 0:
            self.assertFalse((z_vals == 0).all(), "gold_btc_ratio_z is all zeros")

    def test_qqq_momentum_is_not_hardcoded_zero(self):
        """qqq_momentum_5d must be computed when QQQ varies."""
        if "qqq_momentum_5d" not in self.result.columns:
            self.skipTest("qqq_momentum_5d not in output")
        vals = self.result["qqq_momentum_5d"].dropna()
        if len(vals) > 0:
            self.assertFalse((vals == 0).all(), "qqq_momentum_5d is all zeros")

    def test_funding_z_computed_when_funding_present(self):
        fund_z = self.result["fund_z"].dropna()
        self.assertGreater(len(fund_z), 0, "fund_z has no non-NaN values despite funding column present")

    def test_calendar_features_valid_range(self):
        last = self.result.iloc[-1]
        self.assertIn(int(last["hour"]),     range(24))
        self.assertIn(int(last["day"]),      range(7))
        self.assertIn(int(last["is_us"]),    [0, 1])
        self.assertIn(int(last["is_weekend"]),[0, 1])

    def test_no_inf_values_in_last_row(self):
        last = self.result.iloc[-1]
        inf_feats = [f for f in FEATURE_NAMES if f in self.result.columns and np.isinf(last.get(f, 0))]
        self.assertEqual(inf_feats, [], f"Inf values in last row: {inf_feats}")

    def test_without_macro_partial_features_returned(self):
        df_no_macro = _make_df(n=250, include_macro=False)
        pipe   = FeaturePipeline()
        result = pipe.transform_batch(df_no_macro)
        # OHLCV features must be present
        ohlcv_feats = [n for n, s in FEATURE_REGISTRY.items() if s.source_type == "ohlcv"]
        for f in ohlcv_feats:
            self.assertIn(f, result.columns, f"OHLCV feature '{f}' missing without macro")
        # Macro features must be absent (NaN or not in columns)
        macro_feats = [n for n, s in FEATURE_REGISTRY.items() if s.source_type == "macro"]
        for f in macro_feats:
            if f in result.columns:
                self.assertTrue(result[f].isna().all(), f"Macro feature '{f}' has values without macro data")

    def test_row_count_preserved(self):
        self.assertEqual(len(self.result), len(self.df))

    def test_index_preserved(self):
        # Pipeline strips tz in transform_batch (UTC-naive internally).
        # Compare only the timestamps, not the dtype/tz metadata.
        result_ts = self.result.index.tz_localize(None) if self.result.index.tz is None else self.result.index.tz_convert(None)
        input_ts  = self.df.index.tz_localize(None) if self.df.index.tz is None else self.df.index.tz_convert(None)
        pd.testing.assert_index_equal(result_ts, input_ts)


class TestTransformLive(unittest.TestCase):
    """transform_live() correctness and ring-buffer behavior."""

    def setUp(self):
        self.df = _make_df(n=250, include_macro=True)

    def _warm_up_and_feed(self, n_warmup: int = 245, n_live: int = 5, include_macro: bool = True):
        pipe = FeaturePipeline()
        pipe.warm_up(self.df.iloc[:n_warmup])
        for i in range(n_warmup, n_warmup + n_live):
            row = self.df.iloc[i]
            pipe.ingest_candle(
                {
                    "timestamp": int(row.name.timestamp() * 1000),
                    "open": row["open"], "high": row["high"],
                    "low": row["low"],   "close": row["close"],
                    "volume": row["volume"],
                },
                funding_rate=float(row.get("funding", 0)),
            )
            if include_macro:
                pipe.ingest_macro({
                    "vix_current":        float(row.get("vix", 20)),
                    "qqq_current":        float(row.get("qqq", 480)),
                    "us10y_current":      float(row.get("us10y", 4.3)),
                    "gold_current":       float(row.get("gold", 2600)),
                    "fear_greed_value":   float(row.get("fear_greed", 50)),
                })
        return pipe

    def test_returns_feature_vector_after_warmup(self):
        pipe = self._warm_up_and_feed()
        fv   = pipe.transform_live()
        self.assertIsNotNone(fv)
        self.assertIsInstance(fv, FeatureVector)

    def test_quality_high_after_warmup_with_macro(self):
        pipe = self._warm_up_and_feed(include_macro=True)
        fv   = pipe.transform_live()
        self.assertGreaterEqual(fv.quality_score, MIN_QUALITY_SCORE,
            f"Quality {fv.quality_score:.0%} below threshold after warmup with macro")

    def test_no_proxy_features_after_warmup(self):
        """With a ring buffer and macro data, proxy list must be empty."""
        pipe = self._warm_up_and_feed(include_macro=True)
        fv   = pipe.transform_live()
        self.assertEqual(fv.quality.proxy, [],
            f"Proxy features present after full warmup: {fv.quality.proxy}")

    def test_can_predict_true_after_warmup(self):
        pipe = self._warm_up_and_feed()
        fv   = pipe.transform_live()
        self.assertTrue(fv.can_predict)

    def test_can_predict_false_with_sparse_buffer(self):
        """3 bars is not enough — prediction should be blocked."""
        pipe = FeaturePipeline()
        for i in range(3):
            pipe.ingest_candle({"close": 80_000.0, "volume": 1_000.0, "timestamp": i * 3_600_000})
        fv = pipe.transform_live()
        self.assertIsNone(fv)  # returns None when < 5 bars

    def test_batch_live_ret4_consistency(self):
        """ret_4 from live must closely match the batch pipeline."""
        pipe_batch = FeaturePipeline()
        batch_result = pipe_batch.transform_batch(self.df)
        batch_ret4   = float(batch_result.iloc[-1]["ret_4"])

        pipe_live = self._warm_up_and_feed(n_warmup=245, n_live=5, include_macro=False)
        fv = pipe_live.transform_live()
        live_ret4 = fv.values.get("ret_4", float("nan"))

        self.assertFalse(pd.isna(live_ret4), "ret_4 is NaN in live vector")
        # Small delta acceptable due to 1H vs 5-min accumulator boundary
        delta = abs(batch_ret4 - live_ret4)
        self.assertLess(delta, 0.005, f"ret_4 batch/live mismatch too large: {delta:.6f}")

    def test_to_array_ordered_correctly(self):
        pipe = self._warm_up_and_feed()
        fv   = pipe.transform_live()
        arr  = fv.to_array(FEATURE_NAMES)
        self.assertEqual(arr.shape, (len(FEATURE_NAMES),))
        self.assertEqual(arr.dtype, np.float32)

    def test_to_array_with_subset(self):
        pipe   = self._warm_up_and_feed()
        fv     = pipe.transform_live()
        subset = FEATURE_NAMES[:5]
        arr    = fv.to_array(subset)
        self.assertEqual(arr.shape, (5,))

    def test_ring_buffer_capped(self):
        pipe = FeaturePipeline()
        df_long = _make_df(n=600, include_macro=False)
        pipe.warm_up(df_long)
        self.assertLessEqual(len(pipe._ring), pipe.RING_CAPACITY)

    def test_warmup_with_tz_aware_index(self):
        df_utc = _make_df(n=100)
        # Already UTC-aware — should not raise
        pipe = FeaturePipeline()
        pipe.warm_up(df_utc)
        self.assertGreater(len(pipe._ring), 0)

    def test_warmup_with_tz_naive_index(self):
        df_naive = _make_df(n=100)
        df_naive.index = df_naive.index.tz_localize(None)
        pipe = FeaturePipeline()
        pipe.warm_up(df_naive)
        self.assertGreater(len(pipe._ring), 0)


class TestFeatureQuality(unittest.TestCase):
    """FeatureQuality dataclass logic."""

    def test_score_full_quality(self):
        q = FeatureQuality(
            real=list(FEATURE_NAMES[:20]),
            external=list(FEATURE_NAMES[20:]),
        )
        self.assertAlmostEqual(q.score, 1.0, places=2)
        self.assertTrue(q.can_predict)

    def test_score_zero_with_all_defaults(self):
        q = FeatureQuality(default=list(FEATURE_NAMES))
        self.assertAlmostEqual(q.score, 0.0, places=2)
        self.assertFalse(q.can_predict)

    def test_summary_contains_counts(self):
        q = FeatureQuality(real=["a", "b"], external=["c"], proxy=["d"], default=["e"])
        s = q.summary()
        self.assertIn("real=2", s)
        self.assertIn("ext=1", s)
        self.assertIn("proxy=1", s)
        self.assertIn("default=1", s)

    def test_can_predict_at_threshold_boundary(self):
        # Exactly at threshold should pass
        import math
        # Use ceil so n_needed / total >= MIN_QUALITY_SCORE exactly
        total    = len(FEATURE_NAMES)
        n_needed = math.ceil(total * MIN_QUALITY_SCORE)
        q_pass   = FeatureQuality(real=list(FEATURE_NAMES[:n_needed]),
                                  default=list(FEATURE_NAMES[n_needed:]))
        self.assertTrue(q_pass.can_predict,
            f"Expected can_predict=True with {n_needed}/{total} real features (threshold={MIN_QUALITY_SCORE:.0%})")

        # One below threshold should fail
        q_fail = FeatureQuality(real=list(FEATURE_NAMES[:n_needed - 1]),
                                default=list(FEATURE_NAMES[n_needed - 1:]))
        self.assertFalse(q_fail.can_predict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
