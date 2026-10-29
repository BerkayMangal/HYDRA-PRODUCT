import unittest
import numpy as np
import pandas as pd
from research.lab.engine import (
    Features,
    net_returns,
    metrics,
    daily_returns,
    grid,
    holm,
)


class Integrity(unittest.TestCase):
    def frame(self, n=1600):
        rng = np.random.default_rng(42)
        p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        ix = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "ts": ix.astype("int64") // 10**6,
                "open": p,
                "high": p * 1.01,
                "low": p * 0.99,
                "close": p,
                "vol": 1.0,
                "confirm": 1,
            },
            index=ix,
        )

    def test_no_lookahead_every_family_both_timeframes(self):
        df = self.frame()
        cut = 1251
        for fam in sorted({x["family"] for x in grid()}):
            for tf in [1, 4]:
                s = next(
                    x
                    for x in grid()
                    if x["family"] == fam
                    and x["hours"] == tf
                    and x["vol_target"] == 0.6
                )
                full = Features(df).position(s)
                prefix = Features(df.iloc[:cut]).position(s)
                np.testing.assert_allclose(full[:cut], prefix, err_msg=str(s))

    def test_future_mutation_cannot_change_past_positions(self):
        df = self.frame()
        other = df.copy()
        other.iloc[
            1200:, other.columns.get_indexer(["open", "high", "low", "close"])
        ] *= 3
        for fam in sorted({x["family"] for x in grid()}):
            s = next(x for x in grid() if x["family"] == fam and x["hours"] == 4)
            np.testing.assert_allclose(
                Features(df).position(s)[:1200], Features(other).position(s)[:1200]
            )

    def test_next_open_timing(self):
        df = self.frame(300)
        s = {
            "family": "trend",
            "hours": 1,
            "slow": 48,
            "slope": 0,
            "buffer": 0,
            "vol_target": 0,
        }
        p = Features(df).position(s)
        self.assertEqual(p[200], 0.0)
        expected = (
            df.close.iloc[200] > df.close.ewm(span=48, adjust=False).mean().iloc[200]
        )
        self.assertEqual(p[201], float(expected))

    def test_flat_no_yield_or_fees(self):
        np.testing.assert_array_equal(
            net_returns(np.zeros(5), np.array([0.1, -0.2, 0.3, 0.0, 0.0])), 0.0
        )

    def test_round_trip_cost(self):
        r = net_returns(np.ones(5), np.zeros(5), 0.0012)
        self.assertAlmostEqual(np.prod(1 + r), (1 - 0.0012) ** 2)

    def test_mark_to_market_drawdown(self):
        ix = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
        r = net_returns(np.ones(3), np.array([-0.5, 1.0, 0.0]), 0)
        m = metrics(r, np.ones(3), ix)
        self.assertAlmostEqual(m["net_return"], 0.0)
        self.assertAlmostEqual(m["max_dd"], -0.5)

    def test_daily_compounds_to_hourly(self):
        ix = pd.date_range("2020-01-01 17:00", periods=81, freq="h", tz="UTC")
        r = np.linspace(-0.01, 0.02, 81)
        self.assertAlmostEqual(np.prod(1 + daily_returns(r, ix)), np.prod(1 + r))

    def test_cash_and_exposure_constraints(self):
        with self.assertRaises(ValueError):
            net_returns(np.array([1.1]), np.array([0.0]))
        with self.assertRaises(ValueError):
            net_returns(np.array([-1.0]), np.array([0.0]))

    def test_unique_spec_ids(self):
        ids = [x["id"] for x in grid()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
