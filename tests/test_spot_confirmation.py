import json
import sqlite3

import numpy as np
import pandas as pd

from research.spot_confirmation import baseline_exposure, compare, spot_window, replay_paired_portfolios


def history(rising=True):
    index = pd.date_range("2025-01-01", periods=1200, freq="h", tz="UTC")
    prices = np.linspace(100., 200., len(index)) if rising else np.linspace(200., 100., len(index))
    return pd.DataFrame({"close": prices}, index=index)


def database(asof):
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE TABLE health(observed_ms INTEGER,payload TEXT); "
                     "CREATE TABLE trades(instrument TEXT,event_ms INTEGER,observed_ms INTEGER,side TEXT,quote_size REAL);")
    healthy = json.dumps({"coverage": {"BTC-USDT": {"status": "overlap_observed"}}, "errors": {}})
    for ms in range(asof - 17 * 60000, asof + 1, 30000):
        db.execute("INSERT INTO health VALUES(?,?)", (ms, healthy))
    return db


def setup(rising=True):
    hourly = history(rising)
    asof = int((hourly.index[-1] + pd.Timedelta(hours=1)).timestamp() * 1000)
    return hourly, asof, database(asof)


def trade(db, asof, side, quote, observed=None):
    db.execute("INSERT INTO trades VALUES('BTC-USDT',?,?,?,?)",
               (asof - 1000, asof if observed is None else observed, side, quote))


def test_positive_spot_confirms_fixed_ema():
    hourly, asof, db = setup()
    trade(db, asof, "buy", 100.)
    result = compare(db, hourly, asof)
    assert result["baseline_exposure"] == result["confirmed_exposure"] == 1.
    assert not result["orders_enabled"]


def test_negative_spot_blocks_baseline_entry():
    hourly, asof, db = setup()
    trade(db, asof, "sell", 100.)
    result = compare(db, hourly, asof)
    assert result["baseline_exposure"] == 1.
    assert result["confirmed_exposure"] == 0.


def test_spot_does_not_create_trade_against_the_trend():
    hourly, asof, db = setup(False)
    trade(db, asof, "buy", 100.)
    result = compare(db, hourly, asof)
    assert result["baseline_exposure"] == result["confirmed_exposure"] == 0.


def test_later_backfill_cannot_change_a_past_decision():
    hourly, asof, db = setup()
    trade(db, asof, "sell", 100.)
    before = compare(db, hourly, asof)
    trade(db, asof, "buy", 100000., observed=asof + 1)
    assert compare(db, hourly, asof) == before


def test_missing_coverage_is_abstention_not_neutral_flow():
    hourly, asof, db = setup()
    trade(db, asof, "buy", 100.)
    db.execute("DELETE FROM health WHERE observed_ms < ?", (asof - 1000,))
    result = compare(db, hourly, asof)
    assert result["confirmed_exposure"] is None
    assert result["status"] == "abstain_missing_data"


def test_gap_blocks_confirmation_until_window_has_clean_coverage():
    hourly, asof, db = setup()
    trade(db, asof, "buy", 100.)
    bad = json.dumps({"coverage": {"BTC-USDT": {"status": "possible_gap"}}})
    db.execute("UPDATE health SET payload=? WHERE observed_ms=?", (bad, asof - 5 * 60000))
    assert compare(db, hourly, asof)["confirmed_exposure"] is None


def test_unclosed_or_future_candle_cannot_affect_baseline():
    hourly, asof, _ = setup()
    before = baseline_exposure(hourly, asof)
    hourly.loc[hourly.index[-1] + pd.Timedelta(hours=1), "close"] = .0001
    assert baseline_exposure(hourly, asof) == before


def test_missing_hour_suppresses_baseline():
    hourly, asof, _ = setup()
    assert baseline_exposure(hourly.drop(hourly.index[-5]), asof) is None


def test_stale_or_empty_trades_are_not_valid_confirmation():
    _, asof, db = setup()
    assert not spot_window(db, asof, 15)["usable"]
    trade(db, asof - 120000, "buy", 100., observed=asof)
    assert not spot_window(db, asof, 15)["usable"]


def test_paper_fees_are_paid_on_changes_not_every_forecast():
    records = [{"as_of_ms": i + 1, "reference_price": p, "baseline_exposure": target, "confirmed_exposure": target}
               for i, (p, target) in enumerate([(100., 1.), (110., 1.), (120., 0.)])]
    result = replay_paired_portfolios(records, cost_per_side=.01)
    portfolio = result["portfolios"]["baseline"]
    assert np.isclose(portfolio["net_return"], 1.2 * .99**2 - 1)
    assert portfolio["entries"] == portfolio["exits"] == 1


def test_missing_confirmation_exits_paper_to_cash_at_observed_price():
    records = [{"as_of_ms": 1, "reference_price": 100., "baseline_exposure": 1., "confirmed_exposure": 1.},
               {"as_of_ms": 2, "reference_price": 110., "baseline_exposure": 1., "confirmed_exposure": None},
               {"as_of_ms": 3, "reference_price": 90., "baseline_exposure": 1., "confirmed_exposure": None}]
    result = replay_paired_portfolios(records, cost_per_side=0.)
    assert np.isclose(result["portfolios"]["confirmed"]["net_return"], .1)
    assert np.isclose(result["portfolios"]["baseline"]["net_return"], -.1)
