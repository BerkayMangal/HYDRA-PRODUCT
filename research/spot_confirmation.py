"""Frozen 4H EMA baseline versus spot-flow confirmation, for prospective shadow use."""

import json
import math
import hashlib

import pandas as pd

SPECIFICATION = {
    "version": "ema_spot_v1",
    "baseline_id": "806ef45dc42abb7a",
    "baseline": "BTC-USDT spot, completed 4H bars, EMA4 > EMA200, long/flat, no leverage",
    "confirmation": "Both observed 5m and 15m spot buy-minus-sell quote-volume imbalance > 0",
    "data_gate": "Known trade-ID overlap on every spot poll; max 90s poll gap and 60s latest-trade age; as-of event and receipt times",
    "execution": "Research only. Record both variants at the same receipt time and reference quote; no automatic promotion",
    "missing_data_policy": "No confirmed exposure; paper replay exits to cash at the current recorded reference quote",
    "fees_per_side": 0.0012,
    "stress_fees_per_side": 0.0022,
    "status": "Unvalidated hypothesis. Do not tune thresholds using evaluation outcomes.",
}
SPECIFICATION_SHA256 = hashlib.sha256(
    json.dumps(SPECIFICATION, sort_keys=True).encode()
).hexdigest()


def baseline_exposure(hourly, as_of_ms):
    """Only closed, continuous source candles available before this decision."""
    frame = hourly.sort_index().copy()
    if frame.index.tz is None:
        raise ValueError("UTC-aware candle index required")
    cutoff = pd.Timestamp(as_of_ms, unit="ms", tz="UTC")
    frame = frame[frame.index + pd.Timedelta(hours=1) <= cutoff]
    if frame.index.duplicated().any() or frame.empty:
        return None
    if (frame.index.to_series().diff().dropna() != pd.Timedelta(hours=1)).any():
        return None
    bars = frame.resample("4h", label="left", closed="left").agg(
        {"close": ["last", "count"]}
    )
    close = bars.loc[bars[("close", "count")] == 4, ("close", "last")]
    if len(close) <= 200 or close.index[-1] + pd.Timedelta(hours=4) < cutoff.floor(
        "4h"
    ):
        return None
    fast = close.ewm(span=4, adjust=False).mean().iloc[-1]
    slow = close.ewm(span=200, adjust=False).mean().iloc[-1]
    return float(fast > slow)


def spot_window(db, as_of_ms, minutes):
    """No backfilled trade is usable before its actual observed_ms receipt time."""
    start = as_of_ms - minutes * 60000
    records = db.execute(
        "SELECT observed_ms,payload FROM health WHERE observed_ms BETWEEN ? AND ? ORDER BY observed_ms",
        (start - 90000, as_of_ms),
    ).fetchall()
    before = [row for row in records if row[0] <= start]
    if not before or not records or records[-1][0] < as_of_ms - 60000:
        return {"usable": False, "reason": "coverage_warmup_or_stale"}
    records = [before[-1]] + [row for row in records if row[0] > start]
    if any(b[0] - a[0] > 90000 for a, b in zip(records, records[1:])):
        return {"usable": False, "reason": "poll_gap"}
    for _, payload in records:
        status = json.loads(payload)
        coverage = status.get("coverage", {}).get("BTC-USDT", {})
        if coverage.get("status") != "overlap_observed":
            return {"usable": False, "reason": "unverified_trade_coverage"}
        if any(
            key in status.get("errors", {})
            for key in ("trades:BTC-USDT", "backfill:BTC-USDT")
        ):
            return {"usable": False, "reason": "source_error"}
    rows = db.execute(
        "SELECT side,SUM(quote_size),MAX(event_ms),COUNT(*) FROM trades "
        "WHERE instrument='BTC-USDT' AND event_ms>? AND event_ms<=? AND observed_ms<=? GROUP BY side",
        (start, as_of_ms, as_of_ms),
    ).fetchall()
    totals = {side: float(volume) for side, volume, _, _ in rows}
    buy, sell = totals.get("buy", 0.0), totals.get("sell", 0.0)
    total = buy + sell
    newest = max((row[2] for row in rows), default=0)
    if total <= 0 or not math.isfinite(total) or as_of_ms - newest > 60000:
        return {"usable": False, "reason": "no_fresh_valid_trades"}
    return {
        "usable": True,
        "imbalance": (buy - sell) / total,
        "buy_quote": buy,
        "sell_quote": sell,
        "trades": sum(row[3] for row in rows),
        "window_minutes": minutes,
    }


def compare(db, hourly, as_of_ms):
    baseline = baseline_exposure(hourly, as_of_ms)
    windows = {str(minutes): spot_window(db, as_of_ms, minutes) for minutes in (5, 15)}
    usable = baseline is not None and all(row["usable"] for row in windows.values())
    confirmed = (
        float(baseline > 0 and all(row["imbalance"] > 0 for row in windows.values()))
        if usable
        else None
    )
    return {
        "specification": SPECIFICATION,
        "specification_sha256": SPECIFICATION_SHA256,
        "as_of_ms": as_of_ms,
        "baseline_exposure": baseline,
        "confirmed_exposure": confirmed,
        "flow": windows,
        "status": "research_only" if usable else "abstain_missing_data",
        "orders_enabled": False,
    }


def replay_paired_portfolios(records, cost_per_side=0.0012):
    """Causal binary-exposure accounting, with fees only on position changes.

    Quotes are recorded references, not executable fills. Equity is sampled at
    decision times; intraperiod drawdown and fill risk are not measured here.
    """
    portfolios = {
        name: {"cash": 1.0, "btc": 0.0, "entries": 0, "exits": 0, "fees": 0.0}
        for name in ("baseline", "confirmed")
    }
    observations = 0
    eligible = 0
    previous = -1
    last_price = None
    for record in records:
        timestamp = record["as_of_ms"]
        price = float(record["reference_price"])
        if timestamp <= previous or not math.isfinite(price) or price <= 0:
            raise ValueError("Ordered decisions and positive reference prices required")
        previous, last_price = timestamp, price
        observations += 1
        eligible += int(record["confirmed_exposure"] is not None)
        for name, portfolio in portfolios.items():
            target = record[name + "_exposure"]
            if target not in (None, 0.0, 1.0):
                raise ValueError("Binary exposure required")
            target = 0.0 if target is None else target
            if target and not portfolio["btc"]:
                fee = portfolio["cash"] * cost_per_side
                portfolio["btc"] = (portfolio["cash"] - fee) / price
                portfolio["cash"] = 0.0
                portfolio["entries"] += 1
                portfolio["fees"] += fee
            elif not target and portfolio["btc"]:
                value = portfolio["btc"] * price
                fee = value * cost_per_side
                portfolio["cash"], portfolio["btc"] = value - fee, 0.0
                portfolio["exits"] += 1
                portfolio["fees"] += fee
    for portfolio in portfolios.values():
        nav = portfolio["cash"] + portfolio["btc"] * (last_price or 0.0)
        portfolio["marked_equity"] = nav
        portfolio["net_return"] = nav - 1.0
    return {
        "observations": observations,
        "confirmation_available": eligible,
        "cost_per_side": cost_per_side,
        "portfolios": portfolios,
        "limits": "Hypothetical reference-price accounting; no executable fills; limited prospective sample",
    }
