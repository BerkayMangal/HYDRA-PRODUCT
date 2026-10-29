"""Causal long/flat SPOT research engine. No order execution capability."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import ROOT


def grid():
    specs = []

    def add(family, **kwargs):
        for values in itertools.product(*kwargs.values()):
            p = {"family": family, **dict(zip(kwargs, values))}
            if family == "dual_ema" and p["fast"] >= p["slow"]:
                continue
            if family == "breakout" and p["exit_window"] >= p["window"]:
                continue
            for tf in [1, 4]:
                for risk in [0, 0.6, 1.0]:
                    spec = {**p, "hours": tf, "vol_target": risk}
                    spec["id"] = hashlib.sha256(
                        json.dumps(spec, sort_keys=True).encode()
                    ).hexdigest()[:16]
                    specs.append(spec)

    add(
        "trend",
        slow=[48, 72, 96, 144, 200, 288, 400, 600],
        slope=[0, 6, 12, 24, 48, 72],
        buffer=[0, 0.0025, 0.005, 0.01],
    )
    add(
        "dual_ema",
        fast=[4, 8, 12, 24, 36, 48, 72],
        slow=[48, 72, 96, 144, 200, 288, 400, 600],
        buffer=[0, 0.0025, 0.005, 0.01],
    )
    add(
        "breakout",
        window=[12, 24, 36, 48, 72, 96, 144, 240, 336, 504],
        exit_window=[6, 12, 24, 48, 72, 96],
        trend_filter=[0, 100, 200, 400],
        atr_buffer=[0, 0.5, 1.0],
    )
    add(
        "rsi_pullback",
        period=[6, 10, 14, 21, 28],
        entry=[15, 20, 25, 30, 35, 40],
        exit=[45, 50, 55, 60, 65, 70],
        trend_filter=[0, 100, 200, 400],
    )
    add(
        "bollinger",
        window=[12, 24, 36, 48, 72, 96],
        entry_z=[1.0, 1.5, 2.0, 2.5, 3.0],
        exit_z=[-0.5, 0.0, 0.5],
        trend_filter=[0, 100, 200, 400],
    )
    add(
        "momentum",
        window=[6, 12, 24, 48, 72, 96, 168, 336, 504, 720],
        threshold=[0, 0.005, 0.01, 0.02, 0.04, 0.08],
        trend_filter=[0, 100, 200, 400],
    )
    return specs


def load(path=None):
    df = pd.read_csv(path or ROOT / "data/btc-usdt-1h.csv")
    df.index = pd.to_datetime(df.ts, unit="ms", utc=True)
    if df.index.duplicated().any() or not np.all(np.diff(df.ts) == 3600000):
        raise ValueError("Data gaps or duplicates")
    if not (df["confirm"] == 1).all():
        raise ValueError("Unconfirmed candle")
    return df


class Features:
    def __init__(self, df):
        self.hourly = df
        self.cache = {}
        self.frames = {1: df}
        f = df.resample("4h", label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "vol": "sum",
                "ts": "count",
            }
        )
        self.frames[4] = f[f.ts == 4]

    def get(self, tf, kind, n):
        key = (tf, kind, n)
        if key in self.cache:
            return self.cache[key]
        df = self.frames[tf]
        c = df.close
        if kind == "ema":
            v = c.ewm(span=n, adjust=False).mean()
        elif kind == "rsi":
            delta = c.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
            v = 100 - 100 / (1 + gain / loss)
            v = v.where((gain + loss) > 0, 50.0)
        elif kind == "high":
            v = df.high.rolling(n, min_periods=n).max().shift(1)
        elif kind == "low":
            v = df.low.rolling(n, min_periods=n).min().shift(1)
        elif kind == "z":
            v = (c - c.rolling(n).mean()) / c.rolling(n).std(ddof=1).replace(0, np.nan)
        elif kind == "momentum":
            v = c.pct_change(n, fill_method=None)
        elif kind == "atr":
            prev = c.shift(1)
            tr = pd.concat(
                [df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()],
                axis=1,
            ).max(axis=1)
            v = tr.ewm(alpha=1 / n, adjust=False).mean()
        elif kind == "vol":
            v = c.pct_change(fill_method=None).rolling(n).std(ddof=1) * np.sqrt(
                8760 / tf
            )
        else:
            raise KeyError(kind)
        self.cache[key] = v
        return v

    def position(self, spec):
        tf = spec["hours"]
        df = self.frames[tf]
        c = df.close
        fam = spec["family"]
        filt = pd.Series(True, index=df.index)
        if spec.get("trend_filter", 0):
            filt = c > self.get(tf, "ema", spec["trend_filter"])
        if fam == "trend":
            ema = self.get(tf, "ema", spec["slow"])
            b = spec["buffer"]
            slope = (ema > ema.shift(spec["slope"])) if spec["slope"] else filt
            entry = (c > ema * (1 + b)) & slope
            exit = (c < ema * (1 - b)) | ~slope
        elif fam == "dual_ema":
            fast = self.get(tf, "ema", spec["fast"])
            slow = self.get(tf, "ema", spec["slow"])
            b = spec["buffer"]
            entry = fast > slow * (1 + b)
            exit = fast < slow * (1 - b)
        elif fam == "breakout":
            atr = self.get(tf, "atr", 14)
            entry = (
                c > self.get(tf, "high", spec["window"]) + spec["atr_buffer"] * atr
            ) & filt
            exit = (c < self.get(tf, "low", spec["exit_window"])) | ~filt
        elif fam == "rsi_pullback":
            rsi = self.get(tf, "rsi", spec["period"])
            entry = (rsi < spec["entry"]) & filt
            exit = (rsi > spec["exit"]) | ~filt
        elif fam == "bollinger":
            z = self.get(tf, "z", spec["window"])
            entry = (z < -spec["entry_z"]) & filt
            exit = (z > spec["exit_z"]) | ~filt
        elif fam == "momentum":
            mom = self.get(tf, "momentum", spec["window"])
            entry = (mom > spec["threshold"]) & filt
            exit = (mom < 0) | ~filt
        else:
            raise KeyError(fam)
        # Entry/exit events set an explicit state, not a backfilled future state.
        events = np.where(exit, 0.0, np.where(entry, 1.0, np.nan))
        state = pd.Series(events, index=df.index).ffill().fillna(0.0)
        warmup = max(
            200,
            spec.get("window", 0),
            spec.get("slow", 0),
            spec.get("trend_filter", 0),
            spec.get("period", 0) * 5,
        )
        state.iloc[:warmup] = 0.0
        if spec["vol_target"]:
            scale = (
                (spec["vol_target"] / self.get(tf, "vol", 42))
                .clip(upper=1.0)
                .fillna(0.0)
            )
            state = state * scale
        # A bar starting at t is only available at t+timeframe. No 4H bar early use.
        state.index = state.index + pd.Timedelta(hours=tf)
        return (
            state.reindex(self.hourly.index, method="ffill")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )


def net_returns(position, returns, cost=0.0012):
    p = np.asarray(position, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if len(p) != len(r) or not len(p):
        raise ValueError("Aligned, nonempty input required")
    if (
        not np.isfinite(p).all()
        or not np.isfinite(r).all()
        or (p < 0).any()
        or (p > 1).any()
    ):
        raise ValueError("Invalid spot exposure or return")
    turnover = np.abs(np.diff(p, prepend=0.0))
    net = (1 - turnover * cost) * (1 + p * r) - 1
    net[-1] = (1 + net[-1]) * (1 - abs(p[-1]) * cost) - 1
    return net


def daily_returns(net, dates):
    s = pd.Series(np.log1p(net), index=dates)
    return np.expm1(s.resample("1D").sum())


def metrics(net, position, dates):
    net = np.asarray(net)
    p = np.asarray(position)
    daily = daily_returns(net, dates)
    logs = np.cumsum(np.log1p(net))
    eq = np.exp(logs)
    peaks = np.maximum.accumulate(np.maximum(eq, 1.0))
    sd = daily.std(ddof=1)
    total = float(np.expm1(logs[-1]))
    years = len(net) / 8760
    return {
        "net_return": total,
        "cagr": float(np.expm1(logs[-1] / years)),
        "max_dd": float((eq / peaks - 1).min()),
        "sharpe": float(daily.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
        "exposure": float(p.mean()),
        "entries": int(((p > 0) & (np.roll(p, 1) == 0)).sum())
        + int(p[0] > 0 and p[-1] > 0),
        "turnover": float(np.abs(np.diff(p, prepend=0)).sum() + p[-1]),
        "hours": len(net),
    }


def excess_series(net, p, returns, dates):
    daily = daily_returns(net, dates)
    # Descriptive exposure-matched BTC benchmark; not a causal fitted market beta.
    bh = daily_returns(returns, dates)
    return daily - bh * float(np.mean(p))


def bootstrap(excess, n_trials=1, reps=2000, block=7):
    a = np.asarray(excess, dtype=float)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(a), size=(reps, int(np.ceil(len(a) / block))))
    indices = (starts[:, :, None] + np.arange(block)) % len(a)
    sampled = a[indices.reshape(reps, -1)[:, : len(a)]].mean(axis=1)
    mean = float(a.mean())
    centered = sampled - mean
    p = float((1 + (centered >= mean).sum()) / (reps + 1))
    return {
        "mean_daily_excess": mean,
        "ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
        "p_one_sided_block_bootstrap": p,
        "p_bonferroni_search": min(1.0, p * n_trials),
        "block_days": block,
        "replicates": reps,
        "note": "Centered circular block bootstrap; exploratory, exposure matching is not factor alpha",
    }


def holm(pvalues):
    order = np.argsort(pvalues)
    out = np.zeros(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        out[i] = min(1.0, running)
    return out.tolist()
