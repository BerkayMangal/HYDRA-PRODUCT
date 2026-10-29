"""Public-data collector and experimental forecasts. Contains no order API calls.

Runs only in the current machine. SQLite journal survives process restarts;
process survival/publication is not assumed. Never emits production advice.
"""

import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import time
import shutil
from datetime import datetime, timezone

import pandas as pd
import requests

from .engine import ROOT, Features, load
from .paths import write_json
from .runtime import validate_budget
from research.spot_confirmation import compare, SPECIFICATION

BASE = "https://www.okx.com/api/v5/"
ALLOWED = {
    "public/instruments",
    "market/trades",
    "market/history-trades",
    "market/ticker",
    "market/candles",
    "public/open-interest",
    "public/funding-rate",
    "market/books",
}


def get(path, params):
    if path not in ALLOWED:
        raise ValueError("Only read-only public market endpoints are allowed")
    r = requests.get(BASE + path, params=params, timeout=15)
    r.raise_for_status()
    body = r.json()
    if body.get("code") != "0":
        raise ValueError(body.get("msg", "API error"))
    return body


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY,observed_ms INTEGER,kind TEXT,instrument TEXT,payload TEXT,sha256 TEXT);
    CREATE TABLE IF NOT EXISTS trades(instrument TEXT,trade_id TEXT,event_ms INTEGER,observed_ms INTEGER,side TEXT,price REAL,raw_size REAL,btc_size REAL,quote_size REAL,PRIMARY KEY(instrument,trade_id));
    CREATE INDEX IF NOT EXISTS trades_event ON trades(instrument,event_ms);
    CREATE TABLE IF NOT EXISTS candles(event_ms INTEGER PRIMARY KEY,observed_ms INTEGER,open REAL,high REAL,low REAL,close REAL,volume REAL);
    CREATE TABLE IF NOT EXISTS health(observed_ms INTEGER PRIMARY KEY,status TEXT,payload TEXT);
    CREATE TABLE IF NOT EXISTS comparison_decisions(bar_end_ms INTEGER PRIMARY KEY,observed_ms INTEGER,payload TEXT);
    CREATE TABLE IF NOT EXISTS forecasts(id INTEGER PRIMARY KEY,spec_id TEXT,bar_end_ms INTEGER,observed_ms INTEGER,exposure REAL,reference_price REAL,label TEXT,UNIQUE(spec_id,bar_end_ms));
    """)
    return db


def insert_trades(db, instrument, rows, observed_ms, ctval):
    inserted = 0
    for row in rows:
        px = float(row["px"])
        sz = float(row["sz"])
        btc = sz * ctval
        if (
            not math.isfinite(px)
            or not math.isfinite(sz)
            or px <= 0
            or sz < 0
            or row["side"] not in ["buy", "sell"]
        ):
            raise ValueError("Invalid trade")
        cur = db.execute(
            "INSERT OR IGNORE INTO trades VALUES(?,?,?,?,?,?,?,?,?)",
            (
                instrument,
                str(row["tradeId"]),
                int(row["ts"]),
                observed_ms,
                row["side"],
                px,
                sz,
                btc,
                btc * px,
            ),
        )
        inserted += cur.rowcount
    return inserted


def save_snapshot(db, observed, kind, instrument, body):
    payload = json.dumps(body, separators=(",", ":"))
    db.execute(
        "INSERT INTO snapshots(observed_ms,kind,instrument,payload,sha256) VALUES(?,?,?,?,?)",
        (
            observed,
            kind,
            instrument,
            payload,
            hashlib.sha256(payload.encode()).hexdigest(),
        ),
    )


def recover_window(db, instrument, rows):
    """Find an actual previously stored trade ID, not just a timestamp overlap."""

    def overlaps(batch):
        return any(
            db.execute(
                "SELECT 1 FROM trades WHERE instrument=? AND trade_id=?",
                (instrument, str(x["tradeId"])),
            ).fetchone()
            is not None
            for x in batch
        )

    if overlaps(rows):
        return [], True
    backfills = []
    if not rows:
        return backfills, False
    cursor = min(int(x["tradeId"]) for x in rows)
    for _ in range(20):
        body = get(
            "market/history-trades",
            {"instId": instrument, "after": str(cursor), "limit": "100"},
        )
        observed = int(time.time() * 1000)
        batch = body["data"]
        save_snapshot(db, observed, "trade_backfill", instrument, body)
        if not batch:
            break
        backfills.append((observed, batch))
        if overlaps(batch):
            return backfills, True
        oldest = min(int(x["tradeId"]) for x in batch)
        if oldest >= cursor:
            break
        cursor = oldest
        time.sleep(0.12)
    return backfills, False


def cycle(db, ctval, history, specs):
    tasks = {}
    for inst in ["BTC-USDT", "BTC-USDT-SWAP"]:
        tasks["trades:" + inst] = ("market/trades", {"instId": inst, "limit": "500"})
        tasks["ticker:" + inst] = ("market/ticker", {"instId": inst})
        tasks["books:" + inst] = ("market/books", {"instId": inst, "sz": "20"})
    tasks["open_interest:BTC-USDT-SWAP"] = (
        "public/open-interest",
        {"instType": "SWAP", "instId": "BTC-USDT-SWAP"},
    )
    tasks["funding:BTC-USDT-SWAP"] = (
        "public/funding-rate",
        {"instId": "BTC-USDT-SWAP"},
    )
    tasks["candles:BTC-USDT"] = (
        "market/candles",
        {"instId": "BTC-USDT", "bar": "1H", "limit": "100"},
    )
    results = {}
    errors = {}
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(get, *args): key for key, args in tasks.items()}
        for f in cf.as_completed(futures):
            key = futures[f]
            try:
                results[key] = (int(time.time() * 1000), f.result())
            except Exception as e:
                errors[key] = str(e)
    counts = {}
    coverage = {}
    for key, (observed, body) in results.items():
        kind, inst = key.split(":")
        rows = body["data"]
        save_snapshot(db, observed, kind, inst, body)
        if kind == "trades":
            previous = db.execute(
                "SELECT MAX(event_ms) FROM trades WHERE instrument=?", (inst,)
            ).fetchone()[0]
            oldest = min(int(x["ts"]) for x in rows) if rows else None
            backfills = []
            overlap = False
            if previous is not None:
                try:
                    backfills, overlap = recover_window(db, inst, rows)
                except Exception as exc:
                    errors["backfill:" + inst] = str(exc)
            counts[inst] = insert_trades(
                db, inst, rows, observed, 1.0 if inst == "BTC-USDT" else ctval
            )
            for fetched, batch in backfills:
                counts[inst] += insert_trades(
                    db, inst, batch, fetched, 1.0 if inst == "BTC-USDT" else ctval
                )
            coverage[inst] = {
                "previous_event_ms": previous,
                "earliest_returned_ms": oldest,
                "overlaps_previous_window": overlap,
                "backfill_pages": len(backfills),
                "status": (
                    "initial_window"
                    if previous is None
                    else "overlap_observed" if overlap else "possible_gap"
                ),
            }
            if previous is not None and not overlap:
                errors["coverage:" + inst] = (
                    "Potential trade gap; window completeness not certified"
                )
        elif kind == "ticker" and rows:
            age = (observed - int(rows[0]["ts"])) / 1000
            if age > 120 or age < -5:
                errors["freshness:" + inst] = (
                    "Ticker event/receipt time outside freshness bounds"
                )
        elif kind == "candles":
            for row in rows:
                if str(row[-1]) != "1":
                    continue
                ts = int(row[0])
                o, h, l, c, v = map(float, row[1:6])
                if min(o, h, l, c) <= 0 or v < 0 or l > min(o, c) or h < max(o, c):
                    raise ValueError("Invalid candle")
                db.execute(
                    "INSERT OR IGNORE INTO candles VALUES(?,?,?,?,?,?,?)",
                    (ts, observed, o, h, l, c, v),
                )
    now = int(time.time() * 1000)
    flow = {}
    for inst in ["BTC-USDT", "BTC-USDT-SWAP"]:
        rows = db.execute(
            "SELECT side,SUM(quote_size),COUNT(*) FROM trades WHERE instrument=? AND event_ms>=? AND event_ms<=? GROUP BY side",
            (inst, now - 300000, now),
        ).fetchall()
        totals = {r[0]: float(r[1]) for r in rows}
        buy = totals.get("buy", 0.0)
        sell = totals.get("sell", 0.0)
        flow[inst] = {
            "buy_quote_usdt": buy,
            "sell_quote_usdt": sell,
            "net_quote_usdt": buy - sell,
            "imbalance": (buy - sell) / (buy + sell) if buy + sell else None,
            "warning": "Observed sample only; 5-minute completeness not certified. Inspect coverage/gap history.",
        }
    created = 0
    comparison_input = None
    new_comparison = False
    if "candles:BTC-USDT" in results and "ticker:BTC-USDT" in results and specs:
        rows = db.execute(
            "SELECT event_ms,open,high,low,close,volume FROM candles ORDER BY event_ms"
        ).fetchall()
        extra = pd.DataFrame(
            rows, columns=["ts", "open", "high", "low", "close", "vol"]
        )
        extra.index = pd.to_datetime(extra.ts, unit="ms", utc=True)
        merged = pd.concat([history, extra]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        if not (
            (merged.index.to_series().diff().dropna()) == pd.Timedelta(hours=1)
        ).all():
            errors["forecast"] = "History gap; forecasts suppressed until backfilled"
        else:
            # Append the current open as an unobserved placeholder ONLY to provide
            # the next-open index. Its close cannot affect its own exposure.
            next_ts = merged.index[-1] + pd.Timedelta(hours=1)
            latest = results["ticker:BTC-USDT"][1]["data"][0]
            reference = float(latest["last"])
            merged.loc[next_ts, ["open", "high", "low", "close", "vol"]] = [
                reference,
                reference,
                reference,
                reference,
                0.0,
            ]
            features = Features(merged)
            bar_end = int(next_ts.timestamp() * 1000)
            if now - bar_end > 3720000 or now < bar_end:
                errors["forecast"] = (
                    "Completed bar outside hourly freshness bounds; forecast suppressed"
                )
            elif "freshness:BTC-USDT" in errors:
                errors["forecast"] = (
                    "Spot reference price is stale; forecast suppressed"
                )
            else:
                comparison_input = (merged.iloc[:-1].copy(), bar_end, reference)
                for spec in specs:
                    position = float(features.position(spec)[-1])
                    cur = db.execute(
                        "INSERT OR IGNORE INTO forecasts(spec_id,bar_end_ms,observed_ms,exposure,reference_price,label) VALUES(?,?,?,?,?,?)",
                        (
                            spec["id"],
                            bar_end,
                            now,
                            position,
                            reference,
                            "EXPERIMENTAL_RESEARCH_ONLY_NO_ORDERS"
                            + ("_LATE_STARTUP" if now - bar_end > 120000 else ""),
                        ),
                    )
                    created += cur.rowcount
    status = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "public_data_only": True,
        "orders_enabled": False,
        "new_trades": counts,
        "coverage": coverage,
        "flow_5m_observed": flow,
        "new_forecasts": created,
        "errors": errors,
    }
    db.execute(
        "INSERT OR REPLACE INTO health VALUES(?,?,?)",
        (now, "degraded" if errors else "collecting", json.dumps(status)),
    )
    if comparison_input is not None:
        hourly, bar_end, reference = comparison_input
        if (
            db.execute(
                "SELECT 1 FROM comparison_decisions WHERE bar_end_ms=?", (bar_end,)
            ).fetchone()
            is None
        ):
            decision = compare(db, hourly, now)
            new_comparison = True
            decision["reference_price"] = reference
            db.execute(
                "INSERT INTO comparison_decisions VALUES(?,?,?)",
                (bar_end, now, json.dumps(decision)),
            )
            for spec_id, exposure in (
                ("EMA_BASELINE_V1", decision["baseline_exposure"]),
                ("EMA_SPOT_V1", decision["confirmed_exposure"]),
            ):
                if exposure is not None:
                    db.execute(
                        "INSERT OR IGNORE INTO forecasts(spec_id,bar_end_ms,observed_ms,exposure,reference_price,label) VALUES(?,?,?,?,?,?)",
                        (
                            spec_id,
                            bar_end,
                            now,
                            exposure,
                            reference,
                            "PROSPECTIVE_EMA_SPOT_V1_RESEARCH_ONLY",
                        ),
                    )
            status["ema_spot_comparison"] = decision
            db.execute(
                "UPDATE health SET payload=? WHERE observed_ms=?",
                (json.dumps(status), now),
            )
    db.commit()
    if new_comparison:
        from .report import generate

        generate()
    write_json(ROOT / "shadow-status.json", status)
    print(json.dumps(status), flush=True)
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument(
        "--until",
        help="UTC/offset ISO timestamp to stop; e.g. 2026-10-28T05:00:00+00:00",
    )
    args = p.parse_args()
    if args.interval < 15:
        raise ValueError("Minimum polling interval 15 seconds")
    until = validate_budget(args.once, args.until)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "data/shadow.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / "shadow.pid").write_text(str(os.getpid()))
    db = connect(ROOT / "data/shadow.sqlite3")
    metadata = get(
        "public/instruments", {"instType": "SWAP", "instId": "BTC-USDT-SWAP"}
    )
    instrument = metadata["data"][0]
    if instrument["ctValCcy"] != "BTC" or float(instrument.get("ctMult") or 1) != 1:
        raise ValueError("Unsupported contract units")
    ctval = float(instrument["ctVal"])
    save_snapshot(db, int(time.time() * 1000), "metadata", "BTC-USDT-SWAP", metadata)
    db.commit()
    history = load()
    selected = ROOT / "results/frozen-selected.json"
    specs = (
        [x["spec"] for x in json.loads(selected.read_text())]
        if selected.exists()
        else []
    )
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while running and (until is None or datetime.now(timezone.utc) < until):
        if shutil.disk_usage(ROOT).free < 1024**3:
            raise RuntimeError("Less than 1GiB free; journal stopped")
        try:
            cycle(db, ctval, history, specs)
        except Exception as e:
            db.rollback()
            failed = {
                "errors": {"cycle": type(e).__name__},
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "orders_enabled": False,
            }
            db.execute(
                "INSERT OR REPLACE INTO health VALUES(?,?,?)",
                (int(time.time() * 1000), "degraded", json.dumps(failed)),
            )
            db.commit()
            write_json(ROOT / "shadow-status.json", failed)
            print(json.dumps(failed), flush=True)
            if args.once:
                raise
        if args.once:
            break
        deadline = time.monotonic() + args.interval
        while running and time.monotonic() < deadline:
            if until is not None and datetime.now(timezone.utc) >= until:
                break
            time.sleep(0.5)
    db.close()
    from .report import generate

    generate()
    status_path = ROOT / "shadow-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update(
        runtime_state="stopped",
        stopped_at=datetime.now(timezone.utc).isoformat(),
        orders_enabled=False,
    )
    write_json(status_path, status)
    print(
        json.dumps({"stopped_at": status["stopped_at"], "orders_enabled": False}),
        flush=True,
    )


if __name__ == "__main__":
    main()
