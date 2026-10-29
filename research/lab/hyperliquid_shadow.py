"""Bounded public Hyperliquid wallet research journal. No signing or orders."""

import argparse, hashlib, json, signal, sqlite3, sys, time, shutil, os
import re
from datetime import datetime, timezone
from pathlib import Path

from .paths import ROOT as LAB_ROOT, write_json
from .runtime import validate_budget

ROOT = LAB_ROOT / "hyperliquid"
from collectors.hyperliquid_research import PublicHyperliquid
from research.hyperliquid_features import (
    market_contexts,
    sampled_positions,
    realized_liquidation_fills,
    signed_inventory_change,
    SPECIFICATION,
    SPECIFICATION_SHA256,
)
from research.wallet_breadth import btc_wallet_change


def connect():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / "observations.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
 CREATE TABLE IF NOT EXISTS raw_snapshots(id INTEGER PRIMARY KEY,kind TEXT,wallet TEXT,event_ms INTEGER,observed_ms INTEGER,payload TEXT,sha256 TEXT);
 CREATE TABLE IF NOT EXISTS fills(coin TEXT,trade_id TEXT,event_ms INTEGER,first_observed_ms INTEGER,payload TEXT,PRIMARY KEY(coin,trade_id));
 CREATE TABLE IF NOT EXISTS liquidations(coin TEXT,trade_id TEXT,event_ms INTEGER,first_observed_ms INTEGER,payload TEXT,PRIMARY KEY(coin,trade_id));
 CREATE TABLE IF NOT EXISTS features(observed_ms INTEGER PRIMARY KEY,spec_hash TEXT,payload TEXT);
 CREATE TABLE IF NOT EXISTS health(observed_ms INTEGER PRIMARY KEY,payload TEXT);
 CREATE TABLE IF NOT EXISTS fill_requests(wallet TEXT,observed_ms INTEGER,start_ms INTEGER,end_ms INTEGER,rows INTEGER,truncated INTEGER);
 CREATE INDEX IF NOT EXISTS fill_requests_wallet ON fill_requests(wallet,end_ms);
 CREATE INDEX IF NOT EXISTS fills_event ON fills(event_ms);
 """)
    return db


def save_raw(db, kind, wallet, body, observed):
    raw = json.dumps(body, separators=(",", ":"))
    event = body.get("time") if isinstance(body, dict) else None
    db.execute(
        "INSERT INTO raw_snapshots(kind,wallet,event_ms,observed_ms,payload,sha256) VALUES(?,?,?,?,?,?)",
        (kind, wallet, event, observed, raw, hashlib.sha256(raw.encode()).hexdigest()),
    )


def cycle(db, client, cohort, lookback_seconds=180):
    if lookback_seconds < 180:
        raise ValueError("At least 180 seconds of fill lookback required")
    errors = {}
    states = []
    wallet_fills = []
    truncated = []
    body = client.query("metaAndAssetCtxs")
    meta_observed = int(time.time() * 1000)
    contexts = market_contexts(body)
    save_raw(db, "metaAndAssetCtxs", None, body, meta_observed)
    for wallet in cohort["wallets"]:
        try:
            state = client.query("clearinghouseState", user=wallet)
            observed = int(time.time() * 1000)
            save_raw(db, "clearinghouseState", wallet, state, observed)
            states.append({"wallet": wallet, "observed_ms": observed, "state": state})
        except Exception as e:
            errors["state:" + wallet] = type(e).__name__
            states.append(
                {
                    "wallet": wallet,
                    "observed_ms": int(time.time() * 1000),
                    "state": None,
                }
            )
        # Small overlapping recent windows; full historical coverage is never assumed.
        try:
            end = int(time.time() * 1000)
            start = end - int(lookback_seconds * 1000)
            previous_end = db.execute(
                "SELECT MAX(end_ms) FROM fill_requests WHERE wallet=?", (wallet,)
            ).fetchone()[0]
            if previous_end is not None and start > previous_end:
                errors["fill_window_gap:" + wallet] = (
                    "A source interval was not requested; history remains sampled"
                )
            fills = client.query(
                "userFillsByTime",
                user=wallet,
                startTime=start,
                endTime=end,
                aggregateByTime=False,
            )
            observed = int(time.time() * 1000)
            if not isinstance(fills, list):
                raise ValueError("Unexpected fills schema")
            if len(fills) >= 2000:
                truncated.append(wallet)
            db.execute(
                "INSERT INTO fill_requests VALUES(?,?,?,?,?,?)",
                (wallet, observed, start, end, len(fills), int(len(fills) >= 2000)),
            )
            for fill in fills:
                db.execute(
                    "INSERT OR IGNORE INTO fills VALUES(?,?,?,?,?)",
                    (
                        fill["coin"],
                        str(fill["tid"]),
                        int(fill["time"]),
                        observed,
                        json.dumps(fill),
                    ),
                )
            wallet_fills.append((wallet, fills, observed))
        except Exception as e:
            errors["fills:" + wallet] = type(e).__name__
        time.sleep(0.15)
    now = int(time.time() * 1000)
    features = sampled_positions(states, contexts, now)
    features["specification_sha256"] = SPECIFICATION_SHA256
    features["context_observed_ms"] = meta_observed
    features["as_of_ms"] = now
    features["markets"] = {
        k: v for k, v in contexts.items() if k in ["BTC", "ETH", "SOL", "HYPE"]
    }
    features["cohort_frozen_at_ms"] = cohort["frozen_at_ms"]
    features["cohort_sha256"] = hashlib.sha256(
        json.dumps(cohort, sort_keys=True).encode()
    ).hexdigest()
    features["scope"] = cohort["scope"]
    features["cohort_size_selected"] = True
    previous = db.execute(
        "SELECT observed_ms,payload FROM features WHERE observed_ms BETWEEN ? AND ? ORDER BY ABS(observed_ms-?) LIMIT 1",
        (now - 18 * 60000, now - 14 * 60000, now - 15 * 60000),
    ).fetchone()
    features["sampled_inventory_change"] = (
        signed_inventory_change(json.loads(previous[1]), features) if previous else None
    )
    features["sampled_wallet_change_btc"] = (
        btc_wallet_change(json.loads(previous[1]), features) if previous else None
    )
    features["inventory_change_actual_window_minutes"] = (
        (now - previous[0]) / 60000 if previous else None
    )
    events = realized_liquidation_fills(wallet_fills)
    for event in events:
        event["collection_phase"] = (
            "prospective_receipt"
            if event["event_ms"] >= cohort["frozen_at_ms"]
            else "bootstrap_history"
        )
        db.execute(
            "INSERT OR IGNORE INTO liquidations VALUES(?,?,?,?,?)",
            (
                event["coin"],
                str(event["trade_id"]),
                event["event_ms"],
                event["observed_ms"],
                json.dumps(event),
            ),
        )
    features["recent_explicit_liquidation_fills"] = len(events)
    features["fills_may_be_truncated_wallets"] = truncated
    status = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "wallets_requested": len(cohort["wallets"]),
        "wallets_successful": features["successful_wallets"],
        "position_count": len(features["positions"]),
        "explicit_liquidation_fills_in_window": len(events),
        "fills_may_be_truncated": len(truncated),
        "fills_requested_lookback_seconds": lookback_seconds,
        "errors": errors,
        "orders_enabled": False,
        "scope": cohort["scope"],
        "specification_sha256": SPECIFICATION_SHA256,
        "cohort_sha256": features["cohort_sha256"],
    }
    db.execute(
        "INSERT INTO features VALUES(?,?,?)",
        (now, SPECIFICATION_SHA256, json.dumps(features)),
    )
    db.execute("INSERT INTO health VALUES(?,?)", (now, json.dumps(status)))
    db.commit()
    write_json(ROOT / "status.json", status)
    write_json(ROOT / "latest-features.json", features)
    print(json.dumps(status), flush=True)
    return status


def validate_cohort(cohort, interval):
    wallets = cohort["wallets"]
    if not 1 <= len(wallets) <= 24:
        raise ValueError("A frozen cohort of 1 to 24 public wallets is required")
    if any(
        not isinstance(wallet, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}", wallet)
        for wallet in wallets
    ):
        raise ValueError("Valid public account addresses required")
    if len({wallet.lower() for wallet in wallets}) != len(wallets):
        raise ValueError("Duplicate wallet addresses cannot count as extra accounts")
    if len(wallets) > 12 and interval < 240:
        raise ValueError(
            "Cohorts larger than 12 require at least 240 seconds between polls"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--until")
    args = p.parse_args()
    if args.interval < 60:
        raise ValueError("At least 60 seconds between wallet polls")
    until = validate_budget(args.once, args.until)
    ROOT.mkdir(parents=True, exist_ok=True)
    cohort = json.loads((ROOT / "large-wallet-cohort.json").read_text())
    validate_cohort(cohort, args.interval)
    # PID lock only protects this worker; never interfere with the existing OKX worker.
    import fcntl

    lock = (ROOT / "collector.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / "collector.pid").write_text(str(os.getpid()))
    (ROOT / "specification.json").write_text(
        json.dumps(
            {"specification": SPECIFICATION, "sha256": SPECIFICATION_SHA256}, indent=2
        )
    )
    stop = False

    def halt(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)
    db = connect()
    client = PublicHyperliquid()
    while not stop and (until is None or datetime.now(timezone.utc) < until):
        if shutil.disk_usage(ROOT).free < 1024**3:
            raise RuntimeError("Less than 1GiB free; journal stopped")
        try:
            cycle(db, client, cohort, lookback_seconds=max(180, args.interval + 120))
        except Exception as e:
            db.rollback()
            failed = {
                "cycle_error": type(e).__name__,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "orders_enabled": False,
            }
            db.execute(
                "INSERT OR REPLACE INTO health VALUES(?,?)",
                (int(time.time() * 1000), json.dumps(failed)),
            )
            db.commit()
            write_json(ROOT / "status.json", failed)
            print(json.dumps(failed), flush=True)
            if args.once:
                raise
        if args.once:
            break
        deadline = time.monotonic() + args.interval
        while not stop and time.monotonic() < deadline:
            if until is not None and datetime.now(timezone.utc) >= until:
                break
            time.sleep(0.5)
    db.close()
    status_path = ROOT / "status.json"
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
