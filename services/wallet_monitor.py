"""Read-only public wallet monitoring, isolated from directional decisions."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

from collectors.hyperliquid_research import PublicHyperliquid
from research.hyperliquid_features import market_contexts, sampled_positions
from research.wallet_breadth import btc_wallet_change

REPO = Path(__file__).resolve().parents[1]


def frozen_cohorts():
    evidence = REPO / "docs/research/evidence"
    manifest = json.loads((evidence / "bundle-manifest.json").read_text())
    item = next(
        row
        for row in manifest["files"]
        if row["path"] == "hyperliquid/large-wallet-cohort.json"
    )
    packed = (evidence / item["archive"]).read_bytes()
    body = gzip.decompress(packed)
    if (
        hashlib.sha256(packed).hexdigest() != item["archive_sha256"]
        or hashlib.sha256(body).hexdigest() != item["sha256"]
    ):
        raise ValueError("Frozen wallet cohort checksum mismatch")
    primary = json.loads(body)
    extra = json.loads(
        (evidence / "continued-20261029/additional-frozen-cohort.json").read_text()
    )
    cohorts = {"original": primary, "additional": extra}
    for cohort in cohorts.values():
        from research.lab.hyperliquid_shadow import validate_cohort

        validate_cohort(cohort, 240)
    if len(set(primary["wallets"]) | set(extra["wallets"])) > 24:
        raise ValueError("Live monitor supports at most 24 unique accounts")
    return cohorts


class WalletMonitor:
    def __init__(
        self,
        cohorts=None,
        *,
        client_factory=PublicHyperliquid,
        database=None,
        interval=240,
        clock=time.time
    ):
        if interval < 240:
            raise ValueError("Live wallet polling must be at least 240 seconds")
        self.cohorts = cohorts if cohorts is not None else frozen_cohorts()
        self.wallets = sorted(
            {wallet for cohort in self.cohorts.values() for wallet in cohort["wallets"]}
        )
        if not 1 <= len(self.wallets) <= 24:
            raise ValueError("A bounded, nonempty frozen cohort is required")
        from research.lab.hyperliquid_shadow import validate_cohort

        for cohort in self.cohorts.values():
            validate_cohort(cohort, interval)
        self.client_factory, self.interval, self.clock = client_factory, interval, clock
        mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
        self.database = Path(
            database or Path(mount or "data") / "wallet-monitor.sqlite3"
        )
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.thread = None
        self.history = {name: deque(maxlen=12) for name in self.cohorts}
        self.snapshot = {
            "status": "initializing",
            "orders_enabled": False,
            "requested_accounts": len(self.wallets),
            "successful_accounts": 0,
            "observed_ms": None,
            "btc_positions": [],
            "cohorts": {},
            "errors": {},
            "poll_interval_seconds": interval,
            "scope": "Frozen public-account samples; common ownership and external hedges unknown",
        }

    def _persist(self, observed_ms, raw, snapshot):
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS wallet_observations(observed_ms INTEGER PRIMARY KEY, raw_json TEXT, snapshot_json TEXT, sha256 TEXT)"
            )
            payload = json.dumps(raw, sort_keys=True, allow_nan=False)
            db.execute(
                "INSERT INTO wallet_observations VALUES(?,?,?,?)",
                (
                    observed_ms,
                    payload,
                    json.dumps(snapshot, allow_nan=False),
                    hashlib.sha256(payload.encode()).hexdigest(),
                ),
            )

    def _query(self, kind, **params):
        client = self.client_factory()
        try:
            return client.query(kind, **params)
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                close()

    def cycle(self):
        errors = {}
        raw = {"states": []}
        try:
            metadata = self._query("metaAndAssetCtxs")
            meta_ms = int(self.clock() * 1000)
            contexts = market_contexts(metadata)
            raw.update(metadata=metadata, metadata_observed_ms=meta_ms)
        except Exception as exc:
            with self.lock:
                self.snapshot = {
                    **self.snapshot,
                    "status": "unavailable",
                    "btc_positions": [],
                    "cohorts": {},
                    "successful_accounts": 0,
                    "errors": {"market_metadata": type(exc).__name__},
                    "observed_ms": None,
                    "checked_ms": int(self.clock() * 1000),
                }
            return self.get_snapshot()

        def read(wallet):
            try:
                state = self._query("clearinghouseState", user=wallet)
                return {
                    "wallet": wallet,
                    "observed_ms": int(self.clock() * 1000),
                    "state": state,
                }
            except Exception as exc:
                return {
                    "wallet": wallet,
                    "observed_ms": int(self.clock() * 1000),
                    "state": None,
                    "error": type(exc).__name__,
                }

        with ThreadPoolExecutor(max_workers=3) as pool:
            states = list(pool.map(read, self.wallets))
        raw["states"] = states
        as_of = int(self.clock() * 1000)
        if as_of - meta_ms > 120000:
            raise ValueError("Market metadata aged beyond the wallet snapshot budget")
        state_by_wallet = {row["wallet"]: row for row in states}
        all_positions, summaries, successful = [], {}, set()
        for name, cohort in self.cohorts.items():
            features = sampled_positions(
                [state_by_wallet[wallet] for wallet in cohort["wallets"]],
                contexts,
                as_of,
            )
            available = set(cohort["wallets"]) - set(features["unavailable_wallets"])
            successful.update(available)
            for wallet in features["unavailable_wallets"]:
                errors[wallet] = state_by_wallet[wallet].get(
                    "error", "InvalidOrStaleState"
                )
            candidates = [
                (timestamp, previous)
                for timestamp, previous in self.history[name]
                if 14 * 60000 <= as_of - timestamp <= 18 * 60000
            ]
            previous = (
                min(candidates, key=lambda row: abs(as_of - row[0] - 15 * 60000))
                if candidates
                else None
            )
            change = btc_wallet_change(previous[1], features) if previous else None
            summaries[name] = {
                "requested_accounts": features["requested_wallets"],
                "successful_accounts": features["successful_wallets"],
                "sampled_btc": features["sampled_assets"].get("BTC"),
                "btc_inventory_change": change,
                "change_window_minutes": (
                    (as_of - previous[0]) / 60000 if previous else None
                ),
                "frozen_at_ms": cohort["frozen_at_ms"],
                "scope": cohort["scope"],
            }
            self.history[name].append((as_of, features))
            # Shared addresses appear once in the combined BTC table.
            for position in features["positions"]:
                if position["coin"] == "BTC" and position["wallet"] not in {
                    row["wallet"] for row in all_positions
                }:
                    all_positions.append({**position, "cohort": name})
        snapshot = {
            "status": "partial" if errors else "observed",
            "orders_enabled": False,
            "requested_accounts": len(self.wallets),
            "successful_accounts": len(successful),
            "observed_ms": as_of,
            "metadata_observed_ms": meta_ms,
            "observed_at": datetime.fromtimestamp(
                as_of / 1000, timezone.utc
            ).isoformat(),
            "btc_positions": sorted(
                all_positions, key=lambda row: row["notional_usd"], reverse=True
            ),
            "cohorts": summaries,
            "errors": errors,
            "poll_interval_seconds": self.interval,
            "scope": "Sampled default-DEX BTC inventory; estimated liquidation levels, not realized forced trades",
            "storage": (
                "mounted_volume"
                if os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
                else "local_disk"
            ),
            "cold_start": not any(
                row["btc_inventory_change"] is not None for row in summaries.values()
            ),
        }
        try:
            self._persist(as_of, raw, snapshot)
            snapshot["journal_written"] = True
        except (OSError, sqlite3.Error, ValueError) as exc:
            snapshot["journal_written"] = False
            snapshot["status"] = "partial"
            snapshot["errors"]["journal"] = type(exc).__name__
        with self.lock:
            self.snapshot = snapshot
        return self.get_snapshot()

    def get_snapshot(self):
        with self.lock:
            result = deepcopy(self.snapshot)
        observed = result.get("observed_ms")
        age = (self.clock() * 1000 - observed) / 1000 if observed is not None else None
        result["age_seconds"] = age
        result["is_stale"] = age is None or age < 0 or age > self.interval + 120
        if observed is not None and result["is_stale"]:
            result["status"] = "stale"
        return result

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return

        def run():
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    self.cycle()
                except Exception as exc:
                    with self.lock:
                        self.snapshot = {
                            **self.snapshot,
                            "status": "unavailable",
                            "observed_ms": None,
                            "btc_positions": [],
                            "cohorts": {},
                            "successful_accounts": 0,
                            "errors": {"cycle": type(exc).__name__},
                        }
                self.stop_event.wait(
                    max(1, self.interval - (time.monotonic() - started))
                )

        self.thread = threading.Thread(
            target=run, name="public-wallet-monitor", daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
