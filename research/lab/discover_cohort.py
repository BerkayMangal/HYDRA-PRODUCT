"""Freeze an additional activity/size sample before collecting future outcomes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

from collectors.hyperliquid_research import PublicHyperliquid
from research.hyperliquid_features import market_contexts, number
from .paths import ROOT, write_json

SPECIFICATION = {
    "version": "additional_activity_size_cohort_v1",
    "coins": [
        "BTC",
        "ETH",
        "SOL",
        "HYPE",
        "DOGE",
        "XRP",
        "AVAX",
        "LINK",
        "SUI",
        "AAVE",
    ],
    "candidate_order": "Lexicographic public recentTrades participants; exclude original frozen accounts",
    "maximum_states_inspected": 96,
    "btc_notional_strata_usd": [[10000, 100000], [100000, 1000000], [1000000, None]],
    "maximum_selected_per_stratum": 6,
    "selection_within_stratum": "Lexicographic address; no historical/future PnL or performance labels",
    "maximum_concurrent_reads": 3,
    "scope": "Additional active, size-stratified default-DEX BTC accounts; not representative of all traders",
    "orders_enabled": False,
}
SPECIFICATION_SHA256 = hashlib.sha256(
    json.dumps(SPECIFICATION, sort_keys=True).encode()
).hexdigest()


def stratum(notional):
    for index, (minimum, maximum) in enumerate(
        SPECIFICATION["btc_notional_strata_usd"]
    ):
        if notional >= minimum and (maximum is None or notional < maximum):
            return index
    return None


def select_accounts(observations, mark, excluded):
    """Selection depends on current signed size and a frozen mark, never PnL."""
    buckets = {index: [] for index in range(3)}
    for record in sorted(observations, key=lambda item: item["wallet"]):
        wallet, state = record["wallet"], record.get("state")
        if wallet in excluded or not isinstance(state, dict):
            continue
        event = state.get("time")
        if event is None or not 0 <= record["observed_ms"] - event <= 120000:
            continue
        btc_size = sum(
            number(item["position"]["szi"])
            for item in state["assetPositions"]
            if item["position"]["coin"] == "BTC"
        )
        notional = abs(btc_size) * mark
        bucket = stratum(notional)
        if (
            bucket is not None
            and len(buckets[bucket]) < SPECIFICATION["maximum_selected_per_stratum"]
        ):
            buckets[bucket].append(
                {
                    "wallet": wallet,
                    "btc_notional_at_selection": notional,
                    "signed_btc_at_selection": btc_size,
                }
            )
    selected = [row for bucket in buckets.values() for row in bucket]
    return sorted({row["wallet"] for row in selected}), buckets


def discover(output, original):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive declaration stops accidental re-selection after seeing results.
    declared = {
        "declared_at_ms": int(time.time() * 1000),
        "specification": SPECIFICATION,
        "specification_sha256": SPECIFICATION_SHA256,
    }
    with (output / "discovery-spec.json").open("x") as stream:
        json.dump(declared, stream, indent=2)
    original = json.loads(Path(original).read_text())
    excluded = set(original["wallets"])
    client = PublicHyperliquid()
    body = client.query("metaAndAssetCtxs")
    mark_observed = int(time.time() * 1000)
    contexts = market_contexts(body)
    write_json(
        output / "discovery-market.json", {"observed_ms": mark_observed, "body": body}
    )
    candidates, raw_trades, errors = set(), {}, {}
    for coin in SPECIFICATION["coins"]:
        if coin not in contexts:
            errors["trades:" + coin] = "Asset absent from default-DEX metadata"
            continue
        try:
            rows = client.query("recentTrades", coin=coin)
            if not isinstance(rows, list):
                raise ValueError("Expected public recentTrades list")
            observed = int(time.time() * 1000)
            raw_trades[coin] = {"observed_ms": observed, "body": rows}
            for row in rows:
                for wallet in row.get("users", []):
                    if isinstance(wallet, str) and re.fullmatch(
                        r"0x[0-9a-fA-F]{40}", wallet
                    ):
                        candidates.add(wallet.lower())
        except Exception as exc:
            errors["trades:" + coin] = type(exc).__name__
    write_json(output / "discovery-trades.json", raw_trades)
    if "BTC" not in raw_trades:
        raise RuntimeError("BTC participant discovery failed; no cohort frozen")
    candidates = sorted(candidates - excluded)[
        : SPECIFICATION["maximum_states_inspected"]
    ]

    def read(wallet):
        try:
            state = PublicHyperliquid().query("clearinghouseState", user=wallet)
            return {
                "wallet": wallet,
                "observed_ms": int(time.time() * 1000),
                "state": state,
            }
        except Exception as exc:
            return {
                "wallet": wallet,
                "observed_ms": int(time.time() * 1000),
                "state": None,
                "error": type(exc).__name__,
            }

    states = []
    with ThreadPoolExecutor(
        max_workers=SPECIFICATION["maximum_concurrent_reads"]
    ) as pool:
        futures = [pool.submit(read, wallet) for wallet in candidates]
        for future in as_completed(futures):
            states.append(future.result())
            if len(states) % 24 == 0:
                print(
                    json.dumps(
                        {"states_inspected": len(states), "planned": len(candidates)}
                    ),
                    flush=True,
                )
    write_json(
        output / "discovery-states.json", sorted(states, key=lambda row: row["wallet"])
    )
    # A bounded discovery's selection reference must not silently age indefinitely.
    if time.time() * 1000 - mark_observed > 300000:
        raise RuntimeError(
            "Discovery took over five minutes; preserve attempt, no cohort frozen"
        )
    wallets, buckets = select_accounts(states, contexts["BTC"]["mark_price"], excluded)
    if not wallets:
        raise RuntimeError(
            "No eligible additional BTC accounts; preserve unsuccessful discovery"
        )
    cohort = {
        "specification": SPECIFICATION,
        "specification_sha256": SPECIFICATION_SHA256,
        "frozen_at_ms": int(time.time() * 1000),
        "wallets": wallets,
        "sample_size": len(wallets),
        "inspected": len(states),
        "selection_strata": buckets,
        "original_wallets_excluded": sorted(excluded),
        "discovery_errors": errors,
        "failed_state_reads": sum(row["state"] is None for row in states),
        "selection_mark_price": contexts["BTC"]["mark_price"],
        "selection_mark_observed_ms": mark_observed,
        "scope": SPECIFICATION["scope"],
        "orders_enabled": False,
    }
    write_json(output / "large-wallet-cohort.json", cohort)
    print(
        json.dumps(
            {
                "frozen_wallets": len(wallets),
                "inspected": len(states),
                "errors": errors,
                "output": str(output),
            }
        ),
        flush=True,
    )
    return cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--exclude-cohort",
        type=Path,
        default=ROOT / "hyperliquid/large-wallet-cohort.json",
    )
    args = parser.parse_args()
    discover(args.out, args.exclude_cohort)


if __name__ == "__main__":
    main()
