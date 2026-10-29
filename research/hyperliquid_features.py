"""Public Hyperliquid research features with explicit units and sample scope."""

import hashlib
import json
import math

SPECIFICATION = {
    "version": "hyperliquid_research_v1",
    "scope": "Frozen, currently active public-wallet sample on the default perp DEX; not all traders",
    "hypotheses": [
        "Near-liquidation sampled inventory plus spot absorption may identify reversals after realized forced flow",
        "Change in sampled signed base inventory may add information beyond price trend and funding",
        "Realized liquidation continuation versus reversal depends on spot flow; estimated liquidation levels alone are not signals",
    ],
    "inventory_distance_bands": [0.005, 0.01, 0.02],
    "wallet_change_window_minutes": 15,
    "forward_horizons_minutes": [15, 60, 240],
    "orders_enabled": False,
    "rules": "Store event/receipt times; do not reconstruct past risk zones from current positions; distinguish victims from counterparties; do not rank wallets using future PnL",
}
SPECIFICATION_SHA256 = hashlib.sha256(
    json.dumps(SPECIFICATION, sort_keys=True).encode()
).hexdigest()


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Finite market value required")
    return result


def market_contexts(response):
    if not isinstance(response, list) or len(response) != 2:
        raise ValueError("Expected [metadata, asset contexts]")
    meta, contexts = response
    universe = meta["universe"]
    if len(universe) != len(contexts):
        raise ValueError("Metadata/context lengths differ")
    result = {}
    for instrument, ctx in zip(universe, contexts):
        mark, oi = number(ctx["markPx"]), number(ctx["openInterest"])
        funding, volume = number(ctx["funding"]), number(ctx["dayNtlVlm"])
        if mark <= 0 or oi < 0 or volume < 0:
            raise ValueError("Invalid price, open interest or volume")
        result[instrument["name"]] = {
            "coin": instrument["name"],
            "mark_price": mark,
            "open_interest_base": oi,
            "open_interest_usd": oi * mark,
            "volume_24h_usd": volume,
            "funding_hourly": funding,
            "funding_simple_apr_pct": funding * 24 * 365 * 100,
        }
    return result


def sampled_positions(states, contexts, as_of_ms, max_age_ms=120000):
    """Missing wallets remain unknown; a successful empty account is flat."""
    rows, failed = [], []
    for sample in states:
        wallet, observed = sample["wallet"], sample["observed_ms"]
        state = sample.get("state")
        if not isinstance(state, dict) or "assetPositions" not in state:
            failed.append(wallet)
            continue
        event_time = state.get("time")
        if (
            observed > as_of_ms
            or event_time is None
            or not -5000 <= as_of_ms - event_time <= max_age_ms
        ):
            failed.append(wallet)
            continue
        for item in state["assetPositions"]:
            position = item["position"]
            coin = position["coin"]
            if coin not in contexts:
                continue
            size = number(position["szi"])
            if not size:
                continue
            mark = contexts[coin]["mark_price"]
            liquidation = position.get("liquidationPx")
            liquidation = number(liquidation) if liquidation is not None else None
            side = "long" if size > 0 else "short"
            valid_level = (
                liquidation is not None
                and liquidation > 0
                and (liquidation < mark if size > 0 else liquidation > mark)
            )
            rows.append(
                {
                    "wallet": wallet,
                    "coin": coin,
                    "side": side,
                    "signed_base_size": size,
                    "notional_usd": abs(size) * mark,
                    "mark_price": mark,
                    "entry_price": (
                        number(position["entryPx"])
                        if position.get("entryPx") is not None
                        else None
                    ),
                    "estimated_liquidation_price": liquidation,
                    "liquidation_distance_fraction": (
                        abs(liquidation - mark) / mark if valid_level else None
                    ),
                    "liquidation_level_status": (
                        "estimated" if valid_level else "missing_or_inconsistent"
                    ),
                    "margin_type": position.get("leverage", {}).get("type"),
                    "event_ms": event_time,
                    "observed_ms": observed,
                }
            )
    summaries = {}
    for coin in contexts:
        selected = [row for row in rows if row["coin"] == coin]
        if not selected:
            continue
        summary = {
            "sampled_gross_notional_usd": sum(row["notional_usd"] for row in selected),
            "sampled_net_base": sum(row["signed_base_size"] for row in selected),
            "sampled_accounts_with_position": len(selected),
            "missing_liquidation_levels": sum(
                row["liquidation_distance_fraction"] is None for row in selected
            ),
        }
        notionals = [row["notional_usd"] for row in selected]
        gross = sum(notionals)
        summary["largest_account_gross_share"] = (
            max(notionals) / gross if gross else None
        )
        summary["effective_account_count_by_gross"] = (
            gross**2 / sum(value**2 for value in notionals) if gross else 0.0
        )
        for band in SPECIFICATION["inventory_distance_bands"]:
            for side in ("long", "short"):
                key = f"sampled_{side}_inventory_within_{band:.3f}"
                summary[key] = sum(
                    row["notional_usd"]
                    for row in selected
                    if row["side"] == side
                    and row["liquidation_distance_fraction"] is not None
                    and row["liquidation_distance_fraction"] <= band
                )
        summaries[coin] = summary
    return {
        "positions": rows,
        "sampled_assets": summaries,
        "unavailable_wallets": failed,
        "wallets": sorted(sample["wallet"] for sample in states),
        "requested_wallets": len(states),
        "successful_wallets": len(states) - len(failed),
        "scope": SPECIFICATION["scope"],
        "orders_enabled": False,
    }


def signed_inventory_change(previous, current):
    """Use base-size changes, not price-driven changes in dollar position value."""
    if previous["unavailable_wallets"] or current["unavailable_wallets"]:
        return None
    previous_wallets = previous.get("wallets")
    current_wallets = current.get("wallets")
    if previous_wallets != current_wallets or previous_wallets is None:
        return None
    assets = set(previous["sampled_assets"]) | set(current["sampled_assets"])
    return {
        coin: current["sampled_assets"].get(coin, {}).get("sampled_net_base", 0.0)
        - previous["sampled_assets"].get(coin, {}).get("sampled_net_base", 0.0)
        for coin in assets
    }


def realized_liquidation_fills(wallet_fills):
    """Deduplicate counterparties/partial reports by asset+trade ID, not by wallet."""
    events = {}
    for wallet, fills, observed_ms in wallet_fills:
        for fill in fills:
            liquidation = fill.get("liquidation")
            if not liquidation:
                continue
            victim = liquidation.get("liquidatedUser")
            if not victim or fill.get("side") not in ("A", "B"):
                continue
            own_victim = wallet.lower() == victim.lower()
            victim_order_side = (
                fill["side"] if own_victim else ("A" if fill["side"] == "B" else "B")
            )
            key = (fill["coin"], fill["tid"])
            event = {
                "coin": fill["coin"],
                "trade_id": fill["tid"],
                "event_ms": int(fill["time"]),
                "observed_ms": observed_ms,
                "victim": victim.lower(),
                "liquidated_position_side": (
                    "long" if victim_order_side == "A" else "short"
                ),
                "fill_price": number(fill["px"]),
                "base_size": number(fill["sz"]),
                "fill_notional_usd": number(fill["px"]) * number(fill["sz"]),
                "method": liquidation.get("method"),
                "evidence": "explicit_fill_liquidation_object",
            }
            if event["fill_price"] <= 0 or event["base_size"] <= 0:
                raise ValueError("Positive liquidation price and size required")
            if event["event_ms"] > observed_ms:
                raise ValueError("Future liquidation event")
            if key in events:
                earlier = events[key]
                if (earlier["victim"], earlier["liquidated_position_side"]) != (
                    event["victim"],
                    event["liquidated_position_side"],
                ):
                    raise ValueError("Conflicting victim information")
                event["observed_ms"] = min(earlier["observed_ms"], observed_ms)
            events[key] = event
    return sorted(
        events.values(), key=lambda row: (row["event_ms"], row["coin"], row["trade_id"])
    )
