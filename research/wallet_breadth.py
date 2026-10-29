"""Account breadth and single-account sensitivity, without PnL selection."""

import math


def btc_wallet_change(previous, current):
    """Require the same fully observed cohort; a successfully flat wallet is zero."""
    if previous.get("unavailable_wallets") or current.get("unavailable_wallets"):
        return None
    wallets = previous.get("wallets")
    if (
        not wallets
        or wallets != current.get("wallets")
        or len(set(wallets)) != len(wallets)
    ):
        return None
    sizes = []
    current_notionals = {wallet: 0.0 for wallet in wallets}
    for feature in (previous, current):
        inventory = {wallet: 0.0 for wallet in wallets}
        for row in feature["positions"]:
            if row["coin"] != "BTC":
                continue
            wallet = row["wallet"]
            size = float(row["signed_base_size"])
            if wallet not in inventory or not math.isfinite(size):
                raise ValueError("Valid positions from the frozen cohort required")
            inventory[wallet] += size
            if feature is current:
                notional = float(row["notional_usd"])
                if not math.isfinite(notional) or notional < 0:
                    raise ValueError("Finite, nonnegative current notional required")
                current_notionals[wallet] += notional
        sizes.append(inventory)
    changes = {wallet: sizes[1][wallet] - sizes[0][wallet] for wallet in wallets}
    epsilon = 1e-9
    buyers = sum(value > epsilon for value in changes.values())
    sellers = sum(value < -epsilon for value in changes.values())
    total = sum(changes.values())
    absolute = sum(abs(value) for value in changes.values())
    leave_one_out = [total - value for value in changes.values()]
    largest = max(current_notionals, key=current_notionals.get)
    return {
        "per_wallet_base_change": changes,
        "sampled_net_base_change": total,
        "accounts_buying": buyers,
        "accounts_selling": sellers,
        "accounts_unchanged": len(wallets) - buyers - sellers,
        "equal_account_direction_balance": (buyers - sellers) / len(wallets),
        "largest_absolute_change_share": (
            max(abs(value) for value in changes.values()) / absolute
            if absolute
            else None
        ),
        "leave_one_out_net_min": min(leave_one_out),
        "leave_one_out_net_max": max(leave_one_out),
        "aggregate_direction_survives_every_single_account_removal": (
            min(leave_one_out) > epsilon
            if total > epsilon
            else max(leave_one_out) < -epsilon if total < -epsilon else False
        ),
        "net_without_largest_current_btc_account": total - changes[largest],
        "scope": "Account-count diagnostic; common ownership and external hedges are unknown",
        "orders_enabled": False,
    }
