"""Assumed cost hurdles on recorded movements; never executable portfolio PnL."""

import math


def cost_scenario(report, fee_per_side_bps=8.0, slippage_per_side_bps=4.0,
                  funding_allowance_bps=0.0):
    inputs = (fee_per_side_bps, slippage_per_side_bps, funding_allowance_bps)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 100 for value in inputs):
        raise ValueError("Each assumed cost must be finite and between 0 and 100 bp")
    total = 2 * (fee_per_side_bps + slippage_per_side_bps) + funding_allowance_bps
    result = {
        "status": "unavailable",
        "orders_enabled": False,
        "alpha_validated": False,
        "report_generated_at": report.get("generated_at"),
        "assumptions": {
            "fee_per_side_bps": fee_per_side_bps,
            "slippage_per_side_bps": slippage_per_side_bps,
            "funding_allowance_bps": funding_allowance_bps,
            "round_trip_hurdle_bps": total,
        },
        "rows": [],
        "limits": "Recorded primary-cohort price movements, not executed returns or a portfolio. "
                  "Overlapping horizons and a small concentrated sample do not establish alpha. "
                  "Fees and slippage are assumed per side; funding is an assumed allowance for each "
                  "horizon, not measured accrued funding. Positive differences do not promote a strategy.",
    }
    if report.get("status") != "research_only_alpha_unproven" or report.get("orders_enabled") is not False:
        return result
    result["cohort_accounts"] = report.get("hyperliquid", {}).get("requested_wallets")
    diagnostics = report.get("price_movement_diagnostics", {})
    for name in ("inventory", "liquidation_reversal", "spot_absorption"):
        for horizon in (15, 60, 240):
            row = diagnostics.get(name, {}).get(str(horizon))
            if not row:
                continue
            count = row.get("observed_groups")
            nonoverlap = row.get("nonoverlapping_intervals")
            move = row.get("mean_signed_gross_price_move")
            if (type(count) is not int or type(nonoverlap) is not int
                    or not 0 <= nonoverlap <= count):
                raise ValueError("Valid recorded observation counts required")
            if move is not None and (isinstance(move, bool)
                    or not isinstance(move, (int, float)) or not math.isfinite(move)):
                raise ValueError("Finite recorded movement or unknown required")
            if count == 0:
                move = None
            gross = move * 10000 if move is not None else None
            result["rows"].append({
                "hypothesis": name,
                "horizon_minutes": horizon,
                "observed_groups": count,
                "nonoverlapping_intervals": nonoverlap,
                "mean_signed_gross_move_bps": gross,
                "mean_move_less_assumed_cost_bps": gross - total if gross is not None else None,
                "status": "descriptive_only" if move is not None else "unknown",
            })
    if result["rows"]:
        result["status"] = "descriptive_only"
    return result
