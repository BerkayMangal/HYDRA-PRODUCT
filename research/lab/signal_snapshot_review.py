"""Recompute a captured /api/scalp response offline; never place orders.

Usage: python -m research.lab.signal_snapshot_review path/to/scalp.json
This is a snapshot audit, not a portfolio backtest or an independent model test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def review(snapshot: dict) -> dict:
    rows = snapshot["recent"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recent signal IDs")
    closed = [r for r in rows if r.get("ret_pct") is not None and r["status"] != "open"]
    groups = {
        "recent_closed": closed,
        "recent_sent_closed": [r for r in closed if r.get("sent")],
        "recent_hgb": [r for r in closed if r["setup"] == "ml15-hgb-c1x2-long"],
        "recent_sent_hgb": [r for r in closed if r.get("sent") and r["setup"] == "ml15-hgb-c1x2-long"],
    }
    result = {}
    for name, selected in groups.items():
        returns = [r["ret_pct"] for r in selected]
        average = mean(returns) if returns else None
        result[name] = {
            "n": len(returns),
            "wins": sum(r > 0 for r in returns),
            "win_rate": sum(r > 0 for r in returns) / len(returns) if returns else None,
            "mean_reported_pct": average,
            "arithmetic_sum_percentage_points": sum(returns),
            "hypothetical_mean_after_10bp_pct": average - 0.1 if average is not None else None,
            "hypothetical_mean_after_14bp_pct": average - 0.14 if average is not None else None,
            "not_portfolio_return": True,
        }
    differences = []
    for r in closed:
        if r["side"] not in ("long", "short") or r["entry"] <= 0:
            raise ValueError("Invalid side or entry")
        gross = (r["exit_price"] / r["entry"] - 1) * 100
        if r["side"] == "short":
            gross = -gross
        differences.append(abs(r["ret_pct"] - gross))
    result.update({
        "max_gross_price_return_rounding_difference_pct": max(differences) if differences else None,
        "record": snapshot["record"],
        "study": snapshot["study"],
        "api_only_no_source_or_full_ledger": True,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    result = review(json.loads(raw))
    result["snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
