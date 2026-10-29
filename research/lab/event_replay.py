"""Receipt-time replay on prospectively collected data, not a frozen live strategy.

This analysis was authored after the journals were collected. Price movements
are descriptive, with overlapping horizons and sampled liquidation coverage.
"""

from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
import math
import statistics

from research.spot_confirmation import spot_window

SPECIFICATION = {
    "version": "receipt_replay_v1",
    "design": "Retrospective analysis of prospectively journaled public data; not prospective strategy validation",
    "liquidation_groups": "BTC fills grouped by first-receipt UTC minute and liquidated side; decide at the next minute boundary",
    "spot_confirmation": "5m and 15m as-of spot imbalance both oppose the forced trade; unknown coverage abstains",
    "inventory_sampling": "Earliest usable recorded BTC base-inventory change in each UTC quarter hour; unchanged frozen cohort",
    "price_reference": "First recorded OKX spot ticker with event and receipt at/after decision, within 120 seconds; no historical price fetch",
    "forward_horizons_minutes": [15, 60, 240],
    "illustrative_round_trip_cost": 0.0024,
    "statistics": "Report all available movements and greedy non-overlapping interval counts; no significance or promotion claim",
    "orders_enabled": False,
}
SPECIFICATION_SHA256 = hashlib.sha256(
    json.dumps(SPECIFICATION, sort_keys=True).encode()
).hexdigest()


class PriceTape:
    def __init__(self, records):
        self.rows = []
        for observed, event, price in sorted(records):
            if 0 <= observed - event <= 60000 and math.isfinite(price) and price > 0:
                self.rows.append((observed, event, price))
        self.times = [row[0] for row in self.rows]

    @classmethod
    def from_db(cls, db):
        records = []
        for observed, payload in db.execute(
            "SELECT observed_ms,payload FROM snapshots WHERE kind='ticker' AND instrument='BTC-USDT' ORDER BY observed_ms"
        ):
            row = json.loads(payload)["data"][0]
            records.append((observed, int(row["ts"]), float(row["last"])))
        return cls(records)

    def at_or_after(self, timestamp, max_delay_ms=120000):
        start = bisect_left(self.times, timestamp)
        for observed, event, price in self.rows[start:]:
            if observed > timestamp + max_delay_ms:
                break
            if event >= timestamp:
                return {"observed_ms": observed, "event_ms": event, "price": price}
        return None


def outcomes(tape, decision_ms, direction):
    entry = tape.at_or_after(decision_ms)
    result = {"entry_reference": entry, "horizons": {}}
    for horizon in SPECIFICATION["forward_horizons_minutes"]:
        exit_quote = (
            tape.at_or_after(entry["observed_ms"] + horizon * 60000) if entry else None
        )
        row = {
            "status": "observed" if exit_quote else "missing_reference",
            "exit_reference": exit_quote,
        }
        if exit_quote:
            move = direction * (exit_quote["price"] / entry["price"] - 1)
            row["signed_gross_price_move"] = move
            row["move_less_illustrative_cost_hurdle"] = (
                move - SPECIFICATION["illustrative_round_trip_cost"]
            )
        result["horizons"][str(horizon)] = row
    return result


def summarize(rows):
    result = {}
    for horizon in SPECIFICATION["forward_horizons_minutes"]:
        observed = [
            row for row in rows if row["horizons"][str(horizon)]["status"] == "observed"
        ]
        moves = [
            row["horizons"][str(horizon)]["signed_gross_price_move"] for row in observed
        ]
        boundary, nonoverlap = -1, 0
        for row in sorted(
            observed, key=lambda item: item["entry_reference"]["observed_ms"]
        ):
            if row["entry_reference"]["observed_ms"] >= boundary:
                nonoverlap += 1
                boundary = row["horizons"][str(horizon)]["exit_reference"][
                    "observed_ms"
                ]
        result[str(horizon)] = {
            "observed_groups": len(moves),
            "missing_groups": len(rows) - len(moves),
            "nonoverlapping_intervals": nonoverlap,
            "mean_signed_gross_price_move": statistics.mean(moves) if moves else None,
            "median_signed_gross_price_move": (
                statistics.median(moves) if moves else None
            ),
            "positive_move_fraction": (
                sum(move > 0 for move in moves) / len(moves) if moves else None
            ),
            "mean_move_less_illustrative_cost_hurdle": (
                statistics.mean(moves) - SPECIFICATION["illustrative_round_trip_cost"]
                if moves
                else None
            ),
        }
    return result


def liquidation_replay(hl_db, okx_db, tape, frozen_at_ms):
    groups = defaultdict(list)
    rejected = 0
    for event_ms, observed, payload in hl_db.execute(
        "SELECT event_ms,first_observed_ms,payload FROM liquidations WHERE coin='BTC' ORDER BY first_observed_ms"
    ):
        event = json.loads(payload)
        if event_ms < frozen_at_ms or event_ms > observed:
            rejected += 1
            continue
        decision = (observed // 60000 + 1) * 60000
        groups[(decision, event["liquidated_position_side"])].append((event, observed))
    rows = []
    for (decision, side), fills in sorted(groups.items()):
        direction = 1 if side == "long" else -1
        windows = {
            str(minutes): spot_window(okx_db, decision, minutes) for minutes in (5, 15)
        }
        usable = all(window["usable"] for window in windows.values())
        absorption = (
            all(direction * window["imbalance"] > 0 for window in windows.values())
            if usable
            else None
        )
        rows.append(
            {
                "decision_ms": decision,
                "liquidated_position_side": side,
                "fills_known_at_decision": len(fills),
                "victims_observed": len({event["victim"] for event, _ in fills}),
                "sampled_notional_usd": sum(
                    event["fill_notional_usd"] for event, _ in fills
                ),
                "median_receipt_delay_seconds": statistics.median(
                    (observed - event["event_ms"]) / 1000 for event, observed in fills
                ),
                "spot_flow": windows,
                "spot_absorption": absorption,
                **outcomes(tape, decision, direction),
            }
        )
    return {
        "receipt_minute_side_groups": len(rows),
        "rejected_precohort_or_future_events": rejected,
        "spot_absorption_groups": sum(row["spot_absorption"] is True for row in rows),
        "unknown_spot_coverage_groups": sum(
            row["spot_absorption"] is None for row in rows
        ),
        "all_reversals": summarize(rows),
        "spot_absorption_reversals": summarize(
            [row for row in rows if row["spot_absorption"] is True]
        ),
        "rows": rows,
    }


def inventory_replay(features, tape):
    # Earliest usable receipt in each quarter-hour; never select on its outcome.
    chosen = {}
    for observed, feature in features:
        delta = feature.get("sampled_inventory_change")
        if delta is not None and "BTC" in delta:
            chosen.setdefault(observed // 900000, (observed, feature))
    rows, flat = [], 0
    for observed, feature in chosen.values():
        delta = feature["sampled_inventory_change"]["BTC"]
        if not delta:
            flat += 1
            continue
        rows.append(
            {
                "decision_ms": observed,
                "sampled_btc_base_change": delta,
                "actual_window_minutes": feature[
                    "inventory_change_actual_window_minutes"
                ],
                "largest_account_gross_share": feature["sampled_assets"]
                .get("BTC", {})
                .get("largest_account_gross_share"),
                **outcomes(tape, observed, 1 if delta > 0 else -1),
            }
        )
    return {
        "quarter_hour_observations": len(chosen),
        "unchanged_inventory_observations": flat,
        "all_inventory_directions": summarize(rows),
        "rows": rows,
    }
