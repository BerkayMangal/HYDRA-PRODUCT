"""Predeclared staged search. No candidate selection on 2025/2026 results."""

import csv
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .engine import (
    ROOT,
    Features,
    grid,
    load,
    net_returns,
    metrics,
    excess_series,
    bootstrap,
    holm,
)

RESULTS = ROOT / "results"


def write(name, data):
    (RESULTS / name).write_text(json.dumps(data, indent=2, allow_nan=False))


def compact(net, p):
    daily = np.expm1(np.log1p(net).reshape(-1, 24).sum(axis=1))
    sd = daily.std(ddof=1)
    eq = np.exp(np.cumsum(np.log1p(net)))
    dd = float((eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1).min())
    return {
        "net_return": float(eq[-1] - 1),
        "sharpe": float(daily.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
        "max_dd": dd,
        "exposure": float(p.mean()),
        "entries": int(((p > 0) & (np.r_[0.0, p[:-1]] == 0)).sum()),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    data = load()
    quality = json.loads((ROOT / "data/quality.json").read_text())
    if not all(quality["checks"].values()):
        raise ValueError("Historical data failed quality gate")
    actual = hashlib.sha256((ROOT / "data/btc-usdt-1h.csv").read_bytes()).hexdigest()
    if actual != quality["sha256"]:
        raise ValueError("Dataset hash mismatch")
    specs = grid()
    stamp = datetime.now(timezone.utc).isoformat()
    manifest = {
        "created_at": stamp,
        "candidate_count": len(specs),
        "dataset_sha256": actual,
        "dataset": quality,
        "splits": {
            "warmup": "2020-01-01 to 2020-05-31",
            "discovery": "2020-06-01 to 2023-12-31",
            "validation": "2024-01-01 to 2024-12-31",
            "historical_test": "2025-01-01 to 2025-12-31",
            "recent_diagnostic": "2026-01-01 to dataset end",
        },
        "selection": "Discovery: maxDD>-50%, >=20 entries, at least 2 positive yearly Sharpes; rank median yearly Sharpe - 0.5*std(yearly Sharpe) + aggregate maxDD. Freeze top 3 distinct discovery-position paths per family. Validation: choose one per family by Sharpe subject to >0 net and maxDD>-35%; if none, no selection. Never reselect after historical test.",
        "execution": "BTC/USDT spot, long/flat exposure [0,1], no leverage, no cash yield. Completed 1H/4H signals enter at the next bar open. Hourly mark-to-market. Fractional vol-target rebalances pay turnover fees. Each period starts with cash, pays first exposure and liquidates terminal exposure.",
        "cost_per_side": 0.0012,
        "stress_cost_per_side": 0.0022,
        "extra_latency_stress": "one additional hourly bar",
        "promotion": "At least 20 entries in historical test AND recent diagnostic, positive net in both base/stress/delayed, maxDD>-35%, positive mean daily excess and positive lower block-bootstrap 95% bound in both, Holm p<0.05 across selected families in historical test. Statistical search-Bonferroni diagnostic also reported. Only research shadow deployment is supported; no orders.",
        "caveats": "Previous audit saw 2025-10 onward and tried 3 simple rules. 2025/2026 are chronological historical diagnostics, not untouched prospective proof. Historical OHLCV-only rules do not test spot CVD/OI/funding hypotheses. Selection criteria are design choices, not guarantees.",
        "candidates": specs,
    }
    manifest_path = RESULTS / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["dataset_sha256"] != actual or old["candidates"] != specs:
            raise RuntimeError(
                "Existing run has different dataset/specs; preserve it and start a new run"
            )
    else:
        write("manifest.json", manifest)
    dates = data.index[:-1]
    returns = (data.open.shift(-1) / data.open - 1).iloc[:-1].to_numpy()
    features = Features(data)
    dev = (dates >= pd.Timestamp("2020-06-01", tz="UTC")) & (
        dates < pd.Timestamp("2024-01-01", tz="UTC")
    )
    dev_dates = dates[dev]
    dev_year = dev_dates.year
    training = []
    seen = set()
    top = {}
    started = time.time()
    resume = RESULTS / "discovery.json"
    if resume.exists():
        training = json.loads(resume.read_text())
        print("Reusing completed discovery", len(training), flush=True)
    else:
        for i, s in enumerate(specs):
            p = features.position(s)[:-1][dev]
            identity = hashlib.sha256(p.astype("float32").tobytes()).hexdigest()
            # Do not pretend parameter duplicates are independent observed paths.
            if identity in seen:
                continue
            seen.add(identity)
            net = net_returns(p, returns[dev])
            summary = compact(net, p)
            annual = []
            for year in sorted(set(dev_year)):
                m = dev_year == year
                annual.append(compact(net_returns(p[m], returns[dev][m]), p[m]))
            sh = np.array([x["sharpe"] for x in annual])
            score = float(np.median(sh) - 0.5 * np.std(sh) + summary["max_dd"])
            eligible = (
                summary["max_dd"] > -0.5
                and summary["entries"] >= 20
                and int((sh > 0).sum()) >= 2
            )
            training.append(
                {
                    "spec": s,
                    "path_hash": identity,
                    "discovery": summary,
                    "annual": annual,
                    "selection_score": score,
                    "eligible": eligible,
                }
            )
            if i % 500 == 0:
                print(
                    "discovery",
                    i,
                    "/",
                    len(specs),
                    "distinct",
                    len(training),
                    "seconds",
                    round(time.time() - started, 1),
                    flush=True,
                )
        write("discovery.json", training)
    for row in training:
        if row["eligible"]:
            top.setdefault(row["spec"]["family"], []).append(row)
    shortlist = []
    for fam, rows in top.items():
        shortlist.extend(
            sorted(rows, key=lambda r: (-r["selection_score"], r["spec"]["id"]))[:3]
        )
    write("frozen-discovery-shortlist.json", shortlist)
    print(
        "Frozen discovery shortlist",
        len(shortlist),
        "distinct discovery paths",
        len(training),
        flush=True,
    )

    def evaluate(spec, start, end, cost=0.0012, lag=0):
        mask = (dates >= pd.Timestamp(start, tz="UTC")) & (
            dates < pd.Timestamp(end, tz="UTC")
        )
        full = features.position(spec)[:-1]
        if lag:
            full = np.r_[np.zeros(lag), full[:-lag]]
        p = full[mask]
        r = returns[mask]
        d = dates[mask]
        net = net_returns(p, r, cost)
        return metrics(net, p, d), net, p, r, d

    validation = []
    for row in shortlist:
        s = row["spec"]
        m, *_ = evaluate(s, "2024-01-01", "2025-01-01")
        validation.append({"spec": s, "discovery": row["discovery"], "validation": m})
    write("validation.json", validation)
    selected = []
    for fam in sorted(top):
        eligible = [
            x
            for x in validation
            if x["spec"]["family"] == fam
            and x["validation"]["net_return"] > 0
            and x["validation"]["max_dd"] > -0.35
        ]
        if eligible:
            selected.append(
                sorted(
                    eligible,
                    key=lambda x: (-x["validation"]["sharpe"], x["spec"]["id"]),
                )[0]
            )
    write("frozen-selected.json", selected)
    print("Frozen validation selections", len(selected), flush=True)

    evaluations = []
    equity_rows = []
    for row in selected:
        s = row["spec"]
        entry = {
            "spec": s,
            "discovery": row["discovery"],
            "validation": row["validation"],
        }
        for label, start, end in [
            ("test_2025", "2025-01-01", "2026-01-01"),
            ("diagnostic_2026", "2026-01-01", "2027-01-01"),
        ]:
            m, net, p, r, d = evaluate(s, start, end)
            m["excess_statistics"] = bootstrap(excess_series(net, p, r, d), len(specs))
            m["cost_stress"] = evaluate(s, start, end, 0.0022)[0]
            m["extra_hour_latency"] = evaluate(s, start, end, 0.0012, 1)[0]
            m["buy_hold"] = metrics(net_returns(np.ones(len(r)), r), np.ones(len(r)), d)
            m["quarterly_returns"] = {
                str(k): float(np.expm1(np.log1p(v).sum()))
                for k, v in pd.Series(net, index=d).groupby(pd.Grouper(freq="QS"))
            }
            entry[label] = m
            equity_rows.append(
                pd.DataFrame(
                    {
                        "timestamp": d,
                        "id": s["id"],
                        "period": label,
                        "position": p,
                        "net_return": net,
                    }
                )
            )
        evaluations.append(entry)
    ps = [
        x["test_2025"]["excess_statistics"]["p_one_sided_block_bootstrap"]
        for x in evaluations
    ]
    corrected = holm(ps)
    for row, padj in zip(evaluations, corrected):
        row["test_2025"]["excess_statistics"]["p_holm_selected_families"] = padj
        reasons = []
        for label in ["test_2025", "diagnostic_2026"]:
            m = row[label]
            if m["entries"] < 20:
                reasons.append(label + ": fewer than 20 entries")
            if m["net_return"] <= 0:
                reasons.append(label + ": nonpositive net return")
            if m["max_dd"] <= -0.35:
                reasons.append(label + ": drawdown >=35%")
            if m["cost_stress"]["net_return"] <= 0:
                reasons.append(label + ": fails cost stress")
            if m["extra_hour_latency"]["net_return"] <= 0:
                reasons.append(label + ": fails delay stress")
            if m["excess_statistics"]["ci95"][0] <= 0:
                reasons.append(label + ": positive excess not established")
        if padj >= 0.05:
            reasons.append("historical test: fails Holm multiple-family significance")
        row["promotion_eligible"] = not reasons
        row["rejection_reasons"] = reasons
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "declared_variants": len(specs),
        "distinct_discovery_paths": len(training),
        "shortlisted_before_validation": len(shortlist),
        "selected_before_test": len(selected),
        "promotion_eligible_count": sum(x["promotion_eligible"] for x in evaluations),
        "evaluations": evaluations,
        "duration_seconds": round(time.time() - started, 1),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    write("summary.json", summary)
    if equity_rows:
        pd.concat(equity_rows).to_csv(RESULTS / "evaluation-hourly.csv", index=False)
    with (RESULTS / "leaderboard-discovery.csv").open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "family",
                "selection_score",
                "eligible",
                "net_return",
                "sharpe",
                "max_dd",
                "entries",
                "exposure",
            ],
        )
        writer.writeheader()
        for r in sorted(training, key=lambda x: -x["selection_score"]):
            writer.writerow(
                {
                    "id": r["spec"]["id"],
                    "family": r["spec"]["family"],
                    "selection_score": r["selection_score"],
                    "eligible": r["eligible"],
                    **{
                        k: r["discovery"][k]
                        for k in [
                            "net_return",
                            "sharpe",
                            "max_dd",
                            "entries",
                            "exposure",
                        ]
                    },
                }
            )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "evaluations"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
