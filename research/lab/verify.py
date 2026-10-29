"""Audit persisted selection boundaries and equity accounting, without reranking tests."""

import hashlib
import json
import numpy as np
import pandas as pd
from .engine import ROOT


def main():
    p = ROOT / "results"
    manifest = json.loads((p / "manifest.json").read_text())
    summary = json.loads((p / "summary.json").read_text())
    discovery = json.loads((p / "discovery.json").read_text())
    short = json.loads((p / "frozen-discovery-shortlist.json").read_text())
    val = json.loads((p / "validation.json").read_text())
    selected = json.loads((p / "frozen-selected.json").read_text())
    assert (
        hashlib.sha256((ROOT / "data/btc-usdt-1h.csv").read_bytes()).hexdigest()
        == manifest["dataset_sha256"]
    )
    assert (
        hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest()
        == summary["manifest_sha256"]
    )
    expected = []
    for fam in sorted({r["spec"]["family"] for r in discovery}):
        rows = [x for x in discovery if x["spec"]["family"] == fam and x["eligible"]]
        expected.extend(
            sorted(rows, key=lambda r: (-r["selection_score"], r["spec"]["id"]))[:3]
        )
    assert {x["spec"]["id"] for x in expected} == {x["spec"]["id"] for x in short}
    assert {x["spec"]["id"] for x in short} == {x["spec"]["id"] for x in val}
    expected_selected = []
    for fam in sorted({r["spec"]["family"] for r in val}):
        rows = [
            x
            for x in val
            if x["spec"]["family"] == fam
            and x["validation"]["net_return"] > 0
            and x["validation"]["max_dd"] > -0.35
        ]
        if rows:
            expected_selected.append(
                sorted(
                    rows, key=lambda x: (-x["validation"]["sharpe"], x["spec"]["id"])
                )[0]["spec"]["id"]
            )
    assert expected_selected == [x["spec"]["id"] for x in selected]
    assert expected_selected == [x["spec"]["id"] for x in summary["evaluations"]]
    series = pd.read_csv(p / "evaluation-hourly.csv")
    byid = {x["spec"]["id"]: x for x in summary["evaluations"]}
    for (sid, period), df in series.groupby(["id", "period"]):
        r = byid[sid][period]
        assert not df.timestamp.duplicated().any()
        assert np.isclose(np.expm1(np.log1p(df.net_return).sum()), r["net_return"])
        assert np.isclose(
            np.prod(1 + np.array(list(r["quarterly_returns"].values()))) - 1,
            r["net_return"],
        )
        assert (df.position.between(0, 1)).all()
    print(
        "PASS: dataset/manifest hashes; discovery-only shortlist; validation-only model selection; frozen evaluation IDs; all eight out-of-sample equity curves and quarter compounding."
    )


if __name__ == "__main__":
    main()
