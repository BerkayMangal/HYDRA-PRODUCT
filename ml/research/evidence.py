"""
ml/research/evidence.py
────────────────────────
HYDRA Evidence Package Generator (Phase 5)

Produces a complete, reviewable evidence package that answers:
  1. Does the ML model have real OOS alpha?
  2. Do the deterministic engines add measurable signal?
  3. What should be enabled in production?

USAGE
-----
  # Real data (needs OKX API access):
  python -m ml.research.evidence

  # Synthetic data (pipeline validation only — NOT evidence):
  python -m ml.research.evidence --synthetic

  # View saved evidence:
  python -m ml.research.evidence --load data/evidence.json

OUTPUT
------
  data/evidence_YYYYMMDD_HHMMSS.json — full evidence package
  Printed evidence memo to stdout

DATA REQUIREMENTS
-----------------
  Minimum: 270 days of 1H BTC/USDT OHLCV from OKX
  Recommended: 365 days
  Macro: VIX, QQQ, US10Y, Gold daily from yfinance (15min delayed, best-effort)
  Funding: OKX perpetual funding rate history

  Without OKX API access, use --synthetic for pipeline testing only.
  Synthetic results are NOT evidence and will be flagged as such.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.pipeline import FeaturePipeline, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Engine evaluation (Step 4)
# ---------------------------------------------------------------------------

@dataclass
class EngineEvaluation:
    """Empirical evaluation of one deterministic engine."""
    engine_name: str
    n_samples: int
    # Directional accuracy: does engine direction predict next-24h return sign?
    directional_accuracy: float   # fraction of correct direction calls
    directional_accuracy_strong: float  # accuracy when |score| > 30
    n_strong_signals: int
    # Correlation with forward returns
    score_return_correlation: float
    score_return_pvalue: float
    # Signal activity
    pct_neutral: float
    pct_long: float
    pct_short: float
    avg_abs_score: float
    # Verdict
    verdict: str   # "decision_support" / "downweight" / "context_only" / "remove"
    rationale: str

    def to_dict(self) -> Dict:
        return asdict(self)


def evaluate_engine(
    engine_cls,
    feature_df: pd.DataFrame,
    forward_hours: int = 24,
) -> EngineEvaluation:
    """
    Evaluate whether an engine's score predicts future BTC returns.

    Runs the engine on every forward_hours-th row (non-overlapping),
    then measures directional accuracy and correlation with realized returns.
    """
    from scipy.stats import pearsonr

    engine = engine_cls({})
    engine_name = engine_cls.__name__

    # Forward return
    fwd_ret = feature_df["close"].pct_change(forward_hours).shift(-forward_hours)

    # Subsample to non-overlapping
    idx = list(range(0, len(feature_df) - forward_hours, forward_hours))
    if len(idx) < 20:
        return EngineEvaluation(
            engine_name=engine_name, n_samples=len(idx),
            directional_accuracy=0.5, directional_accuracy_strong=0.5,
            n_strong_signals=0, score_return_correlation=0.0,
            score_return_pvalue=1.0, pct_neutral=1.0, pct_long=0.0,
            pct_short=0.0, avg_abs_score=0.0,
            verdict="remove", rationale="insufficient_data",
        )

    scores = []
    directions = []
    returns = []

    for i in idx:
        row = feature_df.iloc[i]
        ret = fwd_ret.iloc[i]
        if pd.isna(ret):
            continue

        try:
            result = engine.compute(pd.Series(row))
            score = result.score if hasattr(result, 'score') else result.get('score', 0)
            direction = result.direction if hasattr(result, 'direction') else result.get('direction', 'NEUTRAL')
        except Exception:
            continue

        scores.append(float(score))
        directions.append(direction)
        returns.append(float(ret))

    if len(scores) < 20:
        return EngineEvaluation(
            engine_name=engine_name, n_samples=len(scores),
            directional_accuracy=0.5, directional_accuracy_strong=0.5,
            n_strong_signals=0, score_return_correlation=0.0,
            score_return_pvalue=1.0, pct_neutral=1.0, pct_long=0.0,
            pct_short=0.0, avg_abs_score=0.0,
            verdict="remove", rationale="insufficient_data",
        )

    scores_arr = np.array(scores)
    returns_arr = np.array(returns)

    # Directional accuracy: engine says LONG and return > 0, or SHORT and return < 0
    correct = 0
    directional_n = 0
    correct_strong = 0
    strong_n = 0

    for s, d, r in zip(scores, directions, returns):
        if d == "NEUTRAL":
            continue
        directional_n += 1
        if (d == "LONG" and r > 0) or (d == "SHORT" and r < 0):
            correct += 1
        if abs(s) > 30:
            strong_n += 1
            if (d == "LONG" and r > 0) or (d == "SHORT" and r < 0):
                correct_strong += 1

    dir_acc = correct / max(directional_n, 1)
    dir_acc_strong = correct_strong / max(strong_n, 1)

    # Correlation
    corr, pval = pearsonr(scores_arr, returns_arr)

    # Activity distribution
    n = len(directions)
    pct_neutral = directions.count("NEUTRAL") / n
    pct_long = directions.count("LONG") / n
    pct_short = directions.count("SHORT") / n

    # Verdict
    if abs(corr) > 0.10 and pval < 0.05 and dir_acc > 0.52:
        verdict = "decision_support"
        rationale = f"corr={corr:.3f}(p={pval:.3f}), dir_acc={dir_acc:.1%}"
    elif abs(corr) > 0.05 or dir_acc > 0.51:
        verdict = "downweight"
        rationale = f"weak signal: corr={corr:.3f}(p={pval:.3f}), dir_acc={dir_acc:.1%}"
    elif pct_neutral > 0.8:
        verdict = "context_only"
        rationale = f"mostly neutral ({pct_neutral:.0%}), insufficient activity"
    else:
        verdict = "remove"
        rationale = f"no predictive value: corr={corr:.3f}, dir_acc={dir_acc:.1%}"

    return EngineEvaluation(
        engine_name=engine_name,
        n_samples=len(scores),
        directional_accuracy=round(dir_acc, 4),
        directional_accuracy_strong=round(dir_acc_strong, 4),
        n_strong_signals=strong_n,
        score_return_correlation=round(float(corr), 4),
        score_return_pvalue=round(float(pval), 4),
        pct_neutral=round(pct_neutral, 3),
        pct_long=round(pct_long, 3),
        pct_short=round(pct_short, 3),
        avg_abs_score=round(float(np.mean(np.abs(scores_arr))), 2),
        verdict=verdict,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Evidence package
# ---------------------------------------------------------------------------

@dataclass
class EvidencePackage:
    """Complete evidence package for CTO/quant lead review."""
    timestamp_utc: str
    data_source: str           # "real_okx" or "synthetic"
    data_period: str           # e.g. "2025-04-01 to 2026-04-01"
    data_bars: int

    # ML verdict (from walk-forward)
    ml_verdict: str
    ml_rationale: str
    ml_median_sharpe: float
    ml_median_auc: float
    ml_total_oos_return: float
    ml_feature_stability: str
    ml_fold_count: int
    ml_negative_fold_pct: float
    ml_warnings: List[str]

    # Engine evaluations (Step 4)
    engine_evaluations: List[Dict]

    # Production posture (Step 5)
    production_posture: str
    posture_rationale: str
    enabled_components: List[str]
    disabled_components: List[str]
    dashboard_only: List[str]

    # Walk-forward result reference
    wf_artifact_path: str

    # Caveats
    caveats: List[str]
    unknowns: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def determine_production_posture(
    ml_verdict: str,
    engine_evals: List[EngineEvaluation],
    data_source: str,
) -> Tuple[str, str, List[str], List[str], List[str]]:
    """
    Step 5: Determine production posture from evidence.

    Returns: (posture, rationale, enabled, disabled, dashboard_only)
    """
    # Count useful engines
    useful_engines = [e for e in engine_evals if e.verdict in ("decision_support", "downweight")]
    context_engines = [e for e in engine_evals if e.verdict == "context_only"]

    if data_source == "synthetic":
        return (
            "RESEARCH_SANDBOX",
            "Cannot determine production posture from synthetic data. Run on real OKX data.",
            [],
            ["ml_signal_engine", "telegram_delivery_signals"],
            ["dashboard", "pulse_engine", "all_engines_display"],
        )

    enabled = ["dashboard", "quality_gate", "event_calendar", "telegram_alerts_basic"]
    disabled = []
    dashboard_only = ["pulse_engine", "morning_briefing", "weekly_report"]

    # ML posture
    if ml_verdict in ("LIMITED_DECISION", "PAPER_TRADE"):
        enabled.append("ml_paper_trade")
    else:
        disabled.append("ml_signal_engine")

    # Engine posture
    for e in engine_evals:
        if e.verdict == "decision_support":
            enabled.append(f"engine_{e.engine_name}")
        elif e.verdict == "downweight":
            enabled.append(f"engine_{e.engine_name}_downweighted")
        else:
            dashboard_only.append(f"engine_{e.engine_name}_display_only")

    # Determine overall posture
    if ml_verdict == "LIMITED_DECISION" and len(useful_engines) >= 2:
        posture = "LIMITED_DECISION_SUPPORT"
        rationale = (
            f"ML shows marginal OOS alpha ({ml_verdict}). "
            f"{len(useful_engines)} engines show some predictive value. "
            "Signals should be used as decision support with strict suppression rules, "
            "NOT as automated trade execution."
        )
    elif ml_verdict == "PAPER_TRADE" or len(useful_engines) >= 1:
        posture = "DASHBOARD_PLUS_ALERTS"
        rationale = (
            "System provides useful market monitoring and context. "
            "Directional signals are informational, not actionable without human judgment. "
            f"ML: {ml_verdict}. Engines: {len(useful_engines)} marginally useful."
        )
    else:
        posture = "DASHBOARD_MONITORING_ONLY"
        rationale = (
            f"ML: {ml_verdict}. No engine shows reliable predictive value. "
            "System is useful as a data aggregation and monitoring dashboard. "
            "Directional signals should not be acted upon."
        )
        disabled.append("telegram_directional_signals")

    return posture, rationale, enabled, disabled, dashboard_only


# ---------------------------------------------------------------------------
# Evidence memo (Step 6)
# ---------------------------------------------------------------------------

def print_evidence_memo(pkg: Dict) -> None:
    """Print a CTO-readable evidence memo."""
    print()
    print("=" * 72)
    print("HYDRA EVIDENCE REVIEW MEMO")
    print("=" * 72)
    print(f"  Date:        {pkg['timestamp_utc']}")
    print(f"  Data source: {pkg['data_source']}")
    print(f"  Period:      {pkg['data_period']}")
    print(f"  Bars:        {pkg['data_bars']:,}")
    print()

    if pkg["data_source"] == "synthetic":
        print("  ⚠  SYNTHETIC DATA — results are pipeline validation only.")
        print("     These metrics do NOT constitute evidence of alpha or system value.")
        print("     Run on real OKX data for production-grade evidence.")
        print()

    print("─" * 72)
    print("1. ML MODEL VERDICT")
    print("─" * 72)
    print(f"  Verdict:          {pkg['ml_verdict']}")
    print(f"  Median Sharpe:    {pkg['ml_median_sharpe']:+.2f}")
    print(f"  Median AUC:       {pkg['ml_median_auc']:.3f}")
    print(f"  OOS Return:       {pkg['ml_total_oos_return']:.1%}")
    print(f"  Feature Stability:{pkg['ml_feature_stability']}")
    print(f"  Folds:            {pkg['ml_fold_count']} ({pkg['ml_negative_fold_pct']:.0%} negative)")
    print(f"  Rationale:        {pkg['ml_rationale']}")
    for w in pkg.get("ml_warnings", []):
        print(f"  ⚠ {w}")
    print()

    print("─" * 72)
    print("2. DETERMINISTIC ENGINE EVALUATION")
    print("─" * 72)
    for e in pkg["engine_evaluations"]:
        print(f"  {e['engine_name']:<25} verdict={e['verdict']:<18} "
              f"dir_acc={e['directional_accuracy']:.1%}  "
              f"corr={e['score_return_correlation']:+.3f}(p={e['score_return_pvalue']:.3f})  "
              f"strong_n={e['n_strong_signals']}")
    print()

    print("─" * 72)
    print("3. PRODUCTION POSTURE")
    print("─" * 72)
    print(f"  Recommendation:  {pkg['production_posture']}")
    print(f"  Rationale:       {pkg['posture_rationale']}")
    print()
    print("  Enabled:")
    for c in pkg["enabled_components"]:
        print(f"    ✓ {c}")
    print("  Disabled:")
    for c in pkg["disabled_components"]:
        print(f"    ✗ {c}")
    print("  Dashboard only:")
    for c in pkg["dashboard_only"]:
        print(f"    ◐ {c}")
    print()

    print("─" * 72)
    print("4. CAVEATS & UNKNOWNS")
    print("─" * 72)
    for c in pkg.get("caveats", []):
        print(f"  • {c}")
    print()
    print("  Unknowns:")
    for u in pkg.get("unknowns", []):
        print(f"  ? {u}")
    print()

    print("─" * 72)
    print("5. RECOMMENDATION")
    print("─" * 72)
    posture = pkg["production_posture"]
    if posture == "RESEARCH_SANDBOX":
        print("  Run `python -m ml.research.evidence` with real OKX data before")
        print("  making any deployment decisions. Synthetic results are not evidence.")
    elif "MONITORING" in posture:
        print("  System is a useful monitoring dashboard. Directional signals should")
        print("  not be acted upon without independent confirmation.")
    elif "ALERT" in posture:
        print("  System provides useful context and alerts. Signals are informational.")
        print("  Human judgment required for all trading decisions.")
    elif "DECISION" in posture:
        print("  System shows marginal evidence of value. Use as decision SUPPORT only,")
        print("  with strict human oversight. Never auto-execute signals.")
    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_synthetic(days: int = 270) -> pd.DataFrame:
    n = days * 24
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    returns = np.random.normal(0.0001, 0.015, n)
    returns[1000:1500] += 0.002
    returns[2500:3000] -= 0.002
    prices = 100_000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.001, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "close": prices,
        "volume": np.random.lognormal(15, 1, n),
        "vix": 20 + np.cumsum(np.random.normal(0, 0.5, n)).clip(-10, 20),
        "qqq": 500 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, n))),
        "us10y": 4.0 + np.cumsum(np.random.normal(0, 0.01, n)).clip(-1, 2),
        "gold": 2000 * np.exp(np.cumsum(np.random.normal(0.00005, 0.003, n))),
        "funding": np.random.normal(0.0001, 0.0003, n).clip(-0.001, 0.001),
    }, index=dates)
    return df


def fetch_real_data(days: int) -> Optional[pd.DataFrame]:
    """Fetch real OKX data. Returns None if unavailable."""
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Install: pip install ccxt")
        return None

    try:
        logger.info("Fetching {} days from OKX...", days)
        ex = ccxt.okx({"options": {"defaultType": "swap"}})
        since = ex.parse8601((datetime.utcnow() - timedelta(days=days + 5)).isoformat())
        candles = []
        while True:
            batch = ex.fetch_ohlcv("BTC/USDT:USDT", "1h", since=since, limit=300)
            if not batch:
                break
            candles.extend(batch)
            since = batch[-1][0] + 1
            if len(batch) < 300:
                break
            time.sleep(0.15)
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        logger.info("Fetched {} candles ({} — {})", len(df), str(df.index[0])[:10], str(df.index[-1])[:10])

        # Macro (best-effort)
        try:
            import yfinance as yf
            for ticker, col in {"^VIX": "vix", "QQQ": "qqq", "^TNX": "us10y", "GC=F": "gold"}.items():
                try:
                    raw = yf.download(ticker, period=f"{days+10}d", interval="1d", progress=False, auto_adjust=True)
                    if raw.empty:
                        continue
                    close = raw["Close"].squeeze()
                    if close.index.tz is None:
                        close.index = close.index.tz_localize("UTC")
                    else:
                        close.index = close.index.tz_convert("UTC")
                    df[col] = close.reindex(df.index, method="ffill")
                except Exception:
                    pass
        except ImportError:
            logger.warning("yfinance not installed — macro features absent")

        # Funding (best-effort)
        try:
            import requests
            records, after = [], ""
            for _ in range(200):
                params = {"instId": "BTC-USDT-SWAP", "limit": "100"}
                if after:
                    params["after"] = after
                r = requests.get("https://www.okx.com/api/v5/public/funding-rate-history",
                                 params=params, timeout=15)
                data = r.json().get("data", [])
                if not data:
                    break
                for row in data:
                    records.append({"ts": int(row["fundingTime"]), "fr": float(row["fundingRate"])})
                after = data[-1]["fundingTime"]
                time.sleep(0.2)
                if len(data) < 100:
                    break
            if records:
                fdf = pd.DataFrame(records)
                fdf["ts"] = pd.to_datetime(fdf["ts"], unit="ms", utc=True)
                fdf = fdf.set_index("ts").sort_index()
                df["funding"] = fdf["fr"].resample("1h").last().ffill().reindex(df.index, method="ffill")
        except Exception:
            pass

        return df
    except Exception as e:
        logger.error("Data fetch failed: {}", e)
        return None


def build_evidence(
    df: pd.DataFrame,
    data_source: str,
    wf_artifact_path: str,
) -> EvidencePackage:
    """Build complete evidence package from data + walk-forward results."""

    # Feature pipeline
    pipe = FeaturePipeline()
    feature_df = pipe.transform_batch(df)
    logger.info("Feature matrix: {} rows × {} features", len(feature_df), len(FEATURE_NAMES))

    # ── ML walk-forward (Step 2) ──────────────────────────────────────
    from ml.research.walk_forward_v2 import WalkForwardEngineV2
    wf_engine = WalkForwardEngineV2(
        forward_hours=24, train_days=90, val_days=15,
        test_days=30, step_days=14,
    )
    wf_result = wf_engine.run(feature_df, expanding=True)
    wf_engine.save(wf_result, wf_artifact_path)

    # ── Engine evaluation (Step 4) ────────────────────────────────────
    from engines.microstructure.engine import MicrostructureEngine
    from engines.flow.engine import FlowEngine
    from engines.macro.engine import MacroEngine

    engine_evals = []
    for cls in (MicrostructureEngine, FlowEngine, MacroEngine):
        logger.info("Evaluating {}...", cls.__name__)
        ev = evaluate_engine(cls, feature_df, forward_hours=24)
        engine_evals.append(ev)
        logger.info("  {} → {} (dir_acc={:.1%}, corr={:+.3f})",
                     cls.__name__, ev.verdict, ev.directional_accuracy,
                     ev.score_return_correlation)

    # ── Production posture (Step 5) ───────────────────────────────────
    posture, rationale, enabled, disabled, dash_only = determine_production_posture(
        wf_result.ml_verdict, engine_evals, data_source,
    )

    # ── Caveats ───────────────────────────────────────────────────────
    caveats = [
        "All metrics are on historical data. Past performance does not predict future.",
        "Macro features sourced from yfinance (15min delayed, no SLA).",
        "Engine weights are unvalidated priors — empirical evaluation is directional only.",
        f"Cost model: {wf_result.cost_model['fee_per_trade']:.3%} fee + "
        f"{wf_result.cost_model['slippage_bps']}bps slippage per side.",
    ]
    if data_source == "synthetic":
        caveats.insert(0, "⚠ SYNTHETIC DATA — not real evidence. Run on OKX data.")

    unknowns = [
        "Live execution slippage may differ from assumed 2bps.",
        "Regime changes may invalidate historical patterns.",
        "Feature importance stability on real data is unknown until real run completes.",
        "yfinance may break at any time — macro features are fragile.",
    ]

    period = f"{str(df.index[0])[:10]} to {str(df.index[-1])[:10]}"
    neg_pct = sum(1 for f in wf_result.folds if f.sharpe_test < 0) / max(len(wf_result.folds), 1)

    return EvidencePackage(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        data_source=data_source,
        data_period=period,
        data_bars=len(df),
        ml_verdict=wf_result.ml_verdict,
        ml_rationale=wf_result.verdict_rationale,
        ml_median_sharpe=wf_result.median_sharpe,
        ml_median_auc=wf_result.median_auc_test,
        ml_total_oos_return=wf_result.total_oos_return,
        ml_feature_stability=wf_result.feature_stability.stability_verdict,
        ml_fold_count=len(wf_result.folds),
        ml_negative_fold_pct=neg_pct,
        ml_warnings=wf_result.warnings,
        engine_evaluations=[e.to_dict() for e in engine_evals],
        production_posture=posture,
        posture_rationale=rationale,
        enabled_components=enabled,
        disabled_components=disabled,
        dashboard_only=dash_only,
        wf_artifact_path=wf_artifact_path,
        caveats=caveats,
        unknowns=unknowns,
    )


def main():
    parser = argparse.ArgumentParser(
        description="HYDRA Evidence Package Generator (Phase 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--days", type=int, default=270, help="Days of history")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--load", type=str, help="Load saved evidence package")
    parser.add_argument("--out", type=str, help="Output path")
    args = parser.parse_args()

    if args.load:
        with open(args.load) as f:
            print_evidence_memo(json.load(f))
        return

    # Get data
    if args.synthetic:
        logger.info("Generating synthetic data ({} days)...", args.days)
        df = generate_synthetic(args.days)
        data_source = "synthetic"
    else:
        df = fetch_real_data(args.days)
        if df is None or len(df) < 500:
            logger.error("Insufficient real data. Use --synthetic for pipeline validation.")
            sys.exit(1)
        data_source = "real_okx"

    # Build evidence
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    wf_path = args.out or f"data/evidence_{ts}.json"
    wf_artifact = wf_path.replace(".json", "_wf.json")

    pkg = build_evidence(df, data_source, wf_artifact)

    # Save
    Path(wf_path).parent.mkdir(parents=True, exist_ok=True)
    with open(wf_path, "w") as f:
        json.dump(pkg.to_dict(), f, indent=2, default=str)
    logger.info("Evidence saved: {}", wf_path)

    # Print memo
    print_evidence_memo(pkg.to_dict())
    print(f"\nArtifacts:\n  Evidence: {wf_path}\n  Walk-forward: {wf_artifact}")


if __name__ == "__main__":
    main()
