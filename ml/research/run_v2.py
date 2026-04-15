"""
ml/research/run_v2.py
──────────────────────
HYDRA ML Research CLI v2 (Phase 4)

This is the ONLY legitimate way to generate ML performance evidence.
Dashboard paper-trade metrics are NOT citable.

Usage:
    python -m ml.research.run_v2                        # full run (needs OKX data)
    python -m ml.research.run_v2 --synthetic            # synthetic data (for testing pipeline)
    python -m ml.research.run_v2 --load data/wf_v2.json # view saved result

Output:
    data/wf_v2_YYYYMMDD_HHMMSS.json — full result with all fold metrics
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
import numpy as np
import pandas as pd

from features.pipeline import FeaturePipeline


def fetch_data(days: int) -> pd.DataFrame | None:
    """Fetch OKX 1H OHLCV + macro. Returns None on failure."""
    try:
        import ccxt
        logger.info("Fetching {} days from OKX...", days)
        ex = ccxt.okx({"options": {"defaultType": "swap"}})
        since = ex.parse8601((datetime.utcnow() - timedelta(days=days + 5)).isoformat())
        candles = []
        while True:
            batch = ex.fetch_ohlcv("BTC/USDT:USDT", "1h", since=since, limit=300)
            if not batch: break
            candles.extend(batch)
            since = batch[-1][0] + 1
            if len(batch) < 300: break
            time.sleep(0.15)
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        logger.info("Fetched {} candles", len(df))

        # Macro
        try:
            import yfinance as yf
            for ticker, col in {"^VIX": "vix", "QQQ": "qqq", "^TNX": "us10y", "GC=F": "gold"}.items():
                try:
                    raw = yf.download(ticker, period=f"{days+10}d", interval="1d", progress=False, auto_adjust=True)
                    if raw.empty: continue
                    close = raw["Close"].squeeze()
                    if close.index.tz is None: close.index = close.index.tz_localize("UTC")
                    else: close.index = close.index.tz_convert("UTC")
                    df[col] = close.reindex(df.index, method="ffill")
                except: pass
        except ImportError:
            pass
        return df
    except Exception as e:
        logger.error("Data fetch failed: {}", e)
        return None


def generate_synthetic(days: int = 270) -> pd.DataFrame:
    """Generate synthetic BTC-like 1H data for pipeline testing."""
    n = days * 24
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    # Random walk with slight drift + regime changes
    returns = np.random.normal(0.0001, 0.015, n)
    # Add a few regime changes
    returns[1000:1500] += 0.002   # bull regime
    returns[2500:3000] -= 0.002   # bear regime
    prices = 100_000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.001, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "close": prices,
        "volume": np.random.lognormal(15, 1, n),
    }, index=dates)
    # Synthetic macro
    df["vix"] = 20 + np.cumsum(np.random.normal(0, 0.5, n)).clip(-10, 20)
    df["qqq"] = 500 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, n)))
    df["us10y"] = 4.0 + np.cumsum(np.random.normal(0, 0.01, n)).clip(-1, 2)
    df["gold"] = 2000 * np.exp(np.cumsum(np.random.normal(0.00005, 0.003, n)))
    # Synthetic funding
    df["funding"] = np.random.normal(0.0001, 0.0003, n).clip(-0.001, 0.001)
    return df


def print_report(result: dict) -> None:
    folds = result.get("folds", [])
    print("\n" + "=" * 70)
    print("HYDRA ML WALK-FORWARD RESEARCH REPORT (v2)")
    print("=" * 70)
    print(f"  Timestamp:  {result.get('timestamp_utc', '?')}")
    print(f"  Folds:      {len(folds)}")
    print(f"  Labels:     non-overlapping (every {result['config']['forward_hours']}h)")
    print(f"  Embargo:    {result['config']['purge_bars']} bars")
    print(f"  Costs:      fee={result['cost_model']['fee_per_trade']:.3%} + "
          f"slippage={result['cost_model']['slippage_bps']}bps/side")
    print()

    print("AGGREGATE METRICS (median across folds):")
    print(f"  AUC:        {result['median_auc_test']:.3f}")
    print(f"  Sharpe:     {result['median_sharpe']:+.2f}")
    print(f"  Sortino:    {result['median_sortino']:+.2f}")
    print(f"  Max DD:     {result['median_max_dd']:.1%}")
    print(f"  Hit Rate:   {result['median_hit_rate']:.1%}")
    print(f"  Avg Trade:  {result['median_avg_trade']:.2%}")
    print(f"  OOS Return: {result['total_oos_return']:.1%}")
    print()

    print("PER-FOLD BREAKDOWN:")
    hdr = f"  {'#':>2}  {'Train':>10}  {'Test':>10}  {'N':>3}  {'AUC':>5}  {'Sharpe':>7}  {'MaxDD':>6}  {'Trades':>6}  {'Return':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in folds:
        print(
            f"  {f['fold_idx']+1:>2}  {f['train_start']:>10}  {f['test_start']:>10}  "
            f"{f['test_samples']:>3}  {f['auc_test']:>5.3f}  {f['sharpe_test']:>+7.2f}  "
            f"{f['max_dd_test']:>5.1%}  {f['n_trades_test']:>6}  {f['total_return_test']:>+6.1%}"
        )
    print()

    stab = result.get("feature_stability", {})
    print(f"FEATURE STABILITY: {stab.get('stability_verdict', '?')}")
    print(f"  Mean rank correlation: {stab.get('mean_rank_correlation', 0):.2f}")
    # Top features across all folds
    fstats = stab.get("feature_stats", {})
    if fstats:
        top = sorted(fstats.items(), key=lambda x: x[1].get("mean_rank", 999))[:10]
        print("  Top 10 most stable features:")
        for feat, s in top:
            print(f"    {feat:<25} mean_rank={s['mean_rank']:.1f}  top10_pct={s['top10_pct']:.0%}")
    print()

    print(f"ML VERDICT: {result['ml_verdict']}")
    print(f"  Rationale: {result['verdict_rationale']}")
    for w in result.get("warnings", []):
        print(f"  ⚠ {w}")
    print()
    print("NOTE: All metrics are fee+slippage adjusted, on non-overlapping OOS data.")
    print("Past performance does not predict future performance.")


def main():
    parser = argparse.ArgumentParser(description="HYDRA ML Research v2")
    parser.add_argument("--days", type=int, default=270, help="Days of history")
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument("--fwd-hours", type=int, default=24)
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--load", type=str, help="Load saved result")
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    if args.load:
        with open(args.load) as f:
            print_report(json.load(f))
        return

    # Data
    if args.synthetic:
        logger.info("Using synthetic data ({} days)", args.days)
        df = generate_synthetic(args.days)
    else:
        df = fetch_data(args.days)
        if df is None or len(df) < 500:
            logger.error("Insufficient data")
            sys.exit(1)

    # Features
    pipe = FeaturePipeline()
    fdf = pipe.transform_batch(df)
    logger.info("Feature matrix: {} rows", len(fdf))

    # Run
    from ml.research.walk_forward_v2 import WalkForwardEngineV2
    engine = WalkForwardEngineV2(
        forward_hours=args.fwd_hours,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    result = engine.run(fdf, expanding=not args.rolling)

    # Report
    d = result.to_dict()
    print_report(d)

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or f"data/wf_v2_{ts}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    engine.save(result, out)
    print(f"\nArtifact saved: {out}")


if __name__ == "__main__":
    main()
