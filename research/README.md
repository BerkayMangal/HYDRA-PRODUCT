# Research workflow

The [wallet/liquidation tools catalog](../docs/research/TOOLS_CATALOG_TR.md) compares 16 product pages and three independent tools, with source-level evidence and explicit gaps in API/UI validation. It prioritizes historical realized liquidations, TWAP data and address-level orderbook response rather than accepting OI drops or leaderboard scores as alpha evidence.

The current [runbook](../docs/research/RUNBOOK_TR.md) and [delivery report](../docs/research/FINAL_TR.md) cover the complete workflow. Repo modules in `research/lab/` restore the checksummed frozen experiment, verify staged selection, collect bounded public-data observations and generate offline reports. Set `HYDRA_RESEARCH_DIR` before starting a module to override the default `data/research/` directory. No live orders or notifications are implemented here.

The [additional-cohort study](../docs/research/CONTINUATION_20261029_TR.md) keeps the original six-account cohort and a separate 16-account sample. `wallet_breadth.py` reports equal-account directions and leave-one-account-out sensitivity; these diagnostics do not infer independent owners or alpha. `python -m research.lab.breadth_report` replays the recorded snapshots as an explicitly exploratory analysis. Larger cohorts use a bounded slower public API budget.

```bash
python -m research.lab.restore
python -m research.lab.verify
python -m research.lab.report
# Public one-cycle reads after restore:
python -m research.lab.okx_shadow --once
python -m research.lab.hyperliquid_shadow --once
```

`/api/research` and Research Evidence display a dated report separately from the production decision engine. The frozen historical experiment found no promotion-eligible model; no alpha claim is made by these tools.

# EMA and spot-flow shadow comparison

This research module compares the frozen candidate `806ef45dc42abb7a` (completed
4H BTC/USDT spot bars, EMA4 above EMA200, long/flat) with the same baseline
requiring positive quote-volume imbalance over both 5-minute and 15-minute
spot windows. Thresholds are fixed at zero; no parameters are fitted here.

Use `compare(connection, hourly_candles, as_of_ms)` with the SQLite journal
produced by the local research collector. Its `health` records must include
spot trade-ID overlap status. Both source event times and actual receipt times
are filtered at the decision timestamp. Unknown coverage, gaps, stale trades,
or incomplete price history cause abstention. Later trade backfills cannot
change past decisions. Confirmation can remove baseline exposure; it cannot
create a position against the baseline.

`replay_paired_portfolios(records)` compares the recorded decisions using cash
and BTC inventory. Costs are charged on exposure changes rather than on every
forecast; positions are marked at subsequent recorded reference quotes.
Missing confirmation exits the experimental portfolio to cash at that decision's
recorded quote. The default cost is 12 bp per side; repeat with 22 bp for stress.
Reference quotes are hypothetical fills. Returns, drawdown between observation
times, slippage, and live profitability are not established by this replay.

There are no order, account, or notification calls in this package. It does not
affect the main application's directional signals. Prospective comparisons
must remain separate from the earlier historical search and its rejected models.

Validation:

```bash
python -m pytest tests/test_data_integrity.py tests/test_spot_confirmation.py -q
python -m pytest tests/ -q
```

The collector `python -m research.lab.okx_shadow` records
`comparison_decisions` in its local SQLite journal. Mutable journals under `data/research/` are local artifacts, not a Railway deployment. Existing
pre-comparison observations must not be relabelled as prospective predictions.

## Public Hyperliquid wallet research

`hyperliquid_features.py` keeps base-unit open interest separate from USD
notional and treats Hyperliquid funding as hourly simple-rate APR. Wallet
snapshots are scoped to a frozen public-account cohort, with receipt/source
times, concentration, missing liquidation levels and inventory changes
independent of price changes. Estimated liquidation levels are not actual
forced trades.

Only fills with an explicit `liquidation` object count as realized evidence;
victim/counterparty sides are distinguished and coin/trade ID deduplicates the
two sides of a trade. These are sampled records, not a market-wide feed.

The client in `collectors/hyperliquid_research.py` allows only public `/info`
reads. No API key, signer, order endpoint or copy-trading code is used. The journal worker is `python -m research.lab.hyperliquid_shadow`; its frozen cohort is restored from the evidence bundle. Raw snapshots and SQLite data remain ignored runtime files.

Run `python -m pytest tests/test_hyperliquid_research.py -q` for regressions.
Product/source findings and website-access limitations are in
[the competitor report](../docs/research/COMPETITORS_TR.md).
