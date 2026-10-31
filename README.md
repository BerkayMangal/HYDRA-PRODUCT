# HYDRA — BTC Market Intelligence System

> **What this is:** A market monitoring, data aggregation, and research platform focused on BTC spot and perpetual markets.
>
> **What this is NOT:** An automated trading system. No live execution. No capital at risk.
>
> **Current posture:** Dashboard + monitoring. ML and directional signals are research-only until validated on real OOS data.

---

## Quick Start

Deployment and live public-wallet monitoring: [deployment guide](docs/DEPLOYMENT_TR.md).
The dashboard includes the frozen 22-account BTC view, dated research evidence
and reviewed data-source catalog. Wallet observations never place orders.

```bash
# 1. Install dependencies
pip install -r requirements-dev.txt

# 2. Optional provider and notification settings
# Public OKX and Hyperliquid research work without account keys.
# The supported config/config_template.yaml fallback needs no local config file.

# 3. Run dashboard
python main.py

# Dashboard at http://localhost:8080
# API docs at http://localhost:8080/api/docs
```

## Architecture

```
Collectors (OKX, CoinGlass, yfinance, CoinGecko, ...)
    ↓
UnifiedDataStore (source-time z-scores, freshness and TTL)
UnifiedFrameBuilder (feature pipeline, quality tracking)
    ↓
3 Engines (Microstructure, Flow, Macro)
    → EngineOutput with signal grading (DECISION/SECONDARY/CONTEXT)
    ↓
Layer1DecisionEngine
    → DecisionExplanation (NO_SIGNAL / NEUTRAL / WEAK_BULLISH / BULLISH / ...)
    ↓
Dashboard (/api/decision, /api/signal, /api/hybrid)
Telegram (actionable signals only — BULLISH/BEARISH)
```

**ML is disabled by default.** The legacy simulator still has unresolved timing/accounting defects; its results are not validation evidence. The audited public-data research workflow is documented in [the research runbook](docs/research/RUNBOOK_TR.md).

## Runtime Modes

Set `HYDRA_MODE` environment variable:

| Mode | What runs | Use case |
|------|-----------|----------|
| `dashboard` (default) | Everything: collectors, engines, dashboard, Telegram | Live monitoring |
| `collectors_only` | Collectors + dashboard, no engines | Data collection |
| `research` | Nothing live — use ML research CLI | Offline research |
| `evidence` | Nothing live — use evidence CLI | Evidence generation |

## Key Entry Points

| Command | Purpose |
|---------|---------|
| `python main.py` | Start live dashboard |
| `python -m research.lab.restore` | Restore the frozen research data and selections offline |
| `python -m research.lab.verify` | Verify data, staged selections and equity accounting |
| `python -m research.lab.report` | Summarize existing journals without network access |
| `python -m research.lab.data_audit` | Read-only full payload, time and OKX unit diagnostics |
| `python -m research.lab.twap_forward --once --out <new-trial>` | Bootstrap a frozen, public-data TWAP trial; no orders |
| `python -m research.lab.forward_report --trial <recorded-trial>` | Reproduce a forward ledger report offline without settling trades |
| `python -m research.lab.forward_diagnostics --trial <recorded-trial>` | Explain recorded costs, flat controls and account concentration |
| `python -m research.lab.twap_discovery --out <new-directory>` | Predeclared public active-TWAP source screen; no PnL selection |
| `python -m research.lab.funding_probe --out <new-directory>` | Fixed historical funding-versus-cost feasibility screen |
| `python -m ml.research.run_v2` | Legacy ML research; unresolved validation defects |
| `python -m ml.research.run_v2 --synthetic` | Pipeline validation (no API needed) |
| `python -m ml.research.evidence` | Full evidence package (needs OKX) |
| `python -m ml.research.evidence --synthetic` | Evidence pipeline validation |
| `python -m ml.research.evidence --load data/evidence.json` | View saved evidence |
| `python -m pytest tests/ -v` | Run all tests |

## Environment Variables

See `.env.example` for full list. Key groups:

| Variable | Required for | Notes |
|----------|-------------|-------|
| `OKX_API_KEY/SECRET/PASS` | Optional account services | Public market-data endpoints need no keys |
| `TELEGRAM_BOT_TOKEN/CHAT_ID` | Signal delivery | Without these, no Telegram messages |
| `COINGLASS_API_KEY` | Derivatives data | Paid tier, $29-79/mo |
| `HYDRA_FRED_KEY` | Optional macro fallback | Does not provide an authoritative event calendar |
| `ANTHROPIC/OPENAI/XAI_API_KEY` | Pulse Engine | Dashboard narrative only — NOT decision-grade |
| `ML_ENABLED` | ML paper-trade | Default: false. Do not enable without evidence |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/decision` | Full Phase 3 decision explanation |
| `GET /api/signal` | Current signal (legacy format) |
| `GET /api/hybrid` | Decision with backward-compat fields |
| `GET /api/status` | System status (collectors, quality) |
| `GET /api/ml` | ML paper-trade status |
| `GET /api/pulse` | AI narrative (dashboard only) |
| `GET /api/events` | Upcoming macro events |
| `GET /api/health` | Process liveness; not data readiness |
| `GET /api/research` | Dated research evidence; separate from live signals |
| `GET /api/research/costs` | Recorded movements less editable assumed costs; not executed returns |
| `GET /api/research/forward` | Recorded prospective TWAP ledger, quality gates and modeled outcomes |
| `GET /api/research/forward/diagnostics` | Read-only cost attribution and concentration; configure `HYDRA_FORWARD_TRIAL` |
| `GET /api/wallets` | Live public BTC wallet snapshots and source health |
| `GET /api/research/tools` | Reviewed wallet/data tools and trial status |
| `GET /api/version` | Release identity and optional build commit |

## Decision States

| State | Meaning | Telegram? |
|-------|---------|-----------|
| `NO_SIGNAL` | Insufficient evidence — system abstains | No |
| `NEUTRAL` | Market is balanced — valid view | No |
| `WEAK_BULLISH` | Directional lean, low confidence | No |
| `WEAK_BEARISH` | Directional lean, low confidence | No |
| `BULLISH` | Directional signal, medium+ confidence | Yes |
| `BEARISH` | Directional signal, medium+ confidence | Yes |

## Signal Grading

Every sub-signal is classified:

| Grade | Can affect score? | Examples |
|-------|-------------------|----------|
| `DECISION` | Yes — primary | CVD, OI, orderbook, ETF flow, VIX regime |
| `SECONDARY` | Capped at 30% | Funding, L/S ratio, DXY, stablecoin |
| `CONTEXT` | No — weight=0 | Fear & Greed, prediction markets, LLM output |

## ML Research

The ML model (XGBoost BTC/USDC market timing) is **disabled by default**.

To evaluate:
```bash
# On real OKX data (requires API access):
python -m ml.research.evidence

# Pipeline validation only (no API needed):
python -m ml.research.evidence --synthetic
```

Evidence artifacts saved to `data/`. The pipeline produces an automated verdict:
- **REMOVE** — no alpha detected
- **RESEARCH_ONLY** — marginal AUC but insufficient Sharpe
- **PAPER_TRADE** — acceptable OOS metrics for paper trading
- **LIMITED_DECISION** — sufficient evidence for limited decision support

**Do not enable ML without a real-data evidence artifact supporting it.**

## Project Structure

```
main.py                      # Live system entry point
config/
  settings.py                # Typed config
  runtime_modes.py           # Runtime mode discipline
  config_template.yaml       # Config template
collectors/                  # Data collectors (OKX, CoinGlass, yfinance, ...)
  unified.py                 # UnifiedDataStore (source-time z-scores, freshness, TTL)
features/
  pipeline.py                # Canonical feature computation (ring buffer)
  quality.py                 # Feature quality tracking
  registry.py                # Feature contracts
  unified_frame.py           # Single source of truth for all consumers
engines/
  output.py                  # Structured engine output + signal grading
  feature_access.py          # Safe feature access (no silent defaults)
  microstructure/engine.py   # CVD, OI, orderbook, funding
  flow/engine.py             # ETF, netflow, stablecoin, sentiment
  macro/engine.py            # VIX, rates, DXY, prediction markets
signals/
  layer1_decision.py         # Phase 3 decision engine (replaces combiner)
  event_calendar.py          # FOMC/CPI event tracking
ml/
  signal_engine.py           # ML paper-trade engine
  research/
    run_v2.py                # Walk-forward CLI
    walk_forward_v2.py       # Walk-forward engine
    evidence.py              # Full evidence package generator
dashboard.py                 # FastAPI dashboard + API
pulse_engine.py              # Triple-AI narrative (dashboard only)
layer2/telegram_delivery.py  # Telegram signal delivery
research/                    # Frozen rules, public workers, receipt-time replay
docs/research/               # Audit, competitor analysis, evidence, runbook
tests/                       # Offline application and research regressions
```

## What This System Is Not

- Not an automated trading bot — it does not execute trades
- Not a backtested strategy with proven returns — historical search produced no promotion-eligible model
- Not a replacement for human judgment — all signals are decision support
- Not a real-time system — data has 5-min to 15-min latency depending on source
- Not production-grade for capital allocation — infrastructure is research-grade

## Known Limitations

1. **Macro data from yfinance** — 15-min delayed, no SLA, breaks randomly
2. **Engine weights are unvalidated priors** — not empirically optimized
3. **Legacy ML simulator** — unresolved label/timing, imputation and overlapping OOS accounting; keep disabled
4. **No risk management** — no position sizing, no portfolio-level constraints
5. **No execution layer** — signals only, no order management

## Tests

```bash
python -m pytest tests/ -v
# Includes causal timing, source freshness, data integrity, public workers and receipt-time joins
```

## Research delivery — 29 October 2026

[30 October update](docs/research/RELEASE_20261030_TR.md) fixes research poll drift
without widening the inventory window, adds per-request cost scenarios and
records a native public TWAP slice trial. These remain unvalidated research.

[The 31 October forward trial](docs/research/FORWARD_20261031_TR.md) freezes one
TWAP hypothesis before collection, preserves first receipt times, and exposes a
read-only signal ledger. Sample thresholds permit review, never automatic orders.

[The subsequent economics review](docs/research/ALPHA_REVIEW_20261031_TR.md)
records the first modeled losses, an 81-account source screen, historical funding
cost hurdles, and resumed source collection. No validated alpha has been found.

[Current findings in Turkish](docs/research/FINAL_TR.md) cover the earlier data audit,
13,368 historical variants, EMA/spot shadow observations, public Hyperliquid
wallets/liquidations, and documented competitor capabilities. No strategy has
met the promotion criteria. The small wallet sample is concentrated and does
not represent the whole market. Pain and some commercial websites remain
blocked by the cloud network policy; their interfaces/prices were not verified.

[The runbook](docs/research/RUNBOOK_TR.md) explains offline reproduction,
bounded collectors and report generation. A checksummed frozen bundle keeps
the historical experiment reproducible without downloading revised prices.
Large raw live journals remain local files and are not committed.
