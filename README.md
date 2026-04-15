# HYDRA — BTC Market Intelligence System

> **What this is:** A market monitoring, data aggregation, and research platform for BTC/USDC.
>
> **What this is NOT:** An automated trading system. No live execution. No capital at risk.
>
> **Current posture:** Dashboard + monitoring. ML and directional signals are research-only until validated on real OOS data.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run dashboard
python main.py

# Dashboard at http://localhost:8080
# API docs at http://localhost:8080/api/docs
```

## Architecture

```
Collectors (OKX, CoinGlass, yfinance, CoinGecko, ...)
    ↓
UnifiedDataStore (z-scores, LOCF decay, TTL)
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

**ML is separate:** XGBoost market timing model runs as paper-trade only. Research pipeline at `python -m ml.research.run_v2`.

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
| `python -m ml.research.run_v2` | Walk-forward ML evaluation |
| `python -m ml.research.run_v2 --synthetic` | Pipeline validation (no API needed) |
| `python -m ml.research.evidence` | Full evidence package (needs OKX) |
| `python -m ml.research.evidence --synthetic` | Evidence pipeline validation |
| `python -m ml.research.evidence --load data/evidence.json` | View saved evidence |
| `python -m pytest tests/ -v` | Run all tests |

## Environment Variables

See `.env.example` for full list. Key groups:

| Variable | Required for | Notes |
|----------|-------------|-------|
| `OKX_API_KEY/SECRET/PASS` | Live data | Public endpoints work without keys |
| `TELEGRAM_BOT_TOKEN/CHAT_ID` | Signal delivery | Without these, no Telegram messages |
| `COINGLASS_API_KEY` | Derivatives data | Paid tier, $29-79/mo |
| `FRED_API_KEY` | Event calendar | Free, recommended |
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
| `GET /api/health` | Health check |

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
  unified.py                 # UnifiedDataStore (z-scores, LOCF, TTL)
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
api_status.py                # /api/status builder
pulse_engine.py              # Triple-AI narrative (dashboard only)
layer2/telegram_delivery.py  # Telegram signal delivery
tests/                       # 135+ tests
```

## What This System Is Not

- Not an automated trading bot — it does not execute trades
- Not a backtested strategy with proven returns — ML verdict is pending real-data run
- Not a replacement for human judgment — all signals are decision support
- Not a real-time system — data has 5-min to 15-min latency depending on source
- Not production-grade for capital allocation — infrastructure is research-grade

## Known Limitations

1. **Macro data from yfinance** — 15-min delayed, no SLA, breaks randomly
2. **Engine weights are unvalidated priors** — not empirically optimized
3. **ML model untested on real data** — synthetic validation only (AUC=0.50, verdict=REMOVE)
4. **No risk management** — no position sizing, no portfolio-level constraints
5. **No execution layer** — signals only, no order management

## Tests

```bash
python -m pytest tests/ -v
# 135 tests covering: features, quality, engines, decision, ML pipeline, integration
```
