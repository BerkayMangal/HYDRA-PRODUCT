"""
main.py — HYDRA Trading Bot v8 (Phase 2–5 rebuild)
====================================================
Single entry point (Procfile: python main.py).

ARCHITECTURE (post-Phase 3)
---------------------------
  Collectors → raw_data
  UnifiedDataStore → collector z-scores
  UnifiedFrameBuilder → unified feature frame + quality gate
  3 engines (Micro/Flow/Macro) → EngineOutput with signal grading
  Layer1DecisionEngine → DecisionExplanation (NO_SIGNAL/NEUTRAL/BULLISH/BEARISH)
  Telegram delivery (actionable signals only)
  Dashboard (all signals for display)
  ML (paper-trade only, research via ml.research.run_v2)

KEY DESIGN DECISIONS
--------------------
  - NO_SIGNAL ≠ NEUTRAL: epistemic abstention vs market balance
  - Context signals (F&G, prediction markets) cannot move score
  - Engines suppress independently on insufficient decision-grade data
  - Quality gate can block entire cycle
  - Pulse/LLM layer is dashboard-only, never enters decision path
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
from loguru import logger

# ── New typed settings (Step 1) ───────────────────────────────────────────
from config.settings import settings

# ── Collectors (unchanged) ────────────────────────────────────────────────
from collectors.defi_collector import DeFiLlamaCollector
from collectors.etf_collector import ETFFlowCollector
from collectors.onchain_collectors import (
    MempoolCollector,
    BybitCollector,
    CoinGeckoKeyCollector,
)
from collectors.extra_collectors import (
    BinanceGlobalCollector,
    FearGreedCollector,
    MarketSentimentCollector,
)
from collectors.kalshi_collector import KalshiCollector
from collectors.macro_collector import MacroCollector
from collectors.okx_collector import OKXCollector
from collectors.unified import UnifiedDataStore

# ── Dashboard (unchanged) ─────────────────────────────────────────────────
from dashboard import start_dashboard, sanitize
from dashboard import state as dashboard_state

# ── Core pipeline ─────────────────────────────────────────────────────────
from layer2 import TelegramDelivery
from collectors.prediction_markets import PredictionMarketsCollector
from collectors.venom_collector import VenomCollector
from services.morning_briefing import MorningBriefingAgent
from services.weekly_report import WeeklyReportGenerator
from services.signal_tracker import SignalTracker
from services.alert_engine import AlertEngine

# ── Engines (Phase 3: accessed directly, not through old combiner) ────────
from engines.microstructure.engine import MicrostructureEngine
from engines.flow.engine import FlowEngine
from engines.macro.engine import MacroEngine

# ── Event calendar ────────────────────────────────────────────────────────
from signals.event_calendar import EventCalendar

# ── New: ML factory — returns _MLEngineStub if disabled (Step 1) ─────────
from ml import build_ml_engine
from engines.pulse_engine import PulseEngine
from telegram_bot.bot import TelegramCommandBot

# ── Phase 2: Unified feature frame ───────────────────────────────────────
from features.unified_frame import UnifiedFrameBuilder

# ── Phase 3: Structured decision engine ──────────────────────────────────
from signals.layer1_decision import Layer1DecisionEngine, DecisionState

# ── Phase 6: Runtime modes ───────────────────────────────────────────────
from config.runtime_modes import (
    get_runtime_mode, validate_startup, print_startup_banner, MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level,
    )
    logger.add(
        "logs/hydra_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )


# ---------------------------------------------------------------------------
# HydraBot
# ---------------------------------------------------------------------------

class HydraBot:
    """
    Top-level orchestrator for HYDRA v7.

    Differences from v6:
      - settings.load() replaces config.load()
      - build_ml_engine() factory replaces direct MLSignalEngine()
      - ML training runs in a daemon thread (never blocks startup)
      - HybridDecisionLayer sits between Layer1 and TelegramDelivery
      - EventCalendar provides real-timestamp event status each cycle
    """

    _MIN_MATURITY_TO_DELIVER: float = 0.5

    def __init__(self) -> None:
        settings.load()
        raw = self._raw_config()

        setup_logging(settings.deployment.log_level)

        # ── Phase 6: Runtime mode validation ─────────────────────────
        self.runtime_mode = get_runtime_mode()
        startup_warnings = validate_startup(self.runtime_mode)
        print_startup_banner(self.runtime_mode, startup_warnings)

        # ── Collectors ────────────────────────────────────────────────
        self.collectors: Dict = {
            "okx":        OKXCollector(raw),
            "etf":        ETFFlowCollector(raw),
            "polymarket": PredictionMarketsCollector(raw),
            "kalshi":     KalshiCollector(raw),
            "macro":      MacroCollector(raw),
            "fear_greed": FearGreedCollector(raw),
            "binance":    BinanceGlobalCollector(raw),
            "sentiment":  MarketSentimentCollector(raw),
            "defi":       DeFiLlamaCollector(raw),
            "venom":      VenomCollector(raw),
            "mempool":    MempoolCollector(raw),
            "bybit":      BybitCollector(raw),
            "cg_plus":    CoinGeckoKeyCollector(raw),
        }

        if settings.api_keys.coinglass_key:
            try:
                from collectors.coinglass_collector import CoinGlassCollector
                self.collectors["coinglass"] = CoinGlassCollector(raw)
                self.use_coinglass = True
                logger.info("CoinGlass enabled")
            except Exception as exc:
                logger.warning("CoinGlass init failed: {}", exc)
                self.use_coinglass = False
        else:
            self.use_coinglass = False

        if settings.api_keys.cryptoquant_key:
            try:
                from collectors.cryptoquant_collector import CryptoQuantCollector
                self.collectors["cryptoquant"] = CryptoQuantCollector(raw)
                self.use_cryptoquant = True
                logger.info("CryptoQuant enabled")
            except Exception as exc:
                logger.warning("CryptoQuant init failed: {}", exc)
                self.use_cryptoquant = False
        else:
            self.use_cryptoquant = False
            logger.info("CryptoQuant key not set — exchange_flow signals will be zero")

        # ── Core pipeline ─────────────────────────────────────────────
        self.data_store = UnifiedDataStore(raw)
        self.tracker    = SignalTracker()
        self.delivery   = TelegramDelivery(raw)

        # ── Phase 3: Layer 1 engines (accessed directly) ─────────────
        self.micro_engine = MicrostructureEngine(raw)
        self.flow_engine  = FlowEngine(raw)
        self.macro_engine = MacroEngine(raw)

        # ── Phase 2: Unified feature frame builder ────────────────────
        self.frame_builder = UnifiedFrameBuilder()

        # ── WARM-UP: Seed ring buffer with historical 1H candles ──────
        # Without this, the pipeline needs 5+ hours of live data before
        # it can compute OHLCV features (EMA, RSI, ATR, etc.).
        # This fetches ~200 1H bars from OKX and seeds the ring buffer
        # so the system is operational within seconds of deployment.
        self._warm_up_pipeline()

        # ── Phase 3: Structured decision engine ──────────────────────
        self.decision_engine = Layer1DecisionEngine(
            signal_threshold=settings.layer1.signal_threshold,
        )

        # ── Event calendar (Step 6) ───────────────────────────────────
        self.event_calendar = EventCalendar(fred_api_key=settings.api_keys.fred_key)
        self.event_calendar.refresh()

        # ── Auxiliary ─────────────────────────────────────────────────
        self.report_gen     = WeeklyReportGenerator()
        self.briefing_agent = MorningBriefingAgent()
        self.pulse_engine   = PulseEngine()
        self.alert_engine   = AlertEngine()
        self.tg_bot         = TelegramCommandBot()

        # ── ML Engine (Steps 1–3) ─────────────────────────────────────
        # build_ml_engine() checks settings.ml.enabled AND xgboost availability.
        # Returns _MLEngineStub (zero-overhead, same interface) if either fails.
        self.ml_engine = build_ml_engine()

        # ML retrain scheduler — fires at startup then on configured interval.
        # Uses a single daemon thread per retrain; zero extra dependencies.
        self._last_ml_retrain: float = 0.0
        if settings.ml.enabled and hasattr(self.ml_engine, 'schedule_training'):
            self._start_ml_training()

        # ── Accumulated feature store ─────────────────────────────────
        self._accumulated_data:       dict = {}
        self._accumulated_timestamps: dict = {}

        # ── Polling intervals ─────────────────────────────────────────
        self.burst_duration  = 600
        self.intervals_burst  = {"fast": 30,  "medium": 120, "slow": 300,  "very_slow": 300}
        self.intervals_normal = {"fast": 300, "medium": 900, "slow": 3600, "very_slow": 14400}
        self.last_poll        = {k: 0 for k in self.intervals_normal}

        # ── State ─────────────────────────────────────────────────────
        self.start_time         = datetime.now(timezone.utc)
        self.heartbeat_interval = settings.deployment.heartbeat_interval_sec

        dashboard_state.bot_start_time = self.start_time
        start_dashboard()

        logger.info("Active collectors: {}", ", ".join(self.collectors.keys()))

        # Pulse Engine — her 60s AI market analizi
        self.pulse_engine.start(lambda: self._accumulated_data)

        # Morning Briefing — 10:00 & 16:00 London Slack
        self.briefing_agent.target_hours = [10, 16]
        self.briefing_agent.start(lambda: self._accumulated_data)

        # Telegram Command Bot
        self.tg_bot.start(
            get_market_data = lambda: self._accumulated_data,
            get_pulse       = lambda: self.pulse_engine.current,
            get_signal      = lambda: dashboard_state.current_signal,
        )

        logger.info("Active collectors: {}", ", ".join(self.collectors.keys()))
        # Wire pulse engine directly to dashboard API (bypasses state.pulse sync)
        import dashboard as _db
        _db._pulse_engine = self.pulse_engine
        _db._signal_tracker = self.tracker  # Phase 4: wire tracker for /api/performance
        _db.set_ml_config(settings.ml)  # FIX: wire ML config to Blueprint API

        logger.info("HydraBot v7 initialised ✅")

    # ------------------------------------------------------------------
    # Pipeline warm-up — fetches historical 1H candles at startup
    # ------------------------------------------------------------------

    def _warm_up_pipeline(self) -> None:
        """
        Fetch historical 1H OHLCV from OKX and seed the ring buffer.

        Without this, the pipeline needs 5+ hours of live 5-min candle
        accumulation before it has enough 1H bars to compute features
        like EMA(50), RSI(14), ATR(14), etc.

        Fetches 250 bars (~10 days) which is enough for EMA(200) to
        start computing meaningful values.
        """
        try:
            exchange = self.collectors["okx"].exchange
            symbol = self.collectors["okx"].symbol

            logger.info("[Warm-up] Fetching historical 1H candles from OKX...")
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=250)

            if not ohlcv or len(ohlcv) < 10:
                logger.warning("[Warm-up] OKX returned {} candles — insufficient", len(ohlcv) if ohlcv else 0)
                return

            # Convert to DataFrame expected by pipeline.warm_up()
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")

            # Also try to fetch funding rate history for the warm-up period
            try:
                funding_resp = exchange.fetch_funding_rate_history(symbol, limit=100)
                if funding_resp:
                    fund_df = pd.DataFrame([
                        {"timestamp": pd.Timestamp(f["timestamp"], unit="ms", tz="UTC"),
                         "funding": float(f.get("fundingRate", 0))}
                        for f in funding_resp if f.get("fundingRate") is not None
                    ])
                    if not fund_df.empty:
                        fund_df = fund_df.set_index("timestamp")
                        fund_df = fund_df.resample("1h").last()
                        df = df.join(fund_df, how="left")
            except Exception:
                pass  # Funding history optional

            self.frame_builder.warm_up(df)
            logger.info("[Warm-up] ✅ Ring buffer seeded with {} 1H bars", len(df))

        except Exception as exc:
            logger.warning("[Warm-up] Failed (non-fatal, system will cold-start): {}", exc)

    # ------------------------------------------------------------------
    # ML background training
    # ------------------------------------------------------------------

    def _start_ml_training(self) -> None:
        """Launch ML training in a daemon thread. Records completion time for schedule tracking."""
        def _train() -> None:
            logger.info("[Main] Background ML training starting...")
            try:
                ok = self.ml_engine.schedule_training()
                if ok:
                    self._last_ml_retrain = time.time()
                    logger.info("[Main] ML training complete ✅")
                else:
                    logger.warning("[Main] ML training failed or insufficient data")
            except Exception as exc:
                logger.error("[Main] ML training exception: {}", exc)

        t = threading.Thread(target=_train, daemon=True, name="ml-training")
        t.start()

    def _check_retrain_schedule(self) -> None:
        """
        Trigger ML retrain if the configured interval has elapsed.
        Called once per main loop iteration — exits immediately if not due.
        """
        if not settings.ml.enabled:
            return
        if not hasattr(self.ml_engine, "schedule_training"):
            return
        interval_sec = settings.ml.retrain_interval_hours * 3600
        if time.time() - self._last_ml_retrain >= interval_sec:
            logger.info(
                "[Main] ML retrain due ({:.1f}h since last). Scheduling...",
                (time.time() - self._last_ml_retrain) / 3600,
            )
            self._start_ml_training()

    # ------------------------------------------------------------------
    # Polling interval
    # ------------------------------------------------------------------

    def _get_intervals(self) -> dict:
        elapsed = time.time() - self.start_time.timestamp()
        if elapsed < self.burst_duration:
            remaining = int(self.burst_duration - elapsed)
            if remaining % 60 == 0 and remaining > 0:
                logger.debug("Burst mode: {}s remaining", remaining)
            return self.intervals_burst
        return self.intervals_normal

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        logger.info("Running health checks...")
        results: dict = {}
        for name, collector in self.collectors.items():
            try:
                ok = collector.health_check()
            except Exception as exc:
                logger.error("[{}] Health check exception: {}", name, exc)
                ok = False
            results[name] = ok
            logger.info("  {} {}", "✅" if ok else "❌", name)

        critical_ok = results.get("okx", False)
        if not critical_ok:
            logger.error("OKX health check FAILED — cannot proceed")
        elif not all(results.values()):
            failed = [k for k, v in results.items() if not v]
            logger.warning("Non-critical failures: {}", failed)
        else:
            logger.info("All health checks passed ✅")
        return critical_ok

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_all(self) -> dict:
        """
        Collect data from all sources at their respective intervals.

        v4 FIX: Previously, if the OKX (fast) polling interval hadn't elapsed,
        the entire method returned {} — discarding medium/slow collector data
        that had just been fetched. This created a hard OKX dependency: if OKX
        failed or was simply between polls, the system was blind.

        Now: Each tier runs independently. Accumulated data is always returned
        if ANY collector produced new data this cycle.
        """
        now       = time.time()
        merged    = {}
        intervals = self._get_intervals()
        any_new   = False

        # ── Fast tier: OKX + CoinGlass (every 30s burst / 300s normal) ────
        if now - self.last_poll["fast"] >= intervals["fast"]:
            okx_data = self.collectors["okx"].safe_fetch()
            if okx_data:
                merged.update(okx_data)
                any_new = True
            if self.use_coinglass:
                cg = self.collectors["coinglass"]
                merged.update(cg.safe_fetch())
                merged.update(cg.fetch_cvd())
                merged.update(cg.fetch_basis_spread())
                any_new = True
            self.last_poll["fast"] = now

        # ── Medium tier: Prediction markets (every 120s burst / 900s normal) ──
        if now - self.last_poll["medium"] >= intervals["medium"]:
            merged.update(self.collectors["polymarket"].safe_fetch())
            merged.update(self.collectors["kalshi"].safe_fetch())
            any_new = True
            self.last_poll["medium"] = now

        # ── Slow tier: Macro, sentiment, DeFi, on-chain (every 300s / 3600s) ──
        if now - self.last_poll["slow"] >= intervals["slow"]:
            for name in ("macro", "fear_greed", "binance", "sentiment", "defi", "mempool", "bybit", "cg_plus"):
                if name in self.collectors:
                    merged.update(self.collectors[name].safe_fetch())
            # Venom: shared_coins'i sentiment'ten al, daha iyi veri
            if "venom" in self.collectors:
                shared = merged.get("all_coins") or merged.get("coin_list", [])
                try:
                    venom_data = self.collectors["venom"].fetch(shared_coins=shared if shared else None)
                    merged.update(venom_data)
                except Exception as e:
                    logger.debug("[Venom] fetch error: {}", e)
                    merged.update(self.collectors["venom"].safe_fetch())
            self.event_calendar.refresh()
            any_new = True
            self.last_poll["slow"] = now

        # ── Very slow tier: ETF, CryptoQuant (every 300s / 14400s) ────────
        if now - self.last_poll["very_slow"] >= intervals["very_slow"]:
            for name in ("etf", "cryptoquant"):
                if name in self.collectors:
                    merged.update(self.collectors[name].safe_fetch())
            any_new = True
            self.last_poll["very_slow"] = now

        # No new data from any tier this cycle
        if not any_new:
            return {}

        # Merge into accumulated store
        for k, v in merged.items():
            self._accumulated_data[k] = v
            self._accumulated_timestamps[k] = now

        # 3D: Derive stablecoin/BTC ratio (free CryptoQuant alternative)
        stable_mcap = self._accumulated_data.get('total_stablecoin_mcap', 0)
        total_mcap_t = self._accumulated_data.get('total_mcap_trillion', 0)
        if stable_mcap > 0 and total_mcap_t > 0:
            btc_mcap = total_mcap_t * 1e12  # trillion → raw
            self._accumulated_data['stablecoin_exchange_ratio'] = stable_mcap / btc_mcap

        return self._accumulated_data

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> Optional[dict]:
        raw_data = self.collect_all()
        if not raw_data:
            return None

        # Dashboard market data
        try:
            dashboard_state.update_market_data(raw_data)
            dashboard_state.update_collectors({name: collector.status for name, collector in self.collectors.items()})
            dashboard_state.update_tracker(self.tracker.get_stats())
        except Exception as exc:
            logger.debug("[Main] Dashboard market update: {}", exc)

        # Phase 4: Evaluate pending signal outcomes against current price
        try:
            current_price = raw_data.get('last_price') or raw_data.get('price_now') or raw_data.get('close', 0)
            if current_price and float(current_price) > 0:
                self.tracker.check_outcomes(float(current_price))
        except Exception as exc:
            logger.debug("[Main] Tracker check_outcomes: {}", exc)

        # ── Phase 2: Feed data into unified frame builder ─────────────
        # Candle ingestion (5-min → 1H ring buffer)
        try:
            self.frame_builder.ingest_candle(
                raw_data,
                funding_rate=raw_data.get("funding_rate"),
            )
        except Exception as exc:
            logger.debug("[Main] Frame candle ingest: {}", exc)

        # Macro ingestion (regime features)
        try:
            macro_keys = {"vix_current", "qqq_current", "us10y_current",
                          "gold_current", "fear_greed_value"}
            if any(raw_data.get(k) for k in macro_keys):
                self.frame_builder.ingest_macro(raw_data)
        except Exception as exc:
            logger.debug("[Main] Frame macro ingest: {}", exc)

        # ML prediction (always safe — stub returns disabled dict)
        ml_signal = None
        try:
            ml_signal = self.ml_engine.predict(raw_data)
            dashboard_state.ml_data           = sanitize(ml_signal)
            dashboard_state.ml_dashboard      = sanitize(self.ml_engine.get_dashboard_data())
            dashboard_state.ml_feature_quality = ml_signal.get("ml_feature_quality")
            dashboard_state.ml_ring_buffer_bars= ml_signal.get("ml_ring_buffer_bars", 0)
        except Exception as exc:
            logger.debug("[ML] predict() error: {}", exc)

        # Pulse Engine state → dashboard
        try:
            dashboard_state.pulse = sanitize(self.pulse_engine.current)
        except Exception:
            pass

        # ── Collector features via UnifiedDataStore (still needed for z-scores) ──
        collector_features = None
        try:
            collector_features = self.data_store.update(raw_data, self._accumulated_timestamps)
        except Exception as exc:
            logger.warning("[Main] DataStore update error: {}", exc)

        if collector_features is None or (hasattr(collector_features, "empty") and collector_features.empty):
            return None

        # ── Phase 2: Build unified feature frame ─────────────────────
        try:
            frame = self.frame_builder.build(collector_features, raw_data)
        except Exception as exc:
            logger.warning("[Main] UnifiedFrame build error: {}", exc)
            return None

        # ── Phase 2: Quality gate — circuit breaker (v4) ──────────────
        cb_level = frame.quality.circuit_breaker_level
        if not frame.quality.can_generate_signal:
            reason = frame.quality.abstain_reason
            logger.warning("[Main] ABSTAIN [RED] — quality gate failed: {}", reason)
            dashboard_state.update_data_store(self.data_store.status)
            return {
                "score": 0, "direction": "NEUTRAL", "confidence": "NONE",
                "suppressed_reason": f"quality_gate: {reason}",
                "feature_completeness": frame.quality.usable_fraction,
                "circuit_breaker": "RED",
                "data_maturity": min(frame.ring_buffer_bars / 20.0, 1.0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        if cb_level != "GREEN":
            logger.info("[Main] Circuit breaker [{}] — running with reduced confidence (multiplier={:.2f})",
                        cb_level, frame.quality.confidence_multiplier())

        # ── Phase 3: Run engines directly on unified features ─────────
        features = frame.to_legacy_series()
        session = str(features.get("session_label", "unknown"))

        try:
            micro_out = self.micro_engine.compute(features)
            flow_out  = self.flow_engine.compute(features)
            macro_out = self.macro_engine.compute(features)
        except Exception as exc:
            logger.warning("[Main] Engine compute error: {}", exc)
            return None

        # Inject real-timestamp event status
        try:
            ev = self.event_calendar.get_status()
            if ev.is_pre_event:
                macro_out.event_outcome = "pre_event_blackout"
            dashboard_state.next_events = self.event_calendar.next_events(n=3)
        except Exception as exc:
            logger.debug("[EventCal] Status error: {}", exc)

        # ── Phase 3: Structured decision ──────────────────────────────
        try:
            explanation = self.decision_engine.decide(
                micro_out, flow_out, macro_out,
                quality_gate_passed=frame.quality.can_generate_signal,
                quality_gate_reason=frame.quality.abstain_reason or f"ok [{cb_level}]",
                data_completeness=frame.quality.usable_fraction,
                session=session,
            )
            l1_signal = self.decision_engine.to_legacy_signal(explanation)

            # v4: Apply circuit breaker confidence multiplier
            conf_mult = frame.quality.confidence_multiplier()
            if "score" in l1_signal and isinstance(l1_signal.get("score"), (int, float)):
                l1_signal["score"] = round(l1_signal["score"] * conf_mult, 1)
            l1_signal["circuit_breaker"] = cb_level
            l1_signal["confidence_multiplier"] = round(conf_mult, 3)

            dashboard_state.hybrid_signal = sanitize(explanation.to_dict())
        except Exception as exc:
            logger.warning("[Decision] Error: {}", exc)
            return None

        try:
            dashboard_state.update_signal(l1_signal)
            dashboard_state.update_data_store(self.data_store.status)
        except Exception as exc:
            logger.debug("[Main] Dashboard update: {}", exc)

        # ML signal (still runs separately, paper-trade only)
        if ml_signal:
            l1_signal["ml_probability"] = ml_signal.get("ml_probability")
            l1_signal["ml_recommendation"] = ml_signal.get("ml_recommendation")

        # Warm-up maturity guard
        maturity = min(frame.ring_buffer_bars / 20.0, 1.0)
        if maturity < self._MIN_MATURITY_TO_DELIVER:
            return l1_signal

        # Telegram delivery — only for actionable signals
        if explanation.state.is_actionable:
            try:
                # Inject current price for signal tracker
                l1_signal['entry_price'] = raw_data.get('last_price') or raw_data.get('price_now') or raw_data.get('close', 0)
                self.delivery.send_signal(l1_signal)
                self.tracker.record(l1_signal)
                dashboard_state.update_tracker(self.tracker.get_stats())
            except Exception as exc:
                logger.warning("[Delivery] Error: {}", exc)

        # Phase 5: Smart alerts — fires Telegram on significant conditions
        try:
            self.alert_engine.check(
                raw_data, l1_signal, self.pulse_engine.current,
                send_fn=self.delivery.send_text,
            )
        except Exception as exc:
            logger.debug("[AlertEngine] check error: {}", exc)

        return l1_signal

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, run_once: bool = False) -> None:
        startup_msg = (
            f"🟢 <b>HYDRA v8 online</b>\n"
            f"ML: {'enabled' if settings.ml.enabled else 'disabled'} | "
            f"Collectors: {len(self.collectors)} | "
            f"Pulse: Quad-AI\n"
            f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>"
        )
        try:
            self.delivery.send_text(startup_msg)
        except Exception:
            pass

        last_heartbeat = 0.0

        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as exc:
                    logger.error("[Main] run_cycle exception: {}", exc)

                now = time.time()
                # ML retrain schedule check
                self._check_retrain_schedule()

                if now - last_heartbeat >= self.heartbeat_interval:
                    self._log_heartbeat()
                    last_heartbeat = now

                if run_once:
                    break

                time.sleep(10)

        except KeyboardInterrupt:
            logger.info("Shutting down HYDRA...")
            try:
                self.delivery.send_text("🔴 <b>HYDRA stopped</b> (manual shutdown)")
            except Exception:
                pass

    def _log_heartbeat(self) -> None:
        uptime     = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        is_trained = getattr(self.ml_engine, "is_trained", False)
        quality    = getattr(self.ml_engine, "_last_quality", 0.0)
        thr_in     = getattr(self.ml_engine, "_threshold_in",  None)
        thr_out    = getattr(self.ml_engine, "_threshold_out", None)
        ring_buf   = 0
        try:
            ring_buf = len(self.ml_engine._pipeline._ring)
        except Exception:
            pass

        retrain_in_h = (
            max(0.0, settings.ml.retrain_interval_hours - (time.time() - self._last_ml_retrain) / 3600)
            if settings.ml.enabled and self._last_ml_retrain > 0 else None
        )

        logger.info(
            "Heartbeat | uptime={:.1f}h | ML={} q={:.0%} buf={} thr={}/{} retrain_in={}h | collectors={}",
            uptime,
            "trained" if is_trained else "pending",
            quality,
            ring_buf,
            f"{thr_in:.2f}" if thr_in else "n/a",
            f"{thr_out:.2f}" if thr_out else "n/a",
            f"{retrain_in_h:.1f}" if retrain_in_h is not None else "n/a",
            len(self.collectors),
        )

    # ------------------------------------------------------------------
    # Config bridge — converts typed settings → raw dict for legacy collectors
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_config() -> dict:
        """
        Build a raw dict for legacy collectors that still expect the old
        config.get('api_keys', 'okx', ...) dict format.
        All new code reads from settings directly.
        """
        s = settings
        return {
            "api_keys": {
                "okx": {
                    "api_key":    s.api_keys.okx.api_key,
                    "secret_key": s.api_keys.okx.secret_key,
                    "passphrase": s.api_keys.okx.passphrase,
                },
                "coinglass":   {"api_key": s.api_keys.coinglass_key},
                "cryptoquant": {"api_key": s.api_keys.cryptoquant_key},
                "fred":        {"api_key": s.api_keys.fred_key},
                "perplexity":  {"api_key": s.api_keys.perplexity_key},
                "telegram": {
                    "bot_token": s.api_keys.telegram.bot_token,
                    "chat_id":   s.api_keys.telegram.chat_id,
                },
            },
            "targets": {"primary": {"symbol": "BTC-USDT-SWAP", "exchange": "okx", "bar_interval": "5m"}},
            "layer1": {
                "signal_threshold":  s.layer1.signal_threshold,
                "engine_weights":    s.layer1.engine_weights,
                "confidence_levels": s.layer1.confidence_levels,
                "event_dampening": {
                    "fomc_hours_before": 24, "fomc_dampen_factor": 0.5,
                    "cpi_hours_before":  6,  "cpi_dampen_factor":  0.5,
                    "post_release_boost_hours": 2,
                },
            },
            "ml": {
                "enabled":                s.ml.enabled,
                "retrain_interval_hours": s.ml.retrain_interval_hours,
                "min_samples_for_train":  s.ml.min_samples_for_train,
            },
            "deployment": {
                "log_level":              s.deployment.log_level,
                "data_persist_path":      s.deployment.data_persist_path,
                "heartbeat_interval_sec": s.deployment.heartbeat_interval_sec,
            },
            "collectors": {
                "microstructure": {"polling_interval_sec": 300},
                "flow":           {"polling_interval_sec": 14400},
                "macro":          {"polling_interval_sec": 3600},
            },
            "normalization": {
                "microstructure": {"window_hours": 24,  "method": "zscore"},
                "flow":           {"window_days":  7,   "method": "zscore"},
                "macro":          {"window_days":  30,  "method": "zscore"},
            },
            "sessions": {
                "tokyo":    {"start_utc": 0,  "end_utc": 8},
                "london":   {"start_utc": 8,  "end_utc": 14},
                "new_york": {"start_utc": 14, "end_utc": 21},
                "off_hours":{"start_utc": 21, "end_utc": 24},
            },
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HYDRA Trading Bot v7")
    parser.add_argument("--once",   action="store_true", help="Run one cycle and exit")
    parser.add_argument("--health", action="store_true", help="Health check and exit")
    args = parser.parse_args()

    bot = HydraBot()

    if args.health:
        ok = bot.health_check()
        sys.exit(0 if ok else 1)

    bot.run(run_once=args.once)


if __name__ == "__main__":
    main()
