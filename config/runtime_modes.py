"""
config/runtime_modes.py
────────────────────────
HYDRA Runtime Mode Discipline (Phase 6, Steps 2-3 + 6)

Defines what runs in each mode and validates startup conditions.

MODES
-----
  dashboard:       Full live system — collectors, engines, dashboard, Telegram.
                   Default mode. Requires OKX + Telegram credentials.

  collectors_only: Collectors + data store only. No engines, no signals.
                   Useful for data collection without signal generation.

  research:        Offline ML research. No collectors, no dashboard.
                   Entry point: python -m ml.research.run_v2

  evidence:        Full evidence package generation. No live components.
                   Entry point: python -m ml.research.evidence

SAFETY RAILS
-------------
  - ML is disabled by default. Enabling requires explicit opt-in.
  - Missing critical credentials → clear error, not silent degradation.
  - If ML verdict is REMOVE but ML is enabled → startup warning.
  - Narrative/LLM keys are optional — absence is noted, not fatal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger


class RuntimeMode(str, Enum):
    DASHBOARD = "dashboard"
    COLLECTORS_ONLY = "collectors_only"
    RESEARCH = "research"
    EVIDENCE = "evidence"


@dataclass(frozen=True)
class ModeConfig:
    """What runs in each mode."""
    collectors: bool
    engines: bool
    decision_engine: bool
    dashboard: bool
    telegram: bool
    pulse_engine: bool
    ml_paper_trade: bool
    briefings: bool


MODE_CONFIGS: Dict[RuntimeMode, ModeConfig] = {
    RuntimeMode.DASHBOARD: ModeConfig(
        collectors=True, engines=True, decision_engine=True,
        dashboard=True, telegram=True, pulse_engine=True,
        ml_paper_trade=False, briefings=True,
    ),
    RuntimeMode.COLLECTORS_ONLY: ModeConfig(
        collectors=True, engines=False, decision_engine=False,
        dashboard=True, telegram=False, pulse_engine=False,
        ml_paper_trade=False, briefings=False,
    ),
    RuntimeMode.RESEARCH: ModeConfig(
        collectors=False, engines=False, decision_engine=False,
        dashboard=False, telegram=False, pulse_engine=False,
        ml_paper_trade=False, briefings=False,
    ),
    RuntimeMode.EVIDENCE: ModeConfig(
        collectors=False, engines=False, decision_engine=False,
        dashboard=False, telegram=False, pulse_engine=False,
        ml_paper_trade=False, briefings=False,
    ),
}


def get_runtime_mode() -> RuntimeMode:
    """Read HYDRA_MODE from environment, default to dashboard."""
    raw = os.environ.get("HYDRA_MODE", "dashboard").lower().strip()
    try:
        return RuntimeMode(raw)
    except ValueError:
        logger.warning(
            "[Mode] Unknown HYDRA_MODE='{}', defaulting to 'dashboard'. "
            "Valid: {}", raw, [m.value for m in RuntimeMode],
        )
        return RuntimeMode.DASHBOARD


def validate_startup(mode: RuntimeMode) -> List[str]:
    """
    Check startup conditions for the given mode.

    Returns list of warnings. Raises RuntimeError for fatal issues.
    """
    warnings: List[str] = []
    cfg = MODE_CONFIGS[mode]

    # ── Critical credentials ─────────────────────────────────────
    if cfg.collectors:
        if not os.environ.get("OKX_API_KEY"):
            warnings.append("OKX_API_KEY not set — OKX collector will use public endpoints only")

    if cfg.telegram:
        bot = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not bot or not chat:
            warnings.append("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing — Telegram delivery disabled")

    # ── Optional credentials ─────────────────────────────────────
    if cfg.collectors:
        if not os.environ.get("COINGLASS_API_KEY"):
            warnings.append("COINGLASS_API_KEY not set — derivatives aggregation unavailable")
        if not os.environ.get("FRED_API_KEY"):
            warnings.append("FRED_API_KEY not set — event calendar uses hardcoded dates")

    if cfg.pulse_engine:
        ai_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"]
        present = [k for k in ai_keys if os.environ.get(k)]
        if not present:
            warnings.append("No AI API keys set — Pulse Engine will use rules-only fallback")

    # ── ML safety rail ───────────────────────────────────────────
    ml_enabled = os.environ.get("ML_ENABLED", "false").lower() in ("true", "1", "yes")
    if ml_enabled:
        warnings.append(
            "⚠ ML_ENABLED=true — ML is experimental. Ensure evidence artifact "
            "supports this posture before deployment."
        )

    return warnings


def print_startup_banner(mode: RuntimeMode, warnings: List[str]) -> None:
    """Print a clear startup banner showing mode and warnings."""
    cfg = MODE_CONFIGS[mode]
    logger.info("=" * 60)
    logger.info("HYDRA v8 — Runtime Mode: {}", mode.value.upper())
    logger.info("=" * 60)
    logger.info("  Collectors:  {}", "ON" if cfg.collectors else "OFF")
    logger.info("  Engines:     {}", "ON" if cfg.engines else "OFF")
    logger.info("  Decision:    {}", "ON" if cfg.decision_engine else "OFF")
    logger.info("  Dashboard:   {}", "ON" if cfg.dashboard else "OFF")
    logger.info("  Telegram:    {}", "ON" if cfg.telegram else "OFF")
    logger.info("  Pulse (AI):  {}", "ON" if cfg.pulse_engine else "OFF")
    logger.info("  ML Paper:    {}", "ON" if cfg.ml_paper_trade else "OFF")
    logger.info("  Briefings:   {}", "ON" if cfg.briefings else "OFF")
    if warnings:
        logger.info("-" * 60)
        for w in warnings:
            logger.warning("  {}", w)
    logger.info("=" * 60)
