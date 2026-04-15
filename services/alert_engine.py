"""
services/alert_engine.py — HYDRA Smart Alert Engine
════════════════════════════════════════════════════
Monitors market data, signals, and pulse every cycle.
Fires Telegram alerts when significant conditions are met.
4-hour cooldown per alert type prevents spam.

RULES:
  1. RegimeChange     — NEUTRAL → BULLISH/BEARISH
  2. WhaleFlow        — ETF net flow $500M+ in a single day
  3. VIXSpike         — VIX 15%+ daily increase
  4. LiqCascade       — $100M+ liquidations in recent window
  5. AIDisagreement   — 3/4 AI models disagree with consensus
  6. FearGreedExtreme — Fear & Greed < 15 (extreme fear) or > 85 (extreme greed)
"""

import time
from typing import Dict, List, Optional, Any
from loguru import logger


class AlertRule:
    """Base class for alert rules."""
    name: str = "BaseRule"
    emoji: str = "🔔"

    def check(self, market_data: Dict, signal: Dict, pulse: Dict) -> Optional[str]:
        """Return alert message string if triggered, else None."""
        raise NotImplementedError


class RegimeChangeAlert(AlertRule):
    """Fires when regime changes from NEUTRAL to a directional signal."""
    name = "regime_change"
    emoji = "🔄"

    def __init__(self):
        self._last_direction: str = "NEUTRAL"

    def check(self, market_data: Dict, signal: Dict, pulse: Dict) -> Optional[str]:
        current = signal.get('direction', 'NEUTRAL')
        if current == self._last_direction:
            return None
        prev = self._last_direction
        self._last_direction = current
        if prev in ('NEUTRAL', 'NO_SIGNAL') and current in ('BULLISH', 'BEARISH'):
            score = signal.get('score', 0)
            conf = signal.get('confidence', '?')
            return (
                f"{self.emoji} <b>Regime Change: {prev} → {current}</b>\n"
                f"Score: {score} | Confidence: {conf}"
            )
        if prev in ('BULLISH', 'BEARISH') and current in ('BULLISH', 'BEARISH') and prev != current:
            return (
                f"{self.emoji} <b>Regime Flip: {prev} → {current}</b>\n"
                f"Score: {signal.get('score', 0)}"
            )
        return None


class WhaleFlowAlert(AlertRule):
    """Fires when ETF net flow exceeds $500M in a single observation."""
    name = "whale_flow"
    emoji = "🐋"

    THRESHOLD_USD = 500_000_000

    def check(self, market_data: Dict, signal: Dict, pulse: Dict) -> Optional[str]:
        flow = market_data.get('etf_net_flow_usd') or market_data.get('btc_etf_net_flow', 0)
        try:
            flow = float(flow)
        except (TypeError, ValueError):
            return None
        if abs(flow) >= self.THRESHOLD_USD:
            direction = "inflow" if flow > 0 else "outflow"
            return (
                f"{self.emoji} <b>Whale ETF {direction.upper()}: ${abs(flow)/1e6:,.0f}M</b>\n"
                f"Significant institutional {'buying' if flow > 0 else 'selling'} pressure"
            )
        return None


class VIXSpikeAlert(AlertRule):
    """Fires when VIX increases 15%+ in 24h."""
    name = "vix_spike"
    emoji = "⚡"

    THRESHOLD_PCT = 15.0

    def check(self, market_data: Dict, signal: Dict, pulse: Dict) -> Optional[str]:
        chg = market_data.get('vix_change_24h')
        current = market_data.get('vix_current')
        if chg is None:
            return None
        try:
            chg = float(chg)
            current = float(current) if current else 0
        except (TypeError, ValueError):
            return None
        if chg >= self.THRESHOLD_PCT:
            return (
                f"{self.emoji} <b>VIX Spike: +{chg:.1f}%</b> (now {current:.1f})\n"
                f"Elevated volatility — risk-off environment"
            )
        return None


class LiqCascadeAlert(AlertRule):
    """Fires when total liquidations exceed $100M."""
    name = "liq_cascade"
    emoji = "💥"

    THRESHOLD_USD = 100_000_000

    def check(self, market_data: Dict, signal: Dict, pulse: Dict) -> Optional[str]:
        liq_total = market_data.get('liq_total', 0)
        try:
            liq_total = float(liq_total)
        except (TypeError, ValueError):
            return None
        if liq_total >= self.THRESHOLD_USD:
            long_vol = float(market_data.get('liq_long_vol', 0))
            short_vol = float(market_data.get('liq_short_vol', 0))
            dominant = "LONG" if long_vol > short_vol else "SHORT"
            return (
                f"{self.emoji} <b>Liquidation Cascade: ${liq_total/1e6:,.0f}M</b>\n"
                f"Dominant: {dominant} liquidations\n"
                f"Longs: ${long_vol/1e6:,.0f}M | Shorts: ${short_vol/1e6:,.0f}M"
            )
        return None


class AIDisagreementAlert(AlertRule):
    """Fires when 3+ of the 4 AI models disagree with the consensus."""
    name = "ai_disagreement"
    emoji = "🤖"

    def check(self, market_data: Dict, signal: Dict, pulse: Dict) -> Optional[str]:
        analyses = pulse.get('analyses') or pulse.get('model_analyses', {})
        if not analyses or not isinstance(analyses, dict):
            return None

        consensus = pulse.get('consensus', {})
        consensus_dir = consensus.get('direction', '').upper() if isinstance(consensus, dict) else ''
        if not consensus_dir:
            return None

        disagree_count = 0
        model_opinions = []
        for model_name, analysis in analyses.items():
            if isinstance(analysis, dict):
                model_dir = (analysis.get('direction') or analysis.get('sentiment', '')).upper()
                if model_dir and model_dir != consensus_dir:
                    disagree_count += 1
                model_opinions.append(f"{model_name}: {model_dir or '?'}")

        if disagree_count >= 3 and len(analyses) >= 4:
            return (
                f"{self.emoji} <b>AI Disagreement Alert</b>\n"
                f"Consensus: {consensus_dir} but {disagree_count}/{len(analyses)} models disagree\n"
                f"{' | '.join(model_opinions)}"
            )
        return None


class FearGreedExtremeAlert(AlertRule):
    """Fires when Fear & Greed index hits extreme levels."""
    name = "fear_greed_extreme"
    emoji = "😱"

    FEAR_THRESHOLD = 15
    GREED_THRESHOLD = 85

    def check(self, market_data: Dict, signal: Dict, pulse: Dict) -> Optional[str]:
        fg = market_data.get('fear_greed_value')
        if fg is None:
            return None
        try:
            fg = float(fg)
        except (TypeError, ValueError):
            return None
        if fg <= self.FEAR_THRESHOLD:
            return (
                f"{self.emoji} <b>EXTREME FEAR: F&G = {fg:.0f}</b>\n"
                f"Historically a contrarian buy zone — exercise caution"
            )
        if fg >= self.GREED_THRESHOLD:
            return (
                f"🤑 <b>EXTREME GREED: F&G = {fg:.0f}</b>\n"
                f"Market euphoria — watch for reversals"
            )
        return None


# ---------------------------------------------------------------------------
# Alert Engine — orchestrator
# ---------------------------------------------------------------------------

class AlertEngine:
    """
    Runs all alert rules each cycle. Enforces per-rule cooldown
    to prevent spam (default: 4 hours between same alert type).
    """

    COOLDOWN_SECONDS = 14400  # 4 hours

    def __init__(self):
        self.rules: List[AlertRule] = [
            RegimeChangeAlert(),
            WhaleFlowAlert(),
            VIXSpikeAlert(),
            LiqCascadeAlert(),
            AIDisagreementAlert(),
            FearGreedExtremeAlert(),
        ]
        # {rule_name: last_fired_timestamp}
        self._cooldowns: Dict[str, float] = {}

    def check(
        self,
        market_data: Dict[str, Any],
        signal: Dict[str, Any],
        pulse: Dict[str, Any],
        send_fn=None,
    ) -> List[str]:
        """
        Run all rules. Returns list of triggered alert messages.
        If send_fn is provided (e.g. TelegramDelivery.send_text),
        alerts are sent immediately.
        """
        now = time.time()
        triggered = []

        for rule in self.rules:
            # Cooldown check
            last_fired = self._cooldowns.get(rule.name, 0)
            if now - last_fired < self.COOLDOWN_SECONDS:
                continue

            try:
                msg = rule.check(market_data, signal or {}, pulse or {})
            except Exception as exc:
                logger.debug("[AlertEngine] Rule {} error: {}", rule.name, exc)
                continue

            if msg:
                self._cooldowns[rule.name] = now
                triggered.append(msg)
                logger.info("[AlertEngine] Triggered: {}", rule.name)

                if send_fn:
                    try:
                        full_msg = f"🚨 <b>HYDRA ALERT</b>\n\n{msg}"
                        send_fn(full_msg)
                    except Exception as exc:
                        logger.warning("[AlertEngine] Send failed: {}", exc)

        return triggered
