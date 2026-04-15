"""
HYDRA Layer 2 — Telegram Delivery Pipeline

Responsibilities:
  - Format Layer 1 signal dicts into structured, readable Telegram messages.
  - Deliver messages non-blocking (runs HTTP call in a daemon thread so the
    main polling loop is never stalled by network latency or Telegram downtime).
  - Enforce Telegram's rate limit: max 1 message/second to the same chat.
  - Circuit breaker: after 5 consecutive delivery failures, pause for 5 minutes
    before retrying. Prevents log-spam and redundant API hammering.
  - Duplicate suppression: do not re-deliver if the same direction with a
    similar score was sent within the last 30 minutes. Prevents alert fatigue
    during choppy markets where the combiner oscillates near the threshold.
  - Pre-event blackout pass-through: signals suppressed by the combiner
    (suppressed_reason = "pre_event_blackout") are never forwarded here.
  - No mock data. No hardcoded placeholder messages. If formatting fails,
    the exception is logged and delivery is skipped — silence is safer than
    a garbled message.

Telegram Bot API:
  Endpoint : https://api.telegram.org/bot{token}/sendMessage
  Auth     : token in URL path (Bearer in path, no header needed)
  Format   : HTML parse_mode for bold/mono formatting
  Rate     : 30 msg/s global, 1 msg/s to same chat_id (we stay well below this)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Telegram Bot API base URL. Token is injected at call time, never logged.
_TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"

# Minimum gap between messages to the same chat_id.
# Telegram hard limit is 1 msg/s; we use 1.5s for safety margin.
_RATE_LIMIT_SECONDS: float = 1.5

# Circuit breaker: open after this many consecutive delivery failures.
_CIRCUIT_OPEN_AFTER: int = 5

# Circuit breaker: seconds to wait before attempting a probe after opening.
_CIRCUIT_COOLDOWN_SECONDS: float = 300.0  # 5 minutes

# Duplicate suppression window. If the same direction + similar score was
# already delivered within this window, skip re-delivery.
_DUPLICATE_WINDOW_SECONDS: float = 1800.0  # 30 minutes

# Score must change by at least this much from the last delivered signal to
# break out of duplicate suppression (even within the time window).
_DUPLICATE_SCORE_DELTA: float = 15.0

# HTTP request timeout for Telegram API calls.
_REQUEST_TIMEOUT_SECONDS: int = 10

# Maximum retry attempts per message before giving up.
_MAX_RETRIES: int = 3

# Base delay for exponential backoff between retries (seconds).
_RETRY_BASE_DELAY: float = 2.0

# Minimum feature completeness to deliver a signal (mirrors combiner gate).
# Belt-and-suspenders: combiner already blocks these, but delivery layer
# applies the same check independently in case of future refactors.
_MIN_COMPLETENESS_TO_DELIVER: float = 0.50

# Direction emojis for the Telegram message header.
_DIRECTION_EMOJI: Dict[str, str] = {
    "LONG":    "??",
    "SHORT":   "??",
    "NEUTRAL": "⚪",
}

# Confidence tier emojis.
_CONFIDENCE_EMOJI: Dict[str, str] = {
    "EXTREME": "??",
    "HIGH":    "⚡",
    "MEDIUM":  "??",
    "LOW":     "??",
    "NONE":    "—",
}

# Agreement state labels.
_AGREEMENT_LABEL: Dict[str, str] = {
    "ALIGNED":    "✅ All engines agree",
    "CONFLICT":   "⚠️ Engines conflict",
    "WEAK":       "?? Weak signal",
    "MIXED":      "〰️ Mixed",
    "SUPPRESSED": "?? Suppressed",
}

# Human-readable event outcome labels.
_EVENT_OUTCOME_LABEL: Dict[str, str] = {
    "no_event":               "—",
    "pre_event_blackout":     "?? Pre-event blackout",
    "fomc_hawkish_confirmed": "?? FOMC Hawkish ↑",
    "fomc_dovish_confirmed":  "?? FOMC Dovish ↓",
    "fomc_mildly_hawkish":    "?? FOMC Mildly Hawkish",
    "fomc_mildly_dovish":     "?? FOMC Mildly Dovish",
    "fomc_ambiguous":         "?? FOMC Ambiguous",
    "cpi_hawkish_confirmed":  "?? CPI Hot ↑",
    "cpi_dovish_confirmed":   "?? CPI Cool ↓",
    "cpi_mildly_hawkish":     "?? CPI Mildly Hot",
    "cpi_mildly_dovish":      "?? CPI Mildly Cool",
    "cpi_ambiguous":          "?? CPI Ambiguous",
}


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

@dataclass
class _CircuitBreaker:
    """
    Simple three-state circuit breaker for the Telegram API connection.

    States:
      CLOSED  — normal operation, requests pass through.
      OPEN    — too many failures, requests are blocked until cooldown expires.
      PROBING — cooldown expired, one probe request is allowed through.
    """
    failure_count: int     = 0
    opened_at:     float   = 0.0
    state:         str     = "CLOSED"   # CLOSED | OPEN | PROBING

    def record_success(self) -> None:
        """Reset failure count and close the circuit."""
        self.failure_count = 0
        self.state         = "CLOSED"

    def record_failure(self) -> None:
        """Increment failure count; open the circuit if threshold is reached."""
        self.failure_count += 1
        if self.failure_count >= _CIRCUIT_OPEN_AFTER:
            if self.state != "OPEN":
                self.state     = "OPEN"
                self.opened_at = time.time()
                logger.error(
                    "[Telegram] Circuit breaker OPENED after {} consecutive failures. "
                    "Pausing delivery for {}s.",
                    self.failure_count,
                    _CIRCUIT_COOLDOWN_SECONDS,
                )

    def allow_request(self) -> bool:
        """
        Return True if a request should be allowed through.

        CLOSED   -> always allow.
        OPEN     -> allow only after cooldown expires (transition to PROBING).
        PROBING  -> allow one probe; outcome determines next state.
        """
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            elapsed = time.time() - self.opened_at
            if elapsed >= _CIRCUIT_COOLDOWN_SECONDS:
                self.state = "PROBING"
                logger.info(
                    "[Telegram] Circuit breaker probing after {}s cooldown.",
                    int(elapsed),
                )
                return True
            return False

        if self.state == "PROBING":
            # Allow the single probe — outcome is recorded via record_success/failure.
            return True

        return False


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

@dataclass
class _RateLimiter:
    """
    Token-bucket rate limiter enforcing a minimum gap between sends.

    Telegram allows 1 message/second to the same chat_id. We use 1.5s
    as a safety margin. The limiter blocks the delivery thread (not the
    main thread) if a message arrives too soon after the previous one.
    """
    _last_send: float = field(default_factory=float)

    def __post_init__(self) -> None:
        self._last_send = 0.0

    def wait_if_needed(self) -> None:
        """Sleep in the delivery thread if the rate limit gap has not elapsed."""
        elapsed = time.time() - self._last_send
        if elapsed < _RATE_LIMIT_SECONDS:
            wait = _RATE_LIMIT_SECONDS - elapsed
            logger.debug("[Telegram] Rate limit: waiting {:.2f}s", wait)
            time.sleep(wait)
        self._last_send = time.time()


# ---------------------------------------------------------------------------
# Delivered Signal Record (for duplicate suppression)
# ---------------------------------------------------------------------------

@dataclass
class _DeliveredRecord:
    """Snapshot of the last delivered non-neutral signal."""
    direction: str
    score:     float
    sent_at:   float   # unix epoch


# ---------------------------------------------------------------------------
# Main Delivery Class
# ---------------------------------------------------------------------------

class TelegramDelivery:
    """
    Non-blocking Telegram delivery pipeline for HYDRA signals.

    Usage:
        delivery = TelegramDelivery(config)
        delivery.send_signal(signal_dict)   # returns immediately

    The actual HTTP call runs in a daemon thread. The main polling loop
    is never blocked by Telegram latency or outages.
    """

    def __init__(self, config: Dict) -> None:
        tg_cfg = config.get("api_keys", {}).get("telegram", {})

        self._token:   str = tg_cfg.get("bot_token", "")
        self._chat_id: str = str(tg_cfg.get("chat_id", ""))

        self._enabled: bool = bool(self._token and self._chat_id)

        if not self._enabled:
            logger.warning(
                "[Telegram] bot_token or chat_id not configured — "
                "delivery disabled. Set api_keys.telegram in config.yaml."
            )
        else:
            logger.info("[Telegram] Delivery pipeline initialized (chat_id={})", self._chat_id)

        self._circuit:      _CircuitBreaker  = _CircuitBreaker()
        self._rate_limiter: _RateLimiter     = _RateLimiter()

        # Last successfully delivered directional signal, for duplicate suppression.
        self._last_delivered: Optional[_DeliveredRecord] = None

        # Counts for observability.
        self._sent_count:      int = 0
        self._suppressed_count: int = 0
        self._failed_count:    int = 0

        # Lock protecting _last_delivered and counters from concurrent threads.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_signal(self, signal: Dict) -> None:
        """
        Evaluate a signal and deliver it asynchronously if appropriate.

        This method returns immediately. Delivery happens in a daemon thread
        so the main polling loop is never stalled.

        Args:
            signal: The dict returned by Layer1BiasCombiner.generate_signal().
        """
        if not self._enabled:
            return

        # --- Pre-flight checks (synchronous, fast) ---
        direction = signal.get("direction", "NEUTRAL")
        score     = float(signal.get("score", 0.0))

        # Never deliver NEUTRAL or explicitly suppressed signals.
        if direction == "NEUTRAL":
            return

        suppressed_reason = signal.get("suppressed_reason")
        if suppressed_reason:
            logger.debug(
                "[Telegram] Skipping suppressed signal (reason={})", suppressed_reason
            )
            return

        # Never deliver when completeness gate would have blocked this.
        completeness = float(signal.get("feature_completeness", 1.0))
        if completeness < _MIN_COMPLETENESS_TO_DELIVER:
            logger.warning(
                "[Telegram] Completeness {:.0%} below threshold — not delivering.",
                completeness,
            )
            return

        # Duplicate suppression check.
        if self._is_duplicate(direction, score):
            with self._lock:
                self._suppressed_count += 1
            logger.debug(
                "[Telegram] Duplicate suppressed: {} score={:+.1f}", direction, score
            )
            return

        # Circuit breaker check (fast path — no lock needed, state is a string).
        if not self._circuit.allow_request():
            logger.warning(
                "[Telegram] Circuit OPEN — signal not delivered "
                "(direction={}, score={:+.1f})", direction, score
            )
            return

        # --- Format message (synchronous, may raise — catch before threading) ---
        try:
            message = self._format_signal(signal)
        except Exception as exc:
            logger.error("[Telegram] Message formatting failed: {}", exc, exc_info=True)
            return

        # --- Deliver in background thread ---
        thread = threading.Thread(
            target=self._deliver_thread,
            args=(message, direction, score),
            daemon=True,
            name=f"tg-deliver-{int(time.time())}",
        )
        thread.start()

    def send_text(self, text: str) -> None:
        """
        Send a plain-text message (for heartbeats, errors, status updates).

        Also runs in a background thread. No duplicate suppression or
        completeness check — caller is responsible for content.
        """
        if not self._enabled:
            return

        if not self._circuit.allow_request():
            return

        thread = threading.Thread(
            target=self._deliver_thread,
            args=(text, None, 0.0),
            daemon=True,
            name=f"tg-text-{int(time.time())}",
        )
        thread.start()

    @property
    def stats(self) -> Dict:
        """Delivery statistics for heartbeat logging and dashboard."""
        with self._lock:
            return {
                "sent":       self._sent_count,
                "suppressed": self._suppressed_count,
                "failed":     self._failed_count,
                "circuit":    self._circuit.state,
            }

    # ------------------------------------------------------------------
    # Duplicate suppression
    # ------------------------------------------------------------------

    def _is_duplicate(self, direction: str, score: float) -> bool:
        """
        Return True if this signal is too similar to the last delivered one.

        Duplicate criteria (both must hold):
          1. Same direction as last delivered signal.
          2. Signal was delivered within _DUPLICATE_WINDOW_SECONDS.
          3. Score has not changed by more than _DUPLICATE_SCORE_DELTA.

        If the score has shifted significantly (large new move), break through
        suppression even within the time window.
        """
        with self._lock:
            last = self._last_delivered

        if last is None:
            return False

        age = time.time() - last.sent_at
        if age > _DUPLICATE_WINDOW_SECONDS:
            return False   # Window expired — always allow

        if last.direction != direction:
            return False   # Direction flipped — always deliver (reversal signal)

        score_delta = abs(score - last.score)
        if score_delta >= _DUPLICATE_SCORE_DELTA:
            return False   # Significant conviction change — deliver

        return True   # Same direction, similar score, within window — suppress

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def _format_signal(self, signal: Dict) -> str:
        """
        Build a structured HTML-formatted Telegram message from a signal dict.

        Layout:
          [emoji] DIRECTION  |  Score: +XX.X  |  Confidence: TIER [emoji]
          ─────────────────────────────────────
          Agreement : ALIGNED / CONFLICT / WEAK
          Regime    : low_vol / normal / high_vol / crisis
          Session   : london / new_york / tokyo / off_hours
          Event     : FOMC Dovish Confirmed / — / etc.
          ─────────────────────────────────────
          Top Signals:
            • engine.signal_name  :  +XX.X
            • …
          ─────────────────────────────────────
          Completeness : 92%   |  Maturity : 100%
          2025-03-28 14:35:02 UTC
        """
        direction    = signal.get("direction",            "NEUTRAL")
        score        = float(signal.get("score",          0.0))
        confidence   = signal.get("confidence",           "NONE")
        agreement    = signal.get("agreement",            "WEAK")
        regime       = signal.get("regime",               "unknown")
        session      = signal.get("session",              "unknown")
        event_out    = signal.get("event_outcome",        "no_event")
        completeness = float(signal.get("feature_completeness", 1.0))
        maturity     = float(signal.get("data_maturity",        1.0))
        timestamp    = signal.get("timestamp",            "")
        top_contrib  = signal.get("top_contributors",     [])

        dir_emoji    = _DIRECTION_EMOJI.get(direction,   "❓")
        conf_emoji   = _CONFIDENCE_EMOJI.get(confidence, "—")
        agree_label  = _AGREEMENT_LABEL.get(agreement,   agreement)
        event_label  = _EVENT_OUTCOME_LABEL.get(event_out, event_out)

        score_str = f"{score:+.1f}"

        # --- Format timestamp ---
        ts_display = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                ts_display = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                ts_display = str(timestamp)

        # --- Top contributors block ---
        contrib_lines: List[str] = []
        for item in top_contrib[:5]:
            feat  = item.get("feature", "?")
            val   = float(item.get("value", 0.0))
            arrow = "▲" if val > 0 else "▼"
            contrib_lines.append(
                f"  {arrow} <code>{feat:<35}</code>  {val:+.1f}"
            )
        contrib_block = "\n".join(contrib_lines) if contrib_lines else "  (no contributor data)"

        # --- Engine sub-scores (Phase 6 fix: handles both flat and nested) ---
        engines     = signal.get("engines", {})
        def _eng_score(key, *aliases):
            for k in (key, *aliases):
                v = engines.get(k)
                if v is None: continue
                if isinstance(v, (int, float)): return float(v)
                if isinstance(v, dict): return float(v.get("score", 0.0))
            return 0.0
        micro_score = _eng_score("micro", "microstructure")
        flow_score  = _eng_score("flow")
        macro_score = _eng_score("macro")

        engine_block = (
            f"  Micro : <code>{micro_score:+.1f}</code>  "
            f"Flow : <code>{flow_score:+.1f}</code>  "
            f"Macro : <code>{macro_score:+.1f}</code>"
        )

        # --- Assemble final message ---
        # HTML parse_mode: <b>, <code>, <i> are supported by Telegram.
        sep = "─" * 38

        message = (
            f"{dir_emoji}  <b>{direction}</b>"
            f"    Score: <b><code>{score_str}</code></b>"
            f"    {conf_emoji} <b>{confidence}</b>\n"
            f"{sep}\n"
            f"<b>Agreement :</b> {agree_label}\n"
            f"<b>Regime    :</b> {regime.replace('_', ' ').title()}\n"
            f"<b>Session   :</b> {session.replace('_', ' ').title()}\n"
            f"<b>Event     :</b> {event_label}\n"
            f"{sep}\n"
            f"<b>Engine Scores</b>\n"
            f"{engine_block}\n"
            f"{sep}\n"
            f"<b>Top Contributors</b>\n"
            f"{contrib_block}\n"
            f"{sep}\n"
            f"Completeness : <code>{completeness:.0%}</code>    "
            f"Maturity : <code>{maturity:.0%}</code>\n"
            f"<i>{ts_display}</i>"
        )

        return message

    # ------------------------------------------------------------------
    # HTTP delivery (runs in background thread)
    # ------------------------------------------------------------------

    def _deliver_thread(
        self,
        message: str,
        direction: Optional[str],
        score: float,
    ) -> None:
        """
        Thread target: apply rate limiting, then attempt delivery with retries.

        On success:
          - Record the delivery in _last_delivered (for duplicate suppression).
          - Notify circuit breaker of success.
          - Increment sent counter.

        On all-retry failure:
          - Notify circuit breaker of failure.
          - Increment failed counter.
          - Signal is silently dropped (caller has already returned).
        """
        # Rate limit: enforce gap INSIDE the thread so main loop is unblocked.
        self._rate_limiter.wait_if_needed()

        success = self._send_with_retry(message)

        with self._lock:
            if success:
                self._circuit.record_success()
                self._sent_count += 1
                if direction and direction != "NEUTRAL":
                    self._last_delivered = _DeliveredRecord(
                        direction=direction,
                        score=score,
                        sent_at=time.time(),
                    )
                logger.info(
                    "[Telegram] Signal delivered (direction={}, score={:+.1f}) | "
                    "Total sent: {}",
                    direction, score, self._sent_count,
                )
            else:
                self._circuit.record_failure()
                self._failed_count += 1
                logger.error(
                    "[Telegram] Delivery failed after {} retries "
                    "(direction={}, score={:+.1f}) | Circuit: {}",
                    _MAX_RETRIES, direction, score, self._circuit.state,
                )

    def _send_with_retry(self, message: str) -> bool:
        """
        Attempt to deliver a message to Telegram with exponential backoff.

        Returns True on success, False when all retries are exhausted.
        The bot token is never logged.
        """
        url = _TELEGRAM_API_BASE.format(token=self._token)

        payload = {
            "chat_id":    self._chat_id,
            "text":       message,
            "parse_mode": "HTML",
            # Disable link preview for cleaner signal messages.
            "disable_web_page_preview": True,
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )

                if resp.status_code == 200:
                    return True

                # 429 = Too Many Requests — respect Retry-After header.
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(
                        "[Telegram] Rate limited (429) — waiting {}s (attempt {}/{})",
                        retry_after, attempt, _MAX_RETRIES,
                    )
                    time.sleep(retry_after)
                    continue

                # 4xx (not 429) — client errors, no point retrying.
                if 400 <= resp.status_code < 500:
                    logger.error(
                        "[Telegram] Client error {} — not retrying. "
                        "Response: {}",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return False

                # 5xx — server error, retry with backoff.
                logger.warning(
                    "[Telegram] Server error {} on attempt {}/{} — retrying.",
                    resp.status_code, attempt, _MAX_RETRIES,
                )

            except requests.exceptions.Timeout:
                logger.warning(
                    "[Telegram] Request timed out (attempt {}/{})", attempt, _MAX_RETRIES
                )
            except requests.exceptions.ConnectionError as exc:
                logger.warning(
                    "[Telegram] Connection error on attempt {}/{}: {}",
                    attempt, _MAX_RETRIES, exc,
                )
            except Exception as exc:
                logger.error(
                    "[Telegram] Unexpected error on attempt {}/{}: {}",
                    attempt, _MAX_RETRIES, exc, exc_info=True,
                )

            # Exponential backoff before next attempt.
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 2s, 4s, 8s
                logger.debug("[Telegram] Retrying in {:.1f}s...", delay)
                time.sleep(delay)

        return False
