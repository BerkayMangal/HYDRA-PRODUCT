"""
HYDRA Layer 2 — Signal Delivery

Exports the TelegramDelivery class for use by the main orchestrator.

Layer 2 is intentionally thin: it receives a fully-formed signal dict from
Layer1BiasCombiner and is responsible only for:
  1. Deciding whether to deliver (suppression, duplicate, blackout checks).
  2. Formatting the signal into a human-readable Telegram message.
  3. Delivering reliably (retry, circuit breaker, rate limiting).

It does NOT re-evaluate signal logic. The signal chain ends at Layer 1.
"""

from layer2.telegram_delivery import TelegramDelivery

__all__ = ["TelegramDelivery"]
