"""Explicit collection budgets, shared by both public-data workers."""

from datetime import datetime, timezone
import math


def next_poll_deadline(started, finished, interval):
    """Schedule from cycle start; skip missed slots without catch-up requests."""
    if not all(math.isfinite(value) for value in (started, finished, interval)):
        raise ValueError("Finite monotonic times and interval required")
    if interval <= 0 or finished < started:
        raise ValueError("Positive interval and ordered monotonic times required")
    slots = max(1, math.ceil((finished - started) / interval))
    return started + slots * interval


def validate_budget(once, until):
    if once and until:
        raise ValueError("Use either --once or --until")
    if once:
        return None
    if not until:
        raise ValueError("A bounded --until timestamp is required unless using --once")
    deadline = datetime.fromisoformat(until)
    if deadline.tzinfo is None:
        raise ValueError("Stop timestamp must include a UTC offset")
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if not 0 < remaining <= 86400:
        raise ValueError("Stop timestamp must be in the next 24 hours")
    return deadline
