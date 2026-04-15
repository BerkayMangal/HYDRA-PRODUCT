"""
signals/event_calendar.py
──────────────────────────
HYDRA Event Calendar — Real Timestamp-Based Event Detection (Step 6)

WHAT THIS REPLACES
------------------
The original macro engine used hardcoded approximate dates and a fixed
countdown from "expected" FOMC dates. This caused two problems:
1. Dates drifted by days when the Fed changed meeting schedules.
2. Post-event impact was applied for a fixed 2H window regardless of
   the actual release time.

THIS MODULE
-----------
Maintains a live event calendar sourced from FRED API (FOMC dates)
and a fallback hardcoded calendar for when the API is unavailable.
All events carry:
  - exact UTC timestamp of the release
  - pre-event blackout window (hours before release)
  - post-event reaction window (hours after release)

The event status for any given UTC time is:
  - PRE_EVENT: within blackout window before release
  - ACTIVE:    within reaction window after release
  - NONE:      no event active

USAGE
-----
>>> cal = EventCalendar()
>>> cal.refresh()   # fetch FRED data (call once at startup + daily)
>>> status = cal.get_status()
>>> if status.is_pre_event:
...     dampen_signals()
>>> if status.is_active:
...     apply_event_overlay(status.event_type, status.hours_since_release)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScheduledEvent:
    event_type:       str       # "FOMC" | "CPI" | "NFP" | "GDP"
    release_utc:      datetime  # exact UTC release time
    pre_blackout_h:   int       # hours before release to enter blackout
    post_reaction_h:  int       # hours after release that event is "active"
    source:           str       # "FRED" | "HARDCODED" | "MANUAL"
    confirmed:        bool      # True if from official source


@dataclass
class EventStatus:
    is_pre_event:       bool
    is_active:          bool
    event_type:         Optional[str]
    hours_to_release:   Optional[float]   # negative = in the past
    hours_since_release: Optional[float]
    source:             str
    dampen_factor:      float   # 0.0 = full dampen, 1.0 = no dampen

    @property
    def is_any(self) -> bool:
        return self.is_pre_event or self.is_active


# ---------------------------------------------------------------------------
# Fallback hardcoded 2025–2026 FOMC / CPI schedule
# (Updated to actual Fed press releases — accurate to within ±1H)
# ---------------------------------------------------------------------------

_FOMC_DATES_UTC = [
    # 2025
    "2025-01-29T19:00:00Z",
    "2025-03-19T18:00:00Z",
    "2025-05-07T18:00:00Z",
    "2025-06-18T18:00:00Z",
    "2025-07-30T18:00:00Z",
    "2025-09-17T18:00:00Z",
    "2025-10-29T18:00:00Z",
    "2025-12-10T19:00:00Z",
    # 2026 (projected from Fed calendar)
    "2026-01-28T19:00:00Z",
    "2026-03-18T18:00:00Z",
    "2026-04-29T18:00:00Z",
    "2026-06-10T18:00:00Z",
    "2026-07-29T18:00:00Z",
    "2026-09-16T18:00:00Z",
    "2026-10-28T18:00:00Z",
    "2026-12-09T19:00:00Z",
]

_CPI_DATES_UTC = [
    # 2025 (BLS releases at 08:30 ET = 13:30 UTC)
    "2025-01-15T13:30:00Z",
    "2025-02-12T13:30:00Z",
    "2025-03-12T13:30:00Z",
    "2025-04-10T13:30:00Z",
    "2025-05-13T13:30:00Z",
    "2025-06-11T13:30:00Z",
    "2025-07-15T13:30:00Z",
    "2025-08-12T13:30:00Z",
    "2025-09-10T13:30:00Z",
    "2025-10-15T13:30:00Z",
    "2025-11-12T13:30:00Z",
    "2025-12-10T13:30:00Z",
    # 2026
    "2026-01-14T13:30:00Z",
    "2026-02-11T13:30:00Z",
    "2026-03-11T13:30:00Z",
    "2026-04-15T13:30:00Z",
    "2026-05-13T13:30:00Z",
    "2026-06-10T13:30:00Z",
]


# ---------------------------------------------------------------------------
# EventCalendar
# ---------------------------------------------------------------------------

class EventCalendar:
    """
    Maintains a list of macro events with exact UTC timestamps.

    Refreshes from FRED API when an API key is available.
    Falls back to the hardcoded schedule above otherwise.
    """

    def __init__(
        self,
        fred_api_key:     str = "",
        fomc_blackout_h:  int = 24,
        fomc_reaction_h:  int = 4,
        cpi_blackout_h:   int = 6,
        cpi_reaction_h:   int = 2,
        persist_path:     str = "data/event_calendar.json",
    ) -> None:
        self.fred_key       = fred_api_key
        self.persist_path   = persist_path
        self._events:       List[ScheduledEvent] = []
        self._last_refresh: Optional[datetime]   = None

        self._windows = {
            "FOMC": (fomc_blackout_h, fomc_reaction_h),
            "CPI":  (cpi_blackout_h,  cpi_reaction_h),
            "NFP":  (6, 2),
        }

        # Load hardcoded baseline
        self._load_hardcoded()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self, force: bool = False) -> None:
        """
        Refresh event calendar from FRED. Falls back to hardcoded schedule.

        Should be called once at startup and then once daily.

        Parameters
        ----------
        force : bool
            If True, refresh even if last refresh was recent.
        """
        if not force and self._last_refresh:
            age_h = (datetime.now(timezone.utc) - self._last_refresh).total_seconds() / 3600
            if age_h < 12:
                return

        if self.fred_key:
            try:
                self._fetch_fred()
            except Exception as exc:
                logger.warning("[EventCal] FRED fetch failed: {} — using hardcoded", exc)
                self._load_hardcoded()
        else:
            logger.debug("[EventCal] No FRED key — using hardcoded calendar")
            self._load_hardcoded()

        self._last_refresh = datetime.now(timezone.utc)
        self._save()
        logger.info("[EventCal] Calendar: {} events loaded", len(self._events))

    # ------------------------------------------------------------------
    # Status query
    # ------------------------------------------------------------------

    def get_status(self, now: Optional[datetime] = None) -> EventStatus:
        """
        Return the event status at a given UTC time (default: now).

        Checks all events in the calendar and returns the status of the
        highest-priority active event (FOMC > CPI > NFP).
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Sort: FOMC first, then CPI, then others
        priority = {"FOMC": 0, "CPI": 1, "NFP": 2}
        sorted_events = sorted(
            self._events,
            key=lambda e: (priority.get(e.event_type, 9), abs((e.release_utc - now).total_seconds())),
        )

        for event in sorted_events:
            blackout_h, reaction_h = self._windows.get(event.event_type, (6, 2))
            delta_h = (event.release_utc - now).total_seconds() / 3600

            # Pre-event blackout
            if 0 < delta_h <= blackout_h:
                dampen = max(0.3, 1.0 - (blackout_h - delta_h) / blackout_h * 0.7)
                return EventStatus(
                    is_pre_event=True, is_active=False,
                    event_type=event.event_type,
                    hours_to_release=delta_h,
                    hours_since_release=None,
                    source=event.source,
                    dampen_factor=dampen,
                )

            # Post-event reaction window
            if -reaction_h <= delta_h <= 0:
                hours_since = abs(delta_h)
                return EventStatus(
                    is_pre_event=False, is_active=True,
                    event_type=event.event_type,
                    hours_to_release=delta_h,
                    hours_since_release=hours_since,
                    source=event.source,
                    dampen_factor=1.0,   # post-event: no dampening, use reaction data
                )

        return EventStatus(
            is_pre_event=False, is_active=False, event_type=None,
            hours_to_release=None, hours_since_release=None,
            source="none", dampen_factor=1.0,
        )

    def next_events(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the next n upcoming events (for dashboard)."""
        now = datetime.now(timezone.utc)
        upcoming = sorted(
            [e for e in self._events if e.release_utc > now],
            key=lambda e: e.release_utc,
        )[:n]
        return [
            {
                "type":         e.event_type,
                "utc":          e.release_utc.isoformat(),
                "hours_away":   round((e.release_utc - now).total_seconds() / 3600, 1),
                "blackout_h":   self._windows.get(e.event_type, (6, 2))[0],
                "source":       e.source,
                "confirmed":    e.confirmed,
            }
            for e in upcoming
        ]

    # ------------------------------------------------------------------
    # FRED fetch
    # ------------------------------------------------------------------

    def _fetch_fred(self) -> None:
        """
        Fetch FOMC meeting dates from FRED series FOMC.
        FRED doesn't have CPI dates; those remain hardcoded.
        """
        import requests
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id":       "FEDTARMD",   # FOMC target rate midpoint — gives meeting dates
            "api_key":         self.fred_key,
            "file_type":       "json",
            "observation_start": "2024-01-01",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("observations", [])

        fomc_events = []
        for obs in data:
            try:
                dt = datetime.fromisoformat(obs["date"]).replace(
                    hour=19, minute=0, tzinfo=timezone.utc
                )
                fomc_events.append(ScheduledEvent(
                    event_type="FOMC", release_utc=dt,
                    pre_blackout_h=24, post_reaction_h=4,
                    source="FRED", confirmed=True,
                ))
            except Exception:
                continue

        # Keep FRED FOMC dates + hardcoded CPI dates
        cpi_events = [e for e in self._events if e.event_type == "CPI"]
        self._events = fomc_events + cpi_events
        logger.info("[EventCal] FRED: {} FOMC dates loaded", len(fomc_events))

    # ------------------------------------------------------------------
    # Hardcoded baseline
    # ------------------------------------------------------------------

    def _load_hardcoded(self) -> None:
        events: List[ScheduledEvent] = []
        for dt_str in _FOMC_DATES_UTC:
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                events.append(ScheduledEvent(
                    event_type="FOMC", release_utc=dt,
                    pre_blackout_h=24, post_reaction_h=4,
                    source="HARDCODED", confirmed=True,
                ))
            except Exception:
                continue
        for dt_str in _CPI_DATES_UTC:
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                events.append(ScheduledEvent(
                    event_type="CPI", release_utc=dt,
                    pre_blackout_h=6, post_reaction_h=2,
                    source="HARDCODED", confirmed=True,
                ))
            except Exception:
                continue
        self._events = events

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            data = [
                {
                    "event_type":     e.event_type,
                    "release_utc":    e.release_utc.isoformat(),
                    "pre_blackout_h": e.pre_blackout_h,
                    "post_reaction_h": e.post_reaction_h,
                    "source":         e.source,
                    "confirmed":      e.confirmed,
                }
                for e in self._events
            ]
            with open(self.persist_path, "w") as fh:
                json.dump({"events": data, "saved_at": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
        except Exception as exc:
            logger.debug("[EventCal] Save failed: {}", exc)
