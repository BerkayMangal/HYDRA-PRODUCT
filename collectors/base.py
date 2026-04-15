"""
collectors/base.py — BaseCollector v2
══════════════════════════════════════
FIXES vs v1:

  FIX 1 — Exponential backoff in safe_fetch()
  ─────────────────────────────────────────────
  v1 used linear backoff: sleep(retry_delay * (attempt + 1)) = 2s, 4s, 6s.
  Changed to true exponential: sleep(retry_delay ** (attempt + 1)) = 2s, 4s, 8s.
  This is the standard approach for external API calls to avoid thundering herd.

  FIX 2 — _api_get logs HTTP response body on 4xx/5xx
  ────────────────────────────────────────────────────
  Previously, only the status code was logged. Added response body
  (truncated to 200 chars) to the error log so API errors are diagnosable
  without having to re-run with print debugging.

  FIX 3 — Separate connect vs. read timeout
  ──────────────────────────────────────────
  Single int timeout was used as both connect and read timeout.
  Changed to tuple (connect_timeout, read_timeout) as recommended by requests.
  Default: 5s connect, 15s read — prevents hanging on slow DNS / TCP handshake.
"""

import time
import requests

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple, Union

from loguru import logger


class BaseCollector(ABC):
    """Abstract base class for all HYDRA data collectors."""

    def __init__(self, name: str, config: Dict):
        self.name               = name
        self.config             = config
        self.last_fetch_time: Optional[datetime] = None
        self.last_data: Optional[Dict]           = None
        self.consecutive_errors: int             = 0
        self.max_retries: int                    = 3
        self.retry_delay: float                  = 2.0   # base for exponential backoff

        logger.info("[{}] Collector initialized", self.name)

    @abstractmethod
    def fetch(self) -> Dict[str, Any]:
        """
        Fetch latest data from source.
        Returns dict with feature names → floats.
        Returns empty dict on failure.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify API connectivity and auth."""
        pass

    def safe_fetch(self) -> Dict[str, Any]:
        """Fetch with exponential backoff and error tracking."""
        for attempt in range(self.max_retries):
            try:
                data = self.fetch()
                if data:
                    self.last_fetch_time    = datetime.now(timezone.utc)
                    self.last_data          = data
                    self.consecutive_errors = 0
                    return data
                else:
                    logger.warning("[{}] Empty data on attempt {}", self.name, attempt + 1)
            except Exception as e:
                self.consecutive_errors += 1
                logger.error("[{}] Fetch error (attempt {}): {}", self.name, attempt + 1, e)

            if attempt < self.max_retries - 1:
                # Exponential backoff: 2s, 4s, 8s …
                wait = self.retry_delay ** (attempt + 1)
                logger.debug("[{}] Retrying in {:.1f}s", self.name, wait)
                time.sleep(wait)

        # All retries failed — return stale data if available
        if self.last_data:
            logger.warning("[{}] All retries failed. Using stale data from {}",
                           self.name, self.last_fetch_time)
            return self.last_data

        return {}

    def _api_get(
        self,
        url: str,
        headers: Optional[Dict] = None,
        params:  Optional[Dict] = None,
        timeout: Union[int, Tuple[int, int]] = (5, 15),  # (connect, read)
    ) -> Optional[Dict]:
        """
        Generic GET with improved error diagnostics.

        FIX: logs the response body on 4xx/5xx so API errors are visible
        in Railway logs without needing to add temporary print() statements.
        """
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.ok:
                return resp.json()

            # Log status + first 300 chars of body for diagnosis
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:300]

            logger.error(
                "[{}] HTTP {} on {}: {}",
                self.name, resp.status_code,
                url.split("?")[0],   # strip query params for readability
                str(err_body)[:200],
            )
            return None

        except requests.exceptions.ConnectTimeout:
            logger.error("[{}] Connect timeout: {}", self.name, url.split("?")[0])
        except requests.exceptions.ReadTimeout:
            logger.error("[{}] Read timeout: {}", self.name, url.split("?")[0])
        except requests.exceptions.ConnectionError as e:
            logger.error("[{}] Connection error: {}", self.name, e)
        except Exception as e:
            logger.error("[{}] Request failed: {}", self.name, e)

        return None

    @property
    def is_stale(self) -> bool:
        if not self.last_fetch_time:
            return True
        age     = (datetime.now(timezone.utc) - self.last_fetch_time).total_seconds()
        max_age = self.config.get("polling_interval_sec", 300) * 2
        return age > max_age

    @property
    def status(self) -> Dict:
        return {
            "name":               self.name,
            "last_fetch":         str(self.last_fetch_time) if self.last_fetch_time else None,
            "is_stale":           self.is_stale,
            "consecutive_errors": self.consecutive_errors,
            "has_data":           self.last_data is not None,
        }
