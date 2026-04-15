"""
collectors/kalshi_collector.py — v3
════════════════════════════════════
CHANGES vs v2:
  - trading.kalshi.com REMOVED: DNS fails on Railway us-west2 (NameResolutionError)
  - Circuit breaker: after 3 DNS/connection failures → 30 min cooldown
    Stops the 18+ error log lines per cycle
  - 429 rate limit handler: backs off 60s before retry
  - Single endpoint: api.elections.kalshi.com (works from Railway)
"""

import time
from typing import Dict, Any, Optional
from loguru import logger
from collectors.base import BaseCollector

_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_KEYWORDS = [
    "fed", "interest rate", "rate cut", "rate hike",
    "recession", "inflation", "cpi", "bitcoin", "btc", "crypto",
]
_CIRCUIT_OPEN_SECS = 1800   # 30 min cooldown after repeated DNS failures
_RATE_LIMIT_WAIT   = 60     # seconds to back off on 429


class KalshiCollector(BaseCollector):

    def __init__(self, config: Dict):
        super().__init__("Kalshi", config)
        self._markets: Dict[str, Dict] = {}
        self._circuit_open_until: float = 0.0   # epoch; 0 = circuit closed (OK)
        self._dns_fail_count: int = 0

    def fetch(self) -> Dict[str, Any]:
        now = time.time()

        # Circuit breaker — skip calls until cooldown expires
        if now < self._circuit_open_until:
            remaining = (self._circuit_open_until - now) / 60
            logger.debug("[Kalshi] Circuit open — cooldown {:.0f}m remaining", remaining)
            return {}

        self._discover_markets()
        data = {"kalshi_last_poll": time.time()}  # sentinel: prevents safe_fetch retry on empty-but-ok
        for ticker, info in self._markets.items():
            fname = ticker.lower().replace("-","_")[:40]
            prob = info.get("yes_price", 0.0)
            if prob > 0:
                data[f"kalshi_{fname}_prob"]   = prob
                data[f"kalshi_{fname}_volume"] = info.get("volume", 0)

        logger.debug("[Kalshi] {} fields from {} markets", len(data)-1, len(self._markets))
        return data

    def health_check(self) -> bool:
        resp = self._api_get(f"{_BASE}/markets", params={"limit": 1, "status": "open"})
        ok = resp is not None and "markets" in resp
        logger.info("[Kalshi] Health check {}", "OK" if ok else "FAILED")
        return ok

    def _discover_markets(self):
        self._markets = {}
        # Events endpoint
        events = self._safe_get(f"{_BASE}/events",
                                {"status": "open", "limit": 50, "with_nested_markets": "true"})
        if events:
            for ev in events.get("events", []):
                text = ((ev.get("title","") or "") + " " + (ev.get("category","") or "")).lower()
                if not self._match(text): continue
                for mk in ev.get("markets", []):
                    self._add_market(mk, fallback_title=ev.get("title",""))

        # Markets endpoint
        markets = self._safe_get(f"{_BASE}/markets", {"limit": 100, "status": "open"})
        if markets:
            for mk in markets.get("markets", []):
                text = " ".join([mk.get(k,"") or "" for k in ("title","subtitle","category")]).lower()
                if self._match(text):
                    self._add_market(mk)

        if self._markets:
            logger.info("[Kalshi] {} relevant markets found", len(self._markets))
        else:
            logger.debug("[Kalshi] No matching markets this cycle")

    def _safe_get(self, url: str, params: Dict) -> Optional[Dict]:
        """GET with 429 handling and circuit-breaker trip on DNS errors."""
        for attempt in range(3):
            try:
                resp = self._api_get(url, params=params)
                if resp is not None:
                    self._dns_fail_count = 0   # reset on success
                    return resp
                # None means HTTP error was logged already; don't trip circuit
                return None
            except Exception as e:
                err_str = str(e).lower()
                if "nameresolution" in err_str or "failed to resolve" in err_str:
                    self._dns_fail_count += 1
                    logger.warning("[Kalshi] DNS failure #{}", self._dns_fail_count)
                    if self._dns_fail_count >= 3:
                        self._circuit_open_until = time.time() + _CIRCUIT_OPEN_SECS
                        logger.warning(
                            "[Kalshi] Circuit OPEN — trading.kalshi.com DNS unreachable "
                            "from Railway. Cooling down {}min.", _CIRCUIT_OPEN_SECS // 60
                        )
                    return None
                if "429" in err_str or "too many" in err_str:
                    logger.warning("[Kalshi] Rate limited — waiting {}s", _RATE_LIMIT_WAIT)
                    time.sleep(_RATE_LIMIT_WAIT)
                    continue
                logger.warning("[Kalshi] Attempt {}/3 error: {}", attempt+1, str(e)[:100])
                time.sleep(2 ** attempt)
        return None

    def _add_market(self, mk: Dict, fallback_title: str = ""):
        ticker = mk.get("ticker","")
        if not ticker or ticker in self._markets:
            return
        price = mk.get("yes_bid") or mk.get("last_price") or mk.get("yes_ask") or 0
        if isinstance(price, (int, float)) and price > 1.0:
            price /= 100.0
        self._markets[ticker] = {
            "title":     mk.get("title","") or fallback_title,
            "yes_price": float(price),
            "volume":    mk.get("volume", 0) or 0,
        }

    @staticmethod
    def _match(text: str) -> bool:
        return any(kw in text for kw in _KEYWORDS)
