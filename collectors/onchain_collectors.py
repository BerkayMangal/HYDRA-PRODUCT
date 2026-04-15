"""
collectors/onchain_collectors.py
══════════════════════════════════
Three new FREE collectors — no paid API key required.

1. MempoolCollector      — mempool.space (BTC onchain metrics)
2. BybitCollector        — Bybit public API (L/S ratio, OI, funding)
3. CoinGeckoProCollector — CoinGecko with optional API key (higher rate limit)

All three are key-less by default and geo-block free.
"""

import os
import time
from typing import Dict, Any, Optional
from loguru import logger
from collectors.base import BaseCollector


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mempool.space — BTC onchain metrics (completely free, no key)
# ─────────────────────────────────────────────────────────────────────────────

class MempoolCollector(BaseCollector):
    """
    BTC onchain data from mempool.space public API.
    Provides: mempool congestion, fee rates, hash rate, tx volume.
    No API key. No rate limit issues. No geo-block.
    """

    BASE = "https://mempool.space/api"

    def __init__(self, config: Dict):
        super().__init__("Mempool", config)

    def fetch(self) -> Dict[str, Any]:
        data = {}
        data.update(self._fetch_fees())
        data.update(self._fetch_mempool())
        data.update(self._fetch_hashrate())
        return data

    def _fetch_fees(self) -> Dict:
        r = self._api_get(f"{self.BASE}/v1/fees/recommended")
        if not r:
            return {}
        return {
            "btc_fee_fastest_sat":   float(r.get("fastestFee",   0)),
            "btc_fee_halfhour_sat":  float(r.get("halfHourFee",  0)),
            "btc_fee_economy_sat":   float(r.get("economyFee",   0)),
            # Fee pressure: high fees = high network activity = bullish signal
            "btc_fee_pressure":      float(r.get("fastestFee", 0)) / 50.0,  # normalized ~0-3
        }

    def _fetch_mempool(self) -> Dict:
        r = self._api_get(f"{self.BASE}/mempool")
        if not r:
            return {}
        count    = r.get("count", 0)
        vsize    = r.get("vsize", 0)
        fee_hist = r.get("fee_histogram", [])
        return {
            "mempool_tx_count":   float(count),
            "mempool_size_vb":    float(vsize),
            # Congestion index: high count = high demand
            "mempool_congestion": min(float(count) / 50000.0, 3.0),
        }

    def _fetch_hashrate(self) -> Dict:
        r = self._api_get(f"{self.BASE}/v1/mining/hashrate/3d")
        if not r or not isinstance(r, dict):
            return {}
        hr = r.get("currentHashrate", 0)
        diff = r.get("currentDifficulty", 0)
        return {
            "btc_hashrate_eh":    float(hr) / 1e18 if hr else 0.0,   # EH/s
            "btc_difficulty":     float(diff) / 1e12 if diff else 0.0,
        }

    def health_check(self) -> bool:
        r = self._api_get(f"{self.BASE}/v1/fees/recommended")
        return r is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bybit — L/S ratio, OI, funding (free public API, no geo-block)
# ─────────────────────────────────────────────────────────────────────────────

class BybitCollector(BaseCollector):
    """
    Bybit public derivatives API.
    Provides: L/S ratio, Open Interest, Funding Rate for BTC.
    Free, no key required.

    AUTO-DISABLE (v4 FIX):
      Railway's us-west2 IP range is blocked by Bybit (HTTP 403).
      After 3 consecutive empty fetches, the collector auto-disables
      for 1 hour to prevent log spam. Re-checks automatically.
    """

    BASE = "https://api.bybit.com/v5/market"
    _MAX_CONSECUTIVE_FAILURES = 3
    _COOLDOWN_SECONDS = 3600  # 1 hour

    def __init__(self, config: Dict):
        super().__init__("Bybit", config)
        self.symbol = "BTCUSDT"
        self._disabled_until: float = 0.0
        self._consecutive_403: int = 0

    def safe_fetch(self) -> Dict[str, Any]:
        """Override: skip all retries when auto-disabled."""
        if time.time() < self._disabled_until:
            remaining = int((self._disabled_until - time.time()) / 60)
            logger.debug("[Bybit] Still disabled ({} min remaining)", remaining)
            return self.last_data or {}
        return super().safe_fetch()

    def fetch(self) -> Dict[str, Any]:
        # Auto-disable check
        if time.time() < self._disabled_until:
            return {}

        data = {}
        data.update(self._fetch_ls_ratio())
        data.update(self._fetch_oi())
        data.update(self._fetch_funding())
        data.update(self._fetch_ticker())

        # If ALL sub-fetches returned empty, likely blocked
        if not data:
            self._consecutive_403 += 1
            if self._consecutive_403 >= self._MAX_CONSECUTIVE_FAILURES:
                self._disabled_until = time.time() + self._COOLDOWN_SECONDS
                logger.warning(
                    "[Bybit] Auto-disabled for {:.0f}min after {} consecutive failures (likely IP blocked)",
                    self._COOLDOWN_SECONDS / 60, self._consecutive_403,
                )
                self._consecutive_403 = 0
        else:
            self._consecutive_403 = 0

        return data

    def _fetch_ls_ratio(self) -> Dict:
        r = self._api_get(
            f"{self.BASE}/account-ratio",
            params={"category": "linear", "symbol": self.symbol, "period": "1h", "limit": 1}
        )
        if not r or r.get("retCode") != 0:
            return {}
        items = r.get("result", {}).get("list", [])
        if not items:
            return {}
        item = items[0]
        try:
            buy_ratio  = float(item.get("buyRatio",  0.5))
            sell_ratio = float(item.get("sellRatio", 0.5))
            ls = buy_ratio / sell_ratio if sell_ratio > 0 else 1.0
            return {
                "bybit_ls_ratio":    round(ls, 4),
                "bybit_buy_ratio":   round(buy_ratio, 4),
                "bybit_sell_ratio":  round(sell_ratio, 4),
            }
        except Exception:
            return {}

    def _fetch_oi(self) -> Dict:
        r = self._api_get(
            f"{self.BASE}/open-interest",
            params={"category": "linear", "symbol": self.symbol, "intervalTime": "1h", "limit": 2}
        )
        if not r or r.get("retCode") != 0:
            return {}
        items = r.get("result", {}).get("list", [])
        if len(items) < 1:
            return {}
        try:
            oi_now  = float(items[0].get("openInterest", 0))
            oi_prev = float(items[1].get("openInterest", 0)) if len(items) > 1 else oi_now
            oi_chg  = ((oi_now - oi_prev) / oi_prev * 100) if oi_prev > 0 else 0.0
            return {
                "bybit_oi_usd":        round(oi_now, 0),
                "bybit_oi_change_pct": round(oi_chg, 4),
            }
        except Exception:
            return {}

    def _fetch_funding(self) -> Dict:
        r = self._api_get(
            f"{self.BASE}/funding/history",
            params={"category": "linear", "symbol": self.symbol, "limit": 1}
        )
        if not r or r.get("retCode") != 0:
            return {}
        items = r.get("result", {}).get("list", [])
        if not items:
            return {}
        try:
            rate = float(items[0].get("fundingRate", 0))
            return {"bybit_funding_rate": round(rate, 8)}
        except Exception:
            return {}

    def _fetch_ticker(self) -> Dict:
        r = self._api_get(
            f"{self.BASE}/tickers",
            params={"category": "linear", "symbol": self.symbol}
        )
        if not r or r.get("retCode") != 0:
            return {}
        items = r.get("result", {}).get("list", [])
        if not items:
            return {}
        t = items[0]
        try:
            return {
                "bybit_mark_price":     float(t.get("markPrice",     0)),
                "bybit_index_price":    float(t.get("indexPrice",    0)),
                "bybit_turnover_24h":   float(t.get("turnover24h",   0)),
                "bybit_volume_24h":     float(t.get("volume24h",     0)),
            }
        except Exception:
            return {}

    def health_check(self) -> bool:
        r = self._api_get(f"{self.BASE}/tickers", params={"category": "linear", "symbol": self.symbol})
        return r is not None and r.get("retCode") == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. CoinGecko Pro — upgraded version that uses COINGECKO_API_KEY if set
# ─────────────────────────────────────────────────────────────────────────────

class CoinGeckoKeyCollector(BaseCollector):
    """
    Injects COINGECKO_API_KEY into existing CoinGecko requests.
    Also fetches additional endpoints only available with Demo+ key:
    - BTC OHLC (7d)
    - Global market data
    - Trending coins (separate from Venom)

    Falls back to keyless if env var not set.
    """

    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self, config: Dict):
        super().__init__("CoinGeckoPlus", config)
        self.api_key = os.environ.get("COINGECKO_API_KEY", "")
        if self.api_key:
            logger.info("[CoinGeckoPlus] API key loaded — higher rate limits active")
        else:
            logger.warning("[CoinGeckoPlus] No API key — using free tier")

    @property
    def _headers(self) -> Dict:
        if self.api_key:
            return {"x-cg-demo-api-key": self.api_key}
        return {}

    def fetch(self) -> Dict[str, Any]:
        data = {}
        data.update(self._fetch_global())
        data.update(self._fetch_btc_ohlc())
        data.update(self._fetch_trending())
        return data

    def _fetch_global(self) -> Dict:
        r = self._api_get(f"{self.BASE}/global", headers=self._headers)
        if not r:
            return {}
        d = r.get("data", {})
        try:
            return {
                "cg_total_market_cap_usd":  float(d.get("total_market_cap", {}).get("usd", 0)),
                "cg_total_volume_usd":      float(d.get("total_volume",     {}).get("usd", 0)),
                "cg_btc_dominance":         float(d.get("market_cap_percentage", {}).get("btc", 0)),
                "cg_eth_dominance":         float(d.get("market_cap_percentage", {}).get("eth", 0)),
                "cg_active_coins":          float(d.get("active_cryptocurrencies", 0)),
                "cg_market_cap_change_24h": float(d.get("market_cap_change_percentage_24h_usd", 0)),
            }
        except Exception:
            return {}

    def _fetch_btc_ohlc(self) -> Dict:
        """BTC 7-day OHLC — useful for momentum signals."""
        r = self._api_get(
            f"{self.BASE}/coins/bitcoin/ohlc",
            headers=self._headers,
            params={"vs_currency": "usd", "days": "7"}
        )
        if not r or not isinstance(r, list) or len(r) < 2:
            return {}
        try:
            # Most recent candle
            latest = r[-1]   # [timestamp, open, high, low, close]
            prev   = r[-2]
            close  = float(latest[4])
            open_  = float(latest[1])
            high   = float(latest[2])
            low    = float(latest[3])
            prev_close = float(prev[4])
            return {
                "cg_btc_ohlc_close":     close,
                "cg_btc_ohlc_high":      high,
                "cg_btc_ohlc_low":       low,
                "cg_btc_candle_body":    round((close - open_) / open_ * 100, 4),
                "cg_btc_candle_chg":     round((close - prev_close) / prev_close * 100, 4),
                "cg_btc_wick_ratio":     round((high - low) / close * 100, 4) if close > 0 else 0.0,
            }
        except Exception:
            return {}

    def _fetch_trending(self) -> Dict:
        """Top trending coins — useful for Venom sniper confirmation."""
        r = self._api_get(f"{self.BASE}/search/trending", headers=self._headers)
        if not r:
            return {}
        try:
            coins = r.get("coins", [])[:7]
            symbols = [c.get("item", {}).get("symbol", "").upper() for c in coins]
            return {
                "cg_trending_symbols": symbols,
                "cg_trending_count":   float(len(symbols)),
            }
        except Exception:
            return {}

    def health_check(self) -> bool:
        r = self._api_get(f"{self.BASE}/ping", headers=self._headers)
        return r is not None
