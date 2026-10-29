"""
collectors/macro_collector.py — Macro Collector v4
══════════════════════════════════════════════════
Traditional finance regime indicators with MULTI-SOURCE FALLBACK.

PRIMARY:   yfinance  (free, 15min delay)
FALLBACK:  FRED API  (free, daily — for VIX, US10Y, SPX)
LAST-RESORT: Cached last-known-good values with age tracking

v4 FIXES:
  - yfinance frequently fails on Railway (timeout, geo-block, empty data).
    Added FRED API as fallback for critical macro features.
  - Cached last-known-good values prevent macro features from going
    permanently MISSING after a single yfinance failure.
  - Each symbol is fetched independently — one failure doesn't block others.

FRED SERIES:
  VIXCLS   → VIX
  DGS10    → US 10-Year
  SP500    → S&P 500
"""

import os
import time
import requests
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from loguru import logger
from collectors.base import BaseCollector


# FRED series mapping: our_key → FRED series ID
_FRED_SERIES = {
    "vix":   "VIXCLS",
    "us10y": "DGS10",
    # DTWEXBGS has different units/basket from DXY; never substitute it silently.
    "spx":   "SP500",
}

# Max age (seconds) for cached values before they become stale.
# Macro data is slow-moving; 12h cache is acceptable.
_CACHE_MAX_AGE = 43200  # 12 hours


class MacroCollector(BaseCollector):
    """Collect macro indicators with yfinance primary + FRED fallback."""

    def observation_times(self, data, fetched_at):
        return {key: self._cache[key][1] if key in self._cache else fetched_at
                for key in data}

    def __init__(self, config: Dict):
        super().__init__("Macro", config)

        self.symbols = {
            'dxy':   'DX-Y.NYB',
            'us10y': '^TNX',
            'us5y':  '^FVX',
            'spx':   '^GSPC',
            'vix':   '^VIX',
            'qqq':   'QQQ',
        }

        self.em_currencies = {
            'USDTRY': 'USDTRY=X',
            'USDNGN': 'USDNGN=X',
            'USDPKR': 'USDPKR=X',
            'USDEGP': 'USDEGP=X',
            'USDBDT': 'USDBDT=X',
            'USDBRL': 'USDBRL=X',
        }

        self._fred_key = (
            os.environ.get("HYDRA_FRED_KEY", "")
            or config.get("api_keys", {}).get("fred", {}).get("api_key", "")
        )

        # Cache: {key: (value, timestamp)}
        self._cache: Dict[str, tuple] = {}
        self._price_history: Dict[str, Any] = {}
        self._yfinance_consecutive_failures: int = 0

    def fetch(self) -> Dict[str, Any]:
        data = {}
        data.update(self._fetch_market_data())
        data.update(self._fetch_calendar_proximity())
        data.update(self._compute_btc_spx_correlation())

        field_count = sum(1 for v in data.values() if v is not None)
        logger.debug("[Macro] Fetched {} fields (cache hits: {})",
                     field_count, sum(1 for k, (v, t) in self._cache.items()
                                      if time.time() - t < 120))
        return data

    # ── Primary: yfinance ─────────────────────────────────────────────────

    def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch macro data. yfinance → FRED fallback → cache fallback."""
        data = {}
        yf_failed_keys = []

        # 1) Try yfinance for each symbol
        for name, ticker in self.symbols.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period='5d', interval='1h')

                if hist is not None and not hist.empty and len(hist) >= 1:
                    current = float(hist['Close'].iloc[-1])
                    source_time = hist.index[-1].timestamp()
                    if not np.isfinite(current) or source_time > time.time() + 5:
                        raise ValueError('Invalid or future macro observation')
                    data[f'{name}_current'] = current
                    self._update_cache(f'{name}_current', current, source_time)

                    # Twenty-four exchange bars can span several trading days.
                    cutoff = hist.index[-1] - timedelta(hours=24)
                    earlier = hist.loc[hist.index <= cutoff]
                    if not earlier.empty and (cutoff - earlier.index[-1]).total_seconds() <= 7200:
                        prev_24h = float(earlier['Close'].iloc[-1])
                        chg = self._change(name, current, prev_24h)
                        data[f'{name}_change_24h'] = chg
                        self._update_cache(f'{name}_change_24h', chg, source_time)

                    if len(hist) >= 20:
                        sma20 = float(hist['Close'].iloc[-20:].mean())
                        vs = (current - sma20) / sma20 * 100
                        data[f'{name}_vs_sma20'] = vs
                        self._update_cache(f'{name}_vs_sma20', vs, source_time)

                    self._price_history[name] = hist['Close'].copy()
                else:
                    yf_failed_keys.append(name)
            except Exception as e:
                yf_failed_keys.append(name)
                logger.debug("[Macro] yfinance {} failed: {}", name, str(e)[:80])

        # 2) FRED fallback for failed symbols
        if yf_failed_keys and self._fred_key:
            fred_data = self._fetch_fred_fallback(yf_failed_keys)
            data.update(fred_data)

        # 3) Cache fallback for anything still missing
        critical_keys = ['vix_current', 'us10y_current', 'dxy_current', 'spx_current', 'qqq_current']
        for key in critical_keys:
            if key not in data:
                cached = self._get_cached(key)
                if cached is not None:
                    data[key] = cached
                    logger.debug("[Macro] Cache hit for {} (age: {:.0f}s)",
                                 key, time.time() - self._cache[key][1])

        # Track consecutive failures for health reporting
        if yf_failed_keys:
            self._yfinance_consecutive_failures += 1
            if self._yfinance_consecutive_failures <= 3:
                logger.warning("[Macro] yfinance failed for: {} (attempt {})",
                               yf_failed_keys, self._yfinance_consecutive_failures)
        else:
            self._yfinance_consecutive_failures = 0

        # EM currencies (best-effort, not critical)
        for name, ticker in self.em_currencies.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period='5d', interval='1d')
                if hist is not None and not hist.empty:
                    current = float(hist['Close'].iloc[-1])
                    data[f'{name.lower()}_rate'] = current
                    if len(hist) >= 2:
                        prev = float(hist['Close'].iloc[-2])
                        data[f'{name.lower()}_change_1d'] = ((current - prev) / prev) * 100
            except Exception:
                pass

        return data

    # ── FRED Fallback ─────────────────────────────────────────────────────

    def _fetch_fred_fallback(self, failed_keys: list) -> Dict[str, Any]:
        """Fetch daily data from FRED API for symbols that yfinance couldn't get."""
        data = {}
        if not self._fred_key:
            return data

        for key in failed_keys:
            series_id = _FRED_SERIES.get(key)
            if not series_id:
                continue

            try:
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": self._fred_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 5,
                        "observation_start": (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d"),
                    },
                    timeout=(5, 15),
                )
                if not resp.ok:
                    logger.debug("[Macro] FRED {} HTTP {}", series_id, resp.status_code)
                    continue

                observations = resp.json().get("observations", [])
                # Find latest non-"." value
                current_val = None
                prev_val = None
                current_date = prev_date = None
                for obs in observations:
                    val_str = obs.get("value", ".")
                    if val_str != "." and val_str:
                        try:
                            fval = float(val_str)
                            if not np.isfinite(fval):
                                continue
                            if current_val is None:
                                current_val = fval
                                current_date = datetime.fromisoformat(obs['date']).replace(tzinfo=timezone.utc)
                            elif prev_val is None:
                                prev_val = fval
                                prev_date = datetime.fromisoformat(obs['date']).replace(tzinfo=timezone.utc)
                                break
                        except ValueError:
                            continue

                if current_val is not None:
                    data[f'{key}_current'] = current_val
                    self._update_cache(f'{key}_current', current_val, current_date.timestamp())
                    logger.info("[Macro] FRED fallback: {}={:.2f}", key, current_val)

                    if prev_val is not None and prev_val > 0 and current_date - prev_date == timedelta(days=1):
                        chg = self._change(key, current_val, prev_val)
                        data[f'{key}_change_24h'] = chg
                        self._update_cache(f'{key}_change_24h', chg, current_date.timestamp())

            except Exception as e:
                logger.debug("[Macro] FRED {} error: {}", series_id, str(e)[:80])

        return data

    # ── Cache helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _change(name, current, previous):
        if not np.isfinite(current) or not np.isfinite(previous) or previous <= 0:
            raise ValueError('Valid macro reference required')
        return current - previous if name in ('us10y', 'us5y') else (current / previous - 1) * 100

    def _update_cache(self, key: str, value: float, source_time=None) -> None:
        self._cache[key] = (value, time.time() if source_time is None else source_time)

    def _get_cached(self, key: str) -> Optional[float]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > _CACHE_MAX_AGE:
            return None  # Too stale even for fallback
        return value

    # ── Calendar proximity ────────────────────────────────────────────────

    def _fetch_calendar_proximity(self) -> Dict[str, Any]:
        # The legacy guessed dates and generic 14:30 UTC clock were not a
        # verified event feed. Do not publish them as decision-grade countdowns.
        return {"macro_calendar_source": "unverified_legacy_schedule", "macro_calendar_confirmed": False}

    # ── Correlation ───────────────────────────────────────────────────────

    def _compute_btc_spx_correlation(self) -> Dict[str, Any]:
        data = {}
        spx_data = self._price_history.get('spx')
        if spx_data is None or len(spx_data) < 20:
            return data

        try:
            btc = yf.Ticker('BTC-USD')
            btc_hist = btc.history(period='5d', interval='1h')
            if btc_hist is not None and not btc_hist.empty:
                import pandas as pd
                aligned = pd.concat([spx_data.rename('spx'), btc_hist['Close'].rename('btc')], axis=1, join='inner').dropna()
                if len(aligned) >= 21:
                    returns = aligned.pct_change(fill_method=None).dropna()
                    corr = returns['spx'].corr(returns['btc'])
                    if not np.isnan(corr):
                        data['btc_spx_correlation'] = round(float(corr), 3)
                        data['btc_spx_decoupled'] = 1.0 if abs(corr) < 0.3 else 0.0
        except Exception as e:
            logger.debug("[Macro] BTC-SPX correlation: {}", str(e)[:60])

        return data

    # ── Health check ──────────────────────────────────────────────────────

    def health_check(self) -> bool:
        # Try yfinance first
        try:
            t = yf.Ticker('^GSPC')
            hist = t.history(period='1d')
            if hist is not None and not hist.empty:
                logger.info("[Macro] Health check OK (yfinance)")
                return True
        except Exception:
            pass

        # Try FRED as fallback health check
        if self._fred_key:
            try:
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": "VIXCLS",
                        "api_key": self._fred_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1,
                    },
                    timeout=(5, 10),
                )
                if resp.ok:
                    logger.info("[Macro] Health check OK (FRED fallback)")
                    return True
            except Exception:
                pass

        logger.warning("[Macro] Health check FAILED (both yfinance and FRED)")
        return False
