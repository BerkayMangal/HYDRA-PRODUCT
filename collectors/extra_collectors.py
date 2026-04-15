"""
Extra Data Collectors - All FREE
1. Fear & Greed Index (alternative.me)
2. Binance global L/S ratio + taker volume (public endpoints)
3. Bitcoin dominance (CoinGecko free)
"""

import os
import requests
from typing import Dict, Any
from loguru import logger
from collectors.base import BaseCollector


class FearGreedCollector(BaseCollector):
    """Crypto Fear & Greed Index - FREE, updates daily.
    
    0-25: Extreme Fear (contrarian buy signal)
    25-50: Fear
    50-75: Greed
    75-100: Extreme Greed (contrarian sell signal)
    """
    
    def __init__(self, config: Dict):
        super().__init__("FearGreed", config)
    
    def fetch(self) -> Dict[str, Any]:
        data = {}
        try:
            resp = self._api_get(
                'https://api.alternative.me/fng/',
                params={'limit': 2, 'format': 'json'}
            )
            
            if resp and resp.get('data'):
                records = resp['data']
                if records:
                    current = records[0]
                    data['fear_greed_value'] = int(current.get('value', 50))
                    data['fear_greed_label'] = current.get('value_classification', 'Neutral')
                    
                    # Change from yesterday
                    if len(records) >= 2:
                        prev = int(records[1].get('value', 50))
                        data['fear_greed_change'] = data['fear_greed_value'] - prev
                    
                    logger.info(f"[FearGreed] {data['fear_greed_value']} ({data['fear_greed_label']})")
        except Exception as e:
            logger.error(f"[FearGreed] {e}")
        return data
    
    def health_check(self) -> bool:
        resp = self._api_get('https://api.alternative.me/fng/', params={'limit': 1})
        ok = resp is not None and 'data' in resp
        logger.info(f"[FearGreed] Health check {'OK' if ok else 'FAILED'}")
        return ok


class BinanceGlobalCollector(BaseCollector):
    """Binance global data - FREE public endpoints.
    
    May be geo-blocked from Railway, but worth trying.
    """
    
    def __init__(self, config: Dict):
        super().__init__("BinanceGlobal", config)
        self.symbol = 'BTCUSDT'
        self.max_retries = 1  # Don't waste time if geo-blocked
        self._is_blocked = False
    
    def fetch(self) -> Dict[str, Any]:
        if self._is_blocked:
            return {}
        data = {}
        data.update(self._fetch_global_ls())
        if not data:
            self._is_blocked = True
            logger.info("[BinanceGlobal] Geo-blocked, disabling future attempts")
            return {}
        data.update(self._fetch_taker_volume())
        logger.debug(f"[BinanceGlobal] Fetched {len(data)} fields")
        return data
    
    def _fetch_global_ls(self) -> Dict[str, Any]:
        """Global long/short account ratio."""
        data = {}
        try:
            resp = self._api_get(
                'https://fapi.binance.com/futures/data/globalLongShortAccountRatio',
                params={'symbol': self.symbol, 'period': '5m', 'limit': 2}
            )
            
            if resp and isinstance(resp, list) and len(resp) >= 1:
                latest = resp[-1]
                data['binance_ls_long'] = float(latest.get('longAccount', 0.5))
                data['binance_ls_short'] = float(latest.get('shortAccount', 0.5))
                data['binance_ls_ratio'] = float(latest.get('longShortRatio', 1.0))
        except Exception as e:
            logger.debug(f"[BinanceGlobal] L/S: {e}")
        return data
    
    def _fetch_taker_volume(self) -> Dict[str, Any]:
        """Taker buy/sell volume ratio."""
        data = {}
        try:
            resp = self._api_get(
                'https://fapi.binance.com/futures/data/takerlongshortRatio',
                params={'symbol': self.symbol, 'period': '5m', 'limit': 2}
            )
            
            if resp and isinstance(resp, list) and len(resp) >= 1:
                latest = resp[-1]
                data['binance_taker_buy_ratio'] = float(latest.get('buyVol', 0)) / (
                    float(latest.get('buyVol', 1)) + float(latest.get('sellVol', 1))
                )
                data['binance_taker_ratio'] = float(latest.get('buySellRatio', 1.0))
        except Exception as e:
            logger.debug(f"[BinanceGlobal] Taker: {e}")
        return data
    
    def health_check(self) -> bool:
        try:
            resp = self._api_get(
                'https://fapi.binance.com/futures/data/globalLongShortAccountRatio',
                params={'symbol': 'BTCUSDT', 'period': '1h', 'limit': 1}
            )
            ok = resp is not None and isinstance(resp, list)
            logger.info(f"[BinanceGlobal] Health check {'OK' if ok else 'FAILED (likely geo-blocked)'}")
            return ok
        except:
            logger.info("[BinanceGlobal] Health check FAILED (geo-blocked)")
            return False


class MarketSentimentCollector(BaseCollector):
    """Market sentiment + top coin prices from CoinGecko — FREE.
    
    Single /coins/markets call: top 15 coins + we derive dominance/mcap from it.
    Much more efficient than separate /global call.
    """
    
    def __init__(self, config: Dict):
        super().__init__("Sentiment", config)
        self._coins_cache = []
        self.max_retries = 1  # CoinGecko rate limit - don't spam retries
    
    def fetch(self) -> Dict[str, Any]:
        data = {}
        data.update(self._fetch_top_coins())
        
        if data:
            logger.debug(f"[Sentiment] Fetched {len(data)} fields")
        return data
    
    def _fetch_top_coins(self) -> Dict[str, Any]:
        """CoinGecko top coins — prices, mcap, 24h change, volume.
        One API call gives us everything: coin prices + global stats."""
        data = {}
        try:
            api_key = os.getenv('COINGECKO_API_KEY', '').strip()
            headers = {'accept': 'application/json'}
            if api_key:
                headers['x-cg-demo-api-key'] = api_key

            resp = self._api_get(
                'https://api.coingecko.com/api/v3/coins/markets',
                params={
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': 15,
                    'page': 1,
                    'sparkline': 'false',
                    'price_change_percentage': '24h',
                },
                headers=headers
            )
            
            if resp and isinstance(resp, list) and len(resp) > 0:
                self._coins_cache = resp
                
                # Top coins list for dashboard
                top_coins = []
                total_mcap = 0
                btc_mcap = 0
                
                for coin in resp:
                    symbol = (coin.get('symbol', '') or '').upper()
                    price = float(coin.get('current_price', 0) or 0)
                    change = float(coin.get('price_change_percentage_24h', 0) or 0)
                    mcap = float(coin.get('market_cap', 0) or 0)
                    vol = float(coin.get('total_volume', 0) or 0)
                    name = coin.get('name', '')
                    
                    total_mcap += mcap
                    if symbol == 'BTC':
                        btc_mcap = mcap
                    
                    top_coins.append({
                        'symbol': symbol,
                        'name': name,
                        'price': price,
                        'change_24h': round(change, 2),
                        'market_cap': mcap,
                        'market_cap_b': round(mcap / 1e9, 1),
                        'volume': vol,
                        'volume_b': round(vol / 1e9, 1),
                        'image': coin.get('image', ''),
                    })
                    
                    # Store individual coin prices
                    data[f'coin_{symbol.lower()}_price'] = price
                    data[f'coin_{symbol.lower()}_change'] = round(change, 2)
                
                data['top_coins'] = top_coins[:10]
                
                # Derive global stats from top coins
                if total_mcap > 0:
                    data['total_mcap_trillion'] = round(total_mcap / 1e12, 2)
                    if btc_mcap > 0:
                        data['btc_dominance'] = round(btc_mcap / total_mcap * 100, 1)
                
                # ETH price specifically
                for coin in resp:
                    if coin.get('symbol', '').upper() == 'ETH':
                        data['eth_price'] = float(coin.get('current_price', 0) or 0)
                        data['eth_change_24h'] = float(coin.get('price_change_percentage_24h', 0) or 0)
                        break
                
                # Total market cap change estimate (from BTC as proxy)
                btc_change = next((c.get('price_change_percentage_24h', 0) for c in resp 
                                   if c.get('symbol', '').upper() == 'BTC'), 0)
                data['total_mcap_change_24h'] = float(btc_change or 0)
                
                logger.info(f"[Sentiment] {len(top_coins)} coins, BTC dom: {data.get('btc_dominance', 0):.1f}%, ETH: ${data.get('eth_price', 0):,.0f}")
        except Exception as e:
            logger.error(f"[Sentiment] {e}")
        return data
    
    def health_check(self) -> bool:
        # Skip CoinGecko ping to preserve rate limit budget
        logger.info("[Sentiment] Health check OK (skipped ping to save rate limit)")
        return True
