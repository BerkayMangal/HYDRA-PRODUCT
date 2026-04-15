"""
CryptoQuant Collector - On-chain data.
Exchange netflow, miner flow, whale transactions, stablecoin flow.
API: $39+/mo (Advanced plan).
Docs: https://cryptoquant.com/docs
"""

from typing import Dict, Any
from loguru import logger
from collectors.base import BaseCollector


BASE_URL = "https://api.cryptoquant.com/v1"


class CryptoQuantCollector(BaseCollector):
    """Collect on-chain flow data from CryptoQuant."""
    
    def __init__(self, config: Dict):
        super().__init__("CryptoQuant", config)
        
        self.api_key = config.get('api_keys', {}).get('cryptoquant', {}).get('api_key', '')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
        }
    
    def fetch(self) -> Dict[str, Any]:
        """Fetch all on-chain flow data."""
        data = {}
        
        data.update(self._fetch_exchange_netflow())
        data.update(self._fetch_miner_flow())
        data.update(self._fetch_stablecoin_flow())
        
        logger.debug(f"[CryptoQuant] Fetched {len(data)} fields")
        return data
    
    def _fetch_exchange_netflow(self) -> Dict[str, Any]:
        """Bitcoin exchange netflow - key on-chain signal.
        
        Negative netflow = BTC leaving exchanges = accumulation (bullish)
        Positive netflow = BTC entering exchanges = selling pressure (bearish)
        """
        data = {}
        
        resp = self._api_get(
            f"{BASE_URL}/bitcoin/exchange-flows/netflow",
            headers=self.headers,
            params={
                'exchange': 'all_exchange',
                'window': 'day',
                'limit': 8,
            }
        )
        
        if resp and resp.get('result') and resp['result'].get('data'):
            records = resp['result']['data']
            if records:
                latest = records[-1]
                data['exchange_netflow_btc'] = latest.get('netflow', 0)
                
                # 7-day rolling sum
                if len(records) >= 7:
                    data['exchange_netflow_7d'] = sum(
                        r.get('netflow', 0) for r in records[-7:]
                    )
                
                # Trend: is netflow getting more negative (bullish) or positive (bearish)?
                if len(records) >= 3:
                    recent_avg = sum(r.get('netflow', 0) for r in records[-3:]) / 3
                    older_avg = sum(r.get('netflow', 0) for r in records[-7:-3]) / max(len(records[-7:-3]), 1)
                    data['exchange_netflow_trend'] = recent_avg - older_avg
        
        return data
    
    def _fetch_miner_flow(self) -> Dict[str, Any]:
        """Miner outflow to exchanges.
        
        High miner selling → bearish pressure.
        Especially important post-halving when miners are stressed.
        """
        data = {}
        
        resp = self._api_get(
            f"{BASE_URL}/bitcoin/miner-flows/miner-to-exchange",
            headers=self.headers,
            params={
                'miner': 'all_miner',
                'window': 'day',
                'limit': 8,
            }
        )
        
        if resp and resp.get('result') and resp['result'].get('data'):
            records = resp['result']['data']
            if records:
                latest = records[-1]
                data['miner_outflow_btc'] = latest.get('value', 0)
                
                # 7-day average for trend
                if len(records) >= 7:
                    data['miner_outflow_7d_avg'] = sum(
                        r.get('value', 0) for r in records[-7:]
                    ) / 7
        
        return data
    
    def _fetch_stablecoin_flow(self) -> Dict[str, Any]:
        """Stablecoin (USDT/USDC) exchange flow.
        
        Stablecoin inflow to exchanges → buying power arriving → bullish
        Large Tether mints → anticipation of demand → leading bullish indicator
        """
        data = {}
        
        resp = self._api_get(
            f"{BASE_URL}/stablecoin/exchange-stablecoins-ratio",
            headers=self.headers,
            params={
                'window': 'day',
                'limit': 2,
            }
        )
        
        if resp and resp.get('result') and resp['result'].get('data'):
            records = resp['result']['data']
            if records:
                latest = records[-1]
                # Exchange Stablecoins Ratio = stablecoin reserves / BTC market cap
                # Higher ratio = more buying power available = potentially bullish
                data['stablecoin_exchange_ratio'] = latest.get('value', 0)
                
                if len(records) >= 2:
                    prev = records[-2]
                    data['stablecoin_ratio_delta'] = (
                        latest.get('value', 0) - prev.get('value', 0)
                    )
        
        return data
    
    def health_check(self) -> bool:
        """Verify CryptoQuant API connectivity."""
        resp = self._api_get(
            f"{BASE_URL}/bitcoin/exchange-flows/netflow",
            headers=self.headers,
            params={'exchange': 'all_exchange', 'window': 'day', 'limit': 1}
        )
        ok = resp is not None and 'result' in resp
        logger.info(f"[CryptoQuant] Health check {'OK' if ok else 'FAILED'}")
        return ok
