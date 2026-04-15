"""
CoinGlass Collector - The backbone of microstructure data.
Aggregated OI, funding, CVD, liquidation, L/S ratio, orderbook, ETF flow.
API: $29-79/mo, V4 endpoints.
Docs: https://docs.coinglass.com/
"""

from typing import Dict, Any
from loguru import logger
from collectors.base import BaseCollector


# CoinGlass API V4 base URL
BASE_URL = "https://open-api-v3.coinglass.com/api"


class CoinGlassCollector(BaseCollector):
    """Collect aggregated derivatives data from CoinGlass."""
    
    def __init__(self, config: Dict):
        super().__init__("CoinGlass", config)
        
        self.api_key = config.get('api_keys', {}).get('coinglass', {}).get('api_key', '')
        self.headers = {
            'accept': 'application/json',
            'CG-API-KEY': self.api_key,
        }
        self.symbol = "BTC"  # CoinGlass uses plain symbol
    
    def fetch(self) -> Dict[str, Any]:
        """Fetch all microstructure data from CoinGlass."""
        data = {}
        
        # Fetch each data type, merge into single dict
        data.update(self._fetch_open_interest())
        data.update(self._fetch_funding_rate())
        data.update(self._fetch_liquidations())
        data.update(self._fetch_long_short_ratio())
        data.update(self._fetch_etf_flow())
        
        logger.debug(f"[CoinGlass] Fetched {len(data)} fields")
        return data
    
    def _fetch_open_interest(self) -> Dict[str, Any]:
        """Aggregated open interest across exchanges."""
        data = {}
        
        # OI aggregated history
        resp = self._api_get(
            f"{BASE_URL}/futures/openInterest/ohlc-aggregated-history",
            headers=self.headers,
            params={'symbol': self.symbol, 'interval': '5m', 'limit': 3}
        )
        
        if resp and resp.get('success') and resp.get('data'):
            records = resp['data']
            if len(records) >= 2:
                current_oi = records[-1].get('o', 0)  # latest OI
                prev_oi = records[-2].get('o', 0)
                
                data['oi_current'] = current_oi
                data['oi_change'] = current_oi - prev_oi
                data['oi_change_pct'] = ((current_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        
        return data
    
    def _fetch_funding_rate(self) -> Dict[str, Any]:
        """Aggregated funding rate across exchanges."""
        data = {}
        
        # Current funding rate
        resp = self._api_get(
            f"{BASE_URL}/futures/funding/current",
            headers=self.headers,
            params={'symbol': self.symbol}
        )
        
        if resp and resp.get('success') and resp.get('data'):
            records = resp['data']
            if records:
                # Weighted average funding across exchanges
                total_oi = 0
                weighted_fr = 0
                for ex in records:
                    ex_oi = ex.get('openInterest', 0)
                    ex_fr = ex.get('fundingRate', 0)
                    if ex_oi and ex_fr is not None:
                        weighted_fr += ex_fr * ex_oi
                        total_oi += ex_oi
                
                if total_oi > 0:
                    data['funding_rate'] = weighted_fr / total_oi
                    data['funding_rate_max'] = max(
                        (ex.get('fundingRate', 0) for ex in records if ex.get('fundingRate')), 
                        default=0
                    )
                    data['funding_rate_min'] = min(
                        (ex.get('fundingRate', 0) for ex in records if ex.get('fundingRate')), 
                        default=0
                    )
        
        return data
    
    def _fetch_liquidations(self) -> Dict[str, Any]:
        """Liquidation data - long/short separately."""
        data = {}
        
        resp = self._api_get(
            f"{BASE_URL}/futures/liquidation/aggregated-history",
            headers=self.headers,
            params={'symbol': self.symbol, 'interval': '5m', 'limit': 1}
        )
        
        if resp and resp.get('success') and resp.get('data'):
            records = resp['data']
            if records:
                latest = records[-1]
                data['liq_long_vol'] = latest.get('longLiquidationUsd', 0)
                data['liq_short_vol'] = latest.get('shortLiquidationUsd', 0)
                data['liq_total'] = data['liq_long_vol'] + data['liq_short_vol']
                
                # Liquidation imbalance: positive = more longs liquidated (bearish pressure)
                total = data['liq_total']
                if total > 0:
                    data['liq_imbalance'] = (data['liq_long_vol'] - data['liq_short_vol']) / total
        
        return data
    
    def _fetch_long_short_ratio(self) -> Dict[str, Any]:
        """Global long/short ratio."""
        data = {}
        
        resp = self._api_get(
            f"{BASE_URL}/futures/globalLongShortAccountRatio/history",
            headers=self.headers,
            params={'symbol': self.symbol, 'interval': '5m', 'limit': 2}
        )
        
        if resp and resp.get('success') and resp.get('data'):
            records = resp['data']
            if records:
                latest = records[-1]
                data['ls_ratio'] = latest.get('longRate', 0.5)
                data['ls_long_pct'] = latest.get('longRate', 0.5)
                data['ls_short_pct'] = latest.get('shortRate', 0.5)
                
                # Change vs previous
                if len(records) >= 2:
                    prev = records[-2]
                    data['ls_ratio_delta'] = (
                        latest.get('longRate', 0.5) - prev.get('longRate', 0.5)
                    )
        
        return data
    
    def _fetch_etf_flow(self) -> Dict[str, Any]:
        """Bitcoin ETF net flow data."""
        data = {}
        
        resp = self._api_get(
            f"{BASE_URL}/etf/bitcoin/net-assets-history",
            headers=self.headers,
            params={'limit': 8}  # ~7 trading days + 1
        )
        
        if resp and resp.get('success') and resp.get('data'):
            records = resp['data']
            if records:
                # Latest daily flow
                latest = records[-1]
                data['etf_net_flow_daily'] = latest.get('totalNetFlow', 0)
                
                # 7-day rolling sum
                if len(records) >= 7:
                    data['etf_net_flow_7d'] = sum(
                        r.get('totalNetFlow', 0) for r in records[-7:]
                    )
                else:
                    data['etf_net_flow_7d'] = sum(
                        r.get('totalNetFlow', 0) for r in records
                    )
        
        return data
    
    def fetch_basis_spread(self) -> Dict[str, Any]:
        """Basis spread: perp price vs spot price."""
        data = {}
        
        # This may need spot + futures price comparison
        # CoinGlass provides this in some endpoints
        resp = self._api_get(
            f"{BASE_URL}/futures/basis/history",
            headers=self.headers,
            params={'symbol': self.symbol, 'interval': '5m', 'limit': 1}
        )
        
        if resp and resp.get('success') and resp.get('data'):
            records = resp['data']
            if records:
                latest = records[-1]
                data['basis_spread'] = latest.get('basis', 0)
                data['basis_spread_pct'] = latest.get('basisRate', 0)
        
        return data
    
    def fetch_cvd(self) -> Dict[str, Any]:
        """Cumulative Volume Delta - aggregated across exchanges."""
        data = {}
        
        resp = self._api_get(
            f"{BASE_URL}/futures/aggregatedCVD/history",
            headers=self.headers,
            params={'symbol': self.symbol, 'interval': '5m', 'limit': 3}
        )
        
        if resp and resp.get('success') and resp.get('data'):
            records = resp['data']
            if records:
                data['cvd_perp'] = records[-1].get('cvd', 0)
                if len(records) >= 2:
                    data['cvd_5m_delta'] = records[-1].get('cvd', 0) - records[-2].get('cvd', 0)
        
        # Spot CVD separately if available
        resp_spot = self._api_get(
            f"{BASE_URL}/spot/aggregatedCVD/history",
            headers=self.headers,
            params={'symbol': self.symbol, 'interval': '5m', 'limit': 2}
        )
        
        if resp_spot and resp_spot.get('success') and resp_spot.get('data'):
            records = resp_spot['data']
            if records:
                data['cvd_spot'] = records[-1].get('cvd', 0)
        
        return data
    
    def fetch_liquidation_heatmap(self) -> Dict[str, Any]:
        """Liquidation cluster proximity data."""
        data = {}
        
        # Heatmap might require Prime subscription
        # Fallback: use aggregated liquidation levels
        resp = self._api_get(
            f"{BASE_URL}/futures/liquidation/heatmap",
            headers=self.headers,
            params={'symbol': self.symbol}
        )
        
        if resp and resp.get('success') and resp.get('data'):
            # Parse nearest liquidation clusters above and below current price
            data['liq_cluster_data'] = resp['data']
        
        return data
    
    def health_check(self) -> bool:
        """Verify CoinGlass API connectivity."""
        resp = self._api_get(
            f"{BASE_URL}/futures/openInterest/ohlc-aggregated-history",
            headers=self.headers,
            params={'symbol': 'BTC', 'interval': '1h', 'limit': 1}
        )
        ok = resp is not None and resp.get('success', False)
        logger.info(f"[CoinGlass] Health check {'OK' if ok else 'FAILED'}")
        return ok
