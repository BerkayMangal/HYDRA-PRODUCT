"""
Polymarket Collector v3 - Fixed discovery.
Fetches active markets and filters client-side by keywords.
Extracts probability from outcomePrices in list response.
"""

import json as json_lib
from typing import Dict, Any
from loguru import logger
from collectors.base import BaseCollector

GAMMA_URL = "https://gamma-api.polymarket.com"


class PolymarketCollector(BaseCollector):
    
    KEYWORDS = ['fed', 'rate cut', 'interest rate', 'bitcoin', 'btc',
                'recession', 'inflation', 'crypto', 'cpi', 'fomc']
    
    def __init__(self, config: Dict):
        super().__init__("Polymarket", config)
        self._markets: Dict[str, Dict] = {}
    
    def fetch(self) -> Dict[str, Any]:
        data = {}
        
        # Discover markets with embedded prices
        self._discover()
        
        for slug, info in self._markets.items():
            fname = slug[:40].replace('-', '_').replace(' ', '_')
            prob = info.get('prob', 0)
            if prob > 0:
                data[f"poly_{fname}_prob"] = float(prob)
                data[f"poly_{fname}_vol"] = float(info.get('volume', 0))
        
        fed_probs = [v for k, v in data.items() if 'fed' in k and 'prob' in k]
        if fed_probs:
            data['poly_fed_sentiment'] = sum(fed_probs) / len(fed_probs)
        
        if data:
            logger.info(f"[Polymarket] {len(self._markets)} markets, {len(data)} features")
        return data
    
    def _discover(self):
        """Fetch active markets, filter by keywords, extract prices."""
        self._markets = {}
        
        # Fetch a batch of active markets
        resp = self._api_get(
            f"{GAMMA_URL}/markets",
            params={
                'closed': 'false',
                'active': 'true',
                'limit': 100,
                'order': 'volume',
                'ascending': 'false',
            }
        )
        
        if not resp or not isinstance(resp, list):
            logger.warning("[Polymarket] No markets returned")
            return
        
        for market in resp:
            question = (market.get('question', '') or '').lower()
            
            # Check if relevant
            if not any(kw in question for kw in self.KEYWORDS):
                continue
            
            slug = market.get('slug', '') or str(market.get('id', ''))[:20]
            if not slug:
                continue
            
            # Extract probability
            prob = 0
            
            # Method 1: outcomePrices field (JSON string like "[0.65, 0.35]")
            op = market.get('outcomePrices', '')
            if op:
                try:
                    prices = json_lib.loads(op) if isinstance(op, str) else op
                    if isinstance(prices, list) and len(prices) >= 1:
                        prob = float(prices[0])
                except (json_lib.JSONDecodeError, ValueError, IndexError):
                    pass
            
            # Method 2: tokens array
            if prob == 0:
                tokens = market.get('tokens', [])
                if isinstance(tokens, list):
                    for t in tokens:
                        if isinstance(t, dict) and t.get('outcome', '').lower() == 'yes':
                            prob = float(t.get('price', 0) or 0)
                            break
                    if prob == 0 and tokens and isinstance(tokens[0], dict):
                        prob = float(tokens[0].get('price', 0) or 0)
            
            # Method 3: bestBid/bestAsk midpoint
            if prob == 0:
                bb = market.get('bestBid')
                ba = market.get('bestAsk')
                if bb and ba:
                    try:
                        prob = (float(bb) + float(ba)) / 2
                    except (ValueError, TypeError):
                        pass
            
            if prob > 0:
                self._markets[slug] = {
                    'question': market.get('question', ''),
                    'prob': prob,
                    'volume': float(market.get('volume', 0) or 0),
                }
        
        if self._markets:
            logger.info(f"[Polymarket] Found {len(self._markets)} relevant markets:")
            for slug, info in list(self._markets.items())[:5]:
                logger.debug(f"  {info['prob']:.0%} | {info['question'][:60]}")
    
    def health_check(self) -> bool:
        resp = self._api_get(f"{GAMMA_URL}/markets", params={'limit': 1, 'active': 'true'})
        ok = resp is not None and isinstance(resp, list)
        logger.info(f"[Polymarket] Health check {'OK' if ok else 'FAILED'}")
        return ok
