"""
ETF Flow Collector - Fixed: uses requests directly for HTML scraping.
"""

import re
import requests
from datetime import datetime, timezone
from typing import Dict, Any
from loguru import logger
from collectors.base import BaseCollector


class ETFFlowCollector(BaseCollector):
    
    def __init__(self, config: Dict):
        super().__init__("ETFFlow", config)
        self._cached_flow = {}
        self._last_scrape_date = None
    
    def fetch(self) -> Dict[str, Any]:
        data = {}
        today = datetime.now(timezone.utc).date()
        
        if self._last_scrape_date == today and self._cached_flow:
            return self._cached_flow
        
        data = self._scrape_farside()
        
        if data:
            self._cached_flow = data
            self._last_scrape_date = today
        
        return data or self._cached_flow
    
    def _scrape_farside(self) -> Dict[str, Any]:
        """Scrape Farside Investors BTC ETF flow."""
        data = {}
        try:
            r = requests.get(
                'https://farside.co.uk/btc/',
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml',
                },
                timeout=15,
            )
            
            if r.status_code != 200:
                logger.warning(f"[ETFFlow] Farside returned {r.status_code}")
                return data
            
            html = r.text
            
            # Find all table cells with numeric values (ETF flows in millions)
            # Pattern matches positive numbers, negative numbers, and decimals
            numbers = re.findall(r'<td[^>]*>\s*([-−]?\d+\.?\d*)\s*</td>', html)
            
            if not numbers:
                # Try alternative pattern
                numbers = re.findall(r'>([-−]?\d+\.?\d*)</', html)
            
            recent_flows = []
            for n in numbers:
                try:
                    # Handle unicode minus sign
                    n = n.replace('−', '-')
                    val = float(n)
                    # ETF flows are typically -2000 to +2000 million range
                    if abs(val) < 5000:
                        recent_flows.append(val)
                except ValueError:
                    continue
            
            if recent_flows:
                # Take last entries as most recent
                data['etf_net_flow_daily'] = recent_flows[-1] * 1_000_000
                
                if len(recent_flows) >= 7:
                    data['etf_net_flow_7d'] = sum(recent_flows[-7:]) * 1_000_000
                else:
                    data['etf_net_flow_7d'] = sum(recent_flows) * 1_000_000
                
                logger.info(
                    f"[ETFFlow] daily={data.get('etf_net_flow_daily', 0)/1e6:.0f}M, "
                    f"7d={data.get('etf_net_flow_7d', 0)/1e6:.0f}M"
                )
            else:
                logger.warning("[ETFFlow] No numeric data found in Farside page")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"[ETFFlow] Farside request failed: {e}")
        except Exception as e:
            logger.error(f"[ETFFlow] Farside parse failed: {e}")
        
        return data
    
    def health_check(self) -> bool:
        try:
            r = requests.get(
                'https://farside.co.uk/btc/',
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            ok = r.status_code == 200
            logger.info(f"[ETFFlow] Health check {'OK' if ok else f'FAILED ({r.status_code})'}")
            return ok
        except Exception as e:
            logger.error(f"[ETFFlow] Health check FAILED: {e}")
            return False
