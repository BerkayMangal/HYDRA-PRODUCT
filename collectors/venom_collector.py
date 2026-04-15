"""
VENOM Collector — Altcoin Intelligence
"Strike before the crowd."

Data from CoinGecko (FREE):
1. Top movers (gainers/losers with volume filter)
2. Trending coins (most searched)
3. Category/sector performance
4. Market breadth (green vs red)
5. Altcoin season indicator
"""

from typing import Dict, Any, List
import os
from loguru import logger
from collectors.base import BaseCollector


CG_BASE = "https://api.coingecko.com/api/v3"
def _cg_headers() -> Dict[str, str]:
    headers = {'accept': 'application/json'}
    api_key = os.getenv('COINGECKO_API_KEY', '').strip()
    if api_key:
        headers['x-cg-demo-api-key'] = api_key
    return headers

# FIX: don't cache at import time — read env var on every request
# CG_HEADERS = _cg_headers()  <-- removed, use _cg_headers() inline

# Sector mapping — group coins into narratives
SECTORS = {
    'AI & Compute': ['fetch-ai', 'render-token', 'bittensor', 'akash-network', 'ocean-protocol', 'singularitynet'],
    'Meme': ['dogecoin', 'shiba-inu', 'pepe', 'dogwifcoin', 'floki', 'bonk'],
    'Layer 2': ['arbitrum', 'optimism', 'polygon-ecosystem-token', 'starknet', 'mantle', 'immutable-x'],
    'DeFi Blue': ['aave', 'uniswap', 'maker', 'lido-dao', 'compound-governance-token', 'curve-dao-token'],
    'RWA': ['ondo-finance', 'mantra-dao', 'polymesh', 'centrifuge', 'maple'],
    'Layer 1': ['solana', 'avalanche-2', 'near', 'sui', 'aptos', 'sei-network', 'injective-protocol'],
    'Gaming': ['immutable-x', 'gala', 'the-sandbox', 'axie-infinity', 'illuvium'],
}


class VenomCollector(BaseCollector):
    """Altcoin intelligence — movers, trending, sectors, breadth."""

    def __init__(self, config: Dict):
        super().__init__("Venom", config)
        self._cache = {}

    def fetch(self, shared_coins: List = None) -> Dict[str, Any]:
        data = {}
        import time
        
        # Use shared coin data if available (from Sentiment collector)
        if shared_coins:
            logger.info(f"[Venom] Processing {len(shared_coins)} shared coins")
            data.update(self._process_coins(shared_coins))
        else:
            # Fallback: fetch our own (will hit CoinGecko)
            data.update(self._fetch_top_movers())
        
        # Trending: separate CoinGecko endpoint, less rate-limited
        time.sleep(4)
        data.update(self._fetch_trending())
        
        # Categories: skip if we're getting rate limited
        time.sleep(4)
        data.update(self._fetch_categories())
        
        data.update(self._fetch_breadth())
        data.update(self._fetch_hyperliquid())
        
        # ===== VENOM SCORE + ENTRY QUALITY =====
        gainers = data.get('venom_gainers', [])
        trending = data.get('venom_trending', [])
        hl = data.get('venom_hyperliquid', [])
        trending_syms = {t['symbol'] for t in trending}
        hl_map = {h['symbol']: h for h in hl}
        
        scored_coins = []
        for coin in gainers:
            sym = coin['symbol']
            score = 0
            reasons = []
            
            # Momentum (0-25): based on 24h change magnitude
            chg = abs(coin.get('change_24h', 0))
            momentum = min(25, chg * 2.5)
            score += momentum
            
            # Volume strength (0-20): volume relative to market cap
            vol = coin.get('volume', 0)
            mcap = coin.get('market_cap_b', 0) * 1e9
            if mcap > 0:
                vol_ratio = vol / mcap
                vol_score = min(20, vol_ratio * 100)
                score += vol_score
            
            # Trending heat (0-20): is it trending?
            if sym in trending_syms:
                score += 20
                reasons.append('Trending')
            
            # Breadth context (0-15): if broad rally, less impressive
            breadth = data.get('venom_breadth_pct', 50)
            if breadth > 80:
                score -= 5  # Everything is up, less special
            elif breadth < 40:
                score += 15  # Pumping against the market = strong
                reasons.append('Against trend')
            else:
                score += 7
            
            # Funding/positioning (0-20): from Hyperliquid
            hl_data = hl_map.get(sym, {})
            funding = hl_data.get('funding', 0)
            if funding > 0.05:
                score -= 10  # Crowded
                reasons.append('Crowded longs')
            elif funding < -0.02:
                score += 15  # Squeeze potential
                reasons.append('Squeeze setup')
            else:
                score += 10
            
            score = max(0, min(100, round(score)))
            
            # Entry quality label
            if funding > 0.05:
                entry = 'CROWDED'
                entry_color = 'red'
            elif chg > 15 and breadth > 70:
                entry = 'EXTENDED'
                entry_color = 'amber'
            elif chg < 5 and sym in trending_syms:
                entry = 'EARLY'
                entry_color = 'green'
            elif chg > 8 and funding < 0.01:
                entry = 'BREAKOUT'
                entry_color = 'blue'
            elif chg > 10:
                entry = 'EXTENDED'
                entry_color = 'amber'
            else:
                entry = 'EARLY'
                entry_color = 'green'
            
            coin['venom_score'] = score
            coin['entry_quality'] = entry
            coin['entry_color'] = entry_color
            coin['score_reasons'] = reasons
            scored_coins.append(coin)
        
        scored_coins.sort(key=lambda x: x['venom_score'], reverse=True)
        data['venom_scored'] = scored_coins[:10]
        
        # Cross-reference for sniper picks
        if data.get('venom_gainers') and data.get('venom_trending') and data.get('venom_hyperliquid'):
            trending_syms = {t['symbol'] for t in data.get('venom_trending', [])}
            gainer_syms = {g['symbol'] for g in data.get('venom_gainers', [])}
            hl_syms = {h['symbol'] for h in data.get('venom_hyperliquid', []) if h['signal'] != 'neutral'}
            
            # Triple overlap = strongest signal
            triple = trending_syms & gainer_syms & hl_syms
            double = (trending_syms & gainer_syms) | (trending_syms & hl_syms) | (gainer_syms & hl_syms)
            
            sniper_picks = []
            for sym in triple:
                sniper_picks.append({'symbol': sym, 'strength': 'triple', 'reasons': 'Trending + Pumping + HL Signal'})
            for sym in (double - triple):
                reasons = []
                if sym in trending_syms: reasons.append('Trending')
                if sym in gainer_syms: reasons.append('Pumping')
                if sym in hl_syms: reasons.append('HL Signal')
                sniper_picks.append({'symbol': sym, 'strength': 'double', 'reasons': ' + '.join(reasons)})
            
            data['venom_sniper_picks'] = sniper_picks[:5]
        
        if data:
            logger.info(f"[Venom] Fetched {len(data)} fields")
        return data

    def _process_coins(self, coin_list: List) -> Dict[str, Any]:
        """Process already-fetched coin data from Sentiment collector."""
        data = {}
        try:
            resp = coin_list
            if resp and isinstance(resp, list):
                # Filter: volume > $10M, exclude stablecoins
                stables = {'usdt', 'usdc', 'dai', 'usds', 'usde', 'tusd', 'busd', 'fdusd', 'pyusd', 'gusd'}
                
                coins = []
                for c in resp:
                    sym = (c.get('symbol', '') or '').lower()
                    if sym in stables:
                        continue
                    vol = float(c.get('total_volume', 0) or 0)
                    if vol < 10_000_000:
                        continue
                    
                    chg_1h = float(c.get('price_change_percentage_1h_in_currency', 0) or c.get('price_change_percentage_1h', 0) or 0)
                    chg_24h = float(c.get('price_change_percentage_24h', 0) or 0)
                    chg_7d = float(c.get('price_change_percentage_7d_in_currency', 0) or 0)
                    mcap = float(c.get('market_cap', 0) or 0)
                    
                    coins.append({
                        'symbol': (c.get('symbol', '') or '').upper(),
                        'name': c.get('name', ''),
                        'price': float(c.get('current_price', 0) or 0),
                        'change_1h': round(chg_1h, 2),
                        'change_24h': round(chg_24h, 2),
                        'change_7d': round(chg_7d, 2),
                        'volume': vol,
                        'volume_m': round(vol / 1e6, 1),
                        'market_cap_b': round(mcap / 1e9, 2),
                        'image': c.get('image', ''),
                        'id': c.get('id', ''),
                    })
                
                # Sort for gainers/losers
                by_change = sorted(coins, key=lambda x: x['change_24h'], reverse=True)
                
                # Exclude BTC/ETH/stables from movers
                movers = [c for c in by_change if c['symbol'] not in ('BTC', 'ETH', 'USDT', 'USDC')]
                
                gainers = [c for c in movers if c['change_24h'] > 0][:8]
                losers = [c for c in reversed(movers) if c['change_24h'] < 0][:8]
                
                data['venom_gainers'] = gainers
                data['venom_losers'] = losers
                data['venom_all_coins'] = coins
                
                # Market breadth
                green = sum(1 for c in coins if c['change_24h'] > 0)
                red = sum(1 for c in coins if c['change_24h'] < 0)
                data['venom_breadth_green'] = green
                data['venom_breadth_red'] = red
                data['venom_breadth_total'] = len(coins)
                data['venom_breadth_pct'] = round(green / max(len(coins), 1) * 100, 0)
                
                # Altcoin season: avg altcoin 24h vs BTC 24h
                btc_chg = next((c['change_24h'] for c in coins if c['symbol'] == 'BTC'), 0)
                alt_changes = [c['change_24h'] for c in coins if c['symbol'] not in ('BTC', 'ETH', 'USDT', 'USDC')]
                if alt_changes:
                    avg_alt = sum(alt_changes) / len(alt_changes)
                    data['venom_alt_avg_24h'] = round(avg_alt, 2)
                    data['venom_btc_24h'] = round(btc_chg, 2)
                    # Positive = alts outperforming BTC = altseason
                    data['venom_alt_season_score'] = round(avg_alt - btc_chg, 2)
                
                # Sector performance
                sector_perf = {}
                coin_map = {c['id']: c for c in coins}
                for sector, ids in SECTORS.items():
                    sector_coins = [coin_map[cid] for cid in ids if cid in coin_map]
                    if sector_coins:
                        avg = sum(c['change_24h'] for c in sector_coins) / len(sector_coins)
                        sector_perf[sector] = {
                            'change_24h': round(avg, 2),
                            'count': len(sector_coins),
                            'coins': [c['symbol'] for c in sector_coins],
                        }
                data['venom_sectors'] = sector_perf
                
                logger.info(f"[Venom] {len(coins)} coins, {green} green / {red} red, alt season: {data.get('venom_alt_season_score', 0):+.1f}")

        except Exception as e:
            logger.error(f"[Venom] Coin processing: {e}")
        
        return data

    def _fetch_top_movers(self) -> Dict[str, Any]:
        """Fallback: fetch from CoinGecko if no shared data."""
        try:
            import time
            time.sleep(3)  # Rate limit respect
            resp = self._api_get(
                f"{CG_BASE}/coins/markets",
                headers=_cg_headers(),
                params={
                    'vs_currency': 'usd',
                    'order': 'volume_desc',
                    'per_page': 100,
                    'page': 1,
                    'sparkline': 'false',
                    'price_change_percentage': '1h,24h,7d',
                }
            )
            if resp and isinstance(resp, list):
                return self._process_coins(resp)
        except Exception as e:
            logger.error(f"[Venom] Fallback movers: {e}")
        return {}

    def _fetch_trending(self) -> Dict[str, Any]:
        """CoinGecko trending — what everyone is searching for."""
        data = {}
        try:
            resp = self._api_get(f"{CG_BASE}/search/trending", headers=_cg_headers())
            
            if resp and resp.get('coins'):
                trending = []
                for item in resp['coins'][:10]:
                    coin = item.get('item', {})
                    trending.append({
                        'symbol': (coin.get('symbol', '') or '').upper(),
                        'name': coin.get('name', ''),
                        'market_cap_rank': coin.get('market_cap_rank', 0),
                        'price_btc': float(coin.get('price_btc', 0) or 0),
                        'score': coin.get('score', 0),
                        'image': coin.get('small', ''),
                        'id': coin.get('id', ''),
                    })
                
                data['venom_trending'] = trending
                logger.debug(f"[Venom] {len(trending)} trending coins")
                
        except Exception as e:
            logger.error(f"[Venom] Trending: {e}")
        
        return data

    def _fetch_categories(self) -> Dict[str, Any]:
        """Top performing categories/sectors."""
        data = {}
        try:
            resp = self._api_get(
                f"{CG_BASE}/coins/categories",
                headers=_cg_headers(),
                params={'order': 'market_cap_change_24h_desc'}
            )
            
            if resp and isinstance(resp, list):
                # Filter meaningful categories
                skip = ['stablecoins', 'wrapped-tokens', 'bridged', 'liquid-staking']
                cats = []
                for cat in resp:
                    cid = (cat.get('id', '') or '').lower()
                    if any(s in cid for s in skip):
                        continue
                    mcap_chg = float(cat.get('market_cap_change_24h', 0) or 0)
                    name = cat.get('name', '')
                    if name and abs(mcap_chg) > 0:
                        cats.append({
                            'name': name,
                            'change_24h': round(mcap_chg, 2),
                            'market_cap': float(cat.get('market_cap', 0) or 0),
                            'volume_24h': float(cat.get('total_volume', 0) or 0),
                        })
                
                # Top gainers and losers by category
                cats.sort(key=lambda x: x['change_24h'], reverse=True)
                data['venom_hot_categories'] = cats[:6]
                data['venom_cold_categories'] = list(reversed(cats))[:6]
                
                logger.debug(f"[Venom] {len(cats)} categories tracked")
                
        except Exception as e:
            logger.error(f"[Venom] Categories: {e}")
        
        return data

    def _fetch_breadth(self) -> Dict[str, Any]:
        """Already computed in _fetch_top_movers, this is for additional signals."""
        return {}

    def _fetch_hyperliquid(self) -> Dict[str, Any]:
        """Hyperliquid perpetuals — funding rates, OI, volume.
        Public API, no auth needed. https://api.hyperliquid.xyz/info
        """
        data = {}
        try:
            # Hyperliquid uses POST (not GET)
            import requests as req
            resp = req.post(
                'https://api.hyperliquid.xyz/info',
                json={"type": "metaAndAssetCtxs"},
                timeout=15
            )
            
            if resp and resp.status_code == 200:
                result = resp.json()
                
                # result is [meta, [assetCtxs]]
                if isinstance(result, list) and len(result) >= 2:
                    meta = result[0]
                    contexts = result[1]
                    universe = meta.get('universe', [])
                    
                    hl_signals = []
                    
                    for i, ctx in enumerate(contexts):
                        if i >= len(universe):
                            break
                        
                        coin = universe[i]
                        name = coin.get('name', '')
                        
                        funding = float(ctx.get('funding', 0) or 0)
                        oi = float(ctx.get('openInterest', 0) or 0)
                        vol_24h = float(ctx.get('dayNtlVlm', 0) or 0)
                        mark = float(ctx.get('markPx', 0) or 0)
                        
                        # Skip tiny markets
                        if vol_24h < 5_000_000 or oi < 1_000_000:
                            continue
                        
                        # Determine signal
                        signal = 'neutral'
                        signal_text = 'Normal'
                        signal_emoji = '⚪'
                        
                        funding_annualized = funding * 365 * 100  # Rough annualized %
                        
                        if funding > 0.0005:  # Very positive funding
                            signal = 'crowded_long'
                            signal_text = 'Crowded long — reversal risk'
                            signal_emoji = '🔴'
                        elif funding > 0.0002:
                            signal = 'long_bias'
                            signal_text = 'Longs paying — mild bullish crowd'
                            signal_emoji = '🟡'
                        elif funding < -0.0003:
                            signal = 'squeeze_setup'
                            signal_text = 'Shorts paying — squeeze setup'
                            signal_emoji = '🟢'
                        elif funding < -0.0001:
                            signal = 'short_bias'
                            signal_text = 'Mild short bias'
                            signal_emoji = '🔵'
                        
                        hl_signals.append({
                            'symbol': name,
                            'funding': round(funding * 100, 4),  # As percentage
                            'funding_annualized': round(funding_annualized, 1),
                            'open_interest': oi,
                            'oi_m': round(oi / 1e6, 1),
                            'volume_24h': vol_24h,
                            'volume_m': round(vol_24h / 1e6, 1),
                            'mark_price': mark,
                            'signal': signal,
                            'signal_text': signal_text,
                            'signal_emoji': signal_emoji,
                        })
                    
                    # Sort by volume
                    hl_signals.sort(key=lambda x: x['volume_24h'], reverse=True)
                    
                    # Only keep interesting ones (non-neutral or top volume)
                    interesting = [s for s in hl_signals if s['signal'] != 'neutral']
                    top_vol = hl_signals[:5]
                    
                    # Merge: interesting first, then top volume
                    seen = set()
                    final = []
                    for s in interesting + top_vol:
                        if s['symbol'] not in seen:
                            seen.add(s['symbol'])
                            final.append(s)
                    
                    data['venom_hyperliquid'] = final[:10]
                    
                    logger.info(f"[Venom] Hyperliquid: {len(hl_signals)} assets, {len(interesting)} with signals")
                    
        except Exception as e:
            logger.error(f"[Venom] Hyperliquid: {e}")
        
        return data

    def health_check(self) -> bool:
        # Use Hyperliquid for health check (CoinGecko shared with Sentiment)
        try:
            import requests as req
            resp = req.post('https://api.hyperliquid.xyz/info',
                json={"type": "metaAndAssetCtxs"}, timeout=10)
            ok = resp.status_code == 200
        except:
            ok = True  # Don't fail on health check — data comes from Sentiment
        logger.info(f"[Venom] Health check {'OK' if ok else 'FAILED'}")
        return ok
