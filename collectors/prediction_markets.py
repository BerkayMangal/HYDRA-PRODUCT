"""
Prediction Markets — FINAL Working Version
Strategy: Fetch from BOTH /events AND /markets endpoints.
Events = geopolitics, macro. Markets = crypto, finance.
Relaxed keyword matching with scoring. Blacklist entertainment/sports.
"""

from typing import Dict, Any, Optional, List
import json as jsonlib
from loguru import logger
from collectors.base import BaseCollector

GAMMA_URL = "https://gamma-api.polymarket.com"

# Keyword categories — ANY single keyword match = include
# Higher weight = more important for our dashboard
CATEGORIES = {
    'geopolitics': {
        'keywords': ['iran', 'ceasefire', 'hormuz', 'nuclear', 'tehran', 'troops',
                      'strike', 'military action', 'regime', 'israel', 'war',
                      'invasion', 'ukraine', 'russia', 'china', 'taiwan', 'nato',
                      'sanctions', 'embargo', 'missile', 'drone'],
        'emoji_map': {
            'iran': '🔴', 'ceasefire': '🔴', 'hormuz': '🚢', 'nuclear': '☢️',
            'ukraine': '🇺🇦', 'russia': '🇺🇦', 'china': '🇨🇳', 'taiwan': '🇨🇳',
            'troops': '⚔️', 'strike': '⚔️', 'military': '⚔️', 'missile': '⚔️',
            'invasion': '⚔️', 'war': '⚔️', 'israel': '⚔️',
        },
        'default_emoji': '🌍',
    },
    'commodities': {
        'keywords': ['oil', 'crude', 'brent', 'wti', 'opec', 'gold', 'silver'],
        'emoji_map': {'oil': '🛢️', 'crude': '🛢️', 'brent': '🛢️', 'gold': '🥇'},
        'default_emoji': '🛢️',
    },
    'macro': {
        'keywords': ['fed ', 'rate cut', 'fomc', 'interest rate', 'federal reserve',
                      'inflation', 'cpi', 'recession', 'gdp', 'unemployment',
                      'tariff', 'trade war'],
        'emoji_map': {'fed': '📉', 'rate cut': '📉', 'inflation': '📈',
                       'recession': '📉', 'tariff': '🏗️'},
        'default_emoji': '📉',
    },
    'markets': {
        'keywords': ['s&p', 'sp500', 'spx', 'nasdaq', 'dow', 'all time high',
                      'stock market', 'bear market', 'bull market'],
        'emoji_map': {},
        'default_emoji': '📊',
    },
    'crypto': {
        'keywords': ['bitcoin', 'btc', 'ethereum', 'eth ', 'crypto', 'solana',
                      'defi', 'stablecoin'],
        'emoji_map': {'bitcoin': '₿', 'btc': '₿', 'ethereum': '⟠', 'crypto': '₿'},
        'default_emoji': '₿',
    },
    'politics': {
        'keywords': ['trump', 'election', 'congress', 'senate', 'governor',
                      'democrat', 'republican', 'biden'],
        'emoji_map': {},
        'default_emoji': '🏛️',
    },
    'tech': {
        'keywords': ['ai ', 'artificial intelligence', 'openai', 'anthropic',
                      'google ai', 'deepseek', 'spacex', 'tesla'],
        'emoji_map': {'spacex': '🚀', 'tesla': '🚀', 'musk': '🚀'},
        'default_emoji': '🤖',
    },
}

# Skip these completely
BLACKLIST = [
    'nhl', 'nba', 'nfl', 'mlb', 'nascar', 'ufc', 'boxing', 'mma',
    'premier league', 'champions league', 'serie a', 'la liga', 'bundesliga',
    'stanley cup', 'super bowl', 'world series', 'world cup',
    'gta', 'rihanna', 'taylor swift', 'kanye', 'drake', 'album',
    'oscar', 'grammy', 'emmy', 'bachelor', 'love island', 'survivor',
    'jesus christ', 'bitboy', 'convicted', 'weinstein', 'megaeth',
    'airdrop', 'meme coin', 'solana meme', 'pump.fun',
    'avalanche win', 'wild win', 'islanders win', 'rangers win',
    'lakers', 'celtics', 'warriors', 'yankees', 'dodgers',
    'touchdown', 'quarterback', 'pitcher', 'goalkeeper',
    'ballon d', 'mbappe', 'ronaldo', 'messi',
    'masters', 'golf', 'pga', 'tennis', 'open championship',
    'formula 1', 'f1 ', 'grand prix', 'winner 2026',
]


class PredictionMarketsCollector(BaseCollector):
    """Aggregates prediction markets from Polymarket."""

    def __init__(self, config: Dict):
        super().__init__("Predictions", config)
        self._cache: List[Dict] = []
        self._last_discovery = 0

    def fetch(self) -> Dict[str, Any]:
        import time
        data = {}
        now = time.time()

        # Rediscover every 30 minutes or if empty
        if not self._cache or (now - self._last_discovery > 1800):
            self._discover()
            self._last_discovery = now

        if not self._cache:
            return data

        events = []
        for i, m in enumerate(self._cache):
            prob = m['probability']
            slug = m.get('slug', f'evt{i}').replace('-', '_')[:40]
            data[f'poly_{slug}_prob'] = prob
            events.append(m)

        data['prediction_events'] = events
        logger.info(f"[Predictions] {len(events)} world events active")
        return data

    def _discover(self):
        """Fetch from both /events and /markets, merge results."""
        self._cache = []
        all_items = []

        # ===== ENDPOINT 1: /events (geopolitics, macro — the big stuff) =====
        for offset in [0, 100]:
            try:
                resp = self._api_get(
                    f"{GAMMA_URL}/events",
                    params={'closed': 'false', 'limit': 100, 'offset': offset}
                )
                if resp and isinstance(resp, list):
                    for event in resp:
                        title = (event.get('title', '') or '').strip()
                        slug = event.get('slug', '')
                        
                        # Collect ALL searchable text: title + slug + all market questions
                        search_texts = [title, slug]
                        markets = event.get('markets', [])
                        for mkt in (markets or []):
                            q = mkt.get('question', '') or mkt.get('title', '') or ''
                            if q:
                                search_texts.append(q)
                        
                        all_text = ' '.join(search_texts).lower()
                        
                        # Get probability from first market
                        prob = None
                        vol = 0
                        if markets:
                            # Find the most relevant market (highest volume)
                            best_mkt = max(markets, key=lambda x: float(x.get('volume', 0) or 0)) if markets else markets[0]
                            prob = self._extract_prob(best_mkt)
                            for vf in ['volume', 'volumeNum']:
                                try:
                                    vol = max(vol, float(best_mkt.get(vf, 0) or 0))
                                except:
                                    pass
                        
                        if prob is None:
                            prob = self._extract_prob(event)
                        
                        if prob is not None and title:
                            all_items.append({
                                'title': title,
                                'slug': slug,
                                'all_text': all_text,
                                'probability': prob,
                                'volume': vol,
                                'source': 'events',
                                'markets': markets,
                            })
                    
                    if len(resp) < 100:
                        break
            except Exception as e:
                logger.debug(f"[Predictions] Events offset={offset}: {e}")
                break

        logger.info(f"[Predictions] Events endpoint: {len(all_items)} items")

        # ===== ENDPOINT 2: /markets (crypto, smaller markets) =====
        try:
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
            if resp and isinstance(resp, list):
                mkt_count = 0
                for m in resp:
                    question = (m.get('question', '') or '').strip()
                    slug = m.get('slug', '')
                    if not question:
                        continue
                    
                    prob = self._extract_prob(m)
                    vol = 0
                    for vf in ['volume', 'volumeNum']:
                        try:
                            vol = max(vol, float(m.get(vf, 0) or 0))
                        except:
                            pass
                    
                    if prob is not None:
                        all_items.append({
                            'title': question,
                            'slug': slug,
                            'all_text': (question + ' ' + slug).lower(),
                            'probability': prob,
                            'volume': vol,
                            'source': 'markets',
                        })
                        mkt_count += 1
                logger.info(f"[Predictions] Markets endpoint: {mkt_count} items")
        except Exception as e:
            logger.debug(f"[Predictions] Markets: {e}")

        logger.info(f"[Predictions] Total raw: {len(all_items)}")

        # ===== FILTER & CATEGORIZE =====
        seen_slugs = set()
        processed = []

        for item in all_items:
            slug = item['slug']
            if slug in seen_slugs:
                continue

            prob = item['probability']
            if prob <= 0.02 or prob >= 0.98:
                continue

            text = item['all_text']

            # Blacklist check
            if any(bl in text for bl in BLACKLIST):
                continue

            # Category matching — find best category
            cat_name, emoji = self._categorize(text)
            if cat_name is None:
                continue  # No category match = skip

            seen_slugs.add(slug)

            title = item['title']
            # If event has markets, use the most descriptive market question
            markets = item.get('markets', [])
            if markets and len(title) < 20:
                # Event title too short, use market question
                best_q = max(markets, key=lambda x: len(x.get('question', '') or ''))
                q = best_q.get('question', '')
                if q and len(q) > len(title):
                    title = q

            name = title if len(title) <= 55 else title[:52] + '...'

            processed.append({
                'id': slug,
                'slug': slug,
                'emoji': emoji,
                'name': name,
                'probability': round(prob, 3),
                'volume': item['volume'],
                'category': cat_name,
                'source': 'polymarket',
            })

        logger.info(f"[Predictions] After categorize+filter: {len(processed)}")

        # Sort by volume, take top 10
        processed.sort(key=lambda x: x['volume'], reverse=True)
        self._cache = processed[:10]

        if self._cache:
            logger.info(f"[Predictions] FINAL top {len(self._cache)}:")
            for m in self._cache:
                logger.info(f"  {m['emoji']} {m['name'][:50]} → {m['probability']*100:.0f}% (${m['volume']/1e6:.1f}M vol)")
        else:
            logger.warning("[Predictions] No markets matched after filtering")

    def _categorize(self, text: str) -> tuple:
        """Match text against categories. Returns (category_name, emoji) or (None, None)."""
        best_cat = None
        best_score = 0
        best_emoji = None

        for cat_name, cat_info in CATEGORIES.items():
            score = 0
            matched_kw = None
            for kw in cat_info['keywords']:
                if kw in text:
                    score += 1
                    if matched_kw is None:
                        matched_kw = kw

            if score > best_score:
                best_score = score
                best_cat = cat_name
                # Find emoji
                best_emoji = cat_info['default_emoji']
                if matched_kw:
                    for ek, ev in cat_info.get('emoji_map', {}).items():
                        if ek in text:
                            best_emoji = ev
                            break

        if best_score >= 1:
            return best_cat, best_emoji
        return None, None

    def _extract_prob(self, m: Dict) -> Optional[float]:
        """Extract probability from market/event data."""
        # Method 1: outcomePrices
        op = m.get('outcomePrices')
        if op:
            try:
                prices = jsonlib.loads(op) if isinstance(op, str) else op
                if isinstance(prices, list) and len(prices) > 0:
                    p = float(prices[0])
                    if 0 < p < 1:
                        return p
            except:
                pass

        # Method 2: tokens array
        tokens = m.get('tokens', [])
        if tokens and isinstance(tokens, list):
            for tk in tokens:
                if isinstance(tk, dict) and (tk.get('outcome', '') or '').lower() == 'yes':
                    try:
                        p = float(tk.get('price', 0))
                        if 0 < p < 1:
                            return p
                    except:
                        pass
            if isinstance(tokens[0], dict):
                try:
                    p = float(tokens[0].get('price', 0))
                    if 0 < p < 1:
                        return p
                except:
                    pass

        # Method 3: bestBid/bestAsk midpoint
        bb, ba = m.get('bestBid'), m.get('bestAsk')
        if bb and ba:
            try:
                p = (float(bb) + float(ba)) / 2
                if 0 < p < 1:
                    return p
            except:
                pass

        return None

    def health_check(self) -> bool:
        resp = self._api_get(f"{GAMMA_URL}/markets", params={'limit': 1})
        ok = resp is not None and isinstance(resp, list)
        logger.info(f"[Predictions] Health check {'OK' if ok else 'FAILED'}")
        return ok
