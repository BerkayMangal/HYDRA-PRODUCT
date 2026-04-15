"""
DeFiLlama Collector - 100% FREE
1. Stablecoin supply (USDT, USDC) - mint/burn as leading indicator
2. DeFi yields - top pool APYs on major chains
3. Protocol TVL - AAVE, Uniswap, etc.

API: https://defillama.com/docs/api - no key needed
"""

import requests
from typing import Dict, Any, List
from loguru import logger
from collectors.base import BaseCollector


LLAMA_BASE = "https://api.llama.fi"
YIELDS_BASE = "https://yields.llama.fi"
STABLES_BASE = "https://stablecoins.llama.fi"


class DeFiLlamaCollector(BaseCollector):
    """DeFiLlama data - stablecoins, yields, TVL. All free."""
    
    # Chains we care about
    TARGET_CHAINS = ['Ethereum', 'Base', 'Polygon', 'Arbitrum', 'Optimism']
    
    # Bluechip lending protocols only
    TARGET_PROTOCOLS = ['aave-v3', 'compound-v3', 'morpho', 'spark',
                        'maker', 'sky', 'fluid']
    
    # Bluechip pool filter: only these projects for yield display
    BLUECHIP_PROJECTS = ['aave-v3', 'compound-v3', 'morpho', 'morpho-blue',
                         'spark', 'fluid', 'maker']
    
    # Stablecoins
    STABLES = {
        'tether': 'USDT',
        'usd-coin': 'USDC',
        'dai': 'DAI',
    }
    
    def __init__(self, config: Dict):
        super().__init__("DeFiLlama", config)
        self._yields_cache = {}
        self._tvl_cache = {}
        self._stables_cache = {}
    
# fetch moved below
    
    def _fetch_stablecoins(self) -> Dict[str, Any]:
        """USDT/USDC total supply + recent change.
        
        Big USDT mint = buying power incoming = bullish leading indicator.
        Big USDC burn = capital leaving = bearish.
        """
        data = {}
        try:
            resp = self._api_get(f"{STABLES_BASE}/stablecoins?includePrices=false")
            
            if resp and resp.get('peggedAssets'):
                for stable in resp['peggedAssets']:
                    slug = stable.get('gecko_id', '') or stable.get('name', '').lower()
                    
                    for target_slug, symbol in self.STABLES.items():
                        if target_slug in slug.lower() or symbol.lower() in stable.get('symbol', '').lower():
                            peg_data = stable.get('circulating', {}).get('peggedUSD', 0)
                            
                            if peg_data:
                                data[f'stable_{symbol.lower()}_supply'] = float(peg_data)
                            
                            # 7d change
                            chain_circ = stable.get('chainCirculating', {})
                            
                            # Try to get total across chains
                            total = 0
                            for chain_data in chain_circ.values():
                                if isinstance(chain_data, dict):
                                    current = chain_data.get('current', {}).get('peggedUSD', 0)
                                    if current:
                                        total += float(current)
                            
                            if total > 0:
                                data[f'stable_{symbol.lower()}_total'] = total
                            
                            break
                
                # Total stablecoin market cap
                total_stable_mcap = sum(
                    float(s.get('circulating', {}).get('peggedUSD', 0) or 0)
                    for s in resp['peggedAssets']
                )
                data['total_stablecoin_mcap'] = total_stable_mcap
                
                if total_stable_mcap > 0:
                    data['total_stablecoin_mcap_b'] = total_stable_mcap / 1e9
                    logger.debug(f"[DeFiLlama] Total stablecoin: ${total_stable_mcap/1e9:.1f}B")
                    
        except Exception as e:
            logger.error(f"[DeFiLlama] Stablecoins: {e}")
        return data
    
    def _fetch_top_yields(self) -> Dict[str, Any]:
        """Bluechip USDC/USDT pools only - AAVE, Compound, Morpho, Spark.
        
        Filters: TVL > $10M, target chains only, lending protocols only.
        Includes 30d average APY from DeFiLlama.
        """
        data = {}
        bluechip_pools = []
        
        try:
            resp = self._api_get(f"{YIELDS_BASE}/pools")
            
            if resp and resp.get('data'):
                for pool in resp['data']:
                    chain = pool.get('chain', '')
                    project = pool.get('project', '')
                    symbol = (pool.get('symbol', '') or '').upper()
                    apy = pool.get('apy', 0)
                    apy_mean30d = pool.get('apyMean30d', 0)
                    tvl = pool.get('tvlUsd', 0)
                    
                    if not apy or not tvl:
                        continue
                    
                    # Bluechip filter
                    if project not in self.BLUECHIP_PROJECTS:
                        continue
                    if chain not in self.TARGET_CHAINS:
                        continue
                    if tvl < 10_000_000:  # $10M minimum
                        continue
                    
                    # Stablecoin pools only
                    is_stable = any(s in symbol for s in ['USDC', 'USDT', 'DAI', 'USDS', 'GHO', 'PYUSD', 'USDE'])
                    if not is_stable:
                        continue
                    
                    bluechip_pools.append({
                        'chain': chain,
                        'project': project,
                        'symbol': symbol,
                        'apy': round(float(apy), 2),
                        'apy_30d': round(float(apy_mean30d or apy), 2),
                        'tvl': float(tvl),
                        'tvl_m': round(float(tvl) / 1e6, 0),
                        'pool_id': pool.get('pool', ''),
                    })
                
                # Sort by TVL (biggest = most trusted)
                bluechip_pools.sort(key=lambda x: x['tvl'], reverse=True)
                
                data['defi_bluechip_pools'] = bluechip_pools[:25]
                data['defi_stable_yields'] = bluechip_pools[:25]  # backward compat
                
                # Summary
                if bluechip_pools:
                    data['defi_avg_stable_apy'] = round(
                        sum(p['apy'] for p in bluechip_pools[:10]) / min(len(bluechip_pools), 10), 2
                    )
                    data['defi_max_stable_apy'] = max(p['apy'] for p in bluechip_pools)
                    
                    # Best yield per chain
                    best_by_chain = {}
                    for p in bluechip_pools:
                        c = p['chain']
                        if c not in best_by_chain or p['apy'] > best_by_chain[c]['apy']:
                            best_by_chain[c] = p
                    data['defi_best_by_chain'] = best_by_chain
                
                logger.info(f"[DeFiLlama] {len(bluechip_pools)} bluechip stablecoin pools")
                    
        except Exception as e:
            logger.error(f"[DeFiLlama] Yields: {e}")
        return data
    
    def _fetch_protocol_tvls(self) -> Dict[str, Any]:
        """TVL for major DeFi protocols (AAVE, Uniswap, etc.)."""
        data = {}
        protocol_tvls = []
        
        try:
            resp = self._api_get(f"{LLAMA_BASE}/protocols")
            
            if resp and isinstance(resp, list):
                for protocol in resp:
                    slug = protocol.get('slug', '')
                    name = protocol.get('name', '')
                    
                    if slug in self.TARGET_PROTOCOLS or name.lower() in [p.split('-')[0] for p in self.TARGET_PROTOCOLS]:
                        tvl = protocol.get('tvl', 0)
                        change_1d = protocol.get('change_1d', 0)
                        change_7d = protocol.get('change_7d', 0)
                        
                        if tvl:
                            protocol_tvls.append({
                                'name': name,
                                'slug': slug,
                                'tvl': float(tvl),
                                'tvl_b': round(float(tvl) / 1e9, 2),
                                'change_1d': round(float(change_1d or 0), 2),
                                'change_7d': round(float(change_7d or 0), 2),
                                'chains': protocol.get('chains', []),
                                'category': protocol.get('category', ''),
                            })
                
                protocol_tvls.sort(key=lambda x: x['tvl'], reverse=True)
                data['defi_protocol_tvls'] = protocol_tvls
                
                # Total TVL of tracked protocols
                total = sum(p['tvl'] for p in protocol_tvls)
                data['defi_tracked_tvl_b'] = round(total / 1e9, 1)
                
        except Exception as e:
            logger.error(f"[DeFiLlama] Protocol TVLs: {e}")
        return data
    
    def _fetch_chain_tvls(self) -> Dict[str, Any]:
        """TVL per chain - Ethereum, Base, Polygon, etc."""
        data = {}
        chain_tvls = []
        
        try:
            resp = self._api_get(f"{LLAMA_BASE}/v2/chains")
            
            if resp and isinstance(resp, list):
                for chain in resp:
                    name = chain.get('name', '')
                    if name in self.TARGET_CHAINS + ['Solana', 'BSC', 'Avalanche']:
                        tvl = chain.get('tvl', 0)
                        if tvl:
                            chain_tvls.append({
                                'name': name,
                                'tvl': float(tvl),
                                'tvl_b': round(float(tvl) / 1e9, 2),
                            })
                
                chain_tvls.sort(key=lambda x: x['tvl'], reverse=True)
                data['defi_chain_tvls'] = chain_tvls
                
        except Exception as e:
            logger.error(f"[DeFiLlama] Chain TVLs: {e}")
        return data
    
    def _fetch_eth_staking_yields(self) -> Dict[str, Any]:
        """ETH staking yields - Lido, Coinbase, Rocket Pool, ether.fi."""
        data = {}
        eth_staking = []
        ETH_PROTOCOLS = ['lido', 'coinbase-wrapped-staked-eth', 'rocket-pool', 'ether.fi-stake',
                         'stader', 'kelp-dao', 'stakewise-v3']
        try:
            resp = self._api_get(f"{YIELDS_BASE}/pools")
            if resp and resp.get('data'):
                for pool in resp['data']:
                    project = pool.get('project', '')
                    symbol = (pool.get('symbol', '') or '').upper()
                    apy = pool.get('apy', 0)
                    apy30 = pool.get('apyMean30d', 0)
                    tvl = pool.get('tvlUsd', 0)
                    chain = pool.get('chain', '')
                    
                    if not apy or not tvl or tvl < 5_000_000:
                        continue
                    if project not in ETH_PROTOCOLS:
                        continue
                    if not any(s in symbol for s in ['STETH', 'CBETH', 'RETH', 'EETH', 'ETHX', 'RSETH', 'WETH']):
                        continue
                    
                    eth_staking.append({
                        'project': project, 'symbol': symbol, 'chain': chain,
                        'apy': round(float(apy), 2), 'apy_30d': round(float(apy30 or apy), 2),
                        'tvl': float(tvl), 'tvl_m': round(float(tvl) / 1e6, 0),
                    })
                
                eth_staking.sort(key=lambda x: x['tvl'], reverse=True)
                data['defi_eth_staking'] = eth_staking[:10]
                
                if eth_staking:
                    # ETH benchmark = TVL-weighted avg of top 4
                    top4 = eth_staking[:4]
                    total_tvl = sum(p['tvl'] for p in top4)
                    if total_tvl > 0:
                        data['eth_benchmark_apy'] = round(
                            sum(p['apy'] * p['tvl'] for p in top4) / total_tvl, 2)
                        data['eth_benchmark_apy_30d'] = round(
                            sum(p['apy_30d'] * p['tvl'] for p in top4) / total_tvl, 2)
                    
                    logger.info(f"[DeFiLlama] {len(eth_staking)} ETH staking pools, benchmark: {data.get('eth_benchmark_apy', 0):.2f}%")
        except Exception as e:
            logger.error(f"[DeFiLlama] ETH staking: {e}")
        return data
    
    def compute_benchmarks(self, data: Dict) -> Dict[str, Any]:
        """Compute yield benchmarks and risk scores."""
        result = {}
        
        # USD Benchmark: TVL-weighted avg of top bluechip pools
        pools = data.get('defi_bluechip_pools', [])
        if pools:
            top5 = pools[:5]
            total_tvl = sum(p['tvl'] for p in top5)
            if total_tvl > 0:
                result['usd_benchmark_apy'] = round(
                    sum(p['apy'] * p['tvl'] for p in top5) / total_tvl, 2)
                result['usd_benchmark_apy_30d'] = round(
                    sum(p['apy_30d'] * p['tvl'] for p in top5) / total_tvl, 2)
                result['usd_benchmark_tvl_b'] = round(total_tvl / 1e9, 2)
            
            # Risk scoring for each pool
            scored_pools = []
            for p in pools:
                risk = 'emerging'
                if p['tvl'] > 500_000_000:
                    risk = 'bluechip'
                elif p['tvl'] > 50_000_000:
                    risk = 'established'
                p['risk'] = risk
                scored_pools.append(p)
            result['defi_bluechip_pools'] = scored_pools
        
        return result

    def fetch(self) -> Dict[str, Any]:
        data = {}
        data.update(self._fetch_stablecoins())
        data.update(self._fetch_top_yields())
        data.update(self._fetch_eth_staking_yields())
        data.update(self._fetch_protocol_tvls())
        data.update(self._fetch_chain_tvls())
        
        # Compute benchmarks on top of raw data
        benchmarks = self.compute_benchmarks(data)
        data.update(benchmarks)
        
        if data:
            logger.info(f"[DeFiLlama] Fetched {len(data)} fields, USD bench: {data.get('usd_benchmark_apy', '?')}%")
        return data

    def health_check(self) -> bool:
        resp = self._api_get(f"{LLAMA_BASE}/v2/chains")
        ok = resp is not None and isinstance(resp, list)
        logger.info(f"[DeFiLlama] Health check {'OK' if ok else 'FAILED'}")
        return ok
