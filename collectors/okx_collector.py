"""
OKX Collector v4 — Enhanced with free alternatives for CoinGlass data
═════════════════════════════════════════════════════════════════════
v4: Spot CVD + Basis Spread + Liquidation data (all free from OKX)
"""

import ccxt
import time
import numpy as np
from typing import Dict, Any
from loguru import logger
from collectors.base import BaseCollector


class OKXCollector(BaseCollector):

    def __init__(self, config: Dict):
        super().__init__("OKX", config)
        api_keys = config.get('api_keys', {}).get('okx', {})
        self.exchange = ccxt.okx({
            'apiKey': api_keys.get('api_key', ''),
            'secret': api_keys.get('secret_key', ''),
            'password': api_keys.get('passphrase', ''),
            'options': {'defaultType': 'swap'},
        })
        self.symbol = 'BTC/USDT:USDT'
        self.inst_id = 'BTC-USDT-SWAP'
        self.spot_symbol = 'BTC/USDT'
        self.ccy = 'BTC'
        self._prev_oi = 0

    def fetch(self) -> Dict[str, Any]:
        data = {}
        data.update(self._fetch_ohlcv())
        data.update(self._fetch_orderbook())
        data.update(self._fetch_ticker())
        data.update(self._fetch_open_interest())
        data.update(self._fetch_funding_rate())
        data.update(self._fetch_long_short_ratio())
        data.update(self._fetch_cvd_from_trades())
        data.update(self._fetch_spot_cvd())
        data.update(self._fetch_basis_spread())
        data.update(self._fetch_liquidations())
        logger.debug("[OKX] Fetched {} fields", len(data))
        return data

    def _fetch_ohlcv(self) -> Dict[str, Any]:
        data = {}
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '5m', limit=3)
            if ohlcv and len(ohlcv) >= 2:
                candle = ohlcv[-2]
                data['timestamp'] = candle[0]
                data['open'] = candle[1]
                data['high'] = candle[2]
                data['low'] = candle[3]
                data['close'] = candle[4]
                data['volume'] = candle[5]
                if candle[1] > 0:
                    data['close_pct_5m'] = ((candle[4] - candle[1]) / candle[1]) * 100
                data['price_now'] = ohlcv[-1][4]
                if len(ohlcv) >= 3 and ohlcv[-3][5] > 0:
                    data['volume_change_pct'] = ((candle[5] - ohlcv[-3][5]) / ohlcv[-3][5]) * 100
        except Exception as e:
            logger.error("[OKX] OHLCV: {}", e)
        return data

    def _fetch_orderbook(self) -> Dict[str, Any]:
        data = {}
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=50)
            if ob:
                bids = ob.get('bids', [])
                asks = ob.get('asks', [])
                if bids and asks:
                    best_bid = bids[0][0]
                    best_ask = asks[0][0]
                    data['spread_bps'] = ((best_ask - best_bid) / best_bid) * 10000
                    bid_vol = sum(b[0] * b[1] for b in bids[:20])
                    ask_vol = sum(a[0] * a[1] for a in asks[:20])
                    total = bid_vol + ask_vol
                    if total > 0:
                        data['ob_imbalance_raw'] = (bid_vol - ask_vol) / total
                    bid_total = sum(b[0] * b[1] for b in bids)
                    ask_total = sum(a[0] * a[1] for a in asks)
                    big_total = bid_total + ask_total
                    if big_total > 0:
                        data['bid_wall_pct'] = bid_total / big_total * 100
                        data['ask_wall_pct'] = ask_total / big_total * 100
        except Exception as e:
            logger.error("[OKX] Orderbook: {}", e)
        return data

    def _fetch_ticker(self) -> Dict[str, Any]:
        data = {}
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            if ticker:
                data['last_price'] = ticker.get('last', 0)
                data['volume_24h'] = ticker.get('baseVolume', 0)
                data['price_change_24h'] = ticker.get('percentage', 0)
                info = ticker.get('info', {})
                if isinstance(info, dict):
                    mp = info.get('markPx') or info.get('markPrice')
                    ip = info.get('idxPx') or info.get('indexPrice')
                    if mp: data['mark_price'] = float(mp)
                    if ip: data['index_price'] = float(ip)
        except Exception as e:
            logger.error("[OKX] Ticker: {}", e)
        return data

    def _fetch_open_interest(self) -> Dict[str, Any]:
        data = {}
        try:
            resp = self._api_get(
                'https://www.okx.com/api/v5/public/open-interest',
                params={'instType': 'SWAP', 'instId': self.inst_id}
            )
            if resp and resp.get('data'):
                oi_list = resp['data']
                if isinstance(oi_list, list) and len(oi_list) > 0:
                    oi_data = oi_list[0]
                    oi_val = float(oi_data.get('oi', 0))
                    oi_ccy = float(oi_data.get('oiCcy', 0))
                    data['oi_value'] = oi_val
                    data['oi_ccy'] = oi_ccy
                    if self._prev_oi > 0:
                        data['oi_change'] = oi_val - self._prev_oi
                        data['oi_change_pct'] = ((oi_val - self._prev_oi) / self._prev_oi * 100)
                    self._prev_oi = oi_val
        except Exception as e:
            logger.error("[OKX] OI: {}", e)
        try:
            resp = self._api_get(
                'https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume',
                params={'ccy': self.ccy, 'period': '5m'}
            )
            if resp and resp.get('code') == '0' and resp.get('data'):
                records = resp['data']
                if isinstance(records, list) and len(records) >= 2:
                    latest = records[0]
                    prev = records[1]
                    try:
                        oi_now = float(latest[1]) if isinstance(latest, (list, tuple)) else float(latest.get('oi', 0))
                        oi_prev = float(prev[1]) if isinstance(prev, (list, tuple)) else float(prev.get('oi', 0))
                        if oi_prev > 0:
                            data['oi_change_pct'] = (oi_now - oi_prev) / oi_prev * 100
                    except (ValueError, IndexError, TypeError):
                        pass
        except Exception as e:
            logger.debug("[OKX] OI history: {}", e)
        return data

    def _fetch_funding_rate(self) -> Dict[str, Any]:
        data = {}
        try:
            resp = self._api_get(
                'https://www.okx.com/api/v5/public/funding-rate',
                params={'instId': self.inst_id}
            )
            if resp and resp.get('data'):
                fr_list = resp['data']
                if isinstance(fr_list, list) and len(fr_list) > 0:
                    fr = fr_list[0]
                    data['funding_rate'] = float(fr.get('fundingRate', 0))
                    nfr = fr.get('nextFundingRate', '')
                    data['next_funding_rate'] = float(nfr) if nfr else 0
        except Exception as e:
            logger.error("[OKX] Funding: {}", e)
        try:
            resp = self._api_get(
                'https://www.okx.com/api/v5/public/funding-rate-history',
                params={'instId': self.inst_id, 'limit': '10'}
            )
            if resp and resp.get('data'):
                rates = []
                for r in resp['data']:
                    try:
                        rates.append(float(r.get('fundingRate', 0) if isinstance(r, dict) else r[1]))
                    except (ValueError, IndexError, TypeError):
                        continue
                if rates:
                    data['funding_rate_avg_8h'] = float(np.mean(rates[:3]))
                    data['funding_rate_max'] = float(max(rates))
                    data['funding_rate_min'] = float(min(rates))
        except Exception as e:
            logger.error("[OKX] Funding history: {}", e)
        return data

    def _fetch_long_short_ratio(self) -> Dict[str, Any]:
        data = {}
        try:
            resp = self._api_get(
                'https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio',
                params={'ccy': self.ccy, 'period': '1H'}
            )
            if resp and resp.get('code') == '0' and resp.get('data'):
                records = resp['data']
                if isinstance(records, list) and len(records) >= 1:
                    latest = records[0]
                    try:
                        if isinstance(latest, (list, tuple)):
                            ratio = float(latest[1])
                        elif isinstance(latest, dict):
                            ratio = float(latest.get('ratio', 1.0))
                        else:
                            ratio = 1.0
                        data['ls_ratio'] = ratio
                        data['ls_long_pct'] = ratio / (1 + ratio)
                        data['ls_short_pct'] = 1 / (1 + ratio)
                        if len(records) >= 2:
                            prev = records[1]
                            prev_ratio = float(prev[1]) if isinstance(prev, (list, tuple)) else float(prev.get('ratio', ratio))
                            data['ls_ratio_delta'] = ratio - prev_ratio
                    except (ValueError, IndexError, TypeError) as e:
                        logger.debug("[OKX] L/S parse: {}", e)
        except Exception as e:
            logger.error("[OKX] L/S ratio: {}", e)
        return data

    def _fetch_cvd_from_trades(self) -> Dict[str, Any]:
        data = {}
        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=500)
            if trades:
                buy_vol = sum(t['amount'] for t in trades if t['side'] == 'buy')
                sell_vol = sum(t['amount'] for t in trades if t['side'] == 'sell')
                data['cvd_perp'] = float(buy_vol - sell_vol)
                total = buy_vol + sell_vol
                data['cvd_buy_ratio'] = float(buy_vol / total) if total > 0 else 0.5
                if len(trades) >= 10:
                    mid = len(trades) // 2
                    h1 = sum(t['amount'] * (1 if t['side'] == 'buy' else -1) for t in trades[:mid])
                    h2 = sum(t['amount'] * (1 if t['side'] == 'buy' else -1) for t in trades[mid:])
                    data['cvd_5m_delta'] = float(h2 - h1)
        except Exception as e:
            logger.error("[OKX] CVD perp: {}", e)
        return data

    # ══ v4 NEW: FREE replacements for CoinGlass ($79/mo) ══

    def _fetch_spot_cvd(self) -> Dict[str, Any]:
        data = {}
        try:
            trades = self.exchange.fetch_trades(self.spot_symbol, limit=300)
            if trades:
                buy_vol = sum(t['amount'] for t in trades if t['side'] == 'buy')
                sell_vol = sum(t['amount'] for t in trades if t['side'] == 'sell')
                data['cvd_spot'] = float(buy_vol - sell_vol)
        except Exception as e:
            logger.debug("[OKX] Spot CVD: {}", str(e)[:80])
        return data

    def _fetch_basis_spread(self) -> Dict[str, Any]:
        data = {}
        try:
            resp = self._api_get(
                'https://www.okx.com/api/v5/public/mark-price',
                params={'instType': 'SWAP', 'instId': self.inst_id}
            )
            if resp and resp.get('data'):
                mark = float(resp['data'][0].get('markPx', 0))
                idx_resp = self._api_get(
                    'https://www.okx.com/api/v5/market/index-tickers',
                    params={'instId': 'BTC-USDT'}
                )
                idx_price = 0
                if idx_resp and idx_resp.get('data'):
                    idx_price = float(idx_resp['data'][0].get('idxPx', 0))
                if mark > 0 and idx_price > 0:
                    data['basis_spread_pct'] = round((mark - idx_price) / idx_price * 100, 6)
                    data['basis_spread'] = round(mark - idx_price, 2)
        except Exception as e:
            logger.debug("[OKX] Basis: {}", str(e)[:80])
        return data

    def _fetch_liquidations(self) -> Dict[str, Any]:
        data = {}
        try:
            resp = self._api_get(
                'https://www.okx.com/api/v5/public/liquidation-orders',
                params={'instType': 'SWAP', 'instId': self.inst_id, 'state': 'filled', 'limit': '100'}
            )
            if resp and resp.get('data'):
                long_vol = 0.0
                short_vol = 0.0
                for item in resp['data']:
                    sub = item.get('details', [])
                    if not sub and isinstance(item, dict): sub = [item]
                    for d in sub:
                        side = d.get('side', '').lower()
                        sz = float(d.get('sz', 0))
                        px = float(d.get('bkPx', 0) or d.get('px', 0))
                        usd_val = sz * px
                        if side == 'sell': long_vol += usd_val
                        elif side == 'buy': short_vol += usd_val
                total = long_vol + short_vol
                if total > 0:
                    data['liq_long_vol'] = round(long_vol, 2)
                    data['liq_short_vol'] = round(short_vol, 2)
                    data['liq_total'] = round(total, 2)
                    data['liq_imbalance'] = round((long_vol - short_vol) / total, 4)
        except Exception as e:
            logger.debug("[OKX] Liquidations: {}", str(e)[:80])
        return data

    def health_check(self) -> bool:
        try:
            self.exchange.fetch_ticker(self.symbol)
            logger.info("[OKX] Health check OK")
            return True
        except Exception as e:
            logger.error("[OKX] Health check FAILED: {}", e)
            return False
