"""
services/signal_tracker.py — Signal Performance Tracker v2
═══════════════════════════════════════════════════════════
Records signals with BTC price at time of generation.
Evaluates accuracy at 1h, 4h, and 24h windows.
Provides win rate, avg return, and Sharpe proxy.

v2 CHANGES:
  - record() convenience method (extracts price from signal dict)
  - 24h evaluation window added
  - get_performance() returns aggregated stats
  - get_recent(n) returns last N signals with results
  - check_outcomes() generalized for all windows
  - EXPIRED tracking for bot-downtime windows
"""

import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger


class SignalTracker:
    """Track signal accuracy by comparing predicted direction vs actual price movement."""

    def __init__(self):
        self.pending_signals: List[Dict] = []
        self.completed_signals: List[Dict] = []
        self.max_completed = 500

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, signal: Dict) -> None:
        """
        Convenience wrapper for main.py — extracts price from signal dict.
        Falls back to record_signal() with explicit price if available.
        """
        price = (
            signal.get('entry_price')
            or signal.get('last_price')
            or signal.get('price_now')
            or signal.get('close')
            or 0
        )
        if price <= 0:
            logger.debug("[Tracker] No price in signal, skipping record")
            return
        self.record_signal(signal, float(price))

    def record_signal(self, signal: Dict, current_price: float) -> None:
        """Record a new signal with entry price for later evaluation."""
        direction = signal.get('direction', 'NEUTRAL')
        if direction in ('NEUTRAL', 'NO_SIGNAL'):
            return

        now = time.time()
        self.pending_signals.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'direction': direction,
            'score': signal.get('score', 0),
            'confidence': signal.get('confidence', ''),
            'entry_price': current_price,
            # Evaluation windows (check_time, expiry_time)
            'check_1h_time':  now + 3600,
            'expiry_1h_time': now + 3600 + 3600,
            'check_4h_time':  now + 14400,
            'expiry_4h_time': now + 14400 + 3600,
            'check_24h_time':  now + 86400,
            'expiry_24h_time': now + 86400 + 7200,
            # Results (filled by check_outcomes)
            'result_1h': None, 'pnl_1h': None, 'price_1h': None,
            'result_4h': None, 'pnl_4h': None, 'price_4h': None,
            'result_24h': None, 'pnl_24h': None, 'price_24h': None,
        })
        logger.info("[Tracker] Recorded {} @ ${:,.0f}", direction, current_price)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def check_outcomes(self, current_price: float) -> None:
        """
        Evaluate pending signals against current price.
        Each window: check_time <= now <= expiry_time -> evaluate.
        now > expiry_time -> EXPIRED (excluded from stats).
        """
        now = time.time()
        still_pending = []

        for sig in self.pending_signals:
            entry = sig['entry_price']
            direction = sig['direction']

            for window in ('1h', '4h', '24h'):
                result_key = f'result_{window}'
                if sig[result_key] is not None:
                    continue

                check_time = sig[f'check_{window}_time']
                expiry_time = sig[f'expiry_{window}_time']

                if now > expiry_time:
                    sig[result_key] = 'EXPIRED'
                    sig[f'pnl_{window}'] = None
                    sig[f'price_{window}'] = None
                elif now >= check_time:
                    pnl_pct = ((current_price - entry) / entry) * 100
                    if direction in ('BEARISH', 'SHORT'):
                        pnl_pct = -pnl_pct
                    sig[result_key] = 'HIT' if pnl_pct > 0 else 'MISS'
                    sig[f'pnl_{window}'] = round(pnl_pct, 3)
                    sig[f'price_{window}'] = current_price
                    logger.info(
                        "[Tracker] {} outcome: {} {} ({:+.2f}%)",
                        window, direction, sig[result_key], pnl_pct,
                    )

            # Completed when all three windows resolved
            all_done = all(
                sig[f'result_{w}'] is not None for w in ('1h', '4h', '24h')
            )
            if all_done:
                self.completed_signals.append(sig)
                if len(self.completed_signals) > self.max_completed:
                    self.completed_signals = self.completed_signals[-self.max_completed:]
            else:
                still_pending.append(sig)

        self.pending_signals = still_pending

    def evaluate(self, hours: int = 24) -> Dict:
        """Evaluate signals that have completed the given window."""
        window = '24h' if hours >= 24 else f'{hours}h'
        all_sigs = self.completed_signals + self.pending_signals
        evaluated = [s for s in all_sigs if s.get(f'result_{window}') in ('HIT', 'MISS')]
        hits = sum(1 for s in evaluated if s[f'result_{window}'] == 'HIT')
        pnls = [s[f'pnl_{window}'] for s in evaluated if s.get(f'pnl_{window}') is not None]
        return {
            'window': window,
            'total': len(evaluated),
            'hits': hits,
            'win_rate': (hits / len(evaluated) * 100) if evaluated else 0,
            'avg_pnl': (sum(pnls) / len(pnls)) if pnls else 0,
        }

    # ------------------------------------------------------------------
    # Performance summary
    # ------------------------------------------------------------------

    def get_performance(self) -> Dict:
        """Aggregated performance stats for dashboard /api/performance."""
        all_sigs = self.completed_signals + [
            s for s in self.pending_signals if s.get('result_1h') is not None
        ]
        if not all_sigs:
            return {
                'win_rate_1h': 0, 'win_rate_4h': 0, 'win_rate_24h': 0,
                'avg_return_1h': 0, 'avg_return_4h': 0, 'avg_return_24h': 0,
                'sharpe_proxy_24h': 0,
                'total_signals': 0, 'pending': len(self.pending_signals),
            }

        stats = {}
        for window in ('1h', '4h', '24h'):
            evaluated = [s for s in all_sigs if s.get(f'result_{window}') in ('HIT', 'MISS')]
            hits = sum(1 for s in evaluated if s[f'result_{window}'] == 'HIT')
            pnls = [s[f'pnl_{window}'] for s in evaluated if s.get(f'pnl_{window}') is not None]
            stats[f'win_rate_{window}'] = round(hits / len(evaluated) * 100, 1) if evaluated else 0
            stats[f'avg_return_{window}'] = round(sum(pnls) / len(pnls), 3) if pnls else 0

        # Sharpe proxy: mean_return / std_return * sqrt(365)
        pnls_24h = [s['pnl_24h'] for s in all_sigs if s.get('pnl_24h') is not None]
        if len(pnls_24h) >= 3:
            mean_r = sum(pnls_24h) / len(pnls_24h)
            var_r = sum((p - mean_r) ** 2 for p in pnls_24h) / len(pnls_24h)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-6
            stats['sharpe_proxy_24h'] = round(mean_r / std_r * math.sqrt(365), 2)
        else:
            stats['sharpe_proxy_24h'] = 0

        stats['total_signals'] = len(all_sigs)
        stats['pending'] = len(self.pending_signals)
        return stats

    def get_recent(self, n: int = 20) -> List[Dict]:
        """Last N signals with results for dashboard display."""
        all_sigs = self.completed_signals + self.pending_signals
        recent = sorted(all_sigs, key=lambda x: x['timestamp'], reverse=True)[:n]
        return [
            {
                'time': s['timestamp'],
                'direction': s['direction'],
                'score': s['score'],
                'confidence': s['confidence'],
                'entry_price': s['entry_price'],
                'price_1h': s.get('price_1h'),
                'price_4h': s.get('price_4h'),
                'price_24h': s.get('price_24h'),
                'result_1h': s.get('result_1h'),
                'result_4h': s.get('result_4h'),
                'result_24h': s.get('result_24h'),
                'pnl_1h': s.get('pnl_1h'),
                'pnl_4h': s.get('pnl_4h'),
                'pnl_24h': s.get('pnl_24h'),
            }
            for s in recent
        ]

    # ------------------------------------------------------------------
    # Legacy compat -- get_stats() still works for dashboard_state
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Backward-compatible stats dict for dashboard_state.update_tracker()."""
        perf = self.get_performance()
        return {
            'total': perf['total_signals'],
            'pending': perf['pending'],
            'hit_rate_1h': perf['win_rate_1h'],
            'hit_rate_4h': perf['win_rate_4h'],
            'hit_rate_24h': perf['win_rate_24h'],
            'avg_pnl_1h': perf['avg_return_1h'],
            'avg_pnl_4h': perf['avg_return_4h'],
            'avg_pnl_24h': perf['avg_return_24h'],
            'sharpe_proxy': perf['sharpe_proxy_24h'],
            'recent': self.get_recent(10),
        }
