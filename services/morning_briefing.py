"""
Morning Briefing Agent
Sends daily intelligence briefing to Slack at 7:00 AM London time.
Uses Claude API to generate natural language summary from HYDRA data.
Slack Incoming Webhook — no bot setup needed.

Setup:
1. Go to api.slack.com/apps → Create App → Incoming Webhooks → Activate
2. Add webhook to your channel → copy URL
3. Set env vars: SLACK_WEBHOOK_URL, ANTHROPIC_API_KEY
"""

import math
import os
import time
import requests
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from loguru import logger

# London timezone offset (UTC+0 winter, UTC+1 BST)
def london_now():
    """Get current London time (handles BST automatically)."""
    utc = datetime.now(timezone.utc)
    year = utc.year
    mar31 = datetime(year, 3, 31, tzinfo=timezone.utc)
    bst_start = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)
    bst_start = bst_start.replace(hour=1)
    oct31 = datetime(year, 10, 31, tzinfo=timezone.utc)
    bst_end = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)
    bst_end = bst_end.replace(hour=1)

    if bst_start <= utc < bst_end:
        return utc + timedelta(hours=1)
    return utc


class MorningBriefingAgent:
    """Generates and sends daily morning briefing to Slack."""

    def __init__(self, config: Dict = None):
        self.webhook_url = os.environ.get('SLACK_WEBHOOK_URL', '')
        self.api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        self.target_hour = 10
        self.target_hours = [10, 16]
        self.target_minute = 0
        self.last_sent_key = None
        self._running = False
        self._thread = None

        if not self.webhook_url:
            logger.warning("[Briefing] SLACK_WEBHOOK_URL not set — briefing disabled")
        if not self.api_key:
            logger.warning("[Briefing] ANTHROPIC_API_KEY not set — will use template mode")

        logger.info(f"[Briefing] Agent initialized (target: {self.target_hour:02d}:{self.target_minute:02d} London)")

    def start(self, get_market_data_fn):
        self._get_data = get_market_data_fn
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        logger.info("[Briefing] Scheduler started")

    def stop(self):
        self._running = False

    def _scheduler_loop(self):
        while self._running:
            try:
                now = london_now()
                today = now.strftime('%Y-%m-%d')
                target_hours = getattr(self, 'target_hours', [self.target_hour])
                this_key = f"{today}_{now.hour:02d}"
                if (
                    now.hour in target_hours
                    and now.minute >= self.target_minute
                    and now.minute < self.target_minute + 2
                    and self.last_sent_key != this_key
                ):
                    logger.info(f"[Briefing] 🌅 It's {now.strftime('%H:%M')} London — generating briefing")
                    self._generate_and_send()
                    self.last_sent_key = this_key
            except Exception as e:
                logger.error(f"[Briefing] Scheduler error: {e}")
            time.sleep(60)

    def _generate_and_send(self):
        try:
            market_data = self._get_data()
            if self.api_key:
                briefing = self._generate_with_claude(market_data)
            else:
                briefing = self._generate_template(market_data)

            if briefing and self.webhook_url:
                self._send_to_slack(briefing)
                logger.info("[Briefing] ✅ Sent to Slack")
            elif briefing:
                logger.info(f"[Briefing] Generated (no webhook):\n{briefing[:200]}...")
        except Exception as e:
            logger.error(f"[Briefing] Failed: {e}")

    def force_send(self, market_data: Dict = None) -> str:
        if market_data is None and hasattr(self, '_get_data'):
            market_data = self._get_data()
        if not market_data:
            return "No market data available"
        if self.api_key:
            briefing = self._generate_with_claude(market_data)
        else:
            briefing = self._generate_template(market_data)
        if briefing and self.webhook_url:
            self._send_to_slack(briefing)
        return briefing

    def _generate_with_claude(self, m: Dict) -> str:
        context = self._build_context(m)
        prompt = f"""You are the morning briefing writer for Cenoa Insights, a crypto intelligence platform.
Write a concise Slack morning briefing for the team. The date is {london_now().strftime('%A, %B %d, %Y')}.

MARKET DATA:
{context}

RULES:
- Write for a finance team (CFO, traders, product people) who need actionable intel
- Use Slack formatting: *bold*, _italic_, emoji sparingly
- Structure: greeting → market status → key numbers → what matters today → yields update → action items
- Be direct, no fluff. Every sentence should add value
- Include at least one insight that connects dots when the data supports it
- Add relevant global context if macro data suggests it (DXY, VIX, rates)
- Omit any field that is unavailable, missing, NaN, or not meaningful
- Never print placeholders like nan, None, N/A, unavailable values, or excessive decimals
- Round sentiment to a whole number, percentages to 1 decimal when needed, and money to readable units
- End with "🐍 HYDRA" signature
- Keep under 220 words
- Write in English
"""
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data['content'][0]['text']
                return self._post_process_brief(text)
            logger.error(f"[Briefing] Claude API {resp.status_code}: {resp.text[:200]}")
            return self._generate_template(m)
        except Exception as e:
            logger.error(f"[Briefing] Claude API error: {e}")
            return self._generate_template(m)

    def _generate_template(self, m: Dict) -> str:
        now = london_now()
        is_afternoon = now.hour >= 14
        day_name = now.strftime('%A')

        # ── Extract all available data ────────────────────────────
        btc = self._num(m.get('last_price', m.get('close')))
        chg = self._num(m.get('price_change_24h'))
        eth = self._num(m.get('eth_price'))
        eth_chg = self._num(m.get('eth_change_24h'))
        fg = self._num(m.get('fear_greed_value'))
        fgl = self._fg_label(fg) if fg is not None else None
        etf_d = self._num(m.get('etf_net_flow_daily'))
        etf_7d = self._num(m.get('etf_net_flow_7d'))
        dom = self._num(m.get('btc_dominance'))
        vix = self._num(m.get('vix_current'))
        dxy = self._num(m.get('dxy_current'))
        fund = self._num(m.get('funding_rate'))
        bench_usd = self._num(m.get('defi_avg_stable_apy'))
        bench_eth = self._num(m.get('eth_benchmark_apy'))
        t3m = self._num(m.get('treasury_3m'))
        best_apy = self._num(m.get('defi_max_stable_apy'))
        breadth = self._num(m.get('venom_breadth_pct'))
        alt_season = self._num(m.get('venom_alt_season_score'))
        mcap_chg = self._num(m.get('total_mcap_change_24h'))

        # ── Greeting — varies by time + day ───────────────────────
        if is_afternoon:
            greeting = f"📊 *Afternoon Update* — {now.strftime('%H:%M')} London"
        elif day_name == 'Monday':
            greeting = f"🌅 *Monday Briefing* — Week ahead"
        else:
            greeting = f"🌅 *Morning Briefing* — {day_name}"
        date_line = f"_{now.strftime('%B %d, %Y')}_"

        # ── Lead — most interesting data point first ──────────────
        lead = ''
        if btc is not None:
            price_str = self._fmt_money(btc)
            if chg is not None:
                if chg >= 3:
                    lead = f"*BTC {price_str}* — up {chg:.1f}% in 24h. Strong move."
                elif chg <= -3:
                    lead = f"*BTC {price_str}* — down {abs(chg):.1f}% in 24h. Selling pressure."
                elif chg >= 0:
                    lead = f"*BTC {price_str}* (+{chg:.1f}%)"
                else:
                    lead = f"*BTC {price_str}* ({chg:.1f}%)"
            else:
                lead = f"*BTC {price_str}*"
            if eth is not None:
                eth_part = f"${eth:,.0f}"
                if eth_chg is not None:
                    eth_part += f" ({eth_chg:+.1f}%)"
                lead += f"  ·  *ETH {eth_part}*"

        # ── Market mood line ──────────────────────────────────────
        mood = ''
        mood_parts = []
        if fg is not None:
            mood_parts.append(f"F&G *{round(fg)}* ({fgl})")
        if vix is not None:
            vix_note = '🔴 elevated' if vix > 25 else '🟡 watching' if vix > 20 else '🟢 calm'
            mood_parts.append(f"VIX *{vix:.0f}* {vix_note}")
        if fund is not None:
            f_pct = fund * 100
            if abs(f_pct) > 0.02:
                side = 'longs paying' if f_pct > 0 else 'shorts paying'
                mood_parts.append(f"Funding *{f_pct:.3f}%* ({side})")
        if mood_parts:
            mood = '  ·  '.join(mood_parts)

        # ── Flows ─────────────────────────────────────────────────
        flow_line = ''
        flow_parts = []
        if etf_d is not None:
            direction = 'inflow' if etf_d > 0 else 'outflow'
            flow_parts.append(f"ETF today: *{self._fmt_money_short(etf_d)}* {direction}")
        if etf_7d is not None:
            flow_parts.append(f"7d: *{self._fmt_money_short(etf_7d)}*")
        if dom is not None:
            flow_parts.append(f"BTC dom: *{dom:.1f}%*")
        if flow_parts:
            flow_line = '  ·  '.join(flow_parts)

        # ── Altcoin pulse ─────────────────────────────────────────
        alt_line = ''
        if breadth is not None:
            alt_line = f"Altcoin breadth: *{breadth:.0f}%* green"
            if alt_season is not None:
                if alt_season > 2:
                    alt_line += f" — alt season vibes (+{alt_season:.1f}% vs BTC)"
                elif alt_season < -2:
                    alt_line += f" — BTC leading ({alt_season:.1f}% vs alts)"

        # ── Macro ─────────────────────────────────────────────────
        macro_line = ''
        macro_parts = []
        if dxy is not None:
            macro_parts.append(f"DXY *{dxy:.1f}*")
        if t3m is not None:
            macro_parts.append(f"T-Bill *{t3m:.2f}%*")
        if macro_parts:
            macro_line = '  ·  '.join(macro_parts)

        # ── Yields ────────────────────────────────────────────────
        yield_line = ''
        yield_parts = []
        if bench_usd is not None:
            yield_parts.append(f"DeFi stable *{bench_usd:.1f}%*")
        if bench_eth is not None:
            yield_parts.append(f"ETH stake *{bench_eth:.1f}%*")
        if best_apy is not None and best_apy > (bench_usd or 0) * 1.5:
            yield_parts.append(f"Best pool *{best_apy:.1f}%*")
        yield_parts.append("Cenoa *5.0%*")
        if yield_parts:
            yield_line = '  ·  '.join(yield_parts)

        # ── Smart insight — connects dots ─────────────────────────
        insight = self._smart_insight(fg, etf_d, etf_7d, vix, chg, bench_usd, t3m, fund, breadth, dom)

        # ── Assemble ──────────────────────────────────────────────
        lines = [greeting, date_line, '']
        if lead: lines.append(lead)
        if mood: lines.extend(['', mood])
        if flow_line: lines.extend(['', f"📈 {flow_line}"])
        if alt_line: lines.extend(['', f"🪙 {alt_line}"])
        if macro_line: lines.extend(['', f"🌍 {macro_line}"])
        if yield_line: lines.extend(['', f"💰 {yield_line}"])
        if insight: lines.extend(['', f"💡 _{insight}_"])
        lines.extend([
            '',
            f"📊 <{os.environ.get('DASHBOARD_URL', 'https://web-production-57fc0.up.railway.app')}|Open Dashboard>",
            '',
            '🐍 _HYDRA_',
        ])
        return self._post_process_brief('\n'.join(lines))

    def _smart_insight(self, fg, etf_d, etf_7d, vix, chg, bench_usd, t3m, fund, breadth, dom):
        """Generate a single insight by connecting multiple data points."""
        # Fear + ETF inflows = dip buying
        if fg is not None and etf_d is not None and fg <= 20 and etf_d > 0:
            return f'Extreme fear ({round(fg)}) but ETF money still flowing in ({self._fmt_money_short(etf_d)}). Institutions buying the dip.'
        # Fear + negative funding = short squeeze setup
        if fg is not None and fund is not None and fg <= 25 and fund < -0.0001:
            return f'Fear at {round(fg)} and shorts are paying funding. Classic squeeze setup if sentiment shifts.'
        # VIX spike + BTC holding
        if vix is not None and chg is not None and vix > 25 and chg > -1:
            return f'VIX at {vix:.0f} but BTC holding steady. Crypto decoupling from equity fear for now.'
        # Breadth divergence
        if breadth is not None and chg is not None and breadth > 70 and chg < 0:
            return f'BTC is red but {breadth:.0f}% of alts are green. Money rotating into alts.'
        if breadth is not None and chg is not None and breadth < 30 and chg > 0:
            return 'BTC green but most alts are red. Flight to quality — not broad risk-on.'
        # DeFi vs T-bill
        if bench_usd is not None and t3m is not None:
            if bench_usd < t3m:
                return f'DeFi yields ({bench_usd:.1f}%) still below T-Bills ({t3m:.1f}%). Capital not being compensated for crypto risk.'
            if bench_usd > t3m * 1.5:
                return f'DeFi yields ({bench_usd:.1f}%) running well above T-Bills ({t3m:.1f}%). Opportunity or risk? Worth watching.'
        # BTC dominance trend
        if dom is not None and dom > 62:
            return f'BTC dominance at {dom:.1f}%. Alt season is far away. Stick to majors.'
        if dom is not None and dom < 55:
            return f'BTC dominance dropped to {dom:.1f}%. Capital rotating into alts.'
        # Simple fear/greed
        if fg is not None and fg <= 20:
            return f'Extreme fear ({round(fg)}). Historically these levels resolve within 1-2 weeks. Patience.'
        if fg is not None and fg >= 80:
            return f'Extreme greed ({round(fg)}). Time to be careful, not aggressive.'
        # Default
        if chg is not None and abs(chg) < 1:
            return 'Quiet session. Nothing screaming for attention — enjoy the calm.'
        return ''

    def _build_context(self, m: Dict) -> str:
        parts = []
        btc = self._num(m.get('last_price', m.get('close')))
        chg = self._num(m.get('price_change_24h'))
        if btc is not None:
            pct = f" ({chg:+.1f}% 24h)" if chg is not None else ""
            parts.append(f"BTC Price: {self._fmt_money(btc)}{pct}")

        fg = self._num(m.get('fear_greed_value'))
        if fg is not None:
            parts.append(f"Fear & Greed: {round(fg):.0f}/100 ({self._fg_label(fg)})")

        etf = self._num(m.get('etf_net_flow_7d'))
        if etf is not None:
            parts.append(f"BTC ETF 7D Flow: {self._fmt_money_short(etf)}")

        dom = self._num(m.get('btc_dominance'))
        if dom is not None:
            parts.append(f"BTC Dominance: {dom:.1f}%")

        for label, key, fmt in [
            ('VIX', 'vix_current', '{:.0f}'),
            ('DXY', 'dxy_current', '{:.1f}'),
            ('SPX', 'spx_current', '{:,.0f}'),
            ('USD DeFi Benchmark', 'usd_benchmark_apy', '{:.2f}%'),
            ('ETH Staking Benchmark', 'eth_benchmark_apy', '{:.2f}%'),
            ('US 3M T-Bill', 'treasury_3m', '{:.2f}%'),
            ('US 10Y', 'treasury_10y', '{:.2f}%'),
        ]:
            value = self._num(m.get(key, m.get('us10y_current') if key == 'treasury_10y' else None))
            if value is not None:
                parts.append(f"{label}: {fmt.format(value)}")

        best = self._num(m.get('defi_max_stable_apy'))
        pool = (m.get('defi_max_stable_pool') or '').strip()
        if best is not None:
            suffix = f" ({pool})" if pool else ''
            parts.append(f"Best USDC Pool: {best:.1f}%{suffix}")

        poly_lines = []
        for k, v in m.items():
            if k.startswith('poly_') and k.endswith('_prob'):
                prob = self._num(v)
                if prob is None:
                    continue
                name = k.replace('poly_', '').replace('_prob', '').replace('_', ' ')
                poly_lines.append(f"  - {name}: {prob*100:.0f}%")
        if poly_lines:
            parts.append('Prediction Markets:')
            parts.extend(poly_lines[:4])

        for pair, name in [('usdtry', 'USD/TRY'), ('usdngn', 'USD/NGN'), ('usdpkr', 'USD/PKR'), ('usdegp', 'USD/EGP'), ('usdbrl', 'USD/BRL')]:
            rate = self._num(m.get(f'{pair}_rate'))
            chg = self._num(m.get(f'{pair}_change_1d'))
            if rate is not None:
                pct = f" ({chg:+.1f}%)" if chg is not None else ''
                parts.append(f"{name}: {rate:,.2f}{pct}")

        usdt = self._num(m.get('stable_usdt_supply', m.get('stable_usdt_total')))
        usdc = self._num(m.get('stable_usdc_supply', m.get('stable_usdc_total')))
        if usdt is not None:
            parts.append(f"USDT Supply: ${usdt/1e9:.1f}B")
        if usdc is not None:
            parts.append(f"USDC Supply: ${usdc/1e9:.1f}B")

        for label, key in [('FOMC', 'fomc_hours_until'), ('CPI', 'cpi_hours_until')]:
            value = self._num(m.get(key))
            if value is not None and 0 < value < 168:
                parts.append(f"{label} in {value:.0f} hours")

        return "\n".join(parts) if parts else 'Limited clean market data available right now.'

    def _send_to_slack(self, text: str):
        if not self.webhook_url:
            logger.warning("[Briefing] No webhook URL")
            return
        try:
            resp = requests.post(self.webhook_url, json={"text": text, "unfurl_links": False}, timeout=10)
            if resp.status_code != 200:
                logger.error(f"[Briefing] Slack error {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"[Briefing] Slack send failed: {e}")

    def _fg_label(self, v):
        if v <= 25:
            return 'Extreme Fear'
        if v <= 45:
            return 'Fear'
        if v <= 55:
            return 'Neutral'
        if v <= 75:
            return 'Greed'
        return 'Extreme Greed'

    def _num(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {'nan', 'na', 'n/a', 'none', 'null', '-'}:
                return None
            try:
                value = float(stripped.replace(',', ''))
            except Exception:
                return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        return None

    def _fmt_money(self, value: float) -> str:
        return f"${value:,.0f}"

    def _fmt_money_short(self, value: float) -> str:
        abs_v = abs(value)
        sign = '+' if value > 0 else '-' if value < 0 else ''
        if abs_v >= 1_000_000_000:
            return f"{sign}${abs_v/1_000_000_000:.1f}B"
        if abs_v >= 1_000_000:
            return f"{sign}${abs_v/1_000_000:.0f}M"
        if abs_v >= 1_000:
            return f"{sign}${abs_v/1_000:.0f}K"
        return f"{sign}${abs_v:,.0f}"

    def _post_process_brief(self, text: str) -> str:
        """Clean up the briefing text — remove NaN lines, collapse blank runs."""
        cleaned = []
        prev_blank = False
        for raw in text.splitlines():
            line = raw.strip()
            lower = line.lower()
            # Skip lines containing NaN/None artifacts
            if any(t in lower for t in ['nan', 'none', 'n/a', 'null']) and '*' not in line and 'http' not in line:
                continue
            if not line:
                if cleaned and not prev_blank:
                    cleaned.append('')
                prev_blank = True
                continue
            cleaned.append(line)
            prev_blank = False
        return '\n'.join(cleaned).strip()
