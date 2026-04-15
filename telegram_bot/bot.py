"""
telegram_bot.py
───────────────
HYDRA Telegram Bot — Komut tabanlı, dashboard'un uzantısı.

Komutlar:
    /status  — Şu anki BTC fiyatı, Fear&Greed, ETF flow
    /pulse   — Son AI Market Pulse yorumu
    /signal  — HYDRA Layer1 sinyal durumu
    /alerts  — Aktif uyarılar (event blackout, extreme readings)
    /help    — Komut listesi

Setup:
    1. @BotFather'dan bot oluştur, token al
    2. Bot'u gruba/kanala ekle
    3. Railway env vars:
       HYDRA_TELEGRAM_TOKEN = bot token
       HYDRA_TELEGRAM_CHAT  = chat id (grup: -100xxx, kanal: @isim)

Mimari:
    Polling tabanlı — webhook gerekmez.
    Daemon thread olarak çalışır, main loop'u bloke etmez.
"""

import os
import time
import threading
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from loguru import logger


class TelegramCommandBot:
    """
    Komut tabanlı Telegram bot.
    Dashboard state'ini okur, komutlara anında yanıt verir.
    """

    POLL_TIMEOUT = 30   # long-polling timeout (saniye)
    BASE         = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self):
        self.token     = os.environ.get("HYDRA_TELEGRAM_TOKEN", "")
        self.chat_id   = os.environ.get("HYDRA_TELEGRAM_CHAT", "")
        self._offset   = 0
        self._running  = False
        self._thread   = None

        # Data provider fonksiyonları — main.py'den set edilir
        self._get_market_data: Optional[Callable] = None
        self._get_pulse:       Optional[Callable] = None
        self._get_signal:      Optional[Callable] = None

        if not self.token:
            logger.warning("[TGBot] HYDRA_TELEGRAM_TOKEN not set — bot disabled")
        else:
            logger.info("[TGBot] Initialized (token: ...{})", self.token[-6:])

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def start(self,
              get_market_data: Callable,
              get_pulse:       Callable,
              get_signal:      Callable):
        """Bot polling thread'ini başlat."""
        if not self.token:
            return

        self._get_market_data = get_market_data
        self._get_pulse       = get_pulse
        self._get_signal      = get_signal
        self._running         = True
        self._thread          = threading.Thread(
            target=self._polling_loop, daemon=True, name="tg-bot"
        )
        self._thread.start()
        logger.info("[TGBot] Polling started")

    def stop(self):
        self._running = False

    def send(self, text: str, chat_id: str = None) -> bool:
        """Mesaj gönder (proaktif bildirimler için de kullanılır)."""
        if not self.token:
            return False
        cid = chat_id or self.chat_id
        if not cid:
            return False
        try:
            r = requests.post(
                self.BASE.format(token=self.token, method="sendMessage"),
                json={
                    "chat_id":    cid,
                    "text":       text,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            return r.status_code == 200
        except Exception as e:
            logger.error("[TGBot] Send failed: {}", e)
            return False

    # ─────────────────────────────────────────────
    # Polling loop
    # ─────────────────────────────────────────────

    def _polling_loop(self):
        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._handle_update(update)
            except Exception as e:
                logger.error("[TGBot] Poll error: {}", e)
                time.sleep(5)

    def _get_updates(self) -> list:
        try:
            r = requests.get(
                self.BASE.format(token=self.token, method="getUpdates"),
                params={"offset": self._offset, "timeout": self.POLL_TIMEOUT},
                timeout=self.POLL_TIMEOUT + 5,
            )
            if r.status_code != 200:
                return []
            updates = r.json().get("result", [])
            if updates:
                self._offset = updates[-1]["update_id"] + 1
            return updates
        except Exception:
            return []

    def _handle_update(self, update: Dict):
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return

        text    = msg.get("text", "").strip()
        chat_id = str(msg["chat"]["id"])
        user    = msg.get("from", {}).get("first_name", "?")

        if not text.startswith("/"):
            return

        cmd = text.split()[0].lower().replace("@", "").split("@")[0]
        logger.info("[TGBot] Command '{}' from {} in {}", cmd, user, chat_id)

        if cmd == "/status":
            self.send(self._cmd_status(), chat_id)
        elif cmd == "/pulse":
            self.send(self._cmd_pulse(), chat_id)
        elif cmd == "/signal":
            self.send(self._cmd_signal(), chat_id)
        elif cmd == "/alerts":
            self.send(self._cmd_alerts(), chat_id)
        elif cmd in ("/help", "/start"):
            self.send(self._cmd_help(), chat_id)
        else:
            self.send(f"Unknown command: {cmd}\nType /help for the list.", chat_id)

    # ─────────────────────────────────────────────
    # Command handlers
    # ─────────────────────────────────────────────

    def _cmd_status(self) -> str:
        d = self._get_market_data() if self._get_market_data else {}
        if not d:
            return "⚠️ No market data yet. System warming up..."

        btc   = d.get("last_price",        "—")
        chg   = d.get("price_change_24h",  "—")
        fg    = d.get("fear_greed_value",  "—")
        etf   = d.get("etf_net_flow_daily","—")
        fund  = d.get("funding_rate",      "—")
        dom   = d.get("btc_dominance",     "—")

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        try:    btc_fmt = f"${float(btc):,.0f}"
        except: btc_fmt = str(btc)
        try:    chg_fmt = f"{float(chg):+.2f}%"
        except: chg_fmt = str(chg)
        try:    etf_fmt = f"${float(etf)/1e6:,.0f}M" if abs(float(etf)) > 1000 else f"${float(etf):,.0f}M"
        except: etf_fmt = str(etf)
        try:    fund_fmt = f"{float(fund):.4f}"
        except: fund_fmt = str(fund)
        try:    dom_fmt = f"{float(dom):.1f}%"
        except: dom_fmt = str(dom)

        return (
            f"📊 <b>HYDRA Status</b> — {now}\n"
            f"{'─'*30}\n"
            f"🟠 BTC:  <code>{btc_fmt}</code>  ({chg_fmt})\n"
            f"😱 Fear&Greed: <code>{fg}/100</code>\n"
            f"📥 ETF Flow: <code>{etf_fmt}</code>\n"
            f"💰 Funding: <code>{fund_fmt}</code>\n"
            f"👑 BTC Dom: <code>{dom_fmt}</code>"
        )

    def _cmd_pulse(self) -> str:
        pulse = self._get_pulse() if self._get_pulse else {}
        if not pulse or not pulse.get("headline"):
            return "⏳ Pulse engine initializing..."

        signal   = pulse.get("signal", "NEUTRAL")
        headline = pulse.get("headline", "")
        detail   = pulse.get("detail", "")
        conf     = pulse.get("confidence", 0)
        pts      = pulse.get("data_points", 0)
        updated  = pulse.get("updated_at", "")

        emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "ALERT": "🟡", "NEUTRAL": "⚪"}.get(signal, "⚪")

        ts = ""
        if updated:
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                ts = dt.strftime("%H:%M UTC")
            except Exception:
                pass

        return (
            f"{emoji} <b>Market Pulse</b>\n"
            f"{'─'*30}\n"
            f"<b>{headline}</b>\n\n"
            f"{detail}\n\n"
            f"Signal: <code>{signal}</code> | Conf: <code>{conf}%</code> | "
            f"Data: <code>{pts} points</code>\n"
            f"<i>Updated: {ts}</i>"
        )

    def _cmd_signal(self) -> str:
        sig = self._get_signal() if self._get_signal else {}
        if not sig:
            return "⏳ HYDRA signal not ready yet (warming up ~100 min)..."

        direction  = sig.get("direction",        "NEUTRAL")
        score      = sig.get("score",            0)
        confidence = sig.get("confidence",       "—")
        regime     = sig.get("regime",           "—")
        session    = sig.get("session",          "—")
        maturity   = sig.get("data_maturity",    0)

        emoji = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}.get(direction, "⚪")

        warm = f"⏳ Warming up ({float(maturity):.0%})" if float(maturity or 0) < 1 else ""

        return (
            f"{emoji} <b>HYDRA Signal</b>\n"
            f"{'─'*30}\n"
            f"Direction:  <code>{direction}</code>\n"
            f"Score:      <code>{float(score):+.1f}/100</code>\n"
            f"Confidence: <code>{confidence}</code>\n"
            f"Regime:     <code>{regime}</code>\n"
            f"Session:    <code>{session}</code>\n"
            f"{warm}"
        )

    def _cmd_alerts(self) -> str:
        d   = self._get_market_data() if self._get_market_data else {}
        sig = self._get_signal()      if self._get_signal      else {}

        alerts = []

        fg = float(d.get("fear_greed_value", 50) or 50)
        if fg < 15:
            alerts.append(f"😱 Extreme Fear: {fg:.0f}/100")
        elif fg > 80:
            alerts.append(f"🤑 Extreme Greed: {fg:.0f}/100")

        fund = float(d.get("funding_rate", 0) or 0)
        if abs(fund) > 0.001:
            side = "long" if fund > 0 else "short"
            alerts.append(f"💰 Extreme Funding ({side}): {fund:.4f}")

        vix = float(d.get("vix_current", 0) or 0)
        if vix > 30:
            alerts.append(f"📈 VIX Spike: {vix:.1f} (>30)")

        if sig.get("event_blackout"):
            alerts.append("🔇 Event Blackout Active (FOMC/CPI imminent)")

        etf = float(d.get("etf_net_flow_daily", 0) or 0)
        if etf > 500_000_000:
            alerts.append(f"🐋 Massive ETF Inflow: ${etf/1e6:,.0f}M")
        elif etf < -300_000_000:
            alerts.append(f"🚨 ETF Outflow: ${etf/1e6:,.0f}M")

        if not alerts:
            return "✅ <b>No active alerts</b>\nAll readings within normal range."

        header = f"🚨 <b>Active Alerts ({len(alerts)})</b>\n{'─'*30}\n"
        return header + "\n".join(f"• {a}" for a in alerts)

    def _cmd_help(self) -> str:
        return (
            "🐍 <b>HYDRA Bot Commands</b>\n"
            "{'─'*30}\n"
            "/status  — BTC price, Fear&Greed, ETF flow\n"
            "/pulse   — AI Market Pulse analysis\n"
            "/signal  — HYDRA directional signal\n"
            "/alerts  — Active market alerts\n"
            "/help    — This message\n\n"
            "<i>Powered by Cenoa Insights</i>"
        )
