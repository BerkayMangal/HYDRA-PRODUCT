"""
pulse_engine.py — HYDRA Quad-AI Pulse Engine v4
═══════════════════════════════════════════════
4 AI providers running IN PARALLEL every 60 seconds.

PROVIDERS:  Claude (Anthropic) + GPT-4o-mini (OpenAI) + Grok-2 (xAI) + Perplexity (sonar)
AGGREGATION: majority vote → HIGH_CONVICTION / CONSENSUS / DIVERGENCE
FALLBACK:    deterministic rules when all APIs down

CREDIT GUARD (v4 FIX):
  When a provider returns 400/402/429 with credit-related message,
  it is disabled for COOLDOWN_SECONDS (default 3600 = 1 hour).
  This eliminates log spam from exhausted API keys and avoids
  wasting network round-trips on providers that cannot respond.

ENV VARS:
  ANTHROPIC_API_KEY
  OPENAI_API_KEY
  XAI_API_KEY          (get from x.ai/api)
  PERPLEXITY_API_KEY   (get from perplexity.ai)
  — or —
  HYDRA_PERPLEXITY_KEY
"""

import json, math, os, time, threading, concurrent.futures, requests
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

_PROVIDERS = {
    "claude": {
        "name": "Claude", "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-haiku-4-5-20251001", "env_key": "ANTHROPIC_API_KEY",
        "style": "anthropic", "timeout": 15,
    },
    "gpt": {
        "name": "GPT-4o-mini", "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini", "env_key": "OPENAI_API_KEY",
        "style": "openai", "timeout": 15,
    },
    "grok": {
        "name": "Grok-2", "url": "https://api.x.ai/v1/chat/completions",
        "model": "grok-2-latest", "env_key": "XAI_API_KEY",
        "style": "openai", "timeout": 20,
    },
    "perplexity": {
        "name": "Perplexity", "url": "https://api.perplexity.ai/chat/completions",
        "model": "sonar", "env_key": "PERPLEXITY_API_KEY",
        "style": "openai", "timeout": 20,
    },
}
_VALID_SIGNALS = {"BULLISH", "BEARISH", "NEUTRAL", "ALERT"}

# Credit-related error patterns — if response body contains any of these,
# the provider is put on cooldown instead of retrying every 60s.
_CREDIT_ERROR_PATTERNS = (
    "credit balance is too low",
    "used all available credits",
    "reached its month",
    "insufficient_quota",
    "billing",
    "exceeded your current quota",
    "rate_limit_exceeded",
    "payment required",
)

# How long (seconds) to disable a provider after credit exhaustion.
COOLDOWN_SECONDS = 3600  # 1 hour


class PulseEngine:
    """Multi-AI consensus engine. Fires up to 4 providers in parallel, aggregates."""

    REFRESH_INTERVAL = 60

    def __init__(self):
        self._keys = {k: os.environ.get(v["env_key"], "") for k, v in _PROVIDERS.items()}
        # Also check HYDRA_PERPLEXITY_KEY as fallback
        if not self._keys.get("perplexity"):
            self._keys["perplexity"] = os.environ.get("HYDRA_PERPLEXITY_KEY", "")
        self._enabled = {k: bool(v) for k, v in self._keys.items()}

        # Credit guard: tracks when a provider was disabled due to credit errors.
        # Key = provider key, Value = epoch timestamp when cooldown expires.
        self._cooldown_until: Dict[str, float] = {}

        self._resolved_xai_model: Optional[str] = None
        self._get_data = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self.current: Dict[str, Any] = {
            "headline": "", "detail": "", "signal": "NEUTRAL",
            "confidence": 0, "conviction": "INITIALIZING",
            "data_points": 0, "updated_at": None, "error": None,
            "providers": {}, "agreement": None,
        }
        active = [_PROVIDERS[k]["name"] for k, on in self._enabled.items() if on]
        logger.info("[Pulse] Quad-AI engine | active: {}", active or ["rules-only"])

    # ── Provider availability (credit guard) ──────────────────────────────────

    def _is_available(self, key: str) -> bool:
        """Check if provider is enabled AND not on credit cooldown."""
        if not self._enabled.get(key, False):
            return False
        cooldown_end = self._cooldown_until.get(key, 0)
        if time.time() < cooldown_end:
            return False
        return True

    def _put_on_cooldown(self, key: str, reason: str) -> None:
        """Disable a provider for COOLDOWN_SECONDS after credit exhaustion."""
        self._cooldown_until[key] = time.time() + COOLDOWN_SECONDS
        remaining_min = COOLDOWN_SECONDS / 60
        logger.warning(
            "[Pulse] {} disabled for {:.0f}min — {}",
            _PROVIDERS[key]["name"], remaining_min, reason[:80],
        )

    def _is_credit_error(self, status_code: int, body_str: str) -> bool:
        """Detect credit/billing related errors from HTTP response."""
        if status_code in (402, 429):
            return True
        if status_code == 400:
            body_lower = body_str.lower()
            return any(pat in body_lower for pat in _CREDIT_ERROR_PATTERNS)
        return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, get_data_fn):
        self._get_data = get_data_fn
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pulse-engine")
        self._thread.start()
        logger.info("[Pulse] Engine started ({}s interval)", self.REFRESH_INTERVAL)

    def stop(self):
        self._running = False

    def force_refresh(self) -> Dict:
        if self._get_data:
            self._run_once()
        return self.current

    def _loop(self):
        while self._running:
            try:
                self._run_once()
            except Exception as e:
                logger.error("[Pulse] Loop error: {}", e)
            time.sleep(self.REFRESH_INTERVAL)

    def _run_once(self):
        if not self._get_data:
            return
        data = self._get_data()
        if not data:
            return
        context, n_points = self._build_context(data)
        if n_points < 3:
            return

        # Check which providers are actually available right now
        available_providers = {k: cfg for k, cfg in _PROVIDERS.items() if self._is_available(k)}

        if available_providers:
            result = self._analyze_multi_ai(context, data, available_providers)
        else:
            result = self._analyze_rules(data)
            result["agreement"] = "RULES_ONLY"
            result["conviction"] = "RULES"

        result["data_points"] = n_points
        result["updated_at"] = datetime.now(timezone.utc).isoformat()
        result["error"] = None
        # Add cooldown status for dashboard visibility
        result["provider_cooldowns"] = {
            _PROVIDERS[k]["name"]: max(0, int(self._cooldown_until.get(k, 0) - time.time()))
            for k in _PROVIDERS
            if self._cooldown_until.get(k, 0) > time.time()
        }
        with self._lock:
            self.current = result

        # Directly push to dashboard
        try:
            from dashboard import state as _ds, sanitize as _san
            _ds.pulse = _san(result)
        except Exception:
            pass

        logger.info("[Pulse] {} | {} | {} | {} pts",
                    result["signal"], result.get("agreement", ""),
                    result["headline"][:65], n_points)

    # ── Context ──────────────────────────────────────────────────────────────

    def _build_context(self, d: Dict) -> Tuple[str, int]:
        lines, n = [], 0
        def add(label, key, fmt="{}"):
            nonlocal n
            val = d.get(key)
            if val is None: return
            try:
                f = float(val)
            except: return
            if math.isnan(f) or math.isinf(f): return
            if f == 0 and key not in ("vix_current", "dxy_current"): return
            try:
                lines.append(f"  {label}: {fmt.format(f)}"); n += 1
            except: pass

        add("BTC Price",       "last_price",         "${:,.0f}")
        add("BTC 24h Change",  "price_change_24h",   "{:+.2f}%")
        add("Fear & Greed",    "fear_greed_value",   "{:.0f}/100")
        add("ETH Price",       "eth_price",          "${:,.0f}")
        for k, lbl in [("etf_net_flow_daily", "ETF daily"), ("etf_net_flow_7d", "ETF 7d")]:
            val = d.get(k)
            if val is not None:
                try:
                    f = float(val)
                    if not math.isnan(f) and f != 0:
                        lines.append(f"  {lbl}: ${f/1e6:,.0f}M"); n += 1
                except: pass
        add("Funding Rate",    "funding_rate",       "{:.4f}")
        add("OI Change %",     "oi_change_pct",      "{:+.2f}%")
        add("L/S Ratio",       "ls_ratio",           "{:.2f}")
        add("VIX",             "vix_current",        "{:.1f}")
        add("DXY",             "dxy_current",        "{:.2f}")
        add("US10Y",           "us10y_current",      "{:.2f}%")
        add("BTC Dominance",   "btc_dominance",      "{:.1f}%")
        add("DeFi TVL",        "defi_tvl_change_24h", "{:+.2f}%")
        add("Fed Cut Prob",    "fed_cut_probability", "{:.0f}%")
        add("HYDRA Score",     "hydra_score",        "{:+.1f}")
        return "\n".join(lines) if lines else "No data", n

    # ── Multi-AI orchestration ────────────────────────────────────────────────

    def _analyze_multi_ai(self, context: str, raw: Dict,
                          available_providers: Dict[str, Dict]) -> Dict:
        prompt = self._build_prompt(context)
        provider_results: Dict[str, Optional[Dict]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(self._call_provider, k, cfg, prompt): k
                for k, cfg in available_providers.items()
            }
            for fut in concurrent.futures.as_completed(futures, timeout=30):
                k = futures[fut]
                try:
                    provider_results[k] = fut.result()
                except Exception as e:
                    logger.warning("[Pulse] {} future err: {}", k, e)
                    provider_results[k] = None

        valid = {k: v for k, v in provider_results.items() if v is not None}
        if not valid:
            logger.warning("[Pulse] All AI providers failed — rules fallback")
            r = self._analyze_rules(raw)
            r.update({"agreement": "RULES_FALLBACK", "conviction": "RULES", "providers": {}})
            return r
        return self._aggregate(valid, raw)

    def _build_prompt(self, context: str) -> str:
        return f"""You are a real-time crypto market analyst. Analyze this live BTC market data.

LIVE DATA:
{context}

RULES:
1. Only reference numbers above. Zero hallucination.
2. Find the most interesting RELATIONSHIP between data points.
3. Be specific and sharp.

Respond ONLY with this JSON (no markdown):
{{
  "headline": "One sharp sentence max 80 chars",
  "detail": "2-3 sentences with specific numbers",
  "signal": "BULLISH or BEARISH or NEUTRAL or ALERT",
  "confidence": 0-100
}}"""

    def _resolve_xai_model(self, api_key: str) -> str:
        """Resolve a model that is actually available to this API key."""
        preferred = [
            "grok-4.20-reasoning",
            "grok-4",
            "grok-4-fast-reasoning",
            "grok-2-latest",
            _PROVIDERS["grok"]["model"],
        ]
        endpoints = (
            "https://api.x.ai/v1/language-models",
            "https://api.x.ai/v1/models",
        )

        for url in endpoints:
            try:
                resp = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
                if not resp.ok:
                    continue
                payload = resp.json()
                models = []
                if isinstance(payload.get("models"), list):
                    models.extend(payload.get("models", []))
                if isinstance(payload.get("data"), list):
                    models.extend(payload.get("data", []))

                available = set()
                for model in models:
                    if not isinstance(model, dict):
                        continue
                    mid = model.get("id")
                    if mid:
                        available.add(mid)
                    for alias in model.get("aliases", []) or []:
                        if alias:
                            available.add(alias)

                for candidate in preferred:
                    if candidate in available:
                        return candidate
                if available:
                    return sorted(available)[0]
            except Exception as e:
                logger.debug("[Pulse] xAI model discovery failed at {}: {}", url, e)

        return preferred[0]

    def _call_provider(self, key: str, cfg: Dict, prompt: str) -> Optional[Dict]:
        api_key = self._keys.get(key, "")
        if not api_key:
            return None
        try:
            cfg_local = dict(cfg)
            if key == "grok":
                if not self._resolved_xai_model:
                    self._resolved_xai_model = self._resolve_xai_model(api_key)
                    logger.info("[Pulse] Grok model resolved: {}", self._resolved_xai_model)
                cfg_local["model"] = self._resolved_xai_model

            if cfg_local["style"] == "anthropic":
                return self._call_anthropic(cfg_local, api_key, prompt, key)
            else:
                return self._call_openai_compat(cfg_local, api_key, prompt, key)
        except Exception as e:
            logger.warning("[Pulse] {} error: {}", cfg["name"], e)
            return None

    def _call_anthropic(self, cfg, api_key, prompt, provider_key) -> Optional[Dict]:
        resp = requests.post(cfg["url"], headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }, json={"model": cfg["model"], "max_tokens": 300,
                 "messages": [{"role": "user", "content": prompt}]},
        timeout=cfg["timeout"])

        if resp.status_code != 200:
            body_str = str(resp.json() if resp.text else "")[:200]
            if self._is_credit_error(resp.status_code, body_str):
                self._put_on_cooldown(provider_key, body_str)
            else:
                logger.warning("[Pulse] {} HTTP {} — {}", cfg["name"], resp.status_code, body_str[:120])
            return None
        return self._parse(resp.json()["content"][0]["text"], cfg["name"])

    def _call_openai_compat(self, cfg, api_key, prompt, provider_key) -> Optional[Dict]:
        messages = [{"role": "user", "content": prompt}]

        # Perplexity needs a strict system prompt — sonar model tends to
        # do web search and return prose instead of JSON without it.
        if provider_key == "perplexity":
            messages = [
                {"role": "system", "content": (
                    "You are a JSON-only API. You analyze crypto market data provided by the user. "
                    "Do NOT search the web. Do NOT add any text outside the JSON object. "
                    "Respond with ONLY a valid JSON object, no markdown, no explanation."
                )},
                {"role": "user", "content": prompt},
            ]

        payload = {
            "model": cfg["model"],
            "max_tokens": 300,
            "temperature": 0.3,
            "messages": messages,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        resp = requests.post(cfg["url"], headers=headers, json=payload, timeout=cfg["timeout"])

        if resp.status_code != 200:
            body_str = str(resp.json() if resp.text else "")[:200]

            # Credit guard
            if self._is_credit_error(resp.status_code, body_str):
                self._put_on_cooldown(provider_key, body_str)
                return None

            logger.warning("[Pulse] {} HTTP {} — {}", cfg["name"], resp.status_code, body_str[:120])

            # Grok model fallback
            if provider_key == "grok" and resp.status_code == 400 and "model" in body_str.lower():
                refreshed = self._resolve_xai_model(api_key)
                if refreshed and refreshed != cfg.get("model"):
                    logger.info("[Pulse] Grok model fallback: {} -> {}", cfg.get("model"), refreshed)
                    payload["model"] = refreshed
                    self._resolved_xai_model = refreshed
                    resp = requests.post(cfg["url"], headers=headers, json=payload, timeout=cfg["timeout"])
                    if resp.status_code != 200:
                        retry_body = str(resp.json() if resp.text else "")[:160]
                        if self._is_credit_error(resp.status_code, retry_body):
                            self._put_on_cooldown(provider_key, retry_body)
                        return None
                else:
                    return None
            else:
                return None

        return self._parse(resp.json()["choices"][0]["message"]["content"], cfg["name"])

    def _parse(self, text: str, name: str) -> Optional[Dict]:
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[1][4:] if len(parts) > 1 and parts[1].startswith("json") else (parts[1] if len(parts) > 1 else text)
        try:
            d = json.loads(text.strip())
            sig = str(d.get("signal", "NEUTRAL")).upper()
            if sig not in _VALID_SIGNALS:
                sig = "NEUTRAL"
            return {
                "headline":   str(d.get("headline", ""))[:100],
                "detail":     str(d.get("detail", ""))[:500],
                "signal":     sig,
                "confidence": max(0, min(100, int(d.get("confidence", 50)))),
                "provider":   name,
            }
        except Exception as e:
            logger.warning("[Pulse] {} parse fail: {} | {}", name, e, text[:80])
            return None

    # ── Aggregation ──────────────────────────────────────────────────────────

    def _aggregate(self, valid: Dict[str, Dict], raw: Dict) -> Dict:
        results = list(valid.values())
        n = len(results)
        signals = [r["signal"] for r in results]
        signal_ct = Counter(signals)

        if n == 1:
            r = results[0]
            safe_providers = {
                k: {kk: vv for kk, vv in v.items() if kk != "providers"}
                for k, v in valid.items()
            }
            result = {kk: vv for kk, vv in r.items() if kk != "providers"}
            result.update({
                "agreement": "SINGLE",
                "conviction": self._conv(r["confidence"]),
                "providers": safe_providers,
            })
            return result

        top_signal, top_count = signal_ct.most_common(1)[0]

        # Divergence: all different
        if top_count == 1 and n >= 3:
            names = [r["provider"] for r in results]
            sigs = [r["signal"] for r in results]
            headline_parts = " / ".join(f"{names[i]} {sigs[i]}" for i in range(min(n, 3)))
            return {
                "headline":   f"AI Divergence: {headline_parts}"[:100],
                "detail":     f"{n} AI models see the same data and disagree — genuinely ambiguous signal. {results[0]['headline'][:60]}",
                "signal":     "ALERT",
                "confidence": 40,
                "agreement":  "DIVERGENCE",
                "conviction": "LOW",
                "providers":  valid,
            }

        agreeing = [r for r in results if r["signal"] == top_signal]
        avg_conf = sum(r["confidence"] for r in agreeing) / len(agreeing)
        agreement = "HIGH_CONVICTION" if top_count == n else "CONSENSUS"
        if top_count == n:
            avg_conf = min(100, avg_conf + 15)   # unanimity bonus
        if top_count >= 3:
            avg_conf = min(100, avg_conf + 5)    # supermajority bonus

        best = max(agreeing, key=lambda x: x["confidence"])
        return {
            "headline":   best["headline"],
            "detail":     best["detail"],
            "signal":     top_signal,
            "confidence": int(avg_conf),
            "agreement":  agreement,
            "conviction": self._conv(int(avg_conf)),
            "providers":  valid,
        }

    @staticmethod
    def _conv(c: int) -> str:
        return "HIGH" if c >= 75 else ("MEDIUM" if c >= 55 else "LOW")

    # ── Rules fallback ────────────────────────────────────────────────────────

    @staticmethod
    def _sf(d, key, default):
        val = d.get(key)
        if val is None: return default
        try:
            f = float(val)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except: return default

    def _analyze_rules(self, d: Dict) -> Dict:
        fg  = self._sf(d, "fear_greed_value", 50.)
        chg = self._sf(d, "price_change_24h", 0.)
        etf = self._sf(d, "etf_net_flow_daily", 0.)
        fun = self._sf(d, "funding_rate", 0.)
        vix = self._sf(d, "vix_current", 20.)

        if fg < 20 and etf > 1e8:
            return {"headline": f"Smart Money Divergence: Fear={fg:.0f} but ETF +${etf/1e6:.0f}M",
                    "detail": "Extreme retail fear contradicted by institutional inflows.",
                    "signal": "BULLISH", "confidence": 72, "providers": {}}
        if fun > 0.001 and chg > 2:
            return {"headline": f"Overleveraged Rally: Funding {fun:.4f}, BTC +{chg:.1f}%",
                    "detail": "High funding rates during price rise. Correction risk elevated.",
                    "signal": "ALERT", "confidence": 65, "providers": {}}
        if vix > 30 and chg < -3:
            return {"headline": f"Risk-Off Event: VIX {vix:.1f}, BTC {chg:.1f}%",
                    "detail": "Macro fear spike correlating with crypto selloff.",
                    "signal": "BEARISH", "confidence": 70, "providers": {}}
        if etf > 2e8:
            return {"headline": f"Strong Institutional Demand: ETF +${etf/1e6:.0f}M today",
                    "detail": "Significant ETF inflows signal institutional accumulation.",
                    "signal": "BULLISH", "confidence": 60, "providers": {}}
        etf_s = f"${etf/1e6:+.0f}M" if etf != 0 else "n/a"
        sig = "BULLISH" if chg > 1 else ("BEARISH" if chg < -1 else "NEUTRAL")
        return {"headline": f"BTC {chg:+.1f}% | Fear={fg:.0f}/100 | ETF {etf_s}",
                "detail": "No strong divergence detected. Monitor for breakout.",
                "signal": sig, "confidence": 45, "providers": {}}
