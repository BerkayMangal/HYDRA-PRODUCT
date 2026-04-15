"""
HYDRA Dashboard v11 — Fintech Elegant Theme
Inspired by Revolut, Monzo, Cenoa
Light background, deep indigo + Cenoa green accents
Mobile-first responsive, clean typography
"""

import json,numpy as np
from datetime import datetime,timezone
from typing import Dict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import threading,uvicorn,os

_pulse_engine = None  # Set by main.py — direct ref bypasses state.pulse sync
_ml_cfg       = None  # Set by main.py via set_ml_config() — used by /api/status
_signal_tracker = None  # Set by main.py — direct ref for /api/performance

def set_ml_config(cfg):
    """Called from main.py after settings are loaded so the Blueprint endpoint knows ML state."""
    global _ml_cfg
    _ml_cfg = cfg

# api_status.py removed (Phase 1 cleanup) — /api/status uses legacy format

def sanitize(obj):
    if isinstance(obj,dict):return{str(k):sanitize(v)for k,v in obj.items()}
    elif isinstance(obj,(list,tuple)):return[sanitize(v)for v in obj]
    elif isinstance(obj,(np.integer,)):return int(obj)
    elif isinstance(obj,(np.floating,)):
        v=float(obj);return 0 if(np.isnan(v)or np.isinf(v))else v
    elif isinstance(obj,(np.bool_,)):return bool(obj)
    elif isinstance(obj,np.ndarray):return sanitize(obj.tolist())
    elif isinstance(obj,float):
        try:
            if np.isnan(obj)or np.isinf(obj):return 0
        except:pass
        return obj
    elif isinstance(obj,(int,bool,str,type(None))):return obj
    elif hasattr(obj,'isoformat'):return obj.isoformat()if obj else None
    elif hasattr(obj,'item'):return sanitize(obj.item())
    else:return str(obj)

app=FastAPI(title="HYDRA",docs_url="/api/docs")

from starlette.middleware.base import BaseHTTPMiddleware
class FrameOptionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        return response
app.add_middleware(FrameOptionsMiddleware)

# ── Static file serving for the new HYDRA terminal ────────────────────────────
# Place your frontend at: static/index.html + static/assets/app.js
# Old Cenoa dashboard stays at /  — zero breaking change.
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(os.path.join(_STATIC_DIR, "assets")):
    from fastapi.staticfiles import StaticFiles as _SF
    app.mount("/assets", _SF(directory=os.path.join(_STATIC_DIR,"assets")), name="assets")

class DashboardState:
    def __init__(self):
        self.current_signal={};self.signal_history=[];self.collector_status={};self.data_store_status={}
        self.bot_start_time=None;self.last_cycle_time=None;self.cycle_count=0;self.raw_market_data={}
        self.tracker_stats={};self.price_history=[];self.ml_data={};self.ml_dashboard={}
        self.hybrid_signal={};self.next_events=[];self.ml_feature_quality=None;self.ml_ring_buffer_bars=0
        self.pulse={}
    def update_signal(self,signal):
        self.current_signal=sanitize(signal);self.last_cycle_time=datetime.now(timezone.utc);self.cycle_count+=1
        self.signal_history.append(sanitize({'timestamp':signal.get('timestamp',''),'score':float(signal.get('score',0)),'direction':signal.get('direction','NEUTRAL'),'confidence':signal.get('confidence','NONE'),'regime':signal.get('regime',''),'session':signal.get('session',''),'micro':float(signal.get('engines',{}).get('microstructure',{}).get('score',0)),'flow':float(signal.get('engines',{}).get('flow',{}).get('score',0)),'macro':float(signal.get('engines',{}).get('macro',{}).get('score',0))}))
        if len(self.signal_history)>500:self.signal_history=self.signal_history[-500:]
    def update_collectors(self,s):self.collector_status=sanitize(s)
    def update_data_store(self,s):self.data_store_status=sanitize(s)
    def update_tracker(self,s):self.tracker_stats=sanitize(s)
    def update_market_data(self,d):
        self.raw_market_data=sanitize(d)
        price=d.get('last_price')or d.get('close')or 0
        if price:
            self.price_history.append({'time':datetime.now(timezone.utc).isoformat(),'price':float(price),'direction':self.current_signal.get('direction','NEUTRAL')})
            if len(self.price_history)>500:self.price_history=self.price_history[-500:]

state=DashboardState()

@app.get("/api/signal")
async def api_signal():return JSONResponse(content=sanitize(state.current_signal or{"direction":"WAITING","score":0}))
@app.get("/api/history")
async def api_history(limit:int=288):return JSONResponse(content=sanitize(state.signal_history[-limit:]))
@app.get("/api/status")
async def api_status():
    """Full status payload — legacy format (api_status.py removed in Phase 1)."""
    up = str(datetime.now(timezone.utc)-state.bot_start_time).split('.')[0] if state.bot_start_time else None
    return JSONResponse(content=sanitize({
        "uptime": up, "cycle_count": state.cycle_count,
        "collectors": state.collector_status, "data_store": state.data_store_status,
        "market": state.raw_market_data, "tracker": state.tracker_stats,
        "prices": state.price_history[-200:],
    }))

@app.get("/api/status/legacy")
async def api_status_legacy():
    """Original /api/status payload - kept for backwards compatibility."""
    up=str(datetime.now(timezone.utc)-state.bot_start_time).split('.')[0]if state.bot_start_time else None
    return JSONResponse(content=sanitize({"uptime":up,"cycle_count":state.cycle_count,"collectors":state.collector_status,"data_store":state.data_store_status,"market":state.raw_market_data,"tracker":state.tracker_stats,"prices":state.price_history[-200:]}))

@app.get("/api/health")
async def api_health():return{"status":"alive","cycles":state.cycle_count}
@app.get("/api/ml")
async def api_ml():
    if _ml_cfg is not None and getattr(_ml_cfg, "enabled", False) is False:
        return JSONResponse(content=sanitize({
            "status":"disabled",
            "label":"Disabled",
            "reason":"ML is disabled in config. Set ml.enabled: true to activate.",
            "is_trained":False,
            "probability":0.5,
            "recommendation":"N/A",
            "state":"OUT",
            "equity_pct":0,
            "win_rate":0,
            "total_trades":0,
            "feature_quality":state.ml_feature_quality,
            "ring_buffer_bars":state.ml_ring_buffer_bars,
        }))
    d=state.ml_dashboard or state.ml_data or {}
    if not d:
        d={"status":"initializing","is_trained":False,"probability":0.5,"recommendation":"WAIT","state":"OUT","equity_pct":0,"win_rate":0,"total_trades":0}
    d["feature_quality"]=state.ml_feature_quality
    d["ring_buffer_bars"]=state.ml_ring_buffer_bars
    return JSONResponse(content=sanitize(d))
@app.get("/api/hybrid")
async def api_hybrid():
    return JSONResponse(content=sanitize(state.hybrid_signal or {"direction":"WAITING","composite_score":0}))
@app.get("/api/decision")
async def api_decision():
    """Full Phase 3 decision explanation — engine scores, confidences, suppressions, conflict."""
    data = state.hybrid_signal or {}
    return JSONResponse(content=sanitize({
        "state": data.get("state", "NO_SIGNAL"),
        "direction": data.get("direction", "NEUTRAL"),
        "score": data.get("composite_score", 0),
        "confidence": data.get("confidence", "NONE"),
        "consensus": data.get("consensus_strength", 0),
        "conflict": data.get("conflict_score", 0),
        "engines": data.get("engine_scores", {}),
        "engine_confidences": data.get("engine_confidences", {}),
        "suppressions": data.get("suppression_reasons", []),
        "quality": data.get("data_completeness", 0),
        "decision_evidence": data.get("decision_grade_evidence", 0),
        "regime": data.get("regime", ""),
        "session": data.get("session", ""),
        "timestamp": data.get("timestamp_utc", ""),
    }))
@app.get("/api/pulse")
async def api_pulse():
    if _pulse_engine is not None:
        cur = _pulse_engine.current
        if cur.get("headline") and len(str(cur.get("headline",""))) > 5:
            return JSONResponse(content=sanitize(cur))
    return JSONResponse(content=sanitize(state.pulse or {"headline":"Initializing...","signal":"NEUTRAL","confidence":0}))
@app.get("/api/events")
async def api_events():
    return JSONResponse(content=sanitize({"upcoming":state.next_events}))
@app.get("/api/briefing")
async def api_briefing():
    try:
        from services.morning_briefing import MorningBriefingAgent
        agent = MorningBriefingAgent()
        text = agent.force_send(state.raw_market_data)
        return JSONResponse(content={"briefing": text, "sent_to_slack": bool(agent.webhook_url)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/report")
async def api_report():
    try:
        from services.weekly_report import WeeklyReportGenerator
        gen=WeeklyReportGenerator()
        report=gen.generate(state.raw_market_data,state.tracker_stats)
        return JSONResponse(content={"report":report,"generated":datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        return JSONResponse(content={"error":str(e)},status_code=500)

@app.get("/api/performance")
async def api_performance():
    """Signal performance tracking — win rates, Sharpe proxy, recent signals with outcomes."""
    if _signal_tracker is not None:
        try:
            perf = _signal_tracker.get_performance()
            perf['recent'] = _signal_tracker.get_recent(20)
            return JSONResponse(content=sanitize(perf))
        except Exception as e:
            logger.error("[/api/performance] Error: {}", e)
    # Fallback to state tracker_stats
    return JSONResponse(content=sanitize(state.tracker_stats or {
        "win_rate_1h": 0, "win_rate_4h": 0, "win_rate_24h": 0,
        "total_signals": 0, "recent": [],
    }))

# ── Page routes — all serve static/index.html ────────────────────────
def _serve_frontend():
    _idx = os.path.join(_STATIC_DIR, "index.html")
    if os.path.isfile(_idx):
        with open(_idx) as _f:
            return _f.read()
    return "<h1>HYDRA</h1><p>Frontend not deployed. API is live at <a href='/api/health'>/api/health</a></p>"

@app.get("/", response_class=HTMLResponse)
async def root(): return _serve_frontend()

@app.get("/market", response_class=HTMLResponse)
async def market(): return _serve_frontend()

@app.get("/yields", response_class=HTMLResponse)
async def yields(): return _serve_frontend()

@app.get("/pro", response_class=HTMLResponse)
async def pro(): return _serve_frontend()

@app.get("/venom", response_class=HTMLResponse)
async def venom(): return _serve_frontend()

@app.get("/hydra", response_class=HTMLResponse)
async def hydra(): return _serve_frontend()

def start_dashboard(host="0.0.0.0",port=None):
    import os
    if port is None:port=int(os.environ.get("PORT",8080))
    t=threading.Thread(target=lambda:uvicorn.run(app,host=host,port=port,log_level="warning"),daemon=True)
    t.start();logger.info(f"Dashboard on http://{host}:{port}");return t
