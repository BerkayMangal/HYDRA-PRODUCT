"""
ml/signal_engine.py  (Step 3 — final production version)
──────────────────────────────────────────────────────────
HYDRA ML Signal Engine — BTC/USDC Market Timing

Integrates Step 1 (config gate) + Step 2 (unified feature pipeline)
+ Step 3 (Platt calibration, validation-derived thresholds, quality gate).

WHAT THIS MODULE DOES
---------------------
- Loads/saves model artifacts from disk (XGBoost native + pickle calibrator)
- Calls FeaturePipeline.transform_live() on every prediction cycle
- Applies Platt-calibrated ensemble probabilities
- Hard-blocks prediction if feature quality < MIN_QUALITY_SCORE
- Runs a paper-trade tracker (NOT connected to live execution)

WHAT THIS MODULE DOES NOT DO
-----------------------------
- Drive live trading decisions (Layer 1 owns that)
- Run walk-forward CV — that lives in ml/research/walk_forward.py
- Train synchronously at startup (use schedule_training() in a daemon thread)
"""

from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from xgboost import XGBClassifier
    _HAS_DEPS = True
except ImportError as _e:
    _HAS_DEPS = False
    logger.error("MLSignalEngine: deps missing ({}). Engine disabled.", _e)

from config.settings import MLConfig
from features.pipeline import FeaturePipeline, FEATURE_NAMES, MIN_QUALITY_SCORE

# Phase 5: Paper-trade cost aligned with ml.research.walk_forward_v2.CostModel
# fee=0.1% + slippage=2bps/side = 0.07% per side (0.14% round-trip)
_COST_PER_SIDE: float = (0.001 + 2 * 2 / 10_000) / 2   # 0.0007

_MODEL_PRIMARY_PATH   = "data/ml_model_primary.json"
_MODEL_SECONDARY_PATH = "data/ml_model_secondary.json"
_CALIBRATOR_PATH      = "data/ml_calibrator.pkl"
_META_PATH            = "data/ml_meta.json"
_PAPER_TRADE_PATH     = "data/ml_paper_trade.json"


class MLSignalEngine:
    """Production ML signal engine for BTC/USDC market timing (paper-trade only)."""

    def __init__(self, cfg: MLConfig) -> None:
        self.cfg              = cfg
        self.is_trained: bool = False
        self.is_enabled: bool = _HAS_DEPS

        self._model_primary:   Optional["XGBClassifier"]      = None
        self._model_secondary: Optional["XGBClassifier"]      = None
        self._calibrator:      Optional["LogisticRegression"] = None
        self._feature_cols:    List[str]                      = []
        self._threshold_in:    float                          = cfg.in_threshold
        self._threshold_out:   float                          = cfg.out_threshold
        self._last_train_time: float                          = 0.0
        self._train_metadata:  Dict[str, Any]                 = {}

        self._pipeline = FeaturePipeline()

        self._pt_state:       str              = "OUT"
        self._pt_entry_price: float            = 0.0
        self._pt_last_switch: Optional[datetime] = None
        self._pt_equity:      float            = 1.0
        self._pt_trades:      List[Dict]       = []
        self._pt_equity_hist: List[Dict]       = []
        self._last_prob:      float            = 0.5
        self._last_rec:       str              = "WAIT"
        self._last_quality:   float            = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────

    def schedule_training(self) -> bool:
        """Fetch history, compute features, train + calibrate ensemble. Thread-safe call."""
        if not _HAS_DEPS:
            return False
        logger.info("[ML] Training started (train_days={})", self.cfg.train_days)
        try:
            df = self._fetch_training_data()
            if df is None:
                return False

            self._pipeline.warm_up(df)
            feature_df = self._pipeline.transform_batch(df)
            X, y, cols = self._prepare_xy(feature_df)

            if len(X) < self.cfg.min_samples_for_train:
                logger.warning("[ML] {} rows < min {}. Skip.", len(X), self.cfg.min_samples_for_train)
                return False

            split  = int(len(X) * 0.70)
            X_fit, X_cal = X[:split], X[split:]
            y_fit, y_cal = y[:split], y[split:]

            n_neg = (y_fit == 0).sum()
            n_pos = (y_fit == 1).sum()
            spw   = n_neg / max(n_pos, 1)
            logger.info("[ML] fit={} rows | pos_rate={:.1%} | spw={:.2f}", len(X_fit), n_pos / max(len(y_fit), 1), spw)

            self._model_primary, self._model_secondary = self._fit_ensemble(X_fit, y_fit, spw)
            self._calibrator = self._fit_calibrator(X_cal, y_cal)

            val_metrics = self._evaluate(X_cal, y_cal, label="calibration-holdout")

            prices_cal = feature_df["close"].values[split: split + len(X_cal)]
            p_val = np.array([self._calibrated_predict(X_cal[[i]]) for i in range(len(X_cal))])
            self._threshold_in  = self._optimise_threshold(p_val, y_cal, prices_cal)
            self._threshold_out = max(0.30, self._threshold_in - 0.05)

            self._feature_cols    = cols
            self.is_trained       = True
            self._last_train_time = time.time()
            self._train_metadata  = {
                "trained_at":     datetime.now(timezone.utc).isoformat(),
                "train_rows":     len(X_fit),
                "feature_cols":   cols,
                "threshold_in":   self._threshold_in,
                "threshold_out":  self._threshold_out,
                "val_auc":        val_metrics.get("auc", 0),
                "val_logloss":    val_metrics.get("logloss", 0),
                "calibration_ece": val_metrics.get("ece", 0),
                "pos_rate_train": float(n_pos / max(len(y_fit), 1)),
                "note": (
                    "Single chronological split. For rigorous OOS evaluation, "
                    "run ml/research/walk_forward.py separately."
                ),
            }
            self._save_model()
            logger.info("[ML] Done ✅ thr_in={:.2f} thr_out={:.2f}", self._threshold_in, self._threshold_out)
            return True
        except Exception as exc:
            logger.error("[ML] Training failed: {}", exc, exc_info=True)
            return False

    # ──────────────────────────────────────────────────────────────────
    # Live prediction
    # ──────────────────────────────────────────────────────────────────

    def predict(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate paper-trade signal. NOT used in live trading decisions."""
        if not self.is_trained or not _HAS_DEPS or not self.is_enabled:
            return self._disabled_signal("not_trained")

        hours_since = (time.time() - self._last_train_time) / 3600
        if hours_since > self.cfg.retrain_interval_hours:
            logger.info("[ML] Retrain due ({}h). Background scheduler should retrain.", int(hours_since))

        funding = market_data.get("funding_rate") or market_data.get("next_funding_rate")
        self._pipeline.ingest_candle(market_data, funding_rate=funding)
        if any(k in market_data for k in ("vix_current", "qqq_current", "fear_greed_value")):
            self._pipeline.ingest_macro(market_data)

        fv = self._pipeline.transform_live()
        if fv is None:
            return self._disabled_signal("pipeline_empty")

        self._last_quality = fv.quality_score

        if not fv.can_predict:
            logger.warning("[ML] Suppressed: {} (min={:.0%})", fv.quality.summary(), MIN_QUALITY_SCORE)
            return {
                **self._disabled_signal("quality_gate"),
                "ml_feature_quality":   fv.quality_score,
                "ml_ring_buffer_bars":  fv.bar_count,
                "ml_quality_detail": {
                    "real": fv.quality.real[:5],
                    "external": fv.quality.external[:3],
                    "default": fv.quality.default[:5],
                },
            }

        try:
            X    = fv.to_array(self._feature_cols).reshape(1, -1)
            prob = self._calibrated_predict(X)
        except Exception as exc:
            logger.error("[ML] Inference error: {}", exc)
            return self._disabled_signal("inference_error")

        self._last_prob = prob
        price = float(market_data.get("last_price") or market_data.get("close") or 0)
        if price > 0:
            self._update_paper_trade(prob, price)
        rec, conf = self._make_recommendation(prob)
        self._last_rec = rec

        return {
            "ml_enabled":          True,
            "ml_probability":      round(prob, 4),
            "ml_recommendation":   rec,
            "ml_confidence":       conf,
            "ml_state":            self._pt_state,
            "ml_equity":           round(self._pt_equity, 6),
            "ml_equity_pct":       round((self._pt_equity - 1) * 100, 2),
            "ml_entry_price":      self._pt_entry_price if self._pt_state == "IN" else 0,
            "ml_unrealized_pnl":   self._unrealised_pnl(price),
            "ml_total_trades":     self._n_closed_trades(),
            "ml_win_rate":         self._win_rate(),
            "ml_last_switch":      self._pt_last_switch.isoformat() if self._pt_last_switch else None,
            "ml_is_trained":       True,
            "ml_threshold_in":     self._threshold_in,
            "ml_threshold_out":    self._threshold_out,
            "ml_feature_quality":  round(fv.quality_score, 3),
            "ml_ring_buffer_bars": fv.bar_count,
            "ml_warning":          None,
        }

    # ──────────────────────────────────────────────────────────────────
    # Inference helpers
    # ──────────────────────────────────────────────────────────────────

    def _calibrated_predict(self, X: np.ndarray) -> float:
        p1  = float(self._model_primary.predict_proba(X)[:, 1][0])
        p2  = float(self._model_secondary.predict_proba(X)[:, 1][0])
        raw = (p1 + p2) / 2
        if self._calibrator is None:
            return raw
        return float(self._calibrator.predict_proba(np.array([[raw]]))[:, 1][0])

    def _make_recommendation(self, prob: float) -> Tuple[str, str]:
        thr_in, thr_out = self._threshold_in, self._threshold_out
        if self._pt_state == "OUT" and prob >= thr_in:
            return "BUY_SIGNAL",  "HIGH" if prob >= thr_in + 0.10 else "MEDIUM"
        if self._pt_state == "IN" and prob < thr_out:
            return "SELL_SIGNAL", "HIGH" if prob < thr_out - 0.10 else "MEDIUM"
        if self._pt_state == "IN":
            return "HOLD_BTC",    "HIGH" if prob >= thr_in else "LOW"
        return "HOLD_USDC",       "HIGH" if prob < thr_out else "LOW"

    # ──────────────────────────────────────────────────────────────────
    # Training helpers
    # ──────────────────────────────────────────────────────────────────

    def _fetch_training_data(self):
        try:
            import ccxt
            from datetime import timedelta
            exchange = ccxt.okx({"options": {"defaultType": "swap"}})
            since_ms = exchange.parse8601(
                (datetime.utcnow() - timedelta(days=self.cfg.train_days)).isoformat()
            )
            candles = []
            while True:
                batch = exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", since=since_ms, limit=300)
                if not batch:
                    break
                candles.extend(batch)
                since_ms = batch[-1][0] + 1
                if len(batch) < 300:
                    break
                time.sleep(0.15)
            if len(candles) < self.cfg.min_samples_for_train:
                return None
            import pandas as pd
            df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            logger.info("[ML] Fetched {} 1H candles", len(df))
            df = self._attach_macro(df)
            df = self._attach_funding(df)
            df = self._attach_fear_greed(df)
            return df
        except Exception as exc:
            logger.error("[ML] Data fetch failed: {}", exc)
            return None

    def _attach_macro(self, df):
        try:
            import yfinance as yf
            for ticker, col in {"^VIX": "vix", "QQQ": "qqq", "^TNX": "us10y", "GC=F": "gold"}.items():
                try:
                    raw = yf.download(ticker, period=f"{self.cfg.train_days+5}d", interval="1d",
                                      progress=False, auto_adjust=True)
                    if raw.empty:
                        continue
                    close = raw["Close"].squeeze()
                    if close.index.tz is None:
                        close.index = close.index.tz_localize("UTC")
                    else:
                        close.index = close.index.tz_convert("UTC")
                    df[col] = close.reindex(df.index, method="ffill")
                except Exception as e:
                    logger.debug("[ML] Macro {} failed: {}", ticker, e)
        except ImportError:
            logger.warning("[ML] yfinance not installed")
        return df

    def _attach_funding(self, df):
        try:
            import requests
            records, after = [], ""
            for _ in range(200):
                params = {"instId": "BTC-USDT-SWAP", "limit": "100"}
                if after:
                    params["after"] = after
                r = requests.get("https://www.okx.com/api/v5/public/funding-rate-history",
                                  params=params, timeout=15)
                data = r.json().get("data", [])
                if not data:
                    break
                for row in data:
                    records.append({"ts": int(row["fundingTime"]), "fr": float(row["fundingRate"])})
                after = data[-1]["fundingTime"]
                time.sleep(0.2)
                if len(data) < 100:
                    break
            if records:
                import pandas as pd
                fdf = pd.DataFrame(records)
                fdf["ts"] = pd.to_datetime(fdf["ts"], unit="ms", utc=True)
                fdf = fdf.set_index("ts").sort_index()
                df["funding"] = fdf["fr"].resample("1h").last().ffill().reindex(df.index, method="ffill")
        except Exception as e:
            logger.debug("[ML] Funding: {}", e)
        return df

    def _attach_fear_greed(self, df):
        try:
            import requests, pandas as pd
            r = requests.get("https://api.alternative.me/fng/", params={"limit": "200"}, timeout=10)
            data = r.json().get("data", [])
            if data:
                fdf = pd.DataFrame(data)
                fdf["ts"] = pd.to_datetime(fdf["timestamp"].astype(int), unit="s", utc=True)
                fdf["value"] = fdf["value"].astype(int)
                fdf = fdf.set_index("ts").sort_index()
                df["fear_greed"] = fdf["value"].resample("1h").last().ffill().reindex(df.index, method="ffill")
        except Exception as e:
            logger.debug("[ML] F&G: {}", e)
        return df

    def _prepare_xy(self, df):
        fwd_h   = self.cfg.forward_return_hours
        fwd_ret = df["close"].pct_change(fwd_h).shift(-fwd_h)
        target  = (fwd_ret >= 0.005).astype(int)
        cols    = [c for c in FEATURE_NAMES if c in df.columns]
        mdf     = df[cols].copy()
        mdf["target"] = target
        mdf     = mdf.dropna(subset=["target"])
        mdf     = mdf.replace([np.inf, -np.inf], np.nan)
        medians = mdf[cols].median()
        mdf[cols] = mdf[cols].fillna(medians)
        mdf     = mdf.dropna(subset=cols)
        X = mdf[cols].values.astype(np.float32)
        y = mdf["target"].values.astype(np.int32)
        return X, y, cols

    def _fit_ensemble(self, X, y, spw):
        shared = dict(eval_metric="logloss", verbosity=0, scale_pos_weight=spw, random_state=42)
        m1 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.02,
                            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0,
                            reg_lambda=3.0, min_child_weight=20, **shared)
        m2 = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5,
                            reg_lambda=2.0, min_child_weight=15, **shared)
        m1.fit(X, y)
        m2.fit(X, y)
        return m1, m2

    def _fit_calibrator(self, X_cal, y_cal):
        if len(X_cal) < 20 or len(np.unique(y_cal)) < 2:
            logger.warning("[ML] Calibration slice too small — Platt skipped")
            return None
        p1  = self._model_primary.predict_proba(X_cal)[:, 1]
        p2  = self._model_secondary.predict_proba(X_cal)[:, 1]
        raw = ((p1 + p2) / 2).reshape(-1, 1)
        cal = LogisticRegression(C=1.0, max_iter=1000)
        cal.fit(raw, y_cal)
        return cal

    def _evaluate(self, X, y, label: str) -> Dict[str, float]:
        try:
            prob = np.array([self._calibrated_predict(X[[i]]) for i in range(len(X))])
            auc  = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.5
            ll   = log_loss(y, prob)
            ece  = self._ece(prob, y)
            logger.info("[ML] {} → AUC={:.3f} LL={:.3f} ECE={:.3f}", label, auc, ll, ece)
            return {"auc": auc, "logloss": ll, "ece": ece}
        except Exception as exc:
            logger.warning("[ML] Eval error: {}", exc)
            return {}

    @staticmethod
    def _ece(probs, labels, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        errs, wts = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (probs >= lo) & (probs < hi)
            if m.sum() == 0:
                continue
            errs.append(abs(probs[m].mean() - labels[m].mean()))
            wts.append(m.sum())
        return float(np.average(errs, weights=wts)) if errs else 0.0

    def _optimise_threshold(self, probs, labels, prices) -> float:
        best_thr, best_sr = self.cfg.in_threshold, -np.inf
        n = min(len(probs), len(prices))
        if n < 10:
            return best_thr
        for thr_f in np.arange(0.35, 0.66, 0.02):
            thr = float(thr_f)
            sr  = self._quick_sharpe(probs[:n], prices[:n], thr, thr - 0.05)
            if sr > best_sr:
                best_sr, best_thr = sr, thr
        logger.info("[ML] Threshold opt: {:.2f} (val Sharpe={:.2f})", best_thr, best_sr)
        return best_thr

    @staticmethod
    def _quick_sharpe(probs, prices, thr_in, thr_out, min_hold=48):
        n = min(len(probs), len(prices))
        state, entry, equity = "OUT", 0.0, 1.0
        last_sw = -min_hold - 1
        daily, prev = [], equity
        for t in range(n):
            if t % 24 == 0 and t > 0:
                daily.append(equity / prev - 1)
                prev = equity
            if (t - last_sw) < min_hold:
                if state == "OUT":
                    equity *= 1 + (0.05 / 365 / 24)
                continue
            if state == "OUT" and probs[t] >= thr_in:
                equity *= 0.9995
                state, entry, last_sw = "IN", prices[t], t
            elif state == "IN" and probs[t] < thr_out:
                ret = (prices[t] - entry) / entry
                equity *= (1 + ret * 0.9995)
                state, entry, last_sw = "OUT", 0.0, t
            elif state == "OUT":
                equity *= 1 + (0.05 / 365 / 24)
        if len(daily) < 5:
            return 0.0
        dr = np.array(daily)
        return float(dr.mean() / (dr.std() + 1e-9) * np.sqrt(365))

    # ──────────────────────────────────────────────────────────────────
    # Paper trade
    # ──────────────────────────────────────────────────────────────────

    def _update_paper_trade(self, prob: float, price: float) -> None:
        now = datetime.now(timezone.utc)
        if self._pt_state == "IN" and self._pt_entry_price > 0:
            loss = (price - self._pt_entry_price) / self._pt_entry_price
            if loss < -self.cfg.stop_loss_pct:
                self._pt_equity *= (1 + loss) * (1 - _COST_PER_SIDE)
                self._pt_trades.append({"time": now.isoformat(), "action": "SL",
                                         "price": price, "pnl_pct": round(loss*100, 2), "prob": prob})
                self._pt_state, self._pt_entry_price, self._pt_last_switch = "OUT", 0.0, now
                logger.warning("[ML:PT] STOP LOSS ${:,.0f} | {:.2%}", price, loss)
                self._save_paper_trade()
                return

        if self._pt_last_switch:
            hold_h = (now - self._pt_last_switch).total_seconds() / 3600
            if hold_h < self.cfg.min_hold_hours:
                if self._pt_state == "OUT":
                    self._pt_equity *= 1 + (0.05 / 365 / 24)
                self._record_equity(price, prob)
                return

        if self._pt_state == "OUT" and prob >= self._threshold_in:
            self._pt_equity *= 1 - _COST_PER_SIDE
            self._pt_state, self._pt_entry_price, self._pt_last_switch = "IN", price, now
            self._pt_trades.append({"time": now.isoformat(), "action": "BUY", "price": price, "prob": prob})
            logger.info("[ML:PT] BUY ${:,.0f} p={:.2%}", price, prob)
            self._save_paper_trade()

        elif self._pt_state == "IN" and prob < self._threshold_out:
            ret = (price - self._pt_entry_price) / self._pt_entry_price
            self._pt_equity *= (1 + ret) * (1 - _COST_PER_SIDE)
            self._pt_trades.append({"time": now.isoformat(), "action": "SELL",
                                     "price": price, "pnl_pct": round(ret*100, 2), "prob": prob})
            logger.info("[ML:PT] SELL ${:,.0f} pnl={:.2%}", price, ret)
            self._pt_state, self._pt_entry_price, self._pt_last_switch = "OUT", 0.0, now
            self._save_paper_trade()
        elif self._pt_state == "OUT":
            self._pt_equity *= 1 + (0.05 / 365 / 24)

        self._record_equity(price, prob)

    def _record_equity(self, price: float, prob: float) -> None:
        self._pt_equity_hist.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "equity": round(self._pt_equity, 6),
            "equity_pct": round((self._pt_equity - 1) * 100, 2),
            "state": self._pt_state, "price": price, "prob": prob,
        })
        if len(self._pt_equity_hist) > 2000:
            self._pt_equity_hist = self._pt_equity_hist[-2000:]

    def _unrealised_pnl(self, price: float) -> float:
        if self._pt_state != "IN" or self._pt_entry_price <= 0:
            return 0.0
        return round((price - self._pt_entry_price) / self._pt_entry_price * 100, 2)

    def _n_closed_trades(self) -> int:
        return sum(1 for t in self._pt_trades if t.get("action") in ("SELL", "SL"))

    def _win_rate(self) -> Optional[float]:
        closed = [t for t in self._pt_trades if t.get("action") in ("SELL", "SL")]
        if not closed:
            return None
        return round(sum(1 for t in closed if t.get("pnl_pct", 0) > 0) / len(closed) * 100, 1)

    # ──────────────────────────────────────────────────────────────────
    # Dashboard
    # ──────────────────────────────────────────────────────────────────

    def get_dashboard_data(self) -> Dict[str, Any]:
        return {
            "enabled":          True,
            "probability":      self._last_prob,
            "recommendation":   self._last_rec,
            "state":            self._pt_state,
            "equity":           round(self._pt_equity, 6),
            "equity_pct":       round((self._pt_equity - 1) * 100, 2),
            "entry_price":      self._pt_entry_price,
            "total_trades":     self._n_closed_trades(),
            "win_rate":         self._win_rate(),
            "is_trained":       self.is_trained,
            "feature_count":    len(self._feature_cols),
            "feature_quality":  round(self._last_quality, 3),
            "ring_buffer_bars": len(self._pipeline._ring),
            "trades":           self._pt_trades[-20:],
            "equity_history":   self._pt_equity_hist[-500:],
            "training_meta":    self._train_metadata,
            "thresholds": {
                "in":             self._threshold_in,
                "out":            self._threshold_out,
                "source":         "validation-derived" if self.is_trained else "config-default",
                "min_hold_hours": self.cfg.min_hold_hours,
                "stop_loss_pct":  self.cfg.stop_loss_pct * 100,
            },
            "disclaimer": (
                "Paper-trade only. NOT used in live trading. "
                "For rigorous OOS metrics: ml/research/walk_forward.py"
            ),
        }

    def _disabled_signal(self, reason: str) -> Dict[str, Any]:
        return {
            "ml_enabled":         self.is_enabled,
            "ml_probability":     None,
            "ml_recommendation":  "ML_DISABLED",
            "ml_confidence":      "NONE",
            "ml_state":           self._pt_state,
            "ml_equity":          round(self._pt_equity, 6),
            "ml_equity_pct":      round((self._pt_equity - 1) * 100, 2),
            "ml_entry_price":     None,
            "ml_unrealized_pnl":  None,
            "ml_total_trades":    self._n_closed_trades(),
            "ml_win_rate":        self._win_rate(),
            "ml_last_switch":     self._pt_last_switch.isoformat() if self._pt_last_switch else None,
            "ml_is_trained":      self.is_trained,
            "ml_threshold_in":    self._threshold_in,
            "ml_threshold_out":   self._threshold_out,
            "ml_feature_quality": self._last_quality,
            "ml_ring_buffer_bars": len(self._pipeline._ring),
            "ml_warning":         f"Signal suppressed: {reason}",
        }

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def _save_model(self) -> None:
        os.makedirs("data", exist_ok=True)
        try:
            self._model_primary.save_model(_MODEL_PRIMARY_PATH)
            self._model_secondary.save_model(_MODEL_SECONDARY_PATH)
            if self._calibrator:
                with open(_CALIBRATOR_PATH, "wb") as fh:
                    pickle.dump(self._calibrator, fh)
            with open(_META_PATH, "w") as fh:
                json.dump(self._train_metadata, fh, indent=2)
            logger.info("[ML] Artifacts saved")
        except Exception as exc:
            logger.error("[ML] Model save failed: {}", exc)

    def load_model(self) -> bool:
        try:
            if not all(os.path.exists(p) for p in [_MODEL_PRIMARY_PATH, _META_PATH]):
                logger.info("[ML] No persisted model — training required")
                return False
            self._model_primary   = XGBClassifier()
            self._model_secondary = XGBClassifier()
            self._model_primary.load_model(_MODEL_PRIMARY_PATH)
            self._model_secondary.load_model(_MODEL_SECONDARY_PATH)
            if os.path.exists(_CALIBRATOR_PATH):
                with open(_CALIBRATOR_PATH, "rb") as fh:
                    self._calibrator = pickle.load(fh)
            with open(_META_PATH) as fh:
                self._train_metadata = json.load(fh)
            self._feature_cols    = self._train_metadata.get("feature_cols", FEATURE_NAMES)
            self._threshold_in    = float(self._train_metadata.get("threshold_in",  self.cfg.in_threshold))
            self._threshold_out   = float(self._train_metadata.get("threshold_out", self.cfg.out_threshold))
            self.is_trained       = True
            self._last_train_time = os.path.getmtime(_META_PATH)
            logger.info("[ML] Model loaded | thr_in={:.2f} thr_out={:.2f}",
                        self._threshold_in, self._threshold_out)
            return True
        except Exception as exc:
            logger.error("[ML] Model load failed: {}", exc)
            return False

    def load_state(self) -> None:
        """Backward-compat alias for main.py."""
        self.load_model()
        self._load_paper_trade()

    def _load_paper_trade(self) -> None:
        if not os.path.exists(_PAPER_TRADE_PATH):
            return
        try:
            with open(_PAPER_TRADE_PATH) as fh:
                s = json.load(fh)
            self._pt_state, self._pt_entry_price, self._pt_equity = (
                s.get("state", "OUT"), float(s.get("entry_price", 0)), float(s.get("equity", 1.0))
            )
            self._pt_trades = s.get("trades", [])
            if raw := s.get("last_switch"):
                self._pt_last_switch = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
            logger.info("[ML:PT] Loaded {} equity={:.4f} ({:+.2f}%)",
                        self._pt_state, self._pt_equity, (self._pt_equity-1)*100)
        except Exception as exc:
            logger.warning("[ML:PT] Load failed: {}", exc)

    def _save_paper_trade(self) -> None:
        try:
            os.makedirs("data", exist_ok=True)
            with open(_PAPER_TRADE_PATH, "w") as fh:
                json.dump({
                    "state": self._pt_state, "entry_price": self._pt_entry_price,
                    "equity": self._pt_equity,
                    "last_switch": self._pt_last_switch.isoformat() if self._pt_last_switch else None,
                    "trades": self._pt_trades[-100:],
                }, fh, indent=2)
        except Exception as exc:
            logger.debug("[ML:PT] Save failed: {}", exc)
