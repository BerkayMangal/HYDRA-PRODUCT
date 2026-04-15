"""
ml/__init__.py — ML subsystem entry point.

build_ml_engine() is the single public factory. It returns either a live
MLSignalEngine or a zero-overhead _MLEngineStub based on:
  1. settings.ml.enabled (config gate)
  2. xgboost importable (library gate)
"""
from __future__ import annotations
from typing import Any, Dict
from loguru import logger


class _MLEngineStub:
    """No-op engine returned when ML is disabled or unavailable."""
    is_trained: bool = False
    is_enabled: bool = False
    _pipeline = None

    def predict(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._disabled()

    def load_state(self) -> None:
        pass

    def schedule_training(self) -> bool:
        return False

    def get_dashboard_data(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "reason": "ML disabled (ml.enabled: false or xgboost missing)",
            "probability": None,
            "recommendation": "ML_DISABLED",
            "state": "DISABLED",
            "equity": None, "equity_pct": None, "is_trained": False,
        }

    @staticmethod
    def _disabled() -> Dict[str, Any]:
        return {
            "ml_enabled": False,
            "ml_probability": None,
            "ml_recommendation": "ML_DISABLED",
            "ml_confidence": "NONE",
            "ml_state": "DISABLED",
            "ml_equity": None, "ml_equity_pct": None,
            "ml_entry_price": None, "ml_unrealized_pnl": None,
            "ml_total_trades": None, "ml_win_rate": None,
            "ml_last_switch": None, "ml_is_trained": False,
            "ml_threshold_in": None, "ml_threshold_out": None,
            "ml_feature_quality": None, "ml_ring_buffer_bars": 0,
            "ml_warning": "ML disabled",
        }


def build_ml_engine():
    """
    Return MLSignalEngine or _MLEngineStub.

    Gates on settings.ml.enabled AND xgboost availability.
    """
    try:
        from config.settings import settings
        settings.assert_loaded()
        if not settings.ml.enabled:
            logger.info("ML engine DISABLED by config (ml.enabled: false)")
            return _MLEngineStub()
    except Exception:
        return _MLEngineStub()

    try:
        import xgboost  # noqa
    except ImportError:
        logger.error("ml.enabled=true but xgboost not installed. Falling back to stub.")
        return _MLEngineStub()

    try:
        from ml.signal_engine import MLSignalEngine
        engine = MLSignalEngine(settings.ml)
        engine.load_state()
        logger.info("ML engine ENABLED (paper-trade / research mode only)")
        return engine
    except Exception as exc:
        logger.error("ML engine init failed: {}. Falling back to stub.", exc)
        return _MLEngineStub()


__all__ = ["build_ml_engine", "_MLEngineStub"]
