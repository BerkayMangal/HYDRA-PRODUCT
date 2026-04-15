"""
config/settings.py
──────────────────
Typed, validated configuration layer for HYDRA.

Design principles
-----------------
* Every field has an explicit type and default.
* Config is loaded once; all subsystems read from the singleton.
* ML is OFF by default. Enabling it requires an explicit opt-in in
  config.yaml AND xgboost being installed. Both conditions must hold.
* No subsystem initialises unless its config section exists AND is enabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Sub-section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OKXConfig:
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class APIKeysConfig:
    okx: OKXConfig = field(default_factory=OKXConfig)
    coinglass_key: str = ""
    cryptoquant_key: str = ""
    fred_key: str = ""
    perplexity_key: str = ""
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


@dataclass
class Layer1Config:
    signal_threshold: float = 35.0
    engine_weights: Dict[str, float] = field(default_factory=lambda: {
        "microstructure": 0.45,
        "flow": 0.30,
        "macro": 0.25,
    })
    confidence_levels: Dict[str, float] = field(default_factory=lambda: {
        "low": 35.0,
        "medium": 50.0,
        "high": 70.0,
        "extreme": 85.0,
    })


@dataclass
class MLConfig:
    """
    ML is disabled by default.

    To enable:
      1. Set ml.enabled: true in config.yaml
      2. Install xgboost (pip install xgboost)

    Even when enabled, ML does NOT participate in live trading decisions.
    It runs as a research/paper-trade module only. See ml/research/ for
    the proper walk-forward research pipeline.
    """
    enabled: bool = False
    retrain_interval_hours: int = 24
    min_samples_for_train: int = 500      # Raised from 200 — 200 is too few for 24-feature XGB
    forward_return_hours: int = 24
    train_days: int = 90
    in_threshold: float = 0.50            # Must be derived from validation, not set by hand
    out_threshold: float = 0.45           # Must be derived from validation, not set by hand
    min_hold_hours: int = 48
    stop_loss_pct: float = 0.08
    fee_pct: float = 0.001


@dataclass
class BacktestConfig:
    train_days: int = 90
    test_days: int = 30
    step_days: int = 7
    min_train_samples: int = 300


@dataclass
class DeploymentConfig:
    log_level: str = "INFO"
    data_persist_path: str = "./data"
    heartbeat_interval_sec: int = 60


@dataclass
class HydraSettings:
    """
    Top-level validated settings for the HYDRA system.

    Usage
    -----
    >>> from config.settings import settings
    >>> settings.load()
    >>> if settings.ml.enabled:
    ...     init_ml()
    """
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    layer1: Layer1Config = field(default_factory=Layer1Config)
    ml: MLConfig = field(default_factory=MLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    _loaded: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, config_path: Optional[str] = None) -> "HydraSettings":
        """
        Load settings from YAML.

        Search order:
          1. Explicit `config_path` argument
          2. HYDRA_CONFIG environment variable
          3. config/config.yaml (next to this file)
          4. config/config_template.yaml (fallback, logs a warning)
        """
        path = self._resolve_path(config_path)
        with open(path) as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        self._load_api_keys(raw)
        self._load_layer1(raw)
        self._load_ml(raw)
        self._load_backtest(raw)
        self._load_deployment(raw)
        self._apply_env_overrides()

        self._loaded = True
        logger.info("Settings loaded from {}", path)
        self._log_enabled_subsystems()
        return self

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _resolve_path(self, config_path: Optional[str]) -> Path:
        """Resolve config file path with fallback chain."""
        candidates = [
            config_path,
            os.environ.get("HYDRA_CONFIG"),
            str(Path(__file__).parent / "config.yaml"),
            str(Path(__file__).parent / "config_template.yaml"),
        ]
        for c in candidates:
            if c and Path(c).exists():
                return Path(c)
        raise FileNotFoundError(
            "No config file found. Copy config/config_template.yaml to "
            "config/config.yaml and fill in your credentials."
        )

    def _load_api_keys(self, raw: Dict) -> None:
        keys = raw.get("api_keys", {})
        okx = keys.get("okx", {})
        self.api_keys = APIKeysConfig(
            okx=OKXConfig(
                api_key=okx.get("api_key", ""),
                secret_key=okx.get("secret_key", ""),
                passphrase=okx.get("passphrase", ""),
            ),
            coinglass_key=keys.get("coinglass", {}).get("api_key", ""),
            cryptoquant_key=keys.get("cryptoquant", {}).get("api_key", ""),
            fred_key=keys.get("fred", {}).get("api_key", ""),
            perplexity_key=keys.get("perplexity", {}).get("api_key", ""),
            telegram=TelegramConfig(
                bot_token=keys.get("telegram", {}).get("bot_token", ""),
                chat_id=keys.get("telegram", {}).get("chat_id", ""),
            ),
        )

    def _load_layer1(self, raw: Dict) -> None:
        l1 = raw.get("layer1", {})
        self.layer1 = Layer1Config(
            signal_threshold=float(l1.get("signal_threshold", 35.0)),
            engine_weights=l1.get("engine_weights", self.layer1.engine_weights),
            confidence_levels=l1.get("confidence_levels", self.layer1.confidence_levels),
        )

    def _load_ml(self, raw: Dict) -> None:
        """
        Load ML config.

        IMPORTANT: ml.enabled defaults to False. The system deliberately
        requires an explicit opt-in. If the key is absent from config.yaml,
        ML remains disabled.
        """
        ml = raw.get("ml", {})
        self.ml = MLConfig(
            enabled=bool(ml.get("enabled", False)),     # ← explicit False default
            retrain_interval_hours=int(ml.get("retrain_interval_hours", 24)),
            min_samples_for_train=int(ml.get("min_samples_for_train", 500)),
            forward_return_hours=int(ml.get("forward_return_hours", 24)),
            train_days=int(ml.get("train_days", 90)),
            in_threshold=float(ml.get("in_threshold", 0.50)),
            out_threshold=float(ml.get("out_threshold", 0.45)),
            min_hold_hours=int(ml.get("min_hold_hours", 48)),
            stop_loss_pct=float(ml.get("stop_loss_pct", 0.08)),
            fee_pct=float(ml.get("fee_pct", 0.001)),
        )

    def _load_backtest(self, raw: Dict) -> None:
        bt = raw.get("backtest", {}).get("walk_forward", {})
        self.backtest = BacktestConfig(
            train_days=int(bt.get("train_days", 90)),
            test_days=int(bt.get("test_days", 30)),
            step_days=int(bt.get("step_days", 7)),
            min_train_samples=int(bt.get("min_train_samples", 300)),
        )

    def _load_deployment(self, raw: Dict) -> None:
        dep = raw.get("deployment", {})
        self.deployment = DeploymentConfig(
            log_level=dep.get("log_level", "INFO"),
            data_persist_path=dep.get("data_persist_path", "./data"),
            heartbeat_interval_sec=int(dep.get("heartbeat_interval_sec", 60)),
        )

    def _apply_env_overrides(self) -> None:
        """
        Apply Railway environment variable overrides.

        These take precedence over file values so secrets are never
        committed to source control.
        """
        if v := os.environ.get("HYDRA_OKX_API_KEY"):
            self.api_keys.okx.api_key = v
        if v := os.environ.get("HYDRA_OKX_SECRET"):
            self.api_keys.okx.secret_key = v
        if v := os.environ.get("HYDRA_OKX_PASSPHRASE"):
            self.api_keys.okx.passphrase = v
        if v := os.environ.get("HYDRA_COINGLASS_KEY"):
            self.api_keys.coinglass_key = v
        if v := os.environ.get("HYDRA_CRYPTOQUANT_KEY"):
            self.api_keys.cryptoquant_key = v
        if v := os.environ.get("HYDRA_FRED_KEY"):
            self.api_keys.fred_key = v
        if v := os.environ.get("HYDRA_PERPLEXITY_KEY"):
            self.api_keys.perplexity_key = v
        if v := os.environ.get("PERPLEXITY_API_KEY"):
            self.api_keys.perplexity_key = v
        if v := os.environ.get("HYDRA_TELEGRAM_TOKEN"):
            self.api_keys.telegram.bot_token = v
        if v := os.environ.get("HYDRA_TELEGRAM_CHAT"):
            self.api_keys.telegram.chat_id = v

    def _log_enabled_subsystems(self) -> None:
        """Log which optional subsystems are active on startup."""
        logger.info("─── Enabled subsystems ───")
        logger.info(
            "  CoinGlass   : {}",
            "✅" if self.api_keys.coinglass_key else "❌ (no key)",
        )
        logger.info(
            "  CryptoQuant : {}",
            "✅" if self.api_keys.cryptoquant_key else "❌ (no key)",
        )
        logger.info(
            "  FRED        : {}",
            "✅" if self.api_keys.fred_key else "❌ (no key)",
        )
        logger.info(
            "  Perplexity  : {}",
            "✅" if self.api_keys.perplexity_key else "❌ (no key)",
        )
        logger.info(
            "  Telegram    : {}",
            "✅" if self.api_keys.telegram.bot_token else "❌ (no token)",
        )
        if self.ml.enabled:
            logger.info("  ML Engine   : ⚠️  ENABLED (paper-trade / research only)")
        else:
            logger.info("  ML Engine   : 🔒 DISABLED (set ml.enabled: true to activate)")

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def assert_loaded(self) -> None:
        """Raise if settings were never loaded — catches missing load() call."""
        if not self._loaded:
            raise RuntimeError(
                "HydraSettings.load() was never called. "
                "Call settings.load() at application startup."
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

settings = HydraSettings()
