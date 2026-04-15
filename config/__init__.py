"""
HYDRA Config Loader
Loads config.yaml and provides access to all settings.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


class HydraConfig:
    """Singleton config manager for HYDRA bot."""
    
    _instance: Optional['HydraConfig'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: str = None):
        """Load configuration from yaml file."""
        if config_path is None:
            # Check env var first, then default paths
            config_path = os.environ.get(
                'HYDRA_CONFIG',
                str(Path(__file__).parent / 'config.yaml')
            )
        
        if not os.path.exists(config_path):
            logger.warning(f"Config not found at {config_path}, using template")
            config_path = str(Path(__file__).parent / 'config_template.yaml')
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Override with environment variables where available
        self._apply_env_overrides()
        
        logger.info(f"Config loaded from {config_path}")
        return self
    
    def _apply_env_overrides(self):
        """Override config values with environment variables.
        
        Railway deployment uses env vars, so these take precedence.
        Pattern: HYDRA_SECTION_KEY (e.g. HYDRA_OKX_API_KEY)
        """
        env_mappings = {
            'HYDRA_OKX_API_KEY': ('api_keys', 'okx', 'api_key'),
            'HYDRA_OKX_SECRET': ('api_keys', 'okx', 'secret_key'),
            'HYDRA_OKX_PASSPHRASE': ('api_keys', 'okx', 'passphrase'),
            'HYDRA_COINGLASS_KEY': ('api_keys', 'coinglass', 'api_key'),
            'HYDRA_CRYPTOQUANT_KEY': ('api_keys', 'cryptoquant', 'api_key'),
            'HYDRA_FRED_KEY': ('api_keys', 'fred', 'api_key'),
            'HYDRA_TELEGRAM_TOKEN': ('api_keys', 'telegram', 'bot_token'),
            'HYDRA_TELEGRAM_CHAT': ('api_keys', 'telegram', 'chat_id'),
        }
        
        for env_var, path in env_mappings.items():
            val = os.environ.get(env_var)
            if val:
                self._set_nested(path, val)
                logger.debug(f"Overriding config with env: {env_var}")
    
    def _set_nested(self, path: tuple, value: Any):
        """Set a nested config value."""
        d = self._config
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value
    
    def get(self, *keys, default=None) -> Any:
        """Get nested config value.
        
        Usage: config.get('api_keys', 'okx', 'api_key')
        """
        d = self._config
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, default)
            else:
                return default
        return d
    
    @property
    def api_keys(self) -> Dict:
        return self._config.get('api_keys', {})
    
    @property
    def targets(self) -> Dict:
        return self._config.get('targets', {})
    
    @property
    def collectors(self) -> Dict:
        return self._config.get('collectors', {})
    
    @property
    def layer1(self) -> Dict:
        return self._config.get('layer1', {})
    
    @property
    def ml(self) -> Dict:
        return self._config.get('ml', {})
    
    @property
    def sessions(self) -> Dict:
        return self._config.get('sessions', {})


# Global config instance
config = HydraConfig()
