"""
Configuration management for Playlister.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from loguru import logger


class Config:
    """Configuration manager."""

    DEFAULT_CONFIG = {
        'paths': {
            'music': '/music',
            'data': '/data',
            'playlists': '/playlists',
            'config': '/config'
        },
        'database': {
            'path': '/data/playlister.db',
            'backup_on_startup': True,
            'backup_count': 5
        },
        'extraction': {
            'workers': 4,
            'batch_size': 100,
            'resume_enabled': True,
            'cache_features': True,
            'force_reextract': False,
            'min_file_age_seconds': 5
        },
        'audio_formats': [
            # Lossless formats
            '.flac', '.wav', '.aiff', '.aif', '.ape', '.wv', '.tta', '.tak',
            # Lossy formats
            '.mp3', '.ogg', '.opus', '.m4a', '.mp4', '.mpc', '.wma', '.asf', '.spx'
        ],
        'logging': {
            'level': 'INFO',
            'format': '[{time:YYYY-MM-DD HH:mm:ss}] {level: <8} | {message}',
            'file': '/data/playlister.log',
            'rotation': '10 MB',
            'retention': '1 week'
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses defaults and env vars.
        """
        self.config: Dict[str, Any] = self.DEFAULT_CONFIG.copy()

        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)

        # Override with environment variables
        self._load_from_env()

    def load_from_file(self, config_path: str):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        path = Path(config_path)

        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return

        try:
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f)

            if file_config:
                self._deep_update(self.config, file_config)
                logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")

    def _load_from_env(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'PLAYLISTER_MUSIC_PATH': ['paths', 'music'],
            'PLAYLISTER_DATA_PATH': ['paths', 'data'],
            'PLAYLISTER_PLAYLIST_PATH': ['paths', 'playlists'],
            'PLAYLISTER_CONFIG_PATH': ['paths', 'config'],
            'PLAYLISTER_LOG_LEVEL': ['logging', 'level'],
            'PLAYLISTER_WORKERS': ['extraction', 'workers'],
        }

        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self._set_nested(self.config, config_path, value)
                logger.debug(f"Config override from env: {env_var} = {value}")

    def _deep_update(self, base: Dict, update: Dict):
        """
        Deep update of nested dictionary.

        Args:
            base: Base dictionary to update
            update: Dictionary with updates
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _set_nested(self, config: Dict, path: list, value: Any):
        """
        Set a nested configuration value.

        Args:
            config: Configuration dictionary
            path: List of keys forming the path
            value: Value to set
        """
        for key in path[:-1]:
            config = config.setdefault(key, {})
        config[path[-1]] = value

    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value by keys.

        Args:
            *keys: Keys forming the path (e.g., 'paths', 'music')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys, value: Any):
        """
        Set configuration value by keys.

        Args:
            *keys: Keys forming the path (e.g., 'paths', 'music')
            value: Value to set
        """
        self._set_nested(self.config, list(keys), value)

    def save_to_file(self, config_path: str):
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save config file
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config file {config_path}: {e}")

    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return self.config[key]

    def __setitem__(self, key, value):
        """Allow dictionary-style setting."""
        self.config[key] = value

    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self.config.copy()