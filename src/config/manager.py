"""Configuration manager for TeCoD application."""

import os
import threading
from pathlib import Path

from hydra import compose
from omegaconf import DictConfig, OmegaConf

from ..exceptions.base import ConfigurationError
from .models import AppConfig


class ConfigManager:
    """Manages application configuration using Hydra and Pydantic."""

    def __init__(self, config_path: str = "conf"):
        self.config_path = config_path
        self._config: AppConfig | None = None
        # Serialises load_config() against concurrent callers so the
        # Pydantic config object can't be partially overwritten.
        self._load_lock = threading.Lock()

    def load_config(
        self, config_name: str = "config", overrides: list[str] | None = None
    ) -> AppConfig:
        """Load configuration from Hydra with optional overrides.

        Args:
            config_name: Name of the config file (without .yaml extension)
            overrides: List of Hydra-style override strings

        Returns:
            Validated AppConfig instance

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        with self._load_lock:
            try:
                # Use initialize_config_dir with absolute path instead of initialize
                from hydra import initialize_config_dir

                # Get absolute path to config directory
                if os.path.isabs(self.config_path):
                    config_dir = self.config_path
                else:
                    # Get project root directory (where main.py is located)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    config_dir = os.path.join(project_root, self.config_path)

                # Support TECOD_ENV env var as override (lower priority than explicit env= overrides)
                env_from_var = os.environ.get("TECOD_ENV")
                if env_from_var and not any(
                    o.startswith("env@_global_=") for o in (overrides or [])
                ):
                    overrides = list(overrides or [])
                    overrides.insert(0, f"env@_global_={env_from_var}")

                with initialize_config_dir(
                    config_dir=os.path.abspath(config_dir), version_base=None
                ):
                    hydra_cfg: DictConfig = compose(
                        config_name=config_name, overrides=overrides or []
                    )

                # Convert OmegaConf to dict and then to Pydantic model
                config_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
                self._config = AppConfig(**config_dict)

                # Set environment variables (canonical set point; CLI does
                # not need to re-set this after load_config returns).
                os.environ["ROOT_DIR"] = self._config.root_dir

                return self._config

            except Exception as e:
                import logging

                logging.getLogger("ConfigManager").exception(
                    f"Failed to load configuration: {str(e)}"
                )
                raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e

    @property
    def config(self) -> AppConfig:
        """Get the current configuration.

        Raises:
            ConfigurationError: If configuration hasn't been loaded yet
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded. Call load_config() first.")
        return self._config

    def validate_environment(self) -> None:
        """Validate that required files and directories exist.

        Raises:
            ConfigurationError: If validation fails
        """
        config = self.config

        # Check if database file exists
        if not config.db_file_path.exists():
            raise ConfigurationError(f"Database file not found: {config.db_file_path}")

        # Ensure data directory exists
        if not config.data_path.exists():
            config.data_path.mkdir(parents=True, exist_ok=True)

    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path based on root directory.

        Args:
            relative_path: Path relative to root directory

        Returns:
            Absolute Path object
        """
        return Path(self.config.root_dir) / relative_path


# Singleton instance for global access
_config_manager: ConfigManager | None = None
_config_manager_lock = threading.Lock()


def get_config_manager() -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        with _config_manager_lock:
            if _config_manager is None:
                _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return get_config_manager().config
