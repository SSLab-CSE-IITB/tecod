"""Base service classes and interfaces."""

import logging
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from ..config.models import AppConfig
from ..exceptions.base import ServiceInitializationError


def resolve_device(*devices: str | None) -> str:
    """Resolve device from priority list. First non-None/empty/auto wins, then auto-detect.

    Args:
        *devices: Device strings in priority order (component override, default, etc.)

    Returns:
        Resolved device string ('cuda' or 'cpu')
    """
    import torch

    for d in devices:
        if d and d not in ("auto", "null"):
            if d == "cuda" and not torch.cuda.is_available():
                return "cpu"
            return d
    return "cuda" if torch.cuda.is_available() else "cpu"


class Service(ABC):
    """Abstract base class for all services."""

    def __init__(self, config: AppConfig, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the service.

        Raises:
            ServiceInitializationError: If initialization fails
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup service resources."""
        pass

    def ensure_initialized(self) -> None:
        """Ensure the service is initialized.

        Raises:
            ServiceInitializationError: If service is not initialized
        """
        if not self._initialized:
            raise ServiceInitializationError(
                self.__class__.__name__, "Service not initialized. Call initialize() first."
            )

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    def _mark_initialized(self) -> None:
        """Mark service as initialized."""
        self._initialized = True


class DeviceAwareService(Service):
    """Base class for services that need device management."""

    def __init__(
        self,
        config: AppConfig,
        device: str | None = None,
        logger: logging.Logger | None = None,
    ):
        super().__init__(config, logger)
        self.device = self._determine_device(device)

    def _determine_device(self, requested_device: str | None) -> str:
        """Determine the best device to use.

        Args:
            requested_device: Requested device ('cuda', 'cpu', 'auto', None)

        Returns:
            Device string to use
        """
        device = resolve_device(requested_device)
        if requested_device == "cuda" and device == "cpu":
            self.logger.warning("CUDA requested but not available, falling back to CPU")
        return device


@runtime_checkable
class ModelServiceProtocol(Protocol):
    """Structural interface for model services (ModelService, OpenAICompatService).

    Both local and API-based model services implement generate_sql(),
    supports_method(), and get_model_info(). This protocol formalises
    that contract so that consumers (TeCoDService, factory) can depend
    on the interface rather than a union type.
    """

    def generate_sql(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> str: ...

    @classmethod
    def supports_method(cls, method: str) -> bool: ...

    def get_model_info(self) -> dict: ...


class ServiceContainer:
    """Container for managing service dependencies."""

    def __init__(self):
        self._services: dict[str, Service] = {}
        self._config: AppConfig | None = None

    def set_config(self, config: AppConfig) -> None:
        """Set the configuration for all services."""
        self._config = config

    def register(self, name: str, service: Service) -> None:
        """Register a service.

        Args:
            name: Service name/identifier
            service: Service instance
        """
        self._services[name] = service

    def get(self, name: str) -> Service:
        """Get a service by name.

        Args:
            name: Service name/identifier

        Returns:
            Service instance

        Raises:
            ServiceInitializationError: If service not found
        """
        if name not in self._services:
            raise ServiceInitializationError(name, f"Service '{name}' not registered")
        return self._services[name]

    def initialize_all(self) -> None:
        """Initialize all registered services.

        On failure, cleans up all already-initialized services in reverse
        order before re-raising, so no service is left holding resources.
        """
        initialized: list[Service] = []
        for name, service in self._services.items():
            try:
                if not service.is_initialized:
                    service.initialize()
                    initialized.append(service)
            except Exception as e:
                logger = logging.getLogger("ServiceContainer")
                logger.exception(f"Failed to initialize service {name}: {str(e)}")
                # Cleanup already-initialized services in reverse order
                for svc in reversed(initialized):
                    try:
                        svc.cleanup()
                    except Exception as cleanup_err:
                        logger.exception(f"Error during rollback cleanup: {cleanup_err}")
                raise ServiceInitializationError(name, f"Failed to initialize: {str(e)}") from e

    def cleanup_all(self) -> None:
        """Cleanup all registered services."""
        for service in self._services.values():
            try:
                service.cleanup()
            except Exception as e:
                # Log cleanup errors but don't raise
                logging.getLogger("ServiceContainer").exception(f"Error during cleanup: {e}")

    @property
    def config(self) -> AppConfig:
        """Get the current configuration."""
        if self._config is None:
            raise ServiceInitializationError("ServiceContainer", "Configuration not set")
        return self._config
