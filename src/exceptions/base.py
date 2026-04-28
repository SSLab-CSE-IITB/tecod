"""Base exception classes for TeCoD application."""

from typing import Any


class TeCoDBaseException(Exception):
    """Base exception for TeCoD application."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(TeCoDBaseException):
    """Raised when there's an issue with configuration."""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})


class ServiceInitializationError(TeCoDBaseException):
    """Raised when a service fails to initialize."""

    def __init__(self, service_name: str, reason: str):
        message = f"Failed to initialize {service_name}: {reason}"
        super().__init__(message, "SERVICE_INIT_ERROR", {"service_name": service_name})


class ModelLoadingError(TeCoDBaseException):
    """Raised when model loading fails."""

    def __init__(self, model_id: str, reason: str):
        message = f"Failed to load model {model_id}: {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR", {"model_id": model_id})


class GenerationError(TeCoDBaseException):
    """Raised when text generation fails."""

    def __init__(self, model_id: str, reason: str):
        message = f"Generation failed for model {model_id}: {reason}"
        super().__init__(message, "GENERATION_ERROR", {"model_id": model_id})


class VectorStoreError(TeCoDBaseException):
    """Raised when vector store operations fail."""

    def __init__(self, operation: str, reason: str):
        message = f"Vector store operation '{operation}' failed: {reason}"
        super().__init__(message, "VECTOR_STORE_ERROR", {"operation": operation})


class TemplateError(TeCoDBaseException):
    """Raised when template operations fail."""

    def __init__(self, template_id: int | None, reason: str):
        message = f"Template operation failed (ID: {template_id}): {reason}"
        super().__init__(message, "TEMPLATE_ERROR", {"template_id": template_id})
