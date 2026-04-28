"""Services package for TeCoD application."""

from .base import DeviceAwareService, Service, ServiceContainer

__all__ = [
    "Service",
    "DeviceAwareService",
    "ServiceContainer",
    "EmbeddingService",
    "VectorStoreService",
    "ModelService",
    "OpenAICompatService",
    "create_model_service",
    "TemplateService",
    "TeCoDService",
]


def __getattr__(name: str):
    """Lazily import heavy service implementations.

    This keeps read-only CLI commands such as `version` and `status` from
    importing model libraries that may initialize local caches.
    """
    if name == "EmbeddingService":
        from .embedding import EmbeddingService

        return EmbeddingService
    if name == "VectorStoreService":
        from .vector_store import VectorStoreService

        return VectorStoreService
    if name == "ModelService":
        from .model import ModelService

        return ModelService
    if name == "OpenAICompatService":
        from .openai_compat import OpenAICompatService

        return OpenAICompatService
    if name == "create_model_service":
        from .factory import create_model_service

        return create_model_service
    if name == "TemplateService":
        from .template import TemplateService

        return TemplateService
    if name == "TeCoDService":
        from .tecod import TeCoDService

        return TeCoDService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
