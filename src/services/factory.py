"""Factory for creating model service instances."""

import logging

from ..exceptions.base import ConfigurationError
from .base import ModelServiceProtocol

log = logging.getLogger("factory")

_SUPPORTED_PROVIDERS = {"local", "openai"}


def create_model_service(config, device=None, logger=None) -> ModelServiceProtocol:
    """Create the appropriate model service based on config.

    Args:
        config: Application configuration
        device: Device for local models (ignored for API models)
        logger: Optional logger instance

    Returns:
        A model service implementing ModelServiceProtocol

    Raises:
        ConfigurationError: If config.tecod.provider is not one of the
            supported providers. A typo silently selecting the local
            path would download a multi-GB HF model onto CPU; failing
            fast is the friendlier behaviour.
    """
    provider = config.tecod.provider
    log.info("Creating model service: provider=%r, model_id=%r", provider, config.tecod.model_id)

    if provider not in _SUPPORTED_PROVIDERS:
        raise ConfigurationError(
            f"Unknown model provider {provider!r}. "
            f"Supported providers: {sorted(_SUPPORTED_PROVIDERS)}"
        )

    if provider == "openai":
        from .openai_compat import OpenAICompatService

        log.info("Instantiating OpenAICompatService")
        return OpenAICompatService(config, logger)

    from .model import ModelService

    log.info("Instantiating ModelService (local)")
    return ModelService(config, device, logger)
