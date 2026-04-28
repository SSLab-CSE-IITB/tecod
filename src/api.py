"""
TeCoD Python API

A clean programmatic interface for using Template-guided Constraint-based Decoding (TeCoD)
for text-to-SQL generation without requiring CLI interaction.
"""

import logging
from typing import Any

from .config.manager import ConfigManager, get_config_manager
from .exceptions.base import TeCoDBaseException
from .models.data import GenerationOutput, GenerationRequest
from .services.base import resolve_device
from .services.embedding import EmbeddingService
from .services.factory import create_model_service
from .services.tecod import TeCoDService
from .services.template import TemplateService
from .services.vector_store import VectorStoreService
from .utils.logging import setup_logging


class TeCoD:
    """
    Main TeCoD API class for programmatic usage.

    This class provides a simple interface to use TeCoD for text-to-SQL generation
    without requiring CLI interaction. It handles all service initialization and
    provides clean methods for SQL generation.

    Example:
        >>> tecod = TeCoD(
        ...     config_overrides=["+data=financial"],
        ... )
        >>> result = tecod.generate("How many accounts from north Bohemia are eligible to receive loans?")
        >>> print(result.pred_sql)
        SELECT COUNT(T1.account_id) ...
        >>> print(f"Generation took {result.total_time*1000:.1f}ms")
        Generation took 1234.5ms
    """

    def __init__(
        self,
        data_dir: str = "data",
        device: str | None = None,
        config_overrides: list[str] | None = None,
        log_level: str = "INFO",
        console_log_level: str = "ERROR",
        file_log_level: str = "DEBUG",
        log_file: str | None = "tecod_api.log",
    ):
        """
        Initialize TeCoD API.

        Args:
            data_dir: Directory containing TeCoD data (examples, templates, etc.)
            device: Device to use ("cuda", "cpu", or None for auto-detection)
            config_overrides: List of Hydra-style config overrides
            log_level: Base logging level
            console_log_level: Console logging level
            file_log_level: File logging level
            log_file: Log file path (None to disable file logging)

        Raises:
            TeCoDBaseException: If initialization fails
        """
        self.config_manager: ConfigManager | None = None
        self.tecod_service: TeCoDService | None = None
        self.logger: logging.Logger | None = None
        self._initialized = False

        # Store initialization parameters
        self._data_dir = data_dir
        self._device = device
        self._config_overrides = config_overrides or [f"data_dir={data_dir}"]
        self._log_level = log_level
        self._console_log_level = console_log_level
        self._file_log_level = file_log_level
        self._log_file = log_file

        # Initialize immediately
        self._initialize()

    def _initialize(self) -> None:
        """Initialize all TeCoD services."""
        try:
            # Setup logging - convert string levels to integer levels
            console_level = (
                getattr(logging, self._console_log_level.upper())
                if isinstance(self._console_log_level, str)
                else self._console_log_level
            )
            file_level = (
                getattr(logging, self._file_log_level.upper())
                if isinstance(self._file_log_level, str)
                else self._file_log_level
            )

            self.logger = setup_logging(
                console_level=console_level, file_level=file_level, log_file=self._log_file
            )

            self.logger.info("Initializing TeCoD API...")

            # Load configuration
            self.config_manager = get_config_manager()
            self.config_manager.load_config(overrides=self._config_overrides)
            config = self.config_manager.config

            # Resolve per-component devices
            default_device = self._device or config.device
            emb_device = resolve_device(config.emb.device, default_device)
            nli_device = resolve_device(config.nli.device, default_device)

            # Initialize services
            self.logger.info("Creating TeCoD services...")
            embedding_service = EmbeddingService(config, emb_device, self.logger)
            vector_store_service = VectorStoreService(config, embedding_service, self.logger)
            model_service = create_model_service(config, default_device, self.logger)
            template_service = TemplateService(config, self.logger)

            # Create main TeCoD service
            self.tecod_service = TeCoDService(
                config=config,
                embedding_service=embedding_service,
                vector_store_service=vector_store_service,
                model_service=model_service,
                template_service=template_service,
                device=nli_device,
                logger=self.logger,
            )

            # Initialize all services
            self.logger.info("Initializing TeCoD services...")
            self.tecod_service.initialize()

            self._initialized = True
            self.logger.info("TeCoD API initialized successfully!")

        except Exception as e:
            error_msg = f"Failed to initialize TeCoD API: {str(e)}"
            if self.logger:
                self.logger.exception(error_msg)
            raise TeCoDBaseException(error_msg) from e

    def generate(
        self,
        query: str,
        max_new_tokens: int = 4096,
        num_beams: int = 1,
        do_sample: bool = False,
        top_k: int = 1000,
        regex_grammar: str | None = None,
    ) -> GenerationOutput:
        """
        Generate SQL from natural language query.

        Args:
            query: Natural language query describing what SQL to generate
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search (1 = greedy)
            do_sample: Whether to use sampling instead of greedy/beam search
            top_k: Number of top similar examples to retrieve for context
            regex_grammar: Optional regex grammar for constrained generation

        Returns:
            GenerationOutput containing the generated SQL and detailed timing info

        Raises:
            TeCoDBaseException: If generation fails
            ValueError: If query is empty or invalid parameters

        Example:
            >>> result = tecod.generate("How many accounts from north Bohemia are eligible to receive loans?")
            >>> print(result.pred_sql)
            SELECT COUNT(T1.account_id) ...
            >>> print(f"Method: {result.method}")
            Method: gcd
            >>> print(f"Total time: {result.total_time*1000:.1f}ms")
            Total time: 1234.5ms
        """
        if not self._initialized:
            raise TeCoDBaseException("TeCoD API not initialized")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            request = GenerationRequest(
                query=query.strip(),
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                regex_grammar=regex_grammar,
            )

            self.logger.info(
                f"Generating SQL for query: '{query[:50]}{'...' if len(query) > 50 else ''}'"
            )
            result = self.tecod_service.generate(request)
            self.logger.info(
                f"Generation completed in {result.total_time * 1000:.1f}ms using {result.method.upper()}"
            )

            return result

        except Exception as e:
            error_msg = f"SQL generation failed: {str(e)}"
            self.logger.exception(error_msg)
            raise TeCoDBaseException(error_msg) from e

    def generate_with_method(self, request: GenerationRequest) -> GenerationOutput:
        """
        Generate SQL with an explicit generation method.

        Args:
            request: Generation request with `method` set to one of gcd,
                base-gcd, sgc, icl, zs, or auto.

        Returns:
            GenerationOutput containing the generated SQL and detailed timing info.
        """
        if not self._initialized or not self.tecod_service:
            raise TeCoDBaseException("TeCoD API not initialized")

        try:
            query_preview = f"{request.query[:50]}{'...' if len(request.query) > 50 else ''}"
            self.logger.info(
                f"Generating SQL with method={request.method} for query: '{query_preview}'"
            )
            result = self.tecod_service.generate_with_method(request)
            self.logger.info(
                f"Generation completed in {result.total_time * 1000:.1f}ms "
                f"using {result.method.upper()}"
            )
            return result

        except Exception as e:
            error_msg = f"SQL generation failed: {str(e)}"
            self.logger.exception(error_msg)
            raise TeCoDBaseException(error_msg) from e

    def get_status(self) -> dict[str, Any]:
        """
        Get current status of TeCoD API and all services.

        Returns:
            Dictionary containing status information

        Example:
            >>> status = tecod.get_status()
            >>> print(status['ready'])
            True
            >>> print(status['device'])
            cuda
        """
        if not self._initialized or not self.tecod_service:
            return {"ready": False, "initialized": False, "error": "API not initialized"}

        try:
            services_status = []

            # Check all services
            for service_name, service in [
                ("embedding", self.tecod_service.embedding_service),
                ("vector_store", self.tecod_service.vector_store_service),
                ("model", self.tecod_service.model_service),
                ("template", self.tecod_service.template_service),
                ("tecod", self.tecod_service),
            ]:
                services_status.append(
                    {
                        "name": service_name,
                        "initialized": service.is_initialized,
                        "device": getattr(service, "device", None),
                    }
                )

            all_ready = all(s["initialized"] for s in services_status)

            return {
                "ready": all_ready,
                "initialized": self._initialized,
                "device": self._device,
                "data_dir": self._data_dir,
                "services": services_status,
                "config_loaded": self.config_manager is not None,
            }

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error getting API status: {str(e)}")
            return {"ready": False, "initialized": self._initialized, "error": str(e)}

    def cleanup(self) -> None:
        """
        Clean up all resources and services.

        Call this when you're done using the TeCoD API to free up GPU memory
        and other resources.

        Example:
            >>> tecod = TeCoD()
            >>> result = tecod.generate("Show users")
            >>> tecod.cleanup()  # Free up resources
        """
        if self.tecod_service:
            try:
                self.logger.info("Cleaning up TeCoD services...")
                self.tecod_service.cleanup()
                self.logger.info("Cleanup completed")
            except Exception as e:
                if self.logger:
                    self.logger.exception(f"Error during cleanup: {str(e)}")

        self._initialized = False
        self.tecod_service = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()

    @property
    def is_ready(self) -> bool:
        """Check if TeCoD API is ready for use."""
        return (
            self._initialized
            and self.tecod_service is not None
            and self.tecod_service.is_initialized
        )


# Convenience function for quick usage
def create_tecod(data_dir: str = "data", device: str | None = None, **kwargs) -> TeCoD:
    """
    Convenience function to create and initialize a TeCoD instance.

    Args:
        data_dir: Directory containing TeCoD data
        device: Device to use ("cuda", "cpu", or None for auto)
        **kwargs: Additional arguments passed to TeCoD constructor

    Returns:
        Initialized TeCoD instance

    Example:
        >>> tecod = create_tecod(data_dir="my_data", device="cuda")
        >>> result = tecod.generate("Find all products")
    """
    return TeCoD(data_dir=data_dir, device=device, **kwargs)
