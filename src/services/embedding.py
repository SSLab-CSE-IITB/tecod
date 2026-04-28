"""Embedding service for TeCoD application."""

from collections.abc import Callable

import numpy as np

from ..exceptions.base import ServiceInitializationError
from ..utils.timing import log_with_time_elapsed
from .base import DeviceAwareService


class EmbeddingService(DeviceAwareService):
    """Service for handling text embeddings using SentenceTransformers."""

    def __init__(self, config, device: str | None = None, logger=None):
        super().__init__(config, device, logger)
        self._model = None
        self._embed_fn: Callable | None = None

    def initialize(self) -> None:
        """Initialize the embedding model and create embedding function."""
        try:
            with log_with_time_elapsed("Loading embedding model", self.logger):
                from sentence_transformers import SentenceTransformer

                self.logger.info(f"Loading embedding model: {self.config.emb.model}")
                self._model = SentenceTransformer(
                    self.config.emb.model, device=self.device, local_files_only=True
                )

                # Create embedding function with prompts
                self._embed_fn = self._create_embedding_function()

            self._mark_initialized()
            self.logger.info(f"Embedding service initialized on device: {self.device}")

        except Exception as e:
            self.logger.exception(f"Failed to initialize EmbeddingService: {str(e)}")
            raise ServiceInitializationError("EmbeddingService", str(e)) from e

    def cleanup(self) -> None:
        """Cleanup embedding model resources."""
        if self._model is not None:
            # Move model to CPU to free GPU memory
            if hasattr(self._model, "_modules"):
                self._model.to("cpu")
            self._model = None
        self._embed_fn = None
        self._initialized = False

    def _create_embedding_function(self) -> Callable:
        """Create the embedding function with prompt templates.

        Returns:
            Callable that takes text and prompt type and returns embeddings
        """
        prompts = {
            "query": "Instruct: Given a user query, retrieve the relevant question skeleton that best matches its semantic structure and intent.\nQuery:{query}",
            "skeleton": "{query}",
        }

        def get_detailed_instruct(prompt: str, query: str) -> str:
            """Format prompt with query text."""
            return prompts[prompt].format(query=query) if prompts[prompt] else query

        def embed_fn(text: str | list[str], prompt: str = "query") -> np.ndarray:
            """Embed text with specified prompt template.

            Args:
                text: Text or list of texts to embed
                prompt: Prompt type ('query' or 'skeleton')

            Returns:
                Numpy array of embeddings
            """
            if not self.is_initialized:
                raise ServiceInitializationError("EmbeddingService", "Service not initialized")

            inputs = None
            if isinstance(text, str):
                inputs = [get_detailed_instruct(prompt, text)]
            else:
                inputs = [get_detailed_instruct(prompt, t) for t in text]

            return self._model.encode(inputs, convert_to_numpy=True, show_progress_bar=False)

        return embed_fn

    def embed(self, text: str | list[str], prompt: str = "query") -> np.ndarray:
        """Embed text using the configured model.

        Args:
            text: Text or list of texts to embed
            prompt: Prompt type ('query' or 'skeleton')

        Returns:
            Numpy array of embeddings
        """
        self.ensure_initialized()
        return self._embed_fn(text, prompt)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model.

        Returns:
            Embedding dimension
        """
        self.ensure_initialized()
        # Test with a short text to get dimension
        test_embedding = self.embed("test")
        return test_embedding.shape[-1]

    @property
    def model(self):
        """Get the underlying SentenceTransformer model."""
        self.ensure_initialized()
        return self._model
