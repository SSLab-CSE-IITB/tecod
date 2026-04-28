"""OpenAI-compatible API model service."""

import time

from ..exceptions.base import GenerationError, ServiceInitializationError
from .base import Service


class OpenAICompatService(Service):
    """Service for OpenAI-compatible API endpoints.

    Works with any provider that implements the OpenAI chat completions spec:
    OpenAI, vLLM, Ollama, Together, Groq, LM Studio, etc.

    Configure via tecod.api_key and tecod.base_url. API keys are required
    for OpenAI-hosted endpoints and optional for local/custom compatible endpoints.
    """

    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self._client = None
        self._model_id = config.tecod.model_id

    def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI

            api_key = self.config.tecod.api_key
            base_url = self.config.tecod.base_url or None

            if not api_key and base_url is None:
                raise ServiceInitializationError(
                    "OpenAICompatService",
                    "No api_key configured for OpenAI-hosted endpoint. "
                    "Set OPENAI_API_KEY or tecod.api_key, or configure "
                    "tecod.base_url for a local OpenAI-compatible endpoint.",
                )

            client_api_key = api_key or "tecod-local-no-auth"
            self._client = OpenAI(api_key=client_api_key, base_url=base_url)
            self._mark_initialized()
            self.logger.info(f"OpenAICompatService initialized for model: {self._model_id}")

        except ServiceInitializationError:
            raise
        except Exception as e:
            self.logger.exception(f"Failed to initialize OpenAICompatService: {str(e)}")
            raise ServiceInitializationError("OpenAICompatService", str(e)) from e

    def cleanup(self) -> None:
        """Cleanup client resources."""
        self._client = None
        self._initialized = False

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Generate text using the OpenAI-compatible chat completions API.

        Args:
            prompt: The prompt to send as a single user message
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text string

        Raises:
            GenerationError: On API errors or empty responses
        """
        self.ensure_initialized()

        from openai import APIStatusError

        self.logger.debug(
            f"OpenAI request: model={self._model_id}, "
            f"max_new_tokens={max_new_tokens}, temperature={temperature}, "
            f"prompt={prompt!r}"
        )

        max_retries = self.config.tecod.retries
        base_delay = self.config.tecod.retry_base_delay
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                break  # success
            except APIStatusError as e:
                if e.status_code < 500:
                    # 4xx — not retryable (auth, bad request, etc.)
                    raise GenerationError(self._model_id, f"Client error: {e}") from e
                # 5xx — transient, retry with backoff
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    self.logger.warning(
                        f"Server error (attempt {attempt}/{max_retries}), retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    raise GenerationError(
                        self._model_id,
                        f"Server error after {max_retries} attempts: {e}",
                    ) from e
            except Exception as e:
                raise GenerationError(self._model_id, f"Unexpected error: {e}") from e

        if not response.choices:
            raise GenerationError(self._model_id, "API returned empty choices list")

        text = response.choices[0].message.content
        self.logger.debug(f"OpenAI extracted text: {text!r}")
        if text is None:
            finish_reason = response.choices[0].finish_reason
            raise GenerationError(
                self._model_id,
                f"Empty response (finish_reason={finish_reason})",
            )
        return text

    def generate_sql(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Generate SQL from a prompt string.

        High-level interface matching ModelService.generate_sql().

        Args:
            prompt: The prompt to send to the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Ignored (for interface compatibility)

        Returns:
            Generated SQL text string
        """
        return self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @classmethod
    def supports_method(cls, method: str) -> bool:
        """Check if this model service supports a given generation method.

        API models do not support methods requiring direct logit/KV-cache access.

        Args:
            method: Generation method name

        Returns:
            True if method is supported
        """
        unsupported = {"gcd", "base-gcd"}
        return method not in unsupported

    def get_model_info(self) -> dict:
        """Get information about the model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_id": self._model_id,
            "provider": "openai",
            "device": "api",
        }
