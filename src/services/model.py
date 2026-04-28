"""Model service for managing LLM and other ML models."""

from typing import Any

import outlines
import torch

from ..exceptions.base import ModelLoadingError, ServiceInitializationError
from ..utils.timing import log_with_time_elapsed
from .base import DeviceAwareService


class ModelService(DeviceAwareService):
    """Service for loading and managing language models."""

    def __init__(self, config, device: str | None = None, logger=None):
        super().__init__(config, device, logger)
        self._model = None
        self._tokenizer = None
        self._outlines_tokenizer = None

    def initialize(self) -> None:
        """Initialize the language model and tokenizer."""
        try:
            model_id = self.config.tecod.model_id

            with log_with_time_elapsed("Loading language model", self.logger):
                self.logger.info(f"Loading language model: {model_id}")
                self._model, self._tokenizer = self._load_model(model_id, self.device)

                # Initialize outlines tokenizer for grammar-guided generation
                self._outlines_tokenizer = outlines.models.TransformerTokenizer(self._tokenizer)

            self._mark_initialized()
            self.logger.info(f"Model service initialized on device: {self.device}")

        except Exception as e:
            self.logger.exception(f"Failed to initialize ModelService: {str(e)}")
            raise ServiceInitializationError("ModelService", str(e)) from e

    def cleanup(self) -> None:
        """Cleanup model resources."""
        if self._model is not None:
            # Move model to CPU to free GPU memory
            try:
                self._model.to("cpu")
            except Exception:
                self.logger.debug("Cleanup error moving model to CPU", exc_info=True)
            self._model = None

        self._tokenizer = None
        self._outlines_tokenizer = None
        self._initialized = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_id: str, device: str) -> tuple[Any, Any]:
        """Load model and tokenizer.

        Args:
            model_id: HuggingFace model identifier
            device: Device to load model on

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelLoadingError: If model loading fails
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Set device_map for model loading
            if device == "cpu":
                device_map = None  # Let transformers handle CPU loading
            else:
                device_map = device

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=True
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=device_map,
                dtype="auto",
                local_files_only=True,
            )
            model.generation_config.pad_token_id = tokenizer.pad_token_id

            return model, tokenizer

        except Exception as e:
            self.logger.exception(f"Failed to load model {model_id}: {str(e)}")
            raise ModelLoadingError(model_id, str(e)) from e

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = 4096,
        num_beams: int = 1,
        do_sample: bool = False,
        regex_grammar: str | None = None,
        **generation_kwargs,
    ) -> dict:
        """Generate text using the loaded model.

        Args:
            inputs: Tokenized inputs with input_ids and attention_mask
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            regex_grammar: Optional regex pattern for constrained generation
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary containing generation results
        """
        self.ensure_initialized()

        # Move inputs to correct device
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        # Set up logits processor for constrained generation
        logits_processor = None
        if regex_grammar:
            grammar_processor = outlines.processors.RegexLogitsProcessor(
                regex_grammar, self._outlines_tokenizer, "torch"
            )
            logits_processor = [grammar_processor]

        # Generate
        output = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self._tokenizer.eos_token_id,
            num_beams=num_beams,
            logits_processor=logits_processor,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
            **generation_kwargs,
        )

        return output

    def tokenize(self, text: str, **kwargs) -> dict:
        """Tokenize text using the model's tokenizer.

        Args:
            text: Text to tokenize
            **kwargs: Additional tokenization parameters

        Returns:
            Tokenized output dictionary
        """
        self.ensure_initialized()
        return self._tokenizer(text, return_tensors="pt", **kwargs)

    def decode(self, token_ids, **kwargs) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional decoding parameters

        Returns:
            Decoded text string
        """
        self.ensure_initialized()
        return self._tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids_batch, **kwargs) -> list[str]:
        """Batch decode token IDs to text.

        Args:
            token_ids_batch: Batch of token IDs to decode
            **kwargs: Additional decoding parameters

        Returns:
            List of decoded text strings
        """
        self.ensure_initialized()
        return self._tokenizer.batch_decode(token_ids_batch, **kwargs)

    @property
    def model(self):
        """Get the underlying model."""
        self.ensure_initialized()
        return self._model

    @property
    def tokenizer(self):
        """Get the underlying tokenizer."""
        self.ensure_initialized()
        return self._tokenizer

    @property
    def outlines_tokenizer(self):
        """Get the outlines tokenizer for grammar-guided generation."""
        self.ensure_initialized()
        return self._outlines_tokenizer

    @classmethod
    def supports_method(cls, method: str) -> bool:
        """Check if this model service supports a given generation method.

        Local models support all generation methods.

        Args:
            method: Generation method name

        Returns:
            True if method is supported
        """
        return True

    def generate_sql(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        regex_grammar: str | None = None,
    ) -> str:
        """Generate SQL from a prompt string.

        High-level interface matching OpenAICompatService.generate_sql().

        Args:
            prompt: The prompt to send to the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            regex_grammar: Optional regex pattern for constrained generation

        Returns:
            Generated SQL text string
        """
        from ..utils.generation import get_data, get_gen_sequences

        self.ensure_initialized()

        inputs = get_data(prompts=[prompt], tokenizer=self._tokenizer)
        output = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            regex_grammar=regex_grammar,
        )
        gen_sequences = get_gen_sequences(
            sequences=output.sequences,
            tokenizer=self._tokenizer,
            inputs=inputs,
        )
        generations = self.batch_decode(
            gen_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return generations[0]

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        self.ensure_initialized()

        return {
            "model_id": self.config.tecod.model_id,
            "device": self.device,
            "vocab_size": len(self._tokenizer),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
