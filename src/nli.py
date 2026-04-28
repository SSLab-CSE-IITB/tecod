"""Natural Language Inference (NLI) module for TeCoD.

This module provides NLI classification to determine semantic relationships
between natural language queries and example templates.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class NLI:
    """Natural Language Inference classifier for query-template matching.

    This class wraps a sequence classification model (typically DeBERTa-based)
    to perform NLI classification, determining whether a query semantically
    matches (entails) a template pattern.

    Attributes:
        model: The underlying transformer model for sequence classification.
        tokenizer: The tokenizer for the model.
        device: The device (CPU/CUDA) where the model runs.

    Example:
        >>> nli = NLI(model_id="tasksource/deberta-base-long-nli", device="cuda")
        >>> results = nli(["Find all [MASK]", "Count [MASK]"], "Find all users")
        >>> print(results[0]["entailment"])  # Probability of entailment
        0.85
    """

    def __init__(
        self,
        model_id: str | None = "tasksource/deberta-base-long-nli",
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        device: str | None = None,
        bf16: bool = False,
    ) -> None:
        """Initialize the NLI classifier.

        Args:
            model_id: HuggingFace model identifier. Used if model/tokenizer not provided.
            model: Pre-loaded transformer model. If provided, tokenizer must also be given.
            tokenizer: Pre-loaded tokenizer. If provided, model must also be given.
            device: Device to run inference on ('cuda', 'cpu', or specific device like 'cuda:0').
            bf16: Whether to load the model in bfloat16 precision for reduced memory usage.

        Raises:
            ValueError: If neither model_id nor (model and tokenizer) are provided.
        """
        if not model_id and not (model and tokenizer):
            raise ValueError("Either model_id or both model and tokenizer must be provided")

        # Resolve 'auto'/None to a concrete device before loading or moving
        # tensors, so the model and tokenized inputs stay aligned.
        if device is None or device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        else:
            resolved_device = device
        self.device: str = resolved_device

        if model is not None and tokenizer is not None:
            self.model: PreTrainedModel = model
            self.tokenizer: PreTrainedTokenizer = tokenizer
        else:
            if bf16:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model.to(self.device)

        # Disable dropout and other training-time layers; we only run inference.
        self.model.eval()

    def __call__(
        self,
        matching_nlqs: list[str],
        nlq: str,
        batch_size: int | None = None,
    ) -> list[dict[str, float]]:
        """Perform NLI classification between a query and multiple templates.

        Args:
            matching_nlqs: List of template/example queries to compare against.
            nlq: The target natural language query.
            batch_size: Number of pairs to process at once. Defaults to all pairs.

        Returns:
            List of dictionaries mapping NLI labels to probabilities.
            Each dict typically contains: 'entailment', 'neutral', 'contradiction'.
        """
        if not matching_nlqs:
            return []

        input_pairs: list[tuple[str, str]] = [(matching_nlq, nlq) for matching_nlq in matching_nlqs]

        if batch_size is None:
            batch_size = len(input_pairs)

        all_prob_dicts: list[dict[str, float]] = []

        for i in range(0, len(input_pairs), batch_size):
            batch_input = input_pairs[i : i + batch_size]

            # Truncate to the model's context length. Long (template, query)
            # pairs would otherwise overflow position embeddings and emit a
            # silent HF warning. Log once when truncation kicks in so the
            # caller has a chance to notice the drift.
            max_len = self.model.config.max_position_embeddings
            inputs = self.tokenizer(
                batch_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)

            prob = nn.Softmax(dim=-1)(outputs.logits).cpu().to(torch.float32).detach().numpy()
            # Softmax on very large negative logits can produce NaN in mixed
            # precision. Replace with zeros so downstream consumers don't
            # propagate garbage probabilities.
            import numpy as np

            prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)

            batch_prob_dict: list[dict[str, float]] = [
                {
                    label: float(prob[j][l_idx])
                    for l_idx, label in self.model.config.id2label.items()
                }
                for j in range(prob.shape[0])
            ]

            all_prob_dicts.extend(batch_prob_dict)

        return all_prob_dicts
