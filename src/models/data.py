"""Data models for TeCoD requests and responses."""

from typing import Literal

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    """Request model for text generation."""

    query: str = Field(description="Natural language query")
    top_k: int = Field(default=1000, description="Top K results for retrieval")
    max_new_tokens: int = Field(default=4096, description="Maximum new tokens to generate")
    num_beams: int = Field(default=1, description="Number of beams for beam search")
    do_sample: bool = Field(default=False, description="Whether to use sampling")
    regex_grammar: str | None = Field(
        default=None, description="Regex grammar for constrained generation"
    )
    method: Literal["gcd", "base-gcd", "zs", "icl", "sgc", "auto"] | None = Field(
        default="auto", description="Generation method"
    )
    schema_sequence: str | None = Field(default=None, description="Schema serialization format")
    content_sequence: str | None = Field(
        default=None, description="Content serialization format"
    )
    zs_prompt: str | None = Field(default=None, description="Zero-shot prompt")
    use_oracle: bool = Field(default=False, description="Whether to use oracle template")
    gold_sql: str | None = Field(default=None, description="Gold SQL")


class TemplateSelectionResult(BaseModel):
    """Result of template selection process."""

    template_id: int = Field(description="Selected template ID")
    entailment_score: float = Field(description="NLI entailment score")
    cosine_score: float = Field(description="Vector similarity score")
    nli_label: str = Field(description="NLI classification label")
    icl_examples: list[tuple[str, str]] = Field(description="In-context learning examples")
    icl_example_indices: list[int] = Field(description="Indices of ICL examples")

    # Debugging info
    retrieved_examples: DataFrame | None = Field(
        default=None, description="Retrieved examples data"
    )
    nli_results: DataFrame | None = Field(default=None, description="NLI results for candidates")
    templates_considered: DataFrame | None = Field(
        default=None, description="All templates considered"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GenerationOutput(BaseModel):
    """Complete generation output from TeCoD."""

    query: str = Field(description="Input natural language query")
    pred_sql: str | None = Field(description="Generated SQL query")
    method: Literal["gcd", "icl", "sgc", "base-gcd", "zs", "auto"] | None = Field(
        description="Generation method used"
    )

    # Template selection results
    template_id: int = Field(description="Selected template ID")
    nli_score: float = Field(description="NLI entailment score")
    cosine_score: float = Field(description="Vector similarity score")
    nli_label: str = Field(description="NLI classification label")

    # ICL information
    icl_examples: list[tuple[str, str]] = Field(description="In-context learning examples used")
    icl_example_indices: list[int] = Field(description="Indices of ICL examples")

    # Generation metrics
    log_prob: float | None = Field(default=None, description="Log probability of generation")
    log_logits_prob: float | None = Field(default=None, description="Log logits probability")
    generation_time: float | None = Field(default=None, description="Time taken for generation")

    # Detailed timing data
    timing_data: dict[str, float] | None = Field(
        default=None, description="Detailed timing for all operations"
    )
    total_time: float | None = Field(
        default=None, description="Total time for complete generation"
    )

    # Intermediate state tracking
    vector_search_time: float | None = Field(default=None, description="Time for vector search")
    nli_processing_time: float | None = Field(
        default=None, description="Time for NLI processing"
    )
    template_selection_time: float | None = Field(
        default=None, description="Time for template selection"
    )
    embedding_time: float | None = Field(default=None, description="Time for query embedding")

    # Retrieved data sizes for tracing
    retrieved_examples_count: int | None = Field(
        default=None, description="Number of examples retrieved"
    )
    nli_examples_count: int | None = Field(
        default=None, description="Number of examples processed by NLI"
    )

    # Metadata
    prompt: str | None = Field(default=None, description="Full prompt used for generation")

    # Debugging info
    template_selection_result: TemplateSelectionResult | None = Field(
        default=None, description="Template selection result for debugging"
    )
    post_processing_failed: bool = Field(
        default=False,
        description="Whether post-processing raised an exception",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchResult(BaseModel):
    """Result from vector search."""

    indices: list[int] = Field(description="Retrieved example indices")
    distances: list[float] = Field(description="Cosine distances/similarities")
    examples: list[dict] | None = Field(default=None, description="Retrieved example data")


class NLIResult(BaseModel):
    """Result from NLI classification."""

    entailment: float = Field(description="Entailment probability")
    contradiction: float = Field(description="Contradiction probability")
    neutral: float = Field(description="Neutral probability")
    predicted_label: str = Field(description="Predicted NLI label")


class ServiceStatus(BaseModel):
    """Status of a service."""

    name: str = Field(description="Service name")
    initialized: bool = Field(description="Whether service is initialized")
    device: str | None = Field(default=None, description="Device used by service")
    error: str | None = Field(default=None, description="Error message if any")


class SystemStatus(BaseModel):
    """Overall system status."""

    services: list[ServiceStatus] = Field(description="Status of all services")
    config_loaded: bool = Field(description="Whether configuration is loaded")
    ready: bool = Field(description="Whether system is ready for requests")
