"""Configuration models using Pydantic for type safety."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model and vector store."""

    model: str = Field(description="Embedding model identifier")
    device: str | None = Field(
        default=None, description="Device override (None = inherit top-level)"
    )
    collection_name: str = Field(description="Vector store collection name")
    emb_field_name: str = Field(description="Embedding field name in vector store")
    index_name: str = Field(description="Index name for vector store")
    masked_nlq_key: str = Field(description="Masked natural language question key")


class NLIConfig(BaseModel):
    """Configuration for Natural Language Inference model."""

    model: str = Field(description="NLI model identifier")
    method: Literal["mean", "max", "min"] = Field(description="Aggregation method for NLI scores")
    device: str | None = Field(
        default=None, description="Device override (None = inherit top-level)"
    )


class TeCoDConfig(BaseModel):
    """Configuration for TeCoD-specific settings."""

    model_id: str = Field(description="Language model identifier")
    provider: Literal["local", "openai"] = Field(default="local", description="Model provider")
    temperature: float = Field(default=0.0, description="Generation temperature for API models")
    max_new_tokens: int = Field(default=4096, description="Maximum new tokens to generate")
    api_key: str = Field(default="", description="API key for OpenAI-compatible endpoint")
    base_url: str = Field(default="", description="Base URL for OpenAI-compatible endpoint")
    retries: int = Field(default=3, description="Max retries on 5xx from API models")
    retry_base_delay: float = Field(default=1.0, description="Base seconds for exponential backoff")
    grammar_type: str = Field(description="Grammar type for parsing")
    prompt_class: str | None = Field(
        default=None,
        description="Prompt class override (llama, qwen, arctic, granite, codes, default). If unset, inferred from model_id.",
    )
    grammar_template_json_path: str = Field(
        default="pdec/complete_sql_template.json", description="Path to grammar template JSON"
    )
    icl_cnt: int = Field(description="Number of in-context learning examples")
    nli_top_k: int = Field(description="Top K results for NLI filtering")
    vectorsearch_top_k: int = Field(description="Top K results for vector search")
    sql_key: str = Field(description="Key for data SQL queries")
    dialect: str = Field(description="Dialect of SQL")

    @property
    def is_api_model(self) -> bool:
        """Check if the model is an API-based model."""
        return self.provider != "local"


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    console_level: str = Field(default="WARNING")
    file_level: str = Field(default="DEBUG")
    log_file: str = Field(default="tecod.log")
    use_json_format: bool = Field(default=False)


class StateConfig(BaseModel):
    """Configuration for state files and paths."""

    examples: str = Field(description="Examples file name")
    templates: str = Field(description="Templates file name")
    schema_prompt: str = Field(description="Schema prompt file name")
    compiled_templates: str = Field(description="Compiled templates directory name")
    index: str = Field(description="Vector index file name")
    masked_questions: str = Field(description="Masked questions file name")


class AppConfig(BaseModel):
    """Main application configuration."""

    root_dir: str = Field(description="Root directory path")
    data_dir: str = Field(description="Data directory path")
    db_path: str = Field(description="Database file path")
    device: str = Field(default="auto", description="Default device for all components")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    emb: EmbeddingConfig
    nli: NLIConfig
    tecod: TeCoDConfig
    state: StateConfig

    @field_validator("root_dir", "data_dir", "db_path")
    @classmethod
    def validate_paths_exist(cls, v: str) -> str:
        """Validate that paths are not empty."""
        if not v:
            raise ValueError("Path cannot be empty")
        return v

    @property
    def data_path(self) -> Path:
        """Get data directory as Path object."""
        return Path(self.data_dir)

    @property
    def db_file_path(self) -> Path:
        """Get database file as Path object."""
        return Path(self.db_path)

    @property
    def examples_path(self) -> Path:
        """Get examples file path."""
        return self.data_path / self.state.examples

    @property
    def templates_path(self) -> Path:
        """Get templates file path."""
        return self.data_path / self.state.templates

    @property
    def schema_prompt_path(self) -> Path:
        """Get schema prompt file path."""
        return self.data_path / self.state.schema_prompt

    @property
    def compiled_templates_path(self) -> Path:
        """Get compiled templates directory path."""
        return self.data_path / self.state.compiled_templates

    @property
    def index_path(self) -> Path:
        """Get index file path."""
        return self.data_path / self.state.index
