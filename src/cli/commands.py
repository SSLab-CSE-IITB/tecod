"""CLI commands for TeCoD application."""

import contextlib
import datetime
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated

import pandas as pd
import typer
from tqdm.auto import tqdm

from ..config.manager import get_config_manager
from ..exceptions.base import ConfigurationError, TeCoDBaseException
from ..models.data import GenerationRequest
from ..services.base import ServiceContainer, resolve_device
from ..services.tecod import pick_icl_example_indices

if TYPE_CHECKING:
    from ..services.tecod import TeCoDService


@contextlib.contextmanager
def cli_error_handler(operation: str) -> Iterator[None]:
    """Context manager for consistent CLI error handling.

    Catches TeCoDBaseException and unexpected Exception, logs them,
    echoes to stderr, and raises typer.Exit(1).
    """
    logger = logging.getLogger("app")
    try:
        yield
    except TeCoDBaseException as e:
        logger.error(f"Error during {operation}: {e.message}")
        logger.debug(f"{operation} error stacktrace:", exc_info=True)
        typer.echo(f"Error during {operation}: {e.message}", err=True)
        raise typer.Exit(1) from None
    except Exception as e:
        logger.exception(f"Unexpected error during {operation}: {e}")
        typer.echo(f"Unexpected error during {operation}: {e}", err=True)
        raise typer.Exit(1) from None


class CLIContext:
    """Context for CLI operations."""

    def __init__(self):
        self.config_manager = get_config_manager()
        self.container = ServiceContainer()
        self.tecod_service: "TeCoDService | None" = None

    def initialize_services(self, device: str | None = None) -> None:
        """Initialize all services."""
        config = self.config_manager.config
        self.container.set_config(config)

        # Resolve per-component devices
        default_device = device or config.device
        emb_device = resolve_device(config.emb.device, default_device)
        nli_device = resolve_device(config.nli.device, default_device)

        from ..services.embedding import EmbeddingService
        from ..services.factory import create_model_service
        from ..services.tecod import TeCoDService
        from ..services.template import TemplateService
        from ..services.vector_store import VectorStoreService

        # Create services
        embedding_service = EmbeddingService(config, emb_device)
        vector_store_service = VectorStoreService(config, embedding_service)
        model_service = create_model_service(config, default_device)
        template_service = TemplateService(config)

        # Register services (embedding must be before vector_store —
        # VectorStoreService requires EmbeddingService to be initialized first)
        self.container.register("embedding", embedding_service)
        self.container.register("vector_store", vector_store_service)
        self.container.register("model", model_service)
        self.container.register("template", template_service)

        # Create main TeCoD service
        self.tecod_service = TeCoDService(
            config=config,
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
            model_service=model_service,
            template_service=template_service,
            device=nli_device,
        )

        self.container.register("tecod", self.tecod_service)
        self.container.initialize_all()


# Global CLI context
cli_context = CLIContext()


def load_config_callback(
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--config-override",
            "-c",
            help="Hydra-style overrides (e.g., 'server.port=9000'). Applied globally.",
        ),
    ] = None,
) -> None:
    """Load configuration with optional overrides."""
    with cli_error_handler("configuration"):
        cli_context.config_manager.load_config(overrides=overrides)


def _initialize_database_schema(config) -> None:
    """Initialize database schema prompt file."""
    from ..utils.codes_db_utils import get_db_schema, get_db_schema_sequence

    prompt_column_content_limit = 3

    schema_dict = get_db_schema(
        config.db_path, {}, None, column_content_limit=prompt_column_content_limit
    )

    schema_prompt = get_db_schema_sequence(schema_dict)

    config.schema_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.schema_prompt_path, "w") as f:
        f.write("database schema: \n")
        f.write(schema_prompt)


def _ensure_data_dir(config) -> None:
    """Create the configured data directory for commands that write state."""
    config.data_path.mkdir(parents=True, exist_ok=True)


def _ensure_database_exists(config) -> None:
    """Fail before SQLite helpers can create an empty database implicitly."""
    if not config.db_file_path.exists():
        raise ConfigurationError(
            f"Database file not found: {config.db_file_path}. "
            "Set db_path/TECOD_DB_PATH to an existing SQLite database before "
            "running this command.",
            "db_path",
        )


def _ensure_schema_prompt(config) -> None:
    """Create schema.prompt only for commands that need database schema text."""
    _ensure_data_dir(config)
    _ensure_database_exists(config)
    if not config.schema_prompt_path.exists():
        _initialize_database_schema(config)


def _validate_data_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate raw/prepared data columns with a concise actionable error."""
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ConfigurationError(
            "Input data is missing required column(s): "
            f"{', '.join(missing)}. Required columns: "
            f"{', '.join(required_columns)}. Columns present: "
            f"{', '.join(map(str, data.columns)) or '(none)'}.",
            "data",
        )


def create_index_command(device: str | None = None) -> None:
    """Create vector index from examples data."""
    with cli_error_handler("index creation"):
        config = cli_context.config_manager.config
        _ensure_data_dir(config)
        cli_context.container = ServiceContainer()
        cli_context.container.set_config(config)
        cli_context.tecod_service = None

        default_device = device or config.device
        emb_device = resolve_device(config.emb.device, default_device)

        from ..services.embedding import EmbeddingService
        from ..services.vector_store import VectorStoreService

        embedding_service = EmbeddingService(config, emb_device)
        vector_store_service = VectorStoreService(config, embedding_service)
        cli_context.container.register("embedding", embedding_service)
        cli_context.container.register("vector_store", vector_store_service)

        examples = pd.read_json(config.examples_path, lines=True)
        _validate_data_columns(
            examples,
            ["text", config.tecod.sql_key, config.emb.masked_nlq_key, "t_id"],
        )

        embedding_service.initialize()
        vector_store_service.create_index(examples)

        typer.echo(f"[OK] Vector index created successfully with {len(examples)} examples")


def compile_templates_command(device: str | None = None) -> None:
    """Compile templates for partitioned decoding."""
    with cli_error_handler("template compilation"):
        config = cli_context.config_manager.config
        if config.tecod.provider == "openai":
            raise ConfigurationError(
                "Template compilation is local-model-only and is not supported "
                "when tecod.provider=openai. Use a local model provider to run "
                "compile-templates.",
                "tecod.provider",
            )
        _ensure_schema_prompt(config)

        from ..pdec.compile_template import generate_token_ids_and_save_to_store
        from ..prompts import generate_prompt

        cli_context.initialize_services(device)

        vector_store_service = cli_context.container.get("vector_store")
        model_service = cli_context.container.get("model")
        template_service = cli_context.container.get("template")

        # Load data
        templates = template_service.get_all_templates()
        examples = pd.read_json(config.examples_path, lines=True)
        _validate_data_columns(
            examples,
            ["text", config.tecod.sql_key, config.emb.masked_nlq_key, "t_id"],
        )

        with open(config.schema_prompt_path) as f:
            schema_prompt = f.read()

        # Ensure compiled templates directory exists
        config.compiled_templates_path.mkdir(parents=True, exist_ok=True)

        # Compile each template
        for i, row in tqdm(templates.iterrows(), total=len(templates), desc="Compiling templates"):
            t_id = i
            q_ids = row["q_ids"]
            k = 0
            while k < len(q_ids):
                q_id = q_ids[k]
                q_row = examples.iloc[q_id]
                question = q_row["text"]
                sql_query = q_row[config.tecod.sql_key]

                # Get ICL examples (simplified version)
                search_results = vector_store_service.search(
                    question, top_k=config.tecod.vectorsearch_top_k
                )
                retrieved_examples = examples.iloc[search_results[0].index].copy()
                retrieved_examples["cosine_score"] = search_results[0]["distance"].values

                icl_example_indices = pick_icl_example_indices(
                    retrieved_examples, config.tecod.icl_cnt
                )

                icl_examples = []
                if config.tecod.icl_cnt > 0:
                    for _, ex_row in examples.iloc[icl_example_indices].iterrows():
                        icl_examples.append((ex_row["text"], ex_row[config.tecod.sql_key]))
                    icl_examples = icl_examples[::-1]

                # Generate prompt
                prompt = generate_prompt(
                    model_id=config.tecod.model_id,
                    prompt_class=config.tecod.prompt_class or None,
                    schema_sequence=schema_prompt,
                    content_sequence="",
                    question_text=question,
                    icl_examples=icl_examples,
                    database_engine=config.tecod.dialect,
                )

                try:
                    compiled_template = generate_token_ids_and_save_to_store(
                        model=model_service.model,
                        template_id=t_id,
                        tokenizer=model_service.tokenizer,
                        prompt=prompt,
                        sql_query=sql_query,
                        db_path=str(config.db_file_path),
                        ebnf_type=config.tecod.grammar_type,
                    )

                    # Save compiled template
                    template_service.save_compiled_template(t_id, compiled_template[t_id])
                    if k > 0:
                        typer.echo(f"[OK] Successfully compiled template {t_id} on retry {k}")
                        logger = logging.getLogger("app")
                        logger.info(f"Successfully compiled template {t_id} on retry {k}")
                    break  # Successfully compiled, exit while loop
                except Exception as e:
                    logger = logging.getLogger("app")
                    logger.exception(f"Error compiling template {t_id}: {str(e)}")
                    logger.error(f"Question: {question}")
                    logger.error(f"SQL: {sql_query}")
                    logger.error(f"Retrying with next question for template {t_id}")
                    typer.echo(f"[ERROR] Error compiling template {t_id}: {e}", err=True)
                    k += 1
                    continue

            if k == len(q_ids):
                typer.echo(
                    f"[ERROR] Failed to compile template {t_id} after {k} attempts", err=True
                )
                logger = logging.getLogger("app")
                logger.error(f"Failed to compile template {t_id} after {k} attempts")

        typer.echo("[OK] Template compilation completed")


def process_data_command(json_path: str, prepare_only: bool = False) -> None:
    """Process raw data into examples and templates."""
    import re

    regex = r"'([^']*)(')([^']*)'"
    subst = "'\\1''\\3'"
    compiled_regex = re.compile(regex)

    def clean_up(text):
        return compiled_regex.sub(subst, text)

    with cli_error_handler("data processing"):
        from sqlglot import parse_one
        from sqlglot.optimizer.optimize_joins import optimize_joins

        config = cli_context.config_manager.config

        # Load and process data
        data = pd.read_json(json_path)
        _validate_data_columns(
            data,
            ["text", config.tecod.sql_key, config.emb.masked_nlq_key],
        )
        _ensure_schema_prompt(config)

        # Add timestamp
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        data["timestamp"] = ts

        for i in range(len(data)):
            try:
                sql = data.at[i, config.tecod.sql_key]
                sql = sql.replace("\\'", "''")
                optimized_sql = optimize_joins(parse_one(sql, dialect=config.tecod.dialect)).sql(
                    dialect=config.tecod.dialect
                )
                data.at[i, config.tecod.sql_key] = optimized_sql
            except Exception as e:
                # Fallback to cleaned up SQL
                logger = logging.getLogger("app")
                logger.exception(f"Error processing SQL at index {i}: {str(e)}")
                logger.error(f"Original SQL: {data.at[i, config.tecod.sql_key]}")
                sql = data.at[i, config.tecod.sql_key]
                sql = sql.replace("\\'", "''")
                sql = clean_up(sql)
                optimized_sql = optimize_joins(parse_one(sql, dialect=config.tecod.dialect)).sql(
                    dialect=config.tecod.dialect
                )
                data.at[i, config.tecod.sql_key] = optimized_sql

        # Save examples
        data.to_json(config.examples_path, lines=True, orient="records")

        typer.echo(f"[OK] Processed {len(data)} examples and saved to {config.examples_path}")
        logger = logging.getLogger("app")
        logger.info(f"Processed {len(data)} examples and saved to {config.examples_path}")

        # Create templates
        _create_templates()

        if prepare_only:
            typer.echo(
                "[OK] Prepare-only mode completed. "
                "Run create-index next to build the vector index."
            )
            return

        # Create index and compile templates
        create_index_command()
        if config.tecod.provider != "openai":
            compile_templates_command()
        else:
            typer.echo("[INFO] Skipping template compilation (not supported for provider=openai)")

        typer.echo("[OK] Data processing completed successfully")


def _create_templates() -> None:
    """Create templates from examples data."""
    from ..pdec.tecod_utils import convert_sql_string_to_template

    config = cli_context.config_manager.config

    data = pd.read_json(config.examples_path, lines=True)
    _validate_data_columns(
        data,
        ["text", config.tecod.sql_key, config.emb.masked_nlq_key],
    )

    # Generate templates
    data["template"] = data[config.tecod.sql_key].apply(
        lambda x: convert_sql_string_to_template(x, db_path=str(config.db_file_path))
    )

    # Group by template
    templates = data.reset_index().groupby("template").agg({"index": list}).reset_index()
    templates = templates.rename(columns={"index": "q_ids"})
    templates["num_questions"] = templates["q_ids"].apply(len)
    templates["timestamp"] = data["timestamp"].max()
    templates = templates.reindex(columns=["timestamp", "template", "q_ids", "num_questions"])

    # Add template IDs to examples
    for i, row in templates.iterrows():
        for q_id in row["q_ids"]:
            data.at[q_id, "t_id"] = int(i)

    # Save files
    templates.to_json(config.templates_path, lines=True, orient="records")
    data.to_json(config.examples_path, lines=True, orient="records")


def tecod_interactive_command(device: str | None = None) -> None:
    """Start interactive TeCoD session."""
    try:
        with cli_error_handler("interactive mode"):
            logger = logging.getLogger("app")
            typer.echo("🚀 Starting TeCoD client...")
            logger.info("Starting TeCoD client...")

            config = cli_context.config_manager.config
            _ensure_schema_prompt(config)
            cli_context.initialize_services(device)

            typer.echo("[OK] TeCoD client ready!")
            logger.info("TeCoD client ready")

            while True:
                query = typer.prompt("Enter a natural language query (or 'exit' to quit)")

                if query.lower() in {"exit", "quit"}:
                    typer.echo("👋 Goodbye!")
                    break

                try:
                    request = GenerationRequest(query=query)
                    output = cli_context.tecod_service.generate(request)

                    # Detailed output with timing information
                    typer.echo("\n" + "=" * 80)
                    typer.echo(f"📝 Query: {output.query}")
                    typer.echo(f"🎯 Method: {output.method.upper()}")
                    typer.echo(f"🏷️ Template ID: {output.template_id}")
                    typer.echo(f"📊 NLI Score: {output.nli_score:.4f} ({output.nli_label})")
                    typer.echo(f"📐 Cosine Score: {output.cosine_score:.4f}")
                    typer.echo(f"💾 Generated SQL: {output.pred_sql}")

                    # Timing information
                    if output.total_time:
                        typer.echo("\n⏱️ TIMING BREAKDOWN:")
                        typer.echo(f"   Total Time: {output.total_time * 1000:.1f}ms")
                        if output.template_selection_time:
                            typer.echo(
                                f"   Template Selection: {output.template_selection_time * 1000:.1f}ms"
                            )
                        if output.vector_search_time:
                            typer.echo(
                                f"   Vector Search: {output.vector_search_time * 1000:.1f}ms"
                            )
                        if output.embedding_time:
                            typer.echo(f"   Query Embedding: {output.embedding_time * 1000:.1f}ms")
                        if output.nli_processing_time:
                            typer.echo(
                                f"   NLI Processing: {output.nli_processing_time * 1000:.1f}ms"
                            )
                        if output.generation_time:
                            typer.echo(f"   Generation: {output.generation_time * 1000:.1f}ms")

                    # Data flow information
                    if output.retrieved_examples_count or output.nli_examples_count:
                        typer.echo("\n📊 DATA FLOW:")
                        if output.retrieved_examples_count:
                            typer.echo(f"   Retrieved Examples: {output.retrieved_examples_count}")
                        if output.nli_examples_count:
                            typer.echo(f"   NLI Processed: {output.nli_examples_count}")
                        if output.icl_example_indices:
                            typer.echo(f"   ICL Examples: {len(output.icl_example_indices)}")

                    # Detailed timing data if available
                    if output.timing_data:
                        typer.echo("\n🔧 DETAILED TIMING:")
                        for operation, value in output.timing_data.items():
                            # Only show actual timing data, not counts
                            if not operation.endswith("_count") and "count" not in operation:
                                typer.echo(f"   {operation}: {value * 1000:.1f}ms")

                    typer.echo("=" * 80 + "\n")

                    # Log the complete output for debugging
                    logger.info(f"Generation completed: {output.model_dump()}")

                except Exception as e:
                    logger = logging.getLogger("app")
                    logger.exception(f"Error processing query '{query}': {str(e)}")
                    typer.echo(f"[ERROR] Error processing query: {str(e)}", err=True)
    finally:
        # Cleanup (typer.Exit inherits from SystemExit, so finally still runs)
        if cli_context.tecod_service:
            cli_context.container.cleanup_all()
