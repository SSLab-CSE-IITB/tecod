"""Refactored main entry point for TeCoD application."""

import typer

from src.cli.commands import (
    compile_templates_command,
    create_index_command,
    load_config_callback,
    process_data_command,
    tecod_interactive_command,
)
from src.utils.logging import setup_tecod_logging

app = typer.Typer(
    pretty_exceptions_enable=False,
    help="TeCoD: Template Constrained Decoding for Text-to-SQL",
)


@app.callback()
def init_system(
    ctx: typer.Context,
    env: str | None = typer.Option(
        None,
        "--env",
        "-e",
        help="Hydra environment config to load (e.g., 'local', 'openai').",
    ),
    overrides: list[str] | None = typer.Option(
        None,
        "--config-override",
        "-c",
        help="Hydra-style overrides (e.g., 'server.port=9000'). Applied globally.",
    ),
):
    """Initialize TeCoD system with configuration."""
    # Set up logging first
    setup_tecod_logging()

    if env:
        overrides = overrides or []
        overrides.insert(0, f"env@_global_={env}")

    # Load configuration
    load_config_callback(overrides)

    # Reconfigure logging from loaded config (e.g. env-specific log file)
    from src.config.manager import get_config_manager

    cfg = get_config_manager().config
    setup_tecod_logging(
        console_level=cfg.logging.console_level,
        file_level=cfg.logging.file_level,
        log_file=cfg.logging.log_file,
        use_json_format=cfg.logging.use_json_format,
    )


@app.command()
def version():
    """Show version information and current configuration."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    from src.config.manager import get_config_manager

    try:
        ver = pkg_version("TeCoD")
    except PackageNotFoundError:
        ver = "0.2.0"  # fallback for development (must stay in sync with pyproject.toml)
    typer.echo(f"TeCoD Version: {ver}")

    try:
        config_manager = get_config_manager()
        config = config_manager.config
        typer.echo("\nCurrent Configuration:")
        typer.echo(f"  Root Directory: {config.root_dir}")
        typer.echo(f"  Data Directory: {config.data_dir}")
        typer.echo(f"  Database Path: {config.db_path}")
        typer.echo(f"  Model: {config.tecod.model_id}")
        typer.echo(f"  Embedding Model: {config.emb.model}")
        typer.echo(f"  NLI Model: {config.nli.model}")
    except Exception as e:
        typer.echo(f"Error loading configuration: {e}", err=True)


@app.command("create-index")
def create_index(
    device: str | None = typer.Option(None, "--device", help="Device to use (cuda, cpu, auto)"),
):
    """Create vector index from examples data."""
    create_index_command(device)


@app.command("compile-templates")
def compile_templates(
    device: str | None = typer.Option(None, "--device", help="Device to use (cuda, cpu, auto)"),
):
    """Compile templates for partitioned decoding."""
    compile_templates_command(device)


@app.command("process-data")
def process_data(
    json_path: str = typer.Argument(help="Path to input JSON data file"),
    prepare_only: bool = typer.Option(
        False,
        "--prepare-only",
        help=(
            "Only write examples.jsonl, templates.jsonl, and schema.prompt. "
            "Skip vector index creation and local template compilation."
        ),
    ),
):
    """Process raw data into examples and templates."""
    process_data_command(json_path, prepare_only=prepare_only)


@app.command("tecod")
def tecod(
    device: str | None = typer.Option(None, "--device", help="Device to use (cuda, cpu, auto)"),
):
    """Start interactive TeCoD session."""
    tecod_interactive_command(device)


@app.command("status")
def status():
    """Show system status and service information."""
    from src.config.manager import get_config_manager

    try:
        config_manager = get_config_manager()
        config = config_manager.config

        typer.echo("TeCoD System Status")
        typer.echo("=" * 50)

        # Check configuration
        typer.echo("[OK]   Configuration loaded")

        # Check file existence
        files_to_check = [
            ("Examples", config.examples_path),
            ("Templates", config.templates_path),
            ("Schema Prompt", config.schema_prompt_path),
            ("Database", config.db_file_path),
        ]

        for name, path in files_to_check:
            if path.exists():
                typer.echo(f"[OK]   {name}: {path}")
            else:
                typer.echo(f"[FAIL] {name}: {path} (missing)")

        # Check compiled templates directory
        compiled_dir = config.compiled_templates_path
        if compiled_dir.exists():
            template_count = len(list(compiled_dir.glob("*.pkl")))
            typer.echo(f"[OK]   Compiled Templates: {template_count} templates found")
        else:
            typer.echo("[FAIL] Compiled Templates: Directory not found")

        # Check vector index
        index_path = config.index_path
        if index_path.exists():
            typer.echo(f"[OK]   Vector Index: {index_path}")
        else:
            typer.echo(f"[FAIL] Vector Index: {index_path} (missing)")

    except Exception as e:
        typer.echo(f"[FAIL] Error checking status: {e}", err=True)
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
