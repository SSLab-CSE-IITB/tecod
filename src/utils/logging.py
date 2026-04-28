"""Logging configuration for TeCoD application."""

import contextvars
import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime

# Per-task/per-thread stack of extra fields contributed by LogContext.
# A ContextVar (not a module-level dict) means two threads — or two
# concurrent asyncio tasks — can each have their own logical context
# without stepping on each other, which is impossible with the old
# setLogRecordFactory approach.
_log_extras: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "tecod_log_extras", default=None
)


class _ContextExtraFilter(logging.Filter):
    """Attaches the current _log_extras dict to each record, if any.

    Installed by setup_logging on each handler so named child loggers that
    propagate to root handlers include context extras.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        extras = _log_extras.get()
        if extras:
            record.extra_data = extras
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs log records as JSON objects for easier parsing and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        # Timestamps are always UTC (the trailing "Z" marks Zulu / UTC),
        # which keeps log aggregation consistent across hosts and tz.
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        if hasattr(record, "extra_data"):
            log_obj["extra"] = record.extra_data

        return json.dumps(log_obj)


def setup_logging(
    console_level: str | int = "ERROR",
    file_level: str | int = "INFO",
    log_file: str | None = "app.log",
    log_to_console: bool = True,
    format_string: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    use_json_format: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Set up logging configuration for TeCoD.

    Args:
        console_level: Console logging level (DEBUG, INFO, WARNING, ERROR).
        file_level: File logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file (None to disable file logging).
        log_to_console: Whether to log to console.
        format_string: Log message format for standard formatter.
        use_json_format: Use JSON format for structured logging.
        max_file_size: Maximum size of log file before rotation (bytes).
        backup_count: Number of backup log files to keep.

    Returns:
        Configured root logger.
    """
    # Convert string levels to logging constants
    if isinstance(console_level, str):
        console_numeric_level = getattr(logging, console_level.upper(), logging.ERROR)
    else:
        console_numeric_level = console_level

    if isinstance(file_level, str):
        file_numeric_level = getattr(logging, file_level.upper(), logging.DEBUG)
    else:
        file_numeric_level = file_level

    # Create formatters
    if use_json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(format_string)

    # Console always uses standard format for readability
    console_formatter = logging.Formatter(format_string)

    # Get root logger and set to most verbose level needed
    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_numeric_level, file_numeric_level))

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    context_filter = _ContextExtraFilter()

    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_numeric_level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)

    # Add file handler with rotation. RotatingFileHandler does not create
    # parent directories, so a fresh container with a configured but
    # non-existent log path (e.g. /var/log/tecod/) would fail on first
    # write. Create the parent and, if that fails (privilege error), fall
    # back to console-only logging rather than taking down startup.
    if log_file:
        from pathlib import Path

        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
            )
            file_handler.setLevel(file_numeric_level)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(context_filter)
            root_logger.addHandler(file_handler)
        except OSError as e:
            root_logger.warning(
                "Could not attach file log at %s (%s); continuing without file logs.",
                log_file,
                e,
            )

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def setup_tecod_logging(
    console_level: str | int = "WARNING",
    file_level: str | int = "DEBUG",
    log_file: str = "tecod.log",
    use_json_format: bool = False,
) -> logging.Logger:
    """Set up TeCoD-specific logging configuration.

    Args:
        console_level: Console logging level.
        file_level: File logging level.
        log_file: Path to log file.
        use_json_format: Use JSON format for structured logging.

    Returns:
        Main application logger.
    """
    # Set up root logging
    setup_logging(
        console_level=console_level,
        file_level=file_level,
        log_file=log_file,
        use_json_format=use_json_format,
    )

    # Get main application logger
    app_logger = logging.getLogger("tecod")
    app_logger.info("TeCoD logging initialized")

    return app_logger


class LogContext:
    """Context manager for adding extra data to log records.

    Uses a contextvars-backed stack so concurrent threads (and asyncio
    tasks) each see their own context. Nested LogContext blocks compose:
    inner fields override outer fields with the same key.

    Example:
        >>> with LogContext(request_id="abc123"):
        ...     logger.info("Processing request")
        # Logs: {"timestamp": "...", "message": "Processing request", "extra": {"request_id": "abc123"}}
    """

    def __init__(self, **kwargs):
        """Initialize log context with extra data.

        Args:
            **kwargs: Extra data to add to log records.
        """
        self.extra_data = kwargs
        self._token: contextvars.Token | None = None

    def __enter__(self):
        current = _log_extras.get() or {}
        merged = {**current, **self.extra_data}
        self._token = _log_extras.set(merged)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _log_extras.reset(self._token)
            self._token = None
        return False
