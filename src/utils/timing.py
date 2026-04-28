"""Timing utilities for TeCoD operations."""

import logging
import time
from contextlib import contextmanager


@contextmanager
def log_with_time_elapsed(
    operation_name: str, logger: logging.Logger | None = None, level: int = logging.INFO
):
    """Context manager to log operation timing.

    Args:
        operation_name: Name of the operation being timed
        logger: Logger instance (uses root logger if None)
        level: Logging level for the timing messages
    """
    if logger is None:
        logger = logging.getLogger()

    logger.log(level, f"{operation_name}...")
    start = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        # Show in milliseconds for better readability while keeping precise storage
        logger.log(level, f"{operation_name}... done in {elapsed * 1000:.1f}ms")


class Timer:
    """Timer class for measuring operation durations."""

    def __init__(self, operation_name: str, logger: logging.Logger | None = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger()
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.logger.debug(f"[TIMING] Starting {self.operation_name}")

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")

        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        # Display in milliseconds for better readability
        self.logger.info(f"[TIMING] {self.operation_name} completed in {elapsed * 1000:.1f}ms")
        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        # Explicit so the intent is obvious: we do not swallow exceptions.
        return False
