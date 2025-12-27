import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    log_format: str | None = None,
    log_file: str | None = None,
) -> None:
    """
    Sets up a project-wide logging configuration.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_format: Custom log format string.
        log_file: Optional path to a file to write logs to.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the given name.
    """
    return logging.getLogger(name)
