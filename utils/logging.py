import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_format: str | None = None,
    log_file: str | Path | None = None,
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
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def setup_run_logging(
    run_dir: str | Path,
    level: int = logging.INFO,
    log_format: str | None = None,
) -> Path:
    """
    Sets up logging for a specific run directory.

    Configures both console output and file logging to run_dir/logs/hydra.log.

    Args:
        run_dir: Path to the run directory
        level: The logging level (e.g., logging.INFO, logging.DEBUG)
        log_format: Custom log format string

    Returns:
        Path to the log file
    """
    run_dir = Path(run_dir)
    log_file = run_dir / "logs" / "hydra.log"

    setup_logging(level=level, log_format=log_format, log_file=log_file)

    return log_file


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the given name.
    """
    return logging.getLogger(name)
