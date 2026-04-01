import logging
import os
import sys

_DEFAULT_FORMAT = "[%(levelname)s] %(message)s"
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def setup_logging(level: str | None = None, force: bool = False) -> None:
    """Configure process-wide logging with a concise console format."""
    log_level = (level or os.getenv("DINO_QPM_LOG_LEVEL", "INFO")).upper()
    if log_level not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid log level '{log_level}'. Expected one of: {sorted(_VALID_LEVELS)}"
        )

    # Persist selected level so later setup_logging(force=True) calls keep it.
    os.environ["DINO_QPM_LOG_LEVEL"] = log_level
    numeric_level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format=_DEFAULT_FORMAT,
        stream=sys.stdout,
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
