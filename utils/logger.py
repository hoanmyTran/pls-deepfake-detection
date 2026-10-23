import logging
import sys
from pathlib import Path
from typing import Optional, Union

LOG_FORMAT = "%(asctime)s | %(levelname)8s | %(message)s"


class CustomFormatter(logging.Formatter):
    """Color console messages by severity without mutating LogRecord objects."""

    COLORS = {
        logging.DEBUG: "\033[38;21m",
        logging.INFO: "\033[38;5;39m",
        logging.WARNING: "\033[38;5;226m",
        logging.ERROR: "\033[38;5;196m",
        logging.CRITICAL: "\033[31;1m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str = LOG_FORMAT, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.COLORS.get(record.levelno) if self.use_color else None
        return f"{color}{message}{self.RESET}" if color else message


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Get a logger without adding duplicate handlers on repeated calls."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(h, "_experiment_console", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._experiment_console = True
        handler.setFormatter(CustomFormatter(use_color=sys.stderr.isatty()))
        logger.addHandler(handler)

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        if not any(getattr(h, "_experiment_file", None) == path for h in logger.handlers):
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler._experiment_file = path
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(handler)

    return logger
