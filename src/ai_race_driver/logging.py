"""Logging configuration shared by the command-line applications."""

import copy
import logging
import sys
from typing import TextIO

from colorama import Fore, Style, just_fix_windows_console

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
}


class ColorFormatter(logging.Formatter):
    """Format log level names with terminal colors without mutating the record."""

    def __init__(self, *, use_color: bool) -> None:
        super().__init__(DEFAULT_LOG_FORMAT)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)

        color = _LEVEL_COLORS.get(record.levelno)
        if color is None:
            return super().format(record)

        colored_record = copy.copy(record)
        colored_record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(colored_record)


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> None:
    """Configure package logging for a CLI process."""
    output = stream if stream is not None else sys.stderr
    just_fix_windows_console()

    handler = logging.StreamHandler(output)
    handler.setFormatter(ColorFormatter(use_color=output.isatty()))

    package_logger = logging.getLogger("ai_race_driver")
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(level.upper())
    package_logger.propagate = False
