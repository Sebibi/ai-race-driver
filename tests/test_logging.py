"""Tests for command-line logging configuration."""

import io
import logging

from colorama import Fore, Style

from ai_race_driver.logging import ColorFormatter, configure_logging


def _record(level: int, message: str = "message") -> logging.LogRecord:
    return logging.LogRecord("ai_race_driver.test", level, __file__, 1, message, (), None)


def test_color_formatter_colors_each_standard_level() -> None:
    formatter = ColorFormatter(use_color=True)
    expected_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    for level, color in expected_colors.items():
        rendered = formatter.format(_record(level))
        assert f"{color}{logging.getLevelName(level)}{Style.RESET_ALL}" in rendered


def test_color_formatter_can_disable_ansi_codes() -> None:
    rendered = ColorFormatter(use_color=False).format(_record(logging.WARNING))

    assert Fore.YELLOW not in rendered
    assert "WARNING" in rendered


def test_configure_logging_sets_package_level_and_stream() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)

    logger = logging.getLogger("ai_race_driver.test")
    logger.debug("diagnostic")

    assert logging.getLogger("ai_race_driver").level == logging.DEBUG
    assert "DEBUG" in stream.getvalue()
    assert "diagnostic" in stream.getvalue()
