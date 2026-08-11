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


def test_configure_logging_handles_direct_script_logger() -> None:
    stream = io.StringIO()
    direct_logger = logging.getLogger("__main__")
    original_handlers = direct_logger.handlers.copy()
    original_level = direct_logger.level
    original_propagate = direct_logger.propagate
    try:
        configure_logging("INFO", stream=stream, entrypoint_logger=direct_logger)

        direct_logger.info("direct invocation")

        assert "INFO" in stream.getvalue()
        assert "direct invocation" in stream.getvalue()
    finally:
        direct_logger.handlers = original_handlers
        direct_logger.setLevel(original_level)
        direct_logger.propagate = original_propagate
