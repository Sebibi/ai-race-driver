"""Tests for reusable process-environment setup."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ai_race_driver import configuration

REQUIRED_VARIABLES = ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY")


@pytest.fixture(autouse=True)
def _restore_wandb_environment() -> Iterator[None]:
    tracked_variables = (*REQUIRED_VARIABLES, "WANDB_NAME", "WANDB_MODE")
    original_values = {
        variable: configuration.os.environ.get(variable) for variable in tracked_variables
    }
    yield
    for variable, value in original_values.items():
        if value is None:
            configuration.os.environ.pop(variable, None)
        else:
            configuration.os.environ[variable] = value


def _clear_required_variables(monkeypatch: Any) -> None:
    for variable in REQUIRED_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_setup_environment_uses_preloaded_variables(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    for variable in REQUIRED_VARIABLES:
        monkeypatch.setenv(variable, f"preloaded-{variable.lower()}")

    def fail_if_called(*_: object, **__: object) -> None:
        pytest.fail("load_dotenv should not run when required variables are preloaded")

    monkeypatch.setattr(configuration, "load_dotenv", fail_if_called)

    with caplog.at_level(logging.INFO, logger=configuration.__name__):
        configuration.setup_environment()

    assert "Using preloaded environment variables" in caplog.text


def test_setup_environment_loads_missing_variables_from_dotenv(
    tmp_path: Path, monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    _clear_required_variables(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "WANDB_API_KEY=test-key\nWANDB_PROJECT=hpc-training\nWANDB_ENTITY=race-team\n"
    )

    with caplog.at_level(logging.INFO, logger=configuration.__name__):
        configuration.setup_environment(dotenv_path=dotenv_path)

    assert configuration.os.environ["WANDB_API_KEY"] == "test-key"
    assert configuration.os.environ["WANDB_PROJECT"] == "hpc-training"
    assert configuration.os.environ["WANDB_ENTITY"] == "race-team"
    assert "Using environment variables loaded from .env" in caplog.text


def test_setup_environment_preserves_preloaded_values(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _clear_required_variables(monkeypatch)
    monkeypatch.setenv("WANDB_PROJECT", "scheduler-project")
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "WANDB_API_KEY=test-key\nWANDB_PROJECT=dotenv-project\nWANDB_ENTITY=race-team\n"
    )

    configuration.setup_environment(dotenv_path=dotenv_path)

    assert configuration.os.environ["WANDB_PROJECT"] == "scheduler-project"


def test_setup_environment_exits_when_required_variables_remain_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _clear_required_variables(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("WANDB_API_KEY=   \nWANDB_PROJECT=hpc-training\n")

    with pytest.raises(SystemExit, match="WANDB_API_KEY, WANDB_ENTITY"):
        configuration.setup_environment(dotenv_path=dotenv_path)
