"""Reusable process-environment configuration helpers."""

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

WANDB_REQUIRED_ENVIRONMENT_VARIABLES = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
)


def _missing_environment_variables(required_variables: Sequence[str]) -> list[str]:
    return [name for name in required_variables if not os.environ.get(name, "").strip()]


def setup_environment(
    required_variables: Sequence[str] = WANDB_REQUIRED_ENVIRONMENT_VARIABLES,
    *,
    dotenv_path: str | Path | None = None,
) -> None:
    """Ensure required variables exist, loading ``.env`` only when necessary."""

    missing_variables = _missing_environment_variables(required_variables)
    if not missing_variables:
        logger.info("Using preloaded environment variables")
        return

    load_dotenv(dotenv_path=dotenv_path, override=False)
    missing_variables = _missing_environment_variables(required_variables)
    if missing_variables:
        rendered_names = ", ".join(missing_variables)
        raise SystemExit(
            "Missing required environment variables after loading .env: " + rendered_names
        )

    logger.info("Using environment variables loaded from .env")
