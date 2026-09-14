"""Host-side source metadata for experiment tracking."""

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_GIT_ENVIRONMENT = {
    "git_branch": "AI_RACE_GIT_BRANCH",
    "git_commit_message": "AI_RACE_GIT_COMMIT_MESSAGE",
    "git_commit_hash": "AI_RACE_GIT_COMMIT_HASH",
}


def _git_output(repository: str | Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _environment_git_metadata() -> dict[str, str]:
    """Read source identifiers injected by an image builder or CI runner."""

    branch = (
        os.environ.get(_IMAGE_GIT_ENVIRONMENT["git_branch"])
        or os.environ.get("GITHUB_HEAD_REF")
        or os.environ.get("GITHUB_REF_NAME")
        or "unknown"
    )
    return {
        "git_branch": branch,
        "git_commit_message": os.environ.get(
            _IMAGE_GIT_ENVIRONMENT["git_commit_message"], "unknown"
        ),
        "git_commit_hash": os.environ.get(
            _IMAGE_GIT_ENVIRONMENT["git_commit_hash"],
            os.environ.get("GITHUB_SHA", "unknown"),
        ),
    }


def get_git_metadata(repository: str | Path = ".") -> dict[str, str]:
    """Return stable Git identifiers for W&B run configuration."""

    try:
        branch = _git_output(repository, "branch", "--show-current")
        if not branch:
            branch = _environment_git_metadata()["git_branch"]
        return {
            "git_branch": branch or "detached",
            "git_commit_message": _git_output(repository, "log", "-1", "--format=%s"),
            "git_commit_hash": _git_output(repository, "rev-parse", "HEAD"),
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        logger.warning("Git metadata is unavailable: %s", error)
        return _environment_git_metadata()


def make_wandb_run_name(git_metadata: dict[str, str]) -> str:
    """Build a normalized W&B run name from the branch and commit message."""

    source_name = f"{git_metadata['git_branch']}_{git_metadata['git_commit_message']}"
    normalized_name = re.sub(r"[^a-z0-9]+", "_", source_name.lower())
    return normalized_name.strip("_")
