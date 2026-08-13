"""Host-side source metadata for experiment tracking."""

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _git_output(repository: str | Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_git_metadata(repository: str | Path = ".") -> dict[str, str]:
    """Return stable Git identifiers for W&B run configuration."""

    try:
        branch = _git_output(repository, "branch", "--show-current")
        if not branch:
            branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME", "")
        return {
            "git_branch": branch or "detached",
            "git_commit_message": _git_output(repository, "log", "-1", "--format=%s"),
            "git_commit_hash": _git_output(repository, "rev-parse", "HEAD"),
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        logger.warning("Git metadata is unavailable: %s", error)
        return {
            "git_branch": "unknown",
            "git_commit_message": "unknown",
            "git_commit_hash": "unknown",
        }


def make_wandb_run_name(git_metadata: dict[str, str]) -> str:
    """Build a normalized W&B run name from the branch and commit message."""

    source_name = f"{git_metadata['git_branch']}_{git_metadata['git_commit_message']}"
    normalized_name = re.sub(r"[^a-z0-9]+", "_", source_name.lower())
    return normalized_name.strip("_")
