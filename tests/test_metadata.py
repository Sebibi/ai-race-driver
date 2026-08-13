"""Tests for source metadata collection."""

import subprocess
from typing import Any

from ai_race_driver import metadata


def test_get_git_metadata_collects_branch_message_and_hash(monkeypatch: Any) -> None:
    outputs = {
        ("branch", "--show-current"): "feature/hpc",
        ("log", "-1", "--format=%s"): "Add online W&B test",
        ("rev-parse", "HEAD"): "abc123",
    }
    monkeypatch.setattr(metadata, "_git_output", lambda _, *args: outputs[args])

    assert metadata.get_git_metadata() == {
        "git_branch": "feature/hpc",
        "git_commit_message": "Add online W&B test",
        "git_commit_hash": "abc123",
    }


def test_get_git_metadata_uses_ci_branch_for_detached_head(monkeypatch: Any) -> None:
    outputs = {
        ("branch", "--show-current"): "",
        ("log", "-1", "--format=%s"): "Detached build",
        ("rev-parse", "HEAD"): "def456",
    }
    monkeypatch.setattr(metadata, "_git_output", lambda _, *args: outputs[args])
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/pull-request")

    assert metadata.get_git_metadata()["git_branch"] == "feature/pull-request"


def test_get_git_metadata_falls_back_when_git_is_unavailable(monkeypatch: Any) -> None:
    def fail(*_: object, **__: object) -> str:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(metadata, "_git_output", fail)

    assert metadata.get_git_metadata() == {
        "git_branch": "unknown",
        "git_commit_message": "unknown",
        "git_commit_hash": "unknown",
    }
