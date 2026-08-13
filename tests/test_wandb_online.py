"""Online W&B acceptance test."""

import os

import wandb

from ai_race_driver.configuration import setup_environment
from ai_race_driver.metadata import get_git_metadata, make_wandb_run_name


def test_wandb_run_is_uploaded_to_portal() -> None:
    setup_environment()
    git_metadata = get_git_metadata()
    run_name = make_wandb_run_name(git_metadata)

    with wandb.init(
        name=run_name,
        job_type="integration-test",
        tags=["integration-test"],
        config=git_metadata,
    ) as run:
        run.log({"integration_test/value": 1})
        run_id = run.id
        run_url = run.url

    assert run_url
    print(f"W&B online test run: {run_url}")

    run_path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{run_id}"
    uploaded_run = wandb.Api().run(run_path)
    assert uploaded_run.name == run_name
    assert uploaded_run.summary["integration_test/value"] == 1
    assert uploaded_run.config["git_branch"] == git_metadata["git_branch"]
    assert uploaded_run.config["git_commit_message"] == git_metadata["git_commit_message"]
    assert uploaded_run.config["git_commit_hash"] == git_metadata["git_commit_hash"]
