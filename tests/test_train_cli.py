"""Tests for the PPO training command."""

from typing import Any

import jax.numpy as jnp

from ai_race_driver.cli import train
from ai_race_driver.training.ppo import PPOConfig, UpdateMetrics


class _FakeRun:
    def __init__(self) -> None:
        self.history: list[tuple[dict[str, float | int], int]] = []
        self.summary: dict[str, Any] = {}
        self.finished = False

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, *_: object) -> None:
        self.finished = True

    def log(self, values: dict[str, float | int], *, step: int) -> None:
        self.history.append((values, step))


def test_log_wandb_run_records_each_ppo_update(monkeypatch: Any) -> None:
    config = PPOConfig.cpu_smoke()
    metrics = UpdateMetrics(
        loss=jnp.array([1.0, 0.5]),
        mean_reward=jnp.array([2.0, 3.0]),
        completed_laps=jnp.array([0, 1]),
        completed_episodes=jnp.array([4, 5]),
        mean_completed_return=jnp.array([6.0, 7.0]),
    )
    run = _FakeRun()
    init_arguments: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> _FakeRun:
        init_arguments.update(kwargs)
        return run

    monkeypatch.setattr(train.wandb, "init", fake_init)

    train.log_wandb_run(
        project="ai-race-driver-test",
        entity="race-team",
        name="smoke",
        mode="offline",
        config=config,
        seed=7,
        metrics=metrics,
        summary={"final_loss": 0.5},
    )

    assert init_arguments["project"] == "ai-race-driver-test"
    assert init_arguments["entity"] == "race-team"
    assert init_arguments["name"] == "smoke"
    assert init_arguments["mode"] == "offline"
    assert init_arguments["config"]["seed"] == 7
    assert init_arguments["config"]["num_envs"] == config.num_envs
    assert run.finished
    assert run.summary == {"final_loss": 0.5}
    assert [step for _, step in run.history] == [1, 2]
    assert [record["train/timesteps"] for record, _ in run.history] == [64, 128]
    assert run.history[-1][0]["train/loss"] == 0.5
    assert run.history[-1][0]["train/completed_laps"] == 1.0


def test_wandb_is_opt_in() -> None:
    args = train.build_parser().parse_args([])

    assert args.wandb_project is None
    assert args.wandb_mode == "online"
