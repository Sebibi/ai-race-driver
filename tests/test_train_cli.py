"""Tests for the PPO training command."""

from typing import Any

import jax.numpy as jnp
import pytest

from ai_race_driver.cli import train
from ai_race_driver.training.evaluation import EvaluationMetrics, EvaluationSuiteMetrics
from ai_race_driver.training.ppo import PPOConfig, UpdateMetrics


def _update_metrics() -> UpdateMetrics:
    return UpdateMetrics(
        loss=jnp.array([1.0, 0.5]),
        policy_loss=jnp.array([0.1, 0.2]),
        value_loss=jnp.array([0.3, 0.4]),
        entropy=jnp.array([1.5, 1.4]),
        approx_kl=jnp.array([0.01, 0.02]),
        clip_fraction=jnp.array([0.03, 0.04]),
        gradient_norm=jnp.array([0.5, 0.6]),
        explained_variance=jnp.array([0.7, 0.8]),
        learning_rate=jnp.array([3e-4, 1.5e-4]),
        mean_reward=jnp.array([2.0, 3.0]),
        completed_laps=jnp.array([0, 1]),
        completed_episodes=jnp.array([0, 2]),
        mean_completed_return=jnp.array([jnp.nan, 7.0]),
        completed_return_sum=jnp.array([0.0, 14.0]),
        completed_return_squared_sum=jnp.array([0.0, 100.0]),
        mean_completed_length=jnp.array([jnp.nan, 11.0]),
        completed_length_sum=jnp.array([0, 22]),
        off_track_episodes=jnp.array([0, 1]),
        time_limit_episodes=jnp.array([0, 0]),
    )


def _evaluation(value: float) -> EvaluationMetrics:
    return EvaluationMetrics(
        return_mean=jnp.asarray(value),
        return_std=jnp.asarray(0.5),
        return_min=jnp.asarray(value - 1.0),
        return_max=jnp.asarray(value + 1.0),
        length_mean=jnp.asarray(20.0),
        progress_mean=jnp.asarray(30.0),
        lap_success_rate=jnp.asarray(0.75),
        off_track_rate=jnp.asarray(0.25),
        time_limit_rate=jnp.asarray(0.0),
    )


def test_training_log_records_every_update_without_false_episode_average() -> None:
    config = PPOConfig.cpu_smoke()
    metrics = _update_metrics()

    first = train.training_log_record(metrics, 0, update=1, config=config)
    second = train.training_log_record(metrics, 1, update=2, config=config)

    assert first["global_step"] == 64
    assert "train/episode/return_mean" not in first
    assert second["global_step"] == 128
    assert second["train/loss/total"] == 0.5
    assert second["train/episode/return_mean"] == 7.0
    assert second["train/episode/lap_success_rate"] == 0.5


def test_evaluation_log_uses_distinct_fixed_and_randomized_names() -> None:
    suite = EvaluationSuiteMetrics(fixed=_evaluation(4.0), randomized=_evaluation(5.0))

    record = train.evaluation_log_record(suite, global_step=128)

    assert record["global_step"] == 128
    assert record["eval/fixed/return_mean"] == 4.0
    assert record["eval/randomized/return_mean"] == 5.0


class _FakeRun:
    def __init__(self) -> None:
        self.definitions: list[tuple[str, dict[str, Any]]] = []
        self.history: list[dict[str, float | int]] = []

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.definitions.append((name, kwargs))

    def log(self, record: dict[str, float | int]) -> None:
        self.history.append(record)


def test_wandb_metrics_use_global_environment_step() -> None:
    run = _FakeRun()

    train.configure_wandb_metrics(run)

    assert ("global_step", {"hidden": True}) in run.definitions
    assert ("train/*", {"step_metric": "global_step"}) in run.definitions
    assert ("eval/*", {"step_metric": "global_step"}) in run.definitions


def test_chunk_logging_keeps_per_update_history_and_evaluates_at_boundary() -> None:
    run = _FakeRun()
    config = PPOConfig.cpu_smoke()
    suite = EvaluationSuiteMetrics(fixed=_evaluation(4.0), randomized=_evaluation(5.0))

    train.log_training_chunk(
        run,
        _update_metrics(),
        suite,
        completed_updates=2,
        config=config,
        chunk_seconds=2.0,
        evaluation_seconds=0.5,
        cumulative_training_seconds=2.0,
    )

    assert [record["global_step"] for record in run.history] == [64, 128]
    assert "eval/fixed/return_mean" not in run.history[0]
    assert run.history[1]["eval/fixed/return_mean"] == 4.0
    assert run.history[1]["performance/train_steps_per_second"] == 64.0


def test_live_tracking_cli_defaults_and_validation() -> None:
    args = train.build_parser().parse_args([])

    assert args.log_every_updates == 4
    assert args.eval_episodes == 32
    assert args.eval_seed == 0
    assert not any(name.startswith("wandb_") for name in vars(args))
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--log-every-updates", "0"])
