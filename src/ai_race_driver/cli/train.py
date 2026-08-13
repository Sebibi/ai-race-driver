"""Train PPO on the built-in oval with live evaluation and progress tracking."""

import argparse
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np
import wandb

from ai_race_driver.configuration import setup_environment
from ai_race_driver.envs.racing import make_default_env
from ai_race_driver.logging import LOG_LEVELS, configure_logging
from ai_race_driver.metadata import get_git_metadata
from ai_race_driver.training.evaluation import (
    EvaluationMetrics,
    EvaluationSuiteMetrics,
    make_evaluate_policy,
)
from ai_race_driver.training.ppo import (
    ActorCritic,
    PPOConfig,
    UpdateMetrics,
    make_training_fns,
    save_checkpoint,
    save_checkpoint_metadata,
)

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=8_388_608)
    parser.add_argument("--num-envs", type=int, default=2_048)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument(
        "--log-every-updates",
        type=_positive_int,
        default=4,
        help="PPO updates per compiled chunk and live logging interval",
    )
    parser.add_argument(
        "--eval-episodes",
        type=_positive_int,
        default=32,
        help="number of randomized-start deterministic evaluation episodes",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="fixed seed used for comparable periodic evaluation starts",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest"))
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    return parser


def _episode_statistics(metrics: UpdateMetrics, index: int) -> dict[str, float]:
    count = float(metrics.completed_episodes[index])
    statistics = {"train/episode/count": count}
    if count == 0:
        return statistics

    return_sum = float(metrics.completed_return_sum[index])
    return_squared_sum = float(metrics.completed_return_squared_sum[index])
    return_mean = return_sum / count
    return_variance = max(return_squared_sum / count - return_mean**2, 0.0)
    statistics.update(
        {
            "train/episode/return_mean": return_mean,
            "train/episode/return_std": math.sqrt(return_variance),
            "train/episode/length_mean": float(metrics.completed_length_sum[index]) / count,
            "train/episode/lap_success_rate": float(metrics.completed_laps[index]) / count,
            "train/episode/off_track_rate": float(metrics.off_track_episodes[index]) / count,
            "train/episode/time_limit_rate": float(metrics.time_limit_episodes[index]) / count,
        }
    )
    return statistics


def training_log_record(
    metrics: UpdateMetrics,
    index: int,
    *,
    update: int,
    config: PPOConfig,
) -> dict[str, float | int]:
    """Convert one device update's metrics into stable host logging names."""

    record: dict[str, float | int] = {
        "global_step": update * config.batch_size,
        "update": update,
        "train/reward/transition_mean": float(metrics.mean_reward[index]),
        "train/episode/completed_laps": float(metrics.completed_laps[index]),
        "train/loss/total": float(metrics.loss[index]),
        "train/loss/policy": float(metrics.policy_loss[index]),
        "train/loss/value": float(metrics.value_loss[index]),
        "train/policy/entropy_sample": float(metrics.entropy[index]),
        "train/policy/approx_kl": float(metrics.approx_kl[index]),
        "train/policy/clip_fraction": float(metrics.clip_fraction[index]),
        "train/policy/gradient_norm": float(metrics.gradient_norm[index]),
        "train/value/explained_variance": float(metrics.explained_variance[index]),
        "train/learning_rate": float(metrics.learning_rate[index]),
    }
    record.update(_episode_statistics(metrics, index))
    return record


def _evaluation_metrics_record(prefix: str, metrics: EvaluationMetrics) -> dict[str, float]:
    return {
        f"{prefix}/return_mean": float(metrics.return_mean),
        f"{prefix}/return_std": float(metrics.return_std),
        f"{prefix}/return_min": float(metrics.return_min),
        f"{prefix}/return_max": float(metrics.return_max),
        f"{prefix}/length_mean": float(metrics.length_mean),
        f"{prefix}/progress_mean": float(metrics.progress_mean),
        f"{prefix}/lap_success_rate": float(metrics.lap_success_rate),
        f"{prefix}/off_track_rate": float(metrics.off_track_rate),
        f"{prefix}/time_limit_rate": float(metrics.time_limit_rate),
    }


def evaluation_log_record(
    metrics: EvaluationSuiteMetrics,
    *,
    global_step: int,
) -> dict[str, float | int]:
    """Convert fixed and randomized evaluation aggregates for logging."""

    return {
        "global_step": global_step,
        **_evaluation_metrics_record("eval/fixed", metrics.fixed),
        **_evaluation_metrics_record("eval/randomized", metrics.randomized),
    }


def log_training_chunk(
    run: Any,
    metrics: UpdateMetrics,
    evaluation: EvaluationSuiteMetrics,
    *,
    completed_updates: int,
    config: PPOConfig,
    chunk_seconds: float,
    evaluation_seconds: float,
    cumulative_training_seconds: float,
) -> None:
    """Log every PPO update after one host synchronization of a compiled chunk."""

    chunk_updates = len(metrics.loss)
    global_step = completed_updates * config.batch_size
    for local_index in range(chunk_updates):
        update = completed_updates - chunk_updates + local_index + 1
        record = training_log_record(
            metrics,
            local_index,
            update=update,
            config=config,
        )
        if local_index == chunk_updates - 1:
            record.update(evaluation_log_record(evaluation, global_step=global_step))
            record.update(
                {
                    "performance/chunk_seconds": chunk_seconds,
                    "performance/evaluation_seconds": evaluation_seconds,
                    "performance/train_steps_per_second": (
                        chunk_updates * config.batch_size / chunk_seconds
                    ),
                    "performance/cumulative_train_steps_per_second": (
                        global_step / cumulative_training_seconds
                    ),
                }
            )
        run.log(record)


def configure_wandb_metrics(run: Any) -> None:
    """Use environment transitions as the x-axis for all live metrics."""

    run.define_metric("global_step", hidden=True)
    run.define_metric("update", hidden=True)
    for prefix in ("train/*", "eval/*", "performance/*"):
        run.define_metric(prefix, step_metric="global_step")


def _chunk_episode_summary(metrics: UpdateMetrics) -> dict[str, float]:
    count = float(np.asarray(metrics.completed_episodes).sum())
    if count == 0:
        return {
            "count": 0.0,
            "return_mean": math.nan,
            "return_std": math.nan,
            "lap_rate": math.nan,
            "off_track_rate": math.nan,
        }
    return_sum = float(np.asarray(metrics.completed_return_sum).sum())
    squared_sum = float(np.asarray(metrics.completed_return_squared_sum).sum())
    return_mean = return_sum / count
    return_variance = max(squared_sum / count - return_mean**2, 0.0)
    return {
        "count": count,
        "return_mean": return_mean,
        "return_std": math.sqrt(return_variance),
        "lap_rate": float(np.asarray(metrics.completed_laps).sum()) / count,
        "off_track_rate": float(np.asarray(metrics.off_track_episodes).sum()) / count,
    }


def _checkpoint_summary(
    evaluation: EvaluationSuiteMetrics,
    *,
    seed: int,
    update: int,
    global_step: int,
    checkpoint_kind: str,
    eval_episodes: int,
    eval_seed: int,
) -> dict[str, Any]:
    return {
        "checkpoint_kind": checkpoint_kind,
        "seed": seed,
        "update": update,
        "global_step": global_step,
        "evaluation_episodes": eval_episodes,
        "evaluation_seed": eval_seed,
        **evaluation_log_record(evaluation, global_step=global_step),
    }


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level, entrypoint_logger=logger)
    setup_environment()
    devices = jax.devices()
    logger.info("JAX devices: %s", devices)
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
    )
    env, env_params = make_default_env(randomize_reset=True)
    model = ActorCritic(action_dim=2, hidden_size=config.hidden_size)
    initialize, make_chunk = make_training_fns(env, env_params, config)
    evaluate_policy = make_evaluate_policy(
        env,
        env_params,
        model,
        randomized_episodes=args.eval_episodes,
    )
    training_key = jax.random.key(args.seed)
    evaluation_key = jax.random.key(args.eval_seed)
    git_metadata = get_git_metadata()
    wandb_config = {
        "seed": args.seed,
        "log_every_updates": args.log_every_updates,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        **asdict(config),
        **git_metadata,
    }

    logger.info("PPO config: %s", config)
    logger.info("Live logging and evaluation every %d PPO update(s)", args.log_every_updates)
    process_started = time.perf_counter()
    with wandb.init(config=wandb_config) as run:
        configure_wandb_metrics(run)

        compilation_started = time.perf_counter()
        compiled_initialize = jax.jit(initialize).lower(training_key).compile()
        initialization_compilation_seconds = time.perf_counter() - compilation_started
        initialization_started = time.perf_counter()
        runner_state = compiled_initialize(training_key)
        jax.block_until_ready(runner_state.observation)
        initialization_seconds = time.perf_counter() - initialization_started
        logger.info(
            "Compiled PPO initialization in %.3f seconds and initialized in %.3f seconds",
            initialization_compilation_seconds,
            initialization_seconds,
        )

        compilation_started = time.perf_counter()
        compiled_evaluate = (
            jax.jit(evaluate_policy)
            .lower(runner_state.train_state.params, evaluation_key)
            .compile()
        )
        evaluation_compilation_seconds = time.perf_counter() - compilation_started
        logger.info(
            "Compiled deterministic evaluation in %.3f seconds",
            evaluation_compilation_seconds,
        )

        evaluation_started = time.perf_counter()
        evaluation = compiled_evaluate(runner_state.train_state.params, evaluation_key)
        jax.block_until_ready(evaluation.fixed.return_mean)
        evaluation_seconds = time.perf_counter() - evaluation_started
        host_evaluation = jax.device_get(evaluation)
        initial_record = evaluation_log_record(host_evaluation, global_step=0)
        initial_record.update(
            {
                "update": 0,
                "performance/evaluation_seconds": evaluation_seconds,
                "performance/initialization_seconds": initialization_seconds,
                "performance/compilation_seconds": (
                    initialization_compilation_seconds + evaluation_compilation_seconds
                ),
            }
        )
        run.log(initial_record)

        best_return = float(host_evaluation.fixed.return_mean)
        best_update = 0
        best_step = 0
        save_checkpoint(
            args.output / "best",
            runner_state.train_state.params,
            config,
            _checkpoint_summary(
                host_evaluation,
                seed=args.seed,
                update=0,
                global_step=0,
                checkpoint_kind="best",
                eval_episodes=args.eval_episodes,
                eval_seed=args.eval_seed,
            ),
        )
        logger.info(
            "Initial evaluation | fixed return %.3f lap %.0f%% | randomized %.3f ± %.3f lap %.1f%%",
            float(host_evaluation.fixed.return_mean),
            100.0 * float(host_evaluation.fixed.lap_success_rate),
            float(host_evaluation.randomized.return_mean),
            float(host_evaluation.randomized.return_std),
            100.0 * float(host_evaluation.randomized.lap_success_rate),
        )

        compiled_chunks: dict[int, Any] = {}
        compilation_seconds = initialization_compilation_seconds + evaluation_compilation_seconds
        training_seconds = 0.0
        evaluation_seconds_total = evaluation_seconds
        final_metrics: UpdateMetrics | None = None

        completed_updates = 0
        while completed_updates < config.num_updates:
            chunk_updates = min(
                args.log_every_updates,
                config.num_updates - completed_updates,
            )
            if chunk_updates not in compiled_chunks:
                compilation_started = time.perf_counter()
                compiled_chunks[chunk_updates] = (
                    jax.jit(make_chunk(chunk_updates)).lower(runner_state).compile()
                )
                chunk_compilation_seconds = time.perf_counter() - compilation_started
                compilation_seconds += chunk_compilation_seconds
                logger.info(
                    "Compiled %d-update training chunk in %.3f seconds",
                    chunk_updates,
                    chunk_compilation_seconds,
                )

            chunk_started = time.perf_counter()
            chunk_output = compiled_chunks[chunk_updates](runner_state)
            jax.block_until_ready(chunk_output.metrics.loss)
            chunk_seconds = time.perf_counter() - chunk_started
            training_seconds += chunk_seconds
            runner_state = chunk_output.runner_state
            host_metrics = jax.device_get(chunk_output.metrics)
            final_metrics = host_metrics

            completed_updates += chunk_updates
            global_step = completed_updates * config.batch_size
            evaluation_started = time.perf_counter()
            evaluation = compiled_evaluate(runner_state.train_state.params, evaluation_key)
            jax.block_until_ready(evaluation.fixed.return_mean)
            evaluation_seconds = time.perf_counter() - evaluation_started
            evaluation_seconds_total += evaluation_seconds
            host_evaluation = jax.device_get(evaluation)

            fixed_return = float(host_evaluation.fixed.return_mean)
            if fixed_return > best_return:
                best_return = fixed_return
                best_update = completed_updates
                best_step = global_step
                save_checkpoint(
                    args.output / "best",
                    runner_state.train_state.params,
                    config,
                    _checkpoint_summary(
                        host_evaluation,
                        seed=args.seed,
                        update=completed_updates,
                        global_step=global_step,
                        checkpoint_kind="best",
                        eval_episodes=args.eval_episodes,
                        eval_seed=args.eval_seed,
                    ),
                )
                logger.info("Saved new best checkpoint with fixed return %.3f", best_return)

            log_training_chunk(
                run,
                host_metrics,
                host_evaluation,
                completed_updates=completed_updates,
                config=config,
                chunk_seconds=chunk_seconds,
                evaluation_seconds=evaluation_seconds,
                cumulative_training_seconds=training_seconds,
            )

            episode_summary = _chunk_episode_summary(host_metrics)
            if episode_summary["count"]:
                return_text = (
                    f"{episode_summary['return_mean']:.3f} ± {episode_summary['return_std']:.3f}"
                )
                lap_text = f"{100.0 * episode_summary['lap_rate']:.1f}%"
                off_track_text = f"{100.0 * episode_summary['off_track_rate']:.1f}%"
            else:
                return_text = "n/a"
                lap_text = "n/a"
                off_track_text = "n/a"
            logger.info(
                "Progress %d/%d updates | %s/%s steps | train return %s "
                "(%d episodes) | lap %s off-track %s | %.0f train steps/s",
                completed_updates,
                config.num_updates,
                f"{global_step:,}",
                f"{config.total_timesteps:,}",
                return_text,
                int(episode_summary["count"]),
                lap_text,
                off_track_text,
                chunk_updates * config.batch_size / chunk_seconds,
            )
            logger.info(
                "Evaluation | fixed return %.3f lap %.0f%% length %.0f | randomized "
                "%.3f ± %.3f lap %.1f%% | %.3fs | loss %.4f KL %.5f clip %.3f",
                fixed_return,
                100.0 * float(host_evaluation.fixed.lap_success_rate),
                float(host_evaluation.fixed.length_mean),
                float(host_evaluation.randomized.return_mean),
                float(host_evaluation.randomized.return_std),
                100.0 * float(host_evaluation.randomized.lap_success_rate),
                evaluation_seconds,
                float(host_metrics.loss[-1]),
                float(host_metrics.approx_kl[-1]),
                float(host_metrics.clip_fraction[-1]),
            )

        if final_metrics is None:
            raise RuntimeError("training completed without producing metrics")

        summary = {
            "seed": args.seed,
            "device": str(devices[0]),
            "compilation_seconds": compilation_seconds,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds_total,
            "train_steps_per_second": config.total_timesteps / training_seconds,
            "best_fixed_return": best_return,
            "best_update": best_update,
            "best_global_step": best_step,
            "final_loss": float(final_metrics.loss[-1]),
            **_checkpoint_summary(
                host_evaluation,
                seed=args.seed,
                update=config.num_updates,
                global_step=config.total_timesteps,
                checkpoint_kind="final",
                eval_episodes=args.eval_episodes,
                eval_seed=args.eval_seed,
            ),
        }
        checkpoint_started = time.perf_counter()
        save_checkpoint(
            args.output,
            runner_state.train_state.params,
            config,
            summary,
        )
        final_checkpoint_seconds = time.perf_counter() - checkpoint_started
        elapsed_seconds = time.perf_counter() - process_started
        summary.update(
            {
                "elapsed_seconds": elapsed_seconds,
                "final_checkpoint_seconds": final_checkpoint_seconds,
                "end_to_end_steps_per_second": config.total_timesteps / elapsed_seconds,
            }
        )
        save_checkpoint_metadata(args.output, config, summary)
        run.summary.update(summary)
        logger.info(
            "Saved final checkpoint to %s and best checkpoint to %s",
            args.output,
            args.output / "best",
        )
        logger.info(
            "Finished in %.3fs | training %.3fs | evaluation %.3fs | %.0f train steps/s",
            elapsed_seconds,
            training_seconds,
            evaluation_seconds_total,
            config.total_timesteps / training_seconds,
        )


if __name__ == "__main__":
    main()
