"""Train PPO on the built-in oval with live evaluation and progress tracking."""

import argparse
import logging
import math
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
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
    make_record_policy,
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


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
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
    parser.add_argument(
        "--video-every-evals",
        type=_nonnegative_int,
        default=0,
        help="record fixed-start video every N evaluations; 0 disables video",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest"))
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    return parser


def should_record_video(
    evaluation_index: int,
    *,
    is_final: bool,
    cadence: int,
) -> bool:
    """Select initial, periodic, and final evaluation video boundaries."""

    if evaluation_index < 0 or cadence < 0:
        raise ValueError("evaluation_index and cadence must be non-negative")
    return cadence > 0 and (evaluation_index == 0 or evaluation_index % cadence == 0 or is_final)


@dataclass
class _VideoTotals:
    compilation_seconds: float = 0.0
    capture_seconds: float = 0.0
    render_seconds: float = 0.0
    wait_seconds: float = 0.0
    logging_seconds: float = 0.0
    completed: int = 0
    failures: int = 0


def _make_video_request(
    compiled_record: Any,
    policy_params: Any,
    evaluation_key: jax.Array,
    *,
    env_params: Any,
    track_geometry: Any,
    output: Path,
    update: int,
    global_step: int,
) -> Any:
    from ai_race_driver.visualization.video import VideoRenderRequest, trajectory_to_telemetry

    capture_started = time.perf_counter()
    trajectory = compiled_record(policy_params, evaluation_key)
    jax.block_until_ready(trajectory.episode_length)
    host_trajectory = jax.device_get(trajectory)
    telemetry = trajectory_to_telemetry(host_trajectory, dt=env_params.vehicle.dt)
    capture_seconds = time.perf_counter() - capture_started
    return VideoRenderRequest(
        telemetry=telemetry,
        track=track_geometry,
        output_path=output / "videos" / f"eval-step-{global_step:012d}.mp4",
        global_step=global_step,
        update=update,
        capture_seconds=capture_seconds,
    )


def _collect_completed_video(
    run: Any,
    renderer: Any,
    totals: _VideoTotals,
    *,
    wait: bool,
) -> bool:
    """Collect and log a completed background render; return false after a failure."""

    collect_started = time.perf_counter()
    try:
        result = renderer.collect(wait=wait)
    except Exception:
        if wait:
            totals.wait_seconds += time.perf_counter() - collect_started
        totals.failures += 1
        logger.exception("Evaluation video rendering failed; disabling subsequent videos")
        return False
    if wait:
        totals.wait_seconds += time.perf_counter() - collect_started
    if result is None:
        return True

    totals.render_seconds += result.render_seconds
    totals.completed += 1
    logging_started = time.perf_counter()
    try:
        run.log(
            {
                "global_step": result.global_step,
                "update": result.update,
                "eval/fixed/video": wandb.Video(
                    str(result.path),
                    caption=f"fixed-start policy at step {result.global_step:,}",
                    format="mp4",
                ),
                "performance/video_capture_seconds": result.capture_seconds,
                "performance/video_render_seconds": result.render_seconds,
                "performance/video_wait_seconds": (
                    time.perf_counter() - collect_started if wait else 0.0
                ),
            }
        )
    except Exception:
        totals.failures += 1
        logger.exception("Evaluation video logging failed; disabling subsequent videos")
        return False
    finally:
        totals.logging_seconds += time.perf_counter() - logging_started
    logger.info(
        "Saved evaluation video to %s (%d frames, %.3fs render)",
        result.path,
        result.frame_count,
        result.render_seconds,
    )
    return True


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
    video_module: Any | None = None
    if args.video_every_evals > 0:
        from ai_race_driver.visualization import video as imported_video_module

        imported_video_module.validate_video_backend()
        video_module = imported_video_module
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
        "video_every_evals": args.video_every_evals,
        **asdict(config),
        **git_metadata,
    }

    logger.info("PPO config: %s", config)
    logger.info("Live logging and evaluation every %d PPO update(s)", args.log_every_updates)
    if args.video_every_evals > 0:
        logger.info(
            "Fixed-start evaluation videos enabled every %d evaluation(s)",
            args.video_every_evals,
        )
    process_started = time.perf_counter()
    with wandb.init(config=wandb_config) as run, ExitStack() as exit_stack:
        configure_wandb_metrics(run)
        video_totals = _VideoTotals()
        compiled_record: Any | None = None
        video_renderer: Any | None = None
        track_geometry: Any | None = None
        video_enabled = video_module is not None

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

        if video_module is not None:
            video_compilation_started = time.perf_counter()
            record_policy = make_record_policy(env, env_params, model)
            compiled_record = (
                jax.jit(record_policy)
                .lower(runner_state.train_state.params, evaluation_key)
                .compile()
            )
            video_totals.compilation_seconds = (
                time.perf_counter() - video_compilation_started
            )
            track_geometry = video_module.make_track_geometry(jax.device_get(env_params.track))
            video_renderer = exit_stack.enter_context(video_module.AsyncVideoRenderer())
            logger.info(
                "Compiled fixed-start video trajectory capture in %.3f seconds",
                video_totals.compilation_seconds,
            )

        def schedule_video(update: int, global_step: int) -> bool:
            if compiled_record is None or video_renderer is None or track_geometry is None:
                return False
            try:
                request = _make_video_request(
                    compiled_record,
                    runner_state.train_state.params,
                    evaluation_key,
                    env_params=env_params,
                    track_geometry=track_geometry,
                    output=args.output,
                    update=update,
                    global_step=global_step,
                )
                video_totals.capture_seconds += request.capture_seconds
                video_renderer.submit(request)
            except Exception:
                video_totals.failures += 1
                logger.exception("Evaluation video capture failed; disabling subsequent videos")
                return False
            logger.info("Scheduled evaluation video for step %s", f"{global_step:,}")
            return True

        compiled_chunks: dict[int, Any] = {}
        compilation_seconds = initialization_compilation_seconds + evaluation_compilation_seconds
        primary_chunk_updates = min(args.log_every_updates, config.num_updates)
        chunk_sizes = {primary_chunk_updates}
        remainder_updates = config.num_updates % args.log_every_updates
        if remainder_updates:
            chunk_sizes.add(remainder_updates)
        for chunk_updates in sorted(chunk_sizes, reverse=True):
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
                "performance/compilation_seconds": compilation_seconds,
            }
        )
        if video_enabled:
            initial_record["performance/video_compilation_seconds"] = (
                video_totals.compilation_seconds
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
        if should_record_video(
            0,
            is_final=config.num_updates == 0,
            cadence=args.video_every_evals,
        ):
            video_enabled = schedule_video(0, 0)

        training_seconds = 0.0
        evaluation_seconds_total = evaluation_seconds
        final_metrics: UpdateMetrics | None = None

        completed_updates = 0
        evaluation_index = 0
        while completed_updates < config.num_updates:
            chunk_updates = min(
                args.log_every_updates,
                config.num_updates - completed_updates,
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
            evaluation_index += 1
            is_final_evaluation = completed_updates == config.num_updates

            if video_renderer is not None and video_enabled:
                video_enabled = _collect_completed_video(
                    run,
                    video_renderer,
                    video_totals,
                    wait=False,
                )
            if video_enabled and should_record_video(
                evaluation_index,
                is_final=is_final_evaluation,
                cadence=args.video_every_evals,
            ):
                if video_renderer is not None and video_renderer.pending:
                    video_enabled = _collect_completed_video(
                        run,
                        video_renderer,
                        video_totals,
                        wait=True,
                    )
                if video_enabled:
                    video_enabled = schedule_video(completed_updates, global_step)

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

        if video_renderer is not None and video_enabled:
            video_enabled = _collect_completed_video(
                run,
                video_renderer,
                video_totals,
                wait=True,
            )

        summary = {
            "seed": args.seed,
            "device": str(devices[0]),
            "compilation_seconds": compilation_seconds,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds_total,
            "video_compilation_seconds": video_totals.compilation_seconds,
            "video_capture_seconds": video_totals.capture_seconds,
            "video_render_seconds": video_totals.render_seconds,
            "video_wait_seconds": video_totals.wait_seconds,
            "video_logging_seconds": video_totals.logging_seconds,
            "videos_completed": video_totals.completed,
            "video_failures": video_totals.failures,
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
            "Finished in %.3fs | training %.3fs | evaluation %.3fs | video %.3fs render "
            "(%d completed, %d failed) | %.0f train steps/s",
            elapsed_seconds,
            training_seconds,
            evaluation_seconds_total,
            video_totals.render_seconds,
            video_totals.completed,
            video_totals.failures,
            config.total_timesteps / training_seconds,
        )


if __name__ == "__main__":
    main()
