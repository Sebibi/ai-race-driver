"""Train PPO on the built-in oval."""

import argparse
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import jax
import wandb

from ai_race_driver.envs.racing import make_default_env
from ai_race_driver.logging import LOG_LEVELS, configure_logging
from ai_race_driver.training.ppo import PPOConfig, make_train, save_checkpoint

logger = logging.getLogger(__name__)

WandbMode = Literal["online", "offline", "disabled"]
WANDB_MODES: tuple[WandbMode, ...] = ("online", "offline", "disabled")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=8_388_608)
    parser.add_argument("--num-envs", type=int, default=2_048)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest"))
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    parser.add_argument("--wandb-project", help="log this training run to a W&B project")
    parser.add_argument("--wandb-entity", help="W&B team or user owning the project")
    parser.add_argument("--wandb-name", help="optional W&B run name")
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default="online")
    return parser


def log_wandb_run(
    *,
    project: str,
    entity: str | None,
    name: str | None,
    mode: WandbMode,
    config: PPOConfig,
    seed: int,
    metrics: Any,
    summary: dict[str, Any],
) -> None:
    """Log completed device-side training metrics from the host."""

    host_metrics = jax.device_get(metrics)
    with wandb.init(
        project=project,
        entity=entity,
        name=name,
        mode=mode,
        config={"seed": seed, **asdict(config)},
    ) as run:
        for update_index in range(config.num_updates):
            run.log(
                {
                    "train/timesteps": (update_index + 1) * config.batch_size,
                    **{
                        f"train/{metric_name}": float(metric_values[update_index])
                        for metric_name, metric_values in host_metrics._asdict().items()
                    },
                },
                step=update_index + 1,
            )
        run.summary.update(summary)


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level, entrypoint_logger=logger)
    logger.info("JAX devices: %s", jax.devices())
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
    )
    env, env_params = make_default_env(randomize_reset=True)
    logger.info("Compiling PPO training function with config: %s", config)
    compile_start = time.perf_counter()
    compiled_train = jax.jit(make_train(env, env_params, config))
    logger.info("Compilation took %.3f seconds", time.perf_counter() - compile_start)

    logger.info("Starting PPO training on %s", jax.devices()[0])
    started = time.perf_counter()
    output = compiled_train(jax.random.key(args.seed))
    jax.block_until_ready(output.metrics.loss)
    elapsed = time.perf_counter() - started
    final_metrics = jax.tree.map(lambda value: float(value[-1]), output.metrics)
    summary = {
        "seed": args.seed,
        "device": str(jax.devices()[0]),
        "elapsed_seconds_including_compile": elapsed,
        "steps_per_second_including_compile": config.total_timesteps / elapsed,
        "final_loss": final_metrics.loss,
        "final_mean_reward": final_metrics.mean_reward,
        "final_completed_laps": final_metrics.completed_laps,
        "final_completed_episodes": final_metrics.completed_episodes,
        "final_mean_completed_return": final_metrics.mean_completed_return,
    }
    save_checkpoint(args.output, output.train_state.params, config, summary)
    logger.info("Saved checkpoint to %s", args.output)
    if args.wandb_project:
        log_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=config,
            seed=args.seed,
            metrics=output.metrics,
            summary=summary,
        )
        logger.info("Logged training metrics to W&B project %s", args.wandb_project)
    for name, value in summary.items():
        logger.info("%s: %s", name, value)


if __name__ == "__main__":
    main()
