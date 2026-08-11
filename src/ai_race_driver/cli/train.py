"""Train PPO on the built-in oval."""

import argparse
import logging
import time
from pathlib import Path

import jax

from ai_race_driver.envs.racing import make_default_env
from ai_race_driver.logging import LOG_LEVELS, configure_logging
from ai_race_driver.training.ppo import PPOConfig, make_train, save_checkpoint

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=8_388_608)
    parser.add_argument("--num-envs", type=int, default=2_048)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest"))
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    return parser


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
    compiled_train = jax.jit(make_train(env, env_params, config))
    logger.info("PPO training function compiled successfully.")

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
    for name, value in summary.items():
        logger.info("%s: %s", name, value)


if __name__ == "__main__":
    main()
