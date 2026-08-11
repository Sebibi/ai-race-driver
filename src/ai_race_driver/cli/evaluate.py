"""Evaluate a saved PPO policy deterministically."""

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp

from ai_race_driver.envs.racing import make_default_env
from ai_race_driver.logging import LOG_LEVELS, configure_logging
from ai_race_driver.training.ppo import ActorCritic, deterministic_action, load_checkpoint

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    logger.info("Loading checkpoint from %s", args.checkpoint)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    hidden_size = int(metadata["ppo"]["hidden_size"])
    model = ActorCritic(action_dim=2, hidden_size=hidden_size)
    env, env_params = make_default_env(randomize_reset=False)
    observation, state = env.reset(jax.random.key(args.seed), env_params)
    template = model.init(jax.random.key(args.seed + 1), observation)
    params = load_checkpoint(args.checkpoint, template)
    policy = jax.jit(lambda obs: deterministic_action(model, params, obs))
    step = jax.jit(env.step)

    key = jax.random.key(args.seed + 2)
    completed = 0
    returns: list[float] = []
    while completed < args.episodes:
        key, step_key = jax.random.split(key)
        action = policy(observation)
        observation, state, _, _, info = step(step_key, state, action, env_params)
        if bool(info["returned_episode"]):
            completed += 1
            returns.append(float(info["returned_episode_return"]))
            logger.info(
                "episode=%d return=%.3f lap=%s off_track=%s",
                completed,
                returns[-1],
                bool(info["lap_complete"]),
                bool(info["off_track"]),
            )
    logger.info("mean_return: %.3f", float(jnp.mean(jnp.asarray(returns))))


if __name__ == "__main__":
    main()
