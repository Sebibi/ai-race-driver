"""Measure steady-state JAX environment and PPO throughput."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from ai_race_driver.envs.racing import make_default_env
from ai_race_driver.training.ppo import PPOConfig, make_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2_048)
    parser.add_argument("--num-steps", type=int, default=1_000)
    parser.add_argument("--ppo", action="store_true", help="also benchmark a small PPO run")
    parser.add_argument("--output", type=Path)
    return parser


def _time_call(function, *args):
    started = time.perf_counter()
    result = function(*args)
    jax.block_until_ready(result)
    return result, time.perf_counter() - started


def main() -> None:
    args = build_parser().parse_args()
    env, env_params = make_default_env(randomize_reset=True)
    vector_reset = jax.vmap(env.reset, in_axes=(0, None))
    vector_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def rollout(key):
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, args.num_envs)
        observation, state = vector_reset(reset_keys, env_params)
        del observation

        def body(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, args.num_envs)
            actions = jnp.zeros((args.num_envs, 2), dtype=jnp.float32)
            _, state, reward, _, _ = vector_step(step_keys, state, actions, env_params)
            return (state, key), reward

        (_, _), rewards = jax.lax.scan(body, (state, key), None, length=args.num_steps)
        return rewards

    compiled_rollout = jax.jit(rollout)
    key = jax.random.key(0)
    _, compile_seconds = _time_call(compiled_rollout, key)
    _, steady_seconds = _time_call(compiled_rollout, key)
    environment_steps = args.num_envs * args.num_steps
    results: dict[str, object] = {
        "device": str(jax.devices()[0]),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "environment_compile_seconds": compile_seconds,
        "environment_steady_seconds": steady_seconds,
        "environment_steps_per_second": environment_steps / steady_seconds,
    }

    if args.ppo:
        ppo_config = PPOConfig(
            total_timesteps=args.num_envs * 16 * 2,
            num_envs=args.num_envs,
            num_steps=16,
            num_minibatches=4,
            update_epochs=2,
        )
        compiled_train = jax.jit(make_train(env, env_params, ppo_config))
        output, ppo_compile_seconds = _time_call(compiled_train, key)
        del output
        output, ppo_steady_seconds = _time_call(compiled_train, key)
        del output
        results.update(
            {
                "ppo_compile_seconds": ppo_compile_seconds,
                "ppo_steady_seconds": ppo_steady_seconds,
                "ppo_steps_per_second": ppo_config.total_timesteps / ppo_steady_seconds,
            }
        )

    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
