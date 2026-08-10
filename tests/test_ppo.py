import os

import jax
import jax.numpy as jnp
import pytest

from ai_race_driver.envs.racing import RacingEnv, RacingEnvParams
from ai_race_driver.track.spline import make_oval_track
from ai_race_driver.training.ppo import (
    ActorCritic,
    PPOConfig,
    deterministic_action,
    make_train,
)


def test_ppo_smoke_updates_parameters_with_finite_metrics() -> None:
    params = RacingEnvParams(track=make_oval_track(samples=64), max_steps_in_episode=32)
    env = RacingEnv(params)
    config = PPOConfig.cpu_smoke()
    train = jax.jit(make_train(env, params, config))

    output = train(jax.random.key(0))
    jax.block_until_ready(output.metrics.loss)

    assert output.metrics.loss.shape == (config.num_updates,)
    assert bool(jnp.all(jnp.isfinite(output.metrics.loss)))
    assert bool(jnp.all(jnp.isfinite(output.metrics.mean_reward)))
    assert output.final_observation.shape == (config.num_envs, 14)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("AI_RACE_RUN_SLOW") != "1",
    reason="set AI_RACE_RUN_SLOW=1 to run learning acceptance",
)
def test_ppo_learns_reproducibly_on_oval() -> None:
    params = RacingEnvParams(track=make_oval_track(samples=256), max_steps_in_episode=1_000)
    env = RacingEnv(params)
    config = PPOConfig(
        total_timesteps=1_048_576,
        num_envs=256,
        num_steps=128,
        hidden_size=128,
    )
    model = ActorCritic(action_dim=2, hidden_size=config.hidden_size)
    evaluation_params = params.replace(randomize_reset=False)
    step = jax.jit(env.step)
    for seed in (0, 1, 2):
        output = jax.jit(make_train(env, params, config))(jax.random.key(seed))
        jax.block_until_ready(output.metrics.loss)
        assert bool(jnp.all(jnp.isfinite(output.metrics.loss)))

        observation, state = env.reset(jax.random.key(10_000 + seed), evaluation_params)
        policy = jax.jit(lambda obs: deterministic_action(model, output.train_state.params, obs))
        key = jax.random.key(20_000 + seed)
        completed_lap = False
        went_off_track = False
        for _ in range(evaluation_params.max_steps_in_episode):
            key, step_key = jax.random.split(key)
            observation, state, _, done, info = step(
                step_key, state, policy(observation), evaluation_params
            )
            if bool(done):
                completed_lap = bool(info["lap_complete"])
                went_off_track = bool(info["off_track"])
                break
        assert completed_lap
        assert not went_off_track
