import jax
import jax.numpy as jnp

from ai_race_driver.envs.racing import OBSERVATION_SIZE, RacingEnv, RacingEnvParams
from ai_race_driver.track.spline import make_oval_track, track_frame_at_s


def test_reset_and_step_are_jittable_and_vectorizable() -> None:
    params = RacingEnvParams(track=make_oval_track(samples=128), max_steps_in_episode=20)
    env = RacingEnv(params)
    keys = jax.random.split(jax.random.key(0), 8)
    observations, states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
    assert observations.shape == (8, OBSERVATION_SIZE)
    assert bool(jnp.all(jnp.isfinite(observations)))

    actions = jnp.zeros((8, 2), dtype=jnp.float32)
    observations, states, rewards, dones, _ = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))(
        keys, states, actions, params
    )
    assert observations.shape == (8, OBSERVATION_SIZE)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(rewards)))


def test_fixed_evaluation_reset_starts_on_centerline() -> None:
    params = RacingEnvParams(track=make_oval_track(samples=128), randomize_reset=False)
    env = RacingEnv(params)
    _, state = env.reset(jax.random.key(0), params)
    assert float(state.centerline_s) == 0.0
    assert float(state.lateral_error) == 0.0
    assert float(state.accumulated_progress) == 0.0


def test_off_track_and_time_limit_terminate() -> None:
    track = make_oval_track(width=4.0, samples=128)
    params = RacingEnvParams(track=track, randomize_reset=False, max_steps_in_episode=1)
    env = RacingEnv(params)
    _, state = env.reset(jax.random.key(0), params)
    _, _, _, done, info = env.step_env(jax.random.key(1), state, jnp.zeros(2), params)
    assert bool(done)
    assert bool(info["time_limit"])

    frame = track_frame_at_s(track, jnp.asarray(0.0))
    off_track_vehicle = state.vehicle.replace(position=frame.center + 3.0 * frame.normal)
    off_track_state = state.replace(vehicle=off_track_vehicle)
    _, _, _, done, info = env.step_env(
        jax.random.key(2), off_track_state, jnp.zeros(2), params.replace(max_steps_in_episode=20)
    )
    assert bool(done)
    assert bool(info["off_track"])
