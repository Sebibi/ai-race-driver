"""Compiled deterministic policy evaluation for racing PPO."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ai_race_driver.envs.racing import RacingEnv, RacingEnvParams
from ai_race_driver.training.ppo import ActorCritic, deterministic_action


class EvaluationMetrics(NamedTuple):
    """Aggregate metrics for a fixed-size evaluation episode batch."""

    return_mean: jax.Array
    return_std: jax.Array
    return_min: jax.Array
    return_max: jax.Array
    length_mean: jax.Array
    progress_mean: jax.Array
    lap_success_rate: jax.Array
    off_track_rate: jax.Array
    time_limit_rate: jax.Array


class EvaluationSuiteMetrics(NamedTuple):
    """Canonical fixed-start and randomized-start policy performance."""

    fixed: EvaluationMetrics
    randomized: EvaluationMetrics


class EvaluationTrajectory(NamedTuple):
    """Fixed-shape deterministic episode data for host-side visualization."""

    position: jax.Array
    heading: jax.Array
    speed: jax.Array
    action: jax.Array
    reward: jax.Array
    accumulated_progress: jax.Array
    lateral_error: jax.Array
    heading_error: jax.Array
    valid: jax.Array
    done: jax.Array
    lap_complete: jax.Array
    off_track: jax.Array
    time_limit: jax.Array
    episode_length: jax.Array
    episode_return: jax.Array


class _EvaluationState(NamedTuple):
    observation: jax.Array
    environment_state: Any
    active: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    progress: jax.Array
    lap_complete: jax.Array
    off_track: jax.Array
    time_limit: jax.Array
    key: jax.Array


class _RecordingState(NamedTuple):
    observation: jax.Array
    environment_state: Any
    active: jax.Array
    key: jax.Array


class _RecordingStep(NamedTuple):
    position: jax.Array
    heading: jax.Array
    speed: jax.Array
    action: jax.Array
    reward: jax.Array
    accumulated_progress: jax.Array
    lateral_error: jax.Array
    heading_error: jax.Array
    valid: jax.Array
    done: jax.Array
    lap_complete: jax.Array
    off_track: jax.Array
    time_limit: jax.Array


def _aggregate(state: _EvaluationState) -> EvaluationMetrics:
    returns = state.episode_return
    lengths = state.episode_length.astype(jnp.float32)
    return EvaluationMetrics(
        return_mean=returns.mean(),
        return_std=returns.std(),
        return_min=returns.min(),
        return_max=returns.max(),
        length_mean=lengths.mean(),
        progress_mean=state.progress.mean(),
        lap_success_rate=state.lap_complete.astype(jnp.float32).mean(),
        off_track_rate=state.off_track.astype(jnp.float32).mean(),
        time_limit_rate=state.time_limit.astype(jnp.float32).mean(),
    )


def make_evaluate_policy(
    env: RacingEnv,
    env_params: RacingEnvParams,
    model: ActorCritic,
    *,
    randomized_episodes: int,
):
    """Build a pure fixed/randomized deterministic evaluation suite."""

    if randomized_episodes < 1:
        raise ValueError("randomized_episodes must be positive")

    fixed_params = env_params.replace(randomize_reset=False)
    randomized_params = env_params.replace(randomize_reset=True)

    def evaluate_batch(
        params: Any,
        key: jax.Array,
        batch_size: int,
        evaluation_params: RacingEnvParams,
    ) -> EvaluationMetrics:
        reset_key, rollout_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, batch_size)
        observations, environment_state = jax.vmap(env.reset, in_axes=(0, None))(
            reset_keys, evaluation_params
        )
        initial_state = _EvaluationState(
            observation=observations,
            environment_state=environment_state,
            active=jnp.ones((batch_size,), dtype=jnp.bool_),
            episode_return=jnp.zeros((batch_size,), dtype=jnp.float32),
            episode_length=jnp.zeros((batch_size,), dtype=jnp.int32),
            progress=jnp.zeros((batch_size,), dtype=jnp.float32),
            lap_complete=jnp.zeros((batch_size,), dtype=jnp.bool_),
            off_track=jnp.zeros((batch_size,), dtype=jnp.bool_),
            time_limit=jnp.zeros((batch_size,), dtype=jnp.bool_),
            key=rollout_key,
        )

        def evaluation_step(state: _EvaluationState, _: None):
            key, step_key = jax.random.split(state.key)
            step_keys = jax.random.split(step_key, batch_size)
            actions = deterministic_action(model, params, state.observation)
            observation, environment_state, _, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_keys, state.environment_state, actions, evaluation_params)
            finished = state.active & done
            next_state = _EvaluationState(
                observation=observation,
                environment_state=environment_state,
                active=state.active & ~done,
                episode_return=jnp.where(
                    finished, info["returned_episode_return"], state.episode_return
                ),
                episode_length=jnp.where(
                    finished, info["returned_episode_length"], state.episode_length
                ),
                progress=jnp.where(finished, info["progress"], state.progress),
                lap_complete=jnp.where(finished, info["lap_complete"], state.lap_complete),
                off_track=jnp.where(finished, info["off_track"], state.off_track),
                time_limit=jnp.where(finished, info["time_limit"], state.time_limit),
                key=key,
            )
            return next_state, None

        final_state, _ = jax.lax.scan(
            evaluation_step,
            initial_state,
            None,
            length=evaluation_params.max_steps_in_episode,
        )
        return _aggregate(final_state)

    def evaluate(params: Any, key: jax.Array) -> EvaluationSuiteMetrics:
        fixed_key, randomized_key = jax.random.split(key)
        return EvaluationSuiteMetrics(
            fixed=evaluate_batch(params, fixed_key, 1, fixed_params),
            randomized=evaluate_batch(
                params,
                randomized_key,
                randomized_episodes,
                randomized_params,
            ),
        )

    return evaluate


def make_record_policy(
    env: RacingEnv,
    env_params: RacingEnvParams,
    model: ActorCritic,
):
    """Build a pure fixed-start deterministic episode recorder.

    This is intentionally separate from aggregate evaluation so trajectory arrays are
    produced only at video boundaries. The scan uses ``step_env`` to retain the terminal
    vehicle state instead of Gymnax's automatic-reset state.
    """

    evaluation_params = env_params.replace(randomize_reset=False)

    def record(params: Any, key: jax.Array) -> EvaluationTrajectory:
        reset_key, rollout_key = jax.random.split(key)
        observation, environment_state = env.reset(reset_key, evaluation_params)
        initial_environment_state = environment_state
        initial_state = _RecordingState(
            observation=observation,
            environment_state=environment_state,
            active=jnp.asarray(True),
            key=rollout_key,
        )

        def recording_step(state: _RecordingState, _: None):
            key, step_key = jax.random.split(state.key)
            action = deterministic_action(model, params, state.observation)
            next_observation, next_environment_state, reward, done, info = env.step_env(
                step_key,
                state.environment_state,
                action,
                evaluation_params,
            )
            valid = state.active
            environment_state = jax.tree.map(
                lambda previous, candidate: jnp.where(valid, candidate, previous),
                state.environment_state,
                next_environment_state,
            )
            observation = jnp.where(valid, next_observation, state.observation)
            terminal = valid & done
            output = _RecordingStep(
                position=environment_state.vehicle.position,
                heading=environment_state.vehicle.heading,
                speed=environment_state.vehicle.speed,
                action=jnp.where(valid, action, jnp.zeros_like(action)),
                reward=jnp.where(valid, reward, 0.0),
                accumulated_progress=environment_state.accumulated_progress,
                lateral_error=environment_state.lateral_error,
                heading_error=environment_state.heading_error,
                valid=valid,
                done=terminal,
                lap_complete=terminal & info["lap_complete"],
                off_track=terminal & info["off_track"],
                time_limit=terminal & info["time_limit"],
            )
            return _RecordingState(
                observation=observation,
                environment_state=environment_state,
                active=valid & ~done,
                key=key,
            ), output

        _, steps = jax.lax.scan(
            recording_step,
            initial_state,
            None,
            length=evaluation_params.max_steps_in_episode,
        )
        return EvaluationTrajectory(
            position=jnp.concatenate(
                (initial_environment_state.vehicle.position[None, :], steps.position),
                axis=0,
            ),
            heading=jnp.concatenate(
                (initial_environment_state.vehicle.heading[None], steps.heading),
                axis=0,
            ),
            speed=jnp.concatenate(
                (initial_environment_state.vehicle.speed[None], steps.speed),
                axis=0,
            ),
            action=steps.action,
            reward=steps.reward,
            accumulated_progress=jnp.concatenate(
                (
                    initial_environment_state.accumulated_progress[None],
                    steps.accumulated_progress,
                ),
                axis=0,
            ),
            lateral_error=jnp.concatenate(
                (initial_environment_state.lateral_error[None], steps.lateral_error),
                axis=0,
            ),
            heading_error=jnp.concatenate(
                (initial_environment_state.heading_error[None], steps.heading_error),
                axis=0,
            ),
            valid=steps.valid,
            done=steps.done,
            lap_complete=steps.lap_complete,
            off_track=steps.off_track,
            time_limit=steps.time_limit,
            episode_length=steps.valid.astype(jnp.int32).sum(),
            episode_return=steps.reward.sum(),
        )

    return record
