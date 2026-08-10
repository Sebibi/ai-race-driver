"""Gymnax-compatible single-car racing environment."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from ai_race_driver.track.spline import (
    TrackData,
    make_oval_track,
    project_to_track,
    signed_progress_delta,
    track_frame_at_s,
)
from ai_race_driver.vehicle.point_mass import (
    PointMassModel,
    PointMassParams,
    PointMassState,
    wrap_angle,
)

PREVIEW_DISTANCES = jnp.asarray((0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0))
OBSERVATION_SIZE = 14


@struct.dataclass
class RacingEnvParams(environment.EnvParams):
    """Track, dynamics, reset, and reward parameters."""

    track: TrackData = struct.field(default_factory=make_oval_track)
    vehicle: PointMassParams = struct.field(default_factory=PointMassParams)
    max_steps_in_episode: int = 2_000
    randomize_reset: bool = True
    reset_lateral_fraction: float = 0.1
    reset_heading_radians: float = 0.15
    reset_speed_min_fraction: float = 0.15
    reset_speed_max_fraction: float = 0.35
    lateral_penalty: float = 0.10
    heading_penalty: float = 0.05
    action_change_penalty: float = 0.01
    off_track_penalty: float = 1.0
    lap_bonus: float = 1.0


@struct.dataclass
class RacingEnvState(environment.EnvState):
    """Complete Markov state for one racing environment."""

    vehicle: PointMassState
    centerline_s: jax.Array
    accumulated_progress: jax.Array
    lateral_error: jax.Array
    heading_error: jax.Array
    nearest_index: jax.Array
    previous_action: jax.Array
    episode_return: jax.Array
    terminated: jax.Array


class RacingEnv(environment.Environment[RacingEnvState, RacingEnvParams]):
    """Fast closed-track environment with a replaceable vehicle model boundary."""

    def __init__(self, params: RacingEnvParams | None = None):
        self._default_params = params if params is not None else RacingEnvParams()
        self.model = PointMassModel()

    @property
    def default_params(self) -> RacingEnvParams:
        return self._default_params

    @partial(jax.jit, static_argnames=("self",))
    def step_env(
        self,
        key: jax.Array,
        state: RacingEnvState,
        action: jax.Array,
        params: RacingEnvParams,
    ) -> tuple[jax.Array, RacingEnvState, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key
        action = jnp.clip(action, -1.0, 1.0)
        vehicle = self.model.step(state.vehicle, action, params.vehicle)
        frame, lateral_error = project_to_track(
            params.track,
            self.model.position(vehicle),
            state.nearest_index,
        )
        tangent_heading = jnp.arctan2(frame.tangent[1], frame.tangent[0])
        heading_error = wrap_angle(self.model.heading(vehicle) - tangent_heading)

        progress_delta = signed_progress_delta(frame.s, state.centerline_s, params.track.length)
        max_step_progress = params.vehicle.max_speed * params.vehicle.dt * 1.5
        progress_delta = jnp.clip(progress_delta, -max_step_progress, max_step_progress)
        accumulated_progress = state.accumulated_progress + progress_delta

        half_width = params.track.width * 0.5
        normalized_lateral = lateral_error / half_width
        normalized_heading = heading_error / jnp.pi
        normalized_progress = progress_delta / jnp.maximum(
            params.vehicle.max_speed * params.vehicle.dt, 1e-6
        )
        reward = (
            normalized_progress
            - params.lateral_penalty * jnp.square(normalized_lateral)
            - params.heading_penalty * jnp.square(normalized_heading)
            - params.action_change_penalty * jnp.sum(jnp.square(action - state.previous_action))
        )

        off_track = jnp.abs(lateral_error) > half_width
        lap_complete = accumulated_progress >= params.track.length
        time_limit = state.time + 1 >= params.max_steps_in_episode
        done = off_track | lap_complete | time_limit
        reward = reward - params.off_track_penalty * off_track + params.lap_bonus * lap_complete
        episode_return = state.episode_return + reward

        next_state = RacingEnvState(
            time=state.time + 1,
            vehicle=vehicle,
            centerline_s=frame.s,
            accumulated_progress=accumulated_progress,
            lateral_error=lateral_error,
            heading_error=heading_error,
            nearest_index=frame.index,
            previous_action=action,
            episode_return=episode_return,
            terminated=done,
        )
        info = {
            "progress": accumulated_progress,
            "progress_delta": progress_delta,
            "lateral_error": lateral_error,
            "heading_error": heading_error,
            "lap_complete": lap_complete,
            "off_track": off_track,
            "time_limit": time_limit,
            "returned_episode": done,
            "returned_episode_return": jnp.where(done, episode_return, 0.0),
            "returned_episode_length": jnp.where(done, state.time + 1, 0),
        }
        return self.get_obs(next_state, params), next_state, reward, done, info

    @partial(jax.jit, static_argnames=("self",))
    def reset_env(
        self, key: jax.Array, params: RacingEnvParams
    ) -> tuple[jax.Array, RacingEnvState]:
        key_s, key_lateral, key_heading, key_speed = jax.random.split(key, 4)
        sampled_s = jax.random.uniform(key_s, (), minval=0.0, maxval=params.track.length)
        sampled_lateral = jax.random.uniform(
            key_lateral,
            (),
            minval=-params.reset_lateral_fraction * params.track.width,
            maxval=params.reset_lateral_fraction * params.track.width,
        )
        sampled_heading = jax.random.uniform(
            key_heading,
            (),
            minval=-params.reset_heading_radians,
            maxval=params.reset_heading_radians,
        )
        sampled_speed = jax.random.uniform(
            key_speed,
            (),
            minval=params.reset_speed_min_fraction * params.vehicle.max_speed,
            maxval=params.reset_speed_max_fraction * params.vehicle.max_speed,
        )

        use_random = jnp.asarray(params.randomize_reset)
        centerline_s = jnp.where(use_random, sampled_s, 0.0)
        lateral_error = jnp.where(use_random, sampled_lateral, 0.0)
        heading_offset = jnp.where(use_random, sampled_heading, 0.0)
        speed = jnp.where(use_random, sampled_speed, 0.25 * params.vehicle.max_speed)

        frame = track_frame_at_s(params.track, centerline_s)
        tangent_heading = jnp.arctan2(frame.tangent[1], frame.tangent[0])
        vehicle = self.model.reset(
            frame.center + lateral_error * frame.normal,
            wrap_angle(tangent_heading + heading_offset),
            speed,
            params.vehicle,
        )
        state = RacingEnvState(
            time=jnp.asarray(0, dtype=jnp.int32),
            vehicle=vehicle,
            centerline_s=centerline_s,
            accumulated_progress=jnp.asarray(0.0),
            lateral_error=lateral_error,
            heading_error=heading_offset,
            nearest_index=frame.index,
            previous_action=jnp.zeros((2,), dtype=jnp.float32),
            episode_return=jnp.asarray(0.0),
            terminated=jnp.asarray(False),
        )
        return self.get_obs(state, params), state

    def get_obs(
        self,
        state: RacingEnvState,
        params: RacingEnvParams | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del key
        if params is None:
            params = self.default_params
        preview_curvature = jax.vmap(
            lambda distance: track_frame_at_s(params.track, state.centerline_s + distance).curvature
        )(PREVIEW_DISTANCES)
        half_width = params.track.width * 0.5
        return jnp.concatenate(
            (
                jnp.asarray(
                    (
                        self.model.speed(state.vehicle) / params.vehicle.max_speed,
                        state.lateral_error / half_width,
                        jnp.sin(state.heading_error),
                        jnp.cos(state.heading_error),
                    )
                ),
                state.previous_action,
                jnp.clip(preview_curvature * half_width, -1.0, 1.0),
            )
        ).astype(jnp.float32)

    def is_terminal(self, state: RacingEnvState, params: RacingEnvParams) -> jax.Array:
        del params
        return state.terminated

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: RacingEnvParams) -> spaces.Box:
        del params
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: RacingEnvParams) -> spaces.Box:
        del params
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=jnp.float32,
        )

    def state_space(self, params: RacingEnvParams) -> spaces.Dict:
        del params
        return spaces.Dict(
            {
                "position": spaces.Box(-jnp.inf, jnp.inf, (2,), dtype=jnp.float32),
                "heading": spaces.Box(-jnp.pi, jnp.pi, (), dtype=jnp.float32),
                "speed": spaces.Box(0.0, jnp.inf, (), dtype=jnp.float32),
            }
        )

    @property
    def name(self) -> str:
        return "JaxPointMassRacing-v0"


def make_default_env(*, randomize_reset: bool = True) -> tuple[RacingEnv, RacingEnvParams]:
    """Construct the built-in oval environment."""

    params = RacingEnvParams(randomize_reset=randomize_reset)
    return RacingEnv(params), params
