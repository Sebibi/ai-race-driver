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

# Track geometry lookahead distances (meters) for preview curvature observations.
# Enables predictive steering by sampling curvature at multiple future points.
# Typical vehicle speed: 5 m/s → 0m @ 0.0s, 2m @ 0.4s, ..., 40m @ 8s (long-horizon planning).
PREVIEW_DISTANCES = jnp.asarray((0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0))

# Observation vector size: 4 (ego state) + 2 (action history) + 8 (curvature preview) = 14.
OBSERVATION_SIZE = 14


@struct.dataclass
class RacingEnvParams(environment.EnvParams):
    """Track, dynamics, reset, and reward parameters.

    Attributes:
        track: Compiled track geometry (periodic cubic spline with constant width).
        vehicle: Point-mass vehicle parameters (mass, max_speed, dt, etc).
        max_steps_in_episode: Episode termination limit (time horizon).
        randomize_reset: If True, randomize initial position/heading/speed; else fixed-start.
        reset_lateral_fraction: Fraction of track width for initial lateral offset range.
        reset_heading_radians: Radians for initial heading offset range.
        reset_speed_min_fraction: Min initial speed as fraction of max_speed.
        reset_speed_max_fraction: Max initial speed as fraction of max_speed.

        Reward parameters:
        lateral_penalty: Weight for (lateral_error / half_width)² penalty. Encourages
            centerline adherence. Default 0.0 (disabled); typical values 0.01-0.1.
        heading_penalty: Weight for (heading_error / π)² penalty. Encourages alignment
            with track tangent. Default 0.0 (disabled); typical values 0.01-0.1.
        action_change_penalty: Weight for Σ(action - previous_action)² penalty.
            Encourages smooth control. Default 0.0 (disabled); typical values 0.001-0.01.
        off_track_penalty: Penalty applied when |lateral_error| > half_width (crash).
            Default 1.0 (enabled); disables agent to learn speed control.
        lap_bonus: Bonus when accumulated_progress >= track.length. Default 1.0
            (enabled); provides goal signal.
    """

    # Track and dynamics
    track: TrackData = struct.field(default_factory=make_oval_track)
    vehicle: PointMassParams = struct.field(default_factory=PointMassParams)
    max_steps_in_episode: int = 2_000
    randomize_reset: bool = True

    # Reset behavior
    reset_lateral_fraction: float = 0.1
    reset_heading_radians: float = 0.15
    reset_speed_min_fraction: float = 0.15
    reset_speed_max_fraction: float = 0.35

    # Reward shaping (all configurable; defaults are conservative)
    # Primary reward: progress delta (always enabled, not configurable)
    lateral_penalty: float = 0.0  # Penalty for centerline deviation
    heading_penalty: float = 0.0  # Penalty for heading misalignment
    action_change_penalty: float = 0.0  # Penalty for abrupt control changes
    off_track_penalty: float = 1.0  # Penalty for crash (hard constraint)
    lap_bonus: float = 1.0  # Bonus for lap completion (goal signal)


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
        """Advance environment by one timestep and compute multi-component shaped reward.

        Reward structure (see REWARD_SPECIFICATION.md for details):
            R_total = R_progress + R_lateral + R_heading + R_action_change + R_terminal

        Where:
        - R_progress: Normalized forward distance along centerline (primary, always enabled)
        - R_lateral: -lateral_penalty * (lateral_error / half_width)² (optional shaping)
        - R_heading: -heading_penalty * (heading_error / π)² (optional shaping)
        - R_action_change: -action_change_penalty * Σ(Δaction)² (optional smoothness)
        - R_terminal: -off_track_penalty*[off-track] + lap_bonus*[lap-complete]

        Returns:
            observation: Current 14-element observation (speed, errors, action history, preview)
            next_state: Updated environment state
            reward: Scalar reward signal for this step
            done: Boolean termination flag
            info: Dictionary of diagnostic information (progress, errors, event flags, etc.)
        """
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

        # Normalize error signals for reward shaping
        normalized_lateral = lateral_error / half_width  # Range: [-∞, ∞] but [-1, 1] on-track
        normalized_heading = heading_error / jnp.pi  # Range: [-1, 1] by construction
        normalized_progress = progress_delta / jnp.maximum(
            params.vehicle.max_speed * params.vehicle.dt, 1e-6
        )  # Dimensionless; ~1.0 for one timestep at max speed

        # Compute continuous reward components
        # R_progress: Primary objective—forward progress along centerline (dense signal every step)
        reward = normalized_progress

        # R_lateral: Quadratic penalty for lateral deviation from centerline
        #   Disabled by default (lateral_penalty=0.0); enables with weight > 0
        reward = reward - params.lateral_penalty * jnp.square(normalized_lateral)

        # R_heading: Quadratic penalty for heading misalignment with track tangent
        #   Disabled by default (heading_penalty=0.0); enables with weight > 0
        reward = reward - params.heading_penalty * jnp.square(normalized_heading)

        # R_action_change: Quadratic penalty for abrupt control changes
        #   Encourages smooth acceleration/steering; disabled by default (action_change_penalty=0.0)
        reward = reward - params.action_change_penalty * jnp.sum(
            jnp.square(action - state.previous_action)
        )

        # Terminal event detection
        off_track = jnp.abs(lateral_error) > half_width
        lap_complete = accumulated_progress >= params.track.length
        time_limit = state.time + 1 >= params.max_steps_in_episode
        done = off_track | lap_complete | time_limit

        # R_off_track: Hard constraint—immediate large penalty for crashing
        #   Applied only on termination (off_track=True); default -1.0
        reward = reward - params.off_track_penalty * off_track

        # R_lap_bonus: Discrete milestone signal—bonus for completing a lap
        #   Applied only on termination (lap_complete=True); default +1.0
        reward = reward + params.lap_bonus * lap_complete
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
        """Construct 14-element normalized observation vector (see OBSERVATION_SPECIFICATION.md).

        Observation structure:
            obs[0]     : normalized_speed = vehicle_speed / max_speed
            obs[1]     : normalized_lateral_error = lateral_error / half_width
            obs[2:4]   : [sin(heading_error), cos(heading_error)] for smooth angular encoding
            obs[4:6]   : previous_action (2D: longitudinal, turning)
            obs[6:14]  : preview_curvature clipped to [-1, 1] at 8 lookahead distances

        The observation enables:
        - Immediate ego state awareness (speed, position, heading)
        - Temporal context (previous action for smooth control)
        - Predictive planning (curvature preview for anticipatory steering)

        All components are normalized to approximately [-1, 1] for stable learning.

        Args:
            state: Current RacingEnvState (vehicle, position, errors, action history)
            params: RacingEnvParams (if None, uses default); needed for track width and car speed
            key: Unused (for Gymnax compatibility)

        Returns:
            obs: Shape (14,), dtype float32. Normalized observation vector.
        """
        del key
        if params is None:
            params = self.default_params

        # Group 1: Ego State (4 elements)
        # ================================
        # [0] Normalized speed: [0, 1] typical; enables speed awareness and control
        normalized_speed = self.model.speed(state.vehicle) / params.vehicle.max_speed

        # [1] Normalized lateral error: [-1, 1] on-track; immediate centerline feedback
        half_width = params.track.width * 0.5
        normalized_lateral_error = state.lateral_error / half_width

        # [2-3] Heading error as [sin, cos] pair: avoids angle discontinuities at ±π
        #   Both ∈ [-1, 1] and provide smooth gradients across wrap-around
        sin_heading_error = jnp.sin(state.heading_error)
        cos_heading_error = jnp.cos(state.heading_error)

        # Group 2: Action History (2 elements)
        # ====================================
        # [4-5] Previous 2D action (longitudinal, turn): enables smooth control learning
        previous_action = state.previous_action

        # Group 3: Track Geometry Lookahead (8 elements)
        # ==============================================
        # Sample curvature at PREVIEW_DISTANCES using vmap for vectorized computation
        #   Enables predictive steering by providing multi-horizon curvature estimates
        preview_curvature = jax.vmap(
            lambda distance: track_frame_at_s(params.track, state.centerline_s + distance).curvature
        )(PREVIEW_DISTANCES)

        # Normalize curvature by track width and clip to [-1, 1] for bounded features
        #   κ is geometric curvature; κ * half_width converts to track-relative scale
        normalized_preview_curvature = jnp.clip(preview_curvature * half_width, -1.0, 1.0)

        # Concatenate all components into single 14-element observation vector
        return jnp.concatenate(
            (
                jnp.asarray(
                    (
                        normalized_speed,
                        normalized_lateral_error,
                        sin_heading_error,
                        cos_heading_error,
                    )
                ),
                previous_action,
                normalized_preview_curvature,
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
