"""A minimal heading-and-speed point vehicle."""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class PointMassState:
    """State of the kinematic point vehicle."""

    position: jax.Array
    heading: jax.Array
    speed: jax.Array


@struct.dataclass
class PointMassParams:
    """Physical limits for the point vehicle, in SI units."""

    dt: float = struct.field(pytree_node=False, default=0.10)
    max_speed: float = struct.field(pytree_node=False, default=40.0)
    max_ax: float = struct.field(pytree_node=False, default=20.0)
    max_ay: float = struct.field(pytree_node=False, default=50.0)


class PointMassModel:
    """Semi-implicit point-vehicle dynamics with acceleration controls."""

    def reset(
        self,
        position: jax.Array,
        heading: jax.Array,
        speed: jax.Array,
        params: PointMassParams,
    ) -> PointMassState:
        del params
        return PointMassState(position=position, heading=heading, speed=speed)

    def step(
        self,
        state: PointMassState,
        action: jax.Array,
        params: PointMassParams,
    ) -> PointMassState:
        action = jnp.clip(action, -1.0, 1.0)
        traction_usage = jnp.linalg.norm(action)
        action = action / jnp.maximum(traction_usage, 1.0)
        ax = action[0] * params.max_ax
        ay = action[1] * params.max_ay

        speed = jnp.clip(state.speed + ax * params.dt, 0.0, params.max_speed)
        yaw_rate = jnp.where(speed > 0.0, ay / jnp.maximum(speed, 1e-6), 0.0)
        heading = wrap_angle(state.heading + yaw_rate * params.dt)
        direction = jnp.stack((jnp.cos(heading), jnp.sin(heading)))
        position = state.position + speed * params.dt * direction
        return PointMassState(position=position, heading=heading, speed=speed)

    def position(self, state: PointMassState) -> jax.Array:
        return state.position

    def heading(self, state: PointMassState) -> jax.Array:
        return state.heading

    def speed(self, state: PointMassState) -> jax.Array:
        return state.speed


def wrap_angle(angle: jax.Array) -> jax.Array:
    """Wrap an angle to [-pi, pi)."""

    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
