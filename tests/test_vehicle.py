import jax
import jax.numpy as jnp
import pytest

from ai_race_driver.vehicle.point_mass import PointMassModel, PointMassParams, wrap_angle


def test_point_mass_accelerates_and_turns_under_jit() -> None:
    model = PointMassModel()
    params = PointMassParams(dt=0.1, max_speed=10.0, max_accel=2.0, max_yaw_rate=1.0)
    state = model.reset(jnp.zeros(2), jnp.asarray(0.0), jnp.asarray(1.0), params)

    next_state = jax.jit(model.step)(state, jnp.asarray((1.0, 1.0)), params)

    assert float(next_state.speed) == pytest.approx(1.2)
    assert float(next_state.heading) == pytest.approx(0.1)
    assert next_state.position.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(next_state.position)))


def test_point_mass_respects_speed_and_action_bounds() -> None:
    model = PointMassModel()
    params = PointMassParams(dt=1.0, max_speed=3.0, max_accel=10.0, max_decel=10.0)
    state = model.reset(jnp.zeros(2), jnp.asarray(0.0), jnp.asarray(2.0), params)
    fast = model.step(state, jnp.asarray((5.0, 0.0)), params)
    stopped = model.step(fast, jnp.asarray((-5.0, 0.0)), params)
    assert float(fast.speed) == 3.0
    assert float(stopped.speed) == 0.0


def test_angle_wrap_range() -> None:
    angles = jnp.asarray((-5.0 * jnp.pi, -jnp.pi, jnp.pi, 5.0 * jnp.pi))
    wrapped = jax.vmap(wrap_angle)(angles)
    assert bool(jnp.all(wrapped >= -jnp.pi))
    assert bool(jnp.all(wrapped < jnp.pi))
