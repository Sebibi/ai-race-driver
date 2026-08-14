import jax
import jax.numpy as jnp
import pytest

from ai_race_driver.vehicle.point_mass import PointMassModel, PointMassParams, wrap_angle


def test_point_mass_accelerates_and_turns_under_jit() -> None:
    model = PointMassModel()
    params = PointMassParams(dt=0.1, max_speed=10.0, max_ax=2.0, max_ay=1.0)
    state = model.reset(jnp.zeros(2), jnp.asarray(0.0), jnp.asarray(1.0), params)

    next_state = jax.jit(model.step)(state, jnp.asarray((0.5, 0.5)), params)

    assert float(next_state.speed) == pytest.approx(1.1)
    assert float(next_state.heading) == pytest.approx(0.5 / 1.1 * 0.1)
    assert next_state.position.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(next_state.position)))


def test_point_mass_respects_speed_and_action_bounds() -> None:
    model = PointMassModel()
    params = PointMassParams(dt=1.0, max_speed=3.0, max_ax=10.0)
    state = model.reset(jnp.zeros(2), jnp.asarray(0.0), jnp.asarray(2.0), params)
    fast = model.step(state, jnp.asarray((5.0, 0.0)), params)
    stopped = model.step(fast, jnp.asarray((-5.0, 0.0)), params)
    assert float(fast.speed) == 3.0
    assert float(stopped.speed) == 0.0


def test_point_mass_projects_acceleration_onto_traction_ellipse_under_vmap() -> None:
    model = PointMassModel()
    params = PointMassParams(dt=0.1, max_speed=20.0, max_ax=4.0, max_ay=8.0)
    state = model.reset(jnp.zeros(2), jnp.asarray(0.0), jnp.asarray(10.0), params)
    actions = jnp.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))

    states = jax.vmap(model.step, in_axes=(None, 0, None))(state, actions, params)
    ax = (states.speed - state.speed) / params.dt
    yaw_rate = wrap_angle(states.heading - state.heading) / params.dt
    ay = states.speed * yaw_rate
    ellipse_usage = jnp.square(ax / params.max_ax) + jnp.square(ay / params.max_ay)

    assert bool(jnp.all(ellipse_usage <= 1.0 + 1e-6))
    assert float(ellipse_usage[0]) == pytest.approx(1.0, abs=3e-6)
    assert float(ellipse_usage[1]) == pytest.approx(1.0, abs=3e-6)
    assert float(ellipse_usage[2]) == pytest.approx(1.0, abs=3e-6)


def test_point_mass_does_not_turn_from_lateral_acceleration_at_rest() -> None:
    model = PointMassModel()
    params = PointMassParams(dt=0.1, max_ay=8.0)
    state = model.reset(jnp.zeros(2), jnp.asarray(0.5), jnp.asarray(0.0), params)

    next_state = jax.jit(model.step)(state, jnp.asarray((0.0, 1.0)), params)

    assert float(next_state.speed) == 0.0
    assert float(next_state.heading) == pytest.approx(0.5)
    assert bool(jnp.all(jnp.isfinite(next_state.position)))


def test_angle_wrap_range() -> None:
    angles = jnp.asarray((-5.0 * jnp.pi, -jnp.pi, jnp.pi, 5.0 * jnp.pi))
    wrapped = jax.vmap(wrap_angle)(angles)
    assert bool(jnp.all(wrapped >= -jnp.pi))
    assert bool(jnp.all(wrapped < jnp.pi))
