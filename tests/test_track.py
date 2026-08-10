import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ai_race_driver.track.spline import (
    compile_closed_track,
    make_oval_track,
    project_to_track,
    signed_progress_delta,
    track_frame_at_s,
)


def test_compiled_track_is_periodic_and_has_requested_width() -> None:
    track = make_oval_track(width=8.0, samples=128)
    start = track_frame_at_s(track, jnp.asarray(0.0))
    wrapped = track_frame_at_s(track, track.length)

    np.testing.assert_allclose(start.center, wrapped.center, atol=1e-5)
    np.testing.assert_allclose(start.tangent, wrapped.tangent, atol=1e-5)
    widths = jnp.linalg.norm(track.left_boundary - track.right_boundary, axis=1)
    np.testing.assert_allclose(widths, 8.0, atol=1e-5)


def test_track_projection_reports_signed_lateral_error_under_jit() -> None:
    track = make_oval_track(samples=128)
    frame = track_frame_at_s(track, jnp.asarray(10.0))
    query = frame.center + 1.5 * frame.normal
    projected, lateral = jax.jit(project_to_track)(track, query, frame.index)

    assert float(lateral) == pytest.approx(1.5, abs=0.08)
    assert float(projected.s) == pytest.approx(10.0, abs=0.2)


def test_progress_delta_wraps_start_line() -> None:
    assert float(
        signed_progress_delta(jnp.asarray(1.0), jnp.asarray(99.0), jnp.asarray(100.0))
    ) == pytest.approx(2.0)
    assert float(
        signed_progress_delta(jnp.asarray(99.0), jnp.asarray(1.0), jnp.asarray(100.0))
    ) == pytest.approx(-2.0)


def test_track_validation() -> None:
    with pytest.raises(ValueError, match="at least four"):
        compile_closed_track([[0, 0], [1, 0], [0, 1]], width=2.0)
    with pytest.raises(ValueError, match="positive"):
        compile_closed_track([[0, 0], [1, 0], [1, 1], [0, 1]], width=0.0)
