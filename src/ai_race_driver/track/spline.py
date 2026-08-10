"""Host-side periodic spline compilation and device-side track queries."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from scipy.interpolate import CubicSpline


@struct.dataclass
class TrackData:
    """Fixed-shape representation of a closed centerline and its boundaries."""

    coefficients: jax.Array
    knots: jax.Array
    sample_u: jax.Array
    sample_s: jax.Array
    center: jax.Array
    tangent: jax.Array
    normal: jax.Array
    curvature: jax.Array
    width: jax.Array
    length: jax.Array

    @property
    def num_samples(self) -> int:
        return self.center.shape[0]

    @property
    def left_boundary(self) -> jax.Array:
        return self.center + self.normal * (self.width * 0.5)

    @property
    def right_boundary(self) -> jax.Array:
        return self.center - self.normal * (self.width * 0.5)


@struct.dataclass
class TrackFrame:
    """Local centerline frame returned by track queries."""

    s: jax.Array
    center: jax.Array
    tangent: jax.Array
    normal: jax.Array
    curvature: jax.Array
    index: jax.Array


def compile_closed_track(
    control_points: Sequence[Sequence[float]] | np.ndarray,
    width: float,
    samples: int = 512,
) -> TrackData:
    """Fit a periodic cubic spline and prepare fixed-shape JAX lookup arrays.

    This function intentionally runs on the host. The returned data contains everything
    needed by compiled environment steps.
    """

    points = np.asarray(control_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("control_points must have shape (n, 2)")
    if points.shape[0] < 4:
        raise ValueError("a closed cubic track requires at least four control points")
    if width <= 0.0:
        raise ValueError("width must be positive")
    if samples < 32:
        raise ValueError("samples must be at least 32")

    if not np.allclose(points[0], points[-1]):
        points = np.concatenate((points, points[:1]), axis=0)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segment_lengths <= 1e-8):
        raise ValueError("consecutive control points must be distinct")

    knots = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    knots /= knots[-1]
    spline = CubicSpline(knots, points, axis=0, bc_type="periodic")

    dense_count = max(4096, samples * 16)
    dense_u = np.linspace(0.0, 1.0, dense_count + 1)
    dense_center = spline(dense_u)
    dense_ds = np.linalg.norm(np.diff(dense_center, axis=0), axis=1)
    dense_s = np.concatenate(([0.0], np.cumsum(dense_ds)))
    length = float(dense_s[-1])
    target_s = np.linspace(0.0, length, samples, endpoint=False)
    sample_u = np.interp(target_s, dense_s, dense_u)

    center = spline(sample_u)
    first = spline(sample_u, 1)
    second = spline(sample_u, 2)
    derivative_norm = np.linalg.norm(first, axis=1, keepdims=True)
    tangent = first / np.maximum(derivative_norm, 1e-12)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / np.maximum(
        np.linalg.norm(first, axis=1) ** 3, 1e-12
    )

    return TrackData(
        coefficients=jnp.asarray(spline.c, dtype=jnp.float32),
        knots=jnp.asarray(knots, dtype=jnp.float32),
        sample_u=jnp.asarray(sample_u, dtype=jnp.float32),
        sample_s=jnp.asarray(target_s, dtype=jnp.float32),
        center=jnp.asarray(center, dtype=jnp.float32),
        tangent=jnp.asarray(tangent, dtype=jnp.float32),
        normal=jnp.asarray(normal, dtype=jnp.float32),
        curvature=jnp.asarray(curvature, dtype=jnp.float32),
        width=jnp.asarray(width, dtype=jnp.float32),
        length=jnp.asarray(length, dtype=jnp.float32),
    )


def make_oval_track(width: float = 8.0, samples: int = 512) -> TrackData:
    """Create the deterministic built-in oval used by examples and acceptance tests."""

    angles = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    points = np.stack((35.0 * np.cos(angles), 20.0 * np.sin(angles)), axis=1)
    return compile_closed_track(points, width=width, samples=samples)


def track_frame_at_s(track: TrackData, s: jax.Array) -> TrackFrame:
    """Interpolate a local track frame at wrapped arc length ``s``."""

    wrapped_s = jnp.mod(s, track.length)
    index = jnp.searchsorted(track.sample_s, wrapped_s, side="right") - 1
    index = jnp.mod(index, track.num_samples)
    next_index = jnp.mod(index + 1, track.num_samples)
    s0 = track.sample_s[index]
    s1 = jnp.where(next_index == 0, track.length, track.sample_s[next_index])
    alpha = jnp.clip((wrapped_s - s0) / jnp.maximum(s1 - s0, 1e-6), 0.0, 1.0)

    center = (1.0 - alpha) * track.center[index] + alpha * track.center[next_index]
    tangent = (1.0 - alpha) * track.tangent[index] + alpha * track.tangent[next_index]
    tangent /= jnp.maximum(jnp.linalg.norm(tangent), 1e-6)
    normal = jnp.stack((-tangent[1], tangent[0]))
    curvature = (1.0 - alpha) * track.curvature[index] + alpha * track.curvature[next_index]
    return TrackFrame(
        s=wrapped_s,
        center=center,
        tangent=tangent,
        normal=normal,
        curvature=curvature,
        index=index,
    )


def project_to_track(
    track: TrackData,
    position: jax.Array,
    previous_index: jax.Array,
    window: int = 16,
) -> tuple[TrackFrame, jax.Array]:
    """Project a position using a fixed local search around the prior sample index."""

    offsets = jnp.arange(-window, window + 1)
    candidate_indices = jnp.mod(previous_index + offsets, track.num_samples)
    candidate_points = track.center[candidate_indices]
    squared_distances = jnp.sum(jnp.square(candidate_points - position), axis=-1)
    nearest = candidate_indices[jnp.argmin(squared_distances)]

    center = track.center[nearest]
    tangent = track.tangent[nearest]
    local_s = track.sample_s[nearest] + jnp.dot(position - center, tangent)
    frame = track_frame_at_s(track, local_s)
    lateral_error = jnp.dot(position - frame.center, frame.normal)
    return frame, lateral_error


def signed_progress_delta(
    current_s: jax.Array, previous_s: jax.Array, length: jax.Array
) -> jax.Array:
    """Return the shortest signed progress delta across a periodic seam."""

    return jnp.mod(current_s - previous_s + 0.5 * length, length) - 0.5 * length
