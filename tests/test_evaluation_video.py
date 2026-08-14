"""Tests for deterministic trajectory capture and headless evaluation videos."""

from pathlib import Path
from threading import Event

import av
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ai_race_driver.envs.racing import RacingEnv, RacingEnvParams
from ai_race_driver.track.spline import make_oval_track
from ai_race_driver.training.evaluation import (
    EvaluationTrajectory,
    make_evaluate_policy,
    make_record_policy,
)
from ai_race_driver.training.ppo import ActorCritic
from ai_race_driver.visualization.video import (
    AsyncVideoRenderer,
    VideoRenderRequest,
    VideoRenderResult,
    make_track_geometry,
    render_evaluation_video,
    trajectory_to_telemetry,
    validate_video_backend,
)


def _synthetic_trajectory() -> EvaluationTrajectory:
    return EvaluationTrajectory(
        position=jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.2))),
        heading=jnp.asarray((jnp.pi - 0.1, -jnp.pi + 0.1, -jnp.pi + 0.2)),
        speed=jnp.asarray((1.0, 2.0, 2.0)),
        action=jnp.asarray(((0.5, 0.2), (0.0, 0.1))),
        reward=jnp.asarray((1.0, 2.0)),
        accumulated_progress=jnp.asarray((0.0, 1.0, 2.0)),
        lateral_error=jnp.asarray((0.0, 0.1, 0.2)),
        heading_error=jnp.asarray((0.0, 0.1, 0.2)),
        valid=jnp.asarray((True, True)),
        done=jnp.asarray((False, True)),
        lap_complete=jnp.asarray((False, True)),
        off_track=jnp.asarray((False, False)),
        time_limit=jnp.asarray((False, False)),
        episode_length=jnp.asarray(2, dtype=jnp.int32),
        episode_return=jnp.asarray(3.0),
    )


def _request(path: Path) -> VideoRenderRequest:
    track = make_track_geometry(make_oval_track(samples=64))
    telemetry = trajectory_to_telemetry(_synthetic_trajectory(), dt=0.1)
    return VideoRenderRequest(
        telemetry=telemetry,
        track=track,
        output_path=path,
        global_step=128,
        update=2,
        capture_seconds=0.01,
        max_frames=3,
    )


def test_compiled_recording_is_reproducible_and_retains_terminal_state() -> None:
    params = RacingEnvParams(track=make_oval_track(samples=64), max_steps_in_episode=16)
    env = RacingEnv(params)
    model = ActorCritic(action_dim=2, hidden_size=32)
    observation, initial_state = env.reset(jax.random.key(0), params.replace(randomize_reset=False))
    model_params = model.init(jax.random.key(1), observation)
    record = jax.jit(make_record_policy(env, params, model))

    first = record(model_params, jax.random.key(2))
    second = record(model_params, jax.random.key(2))
    metrics = jax.jit(make_evaluate_policy(env, params, model, randomized_episodes=1))(
        model_params,
        jax.random.key(2),
    )
    jax.block_until_ready(first.episode_length)

    assert first.position.shape == (params.max_steps_in_episode + 1, 2)
    assert first.action.shape == (params.max_steps_in_episode, 2)
    assert int(first.episode_length) == params.max_steps_in_episode
    assert bool(first.done[params.max_steps_in_episode - 1])
    assert bool(first.time_limit[params.max_steps_in_episode - 1])
    assert not np.array_equal(
        np.asarray(first.position[-1]),
        np.asarray(initial_state.vehicle.position),
    )
    matches = jax.tree.leaves(jax.tree.map(jnp.array_equal, first, second))
    assert all(bool(match) for match in matches)
    np.testing.assert_allclose(first.episode_return, metrics.fixed.return_mean, rtol=1e-6)


def test_telemetry_derives_wrapped_body_accelerations() -> None:
    telemetry = trajectory_to_telemetry(_synthetic_trajectory(), dt=0.1)

    np.testing.assert_allclose(telemetry.longitudinal_acceleration, (0.0, 10.0, 0.0))
    np.testing.assert_allclose(telemetry.yaw_rate, (0.0, 2.0, 1.0), atol=2e-5)
    np.testing.assert_allclose(telemetry.lateral_acceleration, (0.0, 4.0, 2.0), atol=4e-5)
    np.testing.assert_allclose(telemetry.cumulative_return, (0.0, 1.0, 3.0))
    assert telemetry.terminal_reason == "LAP COMPLETE"


def test_rendered_mp4_is_decodable(tmp_path: Path) -> None:
    validate_video_backend()
    output = tmp_path / "evaluation.mp4"

    result = render_evaluation_video(_request(output))

    assert result.path == output
    assert result.frame_count == 3
    assert output.stat().st_size > 0
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (1280, 720)
        assert len(frames) == 3


def test_background_renderer_has_one_explicit_pending_slot(tmp_path: Path) -> None:
    release = Event()
    request = _request(tmp_path / "background.mp4")

    def render(pending_request: VideoRenderRequest) -> VideoRenderResult:
        release.wait(timeout=2.0)
        return VideoRenderResult(
            path=pending_request.output_path,
            global_step=pending_request.global_step,
            update=pending_request.update,
            frame_count=1,
            capture_seconds=pending_request.capture_seconds,
            render_seconds=0.1,
        )

    with AsyncVideoRenderer(render) as renderer:
        renderer.submit(request)
        assert renderer.pending
        assert renderer.collect(wait=False) is None
        release.set()
        result = renderer.collect(wait=True)

    assert result is not None
    assert result.global_step == 128


def test_video_request_rejects_invalid_episode_length() -> None:
    trajectory = _synthetic_trajectory()._replace(episode_length=jnp.asarray(0))

    with pytest.raises(ValueError, match="episode_length"):
        trajectory_to_telemetry(trajectory, dt=0.1)
