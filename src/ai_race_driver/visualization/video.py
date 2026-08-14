"""Lightweight host-only rendering of deterministic racing evaluations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIDEO_FPS = 20
MAX_VIDEO_FRAMES = 300
VIDEO_CODEC = "libx264"

Color = tuple[int, int, int]
Point = tuple[int, int]
Rect = tuple[int, int, int, int]

_BACKGROUND = (13, 18, 26)
_PANEL = (22, 29, 39)
_GRID = (53, 65, 80)
_TEXT = (225, 232, 240)
_MUTED = (132, 147, 164)
_TRACK = (67, 75, 86)
_BOUNDARY = (232, 237, 242)
_CENTERLINE = (241, 196, 83)
_PATH = (51, 205, 255)
_VEHICLE = (255, 102, 102)


@dataclass(frozen=True)
class EpisodeTelemetry:
    """Trimmed NumPy telemetry derived from one recorded evaluation episode."""

    time: np.ndarray
    position: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    longitudinal_acceleration: np.ndarray
    lateral_acceleration: np.ndarray
    yaw_rate: np.ndarray
    lateral_error: np.ndarray
    heading_error: np.ndarray
    action: np.ndarray
    ellipse_utilization: np.ndarray
    reward: np.ndarray
    cumulative_return: np.ndarray
    progress: np.ndarray
    lap_complete: bool
    off_track: bool
    time_limit: bool

    @property
    def terminal_reason(self) -> str:
        if self.lap_complete:
            return "LAP COMPLETE"
        if self.off_track:
            return "OFF TRACK"
        if self.time_limit:
            return "TIME LIMIT"
        return "MAX STEPS"


@dataclass(frozen=True)
class TrackGeometry:
    """Host NumPy geometry needed to draw a closed track."""

    center: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    width: float
    length: float


@dataclass(frozen=True)
class VideoRenderRequest:
    """All host data and presentation metadata for one evaluation video."""

    telemetry: EpisodeTelemetry
    track: TrackGeometry
    output_path: Path
    global_step: int
    update: int
    capture_seconds: float
    width: int = FRAME_WIDTH
    height: int = FRAME_HEIGHT
    fps: int = VIDEO_FPS
    max_frames: int = MAX_VIDEO_FRAMES


@dataclass(frozen=True)
class VideoRenderResult:
    """Completed video metadata returned by the background renderer."""

    path: Path
    global_step: int
    update: int
    frame_count: int
    capture_seconds: float
    render_seconds: float


@dataclass(frozen=True)
class _WorldTransform:
    scale: float
    offset_x: float
    offset_y: float

    def points(self, values: np.ndarray) -> list[Point]:
        pixels = np.empty_like(values, dtype=np.float64)
        pixels[:, 0] = self.offset_x + self.scale * values[:, 0]
        pixels[:, 1] = self.offset_y - self.scale * values[:, 1]
        return [tuple(point) for point in np.rint(pixels).astype(np.int32)]

    def point(self, value: np.ndarray) -> Point:
        return (
            round(self.offset_x + self.scale * float(value[0])),
            round(self.offset_y - self.scale * float(value[1])),
        )


@dataclass(frozen=True)
class _PlotSeries:
    name: str
    values: np.ndarray
    color: Color


@dataclass(frozen=True)
class _Plot:
    title: str
    rect: Rect
    series: tuple[_PlotSeries, ...]
    minimum: float
    maximum: float


def validate_video_backend() -> None:
    """Fail fast when the requested headless H.264 backend is unavailable."""

    try:
        import av
    except ImportError as error:  # pragma: no cover - dependency is required by the package
        raise RuntimeError("evaluation videos require the 'av' package") from error
    if VIDEO_CODEC not in av.codecs_available:
        raise RuntimeError(f"PyAV does not provide the required {VIDEO_CODEC!r} encoder")


def make_track_geometry(track: Any) -> TrackGeometry:
    """Copy device-backed track arrays into a small host-only rendering structure."""

    return TrackGeometry(
        center=np.asarray(track.center, dtype=np.float32),
        left_boundary=np.asarray(track.left_boundary, dtype=np.float32),
        right_boundary=np.asarray(track.right_boundary, dtype=np.float32),
        width=float(track.width),
        length=float(track.length),
    )


def trajectory_to_telemetry(trajectory: Any, *, dt: float) -> EpisodeTelemetry:
    """Trim fixed-shape recorded arrays and derive body-frame physical telemetry."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    episode_length = int(np.asarray(trajectory.episode_length))
    available_transitions = int(np.asarray(trajectory.action).shape[0])
    if not 1 <= episode_length <= available_transitions:
        raise ValueError("episode_length is outside the recorded trajectory")

    state_count = episode_length + 1
    position = np.asarray(trajectory.position, dtype=np.float32)[:state_count]
    heading = np.asarray(trajectory.heading, dtype=np.float32)[:state_count]
    speed = np.asarray(trajectory.speed, dtype=np.float32)[:state_count]
    lateral_error = np.asarray(trajectory.lateral_error, dtype=np.float32)[:state_count]
    heading_error = np.asarray(trajectory.heading_error, dtype=np.float32)[:state_count]
    progress = np.asarray(trajectory.accumulated_progress, dtype=np.float32)[:state_count]
    action = np.asarray(trajectory.action, dtype=np.float32)[:episode_length]
    reward = np.asarray(trajectory.reward, dtype=np.float32)[:episode_length]

    longitudinal_acceleration = np.zeros(state_count, dtype=np.float32)
    longitudinal_acceleration[1:] = np.diff(speed) / dt
    heading_delta = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
    yaw_rate = np.zeros(state_count, dtype=np.float32)
    yaw_rate[1:] = heading_delta / dt
    lateral_acceleration = speed * yaw_rate
    cumulative_return = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(reward, dtype=np.float32))
    )
    normalized_acceleration = np.clip(action, -1.0, 1.0)
    ellipse_utilization = np.concatenate(
        (
            np.zeros(1, dtype=np.float32),
            np.minimum(np.linalg.norm(normalized_acceleration, axis=-1), 1.0),
        )
    )
    time_axis = np.arange(state_count, dtype=np.float32) * dt

    terminal_index = episode_length - 1
    return EpisodeTelemetry(
        time=time_axis,
        position=position,
        heading=heading,
        speed=speed,
        longitudinal_acceleration=longitudinal_acceleration,
        lateral_acceleration=lateral_acceleration,
        yaw_rate=yaw_rate,
        lateral_error=lateral_error,
        heading_error=heading_error,
        action=action,
        ellipse_utilization=ellipse_utilization,
        reward=reward,
        cumulative_return=cumulative_return,
        progress=progress,
        lap_complete=bool(np.asarray(trajectory.lap_complete)[terminal_index]),
        off_track=bool(np.asarray(trajectory.off_track)[terminal_index]),
        time_limit=bool(np.asarray(trajectory.time_limit)[terminal_index]),
    )


def _fit_world_transform(track: TrackGeometry, rect: Rect) -> _WorldTransform:
    points = np.concatenate((track.left_boundary, track.right_boundary), axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    x0, y0, x1, y1 = rect
    scale = min((x1 - x0) / float(span[0]), (y1 - y0) / float(span[1]))
    center = 0.5 * (minimum + maximum)
    return _WorldTransform(
        scale=scale,
        offset_x=0.5 * (x0 + x1) - scale * float(center[0]),
        offset_y=0.5 * (y0 + y1) + scale * float(center[1]),
    )


def _closed(points: Sequence[Point]) -> list[Point]:
    return [*points, points[0]]


def _draw_dashed_line(draw: ImageDraw.ImageDraw, points: Sequence[Point]) -> None:
    closed = _closed(points)
    for index in range(0, len(closed) - 1, 2):
        draw.line((closed[index], closed[index + 1]), fill=_CENTERLINE, width=2)


def _symmetric_range(*values: np.ndarray, minimum_span: float = 1.0) -> tuple[float, float]:
    extent = max((float(np.max(np.abs(value))) for value in values), default=minimum_span)
    extent = max(extent * 1.08, minimum_span)
    return -extent, extent


def _make_plots(telemetry: EpisodeTelemetry, track: TrackGeometry, width: int) -> tuple[_Plot, ...]:
    x0 = width - 394
    x1 = width - 24
    rectangles = (
        (x0, 74, x1, 154),
        (x0, 173, x1, 253),
        (x0, 272, x1, 352),
        (x0, 371, x1, 451),
        (x0, 470, x1, 550),
        (x0, 569, x1, 649),
    )
    speed_max = max(float(np.max(telemetry.speed)) * 1.08, 1.0)
    accel_min, accel_max = _symmetric_range(
        telemetry.longitudinal_acceleration,
        telemetry.lateral_acceleration,
    )
    yaw_min, yaw_max = _symmetric_range(telemetry.yaw_rate, minimum_span=0.25)
    lateral_min, lateral_max = _symmetric_range(
        telemetry.lateral_error,
        minimum_span=0.5 * track.width,
    )
    reward_by_state = np.concatenate((np.zeros(1, dtype=np.float32), telemetry.reward))
    reward_min, reward_max = _symmetric_range(reward_by_state, minimum_span=0.1)
    return (
        _Plot(
            "Speed v [m/s]",
            rectangles[0],
            (_PlotSeries("v", telemetry.speed, (59, 180, 255)),),
            0.0,
            speed_max,
        ),
        _Plot(
            "Acceleration ax / ay [m/s^2]",
            rectangles[1],
            (
                _PlotSeries("ax", telemetry.longitudinal_acceleration, (255, 170, 70)),
                _PlotSeries("ay", telemetry.lateral_acceleration, (255, 92, 150)),
            ),
            accel_min,
            accel_max,
        ),
        _Plot(
            "Ellipse utilization",
            rectangles[2],
            (_PlotSeries("usage", telemetry.ellipse_utilization, (255, 214, 92)),),
            0.0,
            1.0,
        ),
        _Plot(
            "Yaw rate [rad/s]",
            rectangles[3],
            (_PlotSeries("yaw", telemetry.yaw_rate, (102, 220, 145)),),
            yaw_min,
            yaw_max,
        ),
        _Plot(
            "Lateral error [m]",
            rectangles[4],
            (_PlotSeries("error", telemetry.lateral_error, (184, 133, 255)),),
            lateral_min,
            lateral_max,
        ),
        _Plot(
            "Reward",
            rectangles[5],
            (_PlotSeries("reward", reward_by_state, (246, 211, 101)),),
            reward_min,
            reward_max,
        ),
    )


def _y_axis_ticks(plot: _Plot) -> tuple[tuple[float, float], ...]:
    midpoint = 0.5 * (plot.minimum + plot.maximum)
    return ((0.0, plot.maximum), (0.5, midpoint), (1.0, plot.minimum))


def _format_axis_value(value: float, span: float) -> str:
    precision = 2 if span < 1.0 else 1
    return f"{value:.{precision}f}"


def _plot_points(
    time_axis: np.ndarray,
    values: np.ndarray,
    rect: Rect,
    minimum: float,
    maximum: float,
) -> list[Point]:
    x0, y0, x1, y1 = rect
    duration = max(float(time_axis[-1]), 1e-6)
    value_span = max(maximum - minimum, 1e-6)
    x = x0 + (x1 - x0) * time_axis / duration
    y = y1 - (y1 - y0) * (values - minimum) / value_span
    pixels = np.column_stack((x, y))
    return [tuple(point) for point in np.rint(pixels).astype(np.int32)]


def _dim(color: Color) -> Color:
    return (
        max(30, color[0] // 3),
        max(30, color[1] // 3),
        max(30, color[2] // 3),
    )


def _draw_static_frame(
    request: VideoRenderRequest,
    transform: _WorldTransform,
    plots: Sequence[_Plot],
    plot_points: Sequence[Sequence[list[Point]]],
) -> Image.Image:
    image = Image.new("RGB", (request.width, request.height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rounded_rectangle((12, 52, request.width - 446, request.height - 16), 10, fill=_PANEL)
    draw.text((24, 18), "Deterministic fixed-start policy evaluation", fill=_TEXT, font=font)

    left = transform.points(request.track.left_boundary)
    right = transform.points(request.track.right_boundary)
    center = transform.points(request.track.center)
    road_polygon = [*left, *reversed(right)]
    draw.polygon(road_polygon, fill=_TRACK)
    draw.line(_closed(left), fill=_BOUNDARY, width=3, joint="curve")
    draw.line(_closed(right), fill=_BOUNDARY, width=3, joint="curve")
    _draw_dashed_line(draw, center)

    full_path = transform.points(request.telemetry.position)
    if len(full_path) > 1:
        draw.line(full_path, fill=_dim(_PATH), width=3, joint="curve")

    for plot, series_points in zip(plots, plot_points, strict=True):
        x0, y0, x1, y1 = plot.rect
        draw.rounded_rectangle((x0 - 46, y0 - 22, x1 + 10, y1 + 10), 8, fill=_PANEL)
        value_span = plot.maximum - plot.minimum
        for fraction, value in _y_axis_ticks(plot):
            grid_y = round(y0 + fraction * (y1 - y0))
            draw.line((x0, grid_y, x1, grid_y), fill=_GRID, width=1)
            value_text = _format_axis_value(value, value_span)
            value_width = draw.textlength(value_text, font=font)
            draw.text(
                (x0 - value_width - 6, grid_y - 5),
                value_text,
                fill=_MUTED,
                font=font,
            )
        if plot.minimum < 0.0 < plot.maximum:
            zero_y = round(y1 - (y1 - y0) * (-plot.minimum) / (plot.maximum - plot.minimum))
            draw.line((x0, zero_y, x1, zero_y), fill=_MUTED, width=1)
        draw.text((x0, y0 - 18), plot.title, fill=_TEXT, font=font)
        if len(plot.series) > 1:
            legend_x = x1 - 70
            for series in plot.series:
                draw.text((legend_x, y0 - 18), series.name, fill=series.color, font=font)
                legend_x += 34
        for series, points in zip(plot.series, series_points, strict=True):
            if len(points) > 1:
                draw.line(points, fill=_dim(series.color), width=2)
    return image


def _vehicle_polygon(center: Point, heading: float) -> list[Point]:
    forward = np.asarray((math.cos(heading), -math.sin(heading)))
    sideways = np.asarray((-forward[1], forward[0]))
    origin = np.asarray(center, dtype=np.float64)
    triangle = np.stack(
        (
            origin + 14.0 * forward,
            origin - 9.0 * forward + 7.0 * sideways,
            origin - 9.0 * forward - 7.0 * sideways,
        )
    )
    return [tuple(point) for point in np.rint(triangle).astype(np.int32)]


def _draw_dynamic_frame(
    static_frame: Image.Image,
    request: VideoRenderRequest,
    transform: _WorldTransform,
    plots: Sequence[_Plot],
    plot_points: Sequence[Sequence[list[Point]]],
    state_index: int,
) -> Image.Image:
    image = static_frame.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    telemetry = request.telemetry

    path = transform.points(telemetry.position[: state_index + 1])
    if len(path) > 1:
        draw.line(path, fill=_PATH, width=4, joint="curve")
    vehicle_center = transform.point(telemetry.position[state_index])
    vehicle = _vehicle_polygon(vehicle_center, float(telemetry.heading[state_index]))
    draw.polygon(vehicle, fill=_VEHICLE, outline=_TEXT)
    heading_tip = (
        round(vehicle_center[0] + 24.0 * math.cos(float(telemetry.heading[state_index]))),
        round(vehicle_center[1] - 24.0 * math.sin(float(telemetry.heading[state_index]))),
    )
    draw.line((vehicle_center, heading_tip), fill=_TEXT, width=2)

    duration = max(float(telemetry.time[-1]), 1e-6)
    for plot, series_points in zip(plots, plot_points, strict=True):
        for series, points in zip(plot.series, series_points, strict=True):
            history = points[: state_index + 1]
            if len(history) > 1:
                draw.line(history, fill=series.color, width=3)
            elif history:
                x, y = history[0]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=series.color)
        cursor_x = round(
            plot.rect[0]
            + (plot.rect[2] - plot.rect[0]) * float(telemetry.time[state_index]) / duration
        )
        draw.line((cursor_x, plot.rect[1], cursor_x, plot.rect[3]), fill=_TEXT, width=1)

    terminal = state_index == len(telemetry.time) - 1
    status = telemetry.terminal_reason if terminal else "RUNNING"
    progress_fraction = float(telemetry.progress[state_index]) / max(request.track.length, 1e-6)
    details = (
        f"update {request.update:,}  |  step {request.global_step:,}  |  "
        f"t={telemetry.time[state_index]:.2f}s  |  "
        f"return={telemetry.cumulative_return[state_index]:.3f}  |  "
        f"progress={100.0 * progress_fraction:.1f}%  |  {status}"
    )
    draw.rectangle((18, request.height - 45, request.width - 446, request.height - 18), fill=_PANEL)
    draw.text((26, request.height - 38), details, fill=_TEXT, font=font)
    return image


def render_evaluation_video(request: VideoRenderRequest) -> VideoRenderResult:
    """Draw and stream one evaluation episode directly into an MP4 file."""

    if request.width < 320 or request.height < 192:
        raise ValueError("video dimensions are too small")
    if request.width % 2 or request.height % 2:
        raise ValueError("video dimensions must be even for yuv420p")
    if request.fps < 1 or request.max_frames < 1:
        raise ValueError("fps and max_frames must be positive")

    import av

    started = time.perf_counter()
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = request.output_path.with_name(
        f".{request.output_path.stem}.tmp{request.output_path.suffix}"
    )
    telemetry = request.telemetry
    frame_count = min(len(telemetry.time), request.max_frames)
    frame_indices = np.linspace(0, len(telemetry.time) - 1, frame_count, dtype=np.int32)
    transform = _fit_world_transform(request.track, (32, 78, request.width - 470, 660))
    plots = _make_plots(telemetry, request.track, request.width)
    points = tuple(
        tuple(
            _plot_points(
                telemetry.time,
                series.values,
                plot.rect,
                plot.minimum,
                plot.maximum,
            )
            for series in plot.series
        )
        for plot in plots
    )
    static_frame = _draw_static_frame(request, transform, plots, points)

    try:
        with av.open(
            str(temporary_path),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        ) as container:
            stream: Any = container.add_stream(VIDEO_CODEC, rate=request.fps)
            stream.width = request.width
            stream.height = request.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "28", "preset": "veryfast", "threads": "1"}
            for state_index in frame_indices:
                image = _draw_dynamic_frame(
                    static_frame,
                    request,
                    transform,
                    plots,
                    points,
                    int(state_index),
                )
                frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temporary_path.replace(request.output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return VideoRenderResult(
        path=request.output_path,
        global_step=request.global_step,
        update=request.update,
        frame_count=frame_count,
        capture_seconds=request.capture_seconds,
        render_seconds=time.perf_counter() - started,
    )


class AsyncVideoRenderer:
    """Single-slot background renderer with explicit backpressure."""

    def __init__(
        self,
        render: Callable[[VideoRenderRequest], VideoRenderResult] = render_evaluation_video,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation-video")
        self._render = render
        self._pending: Future[VideoRenderResult] | None = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def submit(self, request: VideoRenderRequest) -> None:
        if self._pending is not None:
            raise RuntimeError("a video render is already pending")
        self._pending = self._executor.submit(self._render, request)

    def collect(self, *, wait: bool) -> VideoRenderResult | None:
        if self._pending is None or (not wait and not self._pending.done()):
            return None
        future = self._pending
        self._pending = None
        return future.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
