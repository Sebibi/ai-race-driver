"""Closed-track construction and JAX geometry queries."""

from ai_race_driver.track.spline import (
    TrackData,
    TrackFrame,
    compile_closed_track,
    make_oval_track,
    project_to_track,
    track_frame_at_s,
)

__all__ = [
    "TrackData",
    "TrackFrame",
    "compile_closed_track",
    "make_oval_track",
    "project_to_track",
    "track_frame_at_s",
]
