"""Tests for live-camera cadence decisions."""

import pathlib
import sys

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[2] / "instance_tracking_ros"),
)

from instance_tracking_ros.cadence import (  # noqa: E402
    is_camera_keyframe,
    should_process_camera_frame,
)


def test_camera_keyframes_use_zero_based_camera_frame_index():
    keyframes = [
        index
        for index in range(73)
        if is_camera_keyframe(index, keyframe_interval=24)
    ]

    assert keyframes == [0, 24, 48, 72]


def test_stride_keeps_camera_keyframes_even_when_not_on_stride():
    selected = [
        index
        for index in range(30)
        if should_process_camera_frame(
            index,
            process_stride=5,
            keyframe_interval=24,
        )
    ]

    assert selected == [0, 5, 10, 15, 20, 24, 25]


def test_stride_one_processes_every_camera_frame():
    assert [
        should_process_camera_frame(
            index,
            process_stride=1,
            keyframe_interval=24,
        )
        for index in range(8)
    ] == [True] * 8
