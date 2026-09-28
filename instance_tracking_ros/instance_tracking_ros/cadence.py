"""Frame-cadence helpers for live tracker input."""

from __future__ import annotations


def normalize_process_stride(process_stride: int) -> int:
    """Return a valid process stride."""

    return max(1, int(process_stride))


def is_camera_keyframe(camera_frame_index: int, keyframe_interval: int) -> bool:
    """Return whether a zero-based camera frame should run keyframe inference."""

    interval = int(keyframe_interval)
    if interval <= 0:
        return False
    return int(camera_frame_index) % interval == 0


def should_process_camera_frame(
    camera_frame_index: int,
    *,
    process_stride: int,
    keyframe_interval: int,
) -> bool:
    """Return whether a camera frame should be sent through the tracker."""

    stride = normalize_process_stride(process_stride)
    index = int(camera_frame_index)
    if is_camera_keyframe(index, keyframe_interval):
        return True
    return index % stride == 0
