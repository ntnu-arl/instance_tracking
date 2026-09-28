"""Tests for ROS output mask/confidence selection."""

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[2] / "instance_tracking_ros"),
)

from instance_tracking.tracker import TrackerResult  # noqa: E402
from instance_tracking_ros.output_masks import (  # noqa: E402
    render_mask_overlay,
    select_output_confidence,
    select_output_mask,
)


def test_keyframe_fastsam_output_uses_mask_aligned_confidence():
    probability_mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.uint16)
    probability_confidence = torch.tensor(
        [[0.95, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    fastsam_mask = torch.tensor([[0, 2], [2, 0]], dtype=torch.uint16)
    result = TrackerResult(
        masks=probability_mask,
        tracks=[],
        is_keyframe=True,
        frame_index=7,
        keyframe_fastsam_mask=fastsam_mask,
        final_argmax_confidence=probability_confidence,
    )

    assert torch.equal(select_output_mask(result, True), fastsam_mask)

    confidence = select_output_confidence(result, True)
    expected_confidence = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    assert torch.equal(confidence, expected_confidence)


def test_probability_output_keeps_probability_confidence():
    probability_mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.uint16)
    probability_confidence = torch.tensor(
        [[0.95, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    fastsam_mask = torch.tensor([[0, 2], [2, 0]], dtype=torch.uint16)
    result = TrackerResult(
        masks=probability_mask,
        tracks=[],
        is_keyframe=True,
        frame_index=7,
        keyframe_fastsam_mask=fastsam_mask,
        final_argmax_confidence=probability_confidence,
    )

    assert torch.equal(select_output_mask(result, False), probability_mask)
    assert select_output_confidence(result, False) is probability_confidence


def test_render_mask_overlay_blends_only_foreground_pixels():
    frame = np.array(
        [
            [[100, 100, 100], [20, 40, 60]],
            [[10, 20, 30], [200, 200, 200]],
        ],
        dtype=np.uint8,
    )
    colors = np.array(
        [
            [[200, 0, 100], [220, 140, 60]],
            [[90, 80, 70], [0, 100, 200]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[3, 0], [0, 9]], dtype=np.uint16)

    overlay = render_mask_overlay(frame, colors, mask, alpha=0.5)

    expected = frame.copy()
    expected[0, 0] = [150, 50, 100]
    expected[1, 1] = [100, 150, 200]
    assert np.array_equal(overlay, expected)
    assert not np.shares_memory(overlay, frame)


def test_render_mask_overlay_rejects_misaligned_mask():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    colors = np.zeros_like(frame)

    with pytest.raises(ValueError, match="mask_image must have shape"):
        render_mask_overlay(frame, colors, np.zeros((3, 2), dtype=np.uint16))
