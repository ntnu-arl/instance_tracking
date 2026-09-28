"""Helpers for selecting the primary output mask image."""

import cv2
import numpy as np
import torch
from torch import Tensor

from instance_tracking.tracker import TrackerResult


def uses_keyframe_fastsam_mask(
    result: TrackerResult,
    publish_keyframe_fastsam_mask: bool,
) -> bool:
    """Return whether the remapped FastSAM keyframe mask is the published mask."""

    return (
        publish_keyframe_fastsam_mask
        and result.is_keyframe
        and result.keyframe_fastsam_mask is not None
    )


def select_output_mask(
    result: TrackerResult,
    publish_keyframe_fastsam_mask: bool,
) -> Tensor:
    """Choose the mask image to publish and store as the primary output."""

    if uses_keyframe_fastsam_mask(result, publish_keyframe_fastsam_mask):
        return result.keyframe_fastsam_mask

    return result.masks


def select_output_confidence(
    result: TrackerResult,
    publish_keyframe_fastsam_mask: bool,
) -> Tensor | None:
    """Choose the confidence image that matches the selected primary mask.

    The tracker probability confidence is aligned with ``result.masks``. When the
    primary keyframe output is instead the remapped FastSAM mask, publish a
    mask-aligned binary confidence image so downstream fusion does not suppress
    FastSAM-confirmed pixels that the probability path did not select.
    """

    if uses_keyframe_fastsam_mask(result, publish_keyframe_fastsam_mask):
        return result.keyframe_fastsam_mask.to(dtype=torch.int32).gt(0).to(
            dtype=torch.float32
        )

    return result.final_argmax_confidence


def render_mask_overlay(
    frame_rgb: np.ndarray,
    color_mask_rgb: np.ndarray,
    mask_image: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend mask colors over foreground pixels while preserving the RGB background."""

    frame = np.asarray(frame_rgb)
    color_mask = np.asarray(color_mask_rgb)
    mask = np.asarray(mask_image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame_rgb must have shape [H, W, 3], got {frame.shape}")
    if color_mask.shape != frame.shape:
        raise ValueError(
            "color_mask_rgb must match frame_rgb shape, "
            f"got {color_mask.shape} and {frame.shape}"
        )
    if mask.shape != frame.shape[:2]:
        raise ValueError(
            f"mask_image must have shape {frame.shape[:2]}, got {mask.shape}"
        )
    if frame.dtype != np.uint8 or color_mask.dtype != np.uint8:
        raise ValueError("frame_rgb and color_mask_rgb must use uint8 RGB pixels")

    blend_alpha = float(alpha)
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {blend_alpha}")

    frame = np.ascontiguousarray(frame)
    color_mask = np.ascontiguousarray(color_mask)
    mask = np.ascontiguousarray(mask)
    blended = cv2.addWeighted(
        frame,
        1.0 - blend_alpha,
        color_mask,
        blend_alpha,
        0.0,
    )
    foreground = cv2.compare(mask, 0, cv2.CMP_GT)
    overlay = frame.copy()
    cv2.copyTo(blended, foreground, overlay)
    return overlay
