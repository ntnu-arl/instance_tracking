"""Open-vocabulary feature extraction for tracked keyframe observations."""

from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from spark_config import Config
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from instance_tracking_msgs.msg import OpenVocabTrackFeatures as OpenVocabTrackFeaturesMsg
from instance_tracking_msgs.msg import TrackedInstances
from instance_tracking_msgs.srv import EncodeOpenVocabImage as EncodeOpenVocabImageSrv
from instance_tracking_msgs.srv import EncodeOpenVocabText as EncodeOpenVocabTextSrv
from instance_tracking_ros.ros_conversions import Conversions
from instance_tracking_ros.runtime_metrics import build_numeric_summary


OPEN_VOCAB_KEYFRAME_UPDATE_POLICIES = {
    "all_tracks",
    "source_tracks_only",
}

OPEN_VOCAB_CROP_SAMPLING_MODES = {
    "center_square",
    "warp_bbox",
}


def default_normalization_parameters() -> tuple[Tensor, Tensor]:
    """Return OpenAI CLIP normalization parameters as a fallback."""

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)
    return mean, std


def normalization_parameters_from_preprocess_cfg(
    preprocess_cfg: dict[str, Any] | None,
) -> tuple[Tensor, Tensor]:
    """Return CLIP normalization parameters from an OpenCLIP preprocess config."""

    if preprocess_cfg is None:
        return default_normalization_parameters()

    mean_values = preprocess_cfg.get("mean")
    std_values = preprocess_cfg.get("std")
    if mean_values is None or std_values is None:
        return default_normalization_parameters()

    mean = torch.as_tensor(mean_values, dtype=torch.float32)
    std = torch.as_tensor(std_values, dtype=torch.float32)
    if mean.numel() != 3 or std.numel() != 3:
        raise ValueError(
            "OpenCLIP image preprocessing must provide three-channel mean/std values"
        )
    return mean.reshape(3), std.reshape(3)


def center_crop(img: Tensor, size: int, value: float = 0.0) -> Tensor:
    """Center crop a tensor, padding first if needed."""

    missing = size - torch.tensor(img.shape[-2:], device=img.device)
    missing = torch.maximum(missing, torch.tensor([0], device=img.device))
    padding = torch.ceil(missing / 2.0)
    pad_x = int(padding[1].item())
    pad_y = int(padding[0].item())
    img = F.pad(img, (pad_x, pad_x, pad_y, pad_y), "constant", value=value)
    height, width = img.shape[-2:]
    offset_y = (height - size) // 2
    offset_x = (width - size) // 2
    return img[..., offset_y : offset_y + size, offset_x : offset_x + size]


def _crop_to_bbox(img: Tensor, bbox_xyxy: Tensor) -> Tensor:
    x_min, y_min, x_max, y_max = [int(v) for v in bbox_xyxy.tolist()]
    return img[..., y_min:y_max, x_min:x_max]


def _extract_patch(
    img: Tensor,
    bbox_xyxy: Tensor,
    size: int,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
) -> Tensor:
    crop = _crop_to_bbox(img, bbox_xyxy)
    crop = TF.resize(crop, size, interpolation=interpolation, antialias=True)
    return center_crop(crop, size)


def _extract_mask_patch(mask: Tensor, bbox_xyxy: Tensor, size: int) -> Tensor:
    crop = _crop_to_bbox(mask, bbox_xyxy).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    dims = torch.tensor(crop.shape[-2:], dtype=torch.float32, device=crop.device)
    ratio = float(size) / float(torch.min(dims).item())
    new_h, new_w = torch.round(ratio * dims).to(dtype=torch.int64)
    resized = F.interpolate(
        crop,
        size=(int(new_h.item()), int(new_w.item())),
        mode="nearest",
    )
    resized = resized.squeeze(0).squeeze(0).to(dtype=torch.bool)
    return center_crop(resized, size)


def _compute_bbox(mask: Tensor, crop_padding: int) -> Optional[Tensor]:
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        return None

    y_min = int(nonzero[:, 0].min().item())
    y_max = int(nonzero[:, 0].max().item()) + 1
    x_min = int(nonzero[:, 1].min().item())
    x_max = int(nonzero[:, 1].max().item()) + 1

    if crop_padding > 0:
        height, width = mask.shape[-2:]
        x_min = max(0, x_min - crop_padding)
        y_min = max(0, y_min - crop_padding)
        x_max = min(width, x_max + crop_padding)
        y_max = min(height, y_max + crop_padding)

    return torch.tensor([x_min, y_min, x_max, y_max], device=mask.device)


def compute_instance_bboxes_gpu(
    mask_hw: Tensor,
    instance_ids: Tensor,
    crop_padding: int = 0,
) -> tuple[Tensor, Tensor]:
    """Compute padded xyxy boxes for instance ids without per-instance CPU syncs."""

    if mask_hw.ndim != 2:
        raise ValueError(f"Expected mask_hw to be [H, W], got shape={tuple(mask_hw.shape)}")

    height, width = mask_hw.shape
    ids = instance_ids.to(device=mask_hw.device, dtype=mask_hw.dtype).reshape(-1)
    if ids.numel() == 0:
        empty_boxes = torch.empty((0, 4), device=mask_hw.device, dtype=torch.float32)
        empty_valid = torch.empty((0,), device=mask_hw.device, dtype=torch.bool)
        return empty_boxes, empty_valid

    instance_masks = mask_hw.unsqueeze(0).eq(ids.view(-1, 1, 1))
    valid = instance_masks.flatten(1).any(dim=1)

    has_y = instance_masks.any(dim=2)
    has_x = instance_masks.any(dim=1)
    y_coords = torch.arange(height, device=mask_hw.device, dtype=torch.int64).view(1, -1)
    x_coords = torch.arange(width, device=mask_hw.device, dtype=torch.int64).view(1, -1)
    y_min = torch.where(has_y, y_coords, torch.full_like(y_coords, height)).amin(dim=1)
    x_min = torch.where(has_x, x_coords, torch.full_like(x_coords, width)).amin(dim=1)
    y_max = torch.where(has_y, y_coords, torch.full_like(y_coords, -1)).amax(dim=1) + 1
    x_max = torch.where(has_x, x_coords, torch.full_like(x_coords, -1)).amax(dim=1) + 1

    if crop_padding > 0:
        padding = int(crop_padding)
        y_min = (y_min - padding).clamp(min=0, max=height)
        x_min = (x_min - padding).clamp(min=0, max=width)
        y_max = (y_max + padding).clamp(min=0, max=height)
        x_max = (x_max + padding).clamp(min=0, max=width)

    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    default_box = torch.tensor([0, 0, 1, 1], device=mask_hw.device, dtype=boxes.dtype)
    boxes = torch.where(valid.view(-1, 1), boxes, default_box.view(1, 4))
    return boxes.to(dtype=torch.float32), valid


def _center_square_sampling_grid(
    boxes_xyxy: Tensor,
    image_height: int,
    image_width: int,
    size: int,
) -> Tensor:
    """Build a grid_sample grid for resize-shorter-then-center-crop semantics."""

    if boxes_xyxy.numel() == 0:
        return torch.empty((0, size, size, 2), device=boxes_xyxy.device, dtype=torch.float32)

    boxes = boxes_xyxy.to(dtype=torch.float32)
    x_min, y_min, x_max, y_max = boxes.unbind(dim=1)
    box_w = (x_max - x_min).clamp(min=1.0)
    box_h = (y_max - y_min).clamp(min=1.0)
    side = torch.minimum(box_w, box_h).clamp(min=1.0)
    x_start = x_min + (box_w - side) * 0.5
    y_start = y_min + (box_h - side) * 0.5

    steps = torch.arange(size, device=boxes.device, dtype=torch.float32)
    offsets = (steps + 0.5).view(1, -1) * (side.view(-1, 1) / float(size)) - 0.5
    src_x = x_start.view(-1, 1) + offsets
    src_y = y_start.view(-1, 1) + offsets

    norm_x = (src_x + 0.5) * (2.0 / float(image_width)) - 1.0
    norm_y = (src_y + 0.5) * (2.0 / float(image_height)) - 1.0
    grid_x = norm_x.view(-1, 1, size).expand(-1, size, -1)
    grid_y = norm_y.view(-1, size, 1).expand(-1, -1, size)
    return torch.stack((grid_x, grid_y), dim=-1)


def _warp_bbox_sampling_grid(
    boxes_xyxy: Tensor,
    image_height: int,
    image_width: int,
    size: int,
) -> Tensor:
    """Build a grid_sample grid that warps the full bbox into a square patch."""

    if boxes_xyxy.numel() == 0:
        return torch.empty((0, size, size, 2), device=boxes_xyxy.device, dtype=torch.float32)

    boxes = boxes_xyxy.to(dtype=torch.float32)
    x_min, y_min, x_max, y_max = boxes.unbind(dim=1)
    box_w = (x_max - x_min).clamp(min=1.0)
    box_h = (y_max - y_min).clamp(min=1.0)

    steps = torch.arange(size, device=boxes.device, dtype=torch.float32)
    x_offsets = (steps + 0.5).view(1, -1) * (box_w.view(-1, 1) / float(size)) - 0.5
    y_offsets = (steps + 0.5).view(1, -1) * (box_h.view(-1, 1) / float(size)) - 0.5
    src_x = x_min.view(-1, 1) + x_offsets
    src_y = y_min.view(-1, 1) + y_offsets

    norm_x = (src_x + 0.5) * (2.0 / float(image_width)) - 1.0
    norm_y = (src_y + 0.5) * (2.0 / float(image_height)) - 1.0
    grid_x = norm_x.view(-1, 1, size).expand(-1, size, -1)
    grid_y = norm_y.view(-1, size, 1).expand(-1, -1, size)
    return torch.stack((grid_x, grid_y), dim=-1)


def _sampling_grid(
    boxes_xyxy: Tensor,
    image_height: int,
    image_width: int,
    size: int,
    crop_sampling_mode: str,
) -> Tensor:
    mode = crop_sampling_mode.lower().strip()
    if mode == "center_square":
        return _center_square_sampling_grid(boxes_xyxy, image_height, image_width, size)
    if mode == "warp_bbox":
        return _warp_bbox_sampling_grid(boxes_xyxy, image_height, image_width, size)
    raise ValueError(f"Unsupported open_vocab.crop_sampling_mode={crop_sampling_mode!r}")


def compute_observation_geometry_stats_gpu(
    mask_hw: Tensor,
    instance_ids: Tensor,
    *,
    crop_padding: int = 0,
    crop_sampling_mode: str = "center_square",
) -> dict[str, Tensor]:
    """Compute cheap per-instance crop quality diagnostics on the tensor device."""

    if mask_hw.ndim != 2:
        raise ValueError(f"Expected mask_hw to be [H, W], got shape={tuple(mask_hw.shape)}")

    ids = instance_ids.to(device=mask_hw.device, dtype=mask_hw.dtype).reshape(-1)
    if ids.numel() == 0:
        empty = torch.empty((0,), device=mask_hw.device, dtype=torch.float32)
        empty_bool = torch.empty((0,), device=mask_hw.device, dtype=torch.bool)
        return {
            "valid": empty_bool,
            "mask_area_px": empty,
            "bbox_short_side_px": empty,
            "raw_mask_to_box_fill": empty,
            "mask_to_box_fill": empty,
            "padding_clamp_fraction": empty,
            "touches_image_boundary": empty_bool,
        }

    height, width = mask_hw.shape
    instance_masks = mask_hw.unsqueeze(0).eq(ids.view(-1, 1, 1))
    mask_area = instance_masks.flatten(1).sum(dim=1).to(dtype=torch.float32)

    raw_boxes, raw_valid = compute_instance_bboxes_gpu(mask_hw, ids, crop_padding=0)
    padded_boxes, padded_valid = compute_instance_bboxes_gpu(
        mask_hw,
        ids,
        crop_padding=crop_padding,
    )
    valid = raw_valid & padded_valid

    raw_x_min, raw_y_min, raw_x_max, raw_y_max = raw_boxes.unbind(dim=1)
    pad_x_min, pad_y_min, pad_x_max, pad_y_max = padded_boxes.unbind(dim=1)
    raw_w = (raw_x_max - raw_x_min).clamp(min=1.0)
    raw_h = (raw_y_max - raw_y_min).clamp(min=1.0)
    padded_w = (pad_x_max - pad_x_min).clamp(min=1.0)
    padded_h = (pad_y_max - pad_y_min).clamp(min=1.0)

    raw_box_area = (raw_w * raw_h).clamp(min=1.0)
    raw_fill = mask_area / raw_box_area
    if crop_sampling_mode.lower().strip() == "center_square":
        clip_side = torch.minimum(padded_w, padded_h).clamp(min=1.0)
        clip_crop_area = clip_side * clip_side
    elif crop_sampling_mode.lower().strip() == "warp_bbox":
        clip_crop_area = (padded_w * padded_h).clamp(min=1.0)
    else:
        raise ValueError(f"Unsupported open_vocab.crop_sampling_mode={crop_sampling_mode!r}")
    clip_fill = mask_area / clip_crop_area

    padding = max(0, int(crop_padding))
    requested_w = (raw_w + float(2 * padding)).clamp(min=1.0)
    requested_h = (raw_h + float(2 * padding)).clamp(min=1.0)
    requested_area = (requested_w * requested_h).clamp(min=1.0)
    padding_clamp_fraction = ((padded_w * padded_h) / requested_area).clamp(0.0, 1.0)
    touches_boundary = (
        (raw_x_min <= 0.0)
        | (raw_y_min <= 0.0)
        | (raw_x_max >= float(width))
        | (raw_y_max >= float(height))
    )

    zero = torch.zeros_like(mask_area)
    return {
        "valid": valid,
        "mask_area_px": torch.where(valid, mask_area, zero),
        "bbox_short_side_px": torch.where(valid, torch.minimum(raw_w, raw_h), zero),
        "raw_mask_to_box_fill": torch.where(valid, raw_fill, zero),
        "mask_to_box_fill": torch.where(valid, clip_fill, zero),
        "padding_clamp_fraction": torch.where(valid, padding_clamp_fraction, zero),
        "touches_image_boundary": valid & touches_boundary,
    }


def extract_boxed_and_masked_patches_gpu(
    img_chw: Tensor,
    mask_hw: Tensor,
    track_requests: list[tuple[int, int]],
    size: int,
    crop_padding: int = 4,
    crop_sampling_mode: str = "center_square",
    normalization_parameters: tuple[Tensor, Tensor] | None = None,
    output_dtype: torch.dtype | None = None,
    include_masked: bool = True,
) -> tuple[Tensor, Tensor | None, list[tuple[int, int]], Tensor]:
    """Extract a batch of CLIP boxed and masked patches on the tensor device.

    Returned patch tensors use ``output_dtype`` when provided. When
    ``include_masked`` is false, masked-patch resampling and masking are skipped
    and the second returned value is ``None``.
    """

    if img_chw.ndim != 3 or img_chw.shape[0] != 3:
        raise ValueError(f"Expected img_chw to be [3, H, W], got shape={tuple(img_chw.shape)}")
    if mask_hw.ndim != 2:
        raise ValueError(f"Expected mask_hw to be [H, W], got shape={tuple(mask_hw.shape)}")
    if img_chw.shape[-2:] != mask_hw.shape[-2:]:
        raise ValueError(
            f"RGB/mask shape mismatch: img={tuple(img_chw.shape[-2:])}, "
            f"mask={tuple(mask_hw.shape[-2:])}"
        )

    device = img_chw.device
    patch_dtype = output_dtype or torch.float32
    # Keep interpolation in fp32; CUDA fp16 grid_sample is faster only marginally here
    # and produced noticeably larger CLIP embedding drift in saved-frame checks.
    sample_dtype = torch.float32

    if not track_requests:
        empty = torch.empty((0, 3, size, size), device=device, dtype=patch_dtype)
        valid = torch.empty((0,), device=device, dtype=torch.bool)
        return empty, empty if include_masked else None, [], valid

    metadata = [(int(track_id), int(instance_id)) for track_id, instance_id in track_requests]
    instance_ids = torch.tensor(
        [instance_id for _, instance_id in metadata],
        device=mask_hw.device,
        dtype=mask_hw.dtype,
    )
    boxes_xyxy, valid = compute_instance_bboxes_gpu(mask_hw, instance_ids, crop_padding)
    height, width = mask_hw.shape
    grid = _sampling_grid(boxes_xyxy, height, width, size, crop_sampling_mode).to(
        dtype=sample_dtype
    )

    img_batch = img_chw.to(device=device, dtype=sample_dtype).div(255.0).unsqueeze(0)
    img_batch = img_batch.expand(len(metadata), -1, -1, -1)
    boxed = F.grid_sample(
        img_batch,
        grid,
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    )

    mean, std = normalization_parameters or default_normalization_parameters()
    mean = mean.to(device=device, dtype=boxed.dtype).view(1, 3, 1, 1)
    std = std.to(device=device, dtype=boxed.dtype).view(1, 3, 1, 1)
    boxed = (boxed - mean) / std
    if not include_masked:
        if boxed.dtype != patch_dtype:
            boxed = boxed.to(dtype=patch_dtype)
        return boxed, None, metadata, valid

    instance_masks = mask_hw.unsqueeze(0).eq(instance_ids.view(-1, 1, 1))
    mask_patches = F.grid_sample(
        instance_masks.unsqueeze(1).to(dtype=sample_dtype),
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=False,
    ).to(dtype=torch.bool)
    mask_value = -mean / std
    masked = torch.where(mask_patches, boxed, mask_value)
    if boxed.dtype != patch_dtype:
        boxed = boxed.to(dtype=patch_dtype)
        masked = masked.to(dtype=patch_dtype)
    return boxed, masked, metadata, valid


def extract_full_image_patch_gpu(
    img_chw: Tensor,
    size: int,
    normalization_parameters: tuple[Tensor, Tensor] | None = None,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Extract one CLIP-ready full-image patch using OpenCLIP crop semantics."""

    if img_chw.ndim != 3 or img_chw.shape[0] != 3:
        raise ValueError(f"Expected img_chw to be [3, H, W], got shape={tuple(img_chw.shape)}")

    device = img_chw.device
    sample_dtype = torch.float32
    patch_dtype = output_dtype or torch.float32
    height, width = img_chw.shape[-2:]
    box = torch.tensor([[0, 0, width, height]], device=device, dtype=torch.float32)
    grid = _center_square_sampling_grid(box, height, width, size).to(dtype=sample_dtype)
    img_batch = img_chw.to(device=device, dtype=sample_dtype).div(255.0).unsqueeze(0)
    patch = F.grid_sample(
        img_batch,
        grid,
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    )

    mean, std = normalization_parameters or default_normalization_parameters()
    mean = mean.to(device=device, dtype=patch.dtype).view(1, 3, 1, 1)
    std = std.to(device=device, dtype=patch.dtype).view(1, 3, 1, 1)
    patch = (patch - mean) / std
    return patch.to(dtype=patch_dtype) if patch.dtype != patch_dtype else patch


def extract_boxed_and_masked_patch(
    img_chw: Tensor,
    mask_hw: Tensor,
    size: int,
    crop_padding: int = 4,
    normalization_parameters: tuple[Tensor, Tensor] | None = None,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Extract boxed and masked CLIP patches matching semantic_inference behavior."""

    bbox_xyxy = _compute_bbox(mask_hw, crop_padding)
    if bbox_xyxy is None:
        return None, None

    boxed = _extract_patch(img_chw, bbox_xyxy, size)
    mask_patch = _extract_mask_patch(mask_hw, bbox_xyxy, size)

    mean, std = normalization_parameters or default_normalization_parameters()
    boxed = boxed.to(dtype=torch.float32) / 255.0
    mean = mean.to(device=boxed.device, dtype=boxed.dtype).view(3, 1, 1)
    std = std.to(device=boxed.device, dtype=boxed.dtype).view(3, 1, 1)
    boxed = (boxed - mean) / std

    mask_value = -mean / std
    masked = torch.where(mask_patch.unsqueeze(0), boxed, mask_value)
    return boxed, masked


def get_source_track_requests(tracked_msg: TrackedInstances) -> list[tuple[int, int]]:
    """Return (track_id, instance_id) pairs for new source tracks in a keyframe."""

    if not tracked_msg.is_keyframe:
        return []

    return [
        (int(track.track_id), int(track.instance_id))
        for track in tracked_msg.tracks
        if int(track.age) == 0
    ]


def _present_instance_ids(mask_image: np.ndarray | Tensor | None) -> set[int]:
    """Return the non-background instance ids present in a keyframe mask."""

    if mask_image is None:
        return set()

    if isinstance(mask_image, Tensor):
        if mask_image.numel() == 0:
            return set()
        values = torch.unique(mask_image.detach()).to(device="cpu").tolist()
        return {int(v) for v in values if int(v) > 0}

    array = np.asarray(mask_image).reshape(-1)
    if array.size == 0:
        return set()

    return {int(v) for v in np.unique(array) if int(v) > 0}


def get_keyframe_track_requests(
    tracks,
    *,
    mask_image: np.ndarray | Tensor | None,
    is_keyframe: bool,
    keyframe_update_policy: str,
) -> list[tuple[int, int]]:
    """Return track requests eligible for CLIP updates on this completed keyframe."""

    if not is_keyframe:
        return []

    policy = keyframe_update_policy.lower().strip()
    if policy not in OPEN_VOCAB_KEYFRAME_UPDATE_POLICIES:
        raise ValueError(
            "Unsupported open_vocab.keyframe_update_policy="
            f"{keyframe_update_policy!r}"
        )

    present_ids = _present_instance_ids(mask_image)
    if not present_ids:
        return []

    if policy == "source_tracks_only":
        selected_tracks = [track for track in tracks if int(track.age) == 0]
    else:
        selected_tracks = list(tracks)

    return [
        (int(track.track_id), int(track.instance_id))
        for track in selected_tracks
        if int(track.instance_id) in present_ids
    ]


def normalize_open_vocab_feature(feature: Tensor) -> Optional[Tensor]:
    """Return a finite, L2-normalized feature vector or ``None``."""

    vector = feature.reshape(-1).to(dtype=torch.float32)
    if vector.numel() == 0 or not torch.all(torch.isfinite(vector)):
        return None

    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm.item()) <= 1.0e-9:
        return None

    return vector / norm


@dataclass
class OpenVocabTemporalConsistencyConfig(Config):
    """Optional temporal outlier rejection for per-track CLIP observations."""

    enabled: bool = False
    warmup_observations: int = 3
    reject_below_cosine: float = 0.55
    downweight_below_cosine: float = 0.75
    min_downweight: float = 0.25

    def __post_init__(self) -> None:
        self.warmup_observations = max(0, int(self.warmup_observations))
        self.reject_below_cosine = float(self.reject_below_cosine)
        self.downweight_below_cosine = float(self.downweight_below_cosine)
        self.min_downweight = float(min(max(self.min_downweight, 0.0), 1.0))
        if self.downweight_below_cosine < self.reject_below_cosine:
            raise ValueError(
                "open_vocab.temporal_consistency.downweight_below_cosine must be "
                ">= reject_below_cosine"
            )


@dataclass
class OpenVocabObservationQualityConfig(Config):
    """Optional geometry/visibility gates for keyframe CLIP observations."""

    min_bbox_short_side_px: float = 0.0
    min_mask_to_box_fill: float = 0.0
    padding_clamp_full_fraction: float = 1.0
    padding_clamp_min_weight: float = 1.0
    image_boundary_weight: float = 1.0
    min_update_weight: float = 1.0e-6

    def __post_init__(self) -> None:
        self.min_bbox_short_side_px = max(0.0, float(self.min_bbox_short_side_px))
        self.min_mask_to_box_fill = max(0.0, float(self.min_mask_to_box_fill))
        self.padding_clamp_full_fraction = max(
            1.0e-6,
            float(self.padding_clamp_full_fraction),
        )
        self.padding_clamp_min_weight = float(
            min(max(self.padding_clamp_min_weight, 0.0), 1.0)
        )
        self.image_boundary_weight = float(min(max(self.image_boundary_weight, 0.0), 1.0))
        self.min_update_weight = max(0.0, float(self.min_update_weight))


@dataclass
class RunningFeatureAverage:
    """Running weighted average of normalized CLIP observations."""

    sum_feature: Tensor
    num_observations: int = 0
    total_weight: float = 0.0


@dataclass
class FeatureUpdateResult:
    """Result from one attempted per-track open-vocabulary update."""

    feature: Optional[Tensor]
    accepted: bool
    weight: float = 0.0
    temporal_cosine: Optional[float] = None
    reject_reason: str = ""
    temporal_downweighted: bool = False


def update_track_feature_average(
    aggregates: dict[int, RunningFeatureAverage],
    track_id: int,
    feature: Tensor,
    *,
    observation_weight: float = 1.0,
    temporal_consistency: OpenVocabTemporalConsistencyConfig | None = None,
) -> FeatureUpdateResult:
    """Update one track aggregate and return the normalized published feature."""

    normalized_observation = normalize_open_vocab_feature(feature)
    if normalized_observation is None:
        return FeatureUpdateResult(
            feature=None,
            accepted=False,
            reject_reason="invalid_feature",
        )

    weight = float(observation_weight)
    if not np.isfinite(weight) or weight <= 0.0:
        return FeatureUpdateResult(
            feature=None,
            accepted=False,
            reject_reason="nonpositive_weight",
        )

    state = aggregates.get(track_id)
    temporal_cosine = None
    temporal_downweighted = False
    if (
        state is not None
        and temporal_consistency is not None
        and temporal_consistency.enabled
        and state.num_observations >= temporal_consistency.warmup_observations
        and state.sum_feature.numel() == normalized_observation.numel()
        and state.total_weight > 0.0
    ):
        current = normalize_open_vocab_feature(state.sum_feature / float(state.total_weight))
        if current is not None:
            if current.device != normalized_observation.device:
                current = current.to(normalized_observation.device)
            cosine = torch.dot(current, normalized_observation).clamp(-1.0, 1.0)
            temporal_cosine = float(cosine.item())
            if temporal_cosine < temporal_consistency.reject_below_cosine:
                return FeatureUpdateResult(
                    feature=None,
                    accepted=False,
                    weight=0.0,
                    temporal_cosine=temporal_cosine,
                    reject_reason="temporal_cosine",
                )

            if temporal_cosine < temporal_consistency.downweight_below_cosine:
                denom = max(
                    temporal_consistency.downweight_below_cosine
                    - temporal_consistency.reject_below_cosine,
                    1.0e-6,
                )
                alpha = (
                    temporal_cosine - temporal_consistency.reject_below_cosine
                ) / denom
                temporal_weight = temporal_consistency.min_downweight + alpha * (
                    1.0 - temporal_consistency.min_downweight
                )
                temporal_weight = min(max(float(temporal_weight), 0.0), 1.0)
                weight *= temporal_weight
                temporal_downweighted = temporal_weight < 0.999

    if weight <= 0.0:
        return FeatureUpdateResult(
            feature=None,
            accepted=False,
            temporal_cosine=temporal_cosine,
            reject_reason="nonpositive_weight",
        )

    if state is None or state.sum_feature.numel() != normalized_observation.numel():
        state = RunningFeatureAverage(
            sum_feature=normalized_observation.clone() * weight,
            num_observations=1,
            total_weight=weight,
        )
        aggregates[track_id] = state
    else:
        if state.sum_feature.device != normalized_observation.device:
            normalized_observation = normalized_observation.to(state.sum_feature.device)
        state.sum_feature = state.sum_feature + normalized_observation * weight
        state.num_observations += 1
        state.total_weight += weight

    published = state.sum_feature / max(float(state.total_weight), 1.0e-9)
    return FeatureUpdateResult(
        feature=normalize_open_vocab_feature(published),
        accepted=True,
        weight=weight,
        temporal_cosine=temporal_cosine,
        temporal_downweighted=temporal_downweighted,
    )


def update_track_running_average(
    aggregates: dict[int, RunningFeatureAverage],
    track_id: int,
    feature: Tensor,
    *,
    observation_weight: float = 1.0,
    temporal_consistency: OpenVocabTemporalConsistencyConfig | None = None,
) -> Optional[Tensor]:
    """Update one track aggregate and return the normalized published feature."""

    return update_track_feature_average(
        aggregates,
        track_id,
        feature,
        observation_weight=observation_weight,
        temporal_consistency=temporal_consistency,
    ).feature


@dataclass
class OpenVocabBenchmarkConfig(Config):
    """Benchmark-only CLIP timing controls."""

    enabled: bool = False
    synchronize_timers: bool = False


@dataclass
class OpenVocabConfig(Config):
    """Configuration for tracked keyframe open-vocabulary features."""

    enabled: bool = False
    device: str = "cuda"
    model_name: str = "ViT-L-14"
    pretrained: str = "laion2b_s32b_b82k"
    precision: str = "fp16"
    crop_padding: int = 4
    crop_sampling_mode: str = "center_square"
    image_batch_size: int = 0
    normalize_embeddings_before_fusion: bool = False
    use_full_image_feature: bool = False
    boxed_only: bool = True
    boxed_feature_weight: float = 1.0
    masked_feature_weight: float = 1.0
    full_image_feature_weight: float = 1.0
    rgb_cache_size: int = 8
    job_queue_size: int = 8
    topic: str = "tracking/open_vocab_features"
    keyframe_update_policy: str = "all_tracks"
    verbosity: int = 0
    temporal_consistency: OpenVocabTemporalConsistencyConfig = field(
        default_factory=OpenVocabTemporalConsistencyConfig
    )
    observation_quality: OpenVocabObservationQualityConfig = field(
        default_factory=OpenVocabObservationQualityConfig
    )
    benchmark: OpenVocabBenchmarkConfig = field(default_factory=OpenVocabBenchmarkConfig)

    def __post_init__(self) -> None:
        self.keyframe_update_policy = self.keyframe_update_policy.lower().strip()
        if self.keyframe_update_policy not in OPEN_VOCAB_KEYFRAME_UPDATE_POLICIES:
            raise ValueError(
                "Unsupported open_vocab.keyframe_update_policy="
                f"{self.keyframe_update_policy!r}"
            )
        self.crop_sampling_mode = self.crop_sampling_mode.lower().strip()
        if self.crop_sampling_mode not in OPEN_VOCAB_CROP_SAMPLING_MODES:
            raise ValueError(
                "Unsupported open_vocab.crop_sampling_mode="
                f"{self.crop_sampling_mode!r}"
            )
        self.boxed_feature_weight = max(0.0, float(self.boxed_feature_weight))
        self.masked_feature_weight = max(0.0, float(self.masked_feature_weight))
        self.full_image_feature_weight = max(0.0, float(self.full_image_feature_weight))
        if self.boxed_only and self.boxed_feature_weight <= 0.0:
            raise ValueError(
                "open_vocab.boxed_feature_weight must be positive when boxed_only is enabled"
            )
        if (
            self.boxed_feature_weight <= 0.0
            and self.masked_feature_weight <= 0.0
            and (
                not self.use_full_image_feature
                or self.full_image_feature_weight <= 0.0
            )
        ):
            raise ValueError("At least one enabled open_vocab feature weight must be positive")


@dataclass
class OpenVocabJob:
    """One queued keyframe CLIP request."""

    header: object
    frame_index: int
    source_tracks: list[tuple[int, int]]
    rgb: np.ndarray
    mask_image: np.ndarray


class OpenVocabEncoder:
    """Thin OpenCLIP wrapper for image and text encoding."""

    def __init__(self, config: OpenVocabConfig):
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "Open-vocabulary features require the 'open_clip_torch' package"
            ) from exc

        self._device = torch.device(config.device)
        precision = config.precision.lower().strip()
        if self._device.type == "cpu" and precision != "fp32":
            raise ValueError("Open-vocabulary CPU encoding requires precision='fp32'")
        self._model, _, _ = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=config.pretrained,
            precision=precision,
            device=self._device,
        )
        self._tokenizer = open_clip.get_tokenizer(config.model_name)
        self._model = self._model.to(self._device)
        if self._device.type == "cpu":
            self._model = self._model.float()
        self._model.eval()
        image_size = self._model.visual.image_size
        self.input_size = int(image_size[0] if isinstance(image_size, tuple) else image_size)
        self.normalization_parameters = normalization_parameters_from_preprocess_cfg(
            getattr(self._model.visual, "preprocess_cfg", None)
        )
        self.output_dtype = torch.float32
        self.input_dtype = open_clip.get_input_dtype(precision) or torch.float32
        self.encoder_id = f"openclip:{config.model_name}:{config.pretrained}"
        if precision != "fp32":
            self.encoder_id = f"{self.encoder_id}:precision={precision}"

    @property
    def device(self) -> torch.device:
        """Return the encoder device."""

        return self._device

    @torch.inference_mode()
    def encode(self, patches: Tensor) -> Tensor:
        """Encode a batch of CLIP-ready patches without normalization."""

        if patches.numel() == 0:
            return torch.empty((0, 0), device=self._device, dtype=self.output_dtype)

        embeddings = self._model.encode_image(
            patches.to(device=self._device, dtype=self.input_dtype),
            normalize=False,
        )
        return embeddings.to(dtype=self.output_dtype)

    @torch.inference_mode()
    def encode_text(self, prompts: str | list[str]) -> Tensor:
        """Encode one or more text prompts without normalization."""

        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompt_list:
            return torch.empty((0, 0), device=self._device, dtype=self.output_dtype)

        tokens = self._tokenizer(prompt_list)
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens)

        embeddings = self._model.encode_text(
            tokens.to(device=self._device),
            normalize=False,
        )
        embeddings = embeddings.to(dtype=self.output_dtype)
        return embeddings[0] if isinstance(prompts, str) else embeddings

    @torch.inference_mode()
    def warmup(self, batch_size: int = 1) -> None:
        """Run a dummy forward pass so model setup does not hit the first job."""

        dummy = torch.zeros(
            (max(1, batch_size), 3, self.input_size, self.input_size),
            device=self._device,
            dtype=torch.float32,
        )
        _ = self.encode(dummy)
        _ = self.encode_text(["background"])
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()


def encode_patches_in_batches(
    encoder: OpenVocabEncoder,
    patches: Tensor,
    image_batch_size: int,
) -> tuple[Tensor, int]:
    """Encode CLIP image patches in either one batch or fixed-size chunks."""

    num_patches = int(patches.shape[0]) if patches.ndim > 0 else 0
    if num_patches == 0:
        return encoder.encode(patches), 0

    batch_size = int(image_batch_size)
    if batch_size <= 0 or batch_size >= num_patches:
        return encoder.encode(patches), 1

    features = []
    for start in range(0, num_patches, batch_size):
        features.append(encoder.encode(patches[start : start + batch_size]))

    return torch.cat(features, dim=0), len(features)


def fuse_open_vocab_image_features(
    boxed_features: Tensor,
    masked_features: Tensor,
    *,
    config: OpenVocabConfig,
    full_image_feature: Tensor | None = None,
) -> Tensor:
    """Fuse one observation's boxed/masked/global CLIP embeddings."""

    components: list[tuple[Tensor, float]] = []
    if config.boxed_feature_weight > 0.0:
        components.append((boxed_features, config.boxed_feature_weight))
    if not config.boxed_only and config.masked_feature_weight > 0.0:
        components.append((masked_features, config.masked_feature_weight))
    if (
        config.use_full_image_feature
        and full_image_feature is not None
        and config.full_image_feature_weight > 0.0
    ):
        full = full_image_feature.reshape(1, -1).expand(boxed_features.shape[0], -1)
        components.append((full, config.full_image_feature_weight))

    if not components:
        return torch.empty((boxed_features.shape[0], 0), device=boxed_features.device)

    fused = None
    total_weight = 0.0
    for features, weight in components:
        component = features.to(dtype=torch.float32)
        if config.normalize_embeddings_before_fusion:
            component = F.normalize(component, p=2, dim=-1)
        weighted = component * float(weight)
        fused = weighted if fused is None else fused + weighted
        total_weight += float(weight)

    assert fused is not None
    fused = fused / max(total_weight, 1.0e-9)
    if config.normalize_embeddings_before_fusion:
        fused = F.normalize(fused, p=2, dim=-1)
    return fused


def observation_quality_weight(
    stats: dict[str, Tensor],
    index: int,
    config: OpenVocabObservationQualityConfig,
) -> tuple[float, str, dict[str, float | bool]]:
    """Return geometry/visibility update weight, rejection reason, and diagnostics."""

    valid = bool(stats["valid"][index].detach().cpu().item())
    mask_area = float(stats["mask_area_px"][index].detach().cpu().item())
    bbox_short_side = float(stats["bbox_short_side_px"][index].detach().cpu().item())
    mask_to_box_fill = float(stats["mask_to_box_fill"][index].detach().cpu().item())
    raw_mask_to_box_fill = float(
        stats["raw_mask_to_box_fill"][index].detach().cpu().item()
    )
    padding_clamp_fraction = float(
        stats["padding_clamp_fraction"][index].detach().cpu().item()
    )
    touches_boundary = bool(
        stats["touches_image_boundary"][index].detach().cpu().item()
    )
    diagnostics: dict[str, float | bool] = {
        "valid": valid,
        "mask_area_px": mask_area,
        "bbox_short_side_px": bbox_short_side,
        "mask_to_box_fill": mask_to_box_fill,
        "raw_mask_to_box_fill": raw_mask_to_box_fill,
        "padding_clamp_fraction": padding_clamp_fraction,
        "touches_image_boundary": touches_boundary,
    }

    if not valid:
        return 0.0, "missing_instance", diagnostics
    if (
        config.min_bbox_short_side_px > 0.0
        and bbox_short_side < config.min_bbox_short_side_px
    ):
        return 0.0, "bbox_short_side", diagnostics
    if config.min_mask_to_box_fill > 0.0 and mask_to_box_fill < config.min_mask_to_box_fill:
        return 0.0, "mask_to_box_fill", diagnostics

    weight = 1.0
    if config.padding_clamp_min_weight < 1.0:
        fraction = min(max(padding_clamp_fraction, 0.0), config.padding_clamp_full_fraction)
        alpha = fraction / config.padding_clamp_full_fraction
        clamp_weight = config.padding_clamp_min_weight + alpha * (
            1.0 - config.padding_clamp_min_weight
        )
        weight *= min(max(float(clamp_weight), config.padding_clamp_min_weight), 1.0)
    if touches_boundary and config.image_boundary_weight < 1.0:
        weight *= config.image_boundary_weight

    if weight < config.min_update_weight:
        return 0.0, "quality_weight", diagnostics
    return weight, "", diagnostics


class OpenVocabMetricsRecorder:
    """Persist per-frame and aggregate CLIP timing metrics into the run directory."""

    def __init__(self, node, encoder_id: str):
        self._node = node
        self._lock = threading.Lock()
        self._enabled = False
        self._job_events: list[dict] = []
        self._events_path: Optional[Path] = None
        self._summary_path: Optional[Path] = None
        self._summary = {
            "encoder_id": encoder_id,
            "warmup_ms": None,
            "jobs_seen": 0,
            "jobs_published": 0,
            "jobs_without_features": 0,
            "jobs_dropped": 0,
            "jobs_failed": 0,
            "total_source_tracks_requested": 0,
            "total_source_tracks_encoded": 0,
            "total_model_forward_calls": 0,
            "total_boxed_forward_calls": 0,
            "total_masked_forward_calls": 0,
            "total_full_image_forward_calls": 0,
            "total_preprocess_ms": 0.0,
            "total_boxed_inference_ms": 0.0,
            "total_masked_inference_ms": 0.0,
            "total_full_image_inference_ms": 0.0,
            "total_inference_ms": 0.0,
            "total_publish_ms": 0.0,
            "total_job_ms": 0.0,
            "max_source_tracks_requested": 0,
            "max_source_tracks_encoded": 0,
            "max_source_tracks_published": 0,
            "max_total_job_ms": 0.0,
            "total_observations_accepted": 0,
            "total_observations_rejected_geometry": 0,
            "total_observations_rejected_temporal": 0,
            "total_observations_downweighted": 0,
        }

        run_dir = os.getenv("INSTANCE_TRACKING_OUTPUT_RUN_DIR", "").strip()
        if not run_dir:
            return

        output_dir = Path(os.path.expanduser(run_dir)) / "open_vocab"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = output_dir / "inference_events.jsonl"
        self._summary_path = output_dir / "inference_summary.json"
        self._enabled = True
        self._write_summary_locked()
        self._node.get_logger().info(
            f"Saving open-vocabulary inference metrics under {output_dir}"
        )

    def record_warmup(self, duration_ms: float) -> None:
        if not self._enabled:
            return

        with self._lock:
            self._summary["warmup_ms"] = float(duration_ms)
            self._append_event_locked({"event": "warmup", "warmup_ms": float(duration_ms)})
            self._write_summary_locked()

    def record_drop(self, stamp_ns: int) -> None:
        if not self._enabled:
            return

        with self._lock:
            self._summary["jobs_dropped"] += 1
            self._append_event_locked(
                {
                    "event": "drop",
                    "stamp_ns": int(stamp_ns),
                }
            )
            self._write_summary_locked()

    def record_failure(self, frame_index: int, stamp_ns: int, error: str) -> None:
        if not self._enabled:
            return

        with self._lock:
            self._summary["jobs_failed"] += 1
            self._append_event_locked(
                {
                    "event": "failure",
                    "frame_index": int(frame_index),
                    "stamp_ns": int(stamp_ns),
                    "error": error,
                }
            )
            self._write_summary_locked()

    def record_job(
        self,
        *,
        frame_index: int,
        stamp_ns: int,
        source_tracks_requested: int,
        source_tracks_encoded: int,
        model_forward_calls: int,
        boxed_forward_calls: int,
        masked_forward_calls: int,
        image_batch_size: int,
        preprocess_ms: float,
        boxed_inference_ms: float,
        masked_inference_ms: float,
        publish_ms: float,
        total_job_ms: float,
        preprocess_max_memory_allocated_mb: Optional[float],
        preprocess_max_memory_reserved_mb: Optional[float],
        boxed_inference_max_memory_allocated_mb: Optional[float],
        boxed_inference_max_memory_reserved_mb: Optional[float],
        masked_inference_max_memory_allocated_mb: Optional[float],
        masked_inference_max_memory_reserved_mb: Optional[float],
        total_job_max_memory_allocated_mb: Optional[float],
        total_job_max_memory_reserved_mb: Optional[float],
        published: bool,
        source_tracks_published: Optional[int] = None,
        full_image_forward_calls: int = 0,
        full_image_inference_ms: float = 0.0,
        observations_accepted: int = 0,
        observations_rejected_geometry: int = 0,
        observations_rejected_temporal: int = 0,
        observations_downweighted: int = 0,
        mean_observation_update_weight: Optional[float] = None,
        min_observation_update_weight: Optional[float] = None,
        mean_temporal_cosine: Optional[float] = None,
        min_temporal_cosine: Optional[float] = None,
        mean_mask_to_box_fill: Optional[float] = None,
        min_mask_to_box_fill: Optional[float] = None,
        mean_bbox_short_side_px: Optional[float] = None,
        min_bbox_short_side_px: Optional[float] = None,
        mean_padding_clamp_fraction: Optional[float] = None,
        min_padding_clamp_fraction: Optional[float] = None,
        image_boundary_touch_count: int = 0,
    ) -> None:
        if not self._enabled:
            return

        inference_ms = boxed_inference_ms + masked_inference_ms
        inference_ms += full_image_inference_ms
        encoded_tracks = max(0, int(source_tracks_encoded))
        published_tracks = (
            max(0, int(source_tracks_published))
            if source_tracks_published is not None
            else encoded_tracks
        )
        job_end_wall_time_s = time.time()
        job_end_monotonic_s = time.monotonic()
        total_job_s = float(total_job_ms) / 1000.0

        event = {
            "event": "job",
            "frame_index": int(frame_index),
            "stamp_ns": int(stamp_ns),
            "is_keyframe": True,
            "job_start_wall_time_s": job_end_wall_time_s - total_job_s,
            "job_start_monotonic_s": job_end_monotonic_s - total_job_s,
            "job_end_wall_time_s": job_end_wall_time_s,
            "job_end_monotonic_s": job_end_monotonic_s,
            "source_tracks_requested": int(source_tracks_requested),
            "source_tracks_encoded": encoded_tracks,
            "source_tracks_published": published_tracks,
            "model_forward_calls": int(model_forward_calls),
            "boxed_forward_calls": int(boxed_forward_calls),
            "masked_forward_calls": int(masked_forward_calls),
            "full_image_forward_calls": int(full_image_forward_calls),
            "image_batch_size": int(image_batch_size),
            "preprocess_ms": float(preprocess_ms),
            "boxed_inference_ms": float(boxed_inference_ms),
            "masked_inference_ms": float(masked_inference_ms),
            "full_image_inference_ms": float(full_image_inference_ms),
            "total_inference_ms": float(inference_ms),
            "publish_ms": float(publish_ms),
            "total_job_ms": float(total_job_ms),
            "preprocess_max_memory_allocated_mb": preprocess_max_memory_allocated_mb,
            "preprocess_max_memory_reserved_mb": preprocess_max_memory_reserved_mb,
            "boxed_inference_max_memory_allocated_mb": (
                boxed_inference_max_memory_allocated_mb
            ),
            "boxed_inference_max_memory_reserved_mb": (
                boxed_inference_max_memory_reserved_mb
            ),
            "masked_inference_max_memory_allocated_mb": (
                masked_inference_max_memory_allocated_mb
            ),
            "masked_inference_max_memory_reserved_mb": (
                masked_inference_max_memory_reserved_mb
            ),
            "total_job_max_memory_allocated_mb": total_job_max_memory_allocated_mb,
            "total_job_max_memory_reserved_mb": total_job_max_memory_reserved_mb,
            "published": bool(published),
            "mean_total_inference_ms_per_encoded_track": (
                float(inference_ms) / float(encoded_tracks) if encoded_tracks else 0.0
            ),
            "mean_total_job_ms_per_encoded_track": (
                float(total_job_ms) / float(encoded_tracks) if encoded_tracks else 0.0
            ),
            "observations_accepted": int(observations_accepted),
            "observations_rejected_geometry": int(observations_rejected_geometry),
            "observations_rejected_temporal": int(observations_rejected_temporal),
            "observations_downweighted": int(observations_downweighted),
            "image_boundary_touch_count": int(image_boundary_touch_count),
        }
        optional_metrics = {
            "mean_observation_update_weight": mean_observation_update_weight,
            "min_observation_update_weight": min_observation_update_weight,
            "mean_temporal_cosine": mean_temporal_cosine,
            "min_temporal_cosine": min_temporal_cosine,
            "mean_mask_to_box_fill": mean_mask_to_box_fill,
            "min_mask_to_box_fill": min_mask_to_box_fill,
            "mean_bbox_short_side_px": mean_bbox_short_side_px,
            "min_bbox_short_side_px": min_bbox_short_side_px,
            "mean_padding_clamp_fraction": mean_padding_clamp_fraction,
            "min_padding_clamp_fraction": min_padding_clamp_fraction,
        }
        event.update(
            {
                key: float(value)
                for key, value in optional_metrics.items()
                if value is not None and np.isfinite(float(value))
            }
        )

        with self._lock:
            self._summary["jobs_seen"] += 1
            if published:
                self._summary["jobs_published"] += 1
            else:
                self._summary["jobs_without_features"] += 1
            self._summary["total_source_tracks_requested"] += int(source_tracks_requested)
            self._summary["total_source_tracks_encoded"] += encoded_tracks
            self._summary["total_observations_accepted"] += int(observations_accepted)
            self._summary["total_observations_rejected_geometry"] += int(
                observations_rejected_geometry
            )
            self._summary["total_observations_rejected_temporal"] += int(
                observations_rejected_temporal
            )
            self._summary["total_observations_downweighted"] += int(
                observations_downweighted
            )
            self._summary["total_model_forward_calls"] += int(model_forward_calls)
            self._summary["total_boxed_forward_calls"] += int(boxed_forward_calls)
            self._summary["total_masked_forward_calls"] += int(masked_forward_calls)
            self._summary["total_full_image_forward_calls"] += int(full_image_forward_calls)
            self._summary["total_preprocess_ms"] += float(preprocess_ms)
            self._summary["total_boxed_inference_ms"] += float(boxed_inference_ms)
            self._summary["total_masked_inference_ms"] += float(masked_inference_ms)
            self._summary["total_full_image_inference_ms"] += float(
                full_image_inference_ms
            )
            self._summary["total_inference_ms"] += float(inference_ms)
            self._summary["total_publish_ms"] += float(publish_ms)
            self._summary["total_job_ms"] += float(total_job_ms)
            self._summary["max_source_tracks_requested"] = max(
                int(self._summary["max_source_tracks_requested"]),
                int(source_tracks_requested),
            )
            self._summary["max_source_tracks_encoded"] = max(
                int(self._summary["max_source_tracks_encoded"]),
                encoded_tracks,
            )
            self._summary["max_source_tracks_published"] = max(
                int(self._summary["max_source_tracks_published"]),
                published_tracks,
            )
            self._summary["max_total_job_ms"] = max(
                float(self._summary["max_total_job_ms"]),
                float(total_job_ms),
            )
            self._job_events.append(event)
            self._append_event_locked(event)
            self._write_summary_locked()

    def close(self) -> None:
        if not self._enabled:
            return

        with self._lock:
            self._write_summary_locked()

    def _append_event_locked(self, event: dict) -> None:
        if self._events_path is None:
            return

        event.setdefault("record_wall_time_s", time.time())
        event.setdefault("record_monotonic_s", time.monotonic())

        with self._events_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(event, sort_keys=True) + "\n")

    def _write_summary_locked(self) -> None:
        if self._summary_path is None:
            return

        jobs_seen = int(self._summary["jobs_seen"])
        total_encoded = int(self._summary["total_source_tracks_encoded"])
        summary = dict(self._summary)
        summary["mean_source_tracks_requested_per_job"] = (
            float(summary["total_source_tracks_requested"]) / float(jobs_seen)
            if jobs_seen
            else 0.0
        )
        summary["mean_source_tracks_encoded_per_job"] = (
            float(summary["total_source_tracks_encoded"]) / float(jobs_seen)
            if jobs_seen
            else 0.0
        )
        summary["mean_preprocess_ms_per_job"] = (
            float(summary["total_preprocess_ms"]) / float(jobs_seen) if jobs_seen else 0.0
        )
        summary["mean_boxed_inference_ms_per_job"] = (
            float(summary["total_boxed_inference_ms"]) / float(jobs_seen)
            if jobs_seen
            else 0.0
        )
        summary["mean_masked_inference_ms_per_job"] = (
            float(summary["total_masked_inference_ms"]) / float(jobs_seen)
            if jobs_seen
            else 0.0
        )
        summary["mean_total_inference_ms_per_job"] = (
            float(summary["total_inference_ms"]) / float(jobs_seen) if jobs_seen else 0.0
        )
        summary["mean_publish_ms_per_job"] = (
            float(summary["total_publish_ms"]) / float(jobs_seen) if jobs_seen else 0.0
        )
        summary["mean_total_job_ms_per_job"] = (
            float(summary["total_job_ms"]) / float(jobs_seen) if jobs_seen else 0.0
        )
        summary["mean_total_inference_ms_per_encoded_track"] = (
            float(summary["total_inference_ms"]) / float(total_encoded)
            if total_encoded
            else 0.0
        )
        summary["mean_total_job_ms_per_encoded_track"] = (
            float(summary["total_job_ms"]) / float(total_encoded) if total_encoded else 0.0
        )
        summary["job_stats"] = build_numeric_summary(
            self._job_events,
            numeric_fields=(
                "source_tracks_requested",
                "source_tracks_encoded",
                "source_tracks_published",
                "model_forward_calls",
                "boxed_forward_calls",
                "masked_forward_calls",
                "full_image_forward_calls",
                "image_batch_size",
                "preprocess_ms",
                "boxed_inference_ms",
                "masked_inference_ms",
                "full_image_inference_ms",
                "total_inference_ms",
                "publish_ms",
                "total_job_ms",
                "mean_total_inference_ms_per_encoded_track",
                "mean_total_job_ms_per_encoded_track",
                "observations_accepted",
                "observations_rejected_geometry",
                "observations_rejected_temporal",
                "observations_downweighted",
                "mean_observation_update_weight",
                "min_observation_update_weight",
                "mean_temporal_cosine",
                "min_temporal_cosine",
                "mean_mask_to_box_fill",
                "min_mask_to_box_fill",
                "mean_bbox_short_side_px",
                "min_bbox_short_side_px",
                "mean_padding_clamp_fraction",
                "min_padding_clamp_fraction",
                "image_boundary_touch_count",
                "preprocess_max_memory_allocated_mb",
                "preprocess_max_memory_reserved_mb",
                "boxed_inference_max_memory_allocated_mb",
                "boxed_inference_max_memory_reserved_mb",
                "masked_inference_max_memory_allocated_mb",
                "masked_inference_max_memory_reserved_mb",
                "total_job_max_memory_allocated_mb",
                "total_job_max_memory_reserved_mb",
            ),
        )
        all_frequencies = summary["job_stats"].get("all_frames", {}).get(
            "frequencies_hz", {}
        )
        summary["clip_job_frequency_hz"] = float(
            all_frequencies.get("clip_job_frequency_hz", 0.0)
        )
        summary["clip_inference_frequency_hz"] = float(
            all_frequencies.get("clip_inference_frequency_hz", 0.0)
        )

        with self._summary_path.open("w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, sort_keys=True)


class OpenVocabFeatureWorker:
    """Asynchronous in-process open-vocabulary feature extraction for tracked objects."""

    def __init__(self, node, config: OpenVocabConfig):
        self._node = node
        self._config = config
        self._node.context.on_shutdown(self.stop)

        self._started = False
        self._should_shutdown = False
        self._job_queue: queue.Queue[OpenVocabJob] = queue.Queue(
            maxsize=max(1, config.job_queue_size)
        )

        self._pub = None
        self._image_service = None
        self._text_service = None
        self._encoder = None
        self._encoder_lock = threading.Lock()
        self._metrics = None
        self._aggregates: dict[int, RunningFeatureAverage] = {}

        if not config.enabled:
            self._node.get_logger().info("Open-vocabulary keyframe track features disabled")
            return

        self._encoder = OpenVocabEncoder(config)
        self._metrics = OpenVocabMetricsRecorder(self._node, self._encoder.encoder_id)
        self._node.get_logger().info(
            "Warming up open-vocabulary keyframe encoder "
            f"{self._encoder.encoder_id}"
        )
        warmup_start = time.perf_counter()
        self._encoder.warmup()
        if self._metrics is not None:
            self._metrics.record_warmup((time.perf_counter() - warmup_start) * 1000.0)
        self._pub = self._node.create_publisher(
            OpenVocabTrackFeaturesMsg,
            config.topic,
            1,
        )
        self._text_service = self._node.create_service(
            EncodeOpenVocabTextSrv,
            "tracking/open_vocab/encode_text",
            self._handle_encode_text,
        )
        self._image_service = self._node.create_service(
            EncodeOpenVocabImageSrv,
            "tracking/open_vocab/encode_image",
            self._handle_encode_image,
        )
        self._thread = threading.Thread(target=self._do_work, name="open_vocab_features")
        self._thread.start()
        self._started = True
        self._node.get_logger().info(
            "Open-vocabulary keyframe features enabled with "
            f"{self._encoder.encoder_id} "
            f"(keyframe_update_policy={self._config.keyframe_update_policy})"
        )

    def stop(self) -> None:
        """Stop the background worker."""

        if not self._started:
            return

        self._should_shutdown = True
        self._thread.join()
        self._started = False
        self._should_shutdown = False
        self._image_service = None
        self._text_service = None
        if self._metrics is not None:
            self._metrics.close()

    @staticmethod
    def _stamp_to_ns(header) -> int:
        return rclpy.time.Time.from_msg(header.stamp).nanoseconds

    def _can_sample_cuda_memory(self) -> bool:
        return (
            self._config.benchmark.enabled
            and
            self._encoder is not None
            and self._encoder.device.type == "cuda"
            and torch.cuda.is_available()
        )

    def _synchronize_device(self) -> None:
        if (
            self._config.benchmark.enabled
            and self._config.benchmark.synchronize_timers
            and self._encoder is not None
            and self._encoder.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self._encoder.device)

    def _reset_cuda_peak(self) -> None:
        if self._can_sample_cuda_memory():
            torch.cuda.reset_peak_memory_stats(self._encoder.device)

    def _get_cuda_peak_mb(self) -> tuple[Optional[float], Optional[float]]:
        if not self._can_sample_cuda_memory():
            return None, None

        return (
            float(torch.cuda.max_memory_allocated(self._encoder.device)) / 1024.0**2,
            float(torch.cuda.max_memory_reserved(self._encoder.device)) / 1024.0**2,
        )

    def submit(
        self,
        *,
        header,
        frame_index: int,
        tracks,
        rgb: np.ndarray,
        mask_image: np.ndarray,
        is_keyframe: bool,
    ) -> None:
        """Queue one completed keyframe for CLIP processing."""

        if not self._started or not is_keyframe:
            return

        source_tracks = get_keyframe_track_requests(
            tracks,
            mask_image=mask_image,
            is_keyframe=is_keyframe,
            keyframe_update_policy=self._config.keyframe_update_policy,
        )
        if not source_tracks:
            return

        job = OpenVocabJob(
            header=copy.deepcopy(header),
            frame_index=int(frame_index),
            source_tracks=source_tracks,
            rgb=np.ascontiguousarray(rgb),
            mask_image=np.ascontiguousarray(mask_image),
        )
        stamp_ns = self._stamp_to_ns(job.header)
        try:
            self._job_queue.put_nowait(job)
        except queue.Full:
            if self._metrics is not None:
                self._metrics.record_drop(stamp_ns)
            self._node.get_logger().warn(
                "Dropping open-vocabulary job because the worker queue is full "
                f"(queue={self._config.job_queue_size}, frame_index={frame_index}, stamp_ns={stamp_ns})"
            )

    def _do_work(self) -> None:
        while True:
            if self._should_shutdown and self._job_queue.empty():
                return
            try:
                job = self._job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._process_job(job)
            except Exception as exc:  # pragma: no cover - defensive logging
                import traceback

                if self._metrics is not None:
                    self._metrics.record_failure(
                        job.frame_index,
                        self._stamp_to_ns(job.header),
                        str(exc),
                    )
                self._node.get_logger().error(
                    "Open-vocabulary feature extraction failed: "
                    f"{exc}\n{traceback.format_exc()}"
                )

    def _handle_encode_text(self, request, response):
        """Encode one text prompt using the tracker-owned CLIP model."""

        if self._encoder is None:
            response.encoder_id = ""
            response.feature = []
            return response

        prompt = request.prompt.strip()
        if not prompt:
            response.encoder_id = self._encoder.encoder_id
            response.feature = []
            return response

        try:
            with self._encoder_lock:
                feature = self._encoder.encode_text(prompt)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._node.get_logger().error(
                f"Failed to encode open-vocabulary text prompt '{prompt}': {exc}"
            )
            response.encoder_id = ""
            response.feature = []
            return response

        response.encoder_id = self._encoder.encoder_id
        response.feature = (
            feature.detach().cpu().to(dtype=torch.float32).contiguous().numpy().tolist()
        )
        return response

    def _handle_encode_image(self, request, response):
        """Encode one compressed RGB image using the tracker-owned CLIP model."""

        response.success = False
        response.message = ""
        response.encoder_id = ""
        response.feature = []
        if self._encoder is None:
            response.message = "Open-vocabulary encoder is unavailable"
            return response

        response.encoder_id = self._encoder.encoder_id
        try:
            rgb = Conversions.bridge.compressed_imgmsg_to_cv2(
                request.image,
                desired_encoding="rgb8",
            )
            rgb = np.asarray(rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
                raise ValueError(
                    f"decoded image must have shape [H, W, 3], got {rgb.shape}"
                )

            rgb_tensor = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .to(device=self._encoder.device)
                .permute(2, 0, 1)
            )
            image_batch = extract_full_image_patch_gpu(
                rgb_tensor,
                self._encoder.input_size,
                normalization_parameters=self._encoder.normalization_parameters,
                output_dtype=self._encoder.input_dtype,
            )
            with self._encoder_lock:
                features = self._encoder.encode(image_batch)
            if features.ndim != 2 or features.shape[0] != 1 or features.shape[1] == 0:
                raise ValueError(
                    "image encoder returned an invalid feature batch with "
                    f"shape={tuple(features.shape)}"
                )

            feature = features[0]
            response.feature = (
                feature.detach()
                .cpu()
                .to(dtype=torch.float32)
                .contiguous()
                .numpy()
                .tolist()
            )
            response.success = True
            return response
        except Exception as exc:  # pragma: no cover - error detail varies by decoder
            response.message = f"Failed to encode open-vocabulary query image: {exc}"
            self._node.get_logger().error(response.message)
            return response

    def _process_job(self, job: OpenVocabJob) -> None:
        assert self._encoder is not None
        assert self._pub is not None

        stamp_ns = self._stamp_to_ns(job.header)
        source_tracks_requested = len(job.source_tracks)
        image_batch_size = int(self._config.image_batch_size)
        job_start = time.perf_counter()
        peak_allocated_candidates: list[float] = []
        peak_reserved_candidates: list[float] = []

        def update_total_peak(allocated_mb: Optional[float], reserved_mb: Optional[float]) -> None:
            if allocated_mb is not None:
                peak_allocated_candidates.append(float(allocated_mb))
            if reserved_mb is not None:
                peak_reserved_candidates.append(float(reserved_mb))

        device = self._encoder.device
        self._synchronize_device()
        self._reset_cuda_peak()
        rgb_tensor = torch.from_numpy(job.rgb).to(device=device).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(job.mask_image.astype(np.int32, copy=False)).to(
            device=device
        )

        preprocess_start = time.perf_counter()
        instance_ids = torch.tensor(
            [instance_id for _, instance_id in job.source_tracks],
            device=mask_tensor.device,
            dtype=mask_tensor.dtype,
        )
        geometry_stats = compute_observation_geometry_stats_gpu(
            mask_tensor,
            instance_ids,
            crop_padding=self._config.crop_padding,
            crop_sampling_mode=self._config.crop_sampling_mode,
        )
        geometry_stats_cpu = {
            key: value.detach().cpu() for key, value in geometry_stats.items()
        }
        boxed_batch, masked_batch, metadata, _valid = extract_boxed_and_masked_patches_gpu(
            rgb_tensor,
            mask_tensor,
            job.source_tracks,
            self._encoder.input_size,
            crop_padding=self._config.crop_padding,
            crop_sampling_mode=self._config.crop_sampling_mode,
            normalization_parameters=self._encoder.normalization_parameters,
            output_dtype=self._encoder.input_dtype,
            include_masked=not self._config.boxed_only,
        )
        full_image_batch = None
        if self._config.use_full_image_feature:
            full_image_batch = extract_full_image_patch_gpu(
                rgb_tensor,
                self._encoder.input_size,
                normalization_parameters=self._encoder.normalization_parameters,
                output_dtype=self._encoder.input_dtype,
            )
        self._synchronize_device()
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        (
            preprocess_allocated_mb,
            preprocess_reserved_mb,
        ) = self._get_cuda_peak_mb()
        update_total_peak(preprocess_allocated_mb, preprocess_reserved_mb)

        if not metadata:
            if self._metrics is not None:
                self._metrics.record_job(
                    frame_index=job.frame_index,
                    stamp_ns=stamp_ns,
                    source_tracks_requested=source_tracks_requested,
                    source_tracks_encoded=0,
                    source_tracks_published=0,
                    model_forward_calls=0,
                    boxed_forward_calls=0,
                    masked_forward_calls=0,
                    image_batch_size=image_batch_size,
                    preprocess_ms=preprocess_ms,
                    boxed_inference_ms=0.0,
                    masked_inference_ms=0.0,
                    publish_ms=0.0,
                    preprocess_max_memory_allocated_mb=preprocess_allocated_mb,
                    preprocess_max_memory_reserved_mb=preprocess_reserved_mb,
                    boxed_inference_max_memory_allocated_mb=None,
                    boxed_inference_max_memory_reserved_mb=None,
                    masked_inference_max_memory_allocated_mb=None,
                    masked_inference_max_memory_reserved_mb=None,
                    total_job_max_memory_allocated_mb=(
                        max(peak_allocated_candidates) if peak_allocated_candidates else None
                    ),
                    total_job_max_memory_reserved_mb=(
                        max(peak_reserved_candidates) if peak_reserved_candidates else None
                    ),
                    total_job_ms=(time.perf_counter() - job_start) * 1000.0,
                    published=False,
                )
            return

        self._synchronize_device()
        self._reset_cuda_peak()
        boxed_start = time.perf_counter()
        with self._encoder_lock:
            boxed_features, boxed_forward_calls = encode_patches_in_batches(
                self._encoder,
                boxed_batch,
                image_batch_size,
            )
        self._synchronize_device()
        boxed_ms = (time.perf_counter() - boxed_start) * 1000.0
        boxed_allocated_mb, boxed_reserved_mb = self._get_cuda_peak_mb()
        update_total_peak(boxed_allocated_mb, boxed_reserved_mb)

        masked_features = boxed_features
        masked_forward_calls = 0
        masked_ms = 0.0
        masked_allocated_mb = None
        masked_reserved_mb = None
        if not self._config.boxed_only:
            assert masked_batch is not None
            self._synchronize_device()
            self._reset_cuda_peak()
            masked_start = time.perf_counter()
            with self._encoder_lock:
                masked_features, masked_forward_calls = encode_patches_in_batches(
                    self._encoder,
                    masked_batch,
                    image_batch_size,
                )
            self._synchronize_device()
            masked_ms = (time.perf_counter() - masked_start) * 1000.0
            masked_allocated_mb, masked_reserved_mb = self._get_cuda_peak_mb()
        full_image_feature = None
        full_image_forward_calls = 0
        full_image_ms = 0.0
        if full_image_batch is not None:
            self._synchronize_device()
            self._reset_cuda_peak()
            full_image_start = time.perf_counter()
            with self._encoder_lock:
                full_image_features, full_image_forward_calls = encode_patches_in_batches(
                    self._encoder,
                    full_image_batch,
                    image_batch_size,
                )
            self._synchronize_device()
            full_image_ms = (time.perf_counter() - full_image_start) * 1000.0
            full_image_allocated_mb, full_image_reserved_mb = self._get_cuda_peak_mb()
            update_total_peak(full_image_allocated_mb, full_image_reserved_mb)
            if full_image_features.numel() > 0:
                full_image_feature = full_image_features[0]

        features = fuse_open_vocab_image_features(
            boxed_features,
            masked_features,
            config=self._config,
            full_image_feature=full_image_feature,
        )
        self._synchronize_device()
        combined_allocated_mb, combined_reserved_mb = self._get_cuda_peak_mb()
        update_total_peak(masked_allocated_mb, masked_reserved_mb)
        update_total_peak(combined_allocated_mb, combined_reserved_mb)

        payload = []
        observations_accepted = 0
        observations_rejected_geometry = 0
        observations_rejected_temporal = 0
        observations_downweighted = 0
        update_weights: list[float] = []
        temporal_cosines: list[float] = []
        mask_to_box_fills: list[float] = []
        bbox_short_sides: list[float] = []
        padding_clamp_fractions: list[float] = []
        image_boundary_touch_count = 0
        for (track_id, instance_id), feature in zip(metadata, features, strict=True):
            idx = len(mask_to_box_fills)
            quality_weight, quality_reason, quality_diagnostics = observation_quality_weight(
                geometry_stats_cpu,
                idx,
                self._config.observation_quality,
            )
            mask_to_box_fills.append(float(quality_diagnostics["mask_to_box_fill"]))
            bbox_short_sides.append(float(quality_diagnostics["bbox_short_side_px"]))
            padding_clamp_fractions.append(
                float(quality_diagnostics["padding_clamp_fraction"])
            )
            if bool(quality_diagnostics["touches_image_boundary"]):
                image_boundary_touch_count += 1
            if quality_reason:
                observations_rejected_geometry += 1
                continue

            update_result = update_track_feature_average(
                self._aggregates,
                track_id,
                feature,
                observation_weight=quality_weight,
                temporal_consistency=self._config.temporal_consistency,
            )
            if update_result.temporal_cosine is not None:
                temporal_cosines.append(update_result.temporal_cosine)
            if not update_result.accepted or update_result.feature is None:
                if update_result.reject_reason == "temporal_cosine":
                    observations_rejected_temporal += 1
                else:
                    observations_rejected_geometry += 1
                continue
            observations_accepted += 1
            update_weights.append(update_result.weight)
            if quality_weight < 0.999 or update_result.temporal_downweighted:
                observations_downweighted += 1
            payload.append((track_id, instance_id, update_result.feature))

        model_forward_calls = (
            boxed_forward_calls + masked_forward_calls + full_image_forward_calls
        )

        def mean_or_none(values: list[float]) -> Optional[float]:
            return float(np.mean(values)) if values else None

        def min_or_none(values: list[float]) -> Optional[float]:
            return float(np.min(values)) if values else None

        if not payload:
            if self._metrics is not None:
                self._metrics.record_job(
                    frame_index=job.frame_index,
                    stamp_ns=stamp_ns,
                    source_tracks_requested=source_tracks_requested,
                    source_tracks_encoded=len(metadata),
                    source_tracks_published=0,
                    model_forward_calls=model_forward_calls,
                    boxed_forward_calls=boxed_forward_calls,
                    masked_forward_calls=masked_forward_calls,
                    full_image_forward_calls=full_image_forward_calls,
                    image_batch_size=image_batch_size,
                    preprocess_ms=preprocess_ms,
                    boxed_inference_ms=boxed_ms,
                    masked_inference_ms=masked_ms,
                    full_image_inference_ms=full_image_ms,
                    publish_ms=0.0,
                    preprocess_max_memory_allocated_mb=preprocess_allocated_mb,
                    preprocess_max_memory_reserved_mb=preprocess_reserved_mb,
                    boxed_inference_max_memory_allocated_mb=boxed_allocated_mb,
                    boxed_inference_max_memory_reserved_mb=boxed_reserved_mb,
                    masked_inference_max_memory_allocated_mb=masked_allocated_mb,
                    masked_inference_max_memory_reserved_mb=masked_reserved_mb,
                    total_job_max_memory_allocated_mb=(
                        max(peak_allocated_candidates) if peak_allocated_candidates else None
                    ),
                    total_job_max_memory_reserved_mb=(
                        max(peak_reserved_candidates) if peak_reserved_candidates else None
                    ),
                    total_job_ms=(time.perf_counter() - job_start) * 1000.0,
                    published=False,
                    observations_accepted=observations_accepted,
                    observations_rejected_geometry=observations_rejected_geometry,
                    observations_rejected_temporal=observations_rejected_temporal,
                    observations_downweighted=observations_downweighted,
                    mean_observation_update_weight=mean_or_none(update_weights),
                    min_observation_update_weight=min_or_none(update_weights),
                    mean_temporal_cosine=mean_or_none(temporal_cosines),
                    min_temporal_cosine=min_or_none(temporal_cosines),
                    mean_mask_to_box_fill=mean_or_none(mask_to_box_fills),
                    min_mask_to_box_fill=min_or_none(mask_to_box_fills),
                    mean_bbox_short_side_px=mean_or_none(bbox_short_sides),
                    min_bbox_short_side_px=min_or_none(bbox_short_sides),
                    mean_padding_clamp_fraction=mean_or_none(padding_clamp_fractions),
                    min_padding_clamp_fraction=min_or_none(padding_clamp_fractions),
                    image_boundary_touch_count=image_boundary_touch_count,
                )
            return

        publish_start = time.perf_counter()
        msg = Conversions.to_open_vocab_track_features_msg(
            job.header,
            job.frame_index,
            self._encoder.encoder_id,
            payload,
        )
        self._pub.publish(msg)
        publish_ms = (time.perf_counter() - publish_start) * 1000.0
        if self._metrics is not None:
            self._metrics.record_job(
                frame_index=job.frame_index,
                stamp_ns=stamp_ns,
                source_tracks_requested=source_tracks_requested,
                source_tracks_encoded=len(metadata),
                source_tracks_published=len(payload),
                model_forward_calls=model_forward_calls,
                boxed_forward_calls=boxed_forward_calls,
                masked_forward_calls=masked_forward_calls,
                full_image_forward_calls=full_image_forward_calls,
                image_batch_size=image_batch_size,
                preprocess_ms=preprocess_ms,
                boxed_inference_ms=boxed_ms,
                masked_inference_ms=masked_ms,
                full_image_inference_ms=full_image_ms,
                publish_ms=publish_ms,
                preprocess_max_memory_allocated_mb=preprocess_allocated_mb,
                preprocess_max_memory_reserved_mb=preprocess_reserved_mb,
                boxed_inference_max_memory_allocated_mb=boxed_allocated_mb,
                boxed_inference_max_memory_reserved_mb=boxed_reserved_mb,
                masked_inference_max_memory_allocated_mb=masked_allocated_mb,
                masked_inference_max_memory_reserved_mb=masked_reserved_mb,
                total_job_max_memory_allocated_mb=(
                    max(peak_allocated_candidates) if peak_allocated_candidates else None
                ),
                total_job_max_memory_reserved_mb=(
                    max(peak_reserved_candidates) if peak_reserved_candidates else None
                ),
                total_job_ms=(time.perf_counter() - job_start) * 1000.0,
                published=True,
                observations_accepted=observations_accepted,
                observations_rejected_geometry=observations_rejected_geometry,
                observations_rejected_temporal=observations_rejected_temporal,
                observations_downweighted=observations_downweighted,
                mean_observation_update_weight=mean_or_none(update_weights),
                min_observation_update_weight=min_or_none(update_weights),
                mean_temporal_cosine=mean_or_none(temporal_cosines),
                min_temporal_cosine=min_or_none(temporal_cosines),
                mean_mask_to_box_fill=mean_or_none(mask_to_box_fills),
                min_mask_to_box_fill=min_or_none(mask_to_box_fills),
                mean_bbox_short_side_px=mean_or_none(bbox_short_sides),
                min_bbox_short_side_px=min_or_none(bbox_short_sides),
                mean_padding_clamp_fraction=mean_or_none(padding_clamp_fractions),
                min_padding_clamp_fraction=min_or_none(padding_clamp_fractions),
                image_boundary_touch_count=image_boundary_touch_count,
            )


OpenVocabFeaturePipeline = OpenVocabFeatureWorker
