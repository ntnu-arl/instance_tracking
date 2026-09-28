"""Helpers for recording and replaying cached tracker observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


CACHE_VERSION = 1
CACHE_ROOT_NAME = "cache"
FRAME_FEATURES_DIRNAME = "frame_features"
KEYFRAME_DETECTIONS_DIRNAME = "keyframe_detections"
MANIFEST_FILENAME = "manifest.json"


def frame_name(frame_index: int) -> str:
    """Format a frame index using the recorder naming convention."""

    return f"{int(frame_index):06d}"


def compute_boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """Compute xyxy boxes for a stack of binary masks."""

    masks_arr = np.asarray(masks).astype(bool, copy=False)
    if masks_arr.ndim != 3:
        raise ValueError(f"Expected masks [N, H, W], got {masks_arr.shape}")

    boxes = np.zeros((masks_arr.shape[0], 4), dtype=np.int32)
    for idx, mask in enumerate(masks_arr):
        ys, xs = np.nonzero(mask)
        if ys.size == 0 or xs.size == 0:
            continue
        boxes[idx] = np.array(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
            dtype=np.int32,
        )
    return boxes


@dataclass(frozen=True)
class ObservationCachePaths:
    """Resolved cache paths within a saved run directory."""

    run_dir: Path
    cache_root: Path
    frame_features_dir: Path
    keyframe_detections_dir: Path
    manifest_path: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "ObservationCachePaths":
        resolved = Path(run_dir).expanduser().resolve()
        cache_root = resolved / CACHE_ROOT_NAME
        return cls(
            run_dir=resolved,
            cache_root=cache_root,
            frame_features_dir=cache_root / FRAME_FEATURES_DIRNAME,
            keyframe_detections_dir=cache_root / KEYFRAME_DETECTIONS_DIRNAME,
            manifest_path=cache_root / MANIFEST_FILENAME,
        )


def write_manifest(paths: ObservationCachePaths) -> None:
    """Persist a minimal manifest describing the cache layout."""

    paths.cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": CACHE_VERSION,
        "frame_features_dir": FRAME_FEATURES_DIRNAME,
        "keyframe_detections_dir": KEYFRAME_DETECTIONS_DIRNAME,
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class CachedObservationLoader:
    """Loads prerecorded features and keyframe detections for replay."""

    def __init__(self, run_dir: str | Path, device: torch.device):
        self.paths = ObservationCachePaths.from_run_dir(run_dir)
        self.device = device

        if not self.paths.frame_features_dir.is_dir():
            raise FileNotFoundError(
                "Missing cached frame features directory: "
                f"{self.paths.frame_features_dir}"
            )
        if not self.paths.keyframe_detections_dir.is_dir():
            raise FileNotFoundError(
                "Missing cached keyframe detections directory: "
                f"{self.paths.keyframe_detections_dir}"
            )

    def _feature_path(self, frame_index: int) -> Path:
        return self.paths.frame_features_dir / f"{frame_name(frame_index)}.npz"

    def _detection_path(self, frame_index: int) -> Path:
        return self.paths.keyframe_detections_dir / f"{frame_name(frame_index)}.npz"

    @staticmethod
    def _validate_image_shape(
        stored_shape: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]],
        path: Path,
    ) -> None:
        if stored_shape is None or image_shape is None:
            return

        expected = tuple(int(v) for v in np.asarray(stored_shape).reshape(-1)[:2])
        actual = tuple(int(v) for v in image_shape[:2])
        if expected != actual:
            raise ValueError(
                f"Cached observations at '{path}' expect image shape {expected}, "
                f"but current input is {actual}"
            )

    def load_frame_features(
        self,
        frame_index: int,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """Load cached DINO features for a frame."""

        path = self._feature_path(frame_index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached features for frame {frame_index}: {path}")

        with np.load(path) as data:
            self._validate_image_shape(data.get("image_shape"), image_shape, path)
            features = np.asarray(data["features"], dtype=np.float32)

        return torch.from_numpy(features).to(self.device)

    def load_keyframe_detections(
        self,
        frame_index: int,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """Load cached refined keyframe detection masks."""

        path = self._detection_path(frame_index)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing cached keyframe detections for frame {frame_index}: {path}"
            )

        with np.load(path) as data:
            self._validate_image_shape(data.get("image_shape"), image_shape, path)
            masks = np.asarray(data["masks"]).astype(bool, copy=False)

        return torch.from_numpy(masks).to(self.device)
