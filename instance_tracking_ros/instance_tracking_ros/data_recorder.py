"""Asynchronous data recorder for frames, depth, masks, and visualizations.

Designed to run on a background thread so that the main tracking loop
remains real-time. In ordinary recording mode frames may be dropped when the
queue is full; replay-cache recording switches to blocking writes so the saved
cache stays lossless.
"""

import os
import pathlib
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from spark_config import Config

from instance_tracking.observation_cache import (
    ObservationCachePaths,
    compute_boxes_from_masks,
    write_manifest,
)


@dataclass
class DataRecorderConfig(Config):
    """Configuration for DataRecorder."""

    enabled: bool = False
    save_depth: bool = True  # Save raw aligned depth maps (.npy)
    save_depth_viz: bool = True  # Save human-readable depth visualizations
    save_features: bool = False  # Save DINOv3 features for keyframes (FP16 NPZ)
    save_probabilities: bool = False  # Save winning-label probability maps (FP16 NPZ)
    save_raw_probabilities: bool = False  # Save pre-normalization patch argmax prob (FP16 NPZ)
    save_observation_cache: bool = False  # Save all-frame features + raw keyframe detections
    save_prototypes: bool = True  # Save keyframe prototype snapshots
    output_root: str = "/hostwork/hydra_ws/data/instance_tracking"
    queue_size: int = 16
    video_fps: float = 20.0
    image_format: str = "png"
    viz_format: str = "jpg"

    @classmethod
    def load(cls, filepath):
        return Config.load(cls, filepath)


class DataRecorder:
    """Background writer for frames, depth maps, masks, visualizations, and video."""

    def __init__(self, node, config: DataRecorderConfig):
        self.enabled = bool(config.enabled)
        self._logger = node.get_logger()
        self._should_stop = False
        self._thread: Optional[threading.Thread] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._dropped = 0
        self._config_snapshot_saved = False

        if not self.enabled:
            return

        explicit_output_dir = os.getenv("INSTANCE_TRACKING_OUTPUT_RUN_DIR", "").strip()
        if explicit_output_dir:
            self._output_dir = pathlib.Path(os.path.expanduser(explicit_output_dir))
            self._output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_root = pathlib.Path(os.path.expanduser(config.output_root))
            self._output_dir = self._create_unique_output_dir(output_root)
        self._frame_dir = self._output_dir / "frames"
        self._depth_dir = self._output_dir / "depth"
        self._depth_viz_dir = self._output_dir / "depth_viz"
        self._keyframe_mask_dir = self._output_dir / "masks" / "keyframes"  # Final keyframe masks
        self._propagated_mask_dir = self._output_dir / "masks" / "propagated"  # Final propagated masks
        self._keyframe_fastsam_mask_dir = (
            self._output_dir / "masks" / "keyframes_fastsam_cleaned"
        )
        self._viz_dir = self._output_dir / "viz"
        self._keyframe_fastsam_viz_dir = self._viz_dir / "keyframes_fastsam_cleaned"
        self._features_dir = self._output_dir / "features"
        self._patch_probability_dir = self._output_dir / "probabilities" / "patch_argmax"
        self._patch_keyframe_probability_dir = self._patch_probability_dir / "keyframes"
        self._patch_propagated_probability_dir = self._patch_probability_dir / "propagated"
        self._final_probability_dir = self._output_dir / "probabilities" / "final_argmax"
        self._final_keyframe_probability_dir = self._final_probability_dir / "keyframes"
        self._final_propagated_probability_dir = self._final_probability_dir / "propagated"
        self._patch_raw_probability_dir = self._output_dir / "probabilities" / "patch_argmax_raw"
        self._patch_raw_propagated_probability_dir = (
            self._patch_raw_probability_dir / "propagated"
        )
        self._prototypes_dir = self._output_dir / "prototypes"
        self._video_path = self._output_dir / "viz.mp4"
        self._save_depth = config.save_depth
        self._save_depth_viz = config.save_depth_viz
        self._save_features = config.save_features
        self._save_probabilities = config.save_probabilities
        self._save_raw_probabilities = config.save_raw_probabilities
        self._save_observation_cache = config.save_observation_cache
        self._save_prototypes = config.save_prototypes
        self._cache_paths = ObservationCachePaths.from_run_dir(self._output_dir)

        dirs_to_create = [
            self._frame_dir,
            self._keyframe_mask_dir,
            self._propagated_mask_dir,
            self._keyframe_fastsam_mask_dir,
            self._viz_dir,
            self._keyframe_fastsam_viz_dir,
        ]
        if self._save_depth:
            dirs_to_create.append(self._depth_dir)
        if self._save_depth_viz:
            dirs_to_create.append(self._depth_viz_dir)
        if self._save_features:
            dirs_to_create.append(self._features_dir)
        if self._save_probabilities:
            dirs_to_create.extend(
                [
                    self._patch_keyframe_probability_dir,
                    self._patch_propagated_probability_dir,
                    self._final_keyframe_probability_dir,
                    self._final_propagated_probability_dir,
                ]
            )
        if self._save_raw_probabilities:
            dirs_to_create.append(self._patch_raw_propagated_probability_dir)
        if self._save_observation_cache:
            dirs_to_create.extend(
                [
                    self._cache_paths.cache_root,
                    self._cache_paths.frame_features_dir,
                    self._cache_paths.keyframe_detections_dir,
                ]
            )
        if self._save_prototypes:
            dirs_to_create.append(self._prototypes_dir)
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

        if self._save_observation_cache:
            write_manifest(self._cache_paths)

        self._queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self._config = config

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._logger.info(
            f"DataRecorder enabled. Saving to {self._output_dir} (queue={config.queue_size})."
        )

    def _create_unique_output_dir(self, output_root: pathlib.Path) -> pathlib.Path:
        """Create a unique run_* directory without merging concurrent runs."""

        output_root.mkdir(parents=True, exist_ok=True)
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
        base_suffix = f"_job{slurm_job_id}" if slurm_job_id else ""

        for attempt in range(1000):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            candidate_name = f"run_{timestamp}{base_suffix}"
            if attempt:
                candidate_name = f"{candidate_name}_{attempt:03d}"
            candidate = output_root / candidate_name
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                continue

        raise RuntimeError(f"Failed to create unique run directory under {output_root}")

    def save_config(self, config_obj):
        """Persist the run configuration once.

        Args:
            config_obj: Serializable config (spark_config Config dataclass)
        """
        if (not self.enabled) or self._config_snapshot_saved:
            return

        cfg_path = self._output_dir / "config.yaml"
        try:
            config_obj.save(cfg_path)
            self._config_snapshot_saved = True
            self._logger.info(f"Saved run config to {cfg_path}")
        except Exception as exc:
            self._logger.warn(f"Failed to write run config: {exc}")

    def record(
        self,
        frame_index: int,
        frame_rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        viz_rgb: np.ndarray,
        features=None,  # Tensor [H', W', D] or None
        frame_features=None,  # Tensor [H', W', D] or None, saved for every frame when cache recording enabled
        is_keyframe: bool = False,
        prototypes=None,  # Optional tuple: (instance_ids, prototypes, colors)
        keyframe_fastsam_mask: Optional[np.ndarray] = None,
        keyframe_detection_masks=None,  # Tensor or None, raw refined keyframe detections
        patch_argmax_confidence=None,  # Tensor [H', W'] or None
        final_argmax_confidence=None,  # Tensor [H, W] or None
        patch_argmax_confidence_raw=None,  # Tensor [H', W'] or None, pre-normalization
    ):
        """Queue frame data for saving.

        Args:
            frame_index: Incremental frame index from tracker
            frame_rgb: Original RGB frame (H, W, 3)
            depth: Depth image aligned to frame_rgb
            mask: Primary output instance mask (H, W) integer ids
            viz_rgb: Visualization RGB image (H, W, 3)
            features: Optional DINOv3 feature grid [H', W', D] for keyframes
            is_keyframe: Whether this is a keyframe (for feature saving)
            keyframe_fastsam_mask: Optional cleaned FastSAM keyframe mask [H, W]
                before down/up-sampling.
        """

        if not self.enabled:
            return

        frame_name = self._frame_name(frame_index)
        
        # Only convert and include features for keyframes if feature saving is enabled
        feat_to_save = None
        if self._save_features and is_keyframe and features is not None:
            # Convert tensor to numpy FP16 for storage efficiency
            feat_to_save = features.cpu().numpy().astype(np.float16)

        frame_features_to_save = None
        if self._save_observation_cache and frame_features is not None:
            frame_features_to_save = frame_features.cpu().numpy().astype(np.float16)

        keyframe_detection_masks_to_save = None
        if (
            self._save_observation_cache
            and is_keyframe
            and keyframe_detection_masks is not None
        ):
            keyframe_detection_masks_to_save = (
                keyframe_detection_masks.cpu().numpy().astype(np.uint8)
            )

        patch_argmax_confidence_to_save = None
        if self._save_probabilities and patch_argmax_confidence is not None:
            patch_argmax_confidence_to_save = (
                patch_argmax_confidence.cpu().numpy().astype(np.float16)
            )

        final_argmax_confidence_to_save = None
        if self._save_probabilities and final_argmax_confidence is not None:
            final_argmax_confidence_to_save = (
                final_argmax_confidence.cpu().numpy().astype(np.float16)
            )

        patch_argmax_confidence_raw_to_save = None
        if self._save_raw_probabilities and patch_argmax_confidence_raw is not None:
            patch_argmax_confidence_raw_to_save = (
                patch_argmax_confidence_raw.cpu().numpy().astype(np.float16)
            )

        try:
            payload = (
                frame_name,
                np.ascontiguousarray(frame_rgb),
                np.ascontiguousarray(depth),
                np.ascontiguousarray(mask.astype(np.uint16)),
                np.ascontiguousarray(viz_rgb),
                feat_to_save,
                frame_features_to_save,
                is_keyframe,
                prototypes,
                (
                    np.ascontiguousarray(keyframe_fastsam_mask.astype(np.uint16))
                    if keyframe_fastsam_mask is not None
                    else None
                ),
                keyframe_detection_masks_to_save,
                patch_argmax_confidence_to_save,
                final_argmax_confidence_to_save,
                patch_argmax_confidence_raw_to_save,
            )
            if self._save_observation_cache:
                # Cache generation is meant to be replayable, so avoid silent drops.
                self._queue.put(payload)
            else:
                self._queue.put_nowait(payload)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:  # Log occasionally
                self._logger.warn(
                    f"DataRecorder queue full. Dropped {self._dropped} frames so far."
                )

    def stop(self):
        if not self.enabled:
            return

        self._should_stop = True
        if self._thread is not None:
            self._thread.join()

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if self._dropped > 0:
            self._logger.warn(f"DataRecorder dropped {self._dropped} frames (queue limit).")

    def _init_video_writer(self, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            str(self._video_path), fourcc, float(self._config.video_fps), (width, height)
        )
        if not self._video_writer.isOpened():
            self._logger.error(f"Failed to open video writer at {self._video_path}")
            self._video_writer = None

    @staticmethod
    def _frame_name(frame_index: int) -> str:
        return f"{int(frame_index):06d}"

    @staticmethod
    def _depth_to_viz_bgr(depth: np.ndarray) -> np.ndarray:
        """Convert depth image to a robust, human-readable BGR visualization."""
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim != 2 or depth_arr.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        depth_float = depth_arr.astype(np.float32, copy=False)
        valid = np.isfinite(depth_float) & (depth_float > 0.0)
        normalized = np.zeros(depth_float.shape, dtype=np.uint8)

        if np.any(valid):
            valid_depths = depth_float[valid]
            lo = float(np.percentile(valid_depths, 2.0))
            hi = float(np.percentile(valid_depths, 98.0))
            if hi <= lo:
                lo = float(valid_depths.min())
                hi = float(valid_depths.max())

            if hi > lo:
                clipped = np.clip(depth_float, lo, hi)
                normalized[valid] = (
                    (clipped[valid] - lo) * (255.0 / (hi - lo))
                ).astype(np.uint8)
            else:
                normalized[valid] = 255

        depth_viz_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        depth_viz_bgr[~valid] = (32, 32, 32)
        return depth_viz_bgr

    @staticmethod
    def _mask_to_viz_bgr(mask: np.ndarray) -> np.ndarray:
        """Convert uint16 instance-id mask to deterministic BGR visualization."""
        mask_arr = np.asarray(mask)
        if mask_arr.ndim != 2 or mask_arr.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        labels = mask_arr.astype(np.int32, copy=False)
        viz = np.full((labels.shape[0], labels.shape[1], 3), 24, dtype=np.uint8)
        instance_ids = np.unique(labels)
        for instance_id in instance_ids:
            if instance_id <= 0:
                continue
            hue = int((int(instance_id) * 53) % 180)
            hsv = np.uint8([[[hue, 220, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            viz[labels == instance_id] = bgr
        return viz

    @staticmethod
    def _mask_to_track_color_viz_bgr(mask: np.ndarray, prototypes) -> Optional[np.ndarray]:
        """Colorize mask with persistent track RGB colors from prototypes payload."""
        if prototypes is None:
            return None

        try:
            instance_ids, _, colors = prototypes
        except Exception:
            return None

        if instance_ids is None or colors is None:
            return None

        mask_arr = np.asarray(mask)
        ids_arr = np.asarray(instance_ids).reshape(-1)
        colors_arr = np.asarray(colors)
        if mask_arr.ndim != 2 or mask_arr.size == 0:
            return None
        if ids_arr.size == 0 or colors_arr.ndim != 2 or colors_arr.shape[1] < 3:
            return None

        labels = mask_arr.astype(np.int32, copy=False)
        viz = np.full((labels.shape[0], labels.shape[1], 3), 24, dtype=np.uint8)
        for instance_id, color_rgb in zip(ids_arr, colors_arr):
            inst = int(instance_id)
            if inst <= 0:
                continue
            rgb = np.asarray(color_rgb[:3], dtype=np.uint8)
            viz[labels == inst] = rgb[::-1]  # RGB -> BGR

        return viz

    def _write_sample(
        self,
        frame_name,
        frame_rgb,
        depth,
        mask,
        viz_rgb,
        features,
        frame_features,
        is_keyframe,
        prototypes,
        keyframe_fastsam_mask,
        keyframe_detection_masks,
        patch_argmax_confidence,
        final_argmax_confidence,
        patch_argmax_confidence_raw,
    ):
        frame_path = self._frame_dir / f"{frame_name}.{self._config.image_format}"
        # Save the primary output masks to keyframe/propagated folders.
        mask_dir = self._keyframe_mask_dir if is_keyframe else self._propagated_mask_dir
        mask_path = mask_dir / f"{frame_name}.png"
        viz_path = self._viz_dir / f"{frame_name}.{self._config.viz_format}"

        # Convert RGB -> BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        viz_bgr = cv2.cvtColor(viz_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(frame_path), frame_bgr)
        if self._save_depth:
            depth_path = self._depth_dir / f"{frame_name}.npy"
            np.save(str(depth_path), depth)
        if self._save_depth_viz:
            depth_viz_path = self._depth_viz_dir / f"{frame_name}.png"
            depth_viz_bgr = self._depth_to_viz_bgr(depth)
            cv2.imwrite(str(depth_viz_path), depth_viz_bgr)
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(viz_path), viz_bgr)

        if keyframe_fastsam_mask is not None:
            keyframe_fastsam_mask_path = self._keyframe_fastsam_mask_dir / f"{frame_name}.png"
            keyframe_fastsam_viz_path = (
                self._keyframe_fastsam_viz_dir / f"{frame_name}.{self._config.viz_format}"
            )
            cv2.imwrite(str(keyframe_fastsam_mask_path), keyframe_fastsam_mask)
            keyframe_fastsam_viz_bgr = self._mask_to_track_color_viz_bgr(
                keyframe_fastsam_mask, prototypes
            )
            if keyframe_fastsam_viz_bgr is None:
                keyframe_fastsam_viz_bgr = self._mask_to_viz_bgr(keyframe_fastsam_mask)
            cv2.imwrite(
                str(keyframe_fastsam_viz_path),
                keyframe_fastsam_viz_bgr,
            )

        # Save features if present (keyframes only)
        if features is not None:
            feat_path = self._features_dir / f"{frame_name}.npz"
            np.savez_compressed(str(feat_path), features=features)

        if frame_features is not None:
            frame_features_path = self._cache_paths.frame_features_dir / f"{frame_name}.npz"
            np.savez_compressed(
                str(frame_features_path),
                features=frame_features,
                image_shape=np.asarray(frame_rgb.shape[:2], dtype=np.int32),
            )

        if keyframe_detection_masks is not None:
            detections_path = self._cache_paths.keyframe_detections_dir / f"{frame_name}.npz"
            np.savez_compressed(
                str(detections_path),
                masks=keyframe_detection_masks.astype(np.uint8, copy=False),
                boxes=compute_boxes_from_masks(keyframe_detection_masks),
                image_shape=np.asarray(frame_rgb.shape[:2], dtype=np.int32),
            )

        if self._save_prototypes and prototypes is not None:
            proto_path = self._prototypes_dir / f"{frame_name}.npz"
            ids, protos, colors = prototypes
            np.savez_compressed(
                str(proto_path),
                instance_ids=ids,
                prototypes=protos,
                colors=colors,
            )

        probability_dirs = (
            (
                self._patch_keyframe_probability_dir,
                self._final_keyframe_probability_dir,
            )
            if is_keyframe
            else (
                self._patch_propagated_probability_dir,
                self._final_propagated_probability_dir,
            )
        )
        if patch_argmax_confidence is not None:
            patch_probability_path = probability_dirs[0] / f"{frame_name}.npz"
            np.savez_compressed(
                str(patch_probability_path),
                probabilities=patch_argmax_confidence,
                image_shape=np.asarray(frame_rgb.shape[:2], dtype=np.int32),
            )
        if final_argmax_confidence is not None:
            final_probability_path = probability_dirs[1] / f"{frame_name}.npz"
            np.savez_compressed(
                str(final_probability_path),
                probabilities=final_argmax_confidence,
                image_shape=np.asarray(frame_rgb.shape[:2], dtype=np.int32),
            )

        if not is_keyframe and patch_argmax_confidence_raw is not None:
            raw_path = self._patch_raw_propagated_probability_dir / f"{frame_name}.npz"
            np.savez_compressed(
                str(raw_path),
                probabilities=patch_argmax_confidence_raw,
                image_shape=np.asarray(frame_rgb.shape[:2], dtype=np.int32),
            )

        if self._video_writer is None:
            h, w = viz_bgr.shape[:2]
            self._init_video_writer(w, h)

        if self._video_writer is not None:
            self._video_writer.write(viz_bgr)

    def _worker(self):
        while not self._should_stop:
            try:
                (
                    frame_name,
                    frame_rgb,
                    depth,
                    mask,
                    viz_rgb,
                    features,
                    frame_features,
                    is_keyframe,
                    prototypes,
                    keyframe_fastsam_mask,
                    keyframe_detection_masks,
                    patch_argmax_confidence,
                    final_argmax_confidence,
                    patch_argmax_confidence_raw,
                ) = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._write_sample(
                frame_name,
                frame_rgb,
                depth,
                mask,
                viz_rgb,
                features,
                frame_features,
                is_keyframe,
                prototypes,
                keyframe_fastsam_mask,
                keyframe_detection_masks,
                patch_argmax_confidence,
                final_argmax_confidence,
                patch_argmax_confidence_raw,
            )

        # Flush remaining queue if stop was requested
        while not self._queue.empty():
            try:
                (
                    frame_name,
                    frame_rgb,
                    depth,
                    mask,
                    viz_rgb,
                    features,
                    frame_features,
                    is_keyframe,
                    prototypes,
                    keyframe_fastsam_mask,
                    keyframe_detection_masks,
                    patch_argmax_confidence,
                    final_argmax_confidence,
                    patch_argmax_confidence_raw,
                ) = self._queue.get_nowait()
            except queue.Empty:
                break
            self._write_sample(
                frame_name,
                frame_rgb,
                depth,
                mask,
                viz_rgb,
                features,
                frame_features,
                is_keyframe,
                prototypes,
                keyframe_fastsam_mask,
                keyframe_detection_masks,
                patch_argmax_confidence,
                final_argmax_confidence,
                patch_argmax_confidence_raw,
            )

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
