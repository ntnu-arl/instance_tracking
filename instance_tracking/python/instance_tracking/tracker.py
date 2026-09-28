"""Main instance tracker orchestrating segmentation, features, and propagation."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from spark_config import Config, config_field
from torch import Tensor

from instance_tracking.models.feature_extractor import DINOv3Config, DINOv3FeatureExtractor
from instance_tracking.models.instance_segmenter import FastSAMConfig, InstanceSegmenter
from instance_tracking.models.mask_refinement import MaskRefinementConfig, MaskRefiner
from instance_tracking.observation_cache import CachedObservationLoader
from instance_tracking.models.preprocessing import DINOv3Preprocessing, compute_target_dimensions
from instance_tracking.models.propagation import LabelPropagator, PropagationConfig, postprocess_probs
from instance_tracking.tracking.track import Track
from instance_tracking.tracking.track_manager import TrackManager, TrackManagerConfig

logger = logging.getLogger(__name__)


__all__ = [
    "InstanceTracker",
    "TrackerConfig",
    "TrackerResult",
]


@dataclass
class TrackerResult:
    """Result from processing a frame.
    
    Attributes:
        masks: Instance mask image [H, W] with pixel values = instance_id
        tracks: List of active Track objects
        is_keyframe: Whether this was a keyframe (instance segmenter ran)
        frame_index: Index of this frame in the sequence
        camera_frame_index: Optional zero-based index in the original camera stream
        features: Optional DINOv3 feature grid [H', W', D] (only for keyframes)
        frame_features: Optional DINOv3 feature grid [H', W', D] for the
            current frame (used for observation-cache recording/replay)
        keyframe_fastsam_mask: Optional cleaned FastSAM keyframe mask image [H, W]
            before downsampling/propagation (ids are persistent track instance_id)
        keyframe_detection_masks: Optional refined keyframe detection masks
            [N, H, W] before association to persistent track ids
        patch_argmax_confidence: Optional float image [H', W'] containing the
            winning foreground-label probability at feature resolution.
            Background patches are 0.
        final_argmax_confidence: Optional float image [H, W] containing the
            winning foreground-label probability after upsampling/refinement.
            Background pixels are 0.
        patch_argmax_confidence_raw: Optional float image [H', W'] containing the
            winning foreground-label probability at feature resolution BEFORE
            per-track min-max normalization. Only set for propagated frames.
            Background patches are 0.
    """

    masks: Tensor
    tracks: List[Track]
    is_keyframe: bool
    frame_index: int
    camera_frame_index: Optional[int] = None
    features: Optional[Tensor] = None
    frame_features: Optional[Tensor] = None
    keyframe_fastsam_mask: Optional[Tensor] = None
    keyframe_detection_masks: Optional[Tensor] = None
    patch_argmax_confidence: Optional[Tensor] = None
    final_argmax_confidence: Optional[Tensor] = None
    patch_argmax_confidence_raw: Optional[Tensor] = None
    runtime_profile: Optional["TrackerRuntimeProfile"] = None
    
    @property
    def num_instances(self) -> int:
        """Get number of active instances."""
        return len(self.tracks)
    
    def cpu(self) -> "TrackerResult":
        """Move result to CPU."""
        return TrackerResult(
            masks=self.masks.cpu(),
            tracks=self.tracks,
            is_keyframe=self.is_keyframe,
            frame_index=self.frame_index,
            camera_frame_index=self.camera_frame_index,
            features=self.features.cpu() if self.features is not None else None,
            frame_features=(
                self.frame_features.cpu() if self.frame_features is not None else None
            ),
            keyframe_fastsam_mask=(
                self.keyframe_fastsam_mask.cpu() if self.keyframe_fastsam_mask is not None else None
            ),
            keyframe_detection_masks=(
                self.keyframe_detection_masks.cpu()
                if self.keyframe_detection_masks is not None
                else None
            ),
            patch_argmax_confidence=(
                self.patch_argmax_confidence.cpu()
                if self.patch_argmax_confidence is not None
                else None
            ),
            final_argmax_confidence=(
                self.final_argmax_confidence.cpu()
                if self.final_argmax_confidence is not None
                else None
            ),
            patch_argmax_confidence_raw=(
                self.patch_argmax_confidence_raw.cpu()
                if self.patch_argmax_confidence_raw is not None
                else None
            ),
            runtime_profile=self.runtime_profile,
        )


@dataclass
class TrackerBenchmarkConfig(Config):
    """Benchmark-only tracker timing controls."""

    enabled: bool = False
    synchronize_timers: bool = False


@dataclass
class TrackerRuntimeProfile:
    """Per-frame runtime and GPU allocator peaks for the tracker pipeline."""

    dino_ms: float = 0.0
    fastsam_ms: float = 0.0
    keyframe_mask_refine_ms: float = 0.0
    keyframe_track_update_ms: float = 0.0
    keyframe_label_image_ms: float = 0.0
    propagation_ms: float = 0.0
    propagation_update_ms: float = 0.0
    probability_tensor_ms: float = 0.0
    probability_confidence_ms: float = 0.0
    probability_refine_ms: float = 0.0
    probability_outputs_ms: float = 0.0
    tracker_total_ms: float = 0.0
    dino_max_memory_allocated_mb: Optional[float] = None
    dino_max_memory_reserved_mb: Optional[float] = None
    fastsam_max_memory_allocated_mb: Optional[float] = None
    fastsam_max_memory_reserved_mb: Optional[float] = None
    propagation_max_memory_allocated_mb: Optional[float] = None
    propagation_max_memory_reserved_mb: Optional[float] = None
    tracker_max_memory_allocated_mb: Optional[float] = None
    tracker_max_memory_reserved_mb: Optional[float] = None


@dataclass
class TrackerConfig(Config):
    """Configuration for the instance tracker."""
    
    keyframe_interval: int = 24
    device: str = "cuda"
    observation_cache_dir: str = ""
    log_processing_timings: bool = False
    benchmark: TrackerBenchmarkConfig = field(default_factory=TrackerBenchmarkConfig)
    
    # Sub-component configs
    instance_segmenter: Any = config_field("instance_segmenter", default="fastsam")
    feature_extractor: DINOv3Config = field(default_factory=DINOv3Config)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    track_manager: TrackManagerConfig = field(default_factory=TrackManagerConfig)
    mask_refinement: MaskRefinementConfig = field(default_factory=MaskRefinementConfig)
    
    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class InstanceTracker:
    """Main instance tracker with sparse segmentation and dense propagation.
    
    Runs instance segmentation (e.g., FastSAM) every N frames and propagates
    masks between keyframes using DINOv3 features.
    """
    
    def __init__(self, config: TrackerConfig):
        """Initialize the instance tracker.
        
        Args:
            config: TrackerConfig with all parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        self._observation_loader: Optional[CachedObservationLoader] = None
        if config.observation_cache_dir:
            self._observation_loader = CachedObservationLoader(
                config.observation_cache_dir,
                self.device,
            )

        # Initialize components (lazy loading for models)
        self._segmenter: Optional[InstanceSegmenter] = None
        self._feature_extractor: Optional[DINOv3FeatureExtractor] = None
        self._preprocessor: Optional[DINOv3Preprocessing] = None
        
        self._propagator = LabelPropagator(config.propagation, self.device)
        self._track_manager = TrackManager(config.track_manager, self.device)
        self._mask_refiner = MaskRefiner(config.mask_refinement, self.device)
        
        # State
        self._frame_index = 0
        self._image_shape: Optional[Tuple[int, int]] = None
        self._features_shape: Optional[Tuple[int, int]] = None
        self._initialized = False

        if self._observation_loader is not None:
            logger.info(
                "Tracker using cached observations from %s",
                self._observation_loader.paths.run_dir,
            )
    
    @classmethod
    def construct(cls, **kwargs):
        """Load tracker from configuration dictionary."""
        config = TrackerConfig()
        config.update(kwargs)
        return cls(config)
    
    @property
    def segmenter(self) -> InstanceSegmenter:
        """Lazy-load the instance segmenter."""
        if self._segmenter is None:
            self._segmenter = self.config.instance_segmenter.create()
        return self._segmenter
    
    @property
    def feature_extractor(self) -> DINOv3FeatureExtractor:
        """Lazy-load the feature extractor."""
        if self._feature_extractor is None:
            self._feature_extractor = DINOv3FeatureExtractor(self.config.feature_extractor)
            self._feature_extractor.to(self.device)
        return self._feature_extractor

    @property
    def uses_cached_observations(self) -> bool:
        """Whether the tracker bypasses DINO/FastSAM using prerecorded inputs."""

        return self._observation_loader is not None
    
    def _get_preprocessor(self, height: int, width: int) -> DINOv3Preprocessing:
        """Get or create preprocessor for given image dimensions."""
        if self._preprocessor is None or (height, width) != self._image_shape:
            patch_size = self.feature_extractor.patch_size
            short_side = self.config.feature_extractor.short_side
            
            target_h, target_w = compute_target_dimensions(
                height, width, short_side, patch_size
            )
            
            self._preprocessor = DINOv3Preprocessing(
                target_height=target_h,
                target_width=target_w,
                device=self.device,
            )
            
            self._image_shape = (height, width)
            self._features_shape = (target_h // patch_size, target_w // patch_size)
        
        return self._preprocessor
    
    def reset(self) -> None:
        """Reset tracker state for a new sequence."""
        self._frame_index = 0
        self._initialized = False
        self._propagator.reset()
        self._track_manager.reset()
    
    def warmup(self, height: int = 480, width: int = 720) -> None:
        """Eagerly load models and JIT-compile CUDA kernels.
        
        Call this after initialization to move slow model loading out of
        the first frame processing. Runs a dummy forward pass to trigger
        CUDA kernel compilation.
        
        Args:
            height: Expected input image height
            width: Expected input image width
        """
        if self.uses_cached_observations:
            logger.info("Warmup skipped because tracker is using cached observations")
            return

        logger.info("Warming up models...")
        
        dummy_img = np.zeros((height, width, 3), dtype=np.uint8)
        with torch.no_grad():
            # Run the full keyframe detection path, including FastSAM mask
            # refinement, so the first real keyframe does not pay that cost.
            self._get_keyframe_detection_masks(dummy_img)
        
        # Force load feature extractor (DINOv3) and run dummy forward
        with torch.no_grad():
            _ = self._extract_features(dummy_img)
        
        # Clear CUDA cache after warmup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Warmup complete")
    
    @property
    def is_initialized(self) -> bool:
        """Check if tracker is initialized with first frame."""
        return self._initialized
    
    @property
    def frame_index(self) -> int:
        """Get current frame index."""
        return self._frame_index
    
    @property
    def active_tracks(self) -> List[Track]:
        """Get list of active tracks."""
        return self._track_manager.active_tracks
    
    def colorize_mask_image(self, mask_image: Tensor) -> np.ndarray:
        """Colorize a precomputed instance-id mask image.

        This uses per-track colors derived from initial DINOv3 prototypes,
        but applies them to the final mask image for the current frame.

        Args:
            mask_image: uint16 tensor [H, W] where pixel value = instance_id

        Returns:
            color_image: uint8 array [H, W, 3] RGB with feature-based colors
        """
        return self._track_manager.colorize_mask_image(mask_image)

    @staticmethod
    def _foreground_argmax_confidence(probs: Tensor) -> Tensor:
        """Return the winning foreground probability with background suppressed to 0."""

        label_confidence, label_indices = torch.max(probs, dim=-1)
        return torch.where(label_indices > 0, label_confidence, torch.zeros_like(label_confidence))

    def _get_unrefined_mask_and_confidence(
        self,
        output_shape: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """Return the thresholded output mask and the selected-label confidence map."""

        h, w = output_shape
        mask_image = torch.zeros((h, w), dtype=torch.int32, device=self.device)
        label_confidence = torch.zeros((h, w), dtype=torch.float32, device=self.device)

        for track in self._track_manager.active_tracks:
            if track.mask_probs is None:
                continue

            probs_up = F.interpolate(
                track.mask_probs.unsqueeze(0).unsqueeze(0),
                size=output_shape,
                mode="nearest",
            ).squeeze()
            selected = probs_up > 0.5
            mask_image[selected] = track.instance_id
            label_confidence[selected] = probs_up[selected]

        return mask_image.to(torch.uint16), label_confidence

    def _get_probability_outputs(
        self,
        is_keyframe: bool,
        output_shape: Tuple[int, int],
        runtime_profile: Optional[TrackerRuntimeProfile] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Return the mask image plus patch/final winning-label confidence maps."""

        outputs_start = time.perf_counter()
        if runtime_profile is not None:
            self._synchronize_device()
        tensor_start = time.perf_counter()
        probs_and_ids = self._track_manager.get_probability_tensor_with_instance_ids()
        if runtime_profile is not None:
            self._synchronize_device()
            runtime_profile.probability_tensor_ms = (
                time.perf_counter() - tensor_start
            ) * 1000.0
        h, w = output_shape
        if probs_and_ids is None:
            patch_confidence = None
            if self._features_shape is not None:
                patch_confidence = torch.zeros(
                    self._features_shape,
                    dtype=torch.float32,
                    device=self.device,
                )
            if runtime_profile is not None:
                self._synchronize_device()
                runtime_profile.probability_outputs_ms = (
                    time.perf_counter() - outputs_start
                ) * 1000.0
            return (
                torch.zeros((h, w), dtype=torch.uint16, device=self.device),
                patch_confidence,
                torch.zeros((h, w), dtype=torch.float32, device=self.device),
            )

        probs, instance_ids = probs_and_ids
        if runtime_profile is not None:
            self._synchronize_device()
        confidence_start = time.perf_counter()
        patch_argmax_confidence = self._foreground_argmax_confidence(probs).to(torch.float32)
        if runtime_profile is not None:
            self._synchronize_device()
            runtime_profile.probability_confidence_ms = (
                time.perf_counter() - confidence_start
            ) * 1000.0

        if not self.config.mask_refinement.enabled:
            if runtime_profile is not None:
                self._synchronize_device()
            refine_start = time.perf_counter()
            mask_image, final_argmax_confidence = self._get_unrefined_mask_and_confidence(
                output_shape
            )
            if runtime_profile is not None:
                self._synchronize_device()
                runtime_profile.probability_refine_ms = (
                    time.perf_counter() - refine_start
                ) * 1000.0
                runtime_profile.probability_outputs_ms = (
                    time.perf_counter() - outputs_start
                ) * 1000.0
            return mask_image, patch_argmax_confidence, final_argmax_confidence

        if runtime_profile is not None:
            self._synchronize_device()
        refine_start = time.perf_counter()
        refined = self._mask_refiner.refine_with_confidence(
            probs=probs,
            output_shape=output_shape,
            is_keyframe=is_keyframe,
            instance_ids=instance_ids,
            rgb=None,
        )
        if runtime_profile is not None:
            self._synchronize_device()
            runtime_profile.probability_refine_ms = (
                time.perf_counter() - refine_start
            ) * 1000.0
            runtime_profile.probability_outputs_ms = (
                time.perf_counter() - outputs_start
            ) * 1000.0
        return refined.mask_image, patch_argmax_confidence, refined.label_confidence
    
    def _is_keyframe(self, frame_index: Optional[int] = None) -> bool:
        """Check if current frame should be a keyframe."""
        index = self._frame_index if frame_index is None else int(frame_index)
        return index % self.config.keyframe_interval == 0
    
    def _extract_features(self, img: np.ndarray) -> Tensor:
        """Extract DINOv3 features from image.
        
        Args:
            img: RGB image [H, W, 3] uint8
        
        Returns:
            features: L2-normalized features [H', W', D]
        """
        h, w = img.shape[:2]
        preprocessor = self._get_preprocessor(h, w)
        
        # Convert to tensor and preprocess
        img_tensor = torch.from_numpy(img).to(self.device)
        img_preprocessed = preprocessor(img_tensor).squeeze(0)  # [C, H, W]
        
        # Extract features
        features = self.feature_extractor.extract_features(img_preprocessed)

        return features

    def _benchmark_enabled(self) -> bool:
        return bool(self.config.benchmark.enabled)

    def _should_synchronize_timers(self) -> bool:
        return self._benchmark_enabled() and bool(
            self.config.benchmark.synchronize_timers
        )

    def _can_sample_cuda_memory(self) -> bool:
        return self._benchmark_enabled() and self.device.type == "cuda" and torch.cuda.is_available()

    def _synchronize_device(self) -> None:
        if (
            self._should_synchronize_timers()
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def _reset_cuda_peak(self) -> None:
        if self._can_sample_cuda_memory():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _get_cuda_peak_mb(self) -> tuple[Optional[float], Optional[float]]:
        if not self._can_sample_cuda_memory():
            return None, None

        return (
            float(torch.cuda.max_memory_allocated(self.device)) / 1024.0**2,
            float(torch.cuda.max_memory_reserved(self.device)) / 1024.0**2,
        )

    def _get_frame_features(self, img: np.ndarray) -> Tensor:
        """Return DINO features, either by inference or from cache."""

        image_shape = tuple(int(v) for v in img.shape[:2])
        if self._observation_loader is not None:
            features = self._observation_loader.load_frame_features(
                self._frame_index,
                image_shape=image_shape,
            )
            if features.ndim != 3:
                raise ValueError(
                    f"Cached features for frame {self._frame_index} must be [H', W', D], "
                    f"got {features.shape}"
                )
            self._image_shape = image_shape
            self._features_shape = tuple(int(v) for v in features.shape[:2])
            return features

        return self._extract_features(img)

    def _get_keyframe_detection_masks(
        self,
        img: np.ndarray,
        runtime_profile: Optional[TrackerRuntimeProfile] = None,
    ) -> Tensor:
        """Return refined keyframe detection masks, either by inference or cache."""

        image_shape = tuple(int(v) for v in img.shape[:2])
        if self._observation_loader is not None:
            return self._observation_loader.load_keyframe_detections(
                self._frame_index,
                image_shape=image_shape,
            )

        if runtime_profile is not None:
            self._synchronize_device()
            self._reset_cuda_peak()
            t0 = time.perf_counter()
            masks, _ = self.segmenter.segment(img, device=self.device)
            self._synchronize_device()
            runtime_profile.fastsam_ms = (time.perf_counter() - t0) * 1000.0
            (
                runtime_profile.fastsam_max_memory_allocated_mb,
                runtime_profile.fastsam_max_memory_reserved_mb,
            ) = self._get_cuda_peak_mb()
        else:
            masks, _ = self.segmenter.segment(img, device=self.device)

        if runtime_profile is not None:
            self._synchronize_device()
            refine_start = time.perf_counter()
        masks = masks.to(self.device)
        refined_masks = self._mask_refiner.refine_keyframe_masks(masks)
        if runtime_profile is not None:
            self._synchronize_device()
            runtime_profile.keyframe_mask_refine_ms = (
                time.perf_counter() - refine_start
            ) * 1000.0
        return refined_masks
    
    def _masks_to_probs(self, masks: Tensor) -> Tensor:
        """Convert binary masks to probability tensor.
        
        Args:
            masks: Binary masks [N, H, W]
        
        Returns:
            probs: Probability tensor [H', W', N+1] with background at index 0
        """
        n = masks.shape[0]
        
        if n == 0:
            # No masks - return all background
            h, w = self._features_shape
            return torch.ones((h, w, 1), device=self.device)
        
        # Downsample masks to feature resolution
        masks_down = F.interpolate(
            masks.unsqueeze(1).float(),
            size=self._features_shape,
            mode="nearest-exact",
        ).squeeze(1)  # [N, H', W']
        
        # Convert to one-hot with background
        # Create label image from masks (later masks override earlier ones)
        h, w = self._features_shape
        labels = torch.zeros((h, w), dtype=torch.long, device=self.device)
        
        for i in range(n):
            labels[masks_down[i] > 0.5] = i + 1  # 1-indexed, 0 is background
        
        # Convert to one-hot
        probs = F.one_hot(labels, n + 1).float()  # [H', W', N+1]
        
        return probs

    def _binary_masks_to_instance_id_image(
        self,
        masks: Tensor,
        detection_instance_ids: List[int],
        output_shape: Tuple[int, int],
    ) -> Tensor:
        """Convert binary masks [N, H, W] to a uint16 instance-id image [H, W].

        Later masks overwrite earlier masks in overlap regions.
        """
        h, w = output_shape
        if masks.numel() == 0:
            return torch.zeros((h, w), dtype=torch.uint16, device=self.device)

        # Use int32 for masked assignment; torch does not support masked_fill_ on uint16.
        labels = torch.zeros((h, w), dtype=torch.int32, device=self.device)
        if len(detection_instance_ids) != masks.shape[0]:
            detection_instance_ids = [i + 1 for i in range(masks.shape[0])]
        for i in range(masks.shape[0]):
            instance_id = int(detection_instance_ids[i])
            if instance_id <= 0:
                instance_id = i + 1
            labels[masks[i]] = instance_id
        return labels.to(torch.uint16)
    
    def process_frame(
        self,
        img: np.ndarray,
        *,
        is_keyframe: Optional[bool] = None,
        camera_frame_index: Optional[int] = None,
    ) -> TrackerResult:
        """Process a single frame.
        
        Args:
            img: RGB image [H, W, 3] (uint8 or uint16)
            is_keyframe: Optional explicit keyframe decision. When omitted,
                keyframes use the processed-frame index for backwards compatibility.
            camera_frame_index: Optional zero-based source camera frame index.
        
        Returns:
            TrackerResult with masks and track information
        """
        # Ensure image is uint8 (required by FastSAM and feature extractor)
        original_dtype = img.dtype
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        h, w = img.shape[:2]
        is_keyframe = self._is_keyframe() if is_keyframe is None else bool(is_keyframe)
        
        # Debug: Log input image stats
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Frame {self._frame_index}: shape={img.shape}, original_dtype={original_dtype}, "
                f"camera_frame_index={camera_frame_index}, min={img.min()}, "
                f"max={img.max()}, mean={img.mean():.1f}, is_keyframe={is_keyframe}"
            )
        
        runtime_profile = TrackerRuntimeProfile() if self._benchmark_enabled() else None
        self._synchronize_device()
        tracker_start = time.perf_counter()
        tracker_peak_allocated_candidates: List[float] = []
        tracker_peak_reserved_candidates: List[float] = []

        def update_tracker_peak(allocated_mb: Optional[float], reserved_mb: Optional[float]) -> None:
            if allocated_mb is not None:
                tracker_peak_allocated_candidates.append(float(allocated_mb))
            if reserved_mb is not None:
                tracker_peak_reserved_candidates.append(float(reserved_mb))

        # Always extract features (timed)
        t0 = time.perf_counter()
        if runtime_profile is not None:
            self._synchronize_device()
            self._reset_cuda_peak()
        with torch.no_grad():
            features = self._get_frame_features(img)
        if runtime_profile is not None:
            self._synchronize_device()
            runtime_profile.dino_ms = (time.perf_counter() - t0) * 1000.0
            (
                runtime_profile.dino_max_memory_allocated_mb,
                runtime_profile.dino_max_memory_reserved_mb,
            ) = self._get_cuda_peak_mb()
            update_tracker_peak(
                runtime_profile.dino_max_memory_allocated_mb,
                runtime_profile.dino_max_memory_reserved_mb,
            )
        dino_ms = (
            runtime_profile.dino_ms
            if runtime_profile is not None
            else (time.perf_counter() - t0) * 1000.0
        )

        keyframe_fastsam_mask_image = None
        keyframe_detection_masks = None
        detection_instance_ids: List[int] = []
        raw_patch_confidence = None
        if not self._initialized or is_keyframe:
            # Load or infer keyframe detections (timed)
            t0 = time.perf_counter()
            masks = self._get_keyframe_detection_masks(img, runtime_profile=runtime_profile)
            fastsam_ms = (
                runtime_profile.fastsam_ms
                if runtime_profile is not None
                else (time.perf_counter() - t0) * 1000.0
            )
            if runtime_profile is not None:
                update_tracker_peak(
                    runtime_profile.fastsam_max_memory_allocated_mb,
                    runtime_profile.fastsam_max_memory_reserved_mb,
                )
            keyframe_detection_masks = masks
            
            n_detections = masks.shape[0] if masks.numel() > 0 else 0
            logger.debug(f"Frame {self._frame_index}: keyframe source returned {n_detections} masks")
            
            if runtime_profile is not None:
                self._synchronize_device()
                track_update_start = time.perf_counter()
            if not self._initialized:
                # First frame initialization
                created_tracks = self._track_manager.initialize_from_masks(
                    masks, features, self._features_shape
                )
                detection_instance_ids = [int(t.instance_id) for t in created_tracks]
                probs = self._masks_to_probs(masks)
                
                # Get probability tensor without background for propagator
                if probs.shape[-1] > 1:
                    self._propagator.initialize(features, probs)
                
                self._initialized = True
                logger.debug(f"Frame {self._frame_index}: initialized with {len(self._track_manager.active_tracks)} tracks")
            else:
                # Keyframe update - supervisory signal that corrects drift,
                # introduces new instances, and retires obsolete tracks
                n_tracks_before = len(self._track_manager.active_tracks)
                matched, new, retired, detection_instance_ids = self._track_manager.update_from_keyframe(
                    masks, features, self._features_shape
                )
                n_tracks_after = len(self._track_manager.active_tracks)
                
                logger.debug(
                    f"Frame {self._frame_index}: keyframe update - "
                    f"matched={len(matched)}, new={len(new)}, retired={len(retired)}, "
                    f"tracks: {n_tracks_before} -> {n_tracks_after}"
                )
                
                # Keyframe becomes new anchor - clear context and update reference
                probs = self._track_manager.get_probability_tensor()
                if probs is not None:
                    self._propagator.update_reference(features, probs)
            if runtime_profile is not None:
                self._synchronize_device()
                runtime_profile.keyframe_track_update_ms = (
                    time.perf_counter() - track_update_start
                ) * 1000.0

            if is_keyframe:
                if runtime_profile is not None:
                    self._synchronize_device()
                    label_image_start = time.perf_counter()
                keyframe_fastsam_mask_image = self._binary_masks_to_instance_id_image(
                    masks, detection_instance_ids, (h, w)
                )
                if runtime_profile is not None:
                    self._synchronize_device()
                    runtime_profile.keyframe_label_image_ms = (
                        time.perf_counter() - label_image_start
                    ) * 1000.0
            
            if self.config.log_processing_timings:
                logger.info(
                    "Frame %d timing: dino=%.1fms, fastsam=%.1fms",
                    self._frame_index,
                    dino_ms,
                    fastsam_ms,
                )
        else:
            # Propagate masks (timed)
            if self._propagator.is_initialized:
                t0 = time.perf_counter()
                if runtime_profile is not None:
                    self._synchronize_device()
                    self._reset_cuda_peak()
                propagated_probs = self._propagator.propagate(features)
                if runtime_profile is not None:
                    self._synchronize_device()
                    runtime_profile.propagation_ms = (time.perf_counter() - t0) * 1000.0
                    (
                        runtime_profile.propagation_max_memory_allocated_mb,
                        runtime_profile.propagation_max_memory_reserved_mb,
                    ) = self._get_cuda_peak_mb()
                    update_tracker_peak(
                        runtime_profile.propagation_max_memory_allocated_mb,
                        runtime_profile.propagation_max_memory_reserved_mb,
                    )
                if runtime_profile is not None:
                    self._synchronize_device()
                    propagation_update_start = time.perf_counter()
                raw_patch_confidence = self._foreground_argmax_confidence(propagated_probs).to(
                    torch.float32
                )
                propagated_probs = postprocess_probs(propagated_probs.permute(2, 0, 1)).permute(1, 2, 0)
                self._track_manager.update_from_propagation(propagated_probs)
                if runtime_profile is not None:
                    self._synchronize_device()
                    runtime_profile.propagation_update_ms = (
                        time.perf_counter() - propagation_update_start
                    ) * 1000.0
                prop_ms = (
                    runtime_profile.propagation_ms
                    if runtime_profile is not None
                    else (time.perf_counter() - t0) * 1000.0
                )
            else:
                prop_ms = 0.0
            
            if self.config.log_processing_timings:
                logger.info(
                    "Frame %d timing: dino=%.1fms, prop=%.1fms",
                    self._frame_index,
                    dino_ms,
                    prop_ms,
                )
        
        # Generate output mask image
        if runtime_profile is not None:
            self._synchronize_device()
            self._reset_cuda_peak()
        mask_image, patch_argmax_confidence, final_argmax_confidence = (
            self._get_probability_outputs(
                is_keyframe,
                (h, w),
                runtime_profile=runtime_profile,
            )
        )
        if runtime_profile is not None:
            self._synchronize_device()
            output_allocated_mb, output_reserved_mb = self._get_cuda_peak_mb()
            update_tracker_peak(output_allocated_mb, output_reserved_mb)
        
        # Include features for keyframes (kept on device, converted to numpy by caller if needed)
        result_features = features if is_keyframe else None
        
        result = TrackerResult(
            masks=mask_image,
            tracks=list(self._track_manager.active_tracks),
            is_keyframe=is_keyframe,
            frame_index=self._frame_index,
            camera_frame_index=camera_frame_index,
            features=result_features,
            frame_features=features,
            keyframe_fastsam_mask=keyframe_fastsam_mask_image,
            keyframe_detection_masks=keyframe_detection_masks,
            patch_argmax_confidence=patch_argmax_confidence,
            final_argmax_confidence=final_argmax_confidence,
            patch_argmax_confidence_raw=raw_patch_confidence,
            runtime_profile=runtime_profile,
        )

        if runtime_profile is not None:
            self._synchronize_device()
            runtime_profile.tracker_total_ms = (time.perf_counter() - tracker_start) * 1000.0
            runtime_profile.tracker_max_memory_allocated_mb = (
                max(tracker_peak_allocated_candidates)
                if tracker_peak_allocated_candidates
                else None
            )
            runtime_profile.tracker_max_memory_reserved_mb = (
                max(tracker_peak_reserved_candidates)
                if tracker_peak_reserved_candidates
                else None
            )

        self._frame_index += 1

        return result
    
    def __call__(self, img: np.ndarray) -> TrackerResult:
        """Process frame (alias for process_frame)."""
        return self.process_frame(img)
