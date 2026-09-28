from pathlib import Path

import numpy as np
import pytest
import torch

from instance_tracking.observation_cache import ObservationCachePaths, write_manifest
from instance_tracking.models.mask_refinement import MaskRefinementConfig
from instance_tracking.models.propagation import PropagationConfig
from instance_tracking.tracker import InstanceTracker, TrackerConfig
from instance_tracking.tracking.track_manager import TrackManagerConfig


def _prepare_cache(run_dir: Path) -> ObservationCachePaths:
    paths = ObservationCachePaths.from_run_dir(run_dir)
    paths.frame_features_dir.mkdir(parents=True, exist_ok=True)
    paths.keyframe_detections_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(paths)
    return paths


def _identity_features(height: int = 4, width: int = 4) -> np.ndarray:
    features = np.zeros((height, width, height * width), dtype=np.float16)
    flat_index = 0
    for y in range(height):
        for x in range(width):
            features[y, x, flat_index] = 1.0
            flat_index += 1
    return features


def _single_mask(
    top: int,
    left: int,
    bottom: int,
    right: int,
    image_shape: tuple[int, int] = (4, 4),
) -> np.ndarray:
    masks = np.zeros((1, image_shape[0], image_shape[1]), dtype=np.uint8)
    masks[0, top:bottom, left:right] = 1
    return masks


def _expected_instance_mask(
    instance_id: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
    image_shape: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    mask = torch.zeros(image_shape, dtype=torch.uint16)
    mask[top:bottom, left:right] = int(instance_id)
    return mask


def _write_frame_features(
    paths: ObservationCachePaths,
    frame_index: int,
    features: np.ndarray | None = None,
    image_shape: tuple[int, int] = (4, 4),
) -> None:
    if features is None:
        features = _identity_features(*image_shape)
    np.savez_compressed(
        paths.frame_features_dir / f"{frame_index:06d}.npz",
        features=features,
        image_shape=np.array(image_shape, dtype=np.int32),
    )


def _write_keyframe_masks(
    paths: ObservationCachePaths,
    frame_index: int,
    masks: np.ndarray | None = None,
    image_shape: tuple[int, int] = (4, 4),
) -> None:
    if masks is None:
        masks = _single_mask(0, 0, 2, 2, image_shape=image_shape)
    np.savez_compressed(
        paths.keyframe_detections_dir / f"{frame_index:06d}.npz",
        masks=masks,
        image_shape=np.array(image_shape, dtype=np.int32),
    )


def _make_cached_tracker(
    run_dir: Path,
    *,
    keyframe_interval: int,
    min_iou_gate: float = 0.15,
    eligibility_factor: float = 0.0,
) -> InstanceTracker:
    return InstanceTracker(
        TrackerConfig(
            keyframe_interval=keyframe_interval,
            device="cpu",
            observation_cache_dir=str(run_dir),
            propagation=PropagationConfig(
                max_context_length=7,
                neighborhood_size=0,
                topk=1,
                temperature=0.1,
            ),
            track_manager=TrackManagerConfig(
                min_iou_gate=min_iou_gate,
                eligibility_factor=eligibility_factor,
                alignment_threshold=0.35,
            ),
            mask_refinement=MaskRefinementConfig(enabled=False),
        )
    )


def test_tracker_replay_uses_cached_observations_without_model_loading(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_cache"
    paths = _prepare_cache(run_dir)
    for frame_index in (0, 1):
        _write_frame_features(paths, frame_index)
        _write_keyframe_masks(paths, frame_index)

    tracker = InstanceTracker(
        TrackerConfig(
            keyframe_interval=1,
            device="cpu",
            observation_cache_dir=str(run_dir),
        )
    )
    tracker.warmup()

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    result0 = tracker.process_frame(image)
    result1 = tracker.process_frame(image)

    assert tracker.uses_cached_observations
    assert tracker._feature_extractor is None
    assert tracker._segmenter is None
    assert result0.is_keyframe
    assert result1.is_keyframe
    assert result0.num_instances == 1
    assert result1.num_instances == 1
    assert result1.tracks[0].track_id == result0.tracks[0].track_id
    assert result0.frame_features.shape == (4, 4, 16)
    assert result0.keyframe_detection_masks.shape == (1, 4, 4)


def test_tracker_replay_rejects_cached_shape_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_cache"
    paths = _prepare_cache(run_dir)
    _write_frame_features(paths, 0)
    _write_keyframe_masks(paths, 0)

    tracker = InstanceTracker(
        TrackerConfig(
            keyframe_interval=1,
            device="cpu",
            observation_cache_dir=str(run_dir),
        )
    )

    with pytest.raises(ValueError, match="expect image shape"):
        tracker.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))


def test_tracker_benchmark_sync_is_disabled_unless_explicitly_enabled(monkeypatch) -> None:
    tracker = InstanceTracker.__new__(InstanceTracker)
    tracker.config = TrackerConfig()
    tracker.device = torch.device("cuda")

    sync_calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: sync_calls.append(str(device)),
    )

    tracker.config.benchmark.enabled = False
    tracker.config.benchmark.synchronize_timers = True
    tracker._synchronize_device()
    assert sync_calls == []

    tracker.config.benchmark.enabled = True
    tracker._synchronize_device()
    assert sync_calls == ["cuda"]

    tracker.config.benchmark.synchronize_timers = False
    tracker._synchronize_device()
    assert sync_calls == ["cuda"]


def test_tracker_runtime_profile_only_exists_in_benchmark_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_cache"
    paths = _prepare_cache(run_dir)
    _write_frame_features(paths, 0)
    _write_keyframe_masks(paths, 0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    tracker_without_benchmark = InstanceTracker(
        TrackerConfig(
            keyframe_interval=1,
            device="cpu",
            observation_cache_dir=str(run_dir),
        )
    )
    result_without_benchmark = tracker_without_benchmark.process_frame(image)
    assert result_without_benchmark.runtime_profile is None

    benchmark_config = TrackerConfig(
        keyframe_interval=1,
        device="cpu",
        observation_cache_dir=str(run_dir),
    )
    benchmark_config.benchmark.enabled = True
    benchmark_config.benchmark.synchronize_timers = True
    tracker_with_benchmark = InstanceTracker(
        benchmark_config
    )
    result_with_benchmark = tracker_with_benchmark.process_frame(image)
    assert result_with_benchmark.runtime_profile is not None
    assert result_with_benchmark.runtime_profile.tracker_total_ms >= 0.0
    assert result_with_benchmark.runtime_profile.dino_max_memory_allocated_mb is None


def test_tracker_replay_propagates_on_non_keyframe_without_detection_cache(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_cache"
    paths = _prepare_cache(run_dir)
    _write_frame_features(paths, 0)
    _write_frame_features(paths, 1)
    _write_keyframe_masks(paths, 0, _single_mask(0, 0, 2, 2))

    tracker = _make_cached_tracker(run_dir, keyframe_interval=2)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    result0 = tracker.process_frame(image)
    first_track_id = result0.tracks[0].track_id
    result1 = tracker.process_frame(image)

    assert result0.is_keyframe
    assert not result1.is_keyframe
    assert result1.num_instances == 1
    assert result1.tracks[0].track_id == first_track_id
    assert result1.tracks[0].frames_since_detection == 1
    assert torch.equal(result1.masks.cpu(), _expected_instance_mask(1, 0, 0, 2, 2))
    expected_confidence = torch.zeros((4, 4), dtype=torch.float32)
    expected_confidence[0:2, 0:2] = 1.0
    assert torch.equal(result1.patch_argmax_confidence.cpu(), expected_confidence)
    assert torch.equal(result1.final_argmax_confidence.cpu(), expected_confidence)
    assert len(tracker._propagator._features_queue) == 1
    assert len(tracker._propagator._probs_queue) == 1


def test_tracker_replay_keyframe_refresh_resets_propagation_context(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_cache"
    paths = _prepare_cache(run_dir)
    for frame_index in (0, 1, 2):
        _write_frame_features(paths, frame_index)
    _write_keyframe_masks(paths, 0, _single_mask(0, 0, 2, 2))
    _write_keyframe_masks(paths, 2, _single_mask(0, 0, 2, 2))

    tracker = _make_cached_tracker(run_dir, keyframe_interval=2)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    result0 = tracker.process_frame(image)
    first_track_id = result0.tracks[0].track_id
    tracker.process_frame(image)
    assert len(tracker._propagator._features_queue) == 1
    assert len(tracker._propagator._probs_queue) == 1

    result2 = tracker.process_frame(image)

    assert result2.is_keyframe
    assert result2.num_instances == 1
    assert result2.tracks[0].track_id == first_track_id
    assert result2.tracks[0].frames_since_detection == 0
    assert len(tracker._propagator._features_queue) == 0
    assert len(tracker._propagator._probs_queue) == 0


def test_tracker_replay_keyframe_can_retire_and_replace_track(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_cache"
    paths = _prepare_cache(run_dir)
    for frame_index in (0, 1, 2):
        _write_frame_features(paths, frame_index)
    _write_keyframe_masks(paths, 0, _single_mask(0, 0, 2, 2))
    _write_keyframe_masks(paths, 2, _single_mask(2, 2, 4, 4))

    tracker = _make_cached_tracker(run_dir, keyframe_interval=2, min_iou_gate=0.15)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    result0 = tracker.process_frame(image)
    original_track_id = result0.tracks[0].track_id
    tracker.process_frame(image)
    result2 = tracker.process_frame(image)

    active_tracks = tracker.active_tracks
    assert result2.is_keyframe
    assert result2.num_instances == 1
    assert len(active_tracks) == 1
    assert active_tracks[0].track_id != original_track_id
    assert tracker._track_manager.tracks[original_track_id].is_terminated
    assert torch.equal(
        result2.masks.cpu(),
        _expected_instance_mask(active_tracks[0].instance_id, 2, 2, 4, 4),
    )


def test_tracker_warmup_runs_fastsam_dummy_forward() -> None:
    class _FakeSegmenter:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, int, int], str]] = []

        def segment(self, img, device=None):
            self.calls.append((tuple(img.shape), str(device)))
            h, w = img.shape[:2]
            return (
                torch.zeros((0, h, w), dtype=torch.bool),
                torch.zeros((0, 4), dtype=torch.int32),
            )

    class _FakeFeatureExtractor:
        patch_size = 1

        def __init__(self) -> None:
            self.calls = 0

        def extract_features(self, img: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            h, w = img.shape[-2:]
            return torch.zeros((h, w, 1), dtype=torch.float32, device=img.device)

    tracker = InstanceTracker(TrackerConfig(device="cpu"))
    tracker.config.feature_extractor.short_side = 4
    tracker._segmenter = _FakeSegmenter()
    tracker._feature_extractor = _FakeFeatureExtractor()

    tracker.warmup(height=4, width=4)

    assert tracker._segmenter.calls == [((4, 4, 3), "cpu")]
    assert tracker._feature_extractor.calls == 1
