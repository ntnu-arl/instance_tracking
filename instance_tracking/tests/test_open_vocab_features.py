"""Tests for source-track open-vocabulary feature helpers."""

import pathlib
import sys
import threading
import types

import cv2
import numpy as np
import pytest
import torch

from instance_tracking_msgs.msg import Track, TrackedInstances
from instance_tracking_msgs.srv import EncodeOpenVocabImage
from sensor_msgs.msg import CompressedImage

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[2] / "instance_tracking_ros"),
)

import instance_tracking_ros.open_vocab as open_vocab_module  # noqa: E402
from instance_tracking_ros.open_vocab import (  # noqa: E402
    OpenVocabConfig,
    OpenVocabEncoder,
    OpenVocabFeatureWorker,
    OpenVocabObservationQualityConfig,
    OpenVocabTemporalConsistencyConfig,
    compute_instance_bboxes_gpu,
    compute_observation_geometry_stats_gpu,
    default_normalization_parameters,
    encode_patches_in_batches,
    extract_boxed_and_masked_patch,
    extract_boxed_and_masked_patches_gpu,
    fuse_open_vocab_image_features,
    get_keyframe_track_requests,
    get_source_track_requests,
    normalization_parameters_from_preprocess_cfg,
    observation_quality_weight,
    update_track_feature_average,
    update_track_running_average,
)
from instance_tracking_ros.runtime_config import load_runtime_config  # noqa: E402


def test_extract_boxed_and_masked_patch_uses_normalized_black_fill():
    img = torch.tensor(
        [
            [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]],
            [[15, 25, 35, 45], [55, 65, 75, 85], [95, 105, 115, 125], [135, 145, 155, 165]],
            [[12, 22, 32, 42], [52, 62, 72, 82], [92, 102, 112, 122], [132, 142, 152, 162]],
        ],
        dtype=torch.uint8,
    )
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[1:3, 1:3] = True

    boxed, masked = extract_boxed_and_masked_patch(img, mask, size=4, crop_padding=1)

    assert boxed is not None
    assert masked is not None
    assert boxed.shape == (3, 4, 4)
    assert masked.shape == (3, 4, 4)

    mean, std = default_normalization_parameters()
    normalized_black = -mean / std
    assert torch.allclose(masked[:, 0, 0], normalized_black, atol=1.0e-5)
    assert not torch.allclose(masked[:, 1, 1], normalized_black, atol=1.0e-5)


def test_extract_boxed_and_masked_patch_accepts_model_specific_normalization():
    img = torch.full((3, 4, 4), 128, dtype=torch.uint8)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[1:3, 1:3] = True
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32)

    _, masked = extract_boxed_and_masked_patch(
        img,
        mask,
        size=4,
        crop_padding=1,
        normalization_parameters=(mean, std),
    )

    assert masked is not None
    assert torch.allclose(masked[:, 0, 0], -mean / std, atol=1.0e-5)


def test_normalization_parameters_from_preprocess_cfg_uses_openclip_values():
    mean, std = normalization_parameters_from_preprocess_cfg(
        {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
        }
    )

    assert torch.allclose(mean, torch.full((3,), 0.5))
    assert torch.allclose(std, torch.full((3,), 0.5))


def test_compute_instance_bboxes_gpu_matches_expected_padded_boxes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = torch.zeros((6, 7), dtype=torch.int32, device=device)
    mask[1:3, 2:5] = 5
    mask[5, 0] = 9
    instance_ids = torch.tensor([5, 9, 11], dtype=torch.int32, device=device)

    boxes, valid = compute_instance_bboxes_gpu(mask, instance_ids, crop_padding=1)

    expected = torch.tensor(
        [
            [1, 0, 6, 4],
            [0, 4, 2, 6],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(boxes.cpu(), expected)
    assert valid.cpu().tolist() == [True, True, False]


def test_extract_boxed_and_masked_patches_gpu_batches_and_handles_missing_instances():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.full((3, 6, 6), 128, dtype=torch.uint8, device=device)
    mask = torch.zeros((6, 6), dtype=torch.int32, device=device)
    mask[2:4, 2:4] = 3
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32)

    boxed, masked, metadata, valid = extract_boxed_and_masked_patches_gpu(
        img,
        mask,
        [(10, 3), (11, 99)],
        size=4,
        crop_padding=1,
        normalization_parameters=(mean, std),
    )

    assert boxed.shape == (2, 3, 4, 4)
    assert masked.shape == (2, 3, 4, 4)
    assert metadata == [(10, 3), (11, 99)]
    assert valid.cpu().tolist() == [True, False]
    normalized_black = (-mean / std).to(device=device).view(3, 1, 1)
    assert torch.allclose(masked[1], normalized_black.expand_as(masked[1]), atol=1.0e-5)
    assert not torch.allclose(masked[0], normalized_black.expand_as(masked[0]), atol=1.0e-5)


def test_extract_boxed_only_patches_skips_masked_resampling(monkeypatch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.full((3, 6, 6), 128, dtype=torch.uint8, device=device)
    mask = torch.zeros((6, 6), dtype=torch.int32, device=device)
    mask[2:4, 2:4] = 3
    grid_sample_calls = 0
    original_grid_sample = open_vocab_module.F.grid_sample

    def count_grid_sample(*args, **kwargs):
        nonlocal grid_sample_calls
        grid_sample_calls += 1
        return original_grid_sample(*args, **kwargs)

    monkeypatch.setattr(open_vocab_module.F, "grid_sample", count_grid_sample)
    boxed, masked, metadata, valid = extract_boxed_and_masked_patches_gpu(
        img,
        mask,
        [(10, 3)],
        size=4,
        include_masked=False,
    )

    assert boxed.shape == (1, 3, 4, 4)
    assert masked is None
    assert metadata == [(10, 3)]
    assert valid.cpu().tolist() == [True]
    assert grid_sample_calls == 1


def test_warp_bbox_sampling_preserves_full_padded_bbox_content():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.zeros((3, 4, 8), dtype=torch.uint8, device=device)
    img[:, 1:3, 1:7] = 255
    img[:, 1:3, 1:2] = 32
    img[:, 1:3, 6:7] = 64
    mask = torch.zeros((4, 8), dtype=torch.int32, device=device)
    mask[1:3, 1:7] = 3
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)

    center_boxed, _center_masked, _metadata, _valid = extract_boxed_and_masked_patches_gpu(
        img,
        mask,
        [(10, 3)],
        size=4,
        crop_sampling_mode="center_square",
        normalization_parameters=(mean, std),
    )
    warped_boxed, _warped_masked, _metadata, _valid = extract_boxed_and_masked_patches_gpu(
        img,
        mask,
        [(10, 3)],
        size=4,
        crop_sampling_mode="warp_bbox",
        normalization_parameters=(mean, std),
    )

    center_first_col = center_boxed[0, :, :, 0].mean()
    warped_first_col = warped_boxed[0, :, :, 0].mean()
    assert warped_first_col < center_first_col


def test_observation_geometry_stats_reports_fill_and_boundary_clamp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = torch.zeros((10, 10), dtype=torch.int32, device=device)
    mask[2:6, 3:7] = 4
    mask[0:2, 0:2] = 5
    stats = compute_observation_geometry_stats_gpu(
        mask,
        torch.tensor([4, 5], dtype=torch.int32, device=device),
        crop_padding=2,
        crop_sampling_mode="warp_bbox",
    )

    assert stats["valid"].cpu().tolist() == [True, True]
    assert stats["bbox_short_side_px"].cpu().tolist() == [4.0, 2.0]
    assert torch.allclose(
        stats["raw_mask_to_box_fill"].cpu(),
        torch.tensor([1.0, 1.0]),
    )
    assert stats["mask_to_box_fill"][0].item() < 0.30
    assert stats["padding_clamp_fraction"][1].item() < 1.0
    assert stats["touches_image_boundary"].cpu().tolist() == [False, True]


def test_observation_quality_weight_rejects_tiny_and_low_fill_observations():
    stats = {
        "valid": torch.tensor([True, True]),
        "mask_area_px": torch.tensor([100.0, 100.0]),
        "bbox_short_side_px": torch.tensor([20.0, 30.0]),
        "raw_mask_to_box_fill": torch.tensor([0.5, 0.5]),
        "mask_to_box_fill": torch.tensor([0.5, 0.05]),
        "padding_clamp_fraction": torch.tensor([1.0, 1.0]),
        "touches_image_boundary": torch.tensor([False, False]),
    }
    config = OpenVocabObservationQualityConfig(
        min_bbox_short_side_px=24,
        min_mask_to_box_fill=0.07,
    )

    _weight, reason, _diagnostics = observation_quality_weight(stats, 0, config)
    assert reason == "bbox_short_side"
    _weight, reason, _diagnostics = observation_quality_weight(stats, 1, config)
    assert reason == "mask_to_box_fill"


def test_observation_quality_weight_downweights_clamped_boundary_observations():
    stats = {
        "valid": torch.tensor([True]),
        "mask_area_px": torch.tensor([100.0]),
        "bbox_short_side_px": torch.tensor([30.0]),
        "raw_mask_to_box_fill": torch.tensor([0.5]),
        "mask_to_box_fill": torch.tensor([0.5]),
        "padding_clamp_fraction": torch.tensor([0.5]),
        "touches_image_boundary": torch.tensor([True]),
    }
    config = OpenVocabObservationQualityConfig(
        padding_clamp_min_weight=0.5,
        image_boundary_weight=0.5,
    )

    weight, reason, diagnostics = observation_quality_weight(stats, 0, config)

    assert reason == ""
    assert diagnostics["touches_image_boundary"] is True
    assert 0.0 < weight < 1.0


def test_fuse_open_vocab_features_normalizes_components_before_fusion():
    boxed = torch.tensor([[10.0, 0.0]], dtype=torch.float32)
    masked = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
    config = OpenVocabConfig(
        boxed_only=False,
        normalize_embeddings_before_fusion=True,
        precision="fp32",
    )

    fused = fuse_open_vocab_image_features(boxed, masked, config=config)

    expected = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
    assert torch.allclose(fused, expected, atol=1.0e-5)


def test_fuse_open_vocab_features_boxed_only_ignores_masked_feature():
    boxed = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    masked = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    config = OpenVocabConfig(
        boxed_only=True,
        normalize_embeddings_before_fusion=True,
        precision="fp32",
    )

    fused = fuse_open_vocab_image_features(boxed, masked, config=config)

    assert torch.allclose(fused, boxed)


def test_fuse_open_vocab_features_can_include_full_image_feature():
    boxed = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    masked = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    full = torch.tensor([0.0, 1.0], dtype=torch.float32)
    config = OpenVocabConfig(
        boxed_only=False,
        normalize_embeddings_before_fusion=True,
        use_full_image_feature=True,
        precision="fp32",
    )

    fused = fuse_open_vocab_image_features(boxed, masked, config=config, full_image_feature=full)

    expected = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
    assert torch.allclose(fused, expected, atol=1.0e-5)


def test_encode_patches_in_batches_preserves_order_and_counts_calls():
    class FakeEncoder:
        def __init__(self):
            self.batch_sizes = []

        def encode(self, patches):
            self.batch_sizes.append(int(patches.shape[0]))
            return patches[:, :, 0, 0].to(dtype=torch.float32)

    patches = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
    encoder = FakeEncoder()

    features, calls = encode_patches_in_batches(encoder, patches, image_batch_size=2)

    expected = patches[:, :, 0, 0]
    assert calls == 3
    assert encoder.batch_sizes == [2, 2, 1]
    assert torch.equal(features, expected)


def test_encode_patches_in_batches_keeps_full_batch_default():
    class FakeEncoder:
        def __init__(self):
            self.calls = 0

        def encode(self, patches):
            self.calls += 1
            return torch.ones((patches.shape[0], 4), dtype=torch.float32)

    encoder = FakeEncoder()
    patches = torch.zeros((3, 3, 2, 2), dtype=torch.float32)

    features, calls = encode_patches_in_batches(encoder, patches, image_batch_size=0)

    assert calls == 1
    assert encoder.calls == 1
    assert features.shape == (3, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for runtime fp16 crop check")
def test_extract_boxed_and_masked_patches_gpu_uses_requested_output_dtype():
    img = torch.full((3, 6, 6), 128, dtype=torch.uint8, device="cuda")
    mask = torch.zeros((6, 6), dtype=torch.int32, device="cuda")
    mask[2:4, 2:4] = 3

    boxed, masked, _metadata, _valid = extract_boxed_and_masked_patches_gpu(
        img,
        mask,
        [(10, 3)],
        size=4,
        crop_padding=1,
        output_dtype=torch.float16,
    )

    assert boxed.dtype == torch.float16
    assert masked.dtype == torch.float16


def test_open_vocab_encoder_uses_model_preprocess_cfg(monkeypatch):
    class FakeVisual:
        image_size = (224, 224)
        preprocess_cfg = {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
        }

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = FakeVisual()

        def encode_image(self, patches, normalize=False):
            return torch.ones((patches.shape[0], 4), device=patches.device)

        def encode_text(self, tokens, normalize=False):
            return torch.ones((tokens.shape[0], 4), device=tokens.device)

    fake_open_clip = types.SimpleNamespace(
        create_model_and_transforms=lambda *args, **kwargs: (FakeModel(), None, None),
        get_tokenizer=lambda model_name: lambda prompts: torch.ones(
            (len(prompts), 4),
            dtype=torch.int64,
        ),
        get_input_dtype=lambda precision: None,
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    encoder = OpenVocabEncoder(OpenVocabConfig(device="cpu", precision="fp32"))

    mean, std = encoder.normalization_parameters
    assert torch.allclose(mean, torch.full((3,), 0.5))
    assert torch.allclose(std, torch.full((3,), 0.5))


def test_get_source_track_requests_only_returns_new_keyframe_tracks():
    msg = TrackedInstances()
    msg.is_keyframe = True

    fresh_track = Track()
    fresh_track.track_id = 11
    fresh_track.instance_id = 3
    fresh_track.age = 0

    old_track = Track()
    old_track.track_id = 12
    old_track.instance_id = 4
    old_track.age = 5

    msg.tracks = [fresh_track, old_track]

    assert get_source_track_requests(msg) == [(11, 3)]

    msg.is_keyframe = False
    assert get_source_track_requests(msg) == []


def test_get_keyframe_track_requests_respects_policy_and_mask_presence():
    fresh_track = Track()
    fresh_track.track_id = 11
    fresh_track.instance_id = 3
    fresh_track.age = 0

    old_track = Track()
    old_track.track_id = 12
    old_track.instance_id = 4
    old_track.age = 5

    mask = np.array(
        [
            [0, 3, 3],
            [0, 4, 0],
        ],
        dtype=np.uint16,
    )

    assert get_keyframe_track_requests(
        [fresh_track, old_track],
        mask_image=mask,
        is_keyframe=True,
        keyframe_update_policy="source_tracks_only",
    ) == [(11, 3)]

    assert get_keyframe_track_requests(
        [fresh_track, old_track],
        mask_image=mask,
        is_keyframe=True,
        keyframe_update_policy="all_tracks",
    ) == [(11, 3), (12, 4)]

    missing_old_mask = np.array([[0, 3, 3]], dtype=np.uint16)
    assert get_keyframe_track_requests(
        [fresh_track, old_track],
        mask_image=missing_old_mask,
        is_keyframe=True,
        keyframe_update_policy="all_tracks",
    ) == [(11, 3)]

    assert get_keyframe_track_requests(
        [fresh_track, old_track],
        mask_image=mask,
        is_keyframe=False,
        keyframe_update_policy="all_tracks",
    ) == []


def test_running_average_normalizes_each_keyframe_observation_equally():
    aggregates = {}

    first = update_track_running_average(
        aggregates,
        11,
        torch.tensor([3.0, 0.0, 0.0], dtype=torch.float32),
    )
    second = update_track_running_average(
        aggregates,
        11,
        torch.tensor([0.0, 8.0, 0.0], dtype=torch.float32),
    )

    assert first is not None
    assert second is not None
    assert torch.allclose(first, torch.tensor([1.0, 0.0, 0.0]), atol=1.0e-5)
    expected = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(second, expected, atol=1.0e-5)
    assert aggregates[11].num_observations == 2
    assert torch.allclose(torch.linalg.vector_norm(second), torch.tensor(1.0), atol=1.0e-5)


def test_weighted_running_average_uses_observation_weight():
    aggregates = {}

    update_track_feature_average(
        aggregates,
        11,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        observation_weight=1.0,
    )
    result = update_track_feature_average(
        aggregates,
        11,
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        observation_weight=0.25,
    )

    expected = torch.tensor([1.0, 0.25], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert result.accepted
    assert result.feature is not None
    assert torch.allclose(result.feature, expected, atol=1.0e-5)
    assert aggregates[11].total_weight == pytest.approx(1.25)


def test_temporal_consistency_rejects_outlier_after_warmup():
    aggregates = {}
    temporal = OpenVocabTemporalConsistencyConfig(
        enabled=True,
        warmup_observations=1,
        reject_below_cosine=0.55,
        downweight_below_cosine=0.75,
    )

    update_track_feature_average(
        aggregates,
        11,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        temporal_consistency=temporal,
    )
    result = update_track_feature_average(
        aggregates,
        11,
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        temporal_consistency=temporal,
    )

    assert not result.accepted
    assert result.reject_reason == "temporal_cosine"
    assert result.temporal_cosine == pytest.approx(0.0)
    assert aggregates[11].num_observations == 1


def test_temporal_consistency_downweights_mid_similarity_observation():
    aggregates = {}
    temporal = OpenVocabTemporalConsistencyConfig(
        enabled=True,
        warmup_observations=1,
        reject_below_cosine=0.55,
        downweight_below_cosine=0.75,
        min_downweight=0.25,
    )

    update_track_feature_average(
        aggregates,
        11,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        temporal_consistency=temporal,
    )
    candidate = torch.tensor([0.60, 0.80], dtype=torch.float32)
    result = update_track_feature_average(
        aggregates,
        11,
        candidate,
        temporal_consistency=temporal,
    )

    assert result.accepted
    assert result.temporal_downweighted
    assert result.temporal_cosine == pytest.approx(0.60, abs=1.0e-5)
    assert 0.25 < result.weight < 1.0


def test_open_vocab_config_accepts_only_supported_keyframe_policies():
    assert OpenVocabConfig().keyframe_update_policy == "all_tracks"
    assert OpenVocabConfig().precision == "fp16"
    assert OpenVocabConfig().image_batch_size == 0
    assert OpenVocabConfig().boxed_only is True
    assert OpenVocabConfig(boxed_only=True).boxed_only is True
    assert OpenVocabConfig(crop_sampling_mode="warp_bbox").crop_sampling_mode == "warp_bbox"
    assert OpenVocabConfig(keyframe_update_policy="source_tracks_only").keyframe_update_policy == (
        "source_tracks_only"
    )

    with pytest.raises(ValueError):
        OpenVocabConfig(keyframe_update_policy="invalid")
    with pytest.raises(ValueError):
        OpenVocabConfig(crop_sampling_mode="invalid")
    with pytest.raises(ValueError, match="boxed_feature_weight"):
        OpenVocabConfig(boxed_only=True, boxed_feature_weight=0.0)


def test_runtime_config_defaults_to_all_keyframes_for_open_vocab():
    config = load_runtime_config(None)
    assert config.open_vocab.keyframe_update_policy == "all_tracks"
    assert config.open_vocab.precision == "fp16"


def test_open_vocab_benchmark_sync_is_disabled_unless_explicitly_enabled(monkeypatch):
    worker = OpenVocabFeatureWorker.__new__(OpenVocabFeatureWorker)
    worker._config = OpenVocabConfig(enabled=True)
    worker._encoder = type("EncoderStub", (), {"device": torch.device("cuda")})()

    sync_calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: sync_calls.append(str(device)),
    )

    worker._config.benchmark.enabled = False
    worker._config.benchmark.synchronize_timers = True
    worker._synchronize_device()
    assert sync_calls == []

    worker._config.benchmark.enabled = True
    worker._synchronize_device()
    assert sync_calls == ["cuda"]

    worker._config.benchmark.synchronize_timers = False
    worker._synchronize_device()
    assert sync_calls == ["cuda"]


class _ImageServiceLogger:
    def __init__(self):
        self.errors: list[str] = []

    def error(self, message):
        self.errors.append(str(message))


class _ImageServiceNode:
    def __init__(self):
        self.logger = _ImageServiceLogger()

    def get_logger(self):
        return self.logger


class _ImageServiceEncoder:
    encoder_id = "openclip:test-model:test-weights"
    device = torch.device("cpu")
    input_size = 4
    input_dtype = torch.float16
    normalization_parameters = (
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
    )

    def __init__(self):
        self.encoded_batches: list[torch.Tensor] = []

    def encode(self, image_batch):
        self.encoded_batches.append(image_batch)
        return torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)


def _make_image_service_worker():
    worker = OpenVocabFeatureWorker.__new__(OpenVocabFeatureWorker)
    worker._node = _ImageServiceNode()
    worker._encoder = _ImageServiceEncoder()
    worker._encoder_lock = threading.Lock()
    return worker


def test_open_vocab_image_service_decodes_rgb_and_uses_model_preprocessing(monkeypatch):
    worker = _make_image_service_worker()
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        ],
        dtype=np.uint8,
    )
    encoded_ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert encoded_ok

    request = EncodeOpenVocabImage.Request()
    request.image = CompressedImage(format="png", data=encoded.tobytes())
    response = EncodeOpenVocabImage.Response()
    preprocess_call = {}

    def fake_extract(
        rgb_tensor,
        size,
        normalization_parameters=None,
        output_dtype=None,
    ):
        preprocess_call["rgb_tensor"] = rgb_tensor
        preprocess_call["size"] = size
        preprocess_call["normalization_parameters"] = normalization_parameters
        preprocess_call["output_dtype"] = output_dtype
        return torch.zeros((1, 3, size, size), dtype=output_dtype)

    monkeypatch.setattr(
        open_vocab_module,
        "extract_full_image_patch_gpu",
        fake_extract,
    )

    result = worker._handle_encode_image(request, response)

    assert result is response
    assert response.success is True
    assert response.message == ""
    assert response.encoder_id == worker._encoder.encoder_id
    assert response.feature == pytest.approx([1.0, 2.0, 3.0])
    assert torch.equal(
        preprocess_call["rgb_tensor"],
        torch.from_numpy(rgb).permute(2, 0, 1),
    )
    assert preprocess_call["size"] == worker._encoder.input_size
    assert (
        preprocess_call["normalization_parameters"]
        is worker._encoder.normalization_parameters
    )
    assert preprocess_call["output_dtype"] == worker._encoder.input_dtype
    assert len(worker._encoder.encoded_batches) == 1
    assert worker._encoder.encoded_batches[0].dtype == worker._encoder.input_dtype
    assert worker._node.logger.errors == []


def test_open_vocab_image_service_rejects_invalid_compressed_image():
    worker = _make_image_service_worker()
    request = EncodeOpenVocabImage.Request()
    request.image = CompressedImage(format="jpeg", data=b"not an encoded image")
    response = EncodeOpenVocabImage.Response()

    result = worker._handle_encode_image(request, response)

    assert result is response
    assert response.success is False
    assert "Failed to encode open-vocabulary query image" in response.message
    assert response.encoder_id == worker._encoder.encoder_id
    assert list(response.feature) == []
    assert len(worker._encoder.encoded_batches) == 0
    assert worker._node.logger.errors == [response.message]
