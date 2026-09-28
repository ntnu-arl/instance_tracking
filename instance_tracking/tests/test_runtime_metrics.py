"""Tests for persisted runtime metrics summaries."""

import importlib.util
import json
import pathlib
import sys

import pytest

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[2] / "instance_tracking_ros" / "instance_tracking_ros"),
)

from runtime_metrics import (  # noqa: E402
    RuntimeMetricsRecorder,
    build_numeric_summary,
)

_HAS_OPEN_VOCAB_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "rclpy", "instance_tracking_msgs")
)
if _HAS_OPEN_VOCAB_DEPS:
    from open_vocab import OpenVocabMetricsRecorder  # noqa: E402
else:  # pragma: no cover - environment-dependent
    OpenVocabMetricsRecorder = None


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


class _DummyNode:
    def get_logger(self):
        return _DummyLogger()


def test_build_numeric_summary_splits_keyframes_and_propagated_frames():
    events = [
        {
            "frame_index": 0,
            "is_keyframe": True,
            "dino_ms": 10.0,
            "node_total_ms": 25.0,
        },
        {
            "frame_index": 1,
            "is_keyframe": False,
            "dino_ms": 14.0,
            "node_total_ms": 21.0,
        },
        {
            "frame_index": 2,
            "is_keyframe": False,
            "dino_ms": 16.0,
            "node_total_ms": 23.0,
        },
    ]

    summary = build_numeric_summary(
        events,
        numeric_fields=("dino_ms", "node_total_ms"),
    )

    assert summary["count"] == 3
    assert summary["keyframe_count"] == 1
    assert summary["propagated_count"] == 2
    assert summary["all_frames"]["metrics"]["dino_ms"]["mean"] == pytest.approx(40.0 / 3.0)
    assert summary["all_frames"]["metrics"]["node_total_ms"]["std"] == pytest.approx(
        (8.0 / 3.0) ** 0.5
    )
    assert summary["keyframes"]["metrics"]["dino_ms"]["mean"] == pytest.approx(10.0)
    assert summary["propagated_frames"]["metrics"]["dino_ms"]["mean"] == pytest.approx(
        15.0
    )


def test_runtime_metrics_recorder_persists_null_cuda_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("INSTANCE_TRACKING_OUTPUT_RUN_DIR", str(tmp_path))

    recorder = RuntimeMetricsRecorder(
        _DummyLogger(),
        numeric_fields=("node_total_ms", "node_max_memory_allocated_mb"),
    )
    recorder.record_frame(
        {
            "frame_index": 0,
            "is_keyframe": True,
            "node_total_ms": 12.5,
            "node_max_memory_allocated_mb": None,
        }
    )
    recorder.close()

    events = (tmp_path / "runtime" / "frame_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
    event = json.loads(events[0])
    assert event["node_max_memory_allocated_mb"] is None
    assert event["record_wall_time_s"] > 0.0
    assert event["record_monotonic_s"] > 0.0

    summary = json.loads((tmp_path / "runtime" / "frame_summary.json").read_text(encoding="utf-8"))
    assert summary["all_frames"]["metrics"]["node_total_ms"]["count"] == 1
    assert summary["all_frames"]["metrics"]["node_total_ms"]["mean"] == pytest.approx(12.5)
    assert summary["all_frames"]["metrics"]["node_max_memory_allocated_mb"]["count"] == 0
    assert summary["all_frames"]["frequencies_hz"]["node_frequency_hz"] == pytest.approx(80.0)
    assert summary["node_frequency_hz"] == pytest.approx(80.0)


def test_open_vocab_metrics_summary_includes_job_stats(tmp_path, monkeypatch):
    if OpenVocabMetricsRecorder is None:
        pytest.skip("Open-vocabulary metrics test requires torch + ROS Python deps")

    monkeypatch.setenv("INSTANCE_TRACKING_OUTPUT_RUN_DIR", str(tmp_path))

    recorder = OpenVocabMetricsRecorder(
        _DummyNode(),
        "openclip:ViT-H-14:laion2b_s32b_b79k",
    )
    recorder.record_job(
        frame_index=10,
        stamp_ns=1234,
        source_tracks_requested=3,
        source_tracks_encoded=2,
        model_forward_calls=2,
        boxed_forward_calls=1,
        masked_forward_calls=1,
        image_batch_size=8,
        preprocess_ms=5.0,
        boxed_inference_ms=7.0,
        masked_inference_ms=8.0,
        publish_ms=1.0,
        total_job_ms=21.0,
        preprocess_max_memory_allocated_mb=100.0,
        preprocess_max_memory_reserved_mb=120.0,
        boxed_inference_max_memory_allocated_mb=140.0,
        boxed_inference_max_memory_reserved_mb=160.0,
        masked_inference_max_memory_allocated_mb=180.0,
        masked_inference_max_memory_reserved_mb=200.0,
        total_job_max_memory_allocated_mb=180.0,
        total_job_max_memory_reserved_mb=200.0,
        published=True,
    )
    recorder.record_job(
        frame_index=11,
        stamp_ns=5678,
        source_tracks_requested=2,
        source_tracks_encoded=1,
        model_forward_calls=2,
        boxed_forward_calls=1,
        masked_forward_calls=1,
        image_batch_size=8,
        preprocess_ms=6.0,
        boxed_inference_ms=9.0,
        masked_inference_ms=10.0,
        publish_ms=1.5,
        total_job_ms=26.5,
        preprocess_max_memory_allocated_mb=90.0,
        preprocess_max_memory_reserved_mb=110.0,
        boxed_inference_max_memory_allocated_mb=150.0,
        boxed_inference_max_memory_reserved_mb=170.0,
        masked_inference_max_memory_allocated_mb=190.0,
        masked_inference_max_memory_reserved_mb=210.0,
        total_job_max_memory_allocated_mb=190.0,
        total_job_max_memory_reserved_mb=210.0,
        published=False,
    )
    recorder.close()

    summary = json.loads(
        (tmp_path / "open_vocab" / "inference_summary.json").read_text(encoding="utf-8")
    )
    assert summary["jobs_seen"] == 2
    assert summary["jobs_published"] == 1
    assert summary["jobs_without_features"] == 1
    assert summary["job_stats"]["count"] == 2
    assert summary["job_stats"]["keyframe_count"] == 2
    assert summary["job_stats"]["all_frames"]["metrics"]["total_job_ms"]["max"] == pytest.approx(
        26.5
    )
    assert summary["job_stats"]["all_frames"]["frequencies_hz"]["clip_job_frequency_hz"] == (
        pytest.approx(1000.0 / 23.75)
    )
    assert summary["clip_job_frequency_hz"] == pytest.approx(1000.0 / 23.75)
    assert summary["job_stats"]["all_frames"]["metrics"][
        "total_job_max_memory_reserved_mb"
    ]["max"] == pytest.approx(210.0)
    events = (
        tmp_path / "open_vocab" / "inference_events.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    job_event = json.loads(events[0])
    assert job_event["record_wall_time_s"] > 0.0
    assert job_event["record_monotonic_s"] > 0.0
    assert job_event["job_start_wall_time_s"] < job_event["job_end_wall_time_s"]
    assert job_event["job_start_monotonic_s"] < job_event["job_end_monotonic_s"]
    assert summary["total_model_forward_calls"] == 4
    assert summary["total_boxed_forward_calls"] == 2
    assert summary["total_masked_forward_calls"] == 2
    assert summary["job_stats"]["all_frames"]["metrics"]["image_batch_size"]["mean"] == 8.0
