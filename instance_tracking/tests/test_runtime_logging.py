"""Tests for quiet operational logging and tracker health summaries."""

import pathlib
import queue
import sys

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[2] / "instance_tracking_ros"),
)

from instance_tracking.tracker import TrackerConfig  # noqa: E402
from instance_tracking_ros.image_worker import (  # noqa: E402
    ImageWorker,
    ImageWorkerConfig,
)
from instance_tracking_ros.runtime_config import (  # noqa: E402
    OutputConfig,
    RuntimeLoggingConfig,
)


class _Logger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warn(self, message):
        self.warnings.append(message)


class _Node:
    def __init__(self):
        self.logger = _Logger()

    def get_logger(self):
        return self.logger


def _make_worker():
    worker = ImageWorker.__new__(ImageWorker)
    worker._node = _Node()
    worker._config = ImageWorkerConfig(queue_size=4, drop_warn_every=25)
    worker._keyframe_interval = 24
    worker._status_log_interval_s = 30.0
    worker._log_stride_skips = False
    worker._last_status_log_wall_time_s = 100.0
    worker._received_messages = 80
    worker._enqueued_messages = 40
    worker._processed_messages = 38
    worker._stride_skips = 40
    worker._queue_full_drops = 0
    worker._min_separation_skips = 0
    worker._callback_failures = 0
    worker._queue = queue.Queue(maxsize=4)
    return worker


def test_runtime_logging_defaults_are_sparse():
    config = RuntimeLoggingConfig()

    assert config.log_config is False
    assert config.log_keyframes is False
    assert config.log_stride_skips is False
    assert config.status_interval_s == 30.0
    assert TrackerConfig().log_processing_timings is False


def test_output_visualization_defaults_to_overlay_only():
    config = OutputConfig()

    assert config.publish_color_visualization is False
    assert config.publish_overlay_visualization is True
    assert config.overlay_alpha == 0.55


def test_health_summary_is_sparse_and_distinguishes_drops_from_skips():
    worker = _make_worker()

    worker._maybe_log_status(129.9)
    assert worker._node.logger.infos == []

    worker._maybe_log_status(130.0)
    assert len(worker._node.logger.infos) == 1
    status = worker._node.logger.infos[0]
    assert "queue_drops=0" in status
    assert "intentional_stride_skips=40" in status
    assert "rate_limit_skips=0" in status
    assert "queue_depth=0/4" in status

    worker._maybe_log_status(140.0)
    assert len(worker._node.logger.infos) == 1


def test_stride_skip_detail_is_opt_in():
    worker = _make_worker()
    msg = type(
        "Message",
        (),
        {
            "header": type(
                "Header",
                (),
                {"stamp": type("Stamp", (), {"sec": 1, "nanosec": 2})()},
            )()
        },
    )()

    worker._log_stride_skip_warning(msg, camera_frame_index=79)
    assert worker._node.logger.infos == []

    worker._log_stride_skips = True
    worker._stride_skips = 50
    worker._log_stride_skip_warning(msg, camera_frame_index=99)
    assert len(worker._node.logger.infos) == 1
