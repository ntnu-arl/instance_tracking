"""Shared runtime configuration helpers for instance_tracking ROS apps."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import spark_config as sc

from instance_tracking.tracker import TrackerConfig
from instance_tracking_ros.data_recorder import DataRecorderConfig
from instance_tracking_ros.image_worker import ImageWorkerConfig
from instance_tracking_ros.open_vocab import OpenVocabConfig


@dataclass
class OutputConfig(sc.Config):
    """Configuration for published and recorded mask outputs."""

    publish_keyframe_fastsam_mask: bool = True
    publish_confidence_image: bool = True
    include_track_prototypes: bool = True
    publish_mask_visualization: bool = True
    publish_color_visualization: bool = False
    publish_overlay_visualization: bool = True
    overlay_alpha: float = 0.55

    def __post_init__(self) -> None:
        self.overlay_alpha = float(self.overlay_alpha)
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError(
                f"output.overlay_alpha must be in [0, 1], got {self.overlay_alpha}"
            )


@dataclass
class RuntimeLoggingConfig(sc.Config):
    """Operational logging controls for the ROS tracker node."""

    log_config: bool = False
    log_keyframes: bool = False
    log_stride_skips: bool = False
    status_interval_s: float = 30.0

    def __post_init__(self) -> None:
        self.status_interval_s = max(0.0, float(self.status_interval_s))


@dataclass
class InstanceTrackingNodeConfig(sc.Config):
    """Top-level config shared by live and offline entrypoints."""

    worker: ImageWorkerConfig = field(default_factory=ImageWorkerConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    open_vocab: OpenVocabConfig = field(default_factory=OpenVocabConfig)
    recorder: DataRecorderConfig = field(default_factory=DataRecorderConfig)
    logging: RuntimeLoggingConfig = field(default_factory=RuntimeLoggingConfig)


def resolve_default_config_path() -> pathlib.Path | None:
    """Resolve the installed or in-tree OpenLex-quality config path."""

    candidates = [
        pathlib.Path(__file__).resolve().parents[1]
        / "config"
        / "openlex_quality.yaml",
        pathlib.Path(__file__).resolve().parents[3]
        / "share"
        / "instance_tracking_ros"
        / "config"
        / "openlex_quality.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_runtime_config(config_path: str | None) -> InstanceTrackingNodeConfig:
    """Load a full instance-tracking runtime config from file or defaults."""

    if config_path:
        resolved = pathlib.Path(config_path).expanduser().absolute()
        if resolved.exists():
            return sc.Config.load(InstanceTrackingNodeConfig, resolved)
        raise FileNotFoundError(f"Config path '{resolved}' does not exist")

    default_path = resolve_default_config_path()
    if default_path is not None:
        return sc.Config.load(InstanceTrackingNodeConfig, default_path)

    return InstanceTrackingNodeConfig()
