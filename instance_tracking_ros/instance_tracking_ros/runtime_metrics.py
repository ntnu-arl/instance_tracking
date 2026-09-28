"""Persistent runtime metrics for instance tracking benchmark runs."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional


FREQUENCY_METRIC_NAMES = {
    "dino_ms": "dino_frequency_hz",
    "fastsam_ms": "fastsam_frequency_hz",
    "propagation_ms": "propagation_frequency_hz",
    "tracker_total_ms": "tracker_frequency_hz",
    "node_total_ms": "node_frequency_hz",
    "preprocess_ms": "preprocess_frequency_hz",
    "boxed_inference_ms": "boxed_inference_frequency_hz",
    "masked_inference_ms": "masked_inference_frequency_hz",
    "full_image_inference_ms": "full_image_inference_frequency_hz",
    "total_inference_ms": "clip_inference_frequency_hz",
    "publish_ms": "publish_frequency_hz",
    "total_job_ms": "clip_job_frequency_hz",
}


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(
        float(value)
    )


def frequency_hz(mean_ms: float) -> float:
    """Return Hz for a mean duration in milliseconds."""

    return 1000.0 / float(mean_ms) if mean_ms > 0.0 else 0.0


def compute_scalar_stats(values: Iterable[float]) -> dict[str, float | int]:
    """Return count/mean/std/min/max for a numeric series."""

    series = [float(v) for v in values if math.isfinite(float(v))]
    if not series:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    count = len(series)
    mean = sum(series) / float(count)
    variance = sum((value - mean) ** 2 for value in series) / float(count)
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(series),
        "max": max(series),
    }


def build_numeric_summary(
    events: list[dict[str, Any]],
    *,
    numeric_fields: Iterable[str],
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build split runtime summary for all/keyframe/propagated frame events."""

    numeric_fields = tuple(numeric_fields)
    all_events = list(events)
    keyframes = [event for event in all_events if bool(event.get("is_keyframe", False))]
    propagated = [
        event for event in all_events if not bool(event.get("is_keyframe", False))
    ]

    def summarize_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"count": len(subset), "metrics": {}}
        for field in numeric_fields:
            values = [
                float(event[field])
                for event in subset
                if _is_finite_number(event.get(field))
            ]
            summary["metrics"][field] = compute_scalar_stats(values)
        frequencies = {
            freq_name: frequency_hz(float(summary["metrics"][field]["mean"]))
            for field, freq_name in FREQUENCY_METRIC_NAMES.items()
            if field in summary["metrics"] and int(summary["metrics"][field]["count"]) > 0
        }
        if frequencies:
            summary["frequencies_hz"] = frequencies
        return summary

    result = {
        "count": len(all_events),
        "keyframe_count": len(keyframes),
        "propagated_count": len(propagated),
        "all_frames": summarize_subset(all_events),
        "keyframes": summarize_subset(keyframes),
        "propagated_frames": summarize_subset(propagated),
    }
    if metadata:
        result.update(metadata)
    return result


class RuntimeMetricsRecorder:
    """Persist per-frame benchmark metrics into a run directory."""

    def __init__(
        self,
        logger,
        *,
        output_subdir: str = "runtime",
        events_filename: str = "frame_events.jsonl",
        summary_filename: str = "frame_summary.json",
        metadata: Optional[dict[str, Any]] = None,
        numeric_fields: Optional[Iterable[str]] = None,
    ):
        self._logger = logger
        self._lock = threading.Lock()
        self._enabled = False
        self._events: list[dict[str, Any]] = []
        self._metadata = dict(metadata or {})
        self._numeric_fields = tuple(numeric_fields) if numeric_fields is not None else None
        self._events_path: Optional[Path] = None
        self._summary_path: Optional[Path] = None

        run_dir = os.getenv("INSTANCE_TRACKING_OUTPUT_RUN_DIR", "").strip()
        if not run_dir:
            return

        output_dir = Path(os.path.expanduser(run_dir)) / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = output_dir / events_filename
        self._summary_path = output_dir / summary_filename
        self._enabled = True
        self._write_summary_locked()
        self._logger.info(f"Saving runtime metrics under {output_dir}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_frame(self, event: dict[str, Any]) -> None:
        if not self._enabled:
            return

        event = dict(event)
        event.setdefault("record_wall_time_s", time.time())
        event.setdefault("record_monotonic_s", time.monotonic())

        normalized = {}
        for key, value in event.items():
            if _is_finite_number(value):
                normalized[key] = float(value)
            else:
                normalized[key] = value

        with self._lock:
            self._events.append(normalized)
            self._append_event_locked(normalized)
            self._write_summary_locked()

    def update_metadata(self, updates: dict[str, Any]) -> None:
        """Merge additional metadata into the persisted summary."""

        if not self._enabled:
            return

        with self._lock:
            self._metadata.update(dict(updates))
            self._write_summary_locked()

    def close(self) -> None:
        if not self._enabled:
            return

        with self._lock:
            self._write_summary_locked()

    def _append_event_locked(self, event: dict[str, Any]) -> None:
        if self._events_path is None:
            return

        with self._events_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(event, sort_keys=True) + "\n")

    def _write_summary_locked(self) -> None:
        if self._summary_path is None:
            return

        if self._numeric_fields is None:
            numeric_fields = sorted(
                {
                    key
                    for event in self._events
                    for key, value in event.items()
                    if _is_finite_number(value)
                }
            )
        else:
            numeric_fields = self._numeric_fields
        summary = build_numeric_summary(
            self._events,
            numeric_fields=numeric_fields,
            metadata=self._metadata,
        )
        all_frequencies = summary.get("all_frames", {}).get("frequencies_hz", {})
        if all_frequencies:
            if "tracker_frequency_hz" in all_frequencies:
                summary["tracker_frequency_hz"] = float(
                    all_frequencies["tracker_frequency_hz"]
                )
            if "node_frequency_hz" in all_frequencies:
                summary["node_frequency_hz"] = float(all_frequencies["node_frequency_hz"])
        with self._summary_path.open("w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, sort_keys=True)
