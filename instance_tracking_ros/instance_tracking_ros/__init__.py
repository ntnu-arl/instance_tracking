"""Instance tracking ROS2 interface."""

from importlib import import_module

_LAZY_EXPORTS = {
    "Conversions": ("instance_tracking_ros.ros_conversions", "Conversions"),
    "ImageWorker": ("instance_tracking_ros.image_worker", "ImageWorker"),
    "ImageWorkerConfig": ("instance_tracking_ros.image_worker", "ImageWorkerConfig"),
    "ImageWorkerFrame": ("instance_tracking_ros.image_worker", "ImageWorkerFrame"),
    "OPEN_VOCAB_KEYFRAME_UPDATE_POLICIES": (
        "instance_tracking_ros.open_vocab",
        "OPEN_VOCAB_KEYFRAME_UPDATE_POLICIES",
    ),
    "DataRecorder": ("instance_tracking_ros.data_recorder", "DataRecorder"),
    "DataRecorderConfig": ("instance_tracking_ros.data_recorder", "DataRecorderConfig"),
    "OpenVocabConfig": ("instance_tracking_ros.open_vocab", "OpenVocabConfig"),
    "OpenVocabFeatureWorker": (
        "instance_tracking_ros.open_vocab",
        "OpenVocabFeatureWorker",
    ),
    "PROMPT_BANK_SCHEMA_VERSION": (
        "instance_tracking_ros.prompt_bank",
        "PROMPT_BANK_SCHEMA_VERSION",
    ),
    "PROMPT_BANK_TYPE": ("instance_tracking_ros.prompt_bank", "PROMPT_BANK_TYPE"),
    "build_prompt_bank_payload": (
        "instance_tracking_ros.prompt_bank",
        "build_prompt_bank_payload",
    ),
    "compute_instance_bboxes_gpu": (
        "instance_tracking_ros.open_vocab",
        "compute_instance_bboxes_gpu",
    ),
    "encode_patches_in_batches": (
        "instance_tracking_ros.open_vocab",
        "encode_patches_in_batches",
    ),
    "extract_boxed_and_masked_patch": (
        "instance_tracking_ros.open_vocab",
        "extract_boxed_and_masked_patch",
    ),
    "extract_boxed_and_masked_patches_gpu": (
        "instance_tracking_ros.open_vocab",
        "extract_boxed_and_masked_patches_gpu",
    ),
    "get_keyframe_track_requests": (
        "instance_tracking_ros.open_vocab",
        "get_keyframe_track_requests",
    ),
    "get_source_track_requests": (
        "instance_tracking_ros.open_vocab",
        "get_source_track_requests",
    ),
    "load_prompt_file": ("instance_tracking_ros.prompt_bank", "load_prompt_file"),
    "normalize_open_vocab_feature": (
        "instance_tracking_ros.open_vocab",
        "normalize_open_vocab_feature",
    ),
    "normalize_feature": ("instance_tracking_ros.prompt_bank", "normalize_feature"),
    "save_prompt_bank": ("instance_tracking_ros.prompt_bank", "save_prompt_bank"),
    "update_track_running_average": (
        "instance_tracking_ros.open_vocab",
        "update_track_running_average",
    ),
}

__all__ = [
    "Conversions",
    "ImageWorker",
    "ImageWorkerConfig",
    "ImageWorkerFrame",
    "OPEN_VOCAB_KEYFRAME_UPDATE_POLICIES",
    "DataRecorder",
    "DataRecorderConfig",
    "OpenVocabConfig",
    "OpenVocabFeatureWorker",
    "PROMPT_BANK_SCHEMA_VERSION",
    "PROMPT_BANK_TYPE",
    "build_prompt_bank_payload",
    "compute_instance_bboxes_gpu",
    "encode_patches_in_batches",
    "extract_boxed_and_masked_patch",
    "extract_boxed_and_masked_patches_gpu",
    "get_keyframe_track_requests",
    "get_source_track_requests",
    "load_prompt_file",
    "normalize_open_vocab_feature",
    "normalize_feature",
    "save_prompt_bank",
    "update_track_running_average",
]


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
