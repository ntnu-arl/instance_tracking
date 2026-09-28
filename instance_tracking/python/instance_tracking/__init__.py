"""Instance tracking with sparse segmentation and dense propagation."""

import logging
import os
import pathlib

import torch


# Custom TRACE level (more verbose than DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def root_path():
    """Get root path of package."""
    return pathlib.Path(__file__).absolute().parent


def default_device(use_cuda=True):
    """Get default device to use for pytorch."""
    return "cuda" if torch.cuda.is_available() and use_cuda else "cpu"


def configure_logging():
    """Configure logging for instance_tracking modules.
    
    Log levels:
    - Default: INFO for instance_tracking, WARNING for dependencies
    - INSTANCE_TRACKING_DEBUG=1: DEBUG (image stats, GPU memory, matching summaries)
    - INSTANCE_TRACKING_TRACE=1: TRACE (individual match/retire details per track)
    """
    if os.environ.get("INSTANCE_TRACKING_TRACE"):
        level = TRACE
    elif os.environ.get("INSTANCE_TRACKING_DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Keep dependency chatter out of the normal operational log. The selected
    # level below is applied only to instance_tracking modules.
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] [%(name)s] %(message)s",
    )
    logging.getLogger().setLevel(logging.WARNING)
    
    # Set level specifically for our modules
    for module in [
        "instance_tracking.tracker",
        "instance_tracking.models.instance_segmenter",
        "instance_tracking.models.feature_extractor",
        "instance_tracking.tracking.track_manager",
    ]:
        logging.getLogger(module).setLevel(level)


# Auto-configure logging on import
configure_logging()
