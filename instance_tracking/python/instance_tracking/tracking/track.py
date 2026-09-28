"""Track dataclass for instance tracking."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor


__all__ = [
    "Track",
    "TrackState",
]


class TrackState:
    """Enum-like class for track states."""
    ACTIVE = "active"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass
class Track:
    """Represents a tracked instance across frames.
    
    Attributes:
        track_id: Unique persistent ID for this track
        instance_id: Current pixel value in the mask image (1-indexed)
        age: Number of frames since track creation
        frames_since_detection: Frames since last keyframe detection
        state: Current track state (active, lost, terminated)
        mask_probs: Current probability map at feature resolution [H, W]
        confidence: Track confidence/quality score
        prototype: DINOv3 feature prototype [D] published for downstream use.
            Initialized from the first detection, then updated only on matched
            keyframes using an exponential moving average.
        color: RGB color tuple derived from prototype (0-255). Persistent
            color for visualization, not updated on re-detection.
        last_keyframe_mask_area: Binary mask area at feature resolution from the
            last keyframe detection used to initialize or refresh this track.
    """
    
    track_id: int
    instance_id: int
    age: int = 0
    frames_since_detection: int = 0
    state: str = TrackState.ACTIVE
    mask_probs: Optional[Tensor] = field(default=None, repr=False)
    confidence: float = 1.0
    prototype: Optional[Tensor] = field(default=None, repr=False)
    color: Optional[Tuple[int, int, int]] = None
    last_keyframe_mask_area: int = 0

    @staticmethod
    def _compute_mask_area(mask_probs: Optional[Tensor]) -> int:
        """Compute the current binary support size at feature resolution."""
        if mask_probs is None:
            return 0
        return int((mask_probs > 0.5).sum().item())

    def update_propagated(self, new_probs: Tensor) -> None:
        """Update track with propagated probabilities.
        
        Args:
            new_probs: New probability map [H, W]
        """
        self.mask_probs = new_probs
        self.age += 1
        self.frames_since_detection += 1
    
    def update_detected(self, new_probs: Tensor, confidence: float = 1.0) -> None:
        """Update track with detection from keyframe.
        
        Args:
            new_probs: New probability map from detection [H, W]
            confidence: Detection confidence
        """
        self.mask_probs = new_probs
        self.age += 1
        self.frames_since_detection = 0
        self.confidence = confidence
        self.state = TrackState.ACTIVE
        self.last_keyframe_mask_area = self._compute_mask_area(new_probs)
    
    def mark_lost(self) -> None:
        """Mark track as lost (not detected in keyframe)."""
        self.state = TrackState.LOST
        self.age += 1
        self.frames_since_detection += 1
    
    def terminate(self) -> None:
        """Terminate this track."""
        self.state = TrackState.TERMINATED

    @property
    def current_mask_area(self) -> int:
        """Get the current binary mask support size at feature resolution."""
        return self._compute_mask_area(self.mask_probs)
    
    @property
    def is_active(self) -> bool:
        """Check if track is active."""
        return self.state == TrackState.ACTIVE
    
    @property
    def is_lost(self) -> bool:
        """Check if track is lost."""
        return self.state == TrackState.LOST
    
    @property
    def is_terminated(self) -> bool:
        """Check if track is terminated."""
        return self.state == TrackState.TERMINATED
    
    def to_dict(self) -> dict:
        """Convert track to dictionary (for ROS message)."""
        return {
            "track_id": self.track_id,
            "instance_id": self.instance_id,
            "age": self.age,
            "frames_since_detection": self.frames_since_detection,
            "confidence": self.confidence,
        }
