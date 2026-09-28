"""Track manager for instance tracking lifecycle management."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from spark_config import Config
from torch import Tensor

from instance_tracking import TRACE
from instance_tracking.tracking.track import Track, TrackState

logger = logging.getLogger(__name__)


__all__ = [
    "TrackManager",
    "TrackManagerConfig",
]


@dataclass
class TrackManagerConfig(Config):
    """Configuration for track manager."""
    
    # Matching weights for combined score (should sum to 1.0)
    iou_weight: float = 0.5
    feature_weight: float = 0.5
    # Ignore pairs whose spatial overlap is too small to be a plausible temporal
    # continuation. We still compute the dense cosine matrix for simplicity;
    # this gate constrains assignment rather than feature extraction cost.
    min_iou_gate: float = 0.15
    # Minimum current_propagated_mask_area / last_keyframe_mask_area required
    # for an existing propagated track to be eligible for keyframe association.
    eligibility_factor: float = 0.05
    # Update matched tracks' published prototypes on keyframes using:
    #   normalize((1 - a) * old + a * keyframe_proto)
    prototype_ema_alpha: float = 0.2
    
    # Thresholds
    alignment_threshold: float = 0.35  # Minimum combined score to accept match
    max_lost_frames: int = 10  # Frames before terminating lost track
    
    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


def compute_mask_iou(mask1: Tensor, mask2: Tensor) -> float:
    """Compute IoU between two binary masks.
    
    Args:
        mask1: First binary mask [H, W]
        mask2: Second binary mask [H, W]
    
    Returns:
        IoU score
    """
    intersection = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    
    if union == 0:
        return 0.0
    
    return (intersection / union).item()


def compute_iou_matrix(masks1: Tensor, masks2: Tensor) -> Tensor:
    """Compute pairwise IoU matrix between two sets of masks.
    
    Args:
        masks1: First set of masks [N, H, W]
        masks2: Second set of masks [M, H, W]
    
    Returns:
        IoU matrix of shape [N, M]
    """
    n = masks1.shape[0]
    m = masks2.shape[0]
    
    if n == 0 or m == 0:
        return torch.zeros((n, m), device=masks1.device)
    
    # Flatten spatial dimensions
    masks1_flat = masks1.flatten(1).float()  # [N, H*W]
    masks2_flat = masks2.flatten(1).float()  # [M, H*W]
    
    # Compute intersections
    intersection = torch.mm(masks1_flat, masks2_flat.t())  # [N, M]
    
    # Compute unions
    area1 = masks1_flat.sum(dim=1, keepdim=True)  # [N, 1]
    area2 = masks2_flat.sum(dim=1, keepdim=True)  # [M, 1]
    union = area1 + area2.t() - intersection  # [N, M]
    
    # Compute IoU
    iou = intersection / union.clamp(min=1e-8)
    
    return iou


def compute_mask_prototype(features: Tensor, mask: Tensor) -> Tensor:
    """Compute average feature vector for masked region.
    
    Args:
        features: Feature map [H, W, D]
        mask: Boolean mask [H, W]
    
    Returns:
        prototype: Average feature vector [D]
    """
    masked_features = features[mask]  # [N_pixels, D]
    if masked_features.numel() == 0:
        return torch.zeros(features.shape[-1], device=features.device)
    return masked_features.mean(dim=0)


def compute_prototype_matrix(
    features: Tensor,
    masks: Tensor,
) -> Tensor:
    """Compute prototypes for multiple masks.
    
    Args:
        features: Feature map [H, W, D]
        masks: Binary masks [N, H, W]
    
    Returns:
        prototypes: Feature prototypes [N, D]
    """
    n = masks.shape[0]
    d = features.shape[-1]
    device = features.device
    
    if n == 0:
        return torch.zeros((0, d), device=device)
    
    prototypes = []
    for i in range(n):
        proto = compute_mask_prototype(features, masks[i])
        prototypes.append(proto)
    
    return torch.stack(prototypes, dim=0)


def compute_cosine_similarity_matrix(
    prototypes1: Tensor,
    prototypes2: Tensor,
) -> Tensor:
    """Compute pairwise cosine similarity matrix.
    
    Args:
        prototypes1: First set of prototypes [N, D]
        prototypes2: Second set of prototypes [M, D]
    
    Returns:
        Cosine similarity matrix of shape [N, M]
    """
    n = prototypes1.shape[0]
    m = prototypes2.shape[0]
    
    if n == 0 or m == 0:
        return torch.zeros((n, m), device=prototypes1.device)
    
    # Normalize prototypes
    p1_norm = F.normalize(prototypes1, dim=-1, p=2)
    p2_norm = F.normalize(prototypes2, dim=-1, p=2)
    
    # Compute cosine similarity
    return torch.mm(p1_norm, p2_norm.t())  # [N, M]


def prototype_to_color(
    prototype: Tensor,
    indices: Tuple[int, int, int] = (-1, -2, -3),
    saturation: float = 0.85,
    value: float = 0.95,
) -> Tuple[int, int, int]:
    """Convert feature prototype to vibrant RGB color using HSV mapping.
    
    Maps feature dimensions to HSV color space with fixed high saturation
    and value to ensure bright, distinguishable colors. The hue is derived
    from the prototype features to maintain color consistency per track.
    
    Args:
        prototype: Feature vector [D] (L2-normalized DINOv3 features)
        indices: Which dimensions to use for hue computation (default: last 3)
        saturation: HSV saturation (0-1), higher = more vibrant (default: 0.85)
        value: HSV value/brightness (0-1), higher = brighter (default: 0.95)
    
    Returns:
        RGB tuple with values in range (0-255)
    """
    import colorsys
    
    # Extract feature components
    components = prototype[list(indices)].cpu().numpy()
    
    # Map components to hue: use atan2 of first two components for circular mapping
    # This gives a full 360-degree hue range based on feature direction
    hue = (np.arctan2(components[0], components[1]) + np.pi) / (2 * np.pi)
    
    # Use third component to slightly vary saturation (keep it high though)
    sat = saturation + 0.1 * components[2]
    sat = np.clip(sat, 0.7, 1.0)
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, sat, value)
    
    return (int(255 * r), int(255 * g), int(255 * b))


class TrackManager:
    """Manages track lifecycle for instance tracking.
    
    Responsibilities:
    - Assign unique track IDs
    - Associate detections with existing tracks
    - Handle track creation, update, and termination
    - Maintain track ID consistency across propagation
    """
    
    def __init__(self, config: TrackManagerConfig, device: torch.device):
        """Initialize track manager.
        
        Args:
            config: TrackManagerConfig with parameters
            device: Device for tensor operations
        """
        self.config = config
        self.device = device
        
        self._tracks: Dict[int, Track] = {}
        self._next_track_id: int = 1
        self._next_instance_id: int = 1
    
    def reset(self) -> None:
        """Reset the track manager state."""
        self._tracks.clear()
        self._next_track_id = 1
        self._next_instance_id = 1
    
    @property
    def tracks(self) -> Dict[int, Track]:
        """Get all tracks."""
        return self._tracks
    
    @property
    def active_tracks(self) -> List[Track]:
        """Get active tracks."""
        return [t for t in self._tracks.values() if t.is_active]
    
    @property
    def num_active_tracks(self) -> int:
        """Get number of active tracks."""
        return len(self.active_tracks)
    
    def _create_track(
        self,
        mask_probs: Tensor,
        features: Optional[Tensor] = None,
        mask_binary: Optional[Tensor] = None,
        confidence: float = 1.0,
    ) -> Track:
        """Create a new track with optional feature-based color.

        The prototype is computed from the initial instance mask at track
        creation. Matched keyframes can later update the published prototype
        via EMA, but the display color stays fixed from this initial value so
        the same object keeps a stable visualization color.
        
        Args:
            mask_probs: Initial probability map [H, W]
            features: Feature map [H, W, D] for computing prototype (optional)
            mask_binary: Binary mask [H, W] for computing prototype (optional)
            confidence: Detection confidence
        
        Returns:
            New Track object with prototype and color if features provided
        """
        prototype = None
        color = None
        
        if features is not None and mask_binary is not None:
            prototype = compute_mask_prototype(features, mask_binary)
            color = prototype_to_color(prototype)
        
        track = Track(
            track_id=self._next_track_id,
            instance_id=self._next_instance_id,
            mask_probs=mask_probs.to(self.device),
            confidence=confidence,
            prototype=prototype,
            color=color,
        )
        if mask_binary is not None:
            track.last_keyframe_mask_area = int(mask_binary.sum().item())
        else:
            track.last_keyframe_mask_area = track.current_mask_area
        
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        self._next_instance_id += 1

        return track

    def _update_track_prototype(self, track: Track, keyframe_prototype: Tensor) -> None:
        """Update a matched track's published prototype using keyframe-only EMA."""
        if keyframe_prototype is None:
            return

        prototype = keyframe_prototype.to(self.device)
        if prototype.numel() == 0:
            return

        if track.prototype is None or track.prototype.numel() == 0:
            track.prototype = prototype
            return

        alpha = float(self.config.prototype_ema_alpha)
        if alpha <= 0.0:
            return
        if alpha >= 1.0:
            track.prototype = prototype
            return

        blended = (1.0 - alpha) * track.prototype + alpha * prototype
        track.prototype = F.normalize(blended.unsqueeze(0), dim=-1, p=2).squeeze(0)

    def _compute_track_eligibility(self, tracks: List[Track]) -> Tuple[Tensor, Tensor]:
        """Compute keyframe-association eligibility from propagated support size."""
        if not tracks:
            empty = torch.zeros((0,), dtype=torch.float32, device=self.device)
            return empty.to(torch.bool), empty

        ratios = []
        for track in tracks:
            if track.last_keyframe_mask_area <= 0:
                ratios.append(0.0)
            else:
                ratios.append(track.current_mask_area / float(track.last_keyframe_mask_area))

        ratio_tensor = torch.tensor(ratios, dtype=torch.float32, device=self.device)
        eligible = ratio_tensor > self.config.eligibility_factor
        return eligible, ratio_tensor
    
    def initialize_from_masks(
        self,
        masks: Tensor,
        features: Tensor,
        features_shape: Tuple[int, int],
    ) -> List[Track]:
        """Initialize tracks from initial detection masks.
        
        Args:
            masks: Binary masks [N, H, W] at full resolution
            features: Feature map [H', W', D] for computing prototypes
            features_shape: Feature map shape (H', W') for downsampling
        
        Returns:
            List of created tracks with feature-based colors
        """
        self.reset()
        
        n_masks = masks.shape[0]
        if n_masks == 0:
            return []
        
        # Downsample masks to feature resolution
        masks_down = F.interpolate(
            masks.unsqueeze(1).float(),
            size=features_shape,
            mode="nearest-exact",
        ).squeeze(1)  # [N, H', W']
        
        tracks = []
        for i in range(n_masks):
            mask_binary = masks_down[i] > 0.5
            track = self._create_track(
                masks_down[i],
                features=features,
                mask_binary=mask_binary,
            )
            tracks.append(track)
        
        return tracks
    
    def update_from_keyframe(
        self,
        new_masks: Tensor,
        features: Tensor,
        features_shape: Tuple[int, int],
    ) -> Tuple[List[Track], List[Track], List[Track], List[int]]:
        """Update tracks from new keyframe detections.
        
        Associates new detections with existing tracks using combined
        IoU and feature similarity scores. Creates new tracks for
        unmatched detections and retires unmatched existing tracks.
        
        This is the supervisory signal that introduces new instances,
        removes obsolete tracks, and corrects accumulated drift.
        
        Args:
            new_masks: New detection masks [N, H, W] at full resolution
            features: Current frame features [H', W', D]
            features_shape: Feature map shape for downsampling
        
        Returns:
            Tuple of (matched_tracks, new_tracks, retired_tracks, detection_instance_ids)
            where detection_instance_ids[i] is the persistent instance_id assigned
            to detection i in new_masks.
        """
        n_new = new_masks.shape[0]
        
        if n_new == 0:
            # No detections - retire all active tracks
            retired_tracks = []
            for track in self.active_tracks:
                track.terminate()
                retired_tracks.append(track)
            logger.debug(f"No detections - retiring all {len(retired_tracks)} tracks")
            return [], [], retired_tracks, []
        
        # Downsample new masks to feature resolution
        new_masks_down = F.interpolate(
            new_masks.unsqueeze(1).float(),
            size=features_shape,
            mode="nearest-exact",
        ).squeeze(1)  # [N, H', W']
        
        active = self.active_tracks
        
        # Get binary masks for new detections
        new_masks_binary = new_masks_down > 0.5
        
        if not active:
            # No existing tracks - create new ones for all detections
            new_tracks = []
            detection_instance_ids = [-1] * n_new
            for i in range(n_new):
                track = self._create_track(
                    new_masks_down[i],
                    features=features,
                    mask_binary=new_masks_binary[i],
                )
                new_tracks.append(track)
                detection_instance_ids[i] = int(track.instance_id)
            logger.debug(f"No existing tracks - created {len(new_tracks)} new tracks")
            return [], new_tracks, [], detection_instance_ids
        
        # Get binary masks for existing tracks
        existing_masks = torch.stack([t.mask_probs > 0.5 for t in active], dim=0)
        eligible_tracks, eligibility_ratios = self._compute_track_eligibility(active)

        # Compute IoU matrix [N_existing, N_new]
        iou_matrix = compute_iou_matrix(existing_masks, new_masks_binary)

        # Spatial gate: only consider pairs with sufficient overlap to be a
        # plausible continuation across the keyframe interval.
        valid_pairs = iou_matrix >= self.config.min_iou_gate
        
        # Compute feature prototypes for new masks
        new_prototypes = compute_prototype_matrix(features, new_masks_binary)
        
        # Compute feature prototypes for existing tracks using current features
        existing_prototypes = compute_prototype_matrix(features, existing_masks)
        
        # Compute full cosine similarity matrix [N_existing, N_new] once
        cosine_matrix = compute_cosine_similarity_matrix(existing_prototypes, new_prototypes)
        
        # Clamp cosine similarity to [0, 1] for score computation
        cosine_matrix = cosine_matrix.clamp(min=0.0)
        
        # Compute combined score matrix
        score_matrix = (
            self.config.iou_weight * iou_matrix +
            self.config.feature_weight * cosine_matrix
        )

        # The score definition stays unchanged. We apply IoU gating and
        # propagated-area eligibility only when deciding which pairs can match.
        matching_score_matrix = score_matrix.masked_fill(~valid_pairs, -1.0)
        if matching_score_matrix.numel() > 0:
            matching_score_matrix = matching_score_matrix.masked_fill(
                ~eligible_tracks.unsqueeze(1), -1.0
            )
        
        # Debug: Log score matrix statistics
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Matching {len(active)} existing tracks with {n_new} new detections "
                f"(threshold={self.config.alignment_threshold:.2f})"
            )
            logger.debug(
                f"Score matrix: max={score_matrix.max():.3f}, mean={score_matrix.mean():.3f}"
            )
            logger.debug(
                "Eligible propagated tracks for matching: "
                f"{eligible_tracks.sum().item()}/{len(active)} "
                f"(eligibility_factor={self.config.eligibility_factor:.3f}, "
                f"min_iou_gate={self.config.min_iou_gate:.3f})"
            )
        
        # Store original scores for logging
        original_iou = iou_matrix.clone()
        original_cosine = cosine_matrix.clone()
        
        # Greedy assignment with alignment threshold
        matched_tracks = []
        matched_detection_indices = set()
        matched_track_indices = set()
        detection_instance_ids = [-1] * n_new
        
        while True:
            if matching_score_matrix.numel() == 0:
                break

            # Find best match
            max_score = matching_score_matrix.max()
            if max_score < self.config.alignment_threshold:
                break

            flat_idx = matching_score_matrix.argmax()
            track_idx = (flat_idx // matching_score_matrix.shape[1]).item()
            det_idx = (flat_idx % matching_score_matrix.shape[1]).item()

            # Skip if already matched
            if track_idx in matched_track_indices or det_idx in matched_detection_indices:
                matching_score_matrix[track_idx, det_idx] = -1
                continue
            
            # Update matched track
            track = active[track_idx]
            track.update_detected(new_masks_down[det_idx])
            self._update_track_prototype(track, new_prototypes[det_idx])
            matched_tracks.append(track)
            matched_detection_indices.add(det_idx)
            matched_track_indices.add(track_idx)
            detection_instance_ids[det_idx] = int(track.instance_id)
            
            # Trace: Log individual match details
            if logger.isEnabledFor(TRACE):
                iou_val = original_iou[track_idx, det_idx].item()
                cos_val = original_cosine[track_idx, det_idx].item()
                logger.log(
                    TRACE,
                    f"  MATCH: track {track.track_id} <-> det {det_idx} "
                    f"(score={max_score:.3f}, iou={iou_val:.3f}, cosine={cos_val:.3f})"
                )
            
            # Remove matched pair from consideration
            matching_score_matrix[track_idx, :] = -1
            matching_score_matrix[:, det_idx] = -1
        
        # Create new tracks for unmatched detections
        new_tracks = []
        for i in range(n_new):
            if i not in matched_detection_indices:
                track = self._create_track(
                    new_masks_down[i],
                    features=features,
                    mask_binary=new_masks_binary[i],
                )
                new_tracks.append(track)
                detection_instance_ids[i] = int(track.instance_id)
                # Trace: Log individual new track details
                if logger.isEnabledFor(TRACE):
                    # Find best score this detection had with any track
                    if len(active) > 0:
                        best_score = original_iou[:, i].max().item() * self.config.iou_weight + \
                                     original_cosine[:, i].max().item() * self.config.feature_weight
                        logger.log(
                            TRACE,
                            f"  NEW: det {i} -> track {track.track_id} "
                            f"(best_score={best_score:.3f} < threshold)"
                        )
                    else:
                        logger.log(TRACE, f"  NEW: det {i} -> track {track.track_id}")
        
        # Retire unmatched existing tracks
        retired_tracks = []
        for idx, track in enumerate(active):
            if idx not in matched_track_indices:
                # Trace: Log why track was retired
                if logger.isEnabledFor(TRACE):
                    best_score = original_iou[idx, :].max().item() * self.config.iou_weight + \
                                 original_cosine[idx, :].max().item() * self.config.feature_weight
                    best_det = original_iou[idx, :].argmax().item()
                    iou_val = original_iou[idx, best_det].item()
                    cos_val = original_cosine[idx, best_det].item()
                    reason = "unmatched"
                    if not bool(eligible_tracks[idx].item()):
                        reason = (
                            "ineligible_propagated_area "
                            f"(ratio={eligibility_ratios[idx].item():.3f})"
                        )
                    elif not bool(valid_pairs[idx].any().item()):
                        reason = "iou_gated"
                    logger.log(
                        TRACE,
                        f"  RETIRE: track {track.track_id} (age={track.age}) "
                        f"best match was det {best_det} "
                        f"(score={best_score:.3f}, iou={iou_val:.3f}, cosine={cos_val:.3f}, "
                        f"reason={reason})"
                    )
                track.terminate()
                retired_tracks.append(track)
        
        return matched_tracks, new_tracks, retired_tracks, detection_instance_ids
    
    def update_from_propagation(self, propagated_probs: Tensor) -> None:
        """Update tracks from propagated probability maps.
        
        Args:
            propagated_probs: Propagated probabilities [H, W, M]
                             where M = num_active_tracks + 1 (background)
        """
        active = self.active_tracks
        
        # Skip background (index 0), assign probs to tracks in order
        for i, track in enumerate(active):
            # Probability index is i+1 (0 is background)
            if i + 1 < propagated_probs.shape[-1]:
                track.update_propagated(propagated_probs[:, :, i + 1])
    
    def get_mask_image(
        self,
        output_shape: Tuple[int, int],
    ) -> Tensor:
        """Generate instance mask image from active tracks.
        
        Args:
            output_shape: Output image shape (H, W)
        
        Returns:
            mask_image: uint16 tensor [H, W] where pixel value = instance_id
        """
        h, w = output_shape
        # Use int32 for masked assignment (uint16 doesn't support masked_fill_)
        mask_image = torch.zeros((h, w), dtype=torch.int32, device=self.device)
        
        for track in self.active_tracks:
            if track.mask_probs is None:
                continue
            
            # Upsample probability map to output resolution
            probs_up = F.interpolate(
                track.mask_probs.unsqueeze(0).unsqueeze(0),
                size=output_shape,
                mode="nearest",
            ).squeeze()
            
            # Threshold and assign instance ID
            mask = probs_up > 0.5
            mask_image[mask] = track.instance_id
        
        # Convert to uint16 for output
        return mask_image.to(torch.uint16)
        
    def colorize_mask_image(self, mask_image: Tensor) -> np.ndarray:
        """Colorize a precomputed instance-id mask image.

        Args:
            mask_image: uint16 tensor [H, W] where pixel value = instance_id

        Returns:
            color_image: uint8 array [H, W, 3] RGB
        """
        mask_np = mask_image.detach().cpu().numpy().astype(np.uint16)
        max_id = int(mask_np.max()) if mask_np.size > 0 else 0
        color_lut = np.zeros((max_id + 1, 3), dtype=np.uint8)

        for track in self.active_tracks:
            if track.color is None:
                continue
            inst_id = track.instance_id
            if inst_id <= max_id:
                color_lut[inst_id] = track.color

        return color_lut[mask_np]
    
    def get_probability_tensor(self) -> Optional[Tensor]:
        """Get stacked probability tensor for all active tracks.
        
        Returns:
            probs: Tensor [H, W, M+1] including background, or None if no tracks
        """
        probs_and_ids = self.get_probability_tensor_with_instance_ids()
        if probs_and_ids is None:
            return None
        return probs_and_ids[0]

    def get_probability_tensor_with_instance_ids(
        self,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Get stacked probability tensor and aligned instance ids.

        Returns:
            Tuple of:
              probs: Tensor [H, W, M+1] including background
              instance_ids: Tensor [M] where channel i+1 maps to instance_ids[i]
            or None if no active tracks.
        """
        active = self.active_tracks
        if not active:
            return None
        
        # Get feature map shape from first track
        h, w = active[0].mask_probs.shape
        
        # Create background probability (1 - sum of all track probs)
        all_probs = torch.stack([t.mask_probs for t in active], dim=-1)
        bg_prob = 1.0 - all_probs.sum(dim=-1, keepdim=True).clamp(max=1.0)
        
        # Concatenate background + track probs
        probs = torch.cat([bg_prob, all_probs], dim=-1)
        instance_ids = torch.tensor(
            [t.instance_id for t in active],
            dtype=torch.int32,
            device=self.device,
        )
        return probs, instance_ids
