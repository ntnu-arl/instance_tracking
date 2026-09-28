"""Label propagation using DINOv3 features."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from spark_config import Config
from torch import Tensor


__all__ = [
    "PropagationConfig",
    "LabelPropagator",
    "make_neighborhood_mask",
    "propagate_labels",
]


@dataclass
class PropagationConfig(Config):
    """Configuration for label propagation."""
    
    max_context_length: int = 7
    neighborhood_size: int = 12
    neighborhood_shape: str = "circle"  # "circle" or "square"
    topk: int = 5
    temperature: float = 0.2
    
    def __post_init__(self) -> None:
        shape = self.neighborhood_shape.lower()
        if shape not in {"circle", "square"}:
            raise ValueError(f"Unsupported neighborhood_shape={self.neighborhood_shape}")
        self.neighborhood_shape = shape
    
    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


def make_neighborhood_mask(
    height: int,
    width: int,
    neighborhood_size: int,
    shape: str = "circle",
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create a neighborhood mask for spatial attention.
    
    Creates a boolean mask indicating which spatial positions can attend
    to which other positions, based on distance.
    
    Args:
        height: Feature map height
        width: Feature map width
        neighborhood_size: Maximum distance for attention
        shape: "circle" for L2 distance, "square" for L-inf distance
        device: Device to construct the mask on
    
    Returns:
        mask: Boolean tensor of shape [H, W, H, W] where mask[i,j,k,l]
              indicates if position (i,j) can attend to (k,l)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create coordinate grids
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    # Compute pairwise distances
    # Shape: [H, W, H, W]
    dy = yy.unsqueeze(2).unsqueeze(3) - yy.unsqueeze(0).unsqueeze(1)
    dx = xx.unsqueeze(2).unsqueeze(3) - xx.unsqueeze(0).unsqueeze(1)
    
    if shape == "circle":
        dist = torch.sqrt(dy.float() ** 2 + dx.float() ** 2)
    else:  # square
        dist = torch.maximum(torch.abs(dy), torch.abs(dx)).float()
    
    mask = dist <= neighborhood_size
    
    return mask


def propagate_labels(
    current_features: Tensor,
    context_features: Tensor,
    context_probs: Tensor,
    neighborhood_mask: Tensor,
    topk: int,
    temperature: float,
) -> Tensor:
    """Propagate labels from context frames to current frame.
    
    Uses feature similarity to transfer probability maps from context
    frames to the current frame.
    
    Args:
        current_features: Features for current frame [H, W, D]
        context_features: Features for context frames [T, H, W, D]
        context_probs: Probability maps for context frames [T, H, W, M]
        neighborhood_mask: Spatial attention mask [H, W, H, W]
        topk: Number of top matches to consider per position
        temperature: Softmax temperature for attention weights
    
    Returns:
        current_probs: Propagated probability map [H, W, M]
    """
    t, h, w, M = context_probs.shape
    
    # Compute dot product attention
    # current_features: [H, W, D]
    # context_features: [T, H, W, D]
    # Result: [H, W, T, H, W]
    dot = torch.einsum("ijd, tuvd -> ijtuv", current_features, context_features)
    
    # Apply neighborhood mask (only attend to nearby positions)
    # For each current patch (i,j), only allow attention to (u,v) within neighborhood
    # torch.where returns a tensor of the same shape as condition (neighborhood_mask[…])
    # takes each element from x where condition is True, else from y
    dot = torch.where(neighborhood_mask[:, :, None, :, :], dot, -torch.inf)
    
    # Flatten spatial and temporal dimensions for top-k selection
    # Each row corresponds to a patch-position in current frame
    # Columns correspond to all patch-positions in context frames
    dot = dot.flatten(2, -1).flatten(0, 1)  # [H*W, T*H*W]
    
    # Select top-k matches per position
    # Top-k taken along column-dimension – want only the best k matches among all context positions
    # Everything else set to -inf
    k_largest = torch.topk(dot, dim=1, k=topk).values
    dot = torch.where(dot >= k_largest[:, -1:], dot, -torch.inf)
    
    # Compute attention weights with temperature scaling
    # Again along column-dimension – converts the remaining scores into weights for each current patch
    weights = F.softmax(dot / temperature, dim=1)  # [H*W, T*H*W]
    
    # Apply attention to get propagated probabilities
    context_probs_flat = context_probs.flatten(0, 2)  # [T*H*W, M]
    current_probs = torch.mm(weights, context_probs_flat)  # [H*W, M]
    
    # Normalize probabilities
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    
    # Reshape back to spatial dimensions
    return current_probs.unflatten(0, (h, w))  # [H, W, M]


def postprocess_probs(probs: Tensor) -> Tensor:
    """Normalize probability maps to [0, 1] range per mask.
    
    Args:
        probs: Probability tensor of shape [N, H, W] or [B, N, H, W]
    
    Returns:
        Normalized probabilities
    """
    if probs.ndim == 3:
        probs = probs.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    vmin = probs.flatten(2, 3).min(dim=2).values
    vmax = probs.flatten(2, 3).max(dim=2).values
    
    denom = (vmax - vmin)[:, :, None, None]
    denom = denom.clamp(min=1e-8)
    
    probs = (probs - vmin[:, :, None, None]) / denom
    probs = torch.nan_to_num(probs, nan=0.0)
    
    if squeeze:
        probs = probs.squeeze(0)
    
    return probs


class LabelPropagator:
    """Manages label propagation with context queue.
    
    Maintains a queue of recent features and probability maps for
    propagating labels across frames.
    """
    
    def __init__(self, config: PropagationConfig, device: torch.device):
        """Initialize label propagator.
        
        Args:
            config: PropagationConfig with propagation parameters
            device: Device to store tensors on
        """
        self.config = config
        self.device = device
        
        self._features_queue: List[Tensor] = []
        self._probs_queue: List[Tensor] = []
        self._first_features: Optional[Tensor] = None
        self._first_probs: Optional[Tensor] = None
        self._neighborhood_mask: Optional[Tensor] = None
        self._initialized = False
    
    def reset(self) -> None:
        """Reset the propagator state."""
        self._features_queue.clear()
        self._probs_queue.clear()
        self._first_features = None
        self._first_probs = None
        self._neighborhood_mask = None
        self._initialized = False
    
    def initialize(
        self,
        first_features: Tensor,
        first_probs: Tensor,
    ) -> None:
        """Initialize with first frame features and probabilities.
        
        Args:
            first_features: Features for first frame [H, W, D]
            first_probs: Initial probability map [H, W, M]
        """
        self._first_features = first_features.to(self.device)
        self._first_probs = first_probs.to(self.device)
        
        h, w = first_features.shape[:2]
        self._neighborhood_mask = make_neighborhood_mask(
            h, w,
            self.config.neighborhood_size,
            self.config.neighborhood_shape,
            device=self.device,
        ).to(self.device)
        
        self._features_queue.clear()
        self._probs_queue.clear()
        self._initialized = True
    
    @property
    def is_initialized(self) -> bool:
        """Check if propagator is initialized."""
        return self._initialized
    
    @property
    def num_masks(self) -> int:
        """Get number of masks being tracked."""
        if self._first_probs is None:
            return 0
        return self._first_probs.shape[-1]
    
    def propagate(self, current_features: Tensor) -> Tensor:
        """Propagate labels to current frame.
        
        Args:
            current_features: Features for current frame [H, W, D]
        
        Returns:
            current_probs: Propagated probability map [H, W, M]
        """
        if not self._initialized:
            raise RuntimeError("LabelPropagator not initialized. Call initialize() first.")
        
        current_features = current_features.to(self.device)
        
        # Build context from first frame + queue
        context_features = torch.stack(
            [self._first_features, *self._features_queue], dim=0
        )
        context_probs = torch.stack(
            [self._first_probs, *self._probs_queue], dim=0
        )
        
        # Propagate labels
        current_probs = propagate_labels(
            current_features,
            context_features,
            context_probs,
            self._neighborhood_mask,
            self.config.topk,
            self.config.temperature,
        )
        
        # Update queues
        self._features_queue.append(current_features)
        self._probs_queue.append(current_probs)
        
        # Trim queues if needed
        while len(self._features_queue) > self.config.max_context_length:
            self._features_queue.pop(0)
        while len(self._probs_queue) > self.config.max_context_length:
            self._probs_queue.pop(0)
        
        return current_probs
    
    def update_reference(
        self,
        new_features: Tensor,
        new_probs: Tensor,
    ) -> None:
        """Update the reference frame (for keyframe updates).
        
        This replaces the first frame with new keyframe data.
        
        Args:
            new_features: Features for new reference frame [H, W, D]
            new_probs: Probability map for new reference [H, W, M]
        """
        self._first_features = new_features.to(self.device)
        self._first_probs = new_probs.to(self.device)
        
        # Clear context queue since we have a new reference
        self._features_queue.clear()
        self._probs_queue.clear()
