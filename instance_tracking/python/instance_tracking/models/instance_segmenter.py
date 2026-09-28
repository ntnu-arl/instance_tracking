"""Abstract instance segmenter interface and implementations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from spark_config import Config, register_config
from torch import Tensor, nn

logger = logging.getLogger(__name__)


__all__ = [
    "InstanceSegmenter",
    "FastSAMSegmenter",
    "FastSAMConfig",
]


class InstanceSegmenter(ABC, nn.Module):
    """Abstract base class for instance segmentation models.
    
    Implementations should take an RGB image and return instance masks
    with corresponding bounding boxes.
    """
    
    @abstractmethod
    def segment(
        self, img, device=None
    ) -> Tuple[Tensor, Tensor]:
        """Segment an image into instance masks.
        
        Args:
            img: Input image (np.ndarray uint8 [H, W, 3] RGB or torch.Tensor)
            device: Device to run inference on
            
        Returns:
            masks: Boolean tensor of shape [N, H, W] where N is number of instances
            boxes: Tensor of shape [N, 4] with bounding boxes in xyxy format
        """
        pass


class FastSAMSegmenter(InstanceSegmenter):
    """FastSAM-based instance segmenter.
    
    Wraps ultralytics FastSAM for instance segmentation.
    """
    
    def __init__(self, config, verbose=False):
        """Initialize FastSAM segmenter.
        
        Args:
            config: FastSAMConfig with model parameters
            verbose: Whether to print verbose output during inference
        """
        super().__init__()
        from ultralytics import FastSAM
        
        self.config = config
        self.verbose = verbose
        self.sam = FastSAM(config.model_name)
    
    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = FastSAMConfig()
        config.update(kwargs)
        return cls(config)
    
    def train(self, mode):
        """Don't pass train to underlying model."""
        pass
    
    def segment(
        self, img, device=None
    ) -> Tuple[Tensor, Tensor]:
        """Segment image using FastSAM.
        
        Args:
            img: Input RGB image as numpy array [H, W, 3] uint8
            device: Device to run inference on
            
        Returns:
            masks: Boolean tensor [N, H, W]
            boxes: Int32 tensor [N, 4] in xyxy format
        """
        # Debug: Log image stats and GPU memory before inference
        if logger.isEnabledFor(logging.DEBUG):
            if isinstance(img, np.ndarray):
                logger.debug(
                    f"FastSAM input: shape={img.shape}, dtype={img.dtype}, "
                    f"min={img.min()}, max={img.max()}, mean={img.mean():.1f}"
                )
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.debug(f"GPU memory before FastSAM: {mem_alloc:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        try:
            results = self.sam(
                source=img,
                device=device,
                retina_masks=True,
                imgsz=self.config.output_size,
                conf=self.config.confidence,
                iou=self.config.iou,
                verbose=self.verbose,
            )
        except Exception as e:
            logger.error(f"FastSAM inference failed: {e}")
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.error(f"GPU memory at failure: {mem_alloc:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            raise
        
        # Debug: Log GPU memory after inference
        if logger.isEnabledFor(logging.DEBUG) and torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"GPU memory after FastSAM: {mem_alloc:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        if results[0].masks is None:
            # No detections - return empty tensors
            h, w = img.shape[:2] if hasattr(img, 'shape') else (0, 0)
            logger.warning(f"FastSAM returned no masks for image of shape {img.shape if hasattr(img, 'shape') else 'unknown'}")
            return (
                torch.zeros((0, h, w), dtype=torch.bool),
                torch.zeros((0, 4), dtype=torch.int32),
            )
        
        masks = results[0].masks.data.to(torch.bool)
        n_masks = masks.shape[0]
        
        # Debug: Log coverage statistics
        if logger.isEnabledFor(logging.DEBUG):
            h, w = masks.shape[1], masks.shape[2]
            total_pixels = h * w
            # Union of all masks = foreground, rest = background
            foreground = masks.any(dim=0)  # [H, W]
            fg_pixels = foreground.sum().item()
            fg_pct = 100.0 * fg_pixels / total_pixels
            bg_pct = 100.0 - fg_pct
            logger.debug(
                f"FastSAM detected {n_masks} instances, "
                f"coverage: {fg_pct:.1f}% foreground, {bg_pct:.1f}% background"
            )
        
        return (
            masks,
            results[0].boxes.xyxy.to(torch.int32),
        )
    
    def forward(self, img, device=None) -> Tuple[Tensor, Tensor]:
        """Forward pass (alias for segment)."""
        return self.segment(img, device=device)


@register_config("instance_segmenter", name="fastsam", constructor=FastSAMSegmenter)
@dataclass
class FastSAMConfig(Config):
    """Configuration for FastSAM instance segmenter."""
    
    model_name: str = "FastSAM-x.pt"
    confidence: float = 0.55
    iou: float = 0.85
    output_size: int = 736
    
    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
