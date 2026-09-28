"""GPU preprocessing pipeline for DINOv3 feature extraction."""

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn


__all__ = [
    "DINOv3Preprocessing",
    "compute_target_dimensions",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]


# ImageNet normalization constants used by DINOv3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_target_dimensions(
    height: int,
    width: int,
    short_side: int,
    patch_size: int,
) -> Tuple[int, int]:
    """Compute target (height, width) for images.
    
    Scales the image so the shorter side equals short_side,
    then rounds up to nearest multiple of patch_size.
    
    Args:
        height: Original image height
        width: Original image width
        short_side: Target size for the shorter side
        patch_size: Model patch size (dimensions must be multiples of this)
    
    Returns:
        (target_height, target_width) as multiples of patch_size
    """
    # Scale so shorter side equals short_side while preserving aspect ratio
    if width > height:
        scale = short_side / height
    else:
        scale = short_side / width
    
    new_height = height * scale
    new_width = width * scale
    
    # Round up to nearest multiple of patch_size
    target_height = math.ceil(new_height / patch_size) * patch_size
    target_width = math.ceil(new_width / patch_size) * patch_size
    
    return (target_height, target_width)


class DINOv3Preprocessing(nn.Module):
    """GPU-accelerated preprocessing for DINOv3 using pure PyTorch operations.
    
    Performs all preprocessing (resize, normalization) as tensor operations
    on GPU, avoiding CPU-GPU transfers and PIL conversions in the main loop.
    """
    
    def __init__(
        self,
        target_height: int,
        target_width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize preprocessing module.
        
        Args:
            target_height: Target image height (must be multiple of patch_size)
            target_width: Target image width (must be multiple of patch_size)
            device: Target device for tensors
            dtype: Target dtype for output tensors
        """
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.target_device = device
        self.dtype = dtype
        
        # Store normalization parameters as buffers for GPU execution
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD, dtype=dtype, device=device).view(1, 3, 1, 1)
        )
    
    def forward(self, img: Tensor) -> Tensor:
        """Preprocess image tensor on GPU.
        
        Args:
            img: Image tensor, expected shapes:
                - (H, W, C) uint8 tensor (channels last, RGB)
                - (C, H, W) uint8 tensor (channels first, RGB)
                - (B, C, H, W) float tensor (already batched)
                Can be on CPU or GPU.
        
        Returns:
            Preprocessed tensor of shape (1, 3, target_height, target_width)
        """
        # Move to target device if needed
        if img.device != self.target_device:
            img = img.to(self.target_device)
        
        # Handle channels-last format (H, W, C) -> (C, H, W)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        
        # Convert to float and normalize to [0, 1]
        if img.dtype == torch.uint8:
            img = img.to(self.dtype) / 255.0
        elif img.dtype == torch.uint16:
            img = img.to(self.dtype) / 65535.0
        elif img.dtype != self.dtype:
            img = img.to(self.dtype)
        
        # Add batch dimension if needed
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        # Resize using bicubic interpolation
        img = F.interpolate(
            img,
            size=(self.target_height, self.target_width),
            mode="bicubic",
            align_corners=False,
        )
        
        # Apply ImageNet normalization
        img = (img - self.mean) / self.std
        
        return img.contiguous()
    
    def update_target_size(self, target_height: int, target_width: int) -> None:
        """Update target dimensions (useful when processing different sequences).
        
        Args:
            target_height: New target height
            target_width: New target width
        """
        self.target_height = target_height
        self.target_width = target_width


def create_preprocessing(
    height: int,
    width: int,
    short_side: int,
    patch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> DINOv3Preprocessing:
    """Factory function to create preprocessing pipeline.
    
    Computes target dimensions and creates preprocessing module.
    
    Args:
        height: Original image height
        width: Original image width
        short_side: Target size for shorter side
        patch_size: Model patch size
        device: Target device
        dtype: Target dtype
    
    Returns:
        Configured DINOv3Preprocessing module
    """
    target_height, target_width = compute_target_dimensions(
        height, width, short_side, patch_size
    )
    
    return DINOv3Preprocessing(
        target_height=target_height,
        target_width=target_width,
        device=device,
        dtype=dtype,
    )
