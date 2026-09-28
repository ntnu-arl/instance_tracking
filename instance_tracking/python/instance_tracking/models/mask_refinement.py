"""Mask refinement utilities for instance tracking."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from spark_config import Config
from torch import Tensor


__all__ = [
    "MaskRefinementConfig",
    "RefinedMaskOutput",
    "MaskRefiner",
]


def _ensure_odd(value: int) -> int:
    if value < 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _scale_kernel(value: int, scale: float) -> int:
    scaled = int(round(value * scale))
    return _ensure_odd(max(1, scaled))


def _scale_iters(value: int, scale: float) -> int:
    return max(0, int(round(value * scale)))


@dataclass
class MaskRefinementConfig(Config):
    """Configuration for mask refinement."""

    enabled: bool = True
    stronger_on_keyframes: bool = True
    keyframe_multiplier: float = 1.5

    # Boundary behavior: "balanced", "conservative", "edge_faithful"
    boundary_mode: str = "balanced"

    # Smoothing at feature resolution (reduces patchiness)
    smooth_probs: bool = False
    prob_kernel: int = 3
    prob_iters: int = 1

    # Upsampling probabilities to output resolution
    upsample_mode: str = "bilinear"  # "bilinear" or "nearest"
    upsample_align_corners: bool = False

    # Optional smoothing at output resolution
    post_smooth: bool = True
    post_kernel: int = 3
    post_iters: int = 1

    # Morphological refinement on binary masks
    morph_open: bool = True
    morph_close: bool = True
    morph_kernel: int = 3
    morph_iters: int = 1
    kernel_tolerance: float = 1.0e-3

    # Thresholding and overlap resolution
    threshold: float = 0.5
    resolve_overlaps: bool = True
    overlap_strategy: str = "confidence"  # "confidence" or "order"

    # Keyframe-only refinement of FastSAM masks (before downsampling/propagation)
    keyframe_refine_enabled: bool = True
    keyframe_refine_morph_open: bool = True
    keyframe_refine_morph_close: bool = True
    keyframe_refine_kernel: int = 3
    keyframe_refine_iters: int = 1
    keyframe_refine_dilate: bool = False
    keyframe_refine_dilation_iters: int = 1

    def __post_init__(self) -> None:
        self.boundary_mode = self.boundary_mode.lower()
        if self.boundary_mode not in {"balanced", "conservative", "edge_faithful"}:
            raise ValueError(f"Unsupported boundary_mode={self.boundary_mode}")

        self.upsample_mode = self.upsample_mode.lower()
        if self.upsample_mode not in {"bilinear", "nearest"}:
            raise ValueError(f"Unsupported upsample_mode={self.upsample_mode}")

        self.overlap_strategy = self.overlap_strategy.lower()
        if self.overlap_strategy not in {"confidence", "order"}:
            raise ValueError(f"Unsupported overlap_strategy={self.overlap_strategy}")

        self.prob_kernel = _ensure_odd(self.prob_kernel)
        self.post_kernel = _ensure_odd(self.post_kernel)
        self.morph_kernel = _ensure_odd(self.morph_kernel)
        self.keyframe_refine_kernel = _ensure_odd(self.keyframe_refine_kernel)


@dataclass
class RefinedMaskOutput:
    """Final mask image plus the winning-label confidence at each output pixel."""

    mask_image: Tensor
    label_confidence: Tensor


class MaskRefiner:
    """Refines instance masks derived from probability maps."""

    def __init__(self, config: MaskRefinementConfig, device: torch.device):
        self.config = config
        self.device = device

    def _avg_smooth(self, probs: Tensor, kernel: int, iters: int) -> Tensor:
        if iters <= 0 or kernel <= 1:
            return probs
        x = probs.unsqueeze(1)
        for _ in range(iters):
            x = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
        return x.squeeze(1)

    def _erode(self, masks: Tensor, kernel: int, iters: int, tol: float) -> Tensor:
        if iters <= 0 or kernel <= 1:
            return masks
        weight = torch.ones((1, 1, kernel, kernel), device=masks.device)
        x = masks.float().unsqueeze(1)
        target = kernel * kernel
        for _ in range(iters):
            x = F.conv2d(x, weight, padding=kernel // 2)
            x = (x >= (target - tol)).float()
        return x.squeeze(1).bool()

    def _dilate(self, masks: Tensor, kernel: int, iters: int) -> Tensor:
        if iters <= 0 or kernel <= 1:
            return masks
        weight = torch.ones((1, 1, kernel, kernel), device=masks.device)
        x = masks.float().unsqueeze(1)
        for _ in range(iters):
            x = F.conv2d(x, weight, padding=kernel // 2)
            x = (x > 0).float()
        return x.squeeze(1).bool()

    def _morph_open(self, masks: Tensor, kernel: int, iters: int, tol: float) -> Tensor:
        return self._dilate(self._erode(masks, kernel, iters, tol), kernel, iters)

    def _morph_close(self, masks: Tensor, kernel: int, iters: int, tol: float) -> Tensor:
        return self._erode(self._dilate(masks, kernel, iters), kernel, iters, tol)

    def _scale_for_keyframe(self, is_keyframe: bool) -> float:
        if is_keyframe and self.config.stronger_on_keyframes:
            return max(1.0, float(self.config.keyframe_multiplier))
        return 1.0

    def refine_keyframe_masks(self, masks: Tensor) -> Tensor:
        """Lightweight refinement for FastSAM keyframe masks.

        This is intentionally minimal: resolve overlaps and optionally apply
        a small open/close to remove speckles. Heavy erosion is avoided.

        Args:
            masks: Boolean tensor [N, H, W]

        Returns:
            Refined boolean masks [N, H, W]
        """
        if not self.config.keyframe_refine_enabled:
            return masks

        masks = masks.to(torch.bool)
        num_masks = masks.size(0)
        if num_masks == 0:
            return masks

        # Resolve overlaps by area priority: larger masks claim pixels first.
        mask_sizes = torch.sum(masks, dim=(1, 2))
        sorted_idx = torch.argsort(mask_sizes, descending=True)

        dims = masks.size()
        img = torch.zeros((dims[1], dims[2], 1), dtype=sorted_idx.dtype, device=masks.device)
        for idx in sorted_idx:
            img[masks[idx]] = idx + 1

        m_new = torch.zeros_like(masks)
        for idx in sorted_idx:
            nms_indices = (img == idx + 1)[:, :, 0]
            m_new[idx, nms_indices] = 1

        if self.config.keyframe_refine_morph_open:
            m_new = self._morph_open(
                m_new,
                self.config.keyframe_refine_kernel,
                self.config.keyframe_refine_iters,
                self.config.kernel_tolerance,
            )
        if self.config.keyframe_refine_morph_close:
            m_new = self._morph_close(
                m_new,
                self.config.keyframe_refine_kernel,
                self.config.keyframe_refine_iters,
                self.config.kernel_tolerance,
            )
        if self.config.keyframe_refine_dilate:
            m_new = self._dilate(
                m_new,
                self.config.keyframe_refine_kernel,
                self.config.keyframe_refine_dilation_iters,
            )

        return m_new.to(torch.bool)

    def refine_with_confidence(
        self,
        probs: Tensor,
        output_shape: Tuple[int, int],
        is_keyframe: bool = False,
        instance_ids: Optional[Tensor] = None,
        rgb: Optional[Tensor] = None,
    ) -> RefinedMaskOutput:
        """Refine probability maps into a hard instance-id mask and confidence map.

        Args:
            probs: Probability tensor [H', W', M] or [H', W', M+1] (bg in 0).
            output_shape: Desired output (H, W).
            is_keyframe: Whether this is a keyframe (optional stronger refine).
            instance_ids: Optional tensor [M] mapping foreground channels to
                persistent instance ids.
            rgb: Optional RGB image (unused for now, reserved for edge-aware steps).

        Returns:
            RefinedMaskOutput with:
              mask_image: uint16 tensor [H, W] with pixel values = instance_id
              label_confidence: float tensor [H, W] containing the probability of
                the chosen foreground label. Background pixels are 0.
        """
        if not self.config.enabled:
            raise RuntimeError("MaskRefiner called while disabled")

        if probs.numel() == 0:
            h, w = output_shape
            zeros = torch.zeros((h, w), dtype=torch.float32, device=probs.device)
            return RefinedMaskOutput(
                mask_image=torch.zeros((h, w), dtype=torch.uint16, device=probs.device),
                label_confidence=zeros,
            )

        # Expect background in channel 0 if present; drop it for per-instance processing
        if probs.ndim != 3:
            raise ValueError(f"Expected probs [H, W, M], got {probs.shape}")

        if probs.shape[-1] > 1:
            probs_fg = probs[..., 1:]
        else:
            probs_fg = probs

        num_instances = probs_fg.shape[-1]
        if num_instances == 0:
            h, w = output_shape
            zeros = torch.zeros((h, w), dtype=torch.float32, device=probs.device)
            return RefinedMaskOutput(
                mask_image=torch.zeros((h, w), dtype=torch.uint16, device=probs.device),
                label_confidence=zeros,
            )

        if instance_ids is not None:
            if instance_ids.numel() != num_instances:
                raise ValueError(
                    f"instance_ids length {instance_ids.numel()} != num_instances {num_instances}"
                )
            instance_ids = instance_ids.to(device=probs.device, dtype=torch.int32)

        scale = self._scale_for_keyframe(is_keyframe)

        prob_kernel = _scale_kernel(self.config.prob_kernel, scale)
        prob_iters = _scale_iters(self.config.prob_iters, scale)
        post_kernel = _scale_kernel(self.config.post_kernel, scale)
        post_iters = _scale_iters(self.config.post_iters, scale)
        morph_kernel = _scale_kernel(self.config.morph_kernel, scale)
        morph_iters = _scale_iters(self.config.morph_iters, scale)

        # [H', W', N] -> [N, H', W']
        probs_fg = probs_fg.permute(2, 0, 1).contiguous()

        if self.config.smooth_probs:
            probs_fg = self._avg_smooth(probs_fg, prob_kernel, prob_iters)

        # Upsample to full resolution
        probs_up = F.interpolate(
            probs_fg.unsqueeze(1),
            size=output_shape,
            mode=self.config.upsample_mode,
            align_corners=self.config.upsample_align_corners
            if self.config.upsample_mode == "bilinear"
            else None,
        ).squeeze(1)

        if self.config.post_smooth:
            probs_up = self._avg_smooth(probs_up, post_kernel, post_iters)

        probs_up = probs_up.clamp(min=0.0, max=1.0)
        masks = probs_up > self.config.threshold

        morph_open = self.config.morph_open
        morph_close = self.config.morph_close
        extra_erode = 0

        if self.config.boundary_mode == "conservative":
            extra_erode = max(1, morph_iters // 2 or 1)
        elif self.config.boundary_mode == "edge_faithful":
            morph_open = False

        #TODO: think about effect of this when we begin to back-project.

        if morph_open:
            masks = self._morph_open(
                masks, morph_kernel, morph_iters, self.config.kernel_tolerance
            )
        if morph_close:
            masks = self._morph_close(
                masks, morph_kernel, morph_iters, self.config.kernel_tolerance
            )
        if extra_erode > 0:
            masks = self._erode(masks, morph_kernel, extra_erode, self.config.kernel_tolerance)

        # Resolve overlaps into a single instance-id image
        h, w = output_shape
        label_confidence = torch.zeros((h, w), dtype=probs_up.dtype, device=probs.device)
        if self.config.resolve_overlaps and num_instances > 1:
            if self.config.overlap_strategy == "confidence":
                union = masks.any(dim=0)
                scores = probs_up.clone()
                scores[~masks] = float("-inf")
                label_confidence, labels = torch.max(scores, dim=0)
                label_confidence = torch.where(
                    union, label_confidence, torch.zeros_like(label_confidence)
                )
                if instance_ids is not None:
                    labels = instance_ids[labels]
                else:
                    labels = labels + 1
                mask_image = torch.where(union, labels, torch.zeros_like(labels))
            else:
                mask_image = torch.zeros((h, w), dtype=torch.int32, device=probs.device)
                for i in range(num_instances):
                    selected = masks[i]
                    mask_image[selected] = instance_ids[i] if instance_ids is not None else i + 1
                    label_confidence[selected] = probs_up[i, selected]
        else:
            mask_image = torch.zeros((h, w), dtype=torch.int32, device=probs.device)
            for i in range(num_instances):
                selected = masks[i]
                mask_image[selected] = instance_ids[i] if instance_ids is not None else i + 1
                label_confidence[selected] = probs_up[i, selected]

        return RefinedMaskOutput(
            mask_image=mask_image.to(torch.uint16),
            label_confidence=label_confidence.to(torch.float32),
        )

    def refine(
        self,
        probs: Tensor,
        output_shape: Tuple[int, int],
        is_keyframe: bool = False,
        instance_ids: Optional[Tensor] = None,
        rgb: Optional[Tensor] = None,
    ) -> Tensor:
        """Refine probability maps into a hard instance-id mask image."""

        return self.refine_with_confidence(
            probs=probs,
            output_shape=output_shape,
            is_keyframe=is_keyframe,
            instance_ids=instance_ids,
            rgb=rgb,
        ).mask_image
