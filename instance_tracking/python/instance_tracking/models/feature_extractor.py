"""DINOv3 feature extractor for mask propagation."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from spark_config import Config, register_config
from torch import Tensor, nn

logger = logging.getLogger(__name__)


__all__ = [
    "DINOv3FeatureExtractor",
    "DINOv3Config",
    "MODEL_DINOV3_VITS",
    "MODEL_DINOV3_VITSP",
    "MODEL_DINOV3_VITB",
    "MODEL_DINOV3_VITL",
]


# Model variant names
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

# GitHub repository for DINOv3 model architecture
DINOV3_GITHUB_REPO = "facebookresearch/dinov3"


def _get_workspace_models_path():
    """Get path to workspace models directory.
    
    Returns path to <workspace>/models/dinov3/ which is the recommended
    location for DINOv3 weights.
    """
    # Navigate from this file: python/instance_tracking/models/feature_extractor.py
    # Up to: instance_tracking/
    # Then to workspace: ../../..
    # Then to models: models/dinov3
    this_file = Path(__file__).absolute()

    #TODO: Make this more robust by using the workspace root from the environment variable
    workspace_root = this_file.parent.parent.parent.parent.parent.parent.parent
    return workspace_root / "models" / "dinov3"


# Default weights path - workspace models directory
DEFAULT_WEIGHTS_ROOT = str(_get_workspace_models_path())

# Mapping from model name to weight filename
MODEL_WEIGHT_FILENAMES: Dict[str, str] = {
    MODEL_DINOV3_VITS: "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    MODEL_DINOV3_VITSP: "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    MODEL_DINOV3_VITB: "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    MODEL_DINOV3_VITL: "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def _resolve_path(config_value: str, env_var: str, default: str) -> Path:
    """Resolve a path from config, environment variable, or default.
    
    Priority: config_value > env_var > default
    """
    if config_value and config_value != default:
        return Path(config_value).expanduser()
    
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser()
    
    return Path(default).expanduser()


@register_config("feature_extractor", name="dinov3", constructor=lambda cfg: DINOv3FeatureExtractor(cfg))
@dataclass
class DINOv3Config(Config):
    """Configuration for DINOv3 feature extractor.
    
    The model architecture is loaded from the official GitHub repository
    (facebookresearch/dinov3) using torch.hub. Weights are loaded from
    a local file specified by weights_root.
    """
    
    model_name: str = MODEL_DINOV3_VITS
    weights_root: str = DEFAULT_WEIGHTS_ROOT
    short_side: int = 960
    
    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
    
    def get_weights_root(self) -> Path:
        """Get resolved weights root path.
        
        Can be overridden via DINOV3_WEIGHTS_PATH environment variable.
        """
        return _resolve_path(self.weights_root, "DINOV3_WEIGHTS_PATH", DEFAULT_WEIGHTS_ROOT)


class DINOv3FeatureExtractor(nn.Module):
    """DINOv3 feature extractor for dense feature computation.
    
    Extracts patch-level features from images using a DINOv3 vision transformer.
    Features are L2-normalized and suitable for nearest-neighbor matching.
    """
    
    def __init__(self, config: DINOv3Config):
        """Initialize DINOv3 feature extractor.
        
        Args:
            config: DINOv3Config with model parameters and paths
        """
        super().__init__()
        self.config = config
        self._model: Optional[nn.Module] = None
        self._canary_param = nn.Parameter(torch.empty(0))
    
    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = DINOv3Config()
        config.update(kwargs)
        return cls(config)
    
    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device
    
    @property
    def model(self) -> nn.Module:
        """Lazy-load the DINOv3 model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    @property
    def patch_size(self) -> int:
        """Get the patch size of the loaded model."""
        return self.model.patch_size
    
    def _load_model(self) -> nn.Module:
        """Load DINOv3 model from GitHub repository.
        
        The model architecture is downloaded from the official GitHub repository
        on first run and cached locally. Subsequent runs use the cached version.
        Weights are loaded from a local file.
        """
        weights_root = self.config.get_weights_root()
        model_name = self.config.model_name
        
        if model_name not in MODEL_WEIGHT_FILENAMES:
            raise ValueError(f"Unknown model_name={model_name}")
        
        weights_path = weights_root / MODEL_WEIGHT_FILENAMES[model_name]
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        # Load model architecture from GitHub repository
        # First run downloads and caches, subsequent runs use cache
        # Note: pretrained=False because we'll load weights manually from local file
        model = torch.hub.load(
            repo_or_dir=DINOV3_GITHUB_REPO,
            model=model_name,
            source="github",
            pretrained=False,
        )
        
        # Load weights from local file
        checkpoint = torch.load(weights_path, map_location="cpu")
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def to(self, *args, **kwargs):
        """Move model to device."""
        result = super().to(*args, **kwargs)
        if self._model is not None:
            self._model = self._model.to(*args, **kwargs)
        return result
    
    def extract_features(self, img: Tensor) -> Tensor:
        """Extract dense features from a preprocessed image.
        
        Args:
            img: Preprocessed image tensor of shape [C, H, W] or [B, C, H, W]
                 (normalized, on correct device)
        
        Returns:
            features: L2-normalized features of shape [H', W', D] or [B, H', W', D]
                      where H' = H/patch_size, W' = W/patch_size, D = feature dim
        """
        # Ensure batch dimension
        if img.ndim == 3:
            img = img.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Get intermediate layer features
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(img, n=1, reshape=True)[0]
        
        # Debug: Log GPU memory after DINOv3 forward pass
        if logger.isEnabledFor(logging.DEBUG) and torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"GPU memory after DINOv3: {mem_alloc:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        
        # Reshape from [B, D, H', W'] to [B, H', W', D]
        feats = feats.movedim(-3, -1)
        
        # L2 normalize
        feats = F.normalize(feats, dim=-1, p=2)
        
        if squeeze_output:
            feats = feats.squeeze(0)
        
        return feats
    
    def forward(self, img: Tensor) -> Tensor:
        """Forward pass (alias for extract_features)."""
        return self.extract_features(img)
