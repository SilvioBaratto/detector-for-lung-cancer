"""
Lightweight classification models for nodule detection.

This module provides:
- Model registry for easy model creation
- MIFNet - Multi-scale Interleaved Fusion Network (0.7M params)
- EfficientNet3D - Lightweight 3D variant of EfficientNet

Optimized for Apple Silicon MPS with 16GB RAM.

References:
- MIFNet: https://www.nature.com/articles/s41598-024-79058-y
- EfficientNet: https://www.sciencedirect.com/science/article/pii/S0952197623010862
"""

from typing import Callable, TypeVar

import torch
from torch import nn
import torch.nn.functional as F

from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# ============================================================================
# Model Registry (Open/Closed Principle)
# ============================================================================

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}

T = TypeVar("T", bound=nn.Module)


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a model class."""
    def decorator(cls: type[T]) -> type[T]:
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    """Create a model by name from the registry."""
    if name not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())


# ============================================================================
# Squeeze-and-Excitation Block
# ============================================================================

class SqueezeExcitation3D(nn.Module):
    """3D Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        # Global average pooling
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Scale
        y = y.view(b, c, *([1] * len(spatial)))
        return x * y


# ============================================================================
# Multi-Scale Feature Extraction (MIFNet core)
# ============================================================================

class MultiScaleConv3D(nn.Module):
    """
    Multi-scale 3D convolution block.

    Uses multiple kernel sizes to capture features at different scales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        branch_channels = out_channels // len(kernels)
        remainder = out_channels - branch_channels * len(kernels)

        self.branches = nn.ModuleList()
        for i, k in enumerate(kernels):
            # Add remainder to first branch
            ch = branch_channels + (remainder if i == 0 else 0)
            self.branches.append(nn.Sequential(
                nn.Conv3d(in_channels, ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm3d(ch),
                nn.GELU(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class DepthwiseSeparableConv3D(nn.Module):
    """Depthwise separable 3D convolution for efficiency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class InterleavedFusionBlock(nn.Module):
    """
    Interleaved Fusion Block from MIFNet.

    Combines multi-scale features with channel attention.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Multi-scale convolution
        self.multi_scale = MultiScaleConv3D(in_channels, out_channels)

        # Channel attention
        self.se = SqueezeExcitation3D(out_channels)

        # Skip connection
        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.multi_scale(x)
        out = self.se(out)
        return out + identity


# ============================================================================
# MIFNet Implementation
# ============================================================================

@register_model("mifnet")
class MIFNet(nn.Module):
    """
    Multi-scale Interleaved Fusion Network for nodule classification.

    Features:
    - Only ~0.7M parameters
    - Multi-scale feature extraction
    - Squeeze-and-Excitation attention
    - Depthwise separable convolutions
    - MPS compatible

    Input: (B, 1, D, H, W) - 3D CT patch
    Output: (logits, probabilities) each of shape (B, 2)

    Reference: https://www.nature.com/articles/s41598-024-79058-y
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
        )

        # Interleaved fusion blocks with progressive downsampling
        self.stage1 = nn.Sequential(
            InterleavedFusionBlock(base_channels, base_channels * 2),
            nn.MaxPool3d(2, 2),
        )

        self.stage2 = nn.Sequential(
            InterleavedFusionBlock(base_channels * 2, base_channels * 4),
            nn.MaxPool3d(2, 2),
        )

        self.stage3 = nn.Sequential(
            InterleavedFusionBlock(base_channels * 4, base_channels * 8),
            nn.MaxPool3d(2, 2),
        )

        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv3D(base_channels * 8, base_channels * 8),
            SqueezeExcitation3D(base_channels * 8),
        )

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_channels * 8, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"MIFNet created with {total_params / 1e6:.3f}M parameters")

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Feature extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        probabilities = self.softmax(logits)

        return logits, probabilities


# ============================================================================
# Lightweight EfficientNet-inspired 3D Model
# ============================================================================

class MBConv3D(nn.Module):
    """Mobile Inverted Bottleneck Conv block for 3D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4.0,
        stride: int = 1,
        use_se: bool = True,
    ):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm3d(hidden_channels),
                nn.GELU(),
            ])

        # Depthwise
        layers.extend([
            nn.Conv3d(
                hidden_channels, hidden_channels, 3,
                stride=stride, padding=1, groups=hidden_channels, bias=False
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
        ])

        self.conv = nn.Sequential(*layers)

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation3D(hidden_channels) if use_se else nn.Identity()

        # Projection
        self.project = nn.Sequential(
            nn.Conv3d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + identity
        return out


@register_model("efficientnet3d")
class EfficientNet3D(nn.Module):
    """
    Lightweight 3D EfficientNet-inspired model.

    Features:
    - ~0.5M parameters
    - Mobile Inverted Bottleneck blocks
    - Squeeze-and-Excitation attention
    - MPS compatible

    Input: (B, 1, D, H, W)
    Output: (logits, probabilities) each of shape (B, 2)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
        )

        # MBConv stages
        self.stage1 = MBConv3D(base_channels, base_channels * 2, expand_ratio=1, stride=1)
        self.stage2 = MBConv3D(base_channels * 2, base_channels * 2, expand_ratio=4, stride=2)
        self.stage3 = MBConv3D(base_channels * 2, base_channels * 4, expand_ratio=4, stride=2)
        self.stage4 = MBConv3D(base_channels * 4, base_channels * 8, expand_ratio=4, stride=2)

        # Head
        self.head = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 16, 1, bias=False),
            nn.BatchNorm3d(base_channels * 16),
            nn.GELU(),
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_channels * 16, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"EfficientNet3D created with {total_params / 1e6:.3f}M parameters")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        probabilities = self.softmax(logits)

        return logits, probabilities


