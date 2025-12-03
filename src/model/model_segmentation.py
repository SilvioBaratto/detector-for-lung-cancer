"""
Lightweight segmentation models for nodule detection.

This module provides:
- Model registry for easy model creation
- UltraLightUNet - Ultra-lightweight U-Net with multi-kernel convolutions
- SegmentationAugmentation - Data augmentation for segmentation

Optimized for Apple Silicon MPS with 16GB RAM.

References:
- UltraLightUNet: https://openreview.net/forum?id=BefqqrgdZ1
"""

import math
import random
from collections import namedtuple
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

_SEGMENTATION_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}

T = TypeVar("T", bound=nn.Module)


def register_segmentation_model(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a segmentation model class."""
    def decorator(cls: type[T]) -> type[T]:
        _SEGMENTATION_MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_segmentation_model(name: str, **kwargs) -> nn.Module:
    """Create a segmentation model by name from the registry."""
    if name not in _SEGMENTATION_MODEL_REGISTRY:
        available = ", ".join(_SEGMENTATION_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _SEGMENTATION_MODEL_REGISTRY[name](**kwargs)


def list_segmentation_models() -> list[str]:
    """List all registered segmentation model names."""
    return list(_SEGMENTATION_MODEL_REGISTRY.keys())


# ============================================================================
# Multi-Kernel Inverted Residual Block (Core of UltraLightUNet)
# ============================================================================

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for efficiency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiKernelInvertedResidual(nn.Module):
    """
    Multi-Kernel Inverted Residual (MKIR) block.

    Uses multiple kernel sizes to capture multi-scale features efficiently.
    Based on UltraLightUNet architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 2.0,
        kernels: tuple[int, ...] = (3, 5, 7),
    ):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)

        # Expansion layer
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # Multi-kernel depthwise convolutions
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    hidden_channels // len(kernels),
                    hidden_channels // len(kernels),
                    kernel_size=k,
                    padding=k // 2,
                    groups=hidden_channels // len(kernels),
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels // len(kernels)),
                nn.GELU(),
            )
            for k in kernels
        ])

        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Skip connection
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        # Expand
        out = self.expand(x)

        # Split and process with multiple kernels
        chunks = torch.chunk(out, len(self.branches), dim=1)
        out = torch.cat([
            branch(chunk) for branch, chunk in zip(self.branches, chunks)
        ], dim=1)

        # Project
        out = self.project(out)

        return out + identity


# ============================================================================
# UltraLightUNet Implementation
# ============================================================================

class DownBlock(nn.Module):
    """Encoder block with MKIR and downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mkir = MultiKernelInvertedResidual(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.mkir(x)
        pooled = self.pool(features)
        return pooled, features


class UpBlock(nn.Module):
    """Decoder block with upsampling and MKIR."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # Use bilinear upsampling (MPS compatible) instead of ConvTranspose2d
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.mkir = MultiKernelInvertedResidual(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_reduce(x)

        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.mkir(x)
        return x


@register_segmentation_model("ultralightunet")
class UltraLightUNet(nn.Module):
    """
    Ultra-lightweight U-Net for medical image segmentation.

    Features:
    - Only ~0.3M parameters
    - Multi-kernel inverted residual blocks
    - Depthwise separable convolutions
    - MPS compatible (no ConvTranspose2d)

    Input: (B, in_channels, H, W)
    Output: (B, n_classes, H, W) with sigmoid activation

    Reference: https://openreview.net/forum?id=BefqqrgdZ1
    """

    def __init__(
        self,
        in_channels: int = 7,
        n_classes: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes

        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        channels = base_channels
        encoder_channels = [base_channels]

        for i in range(depth):
            out_ch = min(channels * 2, 128)  # Cap at 128 for lightweight
            self.encoders.append(DownBlock(channels, out_ch))
            channels = out_ch
            encoder_channels.append(channels)

        # Bottleneck
        self.bottleneck = MultiKernelInvertedResidual(channels, channels)

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth):
            skip_ch = encoder_channels[-(i + 2)]
            out_ch = skip_ch
            self.decoders.append(UpBlock(channels, skip_ch, out_ch))
            channels = out_ch

        # Output
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, n_classes, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"UltraLightUNet created with {total_params / 1e6:.3f}M parameters")

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.input_conv(x)

        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        # Output
        return self.output_conv(x)


# ============================================================================
# Segmentation Augmentation
# ============================================================================

class SegmentationAugmentation(nn.Module):
    """Data augmentation for 2D segmentation."""

    def __init__(
        self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(
            transform_t[:, :2], input_g.size(), align_corners=False
        )

        augmented_input_g = F.grid_sample(
            input_g, affine_t, padding_mode='border', align_corners=False
        )
        augmented_label_g = F.grid_sample(
            label_g.to(torch.float32), affine_t,
            padding_mode='border', align_corners=False
        )

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise
            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])

            transform_t @= rotation_t

        return transform_t


# ============================================================================
# Segmentation Mask (kept from original)
# ============================================================================

MaskTuple = namedtuple(
    'MaskTuple',
    'raw_dense_mask, dense_mask, body_mask, air_mask, '
    'raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask'
)


class SegmentationMask(nn.Module):
    """Mask generation for segmentation training."""

    def __init__(self):
        super().__init__()
        self.conv_list = nn.ModuleList([
            self._make_circle_conv(radius) for radius in range(1, 8)
        ])

    def _make_circle_conv(self, radius):
        diameter = 1 + radius * 2
        a = torch.linspace(-1, 1, steps=diameter)**2
        b = (a[None] + a[:, None])**0.5
        circle_weights = (b <= 1.0).to(torch.float32)

        conv = nn.Conv2d(1, 1, kernel_size=diameter, padding=radius, bias=False)
        conv.weight.data.fill_(1)
        conv.weight.data *= circle_weights / circle_weights.sum()

        return conv

    def erode(self, input_mask, radius, threshold=1):
        conv = self.conv_list[radius - 1]
        input_float = input_mask.to(torch.float32)
        result = conv(input_float)
        return result >= threshold

    def deposit(self, input_mask, radius, threshold=0):
        conv = self.conv_list[radius - 1]
        input_float = input_mask.to(torch.float32)
        result = conv(input_float)
        return result > threshold

    def fill_cavity(self, input_mask):
        cumsum = input_mask.cumsum(-1)
        filled_mask = (cumsum > 0)
        filled_mask &= (cumsum < cumsum[..., -1:])
        cumsum = input_mask.cumsum(-2)
        filled_mask &= (cumsum > 0)
        filled_mask &= (cumsum < cumsum[..., -1:, :])
        return filled_mask

    def forward(self, input_g, raw_pos_g):
        gcc_g = input_g + 1

        with torch.no_grad():
            raw_dense_mask = gcc_g > 0.7
            dense_mask = self.deposit(raw_dense_mask, 2)
            dense_mask = self.erode(dense_mask, 6)
            dense_mask = self.deposit(dense_mask, 4)

            body_mask = self.fill_cavity(dense_mask)
            air_mask = self.deposit(body_mask & ~dense_mask, 5)
            air_mask = self.erode(air_mask, 6)

            lung_mask = self.deposit(air_mask, 5)

            raw_candidate_mask = gcc_g > 0.4
            raw_candidate_mask &= air_mask
            candidate_mask = self.erode(raw_candidate_mask, 1)
            candidate_mask = self.deposit(candidate_mask, 1)

            pos_mask = self.deposit((raw_pos_g > 0.5) & lung_mask, 2)

            neg_mask = self.deposit(candidate_mask, 1)
            neg_mask &= ~pos_mask
            neg_mask &= lung_mask

            label_g = (pos_mask).to(torch.float32)
            neg_g = neg_mask.to(torch.float32)
            pos_g = pos_mask.to(torch.float32)

        mask_dict = {
            'raw_dense_mask': raw_dense_mask,
            'dense_mask': dense_mask,
            'body_mask': body_mask,
            'air_mask': air_mask,
            'raw_candidate_mask': raw_candidate_mask,
            'candidate_mask': candidate_mask,
            'lung_mask': lung_mask,
            'neg_mask': neg_mask,
            'pos_mask': pos_mask,
        }

        return label_g, neg_g, pos_g, lung_mask, mask_dict
