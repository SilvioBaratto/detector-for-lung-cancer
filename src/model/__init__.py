"""
Model package for lung cancer detection.

This package provides:
- Classification models (MIFNet, EfficientNet3D)
- Segmentation models (UltraLightUNet)
- Model registries for easy instantiation

Recommended models for Apple Silicon (MPS):
- Classification: 'mifnet' (0.7M params) or 'efficientnet3d' (0.5M params)
- Segmentation: 'ultralightunet' (0.3M params)
"""

from model.model_classification import (
    # Registry
    register_model,
    get_model,
    list_models,
    # Lightweight models
    MIFNet,
    EfficientNet3D,
    # Building blocks
    SqueezeExcitation3D,
    MultiScaleConv3D,
    DepthwiseSeparableConv3D,
    InterleavedFusionBlock,
    MBConv3D,
)

from model.model_segmentation import (
    # Registry
    register_segmentation_model,
    get_segmentation_model,
    list_segmentation_models,
    # Lightweight models
    UltraLightUNet,
    MultiKernelInvertedResidual,
    DepthwiseSeparableConv2d,
    # Augmentation and masks
    SegmentationAugmentation,
    SegmentationMask,
    MaskTuple,
)

__all__ = [
    # Classification Registry
    'register_model',
    'get_model',
    'list_models',
    # Classification Models
    'MIFNet',
    'EfficientNet3D',
    # Classification Blocks
    'SqueezeExcitation3D',
    'MultiScaleConv3D',
    'DepthwiseSeparableConv3D',
    'InterleavedFusionBlock',
    'MBConv3D',
    # Segmentation Registry
    'register_segmentation_model',
    'get_segmentation_model',
    'list_segmentation_models',
    # Segmentation Models
    'UltraLightUNet',
    # Segmentation Blocks
    'MultiKernelInvertedResidual',
    'DepthwiseSeparableConv2d',
    # Augmentation
    'SegmentationAugmentation',
    'SegmentationMask',
    'MaskTuple',
]
