"""
Training package for lung cancer detection.

This package provides:
- Loss functions (losses)
- Metrics calculation (metrics)
- Logging utilities (logging)
- Checkpointing utilities (checkpointing)
"""

from training.losses import (
    LossStrategy,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    get_loss,
    list_losses,
)

from training.metrics import (
    MetricCalculator,
    ClassificationMetrics,
    SegmentationMetrics,
    compute_classification_metrics,
    compute_segmentation_metrics,
)

from training.logging import (
    TensorBoardLogger,
    TrainingLogger,
    CompositeLogger,
    NullLogger,
    create_training_logger,
    create_null_logger,
)

from training.checkpointing import (
    Checkpoint,
    CheckpointInfo,
    ModelCheckpointer,
    create_checkpointer,
)

__all__ = [
    # Losses
    'LossStrategy',
    'CrossEntropyLoss',
    'DiceLoss',
    'FocalLoss',
    'CombinedLoss',
    'get_loss',
    'list_losses',
    # Metrics
    'MetricCalculator',
    'ClassificationMetrics',
    'SegmentationMetrics',
    'compute_classification_metrics',
    'compute_segmentation_metrics',
    # Logging
    'TensorBoardLogger',
    'TrainingLogger',
    'CompositeLogger',
    'NullLogger',
    'create_training_logger',
    'create_null_logger',
    # Checkpointing
    'Checkpoint',
    'CheckpointInfo',
    'ModelCheckpointer',
    'create_checkpointer',
]
