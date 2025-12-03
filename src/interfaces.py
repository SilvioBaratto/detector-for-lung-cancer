"""
Protocol interfaces for dependency inversion.

This module defines abstract interfaces (Protocols) that enable:
- Loose coupling between components
- Easy testing with mock implementations
- Clear contracts for implementations

Following Interface Segregation Principle: interfaces are small and focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterator, Any, Optional

import torch
from torch import nn


# ============================================================================
# Data Access Protocols
# ============================================================================

@dataclass
class CandidateData:
    """Generic candidate data structure."""
    is_nodule: bool
    has_annotation: bool
    is_malignant: bool
    diameter_mm: float
    series_uid: str
    center_xyz: tuple[float, float, float]


class CandidateRepository(Protocol):
    """Protocol for candidate data access."""

    def get_all(self) -> list[CandidateData]:
        """Get all candidates."""
        ...

    def get_by_series(self) -> dict[str, list[CandidateData]]:
        """Get candidates grouped by series UID."""
        ...

    def get_nodules(self) -> list[CandidateData]:
        """Get only nodule candidates."""
        ...

    def get_non_nodules(self) -> list[CandidateData]:
        """Get only non-nodule candidates."""
        ...


class CtLoader(Protocol):
    """Protocol for CT scan loading."""

    def get_classification(self, series_uid: str) -> Any:
        """Load CT for classification."""
        ...

    def get_segmentation(self, series_uid: str) -> Any:
        """Load CT for segmentation."""
        ...

    def clear(self) -> None:
        """Clear cached CT scans."""
        ...


# ============================================================================
# Model Output Data Structures
# ============================================================================

@dataclass
class ClassificationOutput:
    """Standardized output for classification models."""
    logits: torch.Tensor      # Raw model outputs (B, num_classes)
    probabilities: torch.Tensor  # Softmax probabilities (B, num_classes)

    @property
    def predictions(self) -> torch.Tensor:
        """Get predicted class indices."""
        return self.probabilities.argmax(dim=1)

    @property
    def confidence(self) -> torch.Tensor:
        """Get confidence scores for predictions."""
        return self.probabilities.max(dim=1).values


@dataclass
class SegmentationOutput:
    """Standardized output for segmentation models."""
    mask: torch.Tensor  # Predicted mask (B, C, H, W) or (B, C, D, H, W)

    def threshold(self, value: float = 0.5) -> torch.Tensor:
        """Get binary mask at threshold."""
        return (self.mask > value).float()


# ============================================================================
# Model Protocols
# ============================================================================

class NeuralNetworkModel(Protocol):
    """Base protocol for all neural network models."""

    def train(self, mode: bool = True) -> nn.Module:
        """Set training mode."""
        ...

    def eval(self) -> nn.Module:
        """Set evaluation mode."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Get model parameters."""
        ...

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Get named model parameters."""
        ...

    def named_children(self) -> Iterator[tuple[str, nn.Module]]:
        """Get named child modules."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        """Load state dictionary."""
        ...

    def to(self, device: Any) -> nn.Module:
        """Move model to device."""
        ...


class ClassificationModel(NeuralNetworkModel, Protocol):
    """
    Protocol for classification models.

    Returns tuple of (logits, probabilities) for backward compatibility.
    New code should use ClassificationOutput wrapper.
    """

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D

        Returns:
            Tuple of (logits, probabilities), each of shape (B, num_classes)
        """
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (same as __call__)."""
        ...


class SegmentationModel(NeuralNetworkModel, Protocol):
    """
    Protocol for segmentation models.

    Returns probability mask for each pixel/voxel.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D

        Returns:
            Segmentation mask of shape (B, num_classes, ...) with probabilities
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (same as __call__)."""
        ...


# ============================================================================
# Training Protocols
# ============================================================================

class LossFunction(Protocol):
    """Protocol for loss functions."""

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss."""
        ...


class Optimizer(Protocol):
    """Protocol for optimizers."""

    def zero_grad(self) -> None:
        """Zero gradients."""
        ...

    def step(self) -> None:
        """Perform optimization step."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Get optimizer state."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state."""
        ...


class Augmentation(Protocol):
    """Protocol for data augmentation."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation."""
        ...


# ============================================================================
# Logging Protocols
# ============================================================================

class MetricsLogger(Protocol):
    """Protocol for basic metrics logging."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        ...

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        ...

    def flush(self) -> None:
        """Flush pending logs."""
        ...

    def close(self) -> None:
        """Close the logger."""
        ...


class ExtendedMetricsLogger(MetricsLogger, Protocol):
    """Extended logger protocol with image and histogram support."""

    def log_image(self, tag: str, image: Any, step: int, dataformats: str = "CHW") -> None:
        """Log an image."""
        ...

    def log_histogram(self, tag: str, values: Any, step: int, bins: Any = None) -> None:
        """Log a histogram."""
        ...

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure."""
        ...


# ============================================================================
# Checkpointing Protocols
# ============================================================================

@dataclass
class Checkpoint:
    """Checkpoint data structure."""
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    epoch: int
    metrics: dict[str, float]
    config: Optional[dict[str, Any]] = None


class Checkpointer(Protocol):
    """Protocol for model checkpointing."""

    def save(
        self,
        model: nn.Module,
        optimizer: Any,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> str:
        """Save a checkpoint. Returns the file path."""
        ...

    def load(self, path: str) -> Checkpoint:
        """Load a checkpoint."""
        ...

    def get_best_path(self) -> Optional[str]:
        """Get path to best checkpoint if exists."""
        ...

    def get_latest_path(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        ...


# ============================================================================
# Dataset Protocols
# ============================================================================

class TrainableDataset(Protocol):
    """Protocol for datasets that support training operations."""

    def __len__(self) -> int:
        """Get dataset length."""
        ...

    def __getitem__(self, idx: int) -> tuple:
        """Get item by index."""
        ...

    def shuffle(self) -> None:
        """Shuffle the dataset."""
        ...
