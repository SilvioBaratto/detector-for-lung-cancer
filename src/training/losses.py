"""
Loss functions for training.

This module provides loss function strategies following the Strategy Pattern,
enabling easy swapping of loss functions without modifying training code.

Open/Closed Principle: Add new losses by creating new classes, not modifying existing code.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F


# ============================================================================
# Loss Strategy Base
# ============================================================================

class LossStrategy(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            reduction: 'none', 'mean', or 'sum'

        Returns:
            Loss tensor
        """
        pass

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Allow calling loss as a function."""
        return self.compute(predictions, targets, reduction)


# ============================================================================
# Classification Losses
# ============================================================================

class CrossEntropyLoss(LossStrategy):
    """Standard cross-entropy loss for classification."""

    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        """
        Args:
            weight: Class weights for imbalanced data
            label_smoothing: Label smoothing factor (0 to 1)
        """
        self.weight = weight
        self.label_smoothing = label_smoothing

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            predictions: Logits of shape (B, C) or (B, C, ...)
            targets: Target indices of shape (B,) or class probabilities (B, C)
        """
        return F.cross_entropy(
            predictions,
            targets,
            weight=self.weight,
            reduction=reduction,
            label_smoothing=self.label_smoothing,
        )


class BinaryCrossEntropyLoss(LossStrategy):
    """Binary cross-entropy loss."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            pos_weight: Weight for positive class
        """
        self.pos_weight = pos_weight

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            predictions,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction=reduction,
        )


class FocalLoss(LossStrategy):
    """
    Focal loss for handling class imbalance.

    Focuses learning on hard examples by down-weighting easy ones.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Balance factor for positive/negative classes
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        self.alpha = alpha
        self.gamma = gamma

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if reduction == "mean":
            return focal_loss.mean()
        elif reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# Segmentation Losses
# ============================================================================

class DiceLoss(LossStrategy):
    """
    Dice loss for segmentation.

    Measures overlap between prediction and ground truth.
    """

    def __init__(self, epsilon: float = 1.0, smooth: float = 1.0):
        """
        Args:
            epsilon: Small constant to avoid division by zero
            smooth: Smoothing factor
        """
        self.epsilon = epsilon
        self.smooth = smooth

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            predictions: Predicted probabilities (B, 1, H, W) or (B, 1, D, H, W)
            targets: Ground truth masks (same shape as predictions)
        """
        # Flatten spatial dimensions
        pred_flat = predictions.view(predictions.size(0), -1)
        target_flat = targets.view(targets.size(0), -1).float()

        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        loss = 1 - dice

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss


class WeightedDiceLoss(LossStrategy):
    """
    Dice loss with additional weighting for false negatives.

    Useful when missing positives is more costly than false positives.
    """

    def __init__(self, epsilon: float = 1.0, fn_weight: float = 8.0):
        """
        Args:
            epsilon: Small constant to avoid division by zero
            fn_weight: Weight for false negative term
        """
        self.epsilon = epsilon
        self.fn_weight = fn_weight
        self.dice_loss = DiceLoss(epsilon=epsilon)

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        # Standard Dice loss
        dice_loss = self.dice_loss.compute(predictions, targets, reduction)

        # False negative weighted Dice
        fn_dice = self.dice_loss.compute(predictions * targets, targets, reduction)

        return dice_loss + fn_dice * self.fn_weight


class CombinedLoss(LossStrategy):
    """Combine multiple loss functions with weights."""

    def __init__(self, losses: list[tuple[LossStrategy, float]]):
        """
        Args:
            losses: List of (loss_function, weight) tuples
        """
        self.losses = losses

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=predictions.device)
        for loss_fn, weight in self.losses:
            total_loss = total_loss + weight * loss_fn.compute(predictions, targets, reduction)
        return total_loss


# ============================================================================
# Loss Registry
# ============================================================================

_LOSS_REGISTRY: dict[str, type[LossStrategy]] = {
    "cross_entropy": CrossEntropyLoss,
    "bce": BinaryCrossEntropyLoss,
    "focal": FocalLoss,
    "dice": DiceLoss,
    "weighted_dice": WeightedDiceLoss,
}


def get_loss(name: str, **kwargs) -> LossStrategy:
    """Create a loss function by name."""
    if name not in _LOSS_REGISTRY:
        available = ", ".join(_LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")
    return _LOSS_REGISTRY[name](**kwargs)


def register_loss(name: str):
    """Decorator to register a loss function."""
    def decorator(cls: type[LossStrategy]) -> type[LossStrategy]:
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator
