"""
Metrics calculation for training and evaluation.

This module provides:
- Metric data structures
- Metric calculators for classification and segmentation
- Metric aggregation utilities

Single Responsibility: Only handles metrics calculation, not logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ============================================================================
# Metric Data Structures
# ============================================================================

@dataclass
class ClassificationMetrics:
    """Metrics for classification tasks."""
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: float = 0.0

    # Per-class metrics
    pos_correct: float = 0.0
    neg_correct: float = 0.0
    pos_count: int = 0
    neg_count: int = 0

    # Confusion matrix components
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc": self.auc,
            "pos_correct": self.pos_correct,
            "neg_correct": self.neg_correct,
        }


@dataclass
class SegmentationMetrics:
    """Metrics for segmentation tasks."""
    loss: float = 0.0
    dice: float = 0.0
    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Pixel-level counts
    true_positives: float = 0.0
    false_positives: float = 0.0
    false_negatives: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "loss": self.loss,
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


# ============================================================================
# Metric Calculators
# ============================================================================

class ClassificationMetricsCalculator:
    """Calculator for classification metrics."""

    def __init__(self, num_classes: int = 2, threshold: float = 0.5):
        """
        Args:
            num_classes: Number of classes
            threshold: Classification threshold for binary classification
        """
        self.num_classes = num_classes
        self.threshold = threshold

    def compute(
        self,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        losses: torch.Tensor,
    ) -> ClassificationMetrics:
        """
        Compute classification metrics.

        Args:
            predictions: Predicted class indices (B,)
            probabilities: Class probabilities (B, C)
            labels: True labels - either indices (B,) or one-hot (B, C)
            losses: Per-sample losses (B,)

        Returns:
            ClassificationMetrics object
        """
        # Convert to numpy for easier computation
        preds = predictions.detach().cpu().numpy()
        probs = probabilities.detach().cpu().numpy()

        # Handle both index and one-hot labels
        if labels.dim() > 1:
            true_labels = labels[:, 1].detach().cpu().numpy()  # Assume binary
        else:
            true_labels = labels.detach().cpu().numpy()

        loss_values = losses.detach().cpu().numpy()

        # Basic counts
        pos_mask = true_labels == 1
        neg_mask = true_labels == 0

        pos_count = int(pos_mask.sum())
        neg_count = int(neg_mask.sum())

        # Predictions
        pred_pos = preds == 1
        pred_neg = preds == 0

        # Confusion matrix
        tp = int((pos_mask & pred_pos).sum())
        fp = int((neg_mask & pred_pos).sum())
        tn = int((neg_mask & pred_neg).sum())
        fn = int((pos_mask & pred_neg).sum())

        # Metrics
        accuracy = (tp + tn) / max(len(preds), 1) * 100

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        pos_correct = tp / max(pos_count, 1) * 100
        neg_correct = tn / max(neg_count, 1) * 100

        # AUC calculation
        auc = self._compute_auc(probs[:, 1] if probs.shape[1] > 1 else probs.flatten(), true_labels)

        return ClassificationMetrics(
            loss=float(loss_values.mean()),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc=auc,
            pos_correct=pos_correct,
            neg_correct=neg_correct,
            pos_count=pos_count,
            neg_count=neg_count,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )

    def _compute_auc(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute ROC AUC using trapezoidal rule."""
        if len(np.unique(labels)) < 2:
            return 0.5  # Undefined for single class

        # Create thresholds
        thresholds = np.linspace(1, 0, num=89)

        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()

        if pos_count == 0 or neg_count == 0:
            return 0.5

        # Compute TPR and FPR at each threshold
        tpr = np.array([(probs[pos_mask] >= t).sum() / pos_count for t in thresholds])
        fpr = np.array([(probs[neg_mask] >= t).sum() / neg_count for t in thresholds])

        # Trapezoidal integration
        fp_diff = np.diff(fpr)
        tp_avg = (tpr[1:] + tpr[:-1]) / 2
        auc = float((fp_diff * tp_avg).sum())

        return auc

    def compute_from_tensor(
        self,
        metrics_tensor: torch.Tensor,
        label_idx: int = 0,
        pred_idx: int = 1,
        prob_idx: int = 2,
        loss_idx: int = 3,
    ) -> ClassificationMetrics:
        """
        Compute metrics from a pre-collected metrics tensor.

        Args:
            metrics_tensor: Tensor of shape (num_metrics, num_samples)
            label_idx, pred_idx, prob_idx, loss_idx: Indices in the tensor
        """
        labels = metrics_tensor[label_idx]
        pred_classes = metrics_tensor[pred_idx].long()
        probs = metrics_tensor[prob_idx]
        losses = metrics_tensor[loss_idx]

        return self.compute(
            predictions=pred_classes,
            probabilities=torch.stack([1 - probs, probs], dim=1),
            labels=labels.long(),
            losses=losses,
        )


class SegmentationMetricsCalculator:
    """Calculator for segmentation metrics."""

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Segmentation threshold
        """
        self.threshold = threshold

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        losses: torch.Tensor,
    ) -> SegmentationMetrics:
        """
        Compute segmentation metrics.

        Args:
            predictions: Predicted masks (B, 1, H, W) or (B, 1, D, H, W)
            targets: Ground truth masks (same shape)
            losses: Per-sample losses (B,)

        Returns:
            SegmentationMetrics object
        """
        # Threshold predictions
        pred_binary = (predictions > self.threshold).float()

        # Flatten for computation
        pred_flat = pred_binary.view(pred_binary.size(0), -1)
        target_flat = targets.view(targets.size(0), -1).float()

        # Per-sample metrics
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        union = pred_sum + target_sum - intersection

        # Dice
        dice = (2 * intersection + 1) / (pred_sum + target_sum + 1)

        # IoU
        iou = (intersection + 1) / (union + 1)

        # Confusion components (summed over batch)
        tp = intersection.sum().item()
        fp = (pred_flat.sum() - intersection.sum()).item()
        fn = (target_flat.sum() - intersection.sum()).item()

        # Precision, Recall, F1
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return SegmentationMetrics(
            loss=float(losses.mean().item()),
            dice=float(dice.mean().item()),
            iou=float(iou.mean().item()),
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    def compute_from_counts(
        self,
        tp: float,
        fp: float,
        fn: float,
        total_loss: float,
        num_samples: int,
    ) -> SegmentationMetrics:
        """Compute metrics from aggregated counts."""
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Dice from counts
        dice = 2 * tp / max(2 * tp + fp + fn, 1e-8)

        # IoU from counts
        iou = tp / max(tp + fp + fn, 1e-8)

        return SegmentationMetrics(
            loss=total_loss / max(num_samples, 1),
            dice=dice,
            iou=iou,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )


# ============================================================================
# Metric Aggregators
# ============================================================================

class MetricAggregator:
    """Aggregates metrics over batches."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset aggregated values."""
        self._values: dict[str, list[float]] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float], count: int = 1) -> None:
        """Update with new values."""
        for key, value in metrics.items():
            if key not in self._values:
                self._values[key] = []
                self._counts[key] = 0
            self._values[key].append(value * count)
            self._counts[key] += count

    def compute(self) -> dict[str, float]:
        """Compute aggregated metrics."""
        result = {}
        for key in self._values:
            total = sum(self._values[key])
            count = self._counts[key]
            result[key] = total / max(count, 1)
        return result
