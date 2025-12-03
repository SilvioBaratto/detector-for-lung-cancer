"""
Training logging utilities.

This module provides logging implementations following the Strategy Pattern,
enabling easy swapping of logging backends without modifying training code.

Single Responsibility: Only handles metrics logging, not training logic.

Note: MetricsLogger and ExtendedMetricsLogger Protocols are defined in interfaces.py
"""

from __future__ import annotations

import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from interfaces import MetricsLogger


# ============================================================================
# Logger Implementations
# ============================================================================

class TensorBoardLogger:
    """
    TensorBoard logging implementation.

    Implements the MetricsLogger protocol for TensorBoard-based logging.

    Example:
        logger = TensorBoardLogger(log_dir="runs/experiment1")
        logger.log_scalar("loss", 0.5, step=100)
        logger.log_scalars("metrics", {"precision": 0.9, "recall": 0.8}, step=100)
        logger.close()
    """

    def __init__(self, log_dir: str, comment: str = ""):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            comment: Optional comment to append to log directory
        """
        from torch.utils.tensorboard import SummaryWriter

        full_log_dir = f"{log_dir}-{comment}" if comment else log_dir
        self._writer = SummaryWriter(log_dir=full_log_dir)
        self.log_dir = full_log_dir

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log multiple scalar values under a main tag."""
        self._writer.add_scalars(main_tag, values, step)

    def log_image(
        self,
        tag: str,
        image: Any,
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image (numpy array or tensor)."""
        self._writer.add_image(tag, image, step, dataformats=dataformats)

    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: int,
        bins: Any = None,
    ) -> None:
        """Log a histogram of values."""
        if bins is not None:
            self._writer.add_histogram(tag, values, step, bins=bins)
        else:
            self._writer.add_histogram(tag, values, step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure."""
        self._writer.add_figure(tag, figure, step)

    def log_pr_curve(
        self,
        tag: str,
        labels: Any,
        predictions: Any,
        step: int,
    ) -> None:
        """Log precision-recall curve."""
        self._writer.add_pr_curve(tag, labels, predictions, step)

    def flush(self) -> None:
        """Flush pending logs to disk."""
        self._writer.flush()

    def close(self) -> None:
        """Close the logger and release resources."""
        self._writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures logger is closed."""
        self.close()


class CompositeLogger:
    """
    Composite logger that writes to multiple backends.

    Useful for logging to both TensorBoard and console simultaneously.
    """

    def __init__(self, loggers: list[MetricsLogger]):
        """
        Initialize with list of loggers.

        Args:
            loggers: List of logger instances to write to
        """
        self._loggers = loggers

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar to all loggers."""
        for logger in self._loggers:
            logger.log_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log scalars to all loggers."""
        for logger in self._loggers:
            logger.log_scalars(main_tag, values, step)

    def flush(self) -> None:
        """Flush all loggers."""
        for logger in self._loggers:
            logger.flush()

    def close(self) -> None:
        """Close all loggers."""
        for logger in self._loggers:
            logger.close()


class NullLogger:
    """No-op logger for testing or when logging is disabled."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        pass

    def log_image(self, tag: str, image: Any, step: int, dataformats: str = "CHW") -> None:
        pass

    def log_histogram(self, tag: str, values: Any, step: int, bins: Any = None) -> None:
        pass

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# ============================================================================
# Training Logger (Combines train/val writers)
# ============================================================================

class TrainingLogger:
    """
    High-level training logger that manages train and validation writers.

    Provides a clean interface for logging during training loops.

    Example:
        logger = TrainingLogger(
            base_log_dir="runs/classification",
            time_str="2024-01-01_12.00.00",
            comment="experiment1"
        )

        # Log training metrics
        logger.log_training_metrics(epoch=1, metrics={"loss": 0.5}, step=1000)

        # Log validation metrics
        logger.log_validation_metrics(epoch=1, metrics={"accuracy": 0.9}, step=1000)

        logger.close()
    """

    def __init__(
        self,
        base_log_dir: str,
        time_str: str,
        comment: str = "",
        mode_suffix: str = "cls",
    ):
        """
        Initialize training logger with train/val writers.

        Args:
            base_log_dir: Base directory for logs
            time_str: Timestamp string for unique naming
            comment: Optional comment for log directory
            mode_suffix: Suffix for log directories (e.g., 'cls', 'seg')
        """
        log_dir = os.path.join(base_log_dir, time_str)

        self.train_logger = TensorBoardLogger(
            log_dir=log_dir,
            comment=f"trn_{mode_suffix}-{comment}" if comment else f"trn_{mode_suffix}",
        )
        self.val_logger = TensorBoardLogger(
            log_dir=log_dir,
            comment=f"val_{mode_suffix}-{comment}" if comment else f"val_{mode_suffix}",
        )
        self._initialized = True

    def log_training_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar to training writer."""
        self.train_logger.log_scalar(tag, value, step)

    def log_validation_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar to validation writer."""
        self.val_logger.log_scalar(tag, value, step)

    def log_training_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log multiple metrics to training writer."""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.train_logger.log_scalar(tag, value, step)

    def log_validation_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log multiple metrics to validation writer."""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.val_logger.log_scalar(tag, value, step)

    def log_training_image(
        self,
        tag: str,
        image: Any,
        step: int,
        dataformats: str = "HWC",
    ) -> None:
        """Log an image to training writer."""
        self.train_logger.log_image(tag, image, step, dataformats)

    def log_validation_image(
        self,
        tag: str,
        image: Any,
        step: int,
        dataformats: str = "HWC",
    ) -> None:
        """Log an image to validation writer."""
        self.val_logger.log_image(tag, image, step, dataformats)

    def log_training_histogram(
        self,
        tag: str,
        values: Any,
        step: int,
        bins: Any = None,
    ) -> None:
        """Log a histogram to training writer."""
        self.train_logger.log_histogram(tag, values, step, bins)

    def log_validation_histogram(
        self,
        tag: str,
        values: Any,
        step: int,
        bins: Any = None,
    ) -> None:
        """Log a histogram to validation writer."""
        self.val_logger.log_histogram(tag, values, step, bins)

    def log_training_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure to training writer."""
        self.train_logger.log_figure(tag, figure, step)

    def log_validation_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure to validation writer."""
        self.val_logger.log_figure(tag, figure, step)

    def flush(self) -> None:
        """Flush both loggers."""
        self.train_logger.flush()
        self.val_logger.flush()

    def close(self) -> None:
        """Close both loggers."""
        self.train_logger.close()
        self.val_logger.close()

    def __enter__(self) -> "TrainingLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


# ============================================================================
# Factory Functions
# ============================================================================

def create_training_logger(
    base_dir: str,
    tb_prefix: str,
    time_str: str,
    comment: str = "",
    mode: str = "cls",
) -> TrainingLogger:
    """
    Factory function to create a training logger with standard paths.

    Args:
        base_dir: Base directory for logs (e.g., "runs")
        tb_prefix: TensorBoard prefix subdirectory
        time_str: Timestamp string
        comment: Optional comment
        mode: Mode suffix ('cls', 'seg')

    Returns:
        Configured TrainingLogger instance
    """
    base_log_dir = os.path.join(base_dir, tb_prefix)
    return TrainingLogger(
        base_log_dir=base_log_dir,
        time_str=time_str,
        comment=comment,
        mode_suffix=mode,
    )


def create_null_logger() -> NullLogger:
    """Create a no-op logger for testing."""
    return NullLogger()
