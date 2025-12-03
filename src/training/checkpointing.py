"""
Model checkpointing utilities.

This module provides checkpoint management following the Single Responsibility Principle.
Handles saving, loading, and tracking of model checkpoints.

Note: Checkpointer Protocol is defined in interfaces.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from util.logconf import logging

log = logging.getLogger(__name__)


# ============================================================================
# Checkpoint Data Structures
# ============================================================================

@dataclass
class Checkpoint:
    """Immutable checkpoint data structure."""
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    epoch: int
    total_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    optimizer_name: str = ""
    config: Optional[dict[str, Any]] = None


@dataclass
class CheckpointInfo:
    """Metadata about a saved checkpoint."""
    path: str
    epoch: int
    total_samples: int
    is_best: bool
    sha1_hash: str


# ============================================================================
# Checkpointer Implementation
# ============================================================================

class ModelCheckpointer:
    """
    Manages model checkpoint saving and loading.

    Follows Single Responsibility Principle - only handles checkpointing.

    Example:
        checkpointer = ModelCheckpointer(
            save_dir="models/classification",
            model_type="cls",
            comment="experiment1"
        )

        # Save checkpoint
        info = checkpointer.save(model, optimizer, epoch=5, total_samples=1000,
                                  metrics={"loss": 0.5}, is_best=True)

        # Load checkpoint
        checkpoint = checkpointer.load(info.path)
        model.load_state_dict(checkpoint.model_state)
    """

    def __init__(
        self,
        save_dir: str,
        model_type: str,
        time_str: str,
        comment: str = "",
    ):
        """
        Initialize checkpointer.

        Args:
            save_dir: Directory to save checkpoints
            model_type: Type prefix for checkpoint files (e.g., 'cls', 'seg', 'mal')
            time_str: Timestamp string for unique naming
            comment: Optional comment to include in filename
        """
        self.save_dir = Path(save_dir)
        self.model_type = model_type
        self.time_str = time_str
        self.comment = comment
        self._best_path: Optional[str] = None
        self._latest_path: Optional[str] = None

    def save(
        self,
        model: nn.Module,
        optimizer: Any,
        epoch: int,
        total_samples: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> CheckpointInfo:
        """
        Save a model checkpoint.

        Args:
            model: The model to save (handles DataParallel automatically)
            optimizer: The optimizer to save
            epoch: Current epoch number
            total_samples: Total training samples processed
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best checkpoint so far

        Returns:
            CheckpointInfo with path and metadata
        """
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

        # Build filename
        filename = f"{self.model_type}_{self.time_str}_{self.comment}.{total_samples}.state"
        file_path = self.save_dir / filename

        # Handle DataParallel wrapper
        model_to_save = model
        if isinstance(model, nn.DataParallel):
            model_to_save = model.module

        # Build state dictionary
        state = {
            "model_state": model_to_save.state_dict(),
            "model_name": type(model_to_save).__name__,
            "optimizer_state": optimizer.state_dict(),
            "optimizer_name": type(optimizer).__name__,
            "epoch": epoch,
            "total_samples": total_samples,
            "metrics": metrics,
        }

        # Save checkpoint
        torch.save(state, file_path)
        self._latest_path = str(file_path)

        # Compute hash for verification
        with open(file_path, "rb") as f:
            sha1_hash = hashlib.sha1(f.read()).hexdigest()

        log.info(f"Saved checkpoint to {file_path}")
        log.info(f"SHA1: {sha1_hash}")

        # Save best checkpoint
        if is_best:
            best_filename = f"{self.model_type}_{self.time_str}_{self.comment}.best.state"
            best_path = self.save_dir / best_filename
            shutil.copyfile(file_path, best_path)
            self._best_path = str(best_path)
            log.info(f"Saved best checkpoint to {best_path}")

        return CheckpointInfo(
            path=str(file_path),
            epoch=epoch,
            total_samples=total_samples,
            is_best=is_best,
            sha1_hash=sha1_hash,
        )

    def load(self, path: str, map_location: str = "cpu") -> Checkpoint:
        """
        Load a checkpoint from disk.

        Args:
            path: Path to checkpoint file
            map_location: Device to map tensors to

        Returns:
            Checkpoint dataclass with all saved state
        """
        log.info(f"Loading checkpoint from {path}")

        # Verify file and log hash
        with open(path, "rb") as f:
            sha1_hash = hashlib.sha1(f.read()).hexdigest()
        log.debug(f"Checkpoint SHA1: {sha1_hash}")

        state = torch.load(path, map_location=map_location)

        return Checkpoint(
            model_state=state["model_state"],
            optimizer_state=state.get("optimizer_state", {}),
            epoch=state.get("epoch", 0),
            total_samples=state.get("total_samples", state.get("totalTrainingSamples_count", 0)),
            metrics=state.get("metrics", {}),
            model_name=state.get("model_name", ""),
            optimizer_name=state.get("optimizer_name", ""),
            config=state.get("config"),
        )

    def get_best_path(self) -> Optional[str]:
        """Get path to best checkpoint if it exists."""
        if self._best_path and os.path.exists(self._best_path):
            return self._best_path

        # Try to find existing best checkpoint
        best_pattern = f"{self.model_type}_*.best.state"
        best_files = list(self.save_dir.glob(best_pattern))
        if best_files:
            # Return most recent
            best_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(best_files[0])
        return None

    def get_latest_path(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        if self._latest_path and os.path.exists(self._latest_path):
            return self._latest_path

        # Find most recent checkpoint
        pattern = f"{self.model_type}_*.state"
        files = [f for f in self.save_dir.glob(pattern) if ".best." not in f.name]
        if files:
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(files[0])
        return None


# ============================================================================
# Factory Function
# ============================================================================

def create_checkpointer(
    base_dir: str,
    tb_prefix: str,
    model_type: str,
    time_str: str,
    comment: str = "",
) -> ModelCheckpointer:
    """
    Factory function to create a checkpointer with standard paths.

    Args:
        base_dir: Base directory for all models
        tb_prefix: TensorBoard prefix (used as subdirectory)
        model_type: Model type ('cls', 'seg', 'mal')
        time_str: Timestamp string
        comment: Optional comment

    Returns:
        Configured ModelCheckpointer instance
    """
    save_dir = os.path.join(base_dir, tb_prefix)
    return ModelCheckpointer(
        save_dir=save_dir,
        model_type=model_type,
        time_str=time_str,
        comment=comment,
    )
