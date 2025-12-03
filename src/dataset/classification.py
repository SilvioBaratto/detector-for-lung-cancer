"""
Classification datasets for nodule detection.

This module provides PyTorch datasets for:
- Binary classification (nodule vs non-nodule)
- Malignancy classification (benign vs malignant)

Design Principles:
- Single Responsibility: Dataset only handles data access
- Dependency Inversion: Accepts config and repositories via constructor
"""

from __future__ import annotations

import math
import random
from typing import Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import Config, get_config
from dataset.ct_loader import (
    CandidateInfo,
    CandidateRepository,
    CtCache,
    get_candidate_repository,
    get_ct_cache,
)
from dataset.registry import register_dataset, register_augmentation
from util.disk import getCache
from util.logconf import logging

log = logging.getLogger(__name__)


# ============================================================================
# Data Augmentation
# ============================================================================

@register_augmentation("3d")
class Augmentation3D:
    """3D data augmentation for CT chunks."""

    def __init__(
        self,
        flip: bool = True,
        offset: float = 0.1,
        scale: float = 0.0,
        rotate: bool = True,
        noise: float = 0.0,
    ):
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def __call__(self, chunk: np.ndarray) -> torch.Tensor:
        """Apply augmentation to a 3D chunk."""
        ct_t = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).float()
        transform = torch.eye(4)

        # Flip and offset per axis
        for i in range(3):
            if self.flip and random.random() > 0.5:
                transform[i, i] *= -1

            if self.offset > 0:
                transform[i, 3] = self.offset * (random.random() * 2 - 1)

            if self.scale > 0:
                transform[i, i] *= 1.0 + self.scale * (random.random() * 2 - 1)

        # Rotation around z-axis
        if self.rotate:
            angle = random.random() * math.pi * 2
            s, c = math.sin(angle), math.cos(angle)
            rotation = torch.tensor([
                [c, -s, 0, 0],
                [s,  c, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1],
            ], dtype=torch.float32)
            transform = transform @ rotation

        # Apply affine transform
        grid = F.affine_grid(
            transform[:3].unsqueeze(0),
            list(ct_t.size()),
            align_corners=False,
        )
        augmented = F.grid_sample(
            ct_t, grid,
            padding_mode='border',
            align_corners=False,
        )

        # Add noise
        if self.noise > 0:
            augmented += torch.randn_like(augmented) * self.noise

        return augmented[0]  # Remove batch dimension


# ============================================================================
# Disk Cache for Raw Candidates
# ============================================================================

_classification_cache = None


def get_classification_cache():
    """Get disk cache for classification data."""
    global _classification_cache
    if _classification_cache is None:
        _classification_cache = getCache('classification')
    return _classification_cache


# ============================================================================
# Classification Dataset
# ============================================================================

@register_dataset("classification")
class ClassificationDataset(Dataset):
    """
    Dataset for nodule classification (nodule vs non-nodule).

    Supports:
    - Class balancing via ratio parameter
    - Data augmentation
    - Disk caching for performance
    """

    CHUNK_SIZE = (32, 48, 48)  # Standard input size for classifier

    def __init__(
        self,
        config: Optional[Config] = None,
        candidate_repo: Optional[CandidateRepository] = None,
        ct_cache: Optional[CtCache] = None,
        val_stride: int = 0,
        is_validation: bool = False,
        series_uid: Optional[str] = None,
        sort_by: str = 'random',
        balance_ratio: int = 0,
        augmentation: Optional[Augmentation3D] = None,
        use_disk_cache: bool = True,
    ):
        """
        Initialize classification dataset.

        Args:
            config: Configuration object
            candidate_repo: Repository for candidate info
            ct_cache: Cache for CT scans
            val_stride: Stride for train/val split (0 = no split)
            is_validation: If True, use validation subset
            series_uid: Filter to single series
            sort_by: 'random', 'series_uid', or 'label_and_size'
            balance_ratio: Negative:positive ratio (0 = no balancing)
            augmentation: Augmentation transform
            use_disk_cache: Whether to use disk caching
        """
        self.config = config or get_config()
        self.candidate_repo = candidate_repo or get_candidate_repository(self.config)
        self.ct_cache = ct_cache or get_ct_cache(self.config)
        self.augmentation = augmentation
        self.use_disk_cache = use_disk_cache
        self.balance_ratio = balance_ratio

        # Get candidates
        all_candidates = self.candidate_repo.get_all()

        # Filter by series if specified
        if series_uid:
            series_list = [series_uid]
        else:
            series_list = sorted(set(c.series_uid for c in all_candidates))

        # Apply train/val split
        if is_validation:
            assert val_stride > 0
            series_list = series_list[::val_stride]
        elif val_stride > 0:
            del series_list[::val_stride]

        # Filter candidates by selected series
        series_set = set(series_list)
        self.candidates = [c for c in all_candidates if c.series_uid in series_set]

        # Sort candidates
        if sort_by == 'random':
            random.shuffle(self.candidates)
        elif sort_by == 'series_uid':
            self.candidates.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sort_by == 'label_and_size':
            pass  # Already sorted by size
        else:
            raise ValueError(f"Unknown sort_by: {sort_by}")

        # Split into positive/negative lists for balancing
        self.negatives = [c for c in self.candidates if not c.is_nodule]
        self.positives = [c for c in self.candidates if c.is_nodule]

        log.info(
            f"{self.__class__.__name__}: {len(self.candidates)} samples "
            f"({'validation' if is_validation else 'training'}), "
            f"{len(self.negatives)} neg, {len(self.positives)} pos, "
            f"ratio={'unbalanced' if not balance_ratio else f'{balance_ratio}:1'}"
        )

    def shuffle(self) -> None:
        """Shuffle samples (call at start of each epoch)."""
        if self.balance_ratio:
            random.shuffle(self.candidates)
            random.shuffle(self.negatives)
            random.shuffle(self.positives)

    def __len__(self) -> int:
        if self.balance_ratio:
            return 50000  # Fixed epoch size for balanced training
        return len(self.candidates)

    def __getitem__(self, idx: int):
        """Get a sample."""
        # Select candidate based on balancing strategy
        if self.balance_ratio:
            pos_idx = idx // (self.balance_ratio + 1)
            if idx % (self.balance_ratio + 1):  # Negative sample
                neg_idx = (idx - 1 - pos_idx) % len(self.negatives)
                candidate = self.negatives[neg_idx]
            else:  # Positive sample
                pos_idx = pos_idx % len(self.positives)
                candidate = self.positives[pos_idx]
        else:
            candidate = self.candidates[idx]

        return self._load_sample(candidate)

    def _load_sample(self, candidate: CandidateInfo):
        """Load and prepare a sample."""
        # Try disk cache first
        if self.use_disk_cache:
            cache_key = (candidate.series_uid, candidate.center_xyz, self.CHUNK_SIZE)
            cache = get_classification_cache()
            cached = cache.get(cache_key)

            if cached is None:
                chunk = self._extract_chunk(candidate)
                cache.set(cache_key, chunk)
            else:
                chunk = cached  # type: ignore
        else:
            chunk = self._extract_chunk(candidate)

        # Apply augmentation
        if self.augmentation:
            ct_tensor = self.augmentation(chunk)  # type: ignore
        else:
            ct_tensor = torch.from_numpy(chunk).unsqueeze(0).float()

        # Create label
        label = torch.tensor([
            not candidate.is_nodule,  # Index 0: non-nodule
            candidate.is_nodule,       # Index 1: nodule
        ], dtype=torch.long)

        label_idx = 1 if candidate.is_nodule else 0

        return ct_tensor, label, label_idx, candidate.series_uid, candidate.center_xyz

    def _extract_chunk(self, candidate: CandidateInfo) -> np.ndarray:
        """Extract CT chunk for a candidate."""
        ct = self.ct_cache.get_classification(candidate.series_uid)
        chunk, _ = ct.extract_chunk(candidate.center_xyz, self.CHUNK_SIZE)
        return chunk


# ============================================================================
# Malignancy Classification Dataset
# ============================================================================

@register_dataset("malignancy")
class MalignancyDataset(ClassificationDataset):
    """
    Dataset for malignancy classification (benign vs malignant).

    Uses only positive (nodule) samples and classifies malignancy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Further filter to only nodules
        self.benign = [c for c in self.positives if not c.is_malignant]
        self.malignant = [c for c in self.positives if c.is_malignant]

        log.info(
            f"{self.__class__.__name__}: {len(self.benign)} benign, "
            f"{len(self.malignant)} malignant"
        )

    def __len__(self) -> int:
        if self.balance_ratio:
            return 100000
        return len(self.benign) + len(self.malignant)

    def __getitem__(self, idx: int):
        """Get a sample with malignancy label."""
        if self.balance_ratio:
            # Balance: 50% malignant, 25% benign, 25% negative
            if idx % 2:  # Odd: malignant
                candidate = self.malignant[(idx // 2) % len(self.malignant)]
            elif idx % 4 == 0:  # Divisible by 4: benign
                candidate = self.benign[(idx // 4) % len(self.benign)]
            else:  # Other: negative (for context)
                candidate = self.negatives[(idx // 4) % len(self.negatives)]
        else:
            if idx < len(self.benign):
                candidate = self.benign[idx]
            else:
                candidate = self.malignant[idx - len(self.benign)]

        return self._load_malignancy_sample(candidate)

    def _load_malignancy_sample(self, candidate: CandidateInfo):
        """Load sample with malignancy label."""
        ct_tensor, _, _, series_uid, center_xyz = self._load_sample(candidate)

        # Malignancy label
        label = torch.tensor([
            not candidate.is_malignant,  # Index 0: benign
            candidate.is_malignant,       # Index 1: malignant
        ], dtype=torch.long)

        label_idx = 1 if candidate.is_malignant else 0

        return ct_tensor, label, label_idx, series_uid, center_xyz
