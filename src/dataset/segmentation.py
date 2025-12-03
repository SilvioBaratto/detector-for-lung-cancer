"""
Segmentation datasets for nodule detection.

This module provides PyTorch datasets for 2D slice-based segmentation
of lung nodules in CT scans.

Design Principles:
- Single Responsibility: Dataset only handles data access
- Dependency Inversion: Accepts config and repositories via constructor
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config, get_config
from dataset.ct_loader import (
    CandidateInfo,
    CandidateRepository,
    CtCache,
    get_candidate_repository,
    get_ct_cache,
)
from dataset.registry import register_dataset
from util.disk import getCache
from util.logconf import logging

log = logging.getLogger(__name__)


# ============================================================================
# Disk Cache
# ============================================================================

_segmentation_cache = None


def get_segmentation_cache():
    """Get disk cache for segmentation data."""
    global _segmentation_cache
    if _segmentation_cache is None:
        _segmentation_cache = getCache('segmentation')
    return _segmentation_cache


# ============================================================================
# Segmentation Dataset (Full Slices)
# ============================================================================

@register_dataset("segmentation")
class SegmentationDataset(Dataset):
    """
    Dataset for 2D segmentation of full CT slices.

    Returns slices with context (multiple channels) and corresponding masks.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        candidate_repo: Optional[CandidateRepository] = None,
        ct_cache: Optional[CtCache] = None,
        val_stride: int = 0,
        is_validation: Optional[bool] = None,
        series_uid: Optional[str] = None,
        context_slices: int = 3,
        full_ct: bool = False,
    ):
        """
        Initialize segmentation dataset.

        Args:
            config: Configuration object
            candidate_repo: Repository for candidate info
            ct_cache: Cache for CT scans
            val_stride: Stride for train/val split
            is_validation: True for validation, False for training, None for all
            series_uid: Filter to single series
            context_slices: Number of context slices on each side
            full_ct: If True, include all slices; if False, only positive slices
        """
        self.config = config or get_config()
        self.candidate_repo = candidate_repo or get_candidate_repository(self.config)
        self.ct_cache = ct_cache or get_ct_cache(self.config)
        self.context_slices = context_slices
        self.full_ct = full_ct

        # Get series list
        candidates_by_series = self.candidate_repo.get_by_series()
        if series_uid:
            series_list = [series_uid]
        else:
            series_list = sorted(candidates_by_series.keys())

        # Apply train/val split
        if is_validation is True:
            assert val_stride > 0
            series_list = series_list[::val_stride]
        elif is_validation is False and val_stride > 0:
            del series_list[::val_stride]

        self.series_list = series_list

        # Build sample list
        self.samples: list[tuple[str, int]] = []
        for uid in series_list:
            slice_count, positive_indices = self._get_series_info(uid)

            if full_ct:
                self.samples.extend((uid, i) for i in range(slice_count))
            else:
                self.samples.extend((uid, i) for i in positive_indices)

        # Get candidate info for this subset
        series_set = set(series_list)
        all_candidates = self.candidate_repo.get_all()
        self.candidates = [c for c in all_candidates if c.series_uid in series_set]
        self.positives = [c for c in self.candidates if c.is_nodule]

        log.info(
            f"{self.__class__.__name__}: {len(series_list)} series "
            f"({'validation' if is_validation else 'training' if is_validation is False else 'all'}), "
            f"{len(self.samples)} slices, {len(self.positives)} nodules"
        )

    def _get_series_info(self, series_uid: str) -> tuple[int, list[int]]:
        """Get slice count and positive indices for a series."""
        cache = get_segmentation_cache()
        cache_key = f"series_info_{series_uid}"
        cached = cache.get(cache_key)

        if cached is not None:
            return cached  # type: ignore

        ct = self.ct_cache.get_segmentation(series_uid)
        result = (ct.num_slices, ct.positive_slice_indices)
        cache.set(cache_key, result)
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Get a full slice with context."""
        series_uid, slice_idx = self.samples[idx % len(self.samples)]
        return self._get_slice(series_uid, slice_idx)

    def _get_slice(self, series_uid: str, slice_idx: int):
        """Load a slice with context channels."""
        ct = self.ct_cache.get_segmentation(series_uid)

        # Create multi-channel input with context slices
        num_channels = self.context_slices * 2 + 1
        ct_tensor = torch.zeros((num_channels, 512, 512))

        start = slice_idx - self.context_slices
        end = slice_idx + self.context_slices + 1

        for i, ctx_idx in enumerate(range(start, end)):
            # Clamp to valid range (duplicate boundary slices)
            ctx_idx = max(0, min(ctx_idx, ct.num_slices - 1))
            ct_tensor[i] = torch.from_numpy(
                ct.hu_array[ctx_idx].astype(np.float32)
            )

        # Clamp HU values
        ct_tensor.clamp_(-1000, 1000)

        # Get mask for center slice
        mask_tensor = torch.from_numpy(
            ct.positive_mask[slice_idx]
        ).unsqueeze(0)

        return ct_tensor, mask_tensor, series_uid, slice_idx


# ============================================================================
# Training Segmentation Dataset (Random Crops)
# ============================================================================

@register_dataset("training_segmentation")
class TrainingSegmentationDataset(SegmentationDataset):
    """
    Training dataset with random crops and augmentation.

    Uses fixed epoch size and class balancing.
    """

    CROP_CONTEXT_SIZE = (7, 96, 96)  # Context for cropping
    OUTPUT_SIZE = (7, 64, 64)        # Final crop size

    def __init__(self, *args, balance_ratio: int = 2, **kwargs):
        kwargs['full_ct'] = False  # Only positive slices
        super().__init__(*args, **kwargs)
        self.balance_ratio = balance_ratio

    def shuffle(self) -> None:
        """Shuffle samples for new epoch."""
        random.shuffle(self.candidates)
        random.shuffle(self.positives)

    def __len__(self) -> int:
        return 300000  # Fixed epoch size

    def __getitem__(self, idx: int):
        """Get a training crop."""
        candidate = self.positives[idx % len(self.positives)]
        return self._get_training_crop(candidate)

    def _get_training_crop(self, candidate: CandidateInfo):
        """Get a randomly cropped training sample."""
        # Get larger context chunk
        ct = self.ct_cache.get_segmentation(candidate.series_uid)
        chunk, mask, center_irc = ct.extract_chunk_with_mask(
            candidate.center_xyz,
            self.CROP_CONTEXT_SIZE,
        )

        # Take center mask slice
        mask = mask[3:4]  # Keep dimension for channel

        # Random crop within context
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)

        ct_tensor = torch.from_numpy(
            chunk[:, row_offset:row_offset+64, col_offset:col_offset+64]
        ).float()

        mask_tensor = torch.from_numpy(
            mask[:, row_offset:row_offset+64, col_offset:col_offset+64]
        ).long()

        return ct_tensor, mask_tensor, candidate.series_uid, int(center_irc.index)


# ============================================================================
# Cache Preparation Dataset
# ============================================================================

@register_dataset("cache_preparation")
class CachePreparationDataset(Dataset):
    """
    Dataset for pre-warming the disk cache.

    Iterates through all candidates to cache their data.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        candidate_repo: Optional[CandidateRepository] = None,
    ):
        self.config = config or get_config()
        self.candidate_repo = candidate_repo or get_candidate_repository(self.config)
        self.candidates = self.candidate_repo.get_all()
        self.candidates.sort(key=lambda x: x.series_uid)
        self._seen_series: set[str] = set()

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int):
        """Cache data for a candidate."""
        candidate = self.candidates[idx]

        # Cache the raw chunk
        cache = get_segmentation_cache()
        cache_key = (candidate.series_uid, candidate.center_xyz, (7, 96, 96))

        if cache.get(cache_key) is None:
            ct_cache = get_ct_cache(self.config)
            ct = ct_cache.get_segmentation(candidate.series_uid)
            chunk, mask, _ = ct.extract_chunk_with_mask(
                candidate.center_xyz,
                (7, 96, 96),
            )
            cache.set(cache_key, (chunk, mask))

        # Cache series info once per series
        if candidate.series_uid not in self._seen_series:
            self._seen_series.add(candidate.series_uid)

            series_cache_key = f"series_info_{candidate.series_uid}"
            if cache.get(series_cache_key) is None:
                ct_cache = get_ct_cache(self.config)
                ct = ct_cache.get_segmentation(candidate.series_uid)
                cache.set(series_cache_key, (ct.num_slices, ct.positive_slice_indices))

        return 0, 1  # Dummy return
