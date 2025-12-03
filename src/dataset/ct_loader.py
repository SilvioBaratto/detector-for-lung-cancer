"""
CT scan loading and candidate information management.

This module provides clean abstractions for:
- Loading and manipulating CT scans
- Managing candidate nodule information
- Coordinate system transformations

Design Principles:
- Single Responsibility: Each class has one job
- Open/Closed: Extend via inheritance, don't modify
- Dependency Inversion: Depend on Config abstraction
"""

from __future__ import annotations

import csv
import glob
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import SimpleITK as sitk

from config import Config, get_config
from util.util import XyzTuple, IrcTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class CandidateInfo(NamedTuple):
    """Information about a nodule candidate."""
    is_nodule: bool
    has_annotation: bool
    is_malignant: bool
    diameter_mm: float
    series_uid: str
    center_xyz: tuple[float, float, float]


@dataclass
class CtMetadata:
    """Spatial metadata for a CT scan."""
    origin_xyz: XyzTuple
    spacing_xyz: XyzTuple
    direction: np.ndarray

    def xyz_to_irc(self, coord_xyz: tuple[float, float, float]) -> IrcTuple:
        """Convert XYZ (patient space) to IRC (voxel) coordinates."""
        return xyz2irc(coord_xyz, self.origin_xyz, self.spacing_xyz, self.direction)


# ============================================================================
# Candidate Information Repository
# ============================================================================

class CandidateRepository:
    """
    Repository for loading and querying candidate nodule information.

    Follows Repository Pattern - abstracts data access.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._candidates: Optional[list[CandidateInfo]] = None
        self._by_series: Optional[dict[str, list[CandidateInfo]]] = None

    def _load_candidates(self, require_on_disk: bool = True) -> list[CandidateInfo]:
        """Load candidates from CSV files."""
        config = self.config

        # Find series present on disk
        mhd_pattern = str(config.data.data_root / "subset*" / "*.mhd")
        mhd_list = glob.glob(mhd_pattern)
        present_on_disk = {os.path.split(p)[-1][:-4] for p in mhd_list}

        candidates: list[CandidateInfo] = []

        # Load annotations (nodules with malignancy info)
        annotations_path = config.data.annotations_path
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

        with open(annotations_path, "r") as f:
            for row in list(csv.reader(f))[1:]:
                series_uid = row[0]
                center_xyz = (float(row[1]), float(row[2]), float(row[3]))
                diameter_mm = float(row[4])
                is_malignant = row[5] == 'True'

                candidates.append(CandidateInfo(
                    is_nodule=True,
                    has_annotation=True,
                    is_malignant=is_malignant,
                    diameter_mm=diameter_mm,
                    series_uid=series_uid,
                    center_xyz=center_xyz,
                ))

        # Load non-nodule candidates
        candidates_path = config.data.candidates_path
        if not candidates_path.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

        with open(candidates_path, "r") as f:
            for row in list(csv.reader(f))[1:]:
                series_uid = row[0]

                if require_on_disk and series_uid not in present_on_disk:
                    continue

                is_nodule = bool(int(row[4]))
                center_xyz = (float(row[1]), float(row[2]), float(row[3]))

                if not is_nodule:
                    candidates.append(CandidateInfo(
                        is_nodule=False,
                        has_annotation=False,
                        is_malignant=False,
                        diameter_mm=0.0,
                        series_uid=series_uid,
                        center_xyz=center_xyz,
                    ))

        # Sort: nodules first (largest to smallest), then non-nodules
        candidates.sort(key=lambda x: (not x.is_nodule, -x.diameter_mm))
        return candidates

    def get_all(self, require_on_disk: bool = True) -> list[CandidateInfo]:
        """Get all candidates."""
        if self._candidates is None:
            self._candidates = self._load_candidates(require_on_disk)
        return self._candidates

    def get_by_series(self, require_on_disk: bool = True) -> dict[str, list[CandidateInfo]]:
        """Get candidates grouped by series UID."""
        if self._by_series is None:
            candidates = self.get_all(require_on_disk)
            self._by_series = {}
            for candidate in candidates:
                self._by_series.setdefault(candidate.series_uid, []).append(candidate)
        return self._by_series

    def get_for_series(self, series_uid: str) -> list[CandidateInfo]:
        """Get candidates for a specific series."""
        return self.get_by_series().get(series_uid, [])

    def get_nodules(self, require_on_disk: bool = True) -> list[CandidateInfo]:
        """Get only nodule candidates."""
        return [c for c in self.get_all(require_on_disk) if c.is_nodule]

    def get_non_nodules(self, require_on_disk: bool = True) -> list[CandidateInfo]:
        """Get only non-nodule candidates."""
        return [c for c in self.get_all(require_on_disk) if not c.is_nodule]

    def get_malignant(self, require_on_disk: bool = True) -> list[CandidateInfo]:
        """Get malignant nodules."""
        return [c for c in self.get_all(require_on_disk) if c.is_malignant]

    def get_benign(self, require_on_disk: bool = True) -> list[CandidateInfo]:
        """Get benign nodules."""
        return [c for c in self.get_all(require_on_disk) if c.is_nodule and not c.is_malignant]


# Global repository instance (can be replaced for testing)
_candidate_repository: Optional[CandidateRepository] = None


def get_candidate_repository(config: Optional[Config] = None) -> CandidateRepository:
    """Get the candidate repository instance."""
    global _candidate_repository
    if _candidate_repository is None:
        _candidate_repository = CandidateRepository(config)
    return _candidate_repository


# ============================================================================
# CT Scan Classes
# ============================================================================

class CtScan(ABC):
    """
    Abstract base class for CT scan loading and manipulation.

    Template Method Pattern: _post_load() hook for subclass customization.
    """

    def __init__(self, series_uid: str, config: Optional[Config] = None):
        self.series_uid = series_uid
        self.config = config or get_config()

        # Load CT file
        mhd_pattern = self.config.data.get_ct_pattern(series_uid)
        mhd_paths = glob.glob(mhd_pattern)

        if not mhd_paths:
            raise FileNotFoundError(
                f"CT scan not found for series {series_uid}. Pattern: {mhd_pattern}"
            )

        ct_mhd = sitk.ReadImage(mhd_paths[0])
        self.hu_array = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        self.metadata = CtMetadata(
            origin_xyz=XyzTuple(*ct_mhd.GetOrigin()),
            spacing_xyz=XyzTuple(*ct_mhd.GetSpacing()),
            direction=np.array(ct_mhd.GetDirection()).reshape(3, 3),
        )

        self._post_load()

    @abstractmethod
    def _post_load(self) -> None:
        """Hook for subclass-specific post-load processing."""
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self.hu_array.shape

    @property
    def num_slices(self) -> int:
        return self.hu_array.shape[0]

    def xyz_to_irc(self, coord_xyz: tuple[float, float, float]) -> IrcTuple:
        """Convert XYZ to IRC coordinates."""
        return self.metadata.xyz_to_irc(coord_xyz)

    def extract_chunk(
        self,
        center_xyz: tuple[float, float, float],
        size_irc: tuple[int, int, int],
    ) -> tuple[np.ndarray, IrcTuple]:
        """
        Extract a chunk centered on the given coordinates.

        Args:
            center_xyz: Center in patient space (mm)
            size_irc: Size of chunk (index, row, col)

        Returns:
            (chunk_array, center_irc)
        """
        center_irc = self.xyz_to_irc(center_xyz)

        slices = []
        for axis, center_val in enumerate(center_irc):
            half_size = size_irc[axis] // 2
            start = int(round(center_val)) - half_size
            end = start + size_irc[axis]

            # Validate center is within bounds
            if not (0 <= center_val < self.hu_array.shape[axis]):
                raise ValueError(
                    f"Center out of bounds: axis={axis}, center={center_val}, "
                    f"shape={self.hu_array.shape[axis]}"
                )

            # Clamp to array bounds
            if start < 0:
                start, end = 0, size_irc[axis]
            elif end > self.hu_array.shape[axis]:
                end = self.hu_array.shape[axis]
                start = end - size_irc[axis]

            slices.append(slice(start, end))

        chunk = self.hu_array[tuple(slices)]
        return chunk, center_irc


class ClassificationCt(CtScan):
    """CT scan optimized for classification (clips HU values)."""

    def _post_load(self) -> None:
        self.hu_array.clip(-1000, 1000, out=self.hu_array)


class SegmentationCt(CtScan):
    """CT scan optimized for segmentation (includes nodule masks)."""

    def __init__(
        self,
        series_uid: str,
        config: Optional[Config] = None,
        candidate_repo: Optional[CandidateRepository] = None,
    ):
        self._candidate_repo = candidate_repo or get_candidate_repository(config)
        super().__init__(series_uid, config)

    def _post_load(self) -> None:
        candidates = self._candidate_repo.get_for_series(self.series_uid)
        self.nodule_candidates = [c for c in candidates if c.is_nodule]
        self.positive_mask = self._build_nodule_mask()
        self.positive_slice_indices = self._find_positive_slices()

    def _build_nodule_mask(self, hu_threshold: float = -700) -> np.ndarray:
        """Build boolean mask marking nodule locations."""
        mask = np.zeros_like(self.hu_array, dtype=np.bool_)

        for candidate in self.nodule_candidates:
            center_irc = self.xyz_to_irc(candidate.center_xyz)
            ci, cr, cc = int(center_irc.index), int(center_irc.row), int(center_irc.col)

            # Find nodule extent in each dimension
            ri = self._find_extent(ci, cr, cc, axis=0, threshold=hu_threshold)
            rr = self._find_extent(ci, cr, cc, axis=1, threshold=hu_threshold)
            rc = self._find_extent(ci, cr, cc, axis=2, threshold=hu_threshold)

            # Mark bounding box
            mask[
                ci - ri : ci + ri + 1,
                cr - rr : cr + rr + 1,
                cc - rc : cc + rc + 1,
            ] = True

        # Restrict to voxels above density threshold
        return mask & (self.hu_array > hu_threshold)

    def _find_extent(
        self,
        ci: int, cr: int, cc: int,
        axis: int,
        threshold: float,
        initial: int = 2,
    ) -> int:
        """Find extent of nodule along one axis."""
        radius = initial
        try:
            while True:
                if axis == 0:
                    check = (self.hu_array[ci + radius, cr, cc] > threshold and
                             self.hu_array[ci - radius, cr, cc] > threshold)
                elif axis == 1:
                    check = (self.hu_array[ci, cr + radius, cc] > threshold and
                             self.hu_array[ci, cr - radius, cc] > threshold)
                else:
                    check = (self.hu_array[ci, cr, cc + radius] > threshold and
                             self.hu_array[ci, cr, cc - radius] > threshold)

                if check:
                    radius += 1
                else:
                    break
        except IndexError:
            radius = max(1, radius - 1)

        return radius

    def _find_positive_slices(self) -> list[int]:
        """Find slice indices containing nodules."""
        slice_sums = self.positive_mask.sum(axis=(1, 2))
        return slice_sums.nonzero()[0].tolist()

    def extract_chunk_with_mask(
        self,
        center_xyz: tuple[float, float, float],
        size_irc: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, IrcTuple]:
        """
        Extract CT chunk and corresponding mask.

        Returns:
            (ct_chunk, mask_chunk, center_irc)
        """
        chunk, center_irc = self.extract_chunk(center_xyz, size_irc)

        # Extract corresponding mask region
        slices = []
        for axis, center_val in enumerate(center_irc):
            half_size = size_irc[axis] // 2
            start = int(round(center_val)) - half_size
            end = start + size_irc[axis]

            if start < 0:
                start, end = 0, size_irc[axis]
            elif end > self.hu_array.shape[axis]:
                end = self.hu_array.shape[axis]
                start = end - size_irc[axis]

            slices.append(slice(start, end))

        mask_chunk = self.positive_mask[tuple(slices)]
        return chunk, mask_chunk, center_irc


# ============================================================================
# CT Cache (manages memory efficiently)
# ============================================================================

class CtCache:
    """
    LRU-style cache for CT scans.

    Keeps only one CT in memory at a time to manage RAM usage.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config
        self._classification_ct: Optional[ClassificationCt] = None
        self._segmentation_ct: Optional[SegmentationCt] = None

    def get_classification(self, series_uid: str) -> ClassificationCt:
        """Get classification CT, loading if necessary."""
        if self._classification_ct is None or self._classification_ct.series_uid != series_uid:
            self._classification_ct = ClassificationCt(series_uid, self.config)
        return self._classification_ct

    def get_segmentation(self, series_uid: str) -> SegmentationCt:
        """Get segmentation CT, loading if necessary."""
        if self._segmentation_ct is None or self._segmentation_ct.series_uid != series_uid:
            self._segmentation_ct = SegmentationCt(series_uid, self.config)
        return self._segmentation_ct

    def clear(self) -> None:
        """Clear cached CTs."""
        self._classification_ct = None
        self._segmentation_ct = None


# Global cache instance
_ct_cache: Optional[CtCache] = None


def get_ct_cache(config: Optional[Config] = None) -> CtCache:
    """Get the global CT cache."""
    global _ct_cache
    if _ct_cache is None:
        _ct_cache = CtCache(config)
    return _ct_cache
