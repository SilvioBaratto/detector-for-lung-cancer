"""
Dataset package for lung cancer detection.

This package provides:
- CT scan loading and manipulation (ct_loader)
- Classification datasets (classification)
- Segmentation datasets (segmentation)
- Dataset registry for dynamic instantiation (registry)
"""

from dataset.ct_loader import (
    # Data structures
    CandidateInfo,
    CtMetadata,
    # Repository
    CandidateRepository,
    get_candidate_repository,
    # CT classes
    CtScan,
    ClassificationCt,
    SegmentationCt,
    # Cache
    CtCache,
    get_ct_cache,
)

from dataset.classification import (
    Augmentation3D,
    ClassificationDataset,
    MalignancyDataset,
)

from dataset.segmentation import (
    SegmentationDataset,
    TrainingSegmentationDataset,
    CachePreparationDataset,
)

from dataset.registry import (
    # Dataset registry
    register_dataset,
    get_dataset,
    list_datasets,
    # Augmentation registry
    register_augmentation,
    get_augmentation,
    list_augmentations,
)

__all__ = [
    # CT Loading
    'CandidateInfo',
    'CtMetadata',
    'CandidateRepository',
    'get_candidate_repository',
    'CtScan',
    'ClassificationCt',
    'SegmentationCt',
    'CtCache',
    'get_ct_cache',
    # Classification
    'Augmentation3D',
    'ClassificationDataset',
    'MalignancyDataset',
    # Segmentation
    'SegmentationDataset',
    'TrainingSegmentationDataset',
    'CachePreparationDataset',
    # Registry
    'register_dataset',
    'get_dataset',
    'list_datasets',
    'register_augmentation',
    'get_augmentation',
    'list_augmentations',
]
