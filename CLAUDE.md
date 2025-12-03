# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end deep learning pipeline for lung cancer detection using the LUNA16 CT scan dataset. The system implements a three-stage approach:
1. **Segmentation** - U-Net identifies candidate nodule regions in CT slices
2. **Classification** - 3D CNN classifies candidates as nodule vs non-nodule
3. **Malignancy Classification** - Fine-tuned classifier predicts benign vs malignant

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For GPU support (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Commands

### Training

All training scripts must be run from the `src/training/` directory:

```bash
cd src/training

# Segmentation training (U-Net)
python training_segmentation.py --epochs 20 --augmented comment_name

# Classification training (nodule vs non-nodule)
python training_classification.py --epochs 20 comment_name

# Malignancy classification (fine-tuning)
python training_classification.py --malignant --finetune ../../models/classification/cls_*.best.state --finetune-depth 2 comment_name
```

### End-to-End Analysis

```bash
cd src/analysis

# Run on a single CT scan
python nodule_analysis.py <series_uid>

# Run validation set evaluation
python nodule_analysis.py --run-validation
```

## Architecture

### Directory Structure

- `src/` - Main source code
  - `config.py` - Centralized configuration (DataConfig, SegmentationConfig, etc.)
  - `dataset/` - PyTorch Dataset classes for loading CT data
    - `ct_loader.py` - CT scan loading, coordinate transforms, candidate repository
    - `classification.py` - ClassificationDataset, MalignancyDataset
    - `segmentation.py` - SegmentationDataset, TrainingSegmentationDataset
  - `model/` - Neural network architectures
  - `training/` - Training application classes
  - `analysis/` - Inference and evaluation
  - `util/` - Shared utilities (caching, logging, coordinate transforms)
- `data_exploration/` - Jupyter notebook support and visualization utilities

### Key Components

**Configuration (src/config.py):**
- `Config` dataclass with `DataConfig`, `SegmentationConfig`, `ClassificationConfig`, `InferenceConfig`
- Supports environment variables: `LUNA_DATA_ROOT`, `LUNA_CACHE_ROOT`
- Use `get_config()` to get global configuration instance

**Data Pipeline (src/dataset/):**
- `CandidateInfo` NamedTuple: `is_nodule`, `has_annotation`, `is_malignant`, `diameter_mm`, `series_uid`, `center_xyz`
- `CandidateRepository` - Repository pattern for loading and querying candidate data
- `CtScan` (ABC) with `ClassificationCt` and `SegmentationCt` subclasses
- `CtCache` - LRU-style memory cache for CT scans
- Disk caching via `diskcache.FanoutCache` with gzip compression in `util/disk.py`
- Coordinate systems: IRC (Index, Row, Col) for voxels, XYZ for patient space

**Models:**
- `UNetWrapper` (`model/model_segmentation.py`) - 2D U-Net with 7 context slices, depth=3, BatchNorm
- `LunaModel` (`model/model_classification.py`) - 3D CNN with 4 conv blocks, input size (32, 48, 48)
- Both use Kaiming initialization

**Training Features:**
- TensorBoard logging to `runs/` directory
- Data augmentation: `Augmentation3D` class with flip, offset, scale, rotate, noise
- Dice loss for segmentation (with false-negative weighting)
- Cross-entropy loss for classification
- Class balancing via ratio-based sampling

### Data Format

Expected data location: `../../../LUNA/` relative to `src/`:
- `subset*/` - CT scan directories with .mhd/.raw files
- `annotations_with_malignancy.csv` - Nodule annotations with malignancy labels
- `candidates.csv` - Candidate locations (nodule/non-nodule)

Cache location: `../../data-unversioned/cache/`

Models saved to: `../../models/{segmentation,classification}/`

### Key Data Structures

```python
# Named tuples
CandidateInfo: is_nodule, has_annotation, is_malignant, diameter_mm, series_uid, center_xyz
IrcTuple: index, row, col  # voxel coordinates
XyzTuple: x, y, z  # patient space coordinates (mm)

# Access patterns
config = get_config()
candidate_repo = get_candidate_repository(config)
ct_cache = get_ct_cache(config)

# Get all candidates
candidates = candidate_repo.get_all()
nodules = candidate_repo.get_nodules()

# Load CT scan
ct = ct_cache.get_classification(series_uid)
chunk, center_irc = ct.extract_chunk(center_xyz, (32, 48, 48))
```

## Common Patterns

- Training apps use argparse with consistent flags: `--batch-size`, `--num-workers`, `--epochs`
- `enumerateWithEstimate()` wraps iterators with progress logging
- Dependency injection via constructor parameters (config, candidate_repo, ct_cache)
- Multi-GPU support via `nn.DataParallel`
- Global instances via `get_config()`, `get_candidate_repository()`, `get_ct_cache()`
