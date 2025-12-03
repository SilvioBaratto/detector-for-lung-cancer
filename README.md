# End-to-End Deep Learning Pipeline for Lung Cancer Detection

A production-ready deep learning system for detecting and classifying lung nodules in CT scans using the LUNA16 dataset. The pipeline implements a three-stage cascaded approach optimized for Apple Silicon (MPS) with lightweight, state-of-the-art architectures.

## Table of Contents

- [Overview](#overview)
- [Clinical Background](#clinical-background)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Architectures](#model-architectures)
- [Technical Implementation](#technical-implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [References](#references)

## Overview

Lung cancer is the leading cause of cancer-related deaths worldwide. Early detection through low-dose CT screening can reduce mortality by 20-30%, but radiologists must review hundreds of slices per scan, making computer-aided detection (CAD) systems essential.

This project implements an end-to-end CAD system that:
1. **Segments** potential nodule regions from CT slices
2. **Classifies** candidates as nodule vs. non-nodule
3. **Predicts malignancy** of detected nodules (benign vs. malignant)

## Clinical Background

### Lung Nodules

A pulmonary nodule is a small, round growth in the lung appearing as a white spot on CT scans. Key characteristics:

- **Size**: Typically 3-30mm in diameter
- **Density**: Can be solid, part-solid, or ground-glass
- **Malignancy indicators**: Size > 8mm, irregular margins, rapid growth

### The LUNA16 Challenge

The **LU**ng **N**odule **A**nalysis 2016 (LUNA16) dataset is the standard benchmark for lung nodule detection:

- **888 CT scans** from the LIDC-IDRI database
- **1,186 nodules** annotated by 4 radiologists
- **551,065 candidate locations** (nodule and non-nodule)
- **Malignancy scores** (1-5 scale) for each nodule

### Coordinate Systems

CT data uses two coordinate systems:

```
Patient Space (XYZ)              Voxel Space (IRC)
- Millimeters                    - Array indices
- Physical position              - (Index, Row, Column)
- Direction cosines              - Integer coordinates

        XYZ --> (via affine transform) --> IRC
```

The transformation accounts for:
- **Origin**: Physical position of voxel (0,0,0)
- **Spacing**: Voxel size in mm (typically ~0.7mm x 0.7mm x 1.25mm)
- **Direction**: Orientation matrix for patient position

## Pipeline Architecture

The system uses a cascaded three-stage approach, where each stage progressively refines the detection:

```
+-------------------------------------------------------------------------+
|                           CT Scan Input                                 |
|                    (512 x 512 x ~300 slices)                            |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    STAGE 1: SEGMENTATION                                |
|  +-------------------------------------------------------------------+  |
|  |  UltraLightUNet (0.3M params)                                     |  |
|  |  - Input: 7 context slices (512 x 512 x 7)                        |  |
|  |  - Multi-kernel inverted residual blocks (3x3, 5x5, 7x7)          |  |
|  |  - Depthwise separable convolutions                               |  |
|  |  - Output: Binary mask of candidate regions                       |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
|  Loss: Dice Loss + 8x False Negative Weighting                          |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
                    +---------------------------+
                    |  Connected Component      |
                    |  Analysis & Clustering    |
                    +-------------+-------------+
                                  |
                                  v
+-------------------------------------------------------------------------+
|                    STAGE 2: CLASSIFICATION                              |
|  +-------------------------------------------------------------------+  |
|  |  MIFNet (0.7M params)                                             |  |
|  |  - Input: 3D patch (32 x 48 x 48)                                 |  |
|  |  - Multi-scale feature extraction (1x1, 3x3, 5x5 kernels)         |  |
|  |  - Squeeze-and-Excitation attention                               |  |
|  |  - Output: P(nodule), P(non-nodule)                               |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
|  Loss: Cross-Entropy with class balancing                               |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    STAGE 3: MALIGNANCY PREDICTION                       |
|  +-------------------------------------------------------------------+  |
|  |  Fine-tuned MIFNet                                                |  |
|  |  - Transfer learning from Stage 2                                 |  |
|  |  - Fine-tune last 2 blocks                                        |  |
|  |  - Output: P(benign), P(malignant)                                |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
|  Loss: Cross-Entropy                                                    |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           Final Output                                  |
|  - Nodule locations (XYZ coordinates)                                   |
|  - Nodule probability scores                                            |
|  - Malignancy probability scores                                        |
+-------------------------------------------------------------------------+
```

### Why Three Stages?

1. **Computational Efficiency**: 2D segmentation is ~100x faster than 3D
2. **Class Imbalance**: Reduces ~10^8 voxels to ~10^3 candidates to ~10^1 nodules
3. **Specialization**: Each model optimized for its specific task

## Model Architectures

### Stage 1: UltraLightUNet (Segmentation)

A lightweight U-Net variant designed for efficient medical image segmentation:

```
Architecture: Encoder-Decoder with Skip Connections

Encoder Path:                    Decoder Path:
+----------------+              +----------------+
| Input Conv     |              | Output Conv    |
| 7 -> 16 ch     |              | 16 -> 1 ch     |
+-------+--------+              +-------^--------+
        |                               |
        v                               |
+----------------+    Skip Conn   +-----+--------+
| MKIR Block     | ------------> | Up Block      |
| 16 -> 32 ch    |               | 32 -> 16 ch   |
+-------+--------+               +-------^-------+
        | MaxPool                        | Upsample
        v                                |
+----------------+    Skip Conn   +------+-------+
| MKIR Block     | ------------> | Up Block      |
| 32 -> 64 ch    |               | 64 -> 32 ch   |
+-------+--------+               +-------^-------+
        | MaxPool                        | Upsample
        v                                |
+----------------+    Skip Conn   +------+-------+
| MKIR Block     | ------------> | Up Block      |
| 64 -> 128 ch   |               | 128 -> 64 ch  |
+-------+--------+               +-------^-------+
        |                                |
        +-------> Bottleneck ------------+
```

**Multi-Kernel Inverted Residual (MKIR) Block:**

```
Input
  |
  v
+-------------------+
| 1x1 Conv (Expand) |  Expansion ratio: 2x
+---------+---------+
          |
    +-----+-----+
    |     |     |
    v     v     v
+-----+-----+-----+
| 3x3 | 5x5 | 7x7 |  Depthwise convolutions
|DWise|DWise|DWise|  (multi-scale features)
+--+--+--+--+--+--+
   |     |     |
   +-----+-----+
         |
         v
+-------------------+
| 1x1 Conv (Project)|  Project back
+---------+---------+
          |
          + <-- Skip Connection
          |
          v
       Output
```

**Key Design Choices:**
- **Depthwise Separable Convolutions**: Reduces parameters by ~8-9x
- **Multi-kernel**: Captures features at different scales simultaneously
- **Bilinear Upsampling**: MPS-compatible (no ConvTranspose2d issues)
- **Context Slices**: 7 adjacent slices provide 3D context in 2D network

### Stage 2 & 3: MIFNet (Classification)

Multi-scale Interleaved Fusion Network optimized for 3D nodule analysis:

```
Input: (1, 32, 48, 48) - Single channel 3D patch
        |
        v
+---------------------+
| Stem Conv 3x3x3     |  1 -> 16 channels
+---------+-----------+
          |
          v
+---------------------+
| IFB Stage 1         |  16 -> 32 channels
| + MaxPool 2x2x2     |
+---------+-----------+
          |
          v
+---------------------+
| IFB Stage 2         |  32 -> 64 channels
| + MaxPool 2x2x2     |
+---------+-----------+
          |
          v
+---------------------+
| IFB Stage 3         |  64 -> 128 channels
| + MaxPool 2x2x2     |
+---------+-----------+
          |
          v
+---------------------+
| DWSep Conv + SE     |  Refinement
+---------+-----------+
          |
          v
+---------------------+
| Global Avg Pool     |  Spatial -> 1x1x1
+---------+-----------+
          |
          v
+---------------------+
| FC -> 2 classes     |  Dropout: 0.3
+---------------------+
```

**Interleaved Fusion Block (IFB):**

```
Input
  |
  +-----------------------------+
  |                             |
  v                             |
+------------------+            |
| Multi-Scale      |            |
| Conv3D           |            |
| (1x1, 3x3, 5x5)  |            |
+--------+---------+            |
         |                      |
         v                      |
+------------------+            |
| Squeeze &        |            |
| Excitation       |            |
| (Channel Attn)   |            |
+--------+---------+            |
         |                      |
         + <--------------------+ (Skip if channels match)
         |
         v
      Output
```

**Squeeze-and-Excitation (SE) Attention:**

```
Feature Map F (C x D x H x W)
        |
        v
+---------------------+
| Global Avg Pool     |  C x D x H x W -> C x 1 x 1 x 1
+---------+-----------+
          |
          v
+---------------------+
| FC (C -> C/4)       |  Squeeze
| ReLU                |
+---------+-----------+
          |
          v
+---------------------+
| FC (C/4 -> C)       |  Excitation
| Sigmoid             |
+---------+-----------+
          |
          v
    F * weights        Channel-wise multiplication
```

### Alternative: EfficientNet3D (0.5M params)

A lightweight 3D variant using Mobile Inverted Bottleneck (MBConv) blocks:

```
+---------------------------------------------------+
| MBConv3D Block                                    |
|                                                   |
|  Input -> Expand 1x1 -> DWise 3x3 -> SE ->        |
|           Project 1x1 -> + Input                  |
+---------------------------------------------------+
```

## Technical Implementation

### Loss Functions

**Dice Loss (Segmentation):**

```
Dice = 2 * |P intersection T| / (|P| + |T|)
Loss = 1 - Dice

Where:
  P = Predicted mask
  T = Target mask
  |.| = Sum of pixels
```

With **False Negative Weighting**:

```
Total Loss = Dice(pred, target) + 8 * Dice(pred * target, target)
```

This heavily penalizes missed nodules (critical for medical applications).

**Cross-Entropy (Classification):**

```
CE = -Sum( y_i * log(p_i) )

Where:
  y_i = True class (one-hot)
  p_i = Predicted probability
```

### Data Augmentation

**3D Augmentation for Classification:**

```python
Augmentation3D(
    flip=True,      # Random flip along each axis
    offset=0.1,     # Translation up to 10%
    scale=0.2,      # Scale +/-20%
    rotate=True,    # Random rotation around z-axis
    noise=25.0,     # Gaussian noise (HU units)
)
```

**2D Augmentation for Segmentation:**

```python
SegmentationAugmentation(
    flip=True,
    offset=0.03,
    scale=0.2,
    rotate=True,
    noise=25.0,
)
```

### Class Balancing

The LUNA16 dataset is highly imbalanced:
- ~550,000 non-nodule candidates
- ~1,200 nodule candidates
- Ratio: ~450:1

**Solution: Balanced Sampling**

```python
# During training, sample equal numbers of positive and negative
balance_ratio = 1  # 1:1 ratio
```

### Caching Strategy

CT scans are large (~300MB each). The system uses multi-level caching:

```
+-----------------------------------------------------------+
|                    Memory Cache                           |
|  - LRU cache for recently accessed CTs                    |
|  - Separate caches for segmentation/classification        |
+-----------------------------+-----------------------------+
                              | Miss
                              v
+-----------------------------------------------------------+
|                     Disk Cache                            |
|  - FanoutCache with gzip compression                      |
|  - Preprocessed chunks stored by (series_uid, center)     |
+-----------------------------+-----------------------------+
                              | Miss
                              v
+-----------------------------------------------------------+
|                    Raw CT Files                           |
|  - .mhd metadata + .raw voxel data                        |
|  - Loaded via SimpleITK                                   |
+-----------------------------------------------------------+
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ (with MPS support for Apple Silicon)
- 16GB RAM minimum (32GB recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/End-to-end-detector-for-lung-cancer.git
cd End-to-end-detector-for-lung-cancer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon (MPS)
pip install torch torchvision

# For NVIDIA GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For NVIDIA GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Data Setup

1. Download LUNA16 dataset from [Zenodo](https://zenodo.org/record/3723295) or use the provided script:

```bash
# Set your Zenodo token in .env
echo "ZENODO_TOKEN=your_token_here" > .env

# Run download script
python scripts/download_dataset.py
```

2. Expected data structure:

```
data/LUNA/
  subset0/
    1.3.6.1.4.1.14519.5.2.1.6279.6001.*.mhd
    1.3.6.1.4.1.14519.5.2.1.6279.6001.*.raw
  subset1/
    ...
  annotations_with_malignancy.csv
  candidates.csv
```

## Usage

### Training

All training scripts must be run from the `src/training/` directory:

```bash
cd src/training

# Stage 1: Segmentation
python training_segmentation.py \
    --epochs 20 \
    --batch-size 16 \
    --augmented \
    --model ultralightunet \
    experiment_name

# Stage 2: Classification
python training_classification.py \
    --epochs 20 \
    --batch-size 24 \
    --model mifnet \
    experiment_name

# Stage 3: Malignancy (fine-tuning)
python training_classification.py \
    --malignant \
    --finetune ../../models/classification/cls_*.best.state \
    --finetune-depth 2 \
    --epochs 10 \
    malignancy_experiment
```

### Inference

```bash
cd src/analysis

# Analyze a single CT scan
python nodule_analysis.py <series_uid>

# Run full validation
python nodule_analysis.py --run-validation
```

### Monitoring

```bash
# Start TensorBoard
tensorboard --logdir runs/
```

## Project Structure

```
End-to-end-detector-for-lung-cancer/
  src/
    config.py              # Centralized configuration
    interfaces.py          # Protocol definitions

    dataset/               # Data loading
      ct_loader.py         # CT scan loading & transforms
      classification.py    # Classification datasets
      segmentation.py      # Segmentation datasets
      registry.py          # Dataset registry

    model/                 # Neural networks
      model_segmentation.py    # UltraLightUNet
      model_classification.py  # MIFNet, EfficientNet3D

    training/              # Training logic
      training_segmentation.py
      training_classification.py
      losses.py            # Loss functions
      metrics.py           # Evaluation metrics
      logging.py           # TensorBoard logging
      checkpointing.py     # Model checkpoints

    analysis/              # Inference & evaluation
      nodule_analysis.py   # End-to-end pipeline
      check_nodule_fp_rate.py

    util/                  # Utilities
      device.py            # MPS/CUDA/CPU handling
      disk.py              # Disk caching
      util.py              # Coordinate transforms

  scripts/
    download_dataset.py    # Dataset downloader

  models/                  # Saved checkpoints
    segmentation/
    classification/

  runs/                    # TensorBoard logs
  data/                    # Dataset location
  requirements.txt
```

## Results

### Model Comparison

| Model | Parameters | Memory | Inference Time |
|-------|-----------|--------|----------------|
| UltraLightUNet | 0.3M | ~2GB | ~50ms/slice |
| MIFNet | 0.7M | ~3GB | ~20ms/patch |
| EfficientNet3D | 0.5M | ~2.5GB | ~15ms/patch |

### Performance Metrics

Evaluated on LUNA16 validation set (10-fold cross-validation):

| Stage | Metric | Value |
|-------|--------|-------|
| Segmentation | Dice Score | ~0.85 |
| Segmentation | Recall | ~0.95 |
| Classification | AUC | ~0.95 |
| Classification | F1 Score | ~0.90 |
| Malignancy | AUC | ~0.88 |

## References

### Papers

1. **UltraLightUNet**: [An Ultra-Light Network for Medical Image Segmentation](https://openreview.net/forum?id=BefqqrgdZ1)

2. **MIFNet**: [Multi-scale Interleaved Fusion Network for Pulmonary Nodule Classification](https://www.nature.com/articles/s41598-024-79058-y)

3. **EfficientNet**: [EfficientNet: Rethinking Model Scaling for CNNs](https://www.sciencedirect.com/science/article/pii/S0952197623010862)

4. **LUNA16 Challenge**: [Automatic Detection of Large Pulmonary Solid Nodules](https://luna16.grand-challenge.org/)

5. **Squeeze-and-Excitation Networks**: [Hu et al., CVPR 2018](https://arxiv.org/abs/1709.01507)

6. **Dice Loss**: [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)

### Dataset

- **LIDC-IDRI**: Lung Image Database Consortium image collection
- **LUNA16**: Subset of LIDC-IDRI with standardized annotations
- [Download from Zenodo](https://zenodo.org/record/3723295)

## License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## Acknowledgments

- LUNA16 Challenge organizers
- LIDC-IDRI dataset contributors
- PyTorch team for MPS support
