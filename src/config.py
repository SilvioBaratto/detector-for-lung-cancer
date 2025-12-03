"""
Configuration module for the lung cancer detection pipeline.

This module provides centralized configuration management following
the Dependency Inversion Principle - depend on abstractions, not concretions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class DataConfig:
    """Configuration for data paths and loading."""

    # Base paths
    data_root: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "LUNA")
    cache_root: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data-unversioned" / "cache")
    model_root: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")

    # Data files
    annotations_file: str = "annotations_with_malignancy.csv"
    candidates_file: str = "candidates.csv"

    # Cache settings
    cache_shards: int = 64
    cache_timeout: int = 1
    cache_size_limit: float = 3e11  # 300GB

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        if isinstance(self.cache_root, str):
            self.cache_root = Path(self.cache_root)
        if isinstance(self.model_root, str):
            self.model_root = Path(self.model_root)

    @property
    def annotations_path(self) -> Path:
        return self.data_root / self.annotations_file

    @property
    def candidates_path(self) -> Path:
        return self.data_root / self.candidates_file

    def get_ct_pattern(self, series_uid: str) -> str:
        """Get glob pattern for finding CT files."""
        return str(self.data_root / "subset*" / f"{series_uid}.mhd")

    def get_cache_path(self, scope: str) -> Path:
        """Get cache path for a specific scope."""
        return self.cache_root / scope

    def get_model_path(self, model_type: str) -> Path:
        """Get model directory path."""
        return self.model_root / model_type


@dataclass
class SegmentationConfig:
    """Configuration for segmentation model and training."""

    # Model architecture
    in_channels: int = 7  # context slices (3 before + 1 center + 3 after)
    n_classes: int = 1
    depth: int = 3
    wf: int = 4  # width factor
    batch_norm: bool = True
    up_mode: str = "upconv"

    # Training
    batch_size: int = 16
    num_workers: int = 8
    epochs: int = 20
    learning_rate: float = 0.001
    momentum: float = 0.99

    # Data
    context_slices: int = 3
    val_stride: int = 10

    # Augmentation defaults
    augmentation: dict = field(default_factory=lambda: {
        'flip': True,
        'offset': 0.03,
        'scale': 0.2,
        'rotate': True,
        'noise': 25.0
    })


@dataclass
class ClassificationConfig:
    """Configuration for classification model and training."""

    # Model architecture
    in_channels: int = 1
    conv_channels: int = 8

    # Input dimensions (depth, height, width)
    input_size: tuple = (32, 48, 48)

    # Training
    batch_size: int = 24
    num_workers: int = 8
    epochs: int = 20
    learning_rate: float = 0.001
    learning_rate_finetune: float = 0.003
    weight_decay: float = 1e-4

    # Data
    val_stride: int = 10
    balance_ratio: int = 0  # 0 = no balancing, N = 1:N pos:neg ratio

    # Augmentation
    use_augmentation: bool = True


@dataclass
class InferenceConfig:
    """Configuration for inference and analysis."""

    # Thresholds
    segmentation_threshold: float = 0.5
    classification_threshold: float = 0.5
    malignancy_threshold: float = 0.5

    # Matching
    match_distance_threshold: float = 0.7  # Fraction of nodule diameter

    # Batch processing
    batch_size: int = 4


class Config:
    """
    Main configuration class that aggregates all config sections.

    Usage:
        config = Config()
        config.data.data_root = Path("/custom/path")

        # Or load from environment
        config = Config.from_environment()
    """

    def __init__(
        self,
        data: Optional[DataConfig] = None,
        segmentation: Optional[SegmentationConfig] = None,
        classification: Optional[ClassificationConfig] = None,
        inference: Optional[InferenceConfig] = None,
    ):
        self.data = data or DataConfig()
        self.segmentation = segmentation or SegmentationConfig()
        self.classification = classification or ClassificationConfig()
        self.inference = inference or InferenceConfig()

    @classmethod
    def from_environment(cls) -> "Config":
        """
        Create configuration from environment variables.

        Environment variables:
            LUNA_DATA_ROOT: Path to LUNA dataset
            LUNA_CACHE_ROOT: Path to cache directory
            LUNA_MODEL_ROOT: Path to model directory
        """
        data_config = DataConfig()

        if os.environ.get("LUNA_DATA_ROOT"):
            data_config.data_root = Path(os.environ["LUNA_DATA_ROOT"])
        if os.environ.get("LUNA_CACHE_ROOT"):
            data_config.cache_root = Path(os.environ["LUNA_CACHE_ROOT"])
        if os.environ.get("LUNA_MODEL_ROOT"):
            data_config.model_root = Path(os.environ["LUNA_MODEL_ROOT"])

        return cls(data=data_config)

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Check data paths exist
        if not self.data.data_root.exists():
            issues.append(f"Data root does not exist: {self.data.data_root}")

        if not self.data.annotations_path.exists():
            issues.append(f"Annotations file not found: {self.data.annotations_path}")

        if not self.data.candidates_path.exists():
            issues.append(f"Candidates file not found: {self.data.candidates_path}")

        # Ensure cache directory can be created
        try:
            self.data.cache_root.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            issues.append(f"Cannot create cache directory: {self.data.cache_root}")

        return issues


# Default global configuration instance
# Can be overridden by applications
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config.from_environment()
    return _default_config


def set_config(config: Config) -> None:
    """Set the default configuration instance."""
    global _default_config
    _default_config = config


# ============================================================================
# Configurable Mixin (Reduces boilerplate)
# ============================================================================

class Configurable:
    """
    Mixin class that provides standardized configuration handling.

    Reduces the repeated pattern of `config or get_config()` across classes.

    Usage:
        class MyTrainer(Configurable):
            def __init__(self, config: Optional[Config] = None, **kwargs):
                super().__init__(config=config)
                # Now self.config is guaranteed to be set
                print(self.config.data.data_root)

    The mixin:
    - Accepts optional config in __init__
    - Falls back to get_config() if none provided
    - Stores config as self.config for subclass use
    """

    _config: Config

    def __init__(self, config: Optional[Config] = None, **kwargs):
        """
        Initialize with configuration.

        Args:
            config: Optional configuration. If None, uses get_config().
            **kwargs: Passed to parent __init__ for cooperative inheritance.
        """
        self._config = config if config is not None else get_config()
        super().__init__(**kwargs)

    @property
    def config(self) -> Config:
        """Get the configuration instance."""
        return self._config

    @config.setter
    def config(self, value: Config) -> None:
        """Set the configuration instance."""
        self._config = value
