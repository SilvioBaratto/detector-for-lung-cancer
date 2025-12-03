"""
Dataset registry for dynamic dataset instantiation.

This module provides a registry pattern for datasets, following the Open/Closed Principle.
New datasets can be added by registering them without modifying existing code.

Note: TrainableDataset Protocol is defined in interfaces.py

Example:
    @register_dataset("classification")
    class ClassificationDataset(Dataset):
        ...

    # Later, create dataset by name
    dataset = get_dataset("classification", config=config, is_validation=True)
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar
from torch.utils.data import Dataset


# ============================================================================
# Dataset Registry
# ============================================================================

_DATASET_REGISTRY: dict[str, type[Dataset]] = {}

T = TypeVar("T", bound=Dataset)


def register_dataset(name: str) -> Callable[[type[T]], type[T]]:
    """
    Decorator to register a dataset class.

    Args:
        name: Unique name for the dataset

    Returns:
        Decorator function

    Example:
        @register_dataset("classification")
        class ClassificationDataset(Dataset):
            ...
    """
    def decorator(cls: type[T]) -> type[T]:
        if name in _DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered")
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str, **kwargs: Any) -> Dataset:
    """
    Create a dataset by name from the registry.

    Args:
        name: Registered dataset name
        **kwargs: Arguments to pass to the dataset constructor

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset name is not registered

    Example:
        dataset = get_dataset("classification", config=config, is_validation=True)
    """
    if name not in _DATASET_REGISTRY:
        available = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> list[str]:
    """
    List all registered dataset names.

    Returns:
        List of registered dataset names
    """
    return list(_DATASET_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """
    Check if a dataset is registered.

    Args:
        name: Dataset name to check

    Returns:
        True if registered, False otherwise
    """
    return name in _DATASET_REGISTRY


def unregister_dataset(name: str) -> bool:
    """
    Remove a dataset from the registry.

    Primarily useful for testing.

    Args:
        name: Dataset name to unregister

    Returns:
        True if removed, False if not found
    """
    if name in _DATASET_REGISTRY:
        del _DATASET_REGISTRY[name]
        return True
    return False


# ============================================================================
# Augmentation Registry
# ============================================================================

_AUGMENTATION_REGISTRY: dict[str, type] = {}

A = TypeVar("A")


def register_augmentation(name: str) -> Callable[[type[A]], type[A]]:
    """
    Decorator to register an augmentation class.

    Args:
        name: Unique name for the augmentation

    Returns:
        Decorator function

    Example:
        @register_augmentation("3d")
        class Augmentation3D:
            ...
    """
    def decorator(cls: type[A]) -> type[A]:
        if name in _AUGMENTATION_REGISTRY:
            raise ValueError(f"Augmentation '{name}' is already registered")
        _AUGMENTATION_REGISTRY[name] = cls
        return cls
    return decorator


def get_augmentation(name: str, **kwargs: Any) -> Any:
    """
    Create an augmentation by name from the registry.

    Args:
        name: Registered augmentation name
        **kwargs: Arguments to pass to the augmentation constructor

    Returns:
        Augmentation instance

    Raises:
        ValueError: If augmentation name is not registered
    """
    if name not in _AUGMENTATION_REGISTRY:
        available = ", ".join(_AUGMENTATION_REGISTRY.keys())
        raise ValueError(f"Unknown augmentation '{name}'. Available: {available}")
    return _AUGMENTATION_REGISTRY[name](**kwargs)


def list_augmentations() -> list[str]:
    """List all registered augmentation names."""
    return list(_AUGMENTATION_REGISTRY.keys())
