"""
Device utilities for Apple Silicon MPS optimization.

This module provides:
- Automatic device selection (MPS > CUDA > CPU)
- Memory optimization utilities
- Mixed precision helpers

Optimized for Mac M4 with 16GB unified memory.
"""

import os
from typing import Optional

import torch
from torch import nn

from util.logconf import logging

log = logging.getLogger(__name__)


def get_best_device() -> torch.device:
    """
    Get the best available device for training/inference.

    Priority: MPS (Apple Silicon) > CUDA > CPU

    Returns:
        torch.device: Best available device
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log.info("Using MPS (Apple Silicon) device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        log.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        log.info("Using CPU device")
        return torch.device("cpu")


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get device from string or auto-detect best device.

    Args:
        device_str: Device string ('mps', 'cuda', 'cpu') or None for auto-detect

    Returns:
        torch.device: Selected device
    """
    if device_str is None:
        return get_best_device()

    device_str = device_str.lower()

    if device_str == "mps":
        if not torch.backends.mps.is_available():
            log.warning("MPS not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")

    if device_str == "cuda":
        if not torch.cuda.is_available():
            log.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")

    return torch.device("cpu")


def is_mps_device(device: torch.device) -> bool:
    """Check if device is MPS (Apple Silicon)."""
    return device.type == "mps"


def is_cuda_device(device: torch.device) -> bool:
    """Check if device is CUDA."""
    return device.type == "cuda"


def get_num_workers(device: torch.device) -> int:
    """
    Get optimal number of data loader workers for device.

    Args:
        device: Target device

    Returns:
        Optimal number of workers
    """
    if is_mps_device(device):
        # MPS works best with fewer workers due to unified memory
        return min(4, os.cpu_count() or 4)
    elif is_cuda_device(device):
        return min(8, os.cpu_count() or 8)
    else:
        return 0  # CPU: main process only


def get_optimal_batch_size(
    device: torch.device,
    model_size_mb: float = 50,
    input_size_mb: float = 10,
    memory_fraction: float = 0.7,
) -> int:
    """
    Estimate optimal batch size for device memory.

    Args:
        device: Target device
        model_size_mb: Estimated model size in MB
        input_size_mb: Size of single input in MB
        memory_fraction: Fraction of memory to use

    Returns:
        Estimated optimal batch size
    """
    if is_mps_device(device):
        # Apple Silicon unified memory (estimate 16GB, use 70%)
        available_mb = 16 * 1024 * memory_fraction
    elif is_cuda_device(device):
        props = torch.cuda.get_device_properties(0)
        available_mb = props.total_memory / (1024 * 1024) * memory_fraction
    else:
        # CPU: use system RAM estimation
        available_mb = 8 * 1024 * memory_fraction  # Conservative 8GB estimate

    # Account for model, gradients, optimizer states (~4x model size)
    memory_for_data = available_mb - (model_size_mb * 4)

    # Each sample needs forward + backward memory (~3x input)
    batch_size = int(memory_for_data / (input_size_mb * 3))

    return max(1, min(batch_size, 64))


def optimize_for_mps(model: nn.Module) -> nn.Module:
    """
    Apply MPS-specific optimizations to a model.

    - Ensures all operations are MPS-compatible
    - Converts to FP32 (MPS has limited FP16 support)

    Args:
        model: PyTorch model

    Returns:
        Optimized model
    """
    # Ensure float32 for MPS stability
    model = model.float()

    # Check for unsupported operations
    for name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose3d):
            log.warning(
                f"Module {name} uses ConvTranspose3d which has limited MPS support. "
                "Consider using Upsample + Conv3d instead."
            )

    return model


def clear_memory(device: torch.device) -> None:
    """
    Clear device memory cache.

    Args:
        device: Device to clear
    """
    if is_mps_device(device):
        # MPS uses unified memory, trigger garbage collection
        import gc
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    elif is_cuda_device(device):
        torch.cuda.empty_cache()


def memory_stats(device: torch.device) -> dict[str, float | str]:
    """
    Get memory statistics for device.

    Args:
        device: Target device

    Returns:
        Dictionary with memory stats in MB
    """
    if is_cuda_device(device):
        return {
            "allocated": torch.cuda.memory_allocated(device) / (1024 * 1024),
            "reserved": torch.cuda.memory_reserved(device) / (1024 * 1024),
            "max_allocated": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        }
    elif is_mps_device(device):
        # MPS has limited memory introspection
        if hasattr(torch.mps, 'current_allocated_memory'):
            return {
                "allocated": torch.mps.current_allocated_memory() / (1024 * 1024),
            }
        return {"allocated": 0, "note": "MPS memory stats not available"}
    else:
        return {"note": "CPU memory stats not tracked"}


class DeviceContext:
    """
    Context manager for device-specific settings.

    Usage:
        with DeviceContext(device) as ctx:
            model = ctx.prepare_model(model)
            # Training loop
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_best_device()
        self._original_grad_enabled = torch.is_grad_enabled()

    def __enter__(self) -> "DeviceContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_memory(self.device)
        return False

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for device."""
        model = model.to(self.device)
        if is_mps_device(self.device):
            model = optimize_for_mps(model)
        return model

    def prepare_batch(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Move batch tensors to device."""
        return tuple(t.to(self.device) for t in tensors)


# Convenience function for training scripts
def setup_device(use_cuda: bool = True) -> torch.device:
    """
    Setup device for training (backward compatible with existing scripts).

    Args:
        use_cuda: If True, prefer GPU (MPS or CUDA)

    Returns:
        Selected device
    """
    if not use_cuda:
        return torch.device("cpu")
    return get_best_device()
