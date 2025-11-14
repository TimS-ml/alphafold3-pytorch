"""Core utility functions for AlphaFold3 PyTorch implementation.

This module provides fundamental helper functions used throughout the codebase:

- Existence checking: Check if values are None or not
- Default value handling: Return default values when inputs are None
- Array/list utilities: Get first element, find mode
- GPU detection: Identify NVIDIA, ROCm, or unknown GPU types
- Function utilities: Identity and constant functions

These are low-level utilities that form the building blocks for more complex
operations in other modules. They handle common patterns like None-checking,
default values, and basic data manipulations.
"""

import torch

import numpy as np

from beartype.typing import Any, Iterable, List


def exists(val: Any) -> bool:
    """Check if a value exists (is not None).

    This is a common pattern used throughout the codebase to check for the
    presence of optional values. It's more readable than `val is not None`
    and can be easily used in filter operations.

    Args:
        val: The value to check for existence.

    Returns:
        bool: True if the value is not None, False otherwise.

    Example:
        >>> exists(5)
        True
        >>> exists(None)
        False
        >>> exists(0)  # Note: 0 and False are not None
        True
    """
    return val is not None


def not_exists(val: Any) -> bool:
    """Check if a value does not exist (is None).

    Inverse of exists(). Used for readability when checking if optional
    values are missing.

    Args:
        val: The value to check for non-existence.

    Returns:
        bool: True if the value is None, False otherwise.

    Example:
        >>> not_exists(None)
        True
        >>> not_exists(5)
        False
    """
    return val is None


def default(v: Any, d: Any) -> Any:
    """Return a default value if the input is None.

    A common pattern for handling optional parameters with defaults. Returns
    the input value if it exists, otherwise returns the default value.

    Args:
        v: The value to check. Can be any type.
        d: The default value to return if v is None. Can be any type.

    Returns:
        Any: The value v if it exists (is not None), otherwise the default value d.

    Example:
        >>> default(5, 10)
        5
        >>> default(None, 10)
        10
        >>> default(0, 10)  # 0 is not None, so it's returned
        0
    """
    return v if exists(v) else d


def first(arr: Iterable[Any]) -> Any:
    """Get the first element of an iterable.

    Convenience function to get the first element. Assumes the iterable
    supports indexing (like lists, tuples, or arrays).

    Args:
        arr: An iterable object that supports indexing (e.g., list, tuple, array).

    Returns:
        Any: The first element of the iterable.

    Raises:
        IndexError: If the iterable is empty.
        TypeError: If the iterable doesn't support indexing.

    Example:
        >>> first([1, 2, 3])
        1
        >>> first("hello")
        'h'
    """
    return arr[0]


def always(value):
    """Create a function that always returns the same value.

    Returns a constant function that ignores all arguments and always returns
    the specified value. Useful for creating default callbacks or placeholder
    functions.

    Args:
        value: The value to always return.

    Returns:
        Callable: A function that takes any arguments and always returns value.

    Example:
        >>> const_five = always(5)
        >>> const_five()
        5
        >>> const_five(1, 2, 3, x=10)
        5
    """

    def inner(*args, **kwargs):
        """Inner function that ignores arguments and returns the constant value."""
        return value

    return inner


def identity(x, *args, **kwargs):
    """Return the input unchanged (identity function).

    The mathematical identity function that returns its first argument.
    Additional arguments are ignored. Useful as a default transformation
    or no-op function.

    Args:
        x: The value to return.
        *args: Additional positional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Any: The input value x unchanged.

    Example:
        >>> identity(5)
        5
        >>> identity([1, 2, 3])
        [1, 2, 3]
        >>> identity(5, 10, 20, foo='bar')
        5
    """
    return x


def np_mode(x: np.ndarray) -> Any:
    """Find the mode (most common value) in a 1D NumPy array.

    Computes the statistical mode by finding the value that appears most
    frequently in the array. Returns both the mode value and its count.

    Args:
        x: A 1D NumPy array containing the data.

    Returns:
        tuple: A tuple containing:
            - mode_value: The most frequently occurring value in the array
            - mode_count: The number of times the mode appears

    Raises:
        AssertionError: If the input array is not 1-dimensional.

    Example:
        >>> x = np.array([1, 2, 2, 3, 2, 4])
        >>> np_mode(x)
        (2, 3)  # 2 appears 3 times
        >>> x = np.array([5, 5, 5, 1, 1])
        >>> np_mode(x)
        (5, 3)  # 5 appears 3 times
    """
    assert x.ndim == 1, f"Input NumPy array must be 1D, not {x.ndim}D."
    # Get unique values and their counts
    values, counts = np.unique(x, return_counts=True)
    # Find the index of the maximum count
    m = counts.argmax()
    # Return the value with maximum count and the count itself
    return values[m], counts[m]


def get_gpu_type() -> str:
    """Detect and return the type of GPU available in the system.

    Checks if CUDA is available and attempts to identify the GPU vendor
    (NVIDIA, AMD/ROCm, or unknown). Useful for conditional code paths
    based on GPU type or for logging system information.

    Returns:
        str: A descriptive string indicating the GPU type:
            - "NVIDIA GPU detected" for NVIDIA GPUs
            - "ROCm GPU detected" for AMD GPUs with ROCm
            - "Unknown GPU type" for other CUDA-compatible GPUs
            - "No GPU available" if CUDA is not available

    Example:
        >>> get_gpu_type()
        'NVIDIA GPU detected'  # On a system with NVIDIA GPU
        >>> get_gpu_type()
        'ROCm GPU detected'    # On a system with AMD GPU
        >>> get_gpu_type()
        'No GPU available'     # On a CPU-only system

    Note:
        This function only checks the first GPU (index 0) if multiple GPUs
        are present.
    """
    if torch.cuda.is_available():
        # Get the name of the first GPU device
        device_name = torch.cuda.get_device_name(0).lower()
        # Check for NVIDIA GPUs
        if "nvidia" in device_name:
            return "NVIDIA GPU detected"
        # Check for AMD GPUs with ROCm
        elif "amd" in device_name or "gfx" in device_name:
            return "ROCm GPU detected"
        else:
            return "Unknown GPU type"
    else:
        return "No GPU available"
