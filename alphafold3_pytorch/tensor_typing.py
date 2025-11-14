"""
Tensor Typing Module for AlphaFold3 PyTorch Implementation.

This module provides tensor type checking utilities using jaxtyping and beartype.
It handles environment-based configuration for type checking, debugging, and
specialized features like DeepSpeed checkpointing and Nim compilation.

The module defines custom type annotations for PyTorch tensors and provides
utilities to enable/disable runtime type checking based on environment variables.
"""

from __future__ import annotations

import sh
from functools import partial
import importlib.metadata
from packaging import version

import torch
import numpy as np

from beartype import beartype
from beartype.door import is_bearable

from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import DisorderedResidue, Residue

from environs import Env
from jaxtyping import (
    Float,
    Int,
    Bool,
    Shaped,
    jaxtyped
)
from loguru import logger

from torch import Tensor

# Environment variable configuration

env = Env()
env.read_env()

# Helper functions

def always(value):
    """
    Create a function that always returns the same value regardless of arguments.

    Args:
        value: The value to always return.

    Returns:
        A function that ignores all arguments and returns the given value.
    """
    def inner(*args, **kwargs):
        return value
    return inner

def identity(t):
    """
    Identity function that returns its input unchanged.

    Args:
        t: Input value of any type.

    Returns:
        The input value unchanged.
    """
    return t

# Custom PyTorch tensor typing (jaxtyping works for PyTorch despite the name)

class TorchTyping:
    """
    Wrapper class to adapt jaxtyping type annotations for PyTorch tensors.

    Jaxtyping natively supports JAX arrays, but can be adapted to work with PyTorch.
    This class wraps jaxtyping's abstract data types to work with torch.Tensor.

    Args:
        abstract_dtype: A jaxtyping abstract dtype (Float, Int, Bool, or Shaped).
    """
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """
        Create a type annotation for a PyTorch tensor with specific shape.

        Args:
            shapes: Shape specification string (e.g., "b n d" for batch, sequence, dimension).

        Returns:
            A jaxtyping type annotation configured for torch.Tensor with the given shape.
        """
        return self.abstract_dtype[Tensor, shapes]

# Create PyTorch-specific type annotations
Shaped = TorchTyping(Shaped)  # For tensors with specific shapes
Float  = TorchTyping(Float)   # For float tensors
Int    = TorchTyping(Int)     # For integer tensors
Bool   = TorchTyping(Bool)    # For boolean tensors

# Type aliases for biomolecular structure types

IntType = int | np.int32 | np.int64  # Integer type from Python or NumPy
AtomType = Atom | DisorderedAtom  # Atom types from BioPython
ResidueType = Residue | DisorderedResidue  # Residue types from BioPython
ChainType = Chain  # Chain type from BioPython
TokenType = AtomType | ResidueType  # Token can be either an atom or residue

# Package availability and checkpointing configuration

def package_available(package_name: str) -> bool:
    """
    Check if a Python package is available in the current environment.

    Args:
        package_name: The name of the package to check (e.g., 'deepspeed').

    Returns:
        True if the package is installed and available, False otherwise.
    """
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

# Configure gradient checkpointing (DeepSpeed or PyTorch)
# Gradient checkpointing trades compute for memory by recomputing activations during backward pass

DEEPSPEED_CHECKPOINTING = env.bool('DEEPSPEED_CHECKPOINTING', False)

if DEEPSPEED_CHECKPOINTING:
    # Use DeepSpeed's checkpointing for distributed training optimization
    assert package_available("deepspeed"), "DeepSpeed must be installed for checkpointing."

    import deepspeed

    checkpoint = deepspeed.checkpointing.checkpoint
else:
    # Use PyTorch's native checkpointing with non-reentrant mode for better compatibility
    checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant = False)

# Nim language integration for performance-critical code
# Nim is a statically typed compiled language that can provide performance benefits

try:
    # Check if Nim compiler is available on the system
    sh.which('nim')
    HAS_NIM = True
    # Get the Nim version for compatibility checking
    NIM_VERSION = sh.nim(eval = 'echo NimVersion', hints = 'off')
except sh.ErrorReturnCode_1:
    HAS_NIM = False
    NIM_VERSION = None

# Allow enabling Nim via environment variable, defaulting to availability
USE_NIM = env.bool('USE_NIM', HAS_NIM)

# Validate Nim configuration
assert not (USE_NIM and not HAS_NIM), 'you cannot use Nim if it is not available'
assert not HAS_NIM or version.parse(NIM_VERSION) >= version.parse('2.0.8'), 'nim version must be 2.0.8 or above'

# Detect if running in GitHub CI for conditional behavior
IS_GITHUB_CI = env.bool('IS_GITHUB_CI', False)

# Runtime type checking configuration
# Type checking can be disabled for performance in production while enabled during development

should_typecheck = env.bool('TYPECHECK', False)  # Enable with TYPECHECK=True environment variable
IS_DEBUGGING = env.bool('DEBUG', False)  # Enable with DEBUG=True environment variable

# Create the typecheck decorator that can be enabled/disabled
# When enabled, uses jaxtyping + beartype for runtime type validation
# When disabled, acts as identity decorator (no overhead)
typecheck = jaxtyped(typechecker = beartype) if should_typecheck else identity

# Create a beartype isinstance check that can be disabled for performance
beartype_isinstance = is_bearable if should_typecheck else always(True)

# Log the current configuration
if should_typecheck:
    logger.info("Type checking is enabled.")
else:
    logger.info("Type checking is disabled.")

if IS_DEBUGGING:
    logger.info("Debugging is enabled.")
else:
    logger.info("Debugging is disabled.")

__all__ = [
    Shaped,
    Float,
    Int,
    Bool,
    typecheck,
    should_typecheck,
    beartype_isinstance,
    checkpoint,
    IS_DEBUGGING,
    IS_GITHUB_CI,
    USE_NIM
]
