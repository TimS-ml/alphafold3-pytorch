"""
Test suite for the AlphaFold3 command-line interface (CLI).

This module contains tests that verify the functionality of the CLI tool,
which allows users to run AlphaFold3 inference from the command line without
writing Python code.
"""

import os
# Enable type checking and debug mode for more verbose error messages during testing
os.environ['TYPECHECK'] = 'True'
os.environ['DEBUG'] = 'True'
from shutil import rmtree
from pathlib import Path

import torch

from alphafold3_pytorch.cli import cli

from alphafold3_pytorch.alphafold3 import (
    Alphafold3
)

def test_cli():
    """
    Test the command-line interface for AlphaFold3 inference.

    This test verifies that the CLI can:
    1. Load a pre-trained AlphaFold3 model from a checkpoint
    2. Accept protein sequence inputs via command-line arguments
    3. Generate predictions and save them to an mmCIF output file

    The test is important because it validates the primary user-facing interface
    for running AlphaFold3 predictions, ensuring that users can perform inference
    without writing Python code.
    """
    # Create a minimal AlphaFold3 model with small dimensions for testing
    # These small dimensions make the test run quickly while still validating functionality
    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,  # Number of input features per atom
        dim_template_feats = 44,  # Number of template features
        num_molecule_mods = 0  # No modified molecules in this test
    )

    # Save the model to a checkpoint file for the CLI to load
    checkpoint_path = './test-folder/test-cli-alphafold3.pt'
    alphafold3.save(checkpoint_path, overwrite = True)

    # Run the CLI with command-line arguments:
    # - Two protein chains: 'AG' (Alanine-Glycine) and 'TC' (Threonine-Cysteine)
    # - Output path for the predicted structure
    cli([
        '--checkpoint', checkpoint_path,  # Path to model checkpoint
        '-prot', 'AG',  # First protein chain sequence
        '-prot', 'TC',  # Second protein chain sequence
        '--output',
        './test-folder/output.mmcif'  # Output file for predicted structure
    ], standalone_mode = False)

    # Verify that the CLI successfully generated the output file
    assert Path('./test-folder/output.mmcif').exists()

    # Clean up test artifacts
    rmtree('./test-folder')
