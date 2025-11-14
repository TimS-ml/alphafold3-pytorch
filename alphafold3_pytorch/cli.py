"""
Command-Line Interface for AlphaFold3 Structure Prediction

This module provides a simple CLI for running AlphaFold3 structure predictions
from the command line. Users can specify protein, RNA, and DNA sequences, and
the model will generate predicted structures in mmCIF format.

The CLI uses Click for argument parsing and supports:
- Loading pre-trained model checkpoints
- Multi-sequence input (proteins, RNA, DNA)
- Configurable sampling steps
- CUDA acceleration support
- mmCIF output format

Example Usage:
    # Predict structure for a single protein
    $ alphafold3 --checkpoint model.ckpt --protein MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL

    # Predict protein-RNA complex
    $ alphafold3 --checkpoint model.ckpt --protein SEQUENCE1 --rna AUGCAUGC

    # With custom sampling steps and output
    $ alphafold3 --checkpoint model.ckpt --protein SEQ --steps 100 --output pred.cif

Note:
    The output is saved in mmCIF format, which is the modern standard for
    macromolecular structure representation and is compatible with most
    structural biology software.
"""

from __future__ import annotations

import click
from pathlib import Path

import torch

from alphafold3_pytorch import (
    Alphafold3,
    Alphafold3Input
)

from Bio.PDB.mmcifio import MMCIFIO

# Simple CLI using Click framework

@click.command()
@click.option('-ckpt', '--checkpoint', type = str, help = 'path to alphafold3 checkpoint')
@click.option('-prot', '--protein', type = str, multiple = True, help = 'protein sequences')
@click.option('-rna', '--rna', type = str, multiple = True, help = 'single stranded rna sequences')
@click.option('-dna', '--dna', type = str, multiple = True, help = 'single stranded dna sequences')
@click.option('-steps', '--num-sample-steps', type = int, help = 'number of sampling steps to take')
@click.option('-cuda', '--use-cuda', type = bool, help = 'use cuda if available')
@click.option('-o', '--output', type = str, help = 'output path', default = 'output.cif')
def cli(
    checkpoint: str,
    protein: list[str],
    rna: list[str],
    dna: list[str],
    num_sample_steps: int,
    use_cuda: bool,
    output: str
):
    """
    Predict biomolecular structures using AlphaFold3.

    This command loads a pre-trained AlphaFold3 model and predicts the 3D structure
    of the input sequences. The prediction is saved as an mmCIF file.

    Args:
        checkpoint: Path to the trained AlphaFold3 model checkpoint file
        protein: One or more protein sequences (amino acid sequences)
                Can be specified multiple times for complexes
        rna: One or more single-stranded RNA sequences (nucleotide sequences)
             Can be specified multiple times for complexes
        dna: One or more single-stranded DNA sequences (nucleotide sequences)
             Can be specified multiple times for complexes
        num_sample_steps: Number of diffusion sampling steps (higher = slower but potentially better)
        use_cuda: Whether to use CUDA acceleration if available
        output: Path where the predicted structure will be saved (mmCIF format)

    Raises:
        AssertionError: If checkpoint file doesn't exist

    Example:
        # Predict a protein structure
        $ alphafold3 --checkpoint weights.ckpt --protein MKILVTALALAALAVASAAG

        # Predict a protein-RNA complex
        $ alphafold3 --checkpoint weights.ckpt \\
                     --protein MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAP \\
                     --rna AUGCAUGCUAGC

        # Use GPU with custom sampling steps
        $ alphafold3 --checkpoint weights.ckpt \\
                     --protein SEQUENCE \\
                     --use-cuda True \\
                     --num-sample-steps 200 \\
                     --output my_prediction.cif
    """

    # Validate that checkpoint file exists
    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f'AlphaFold 3 checkpoint must exist at {str(checkpoint_path)}'

    # Prepare input data structure from sequences
    # Alphafold3Input handles conversion of raw sequences to model-compatible format
    alphafold3_input = Alphafold3Input(
        proteins = list(protein),  # Convert tuple to list for processing
        ss_rna = list(rna),        # Single-stranded RNA sequences
        ss_dna = list(dna),        # Single-stranded DNA sequences
    )

    # Load pre-trained model from checkpoint
    # init_and_load reconstructs model architecture and loads weights
    alphafold3 = Alphafold3.init_and_load(checkpoint_path)

    # Move model to GPU if requested and available
    if use_cuda and torch.cuda.is_available():
        alphafold3 = alphafold3.cuda()

    # Set model to evaluation mode (disables dropout, etc.)
    alphafold3.eval()

    # Run structure prediction
    # The model uses diffusion sampling to generate coordinates
    # return_bio_pdb_structures=True converts output to BioPython Structure format
    structure, = alphafold3.forward_with_alphafold3_inputs(
        alphafold3_input,
        return_bio_pdb_structures = True,
        num_sample_steps = num_sample_steps
    )

    # Ensure output directory exists
    output_path = Path(output)
    output_path.parents[0].mkdir(exist_ok = True, parents = True)

    # Write structure to mmCIF file format
    # mmCIF is the modern standard for macromolecular structures
    pdb_writer = MMCIFIO()
    pdb_writer.set_structure(structure)
    pdb_writer.save(str(output_path))

    print(f'mmCIF file saved to {str(output_path)}')
