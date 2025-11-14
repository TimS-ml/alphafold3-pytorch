"""
Mock Datasets Module for AlphaFold3.

This module provides mock dataset classes for testing and development purposes.
These datasets generate synthetic data that matches the expected format for
training AlphaFold3, allowing for rapid prototyping and testing without
requiring real protein structure data.

The main components are:
- MockAtomDataset: A PyTorch Dataset that generates random AtomInput samples
                   with realistic structure and constraints
"""

import random

import torch
from torch.utils.data import Dataset
from alphafold3_pytorch import AtomInput

from alphafold3_pytorch.inputs import (
    IS_MOLECULE_TYPES,
    DEFAULT_NUM_MOLECULE_MODS
)

from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

# Mock dataset classes

class MockAtomDataset(Dataset):
    """
    A mock dataset that generates random AtomInput samples for testing.

    This dataset is useful for:
    - Testing AlphaFold3 training loops without real data
    - Debugging model architectures
    - Rapid prototyping of new features
    - Unit testing

    Each sample contains randomly generated but structurally valid data including:
    - Atom and atompair features
    - Molecule-level features and types
    - MSA (Multiple Sequence Alignment) data
    - Template features
    - Ground truth labels for training

    Attributes:
        data_length: Number of samples in the dataset.
        max_seq_len: Maximum sequence length (number of tokens/molecules).
        atoms_per_window: Number of atoms per windowed attention window.
        dim_atom_inputs: Dimensionality of atom-level input features.
        has_molecule_mods: Whether to include molecule modification flags.
    """

    def __init__(
        self,
        data_length,
        max_seq_len = 16,
        atoms_per_window = 4,
        dim_atom_inputs = 77,
        has_molecule_mods = True
    ):
        """
        Initialize the mock atom dataset.

        Args:
            data_length: Total number of samples the dataset should contain.
            max_seq_len: Maximum sequence length (in tokens). Each sample will
                        have a random length between 1 and max_seq_len.
            atoms_per_window: Number of atoms per window for windowed attention.
            dim_atom_inputs: Dimensionality of atom-level feature vectors.
            has_molecule_mods: Whether to generate molecule modification data.
        """
        self.data_length = data_length
        self.max_seq_len = max_seq_len
        self.atoms_per_window = atoms_per_window
        self.dim_atom_inputs = dim_atom_inputs
        self.has_molecule_mods = has_molecule_mods

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.data_length

    def __getitem__(self, idx):
        """
        Generate a random AtomInput sample.

        Args:
            idx: Index of the sample (unused, but required by PyTorch Dataset).

        Returns:
            A randomly generated AtomInput with all required fields populated.

        Note:
            The generated data is random but structurally consistent, e.g.,
            molecule-atom length mappings are properly aligned with atom counts.
        """
        # Generate a random sequence length for this sample
        seq_len = random.randrange(1, self.max_seq_len)
        atom_seq_len = self.atoms_per_window * seq_len

        # Generate random atom and atompair features
        atom_inputs = torch.randn(atom_seq_len, self.dim_atom_inputs)
        atompair_inputs = torch.randn(atom_seq_len, atom_seq_len, 5)

        # Generate molecule-level atom counts and compute offsets
        molecule_atom_lens = torch.randint(1, self.atoms_per_window, (seq_len,))
        atom_offsets = exclusive_cumsum(molecule_atom_lens)

        # Generate additional molecular and token features
        additional_molecule_feats = torch.randint(0, 2, (seq_len, 5))
        additional_token_feats = torch.randn(seq_len, 33)
        is_molecule_types = torch.randint(0, 2, (seq_len, IS_MOLECULE_TYPES)).bool()

        # Ensure the molecule-atom length mappings match the total atom sequence length
        if molecule_atom_lens.sum() < atom_seq_len:
            molecule_atom_lens[-1] = atom_seq_len - molecule_atom_lens[:-1].sum()

        # Ensure each unique asymmetric ID has at least one molecule type
        # This maintains biological validity (each chain must have a molecule type)
        asym_id = additional_molecule_feats[:, 2]
        unique_asym_id = asym_id.unique()
        for asym in unique_asym_id:
            if any(not row.any() for row in is_molecule_types[asym_id == asym]):
                rand_molecule_type_idx = random.randint(0, IS_MOLECULE_TYPES - 1)  # nosec
                is_molecule_types[asym_id == asym, rand_molecule_type_idx] = True

        # Generate molecule modification flags if requested
        is_molecule_mod = None
        if self.has_molecule_mods:
            # 5% chance of modification per molecule per modification type
            is_molecule_mod = torch.rand((seq_len, DEFAULT_NUM_MOLECULE_MODS)) < 0.05

        # Generate molecule IDs and token bonds
        molecule_ids = torch.randint(0, 32, (seq_len,))
        token_bonds = torch.randint(0, 2, (seq_len, seq_len)).bool()

        # Generate template features (2 templates with 108-dimensional features)
        templates = torch.randn(2, seq_len, seq_len, 108)
        template_mask = torch.ones((2,)).bool()

        # Generate MSA (Multiple Sequence Alignment) features
        msa = torch.randn(7, seq_len, 32)

        # Randomly include or exclude MSA mask (50% chance)
        msa_mask = None
        if random.random() > 0.5:
            msa_mask = torch.ones((7,)).bool()

        additional_msa_feats = torch.randn(7, seq_len, 2)

        # Generate ground truth labels (required for training)

        # Atom positions in 3D space
        atom_pos = torch.randn(atom_seq_len, 3)

        # Indices pointing to specific atoms for each molecule
        molecule_atom_indices = molecule_atom_lens - 1
        distogram_atom_indices = molecule_atom_lens - 1

        # Apply offsets to make indices global rather than local to each molecule
        molecule_atom_indices += atom_offsets
        distogram_atom_indices += atom_offsets

        # Distance labels for distogram prediction (64 bins)
        distance_labels = torch.randint(0, 64, (seq_len, seq_len))
        # Binary labels for whether each atom is resolved
        resolved_labels = torch.randint(0, 2, (atom_seq_len,))

        # Chain information - use the most common asymmetric ID
        majority_asym_id = asym_id.mode().values.item()
        chains = torch.tensor([majority_asym_id, -1]).long()

        # Return a complete AtomInput dataclass
        return AtomInput(
            atom_inputs = atom_inputs,
            atompair_inputs = atompair_inputs,
            molecule_ids = molecule_ids,
            token_bonds = token_bonds,
            molecule_atom_lens = molecule_atom_lens,
            additional_molecule_feats = additional_molecule_feats,
            additional_msa_feats = additional_msa_feats,
            additional_token_feats = additional_token_feats,
            is_molecule_types = is_molecule_types,
            is_molecule_mod = is_molecule_mod,
            templates = templates,
            template_mask = template_mask,
            msa = msa,
            msa_mask = msa_mask,
            atom_pos = atom_pos,
            molecule_atom_indices = molecule_atom_indices,
            distogram_atom_indices = distogram_atom_indices,
            distance_labels = distance_labels,
            resolved_labels = resolved_labels,
            chains = chains
        )
