"""Ribonucleic acid (RNA) constants used in AlphaFold.

This module contains all the constants and mappings needed for working with RNA
in biomolecular structures. It defines:
- Standard atom types and their ordering for RNA nucleotides
- The 4 standard RNA bases (A, C, G, U) and their representations
- Mappings between 1-letter and 3-letter RNA codes
- Compact atom encodings for efficient storage
- MSA (Multiple Sequence Alignment) character mappings
- Biomolecule chain type identifiers

RNA differs from DNA by:
- Having ribose sugar (with 2'-OH group)
- Using uracil (U) instead of thymine (T)
- Using single-letter codes (A, C, G, U) as 3-letter codes

These constants follow the same structure as amino_acid_constants.py and dna_constants.py
but are specific to RNA nucleotides.

References:
    RCSB PDB RNA components: https://files.rcsb.org/ligands/view/A.cif through U.cif
"""

from beartype.typing import Final

import numpy as np

from alphafold3_pytorch.common import amino_acid_constants

# Atom types list for RNA residues
# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
# The list contains 28 standard atom types (phosphate backbone + ribose sugar + bases)
# plus 19 null types (represented by "_") to create a fixed-size array of 47 atom types.
# This enables efficient vectorized operations on RNA structures.
# Note: RNA has O2' (2'-hydroxyl) which DNA lacks
# From: https://files.rcsb.org/ligands/view/A.cif - https://files.rcsb.org/ligands/view/U.cif
# Derived via: `list(dict.fromkeys([name for atom_names in rna_constants.restype_name_to_compact_atom_names.values() for name in atom_names if name]))`
atom_types = [
    "OP3",
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    "N9",
    "C8",
    "N7",
    "C5",
    "C6",
    "N6",
    "N1",
    "C2",
    "N3",
    "C4",
    "O2",
    "N4",
    "O6",
    "N2",
    "O4",  # Carbonyl oxygen in uracil
    "ATM",  # Catch-all atom type for non-standard or modified RNA residues
    "_",  # Null type placeholder
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",
    "_",  # 19th null type placeholder (total 19 null types)
]

# Element types derived from atom types (first character, except "ATM")
element_types = [atom_type if atom_type == "ATM" else atom_type[0] for atom_type in atom_types]

# Set of all valid RNA atom types for quick membership testing
atom_types_set = set(atom_types)

# Mapping from atom type name to its index in the atom_types list
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

# Total number of atom types (28 standard + 19 null = 47)
atom_type_num = len(atom_types)  # := 28 + 19 null types := 47.

# Index of the representative atom for each RNA residue (C1' = ribose C1 carbon)
# C1' is where the base attaches to the ribose sugar
res_rep_atom_index = 12  # The index of the atom used to represent the center of the residue.


# Standard RNA nucleotide residues (1-letter codes)
# This is the standard residue order when coding RNA type as a number.
# Reproduce it by taking 3-letter RNA codes and sorting them alphabetically.
# Order: A (adenine), C (cytosine), G (guanine), U (uracil)
restypes = ["A", "C", "G", "U"]

# Minimum residue type number (starting index for RNA nucleotides)
# RNA follows amino acids (0-20) in the numbering scheme
min_restype_num = len(amino_acid_constants.restypes) + 1  # := 21.

# Mapping from 1-letter RNA code to its numeric index
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}

# Total number of standard RNA residue types (25 = 21 base + 4 RNA types)
restype_num = min_restype_num + len(restypes)  # := 21 + 4 := 25.


# Mapping from 1-letter to 3-letter RNA codes
# Unlike DNA, RNA uses the same 1-letter codes as 3-letter codes (A, C, G, U)
# "X" represents an unknown RNA nucleotide, mapped to "N"
restype_1to3 = {"A": "A", "C": "C", "G": "G", "U": "U", "X": "N"}

# MSA (Multiple Sequence Alignment) character to integer ID mapping for RNA
# RNA nucleotides are mapped to indices 21-25 (after amino acids 0-20)
MSA_CHAR_TO_ID = {
    "A": 21,  # RNA adenine
    "C": 22,  # RNA cytosine
    "G": 23,  # RNA guanine
    "U": 24,  # RNA uracil
    "X": 25,  # Unknown RNA nucleotide
    "-": 31,  # Gap character in alignment
}

# Biomolecule chain type identifier for RNA chains
# "polyribonucleotide" indicates ribose-based nucleic acid polymer (RNA)
BIOMOLECULE_CHAIN: Final[str] = "polyribonucleotide"

# General polymer chain type identifier
POLYMER_CHAIN: Final[str] = "polymer"

# Reverse mapping from 3-letter to 1-letter RNA codes
# NB: restype_3to1 differs from e.g., Bio.Data.PDBData.nucleic_letters_3to1
# by being a simple 1-to-1 mapping of 3 letter names to one letter names.
# The latter contains many more, and less common, three letter names as
# keys and maps many of these to the same one letter name.
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Unknown RNA residue metadata
# These values are used when encountering non-standard or modified RNA nucleotides
unk_restype = "N"  # 3-letter code for unknown RNA residue
unk_chemtype = "RNA linking"  # Chemical type classification
unk_chemname = "UNKNOWN RNA RESIDUE"  # Full descriptive name

# List of all RNA residue names including the 4 standard bases plus unknown
resnames = [restype_1to3[r] for r in restypes] + [unk_restype]

# Chemical type index for RNA residues
# This represents the residue chemical type (i.e., `chemtype`) index of RNA residues.
# 0 = protein/peptide, 1 = RNA, 2 = DNA, 3 = ligand
chemtype_num = amino_acid_constants.chemtype_num + 1  # := 1.

# Compact atom encoding with 24 columns for RNA residues
# This dictionary maps each RNA nucleotide's 3-letter code to an ordered list of its atoms.
# Each list has exactly 24 positions - empty strings represent positions where the
# nucleotide doesn't have an atom.
# RNA atoms include:
# - Phosphate group: OP3, P, OP1, OP2
# - Ribose sugar: O5', C5', C4', O4', C3', O3', C2', O2', C1' (O2' present vs DNA)
# - Nitrogenous base: varies by nucleotide (purines have more atoms than pyrimidines)
# From: https://files.rcsb.org/ligands/view/A.cif - https://files.rcsb.org/ligands/view/U.cif
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_compact_atom_names = {
    "A": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
        "",
    ],
    "C": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
        "",
        "",
        "",
    ],
    "G": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],
    "U": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C6",
        "",
        "",
        "",
    ],
    "N": [
        "OP3",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
        "",
    ],
}

# Mapping array from 47-atom representation to compact 24-atom representation
# Shape: [5 residue types (A, C, G, U, N), 47 possible atom positions]
# For each residue type and atom position in the full 47-atom array,
# this gives the corresponding index in the compact 24-atom array.
restype_atom47_to_compact_atom = np.zeros([5, 47], dtype=int)


def _make_constants():
    """Initialize the atom mapping constants for RNA.

    This function populates the restype_atom47_to_compact_atom array, which maps
    from the full 47-atom representation to the compact 24-atom representation
    for each RNA nucleotide residue type.

    For each of the 5 residue types (A, C, G, U, N/unknown):
        - Iterates through the compact atom names (up to 24 atoms)
        - Finds the corresponding index in the full 47-atom array
        - Stores this mapping for efficient conversion between representations

    This enables conversion between:
        - Full representation: 47 atom positions (standardized across all biomolecule types)
        - Compact representation: 24 atom positions (RNA-specific, minimal storage)
    """
    for restype, restype_letter in enumerate(restype_1to3.keys()):
        resname = restype_1to3[restype_letter]
        for atomname in restype_name_to_compact_atom_names[resname]:
            if not atomname:
                continue
            atomtype = atom_order[atomname]
            compact_atom_idx = restype_name_to_compact_atom_names[resname].index(atomname)
            restype_atom47_to_compact_atom[restype, atomtype] = compact_atom_idx


# Initialize the constants immediately upon module import
_make_constants()
