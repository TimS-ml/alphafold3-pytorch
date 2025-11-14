"""Amino acid constants used in AlphaFold.

This module contains all the constants and mappings needed for working with amino acids
in protein structures. It defines:
- Standard atom types and their ordering for amino acid residues
- The 20 standard amino acids and their representations
- Mappings between 1-letter and 3-letter amino acid codes
- Compact atom encodings for efficient storage
- MSA (Multiple Sequence Alignment) character mappings
- Biomolecule chain type identifiers

These constants are derived from the original AlphaFold codebase and are used throughout
the AlphaFold3-PyTorch implementation for structure prediction and analysis.

References:
    AlphaFold repository: https://github.com/google-deepmind/alphafold
"""

from beartype.typing import Final

import numpy as np

# Atom types list for amino acid residues
# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
# The list contains 37 standard atom types plus 10 null types (represented by "_")
# to create a fixed-size array of 47 atom types per residue.
# This enables efficient vectorized operations on protein structures.
# From: https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/common/residue_constants.py#L492C1-L497C2
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    # "OXT",  # NOTE: This often appears in mmCIF files, but it will not be used for any amino acid type in AlphaFold.
    "ATM",  # NOTE: This represents a catch-all atom type for non-standard or modified residues.
    "_",  # Null type placeholder
    "_",  # Null type placeholder
    "_",  # Null type placeholder
    "_",  # Null type placeholder
    "_",  # Null type placeholder
    "_",  # Null type placeholder
    "_",  # Null type placeholder
    "_",  # Null type placeholder
    "_",  # Null type placeholder
    "_",  # 10th null type placeholder (total 10 null types)
]

# Element types derived from atom types
# For most atoms, this is the first character (e.g., "CA" -> "C")
# "ATM" is kept as-is to represent generic atoms
element_types = [atom_type if atom_type == "ATM" else atom_type[0] for atom_type in atom_types]

# Set of all valid atom types for quick membership testing
atom_types_set = set(atom_types)

# Mapping from atom type name to its index in the atom_types list
# This is used to quickly look up an atom's position in the fixed-size array
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

# Total number of atom types (37 standard + 10 null = 47)
atom_type_num = len(atom_types)  # := 37 + 10 null types := 47.

# Index of the representative atom for each residue (CA = alpha carbon)
# This is used to determine the center position of a residue
res_rep_atom_index = 1  # The index of the atom used to represent the center of the residue.


# Standard amino acid residues (1-letter codes)
# This is the standard residue order when coding AA type as a number.
# The order is alphabetical by 3-letter code (ALA, ARG, ASN, ..., VAL)
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

# Minimum residue type number (starting index for amino acids)
min_restype_num = 0  # := 0.

# Mapping from 1-letter amino acid code to its numeric index
# This is used for encoding sequences as integer arrays
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}

# Total number of standard amino acid residue types (20)
restype_num = min_restype_num + len(restypes)  # := 0 + 20 := 20.


# Mapping from 1-letter to 3-letter amino acid codes
# Example: "A" -> "ALA", "R" -> "ARG", etc.
# "X" represents an unknown amino acid
restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",  # Unknown amino acid
}

# MSA (Multiple Sequence Alignment) character to integer ID mapping
# This is used to encode MSA sequences where each character represents either:
# - A standard amino acid (A-V): mapped to indices 0-19
# - An unknown amino acid (X): mapped to index 20
# - A gap character (-): mapped to index 31
# - Alternative representations (U, Z, J, O): mapped to their biochemically similar amino acids
#   - U (Selenocysteine): mapped to 1 (like R/Arginine)
#   - Z (Glutamic acid/Glutamine): mapped to 3 (like D/Aspartic acid)
#   - J (Leucine/Isoleucine): mapped to 20 (unknown)
#   - O (Pyrrolysine): mapped to 20 (unknown)
MSA_CHAR_TO_ID = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    "X": 20,  # Unknown amino acid
    "-": 31,  # Gap character in alignment
    "U": 1,   # Selenocysteine (rare, mapped to similar residue)
    "Z": 3,   # Glutamic acid or Glutamine (ambiguous, mapped to Aspartic acid)
    "J": 20,  # Leucine or Isoleucine (ambiguous, mapped to unknown)
    "O": 20,  # Pyrrolysine (rare, mapped to unknown)
}

# Biomolecule chain type identifier for amino acid chains
# "polypeptide(L)" indicates L-amino acid polymer (standard proteins)
BIOMOLECULE_CHAIN: Final[str] = "polypeptide(L)"

# General polymer chain type identifier
POLYMER_CHAIN: Final[str] = "polymer"


# Reverse mapping from 3-letter to 1-letter amino acid codes
# NB: restype_3to1 differs from e.g., Bio.Data.PDBData.protein_letters_3to1
# by being a simple 1-to-1 mapping of 3 letter names to one letter names.
# The latter contains many more, and less common, three letter names as
# keys and maps many of these to the same one letter name
# (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Unknown amino acid residue metadata
# These values are used when encountering non-standard or modified amino acids
unk_restype = "UNK"  # 3-letter code for unknown residue
unk_chemtype = "peptide linking"  # Chemical type classification
unk_chemname = "UNKNOWN AMINO ACID RESIDUE"  # Full descriptive name

# List of all residue names including the 20 standard amino acids plus unknown
resnames = [restype_1to3[r] for r in restypes] + [unk_restype]

# Chemical type index for amino acid residues
# This represents the residue chemical type (i.e., `chemtype`) index of amino acid residues.
# 0 = protein/peptide, 1 = RNA, 2 = DNA, 3 = ligand
chemtype_num = 0

# Compact atom encoding with 14 columns for amino acid residues
# This dictionary maps each amino acid's 3-letter code to an ordered list of its atoms.
# Each list has exactly 14 positions - empty strings represent positions where the
# residue doesn't have an atom (e.g., GLY has no CB atom).
# This enables efficient storage and processing of protein structures in fixed-size arrays.
#
# The atom ordering follows standard PDB conventions:
# - N, CA, C, O are the backbone atoms (present in all residues)
# - CB is the first side-chain atom (absent only in GLY)
# - Remaining positions contain side-chain atoms specific to each residue
#
# From: https://github.com/google-deepmind/alphafold/blob/f251de6613cb478207c732bf9627b1e853c99c2f/alphafold/common/residue_constants.py#L505
# Also matches: https://files.rcsb.org/ligands/view/ALA.cif - https://files.rcsb.org/ligands/view/VAL.cif
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_compact_atom_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],  # Unknown residue (minimal atoms)
}

# Mapping array from 47-atom representation to compact 14-atom representation
# Shape: [21 residue types, 47 possible atom positions]
# For each residue type and atom position in the full 47-atom array,
# this gives the corresponding index in the compact 14-atom array.
# Initialized with zeros; filled by _make_constants() below.
restype_atom47_to_compact_atom = np.zeros([21, 47], dtype=int)


def _make_constants():
    """Initialize the atom mapping constants.

    This function populates the restype_atom47_to_compact_atom array, which maps
    from the full 47-atom representation to the compact 14-atom representation
    for each amino acid residue type.

    For each of the 21 residue types (20 standard amino acids + unknown):
        - Iterates through the compact atom names (up to 14 atoms)
        - Finds the corresponding index in the full 47-atom array
        - Stores this mapping for efficient conversion between representations

    This enables conversion between:
        - Full representation: 47 atom positions (many empty for small residues)
        - Compact representation: 14 atom positions (minimal storage)
    """
    for restype, restype_letter in enumerate(restype_1to3.keys()):
        resname = restype_1to3[restype_letter]
        for compact_atomidx, atomname in enumerate(restype_name_to_compact_atom_names[resname]):
            if not atomname:
                continue
            atomtype = atom_order[atomname]
            restype_atom47_to_compact_atom[restype, atomtype] = compact_atomidx


# Initialize the constants immediately upon module import
_make_constants()
