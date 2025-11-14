"""Deoxyribonucleic acid (DNA) constants used in AlphaFold.

This module contains all the constants and mappings needed for working with DNA
in biomolecular structures. It defines:
- Standard atom types and their ordering for DNA nucleotides
- The 4 standard DNA bases (A, C, G, T) and their representations
- Mappings between 1-letter and 3-letter DNA codes
- Compact atom encodings for efficient storage
- MSA (Multiple Sequence Alignment) character mappings
- Biomolecule chain type identifiers

DNA differs from RNA by:
- Having deoxyribose sugar (lacking 2'-OH group)
- Using thymine (T) instead of uracil (U)
- Using "DA", "DC", "DG", "DT" as 3-letter codes

These constants follow the same structure as amino_acid_constants.py and rna_constants.py
but are specific to DNA nucleotides.

References:
    RCSB PDB DNA components: https://files.rcsb.org/ligands/view/DA.cif through DT.cif
"""

from beartype.typing import Final

import numpy as np

from alphafold3_pytorch.common import amino_acid_constants, rna_constants

# Atom types list for DNA residues
# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
# The list contains 28 standard atom types (phosphate backbone + deoxyribose sugar + bases)
# plus 19 null types (represented by "_") to create a fixed-size array of 47 atom types.
# This enables efficient vectorized operations on DNA structures.
# From: https://files.rcsb.org/ligands/view/DA.cif - https://files.rcsb.org/ligands/view/DT.cif
# Derived via: `list(dict.fromkeys([name for atom_names in dna_constants.restype_name_to_compact_atom_names.values() for name in atom_names if name]))`
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
    "O4",
    "C7",  # Methyl carbon in thymine
    "ATM",  # Catch-all atom type for non-standard or modified DNA residues
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

# Set of all valid DNA atom types for quick membership testing
atom_types_set = set(atom_types)

# Mapping from atom type name to its index in the atom_types list
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

# Total number of atom types (28 standard + 19 null = 47)
atom_type_num = len(atom_types)  # := 28 + 19 null types := 47.

# Index of the representative atom for each DNA residue (C1' = ribose C1 carbon)
# C1' is where the base attaches to the deoxyribose sugar
res_rep_atom_index = 11  # The index of the atom used to represent the center of the residue.


# Standard DNA nucleotide residues (1-letter codes)
# This is the standard residue order when coding DNA type as a number.
# Reproduce it by taking 3-letter DNA codes and sorting them alphabetically.
# Order: A (adenine), C (cytosine), G (guanine), T (thymine)
restypes = ["A", "C", "G", "T"]

# Minimum residue type number (starting index for DNA nucleotides)
# DNA follows amino acids (0-20) and RNA (21-25) in the numbering scheme
min_restype_num = (len(amino_acid_constants.restypes) + 1) + (
    len(rna_constants.restypes) + 1
)  # := 26.

# Mapping from 1-letter DNA code to its numeric index
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}

# Total number of standard DNA residue types (30 = 26 base + 4 DNA types)
restype_num = min_restype_num + len(restypes)  # := 26 + 4 := 30.


# Mapping from 1-letter to 3-letter DNA codes
# "D" prefix indicates deoxyribonucleotide (DNA vs RNA)
# "X" represents an unknown DNA nucleotide
restype_1to3 = {
    "A": "DA",
    "C": "DC",
    "G": "DG",
    "T": "DT",
    "X": "DN",  # Unknown DNA nucleotide
}

# MSA (Multiple Sequence Alignment) character to integer ID mapping for DNA
# DNA nucleotides are mapped to indices 26-30 (after amino acids 0-20 and RNA 21-25)
MSA_CHAR_TO_ID = {
    "A": 26,  # DNA adenine
    "C": 27,  # DNA cytosine
    "G": 28,  # DNA guanine
    "T": 29,  # DNA thymine
    "X": 30,  # Unknown DNA nucleotide
    "-": 31,  # Gap character in alignment
}

# Biomolecule chain type identifier for DNA chains
# "polydeoxyribonucleotide" indicates deoxyribose-based nucleic acid polymer (DNA)
BIOMOLECULE_CHAIN: Final[str] = "polydeoxyribonucleotide"

# General polymer chain type identifier
POLYMER_CHAIN: Final[str] = "polymer"


# Reverse mapping from 3-letter to 1-letter DNA codes
# NB: restype_3to1 differs from e.g., Bio.Data.PDBData.nucleic_letters_3to1
# by being a simple 1-to-1 mapping of 3 letter names to one letter names.
# The latter contains many more, and less common, three letter names as
# keys and maps many of these to the same one letter name.
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Unknown DNA residue metadata
# These values are used when encountering non-standard or modified DNA nucleotides
unk_restype = "DN"  # 3-letter code for unknown DNA residue
unk_chemtype = "DNA linking"  # Chemical type classification
unk_chemname = "UNKNOWN DNA RESIDUE"  # Full descriptive name

# List of all DNA residue names including the 4 standard bases plus unknown
resnames = [restype_1to3[r] for r in restypes] + [unk_restype]

# Chemical type index for DNA residues
# This represents the residue chemical type (i.e., `chemtype`) index of DNA residues.
# 0 = protein/peptide, 1 = RNA, 2 = DNA, 3 = ligand
chemtype_num = rna_constants.chemtype_num + 1  # := 2.

# Compact atom encoding with 24 columns for DNA residues
# This dictionary maps each DNA nucleotide's 3-letter code to an ordered list of its atoms.
# Each list has exactly 24 positions - empty strings represent positions where the
# nucleotide doesn't have an atom.
# DNA atoms include:
# - Phosphate group: OP3, P, OP1, OP2
# - Deoxyribose sugar: O5', C5', C4', O4', C3', O3', C2', C1' (no O2' vs RNA)
# - Nitrogenous base: varies by nucleotide (purines have more atoms than pyrimidines)
# From: https://files.rcsb.org/ligands/view/DA.cif - https://files.rcsb.org/ligands/view/DT.cif
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_compact_atom_names = {
    "DA": [
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
        "",
    ],
    "DC": [
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
        "",
    ],
    "DG": [
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
        "",
    ],
    "DT": [
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
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C7",
        "C6",
        "",
        "",
        "",
    ],
    "DN": [
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
        "",
    ],
}

# Mapping array from 47-atom representation to compact 24-atom representation
# Shape: [5 residue types (A, C, G, T, DN), 47 possible atom positions]
# For each residue type and atom position in the full 47-atom array,
# this gives the corresponding index in the compact 24-atom array.
restype_atom47_to_compact_atom = np.zeros([5, 47], dtype=int)


def _make_constants():
    """Initialize the atom mapping constants for DNA.

    This function populates the restype_atom47_to_compact_atom array, which maps
    from the full 47-atom representation to the compact 24-atom representation
    for each DNA nucleotide residue type.

    For each of the 5 residue types (A, C, G, T, DN/unknown):
        - Iterates through the compact atom names (up to 24 atoms)
        - Finds the corresponding index in the full 47-atom array
        - Stores this mapping for efficient conversion between representations

    This enables conversion between:
        - Full representation: 47 atom positions (standardized across all biomolecule types)
        - Compact representation: 24 atom positions (DNA-specific, minimal storage)
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
