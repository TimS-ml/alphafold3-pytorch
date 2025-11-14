"""Ligand constants used in AlphaFold.

This module contains all the constants and mappings needed for working with small molecule
ligands and non-polymer residues in biomolecular structures. It defines:
- Standard atom types covering common elements in ligands
- A simplified residue representation (all ligands mapped to "X"/"UNL")
- Compact atom encodings using all 47 atom positions
- MSA character mappings
- Biomolecule chain type identifiers

Ligands differ from proteins, RNA, and DNA by:
- Being non-polymeric (not connected in a chain)
- Having variable atom compositions
- All being mapped to a single residue type index (20, same as unknown amino acid)
- Using element symbols as atom types rather than PDB atom names

These constants enable AlphaFold 3 to handle diverse small molecules including:
- Cofactors (ATP, NAD, heme)
- Metabolites
- Drug molecules
- Metal ions and clusters
- Covalently modified residues

References:
    RoseTTAFold-All-Atom: https://github.com/baker-laboratory/RoseTTAFold-All-Atom
"""

from beartype.typing import Final

import numpy as np

from alphafold3_pytorch.common import amino_acid_constants, dna_constants

# Atom types list for ligand residues
# This mapping uses chemical element symbols rather than PDB atom names.
# It covers the most common elements found in biological ligands and cofactors.
# The list contains 47 element symbols (including metals and a catch-all "ATM")
# to match the standardized 47-atom array size used across all biomolecule types.
# NOTE: Taken from: https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L117C13-L126C18
atom_types = [
    "AL",
    "AS",
    "AU",
    "B",
    "BE",
    "BR",
    "C",
    "CA",
    "CL",
    "CO",
    "CR",
    "CU",
    "F",
    "FE",
    "HG",
    "I",
    "IR",
    "K",
    "LI",
    "MG",
    "MN",
    "MO",
    "N",
    "NI",
    "O",
    "OS",
    "P",
    "PB",
    "PD",
    "PR",
    "PT",
    "RE",
    "RH",
    "RU",
    "S",
    "SB",
    "SE",
    "SI",
    "SN",
    "TB",
    "TE",
    "U",
    "W",
    "V",
    "Y",
    "ZN",  # Zinc - common metal in metalloproteins
    "ATM",  # Catch-all atom type for any element not explicitly listed
]

# Element types for ligands (same as atom_types but with proper capitalization)
# This list provides the standard chemical element symbols with proper case:
# - First letter capitalized (e.g., "Ca" for calcium)
# - Second letter lowercase for two-letter symbols
# Used for mmCIF output and chemical validation
# NOTE: Taken from: https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L117C13-L126C18
element_types = [
    "Al",
    "As",
    "Au",
    "B",
    "Be",
    "Br",
    "C",
    "Ca",
    "Cl",
    "Co",
    "Cr",
    "Cu",
    "F",
    "Fe",
    "Hg",
    "I",
    "Ir",
    "K",
    "Li",
    "Mg",
    "Mn",
    "Mo",
    "N",
    "Ni",
    "O",
    "Os",
    "P",
    "Pb",
    "Pd",
    "Pr",
    "Pt",
    "Re",
    "Rh",
    "Ru",
    "S",
    "Sb",
    "Se",
    "Si",
    "Sn",
    "Tb",
    "Te",
    "U",
    "W",
    "V",
    "Y",
    "Zn",  # Zinc
    "ATM",  # Catch-all for any element
]

# Set of all valid ligand atom types for quick membership testing
atom_types_set = set(atom_types)

# Mapping from atom type (element symbol) to its index in the atom_types list
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

# Total number of atom types for ligands (exactly 47 to match other biomolecule types)
atom_type_num = len(atom_types)  # := 47.

# Index of the representative atom for each ligand "pseudoresidue"
# Set to the last position (ATM), but in practice each ligand atom is its own "pseudoresidue"
res_rep_atom_index = (
    len(atom_types) - 1
)  # := 46  # The index of the atom used to represent the center of a ligand pseudoresidue.


# Ligand residue type
# All ligand residues are mapped to the unknown amino acid type index (:= 20).
# This means ligands share the same residue type space as unknown amino acids.
restypes = ["X"]  # Single residue type for all ligands

# Minimum residue type number (same as unknown amino acid)
min_restype_num = len(amino_acid_constants.restypes)  # := 20.

# Mapping from ligand "residue type" to its numeric index
restype_order = {restype: min_restype_num + i for i, restype in enumerate(restypes)}

# Total residue type number (ligands don't add new residue types beyond amino acids)
restype_num = len(amino_acid_constants.restypes)  # := 20.


# Mapping from 1-letter to 3-letter ligand codes
# "X" (unknown) maps to "UNL" (unknown ligand)
restype_1to3 = {"X": "UNL"}

# MSA character to integer ID mapping for ligands
# Ligands are treated as unknown ("X") in alignments
MSA_CHAR_TO_ID = {
    "X": 20,  # Unknown/ligand
    "-": 31,  # Gap character in alignment
}

# Biomolecule chain type identifier for ligand chains
# "other" indicates this is not a standard polymer type
BIOMOLECULE_CHAIN: Final[str] = "other"

# Polymer classification for ligands
# "non-polymer" indicates ligands are not polymeric chains
POLYMER_CHAIN: Final[str] = "non-polymer"


# Reverse mapping from 3-letter to 1-letter ligand codes
# NB: restype_3to1 serves as a placeholder for mapping all
# ligand residues to the unknown amino acid type index (:= 20).
# Empty dict because ligands use their actual CCD codes, not standard 3-letter codes
restype_3to1 = {}

# Unknown ligand residue metadata
# These values are used for all ligand residues
unk_restype = "UNL"  # 3-letter code for unknown ligand
unk_chemtype = "non-polymer"  # Chemical type classification
unk_chemname = "UNKNOWN LIGAND RESIDUE"  # Full descriptive name

# Chemical type index for ligand residues
# This represents the residue chemical type (i.e., `chemtype`) index of ligand residues.
# 0 = protein/peptide, 1 = RNA, 2 = DNA, 3 = ligand
chemtype_num = dna_constants.chemtype_num + 1  # := 3.

# Compact atom encoding with 47 columns for ligand residues
# For ligands, all 47 atom positions are available since we use element symbols.
# Unlike proteins/nucleic acids, there's no fixed atom layout - each position
# can hold any element type, and the actual composition varies per ligand.
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_compact_atom_names = {
    "UNL": atom_types,  # All 47 element types available
}

# Mapping array from 47-atom representation to compact representation
# Shape: [1 residue type (UNL/unknown ligand), 47 possible atom positions]
# For ligands, the mapping is identity since we use all 47 positions
restype_atom47_to_compact_atom = np.zeros([1, 47], dtype=int)


def _make_constants():
    """Initialize the atom mapping constants for ligands.

    This function populates the restype_atom47_to_compact_atom array.
    For ligands, this is a simple identity mapping since all 47 atom positions
    are used (one for each element type).

    The mapping allows ligands to integrate seamlessly with the same data structures
    used for proteins and nucleic acids, even though ligands have fundamentally
    different atom naming (elements vs. PDB atom names) and structure (non-polymeric).
    """
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for atomname in restype_name_to_compact_atom_names[resname]:
            if not atomname:
                continue
            atomtype = atom_order[atomname]
            compact_atom_idx = restype_name_to_compact_atom_names[resname].index(atomname)
            restype_atom47_to_compact_atom[restype, atomtype] = compact_atom_idx


# Initialize the constants immediately upon module import
_make_constants()
