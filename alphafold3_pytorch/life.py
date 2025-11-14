"""
Biomolecular Building Blocks and Chemistry Module.

This module defines the fundamental building blocks of life used in AlphaFold3:
- Amino acids (proteins)
- Nucleotides (DNA and RNA)
- Metal ions
- Ligands and other molecules

Each biomolecule is defined with:
- SMILES representation for chemical structure
- Atom indices for key structural points (N-terminus, C-terminus, distogram atoms, etc.)
- RDKit molecule objects for 3D structure manipulation

The module also provides utilities for:
- Converting SMILES strings to 3D molecular structures
- Loading molecules from mmCIF template files
- Computing reverse complements of nucleic acid sequences
- Bond type definitions and atom type registries

These definitions are critical for AlphaFold3's atom-level modeling approach,
which requires explicit 3D coordinates and chemical information for all atoms.
"""

import os
from beartype.typing import Literal

import gemmi
import rdkit.Geometry.rdGeometry as rdGeometry
import torch
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol

from alphafold3_pytorch.tensor_typing import Int, typecheck

# Helper functions


def exists(v):
    """
    Check if a value is not None.

    Args:
        v: Any value to check.

    Returns:
        True if the value is not None, False otherwise.
    """
    return v is not None

def is_unique(arr):
    """
    Check if all elements in an array are unique.

    Args:
        arr: A list or array-like object.

    Returns:
        True if all elements are unique, False otherwise.
    """
    return len(arr) == len({*arr})


# Human amino acids

# NOTE: Template SMILES were derived using non-canonical RDKit SMILES generation
# to guarantee the order and quantity of atoms in the SMILES string perfectly
# matches the atoms in the residue template structure files.
#
# Each amino acid entry contains:
# - resname: Three-letter PDB residue name
# - smile: SMILES string representation
# - first_atom_idx: Index of the N-terminus nitrogen atom (for peptide bonds)
# - last_atom_idx: Index of the C-terminus carbon atom (for peptide bonds)
# - distogram_atom_idx: Index of the beta carbon (used for distance predictions)
# - token_center_atom_idx: Index of the alpha carbon (center of the residue)
# - three_atom_indices_for_frame: Indices of N, CA, C for backbone frame calculation

HUMAN_AMINO_ACIDS = dict(
    A=dict(
        resname="ALA",
        smile="NC(C=O)C",
        first_atom_idx=0,
        last_atom_idx=4,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    R=dict(
        resname="ARG",
        smile="NC(C=O)CCCNC(N)=N",
        first_atom_idx=0,
        last_atom_idx=10,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    N=dict(
        resname="ASN",
        smile="NC(C=O)CC(=O)N",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    D=dict(
        resname="ASP",
        smile="NC(C=O)CC(=O)O",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    C=dict(
        resname="CYS",
        smile="NC(C=O)CS",
        first_atom_idx=0,
        last_atom_idx=5,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    Q=dict(
        resname="GLN",
        smile="NC(C=O)CCC(=O)N",
        first_atom_idx=0,
        last_atom_idx=8,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    E=dict(
        resname="GLU",
        smile="NC(C=O)CCC(=O)O",
        first_atom_idx=0,
        last_atom_idx=8,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    G=dict(
        resname="GLY",
        smile="NCC=O",
        first_atom_idx=0,
        last_atom_idx=3,
        distogram_atom_idx=1,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    H=dict(
        resname="HIS",
        smile="NC(C=O)CC1=CNC=N1",
        first_atom_idx=0,
        last_atom_idx=9,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    I=dict(
        resname="ILE",
        smile="NC(C=O)C(CC)C",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    L=dict(
        resname="LEU",
        smile="NC(C=O)CC(C)C",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    K=dict(
        resname="LYS",
        smile="NC(C=O)CCCCN",
        first_atom_idx=0,
        last_atom_idx=8,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    M=dict(
        resname="MET",
        smile="NC(C=O)CCSC",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    F=dict(
        resname="PHE",
        smile="NC(C=O)CC1=CC=CC=C1",
        first_atom_idx=0,
        last_atom_idx=10,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    P=dict(
        resname="PRO",
        smile="N1C(C=O)CCC1",
        first_atom_idx=0,
        last_atom_idx=6,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    S=dict(
        resname="SER",
        smile="NC(C=O)CO",
        first_atom_idx=0,
        last_atom_idx=5,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    T=dict(
        resname="THR",
        smile="NC(C=O)C(O)C",
        first_atom_idx=0,
        last_atom_idx=6,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    W=dict(
        resname="TRP",
        smile="NC(C=O)CC1=CNC2=C1C=CC=C2",
        first_atom_idx=0,
        last_atom_idx=13,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    Y=dict(
        resname="TYR",
        smile="NC(C=O)CC1=CC=C(O)C=C1",
        first_atom_idx=0,
        last_atom_idx=11,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    V=dict(
        resname="VAL",
        smile="NC(C=O)C(C)C",
        first_atom_idx=0,
        last_atom_idx=6,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    X=dict(
        resname="UNK",
        smile="NC(C=O)C",
        first_atom_idx=0,
        last_atom_idx=4,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=None,
    ),
)

# Nucleotides

# DNA nucleotides with phosphate, deoxyribose sugar, and nitrogenous bases.
# Each entry includes complement information for base-pairing (A-T, G-C).
# The distogram_atom_idx points to the last atom in the base for distance calculations.
# The token_center_atom_idx points to a central atom in the sugar ring.
# three_atom_indices_for_frame provides atoms for calculating the nucleotide's local frame.

DNA_NUCLEOTIDES = dict(
    A=dict(
        resname="DA",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)CC1O",
        first_atom_idx=0,
        last_atom_idx=21,
        complement="T",
        distogram_atom_idx=21,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=(11, 8, 6),
    ),
    C=dict(
        resname="DC",
        smile="OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1O",
        first_atom_idx=0,
        last_atom_idx=19,
        complement="G",
        distogram_atom_idx=13,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=(11, 8, 6),
    ),
    G=dict(
        resname="DG",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1O",
        first_atom_idx=0,
        last_atom_idx=22,
        complement="C",
        distogram_atom_idx=22,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=(11, 8, 6),
    ),
    T=dict(
        resname="DT",
        smile="OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1O",
        first_atom_idx=0,
        last_atom_idx=20,
        complement="A",
        distogram_atom_idx=13,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=(11, 8, 6),
    ),
    X=dict(
        resname="DN",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)CC1O",
        first_atom_idx=0,
        last_atom_idx=21,
        complement="N",
        distogram_atom_idx=21,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=None,
    ),
)

# RNA nucleotides with phosphate, ribose sugar, and nitrogenous bases.
# Similar structure to DNA but with ribose (has 2'-OH) instead of deoxyribose,
# and uracil (U) instead of thymine (T).

RNA_NUCLEOTIDES = dict(
    A=dict(
        resname="A",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=22,
        complement="U",
        distogram_atom_idx=22,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=(12, 8, 6),
    ),
    C=dict(
        resname="C",
        smile="OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=20,
        complement="G",
        distogram_atom_idx=14,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=(12, 8, 6),
    ),
    G=dict(
        resname="G",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=23,
        complement="C",
        distogram_atom_idx=23,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=(12, 8, 6),
    ),
    U=dict(
        resname="U",
        smile="OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C=C2)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=20,
        complement="A",
        distogram_atom_idx=14,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=(12, 8, 6),
    ),
    X=dict(
        resname="N",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=22,
        complement="N",
        distogram_atom_idx=22,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=None,
    ),
)

# Ligands

# Generic unknown ligand placeholder.
# For actual ligands, SMILES should be provided dynamically.
# The single dot "." represents an empty/unknown molecule in SMILES notation.

LIGANDS = dict(
    X=dict(
        resname="UNK",
        smile=".",  # Empty molecule placeholder
        first_atom_idx=0,
        last_atom_idx=0,
        distogram_atom_idx=0,
        token_center_atom_idx=0,
        three_atom_indices_for_frame=None,
    )
)

# Nucleic acid complement lookup tensor

# Maps nucleotide indices to their Watson-Crick complements.
# Order follows ACG(T|U)N where:
# - Index 0 (A) -> Index 3 (T/U)
# - Index 1 (C) -> Index 2 (G)
# - Index 2 (G) -> Index 1 (C)
# - Index 3 (T/U) -> Index 0 (A)
# - Index 4 (N) -> Index 4 (N) (unknown complements to unknown)
NUCLEIC_ACID_COMPLEMENT_TENSOR = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)

# Nucleic acid utility functions


@typecheck
def reverse_complement(seq: str, nucleic_acid_type: Literal["dna", "rna"] = "dna"):
    """
    Get the reverse complement of a nucleic acid sequence.

    This is essential for DNA/RNA double helix modeling, as the two strands
    run antiparallel with complementary base pairing (A-T/U, G-C).

    Args:
        seq: A nucleotide sequence string (e.g., "ATCG" for DNA or "AUCG" for RNA).
        nucleic_acid_type: Type of nucleic acid, either "dna" or "rna".

    Returns:
        The reverse complement sequence as a string.

    Example:
        >>> reverse_complement("ATCG", "dna")
        "CGAT"
        >>> reverse_complement("AUCG", "rna")
        "CGAU"

    Raises:
        AssertionError: If the sequence contains unknown nucleotides.
    """
    if nucleic_acid_type == "dna":
        nucleic_acid_entries = DNA_NUCLEOTIDES
    elif nucleic_acid_type == "rna":
        nucleic_acid_entries = RNA_NUCLEOTIDES

    assert all(
        [nuc in nucleic_acid_entries for nuc in seq]
    ), "unknown nucleotide for given nucleic acid type"

    complement = [nucleic_acid_entries[nuc]["complement"] for nuc in seq]
    return "".join(complement[::-1])


@typecheck
def reverse_complement_tensor(t: Int[" n"]):  # type: ignore
    """
    Get the reverse complement of a nucleic acid sequence represented as indices.

    This tensor-based version is more efficient for batch processing in neural networks.

    Args:
        t: A tensor of nucleotide indices where each index corresponds to A, C, G, T/U, or N.

    Returns:
        A tensor containing the reverse complement sequence.

    Example:
        >>> t = torch.tensor([0, 1, 2, 3])  # ACGT
        >>> reverse_complement_tensor(t)
        tensor([0, 2, 1, 3])  # ACGT -> CGAT (reversed: TACG)
    """
    complement = NUCLEIC_ACID_COMPLEMENT_TENSOR[t]
    reverse_complement = complement.flip(dims=(-1,))
    return reverse_complement


# Metal ions

# Common metal ions found in biological structures.
# These often serve as cofactors in enzymes or stabilize protein structures.
# SMILES notation includes formal charges (e.g., +2, +3).

METALS = dict(
    Mg=dict(resname="Mg", smile="[Mg+2]"),
    Mn=dict(resname="Mn", smile="[Mn+2]"),
    Fe=dict(resname="Fe", smile="[Fe+3]"),
    Co=dict(resname="Co", smile="[Co+2]"),
    Ni=dict(resname="Ni", smile="[Ni+2]"),
    Cu=dict(resname="Cu", smile="[Cu+2]"),
    Zn=dict(resname="Zn", smile="[Zn+2]"),
    Na=dict(resname="Na", smile="[Na+]"),
    Cl=dict(resname="Cl", smile="[Cl-]"),
    Ca=dict(resname="Ca", smile="[Ca+2]"),
    K=dict(resname="K", smile="[K+]"),
)

# Miscellaneous molecules

# Other biologically relevant molecules that don't fit standard categories.
# Example: phospholipids for membrane modeling.

MISC = dict(
    Phospholipid=dict(
        resname="UNL",  # Generic "unknown ligand" residue name
        smile="CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)O)OC(=O)CCCCCCCC1CC1CCCCCC"
    )
)

# Atom types for embeddings

# Defines the vocabulary of atom types used in atom-level embeddings.
# Covers the most common biological elements plus all defined metal ions.
ATOMS = ["C", "O", "N", "S", "P", *METALS]

assert is_unique(ATOMS), "Atom types must be unique"

# Bond types for embeddings

# Standard chemical bond types used in molecular structures.
# These are used to create bond-aware atompair embeddings.
ATOM_BONDS = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# Mapping from PDB mmCIF bond notation to RDKit bond types
BOND_ORDER = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "AROM": Chem.BondType.AROMATIC,
}

assert is_unique(ATOM_BONDS), "Bond types must be unique"

# RDKit molecular structure utilities


@typecheck
def generate_conformation(mol: Mol) -> Mol:
    """
    Generate a 3D conformation for a molecule using RDKit's embedding algorithm.

    This function:
    1. Adds hydrogen atoms (needed for accurate 3D structure)
    2. Generates a 3D conformer using distance geometry
    3. Removes hydrogens to match typical PDB representation

    Args:
        mol: An RDKit molecule object (can be 2D or without conformers).

    Returns:
        The same molecule with an embedded 3D conformer.

    Note:
        The generated conformation may not be the global energy minimum,
        but provides a reasonable starting geometry.
    """
    mol = Chem.AddHs(mol)
    Chem.EmbedMultipleConfs(mol, numConfs=1)
    mol = Chem.RemoveHs(mol)
    return mol


@typecheck
def mol_from_smile(smile: str) -> Mol:
    """
    Generate an RDKit molecule with 3D coordinates from a SMILES string.

    Args:
        smile: A SMILES (Simplified Molecular Input Line Entry System) string
               representing the molecule's chemical structure.

    Returns:
        An RDKit Mol object with an embedded 3D conformer.

    Example:
        >>> mol = mol_from_smile("CC(C)C")  # Isobutane
        >>> mol.GetNumAtoms()
        4  # Carbon atoms (hydrogens removed)
    """
    mol = Chem.MolFromSmiles(smile)
    return generate_conformation(mol)


@typecheck
def mol_from_template_mmcif_file(
    mmcif_filepath: str, remove_hs: bool = True, remove_hydroxyl_oxygen: bool = True
) -> Chem.Mol:
    """
    Load an RDKit molecule from a template mmCIF (macromolecular Crystallographic Information File).

    This function parses chemical component dictionary files from the PDB to create
    RDKit molecules with proper atom positions, bonds, and stereochemistry.

    Args:
        mmcif_filepath: Path to a residue/ligand template mmCIF file.
        remove_hs: Whether to remove hydrogen atoms from the molecule.
                  Hydrogens are often omitted in PDB structures.
        remove_hydroxyl_oxygen: Whether to remove the C-terminal hydroxyl oxygen (OXT).
                               This atom is typically not present in internal residues.

    Returns:
        An RDKit Mol object with 3D coordinates from the template file.

    Note:
        The template atom positions are preserved from the mmCIF file.
        These should be overridden with actual coordinates when building structures.

    The function handles:
    - Atom parsing with 3D coordinates
    - Bond creation with proper bond orders (single, double, triple, aromatic)
    - Stereochemistry (E/Z configuration)
    - Aromaticity flags
    """
    # Parse the mmCIF file using Gemmi
    doc = gemmi.cif.read(mmcif_filepath)
    block = doc.sole_block()

    # Extract atoms and bonds
    atom_table = block.find(
        "_chem_comp_atom.",
        ["atom_id", "type_symbol", "model_Cartn_x", "model_Cartn_y", "model_Cartn_z"],
    )
    bond_table = block.find(
        "_chem_comp_bond.",
        ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag", "pdbx_stereo_config"],
    )

    # Create an empty `rdkit.Chem.RWMol` object
    mol = Chem.RWMol()

    # Dictionary to map atom ids to RDKit atom indices
    atom_id_to_idx = {}

    # Add atoms to the molecule
    for row in atom_table:
        element = row["type_symbol"]
        atom_id = row["atom_id"]
        if remove_hs and element == "H":
            continue
        elif remove_hydroxyl_oxygen and atom_id == "OXT":
            # NOTE: Hydroxyl oxygens are not present in the PDB's nucleotide residue templates
            continue
        rd_atom = Chem.Atom(element)
        idx = mol.AddAtom(rd_atom)
        atom_id_to_idx[atom_id] = idx

    # Create a conformer to store atom positions
    conf = Chem.Conformer(mol.GetNumAtoms())

    # Set atom coordinates
    for row in atom_table:
        atom_id = row["atom_id"]
        if atom_id not in atom_id_to_idx:
            continue
        idx = atom_id_to_idx[atom_id]
        x = float(row["model_Cartn_x"])
        y = float(row["model_Cartn_y"])
        z = float(row["model_Cartn_z"])
        conf.SetAtomPosition(idx, rdGeometry.Point3D(x, y, z))

    # Add conformer to the molecule
    mol.AddConformer(conf)

    for row in bond_table:
        atom_id1 = row["atom_id_1"]
        atom_id2 = row["atom_id_2"]
        if atom_id1 not in atom_id_to_idx or atom_id2 not in atom_id_to_idx:
            continue
        order = row["value_order"]
        aromatic_flag = row["pdbx_aromatic_flag"]
        stereo_config = row["pdbx_stereo_config"]

        idx1 = atom_id_to_idx[atom_id1]
        idx2 = atom_id_to_idx[atom_id2]

        mol.AddBond(idx1, idx2, BOND_ORDER[order])

        if aromatic_flag == "Y":
            mol.GetBondBetweenAtoms(idx1, idx2).SetIsAromatic(True)

        # Handle stereochemistry
        if stereo_config == "N":
            continue
        elif stereo_config == "E":
            mol.GetBondBetweenAtoms(idx1, idx2).SetStereo(Chem.BondStereo.STEREOE)
        elif stereo_config == "Z":
            mol.GetBondBetweenAtoms(idx1, idx2).SetStereo(Chem.BondStereo.STEREOZ)

    # Convert `RWMol` to `Mol`
    mol = mol.GetMol()

    return mol


# Initialize RDKit molecules for all biomolecular building blocks

# Separate biomolecules that can form polymers (proteins, DNA, RNA)
# from standalone molecules (metals, misc)
CHAINABLE_BIOMOLECULES = [
    HUMAN_AMINO_ACIDS,
    DNA_NUCLEOTIDES,
    RNA_NUCLEOTIDES,
]

METALS_AND_MISC = [
    METALS,
    MISC,
]

# Load RDKit molecule objects for each residue/molecule
# Prefer loading from template mmCIF files when available (more accurate),
# otherwise generate from SMILES strings
for entries in [*CHAINABLE_BIOMOLECULES, *METALS_AND_MISC]:
    for rescode in entries:
        entry = entries[rescode]
        resname = entry["resname"]
        # Check for a template file in the chemical/ subdirectory
        template_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chemical", f"{resname}.cif"
        )
        if os.path.exists(template_filepath):
            # Load from high-quality template file
            mol = mol_from_template_mmcif_file(template_filepath)
        else:
            # Generate from SMILES as fallback
            mol = mol_from_smile(entry["smile"])
        # Store the RDKit molecule object
        entry["rdchem_mol"] = mol

# Validate atom indices for chainable biomolecules
# These indices are critical for proper structure assembly
for entries in CHAINABLE_BIOMOLECULES:
    for rescode in entries:
        entry = entries[rescode]
        mol = entry['rdchem_mol']
        num_atoms = mol.GetNumAtoms()

        # Ensure all specified atom indices are valid
        assert 0 <= entry["first_atom_idx"] < num_atoms, \
            f"Invalid first_atom_idx for {rescode}"
        assert 0 <= entry["last_atom_idx"] < num_atoms, \
            f"Invalid last_atom_idx for {rescode}"
        assert 0 <= entry["distogram_atom_idx"] < num_atoms, \
            f"Invalid distogram_atom_idx for {rescode}"
        assert 0 <= entry["token_center_atom_idx"] < num_atoms, \
            f"Invalid token_center_atom_idx for {rescode}"

        # Validate frame calculation atoms if specified
        if exists(entry.get('three_atom_indices_for_frame', None)):
            assert all([(0 <= i < num_atoms) for i in entry["three_atom_indices_for_frame"]]), \
                f"Invalid three_atom_indices_for_frame for {rescode}"

        # Ensure first and last atoms are different (needed for bond formation)
        assert entry["first_atom_idx"] != entry["last_atom_idx"], \
            f"first_atom_idx and last_atom_idx must be different for {rescode}"
