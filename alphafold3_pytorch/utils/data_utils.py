"""Data utility functions for AlphaFold3 PyTorch implementation.

This module provides utility functions for processing molecular data, including:
- Residue type checking and classification (polymer, water, atomized)
- Molecule type determination and conversion
- Biopython structure manipulation
- Coordinate transformations and rotations
- Dictionary merging and data coercion
- mmCIF metadata extraction
- One-hot encoding utilities
- TSV file I/O operations

These utilities are primarily used for data preprocessing, validation, and
transformation in the AlphaFold3 structure prediction pipeline.
"""

import csv

import numpy as np
import torch
from beartype.typing import Any, Dict, Iterable, List, Literal, Set, Tuple
from torch import Tensor

from alphafold3_pytorch.tensor_typing import ChainType, ResidueType, typecheck
from alphafold3_pytorch.utils.utils import exists

# Constants defining molecule types and metadata fields

RESIDUE_MOLECULE_TYPE = Literal["protein", "rna", "dna", "ligand"]
PDB_INPUT_RESIDUE_MOLECULE_TYPE = Literal[
    "protein", "rna", "dna", "mod_protein", "mod_rna", "mod_dna", "ligand"
]
MMCIF_METADATA_FIELD = Literal[
    "structure_method", "release_date", "resolution", "structure_connectivity"
]


@typecheck
def is_polymer(
    res_chem_type: str, polymer_chem_types: Set[str] = {"peptide", "dna", "rna"}
) -> bool:
    """Check if a residue is polymeric using its chemical type string.

    A residue is considered polymeric if its chemical type string contains
    any of the standard polymer identifiers (peptide, DNA, or RNA). This is
    used to distinguish between polymer chains and small molecules/ligands.

    Args:
        res_chem_type: The chemical type of the residue as a descriptive string
            (e.g., "L-peptide linking", "DNA linking").
        polymer_chem_types: The set of polymer chemical types to check against.
            Defaults to {"peptide", "dna", "rna"}.

    Returns:
        bool: True if the residue is polymeric, False otherwise.

    Example:
        >>> is_polymer("L-peptide linking")
        True
        >>> is_polymer("non-polymer")
        False
    """
    return any(chem_type in res_chem_type.lower() for chem_type in polymer_chem_types)


@typecheck
def is_water(res_name: str, water_res_names: Set[str] = {"HOH", "WAT"}) -> bool:
    """Check if a residue is a water residue using its residue name string.

    Water molecules are commonly represented in PDB/mmCIF files with standard
    residue names like "HOH" or "WAT". This function identifies water residues
    for filtering or special handling during structure processing.

    Args:
        res_name: The name of the residue as a descriptive string (e.g., "HOH", "WAT").
        water_res_names: The set of water residue names to check against.
            Defaults to {"HOH", "WAT"}.

    Returns:
        bool: True if the residue is a water residue, False otherwise.

    Example:
        >>> is_water("HOH")
        True
        >>> is_water("ALA")
        False
    """
    return any(water_res_name in res_name.upper() for water_res_name in water_res_names)


@typecheck
def is_atomized_residue(
    res_name: str, atomized_res_mol_types: Set[str] = {"ligand", "mod"}
) -> bool:
    """Check if a residue is an atomized residue using its residue molecule type string.

    Atomized residues are represented at the individual atom level rather than
    with standard residue templates. This includes ligands and modified residues
    that don't follow standard amino acid, nucleotide, or DNA structures.

    Args:
        res_name: The name of the residue as a descriptive string
            (e.g., "ligand", "mod_protein").
        atomized_res_mol_types: The set of atomized residue molecule types as strings.
            Defaults to {"ligand", "mod"}.

    Returns:
        bool: True if the residue is an atomized residue, False otherwise.

    Example:
        >>> is_atomized_residue("ligand_ATP")
        True
        >>> is_atomized_residue("ALA")
        False
    """
    return any(mol_type in res_name.lower() for mol_type in atomized_res_mol_types)


@typecheck
def get_residue_molecule_type(res_chem_type: str) -> RESIDUE_MOLECULE_TYPE:
    """Get the molecule type of a residue from its chemical type string.

    This function classifies residues into one of four categories: protein, RNA, DNA,
    or ligand. The classification is based on keywords in the chemical type string
    from PDB/mmCIF files. Any residue that doesn't match protein, RNA, or DNA is
    classified as a ligand.

    Args:
        res_chem_type: The chemical type of the residue as a descriptive string
            (e.g., "L-peptide linking", "RNA linking", "DNA linking").

    Returns:
        RESIDUE_MOLECULE_TYPE: One of "protein", "rna", "dna", or "ligand".

    Example:
        >>> get_residue_molecule_type("L-peptide linking")
        'protein'
        >>> get_residue_molecule_type("RNA linking")
        'rna'
        >>> get_residue_molecule_type("non-polymer")
        'ligand'
    """
    if "peptide" in res_chem_type.lower():
        return "protein"
    elif "rna" in res_chem_type.lower():
        return "rna"
    elif "dna" in res_chem_type.lower():
        return "dna"
    else:
        return "ligand"


@typecheck
def get_pdb_input_residue_molecule_type(
    res_chem_type: str, is_modified_polymer_residue: bool = False
) -> PDB_INPUT_RESIDUE_MOLECULE_TYPE:
    """Get the molecule type of a residue, including modified polymer types.

    This function extends get_residue_molecule_type() by also distinguishing between
    standard and modified polymer residues. Modified residues are those with non-standard
    chemical modifications (e.g., post-translational modifications in proteins, modified
    nucleotides in RNA/DNA).

    Args:
        res_chem_type: The chemical type of the residue as a descriptive string
            (e.g., "L-peptide linking", "RNA linking").
        is_modified_polymer_residue: Whether the residue is a modified polymer residue.
            Defaults to False.

    Returns:
        PDB_INPUT_RESIDUE_MOLECULE_TYPE: One of "protein", "mod_protein", "rna", "mod_rna",
            "dna", "mod_dna", or "ligand".

    Example:
        >>> get_pdb_input_residue_molecule_type("L-peptide linking", False)
        'protein'
        >>> get_pdb_input_residue_molecule_type("L-peptide linking", True)
        'mod_protein'
    """
    if "peptide" in res_chem_type.lower():
        return "mod_protein" if is_modified_polymer_residue else "protein"
    elif "rna" in res_chem_type.lower():
        return "mod_rna" if is_modified_polymer_residue else "rna"
    elif "dna" in res_chem_type.lower():
        return "mod_dna" if is_modified_polymer_residue else "dna"
    else:
        return "ligand"


@typecheck
def get_biopython_chain_residue_by_composite_id(
    chain: ChainType, res_name: str, res_id: int
) -> ResidueType:
    """Get a Biopython `Residue` or `DisorderedResidue` object by its composite ID.

    Biopython uses a composite ID structure for residues: (hetero_flag, sequence_id, insertion_code).
    This function tries multiple common ID patterns to retrieve a residue from a chain, handling
    various edge cases including hetero atoms and disordered residues.

    The function attempts to find the residue using the following ID patterns in order:
    1. ("", res_id, " ") - Standard residue
    2. (" ", res_id, " ") - Alternative standard residue format
    3. (f"H_{res_name}", res_id, " ") - Hetero residue
    4. (f"H_{res_name}", res_id, "A") - Disordered residue version A

    Args:
        chain: Biopython `Chain` object containing the residues.
        res_name: Residue name (e.g., "ALA", "GLY").
        res_id: Residue sequence index/number.

    Returns:
        ResidueType: Biopython `Residue` or `DisorderedResidue` object.

    Raises:
        AssertionError: If the residue cannot be found using any of the standard ID patterns.

    Example:
        >>> from Bio.PDB import PDBParser
        >>> parser = PDBParser()
        >>> structure = parser.get_structure('protein', 'protein.pdb')
        >>> chain = structure[0]['A']
        >>> residue = get_biopython_chain_residue_by_composite_id(chain, 'ALA', 10)
    """
    # Try standard residue ID format (empty hetero flag)
    if ("", res_id, " ") in chain:
        res = chain[("", res_id, " ")]
    # Try alternative standard format (space hetero flag)
    elif (" ", res_id, " ") in chain:
        res = chain[(" ", res_id, " ")]
    # Try hetero residue format
    elif (
        f"H_{res_name}",
        res_id,
        " ",
    ) in chain:
        res = chain[
            (
                f"H_{res_name}",
                res_id,
                " ",
            )
        ]
    # Try disordered residue version A
    else:
        assert (
            f"H_{res_name}",
            res_id,
            "A",
        ) in chain, f"Version A of residue {res_name} of ID {res_id} in chain {chain.id} was missing from the chain's structure."
        res = chain[
            (
                f"H_{res_name}",
                res_id,
                "A",
            )
        ]
    return res


@typecheck
def matrix_rotate(v: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Perform a rotation using a rotation matrix.

    This function applies a 3x3 rotation matrix to coordinates, handling arrays
    of arbitrary dimensions. The function reshapes multi-dimensional arrays to 2D,
    applies the rotation, and then restores the original shape.

    Args:
        v: The coordinates to rotate. Can be of shape (..., 3) where ... represents
            any number of leading dimensions.
        matrix: The 3x3 rotation matrix to apply.

    Returns:
        np.ndarray: The rotated coordinates with the same shape as the input.

    Example:
        >>> coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> rotated = matrix_rotate(coords, rotation_matrix)
    """
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = v.ndim
    if orig_ndim > 2:
        orig_shape = v.shape
        v = v.reshape(-1, 3)
    # Apply rotation: R @ v^T gives rotated vectors, then transpose back
    v = np.dot(matrix, v.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        v = v.reshape(*orig_shape)
    return v


@typecheck
def deep_merge_dicts(
    dict1: Dict[Any, Any], dict2: Dict[Any, Any], value_op: Literal["union", "concat"]
) -> Dict[Any, Any]:
    """Deeply merge two dictionaries, combining values where possible.

    This function merges two dictionaries by combining values for keys that exist
    in both dictionaries. Values are assumed to be lists or list-like objects that
    support the + operator for concatenation.

    Args:
        dict1: The first dictionary to merge (will be modified in place).
        dict2: The second dictionary to merge.
        value_op: The merge operation to perform on the values of matching keys:
            - "union": Combines values and removes duplicates while preserving order.
            - "concat": Concatenates values without removing duplicates.

    Returns:
        Dict[Any, Any]: The merged dictionary (same object as dict1, modified in place).

    Example:
        >>> d1 = {'a': [1, 2], 'b': [3]}
        >>> d2 = {'a': [2, 3], 'c': [4]}
        >>> deep_merge_dicts(d1, d2, value_op="union")
        {'a': [1, 2, 3], 'b': [3], 'c': [4]}
    """
    # Iterate over items in dict2
    for key, value in dict2.items():
        # If key is in dict1, merge the values
        if key in dict1:
            merged_value = dict1[key] + value
            if value_op == "union":
                # Remove duplicates while preserving order
                dict1[key] = list(dict.fromkeys(merged_value))
            else:
                # Concatenate without removing duplicates
                dict1[key] = merged_value
        else:
            # Otherwise, set/overwrite the key in dict1 with dict2's value
            dict1[key] = value
    return dict1


@typecheck
def coerce_to_float(obj: Any) -> float | None:
    """Coerce an object to a float, returning `None` if conversion fails.

    This function attempts to convert various types to float, with special handling
    for lists (converts the first element). If conversion fails, it returns None
    instead of raising an exception.

    Args:
        obj: The object to coerce to a float. Can be int, float, str, list, or other types.

    Returns:
        float | None: The object coerced to a float if possible, otherwise `None`.

    Example:
        >>> coerce_to_float(42)
        42.0
        >>> coerce_to_float("3.14")
        3.14
        >>> coerce_to_float([2.5, 3.0])
        2.5
        >>> coerce_to_float("invalid")
        None
    """
    try:
        if isinstance(obj, (int, float, str)):
            return float(obj)
        elif isinstance(obj, list):
            # If it's a list, try to convert the first element
            return float(obj[0])
        else:
            return None
    except (ValueError, TypeError):
        # Return None if conversion fails
        return None


@typecheck
def extract_mmcif_metadata_field(
    mmcif_object: Any,
    metadata_field: MMCIF_METADATA_FIELD,
    min_resolution: float = 0.0,
    max_resolution: float = 1000.0,
) -> str | float | None:
    """Extract a metadata field from an mmCIF object.

    This function extracts various metadata fields from Biopython mmCIF objects,
    including experimental method, release date, and resolution. For resolution,
    it tries multiple fields in order of preference and validates the value against
    min/max bounds.

    The function handles different experimental methods (X-ray crystallography,
    cryo-EM, etc.) by checking multiple possible resolution fields in the mmCIF format.

    Args:
        mmcif_object: The Biopython mmCIF object with a `raw_string` attribute
            containing the parsed mmCIF data dictionary.
        metadata_field: The type of metadata field to extract. One of:
            - "structure_method": Experimental method (e.g., "X-RAY DIFFRACTION")
            - "release_date": PDB release date
            - "resolution": Structure resolution in Angstroms
            - "structure_connectivity": Connectivity information
        min_resolution: Minimum acceptable resolution value in Angstroms. Defaults to 0.0.
        max_resolution: Maximum acceptable resolution value in Angstroms. Defaults to 1000.0.

    Returns:
        str | float | None: The extracted metadata value. Returns None if the field
            is not found or if resolution is outside the acceptable range.

    Note:
        For resolution, the function checks three possible mmCIF fields in order:
        1. "_refine.ls_d_res_high" (X-ray refinement)
        2. "_em_3d_reconstruction.resolution" (Cryo-EM)
        3. "_reflns.d_resolution_high" (Reflection data)

    Example:
        >>> from Bio.PDB.MMCIFParser import MMCIFParser
        >>> parser = MMCIFParser()
        >>> structure = parser.get_structure('1abc', '1abc.cif')
        >>> mmcif_dict = parser._mmcif_dict
        >>> extract_mmcif_metadata_field(mmcif_dict, "resolution", min_resolution=0.0, max_resolution=5.0)
        2.1
    """
    # Extract structure method (e.g., X-RAY DIFFRACTION, ELECTRON MICROSCOPY)
    if metadata_field == "structure_method" and "_exptl.method" in mmcif_object.raw_string:
        return mmcif_object.raw_string["_exptl.method"]

    # Extract release date - return the earliest date if multiple revisions exist
    if (
        metadata_field == "release_date"
        and "_pdbx_audit_revision_history.revision_date" in mmcif_object.raw_string
    ):
        return min(mmcif_object.raw_string["_pdbx_audit_revision_history.revision_date"])

    # Extract resolution - try multiple fields depending on experimental method
    # Field 1: X-ray crystallography refinement resolution
    if metadata_field == "resolution" and "_refine.ls_d_res_high" in mmcif_object.raw_string:
        resolution = coerce_to_float(mmcif_object.raw_string["_refine.ls_d_res_high"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution
    # Field 2: Cryo-EM 3D reconstruction resolution
    elif (
        metadata_field == "resolution"
        and "_em_3d_reconstruction.resolution" in mmcif_object.raw_string
    ):
        resolution = coerce_to_float(mmcif_object.raw_string["_em_3d_reconstruction.resolution"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution
    # Field 3: Alternative reflection data resolution
    elif metadata_field == "resolution" and "_reflns.d_resolution_high" in mmcif_object.raw_string:
        resolution = coerce_to_float(mmcif_object.raw_string["_reflns.d_resolution_high"])
        if exists(resolution) and min_resolution <= resolution <= max_resolution:
            return resolution


@typecheck
def make_one_hot(x: Tensor, num_classes: int) -> Tensor:
    """Convert a tensor of class indices to a one-hot encoded tensor.

    Creates a one-hot encoded representation where each index is converted to a
    binary vector with a 1 at the index position and 0s elsewhere.

    Args:
        x: A tensor of integer class indices, shape (...,) with values in [0, num_classes).
        num_classes: The total number of classes (determines the size of the one-hot dimension).

    Returns:
        Tensor: One-hot encoded tensor of shape (..., num_classes) where the last dimension
            contains the one-hot encoding.

    Example:
        >>> indices = torch.tensor([0, 2, 1])
        >>> make_one_hot(indices, num_classes=3)
        tensor([[1., 0., 0.],
                [0., 0., 1.],
                [0., 1., 0.]])
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    # scatter_ writes value 1 at positions specified by x along the last dimension
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


@typecheck
def make_one_hot_np(x: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a NumPy array of class indices to a one-hot encoded array.

    NumPy version of make_one_hot() that creates a one-hot encoded representation
    of integer class indices.

    Args:
        x: A NumPy array of integer class indices, shape (...,) with values in [0, num_classes).
        num_classes: The total number of classes (determines the size of the one-hot dimension).

    Returns:
        np.ndarray: One-hot encoded array of shape (..., num_classes) with dtype int64.

    Example:
        >>> indices = np.array([0, 2, 1])
        >>> make_one_hot_np(indices, num_classes=3)
        array([[1, 0, 0],
               [0, 0, 1],
               [0, 1, 0]])
    """
    x_one_hot = np.zeros((*x.shape, num_classes), dtype=np.int64)
    # put_along_axis writes value 1 at positions specified by x along the last axis
    np.put_along_axis(x_one_hot, np.expand_dims(x, axis=-1), 1, axis=-1)
    return x_one_hot


@typecheck
def get_sorted_tuple_indices(
    tuples_list: List[Tuple[str, Any]], order_list: List[str]
) -> List[int]:
    """Get the indices of tuples reordered according to a specified order.

    This function creates a mapping from string keys in tuples to their original
    indices, then returns indices reordered according to a specified order list.
    Useful for reordering data structures based on a desired key ordering.

    Args:
        tuples_list: A list of tuples where each tuple contains a string key as
            the first element and any value as the second element.
            Example: [("apple", 1), ("banana", 2), ("cherry", 3)]
        order_list: A list of strings specifying the desired order of the tuples
            by their first element.
            Example: ["banana", "apple", "cherry"]

    Returns:
        List[int]: A list of indices corresponding to the reordered tuples.
            Example: [1, 0, 2] (indices that would reorder tuples_list according to order_list)

    Example:
        >>> tuples = [("apple", 1), ("banana", 2), ("cherry", 3)]
        >>> order = ["banana", "apple", "cherry"]
        >>> get_sorted_tuple_indices(tuples, order)
        [1, 0, 2]
    """
    # Create a mapping from the string values to their indices
    index_map = {value: index for index, (value, _) in enumerate(tuples_list)}

    # Generate the indices in the order specified by the order_list
    sorted_indices = [index_map[value] for value in order_list]

    return sorted_indices


@typecheck
def load_tsv_to_dict(filepath):
    """Load a two-column TSV file into a dictionary.

    Reads a tab-separated values (TSV) file with exactly two columns and converts
    it into a dictionary where the first column becomes keys and the second column
    becomes values. Useful for loading simple key-value mappings from files.

    Args:
        filepath: The path to the TSV file to load. File should have two columns
            separated by tabs, with no header row.

    Returns:
        Dict[str, str]: A dictionary mapping first column values to second column values.

    Example:
        Given a file 'mapping.tsv' with contents:
        ```
        protein_A\tP12345
        protein_B\tP67890
        ```

        >>> load_tsv_to_dict('mapping.tsv')
        {'protein_A': 'P12345', 'protein_B': 'P67890'}
    """
    result = {}
    with open(filepath, mode="r", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            # Map first column to second column
            result[row[0]] = row[1]
    return result


@typecheck
def join(arr: Iterable[Any], delimiter: str = "") -> str:
    """Join the elements of an iterable into a string using a delimiter.

    A convenience wrapper around Python's str.join() method that reverses the
    argument order to put the array first, which can be more intuitive in some
    contexts. Automatically converts elements to strings.

    Args:
        arr: The iterable of elements to join (will be converted to strings).
        delimiter: The delimiter string to insert between elements. Defaults to
            empty string (no delimiter).

    Returns:
        str: The joined string.

    Example:
        >>> join(['a', 'b', 'c'], delimiter=', ')
        'a, b, c'
        >>> join([1, 2, 3], delimiter='-')
        '1-2-3'
    """
    # Provide a more intuitive argument order than Python's built-in str.join()
    return delimiter.join(arr)
