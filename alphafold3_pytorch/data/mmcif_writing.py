"""
mmCIF File Format Writer

This module provides utilities for writing macromolecular structures to mmCIF format files.
It handles the conversion from internal Biomolecule representations back to mmCIF format,
which is useful for outputting predicted structures or writing modified structures.

The module supports:
- Writing structures from mmCIF objects
- Inserting sampled/predicted atom coordinates
- Updating B-factors (confidence scores)
- Preserving original atom names and metadata
- Handling biological assemblies
- Optional AlphaFold-specific metadata insertion

Key functions:
- write_mmcif_from_filepath_and_id: Read and rewrite mmCIF with optional modifications
- write_mmcif: Write a Biomolecule to mmCIF file with customization options

The writer is commonly used to:
1. Output predicted structures from AlphaFold 3
2. Save cropped or modified structures
3. Update coordinates while preserving metadata
4. Generate mmCIF files for validation or visualization
"""

import numpy as np

from loguru import logger

from alphafold3_pytorch.common.biomolecule import _from_mmcif_object, to_mmcif
from alphafold3_pytorch.data.data_pipeline import get_assembly
from alphafold3_pytorch.data.mmcif_parsing import MmcifObject, parse_mmcif_object
from alphafold3_pytorch.utils.utils import exists


def write_mmcif_from_filepath_and_id(
    input_filepath: str, output_filepath: str, file_id: str, **kwargs
):
    """
    Read an mmCIF file and write it to a new location with optional modifications.

    This is a convenience function that combines parsing and writing. It's useful
    for updating coordinates, B-factors, or other properties of an existing structure.

    Args:
        input_filepath: Path to the input mmCIF file to read.
        output_filepath: Path where the output mmCIF file should be written.
        file_id: Identifier for the structure (typically PDB ID or filename).
        **kwargs: Additional keyword arguments passed to write_mmcif(), such as:
            - sampled_atom_positions: New atom coordinates to insert
            - b_factors: New B-factors (confidence scores) to insert
            - gapless_poly_seq: Whether to write gapless polymer sequences
            - insert_alphafold_mmcif_metadata: Whether to add AlphaFold metadata

    Note:
        If parsing or writing fails (e.g., due to prior cropping), a warning is
        logged but no exception is raised.

    Example:
        >>> write_mmcif_from_filepath_and_id(
        ...     "input.cif", "output.cif", "7a4d",
        ...     sampled_atom_positions=new_coords
        ... )
    """
    try:
        mmcif_object = parse_mmcif_object(filepath=input_filepath, file_id=file_id)
        return write_mmcif(mmcif_object, output_filepath=output_filepath, **kwargs)
    except Exception as e:
        logger.warning(
            f"Failed to write mmCIF file {output_filepath} due to: {e}. Perhaps cropping was performed on this example?"
        )


def write_mmcif(
    mmcif_object: MmcifObject,
    output_filepath: str,
    gapless_poly_seq: bool = True,
    insert_orig_atom_names: bool = True,
    insert_alphafold_mmcif_metadata: bool = True,
    sampled_atom_positions: np.ndarray | None = None,
    b_factors: np.ndarray | None = None,
):
    """
    Write a Biomolecule structure to an mmCIF file with optional modifications.

    This function converts an MmcifObject to a Biomolecule, optionally updates
    coordinates and B-factors, and writes the result to an mmCIF file. It handles
    biological assemblies automatically.

    Args:
        mmcif_object: The parsed mmCIF object containing structure data.
        output_filepath: Path where the output mmCIF file should be written.
        gapless_poly_seq: If True, write polymer sequences without gaps (default: True).
            This follows the standard mmCIF convention.
        insert_orig_atom_names: If True, preserve original atom names from the input
            structure (default: True). Otherwise, use standard atom naming.
        insert_alphafold_mmcif_metadata: If True, add AlphaFold-specific metadata
            to the mmCIF file (default: True). This includes model confidence scores.
        sampled_atom_positions: Optional numpy array of new atom coordinates to insert.
            Shape should match the masked (non-missing) atoms in the structure.
        b_factors: Optional numpy array of new B-factors (confidence/uncertainty values).
            Shape should match sampled_atom_positions if provided.

    Raises:
        AssertionError: If sampled_atom_positions or b_factors shapes don't match
            the expected masked atom count.

    Example:
        >>> write_mmcif(
        ...     mmcif_obj, "output.cif",
        ...     sampled_atom_positions=predicted_coords,
        ...     b_factors=confidence_scores
        ... )

    Note:
        - The function automatically expands biological assemblies if the file_id
          doesn't already contain "assembly"
        - Atom positions and B-factors are only updated for atoms that exist in
          the original structure (according to atom_mask)
    """
    # Convert mmCIF object to Biomolecule, expanding assembly if needed
    biomol = (
        _from_mmcif_object(mmcif_object)
        if "assembly" in mmcif_object.file_id
        else get_assembly(_from_mmcif_object(mmcif_object))
    )

    # Update atom positions and B-factors if provided
    if exists(sampled_atom_positions):
        # Only update positions for atoms that exist (atom_mask == True)
        atom_mask = biomol.atom_mask.astype(bool)
        assert biomol.atom_positions[atom_mask].shape == sampled_atom_positions.shape, (
            f"Expected sampled atom positions to have masked shape {biomol.atom_positions[atom_mask].shape}, "
            f"but got {sampled_atom_positions.shape}."
        )
        biomol.atom_positions[atom_mask] = sampled_atom_positions

        # Update B-factors if provided
        if exists(b_factors):
            assert biomol.b_factors[atom_mask].shape == b_factors.shape, (
                f"Expected B-factors to have shape {biomol.b_factors[atom_mask].shape}, "
                f"but got {b_factors.shape}."
            )
            biomol.b_factors[atom_mask] = b_factors

    # Preserve original atom names if requested
    unique_res_atom_names = biomol.unique_res_atom_names if insert_orig_atom_names else None

    # Convert Biomolecule to mmCIF string format
    mmcif_string = to_mmcif(
        biomol,
        mmcif_object.file_id,
        gapless_poly_seq=gapless_poly_seq,
        insert_alphafold_mmcif_metadata=insert_alphafold_mmcif_metadata,
        unique_res_atom_names=unique_res_atom_names,
    )

    # Write to file
    with open(output_filepath, "w") as f:
        f.write(mmcif_string)
