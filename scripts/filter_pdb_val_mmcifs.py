"""
PDB Validation Dataset Filtering for AlphaFold 3.

This script filters and curates mmCIF files to create the AlphaFold 3 PDB validation dataset
used for model selection during training. The validation set consists of low-homology chains
and interfaces from PDB structures released between 2021-10-01 and 2023-01-13.

The filtering procedure follows a modified, more stringent version of the validation procedure
outlined in Abramson et al (2024). The process includes two stages:

Multimer Selection:
    1. Select all targets released within the date range
    2. Remove targets with >2048 tokens, >1000 chains, or resolution >4.5 Å
    3. Generate interface chain pairs for remaining targets
    4. Filter for low homology interfaces (see clustering script)

Monomer Selection:
    1. Select polymer monomer targets (may include ligands) within the date range
    2. Remove targets with >2048 tokens or resolution >4.5 Å
    3. Filter for low homology polymers (see clustering script)

Filtering Criteria:
    - Release date: 2021-10-01 to 2023-01-13
    - Maximum tokens: 2048
    - Maximum chains: 1000
    - Maximum resolution: 4.5 Å
    - Ligands in exclusion set are removed
    - Crystallization aids are removed
    - Hydrogens and waters are removed

Usage:
    python filter_pdb_val_mmcifs.py -i <input_dir> -a <asym_dir> -o <output_dir>

See Also:
    - cluster_pdb_val_mmcifs.py: For low-homology clustering of the filtered structures
    - filter_pdb_train_mmcifs.py: For training dataset filtering procedures
"""

# %% [markdown]
# # Curating AlphaFold 3 PDB Validation Dataset
#
# For validating AlphaFold 3 during model training, we propose a modified (i.e., more stringent) version of the
# validation procedure outlined in Abramson et al (2024).
#
# The validation set for model selection during training was composed of all low homology chains and interfaces from
# a subset of all PDB targets released after 2021-09-30 and before 2023-01-13, with maximum length 2048 tokens.
# The process for selecting these targets was broken up into two separate stages. The first was for selecting multimers,
# the second for selecting monomers. Multimer selection proceeded as follows:
#
# 1. Take all targets released after 2021-09-30 and before 2023-01-13 and remove targets with total number of tokens
# greater than 2048, more than one thousand chains or resolution greater than 4.5, then generate a list of all interface
# chain pairs for all remaining targets.
# ... (see the PDB validation set clustering script)
#
# Monomer selection proceeded similarly:
#
# 1. Take all polymer monomer targets released after 2021-09-30 and before 2023-01-13 (can include monomer poly-
# mers with ligand chains) and remove targets with total number of tokens greater than 2048 or resolution greater
# than 4.5.
# ... (see the PDB validation set clustering script)
#

# %%
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import timeout_decorator
from tqdm.contrib.concurrent import process_map

from alphafold3_pytorch.common.biomolecule import _from_mmcif_object
from alphafold3_pytorch.common.paper_constants import (
    CRYSTALLOGRAPHY_METHODS,
    LIGAND_EXCLUSION_SET,
)
from alphafold3_pytorch.data import mmcif_parsing, mmcif_writing
from alphafold3_pytorch.data.data_pipeline import get_assembly
from alphafold3_pytorch.data.mmcif_parsing import MmcifObject
from alphafold3_pytorch.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists
from scripts.filter_pdb_train_mmcifs import (
    FILTER_STRUCTURE_MAX_SECONDS_PER_INPUT,
    filter_pdb_release_date,
    filter_resolution,
    impute_missing_assembly_metadata,
    remove_crystallization_aids,
    remove_excluded_ligands,
    remove_hydrogens,
)

# Helper functions


@typecheck
def filter_num_tokens(
    mmcif_object: MmcifObject, max_tokens: int = 2048, exclusive_max: bool = False
) -> bool:
    """
    Filter structures based on the number of tokens (atoms).

    This function converts an mmCIF structure to a Biomolecule object and counts
    the number of atoms (tokens) in the structure. This is important for ensuring
    structures fit within memory and computational constraints during training.

    Args:
        mmcif_object: Parsed mmCIF structure object
        max_tokens: Maximum number of tokens allowed. Defaults to 2048.
        exclusive_max: If True, use strict inequality (<). If False, use <= (default).

    Returns:
        bool: True if the structure passes the token count filter, False otherwise.

    Note:
        - For assembly files, the structure is used directly
        - For non-assembly files, the biological assembly is generated first
        - Each atom in the structure corresponds to one token
    """
    # Generate biomolecule representation from mmCIF object
    # Assembly files are used directly; others need assembly generation
    biomol = (
        _from_mmcif_object(mmcif_object)
        if "assembly" in mmcif_object.file_id
        else get_assembly(_from_mmcif_object(mmcif_object))
    )
    # Check if token count is within the specified limit
    return (
        len(biomol.atom_mask) < max_tokens
        if exclusive_max
        else len(biomol.atom_mask) <= max_tokens
    )


@typecheck
def filter_num_chains(mmcif_object: MmcifObject, max_chains: int = 1000) -> bool:
    """
    Filter structures based on the number of chains.

    This function counts the number of polymer and ligand chains in a structure
    to ensure it doesn't exceed computational limits. Very large assemblies with
    thousands of chains are typically excluded from training datasets.

    Args:
        mmcif_object: Parsed mmCIF structure object
        max_chains: Maximum number of chains allowed. Defaults to 1000.

    Returns:
        bool: True if the number of chains is within the limit, False otherwise.

    Note:
        All chains are counted, including protein, nucleic acid, peptide, and ligand chains.
    """
    # Count all chains in the structure
    return len(list(mmcif_object.structure.get_chains())) <= max_chains


@typecheck
def prefilter_target(
    mmcif_object: MmcifObject,
    min_cutoff_date: datetime = datetime(2021, 10, 1),
    max_cutoff_date: datetime = datetime(2023, 1, 13),
    max_tokens: int = 2048,
    max_chains: int = 1000,
    max_resolution: float = 4.5,
) -> MmcifObject | None:
    """
    Apply initial filtering criteria to select validation dataset structures.

    This function applies multiple criteria to determine if a structure should be
    included in the validation dataset. All criteria must be satisfied for a
    structure to pass the pre-filtering stage.

    Args:
        mmcif_object: Parsed mmCIF structure object to filter
        min_cutoff_date: Minimum PDB release date (inclusive). Defaults to 2021-10-01.
        max_cutoff_date: Maximum PDB release date (inclusive). Defaults to 2023-01-13.
        max_tokens: Maximum number of atoms allowed. Defaults to 2048.
        max_chains: Maximum number of chains allowed. Defaults to 1000.
        max_resolution: Maximum resolution in Angstroms. Defaults to 4.5.

    Returns:
        MmcifObject | None: The original mmCIF object if it passes all filters,
            None if it fails any filter.

    Note:
        The validation set date range (2021-10-01 to 2023-01-13) falls between
        the training set (up to 2021-09-30) and test set (from 2023-01-14 onwards).
    """
    target_passes_prefilters = (
        filter_pdb_release_date(
            mmcif_object, min_cutoff_date=min_cutoff_date, max_cutoff_date=max_cutoff_date
        )
        and filter_num_tokens(mmcif_object, max_tokens=max_tokens)
        and filter_num_chains(mmcif_object, max_chains=max_chains)
        and filter_resolution(mmcif_object, max_resolution=max_resolution)
    )
    return mmcif_object if target_passes_prefilters else None


@typecheck
@timeout_decorator.timeout(FILTER_STRUCTURE_MAX_SECONDS_PER_INPUT, use_signals=False)
def filter_structure_with_timeout(
    filepath: str,
    output_dir: str,
    min_cutoff_date: datetime = datetime(2021, 10, 1),
    max_cutoff_date: datetime = datetime(2023, 1, 13),
    keep_ligands_in_exclusion_set: bool = False,
):
    """
    Filter a single mmCIF file for the validation dataset with timeout protection.

    This function applies the complete filtering pipeline to create a validation dataset
    structure from raw PDB assembly and asymmetric unit mmCIF files. The filtering includes
    date range checks, size constraints, removal of unwanted molecules, and cleanup of
    problematic structural features.

    The function operates under a timeout constraint to prevent hanging on problematic files.

    Args:
        filepath: Path to the input assembly mmCIF file to filter
        output_dir: Directory where the filtered mmCIF file will be written
        min_cutoff_date: Minimum PDB release date. Defaults to 2021-10-01.
        max_cutoff_date: Maximum PDB release date. Defaults to 2023-01-13.
        keep_ligands_in_exclusion_set: If True, retain ligands that would normally
            be excluded. Defaults to False.

    Returns:
        None

    Raises:
        timeout_decorator.TimeoutError: If processing exceeds the time limit

    Note:
        The function imputes missing metadata from the asymmetric unit file, applies
        pre-filters, removes hydrogens/waters, removes excluded ligands (unless specified),
        removes crystallization aids, and writes the cleaned structure to a new file.

        According to the AlphaFold 3 supplement, validation and test datasets should
        remove excluded ligands, unlike the training dataset which retains them.
    """
    # Determine paths for assembly and asymmetric unit mmCIF files
    # Assembly files contain the biological assembly; asym files contain metadata
    asym_filepath = os.path.join(
        os.path.dirname(filepath).replace("unfiltered_assembly", "unfiltered_asym"),
        os.path.basename(filepath).replace("-assembly1", ""),
    )
    # Extract file identifier and setup output paths
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    # Organize outputs in subdirectories based on middle two characters of PDB ID
    output_file_dir = os.path.join(output_dir, file_id[1:3])
    output_filepath = os.path.join(output_file_dir, f"{file_id}.cif")
    os.makedirs(output_file_dir, exist_ok=True)

    # Parse both assembly and asymmetric unit mmCIF files
    mmcif_object = mmcif_parsing.parse_mmcif_object(filepath, file_id)
    asym_mmcif_object = mmcif_parsing.parse_mmcif_object(asym_filepath, file_id)

    # Impute missing metadata from asymmetric unit
    # Assembly files often lack header information and bond connectivity
    mmcif_object = impute_missing_assembly_metadata(mmcif_object, asym_mmcif_object)

    # Apply pre-filtering criteria (date range, size, resolution)
    mmcif_object = prefilter_target(
        mmcif_object,
        min_cutoff_date=min_cutoff_date,
        max_cutoff_date=max_cutoff_date,
        max_tokens=2048,
        max_chains=1000,
        max_resolution=4.5,
    )
    # Skip structures that don't pass pre-filtering
    if not exists(mmcif_object):
        print(f"Skipping target due to prefiltering: {file_id}")
        return

    # Apply bioassembly-level filters to clean up the structure
    # Remove hydrogens and waters (not explicitly mentioned in AF3 supplement but standard practice)
    mmcif_object = remove_hydrogens(mmcif_object, remove_waters=True)

    # Remove excluded ligands (standard practice for validation/test sets)
    if not keep_ligands_in_exclusion_set:
        mmcif_object = remove_excluded_ligands(mmcif_object, LIGAND_EXCLUSION_SET)

    # Remove crystallization aids that don't represent biological structure
    mmcif_object = remove_crystallization_aids(mmcif_object, CRYSTALLOGRAPHY_METHODS)

    # Only save if there are chains remaining after filtering
    if len(mmcif_object.chains_to_remove) < len(mmcif_object.structure):
        # Apply all accumulated filters to the structure
        mmcif_object = mmcif_parsing.filter_mmcif(mmcif_object)
        # Write the filtered structure to a new mmCIF file
        mmcif_writing.write_mmcif(
            mmcif_object,
            output_filepath,
            gapless_poly_seq=True,  # Ensure no gaps in polymer sequences
            insert_orig_atom_names=True,  # Preserve original atom naming
            insert_alphafold_mmcif_metadata=False,  # Don't add AF3-specific metadata
        )
        print(f"Finished filtering structure: {mmcif_object.file_id}")


@typecheck
def filter_structure(args: Tuple[str, str, datetime, datetime, bool]):
    """
    Wrapper function for filtering a structure with error handling.

    This function unpacks arguments and calls the timeout-protected filtering function.
    It catches all exceptions to ensure batch processing continues even when individual
    files fail, and cleans up any partially written files on failure.

    Args:
        args: Tuple containing:
            - filepath (str): Path to input assembly mmCIF file
            - output_dir (str): Output directory for filtered file
            - min_cutoff_date (datetime): Minimum PDB release date
            - max_cutoff_date (datetime): Maximum PDB release date
            - keep_ligands_in_exclusion_set (bool): Whether to keep excluded ligands

    Returns:
        None

    Note:
        Errors are printed but don't stop the overall processing pipeline.
        Partial output files are removed if filtering fails.
    """
    filepath, output_dir, min_cutoff_date, max_cutoff_date, keep_ligands_in_exclusion_set = args
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    output_file_dir = os.path.join(output_dir, file_id[1:3])
    output_filepath = os.path.join(output_file_dir, f"{file_id}.cif")

    try:
        filter_structure_with_timeout(
            filepath,
            output_dir,
            min_cutoff_date=min_cutoff_date,
            max_cutoff_date=max_cutoff_date,
            keep_ligands_in_exclusion_set=keep_ligands_in_exclusion_set,
        )
    except Exception as e:
        print(f"Skipping structure filtering of {filepath} due to: {e}")
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except Exception as e:
                print(
                    f"Failed to remove partially filtered file {output_filepath} due to: {e}. Skipping its removal..."
                )


if __name__ == "__main__":
    # Parse command-line arguments

    parser = argparse.ArgumentParser(
        description="Filter mmCIF files to curate the AlphaFold 3 PDB validation dataset."
    )
    parser.add_argument(
        "-i",
        "--mmcif_assembly_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "unfiltered_assembly_mmcifs"),
        help="Path to the input directory containing `assembly1` mmCIF files to filter.",
    )
    parser.add_argument(
        "-a",
        "--mmcif_asym_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "unfiltered_asym_mmcifs"),
        help="Path to the input directory containing asymmetric unit mmCIF files with which to filter the `assembly1` mmCIF files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "val_mmcifs"),
        help="Path to the output directory in which to store filtered mmCIF dataset files.",
    )
    parser.add_argument(
        "-f",
        "--min_cutoff_date",
        type=lambda t: datetime.strptime(t, "%Y-%m-%d"),
        default=datetime(2021, 10, 1),
        help="Minimum cutoff date for filtering PDB release dates.",
    )
    parser.add_argument(
        "-l",
        "--max_cutoff_date",
        type=lambda t: datetime.strptime(t, "%Y-%m-%d"),
        default=datetime(2023, 1, 13),
        help="Maximum cutoff date for filtering PDB release dates.",
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="Skip filtering of existing output files.",
    )
    parser.add_argument(
        "-e",
        "--keep_ligands_in_exclusion_set",
        action="store_true",
        help="Keep ligands in the exclusion set during filtering.",
    )
    parser.add_argument(
        "-n",
        "--no_workers",
        type=int,
        default=16,
        help="Number of workers to use for filtering.",
    )
    parser.add_argument(
        "-w",
        "--chunksize",
        type=int,
        default=1,
        help="How many files should be distributed to each worker at a time.",
    )
    args = parser.parse_args()

    assert os.path.exists(
        args.mmcif_assembly_dir
    ), f"Input assembly directory {args.mmcif_assembly_dir} does not exist."
    assert os.path.exists(
        args.mmcif_asym_dir
    ), f"Input asymmetric unit directory {args.mmcif_asym_dir} does not exist."

    # Filter structures across all worker processes

    args_tuples = [
        (
            filepath,
            args.output_dir,
            args.min_cutoff_date,
            args.max_cutoff_date,
            args.keep_ligands_in_exclusion_set,
        )
        for filepath in glob.glob(os.path.join(args.mmcif_assembly_dir, "*", "*.cif"))
        if "assembly1" in os.path.basename(filepath)
        and os.path.exists(
            os.path.join(
                os.path.dirname(filepath).replace("unfiltered_assembly", "unfiltered_asym"),
                os.path.basename(filepath).replace("-assembly1", ""),
            )
        )
        and not (
            args.skip_existing
            and os.path.exists(
                os.path.join(
                    args.output_dir,
                    os.path.splitext(os.path.basename(filepath))[0][1:3],
                    f"{os.path.splitext(os.path.basename(filepath))[0]}.cif",
                )
            )
        )
    ]
    process_map(
        filter_structure,
        args_tuples,
        max_workers=args.no_workers,
        chunksize=args.chunksize,
    )
