"""
UniProt Structure Prediction Reducer to PDB Dataset.

This script filters and processes AlphaFold Database (AFDB) structure predictions,
retaining only those predictions that correspond to proteins with known PDB structures.
It extracts compressed prediction files, updates their metadata with PDB release dates,
and organizes them by UniProt accession ID for downstream training data augmentation.

The script performs the following operations:
1. Maps UniProt accession IDs to corresponding PDB entries
2. Filters AFDB predictions to keep only those with PDB matches
3. Decompresses and processes prediction files
4. Updates mmCIF metadata with appropriate PDB release dates
5. Organizes outputs by UniProt accession ID

Usage:
    python reduce_uniprot_predictions_to_pdb.py

Input:
    - Compressed AFDB prediction files (*.cif.gz)
    - UniProt-to-PDB ID mapping file
    - Filtered PDB mmCIF training files

Output:
    - Decompressed and metadata-updated AFDB prediction files organized by UniProt ID

The script uses multiprocessing for efficient parallel processing of large prediction datasets.
"""

import glob
import gzip
import os
import shutil
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

import polars as pl
import timeout_decorator
from beartype.typing import Dict, Set, Tuple
from tqdm import tqdm

from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.utils.data_utils import extract_mmcif_metadata_field

# Maximum time (in seconds) allowed for processing a single archive file
# This prevents the script from hanging on problematic files
PROCESS_ARCHIVE_MAX_SECONDS_PER_INPUT = 15


@timeout_decorator.timeout(PROCESS_ARCHIVE_MAX_SECONDS_PER_INPUT, use_signals=True)
def process_archive_with_timeout(archive_info: Tuple[str, Dict[str, Set[str]], str, str]):
    """
    Process a single AFDB archive file with timeout protection.

    This function extracts a compressed AlphaFold Database prediction file, finds the
    maximum PDB release date among all associated PDB structures, and updates the
    prediction's mmCIF metadata with this release date. This ensures that AFDB predictions
    are treated with appropriate temporal information for training/validation splits.

    The function operates under a timeout constraint to prevent hanging on problematic files.

    Args:
        archive_info: A tuple containing:
            - archive (str): Path to the compressed prediction archive file (.cif.gz)
            - uniprot_to_pdb_id_mapping (Dict[str, Set[str]]): Mapping from UniProt accession IDs to sets of PDB IDs
            - input_pdb_dir (str): Directory containing filtered PDB mmCIF training files
            - output_dir (str): Directory where processed files will be written

    Returns:
        None

    Raises:
        timeout_decorator.TimeoutError: If processing exceeds PROCESS_ARCHIVE_MAX_SECONDS_PER_INPUT
        Exception: For various parsing or I/O errors during processing

    Note:
        The function extracts the UniProt accession ID from the archive filename, looks up
        associated PDB entries, finds the latest release date, and injects this date into
        the prediction's mmCIF metadata for proper temporal tracking.
    """
    # Unpack the input tuple
    archive, uniprot_to_pdb_id_mapping, input_pdb_dir, output_dir = archive_info

    # Extract the UniProt accession ID from the archive filename
    # Filename format: AF-{accession_id}-F1-model_v4.cif.gz
    archive_accession_id = os.path.splitext(os.path.basename(archive))[0].split("-")[1]
    output_subdir = os.path.join(output_dir, archive_accession_id)

    # Initialize with epoch date; will be updated to the maximum PDB release date
    pdb_release_date = datetime(1970, 1, 1)

    # Iterate through all PDB IDs associated with this UniProt accession
    for pdb_id in list(uniprot_to_pdb_id_mapping[archive_accession_id]):
        pdb_id = pdb_id.lower()
        # PDB files are organized in subdirectories based on the middle two characters
        # e.g., 1abc is stored in directory "ab"
        pdb_group_code = pdb_id[1:3]
        pdb_filepath = os.path.join(input_pdb_dir, pdb_group_code, f"{pdb_id}-assembly1.cif")

        # Check if the corresponding PDB file exists
        if os.path.exists(pdb_filepath):
            try:
                # Parse the mmCIF file to extract metadata
                mmcif_object = mmcif_parsing.parse_mmcif_object(
                    filepath=pdb_filepath, file_id=f"{pdb_id}-assembly1.cif"
                )
                # Extract the release date from the mmCIF metadata
                mmcif_release_date = extract_mmcif_metadata_field(mmcif_object, "release_date")

                # Keep track of the maximum (most recent) release date
                # This ensures the AFDB prediction is dated to when all related structures were public
                pdb_release_date = max(
                    pdb_release_date, datetime.strptime(mmcif_release_date, "%Y-%m-%d")
                )
            except Exception as e:
                print(
                    f"An error occurred while processing PDB ID {pdb_id} associated with {pdb_filepath}: {e}. Skipping this prediction..."
                )
                return

    # If no valid PDB release date was found, skip this prediction
    if pdb_release_date == datetime(1970, 1, 1):
        print(
            f"Could not find PDB release date for {archive_accession_id}. Skipping this prediction..."
        )
        return

    # Create output subdirectory for this UniProt accession
    os.makedirs(output_subdir, exist_ok=True)

    # Decompress the archive file to the output directory
    output_file = os.path.join(output_subdir, os.path.basename(archive).removesuffix(".gz"))
    with gzip.open(archive, "rb") as f_in, open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    # Update the mmCIF file with the PDB release date
    # This inserts a new revision date entry after the revision_date header line
    with open(output_file, "r") as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            # Find the revision date header in the mmCIF file
            if "_pdbx_audit_revision_history.revision_date" in line:
                new_lines.append(line)
                # Insert a new revision entry with the PDB release date
                # Format: "Description" major_rev minor_rev alt_ver date
                new_lines.append(f'"Structure model" 1 0 1 {pdb_release_date.date()} \n')
            else:
                new_lines.append(line)

    # Write the updated content back to the file
    with open(output_file, "w") as f:
        f.writelines(new_lines)


def process_archive(archive_info: Tuple[str, Dict[str, Set[str]], str, str]):
    """
    Wrapper function for archive processing with error handling.

    This function wraps the timeout-protected archive processing function and catches
    any exceptions (including timeouts) to ensure that processing continues even if
    individual files fail. Failed files are logged but don't stop the overall pipeline.

    Args:
        archive_info: A tuple containing:
            - archive (str): Path to the compressed prediction archive file
            - uniprot_to_pdb_id_mapping (Dict[str, Set[str]]): UniProt-to-PDB ID mapping
            - input_pdb_dir (str): Directory with PDB mmCIF files
            - output_dir (str): Output directory for processed files

    Returns:
        None

    Note:
        Errors are printed to stdout but don't raise exceptions, allowing batch
        processing to continue.
    """
    try:
        # Attempt to process the archive with timeout protection
        process_archive_with_timeout(archive_info)
    except Exception as e:
        # Log the error and continue with the next archive
        print(
            f"Processing of archive info {archive_info} took too long and was terminated due to: {e}. Skipping this prediction..."
        )


def filter_pdb_files(
    input_archive_dir: str,
    input_pdb_dir: str,
    output_dir: str,
    uniprot_to_pdb_id_mapping_filepath: str,
):
    """
    Filter and process AFDB predictions based on PDB associations.

    This function is the main entry point for filtering AlphaFold Database predictions.
    It reads the UniProt-to-PDB mapping, identifies AFDB predictions that correspond to
    proteins with known PDB structures, and processes these predictions in parallel.

    The workflow is:
    1. Load UniProt-to-PDB ID mappings from file
    2. Scan input directory for AFDB prediction archives
    3. Filter predictions to keep only those with PDB associations
    4. Process filtered predictions in parallel using multiprocessing
    5. Extract, update metadata, and organize outputs by UniProt ID

    Args:
        input_archive_dir: Directory containing compressed AFDB prediction files (*.cif.gz)
        input_pdb_dir: Directory containing filtered PDB mmCIF training files
        output_dir: Directory where processed prediction files will be written
        uniprot_to_pdb_id_mapping_filepath: Path to TSV file mapping UniProt IDs to PDB IDs

    Returns:
        None

    Note:
        Uses multiprocessing with 12 worker processes for parallel processing.
        Progress is tracked using tqdm progress bars.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the UniProt-to-PDB ID mapping file
    # This file contains tab-separated values: UniProt_ID, Database (PDB), PDB_ID
    uniprot_to_pdb_id_mapping_df = pl.read_csv(
        uniprot_to_pdb_id_mapping_filepath,
        has_header=False,
        separator="\t",
        new_columns=["uniprot_accession", "database", "pdb_id"],
    )
    # Remove the database column as it's always "PDB" after filtering
    uniprot_to_pdb_id_mapping_df.drop_in_place("database")

    # Convert the DataFrame to a dictionary mapping UniProt IDs to sets of PDB IDs
    # Using a defaultdict with sets allows efficient lookup and deduplication
    uniprot_to_pdb_id_mapping = defaultdict(set)
    for row in uniprot_to_pdb_id_mapping_df.iter_rows():
        uniprot_to_pdb_id_mapping[row[0]].add(row[1])

    # Identify which archive files to keep based on PDB associations
    archives_to_keep = defaultdict(set)
    archive_file_pattern = os.path.join(input_archive_dir, "*model_v4.cif.gz")

    # Scan all AFDB prediction archives in the input directory
    for archive_file in tqdm(
        glob.glob(archive_file_pattern),
        desc="Filtering prediction files by PDB ID association",
    ):
        # Extract the UniProt accession ID from the archive filename
        # Format: AF-{accession_id}-F1-model_v4.cif.gz
        archive_accession_id = os.path.splitext(os.path.basename(archive_file))[0].split("-")[1]

        # Keep only archives that have associated PDB structures
        if archive_accession_id in uniprot_to_pdb_id_mapping:
            archives_to_keep[archive_accession_id].add(archive_file)

    # Initialize multiprocessing pool with 12 worker processes
    # This allows parallel processing of multiple archives simultaneously
    pool = Pool(processes=12)

    # Prepare arguments for each worker process
    # Each worker will receive: (archive_path, mapping_dict, pdb_dir, output_dir)
    archive_infos = [
        (archive, uniprot_to_pdb_id_mapping, input_pdb_dir, output_dir)
        for accession_id in archives_to_keep
        for archive in archives_to_keep[accession_id]
    ]

    # Process all archives in parallel with progress tracking
    # Using imap_unordered for better performance (order doesn't matter)
    for _ in tqdm(
        pool.imap_unordered(process_archive, archive_infos),
        total=len(archive_infos),
        desc="Processing archives",
    ):
        pass

    # Clean up the multiprocessing pool
    pool.close()
    pool.join()


if __name__ == "__main__":
    # Define input and output directories for the prediction filtering pipeline
    # Input: Directory containing unfiltered AFDB prediction archives
    input_archive_dir = os.path.join("data", "afdb_data", "unfiltered_train_mmcifs")

    # Directory containing filtered PDB training mmCIF files
    input_pdb_dir = os.path.join("data", "pdb_data", "train_mmcifs")

    # Output directory for processed AFDB predictions
    output_dir = os.path.join("data", "afdb_data", "train_mmcifs")

    # Path to the UniProt-to-PDB ID mapping file
    uniprot_to_pdb_id_mapping_filepath = os.path.join(
        "data", "afdb_data", "data_caches", "uniprot_to_pdb_id_mapping.dat"
    )

    # Execute the filtering and processing pipeline
    filter_pdb_files(
        input_archive_dir, input_pdb_dir, output_dir, uniprot_to_pdb_id_mapping_filepath
    )
