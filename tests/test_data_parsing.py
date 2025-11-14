"""This file prepares unit tests for data parsing (e.g., mmCIF file I/O)."""

import glob
import os
import random

import pytest

from alphafold3_pytorch.common.biomolecule import _from_mmcif_object
from alphafold3_pytorch.data import mmcif_parsing

# Enable type checking for more detailed error messages
os.environ["TYPECHECK"] = "True"

# Known problematic PDB IDs that have issues in their mmCIF files
# These are excluded from testing to avoid false failures

ERRONEOUS_PDB_IDS = [
    "3tob"  # NOTE: At residue index 97, ALY and LYS are assigned the same residue ID of 118 by the authors.
            # This violates mmCIF conventions and cannot be parsed correctly
]


@pytest.mark.parametrize(
    "mmcif_dir",
    [
        os.path.join("data", "pdb_data", "unfiltered_assembly_mmcifs"),
        os.path.join("data", "pdb_data", "unfiltered_asym_mmcifs"),
        os.path.join("data", "pdb_data", "train_mmcifs"),
    ],
)
@pytest.mark.parametrize(
    "complex_id", ["100d", "1k7a", "384d", "4xij", "6adq", "7a4d", "7akd", "8a3j"]
)
def test_mmcif_object_parsing(mmcif_dir: str, complex_id: str) -> None:
    """
    Test parsing of specific PDB mmCIF files and conversion to Biomolecule objects.

    This validates that:
    1. mmCIF files can be read and parsed correctly
    2. Parsing results can be converted to internal Biomolecule representation
    3. The resulting Biomolecule contains valid atomic coordinates

    mmCIF (macromolecular Crystallographic Information File) is the standard
    format for structural biology data. Correct parsing is essential for loading
    experimental structures for training and validation.

    :param mmcif_dir: A directory containing PDB mmCIF files.
    :param complex_id: The PDB ID of the complex to be tested.
    """
    # Construct file path (mmCIF files are organized by middle two characters)
    complex_filepath = os.path.join(mmcif_dir, complex_id[1:3], f"{complex_id}.cif")
    # Handle assembly files (may have suffixes like -assembly1.cif)
    complex_filepaths = glob.glob(f"{complex_filepath[:-4]}*.cif")

    if not complex_filepaths:
        pytest.skip(f"File '{complex_filepath}' does not exist.")

    # Randomly select one assembly if multiple exist
    complex_filepath = random.choice(complex_filepaths)
    with open(complex_filepath, "r") as f:
        mmcif_string = f.read()

    # Parse mmCIF string using author chain IDs and residue numbers
    # (as opposed to entity/label IDs used internally by PDB)
    parsing_result = mmcif_parsing.parse(
        file_id=complex_id,
        mmcif_string=mmcif_string,
        auth_chains=True,  # Use author-assigned chain IDs
        auth_residues=True,  # Use author-assigned residue numbers
    )

    # Check if parsing succeeded
    if parsing_result.mmcif_object is None:
        print(f"Failed to parse file '{complex_filepath}'.")
        raise list(parsing_result.errors.values())[0]
    else:
        # Convert parsed mmCIF object to Biomolecule
        try:
            biomol = _from_mmcif_object(parsing_result.mmcif_object)
        except Exception as e:
            # Skip files with insertion codes (alternate residue positions)
            # These require special handling not yet implemented
            if "mmCIF contains an insertion code" in str(e):
                pytest.skip(f"File '{complex_filepath}' contains an insertion code.")
            else:
                raise e
        # Verify that the Biomolecule contains atomic coordinates
        assert (
            len(biomol.atom_positions) > 0
        ), f"Failed to parse file '{complex_filepath}' into a `Biomolecule` object."


@pytest.mark.parametrize(
    "mmcif_dir",
    [
        os.path.join("data", "pdb_data", "unfiltered_assembly_mmcifs"),
        os.path.join("data", "pdb_data", "unfiltered_asym_mmcifs"),
        os.path.join("data", "pdb_data", "train_mmcifs"),
    ],
)
@pytest.mark.parametrize("num_random_complexes_to_parse", [500])
@pytest.mark.parametrize("random_seed", [1])
def test_random_mmcif_objects_parsing(
    mmcif_dir: str,
    num_random_complexes_to_parse: int,
    random_seed: int,
) -> None:
    """
    Test parsing of a random sample of mmCIF files for robustness.

    This validates that the mmCIF parser can handle a diverse set of structures
    from the PDB without failing. By testing hundreds of random files, we ensure
    the parser is robust to various edge cases and formatting variations found
    in real PDB data.

    This test is particularly important because:
    - PDB files vary widely in complexity (small proteins to large complexes)
    - Different deposition tools create slightly different mmCIF formats
    - Historical files may have non-standard formatting

    :param mmcif_dir: A directory containing PDB mmCIF files.
    :param num_random_complexes_to_parse: The number of random complexes to parse.
    :param random_seed: The random seed for reproducibility.
    """
    # Set random seed for reproducible test results
    random.seed(random_seed)

    if not os.path.exists(mmcif_dir):
        pytest.skip(f"Directory '{mmcif_dir}' does not exist.")

    # Track parsing failures for detailed error reporting
    parsing_errors = []
    failed_complex_indices = []
    failed_random_complex_filepaths = []

    # Get list of subdirectories containing mmCIF files
    # (PDB files are organized by middle two characters of PDB ID)
    mmcif_subdirs = [
        os.path.join(mmcif_dir, subdir)
        for subdir in os.listdir(mmcif_dir)
        if os.path.isdir(os.path.join(mmcif_dir, subdir))
        and os.listdir(os.path.join(mmcif_dir, subdir))
    ]

    # Test parsing on random sample of complexes
    for complex_index in range(num_random_complexes_to_parse):
        # Randomly select a subdirectory
        random_mmcif_subdir = random.choice(mmcif_subdirs)

        # Get all mmCIF files in that subdirectory
        mmcif_subdir_files = [
            os.path.join(random_mmcif_subdir, mmcif_subdir_file)
            for mmcif_subdir_file in os.listdir(random_mmcif_subdir)
            if os.path.isfile(os.path.join(random_mmcif_subdir, mmcif_subdir_file))
            and mmcif_subdir_file.endswith(".cif")
        ]

        # Randomly select one file
        random_complex_filepath = random.choice(mmcif_subdir_files)
        complex_id = os.path.splitext(os.path.basename(random_complex_filepath))[0]

        # Skip if file doesn't exist (shouldn't happen, but be defensive)
        if not os.path.exists(random_complex_filepath):
            print(f"File '{random_complex_filepath}' does not exist.")
            continue

        # Skip known problematic PDB IDs
        if any(
            id in os.path.basename(random_complex_filepath)[:4].lower() for id in ERRONEOUS_PDB_IDS
        ):
            continue

        # Read and parse mmCIF file
        with open(random_complex_filepath, "r") as f:
            mmcif_string = f.read()

        parsing_result = mmcif_parsing.parse(
            file_id=complex_id,
            mmcif_string=mmcif_string,
            auth_chains=True,
            auth_residues=True,
        )

        # Record failures for later reporting
        if parsing_result.mmcif_object is None:
            parsing_errors.append(list(parsing_result.errors.values())[0])
            failed_complex_indices.append(complex_index)
            failed_random_complex_filepaths.append(random_complex_filepath)
        else:
            # Try to convert to Biomolecule
            try:
                biomol = _from_mmcif_object(parsing_result.mmcif_object)
            except Exception as e:
                # Skip insertion codes (not yet supported)
                if "mmCIF contains an insertion code" in str(e):
                    continue
                else:
                    # Record other errors
                    parsing_errors.append(e)
                    failed_complex_indices.append(complex_index)
                    failed_random_complex_filepaths.append(random_complex_filepath)
                    continue

            # Verify Biomolecule has atom positions
            if len(biomol.atom_positions) == 0:
                parsing_errors.append(
                    AssertionError(
                        f"Failed to parse file '{random_complex_filepath}' into a `Biomolecule` object."
                    )
                )
                failed_complex_indices.append(complex_index)
                failed_random_complex_filepaths.append(random_complex_filepath)

    # Report all failures together for easier debugging
    if parsing_errors:
        print(
            f"Failed to parse {len(parsing_errors)} files at indices {failed_complex_indices}: '{failed_random_complex_filepaths}'."
        )
        for error in parsing_errors:
            print(error)
        raise error
