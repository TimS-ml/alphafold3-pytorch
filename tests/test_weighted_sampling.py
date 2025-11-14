"""
Test suite for weighted PDB sampling strategies.

This module tests the WeightedPDBSampler class, which provides balanced sampling
across different molecular types and interface configurations. Weighted sampling is
critical for training on diverse structural data, ensuring the model sees a balanced
mix of:
- Different molecule types (proteins, nucleic acids, ligands, etc.)
- Different interface types (protein-protein, protein-DNA, etc.)
- Different structural complexity levels

The sampler uses cluster mappings to avoid over-representing similar structures
while ensuring adequate coverage of all molecular types.
"""

import os
import shutil
from pathlib import Path

import polars as pl
import pytest

from torch.utils.data import Sampler

from alphafold3_pytorch.data.weighted_pdb_sampler import (
    WeightedPDBSampler
)

from alphafold3_pytorch import (
    create_trainer_from_yaml
)

# PDB ID used for testing
DATA_TEST_PDB_ID = '209d'

# Path to test data containing cluster mappings
TEST_FOLDER = Path("data", "test", "pdb_data", "data_caches", "clusterings")

# Path to interface cluster mappings (groups similar interfaces together)
INTERFACE_MAPPING_PATH = str(TEST_FOLDER / "interface_cluster_mapping.csv")

# Paths to chain cluster mappings for different molecule types
# These files group similar chains together to avoid redundancy in sampling
CHAIN_MAPPING_PATHS = [
    str(TEST_FOLDER / "ligand_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "nucleic_acid_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "peptide_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "protein_chain_cluster_mapping.csv"),
]

@pytest.fixture
def sampler():
    """
    Create a WeightedPDBSampler fixture for testing.

    This sampler will be used across multiple tests to verify sampling behavior.
    The fixture pattern allows us to reuse the same sampler setup.
    """
    return WeightedPDBSampler(
        chain_mapping_paths=CHAIN_MAPPING_PATHS,
        interface_mapping_path=INTERFACE_MAPPING_PATH,
        batch_size=4,
    )


def test_sample(sampler: Sampler):
    """
    Test basic sampling functionality.

    This validates that the sampler can generate batches of the requested size.
    The sampler should return exactly the number of samples requested, drawing
    from the available PDB IDs according to their weights.
    """
    assert len(sampler.sample(4)) == 4, "The sampled batch size does not match the expected size."


def test_cluster_based_sample(sampler: Sampler):
    """
    Test cluster-based sampling functionality.

    This validates that the sampler can perform cluster-aware sampling, which
    ensures diverse coverage of different structural clusters. Cluster-based
    sampling prevents over-sampling of similar structures by treating each
    cluster as a unit during sampling.
    """
    assert (
        len(sampler.cluster_based_sample(4)) == 4
    ), "The cluster-based sampled batch size does not match the expected size."

# End-to-end tests with weighted PDB sampler and trainer

@pytest.fixture()
def populate_mock_pdb_and_remove_test_folders():
    """
    Pytest fixture to set up and tear down mock PDB data for testing.

    This fixture:
    1. Creates a temporary test folder structure
    2. Copies test mmCIF files to simulate a PDB dataset
    3. Yields control back to the test
    4. Cleans up all created files after the test completes

    The cleanup happens automatically even if the test fails, ensuring
    no leftover test artifacts.
    """
    # Get path to test mmCIF file
    proj_root = Path('.')
    working_cif_file = proj_root / 'data' / 'test' / 'pdb_data' / 'mmcifs' / DATA_TEST_PDB_ID[1:3] / f'{DATA_TEST_PDB_ID}-assembly1.cif'

    # Create temporary test folder structure
    pytest_root_folder = Path('./test-folder')
    data_folder = pytest_root_folder / 'data'
    train_folder = data_folder / 'train'
    train_folder.mkdir(exist_ok = True, parents = True)

    # Collect all PDB IDs from cluster mapping files
    pdb_ids = []

    for path in [*CHAIN_MAPPING_PATHS, INTERFACE_MAPPING_PATH]:
        dataset = pl.read_csv(path)
        pdb_ids.extend(list(dataset.get_column('pdb_id')))

    # Copy test mmCIF file for each PDB ID (using same file for all to save space)
    for pdb_id in {*pdb_ids}:
        shutil.copy2(str(working_cif_file), str(train_folder / f'{pdb_id}.cif'))

    # Yield control back to test
    yield

    # Cleanup: remove test folder after test completes
    shutil.rmtree('./test-folder')

def test_weighted_sampling_from_trainer_config(populate_mock_pdb_and_remove_test_folders):
    """
    Test end-to-end training with weighted PDB sampling.

    This validates that:
    1. A trainer can be created from YAML configuration
    2. The trainer can use WeightedPDBSampler with PDBDataset
    3. Training can proceed with weighted sampling of diverse structures

    This is an integration test that ensures all components work together
    correctly in a realistic training scenario.
    """
    # Create trainer from YAML config that includes weighted sampling
    trainer = create_trainer_from_yaml('./tests/configs/trainer_with_pdb_dataset_and_weighted_sampling.yaml')

    # Run training (will execute for configured number of steps)
    trainer()
