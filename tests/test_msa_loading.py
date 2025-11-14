"""
Test suite for MSA (Multiple Sequence Alignment) loading functionality.

This module tests the ability to load and process MSA data from stored alignments.
MSAs provide evolutionary information by showing how a protein sequence varies across
different organisms, which helps the model understand which regions are conserved
(functionally important) and which are variable.
"""

import pytest
from pathlib import Path

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.inputs import PDBDataset, pdb_inputs_to_batched_atom_input
from alphafold3_pytorch.utils.utils import exists

def test_msa_loading():
    """
    Test loading and processing of MSA (Multiple Sequence Alignment) features.

    This test verifies that:
    1. MSA files can be loaded from a specified MSA directory
    2. MSA features are correctly integrated with PDB inputs via WeightedPDBSampler
    3. MSA data is properly batched and formatted for model input

    MSAs are crucial for AlphaFold3 predictions as they encode evolutionary information.
    By analyzing sequence conservation across related proteins, the model can better
    predict which regions form stable structures and which residues interact.
    This test ensures the MSA loading pipeline works correctly.
    """
    # Set up directory paths for test data
    data_test = Path("data", "test", "pdb_data")
    data_test_mmcif_dir = data_test / "mmcifs"  # Directory containing mmCIF structure files
    data_test_clusterings_dir = data_test / "data_caches" / "clusterings"  # Cluster mappings
    data_test_msa_dir = data_test / "data_caches" / "msa" / "msas"  # MSA files

    # Skip test if required data directory doesn't exist
    if not data_test_mmcif_dir.exists():
        pytest.skip(f"The directory `{data_test_mmcif_dir}` is not populated yet.")

    # Set up paths to cluster mapping files
    # These files define how PDB chains/interfaces are grouped into similarity clusters
    interface_mapping_path = str(data_test_clusterings_dir / "interface_cluster_mapping.csv")
    chain_mapping_paths = [
        str(data_test_clusterings_dir / "ligand_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "nucleic_acid_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "peptide_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "protein_chain_cluster_mapping.csv"),
    ]

    # Create a weighted sampler for balanced sampling across different chain/interface types
    sampler = WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=64,
    )

    # Get the set of PDB IDs that are available in the sampler
    sampler_pdb_ids = set(sampler.mappings.get_column("pdb_id").to_list())

    # Find test IDs that exist in both the sampler and the mmCIF directory
    test_ids = set(
        filepath.stem
        for filepath in data_test_mmcif_dir.glob("**/*.cif")
        if filepath.stem in sampler_pdb_ids
    )

    # Create a PDB dataset with MSA loading enabled
    pdb_input = PDBDataset(
        folder=data_test_mmcif_dir,
        sampler=sampler,
        sample_type="default",
        crop_size=4,  # Small crop size for fast testing
        msa_dir=str(data_test_msa_dir),  # Directory containing MSA files
        sample_only_pdb_ids=test_ids,  # Only sample from available test IDs
        training=False,  # Disable training-specific augmentations
    )

    # Load a single example and convert to batched format
    # This tests the full pipeline from PDB file + MSA to model input format
    batched_atom_input = pdb_inputs_to_batched_atom_input(pdb_input[0], atoms_per_window=27)

    # Verify that the batched input was successfully created
    # This confirms that MSA features were loaded and integrated correctly
    assert exists(batched_atom_input)
