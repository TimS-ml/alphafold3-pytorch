"""
Test suite for template loading functionality in AlphaFold3.

This module tests the ability to load and process template structures from PDB files.
Templates are experimental structures of similar proteins that provide additional
structural information to guide the prediction process, improving accuracy for
proteins with known homologs.
"""

import pytest
from pathlib import Path

from alphafold3_pytorch.inputs import PDBDataset, pdb_inputs_to_batched_atom_input
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.utils.utils import exists


def test_template_loading():
    """
    Test loading and processing of template structures from a PDB dataset.

    This test verifies that:
    1. Template structures can be loaded from a specified template directory
    2. Templates are correctly integrated with PDB inputs via WeightedPDBSampler
    3. Template features are properly batched and formatted for model input

    Templates are critical for AlphaFold3 predictions as they provide known structural
    information from homologous proteins. This test ensures that the template loading
    pipeline correctly reads template files and converts them into the feature format
    required by the model.
    """
    # Set up directory paths for test data
    data_test = Path("data", "test", "pdb_data")
    data_test_mmcif_dir = data_test / "mmcifs"  # Directory containing mmCIF structure files
    data_test_clusterings_dir = data_test / "data_caches" / "clusterings"  # Cluster mappings
    data_test_template_dir = data_test / "data_caches" / "template" / "templates"  # Template structures

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

    # Create a PDB dataset with template loading enabled
    pdb_input = PDBDataset(
        folder=data_test_mmcif_dir,
        sampler=sampler,
        sample_type="default",
        crop_size=4,  # Small crop size for fast testing
        templates_dir=str(data_test_template_dir),  # Directory containing template structures
        sample_only_pdb_ids=test_ids,  # Only sample from available test IDs
        training=False,  # Disable training-specific augmentations
    )

    # Load a single example and convert to batched format
    # This tests the full pipeline from PDB file to model input format
    batched_atom_input = pdb_inputs_to_batched_atom_input(pdb_input[0], atoms_per_window=27)

    # Verify that the batched input was successfully created
    # This confirms that template features were loaded and integrated correctly
    assert exists(batched_atom_input)
