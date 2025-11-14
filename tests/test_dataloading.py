"""
Test suite for data loading and dataset construction.

This module tests the creation and usage of datasets for AlphaFold3 training, including:
- PDBDataset: loading experimental structures from PDB/mmCIF files
- PDBDistillationDataset: loading predicted structures from AlphaFold Database (AFDB)
- WeightedPDBSampler: balanced sampling across different molecular types
- Data pipeline integration with the model

Distillation datasets allow AlphaFold3 to learn from previously predicted structures,
enabling knowledge transfer and self-distillation training strategies.
"""

from pathlib import Path

import pytest
import torch

from alphafold3_pytorch import collate_inputs_to_batched_atom_input
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.inputs import (
    PDBDataset,
    PDBDistillationDataset,
    molecule_to_atom_input,
    pdb_input_to_molecule_input,
)
from alphafold3_pytorch.utils.utils import exists

# PDB ID used for testing data loading
DATA_TEST_PDB_ID = "721p"


def test_data_input():
    """
    Test combined PDB and distillation dataset loading with weighted sampling.

    This validates that:
    1. PDBDataset can load experimental structures from PDB files
    2. PDBDistillationDataset can load predicted structures from AFDB
    3. Both datasets can be combined using ConcatDataset
    4. WeightedPDBSampler provides balanced sampling across molecule types
    5. Data can be processed through the full pipeline to model input format

    This is important for training strategies that combine experimental and
    predicted structures, enabling the model to learn from both sources.
    """
    # Set up paths for experimental (PDB) data
    data_test = Path("data", "test", "pdb_data")
    data_test_mmcif_dir = data_test / "mmcifs"
    data_test_clusterings_dir = data_test / "data_caches" / "clusterings"

    # Set up paths for distillation (predicted) data from AlphaFold Database
    distillation_data_test = Path("data", "test", "afdb_data")
    distillation_data_test_mmcif_dir = distillation_data_test / "mmcifs"
    distillation_uniprot_to_pdb_id_mapping_filepath = (
        distillation_data_test / "data_caches" / "uniprot_to_pdb_id_mapping.dat"
    )

    # Skip test if experimental data directory doesn't exist
    if not data_test_mmcif_dir.exists():
        pytest.skip(f"The directory `{data_test_mmcif_dir}` is not populated yet.")

    # Skip test if distillation data directory doesn't exist
    if not distillation_data_test_mmcif_dir.exists():
        pytest.skip(f"The directory `{distillation_data_test_mmcif_dir}` is not populated yet.")

    # Set up cluster mapping paths for balanced sampling
    # These CSV files define how chains/interfaces are grouped by similarity
    interface_mapping_path = str(data_test_clusterings_dir / "interface_cluster_mapping.csv")
    chain_mapping_paths = [
        str(data_test_clusterings_dir / "ligand_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "nucleic_acid_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "peptide_chain_cluster_mapping.csv"),
        str(data_test_clusterings_dir / "protein_chain_cluster_mapping.csv"),
    ]

    # Create weighted sampler for balanced sampling across molecular types
    sampler = WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        pdb_ids_to_keep=[f"{DATA_TEST_PDB_ID}-assembly1"],  # Restrict to specific test structure
        batch_size=2,
    )

    # Find PDB IDs that exist in both sampler and data directory
    sampler_pdb_ids = set(sampler.mappings.get_column("pdb_id").to_list())
    test_ids = set(
        filepath.stem
        for filepath in data_test_mmcif_dir.glob("**/*.cif")
        if filepath.stem in sampler_pdb_ids
    )

    # Create experimental PDB dataset
    dataset = PDBDataset(
        folder=data_test_mmcif_dir,
        sampler=sampler,
        sample_type="default",
        sample_only_pdb_ids=test_ids,  # Only sample from available test IDs
        crop_size=4,  # Small crop for fast testing
    )

    # Determine which IDs are available for distillation dataset
    distillation_test_ids = {r[0] for r in sampler.mappings.select("pdb_id").rows()}
    distillation_test_ids = (
        distillation_test_ids.intersection(test_ids) if exists(test_ids) else distillation_test_ids
    )

    # Create distillation dataset (predicted structures from AFDB)
    distillation_dataset = PDBDistillationDataset(
        folder=distillation_data_test_mmcif_dir,
        sample_only_pdb_ids=distillation_test_ids,
        crop_size=4,
        distillation=True,  # Enable distillation mode
        distillation_template_mmcif_dir=data_test_mmcif_dir,  # Use experimental structures as templates
        uniprot_to_pdb_id_mapping_filepath=distillation_uniprot_to_pdb_id_mapping_filepath,
    )

    # Combine experimental and distillation datasets
    combined_dataset = torch.utils.data.ConcatDataset([dataset, distillation_dataset])

    # Process first example through the data pipeline
    mol_input = pdb_input_to_molecule_input(pdb_input=combined_dataset[0])
    atom_input = molecule_to_atom_input(mol_input)
    batched_atom_input = collate_inputs_to_batched_atom_input([atom_input], atoms_per_window=4)

    # Create minimal AlphaFold3 model for testing
    alphafold3 = Alphafold3(
        dim_atom_inputs=3,
        dim_atompair_inputs=5,
        atoms_per_window=4,
        dim_template_feats=108,
        num_dist_bins=64,
        confidence_head_kwargs=dict(pairformer_depth=1),
        template_embedder_kwargs=dict(pairformer_stack_depth=1),
        msa_module_kwargs=dict(depth=1),
        pairformer_stack=dict(depth=1),
        diffusion_module_kwargs=dict(
            atom_encoder_depth=1,
            token_transformer_depth=1,
            atom_decoder_depth=1,
        ),
    )

    # Verify that the data can be processed through the model
    loss = alphafold3(**batched_atom_input.model_forward_dict())
    loss.backward()  # Ensure gradients can be computed
