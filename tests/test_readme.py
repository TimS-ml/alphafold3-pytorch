"""
Test suite validating code examples from the project README.

This module ensures that the usage examples provided in the project documentation
actually work as advertised. These tests serve two purposes:
1. Validate that the API works as documented
2. Catch breaking changes that would invalidate the README examples
"""

def test_readme1():
    """
    Test the first README example: low-level API with manual tensor construction.

    This example demonstrates how to use AlphaFold3 with manually constructed tensors,
    giving users full control over the input features. This is useful for advanced users
    who want to customize the input features or integrate AlphaFold3 into custom pipelines.

    The test validates:
    - Model initialization with basic configuration
    - Training with manually constructed atom and molecule features
    - Inference/sampling to generate predicted atomic coordinates
    """
    import torch
    from alphafold3_pytorch import Alphafold3
    from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

    # Create AlphaFold3 model with standard input dimensions
    alphafold3 = Alphafold3(
        dim_atom_inputs = 77,  # Number of atom-level input features
        dim_template_feats = 108  # Number of template features per residue pair
    )

    # Create mock input tensors for testing
    # In practice, these would be derived from actual protein/molecule data

    seq_len = 16  # Number of residues/molecules in the sequence

    # Molecule-level features
    molecule_atom_indices = torch.randint(0, 2, (2, seq_len)).long()  # Index of representative atom per molecule
    molecule_atom_lens = torch.full((2, seq_len), 2).long()  # Number of atoms per molecule

    # Calculate atom sequence length and offsets for indexing
    atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
    atom_offsets = exclusive_cumsum(molecule_atom_lens)  # Cumulative sum for atom indexing

    # Atom-level features: positions, types, bonds, etc.
    atom_inputs = torch.randn(2, atom_seq_len, 77)  # Features for each atom
    atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)  # Pairwise atom features

    # Additional molecular features
    additional_molecule_feats = torch.randint(0, 2, (2, seq_len, 5))  # Residue IDs, chain IDs, etc.
    additional_token_feats = torch.randn(2, seq_len, 33)  # Token-level features
    is_molecule_types = torch.randint(0, 2, (2, seq_len, 5)).bool()  # One-hot molecule type (protein, DNA, RNA, etc.)
    is_molecule_mod = torch.randint(0, 2, (2, seq_len, 4)).bool()  # Molecule modifications
    molecule_ids = torch.randint(0, 32, (2, seq_len))  # Unique ID for each molecule

    # Template features: structural information from homologous proteins
    template_feats = torch.randn(2, 2, seq_len, seq_len, 108)  # Features from 2 template structures
    template_mask = torch.ones((2, 2)).bool()  # Mask indicating which templates are valid

    # MSA (Multiple Sequence Alignment) features for evolutionary information
    msa = torch.randn(2, 7, seq_len, 32)  # 7 aligned sequences
    msa_mask = torch.ones((2, 7)).bool()  # Mask indicating which MSA rows are valid

    additional_msa_feats = torch.randn(2, 7, seq_len, 2)  # Additional MSA features

    # Training labels (required for training, omitted during inference)

    atom_pos = torch.randn(2, atom_seq_len, 3)  # Ground truth 3D atomic coordinates

    distogram_atom_indices = molecule_atom_lens - 1  # Indices for distance prediction

    distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))  # Distance bins for distogram loss
    resolved_labels = torch.randint(0, 2, (2, atom_seq_len))  # Binary labels for atom resolution

    # Offset indices to account for variable molecule sizes
    # This ensures indices correctly point to atoms across the batched sequence
    distogram_atom_indices += atom_offsets
    molecule_atom_indices += atom_offsets

    # Training: compute loss with all features and labels
    loss = alphafold3(
        num_recycling_steps = 2,  # Number of times to recycle predictions through the network
        atom_inputs = atom_inputs,
        atompair_inputs = atompair_inputs,
        molecule_ids = molecule_ids,
        molecule_atom_lens = molecule_atom_lens,
        additional_molecule_feats = additional_molecule_feats,
        additional_msa_feats = additional_msa_feats,
        additional_token_feats = additional_token_feats,
        is_molecule_types = is_molecule_types,
        is_molecule_mod = is_molecule_mod,
        msa = msa,
        msa_mask = msa_mask,
        templates = template_feats,
        template_mask = template_mask,
        atom_pos = atom_pos,  # Ground truth coordinates for computing loss
        distogram_atom_indices = distogram_atom_indices,
        molecule_atom_indices = molecule_atom_indices,
        distance_labels = distance_labels,
        resolved_labels = resolved_labels
    )

    # Backpropagate to update model weights
    loss.backward()

    # Inference: generate predicted structure without labels
    # In practice, this would be done after training the model

    sampled_atom_pos = alphafold3(
        num_recycling_steps = 4,  # More recycling steps for better predictions
        num_sample_steps = 16,  # Number of diffusion sampling steps
        atom_inputs = atom_inputs,
        atompair_inputs = atompair_inputs,
        molecule_ids = molecule_ids,
        molecule_atom_lens = molecule_atom_lens,
        additional_molecule_feats = additional_molecule_feats,
        additional_msa_feats = additional_msa_feats,
        additional_token_feats = additional_token_feats,
        is_molecule_types = is_molecule_types,
        is_molecule_mod = is_molecule_mod,
        msa = msa,
        msa_mask = msa_mask,
        templates = template_feats,
        template_mask = template_mask
        # Note: no labels provided during inference
    )

def test_readme2():
    """
    Test the second README example: high-level API with Alphafold3Input.

    This example demonstrates the user-friendly API that automatically handles
    feature extraction from simple sequence strings. This is the recommended
    approach for most users as it abstracts away the complexity of manual
    feature engineering.

    The test validates:
    - Creating inputs from simple protein sequences
    - Automatic feature extraction and batching
    - Training and inference with the simplified API
    """
    import torch
    from alphafold3_pytorch import Alphafold3, Alphafold3Input

    # Simple two-residue protein: Alanine-Glycine
    contrived_protein = 'AG'

    # Mock atomic positions for training
    # These would come from experimental structures (e.g., PDB files) in real use
    mock_atompos = [
        torch.randn(5, 3),   # alanine has 5 non-hydrogen atoms (N, CA, C, O, CB)
        torch.randn(4, 3)    # glycine has 4 non-hydrogen atoms (N, CA, C, O)
    ]

    # Create training input with ground truth atomic positions
    train_alphafold3_input = Alphafold3Input(
        proteins = [contrived_protein],  # Protein sequence(s)
        atom_pos = mock_atompos  # Ground truth positions for training
    )

    # Create evaluation input without positions (for prediction)
    eval_alphafold3_input = Alphafold3Input(
        proteins = [contrived_protein]  # Only sequence needed for inference
    )

    # Initialize AlphaFold3 model with reduced dimensions for fast testing
    alphafold3 = Alphafold3(
        dim_atom_inputs = 3,  # Reduced atom feature dimension
        dim_atompair_inputs = 5,  # Reduced atompair feature dimension
        atoms_per_window = 27,  # Maximum atoms per attention window
        dim_template_feats = 108,  # Template feature dimension
        num_molecule_mods = 0,  # No molecule modifications in this example
        confidence_head_kwargs = dict(
            pairformer_depth = 1  # Shallow network for testing
        ),
        template_embedder_kwargs = dict(
            pairformer_stack_depth = 1  # Shallow template processor
        ),
        msa_module_kwargs = dict(
            depth = 1  # Shallow MSA module
        ),
        pairformer_stack = dict(
            depth = 2  # Two layers in main pairformer
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,  # Shallow atom encoder
            token_transformer_depth = 1,  # Shallow token transformer
            atom_decoder_depth = 1,  # Shallow atom decoder
        )
    )

    # Training: forward pass with ground truth positions
    loss = alphafold3.forward_with_alphafold3_inputs([train_alphafold3_input])
    loss.backward()  # Compute gradients

    # Inference: generate predicted atomic positions
    alphafold3.eval()  # Switch to evaluation mode (disables dropout, etc.)
    sampled_atom_pos = alphafold3.forward_with_alphafold3_inputs(eval_alphafold3_input)

    # Verify output shape: 1 batch × (5 + 4) atoms × 3 coordinates
    assert sampled_atom_pos.shape == (1, (5 + 4), 3)
