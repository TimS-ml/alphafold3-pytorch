"""
AlphaFold3 PyTorch Implementation

This module provides a PyTorch implementation of AlphaFold3, a deep learning model for
predicting protein structure and molecular interactions. The package includes all necessary
components for training and inference, including:

- Neural network architectures (attention mechanisms, pairformer, diffusion modules)
- Input processing and feature embedding
- Training utilities and data loaders
- Configuration management
- Command-line and web interfaces

Main Components:
    - Alphafold3: Main model class for structure prediction
    - Trainer: Training loop management
    - AtomInput/PDBInput: Input data structures
    - Various attention and transformer modules
"""

from alphafold3_pytorch.attention import (
    Attention,
    Attend,
    full_pairwise_repr_to_windowed
)

from alphafold3_pytorch.alphafold3 import (
    RelativePositionEncoding,
    SmoothLDDTLoss,
    WeightedRigidAlign,
    MultiChainPermutationAlignment,
    ComputeAlignmentError,
    CentreRandomAugmentation,
    TemplateEmbedder,
    PreLayerNorm,
    AdaptiveLayerNorm,
    ConditionWrapper,
    OuterProductMean,
    MSAPairWeightedAveraging,
    TriangleMultiplication,
    AttentionPairBias,
    TriangleAttention,
    Transition,
    MSAModule,
    PairformerStack,
    DiffusionTransformer,
    DiffusionModule,
    ElucidatedAtomDiffusion,
    InputFeatureEmbedder,
    ConfidenceHead,
    DistogramHead,
    Alphafold3,
    Alphafold3WithHubMixin,
    ConfidenceHeadLogits,
    ComputeRankingScore,
    ComputeModelSelectionScore
)

from alphafold3_pytorch.inputs import (
    register_input_transform,
    AtomInput,
    BatchedAtomInput,
    MoleculeInput,
    Alphafold3Input,
    atom_input_to_file,
    file_to_atom_input,
    pdb_dataset_to_atom_inputs,
    AtomDataset,
    PDBInput,
    PDBDataset,
    maybe_transform_to_atom_input,
    maybe_transform_to_atom_inputs,
    alphafold3_inputs_to_batched_atom_input,
    alphafold3_input_to_biomolecule,
    collate_inputs_to_batched_atom_input,
    pdb_inputs_to_batched_atom_input,
)

from alphafold3_pytorch.trainer import (
    Trainer,
    DataLoader,
)

from alphafold3_pytorch.configs import (
    Alphafold3Config,
    TrainerConfig,
    ConductorConfig,
    create_alphafold3_from_yaml,
    create_trainer_from_yaml,
    create_trainer_from_conductor_yaml
)

from alphafold3_pytorch.utils.model_utils import (
    ExpressCoordinatesInFrame,
    RigidFrom3Points,
    RigidFromReference3Points,
)

from alphafold3_pytorch.cli import cli
from alphafold3_pytorch.app import app

__all__ = [
    Attention,
    Attend,
    RelativePositionEncoding,
    RigidFrom3Points,
    RigidFromReference3Points,
    SmoothLDDTLoss,
    WeightedRigidAlign,
    MultiChainPermutationAlignment,
    ExpressCoordinatesInFrame,
    ComputeAlignmentError,
    CentreRandomAugmentation,
    TemplateEmbedder,
    PreLayerNorm,
    AdaptiveLayerNorm,
    ConditionWrapper,
    OuterProductMean,
    MSAPairWeightedAveraging,
    TriangleMultiplication,
    AttentionPairBias,
    TriangleAttention,
    Transition,
    MSAModule,
    PairformerStack,
    DiffusionTransformer,
    DiffusionModule,
    ElucidatedAtomDiffusion,
    InputFeatureEmbedder,
    ConfidenceHead,
    DistogramHead,
    Alphafold3,
    Alphafold3WithHubMixin,
    Alphafold3Config,
    AtomInput,
    PDBInput,
    Trainer,
    TrainerConfig,
    ConductorConfig,
    create_alphafold3_from_yaml,
    create_trainer_from_yaml,
    create_trainer_from_conductor_yaml,
    pdb_inputs_to_batched_atom_input,
    ComputeRankingScore,
    ConfidenceHeadLogits,
    ComputeModelSelectionScore,
]
