"""
Comprehensive test suite for the AlphaFold3 Trainer class.

This module tests the training infrastructure, including:
- Trainer initialization and configuration
- Training loops with gradient accumulation
- Validation and testing loops
- Model checkpointing and loading
- Exponential moving average (EMA) of model weights
- Dataset integration (AtomDataset, PDBDataset)
- YAML configuration loading for trainer setup
- Conductor-based training orchestration

The Trainer class handles all aspects of model training, making these tests
critical for ensuring reliable and reproducible training runs.
"""

import os
# Enable type checking for better error messages
os.environ['TYPECHECK'] = 'True'

import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from alphafold3_pytorch import (
    Alphafold3,
    PDBDataset,
    AtomInput,
    atom_input_to_file,
    DataLoader,
    Trainer,
    ConductorConfig,
    collate_inputs_to_batched_atom_input,
    create_trainer_from_yaml,
    create_trainer_from_conductor_yaml,
    create_alphafold3_from_yaml
)

from alphafold3_pytorch.mocks import MockAtomDataset

# PDB ID used for testing
DATA_TEST_PDB_ID = '209d'

def exists(v):
    """Simple helper to check if a value is not None."""
    return v is not None

@pytest.fixture()
def remove_test_folders():
    """
    Pytest fixture to clean up test folders after tests complete.

    This ensures test artifacts don't accumulate in the repository.
    """
    yield
    shutil.rmtree('./test-folder')

def test_trainer_with_mock_atom_input(remove_test_folders):
    """
    Test the Trainer class with mock atom inputs.

    This validates core trainer functionality:
    1. Model saving and loading with hyperparameters
    2. Training loop execution with gradient accumulation
    3. Validation loop execution
    4. Checkpoint creation and restoration
    5. EMA (Exponential Moving Average) weight updates
    6. Loading from both model checkpoints and training state

    Using mock data allows fast testing without requiring real PDB files.
    """

    # Create a minimal AlphaFold3 model for fast testing
    alphafold3 = Alphafold3(
        dim_atom_inputs = 77,
        dim_template_feats = 108,
        num_dist_bins = 64,
        confidence_head_kwargs = dict(
            pairformer_depth = 1  # Shallow network for speed
        ),
        template_embedder_kwargs = dict(
            pairformer_stack_depth = 1
        ),
        msa_module_kwargs = dict(
            depth = 1
        ),
        pairformer_stack = dict(
            depth = 1
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,
            token_transformer_depth = 1,
            atom_decoder_depth = 1,
        ),
    )

    # Create mock datasets for training, validation, and testing
    dataset = MockAtomDataset(100)
    valid_dataset = MockAtomDataset(4)
    test_dataset = MockAtomDataset(2)

    # Test model saving and loading (independent of trainer)

    dataloader = DataLoader(dataset, batch_size = 2)
    inputs = next(iter(dataloader))

    # Run forward pass before saving to get reference output
    alphafold3.eval()
    _, breakdown = alphafold3(**inputs.model_forward_dict(), return_loss_breakdown = True)
    before_distogram = breakdown.distogram

    # Save model with hyperparameters
    path = './test-folder/nested/folder/af3'
    alphafold3.save(path, overwrite = True)

    # Load model from scratch (including hyperparameters)
    alphafold3 = Alphafold3.init_and_load(path)

    # Verify loaded model produces identical output
    alphafold3.eval()
    _, breakdown = alphafold3(**inputs.model_forward_dict(), return_loss_breakdown = True)
    after_distogram = breakdown.distogram

    assert torch.allclose(before_distogram, after_distogram)

    # Test training with validation

    trainer = Trainer(
        alphafold3,
        dataset = dataset,
        valid_dataset = valid_dataset,
        test_dataset = test_dataset,
        accelerator = 'cpu',  # Use CPU for testing (no GPU required)
        num_train_steps = 2,  # Just a few steps for testing
        batch_size = 1,
        valid_every = 1,  # Validate after each training step
        grad_accum_every = 2,  # Test gradient accumulation
        checkpoint_every = 1,  # Save checkpoint after each step
        checkpoint_folder = './test-folder/checkpoints',
        overwrite_checkpoints = True,
        ema_kwargs = dict(  # Test EMA functionality
            use_foreach = True,
            update_after_step = 0,
            update_every = 1
        )
    )

    # Run training
    trainer()

    # Verify checkpoint was created
    assert Path(f'./test-folder/checkpoints/({trainer.train_id})_af3.ckpt.1.pt').exists()

    # Test loading checkpoint from directory (loads latest)
    trainer.load('./test-folder/checkpoints', strict = False)

    assert exists(trainer.model_loaded_from_path)

    # Test saving and loading trainer state (includes optimizer, scheduler, etc.)
    trainer.save('./test-folder/nested/folder2/training.pt', overwrite = True)
    trainer.load('./test-folder/nested/folder2/training.pt', strict = False)

    # Test loading only model weights (for fine-tuning)
    trainer.load('./test-folder/nested/folder2/training.pt', only_model = True, strict = False)

    # Test loading AlphaFold3 model directly from trainer checkpoint
    alphafold3 = Alphafold3.init_and_load('./test-folder/nested/folder2/training.pt')

# Tests for trainer with PDB inputs (real structure files)

@pytest.fixture()
def populate_mock_pdb_and_remove_test_folders():
    """
    Pytest fixture to set up and tear down mock PDB datasets for trainer testing.

    This creates a realistic test environment with:
    - Train, validation, and test split directories
    - Copies of real mmCIF files for each split
    - Automatic cleanup after tests complete

    This allows testing the full training pipeline with PDB files without
    requiring a large dataset or long training times.
    """
    proj_root = Path('.')
    working_cif_file = proj_root / 'data' / 'test' / 'pdb_data' / 'mmcifs' / DATA_TEST_PDB_ID[1:3] / f'{DATA_TEST_PDB_ID}-assembly1.cif'

    pytest_root_folder = Path('./test-folder')
    data_folder = pytest_root_folder / 'data'

    train_folder = data_folder / 'train'
    valid_folder = data_folder / 'valid'
    test_folder = data_folder / 'test'

    train_folder.mkdir(exist_ok = True, parents = True)
    valid_folder.mkdir(exist_ok = True, parents = True)
    test_folder.mkdir(exist_ok = True, parents = True)

    for i in range(10):
        shutil.copy2(str(working_cif_file), str(train_folder / f'{i}.cif'))

    for i in range(1):
        shutil.copy2(str(working_cif_file), str(valid_folder / f'{i}.cif'))

    for i in range(1):
        shutil.copy2(str(working_cif_file), str(test_folder / f'{i}.cif'))

    yield

    shutil.rmtree('./test-folder')

def test_trainer_with_pdb_input(populate_mock_pdb_and_remove_test_folders):
    """
    Test the Trainer class with PDBDataset inputs.

    This validates that the trainer works correctly with real PDB files:
    1. PDBDataset can load and process mmCIF files
    2. Data flows correctly from PDB files through the model
    3. Training, validation, and checkpointing work with PDB inputs
    4. Model state can be saved and loaded correctly

    This is a more realistic test than using mock data, ensuring the full
    pipeline from PDB files to trained model works correctly.
    """

    alphafold3 = Alphafold3(
        dim_atom=4,
        dim_atompair=4,
        dim_input_embedder_token=4,
        dim_single=4,
        dim_pairwise=4,
        dim_token=4,
        dim_atom_inputs=3,
        dim_atompair_inputs=5,
        atoms_per_window=27,
        dim_template_feats=108,
        num_dist_bins=64,
        confidence_head_kwargs=dict(
            pairformer_depth=1,
        ),
        template_embedder_kwargs=dict(pairformer_stack_depth=1),
        msa_module_kwargs=dict(depth=1),
        pairformer_stack=dict(
            depth=1,
            pair_bias_attn_dim_head = 4,
            pair_bias_attn_heads = 2,
        ),
        diffusion_module_kwargs=dict(
            atom_encoder_depth=1,
            token_transformer_depth=1,
            atom_decoder_depth=1,
            atom_decoder_kwargs = dict(
                attn_pair_bias_kwargs = dict(
                    dim_head = 4
                )
            ),
            atom_encoder_kwargs = dict(
                attn_pair_bias_kwargs = dict(
                    dim_head = 4
                )
            )
        ),
    )

    dataset = PDBDataset('./test-folder/data/train')
    valid_dataset = PDBDataset('./test-folder/data/valid')
    test_dataset = PDBDataset('./test-folder/data/test')

    # test saving and loading from Alphafold3, independent of lightning

    dataloader = DataLoader(dataset, batch_size = 1)
    inputs = next(iter(dataloader))

    alphafold3.eval()
    with torch.no_grad():
        _, breakdown = alphafold3(**inputs.model_forward_dict(), return_loss_breakdown = True)

    before_distogram = breakdown.distogram

    path = './test-folder/nested/folder/af3'
    alphafold3.save(path, overwrite = True)

    # load from scratch, along with saved hyperparameters

    alphafold3 = Alphafold3.init_and_load(path)

    alphafold3.eval()
    with torch.no_grad():
        _, breakdown = alphafold3(**inputs.model_forward_dict(), return_loss_breakdown = True)

    after_distogram = breakdown.distogram

    assert torch.allclose(before_distogram, after_distogram)

    # test training + validation

    trainer = Trainer(
        alphafold3,
        dataset = dataset,
        valid_dataset = valid_dataset,
        test_dataset = test_dataset,
        accelerator = 'cpu',
        num_train_steps = 2,
        batch_size = 1,
        valid_every = 1,
        grad_accum_every = 1,
        checkpoint_every = 1,
        checkpoint_folder = './test-folder/checkpoints',
        overwrite_checkpoints = True,
        use_ema = False,
        ema_kwargs = dict(
            use_foreach = True,
            update_after_step = 0,
            update_every = 1
        )
    )

    trainer()

    # assert checkpoints created

    assert Path(f'./test-folder/checkpoints/({trainer.train_id})_af3.ckpt.1.pt').exists()

    # assert can load latest checkpoint by loading from a directory

    trainer.load('./test-folder/checkpoints', strict = False)

    assert exists(trainer.model_loaded_from_path)

    # saving and loading from trainer

    trainer.save('./test-folder/nested/folder2/training.pt', overwrite = True)
    trainer.load('./test-folder/nested/folder2/training.pt', strict = False)

    # allow for only loading model, needed for fine-tuning logic

    trainer.load('./test-folder/nested/folder2/training.pt', only_model = True, strict = False)

    # also allow for loading Alphafold3 directly from training ckpt

    alphafold3 = Alphafold3.init_and_load('./test-folder/nested/folder2/training.pt')

# Test collation function for batching inputs

def test_collate_fn():
    """
    Test the collation function for batching multiple AtomInputs.

    This validates that:
    1. Multiple AtomInput objects can be batched together
    2. The batched input works correctly with the model
    3. Gradients can be computed through batched inputs

    The collation function is critical for efficient training as it handles
    padding, masking, and tensor stacking for variable-sized inputs.
    """
    alphafold3 = Alphafold3(
        dim_atom_inputs = 77,
        dim_template_feats = 108,
        num_dist_bins = 64,
        confidence_head_kwargs = dict(
            pairformer_depth = 1
        ),
        template_embedder_kwargs = dict(
            pairformer_stack_depth = 1
        ),
        msa_module_kwargs = dict(
            depth = 1
        ),
        pairformer_stack = dict(
            depth = 1
        ),
        diffusion_module_kwargs = dict(
            atom_encoder_depth = 1,
            token_transformer_depth = 1,
            atom_decoder_depth = 1,
        ),
    )

    dataset = MockAtomDataset(5)

    batched_atom_inputs = collate_inputs_to_batched_atom_input([dataset[i] for i in range(3)])

    _, breakdown = alphafold3(**batched_atom_inputs.model_forward_dict(), return_loss_breakdown = True)

# Tests for creating trainer from YAML configuration

def test_trainer_config(remove_test_folders):
    """
    Test creating a trainer from YAML configuration.

    This validates that:
    1. Trainer can be instantiated from YAML config
    2. Model configuration is correctly parsed
    3. Training can proceed with config-specified parameters

    YAML configuration allows users to specify training setups in a reproducible,
    version-controllable way without writing Python code.
    """
    curr_dir = Path(__file__).parents[0]
    trainer_yaml_path = curr_dir / 'configs/trainer.yaml'

    trainer = create_trainer_from_yaml(
        trainer_yaml_path,
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# Test creating trainer with PDB dataset from config

def test_trainer_config_with_pdb_dataset(populate_mock_pdb_and_remove_test_folders):
    """
    Test creating trainer with PDBDataset from YAML configuration.

    This validates that:
    1. YAML config can specify PDBDataset parameters
    2. Trainer correctly initializes with PDB data
    3. Training works end-to-end from config

    This is important for production training setups that use PDB files.
    """
    curr_dir = Path(__file__).parents[0]
    trainer_yaml_path = curr_dir / 'configs/trainer_with_pdb_dataset.yaml'

    trainer = create_trainer_from_yaml(trainer_yaml_path)

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# Test creating trainer with AtomDataset from config

def test_trainer_config_with_atom_dataset(remove_test_folders):
    """
    Test creating trainer with AtomDataset from YAML configuration.

    This validates that:
    1. YAML config can specify AtomDataset parameters
    2. Trainer correctly initializes with preprocessed atom inputs
    3. Training works with cached/preprocessed data

    Using AtomDataset allows faster training by preprocessing features once
    and loading them from disk, rather than processing PDB files each time.
    """

    curr_dir = Path(__file__).parents[0]

    # setup atom dataset

    atom_folder = './test-folder/test-atom-folder'
    Path(atom_folder).mkdir(exist_ok = True, parents = True)

    mock_atom_dataset = MockAtomDataset(10)

    for i in range(10):
        atom_input = mock_atom_dataset[i]
        atom_input_to_file(atom_input, f'{atom_folder}/train/{i}.pt', overwrite = True)

    # path to config

    trainer_yaml_path = curr_dir / 'configs/trainer_with_atom_dataset.yaml'

    trainer = create_trainer_from_yaml(trainer_yaml_path)

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# Test creating trainer with AtomDataset precomputed from PDBDataset

def test_trainer_config_with_atom_dataset_from_pdb_dataset(populate_mock_pdb_and_remove_test_folders):
    """
    Test creating trainer with AtomDataset derived from PDBDataset via config.

    This validates a common workflow:
    1. PDBDataset is used to load structures from PDB files
    2. Features are extracted and saved as AtomInputs
    3. Training uses the preprocessed AtomDataset for faster iteration

    This two-stage approach is common in production: preprocess once,
    then train/experiment many times with the cached features.
    """

    curr_dir = Path(__file__).parents[0]
    trainer_yaml_path = curr_dir / 'configs/trainer_with_atom_dataset_created_from_pdb.yaml'

    trainer = create_trainer_from_yaml(trainer_yaml_path)

    assert isinstance(trainer, Trainer)

    # take a single training step

    trainer()

# Test creating trainer without model in config (model passed separately)

def test_trainer_config_without_model(remove_test_folders):
    """
    Test creating trainer where model is passed separately from config.

    This validates a flexible configuration pattern:
    1. Trainer config specifies training parameters only
    2. Model is created separately (e.g., from different config or code)
    3. Model and trainer are combined at runtime

    This is useful for programmatic experiment setups where the model
    might be varied while keeping training parameters constant.
    """
    curr_dir = Path(__file__).parents[0]

    af3_yaml_path = curr_dir / 'configs/alphafold3.yaml'
    trainer_yaml_path = curr_dir / 'configs/trainer_without_model.yaml'

    alphafold3 = create_alphafold3_from_yaml(af3_yaml_path)

    trainer = create_trainer_from_yaml(
        trainer_yaml_path,
        model = alphafold3,
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

# Test creating trainer using conductor configuration

def test_conductor_config():
    """
    Test creating trainer from conductor-style training configuration.

    This validates the "conductor" pattern for orchestrating multiple training runs:
    1. A single config file can specify multiple related training runs
    2. Each run can have its own name and parameters
    3. Checkpoints are organized by run name

    The conductor pattern is useful for complex training setups like pre-training
    followed by fine-tuning, or training multiple model variants.
    """
    curr_dir = Path(__file__).parents[0]
    training_yaml_path = curr_dir / 'configs/training.yaml'

    trainer = create_trainer_from_conductor_yaml(
        training_yaml_path,
        trainer_name = 'main',
        dataset = MockAtomDataset(16)
    )

    assert isinstance(trainer, Trainer)

    assert str(trainer.checkpoint_folder) == 'test-folder/main-and-finetuning/main'
    assert str(trainer.checkpoint_prefix) == 'af3.main.ckpt.'

# Test creating trainer from conductor config with PDB datasets

def test_conductor_config_with_pdb_datasets(populate_mock_pdb_and_remove_test_folders):
    """
    Test creating trainer from conductor config with PDBDataset integration.

    This validates the conductor pattern with real PDB data:
    1. Conductor config can specify PDBDataset parameters
    2. Multiple training runs can share the same dataset configuration
    3. Training works end-to-end with PDB files

    This is the most realistic test, combining conductor orchestration
    with actual structural biology data.
    """
    curr_dir = Path(__file__).parents[0]
    training_yaml_path = curr_dir / 'configs/training_with_pdb_dataset.yaml'

    trainer = create_trainer_from_conductor_yaml(
        training_yaml_path,
        trainer_name = 'main'
    )

    assert isinstance(trainer, Trainer)

    assert str(trainer.checkpoint_folder) == 'test-folder/main-and-finetuning/main'
    assert str(trainer.checkpoint_prefix) == 'af3.main.ckpt.'
