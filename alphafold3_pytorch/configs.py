"""
Configuration Module for AlphaFold3 Training and Inference

This module provides Pydantic-based configuration classes for managing AlphaFold3
model parameters, training settings, and dataset configurations. It enables:

- Type-safe configuration with automatic validation
- YAML-based configuration file loading
- Easy serialization/deserialization of settings
- Structured creation of model and trainer instances

Main Features:
    - Alphafold3Config: Model architecture and hyperparameter configuration
    - TrainerConfig: Training loop settings and optimization parameters
    - DatasetConfig: Dataset loading and preprocessing configuration
    - ConductorConfig: Multi-stage training pipeline management
    - YAML file parsing with nested configuration support

The configuration system is designed to be flexible and extensible, allowing
users to specify all training parameters in YAML files or programmatically.

Example:
    >>> # Load config from YAML and create model
    >>> model = create_alphafold3_from_yaml('config.yaml')
    >>>
    >>> # Load config and create trainer
    >>> trainer = create_trainer_from_yaml('training_config.yaml')
"""

from __future__ import annotations

from alphafold3_pytorch.tensor_typing import typecheck
from beartype.typing import Callable, List, Dict, Literal

from alphafold3_pytorch.alphafold3 import Alphafold3

from alphafold3_pytorch.inputs import (
    AtomDataset,
    PDBDataset,
    pdb_dataset_to_atom_inputs
)

from alphafold3_pytorch.trainer import (
    Trainer,
    Dataset,
    Fabric,
    Optimizer,
    LRScheduler
)

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler

import yaml
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from pydantic.types import (
    FilePath,
    DirectoryPath
)

from lightning.fabric.loggers import TensorBoardLogger

# Helper functions

def exists(v):
    """
    Check if a value exists (is not None).

    Args:
        v: Any value to check

    Returns:
        bool: True if v is not None, False otherwise
    """
    return v is not None

@typecheck
def safe_deep_get(
    d: dict,
    dotpath: str | List[str],  # dotpath notation, so accessing {'a': {'b': {'c': 1}}} would be "a.b.c"
    default = None
):
    """
    Safely retrieve a nested value from a dictionary using dot notation.

    This function allows accessing nested dictionary values using a string path
    like "a.b.c" instead of d['a']['b']['c'], with safe handling of missing keys.

    Args:
        d: Dictionary to retrieve value from
        dotpath: Dot-separated path string (e.g., "model.layers.attention")
                 or list of keys (e.g., ['model', 'layers', 'attention'])
        default: Value to return if path doesn't exist (default: None)

    Returns:
        The value at the specified path, or default if path doesn't exist

    Example:
        >>> config = {'model': {'dim': 512, 'layers': {'depth': 12}}}
        >>> safe_deep_get(config, 'model.layers.depth')
        12
        >>> safe_deep_get(config, 'model.missing.key', default=0)
        0
    """
    if isinstance(dotpath, str):
        dotpath = dotpath.split('.')

    for key in dotpath:
        if (
            not isinstance(d, dict) or \
            key not in d
        ):
            return default

        d = d[key]

    return d

@typecheck
def yaml_config_path_to_dict(
    path: str | Path
) -> dict:
    """
    Load and parse a YAML configuration file into a dictionary.

    Args:
        path: Path to the YAML configuration file

    Returns:
        dict: Parsed configuration dictionary

    Raises:
        AssertionError: If file doesn't exist, can't be parsed, or isn't a dict

    Example:
        >>> config = yaml_config_path_to_dict('config.yaml')
        >>> print(config['model']['dim_atom'])
        128
    """
    if isinstance(path, str):
        path = Path(path)

    assert path.is_file(), f'cannot find {str(path)}'

    with open(str(path), 'r') as f:
        maybe_config_dict = yaml.safe_load(f)

    assert exists(maybe_config_dict), f'unable to parse yaml config at {str(path)}'
    assert isinstance(maybe_config_dict, dict), 'yaml config file is not a dictionary'

    return maybe_config_dict

# Base Pydantic classes for constructing AlphaFold3 and trainer from config files

class BaseModelWithExtra(BaseModel):
    """
    Base Pydantic model that allows extra fields and uses enum values.

    This base class provides common configuration for all config models,
    allowing flexibility in adding extra fields without validation errors.
    """
    model_config = ConfigDict(
        extra = 'allow',
        use_enum_values = True,
    )

class Alphafold3Config(BaseModelWithExtra):
    """
    Configuration for AlphaFold3 model architecture and hyperparameters.

    This class encapsulates all settings needed to instantiate an AlphaFold3 model,
    including dimensions for various representations, loss weights, and training
    parameters. Defaults are based on the AlphaFold3 paper specifications.

    Attributes:
        dim_atom_inputs: Input feature dimension per atom
        dim_template_feats: Dimension of template features
        dim_template_model: Hidden dimension in template embedder
        atoms_per_window: Number of atoms per attention window (for efficiency)
        dim_atom: Hidden dimension for atom representations
        dim_atompair_inputs: Input feature dimension per atom pair
        dim_atompair: Hidden dimension for atom pair representations
        dim_input_embedder_token: Token dimension in input feature embedder
        dim_single: Dimension of single (node) representations in trunk
        dim_pairwise: Dimension of pairwise (edge) representations in trunk
        dim_token: Dimension of token representations in main architecture
        ignore_index: Index to ignore in loss computation (default: -1)
        num_dist_bins: Number of bins for distance predictions
        num_plddt_bins: Number of bins for pLDDT (predicted lDDT) scores
        num_pde_bins: Number of bins for predicted distance error
        num_pae_bins: Number of bins for predicted aligned error
        sigma_data: Noise scale for diffusion model
        diffusion_num_augmentations: Number of data augmentations during training
        loss_confidence_weight: Weight for confidence head loss
        loss_distogram_weight: Weight for distogram prediction loss
        loss_diffusion_weight: Weight for diffusion/denoising loss

    Example:
        >>> config = Alphafold3Config(
        ...     dim_atom_inputs=77,
        ...     dim_template_feats=108,
        ...     # ... other required parameters
        ... )
        >>> model = config.create_instance()
    """
    dim_atom_inputs: int
    dim_template_feats: int
    dim_template_model: int
    atoms_per_window: int
    dim_atom: int
    dim_atompair_inputs: int
    dim_atompair: int
    dim_input_embedder_token: int
    dim_single: int
    dim_pairwise: int
    dim_token: int
    ignore_index: int = -1
    num_dist_bins: int | None
    num_plddt_bins: int
    num_pde_bins: int
    num_pae_bins: int
    sigma_data: int | float
    diffusion_num_augmentations: int
    loss_confidence_weight: int | float
    loss_distogram_weight: int | float
    loss_diffusion_weight: int | float

    @classmethod
    @typecheck
    def from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ):
        """
        Load AlphaFold3 configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file
            dotpath: Optional nested path within the YAML file (e.g., 'model.alphafold3')

        Returns:
            Alphafold3Config: Configuration instance loaded from YAML

        Example:
            >>> config = Alphafold3Config.from_yaml_file('configs/model.yaml')
            >>> # Or for nested configs:
            >>> config = Alphafold3Config.from_yaml_file('configs/full.yaml', 'model.alphafold3')
        """
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'config not found at path {".".join(dotpath)}'

        return cls(**config_dict)

    def create_instance(self) -> Alphafold3:
        """
        Create an AlphaFold3 model instance from this configuration.

        Returns:
            Alphafold3: Initialized AlphaFold3 model with configured parameters

        Example:
            >>> config = Alphafold3Config(dim_atom_inputs=77, ...)
            >>> model = config.create_instance()
            >>> predictions = model(inputs)
        """
        alphafold3 = Alphafold3(**self.model_dump())
        return alphafold3

    @classmethod
    def create_instance_from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ) -> Alphafold3:
        """
        Load configuration from YAML and create an AlphaFold3 model instance.

        This is a convenience method that combines from_yaml_file() and create_instance().

        Args:
            path: Path to the YAML configuration file
            dotpath: Optional nested path within the YAML file

        Returns:
            Alphafold3: Initialized AlphaFold3 model

        Example:
            >>> model = Alphafold3Config.create_instance_from_yaml_file('config.yaml')
        """
        af3_config = cls.from_yaml_file(path, dotpath)
        return af3_config.create_instance()

class WeightedPDBSamplerConfig(BaseModelWithExtra):
    """
    Configuration for weighted PDB (Protein Data Bank) sampling during training.

    This config manages sampling strategies for training on PDB datasets,
    allowing prioritization of certain structure types or interfaces.

    Attributes:
        chain_mapping_paths: List of file paths containing chain mapping information
        interface_mapping_path: Path to file containing interface mapping data
    """
    chain_mapping_paths: List[FilePath]
    interface_mapping_path: FilePath

    def create_instance(self, batch_size: int):
        """
        Create a WeightedPDBSampler instance with specified batch size.

        Args:
            batch_size: Number of samples per batch

        Returns:
            WeightedPDBSampler: Initialized sampler instance
        """
        return WeightedPDBSampler(**{
            'batch_size': batch_size,
            **self.model_dump()
        })

class DatasetConfig(BaseModelWithExtra):
    """
    Configuration for dataset loading and preprocessing.

    This class configures how training, validation, and test datasets are loaded,
    including support for both PDB and pre-processed atom datasets.

    Attributes:
        dataset_type: Type of dataset - 'pdb' for PDB files or 'atom' for preprocessed atom data
        train_folder: Directory containing training data
        valid_folder: Directory containing validation data (optional)
        test_folder: Directory containing test data (optional)
        convert_pdb_to_atom: Whether to convert PDB files to atom inputs on-the-fly
        pdb_to_atom_kwargs: Additional arguments for PDB to atom conversion
        train_weighted_sampler: Optional weighted sampler config for biased sampling
        kwargs: Additional dataset-specific keyword arguments

    Example:
        >>> dataset_config = DatasetConfig(
        ...     dataset_type='pdb',
        ...     train_folder='data/train',
        ...     valid_folder='data/valid'
        ... )
    """
    dataset_type: Literal['pdb', 'atom'] = 'pdb'
    train_folder: DirectoryPath
    valid_folder: DirectoryPath | None = None
    test_folder: DirectoryPath | None = None
    convert_pdb_to_atom: bool = False
    pdb_to_atom_kwargs: dict = dict()
    train_weighted_sampler: WeightedPDBSamplerConfig | None = None
    kwargs: dict = dict()

class TrainerConfig(BaseModelWithExtra):
    """
    Configuration for the AlphaFold3 training loop.

    This class encapsulates all parameters needed to set up and run a training
    session, including optimization settings, checkpointing, logging, and datasets.

    Attributes:
        model: Optional AlphaFold3 model configuration
        num_train_steps: Total number of training steps
        batch_size: Number of samples per batch
        grad_accum_every: Gradient accumulation steps (for larger effective batch size)
        valid_every: Run validation every N steps
        ema_decay: Exponential moving average decay for model weights
        lr: Learning rate
        clip_grad_norm: Maximum gradient norm for clipping
        accelerator: Hardware accelerator type ('cpu', 'gpu', 'tpu', etc.)
        checkpoint_prefix: Prefix for checkpoint file names
        checkpoint_every: Save checkpoint every N steps
        checkpoint_folder: Directory to save checkpoints
        overwrite_checkpoints: Whether to overwrite existing checkpoints
        dataset_config: Configuration for dataset loading
        use_tensorboard: Whether to use TensorBoard logging
        tensorboard_log_dir: Directory for TensorBoard logs
        logger_kwargs: Additional arguments for logger configuration

    Example:
        >>> trainer_config = TrainerConfig(
        ...     num_train_steps=100000,
        ...     batch_size=32,
        ...     lr=1e-4,
        ...     checkpoint_folder='checkpoints'
        ... )
    """
    model: Alphafold3Config | None = None
    num_train_steps: int
    batch_size: int
    grad_accum_every: int
    valid_every: int
    ema_decay: float
    lr: float
    clip_grad_norm: int | float
    accelerator: str
    checkpoint_prefix: str
    checkpoint_every: int
    checkpoint_folder: str
    overwrite_checkpoints: bool
    dataset_config: DatasetConfig | None = None
    use_tensorboard: bool = True
    tensorboard_log_dir: str = './logs'
    logger_kwargs: dict = dict()

    @classmethod
    @typecheck
    def from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ):
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'config not found at path {".".join(dotpath)}'

        return cls(**config_dict)

    def create_instance(
        self,
        dataset: Dataset | None = None,
        model: Alphafold3 | None = None,
        fabric: Fabric | None = None,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        valid_dataset: Dataset | None = None,
        map_dataset_input_fn: Callable | None = None,
    ) -> Trainer:
        """
        Create a Trainer instance from this configuration.

        This method constructs a complete training pipeline including model, datasets,
        optimizer, and logging. Parameters can be passed explicitly or loaded from
        the configuration's dataset_config.

        Args:
            dataset: Training dataset (overrides config if provided)
            model: AlphaFold3 model (overrides config if provided)
            fabric: Lightning Fabric instance for distributed training
            test_dataset: Test dataset (optional)
            optimizer: Optimizer instance (optional, uses default if None)
            scheduler: Learning rate scheduler (optional)
            valid_dataset: Validation dataset (optional)
            map_dataset_input_fn: Function to map dataset items to model inputs

        Returns:
            Trainer: Configured trainer instance ready for training

        Example:
            >>> config = TrainerConfig.from_yaml_file('training_config.yaml')
            >>> trainer = config.create_instance()
            >>> trainer.train()
        """

        trainer_kwargs = self.model_dump(
            exclude = {
                'dataset_config',
                'use_tensorboard',
                'tensorboard_log_dir',
                'logger_kwargs'
            }
        )

        assert exists(self.model) ^ exists(model), 'either model is available on the trainer config, or passed in when creating the instance, but not both or neither'

        # handle model

        if exists(self.model):
            alphafold3 = self.model.create_instance()
        else:
            alphafold3 = model

        # handle dataset

        if exists(dataset):
            trainer_kwargs.update(dataset = dataset)

        if exists(valid_dataset):
            trainer_kwargs.update(valid_dataset = dataset)

        if exists(test_dataset):
            trainer_kwargs.update(test_dataset = dataset)

        if exists(self.dataset_config):
            dataset_config = self.dataset_config

            dataset_type = dataset_config.dataset_type
            dataset_kwargs = dataset_config.kwargs

            convert_pdb_to_atom = dataset_config.convert_pdb_to_atom
            pdb_to_atom_kwargs = dataset_config.pdb_to_atom_kwargs

            if convert_pdb_to_atom:
                assert dataset_type == 'pdb', 'must be `pdb` dataset_type if `convert_pdb_to_atom` is set to True'

            if dataset_type == 'pdb':
                dataset_klass = PDBDataset
            elif dataset_type == 'atom':
                dataset_klass = AtomDataset
            else:
                raise ValueError(f'unhandled dataset_type {dataset_type}')

            # create dataset for train, valid, and test

            for trainer_kwarg_key, config_key in (('dataset', 'train_folder'), ('valid_dataset', 'valid_folder'), ('test_dataset', 'test_folder')):
                folder = getattr(dataset_config, config_key, None)

                if not exists(folder):
                    continue

                assert trainer_kwarg_key not in trainer_kwargs

                dataset = dataset_klass(folder, **dataset_kwargs)

                if convert_pdb_to_atom:
                    dataset = pdb_dataset_to_atom_inputs(dataset, return_atom_dataset = True, **pdb_to_atom_kwargs)

                trainer_kwargs.update(**{trainer_kwarg_key: dataset})

            # handle weighted pdb sampling

            if exists(dataset_config.train_weighted_sampler):
                sampler = dataset_config.train_weighted_sampler.create_instance(batch_size = self.batch_size)

                trainer_kwargs.update(train_sampler = sampler)

        assert 'dataset' in trainer_kwargs, 'dataset is absent - dataset_type must be specified along with train folders (pdb for now), or the Dataset instance must be passed in'

        # handle loggers

        loggers = []

        if self.use_tensorboard:
            loggers.append(TensorBoardLogger(self.tensorboard_log_dir, **self.logger_kwargs))

        # handle rest

        trainer_kwargs.update(dict(
            model = alphafold3,
            fabric = fabric,
            test_dataset = test_dataset,
            optimizer = optimizer,
            scheduler = scheduler,
            valid_dataset = valid_dataset,
            map_dataset_input_fn = map_dataset_input_fn,
            loggers = loggers
        ))

        trainer = Trainer(**trainer_kwargs)
        return trainer

    @classmethod
    def create_instance_from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = [],
        **kwargs
    ) -> Trainer:

        trainer_config = cls.from_yaml_file(path, dotpath)
        return trainer_config.create_instance(**kwargs)

# Conductor config
# Manages multiple training stages for main training and various finetuning phases

class ConductorConfig(BaseModelWithExtra):
    """
    Configuration for multi-stage training pipeline (conductor pattern).

    This class orchestrates multiple training stages, such as pre-training followed
    by fine-tuning stages. It ensures consistent model usage across stages and
    manages checkpoint organization.

    Attributes:
        model: Optional AlphaFold3 model configuration shared across all training stages
        checkpoint_folder: Root directory for all checkpoints
        checkpoint_prefix: Prefix prepended to all checkpoint names
        training_order: Ordered list of training stage names to execute
        training: Dictionary mapping stage names to their TrainerConfig

    Example:
        >>> config = ConductorConfig(
        ...     model=Alphafold3Config(...),
        ...     checkpoint_folder='checkpoints',
        ...     checkpoint_prefix='af3_',
        ...     training_order=['pretrain', 'finetune'],
        ...     training={
        ...         'pretrain': TrainerConfig(...),
        ...         'finetune': TrainerConfig(...)
        ...     }
        ... )
        >>> trainer = config.create_instance('pretrain')
        >>> trainer.train()
        >>> # Then move to next stage
        >>> trainer = config.create_instance('finetune')
        >>> trainer.train()
    """
    model: Alphafold3Config | None = None
    checkpoint_folder: str
    checkpoint_prefix: str
    training_order: List[str]
    training: Dict[str, TrainerConfig]

    @model_validator(mode = 'after')
    def check_valid_conductor_order(self) -> 'ConductorConfig':
        """
        Validate that training_order contains exactly the keys in training dict.

        This ensures that all defined training stages are referenced in the execution order,
        and no undefined stages are referenced.

        Returns:
            ConductorConfig: Validated configuration instance

        Raises:
            ValueError: If training_order doesn't match training keys
        """
        training_order = set(self.training_order)
        trainer_names = set(self.training.keys())

        if training_order != trainer_names:
            raise ValueError('`training_order` needs to contain all the keys (trainer name) under the `training` field')

        return self

    @classmethod
    @typecheck
    def from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = []
    ):
        config_dict = yaml_config_path_to_dict(path)
        config_dict = safe_deep_get(config_dict, dotpath)
        assert exists(config_dict), f'config not found at path {".".join(dotpath)}'

        return cls(**config_dict)

    def create_instance(
        self,
        trainer_name: str,
        **kwargs
    ) -> Trainer:

        assert trainer_name in self.training, f'{trainer_name} not found among available trainers {tuple(self.training.keys())}'

        model = self.model.create_instance()

        trainer_config = self.training[trainer_name]

        # nest the checkpoint_folder of the trainer within the main checkpoint_folder

        nested_checkpoint_folder = str(Path(self.checkpoint_folder) / Path(trainer_config.checkpoint_folder))

        trainer_config.checkpoint_folder = nested_checkpoint_folder

        # prepend the main training checkpoint_prefix

        nested_checkpoint_prefix = self.checkpoint_prefix + trainer_config.checkpoint_prefix

        trainer_config.checkpoint_prefix = nested_checkpoint_prefix

        # create the Trainer, accounting for root level config

        trainer = trainer_config.create_instance(
            model = model,
            **kwargs
        )

        return trainer

    @classmethod
    def create_instance_from_yaml_file(
        cls,
        path: str | Path,
        dotpath: str | List[str] = [],
        **kwargs
    ) -> Trainer:

        training_config = cls.from_yaml_file(path, dotpath)
        return training_config.create_instance(**kwargs)

# convenience functions

create_alphafold3_from_yaml = Alphafold3Config.create_instance_from_yaml_file
create_trainer_from_yaml = TrainerConfig.create_instance_from_yaml_file
create_trainer_from_conductor_yaml = ConductorConfig.create_instance_from_yaml_file
