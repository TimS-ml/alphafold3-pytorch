"""
Training Module for AlphaFold3.

This module provides a comprehensive Trainer class for training AlphaFold3 models,
implementing the training procedure described in Section 5.4 of the AlphaFold3 paper.

Key features:
- Gradient accumulation for large effective batch sizes
- Exponential moving average (EMA) of model weights
- Learning rate scheduling with warmup and decay
- Distributed training support via PyTorch Lightning Fabric
- Automatic checkpointing and model saving
- Validation and testing on held-out datasets
- Support for multiple optimizer variants (Adam, AdamAtan2, Lion, etc.)

The Trainer handles:
- Data loading and batching with custom collation
- Forward and backward passes with mixed precision training
- Gradient clipping for training stability
- Learning rate scheduling
- Model checkpointing and resuming
- Logging and monitoring via Lightning loggers
- Distributed evaluation strategies

Example:
    >>> from alphafold3_pytorch import Alphafold3, Trainer
    >>> model = Alphafold3(...)
    >>> trainer = Trainer(
    ...     model=model,
    ...     dataset=train_dataset,
    ...     num_train_steps=100000,
    ...     batch_size=1,
    ...     valid_dataset=valid_dataset
    ... )
    >>> trainer()  # Start training
"""

from __future__ import annotations

from functools import wraps, partial
from contextlib import contextmanager
from pathlib import Path

from alphafold3_pytorch.alphafold3 import Alphafold3

from beartype.typing import TypedDict, List, Callable

from alphafold3_pytorch.tensor_typing import (
    should_typecheck,
    typecheck,
    Int, Bool, Float
)

from alphafold3_pytorch.inputs import (
    AtomInput,
    BatchedAtomInput,
    Alphafold3Input,
    PDBInput,
    maybe_transform_to_atom_inputs,
    alphafold3_inputs_to_batched_atom_input,
    collate_inputs_to_batched_atom_input,
    UNCOLLATABLE_ATOM_INPUT_FIELDS,
    ATOM_DEFAULT_PAD_VALUES,
)

from alphafold3_pytorch.data import (
    mmcif_writing
)

import torch
from torch import tensor
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, Dataset, DataLoader as OrigDataLoader
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lion_pytorch.foreach import Lion
from adam_atan2_pytorch.foreach import AdamAtan2
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from ema_pytorch import EMA

from lightning import Fabric
from lightning.fabric.loggers import Logger
from lightning.fabric.wrappers import _unwrap_objects

from shortuuid import uuid

# Helper functions

def exists(val):
    """
    Check if a value is not None.

    Args:
        val: Any value to check.

    Returns:
        True if the value is not None, False otherwise.
    """
    return val is not None

def default(v, d):
    """
    Return a value if it exists, otherwise return a default.

    Args:
        v: The value to check.
        d: The default value to return if v is None.

    Returns:
        v if it's not None, otherwise d.
    """
    return v if exists(v) else d

def divisible_by(num, den):
    """
    Check if num is divisible by den.

    Args:
        num: The numerator.
        den: The denominator.

    Returns:
        True if num is divisible by den, False otherwise.
    """
    return (num % den) == 0

def at_most_one_of(*flags: bool) -> bool:
    """
    Check that at most one of the given flags is True.

    This is useful for validating mutually exclusive options.

    Args:
        *flags: Variable number of boolean flags.

    Returns:
        True if at most one flag is True, False if multiple are True.
    """
    return sum([*map(int, flags)]) <= 1

@contextmanager
def to_device_and_back(
    module: Module,
    device: torch.device
):
    """
    Context manager to temporarily move a module to a device and back.

    This is useful for operations like validation with EMA models that might
    be stored on CPU but need to run on GPU.

    Args:
        module: The PyTorch module to move.
        device: The target device.

    Yields:
        None (the module is moved in-place).

    Example:
        >>> with to_device_and_back(ema_model, device='cuda'):
        ...     outputs = ema_model(inputs)  # Runs on CUDA
        # ema_model is back on its original device
    """
    orig_device = next(module.parameters()).device
    need_move_device = orig_device != device

    if need_move_device:
        module.to(device)

    yield

    if need_move_device:
        module.to(orig_device)

def cycle(dataloader: DataLoader):
    """
    Infinitely cycle through a DataLoader.

    This is useful for training where you want to specify a fixed number of
    steps rather than epochs.

    Args:
        dataloader: A PyTorch DataLoader.

    Yields:
        Batches from the dataloader, cycling indefinitely.

    Example:
        >>> dl = cycle(train_dataloader)
        >>> for step in range(num_steps):
        ...     batch = next(dl)
    """
    while True:
        for batch in dataloader:
            yield batch

@typecheck
def accum_dict(
    past_losses: dict | None,
    losses: dict,
    scale: float = 1.
):
    """
    Accumulate loss dictionaries with optional scaling.

    This is used for gradient accumulation where losses need to be averaged
    across multiple micro-batches.

    Args:
        past_losses: Previously accumulated losses (or None for first batch).
        losses: New losses to accumulate.
        scale: Scaling factor to apply to new losses (typically 1/num_accum_steps).

    Returns:
        Updated loss dictionary with accumulated values.

    Example:
        >>> loss_dict = None
        >>> for i in range(4):  # 4 gradient accumulation steps
        ...     batch_losses = {'loss': 10.0, 'plddt_loss': 2.0}
        ...     loss_dict = accum_dict(loss_dict, batch_losses, scale=0.25)
    """
    # Scale the new losses
    losses = {k: v * scale for k, v in losses.items()}

    # If no past losses, just return the scaled current losses
    if not exists(past_losses):
        return losses

    # Accumulate into past losses
    for loss_name in past_losses.keys():
        past_losses[loss_name] += losses.get(loss_name, 0.)

    return past_losses

# DataLoader and collation

@typecheck
def DataLoader(
    *args,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
    transform_to_atom_inputs: bool = True,
    **kwargs
):
    """
    Custom DataLoader for AlphaFold3 with specialized collation.

    This DataLoader wraps PyTorch's DataLoader and provides custom collation
    that handles AtomInput/Alphafold3Input batching and windowing.

    Args:
        *args: Positional arguments passed to PyTorch DataLoader (dataset, etc.).
        atoms_per_window: Number of atoms per window for windowed attention.
                         Should match the model's atoms_per_window setting.
        map_input_fn: Optional function to transform inputs before collation.
        transform_to_atom_inputs: Whether to transform inputs to AtomInput format.
        **kwargs: Additional keyword arguments passed to PyTorch DataLoader.

    Returns:
        A PyTorch DataLoader with custom collation for AlphaFold3.

    Note:
        The collation function handles padding, windowing, and batching of
        complex hierarchical data (atoms, molecules, chains).
    """
    # Create a partial collation function with the specified parameters
    collate_fn = partial(
        collate_inputs_to_batched_atom_input,
        atoms_per_window = atoms_per_window,
        transform_to_atom_inputs = transform_to_atom_inputs,
    )

    # Add input transformation if provided
    if exists(map_input_fn):
        collate_fn = partial(collate_fn, map_input_fn = map_input_fn)

    return OrigDataLoader(*args, collate_fn = collate_fn, **kwargs)

# Learning rate scheduling

def default_lambda_lr_fn(steps):
    """
    Default learning rate schedule used in the AlphaFold3 paper.

    This implements:
    1. Linear warmup for the first 1000 steps (0 to 1.0)
    2. Exponential decay by 0.95 every 50,000 steps after warmup

    Args:
        steps: Current training step number.

    Returns:
        Learning rate multiplier (relative to base learning rate).

    Example:
        At step 0: returns 0.0 (0% of base LR)
        At step 500: returns 0.5 (50% of base LR)
        At step 1000: returns 1.0 (100% of base LR)
        At step 51000: returns 0.95 (95% of base LR)
        At step 101000: returns 0.95^2 = 0.9025 (90.25% of base LR)
    """
    # Linear warmup for first 1000 steps
    if steps < 1000:
        return steps / 1000

    # Exponential decay: 0.95 every 50,000 steps
    steps -= 1000  # Adjust for warmup steps
    return 0.95 ** (steps / 5e4)

# Main Trainer class

class Trainer:
    """
    Trainer for AlphaFold3 models implementing the training procedure from Section 5.4.

    This class handles the complete training loop including:
    - Gradient accumulation for large effective batch sizes
    - Exponential moving average (EMA) of weights
    - Learning rate scheduling
    - Distributed training via PyTorch Lightning Fabric
    - Checkpointing and model persistence
    - Validation and testing
    - Logging and monitoring

    The trainer is designed to be flexible and supports various configurations
    including different optimizers, learning rate schedules, and training strategies.

    Attributes:
        model: The AlphaFold3 model being trained.
        optimizer: The optimizer (Adam, AdamAtan2, Lion, etc.).
        scheduler: Learning rate scheduler.
        ema_model: Exponential moving average of model weights (if enabled).
        dataloader: Training data loader.
        valid_dataloader: Validation data loader (if provided).
        test_dataloader: Test data loader (if provided).
        fabric: PyTorch Lightning Fabric for distributed training.
        steps: Current training step counter.
    """

    @typecheck
    def __init__(
        self,
        model: Alphafold3,
        *,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int = 1,
        map_dataset_input_fn: Callable | None = None,
        valid_dataset: Dataset | None = None,
        valid_every: int = 1000,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        ema_decay = 0.999,
        lr = 1.8e-3,
        default_adam_kwargs: dict = dict(
            betas = (0.9, 0.95),
            eps = 1e-8
        ),
        clip_grad_norm = 10.,
        default_lambda_lr = default_lambda_lr_fn,
        train_sampler: Sampler | None = None,
        fabric: Fabric | None = None,
        loggers: List[Logger] = [],
        accelerator = 'auto',
        checkpoint_prefix = 'af3.ckpt.',
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints',
        overwrite_checkpoints: bool = False,
        fabric_kwargs: dict = dict(),
        distributed_eval: bool = True,
        fp16: bool = False,
        use_ema: bool = True,
        ema_kwargs: dict = dict(
            use_foreach = True
        ),
        ema_on_cpu = False,
        ema_update_model_with_ema_every: int | None = None,
        use_adam_atan2: bool = False,
        use_adopt_atan2: bool = False,
        use_lion: bool = False,
        use_torch_compile: bool = False
    ):
        """
        Initialize the AlphaFold3 Trainer.

        Args:
            model: The AlphaFold3 model to train.
            dataset: Training dataset (must be a PyTorch Dataset).
            num_train_steps: Total number of training steps to run.
            batch_size: Batch size for training.
            grad_accum_every: Number of gradient accumulation steps. Effective batch
                             size is batch_size * grad_accum_every.
            map_dataset_input_fn: Optional function to transform dataset samples.
            valid_dataset: Optional validation dataset.
            valid_every: Run validation every N steps.
            test_dataset: Optional test dataset (evaluated at end of training).
            optimizer: Custom optimizer (if None, creates Adam with default settings).
            scheduler: Custom learning rate scheduler (if None, uses default schedule).
            ema_decay: Decay rate for exponential moving average (default: 0.999).
            lr: Base learning rate (default: 1.8e-3 from paper).
            default_adam_kwargs: Default Adam optimizer parameters.
            clip_grad_norm: Maximum gradient norm for clipping (default: 10.0).
            default_lambda_lr: Learning rate schedule function.
            train_sampler: Optional custom sampler for training data.
            fabric: Optional pre-configured Lightning Fabric instance.
            loggers: List of Lightning loggers for tracking metrics.
            accelerator: Device accelerator ('auto', 'gpu', 'cpu', 'tpu', etc.).
            checkpoint_prefix: Prefix for checkpoint filenames.
            checkpoint_every: Save checkpoint every N steps.
            checkpoint_folder: Directory to save checkpoints.
            overwrite_checkpoints: Whether to overwrite existing checkpoints.
            fabric_kwargs: Additional arguments for Lightning Fabric.
            distributed_eval: Whether to run evaluation on all distributed nodes.
            fp16: Enable mixed precision training (FP16).
            use_ema: Whether to maintain an exponential moving average of weights.
            ema_kwargs: Additional arguments for EMA.
            ema_on_cpu: Store EMA model on CPU to save GPU memory.
            ema_update_model_with_ema_every: For "switch EMA" - update main model
                                            with EMA weights every N steps.
            use_adam_atan2: Use AdamAtan2 optimizer variant.
            use_adopt_atan2: Use AdoptAtan2 optimizer variant.
            use_lion: Use Lion optimizer.
            use_torch_compile: Enable torch.compile() for faster training
                              (incompatible with type checking).

        Note:
            Only one of use_adam_atan2, use_adopt_atan2, or use_lion can be True.
            If torch.compile is enabled, type checking must be disabled via
            the TYPECHECK environment variable.
        """
        super().__init__()

        # fp16 precision is a root level kwarg

        if fp16:
            assert 'precision' not in fabric_kwargs
            fabric_kwargs.update(precision = '16-mixed')

        # instantiate fabric

        if not exists(fabric):
            fabric = Fabric(
                accelerator = accelerator,
                loggers = loggers,
                **fabric_kwargs
            )

        self.fabric = fabric
        fabric.launch()

        # whether evaluating only on root node or not
        # to save on each machine keeping track of EMA

        self.distributed_eval = distributed_eval
        self.will_eval_or_test = self.is_main or distributed_eval

        # using "switch" ema

        self.switch_ema = exists(ema_update_model_with_ema_every)

        # exponential moving average

        self.ema_model = None
        self.has_ema = (self.will_eval_or_test or self.switch_ema) and use_ema

        if self.has_ema:
            self.ema_model = EMA(
                model,
                beta = ema_decay,
                include_online_model = False,
                allow_different_devices = True,
                coerce_dtype = True,
                update_model_with_ema_every = ema_update_model_with_ema_every,
                **ema_kwargs
            )

            self.ema_device = 'cpu' if ema_on_cpu else self.device
            self.ema_model.to(self.ema_device)

        # maybe torch compile

        if use_torch_compile:
            assert not should_typecheck, 'does not work well with jaxtyping + beartype, please invoke your training script with the environment flag `TYPECHECK=False` - ex. `TYPECHECK=False python train_af3.py`'
            model = torch.compile(model)

        # model

        self.model = model

        # optimizer

        if not exists(optimizer):
            optimizer_klass = Adam

            assert at_most_one_of(use_adam_atan2, use_adopt_atan2, use_lion)

            if use_adam_atan2:
                default_adam_kwargs.pop('eps', None)
                optimizer_klass = AdamAtan2
            elif use_adopt_atan2:
                default_adam_kwargs.pop('eps', None)
                optimizer_klass = AdoptAtan2
            elif use_lion:
                default_adam_kwargs.pop('eps', None)
                optimizer_klass = Lion

            optimizer = optimizer_klass(
                model.parameters(),
                lr = lr,
                **default_adam_kwargs
            )

        self.optimizer = optimizer

        # if map dataset function given, curry into DataLoader

        DataLoader_ = partial(DataLoader, atoms_per_window = model.atoms_per_window)

        if exists(map_dataset_input_fn):
            DataLoader_ = partial(DataLoader_, map_input_fn = map_dataset_input_fn)

        # maybe weighted sampling

        train_dl_kwargs = dict()

        if exists(train_sampler):
            train_dl_kwargs.update(sampler = train_sampler)
        else:
            train_dl_kwargs.update(
                shuffle = True,
                drop_last = True
            )

        # train dataloader

        self.dataloader = DataLoader_(
            dataset,
            batch_size = batch_size,
            **train_dl_kwargs
        )

        # validation dataloader on the EMA model

        self.valid_every = valid_every

        self.needs_valid = exists(valid_dataset)
        self.valid_dataloader = None

        if self.needs_valid and self.will_eval_or_test:
            self.valid_dataset_size = len(valid_dataset)
            self.valid_dataloader = DataLoader_(valid_dataset, batch_size = batch_size)

        # testing dataloader on EMA model

        self.needs_test = exists(test_dataset)
        self.test_dataloader = None

        if self.needs_test and self.will_eval_or_test:
            self.test_dataset_size = len(test_dataset)
            self.test_dataloader = DataLoader_(test_dataset, batch_size = batch_size)

        # training steps and num gradient accum steps

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every

        # setup fabric

        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

        fabric.setup_dataloaders(self.dataloader)

        if exists(self.valid_dataloader) and self.distributed_eval:
            fabric.setup_dataloaders(self.valid_dataloader)

        if exists(self.test_dataloader) and self.distributed_eval:
            fabric.setup_dataloaders(self.test_dataloader)

        # scheduler

        if not exists(scheduler):
            scheduler = LambdaLR(optimizer, lr_lambda = default_lambda_lr)

        self.scheduler = scheduler

        # gradient clipping norm

        self.clip_grad_norm = clip_grad_norm

        # steps

        self.steps = 0

        # checkpointing logic

        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_every = checkpoint_every
        self.overwrite_checkpoints = overwrite_checkpoints
        self.checkpoint_folder = Path(checkpoint_folder)

        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # save the path for the last loaded model, if any

        self.train_id = None

        self.last_loaded_train_id = None
        self.model_loaded_from_path: Path | None = None

    @property
    def device(self):
        """
        Get the device where the model and data are located.

        Returns:
            The PyTorch device (e.g., 'cuda:0', 'cpu').
        """
        return self.fabric.device

    @property
    def is_main(self):
        """
        Check if this is the main process in distributed training.

        Returns:
            True if this is the main process (rank 0), False otherwise.
        """
        return self.fabric.global_rank == 0

    def generate_train_id(self):
        """
        Generate a unique identifier for this training run.

        The ID is a 4-character lowercase UUID used to track training sessions
        across checkpoints and resumptions.
        """
        if exists(self.train_id):
            return

        self.train_id = uuid()[:4].lower()

    @property
    def train_id_with_prev(self) -> str:
        """
        Get a training ID that includes the previous training session if resumed.

        This creates a training lineage like: 'abc1.5000-def2' meaning
        training was resumed from checkpoint 5000 of run 'abc1' and continued
        as run 'def2'.

        Returns:
            A string representing the training lineage.
        """
        if not exists(self.last_loaded_train_id):
            return self.train_id

        # Extract checkpoint number from the loaded model path
        ckpt_num = str(self.model_loaded_from_path).split('.')[-2]

        return f'{self.last_loaded_train_id}.{ckpt_num}-{self.train_id}'

    # Saving and loading methods

    def save_checkpoint(self):
        """
        Save a training checkpoint at the current step.

        The checkpoint filename includes the training lineage and step number,
        e.g., '(abc1)_af3.ckpt.1000.pt' or '(abc1.5000-def2)_af3.ckpt.6000.pt'.

        The checkpoint contains:
        - Model state dict with initialization arguments
        - Optimizer state
        - Scheduler state
        - Current training step
        - Training ID
        """
        assert exists(self.train_id_with_prev)

        # Formulate checkpoint path with training lineage and step number
        checkpoint_path = self.checkpoint_folder / f'({self.train_id_with_prev})_{self.checkpoint_prefix}{self.steps}.pt'

        self.save(checkpoint_path, overwrite = self.overwrite_checkpoints)

    def save(
        self,
        path: str | Path,
        overwrite = False,
        prefix: str | None = None
    ):
        """
        Save model, optimizer, and training state to a file.

        Args:
            path: Path where to save the checkpoint.
            overwrite: Whether to overwrite if file exists.
            prefix: Optional prefix for checkpoint filename (unused, kept for compatibility).

        Note:
            The saved checkpoint includes init args for model reconstruction,
            allowing the model to be loaded even without knowing its configuration.
        """
        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (not path.exists() or overwrite)

        path.parent.mkdir(exist_ok = True, parents = True)

        unwrapped_model = _unwrap_objects(self.model)
        unwrapped_optimizer = _unwrap_objects(self.optimizer)

        package = dict(
            model = unwrapped_model.state_dict_with_init_args,
            optimizer = unwrapped_optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            steps = self.steps,
            id = self.train_id
        )

        torch.save(package, str(path))

    def load_from_checkpoint_folder(
        self,
        **kwargs
    ):
        """
        Load the latest checkpoint from the checkpoint folder.

        Args:
            **kwargs: Arguments passed to the load() method.

        Note:
            Automatically finds and loads the most recent checkpoint based
            on the step number in the filename.
        """
        self.load(path = self.checkpoint_folder, **kwargs)

    def load(
        self,
        path: str | Path,
        strict = True,
        prefix = None,
        only_model = False,
        reset_steps = False
    ):
        """
        Load a checkpoint from a file or directory.

        Args:
            path: Path to checkpoint file or directory containing checkpoints.
            strict: Whether to strictly enforce state dict loading (unused).
            prefix: Prefix for finding checkpoints in a directory.
            only_model: If True, only load model weights (not optimizer/scheduler).
            reset_steps: If True, reset step counter to 0 after loading.

        Note:
            If path is a directory, automatically loads the latest checkpoint.
            The training ID from the loaded checkpoint is tracked for lineage.
        """
        if isinstance(path, str):
            path = Path(path)

        assert path.exists(), f'{str(path)} cannot be found for loading'

        # if the path is a directory, then automatically load latest checkpoint

        if path.is_dir():
            prefix = default(prefix, self.checkpoint_prefix)

            model_paths = [*path.glob(f'**/*_{prefix}*.pt')]

            assert len(model_paths) > 0, f'no files found in directory {path}'

            model_paths = sorted(model_paths, key = lambda p: int(str(p).split('.')[-2]))

            path = model_paths[-1]

        # get unwrapped model and optimizer

        unwrapped_model = _unwrap_objects(self.model)

        # load model from path

        model_id = unwrapped_model.load(path)

        # for eventually saving entire training history in filename

        self.model_loaded_from_path = path
        self.last_loaded_train_id = model_id

        if only_model:
            return

        # load optimizer and scheduler states

        package = torch.load(str(path), weights_only = True)

        unwrapped_optimizer = _unwrap_objects(self.optimizer)

        if 'optimizer' in package:
            unwrapped_optimizer.load_state_dict(package['optimizer'])

        if 'scheduler' in package:
            self.scheduler.load_state_dict(package['scheduler'])

        if reset_steps:
            self.steps = 0
        else:
            self.steps = package.get('steps', 0)

    # Utility shortcut methods

    def wait(self):
        """
        Synchronize all distributed processes.

        This ensures all processes reach this point before any continue.
        Useful for coordinating checkpointing and evaluation.
        """
        self.fabric.barrier()

    def print(self, *args, **kwargs):
        """
        Print only on the main process in distributed training.

        Args:
            *args: Arguments to print.
            **kwargs: Keyword arguments for print.
        """
        self.fabric.print(*args, **kwargs)

    def log(self, **log_data):
        """
        Log metrics to all configured loggers.

        Args:
            **log_data: Dictionary of metric names and values to log.
                       The current training step is automatically included.
        """
        self.fabric.log_dict(log_data, step = self.steps)

    # Main training loop

    def __call__(
        self
    ):
        """
        Execute the main training loop.

        This method implements the complete training procedure:
        1. Generate a unique training ID
        2. Loop through training steps with gradient accumulation
        3. Perform forward and backward passes
        4. Update model weights with gradient clipping
        5. Update EMA if enabled
        6. Step learning rate scheduler
        7. Validate periodically if validation set provided
        8. Save checkpoints periodically
        9. Test on test set after training completes

        The training loop runs for num_train_steps iterations, with each
        iteration potentially accumulating gradients over multiple batches
        (controlled by grad_accum_every).

        Returns:
            None. Training progress is logged and checkpoints are saved.

        Note:
            This method blocks until training completes. Progress can be
            monitored through the configured loggers.
        """

        self.generate_train_id()

        # cycle through dataloader

        dl = cycle(self.dataloader)

        # while less than required number of training steps

        while self.steps < self.num_train_steps:

            self.model.train()

            # gradient accumulation

            total_loss = 0.
            train_loss_breakdown = None

            for grad_accum_step in range(self.grad_accum_every):
                is_accumulating = grad_accum_step < (self.grad_accum_every - 1)

                inputs = next(dl)

                with self.fabric.no_backward_sync(self.model, enabled = is_accumulating):

                    # model forwards

                    loss, loss_breakdown = self.model(
                        **inputs.model_forward_dict(),
                        return_loss_breakdown = True
                    )

                    # accumulate

                    scale = self.grad_accum_every ** -1

                    total_loss += loss.item() * scale
                    train_loss_breakdown = accum_dict(train_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                    # backwards

                    self.fabric.backward(loss / self.grad_accum_every)

            # log entire loss breakdown

            self.log(**train_loss_breakdown)

            self.print(f'loss: {total_loss:.3f}')

            # clip gradients

            self.fabric.clip_gradients(self.model, self.optimizer, max_norm = self.clip_grad_norm)

            # optimizer step

            self.optimizer.step()

            # update exponential moving average

            self.wait()

            if self.has_ema:
                self.ema_model.update()

            self.wait()

            # scheduler

            self.scheduler.step()
            self.optimizer.zero_grad()

            self.steps += 1

            # maybe validate, for now, only on main with EMA model

            if (
                self.will_eval_or_test and
                self.needs_valid and
                divisible_by(self.steps, self.valid_every)
            ):
                eval_model = default(self.ema_model, self.model)

                with torch.no_grad(), to_device_and_back(eval_model, self.device):
                    eval_model.eval()

                    total_valid_loss = 0.
                    valid_loss_breakdown = None

                    for valid_batch in self.valid_dataloader:
                        valid_loss, loss_breakdown = eval_model(
                            **valid_batch.model_forward_dict(),
                            return_loss_breakdown = True
                        )

                        valid_batch_size = valid_batch.atom_inputs.shape[0]
                        scale = valid_batch_size / self.valid_dataset_size

                        total_valid_loss += valid_loss.item() * scale
                        valid_loss_breakdown = accum_dict(valid_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                    self.print(f'valid loss: {total_valid_loss:.3f}')

                # prepend valid_ to all losses for logging

                valid_loss_breakdown = {f'valid_{k}':v for k, v in valid_loss_breakdown.items()}

                # reduce valid loss breakdown

                if self.distributed_eval:
                    valid_loss_breakdown = self.fabric.all_reduce(valid_loss_breakdown, reduce_op = 'sum')

                # log

                self.log(**valid_loss_breakdown)

            self.wait()

            if self.is_main and divisible_by(self.steps, self.checkpoint_every):
                self.save_checkpoint()

            self.wait()

        # maybe test

        if self.will_eval_or_test and self.needs_test:
            eval_model = default(self.ema_model, self.model)

            with torch.no_grad(), to_device_and_back(eval_model, self.device):
                eval_model.eval()

                total_test_loss = 0.
                test_loss_breakdown = None

                for test_batch in self.test_dataloader:
                    test_loss, loss_breakdown = eval_model(
                        **test_batch.model_forward_dict(),
                        return_loss_breakdown = True
                    )

                    test_batch_size = test_batch.atom_inputs.shape[0]
                    scale = test_batch_size / self.test_dataset_size

                    total_test_loss += test_loss.item() * scale
                    test_loss_breakdown = accum_dict(test_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                self.print(f'test loss: {total_test_loss:.3f}')

            # prepend test_ to all losses for logging

            test_loss_breakdown = {f'test_{k}':v for k, v in test_loss_breakdown.items()}

            # reduce

            if self.distributed_eval:
                test_loss_breakdown = self.fabric.all_reduce(test_loss_breakdown, reduce_op = 'sum')

            # log

            self.log(**test_loss_breakdown)

        print('training complete')
