"""
Protein Language Model (PLM) Module.

This module provides wrappers for protein language models (PLMs) such as ESM-2 and ProstT5,
which generate embeddings for protein sequences. These embeddings capture evolutionary and
structural information that can enhance AlphaFold3's predictions.

The main components are:
- ESMWrapper: Wrapper for Meta's ESM-2 (Evolutionary Scale Modeling) protein language model
- ProstT5Wrapper: Wrapper for the ProstT5 model which uses T5 architecture for proteins
- remove_plms: A decorator to temporarily remove PLMs from models during certain operations

Both wrappers handle the conversion of amino acid sequences to model-specific formats
and generate contextualized embeddings for each residue.
"""

import re
from functools import partial, wraps

import torch
from beartype.typing import Literal
from torch import tensor
from torch.nn import Module

from alphafold3_pytorch.common.biomolecule import get_residue_constants
from alphafold3_pytorch.inputs import IS_PROTEIN
from alphafold3_pytorch.tensor_typing import Float, Int, typecheck
from alphafold3_pytorch.utils.data_utils import join

# Decorator functions


def remove_plms(fn):
    """
    Decorator to temporarily remove PLMs from a model during function execution.

    This is useful when saving or serializing models, as PLM models can be very large
    (several GB) and may not need to be saved with the main model weights. Temporarily
    removing them reduces checkpoint sizes.

    Args:
        fn: The function to wrap.

    Returns:
        A wrapped function that temporarily removes PLMs during execution.

    Example:
        @remove_plms
        def save_checkpoint(self, path):
            torch.save(self.state_dict(), path)
    """

    @wraps(fn)
    def inner(self, *args, **kwargs):
        # Check if the model has PLMs attached
        has_plms = hasattr(self, "plms")
        if has_plms:
            # Temporarily store and remove PLMs
            plms = self.plms
            delattr(self, "plms")

        # Execute the wrapped function
        out = fn(self, *args, **kwargs)

        if has_plms:
            # Restore PLMs after function execution
            self.plms = plms

        return out

    return inner


# Constants for amino acid residue types

# Get amino acid constants from biomolecule definitions
aa_constants = get_residue_constants(res_chem_index=IS_PROTEIN)
# Standard amino acids plus unknown 'X'
restypes = aa_constants.restypes + ["X"]

# Special tokens for missing/masked amino acids in different models
ESM_MASK_TOKEN = "-"  # nosec - Used by ESM models
PROST_T5_MASK_TOKEN = "X"  # nosec - Used by ProstT5 model

# PLM Wrapper Classes


class ESMWrapper(Module):
    """
    Wrapper for Meta's ESM (Evolutionary Scale Modeling) protein language models.

    ESM models are pretrained transformer models that learn protein sequence representations
    from millions of sequences. They capture evolutionary and structural information useful
    for downstream tasks like structure prediction.

    The wrapper supports various ESM model sizes:
    - esm2_t33_650M_UR50D: 650M parameters, 33 layers (commonly used)
    - esm2_t36_3B_UR50D: 3B parameters, 36 layers (larger variant)
    - And other variants from the ESM model zoo

    Attributes:
        repr_layer: Which layer's representations to extract (default: 33).
        model: The loaded ESM model.
        batch_converter: Utility to convert sequences to model input format.
        embed_dim: Dimensionality of the output embeddings.
    """

    def __init__(
        self,
        esm_name: str,
        repr_layer: int = 33,
    ):
        """
        Initialize the ESM wrapper.

        Args:
            esm_name: Name of the ESM model to load (e.g., "esm2_t33_650M_UR50D").
            repr_layer: Layer index from which to extract representations.
                       Default is 33, which is the final layer for the 33-layer model.
        """
        super().__init__()
        import esm

        self.repr_layer = repr_layer
        # Load the pretrained ESM model and its alphabet from HuggingFace
        self.model, alphabet = esm.pretrained.load_model_and_alphabet_hub(esm_name)
        # Get batch converter for processing sequences
        self.batch_converter = alphabet.get_batch_converter()

        # Store embedding dimension for downstream use
        self.embed_dim = self.model.embed_dim
        # Register a dummy buffer to track the device of this module
        self.register_buffer("dummy", tensor(0), persistent=False)

    @torch.no_grad()
    @typecheck
    def forward(
        self, aa_ids: Int["b n"]  # type: ignore
    ) -> Float["b n dpe"]:  # type: ignore
        """
        Generate PLM embeddings for a batch of protein sequences.

        This method converts amino acid residue indices to sequence strings,
        processes them through the ESM batch converter, and extracts embeddings
        from the specified representation layer.

        Args:
            aa_ids: Batch of amino acid residue indices with shape (batch_size, seq_len).
                   Values are integer indices where -1 represents missing/masked amino acids,
                   and other values index into the standard amino acid types.

        Returns:
            PLM embeddings with shape (batch_size, seq_len, embed_dim).
            Each amino acid position gets a contextual embedding vector from ESM.

        Note:
            This function runs with torch.no_grad() for efficient inference.
            The model automatically adds special tokens which are removed from output.
        """
        device, seq_len, repr_layer = self.dummy.device, aa_ids.shape[-1], self.repr_layer

        # Convert amino acid indices to sequence strings
        # Following the format expected by ESM models (see https://github.com/facebookresearch/esm)
        sequence_data = [
            (
                f"molecule{mol_idx}",  # Sequence identifier
                join([(ESM_MASK_TOKEN if i == -1 else restypes[i]) for i in ids]),  # Sequence string
            )
            for mol_idx, ids in enumerate(aa_ids)
        ]

        # Convert sequences to tokenized input IDs for the model
        _, _, batch_tokens = self.batch_converter(sequence_data)
        batch_tokens = batch_tokens.to(device)

        # Generate embeddings through the pretrained ESM model
        self.model.eval()
        results = self.model(batch_tokens, repr_layers=[repr_layer])

        # Extract embeddings from the specified representation layer
        embeddings = results["representations"][repr_layer]

        # Remove the prefix token (ESM adds [CLS] token at position 0)
        plm_embeddings = embeddings[:, 1 : (seq_len + 1)]

        return plm_embeddings


class ProstT5Wrapper(Module):
    """
    Wrapper for the ProstT5 protein language model.

    ProstT5 is a T5-based encoder model pretrained on protein sequences.
    It uses the T5 (Text-To-Text Transfer Transformer) architecture adapted
    for proteins, providing high-quality protein sequence embeddings.

    The model processes protein sequences with spaces between amino acids and
    handles special cases like unusual amino acids by replacing them with 'X'.

    Attributes:
        tokenizer: T5Tokenizer for converting sequences to model inputs.
        model: The pretrained ProstT5 encoder model.
        embed_dim: Dimensionality of the output embeddings (1024 for ProstT5).
    """

    def __init__(self):
        """
        Initialize the ProstT5 wrapper.

        Loads the pretrained ProstT5 model and tokenizer from HuggingFace.
        The model has a fixed embedding dimension of 1024.
        """
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer

        # Register a dummy buffer to track the device of this module
        self.register_buffer("dummy", tensor(0), persistent=False)

        # Load the tokenizer and encoder model from HuggingFace
        # do_lower_case=False preserves the case of amino acid letters
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        # ProstT5 has a fixed embedding dimension of 1024
        self.embed_dim = 1024

    @torch.no_grad()
    @typecheck
    def forward(
        self, aa_ids: Int["b n"]  # type: ignore
    ) -> Float["b n dpe"]:  # type: ignore
        """
        Generate PLM embeddings for a batch of protein sequences.

        This method converts amino acid residue indices to space-separated sequence
        strings, handles special amino acids, tokenizes them, and generates embeddings
        through the ProstT5 encoder.

        Args:
            aa_ids: Batch of amino acid residue indices with shape (batch_size, seq_len).
                   Values are integer indices where -1 represents missing/masked amino acids.

        Returns:
            PLM embeddings with shape (batch_size, seq_len, 1024).
            Each amino acid position gets a contextual embedding vector from ProstT5.

        Note:
            This function runs with torch.no_grad() for efficient inference.
            Unusual amino acids (U, Z, O, B) are replaced with 'X' (unknown).
            Sequences are space-separated as required by ProstT5.
        """
        device, seq_len = self.dummy.device, aa_ids.shape[-1]

        # Convert amino acid indices to sequence strings
        # Following the format at https://github.com/mheinzinger/ProstT5
        sequence_data = [
            join([(PROST_T5_MASK_TOKEN if i == -1 else restypes[i]) for i in ids])
            for ids in aa_ids
        ]

        # Replace unusual amino acids with 'X' and add spaces between amino acids
        # ProstT5 expects space-separated sequences (e.g., "M A T Q K" instead of "MATQK")
        # UZOB are non-standard amino acids that should be treated as unknown
        sequence_data = [
            join(list(re.sub(r"[UZOB]", "X", str_seq)), " ") for str_seq in sequence_data
        ]

        # Tokenize the sequences for the T5 model
        inputs = self.tokenizer.batch_encode_plus(
            sequence_data, add_special_tokens=True, padding="longest", return_tensors="pt"
        ).to(device)

        # Generate embeddings through the pretrained ProstT5 encoder
        embeddings = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)

        # Remove the prefix token (T5 adds special tokens at the beginning)
        plm_embeddings = embeddings.last_hidden_state[:, 1 : (seq_len + 1)]
        return plm_embeddings


# PLM embedding type and registry

# Registry mapping PLM names to their wrapper classes
# This allows for easy instantiation of PLM models by name
# ESM models are created with partial application to specify the model variant
PLMRegistry = dict(
    esm2_t33_650M_UR50D=partial(ESMWrapper, "esm2_t33_650M_UR50D"),  # ESM-2 650M parameter model
    prostT5=ProstT5Wrapper  # ProstT5 model
)

# Type annotation for valid PLM embedding types
# This ensures type safety when specifying which PLM to use
PLMEmbedding = Literal["esm2_t33_650M_UR50D", "prostT5"]
