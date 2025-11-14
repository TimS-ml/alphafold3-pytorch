"""
Nucleic Acid Language Model (NLM) Module.

This module provides wrappers for nucleic acid language models (NLMs) such as RiNALMo,
which generate embeddings for RNA and DNA sequences. These embeddings can be used to
enhance the AlphaFold3 model's understanding of nucleic acid structures.

The main components are:
- RiNALMoWrapper: A wrapper for the RiNALMo model for RNA/DNA sequence embeddings
- remove_nlms: A decorator to temporarily remove NLMs from models during certain operations
"""

from functools import wraps

import torch
from beartype.typing import Literal
from torch import tensor
from torch.nn import Module

from alphafold3_pytorch.common.biomolecule import get_residue_constants
from alphafold3_pytorch.inputs import IS_DNA, IS_RNA
from alphafold3_pytorch.tensor_typing import Float, Int, typecheck
from alphafold3_pytorch.utils.data_utils import join

# Decorator functions

def remove_nlms(fn):
    """
    Decorator to temporarily remove NLMs from a model during function execution.

    This is useful when saving or serializing models, as NLM models can be large
    and may not need to be saved with the main model weights.

    Args:
        fn: The function to wrap.

    Returns:
        A wrapped function that temporarily removes NLMs during execution.

    Example:
        @remove_nlms
        def save_model(self, path):
            torch.save(self.state_dict(), path)
    """

    @wraps(fn)
    def inner(self, *args, **kwargs):
        # Check if the model has NLMs attached
        has_nlms = hasattr(self, "nlms")
        if has_nlms:
            # Temporarily store and remove NLMs
            nlms = self.nlms
            delattr(self, "nlms")

        # Execute the wrapped function
        out = fn(self, *args, **kwargs)

        if has_nlms:
            # Restore NLMs after function execution
            self.nlms = nlms

        return out

    return inner


# Constants for nucleic acid residue types

# Get residue constants for RNA and DNA from biomolecule definitions
rna_constants = get_residue_constants(res_chem_index=IS_RNA)
dna_constants = get_residue_constants(res_chem_index=IS_DNA)

# RNA residue types: A, C, G, U, plus unknown 'X'
rna_restypes = rna_constants.restypes + ["X"]
# DNA residue types: A, C, G, T, plus unknown 'X'
dna_restypes = dna_constants.restypes + ["X"]

# Minimum residue type numbers for indexing
rna_min_restype_num = rna_constants.min_restype_num
dna_min_restype_num = dna_constants.min_restype_num

# Special token used by RiNALMo to represent masked/missing nucleotides
RINALMO_MASK_TOKEN = "-"  # nosec

# NLM Wrapper Classes


class RiNALMoWrapper(Module):
    """
    Wrapper for the RiNALMo (RNA Language Model) to provide nucleic acid embeddings.

    RiNALMo is a pretrained language model for RNA sequences that can generate
    contextual embeddings for nucleotides. This wrapper adapts the model for use
    in AlphaFold3 by handling both RNA and DNA sequences.

    The model supports multiple variants with different sizes:
    - rinalmo-giga: Largest variant with best performance
    - Other variants may be available from the multimolecule library

    Attributes:
        tokenizer: RnaTokenizer for converting sequences to model inputs.
        model: The pretrained RiNALMo model.
        embed_dim: Dimensionality of the output embeddings.
    """

    def __init__(self, variant: str = "rinalmo-giga"):
        """
        Initialize the RiNALMo wrapper.

        Args:
            variant: The RiNALMo model variant to use. Default is "rinalmo-giga".
                    This specifies which pretrained model checkpoint to load.
        """
        super().__init__()
        from multimolecule import RiNALMoModel, RnaTokenizer

        # Register a dummy buffer to track the device of this module
        self.register_buffer("dummy", tensor(0), persistent=False)

        # Load the tokenizer and model from HuggingFace/multimolecule
        # replace_T_with_U=False keeps DNA thymine (T) distinct from RNA uracil (U)
        self.tokenizer = RnaTokenizer.from_pretrained("multimolecule/" + variant, replace_T_with_U=False)
        self.model = RiNALMoModel.from_pretrained("multimolecule/" + variant)

        # Store the embedding dimension for downstream layers
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    @typecheck
    def forward(
        self, na_ids: Int["b n"]  # type: ignore
    ) -> Float["b n dne"]:  # type: ignore
        """
        Generate NLM embeddings for a batch of nucleotide sequences.

        This method converts nucleotide residue indices to sequence strings,
        tokenizes them, and passes them through the RiNALMo model to generate
        contextual embeddings for each nucleotide position.

        Args:
            na_ids: Batch of nucleotide residue indices with shape (batch_size, seq_len).
                   Values are integer indices where:
                   - -1 represents missing/masked nucleotides
                   - Indices < dna_min_restype_num are RNA residues
                   - Indices >= dna_min_restype_num are DNA residues

        Returns:
            NLM embeddings with shape (batch_size, seq_len, embed_dim).
            Each nucleotide position gets a contextual embedding vector.

        Note:
            This function runs with torch.no_grad() for efficient inference.
            The embeddings from the model include a prefix token which is removed.
        """
        # Get the target device and sequence length
        device, seq_len = self.dummy.device, na_ids.shape[-1]

        # Convert nucleotide indices to letter sequences
        # Handle missing nucleotides (-1) and distinguish between RNA and DNA
        sequence_data = [
            join(
                [
                    (
                        RINALMO_MASK_TOKEN  # Use mask token for missing nucleotides
                        if i == -1
                        else (
                            dna_restypes[i - dna_min_restype_num]  # DNA nucleotides
                            if i >= dna_min_restype_num
                            else rna_restypes[i - rna_min_restype_num]  # RNA nucleotides
                        )
                    )
                    for i in ids
                ]
            )
            for ids in na_ids
        ]

        # Tokenize the sequences for the model
        inputs = self.tokenizer(sequence_data, return_tensors="pt").to(device)

        # Generate embeddings through the pretrained RiNALMo model
        embeddings = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)

        # Remove the prefix token (typically a [CLS] or similar special token)
        # The model adds a prefix token, so we slice it out: [1 : seq_len + 1]
        nlm_embeddings = embeddings.last_hidden_state[:, 1 : (seq_len + 1)]

        return nlm_embeddings


# NLM embedding type and registry

# Registry mapping NLM names to their wrapper classes
# This allows for easy instantiation of NLM models by name
NLMRegistry = dict(rinalmo=RiNALMoWrapper)

# Type annotation for valid NLM embedding types
# This ensures type safety when specifying which NLM to use
NLMEmbedding = Literal["rinalmo"]
