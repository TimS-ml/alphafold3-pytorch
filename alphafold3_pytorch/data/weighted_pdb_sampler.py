"""
Weighted PDB Sampler for Training Data Selection

This module implements a weighted sampling strategy for selecting PDB structures during
training, following the methodology described in AlphaFold 3 Supplement Section 2.5.1.

The sampler balances training data by:
1. Clustering similar chains and interfaces to reduce redundancy
2. Assigning weights based on cluster sizes and molecular composition
3. Down-weighting over-represented structure types
4. Up-weighting under-represented structure types (e.g., nucleic acids, ligands)

Weighting formula (from AF3 Supplement 2.5.1):
    weight = (beta / cluster_size) * (alpha_prot * n_prot + alpha_nuc * n_nuc + alpha_ligand * n_ligand)

Where:
- beta: Base weight factor (different for chains vs interfaces)
- cluster_size: Number of similar structures in the cluster
- alpha_*: Molecule type weights (default: 3.0 for protein/nucleic, 1.0 for ligands)
- n_*: Count of each molecule type in the structure

Key components:
- Chain sampling: Individual polymer chains (proteins, DNA, RNA, peptides)
- Interface sampling: Pairs of interacting chains
- Cluster-based weighting: Reduces sampling of redundant structures

Main class:
- WeightedPDBSampler: PyTorch Sampler for weighted PDB selection

The sampler supports:
- Multiple cluster mapping files for different molecule types
- Filtering by PDB ID (inclusion/exclusion lists)
- Both random and cluster-based sampling strategies
- Configurable weighting parameters

Usage:
    sampler = WeightedPDBSampler(
        chain_mapping_paths=['protein_chains.csv', 'nucleic_chains.csv'],
        interface_mapping_path='interfaces.csv',
        batch_size=64
    )
    for pdb_id in sampler:
        # Train on this PDB structure
        ...
"""

from __future__ import annotations

import os
from beartype.typing import Dict, Iterator, List, Literal, Tuple

import numpy as np
import polars as pl
from loguru import logger
from torch.utils.data import Sampler

from alphafold3_pytorch.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists

# constants

CLUSTERING_RESIDUE_MOLECULE_TYPE = Literal["protein", "rna", "dna", "ligand", "peptide"]

# helper functions


def get_chain_count(molecule_type: CLUSTERING_RESIDUE_MOLECULE_TYPE) -> Tuple[int, int, int]:
    """
    Returns the composition counts for a given molecule type.

    This function converts a molecule type string into a tuple of counts indicating
    how many protein, nucleic acid, and ligand chains are represented. This is used
    for calculating sampling weights based on molecular composition.

    Args:
        molecule_type: The type of molecule ('protein', 'rna', 'dna', 'ligand', or 'peptide').

    Returns:
        A tuple of (n_protein, n_nucleic_acid, n_ligand) counts.
        - n_protein: 1 for protein/peptide, 0 otherwise
        - n_nucleic_acid: 1 for RNA/DNA, 0 otherwise
        - n_ligand: 1 for ligand, 0 otherwise

    Raises:
        ValueError: If molecule_type is not one of the recognized types.

    Example:
        >>> get_chain_count("protein")
        (1, 0, 0)
        >>> get_chain_count("rna")
        (0, 1, 0)
        >>> get_chain_count("ligand")
        (0, 0, 1)
    """
    if molecule_type == "protein":
        return 1, 0, 0
    if molecule_type == "rna":
        return 0, 1, 0
    if molecule_type == "dna":
        return 0, 1, 0
    if molecule_type == "ligand":
        return 0, 0, 1
    if molecule_type == "peptide":
        return 1, 0, 0

    raise ValueError(f"Unknown molecule type: {molecule_type}")


def calculate_weight(
    alphas: Dict[str, float],
    beta: float,
    n_prot: int,
    n_nuc: int,
    n_ligand: int,
    cluster_size: int,
) -> float:
    """
    Calculates the sampling weight for a chain or interface.

    This implements the weighting formula from AlphaFold 3 Supplement Section 2.5.1,
    which balances training data by down-weighting over-represented structure types
    and up-weighting under-represented types.

    The formula is:
        weight = (beta / cluster_size) * (alpha_prot * n_prot + alpha_nuc * n_nuc + alpha_ligand * n_ligand)

    Args:
        alphas: Dictionary of molecule type weights with keys 'prot', 'nuc', 'ligand'.
            Typical values: {'prot': 3.0, 'nuc': 3.0, 'ligand': 1.0}
        beta: Base weight factor. Different values are used for chains (0.5) and
            interfaces (1.0) to balance their sampling.
        n_prot: Number of protein/peptide molecules in this chain/interface.
        n_nuc: Number of nucleic acid (DNA/RNA) molecules in this chain/interface.
        n_ligand: Number of ligand molecules in this chain/interface.
        cluster_size: Size of the cluster this structure belongs to. Larger clusters
            (more redundant structures) receive lower weights.

    Returns:
        The sampling weight (non-negative float). Higher weights mean higher
        probability of sampling during training.

    Example:
        >>> alphas = {'prot': 3.0, 'nuc': 3.0, 'ligand': 1.0}
        >>> calculate_weight(alphas, beta=0.5, n_prot=1, n_nuc=0, n_ligand=0, cluster_size=100)
        0.015  # Single protein in large cluster gets low weight
        >>> calculate_weight(alphas, beta=0.5, n_prot=0, n_nuc=1, n_ligand=0, cluster_size=10)
        0.15   # Nucleic acid in small cluster gets higher weight
    """
    return (beta / cluster_size) * (
        alphas["prot"] * n_prot + alphas["nuc"] * n_nuc + alphas["ligand"] * n_ligand
    )


@typecheck
def get_chain_weight(
    molecule_type: CLUSTERING_RESIDUE_MOLECULE_TYPE,
    cluster_size: int,
    alphas: Dict[str, float],
    beta: float,
) -> float:
    """Calculates the weight of a chain based on its type."""
    n_prot, n_nuc, n_ligand = get_chain_count(molecule_type)
    return calculate_weight(alphas, beta, n_prot, n_nuc, n_ligand, cluster_size)


@typecheck
def get_interface_weight(
    molecule_type_1: CLUSTERING_RESIDUE_MOLECULE_TYPE,
    molecule_type_2: CLUSTERING_RESIDUE_MOLECULE_TYPE,
    cluster_size: int,
    alphas: Dict[str, float],
    beta: float,
) -> float:
    """Calculates the weight of an interface based on the types of the two molecules."""
    p1, n1, l1 = get_chain_count(molecule_type_1)
    p2, n2, l2 = get_chain_count(molecule_type_2)

    n_prot = p1 + p2
    n_nuc = n1 + n2
    n_ligand = l1 + l2

    return calculate_weight(alphas, beta, n_prot, n_nuc, n_ligand, cluster_size)


@typecheck
def get_cluster_sizes(
    mapping: pl.DataFrame,
    cluster_id_col: str,
) -> Dict[int, int]:
    """
    Returns a dictionary where keys are cluster IDs and values are the number
    of chains/interfaces in the cluster.
    """
    cluster_sizes = mapping.group_by(cluster_id_col).agg(pl.len()).sort(cluster_id_col)
    return {row[0]: row[1] for row in cluster_sizes.iter_rows()}


@typecheck
def compute_chain_weights(
    chains: pl.DataFrame, alphas: Dict[str, float], beta: float
) -> pl.Series:
    """Computes the weights of the chains based on the cluster sizes."""
    molecule_idx = chains.get_column_index("molecule_id")
    cluster_idx = chains.get_column_index("cluster_id")
    cluster_sizes = get_cluster_sizes(chains, "cluster_id")

    return (
        chains.map_rows(
            lambda row: get_chain_weight(
                row[molecule_idx].split("-")[0],
                cluster_sizes[row[cluster_idx]],
                alphas,
                beta,
            ),
            return_dtype=pl.Float32,
        )
        .to_series(0)
        .rename("weight")
    )


@typecheck
def compute_interface_weights(
    interfaces: pl.DataFrame, alphas: Dict[str, float], beta: float
) -> pl.Series:
    """Computes the weights of the interfaces based on the chain weights."""
    molecule_idx_1 = interfaces.get_column_index("interface_molecule_id_1")
    molecule_idx_2 = interfaces.get_column_index("interface_molecule_id_2")
    cluster_idx = interfaces.get_column_index("interface_cluster_id")
    cluster_sizes = get_cluster_sizes(interfaces, "interface_cluster_id")

    return (
        interfaces.map_rows(
            lambda row: get_interface_weight(
                row[molecule_idx_1].split("-")[0],
                row[molecule_idx_2].split("-")[0],
                cluster_sizes[row[cluster_idx]],
                alphas,
                beta,
            ),
            return_dtype=pl.Float32,
        )
        .to_series(0)
        .rename("weight")
    )


class WeightedPDBSampler(Sampler[List[str]]):
    """
    Initializes a sampler for weighted sampling of PDB and chain/interface IDs.

    :param chain_mapping_paths: Path to the CSV file containing chain cluster
        mappings. If multiple paths are provided, they will be concatenated.
    :param interface_mapping_path: Path to the CSV file containing interface
        cluster mappings.
    :param batch_size: Number of PDB IDs to sample in each batch.
    :param beta_chain: Weighting factor for chain clusters.
    :param beta_interface: Weighting factor for interface clusters.
    :param alpha_prot: Weighting factor for protein chains.
    :param alpha_nuc: Weighting factor for nucleic acid chains.
    :param alpha_ligand: Weighting factor for ligand chains.
    :param pdb_ids_to_skip: List of PDB IDs to skip during sampling.
        Allow extra data filtering to ensure we avoid training
        on anomolous complexes that passed through all filtering
        and clustering steps.
    :param subset_to_ids: An optional list of mapping DataFrame indices
        to which to subset the original combined mapping DataFrame. This
        is primarily useful for debugging using a smaller set of clusters.

    Example:
    ```
    sampler = WeightedPDBSampler(...)
    for batch in sampler:
        print(batch)
    ```
    """

    def __init__(
        self,
        chain_mapping_paths: str | List[str],
        interface_mapping_path: str,
        batch_size: int,
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
        alpha_prot: float = 3.0,
        alpha_nuc: float = 3.0,
        alpha_ligand: float = 1.0,
        pdb_ids_to_skip: List[str] | None = None,
        pdb_ids_to_keep: list[str] | None = None,
    ):
        # Load chain and interface mappings
        if not isinstance(chain_mapping_paths, list):
            chain_mapping_paths = [chain_mapping_paths]

        chain_mapping = [pl.read_csv(path) for path in chain_mapping_paths]
        # Increment chain cluster IDs to avoid overlap
        chain_cluster_nums = [mapping.get_column("cluster_id").max() for mapping in chain_mapping]
        for i in range(1, len(chain_mapping)):
            chain_mapping[i] = chain_mapping[i].with_columns(
                (pl.col("cluster_id") + sum(chain_cluster_nums[:i])).alias("cluster_id")
            )

        chain_mapping = pl.concat(chain_mapping)
        interface_mapping = pl.read_csv(interface_mapping_path)

        # Filter out unwanted PDB IDs
        if exists(pdb_ids_to_skip) and len(pdb_ids_to_skip) > 0:
            chain_mapping = chain_mapping.filter(pl.col("pdb_id").is_in(pdb_ids_to_skip).not_())
            interface_mapping = interface_mapping.filter(
                pl.col("pdb_id").is_in(pdb_ids_to_skip).not_()
            )

        # Calculate weights for chains and interfaces
        self.alphas = {"prot": alpha_prot, "nuc": alpha_nuc, "ligand": alpha_ligand}
        self.betas = {"chain": beta_chain, "interface": beta_interface}
        self.batch_size = batch_size

        logger.info(
            "Precomputing chain and interface weights. This may take several minutes to complete."
        )

        # Subset to specific PDB IDs if provided
        if exists(pdb_ids_to_keep):
            chain_mapping = chain_mapping.filter(pl.col("pdb_id").is_in(pdb_ids_to_keep)).select(
                ["pdb_id", "chain_id", "molecule_id", "cluster_id"]
            )
            interface_mapping = interface_mapping.filter(
                pl.col("pdb_id").is_in(pdb_ids_to_keep)
            ).select(
                [
                    "pdb_id",
                    "interface_chain_id_1",
                    "interface_chain_id_2",
                    "interface_molecule_id_1",
                    "interface_molecule_id_2",
                    "interface_chain_cluster_id_1",
                    "interface_chain_cluster_id_2",
                    "interface_cluster_id",
                ]
            )

        chain_mapping.insert_column(
            len(chain_mapping.columns),
            compute_chain_weights(chain_mapping, self.alphas, self.betas["chain"]),
        )
        interface_mapping.insert_column(
            len(interface_mapping.columns),
            compute_interface_weights(interface_mapping, self.alphas, self.betas["interface"]),
        )

        logger.info("Finished precomputing chain and interface weights.")

        # Concatenate chain and interface mappings
        chain_mapping = chain_mapping.with_columns(
            [
                pl.col("chain_id").alias("chain_id_1"),
                pl.lit("").alias("chain_id_2"),
            ]
        )
        chain_mapping = chain_mapping.select(
            ["pdb_id", "chain_id_1", "chain_id_2", "cluster_id", "weight"]
        )

        interface_mapping = interface_mapping.with_columns(
            [
                pl.col("interface_chain_id_1").alias("chain_id_1"),
                pl.col("interface_chain_id_2").alias("chain_id_2"),
                (
                    pl.col("interface_cluster_id") + chain_mapping.get_column("cluster_id").max()
                ).alias("cluster_id"),
            ]
        )
        interface_mapping = interface_mapping.select(
            ["pdb_id", "chain_id_1", "chain_id_2", "cluster_id", "weight"]
        )
        self.mappings = chain_mapping.extend(interface_mapping)

        # Normalize weights
        self.weights = self.mappings.get_column("weight").to_numpy()
        self.weights = self.weights / self.weights.sum()

    def __len__(self) -> int:
        """Returns the number of batches in the dataset."""
        return len(self.mappings) // self.batch_size

    def __iter__(self) -> Iterator[List[str]]:
        """Returns an iterator that yields batches of PDB IDs."""
        while True:
            sampled = self.sample(self.batch_size)

            for (
                pdb_id,
                _,
                _,
            ) in sampled:
                yield pdb_id

    @typecheck
    def sample(self, batch_size: int) -> List[Tuple[str, str, str]]:
        """Samples a chain ID or interface ID based on the weights of the chains/interfaces."""
        indices = np.random.choice(len(self.mappings), size=batch_size, p=self.weights)
        return self.mappings[indices].select(["pdb_id", "chain_id_1", "chain_id_2"]).rows()

    @typecheck
    def cluster_based_sample(self, batch_size: int) -> List[Tuple[str, str, str]]:
        """
        Samples a chain ID or interface ID based on cluster IDs. For each batch, a number of cluster IDs
        are selected randomly, and a chain ID or interface ID is sampled from each cluster based on the
        weights of the chains/interfaces in the cluster.

        Warning! Significantly slower than the regular `sample` method.
        """
        cluster_ids = self.mappings.get_column("cluster_id").unique().sample(batch_size)

        pdb_chain_ids = []
        for cluster_id in cluster_ids:
            cluster = self.mappings.filter(pl.col("cluster_id") == cluster_id)
            if len(cluster) == 1:
                pdb_chain_ids.append(
                    cluster[0].select(["pdb_id", "chain_id_1", "chain_id_2"]).row(0)
                )
                continue
            cluster_weights = cluster.get_column("weight").to_numpy()
            cluster_weights = cluster_weights / cluster_weights.sum()
            idx = np.random.choice(len(cluster), p=cluster_weights)
            pdb_chain_ids.append(
                cluster[idx].select(["pdb_id", "chain_id_1", "chain_id_2"]).row(0)
            )

        return pdb_chain_ids


if __name__ == "__main__":
    interface_mapping_path = os.path.join(
        "data", "pdb_data", "data_caches", "clusterings", "interface_cluster_mapping.csv"
    )
    chain_mapping_paths = [
        os.path.join(
            "data", "pdb_data", "data_caches", "clusterings", "ligand_chain_cluster_mapping.csv"
        ),
        os.path.join(
            "data",
            "pdb_data",
            "data_caches",
            "clusterings",
            "nucleic_acid_chain_cluster_mapping.csv",
        ),
        os.path.join(
            "data", "pdb_data", "data_caches", "clusterings", "peptide_chain_cluster_mapping.csv"
        ),
        os.path.join(
            "data", "pdb_data", "data_caches", "clusterings", "protein_chain_cluster_mapping.csv"
        ),
    ]

    sampler = WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=64,
    )

    print(sampler.sample(64))
    print(sampler.cluster_based_sample(64))
