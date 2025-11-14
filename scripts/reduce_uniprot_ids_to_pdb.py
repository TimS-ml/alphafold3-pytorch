"""
UniProt ID to PDB ID Mapping Reducer.

This script extracts PDB-related entries from a compressed UniProt ID mapping file.
It filters a large UniProt ID mapping archive to retain only those entries that map
UniProt accession IDs to PDB database entries, significantly reducing the file size
for downstream processing.

Usage:
    python reduce_uniprot_ids_to_pdb.py

Input:
    - idmapping.dat.gz: Compressed UniProt ID mapping file containing mappings to various databases

Output:
    - uniprot_to_pdb_id_mapping.dat: Text file with only PDB-related mapping entries

The script efficiently processes large compressed files using generator expressions to
minimize memory usage.
"""

import gzip
import os


def filter_pdb_lines(file_path: str, output_file_path: str):
    """
    Filter lines containing PDB entries from a compressed UniProt mapping file.

    This function reads a compressed `.dat.gz` file line by line, identifies entries
    that contain mappings to the PDB (Protein Data Bank) database, and writes only
    those filtered lines to an output file. This reduces the file size by excluding
    mappings to other databases like GO, RefSeq, etc.

    The function uses a memory-efficient generator expression to process the file
    without loading the entire contents into memory, which is important for large
    UniProt mapping files that can be several gigabytes in size.

    Args:
        file_path: Path to the compressed `.dat.gz` UniProt ID mapping file to be read.
        output_file_path: Path to the output file where filtered PDB mapping lines will be written.

    Returns:
        None

    Example:
        >>> filter_pdb_lines('idmapping.dat.gz', 'uniprot_to_pdb_mapping.dat')
        # Creates a new file with only PDB-related mappings

    Note:
        The function looks for lines containing the tab-separated pattern '\tPDB\t' which
        indicates a mapping to the PDB database in the UniProt ID mapping file format.
    """
    # Open the compressed input file in text mode and the output file for writing
    with gzip.open(file_path, "rt") as infile, open(output_file_path, "w") as outfile:
        # Use a generator expression to efficiently filter lines containing PDB mappings
        # This avoids loading the entire file into memory at once
        pdb_lines = (line for line in infile if "\tPDB\t" in line)
        # Write all filtered lines to the output file
        outfile.writelines(pdb_lines)


if __name__ == "__main__":
    # Define the input compressed UniProt ID mapping file
    # This file should be downloaded from UniProt: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/
    input_archive_file = "idmapping.dat.gz"

    # Define the output path for the filtered PDB mappings
    # The output will contain only UniProt-to-PDB ID mappings
    output_file = os.path.join(
        "..", "data", "afdb_data", "data_caches", "uniprot_to_pdb_id_mapping.dat"
    )

    # Execute the filtering process to extract PDB-related entries
    filter_pdb_lines(input_archive_file, output_file)
