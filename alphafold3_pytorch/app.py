"""
Web Application Interface for AlphaFold3 Structure Prediction

This module provides an interactive web-based user interface for AlphaFold3 using
the Gradio framework. It allows users to:

- Build molecular complexes through an intuitive GUI
- Input proteins, RNA, DNA, ligands, and metal ions
- Visualize predicted 3D structures interactively
- Download predicted structures in PDB format

The application features:
    - Real-time input validation for sequence data
    - Support for multiple molecule types and copies
    - Interactive 3D structure visualization
    - Session-based caching for predictions
    - Automatic cleanup of temporary files

Technical Details:
    The web UI is built with Gradio and uses Molecule3D for structure visualization.
    Structure predictions are cached per-session and automatically cleaned up after
    timeout (10-60 minutes of inactivity).

Usage:
    $ python -m alphafold3_pytorch.app --checkpoint weights.ckpt --cache-dir ./cache

Example:
    # Start the web server
    $ alphafold3-app --checkpoint model.ckpt
    # Then navigate to the provided URL (usually http://localhost:7860)
"""

import click
from pathlib import Path

import secrets
import shutil
from Bio.PDB import PDBIO

from alphafold3_pytorch import Alphafold3, Alphafold3Input

# Global constants and state
# These are module-level variables shared across requests

model = None        # Loaded AlphaFold3 model (loaded once on startup)
cache_path = None   # Directory for caching prediction outputs
pdb_writer = PDBIO()  # BioPython writer for PDB format output

# Main structure prediction function

def fold(entities, request):
    """
    Run AlphaFold3 structure prediction for the given molecular entities.

    This function takes a list of molecular entities (proteins, nucleic acids, ligands, ions),
    prepares them as input to the AlphaFold3 model, runs inference, and saves the predicted
    structure to a PDB file.

    Args:
        entities: List of dictionaries, each containing:
                  - mol_type: Type of molecule ('Protein', 'RNA', 'DNA', 'Ligand', 'Ion')
                  - sequence: Sequence string or identifier
                  - num_copies: Number of copies to include in the complex
        request: Gradio request object containing session information

    Returns:
        str: Path to the saved PDB file containing the predicted structure

    Example:
        >>> entities = [
        ...     {'mol_type': 'Protein', 'sequence': 'MKTAY...', 'num_copies': 1},
        ...     {'mol_type': 'DNA', 'sequence': 'ATCG', 'num_copies': 2}
        ... ]
        >>> pdb_path = fold(entities, request)
    """
    # Initialize lists for each molecule type
    proteins = []
    rnas = []
    dnas = []
    ligands = []
    ions = []

    # Organize entities by molecule type and duplicate based on num_copies
    # This allows modeling multi-chain complexes and homomers
    for entity in entities:
        if entity["mol_type"] == "Protein":
            proteins.extend([entity["sequence"]] * entity["num_copies"])
        elif entity["mol_type"] == "RNA":
            rnas.extend([entity["sequence"]] * entity["num_copies"])
        elif entity["mol_type"] == "DNA":
            dnas.extend([entity["sequence"]] * entity["num_copies"])
        elif entity["mol_type"] == "Ligand":
            ligands.extend([entity["sequence"]] * entity["num_copies"])
        elif entity["mol_type"] == "Ion":
            ions.extend([entity["sequence"]] * entity["num_copies"])

    # Prepare the input data structure for AlphaFold3
    # Alphafold3Input handles feature generation and format conversion
    alphafold3_input = Alphafold3Input(
        proteins=proteins,
        ss_dna=dnas,     # Single-stranded DNA
        ss_rna=rnas,     # Single-stranded RNA
        ligands=ligands,
        metal_ions=ions,
    )

    # Run model inference to predict structure
    # Set to eval mode to disable dropout and other training-specific behaviors
    model.eval()
    (structure,) = model.forward_with_alphafold3_inputs(
        alphafold3_inputs=alphafold3_input,
        return_bio_pdb_structures=True,  # Return BioPython Structure object
    )

    # Generate unique output path for this prediction
    # Uses session hash for isolation and random token for uniqueness
    global cache_path, pdb_writer
    output_path = cache_path / str(request.session_hash) / f"{secrets.token_urlsafe(8)}.pdb"
    output_path.parent.mkdir(exist_ok=True)

    # Save predicted structure to PDB file
    pdb_writer.set_structure(structure)
    pdb_writer.save(str(output_path))

    return str(output_path)

# Gradio-specific helper functions

def delete_cache(request):
    """
    Clean up cached prediction files for a user session.

    This function is called automatically when a user's session ends or times out.
    It removes all prediction files associated with the session to free up disk space.

    Args:
        request: Gradio request object containing session_hash

    Note:
        This function is registered with Gradio's unload event and runs automatically.
    """
    if not request.session_hash:
        return

    user_dir: Path = cache_path / request.session_hash
    if user_dir.exists():
        shutil.rmtree(str(user_dir))


def start_gradio_app():
    """
    Initialize and launch the Gradio web application interface.

    This function sets up the complete web UI including:
    - Entity input controls (molecule type, sequence, copies)
    - Validation for protein, DNA, and RNA sequences
    - Support for ligands and metal ions from predefined lists
    - Interactive 3D structure visualization
    - Add/delete functionality for molecular entities
    - Prediction button that triggers structure folding

    The UI layout includes:
    - Input section for adding molecular entities
    - Entity list showing all added components
    - Prediction button and 3D structure viewer
    - Automatic session cleanup

    Note:
        Cache cleanup is configured with a timeout of 600-3600 seconds (10-60 minutes).
        The app automatically cleans up old prediction files after this period.
    """
    import gradio as gr
    from gradio_molecule3d import Molecule3D

    # Configure Gradio app with automatic cache cleanup
    # delete_cache=(min_seconds, max_seconds) sets cleanup interval
    with gr.Blocks(delete_cache=(600, 3600)) as gradio_app:
        entities = gr.State([])

        with gr.Row():
            gr.Markdown("### AlphaFold3 PyTorch Web UI")

        with gr.Row():
            gr.Column(scale=8)
            # upload_json_button = gr.Button("Upload JSON", scale=1, min_width=100)
            clear_button = gr.Button("Clear", scale=1, min_width=100)

        with gr.Row():
            with gr.Column(scale=1, min_width=150):
                mtype = gr.Dropdown(
                    value="Protein",
                    label="Molecule type",
                    choices=["Protein", "DNA", "RNA", "Ligand", "Ion"],
                    interactive=True,
                )
            with gr.Column(scale=1, min_width=80):
                c = gr.Number(
                    value=1,
                    label="Copies",
                    interactive=True,
                )

            with gr.Column(scale=8, min_width=200):

                @gr.render(inputs=mtype)
                def render_sequence(mol_type):
                    if mol_type in ["Protein", "DNA", "RNA"]:
                        seq = gr.Textbox(
                            label="Paste sequence or fasta",
                            placeholder="Input",
                            interactive=True,
                        )
                    elif mol_type == "Ligand":
                        seq = gr.Dropdown(
                            label="Select ligand",
                            choices=[
                                "ADP - Adenosine disphosphate",
                                "ATP - Adenosine triphosphate",
                                "AMP - Adenosine monophosphate",
                                "GTP - Guanosine-5'-triphosphate",
                                "GDP - Guanosine-5'-diphosphate",
                                "FAD - Flavin adenine dinucleotide",
                                "NAD - Nicotinamide-adenine-dinucleotide",
                                "NAP - Nicotinamide-adenine-dinucleotide phosphate (NADP)",
                                "NDP - Dihydro-nicotinamide-adenine-dinucleotide-phosphate (NADPH)",
                                "HEM - Heme",
                                "HEC - Heme C",
                                "OLA - Oleic acid",
                                "MYR - Myristic acid",
                                "CIT - Citric acid",
                                "CLA - Chlorophyll A",
                                "CHL - Chlorophyll B",
                                "BCL - Bacteriochlorophyll A",
                                "BCB - Bacteriochlorophyll B",
                            ],
                            interactive=True,
                        )
                    elif mol_type == "Ion":
                        seq = gr.Dropdown(
                            label="Select ion",
                            choices=[
                                "Mg²⁺",
                                "Zn²⁺",
                                "Cl⁻",
                                "Ca²⁺",
                                "Na⁺",
                                "Mn²⁺",
                                "K⁺",
                                "Fe³⁺",
                                "Cu²⁺",
                                "Co²⁺",
                            ],
                            interactive=True,
                        )

                    add_button.click(add_entity, inputs=[entities, mtype, c, seq], outputs=[entities])
                    clear_button.click(lambda: ("Protein", 1, None), None, outputs=[mtype, c, seq])

        add_button = gr.Button("Add entity", scale=1, min_width=100)

        def add_entity(entities, mtype="Protein", c=1, seq=""):
            if seq is None or len(seq) == 0:
                gr.Info("Input required")
                return entities

            seq_norm = seq.strip(" \t\n\r").upper()

            if mtype in ["Protein", "DNA", "RNA"]:
                if mtype == "Protein" and any([x not in "ARDCQEGHILKMNFPSTWYV" for x in seq_norm]):
                    gr.Info("Invalid protein sequence. Allowed characters: A, R, D, C, Q, E, G, H, I, L, K, M, N, F, P, S, T, W, Y, V")
                    return entities

                if mtype == "DNA" and any([x not in "ACGT" for x in seq_norm]):
                    gr.Info("Invalid DNA sequence. Allowed characters: A, C, G, T")
                    return entities

                if mtype == "RNA" and any([x not in "ACGU" for x in seq_norm]):
                    gr.Info("Invalid RNA sequence. Allowed characters: A, C, G, U")
                    return entities

                if len(seq) < 4:
                    gr.Info("Minimum 4 characters required")
                    return entities

            elif mtype == "Ligand":
                if seq is None or len(seq) == 0:
                    gr.Info("Select a ligand")
                    return entities
                seq_norm = seq.split(" - ")[0]
            elif mtype == "Ion":
                if seq is None or len(seq) == 0:
                    gr.Info("Select an ion")
                    return entities
                seq_norm = "".join([x for x in seq if x.isalpha()])

            new_entity = {"mol_type": mtype, "num_copies": c, "sequence": seq_norm}

            return entities + [new_entity]

        @gr.render(inputs=entities)
        def render_entities(entity_list):
            for idx, entity in enumerate(entity_list):
                with gr.Row():
                    gr.Text(
                        value=entity["mol_type"],
                        label="Type",
                        scale=1,
                        min_width=90,
                        interactive=False,
                    )
                    gr.Text(
                        value=entity["num_copies"],
                        label="Copies",
                        scale=1,
                        min_width=80,
                        interactive=False,
                    )

                    sequence = entity["sequence"]
                    if entity["mol_type"] not in ["Ligand", "Ion"]:
                        # Split every 10 characters, and add a \t after each split
                        sequence = "\t".join([sequence[i : i + 10] for i in range(0, len(sequence), 10)])

                    gr.Text(
                        value=sequence,
                        label="Sequence",
                        placeholder="Input",
                        scale=7,
                        min_width=200,
                        interactive=False,
                    )

                    del_button = gr.Button("🗑️", scale=0, min_width=50)

                    def delete(entity_id=idx):
                        entity_list.pop(entity_id)
                        return entity_list

                    del_button.click(delete, None, outputs=[entities])

        pred_button = gr.Button("Predict", scale=1, min_width=100)
        output_mol = Molecule3D(label="Output structure", config={"backgroundColor": "black"})

        pred_button.click(fold, inputs=entities, outputs=output_mol)
        clear_button.click(lambda: ([], None), None, outputs=[entities, output_mol])

        gradio_app.unload(delete_cache)
        gradio_app.launch()

# CLI entry point for launching the web app
@click.command()
@click.option("-ckpt", "--checkpoint", type=str, help="path to alphafold3 checkpoint", required=True)
@click.option("-cache", "--cache-dir", type=str, help="path to output cache", required=False, default="cache")
@click.option("-prec", "--precision", type=str, help="precision to use", required=False, default="float32")
def app(checkpoint: str, cache_dir: str, precision: str):
    """
    Launch the AlphaFold3 web application.

    This command starts a Gradio web server that provides an interactive interface
    for structure prediction. The server loads the model once at startup and serves
    predictions to multiple users.

    Args:
        checkpoint: Path to the pre-trained AlphaFold3 model checkpoint file
        cache_dir: Directory for caching prediction outputs (created if doesn't exist)
        precision: Precision for model inference ('float32', 'float16', 'bfloat16')
                   Note: Currently not fully implemented

    Example:
        # Start web app with default cache directory
        $ alphafold3-app --checkpoint weights.ckpt

        # Start with custom cache directory
        $ alphafold3-app --checkpoint weights.ckpt --cache-dir /tmp/af3_cache

    Note:
        The cache directory is cleared on startup to ensure fresh predictions.
        Old predictions from previous sessions will be deleted.
    """
    path = Path(checkpoint)
    assert path.exists(), "checkpoint does not exist at path"

    global cache_path
    cache_path = Path(cache_dir)

    if cache_path.exists():
        shutil.rmtree(str(cache_path))

    cache_path.mkdir(exist_ok=True)

    global model
    model = Alphafold3.init_and_load(str(path))
    # To device and quantize?
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # try:
    #     dtype = getattr(torch, precision)
    # except AttributeError:
    #     print(f"Invalid precision: {precision}. Using float32")
    #     dtype = torch.float32
    # model.to(device, dtype=dtype)

    start_gradio_app()
