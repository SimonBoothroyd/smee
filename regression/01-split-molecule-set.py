import gzip
import os
import shutil
from tempfile import NamedTemporaryFile

import click
from rdkit import Chem, RDLogger


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    default="NCI-molecules.sdf.gz",
    show_default=True,
    help="The path to the input GZipped tarball of SDF files.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "-n",
    "--n-molecules",
    "chunk_size",
    default=250,
    show_default=True,
    help="The number of molecules to include in each chunk.",
    type=click.INT,
)
@click.option(
    "-o",
    "--output",
    "output_directory",
    default="01-split-molecules",
    show_default=True,
    help="The directory to save the split files in.",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
)
def split_molecules(input_path, chunk_size, output_directory):
    """Split a GZipped SDF file into smaller chunks.

    Splitting the molecule set into chunks allows the computation of the energies
    to be more easily distributed across molecule compute resources.

    Any molecules which could not be processed by RDKit will be skipped.
    """

    rdkit_logger = RDLogger.logger()
    rdkit_logger.setLevel(RDLogger.CRITICAL)

    os.makedirs(output_directory, exist_ok=True)

    output_name = os.path.basename(input_path).split(".")[0]

    with NamedTemporaryFile(suffix=".sdf") as unzipped_file:

        with gzip.open(input_path, "rb") as zipped_file:
            shutil.copyfileobj(zipped_file, unzipped_file)

        chunk_counter = 0
        chunk_index = 1

        output_stream = Chem.SDWriter(
            os.path.join(output_directory, f"{output_name}-{chunk_index}.sdf")
        )

        with click.progressbar(
            (
                rd_molecule
                for rd_molecule in Chem.SupplierFromFilename(
                    unzipped_file.name,
                    removeHs=False,
                    sanitize=True,
                    strictParsing=True,
                )
                if isinstance(rd_molecule, Chem.Mol)
            ),
        ) as rdkit_molecules:

            for rdkit_molecule in rdkit_molecules:

                if chunk_counter >= chunk_size:
                    chunk_index += 1

                    output_stream.close()

                    output_stream = Chem.SDWriter(
                        os.path.join(
                            output_directory, f"{output_name}-{chunk_index}.sdf"
                        )
                    )

                    chunk_counter = 0

                chunk_counter += 1

                output_stream.write(rdkit_molecule)

        output_stream.close()


if __name__ == "__main__":
    split_molecules()
