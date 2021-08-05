import functools
import os
from multiprocessing import Pool
from typing import Tuple

import click
import pandas
import torch
from openff.interchange.components.interchange import Interchange
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from rdkit import Chem, RDLogger
from simtk import openmm, unit

from smirnoffee.potentials.potentials import evaluate_system_energy


def _run_openmm(molecule: Molecule, system: openmm.System):
    """
    Minimize molecule with specified system and return the positions of the optimized
    molecule.
    """

    integrator = openmm.VerletIntegrator(0.1 * unit.femtoseconds)

    platform = openmm.Platform.getPlatformByName("Reference")

    openmm_context = openmm.Context(system, integrator, platform)
    openmm_context.setPositions(molecule.conformers[0].value_in_unit(unit.nanometer))

    openmm.LocalEnergyMinimizer.minimize(openmm_context)

    conformer = openmm_context.getState(getPositions=True).getPositions(asNumpy=True)
    energy = openmm_context.getState(getEnergy=True).getPotentialEnergy()

    return conformer.value_in_unit(unit.angstrom), energy.value_in_unit(
        unit.kilojoules_per_mole
    )


def _evaluate_energies(
    rd_molecule: Chem.Mol, force_field_path: str
) -> Tuple[str, float, float, bool]:

    try:

        molecule: Molecule = Molecule.from_rdkit(
            rd_molecule, allow_undefined_stereo=True
        )

        molecule = (
            [molecule]
            + molecule.enumerate_stereoisomers(max_isomers=1, rationalise=False)
        )[-1]

        smiles = molecule.to_smiles(explicit_hydrogens=False)

        molecule.generate_conformers(n_conformers=1)

        force_field = ForceField(force_field_path)

        openmm_system = force_field.create_openmm_system(molecule.to_topology())
        openff_system = Interchange.from_smirnoff(force_field, molecule.to_topology())

        conformer, openmm_energy = _run_openmm(molecule, openmm_system)
        conformer = torch.from_numpy(conformer)

        openff_energy = evaluate_system_energy(openff_system, conformer)

    except:
        return "", 0.0, 0.0, True

    print(smiles, openff_energy.numpy(), openmm_energy)
    return smiles, float(openff_energy.numpy()), float(openmm_energy), False


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    show_default=True,
    help="The path to the input SDF file.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "-ff",
    "--force-field",
    "force_field_path",
    default="openff_unconstrained-1.0.0.offxml",
    show_default=True,
    help="The force field to use to evaluate the energies.",
    type=click.STRING,
)
@click.option(
    "-n",
    "--n-procs",
    "n_processes",
    default=1,
    show_default=True,
    help="The number of processes to parallelize over.",
    type=click.INT,
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default="02-energies",
    show_default=True,
    help="The path to save the computed energies to (.csv).",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
def evaluate_energies(input_path, force_field_path, output_path, n_processes):
    """Computes the energies of a set of molecules using `smirnoffee` and OpenMM.

    The output is stored in a pandas friendly CSV file.
    """

    rdkit_logger = RDLogger.logger()
    rdkit_logger.setLevel(RDLogger.CRITICAL)

    click.echo("1) Evaluating the molecule energies.")

    with Pool(n_processes) as pool:

        energies_iterator = pool.imap_unordered(
            functools.partial(_evaluate_energies, force_field_path=force_field_path),
            (
                rd_molecule
                for rd_molecule in Chem.SupplierFromFilename(
                    input_path, removeHs=False, sanitize=True, strictParsing=True
                )
                if isinstance(rd_molecule, Chem.Mol)
            ),
        )

        energies = [
            {
                "SMILES": smiles,
                "E OpenFF": openff_energy,
                "E OpenMM": openmm_energy,
            }
            for smiles, openff_energy, openmm_energy, skip in energies_iterator
            if not skip
        ]

    click.echo("2) Saving the energies.")

    output_directory = os.path.dirname(output_path)

    if len(output_directory) > 0 and output_directory != ".":
        os.makedirs(output_directory, exist_ok=True)

    data_frame = pandas.DataFrame(energies)
    data_frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    evaluate_energies()
