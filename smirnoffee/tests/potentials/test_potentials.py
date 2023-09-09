import openff.interchange
import openff.toolkit
import openff.units
import pytest
import torch

from smirnoffee.ff.smirnoff import convert_interchange
from smirnoffee.potentials import evaluate_energy


def evaluate_openmm_energy(
    interchange: openff.interchange.Interchange,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the potential energy of a molecule in a specified conformer using a
    specified force field.

    Args:
        interchange: The interchange object containing the applied force field
            parameters.
        conformer: The conformer [Å] of the molecule.

    Returns:
        The energy in units of [kJ / mol].
    """

    import openmm.unit

    openmm_system = interchange.to_openmm()

    openmm_context = openmm.Context(
        openmm_system,
        openmm.VerletIntegrator(0.1),
        openmm.Platform.getPlatformByName("Reference"),
    )
    openmm_context.setPositions(
        (conformer.numpy() * openmm.unit.angstrom).value_in_unit(openmm.unit.nanometers)
    )

    state = openmm_context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)

    return torch.tensor(energy)


@pytest.mark.parametrize(
    "smiles",
    [
        "C1=NC2=C(N1)C(=O)NC(=N2)N",
        "C",
        "CO",
        "C=O",
        "c1ccccc1",
        "Cc1ccccc1",
        "c1cocc1",
        "CC(=O)NC1=CC=C(C=C1)O",
    ],
)
def test_evaluate_energy(smiles: str):
    molecule = openff.toolkit.Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)

    conformer = torch.tensor(molecule.conformers[0].m_as(openff.units.unit.angstrom))
    conformer += torch.randn((molecule.n_atoms, 1)) * 0.25

    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff_unconstrained-2.0.0.offxml"),
        molecule.to_topology(),
    )

    force_field, parameters_per_topology = convert_interchange(interchange)

    energy_smirnoffee = evaluate_energy(
        parameters_per_topology[0], conformer, force_field
    )
    energy_openmm = evaluate_openmm_energy(interchange, conformer)

    assert torch.isclose(energy_smirnoffee, energy_openmm)
