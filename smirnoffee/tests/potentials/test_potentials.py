import numpy
import openff.interchange
import openff.toolkit
import openff.units
import openmm
import openmm.unit
import pytest
import torch

from smirnoffee.ff import convert_interchange
from smirnoffee.potentials import compute_energy


def place_v_sites(
    conformer: torch.Tensor, interchange: openff.interchange.Interchange
) -> torch.Tensor:
    conformer = conformer.numpy() * openmm.unit.angstrom

    openmm_system = interchange.to_openmm()
    openmm_context = openmm.Context(
        openmm_system,
        openmm.VerletIntegrator(0.1),
        openmm.Platform.getPlatformByName("Reference"),
    )
    openmm_context.setPositions(conformer)
    openmm_context.computeVirtualSites()
    conformer = openmm_context.getState(getPositions=True).getPositions(asNumpy=True)
    return torch.tensor(conformer.value_in_unit(openmm.unit.angstrom))


def compute_openmm_energy(
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

    if openmm_system.getNumParticles() != interchange.topology.n_atoms:
        for _ in range(interchange.topology.n_atoms - openmm_system.getNumParticles()):
            openmm_system.addParticle(1.0)

    openmm_context = openmm.Context(
        openmm_system,
        openmm.VerletIntegrator(0.1),
        openmm.Platform.getPlatformByName("Reference"),
    )
    openmm_context.setPositions(conformer.numpy() * openmm.unit.angstrom)
    openmm_context.computeVirtualSites()

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
def test_compute_energy(smiles: str):
    molecule = openff.toolkit.Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)

    conformer = torch.tensor(molecule.conformers[0].m_as(openff.units.unit.angstrom))
    conformer += torch.randn((molecule.n_atoms, 1)) * 0.25

    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff_unconstrained-2.0.0.offxml"),
        molecule.to_topology(),
    )

    force_field, parameters_per_topology = convert_interchange(interchange)

    energy_smirnoffee = compute_energy(
        parameters_per_topology[0].parameters, conformer, force_field
    )
    energy_openmm = compute_openmm_energy(interchange, conformer)

    assert torch.isclose(energy_smirnoffee, energy_openmm)


def test_compute_energy_v_sites():
    molecule_a = openff.toolkit.Molecule.from_smiles("O")
    molecule_a.generate_conformers(n_conformers=1)
    molecule_b = openff.toolkit.Molecule.from_smiles("O")
    molecule_b.generate_conformers(n_conformers=1)

    topology = openff.toolkit.Topology()
    topology.add_molecule(molecule_a)
    topology.add_molecule(molecule_b)

    conformer_a = molecule_a.conformers[0].m_as(openff.units.unit.angstrom)
    conformer_b = molecule_b.conformers[0].m_as(openff.units.unit.angstrom)

    conformer = torch.vstack(
        [
            torch.tensor(conformer_a),
            torch.zeros((1, 3)),  # interchange for now interleaves v-sites
            torch.tensor(conformer_b + numpy.array([[3.0, 0.0, 0.0]])),
            torch.zeros((1, 3)),
        ]
    )

    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("tip4p_fb.offxml"), topology
    )
    conformer = place_v_sites(conformer, interchange)

    force_field, topologies = convert_interchange(interchange)

    energy_openmm = compute_openmm_energy(interchange, conformer)
    energy_smirnoffee = compute_energy(
        topologies[0].parameters,
        conformer[[0, 1, 2, 4, 5, 6, 3, 7], :],
        force_field,
    )

    assert torch.isclose(energy_smirnoffee, energy_openmm)
