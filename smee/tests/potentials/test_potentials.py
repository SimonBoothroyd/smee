import numpy
import openff.interchange
import openff.interchange.models
import openff.toolkit
import openff.units
import openmm
import openmm.unit
import pytest
import torch

import smee.converters
import smee.tests.utils
from smee.potentials import broadcast_parameters, compute_energy


def _place_v_sites(
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


def _compute_openmm_energy(
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
        The energy in units of [kcal / mol].
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
    energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)

    return torch.tensor(energy)


def test_broadcast_parameters():
    system, force_field = smee.tests.utils.system_from_smiles(["C", "O"], [2, 3])
    vdw_potential = force_field.potentials_by_type["vdW"]

    methane_top, water_top = system.topologies

    parameters = broadcast_parameters(system, vdw_potential)

    expected_methane_parameters = (
        methane_top.parameters["vdW"].assignment_matrix @ vdw_potential.parameters
    )
    expected_water_parameters = (
        water_top.parameters["vdW"].assignment_matrix @ vdw_potential.parameters
    )

    expected_parameters = torch.vstack(
        [expected_methane_parameters] * 2 + [expected_water_parameters] * 3
    )
    assert parameters.shape == expected_parameters.shape
    assert torch.allclose(parameters, expected_parameters)


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
    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    energy_smee = compute_energy(tensor_top, tensor_ff, conformer, None)
    energy_openmm = _compute_openmm_energy(interchange, conformer)

    assert torch.isclose(energy_smee, energy_openmm)


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
            torch.tensor(conformer_b + numpy.array([[3.0, 0.0, 0.0]])),
            torch.zeros((2, 3)),
        ]
    )

    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("tip4p_fb.offxml"), topology
    )
    conformer = _place_v_sites(conformer, interchange)

    tensor_ff, [tensor_top] = smee.converters.convert_interchange(interchange)

    energy_openmm = _compute_openmm_energy(interchange, conformer)
    energy_smee = compute_energy(tensor_top, tensor_ff, conformer)

    assert torch.isclose(energy_smee, energy_openmm)
