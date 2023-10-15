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
from smee.potentials import broadcast_exclusions, broadcast_parameters, compute_energy


@pytest.fixture()
def mock_lj_potential() -> smee.TensorPotential:
    return smee.TensorPotential(
        type="vdW",
        fn="LJ",
        parameters=torch.tensor([[0.1, 1.1], [0.2, 2.1], [0.3, 3.1]]),
        parameter_keys=[
            openff.interchange.models.PotentialKey(id="[#1:1]"),
            openff.interchange.models.PotentialKey(id="[#6:1]"),
            openff.interchange.models.PotentialKey(id="[#8:1]"),
        ],
        parameter_cols=("epsilon", "sigma"),
        parameter_units=(
            openff.units.unit.kilojoule_per_mole,
            openff.units.unit.angstrom,
        ),
        attributes=torch.tensor([0.0, 0.0, 0.5, 1.0, 9.0, 2.0]),
        attribute_cols=(
            "scale_12",
            "scale_13",
            "scale_14",
            "scale_15",
            "cutoff",
            "switch_width",
        ),
        attribute_units=(
            openff.units.unit.dimensionless,
            openff.units.unit.dimensionless,
            openff.units.unit.dimensionless,
            openff.units.unit.dimensionless,
            openff.units.unit.angstrom,
            openff.units.unit.angstrom,
        ),
    )


@pytest.fixture()
def mock_methane_top() -> smee.TensorTopology:
    methane_top = smee.tests.utils.topology_from_smiles("C")
    methane_top.parameters = {
        "vdW": smee.NonbondedParameterMap(
            assignment_matrix=torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ).to_sparse(),
            exclusions=torch.tensor(
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [1, 2],
                    [1, 3],
                    [1, 4],
                    [2, 3],
                    [2, 4],
                    [3, 4],
                ]
            ),
            exclusion_scale_idxs=torch.tensor([[0] * 4 + [1] * 6]),
        )
    }
    return methane_top


@pytest.fixture()
def mock_water_top() -> smee.TensorTopology:
    methane_top = smee.tests.utils.topology_from_smiles("O")
    methane_top.parameters = {
        "vdW": smee.NonbondedParameterMap(
            assignment_matrix=torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ).to_sparse(),
            exclusions=torch.tensor([[0, 1], [0, 2], [1, 2]]),
            exclusion_scale_idxs=torch.tensor([[0], [0], [1]]),
        )
    }
    return methane_top


def test_broadcast_parameters(mock_lj_potential, mock_methane_top, mock_water_top):
    system = smee.TensorSystem([mock_methane_top, mock_water_top], [2, 3], True)

    parameters = broadcast_parameters(system, mock_lj_potential)

    methane_parameters = (
        mock_methane_top.parameters["vdW"].assignment_matrix
        @ mock_lj_potential.parameters
    )
    water_parameters = (
        mock_water_top.parameters["vdW"].assignment_matrix
        @ mock_lj_potential.parameters
    )

    expected_parameters = torch.vstack(
        [methane_parameters] * 2 + [water_parameters] * 3
    )
    assert parameters.shape == expected_parameters.shape
    assert torch.allclose(parameters, expected_parameters)


def test_broadcast_exclusions(mock_lj_potential, mock_methane_top, mock_water_top):
    mock_lj_potential.attributes = torch.tensor([0.01, 0.02, 0.5, 1.0, 9.0, 2.0])

    system = smee.TensorSystem([mock_methane_top, mock_water_top], [2, 3], True)

    scales = broadcast_exclusions(system, mock_lj_potential)

    # fmt: off
    expected_scale_matrix = torch.tensor(
        [
            [1.0, 0.01, 0.01, 0.01, 0.01] + [1.0] * (system.n_particles - 5),
            [0.01, 1.0, 0.02, 0.02, 0.02] + [1.0] * (system.n_particles - 5),
            [0.01, 0.02, 1.0, 0.02, 0.02] + [1.0] * (system.n_particles - 5),
            [0.01, 0.02, 0.02, 1.0, 0.02] + [1.0] * (system.n_particles - 5),
            [0.01, 0.02, 0.02, 0.02, 1.0] + [1.0] * (system.n_particles - 5),
            #
            [1.0] * 5 + [1.0, 0.01, 0.01, 0.01, 0.01] + [1.0] * (system.n_particles - 10),
            [1.0] * 5 + [0.01, 1.0, 0.02, 0.02, 0.02] + [1.0] * (system.n_particles - 10),
            [1.0] * 5 + [0.01, 0.02, 1.0, 0.02, 0.02] + [1.0] * (system.n_particles - 10),
            [1.0] * 5 + [0.01, 0.02, 0.02, 1.0, 0.02] + [1.0] * (system.n_particles - 10),
            [1.0] * 5 + [0.01, 0.02, 0.02, 0.02, 1.0] + [1.0] * (system.n_particles - 10),
            #
            [1.0] * 10 + [1.0, 0.01, 0.01] + [1.0] * (system.n_particles - 13),
            [1.0] * 10 + [0.01, 1.0, 0.02] + [1.0] * (system.n_particles - 13),
            [1.0] * 10 + [0.01, 0.02, 1.0] + [1.0] * (system.n_particles - 13),
            #
            [1.0] * 13 + [1.0, 0.01, 0.01] + [1.0] * (system.n_particles - 16),
            [1.0] * 13 + [0.01, 1.0, 0.02] + [1.0] * (system.n_particles - 16),
            [1.0] * 13 + [0.01, 0.02, 1.0] + [1.0] * (system.n_particles - 16),
            #
            [1.0] * 16 + [1.0, 0.01, 0.01],
            [1.0] * 16 + [0.01, 1.0, 0.02],
            [1.0] * 16 + [0.01, 0.02, 1.0],
        ]
    )
    # fmt: on

    i, j = torch.triu_indices(system.n_particles, system.n_particles, 1)
    expected_scales = expected_scale_matrix[i, j]

    assert scales.shape == expected_scales.shape
    assert torch.allclose(scales, expected_scales)


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

    force_field, parameters_per_topology = smee.converters.convert_interchange(
        interchange
    )

    energy_smee = compute_energy(
        parameters_per_topology[0].parameters, conformer, force_field
    )
    energy_openmm = compute_openmm_energy(interchange, conformer)

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
    conformer = place_v_sites(conformer, interchange)

    force_field, topologies = smee.converters.convert_interchange(interchange)

    energy_openmm = compute_openmm_energy(interchange, conformer)
    energy_smee = compute_energy(topologies[0].parameters, conformer, force_field)

    assert torch.isclose(energy_smee, energy_openmm)
