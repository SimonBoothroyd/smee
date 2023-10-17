import numpy
import openmm.unit
import pytest
import torch

import smee.converters
import smee.converters.openmm
import smee.mm
import smee.tests.utils
from smee.potentials.nonbonded import (
    _COULOMB_PRE_FACTOR,
    _compute_pme_exclusions,
    compute_coulomb_energy,
    compute_lj_energy,
    compute_pairwise,
    compute_pairwise_scales,
)


def _compute_openmm_energy(
    system: smee.TensorSystem,
    coords: torch.Tensor,
    box_vectors: torch.Tensor | None,
    potential: smee.TensorPotential,
) -> torch.Tensor:
    coords = coords.numpy() * openmm.unit.angstrom

    if box_vectors is not None:
        box_vectors = box_vectors.numpy() * openmm.unit.angstrom

    omm_force = smee.converters.convert_to_openmm_force(potential, system)

    omm_system = smee.converters.openmm.create_openmm_system(system)
    omm_system.addForce(omm_force)

    if box_vectors is not None:
        omm_system.setDefaultPeriodicBoxVectors(*box_vectors)

    omm_integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    omm_context = openmm.Context(omm_system, omm_integrator)

    if box_vectors is not None:
        omm_context.setPeriodicBoxVectors(*box_vectors)

    omm_context.setPositions(coords)

    omm_energy = omm_context.getState(getEnergy=True).getPotentialEnergy()
    omm_energy = omm_energy.value_in_unit(openmm.unit.kilocalories_per_mole)

    return torch.tensor(omm_energy, dtype=torch.float64)


def test_compute_pairwise_scales():
    system, force_field = smee.tests.utils.system_from_smiles(["C", "O"], [2, 3])

    vdw_potential = force_field.potentials_by_type["vdW"]
    vdw_potential.attributes = torch.tensor(
        [0.01, 0.02, 0.5, 1.0, 9.0, 2.0], dtype=torch.float64
    )

    scales = compute_pairwise_scales(system, vdw_potential)

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
        ],
        dtype=torch.float64
    )
    # fmt: on

    i, j = torch.triu_indices(system.n_particles, system.n_particles, 1)
    expected_scales = expected_scale_matrix[i, j]

    assert scales.shape == expected_scales.shape
    assert torch.allclose(scales, expected_scales)


def test_compute_pairwise_periodic():
    system = smee.TensorSystem(
        [
            smee.tests.utils.topology_from_smiles("[Ar]"),
            smee.tests.utils.topology_from_smiles("[Ne]"),
        ],
        [2, 3],
        True,
    )

    coords = torch.tensor(
        [
            [+0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [+4.0, 0.0, 0.0],
            [-8.0, 0.0, 0.0],
            [+8.0, 0.0, 0.0],
        ]
    )
    box_vectors = torch.eye(3) * 24.0

    cutoff = torch.tensor(9.0)

    pairwise = compute_pairwise(system, coords, box_vectors, cutoff)

    expected_idxs = torch.tensor(
        [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [2, 4], [3, 4]],
        dtype=torch.int32,
    )
    n_expected_pairs = len(expected_idxs)

    assert pairwise.idxs.shape == (n_expected_pairs, 2)
    assert torch.allclose(pairwise.idxs, expected_idxs)
    assert pairwise.idxs.dtype == torch.int32

    assert pairwise.deltas.shape == (n_expected_pairs, 3)
    assert pairwise.deltas.dtype == torch.float32

    expected_distances = torch.tensor([4.0, 4.0, 8.0, 8.0, 4.0, 8.0, 4.0, 8.0])
    assert torch.allclose(pairwise.distances, expected_distances)
    assert pairwise.distances.shape == (n_expected_pairs,)
    assert pairwise.distances.dtype == torch.float32

    assert torch.isclose(cutoff, pairwise.cutoff)


@pytest.mark.parametrize("with_batch", [True, False])
def test_compute_pairwise_non_periodic(with_batch):
    system = smee.TensorSystem(
        [
            smee.tests.utils.topology_from_smiles("[Ar]"),
            smee.tests.utils.topology_from_smiles("[Ne]"),
        ],
        [2, 3],
        False,
    )

    coords = torch.tensor(
        [
            [+0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [+4.0, 0.0, 0.0],
            [-8.0, 0.0, 0.0],
            [+8.0, 0.0, 0.0],
        ]
    )
    coords = coords if not with_batch else coords.unsqueeze(0)

    pairwise = compute_pairwise(system, coords, None, torch.tensor(9.0))

    expected_idxs = torch.tensor(
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
        ],
        dtype=torch.int32,
    )
    n_expected_pairs = len(expected_idxs)

    expected_batch_size = tuple() if not with_batch else (1,)

    assert pairwise.idxs.shape == (n_expected_pairs, 2)
    assert torch.allclose(pairwise.idxs, expected_idxs)
    assert pairwise.idxs.dtype == torch.int32

    assert pairwise.deltas.shape == (*expected_batch_size, n_expected_pairs, 3)
    assert pairwise.deltas.dtype == torch.float32

    expected_distances = torch.tensor(
        [4.0, 4.0, 8.0, 8.0, 8.0, 4.0, 12.0, 12.0, 4.0, 16.0]
    )
    expected_distances = (
        expected_distances if not with_batch else expected_distances.unsqueeze(0)
    )

    assert torch.allclose(pairwise.distances, expected_distances)
    assert pairwise.distances.shape == (*expected_batch_size, n_expected_pairs)
    assert pairwise.distances.dtype == torch.float32

    assert pairwise.cutoff is None


def test_compute_lj_energy_periodic(etoh_water_system):
    tensor_sys, tensor_ff, coords, box_vectors = etoh_water_system

    vdw_potential = tensor_ff.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True

    energy = compute_lj_energy(
        tensor_sys, vdw_potential, coords.float(), box_vectors.float()
    )
    energy.backward()

    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, box_vectors, vdw_potential
    )

    assert torch.isclose(energy, expected_energy, atol=1.0e-3)


def test_compute_lj_energy_non_periodic():
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(["CCC", "O"], [2, 3])
    tensor_sys.is_periodic = False

    coords, _ = smee.mm.generate_system_coords(tensor_sys)
    coords = torch.tensor(coords.value_in_unit(openmm.unit.angstrom))

    vdw_potential = tensor_ff.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True

    energy = compute_lj_energy(tensor_sys, vdw_potential, coords.float(), None)
    expected_energy = _compute_openmm_energy(tensor_sys, coords, None, vdw_potential)

    assert torch.isclose(energy, expected_energy, atol=1.0e-5)


def test_coulomb_pre_factor():
    # Compare against a value computed directly from C++ using the OpenMM 7.5.1
    # ONE_4PI_EPS0 define constant multiplied by 10 for nm -> A
    _KCAL_TO_KJ = 4.184

    assert numpy.isclose(_COULOMB_PRE_FACTOR * _KCAL_TO_KJ, 1389.3545764, atol=1.0e-7)


def test_compute_pme_exclusions():
    system, force_field = smee.tests.utils.system_from_smiles(["C", "O"], [2, 3])

    coulomb_potential = force_field.potentials_by_type["Electrostatics"]
    exclusions = _compute_pme_exclusions(system, coulomb_potential)

    expected_exclusions = torch.tensor(
        [
            # C #1
            [1, 2, 3, 4],
            [0, 2, 3, 4],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 2, 3],
            # C #2
            [6, 7, 8, 9],
            [5, 7, 8, 9],
            [5, 6, 8, 9],
            [5, 6, 7, 9],
            [5, 6, 7, 8],
            # O #1
            [11, 12, -1, -1],
            [10, 12, -1, -1],
            [10, 11, -1, -1],
            # O #2
            [14, 15, -1, -1],
            [13, 15, -1, -1],
            [13, 14, -1, -1],
            # O #3
            [17, 18, -1, -1],
            [16, 18, -1, -1],
            [16, 17, -1, -1],
        ]
    )

    assert exclusions.shape == expected_exclusions.shape
    assert torch.allclose(exclusions, expected_exclusions)


def test_compute_coulomb_energy_periodic(etoh_water_system):
    tensor_sys, tensor_ff, coords, box_vectors = etoh_water_system

    coulomb_potential = tensor_ff.potentials_by_type["Electrostatics"]
    coulomb_potential.parameters.requires_grad = True

    energy = compute_coulomb_energy(tensor_sys, coulomb_potential, coords, box_vectors)
    energy.backward()

    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, box_vectors, coulomb_potential
    )

    assert torch.isclose(energy, expected_energy, atol=1.0e-2)


def test_compute_coulomb_energy_non_periodic():
    tensor_sys, tensor_ff = smee.tests.utils.system_from_smiles(["CCC", "O"], [2, 3])
    tensor_sys.is_periodic = False

    coords, _ = smee.mm.generate_system_coords(tensor_sys)
    coords = torch.tensor(coords.value_in_unit(openmm.unit.angstrom))

    coulomb_potential = tensor_ff.potentials_by_type["Electrostatics"]
    coulomb_potential.parameters.requires_grad = True

    energy = compute_coulomb_energy(tensor_sys, coulomb_potential, coords.float(), None)
    expected_energy = _compute_openmm_energy(
        tensor_sys, coords, None, coulomb_potential
    )

    assert torch.isclose(energy, expected_energy, atol=1.0e-5)
