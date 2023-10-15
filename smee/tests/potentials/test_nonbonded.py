import numpy
import openff.interchange
import openff.toolkit
import openmm.unit
import torch

import smee.converters
import smee.converters.openmm
import smee.mm
from smee.potentials.nonbonded import (
    _COULOMB_PRE_FACTOR,
    _compute_lj_energy_periodic,
    compute_coulomb_energy,
    compute_lj_energy,
)


def test_coulomb_pre_factor():
    # Compare against a value computed directly from C++ using the OpenMM 7.5.1
    # ONE_4PI_EPS0 define constant multiplied by 10 for nm -> A
    _KCAL_TO_KJ = 4.184

    assert numpy.isclose(_COULOMB_PRE_FACTOR * _KCAL_TO_KJ, 1389.3545764, atol=1.0e-7)


def test_compute_coulomb_energy_two_particle():
    scale_factor = 5.0

    distance = 2.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    exclusions = torch.tensor([[0, 1]])

    q_a, q_b = 0.25, 0.75

    parameters = torch.tensor([[q_a], [q_b]])
    exclusion_scales = torch.tensor([scale_factor])

    actual_energy = compute_coulomb_energy(
        conformer, parameters, exclusions, exclusion_scales=exclusion_scales
    )
    expected_energy = scale_factor * _COULOMB_PRE_FACTOR * q_a * q_b / distance

    assert torch.isclose(torch.tensor(expected_energy), actual_energy)


def test_compute_coulomb_energy_three_particle():
    scale_factor = 5.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    exclusions = torch.tensor([[0, 2], [0, 1], [1, 2]])

    q_a, q_b, q_c = 0.25, 0.75, 0.50

    parameters = torch.tensor([[q_a], [q_b], [q_c]])
    exclusion_scales = torch.tensor([[scale_factor], [scale_factor], [scale_factor]])

    actual_energy = compute_coulomb_energy(
        conformer, parameters, exclusions, exclusion_scales
    )

    expected_energy = torch.tensor(
        _COULOMB_PRE_FACTOR
        * scale_factor
        * (q_a * q_b / 3.0 + q_a * q_c / 4.0 + q_b * q_c / 5.0),
    )

    assert torch.isclose(expected_energy, actual_energy)


def test_compute_lj_energy_two_particle():
    distance = 2.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])

    eps_a, sigma_a = 0.1, 2.0
    eps_b, sigma_b = 0.3, 4.0

    scale_factor = 5.0

    parameters = torch.tensor([[eps_a, sigma_a], [eps_b, sigma_b]])

    eps_ab = torch.sqrt(torch.tensor(eps_a * eps_b))
    sigma_ab = 0.5 * torch.tensor(sigma_a + sigma_b)

    actual_energy = compute_lj_energy(
        conformer,
        parameters,
        torch.tensor([[0, 1]]),
        torch.tensor([[scale_factor]]),
    )
    expected_energy = (
        (1.0 + (scale_factor - 1.0))
        * 4.0
        * eps_ab
        * ((sigma_ab / distance) ** 12 - (sigma_ab / distance) ** 6)
    )

    assert torch.isclose(expected_energy, actual_energy)


def test_compute_lj_energy_three_particle():
    scale_factor = 5.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

    eps_a, sigma_a = 0.1, 2.0
    eps_b, sigma_b = 0.5, 6.0
    eps_c, sigma_c = 0.3, 4.0

    parameters = torch.tensor([[eps_a, sigma_a], [eps_b, sigma_b], [eps_c, sigma_c]])

    actual_energy = compute_lj_energy(
        conformer,
        parameters,
        torch.tensor([[1, 0], [2, 0], [1, 2]]),
        torch.tensor([[scale_factor], [scale_factor], [scale_factor]]),
    )

    eps_ab = torch.sqrt(torch.tensor(eps_a * eps_b))
    sigma_ab = 0.5 * torch.tensor(sigma_a + sigma_b)
    eps_ac = torch.sqrt(torch.tensor(eps_a * eps_c))
    sigma_ac = 0.5 * torch.tensor(sigma_a + sigma_c)
    eps_bc = torch.sqrt(torch.tensor(eps_b * eps_c))
    sigma_bc = 0.5 * torch.tensor(sigma_b + sigma_c)

    expected_energy = (
        scale_factor * 4.0 * eps_ab * ((sigma_ab / 3.0) ** 12 - (sigma_ab / 3.0) ** 6)
        + scale_factor * 4.0 * eps_ac * ((sigma_ac / 4.0) ** 12 - (sigma_ac / 4.0) ** 6)
        + scale_factor * 4.0 * eps_bc * ((sigma_bc / 5.0) ** 12 - (sigma_bc / 5.0) ** 6)
    )

    assert torch.isclose(expected_energy, actual_energy)


def test_compute_lj_energy_periodic():
    interchanges = [
        openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField("openff-2.0.0.offxml"),
            openff.toolkit.Molecule.from_smiles(smiles).to_topology(),
        )
        for smiles in ["CCO", "O"]
    ]

    tensor_ff, tensor_tops = smee.converters.convert_interchange(interchanges)
    tensor_sys = smee.TensorSystem(tensor_tops, [67, 123], is_periodic=True)

    vdw_potential = tensor_ff.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True

    coords, box_vectors = smee.mm.generate_system_coords(tensor_sys)

    energy = _compute_lj_energy_periodic(
        tensor_sys,
        torch.tensor(coords.value_in_unit(openmm.unit.angstrom), dtype=torch.float32),
        torch.tensor(
            box_vectors.value_in_unit(openmm.unit.angstrom), dtype=torch.float32
        ),
        vdw_potential,
    )
    energy.backward()

    omm_force = smee.converters.convert_to_openmm_force(vdw_potential, tensor_sys)
    omm_force.setUseSwitchingFunction(True)
    omm_force.setUseDispersionCorrection(True)

    omm_system = smee.converters.openmm.create_openmm_system(tensor_sys)
    omm_system.addForce(omm_force)
    omm_system.setDefaultPeriodicBoxVectors(*box_vectors)

    omm_integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    omm_context = openmm.Context(omm_system, omm_integrator)
    omm_context.setPeriodicBoxVectors(*box_vectors)
    omm_context.setPositions(coords)

    omm_energy = omm_context.getState(getEnergy=True).getPotentialEnergy()
    omm_energy = omm_energy.value_in_unit(openmm.unit.kilocalories_per_mole)

    assert torch.isclose(
        energy, torch.tensor(omm_energy, dtype=torch.float64), atol=1.0e-3
    )
