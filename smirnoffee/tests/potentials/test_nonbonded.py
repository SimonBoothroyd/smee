import numpy
import torch

from smirnoffee.potentials.nonbonded import (
    _COULOMB_PRE_FACTOR,
    compute_coulomb_energy,
    compute_lj_energy,
)


def test_coulomb_pre_factor():
    # Compare against a value computed directly from C++ using the OpenMM 7.5.1
    # ONE_4PI_EPS0 define constant multiplied by 10 for nm -> A
    assert numpy.isclose(_COULOMB_PRE_FACTOR, 1389.3545764, atol=1.0e-7)


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
