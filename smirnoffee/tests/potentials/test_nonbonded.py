import numpy
import torch

from smirnoffee.potentials.nonbonded import (
    _COULOMB_PRE_FACTOR,
    evaluate_coulomb_energy,
    evaluate_lj_energy,
)


def test_coulomb_pre_factor():

    # Compare against a value computed directly from C++ using the OpenMM 7.5.1
    # ONE_4PI_EPS0 define constant multiplied by 10 for nm -> A
    assert numpy.isclose(_COULOMB_PRE_FACTOR, 1389.3545764, atol=1.0e-7)


def test_evaluate_coulomb_energy_two_particle():

    scale_factor = 5.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atom_indices = torch.tensor([[0, 1]])

    parameters = torch.tensor([[0.25, 0.75, scale_factor]])

    actual_energy = evaluate_coulomb_energy(conformer, atom_indices, parameters)
    expected_energy = _COULOMB_PRE_FACTOR * 0.25 * 0.75 * 5.0 / 2.0

    assert torch.isclose(torch.tensor(expected_energy), actual_energy)


def test_evaluate_coulomb_energy_three_particle():

    scale_factor = 5.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    atom_indices = torch.tensor([[0, 2], [0, 1], [1, 2]])

    parameters = torch.tensor(
        [
            [0.25, 0.75, scale_factor],
            [0.25, 0.50, scale_factor],
            [0.50, 0.75, scale_factor],
        ]
    )

    actual_energy = evaluate_coulomb_energy(conformer, atom_indices, parameters)

    expected_energy = torch.tensor(
        _COULOMB_PRE_FACTOR
        * 5.0
        * (0.25 * 0.75 / 4.0 + 0.25 * 0.50 / 3.0 + 0.50 * 0.75 / 5.0),
    )

    assert torch.isclose(expected_energy, actual_energy)


def test_evaluate_lj_energy_two_particle():

    scale_factor = 5.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atom_indices = torch.tensor([[0, 1]])

    parameters = torch.tensor([[0.25, 4.0, scale_factor]])

    actual_energy = evaluate_lj_energy(conformer, atom_indices, parameters)
    expected_energy = (
        scale_factor
        * 4.0
        * parameters[0, 0]
        * ((parameters[0, 1] / 2.0) ** 12 - (parameters[0, 1] / 2.0) ** 6)
    )

    assert torch.isclose(torch.tensor(expected_energy), actual_energy)


def test_evaluate_lj_energy_three_particle():

    scale_factor = 5.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    atom_indices = torch.tensor([[0, 2], [0, 1], [1, 2]])

    parameters = torch.tensor(
        [
            [0.25, 0.75, scale_factor],
            [0.25, 0.50, scale_factor],
            [0.50, 0.75, scale_factor],
        ]
    )

    actual_energy = evaluate_lj_energy(conformer, atom_indices, parameters)

    expected_energy = torch.tensor(
        scale_factor
        * 4.0
        * parameters[0, 0]
        * ((parameters[0, 1] / 4.0) ** 12 - (parameters[0, 1] / 4.0) ** 6)
        + scale_factor
        * 4.0
        * parameters[1, 0]
        * ((parameters[1, 1] / 3.0) ** 12 - (parameters[1, 1] / 3.0) ** 6)
        + scale_factor
        * 4.0
        * parameters[2, 0]
        * ((parameters[2, 1] / 5.0) ** 12 - (parameters[2, 1] / 5.0) ** 6)
    )

    assert torch.isclose(expected_energy, actual_energy)
