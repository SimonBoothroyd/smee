import pytest
import torch

from smirnoffee.potentials.valence import (
    _evaluate_cosine_torsion_energy,
    evaluate_cosine_improper_torsion_energy,
    evaluate_cosine_proper_torsion_energy,
    evaluate_harmonic_angle_energy,
    evaluate_harmonic_bond_energy,
)


def test_evaluate_harmonic_bond_energy():

    conformer = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atom_indices = torch.tensor([[0, 1], [0, 2]])
    parameters = torch.tensor([[2.0, 0.95], [0.5, 1.01]], requires_grad=True)

    energy = evaluate_harmonic_bond_energy(conformer, atom_indices, parameters)
    energy.backward()

    assert torch.isclose(energy, torch.tensor(1.0 * 0.05 ** 2 + 0.25 * 0.01 ** 2))
    assert not torch.allclose(parameters.grad, torch.tensor(0.0))


def test_evaluate_harmonic_angle_energy():

    conformer = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atom_indices = torch.tensor([[0, 1, 2]])
    parameters = torch.tensor([[2.0, 92.5]], requires_grad=True)

    energy = evaluate_harmonic_angle_energy(conformer, atom_indices, parameters)
    energy.backward()

    expected_energy = 0.5 * parameters[0, 0] * (90.0 - parameters[0, 1]) ** 2
    expected_gradient = torch.tensor(
        [
            0.5 * (90.0 - parameters[0, 1]) ** 2,
            parameters[0, 0] * (parameters[0, 1] - 90.0),
        ]
    )

    assert torch.isclose(energy, expected_energy)
    assert torch.allclose(parameters.grad, expected_gradient)


@pytest.mark.parametrize(
    "energy_function",
    [
        _evaluate_cosine_torsion_energy,
        evaluate_cosine_proper_torsion_energy,
        evaluate_cosine_improper_torsion_energy,
    ],
)
@pytest.mark.parametrize("phi_sign", [-1.0, 1.0])
def test_evaluate_cosine_torsion_energy(energy_function, phi_sign):

    conformer = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, phi_sign]]
    )
    atom_indices = torch.tensor([[0, 1, 2, 3]])
    parameters = torch.tensor([[2.0, 2.0, 20.0, 1.5]], requires_grad=True)

    energy = energy_function(conformer, atom_indices, parameters)
    energy.backward()

    expected_energy = (
        parameters[0, 0]
        / parameters[0, 3]
        * (
            1.0
            + torch.cos(
                torch.tensor(
                    [
                        parameters[0, 1] * torch.deg2rad(torch.tensor(phi_sign * 45.0))
                        - torch.deg2rad(parameters[0, 2])
                    ]
                )
            )
        )
    )

    assert torch.isclose(energy, expected_energy)
