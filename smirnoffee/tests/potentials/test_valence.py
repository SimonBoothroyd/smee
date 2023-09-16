import pytest
import torch

from smirnoffee.potentials.valence import (
    _compute_cosine_torsion_energy,
    compute_cosine_improper_torsion_energy,
    compute_cosine_proper_torsion_energy,
    compute_harmonic_angle_energy,
    compute_harmonic_bond_energy,
)


@pytest.mark.parametrize(
    "conformer, expected_shape",
    [
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.Size([]),
        ),
        (torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (1,)),
    ],
)
def test_compute_harmonic_bond_energy(conformer, expected_shape):
    atom_indices = torch.tensor([[0, 1], [0, 2]])
    parameters = torch.tensor([[2.0, 0.95], [0.5, 1.01]], requires_grad=True)

    energy = compute_harmonic_bond_energy(conformer, atom_indices, parameters)
    energy.backward()

    assert energy.shape == expected_shape

    assert torch.isclose(energy, torch.tensor(1.0 * 0.05**2 + 0.25 * 0.01**2))
    assert not torch.allclose(parameters.grad, torch.tensor(0.0))


@pytest.mark.parametrize(
    "conformer, expected_shape",
    [
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.Size([]),
        ),
        (torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (1,)),
    ],
)
def test_compute_harmonic_angle_energy(conformer, expected_shape):
    atom_indices = torch.tensor([[0, 1, 2]])
    parameters = torch.tensor([[2.0, 92.5]], requires_grad=True)

    energy = compute_harmonic_angle_energy(conformer, atom_indices, parameters)
    energy.backward()

    assert energy.shape == expected_shape

    expected_energy = 0.5 * parameters[0, 0] * (torch.pi / 2.0 - parameters[0, 1]) ** 2
    expected_gradient = torch.tensor(
        [
            0.5 * (torch.pi / 2.0 - parameters[0, 1]) ** 2,
            parameters[0, 0] * (parameters[0, 1] - torch.pi / 2.0),
        ]
    )

    assert torch.isclose(energy, expected_energy)
    assert torch.allclose(parameters.grad, expected_gradient)


@pytest.mark.parametrize("expected_shape", [torch.Size([]), (1,)])
@pytest.mark.parametrize(
    "energy_function",
    [
        _compute_cosine_torsion_energy,
        compute_cosine_proper_torsion_energy,
        compute_cosine_improper_torsion_energy,
    ],
)
@pytest.mark.parametrize("phi_sign", [-1.0, 1.0])
def test_compute_cosine_torsion_energy(expected_shape, energy_function, phi_sign):
    conformer = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, phi_sign]]
    )

    if expected_shape == (1,):
        conformer = torch.unsqueeze(conformer, 0)

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
                        parameters[0, 1] * torch.tensor(phi_sign * torch.pi / 4.0)
                        - parameters[0, 2]
                    ]
                )
            )
        )
    )

    assert torch.isclose(energy, expected_energy)
    assert energy.shape == expected_shape
