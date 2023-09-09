import numpy
import pytest
import torch
import torch.autograd.functional

from smirnoffee.geometry import compute_angles, compute_bond_vectors, compute_dihedrals


@pytest.mark.parametrize(
    "geometry_function", [compute_angles, compute_dihedrals, compute_bond_vectors]
)
def test_compute_geometry_no_atoms(geometry_function):
    valence_terms = geometry_function(torch.tensor([]), torch.tensor([]))

    if not isinstance(valence_terms, tuple):
        valence_terms = (valence_terms,)

    assert all(term.shape == torch.Size([0]) for term in valence_terms)


def test_compute_bond_vectors():
    conformer = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    atom_indices = torch.tensor([[2, 0], [0, 1]])

    bond_vectors, bond_norms = compute_bond_vectors(conformer, atom_indices)

    assert bond_vectors.shape == (2, 3)
    assert bond_norms.shape == (2,)

    assert torch.allclose(
        bond_vectors, torch.tensor([[0.0, -3.0, 0.0], [2.0, 0.0, 0.0]])
    )
    assert torch.allclose(bond_norms, torch.tensor([3.0, 2.0]))


def test_compute_bond_vectors_batched():
    conformer = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ]
    )

    atom_indices = torch.tensor([[2, 0], [0, 1]])

    bond_vectors, bond_norms = compute_bond_vectors(conformer, atom_indices)

    assert bond_vectors.shape == (2, 2, 3)
    assert bond_norms.shape == (2, 2)

    assert torch.allclose(
        bond_vectors,
        torch.tensor(
            [
                [[-4.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[-9.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ]
        ),
    )
    assert torch.allclose(bond_norms, torch.tensor([[4.0, 2.0], [9.0, 3.0]]))


def test_compute_angles():
    conformer = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    atom_indices = torch.tensor([[0, 1, 2], [1, 0, 2], [0, 1, 3]])

    angles = compute_angles(conformer, atom_indices)
    assert angles.shape == (3,)

    assert torch.allclose(angles, torch.tensor([numpy.pi / 2, numpy.pi / 4, numpy.pi]))

    # Make sure there are no singularities in the gradients.
    gradients = torch.autograd.functional.jacobian(
        lambda x: compute_angles(x, atom_indices), conformer
    )
    assert not torch.isnan(gradients).any() and not torch.isinf(gradients).any()


def test_compute_angles_batched():
    conformer = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]],
            [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    atom_indices = torch.tensor([[0, 1, 2]])

    angles = compute_angles(conformer, atom_indices)
    assert angles.shape == (2, 1)

    assert torch.allclose(angles, torch.tensor([[numpy.pi / 4], [numpy.pi / 2]]))


@pytest.mark.parametrize("phi_sign", [-1.0, 1.0])
def test_compute_dihedrals(phi_sign):
    conformer = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, phi_sign]]
    )
    atom_indices = torch.tensor([[0, 1, 2, 3]])

    dihedrals = compute_dihedrals(conformer, atom_indices)
    assert dihedrals.shape == (1,)

    assert torch.allclose(dihedrals, torch.tensor([phi_sign * numpy.pi / 4.0]))

    # Make sure there are no singularities in the gradients.
    gradients = torch.autograd.functional.jacobian(
        lambda x: compute_dihedrals(x, atom_indices), conformer
    )
    assert not torch.isnan(gradients).any() and not torch.isinf(gradients).any()


def test_compute_dihedrals_batched():
    conformer = torch.tensor(
        [
            [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]],
        ]
    )
    atom_indices = torch.tensor([[0, 1, 2, 3]])

    dihedrals = compute_dihedrals(conformer, atom_indices)
    assert dihedrals.shape == (2, 1)

    assert torch.allclose(dihedrals, torch.tensor([[0.0], [numpy.pi / 4.0]]))

    # Make sure there are no singularities in the gradients.
    gradients = torch.autograd.functional.jacobian(
        lambda x: compute_dihedrals(x, atom_indices), conformer
    )
    assert not torch.isnan(gradients).any() and not torch.isinf(gradients).any()
