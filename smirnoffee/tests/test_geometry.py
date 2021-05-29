import numpy
import pytest
import torch

from smirnoffee.geometry import compute_angles, compute_bond_vectors, compute_dihedrals


def test_compute_bond_vectors():

    conformer = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    atom_indices = torch.tensor([[2, 0], [0, 1]])

    bond_vectors, bond_norms = compute_bond_vectors(conformer, atom_indices)

    assert torch.allclose(
        bond_vectors, torch.tensor([[0.0, -3.0, 0.0], [2.0, 0.0, 0.0]])
    )
    assert torch.allclose(bond_norms, torch.tensor([3.0, 2.0]))


def test_compute_angles():

    conformer = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    atom_indices = torch.tensor([[0, 1, 2], [1, 0, 2], [0, 1, 3]])

    angles = compute_angles(conformer, atom_indices)

    assert torch.allclose(angles, torch.tensor([numpy.pi / 2, numpy.pi / 4, numpy.pi]))


@pytest.mark.parametrize("phi_sign", [-1.0, 1.0])
def test_compute_dihedrals(phi_sign):

    conformer = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, phi_sign]]
    )
    atom_indices = torch.tensor([[0, 1, 2, 3]])

    dihedrals = compute_dihedrals(conformer, atom_indices)

    assert torch.allclose(dihedrals, torch.tensor([phi_sign * numpy.pi / 4.0]))
