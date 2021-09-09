"""Common functions for computing the internal coordinates (e.g. bond lengths)."""

from typing import Tuple

import torch


def compute_bond_vectors(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the vectors between each atom pair specified by the ``atom_indices`` as
    well as their norms.

    Returns:
        A tuple of the vectors and norms.
    """

    if len(atom_indices) == 0:
        return torch.tensor([]), torch.tensor([])

    directions = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]]
    distances = torch.norm(directions, dim=1)

    return directions, distances


def compute_angles(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
) -> torch.Tensor:
    """Computes the angles [rad] between each atom triplet specified by the
    ``atom_indices``.

    Returns:
        A tensor of the valence angles.
    """

    if len(atom_indices) == 0:
        return torch.tensor([])

    vector_ab = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]]
    vector_ac = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 2]]

    # tan theta = sin theta / cos theta
    #
    # ||a x b|| = ||a|| ||b|| sin theta
    #   a . b   = ||a|| ||b|| cos theta
    #
    # => tan theta = (a x b) / (a . b)
    angles = torch.atan2(
        torch.norm(torch.cross(vector_ab, vector_ac, dim=-1), dim=-1),
        (vector_ab * vector_ac).sum(dim=-1),
    )

    return angles


def compute_dihedrals(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
) -> torch.Tensor:
    """Computes the dihedral angles [rad] between each atom quartet specified by the
    ``atom_indices``.

    Returns:
        A tensor of the dihedral angles.
    """

    if len(atom_indices) == 0:
        return torch.tensor([])

    # Based on the OpenMM formalism.
    vector_ab = conformer[atom_indices[:, 0]] - conformer[atom_indices[:, 1]]
    vector_cb = conformer[atom_indices[:, 2]] - conformer[atom_indices[:, 1]]
    vector_cd = conformer[atom_indices[:, 2]] - conformer[atom_indices[:, 3]]

    vector_ab_cross_cb = torch.cross(vector_ab, vector_cb, dim=1)
    vector_cb_cross_cd = torch.cross(vector_cb, vector_cd, dim=1)

    vector_cb_norm = torch.norm(vector_cb, dim=1).unsqueeze(1)

    y = (
        torch.cross(vector_ab_cross_cb, vector_cb_cross_cd, dim=1)
        * vector_cb
        / vector_cb_norm
    ).sum(axis=-1)

    x = (vector_ab_cross_cb * vector_cb_cross_cd).sum(axis=-1)

    phi = torch.atan2(y, x)
    return phi


def compute_linear_displacement(
    conformer: torch.Tensor, atom_indices: torch.Tensor
) -> torch.Tensor:
    """Computes the the displacement [Å] of the BA and BC unit vectors in the linear
    angle "ABC". The displacements are measured along two axes that are perpendicular to
    the AC unit vector.

    Notes:
        * This function is a port of the geomeTRIC ``LinearAngle.value`` function. See
          the main ``smirnoffee`` README for license information.

    Args:
        conformer: The cartesian coordinates of a conformer with shape=(n_atoms, 3) and
            units of [Å].
        atom_indices: A tensor containing the indices of the atoms in each linear angle
            (first three columns) and the index of the axis to compute the displacement
            along (last column) with shape=(n_linear_angles, 4).

    Returns:
        A tensor of the linear displacements.
    """

    vector_ab = conformer[atom_indices[:, 0]] - conformer[atom_indices[:, 1]]
    vector_ab = vector_ab / torch.norm(vector_ab, dim=1).unsqueeze(1)

    vector_cb = conformer[atom_indices[:, 2]] - conformer[atom_indices[:, 1]]
    vector_cb = vector_cb / torch.norm(vector_cb, dim=1).unsqueeze(1)

    vector_ca = conformer[atom_indices[:, 2]] - conformer[atom_indices[:, 0]]
    vector_ca = vector_ca / torch.norm(vector_ca, dim=1).unsqueeze(1)

    # Take the dot product of each row of ``vector_ca`` with the x, y and z axis
    # and find the index (0 = x, 1 = y, 2 = z) of the axis that is most perpendicular
    # to each row. This ensures we don't try and take the cross-product of two
    # co-linear vectors.
    #
    # This is the same approach taken by geomeTRIC albeit more-so vectorized.
    basis = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=conformer.dtype
    )
    basis_index = torch.argmin((vector_ca @ basis).square(), dim=-1)

    axis_0 = basis[basis_index]

    axis_1 = torch.cross(vector_ca, axis_0)
    axis_1 = axis_1 / torch.norm(axis_1, dim=1).unsqueeze(1)

    axis_2 = torch.cross(vector_ca, axis_1)
    axis_2 = axis_2 / torch.norm(axis_2, dim=1).unsqueeze(1)

    return torch.where(
        atom_indices[:, 3] == 0,
        (vector_ab * axis_1).sum(dim=-1) + (vector_cb * axis_1).sum(dim=-1),
        (vector_ab * axis_2).sum(dim=-1) + (vector_cb * axis_2).sum(dim=-1),
    )
