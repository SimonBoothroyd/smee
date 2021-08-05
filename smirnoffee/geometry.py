"""Common functions for computing the internal coordinates (e.g. bond lengths)."""

from typing import Tuple

import torch

_EPSILON = 1.0e-8


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
    vector_ab = vector_ab / torch.norm(vector_ab, dim=1).unsqueeze(1)

    vector_ac = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 2]]
    vector_ac = vector_ac / torch.norm(vector_ac, dim=1).unsqueeze(1)

    cos_angle = (vector_ab * vector_ac).sum(dim=1)
    # TODO: properly handle the acos singularity.
    cos_angle = torch.clamp(cos_angle, -1.0 + _EPSILON, 1.0 - _EPSILON)

    angles = torch.acos(cos_angle)
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
