"""Compute internal coordinates (e.g. bond lengths)."""

import torch


def compute_bond_vectors(
    conformer: torch.Tensor, atom_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the vectors between each atom pair specified by the ``atom_indices`` as
    well as their norms.

    Args:
        conformer: The conformer [Å] to compute the bond vectors for with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        atom_indices: The indices of the atoms involved in each bond with
            ``shape=(n_bonds, 2)``

    Returns:
        The bond vectors and their norms [Å].
    """

    if len(atom_indices) == 0:
        return torch.tensor([]), torch.tensor([])

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    directions = conformer[:, atom_indices[:, 1]] - conformer[:, atom_indices[:, 0]]
    distances = torch.norm(directions, dim=-1)

    if not is_batched:
        directions = torch.squeeze(directions, dim=0)
        distances = torch.squeeze(distances, dim=0)

    return directions, distances


def compute_angles(conformer: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
    """Computes the angles [rad] between each atom triplet specified by the
    ``atom_indices``.

    Args:
        conformer: The conformer [Å] to compute the angles for with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        atom_indices: The indices of the atoms involved in each angle with
            ``shape=(n_angles, 3)``.

    Returns:
        The valence angles [rad].
    """

    if len(atom_indices) == 0:
        return torch.tensor([])

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    vector_ab = conformer[:, atom_indices[:, 1]] - conformer[:, atom_indices[:, 0]]
    vector_ac = conformer[:, atom_indices[:, 1]] - conformer[:, atom_indices[:, 2]]

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

    if not is_batched:
        angles = torch.squeeze(angles, dim=0)

    return angles


def compute_dihedrals(
    conformer: torch.Tensor, atom_indices: torch.Tensor
) -> torch.Tensor:
    """Computes the dihedral angles [rad] between each atom quartet specified by the
    ``atom_indices``.

    Args:
        conformer: The conformer [Å] to compute the dihedral angles for with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        atom_indices: The indices of the atoms involved in each dihedral angle with
            ``shape=(n_dihedrals, 4)``.

    Returns:
        The dihedral angles [rad].
    """

    if len(atom_indices) == 0:
        return torch.tensor([])

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    # Based on the OpenMM formalism.
    vector_ab = conformer[:, atom_indices[:, 0]] - conformer[:, atom_indices[:, 1]]
    vector_cb = conformer[:, atom_indices[:, 2]] - conformer[:, atom_indices[:, 1]]
    vector_cd = conformer[:, atom_indices[:, 2]] - conformer[:, atom_indices[:, 3]]

    vector_ab_cross_cb = torch.cross(vector_ab, vector_cb, dim=-1)
    vector_cb_cross_cd = torch.cross(vector_cb, vector_cd, dim=-1)

    vector_cb_norm = torch.norm(vector_cb, dim=-1).unsqueeze(-1)

    y = (
        torch.cross(vector_ab_cross_cb, vector_cb_cross_cd, dim=-1)
        * vector_cb
        / vector_cb_norm
    ).sum(axis=-1)

    x = (vector_ab_cross_cb * vector_cb_cross_cd).sum(axis=-1)

    phi = torch.atan2(y, x)

    if not is_batched:
        phi = torch.squeeze(phi, dim=0)

    return phi
