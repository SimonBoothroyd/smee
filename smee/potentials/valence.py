"""Valence potential energy functions."""

import torch

import smee.geometry
import smee.potentials

_EPSILON = 1.0e-8


@smee.potentials.potential_energy_fn("Bonds", "k/2*(r-length)**2")
def compute_harmonic_bond_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kcal / mol] of a set of bonds for a given
    conformer using a harmonic potential of the form:

    `1/2 * k * (r - length) ** 2`

    Args:
        conformer: The conformer [Å] to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each bond with
            shape=(n_bonds, 2).
        parameters: A tensor with shape=(n_bonds, 2) where there first column
            contains the force constants ``k`` [kcal / mol / Å^2], and the second the
            equilibrium bond ``length`` [Å].

    Returns:
        The evaluated potential energy [kcal / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1 if conformer.ndim == 2 else (conformer.shape[0],))

    _, distances = smee.geometry.compute_bond_vectors(conformer, atom_indices)

    return (0.5 * parameters[:, 0] * (distances - parameters[:, 1]) ** 2).sum(-1)


@smee.potentials.potential_energy_fn("Angles", "k/2*(theta-angle)**2")
def compute_harmonic_angle_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kcal / mol] of a set of valence angles
    for a given conformer using a harmonic potential of the form:

    `1/2 * k * (theta - angle) ** 2`

    Args:
        conformer: The conformer [Å] to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each valence angle with
            shape=(n_angles, 3).
        parameters: A tensor with shape=(n_angles, 2) where there first column
            contains the force constants ``k`` [kJ / mol / rad^2], and the second the
            equilibrium ``angle`` [rad].

    Returns:
        The evaluated potential energy [kcal / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1 if conformer.ndim == 2 else (conformer.shape[0],))

    angles = smee.geometry.compute_angles(conformer, atom_indices)

    return (0.5 * parameters[:, 0] * (angles - parameters[:, 1]) ** 2).sum(-1)


def _compute_cosine_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kcal / mol] of a set of torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer [Å] to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each proper torsion with
            shape=(n_torsions, 4).
        parameters: A tensor with shape=(n_torsions, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kcal / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1 if conformer.ndim == 2 else (conformer.shape[0],))

    phi = smee.geometry.compute_dihedrals(conformer, atom_indices)

    return (
        parameters[:, 0]
        / parameters[:, 3]
        * (1.0 + torch.cos(parameters[:, 1] * phi - parameters[:, 2]))
    ).sum(-1)


@smee.potentials.potential_energy_fn(
    "ProperTorsions", "k*(1+cos(periodicity*theta-phase))"
)
def compute_cosine_proper_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kcal / mol] of a set of proper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer [Å] to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each proper torsion with
            shape=(n_propers, 4).
        parameters: A tensor with shape=(n_propers, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kcal / mol].
    """
    return _compute_cosine_torsion_energy(conformer, atom_indices, parameters)


@smee.potentials.potential_energy_fn(
    "ImproperTorsions", "k*(1+cos(periodicity*theta-phase))"
)
def compute_cosine_improper_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kcal / mol] of a set of improper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer [Å] to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each improper torsion with
            shape=(n_impropers, 4).
        parameters: A tensor with shape=(n_impropers, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kcal / mol].
    """
    return _compute_cosine_torsion_energy(conformer, atom_indices, parameters)