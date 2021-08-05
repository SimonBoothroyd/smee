"""Functions for evaluating non-bonded potential energy functions."""

import torch
from openff.units import unit

from smirnoffee.potentials import potential_energy_function

_COULOMB_PRE_FACTOR_UNITS = unit.kilojoule / unit.mole * unit.angstrom / unit.e ** 2
_COULOMB_PRE_FACTOR = (
    (unit.avogadro_constant / (4.0 * unit.pi * unit.eps0))
    .to(_COULOMB_PRE_FACTOR_UNITS)
    .magnitude
)


@potential_energy_function("vdW", "4*epsilon*((sigma/r)**12-(sigma/r)**6)")
def evaluate_lj_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of the vdW interactions using the
    standard Lennard-Jones potential.

    Notes:
        * No cutoff will be applied - this is consistent with OpenFF toolkit
          using the OpenMM `NoCutoff` method when creating an OpenMM system for
          a molecule in vacuum.

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The pairs of atom indices of the atoms involved in each
            interaction with shape=(n_pairs, 2).
        parameters: A tensor with shape=(n_pairs, 3) where there first and second column
            contains the epsilon [kJ / mol] and sigma [A] values of each pair
            respectively, and the third column a scale factor.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1)

    directions = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]]
    distances_sqr = (directions * directions).sum(dim=1)

    # TODO: Do you get any noticeable speed-up for x2 -> x4 -> x6 vs pow?
    inverse_distances_6 = torch.rsqrt(distances_sqr) ** 6
    inverse_distances_12 = inverse_distances_6 * inverse_distances_6

    sigma_6 = parameters[:, 1] ** 6
    sigma_12 = sigma_6 * sigma_6

    return (
        4.0
        * parameters[:, 0]
        * parameters[:, 2]
        * (sigma_12 * inverse_distances_12 - sigma_6 * inverse_distances_6)
    ).sum()


@potential_energy_function("Electrostatics", "coul")
def evaluate_coulomb_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of the electrostatic interactions
    using the standard Coulomb potential.

    Notes:
        * No cutoff will be applied - this is consistent with OpenFF toolkit
          using the OpenMM `NoCutoff` method when creating an OpenMM system for
          a molecule in vacuum.

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The pairs of atom indices of the atoms involved in each
            interaction with shape=(n_pairs, 2).
        parameters: A tensor with shape=(n_pairs, 2) where there first column
            contains the charge [e] on the first atom, the second column the charge [e]
            on the second atom, and the third column a scale factor.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1)

    directions = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]]
    distances_sqr = (directions * directions).sum(dim=1)

    inverse_distances = torch.rsqrt(distances_sqr)

    return (
        _COULOMB_PRE_FACTOR
        * parameters[:, 0]
        * parameters[:, 1]
        * parameters[:, 2]
        * inverse_distances
    ).sum()
