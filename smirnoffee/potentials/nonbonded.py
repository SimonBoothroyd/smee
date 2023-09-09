"""Non-bonded potential energy functions."""

import openff.units
import torch

import smirnoffee.potentials

_UNIT = openff.units.unit

_COULOMB_PRE_FACTOR_UNITS = _UNIT.kilojoule / _UNIT.mole * _UNIT.angstrom / _UNIT.e**2
_COULOMB_PRE_FACTOR = (_UNIT.avogadro_constant / (4.0 * _UNIT.pi * _UNIT.eps0)).m_as(
    _COULOMB_PRE_FACTOR_UNITS
)
_COULOMB_POTENTIAL = "coul"

_LJ_POTENTIAL = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"


def _lorentz_berthelot(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon_a = parameters[:, 0, 0]
    epsilon_b = parameters[:, 1, 0]

    sigma_a = parameters[:, 0, 1]
    sigma_b = parameters[:, 1, 1]

    return (epsilon_a * epsilon_b).sqrt(), 0.5 * (sigma_a + sigma_b)


@smirnoffee.potentials.potential_energy_fn("vdW", _LJ_POTENTIAL)
def evaluate_lj_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
    global_parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of the vdW interactions using the
    standard Lennard-Jones potential.

    Notes:
        * No cutoff will be applied.

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The pairs of atom indices of the atoms involved in each
            interaction with shape=(n_pairs, 2).
        parameters: A tensor containing the epsilon [kJ / mol] and sigma [Å] values
            of each interaction pair, with ``shape=(n_pairs, 2, 2)``. Here
            ``parameters[pair_idx][atom_idx][0] = epsilon``,
            ``parameters[pair_idx][atom_idx][1] = sigma``.
        global_parameters: A tensor containing the scale factor for each pair, with
            ``shape=(n_pairs, 1)``.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1)

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    directions = conformer[:, atom_indices[:, 1]] - conformer[:, atom_indices[:, 0]]
    distances_sqr = (directions * directions).sum(dim=-1)

    inverse_distances_6 = torch.rsqrt(distances_sqr) ** 6
    inverse_distances_12 = inverse_distances_6 * inverse_distances_6

    epsilon, sigma = _lorentz_berthelot(parameters)

    sigma_6 = sigma**6
    sigma_12 = sigma_6 * sigma_6

    energy = (
        4.0
        * epsilon
        * (sigma_12 * inverse_distances_12 - sigma_6 * inverse_distances_6)
    )
    # 1-n scale factors
    energy *= global_parameters

    energy = energy.sum(-1)

    if not is_batched:
        energy = torch.squeeze(energy, 0)

    return energy


@smirnoffee.potentials.potential_energy_fn("Electrostatics", _COULOMB_POTENTIAL)
def evaluate_coulomb_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
    global_parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of the electrostatic interactions
    using the standard Coulomb potential.

    Notes:
        * No cutoff will be applied.

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The pairs of atom indices of the atoms involved in each
            interaction with shape=(n_pairs, 2).
        parameters: A tensor containing the charges [e] on each atom in each interaction
            pair, with ``shape=(n_pairs, 2, 1)``.
        global_parameters: A tensor containing the scale factor for each pair, with
            ``shape=(n_pairs, 1)``.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    if len(atom_indices) == 0:
        return torch.zeros(1)

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    directions = conformer[:, atom_indices[:, 1]] - conformer[:, atom_indices[:, 0]]
    distances_sqr = (directions * directions).sum(dim=-1)

    inverse_distances = torch.rsqrt(distances_sqr)

    energy = (
        _COULOMB_PRE_FACTOR
        * parameters[:, 0, 0]
        * parameters[:, 1, 0]
        * global_parameters
        * inverse_distances
    ).sum(-1)

    if not is_batched:
        energy = torch.squeeze(energy, 0)

    return energy
