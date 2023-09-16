"""Non-bonded potential energy functions."""

import openff.units
import torch

import smee.potentials

_UNIT = openff.units.unit

_COULOMB_PRE_FACTOR_UNITS = (
    _UNIT.kilocalorie / _UNIT.mole * _UNIT.angstrom / _UNIT.e**2
)
_COULOMB_PRE_FACTOR = (_UNIT.avogadro_constant / (4.0 * _UNIT.pi * _UNIT.eps0)).m_as(
    _COULOMB_PRE_FACTOR_UNITS
)
_COULOMB_POTENTIAL = "coul"

_LJ_POTENTIAL = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"


def _lorentz_berthelot(
    epsilon_a: torch.Tensor,
    epsilon_b: torch.Tensor,
    sigma_a: torch.Tensor,
    sigma_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (epsilon_a * epsilon_b).sqrt(), 0.5 * (sigma_a + sigma_b)


def compute_pairwise(
    conformer: torch.Tensor,
    exclusions: torch.Tensor,
    exclusion_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the pairwise squared distance between all pairs of particles, and the
    corresponding scale factor based off any exclusions.

    Args:
        conformer: The conformer(s).
        exclusions: A tensor containing pairs of atom indices whose interaction should
            be scaled by ``exclusion_scales`` with ``shape=(n_exclusions, 2)``.
        exclusion_scales: A tensor containing the scale factor for each exclusion pair
            with ``shape=(n_exclusions, 1)``.

    Returns:
        The particle indices of each pair with ``shape=(n_pairs, 2)``, the squared
        distance between each pair with ``shape=(B, n_pairs)``, and the scale factor
        for each pair with ``shape=(n_pairs,)``.
    """

    n_particles = conformer.shape[-2]

    pair_idxs = torch.triu_indices(n_particles, n_particles, 1).T
    pair_scales = torch.ones(len(pair_idxs))

    if len(exclusions) > 0:
        exclusions, _ = exclusions.sort(dim=1)

        i, j = exclusions[:, 0], exclusions[:, 1]
        exclusions_1d = ((i * (2 * n_particles - i - 1)) / 2 + j - i - 1).int()

        pair_scales[exclusions_1d] = exclusion_scales.squeeze(-1)

    directions = conformer[:, pair_idxs[:, 1], :] - conformer[:, pair_idxs[:, 0], :]
    distances_sqr = (directions * directions).sum(dim=-1)

    return pair_idxs, distances_sqr, pair_scales


@smee.potentials.potential_energy_fn("vdW", _LJ_POTENTIAL)
def compute_lj_energy(
    conformer: torch.Tensor,
    parameters: torch.Tensor,
    exclusions: torch.Tensor,
    exclusion_scales: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kcal / mol] of the vdW interactions using the
    standard Lennard-Jones potential.

    Notes:
        * No cutoff will be applied.

    Args:
        conformer: The conformer [Å] to evaluate the potential at.
        parameters: A tensor containing the epsilon [kcal / mol] and sigma [Å] values
            of each particle, with ``shape=(n_particles, 2)``.
        exclusions: A tensor containing pairs of atom indices whose interaction should
            be scaled by ``exclusion_scales`` with ``shape=(n_exclusions, 2)``.
        exclusion_scales: A tensor containing the scale factor for each exclusion pair
            with ``shape=(n_exclusions, 1)``.

    Returns:
        The evaluated potential energy [kcal / mol].
    """

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    pair_idxs, distances_sqr, pair_scales = compute_pairwise(
        conformer, exclusions, exclusion_scales
    )

    epsilon, sigma = _lorentz_berthelot(
        parameters[pair_idxs[:, 0], 0],
        parameters[pair_idxs[:, 1], 0],
        parameters[pair_idxs[:, 0], 1],
        parameters[pair_idxs[:, 1], 1],
    )

    x = sigma**6 / (distances_sqr**3)

    energy = (pair_scales * 4.0 * epsilon * (x * (x - 1.0))).sum(-1)

    if not is_batched:
        energy = torch.squeeze(energy, 0)

    return energy


@smee.potentials.potential_energy_fn("Electrostatics", _COULOMB_POTENTIAL)
def compute_coulomb_energy(
    conformer: torch.Tensor,
    parameters: torch.Tensor,
    exclusions: torch.Tensor,
    exclusion_scales: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kcal / mol] of the electrostatic interactions
    using the standard Coulomb potential.

    Notes:
        * No cutoff will be applied.

    Args:
        conformer: The conformer [Å] to evaluate the potential at.
        parameters: A tensor containing the charge [e] of each particle, with
            ``shape=(n_particles, 1)``.
        exclusions: A tensor containing pairs of atom indices whose interaction should
            be scaled by ``exclusion_scales`` with ``shape=(n_exclusions, 2)``.
        exclusion_scales: A tensor containing the scale factor for each exclusion pair
            with ``shape=(n_exclusions, 1)``.

    Returns:
        The evaluated potential energy [kcal / mol].
    """

    is_batched = conformer.ndim == 3

    if not is_batched:
        conformer = torch.unsqueeze(conformer, 0)

    pair_idxs, distances_sqr, pair_scales = compute_pairwise(
        conformer, exclusions, exclusion_scales
    )
    inverse_distances = torch.rsqrt(distances_sqr)

    energy = (
        _COULOMB_PRE_FACTOR
        * pair_scales
        * parameters[pair_idxs[:, 0], 0]
        * parameters[pair_idxs[:, 1], 0]
        * inverse_distances
    ).sum(-1)

    if not is_batched:
        energy = torch.squeeze(energy, 0)

    return energy
