"""Non-bonded potential energy functions."""
import collections
import math
import typing

import openff.units
import torch

import smee.potentials
import smee.utils

_UNIT = openff.units.unit

_COULOMB_PRE_FACTOR_UNITS = (
    _UNIT.kilocalorie / _UNIT.mole * _UNIT.angstrom / _UNIT.e**2
)
_COULOMB_PRE_FACTOR = (_UNIT.avogadro_constant / (4.0 * _UNIT.pi * _UNIT.eps0)).m_as(
    _COULOMB_PRE_FACTOR_UNITS
)
_COULOMB_POTENTIAL = "coul"

_PME_MIN_NODES = torch.tensor(6)  # taken to match OpenMM 8.0.0
_PME_ORDER = 5  # see OpenMM issue #2567

_LJ_POTENTIAL = "4*epsilon*((sigma/r)**12-(sigma/r)**6)"


class PairwiseDistances(typing.NamedTuple):
    """A container for the pairwise distances between all particles, possibly within
    a given cutoff."""

    idxs: torch.Tensor
    """The particle indices of each pair with ``shape=(n_pairs, 2)``."""
    deltas: torch.Tensor
    """The vector between each pair with ``shape=(n_pairs, 3)``."""
    distances: torch.Tensor
    """The distance between each pair with ``shape=(n_pairs,)``."""

    cutoff: torch.Tensor | None = None
    """The cutoff used when computing the distances."""


def _broadcast_exclusions(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcasts the exclusions (indices and scale factors) of each topology to the
    full system.

    Args:
        system: The system.
        potential: The potential containing the scale factors to broadcast.

    Returns:
        The exception indices with shape ``(n_exceptions, 2)`` and the scale factors
        with shape ``(n_exceptions,)``.
    """

    idx_offset = 0

    per_topology_exclusion_idxs = []
    per_topology_exclusion_scales = []

    for topology, n_copies in zip(system.topologies, system.n_copies):
        exclusion_idxs = topology.parameters[potential.type].exclusions

        exclusion_offset = (
            idx_offset
            + smee.utils.arange_like(n_copies, exclusion_idxs) * topology.n_particles
        )
        idx_offset += n_copies * topology.n_particles

        if len(exclusion_idxs) == 0:
            continue

        exclusion_idxs = exclusion_offset[:, None, None] + exclusion_idxs[None, :, :]

        exclusion_scales = potential.attributes[
            topology.parameters[potential.type].exclusion_scale_idxs
        ]
        exclusion_scales = torch.broadcast_to(
            exclusion_scales, (n_copies, *exclusion_scales.shape)
        )

        per_topology_exclusion_idxs.append(exclusion_idxs.reshape(-1, 2))
        per_topology_exclusion_scales.append(exclusion_scales.reshape(-1))

    system_idxs = (
        torch.zeros((0, 2), dtype=torch.int32)
        if len(per_topology_exclusion_idxs) == 0
        else torch.vstack(per_topology_exclusion_idxs)
    )
    system_scales = (
        torch.zeros((0,), dtype=torch.float32)
        if len(per_topology_exclusion_scales) == 0
        else torch.cat(per_topology_exclusion_scales)
    )

    return system_idxs, system_scales


def compute_pairwise_scales(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> torch.Tensor:
    """Returns the scale factor for each pair of particles in the system by
    broadcasting and stacking the exclusions of each topology.

    Args:
        system: The system.
        potential: The potential containing the scale factors to broadcast.

    Returns:
        The scales for each pair of particles as a flattened upper triangular matrix
        with ``shape=(n_particles * (n_particles - 1) / 2,)``.
    """

    n_particles = system.n_particles
    n_pairs = (n_particles * (n_particles - 1)) // 2

    exclusion_idxs, exclusion_scales = _broadcast_exclusions(system, potential)

    pair_scales = smee.utils.ones_like(n_pairs, other=potential.parameters)

    if len(exclusion_idxs) > 0:
        exclusion_idxs, _ = exclusion_idxs.sort(dim=1)  # ensure upper triangle

        pair_idxs = smee.utils.to_upper_tri_idx(
            exclusion_idxs[:, 0], exclusion_idxs[:, 1], n_particles
        )
        pair_scales[pair_idxs] = exclusion_scales

    return pair_scales


def _compute_pairwise_periodic(
    conformer: torch.Tensor, box_vectors: torch.Tensor, cutoff: torch.Tensor
) -> PairwiseDistances:
    import NNPOps.neighbors

    assert box_vectors is not None, "box vectors must be specified for PBC."
    assert len(conformer.shape) == 2, "the conformer must not have a batch dimension."

    (
        pair_idxs,
        deltas,
        distances,
        _,
    ) = NNPOps.neighbors.getNeighborPairs(conformer, cutoff.item(), -1, box_vectors)

    are_interacting = ~torch.isnan(distances)

    pair_idxs, _ = pair_idxs[:, are_interacting].sort(dim=0)
    distances = distances[are_interacting]
    deltas = deltas[are_interacting, :]

    return PairwiseDistances(pair_idxs.T.contiguous(), deltas, distances, cutoff)


def _compute_pairwise_non_periodic(conformer: torch.Tensor) -> PairwiseDistances:
    n_particles = conformer.shape[-2]

    pair_idxs = torch.triu_indices(n_particles, n_particles, 1, dtype=torch.int32).T

    if conformer.ndim == 2:
        deltas = conformer[pair_idxs[:, 1], :] - conformer[pair_idxs[:, 0], :]
    else:
        deltas = conformer[:, pair_idxs[:, 1], :] - conformer[:, pair_idxs[:, 0], :]
    distances = deltas.norm(dim=-1)

    return PairwiseDistances(pair_idxs.contiguous(), deltas, distances)


def compute_pairwise(
    system: smee.TensorSystem,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None,
    cutoff: torch.Tensor,
) -> PairwiseDistances:
    """Computes all pairwise distances between particles in the system.

    Notes:
        If the system is not periodic, no cutoff and no PBC will be applied.

    Args:
        system: The system to compute the distances for.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.
        box_vectors: The box vectors [Å] of the system with ``shape=(n_confs3, 3)``
            or ``shape=(3, 3)`` if the system is periodic, or ``None`` otherwise.
        cutoff: The cutoff [Å] to apply for periodic systems.

    Returns:
        The pairwise distances between each pair of particles within the cutoff.
    """
    if system.is_periodic:
        return _compute_pairwise_periodic(conformer, box_vectors, cutoff)
    else:
        return _compute_pairwise_non_periodic(conformer)


def lorentz_berthelot(
    epsilon_a: torch.Tensor,
    epsilon_b: torch.Tensor,
    sigma_a: torch.Tensor,
    sigma_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the Lorentz-Berthelot combination rules for the given parameters.

    Args:
        epsilon_a: The epsilon [kcal / mol] values of the first particle in each pair
            with ``shape=(n_pairs, 1)``.
        epsilon_b: The epsilon [kcal / mol] values of the second particle in each pair
            with ``shape=(n_pairs, 1)``.
        sigma_a: The sigma [kcal / mol] values of the first particle in each pair
            with ``shape=(n_pairs, 1)``.
        sigma_b: The sigma [kcal / mol] values of the second particle in each pair
            with ``shape=(n_pairs, 1)``.

    Returns:
        The epsilon [kcal / mol] and sigma [Å] values of each pair, each with
        ``shape=(n_pairs, 1)``.
    """
    return (epsilon_a * epsilon_b).sqrt(), 0.5 * (sigma_a + sigma_b)


def _compute_dispersion_integral(
    r: torch.Tensor, rs: torch.Tensor, rc: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Evaluate the integral needed to compute the LJ long range dispersion correction
    due to the switching function.

    Notes:
        The math was very gratefully copied from OpenMM: https://github.com/openmm/openmm/blob/0363c38dc7ba5abc40d5d4c72efbca0718ff09ab/openmmapi/src/NonbondedForceImpl.cpp#L234C32-L234C32
        See LICENSE_3RD_PARTY for the OpenMM license and copyright notice.

    Args:
        r: The distance to evaluate the integral at.
        rs: The switching distance.
        rc: The cutoff distance.
        sigma: The sigma value of the pair.

    Returns:
        The evaluated integral.
    """
    A = 1 / (rc - rs)
    A2 = A * A
    A3 = A2 * A
    sig2 = sigma * sigma
    sig6 = sig2 * sig2 * sig2
    rs2 = rs * rs
    rs3 = rs * rs2
    r2 = r * r
    r3 = r * r2
    r4 = r * r3
    r5 = r * r4
    r6 = r * r5
    r9 = r3 * r6
    # fmt: off
    return (
        sig6 * A3 * ((
            sig6 * (
                + rs3 * 28 * (6 * rs2 * A2 + 15 * rs * A + 10)
                - r * rs2 * 945 * (rs2 * A2 + 2 * rs * A + 1)
                + r2 * rs * 1080 * (2 * rs2 * A2 + 3 * rs * A + 1)
                - r3 * 420 * (6 * rs2 * A2 + 6 * rs * A + 1)
                + r4 * 756 * (2 * rs * A2 + A)
                - r5 * 378 * A2)
            - r6 * (
                +rs3 * 84 * (6 * rs2 * A2 + 15 * rs * A + 10)
                - r * rs2 * 3780 * (rs2 * A2 + 2 * rs * A + 1)
                + r2 * rs * 7560 * (2 * rs2 * A2 + 3 * rs * A + 1)))
            / (252 * r9)
            - torch.log(r) * 10 * (6 * rs2 * A2 + 6 * rs * A + 1)
            + r * 15 * (2 * rs * A2 + A)
            - r2 * 3 * A2)
    )
    # fmt: on


def _compute_dispersion_term(
    count: float,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    cutoff: torch.Tensor | None,
    switch_width: torch.Tensor | None,
) -> torch.Tensor:
    """Computes the terms of the LJ dispersion correction for a particular type of
    interactions (i.e., ii and ij).

    Args:
        count: The number of interactions of this type with ``shape=(n_parameters,)``.
        epsilon: The epsilon values of each interaction with ``shape=(n_parameters,)``.
        sigma: The sigma values of each interaction with ``shape=(n_parameters,)``.
        cutoff: The cutoff distance.
        switch_width: The distance at which the switching function begins to apply.

    """
    sigma6 = sigma**6

    terms = [sigma6 * sigma6, sigma6]

    if switch_width is not None:
        assert cutoff is not None

        terms.append(
            _compute_dispersion_integral(cutoff, switch_width, cutoff, sigma)
            - _compute_dispersion_integral(switch_width, switch_width, cutoff, sigma)
        )

    return (count * epsilon * torch.stack(terms)).sum(dim=-1)


def _compute_dispersion_correction(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    cutoff: torch.Tensor | None,
    switch_width: torch.Tensor | None,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Computes the long range dispersion correction due to the switching function.

    Args:
        system: The system to compute the correction for.
        potential: The LJ potential.
        cutoff: The cutoff distance.
        switch_width: The distance at which the switching function begins to apply.
        volume: The volume of the system.

    Returns:

    """
    n_by_type = collections.defaultdict(int)

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_counts = topology.parameters["vdW"].assignment_matrix.abs().sum(dim=0)

        for key, count in zip(potential.parameter_keys, parameter_counts):
            n_by_type[key] += count.item() * n_copies

    counts = smee.utils.tensor_like(
        [n_by_type[key] for key in potential.parameter_keys], potential.parameters
    )

    # particles of the same type interacting
    n_ii_interactions = (counts * (counts + 1.0)) / 2.0

    eps_ii, sig_ii = potential.parameters[:, 0], potential.parameters[:, 1]

    terms = _compute_dispersion_term(
        n_ii_interactions, eps_ii, sig_ii, cutoff, switch_width
    )

    # particles of different types interacting
    idx_i, idx_j = torch.triu_indices(len(counts), len(counts), 1)
    n_ij_interactions = counts[idx_i] * counts[idx_j]

    eps_ij, sig_ij = lorentz_berthelot(
        eps_ii[idx_i], eps_ii[idx_j], sig_ii[idx_i], sig_ii[idx_j]
    )
    terms += _compute_dispersion_term(
        n_ij_interactions, eps_ij, sig_ij, cutoff, switch_width
    )

    n_particles = system.n_particles
    n_interactions = (n_particles * (n_particles + 1)) / 2

    terms /= n_interactions

    return (
        8.0
        * n_particles**2
        * torch.pi
        * (terms[0] / (9 * cutoff**9) - terms[1] / (3 * cutoff**3) + terms[2])
        / volume
    )


def _compute_switch_fn(
    potential: smee.TensorPotential,
    pairwise: PairwiseDistances,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if "switch_width" not in potential.attribute_cols:
        return torch.ones(1), None

    switch_width_idx = potential.attribute_cols.index("switch_width")
    switch_width = pairwise.cutoff - potential.attributes[switch_width_idx]

    x_switch = (pairwise.distances - switch_width) / (pairwise.cutoff - switch_width)

    switch_fn = 1.0 - 6.0 * x_switch**5 + 15.0 * x_switch**4 - 10.0 * x_switch**3

    switch_fn = torch.where(
        pairwise.distances < switch_width, torch.tensor(1.0), switch_fn
    )
    switch_fn = torch.where(
        pairwise.distances > pairwise.cutoff, torch.tensor(0.0), switch_fn
    )

    return switch_fn, switch_width


@smee.potentials.potential_energy_fn("vdW", _LJ_POTENTIAL)
def compute_lj_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
    pairwise: PairwiseDistances | None = None,
) -> torch.Tensor:
    """Computes the potential energy [kcal / mol] of the vdW interactions using the
    standard Lennard-Jones potential.

    Notes:
        * No cutoff / switching function will be applied if the system is not
          periodic.
        * A switching function will only be applied if the potential has a
          ``switch_width`` attribute.

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.
        box_vectors: The box vectors [Å] of the system with ``shape=(n_confs, 3, 3)``
            or ``shape=(3, 3)`` if the system is periodic, or ``None`` otherwise.
        pairwise: The pre-computed pairwise distances between each pair of particles
            in the system. If none, these will be computed within the function.

    Returns:
        The computed potential energy [kcal / mol] with ``shape=(n_confs,)`` if the
        input conformer has a batch dimension, or ``shape=()`` otherwise.
    """

    box_vectors = None if not system.is_periodic else box_vectors

    cutoff = potential.attributes[potential.attribute_cols.index("cutoff")]

    pairwise = (
        pairwise
        if pairwise is not None
        else compute_pairwise(system, conformer, box_vectors, cutoff)
    )

    if system.is_periodic and not torch.isclose(pairwise.cutoff, cutoff):
        raise ValueError("the pairwise cutoff does not match the potential.")

    parameters = smee.potentials.broadcast_parameters(system, potential)
    pair_scales = compute_pairwise_scales(system, potential)

    pairs_1d = smee.utils.to_upper_tri_idx(
        pairwise.idxs[:, 0], pairwise.idxs[:, 1], len(parameters)
    )
    pair_scales = pair_scales[pairs_1d]

    epsilon_column = potential.parameter_cols.index("epsilon")
    sigma_column = potential.parameter_cols.index("sigma")

    epsilon, sigma = lorentz_berthelot(
        parameters[pairwise.idxs[:, 0], epsilon_column],
        parameters[pairwise.idxs[:, 1], epsilon_column],
        parameters[pairwise.idxs[:, 0], sigma_column],
        parameters[pairwise.idxs[:, 1], sigma_column],
    )

    x = (sigma / pairwise.distances) ** 6
    energies = pair_scales * 4.0 * epsilon * (x * (x - 1.0))

    if not system.is_periodic:
        return energies.sum(-1)

    switch_fn, switch_width = _compute_switch_fn(potential, pairwise)
    energies *= switch_fn

    energy = energies.sum(-1)
    energy += _compute_dispersion_correction(
        system,
        potential.to(precision="double"),
        switch_width.double(),
        pairwise.cutoff.double(),
        torch.det(box_vectors),
    )

    return energy


def _compute_pme_exclusions(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> torch.Tensor:
    """Builds the exclusion tensor required by NNPOps pme functions

    Args:
        system: The system to compute the exclusions for.
        potential: The electrostatics potential.

    Returns:
        The exclusion tensor with ``shape=(n_particles, max_exclusions)`` where
        ``max_exclusions`` is the maximum number of exclusions of any atom. A value
        of -1 is used for padding.
    """
    exclusion_templates = [
        [[] for _ in range(topology.n_particles)] for topology in system.topologies
    ]
    max_exclusions = 0

    for exclusions, topology, n_copies in zip(
        exclusion_templates, system.topologies, system.n_copies
    ):
        for i, j in topology.parameters[potential.type].exclusions:
            exclusions[i].append(int(j))
            exclusions[j].append(int(i))

            max_exclusions = max(len(exclusions[i]), max_exclusions)
            max_exclusions = max(len(exclusions[j]), max_exclusions)

    idx_offset = 0

    exclusions_per_type = []

    for exclusions, topology, n_copies in zip(
        exclusion_templates, system.topologies, system.n_copies
    ):
        for atom_exclusions in exclusions:
            n_padding = max_exclusions - len(atom_exclusions)
            atom_exclusions.extend([-1] * n_padding)

        exclusions = torch.tensor(exclusions, dtype=torch.int32)

        exclusion_offset = idx_offset + torch.arange(n_copies) * topology.n_particles
        idx_offset += n_copies * topology.n_particles

        if exclusions.shape[-1] == 0:
            continue

        exclusions = torch.broadcast_to(
            exclusions, (n_copies, len(exclusions), max_exclusions)
        )
        exclusions = torch.where(
            exclusions >= 0, exclusions + exclusion_offset[:, None, None], exclusions
        )

        exclusions_per_type.append(exclusions.reshape(-1, max_exclusions))

    return (
        torch.zeros((0, 0), dtype=torch.int32)
        if len(exclusions_per_type) == 0
        else torch.vstack(exclusions_per_type)
    )


def _compute_pme_grid(
    box_vectors: torch.Tensor, cutoff: torch.Tensor, error_tolerance: torch.Tensor
) -> tuple[int, int, int, float]:
    alpha = torch.sqrt(-torch.log(2.0 * error_tolerance)) / cutoff

    factor = 2.0 * alpha / (3 * error_tolerance ** (1.0 / 5.0))

    grid_x = torch.maximum(torch.ceil(factor * box_vectors[0, 0]), _PME_MIN_NODES)
    grid_y = torch.maximum(torch.ceil(factor * box_vectors[1, 1]), _PME_MIN_NODES)
    grid_z = torch.maximum(torch.ceil(factor * box_vectors[2, 2]), _PME_MIN_NODES)

    return int(grid_x), int(grid_y), int(grid_z), float(alpha)


def _compute_coulomb_energy_non_periodic(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    pairwise: PairwiseDistances,
):
    parameters = smee.potentials.broadcast_parameters(system, potential)
    pair_scales = compute_pairwise_scales(system, potential)

    energy = (
        _COULOMB_PRE_FACTOR
        * pair_scales
        * parameters[pairwise.idxs[:, 0], 0]
        * parameters[pairwise.idxs[:, 1], 0]
        / pairwise.distances
    ).sum(-1)

    return energy


def _compute_coulomb_energy_periodic(
    system: smee.TensorSystem,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor,
    potential: smee.TensorPotential,
    pairwise: PairwiseDistances,
) -> torch.Tensor:
    import NNPOps.pme

    assert system.is_periodic, "the system must be periodic."

    charges = smee.potentials.broadcast_parameters(system, potential).squeeze(-1)

    cutoff = potential.attributes[potential.attribute_cols.index("cutoff")]
    error_tol = torch.tensor(0.0001)

    exceptions = _compute_pme_exclusions(system, potential).to(charges.device)

    grid_x, grid_y, grid_z, alpha = _compute_pme_grid(box_vectors, cutoff, error_tol)

    pme = NNPOps.pme.PME(
        grid_x, grid_y, grid_z, _PME_ORDER, alpha, _COULOMB_PRE_FACTOR, exceptions
    )

    energy_direct = torch.ops.pme.pme_direct(
        conformer.float(),
        charges.float(),
        pairwise.idxs.T,
        pairwise.deltas,
        pairwise.distances,
        pme.exclusions,
        pme.alpha,
        pme.coulomb,
    )
    energy_self = (
        -torch.sum(charges**2) * pme.coulomb * pme.alpha / math.sqrt(torch.pi)
    )
    energy_recip = energy_self + torch.ops.pme.pme_reciprocal(
        conformer.float(),
        charges.float(),
        box_vectors.float(),
        pme.gridx,
        pme.gridy,
        pme.gridz,
        pme.order,
        pme.alpha,
        pme.coulomb,
        pme.moduli[0].to(charges.device),
        pme.moduli[1].to(charges.device),
        pme.moduli[2].to(charges.device),
    )

    exclusion_idxs, exclusion_scales = _broadcast_exclusions(system, potential)

    exclusion_distances = (
        conformer[exclusion_idxs[:, 0], :] - conformer[exclusion_idxs[:, 1], :]
    ).norm(dim=-1)

    energy_exclusion = (
        _COULOMB_PRE_FACTOR
        * exclusion_scales
        * charges[exclusion_idxs[:, 0]]
        * charges[exclusion_idxs[:, 1]]
        / exclusion_distances
    ).sum(-1)

    return energy_direct + energy_recip + energy_exclusion


@smee.potentials.potential_energy_fn("Electrostatics", _COULOMB_POTENTIAL)
def compute_coulomb_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
    pairwise: PairwiseDistances | None = None,
) -> torch.Tensor:
    """Computes the potential energy [kcal / mol] of the electrostatic interactions
    using the Coulomb potential.

    Notes:
        * No cutoff will be applied if the system is not periodic.
        * PME will be used to compute the energy if the system is periodic.

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.
        box_vectors: The box vectors [Å] of the system with ``shape=(n_confs, 3, 3)``
            or ``shape=(3, 3)`` if the system is periodic, or ``None`` otherwise.
        pairwise: The pre-computed pairwise distances between each pair of particles
            in the system. If none, these will be computed within the function.

    Returns:
        The computed potential energy [kcal / mol] with ``shape=(n_confs,)`` if the
        input conformer has a batch dimension, or ``shape=()`` otherwise.
    """

    box_vectors = None if not system.is_periodic else box_vectors

    cutoff = potential.attributes[potential.attribute_cols.index("cutoff")]

    pairwise = (
        pairwise
        if pairwise is not None
        else compute_pairwise(system, conformer, box_vectors, cutoff)
    )

    if system.is_periodic and not torch.isclose(pairwise.cutoff, cutoff):
        raise ValueError("the distance cutoff does not match the potential.")

    if system.is_periodic:
        return _compute_coulomb_energy_periodic(
            system, conformer, box_vectors, potential, pairwise
        )
    else:
        return _compute_coulomb_energy_non_periodic(system, potential, pairwise)
