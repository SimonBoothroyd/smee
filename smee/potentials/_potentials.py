"""Compute the potential energy of parameterized systems / topologies."""

import importlib
import inspect
import typing

import torch

import smee
import smee.utils

if typing.TYPE_CHECKING:
    import smee.potentials.nonbonded


_POTENTIAL_ENERGY_FUNCTIONS = {}


def potential_energy_fn(handler_type: str, energy_expression: str):
    """A decorator used to flag a function as being able to compute the potential for a
    specific handler and its associated energy expression."""

    def _potential_function_inner(func):
        if (handler_type, energy_expression) in _POTENTIAL_ENERGY_FUNCTIONS:
            raise KeyError(
                f"A potential energy function is already defined for "
                f"handler={handler_type} fn={energy_expression}."
            )

        _POTENTIAL_ENERGY_FUNCTIONS[(handler_type, energy_expression)] = func
        return func

    return _potential_function_inner


def broadcast_parameters(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> torch.Tensor:
    """Returns parameters for the full system by broadcasting and stacking the
    parameters of each topology.

    Args:
        system: The system.
        potential: The potential whose parameters should be broadcast.

    Returns:
        The parameters for the full system with
        ``shape=(n_parameters, n_parameter_cols)``.
    """

    parameters = []

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameter_map = topology.parameters[potential.type]

        topology_parameters = parameter_map.assignment_matrix @ potential.parameters

        n_interactions = len(topology_parameters)
        n_cols = len(potential.parameter_cols)

        topology_parameters = torch.broadcast_to(
            topology_parameters[None, :, :],
            (n_copies, n_interactions, n_cols),
        ).reshape(-1, n_cols)

        parameters.append(topology_parameters)

    return (
        torch.zeros((0, len(potential.parameter_cols)))
        if len(parameters) == 0
        else torch.vstack(parameters)
    )


def broadcast_exceptions(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    idxs_a: torch.Tensor,
    idxs_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the indices of the parameters that should be used to model interactions
    between pairs of particles

    Args:
        system: The system.
        potential: The potential whose parameters should be broadcast.
        idxs_a: The indices of the first particle in each interaction with
            ``shape=(n_interactions,)``.
        idxs_b: The indices of the second particle in each interaction with
            ``shape=(n_interactions,)``.

    Returns:
        The indices of the interactions that require an exception, and the parameters
        to use for those interactions.
    """
    assert potential.exceptions is not None

    parameter_idxs = []

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameter_map = topology.parameters[potential.type]

        if isinstance(parameter_map, smee.ValenceParameterMap):
            raise NotImplementedError("valence exceptions are not supported")

        # check that each particle is assigned to exactly one parameter
        assignment_dense = parameter_map.assignment_matrix.to_dense()

        if not (assignment_dense.abs().sum(axis=-1) == 1).all():
            raise NotImplementedError(
                f"exceptions can only be used when each particle is assigned exactly "
                f"one {potential.type} parameter"
            )

        assigned_idxs = assignment_dense.argmax(axis=-1)

        n_particles = len(assigned_idxs)

        assigned_idxs = torch.broadcast_to(
            assigned_idxs[None, :], (n_copies, n_particles)
        ).flatten()

        parameter_idxs.append(assigned_idxs)

    if len(parameter_idxs) == 0:
        return torch.zeros((0,), dtype=torch.int64), torch.zeros(
            (0, 0, len(potential.parameter_cols))
        )

    parameter_idxs = torch.concat(parameter_idxs)
    parameter_idxs_a = parameter_idxs[idxs_a]
    parameter_idxs_b = parameter_idxs[idxs_b]

    if len({(min(i, j), max(i, j)) for i, j in potential.exceptions}) != len(
        potential.exceptions
    ):
        raise NotImplementedError("cannot define different exceptions for i-j and j-i")

    exception_lookup = torch.full(
        (len(potential.parameters), len(potential.parameters)), -1
    )

    for (i, j), v in potential.exceptions.items():
        exception_lookup[(min(i, j), max(i, j))] = v
        exception_lookup[(max(i, j), min(i, j))] = v

    exceptions_parameter_idxs = exception_lookup[parameter_idxs_a, parameter_idxs_b]
    exception_mask = exceptions_parameter_idxs >= 0

    exceptions = potential.parameters[exceptions_parameter_idxs[exception_mask]]
    exception_idxs = exception_mask.nonzero().flatten()

    return exception_idxs, exceptions


def broadcast_idxs(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> torch.Tensor:
    """Broadcasts the particle indices of each topology for a given potential
    to the full system.

    Args:
        system: The system.
        potential: The potential.

    Returns:
        The indices with shape ``(n_interactions, n_interacting_particles)`` where
        ``n_interacting_particles`` is 2 for bonds, 3 for angles, etc.
    """

    idx_offset = 0

    per_topology_idxs = []

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameter_map = topology.parameters[potential.type]
        n_interacting_particles = parameter_map.particle_idxs.shape[-1]

        idxs = parameter_map.particle_idxs

        offset = (
            idx_offset + smee.utils.arange_like(n_copies, idxs) * topology.n_particles
        )

        if len(idxs) > 0:
            idxs = offset[:, None, None] + idxs[None, :, :]
            per_topology_idxs.append(idxs.reshape(-1, n_interacting_particles))

        idx_offset += n_copies * topology.n_particles

    return (
        torch.zeros((0, 0))
        if len(per_topology_idxs) == 0
        else torch.vstack(per_topology_idxs)
    )


def _prepare_inputs(
    system: smee.TensorSystem | smee.TensorTopology,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
) -> tuple[smee.TensorSystem, torch.Tensor, torch.Tensor | None]:
    """Prepare inputs for a potential energy function.

    Args:
        system: The system or topology to compute the potential energy of.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_particles, 3)`` or ``shape=(n_particles, n_particles, 3)``.
        box_vectors: The box vectors of the system with ``shape=(3, 3)`` or
            shape=(n_confs, 3, 3)`` if the system is periodic, or ``None`` if the system
            is non-periodic.

    Returns:
        The system representation, the conformer with ``dtype=float32``, and the box
        vectors (if present) with ``dtype=float32``.
    """

    conformer = conformer.float()
    box_vectors = box_vectors.float() if box_vectors is not None else None

    if isinstance(system, smee.TensorTopology):
        system = smee.TensorSystem([system], [1], False)

    if conformer.ndim == 3 and system.is_periodic:
        raise NotImplementedError("batched periodic systems are not supported")

    if system.is_periodic and box_vectors is None:
        raise ValueError("box vectors must be provided for periodic systems")

    return system, conformer, box_vectors


def _precompute_pairwise(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
) -> typing.Optional["smee.potentials.nonbonded.PairwiseDistances"]:
    """Pre-compute pairwise distances for a system if required by any of the
    potential energy functions.

    Args:
        system: The system to compute the potential energy of.
        force_field: The force field that defines the potential energy function.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_particles, 3)`` or ``shape=(n_confs, n_particles, 3)``.
        box_vectors: The box vectors of the system with ``shape=(3, 3)`` or
            shape=(n_confs, 3, 3)`` if the system is periodic, or ``None`` if the system
            is non-periodic.

    Returns:
        The pre-computed pairwise distances if required by any of the potential energy
        functions, or ``None`` otherwise.
    """
    requires_pairwise = False
    cutoffs = []

    for potential in force_field.potentials:
        energy_fn = _POTENTIAL_ENERGY_FUNCTIONS[(potential.type, potential.fn)]
        energy_fn_spec = inspect.signature(energy_fn)

        if "pairwise" not in energy_fn_spec.parameters:
            continue

        requires_pairwise = True

        if smee.CUTOFF_ATTRIBUTE in potential.attribute_cols:
            cutoff = potential.attributes[
                potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)
            ]
            cutoffs.append(cutoff)

        break

    if not requires_pairwise:
        return

    if len(cutoffs) > 1 and not torch.allclose(torch.cat(cutoffs), cutoffs[0]):
        return

    cutoff = None if len(cutoffs) == 0 else cutoffs[0]

    return smee.potentials.nonbonded.compute_pairwise(
        system, conformer, box_vectors, cutoff
    )


def compute_energy_potential(
    system: smee.TensorSystem | smee.TensorTopology,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
    pairwise: typing.Optional["smee.potentials.nonbonded.PairwiseDistances"] = None,
) -> torch.Tensor:
    """Computes the potential energy [kcal / mol] due to a SMIRNOFF potential
    handler for a given conformer(s).

    Args:
        system: The system or topology to compute the potential energy of.
        potential: The potential describing the energy function to compute.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_particles, 3)`` or ``shape=(n_confs, n_particles, 3)``.
        box_vectors: The box vectors of the system with ``shape=(3, 3)`` or
            shape=(n_confs, 3, 3)`` if the system is periodic, or ``None`` if the system
            is non-periodic.
        pairwise: Pre-computed pairwise distances between particles in the system.

    Returns:
        The potential energy of the conformer(s) [kcal / mol].
    """

    # register the built-in potential energy functions
    importlib.import_module("smee.potentials.nonbonded")
    importlib.import_module("smee.potentials.valence")

    system, conformer, box_vectors = _prepare_inputs(system, conformer, box_vectors)

    energy_fn = _POTENTIAL_ENERGY_FUNCTIONS[(potential.type, potential.fn)]
    energy_fn_spec = inspect.signature(energy_fn)

    energy_fn_kwargs = {}

    if "box_vectors" in energy_fn_spec.parameters:
        energy_fn_kwargs["box_vectors"] = box_vectors
    if "pairwise" in energy_fn_spec.parameters:
        energy_fn_kwargs["pairwise"] = pairwise

    return energy_fn(system, potential, conformer, **energy_fn_kwargs)


def compute_energy(
    system: smee.TensorSystem | smee.TensorTopology,
    force_field: smee.TensorForceField,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the potential energy [kcal / mol] of a system / topology in a given
    conformation(s).

    Args:
        system: The system or topology to compute the potential energy of.
        force_field: The force field that defines the potential energy function.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_particles, 3)`` or ``shape=(n_confs, n_particles, 3)``.
        box_vectors: The box vectors of the system with ``shape=(3, 3)`` if the
            system is periodic, or ``None`` if the system is non-periodic.

    Returns:
        The potential energy of the conformer(s) [kcal / mol].
    """

    # register the built-in potential energy functions
    importlib.import_module("smee.potentials.nonbonded")
    importlib.import_module("smee.potentials.valence")

    system, conformer, box_vectors = _prepare_inputs(system, conformer, box_vectors)
    pairwise = _precompute_pairwise(system, force_field, conformer, box_vectors)

    energy = smee.utils.zeros_like(
        conformer.shape[0] if conformer.ndim == 3 else 1, conformer
    )

    for potential in force_field.potentials:
        energy += compute_energy_potential(
            system, potential, conformer, box_vectors, pairwise
        )

    return energy
