"""Compute the potential energy of parameterized systems / topologies."""
import importlib
import inspect

import torch

import smee
import smee.utils

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
        ``shape=(n_particles, n_parameter_cols)``.
    """
    parameters = torch.vstack(
        [
            torch.broadcast_to(
                (
                    topology.parameters[potential.type].assignment_matrix
                    @ potential.parameters
                )[None, :, :],
                (n_copies, topology.n_atoms, potential.parameters.shape[-1]),
            ).reshape(-1, potential.parameters.shape[-1])
            for topology, n_copies in zip(system.topologies, system.n_copies)
        ]
    )

    return parameters


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

    return system, conformer, box_vectors


def compute_energy_potential(
    system: smee.TensorSystem | smee.TensorTopology,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
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

    if "pairwise" in energy_fn_spec.parameters:
        # TODO: pairwise...
        energy_fn_kwargs["pairwise"] = None

    return energy_fn(system, potential, conformer, box_vectors**energy_fn_kwargs)


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

    system, conformer, box_vectors = _prepare_inputs(system, conformer, box_vectors)

    energy = torch.zeros(conformer.shape[0])

    for potential in force_field.potentials:
        energy += compute_energy_potential(system, potential, conformer, box_vectors)

    return energy
