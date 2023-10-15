"""Evaluate the potential energy of parameterized topologies."""
import importlib
import inspect

import torch

import smee
import smee.utils

_POTENTIAL_ENERGY_FUNCTIONS = {}


def potential_energy_fn(handler_type: str, energy_expression: str):
    """A decorator used to flag a function as being able to compute the potential for a
    specific handler and its associated energy expression.."""

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
            )
            for topology, n_copies in zip(system.topologies, system.n_copies)
        ]
    ).reshape(-1, 2)

    return parameters


def broadcast_exclusions(
    system: smee.TensorSystem, potential: smee.TensorPotential
) -> torch.Tensor:
    """Returns the scale factor for each interaction in the full system by broadcasting
    and stacking the exclusions of each topology.

    Args:
        system: The system.
        potential: The potential containing the scale factors to broadcast.

    Returns:
        The parameters for the full system with
        ``shape=(n_particles * (n_particles - 1) / 2,)``.
    """

    n_particles = system.n_particles
    n_pairs = (n_particles * (n_particles - 1)) // 2

    pair_scales = smee.utils.ones_like(n_pairs, other=potential.parameters)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        exclusion_offset = idx_offset + torch.arange(n_copies) * topology.n_particles

        exclusion_idxs = topology.parameters[potential.type].exclusions
        exclusion_idxs = exclusion_offset[:, None, None] + exclusion_idxs[None, :, :]

        exclusion_scales = potential.attributes[
            topology.parameters[potential.type].exclusion_scale_idxs
        ]
        exclusion_scales = torch.broadcast_to(
            exclusion_scales, (n_copies, *exclusion_scales.shape)
        )

        exclusion_idxs = exclusion_idxs.reshape(-1, 2)
        exclusion_scales = exclusion_scales.reshape(-1)

        if len(exclusion_idxs) > 0:
            exclusion_idxs, _ = exclusion_idxs.sort(dim=1)  # ensure upper triangle

            pair_idxs = smee.utils.to_upper_tri_idx(
                exclusion_idxs[:, 0], exclusion_idxs[:, 1], n_particles
            )
            pair_scales[pair_idxs] = exclusion_scales

        idx_offset += n_copies * topology.n_particles

    return pair_scales


def compute_energy_potential(
    parameters: smee.ParameterMap,
    conformer: torch.Tensor,
    potential: smee.TensorPotential,
) -> torch.Tensor:
    """Evaluates the potential energy [kcal / mol] due to a SMIRNOFF potential
    handler for a given conformer(s).

    Args:
        parameters: A map of the parameters of the potential that were applied to a
            particular topology / molecule.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        potential: The potential to evaluate.

    Returns:
        The potential energy of the conformer(s) [kcal / mol].
    """

    conformer = conformer.float()

    if len(conformer.shape) == 2:
        conformer = torch.unsqueeze(conformer, 0)

    parameter_values = parameters.assignment_matrix @ potential.parameters

    importlib.import_module("smee.potentials.nonbonded")
    importlib.import_module("smee.potentials.valence")

    energy_fn = _POTENTIAL_ENERGY_FUNCTIONS[(potential.type, potential.fn)]
    energy_fn_spec = inspect.signature(energy_fn)

    energy_fn_kwargs = {}

    if "attributes" in energy_fn_spec.parameters:
        energy_fn_kwargs["attributes"] = potential.attributes

    if isinstance(parameters, smee.NonbondedParameterMap):
        energy = energy_fn(
            conformer,
            parameter_values,
            parameters.exclusions,
            potential.attributes[parameters.exclusion_scale_idxs],
            **energy_fn_kwargs,
        )
    elif isinstance(parameters, smee.ValenceParameterMap):
        energy = energy_fn(
            conformer, parameters.particle_idxs, parameter_values, **energy_fn_kwargs
        )
    else:
        raise NotImplementedError

    return energy


def compute_energy(
    parameters: dict[str, smee.ParameterMap],
    conformer: torch.Tensor,
    force_field: smee.TensorForceField,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a topology in a given
    conformation(s).

    Args:
        parameters: The parameters that were applied to the topology. This should be
            a dictionary with keys corresponding to a SMIRNOFF handler, and values
            of maps from interactions to corresponding parameters.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_atoms + n_v_sites, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        force_field: The values of the force field parameters.

    Returns:
        The potential energy of the conformer(s) [kcal / mol].
    """

    conformer = conformer.float()

    if conformer.ndim == 2:
        conformer = torch.unsqueeze(conformer, 0)

    energy = torch.zeros(conformer.shape[0])

    for potential in force_field.potentials:
        parameter_map = parameters[potential.type]
        energy += compute_energy_potential(parameter_map, conformer, potential)

    return energy
