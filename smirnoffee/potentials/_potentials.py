"""Evaluate the potential energy of parameterized topolgies."""
import importlib
import inspect

import torch

import smirnoffee.ff

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


def evaluate_energy_potential(
    parameters: smirnoffee.ff.ParameterMap,
    conformer: torch.Tensor,
    potential: smirnoffee.ff.TensorPotential,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] due to a SMIRNOFF potential
    handler for a given conformer(s).

    Args:
        parameters: A map of the parameters of the potential that were applied to a
            particular topology / molecule.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        potential: The potential to evaluate.

    Returns:
        The potential energy of the conformer(s) [kJ / mol].
    """

    if len(conformer.shape) == 2:
        conformer = torch.unsqueeze(conformer, 0)

    parameter_values = potential.parameters[parameters.parameter_idxs]
    global_parameter_values = (
        None
        if potential.global_parameters is None
        else potential.global_parameters[parameters.global_parameter_idxs]
    )

    importlib.import_module("smirnoffee.potentials.nonbonded")
    importlib.import_module("smirnoffee.potentials.valence")

    energy_fn = _POTENTIAL_ENERGY_FUNCTIONS[(potential.type, potential.fn)]
    energy_fn_spec = inspect.signature(energy_fn)

    energy_fn_kwargs = {}

    if "global_parameters" in energy_fn_spec.parameters:
        energy_fn_kwargs["global_parameters"] = global_parameter_values

    energy = energy_fn(
        conformer, parameters.atom_idxs, parameter_values, **energy_fn_kwargs
    )
    return energy


def evaluate_energy(
    parameters: smirnoffee.ff.AppliedParameters,
    conformer: torch.Tensor,
    force_field: smirnoffee.ff.TensorForceField,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a topology / molecule in a given
    conformation(s).

    Args:
        parameters: The parameters that were applied to the molecule. This should be
            a dictionary with keys corresponding to a SMIRNOFF handler, and values
            of maps from interactions to corresponding parameters.
        conformer: The conformer(s) to evaluate the potential at with
            ``shape=(n_atoms, 3)`` or ``shape=(n_confs, n_atoms, 3)``.
        force_field: The values of the force field parameters.

    Returns:
        The potential energy of the conformer(s) [kJ / mol].
    """

    if conformer.ndim == 2:
        conformer = torch.unsqueeze(conformer, 0)

    energy = torch.zeros(conformer.shape[0])

    for potential in force_field.potentials:
        parameter_map = parameters[potential.type]
        energy += evaluate_energy_potential(parameter_map, conformer, potential)

    return energy
