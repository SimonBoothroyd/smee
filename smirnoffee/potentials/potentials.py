from typing import Dict, List, Optional, Tuple

import torch
from openff.system.components.smirnoff import SMIRNOFFPotentialHandler
from openff.system.components.system import System
from openff.system.models import PotentialKey
from openff.toolkit.topology import Molecule

from smirnoffee.exceptions import MissingArgumentsError
from smirnoffee.smirnoff import VectorizedHandler, vectorize_handler, vectorize_system

_POTENTIAL_ENERGY_FUNCTIONS = {}


def potential_energy_function(handler_type, energy_expression):
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


def add_parameter_delta(
    parameters: torch.Tensor,
    parameter_ids: List[Tuple[PotentialKey, Tuple[str, ...]]],
    delta: torch.Tensor,
    delta_ids: List[Tuple[PotentialKey, str]],
) -> torch.Tensor:
    """Adds a 1D vector of parameter 'deltas' to an existing 2D tensor of parameter
    values, whereby each parameter in the flat delta is matched to a parameter in the
    2D tensor according to a combination of its 'potential key' and the specific
    attribute it represents.

    Care is taken such that gradients may be backwards propagated through the new
    parameter tensor back to the original ``delta`` tensor.

    Args:
        parameters: A 2D tensor of parameters to add the deltas to.
        parameter_ids: A list of tuples of the form ``(potential_key, (attrs, ...))``.
            The attributes may include, for example, ``['k', 'length', ...]``.
        delta: A 1D tensor of the values to add to the 2D parameters.
        delta_ids: A list of tuples of the form ``(potential_key, attr)`` which
            identify which delta is associated with which parameter. This must be
            the same length as the ``delta`` tensor. Not all values in the list
            need to appear in ``parameter_ids`` and vice versa.

    Returns:
        A new parameter tensor of the form ``parameters + delta[map_indices]`` where
        ``map_indices`` is constructed by this function by matching ``parameter_ids``
        to ``delta_ids``.
    """

    delta = torch.cat([delta, torch.zeros(1)])
    zero_index = len(delta) - 1

    delta_indices = torch.tensor(
        [
            [
                zero_index
                if (parameter_id, attribute_name) not in delta_ids
                else delta_ids.index((parameter_id, attribute_name))
                for attribute_name in attribute_names
            ]
            for parameter_id, attribute_names in parameter_ids
        ]
    )

    delta = delta[delta_indices]

    return parameters + delta


def evaluate_handler_energy(
    handler: SMIRNOFFPotentialHandler,
    molecule: Molecule,
    conformer: torch.Tensor,
    parameter_delta: Optional[torch.Tensor] = None,
    parameter_delta_ids: Optional[List[Tuple[PotentialKey, str]]] = None,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] contribution of a particular SMIRNOFF
    potential handler for a given conformer.

    Args:
        handler: The potential handler that encodes the potential energy function to
            evaluate.
        molecule: The molecule that the handler is associated with.
        conformer: The conformer to evaluate the potential at.
        parameter_delta: An optional tensor of values to perturb the assigned
            valence parameters by before evaluating the potential energy. If this
            option is specified then ``parameter_delta_ids`` must also be.
        parameter_delta_ids: An optional list of ids associated with the
            ``parameter_delta`` tensor which is used to identify which parameter
            delta matches which assigned parameter in the ``valence_handler``. If this
            option is specified then ``parameter_delta`` must also be.

    Returns:
        The potential energy of the conformer [kJ / mol].
    """

    return evaluate_vectorized_handler_energy(
        vectorize_handler(handler, molecule),
        handler.type,
        handler.expression,
        conformer,
        parameter_delta,
        parameter_delta_ids,
    )


def evaluate_vectorized_handler_energy(
    handler: VectorizedHandler,
    handler_type: str,
    handler_expression: str,
    conformer: torch.Tensor,
    parameter_delta: Optional[torch.Tensor] = None,
    parameter_delta_ids: Optional[List[Tuple[PotentialKey, str]]] = None,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] contribution of a vectorized SMIRNOFF
    potential handler for a given conformer.

    Args:
        handler: The vectorized handler.
        handler_type: The type of handler which was vectorised.
        handler_expression: The energy expression associated with the handler.
        conformer: The conformer to evaluate the potential at.
        parameter_delta: An optional tensor of values to perturb the assigned
            valence parameters by before evaluating the potential energy. If this
            option is specified then ``parameter_delta_ids`` must also be.
        parameter_delta_ids: An optional list of ids associated with the
            ``parameter_delta`` tensor which is used to identify which parameter
            delta matches which assigned parameter in the ``valence_handler``. If this
            option is specified then ``parameter_delta`` must also be.

    Returns:
        The potential energy of the conformer [kJ / mol].
    """

    if not (
        parameter_delta is None
        and parameter_delta_ids is None
        or parameter_delta is not None
        and parameter_delta_ids is not None
    ):

        raise MissingArgumentsError(
            "Either both ``parameter_delta`` and ``parameter_delta_ids`` must be "
            "specified or neither must be."
        )

    if parameter_delta is not None:

        assert len(parameter_delta_ids) == parameter_delta.shape[0], (
            f"each parameter delta (n={len(parameter_delta_ids)}) must have an "
            f"associated id (n={parameter_delta.shape[0]})"
        )

    indices, parameters, parameter_ids = handler

    if len(parameter_ids) == 0:
        return torch.zeros(1)

    if parameter_delta is not None:

        parameters = add_parameter_delta(
            parameters, parameter_ids, parameter_delta, parameter_delta_ids
        )

    energy_expression = _POTENTIAL_ENERGY_FUNCTIONS[(handler_type, handler_expression)]
    handler_energy = energy_expression(conformer, indices, parameters)

    return handler_energy


def evaluate_system_energy(
    system: System,
    conformer: torch.Tensor,
    parameter_delta: Optional[torch.Tensor] = None,
    parameter_delta_ids: Optional[List[Tuple[str, PotentialKey, str]]] = None,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a full OpenFF system containing
    a single molecule.

    Args:
        system: The system that encodes the potential energy function to evaluate.
        conformer: The conformer to evaluate the potential at.
        parameter_delta: An optional tensor of values to perturb the assigned
            parameters by before evaluating the potential energy. If this
            option is specified then ``parameter_delta_ids`` must also be.
        parameter_delta_ids: An optional list of ids associated with the
            ``parameter_delta`` tensor which is used to identify which parameter
            delta matches which assigned parameter. If this option is specified then
            ``parameter_delta`` must also be.

    Returns:
        The potential energy of the conformer [kJ / mol].
    """

    return evaluate_vectorized_system_energy(
        vectorize_system(system), conformer, parameter_delta, parameter_delta_ids
    )


def evaluate_vectorized_system_energy(
    system: Dict[Tuple[str, str], VectorizedHandler],
    conformer: torch.Tensor,
    parameter_delta: Optional[torch.Tensor] = None,
    parameter_delta_ids: Optional[List[Tuple[str, PotentialKey, str]]] = None,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a vectorized OpenFF system containing
    a single molecule.

    Args:
        system: The vectorized system that encodes the potential energy function to
            evaluate.
        conformer: The conformer to evaluate the potential at.
        parameter_delta: An optional tensor of values to perturb the assigned
            parameters by before evaluating the potential energy. If this
            option is specified then ``parameter_delta_ids`` must also be.
        parameter_delta_ids: An optional list of ids associated with the
            ``parameter_delta`` tensor which is used to identify which parameter
            delta matches which assigned parameter. If this option is specified then
            ``parameter_delta`` must also be.

    Returns:
        The potential energy of the conformer [kJ / mol].
    """

    if not (
        parameter_delta is None
        and parameter_delta_ids is None
        or parameter_delta is not None
        and parameter_delta_ids is not None
    ):

        raise MissingArgumentsError(
            "Either both ``parameter_delta`` and ``parameter_delta_ids`` must be "
            "specified or neither must be."
        )

    total_energy = torch.zeros(1)

    for (handler_type, handler_expression), handler in system.items():

        handler_delta, handler_delta_ids = None, None

        if parameter_delta is not None:

            handler_delta_ids = [
                (parameter_id, attribute_id)
                for handler_type, parameter_id, attribute_id in parameter_delta_ids
                if handler_type == handler_type
            ]
            handler_delta_indices = torch.tensor(
                [
                    i
                    for i, (handler_type, _, _) in enumerate(parameter_delta_ids)
                    if handler_type == handler_type
                ]
            )
            handler_delta = parameter_delta[handler_delta_indices]

        handler_energy = evaluate_vectorized_handler_energy(
            handler,
            handler_type,
            handler_expression,
            conformer,
            handler_delta,
            handler_delta_ids,
        )

        total_energy += handler_energy

    return total_energy
