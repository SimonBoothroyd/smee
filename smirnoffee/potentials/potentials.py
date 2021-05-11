from typing import List, Tuple

import torch
from openff.system.models import PotentialKey

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
