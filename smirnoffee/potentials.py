from typing import List, Optional, Tuple

import torch
from openff.system.components.potentials import PotentialHandler
from openff.system.models import PotentialKey

from smirnoffee.exceptions import MissingArguments
from smirnoffee.smirnoff import vectorize_handler

_POTENTIAL_FUNCTIONS = {}


def potential_energy_function(handler_type, energy_expression):
    """A decorator used to flag a function as being able to compute the potential for a
    specific handler and its associated energy expression.."""

    def _potential_function_inner(func):

        if energy_expression in _POTENTIAL_FUNCTIONS:

            raise KeyError(
                f"A potential energy function is already defined for "
                f"handler={handler_type} fn={energy_expression}."
            )

        _POTENTIAL_FUNCTIONS[(handler_type, energy_expression)] = func
        return func

    return _potential_function_inner


def _add_parameter_delta(
    parameters: torch.Tensor,
    parameter_ids: List[Tuple[PotentialKey, Tuple[str, ...]]],
    delta: torch.Tensor,
    delta_ids: List[Tuple[PotentialKey, str]],
) -> torch.Tensor:
    """Adds a 1D vector of parameter 'deltas' to an existing 2D tensor of parameter
    values, whereby each parameter in the flat delta is matched to a parameter in the
    2D tensor according to a combination of its 'potential key' and the specific
    attribute it represents.

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


@potential_energy_function("Bonds", "1/2 * k * (r - length) ** 2")
def evaluate_harmonic_bond_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:

    if len(atom_indices) == 0:
        return torch.zeros(1)

    distances = torch.norm(
        conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]], dim=1
    )

    return (0.5 * parameters[:, 0] * (distances - parameters[:, 1]) ** 2).sum()


@potential_energy_function("Angles", "1/2 * k * (theta - angle) ** 2")
def evaluate_harmonic_angle_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:

    if len(atom_indices) == 0:
        return torch.zeros(1)

    vector_ab = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]]
    vector_ab /= torch.norm(vector_ab, dim=1).unsqueeze(1)

    vector_ac = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 2]]
    vector_ac /= torch.norm(vector_ac, dim=1).unsqueeze(1)

    # TODO: handle the ACOS singularity.
    angles = torch.rad2deg(torch.acos((vector_ab * vector_ac).sum(dim=1)))

    return (0.5 * parameters[:, 0] * (angles - parameters[:, 1]) ** 2).sum()


def _evaluate_cosine_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:

    if len(atom_indices) == 0:
        return torch.zeros(1)

    # Based on the OpenMM formalism.
    vector_ab = conformer[atom_indices[:, 0]] - conformer[atom_indices[:, 1]]
    vector_cb = conformer[atom_indices[:, 2]] - conformer[atom_indices[:, 1]]
    vector_cd = conformer[atom_indices[:, 2]] - conformer[atom_indices[:, 3]]

    vector_ab_cross_cb = torch.cross(vector_ab, vector_cb, dim=1)
    vector_cb_cross_cd = torch.cross(vector_cb, vector_cd, dim=1)

    vector_ab_cross_cb /= torch.norm(vector_ab_cross_cb, dim=1).unsqueeze(1)
    vector_cb_cross_cd /= torch.norm(vector_cb_cross_cd, dim=1).unsqueeze(1)

    cos_phi = (vector_ab_cross_cb * vector_cb_cross_cd).sum(dim=1)

    # TODO: handle the ACOS singularity.
    phi = torch.acos(cos_phi)
    phi *= torch.where((vector_ab * vector_cb_cross_cd).sum(dim=1) < 0, -1.0, 1.0)

    return (
        parameters[:, 0]
        / parameters[:, 3]
        * (1.0 + torch.cos(parameters[:, 1] * phi - torch.deg2rad(parameters[:, 2])))
    ).sum()


@potential_energy_function("ProperTorsions", "k*(1+cos(periodicity*theta-phase))")
def evaluate_cosine_proper_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:

    return _evaluate_cosine_torsion_energy(conformer, atom_indices, parameters)


@potential_energy_function("ImproperTorsions", "k*(1+cos(periodicity*theta-phase))")
def evaluate_cosine_improper_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:

    return _evaluate_cosine_torsion_energy(conformer, atom_indices, parameters)


def evaluate_handler_energy(
    handler: PotentialHandler,
    conformer: torch.Tensor,
    parameter_delta: Optional[torch.Tensor] = None,
    parameter_delta_ids: Optional[List[Tuple[PotentialKey, str]]] = None,
) -> torch.Tensor:

    if not (
        parameter_delta is None
        and parameter_delta_ids is None
        or parameter_delta is not None
        and parameter_delta_ids is not None
    ):

        raise MissingArguments(
            "Either both ``parameter_delta`` and ``parameter_delta_ids`` must be "
            "specified or neither must be."
        )

    if parameter_delta is not None:

        assert len(parameter_delta_ids) == parameter_delta.shape[0], (
            f"each parameter delta (n={len(parameter_delta_ids)}) must have an "
            f"associated id (n={parameter_delta.shape[0]})"
        )

    indices, parameter_ids, parameters = vectorize_handler(handler)

    if len(parameter_ids) == 0:
        return torch.zeros(1)

    if parameter_delta is not None:

        parameters = _add_parameter_delta(
            parameters, parameter_ids, parameter_delta, parameter_delta_ids
        )

    energy_expression = _POTENTIAL_FUNCTIONS[(handler.name, handler.expression)]
    handler_energy = energy_expression(conformer, indices, parameters)

    return handler_energy


# def evaluate_system_energy(
#     system: System,
#     conformer: torch.Tensor,
#     parameter_delta: Optional[torch.Tensor] = None,
#     parameter_delta_ids: Optional[Tuple[str, PotentialKey, str]] = None,
# ) -> torch.Tensor:
#
#     if not (
#         parameter_delta is None
#         and parameter_delta_ids is None
#         or parameter_delta is not None
#         and parameter_delta_ids is not None
#     ):
#
#         raise MissingArguments(
#             "Either both ``parameter_delta`` and ``parameter_delta_ids`` must be "
#             "specified or neither must be."
#         )
#
#     total_energy = torch.zeros(1)
#
#     for handler in system.handlers.values():
#
#         handler_delta_ids = [
#             (parameter_id, attribute_id)
#             for handler_type, parameter_id, attribute_id in parameter_delta_ids
#             if handler_type == handler.name
#         ]
#         handler_delta_indices = torch.tensor(
#             [
#                 i
#                 for i, (handler_type, _, _) in enumerate(parameter_delta_ids)
#                 if handler_type == handler.name
#             ]
#         )
#         handler_delta = parameter_delta[handler_delta_indices]
#
#         total_energy += evaluate_handler_energy(
#             handler, conformer, handler_delta, handler_delta_ids
#         )
#
#     return total_energy
