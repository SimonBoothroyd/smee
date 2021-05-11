from typing import List, Optional, Tuple

import torch
from openff.system.components.potentials import PotentialHandler
from openff.system.models import PotentialKey

from smirnoffee.exceptions import MissingArgumentsError
from smirnoffee.potentials import (
    _POTENTIAL_ENERGY_FUNCTIONS,
    add_parameter_delta,
    potential_energy_function,
)
from smirnoffee.smirnoff import vectorize_valence_handler


@potential_energy_function("Bonds", "1/2 * k * (r - length) ** 2")
def evaluate_harmonic_bond_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a set of bonds for a given conformer
    using a harmonic potential of the form:

    `1/2 * k * (r - length) ** 2`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each bond with
            shape=(n_bonds, 4).
        parameters: A tensor with shape=(n_bonds, 2) where there first column
            contains the force constants ``k``, and the second the equilibrium
            bond ``length``.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

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
    """Evaluates the potential energy [kJ / mol] of a set of valence angles
    for a given conformer using a harmonic potential of the form:

    `1/2 * k * (theta - angle) ** 2`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each valence angle with
            shape=(n_angles, 4).
        parameters: A tensor with shape=(n_angles, 2) where there first column
            contains the force constants ``k``, and the second the equilibrium
            ``angle``.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

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
    """Evaluates the potential energy [kJ / mol] of a set of torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each proper torsion with
            shape=(n_torsions, 4).
        parameters: A tensor with shape=(n_torsions, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

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
    """Evaluates the potential energy [kJ / mol] of a set of proper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each proper torsion with
            shape=(n_propers, 4).
        parameters: A tensor with shape=(n_propers, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kJ / mol].
    """
    return _evaluate_cosine_torsion_energy(conformer, atom_indices, parameters)


@potential_energy_function("ImproperTorsions", "k*(1+cos(periodicity*theta-phase))")
def evaluate_cosine_improper_torsion_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of a set of improper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The indices of the atoms involved in each improper torsion with
            shape=(n_impropers, 4).
        parameters: A tensor with shape=(n_impropers, 4) where there first column
            contains the force constants ``k``, the second the ``periodicities``,
            the third the ``phase`` and the fourth an ``idivf`` factor to divide the
            force constant by.

    Returns:
        The evaluated potential energy [kJ / mol].
    """
    return _evaluate_cosine_torsion_energy(conformer, atom_indices, parameters)


def evaluate_valence_energy(
    valence_handler: PotentialHandler,
    conformer: torch.Tensor,
    parameter_delta: Optional[torch.Tensor] = None,
    parameter_delta_ids: Optional[List[Tuple[PotentialKey, str]]] = None,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] contribution of a particular valence
    potential handler for a given conformer.

    Args:
        valence_handler: The valence potential handler that encodes the potential
            energy function to evaluate.
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

    indices, parameters, parameter_ids = vectorize_valence_handler(valence_handler)

    if len(parameter_ids) == 0:
        return torch.zeros(1)

    if parameter_delta is not None:

        parameters = add_parameter_delta(
            parameters, parameter_ids, parameter_delta, parameter_delta_ids
        )

    energy_expression = _POTENTIAL_ENERGY_FUNCTIONS[
        (valence_handler.name, valence_handler.expression)
    ]
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
#         raise MissingArgumentsError(
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
