from typing import List, Tuple

import torch
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.components.smirnoff import SMIRNOFFAngleHandler, SMIRNOFFBondHandler
from openff.system.models import PotentialKey
from openff.units import unit

_DEFAULT_UNITS = {
    "Bonds": {
        "k": unit.kilojoules / unit.mole / unit.angstrom ** 2,
        "length": unit.angstrom,
    },
    "Angles": {
        "k": unit.kilojoules / unit.mole / unit.degree ** 2,
        "angle": unit.degree,
    },
    "ProperTorsions": {
        "k": unit.kilojoules / unit.mole,
        "periodicity": unit.dimensionless,
        "phase": unit.degree,
        "idivf": unit.dimensionless,
    },
    "ImproperTorsions": {
        "k": unit.kilojoules / unit.mole,
        "periodicity": unit.dimensionless,
        "phase": unit.degree,
        "idivf": unit.dimensionless,
    },
}


_HANDLER_TO_VECTORIZER = {}


def handler_vectorizer(handler_type):
    """A decorator used to flag a function as being able to vectorize a handler."""

    def _handler_vectorizer_inner(func):

        if handler_type in _HANDLER_TO_VECTORIZER:

            raise KeyError(
                f"A vectorizer for the {handler_type} handler is already registered."
            )

        _HANDLER_TO_VECTORIZER[handler_type] = func
        return func

    return _handler_vectorizer_inner


def _get_parameter_value(potential: Potential, handler: str, parameter: str) -> float:
    """Returns the value of a parameter in its default units"""
    return (
        potential.parameters[parameter].to(_DEFAULT_UNITS[handler][parameter]).magnitude
    )


def _vectorize_smirnoff_handler(
    handler: PotentialHandler, attributes: List[str]
) -> Tuple[torch.Tensor, List[Tuple[PotentialKey, Tuple[str, ...]]], torch.Tensor]:
    """Maps a SMIRNOFF potential handler into a tensor of the atom indices involved
    in the potential (e.g. for an angle handler this would be a ``(n_angles, 3)``
    tensor), a list of identifiers which uniquely maps an assigned parameter back
    to the original force field parameter, and a tensor of the values of the
    parameters.

    Args:
        handler: The handler to vectorize.
        attributes: The attributes of a parameter (e.g. ``'k'``, ``'length'``) to
            include in the parameters tensor.

    Returns:
        The atom indices involved in the potential, a list of identifiers which
        uniquely maps an assigned parameter back to the original force field parameter,
        and a tensor of the values of the parameters.
    """

    if len(handler.potentials) == 0:
        return torch.tensor([]), [], torch.tensor([])

    first_parameter = next(iter(handler.potentials.values()))
    assert {*first_parameter.parameters} == {*attributes}

    parameter_ids, parameter_tuples = zip(
        *(
            (
                potential_key,
                tuple(
                    _get_parameter_value(potential, handler.name, attribute)
                    for attribute in attributes
                ),
            )
            for potential_key, potential in handler.potentials.items()
        )
    )
    parameters = torch.tensor(parameter_tuples)

    atom_indices, assigned_ids = zip(
        *(
            (topology_key.atom_indices, potential_key)
            for topology_key, potential_key in handler.slot_map.items()
        )
    )

    assignment_matrix = torch.tensor(
        [parameter_ids.index(parameter_id) for parameter_id in assigned_ids]
    )

    assigned_parameters = parameters[assignment_matrix]

    return (
        torch.tensor(atom_indices),
        [(assigned_id, tuple(attributes)) for assigned_id in assigned_ids],
        assigned_parameters,
    )


@handler_vectorizer("Bonds")
def vectorize_bond_handler(
    handler: SMIRNOFFBondHandler,
) -> Tuple[torch.Tensor, List[Tuple[PotentialKey, Tuple[str, str]]], torch.Tensor]:
    """Maps a SMIRNOFF bond potential handler into a ``(n_bonds, 2)`` tensor of the
    atom indices involved in each bond, a list of identifiers which uniquely maps an
    assigned parameters back to the original force field parameter, and a 2D tensor of
    the values of the parameters where the first column are force constants and the
    second column bond lengths.

    Args:
        handler: The handler to vectorize.

    Returns:
        The atom indices involved in the potential, a list of identifiers which
        uniquely maps an assigned parameter back to the original force field parameter,
        and a tensor of the values of the parameters.
    """

    return _vectorize_smirnoff_handler(handler, ["k", "length"])


@handler_vectorizer("Angles")
def vectorize_angle_handler(
    handler: SMIRNOFFAngleHandler,
) -> Tuple[torch.Tensor, List[Tuple[PotentialKey, Tuple[str, str]]], torch.Tensor]:
    """Maps a SMIRNOFF angle potential handler into a ``(n_bonds, 3)`` tensor of the
    atom indices involved in each angle, a list of identifiers which uniquely maps an
    assigned parameters back to the original force field parameter, and a 2D tensor of
    the values of the parameters where the first column are force constants and the
    second column equilibrium angles.

    Args:
        handler: The handler to vectorize.

    Returns:
        The atom indices involved in the potential, a list of identifiers which
        uniquely maps an assigned parameter back to the original force field parameter,
        and a tensor of the values of the parameters.
    """

    return _vectorize_smirnoff_handler(handler, ["k", "angle"])


@handler_vectorizer("ProperTorsions")
def vectorize_proper_handler(
    handler: SMIRNOFFBondHandler,
) -> Tuple[
    torch.Tensor, List[Tuple[PotentialKey, Tuple[str, str, str, str]]], torch.Tensor
]:
    """Maps a SMIRNOFF proper torsion potential handler into a ``(n_bonds, 4)`` tensor
    of the atom indices involved in each torsion, a list of identifiers which uniquely
    maps an assigned parameters back to the original force field parameter, and a 4D
    tensor of the values of the parameters where the first column are force constants,
    the second column periodicities, the third column phases, and the last column
    ``idivf`` values.

    Args:
        handler: The handler to vectorize.

    Returns:
        The atom indices involved in the potential, a list of identifiers which
        uniquely maps an assigned parameter back to the original force field parameter,
        and a tensor of the values of the parameters.
    """

    return _vectorize_smirnoff_handler(handler, ["k", "periodicity", "phase", "idivf"])


@handler_vectorizer("ImproperTorsions")
def vectorize_improper_handler(
    handler: SMIRNOFFBondHandler,
) -> Tuple[torch.Tensor, List[Tuple[PotentialKey, Tuple[str, str]]], torch.Tensor]:
    """Maps a SMIRNOFF improper torsion potential handler into a ``(n_bonds, 4)`` tensor
    of the atom indices involved in each torsion, a list of identifiers which uniquely
    maps an assigned parameters back to the original force field parameter, and a 4D
    tensor of the values of the parameters where the first column are force constants,
    the second column periodicities, the third column phases, and the last column
    ``idivf`` values.

    Args:
        handler: The handler to vectorize.

    Returns:
        The atom indices involved in the potential, a list of identifiers which
        uniquely maps an assigned parameter back to the original force field parameter,
        and a tensor of the values of the parameters.
    """

    return _vectorize_smirnoff_handler(handler, ["k", "periodicity", "phase", "idivf"])


def vectorize_handler(
    handler: PotentialHandler,
) -> Tuple[torch.Tensor, List[Tuple[PotentialKey, Tuple[str, ...]]], torch.Tensor]:
    """Maps a SMIRNOFF potential handler into a tensor of the atom indices involved in
    each slot, a list of identifiers which uniquely maps an assigned parameters back to
    the original force field parameter, and a 4D tensor of the values of the parameters.

    Args:
        handler: The handler to vectorize.

    Returns:
        The atom indices involved in the potential, a list of identifiers which
        uniquely maps an assigned parameter back to the original force field parameter,
        and a tensor of the values of the parameters.
    """

    if handler.name not in _HANDLER_TO_VECTORIZER:

        raise NotImplementedError(
            f"Vectorizing {handler.name} handlers is not yet supported."
        )

    vectorizer = _HANDLER_TO_VECTORIZER[handler.name]
    return vectorizer(handler)


# def vectorize_system(
#     system: System,
# ) -> Dict[
#     str, Tuple[torch.Tensor, List[Tuple[PotentialKey, Tuple[str, ...]]], torch.Tensor]
# ]:
#     """Maps an OpenFF SMIRNOFF system into a dictionary representation. Each key
#     corresponds to a particular potential handler, and each value contain a tensor of
#     the atom indices involved in each slot, a list of identifiers which uniquely maps
#     an assigned parameters back to the original force field parameter, and a 4D tensor
#     of the values of the parameters.
#
#     Args:
#         system: The system to vectorize.
#
#     Returns:
#         The atom indices involved in the potential, a list of identifiers which
#         uniquely maps an assigned parameter back to the original force field parameter,
#         and a tensor of the values of the parameters.
#     """
#
#     return {
#         handler_type: vectorize_handler(handler)
#         for handler_type, handler in system.handlers.items()
#     }
