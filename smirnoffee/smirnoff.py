import itertools
from typing import List, Tuple, Union

import numpy
import torch
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.components.smirnoff import (
    ElectrostaticsMetaHandler,
    SMIRNOFFAngleHandler,
    SMIRNOFFBondHandler,
    SMIRNOFFImproperTorsionHandler,
    SMIRNOFFProperTorsionHandler,
    SMIRNOFFvdWHandler,
)
from openff.system.models import PotentialKey, TopologyKey
from openff.toolkit.topology import Molecule
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
    "vdW": {
        "epsilon": unit.kilojoules / unit.mole,
        "sigma": unit.angstrom,
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
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[PotentialKey, Tuple[str, ...]]]]:
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
        return torch.tensor([]), torch.tensor([]), []

    first_parameter = next(iter(handler.potentials.values()))
    assert {*first_parameter.parameters} == {*attributes}

    parameter_ids, parameter_tuples = zip(
        *(
            (
                potential_key,
                tuple(
                    _get_parameter_value(potential, handler.type, attribute)
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
        assigned_parameters,
        [(assigned_id, tuple(attributes)) for assigned_id in assigned_ids],
    )


@handler_vectorizer("Bonds")
def vectorize_bond_handler(
    handler: SMIRNOFFBondHandler,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[PotentialKey, Tuple[str, str]]]]:
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

    # noinspection PyTypeChecker
    return _vectorize_smirnoff_handler(handler, ["k", "length"])


@handler_vectorizer("Angles")
def vectorize_angle_handler(
    handler: SMIRNOFFAngleHandler,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[PotentialKey, Tuple[str, str]]]]:
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

    # noinspection PyTypeChecker
    return _vectorize_smirnoff_handler(handler, ["k", "angle"])


@handler_vectorizer("ProperTorsions")
def vectorize_proper_handler(
    handler: SMIRNOFFProperTorsionHandler,
) -> Tuple[
    torch.Tensor, torch.Tensor, List[Tuple[PotentialKey, Tuple[str, str, str, str]]]
]:
    """Maps a SMIRNOFF proper torsion potential handler into a ``(n_propers, 4)`` tensor
    of the atom indices involved in each torsion, a list of identifiers which uniquely
    maps an assigned parameters back to the original force field parameter, and a 2D
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

    # noinspection PyTypeChecker
    return _vectorize_smirnoff_handler(handler, ["k", "periodicity", "phase", "idivf"])


@handler_vectorizer("ImproperTorsions")
def vectorize_improper_handler(
    handler: SMIRNOFFImproperTorsionHandler,
) -> Tuple[
    torch.Tensor, torch.Tensor, List[Tuple[PotentialKey, Tuple[str, str, str, str]]]
]:
    """Maps a SMIRNOFF improper torsion potential handler into a ``(n_impropers, 4)``
    tensor of the atom indices involved in each torsion, a list of identifiers which
    uniquely maps an assigned parameters back to the original force field parameter,
    and a 2D tensor of the values of the parameters where the first column are force
    constants, the second column periodicities, the third column phases, and the last
    column ``idivf`` values.

    Args:
        handler: The handler to vectorize.

    Returns:
        The atom indices involved in the potential, a list of identifiers which
        uniquely maps an assigned parameter back to the original force field parameter,
        and a tensor of the values of the parameters.
    """

    # noinspection PyTypeChecker
    return _vectorize_smirnoff_handler(handler, ["k", "periodicity", "phase", "idivf"])


def vectorize_valence_handler(
    handler: PotentialHandler,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[PotentialKey, Tuple[str, ...]]]]:
    """Maps a valence (i.e. not vdW or electrostatics) SMIRNOFF potential handler into
    a tensor of the atom indices involved in each slot, a list of identifiers which
    uniquely maps an assigned parameters back to the original force field parameter,
    and a 2D tensor of the values of the parameters.

    Args:
        handler: The handler to vectorize.

    Returns:
        The atom indices involved in the potential, a list of identifiers which
        uniquely maps an assigned parameter back to the original force field parameter,
        and a tensor of the values of the parameters.
    """

    if handler.type not in _HANDLER_TO_VECTORIZER:

        raise NotImplementedError(
            f"Vectorizing {handler.type} handlers is not yet supported."
        )

    vectorizer = _HANDLER_TO_VECTORIZER[handler.type]
    return vectorizer(handler)


def _vectorize_nonbonded_scales(
    handler: Union[SMIRNOFFvdWHandler, ElectrostaticsMetaHandler], molecule: Molecule
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[float, ...], Tuple[str, ...]]:
    """ """

    if not numpy.isclose(handler.scale_15, 1.0):
        raise NotImplementedError()

    handler_scale_factors = {
        "scale_12": 0.0,
        "scale_13": handler.scale_13,
        "scale_14": handler.scale_14,
        "scale_1n": 1.0,
    }

    interaction_pairs = {
        **{
            tuple(sorted((bond.atom1_index, bond.atom2_index))): "scale_12"
            for bond in molecule.bonds
        },
        **{
            tuple(
                sorted((angle[0].molecule_atom_index, angle[2].molecule_atom_index))
            ): "scale_13"
            for angle in molecule.angles
        },
        **{
            tuple(
                sorted((proper[0].molecule_atom_index, proper[3].molecule_atom_index))
            ): "scale_14"
            for proper in molecule.propers
        },
    }
    interaction_pairs.update(
        {
            tuple(sorted(pair)): "scale_1n"
            for pair in itertools.combinations(range(molecule.n_atoms), 2)
            if tuple(sorted(pair)) not in interaction_pairs
        }
    )

    pair_indices, scale_factors, scale_types = zip(
        *(
            (pair, handler_scale_factors[scale_type], scale_type)
            for pair, scale_type in interaction_pairs.items()
            if not numpy.isclose(handler_scale_factors[scale_type], 0.0)
        )
    )

    return pair_indices, scale_factors, scale_types


def _lorentz_berthelot(
    epsilon_1: float, epsilon_2: float, sigma_1: float, sigma_2: float
) -> Tuple[float, float]:

    return numpy.sqrt(epsilon_1 * epsilon_2), 0.5 * (sigma_1 + sigma_2)


@handler_vectorizer("vdW")
def vectorize_vdw_handler(
    handler: SMIRNOFFvdWHandler, molecule: Molecule
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[Tuple[PotentialKey, Tuple[str, str, str]]],
]:
    """Maps a SMIRNOFF vdW potential handler into a vectorized form which
    which more readily allows evaluating the potential energy.

    Args:
        handler: The handler to vectorize.
        molecule: The molecule that the handler was created for.

    Returns:
        A tuple of:

        * a tensor of the atom indices involved in each pair of electrostatics
          interactions with shape=(n_pairs, 2)
        * a 2D tensor of the values of the parameters with shape=(n_pairs, 3) where
          the first and second column are the epsilon and sigma parameters respectively
          combined using the handler specified mixing rule, and the final column a
          scale factor
        * a list of identifiers which uniquely maps an assigned parameters back to the
          original force field parameter
    """

    pair_indices, scale_factors, scale_types = _vectorize_nonbonded_scales(
        handler, molecule
    )

    parameter_ids = {
        pair: (
            handler.potentials[handler.slot_map[TopologyKey(atom_indices=(pair[0],))]],
            handler.potentials[handler.slot_map[TopologyKey(atom_indices=(pair[1],))]],
        )
        for pair in pair_indices
    }

    parameters = torch.tensor(
        [
            [
                *_lorentz_berthelot(
                    _get_parameter_value(parameter_ids[pair][0], "vdW", "epsilon"),
                    _get_parameter_value(parameter_ids[pair][1], "vdW", "epsilon"),
                    _get_parameter_value(parameter_ids[pair][0], "vdW", "sigma"),
                    _get_parameter_value(parameter_ids[pair][1], "vdW", "sigma"),
                ),
                scale_factor,
            ]
            for pair, scale_factor in zip(pair_indices, scale_factors)
        ]
    )

    parameter_ids = [
        (PotentialKey(id="[*:1]"), ("epsilon", "sigma", scale_type))
        for scale_type in scale_types
    ]

    return torch.tensor(pair_indices), parameters, parameter_ids


@handler_vectorizer("Electrostatics")
def vectorize_electrostatics_handler(
    handler: ElectrostaticsMetaHandler, molecule: Molecule
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[Tuple[PotentialKey, Tuple[str, str, str]]],
]:
    """Maps a SMIRNOFF electrostatics potential handler into a vectorized form which
    which more readily allows evaluating the potential energy.

    Args:
        handler: The handler to vectorize.
        molecule: The molecule that the handler was created for.

    Returns:
        A tuple of:

        * a tensor of the atom indices involved in each pair of electrostatics
          interactions with shape=(n_pairs, 2)
        * a 2D tensor of the values of the parameters with shape=(n_pairs, 3) where
          the first column is the charge on the first atom, the second column the charge
          on the second atom, and the final column a scale factor
        * a list of identifiers which uniquely maps an assigned parameters back to the
          original force field parameter
    """

    pair_indices, scale_factors, scale_types = _vectorize_nonbonded_scales(
        handler, molecule
    )

    atom_charges = torch.tensor(
        [charge.to(unit.e).magnitude for charge in handler.charges.values()]
    )

    parameters = torch.tensor(
        [
            [atom_charges[pair[0]], atom_charges[pair[1]], scale_factor]
            for pair, scale_factor in zip(pair_indices, scale_factors)
        ]
    )

    parameter_ids = [
        (PotentialKey(id="[*:1]"), ("q1", "q2", scale_type))
        for scale_type in scale_types
    ]

    return torch.tensor(pair_indices), parameters, parameter_ids


def vectorize_nonbonded_handler(
    handler: PotentialHandler,
    molecule: Molecule,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[Tuple[PotentialKey, Tuple[str, str, str]]],
]:
    """Maps a SMIRNOFF nonbonded potential handler into a vectorized form which
    which more readily allows evaluating the potential energy.

    Args:
        handler: The handler to vectorize.
        molecule: The molecule that the handler was created for.

    Returns:
        A tuple of:

        * a tensor of the atom indices involved in each pair of nonbonded
          interactions with shape=(n_pairs, 2)
        * a 2D tensor of the values of the parameters with shape=(n_pairs, n_params)
        * a list of identifiers which uniquely maps an assigned parameters back to the
          original force field parameter
    """

    if handler.type not in _HANDLER_TO_VECTORIZER:

        raise NotImplementedError(
            f"Vectorizing {handler.type} handlers is not yet supported."
        )

    vectorizer = _HANDLER_TO_VECTORIZER[handler.type]
    return vectorizer(handler, molecule)


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
