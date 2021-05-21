import inspect
import itertools
from typing import Dict, List, Tuple, Union

import numpy
import torch
from openff.system.components.potentials import Potential, PotentialHandler
from openff.system.components.smirnoff import (
    ElectrostaticsMetaHandler,
    SMIRNOFFAngleHandler,
    SMIRNOFFBondHandler,
    SMIRNOFFImproperTorsionHandler,
    SMIRNOFFPotentialHandler,
    SMIRNOFFProperTorsionHandler,
    SMIRNOFFvdWHandler,
)
from openff.system.components.system import System
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

VectorizedHandler = Tuple[
    torch.Tensor, torch.Tensor, List[Tuple[PotentialKey, Tuple[str, ...]]]
]


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


def _vectorize_valence_handler(
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
    return _vectorize_valence_handler(handler, ["k", "length"])


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
    return _vectorize_valence_handler(handler, ["k", "angle"])


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
    return _vectorize_valence_handler(handler, ["k", "periodicity", "phase", "idivf"])


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
    return _vectorize_valence_handler(handler, ["k", "periodicity", "phase", "idivf"])


def _vectorize_nonbonded_scales(
    handler: Union[SMIRNOFFvdWHandler, ElectrostaticsMetaHandler], molecule: Molecule
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[float, ...], Tuple[str, ...]]:
    """Vectorizes the 1-n scale factors associated with a set of nonbonded interaction
    pairs.
    """

    if not numpy.isclose(handler.scale_15, 1.0):
        raise NotImplementedError()

    handler_scale_factors = {
        "scale_12": 0.0,
        "scale_13": handler.scale_13,
        "scale_14": handler.scale_14,
        "scale_1n": 1.0,
    }

    interaction_pairs = {
        tuple(sorted((bond.atom1_index, bond.atom2_index))): "scale_12"
        for bond in molecule.bonds
    }

    pairs_13 = {
        tuple(
            sorted((angle[0].molecule_atom_index, angle[2].molecule_atom_index))
        ): "scale_13"
        for angle in molecule.angles
    }
    interaction_pairs.update(
        {key: value for key, value in pairs_13.items() if key not in interaction_pairs}
    )

    pairs_14 = {
        tuple(
            sorted((proper[0].molecule_atom_index, proper[3].molecule_atom_index))
        ): "scale_14"
        for proper in molecule.propers
    }
    interaction_pairs.update(
        {key: value for key, value in pairs_14.items() if key not in interaction_pairs}
    )

    pairs_1n = {
        tuple(sorted(pair)): "scale_1n"
        for pair in itertools.combinations(range(molecule.n_atoms), 2)
    }
    interaction_pairs.update(
        {key: value for key, value in pairs_1n.items() if key not in interaction_pairs}
    )

    if len(interaction_pairs) == 0 or all(
        numpy.isclose(handler_scale_factors[scale_type], 0.0)
        for scale_type in interaction_pairs.values()
    ):
        return (), (), ()

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
    """Apply Lorentz-Berthelot mixing rules to a pair of LJ parameters."""
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

    if handler.mixing_rule != "lorentz-berthelot":

        raise NotImplementedError(
            "Only the Lorentz-Berthelot vdW mixing rule is currently supported."
        )

    pair_indices, scale_factors, scale_types = _vectorize_nonbonded_scales(
        handler, molecule
    )

    if len(pair_indices) == 0:
        return torch.tensor([]), torch.tensor([]), []

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

    if len(pair_indices) == 0:
        return torch.tensor([]), torch.tensor([]), []

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


def vectorize_handler(
    handler: SMIRNOFFPotentialHandler, molecule: Molecule = None
) -> VectorizedHandler:
    """Maps a SMIRNOFF potential handler into a tensor of the atom indices involved in
    each slot, a 2D tensor of the values of the parameters, and a list of identifiers
    which uniquely maps each value in the parameter tensor back to the original force
    field parameter

    Args:
        handler: The handler to vectorize.
        molecule: The molecule that the handler is associated with.

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

    vectorizer_spec = inspect.signature(vectorizer)

    if "molecule" in vectorizer_spec.parameters:

        if molecule is None:

            raise TypeError(
                "The `molecule` attribute must be provided when vectorizing a "
                "nonbonded handler."
            )

        return vectorizer(handler, molecule)

    return vectorizer(handler)


def vectorize_system(system: System) -> Dict[Tuple[str, str], VectorizedHandler]:
    """Maps a SMIRNOFF parameterized system object into a collection of tensor
    representations.

    For each potential handler in the system, a tensor of the atom indices involved in
    each handler slot, a 2D tensor of the values of the parameters, and a list of
    identifiers which uniquely maps each value in the parameter tensor back to the
    original force field parameter will be returned.

    Args:
        system: The system to vectorize.

    Returns:
        A dictionary where each key is a tuple of a handler name and an energy
        expresion, and each value is a vectorized representation of the associated
        handler. See ``vectorize_handler`` for more information.
    """

    reference_molecules = [*system.topology.reference_molecules]

    if len(reference_molecules) != 1:

        raise NotImplementedError(
            "Only systems containing a single molecules can safely be vectorized."
        )

    return {
        (handler_type, handler.expression): vectorize_handler(
            handler, reference_molecules[0]
        )
        for handler_type, handler in system.handlers.items()
    }
