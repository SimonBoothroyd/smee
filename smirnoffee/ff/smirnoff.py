"""Convert SMIRNOFF parameters into tensors."""
import collections
import inspect
import itertools

import openff.interchange.components.potentials
import openff.interchange.models
import openff.interchange.smirnoff._base
import openff.interchange.smirnoff._nonbonded
import openff.interchange.smirnoff._valence
import openff.toolkit
import openff.units
import torch

import smirnoffee.ff

_UNITLESS = openff.units.unit.dimensionless
_ANGSTROM = openff.units.unit.angstrom
_DEGREES = openff.units.unit.degrees
_KJ_PER_MOL = openff.units.unit.kilojoules / openff.units.unit.mole
_ELEMENTARY_CHARGE = openff.units.unit.elementary_charge

_HANDLER_CONVERTERS = {}
_HANDLER_DEFAULT_UNITS = {}


ConvertedHandler = tuple[
    smirnoffee.ff.TensorPotential, list[smirnoffee.ff.ParameterMap]
]


def _get_value(
    potential: openff.interchange.components.potentials.Potential,
    handler: str,
    parameter: str,
) -> float:
    """Returns the value of a parameter in its default units"""
    default_units = _HANDLER_DEFAULT_UNITS[handler][parameter]
    return potential.parameters[parameter].m_as(default_units)


def _find_interaction_pairs(
    topology: openff.toolkit.Topology,
) -> dict[tuple[int, int], str]:
    """Find the 1-n scale factors associated with a set of non-bonded interaction
    pairs.
    """

    interaction_pairs = {
        tuple(sorted((bond.atom1_index, bond.atom2_index))): "scale_12"
        for bond in topology.bonds
    }

    pairs_13 = {
        tuple(
            sorted((angle[0].molecule_atom_index, angle[2].molecule_atom_index))
        ): "scale_13"
        for angle in topology.angles
    }
    interaction_pairs.update(
        {key: value for key, value in pairs_13.items() if key not in interaction_pairs}
    )

    pairs_14 = {
        tuple(
            sorted((proper[0].molecule_atom_index, proper[3].molecule_atom_index))
        ): "scale_14"
        for proper in topology.propers
    }
    interaction_pairs.update(
        {key: value for key, value in pairs_14.items() if key not in interaction_pairs}
    )

    pairs_1n = {
        tuple(sorted(pair)): "scale_1n"
        for pair in itertools.combinations(range(topology.n_atoms), 2)
    }
    interaction_pairs.update(
        {key: value for key, value in pairs_1n.items() if key not in interaction_pairs}
    )

    return interaction_pairs


def _handler_converter(type_: str, default_units: dict[str, openff.units.Unit]):
    """A decorator used to flag a function as being able to vectorize a handler."""

    def _handler_converter_inner(func):
        if type_ in _HANDLER_CONVERTERS:
            raise KeyError(f"A {type_} converter is already registered.")

        _HANDLER_CONVERTERS[type_] = func
        _HANDLER_DEFAULT_UNITS[type_] = default_units

        return func

    return _handler_converter_inner


def _convert_handlers(
    handlers: list[openff.interchange.smirnoff._base.SMIRNOFFCollection],
    parameter_cols: tuple[str, ...],
    global_parameter_cols: tuple[str, ...] | None,
) -> smirnoffee.ff.TensorPotential:
    potential_types = {handler.type for handler in handlers}
    assert len(potential_types) == 1, "multiple handler types found"
    potential_type = next(iter(potential_types))

    potential_fns = {handler.expression for handler in handlers}
    assert len(potential_fns) == 1, "multiple handler functions found"
    potential_fn = next(iter(potential_fns))

    parameters_by_key = {
        parameter_key: parameter
        for handler in handlers
        for parameter_key, parameter in handler.potentials.items()
    }

    parameter_keys = [*parameters_by_key]
    parameters = torch.tensor(
        [
            [
                _get_value(parameters_by_key[parameter_key], potential_type, column)
                for column in parameter_cols
            ]
            for parameter_key in parameter_keys
        ]
    )

    global_parameters = None

    if global_parameter_cols is not None:
        global_parameters_by_column = {
            column: getattr(handler, column)
            for column in global_parameter_cols
            for handler in handlers
        }
        global_parameters_by_column = {
            k: (
                v
                if not hasattr(v, "m_as")
                else v.m_as(_HANDLER_DEFAULT_UNITS[potential_type][k])
            )
            for k, v in global_parameters_by_column.items()
        }

        global_parameters = torch.tensor(
            [global_parameters_by_column[column] for column in global_parameter_cols]
        )

    potential = smirnoffee.ff.TensorPotential(
        type=potential_type,
        fn=potential_fn,
        parameters=parameters,
        parameter_keys=parameter_keys,
        parameter_cols=parameter_cols,
        global_parameters=global_parameters,
        global_parameter_cols=global_parameter_cols,
    )
    return potential


def _convert_valence_handlers(
    handlers: list[openff.interchange.smirnoff._base.SMIRNOFFCollection],
    parameter_cols: tuple[str, ...],
) -> ConvertedHandler:
    potential = _convert_handlers(handlers, parameter_cols, None)

    parameter_key_to_idx = {
        parameter_key: i for i, parameter_key in enumerate(potential.parameter_keys)
    }

    parameter_maps = []

    for handler in handlers:
        atom_idxs = [topology_key.atom_indices for topology_key in handler.key_map]
        parameter_idxs = [
            parameter_key_to_idx[parameter_key]
            for parameter_key in handler.key_map.values()
        ]

        parameter_map = smirnoffee.ff.ParameterMap(
            torch.tensor(atom_idxs, dtype=torch.int64),
            torch.tensor(parameter_idxs, dtype=torch.int64),
        )
        parameter_maps.append(parameter_map)

    return potential, parameter_maps


def _convert_nonbonded_handlers(
    handlers: list[openff.interchange.smirnoff._base.SMIRNOFFCollection],
    topologies: list[openff.toolkit.Topology],
    parameter_cols: tuple[str, ...],
) -> ConvertedHandler:
    potential = _convert_handlers(handlers, parameter_cols, ("scale_13", "scale_14"))

    potential.global_parameter_cols = (
        "scale_12",
        *potential.global_parameter_cols,
        "scale_1n",
    )
    potential.global_parameters = torch.cat(
        [torch.tensor([0.0]), potential.global_parameters, torch.tensor([1.0])]
    )

    parameter_key_to_idx = {
        parameter_key: i for i, parameter_key in enumerate(potential.parameter_keys)
    }

    parameter_maps = []

    for handler, topology in zip(handlers, topologies):
        interaction_pairs = _find_interaction_pairs(topology)

        atom_idxs = torch.tensor([*interaction_pairs])

        global_parameter_idxs = torch.tensor(
            [
                potential.global_parameter_cols.index(scale_type)
                for scale_type in interaction_pairs.values()
            ]
        )

        atom_idx_to_parameter_idx = {
            topology_key.atom_indices[0]: parameter_key_to_idx[parameter_key]
            for topology_key, parameter_key in handler.key_map.items()
        }
        parameter_idxs = torch.tensor(
            [
                (atom_idx_to_parameter_idx[pair[0]], atom_idx_to_parameter_idx[pair[1]])
                for pair in interaction_pairs
            ]
        )

        parameter_map = smirnoffee.ff.ParameterMap(
            atom_idxs, parameter_idxs, global_parameter_idxs
        )
        parameter_maps.append(parameter_map)

    return potential, parameter_maps


@_handler_converter("Bonds", {"k": _KJ_PER_MOL / _ANGSTROM**2, "length": _ANGSTROM})
def _convert_bonds(
    handlers: list[openff.interchange.smirnoff._valence.SMIRNOFFBondCollection],
) -> ConvertedHandler:
    return _convert_valence_handlers(handlers, ("k", "length"))


@_handler_converter("Angles", {"k": _KJ_PER_MOL / _DEGREES**2, "angle": _DEGREES})
def _convert_angles(
    handlers: list[openff.interchange.smirnoff._valence.SMIRNOFFAngleCollection],
) -> ConvertedHandler:
    return _convert_valence_handlers(handlers, ("k", "angle"))


@_handler_converter(
    "ProperTorsions",
    {"k": _KJ_PER_MOL, "periodicity": _UNITLESS, "phase": _DEGREES, "idivf": _UNITLESS},
)
def _convert_propers(
    handlers: list[
        openff.interchange.smirnoff._valence.SMIRNOFFProperTorsionCollection
    ],
) -> ConvertedHandler:
    return _convert_valence_handlers(handlers, ("k", "periodicity", "phase", "idivf"))


@_handler_converter(
    "ImproperTorsions",
    {"k": _KJ_PER_MOL, "periodicity": _UNITLESS, "phase": _DEGREES, "idivf": _UNITLESS},
)
def _convert_impropers(
    handlers: list[
        openff.interchange.smirnoff._valence.SMIRNOFFImproperTorsionCollection
    ],
) -> ConvertedHandler:
    return _convert_valence_handlers(handlers, ("k", "periodicity", "phase", "idivf"))


@_handler_converter("vdW", {"epsilon": _KJ_PER_MOL, "sigma": _ANGSTROM})
def _convert_vdw(
    handlers: list[openff.interchange.smirnoff._nonbonded.SMIRNOFFvdWCollection],
    topologies: list[openff.toolkit.Topology],
) -> ConvertedHandler:
    mixing_rules = {handler.mixing_rule for handler in handlers}
    assert len(mixing_rules) == 1, "multiple mixing rules found"
    mixing_rule = next(iter(mixing_rules))

    if mixing_rule != "lorentz-berthelot":
        raise NotImplementedError("only Lorentz-Berthelot mixing rules are supported.")

    return _convert_nonbonded_handlers(handlers, topologies, ("epsilon", "sigma"))


@_handler_converter("Electrostatics", {"charge": _ELEMENTARY_CHARGE})
def _convert_electrostatics(
    handlers: list[
        openff.interchange.smirnoff._nonbonded.SMIRNOFFElectrostaticsCollection
    ],
    topologies: list[openff.toolkit.Topology],
) -> ConvertedHandler:
    handler_types = {
        potential_key.associated_handler
        for handler in handlers
        for potential_key in handler.potentials
    }

    if handler_types != {"ToolkitAM1BCCHandler"}:
        raise NotImplementedError("only am1bcc electrostatics are supported.")

    return _convert_nonbonded_handlers(handlers, topologies, ("charge",))


def convert_handlers(
    handlers: list[openff.interchange.smirnoff._base.SMIRNOFFCollection],
    topologies: list[openff.toolkit.Topology],
) -> ConvertedHandler:
    """Convert a set of SMIRNOFF parameter handlers into a set of tensor potentials.

    Args:
        handlers: The SMIRNOFF parameter handler collections for a set of interchange
            objects to convert.
        topologies: The topologies associated with each interchange object.

    Returns:
        The potential containing the values of the parameters in each handler
        collection, and a list of maps (one per topology) between molecule elements
        (e.g. bond indices) and parameter indices.


    Examples:

        >>> from openff.toolkit import ForceField, Molecule
        >>> from openff.interchange import Interchange
        >>>
        >>> force_field = ForceField("openff_unconstrained-2.0.0.offxml")
        >>> molecules = [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        >>>
        >>> interchanges = [
        ...     Interchange.from_smirnoff(force_field, molecule.to_topology())
        ...     for molecule in molecules
        ... ]
        >>> vdw_handlers = [
        ...     interchange.collections["vdW"] for interchange in interchanges
        ... ]
        >>>
        >>> vdw_potential, applied_vdw_parameters = convert_handlers(interchanges)
    """
    handler_types = {handler.type for handler in handlers}
    assert len(handler_types) == 1, "multiple handler types found"
    handler_type = next(iter(handler_types))

    assert len(handlers) == len(topologies), "mismatched number of topologies"

    if handler_type not in _HANDLER_CONVERTERS:
        raise NotImplementedError(f"{handler_type} handlers is not yet supported.")

    vectorizer = _HANDLER_CONVERTERS[handler_type]
    vectorizer_spec = inspect.signature(vectorizer)

    vectorizer_kwargs = {}

    if "topologies" in vectorizer_spec.parameters:
        vectorizer_kwargs["topologies"] = topologies

    return vectorizer(handlers, **vectorizer_kwargs)


def convert_interchange(
    interchange: openff.interchange.Interchange | list[openff.interchange.Interchange],
) -> tuple[smirnoffee.ff.TensorForceField, list[smirnoffee.ff.AppliedParameters]]:
    """Convert a list of interchange objects into tensor potentials.

    Args:
        interchange: The list of (or singile) interchange objects to convert into
            tensor potentials.

    Returns:
        The tensor force field containing the parameters of each handler, and a list
        (one per interchange) of objects mapping molecule elements (e.g. bonds, angles)
        to corresponding handler parameters.

    Examples:

        >>> from openff.toolkit import ForceField, Molecule
        >>> from openff.interchange import Interchange
        >>>
        >>> force_field = ForceField("openff_unconstrained-2.0.0.offxml")
        >>> molecules = [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        >>>
        >>> interchanges = [
        ...     Interchange.from_smirnoff(force_field, molecule.to_topology())
        ...     for molecule in molecules
        ... ]
        >>>
        >>> tensor_ff, applied_parameters = convert_interchange(interchanges)
    """
    interchanges = (
        [interchange]
        if isinstance(interchange, openff.interchange.Interchange)
        else interchange
    )
    topologies = []

    handlers = collections.defaultdict(list)

    for interchange in interchanges:
        for handler in interchange.collections.values():
            handlers[handler.type].append(handler)

        topologies.append(interchange.topology)

    potentials, parameter_maps_by_handler = [], {}

    for handler_type, handlers in handlers.items():
        if sum(len(handler.potentials) for handler in handlers) == 0:
            continue

        potential, parameter_map = convert_handlers(handlers, topologies)
        potentials.append(potential)

        parameter_maps_by_handler[handler_type] = parameter_map

    parameter_maps = [
        {
            potential.type: parameter_maps_by_handler[potential.type][i]
            for potential in potentials
        }
        for i in range(len(topologies))
    ]

    force_field = smirnoffee.ff.TensorForceField(potentials)
    return force_field, parameter_maps
