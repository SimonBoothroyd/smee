import dataclasses
import importlib
import inspect
import typing

import openff.interchange.components.potentials
import openff.interchange.models
import openff.interchange.smirnoff._base
import openff.interchange.smirnoff._virtual_sites
import openff.toolkit
import openff.units
import torch

import smee.geometry

_VSiteParameters = (
    openff.interchange.smirnoff._virtual_sites.SMIRNOFFVirtualSiteCollection
)

_CONVERTERS = {}
_DEFAULT_UNITS = {}

_IGNORED_HANDLERS = {"Constraints"}

_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians

_V_SITE_DEFAULT_UNITS = {
    "distance": _ANGSTROM,
    "inPlaneAngle": _RADIANS,
    "outOfPlaneAngle": _RADIANS,
}
_V_SITE_DEFAULT_VALUES = {
    "distance": 0.0 * _ANGSTROM,
    "inPlaneAngle": torch.pi * _RADIANS,
    "outOfPlaneAngle": 0.0 * _RADIANS,
}


@dataclasses.dataclass
class ValenceParameterMap:
    """A map between atom indices part of a particular valence interaction (e.g.
    torsion indices) and the corresponding parameter in a ``TensorPotential``"""

    particle_idxs: torch.Tensor
    """The indices of the particles (e.g. atoms or virtual sites) involved in an
    interaction with ``shape=(n_interactions, n_cols)``. For a bond ``n_cols=2``,
    for angles ``n_cols=3`` etc.
    """
    assignment_matrix: torch.sparse.Tensor
    """A sparse tensor that yields the assigned parameters when multiplied with the
    corresponding handler parameters, with ``shape=(n_interacting, n_parameters)``.
    """


@dataclasses.dataclass
class NonbondedParameterMap:
    """A map between atom indices part of a particular valence interaction (e.g.
    torsion indices) and the corresponding parameter in a ``TensorPotential``"""

    assignment_matrix: torch.sparse.Tensor
    """A sparse tensor that yields the parameters assigned to each particle in the
    system when multiplied with the corresponding handler parameters, with
    ``shape=(n_particles, n_parameters)``.
    """

    exclusions: torch.Tensor
    """Indices of pairs of particles (i.e. atoms or virtual sites) that should
    have their interactions scaled by some factor with ``shape=(n_exclusions, 2)``.
    """
    exclusion_scale_idxs: torch.Tensor
    """Indices into the tensor of handler attributes defining the 1-n scaling factors
    with ``shape=(n_exclusions, 1)``.
    """


ParameterMap = ValenceParameterMap | NonbondedParameterMap


@dataclasses.dataclass
class VSiteMap:
    """A map between virtual sites that have been added to a topology and their
    corresponding 'parameters' used to position them."""

    keys: list[openff.interchange.models.VirtualSiteKey]
    """The keys used to identify each v-site."""
    key_to_idx: dict[openff.interchange.models.VirtualSiteKey, int]
    """A map between the unique keys associated with each v-site and their index in
    the topology"""

    parameter_idxs: torch.Tensor
    """The indices of the corresponding v-site parameters with ``shape=(n_v_sites, 1)``
    """


@dataclasses.dataclass
class TensorTopology:
    """A tensor representation of a molecular topology that has been assigned force
    field parameters."""

    n_atoms: int
    """The number of atoms in the topology."""

    parameters: dict[str, ParameterMap]
    """The parameters that have been assigned to the topology."""
    v_sites: VSiteMap | None = None
    """The v-sites that have been assigned to the topology."""


@dataclasses.dataclass
class TensorPotential:
    """A tensor representation of a valence SMIRNOFF parameter handler"""

    type: str
    """The type of handler associated with these parameters"""
    fn: str
    """The associated potential energy function"""

    parameters: torch.Tensor
    """The values of the parameters with ``shape=(n_parameters, n_parameter_cols)``"""
    parameter_keys: list[openff.interchange.models.PotentialKey]
    """Unique keys associated with each parameter with ``length=(n_parameters)``"""
    parameter_cols: tuple[str, ...]
    """The names of each column of ``parameters``."""
    parameter_units: tuple[openff.units.Unit, ...]
    """The units of each parameter in ``parameters``."""

    attributes: torch.Tensor | None = None
    """The attributes defined on a handler such as 1-4 scaling factors with
    ``shape=(n_attributes, n_attribute_cols)``"""
    attribute_cols: tuple[str, ...] | None = None
    """The names of each column of ``attributes``."""
    attribute_units: tuple[openff.units.Unit, ...] = None
    """The units of each attribute in ``attributes``."""


@dataclasses.dataclass
class TensorVSites:
    """A tensor representation of a set of virtual sites parameters."""

    keys: typing.List[openff.interchange.models.VirtualSiteKey]
    """The unique keys associated with each v-site with ``length=(n_v_sites)``"""
    weights: list[torch.Tensor]
    """A matrix of weights that, when applied to the 'orientiational' atoms, yields a
    basis that the virtual site coordinate parameters can be projected onto with
    ``shape=(n_v_sites, 3, 3)``"""
    parameters: torch.Tensor
    """The distance, in-plane and out-of-plane angles with ``shape=(n_v_sites, 3)``"""

    @property
    def parameter_units(self) -> dict[str, openff.units.Unit]:
        """The units of each v-site parameter."""
        return {**_V_SITE_DEFAULT_UNITS}


@dataclasses.dataclass
class TensorForceField:
    """A tensor representation of a SMIRNOFF force field."""

    potentials: list[TensorPotential]
    """The terms and associated parameters of the potential energy function."""

    v_sites: TensorVSites | None = None
    """Parameters used to add and define the coords of v-sites in the system. The
    non-bonded parameters of any v-sites are stored in relevant potentials, e.g. 'vdW'
    or 'Electrostatics'.
    """

    @property
    def potentials_by_type(self) -> dict[str, TensorPotential]:
        potentials = {potential.type: potential for potential in self.potentials}
        assert len(potentials) == len(self.potentials), "duplicate potentials found"

        return potentials


def parameter_converter(type_: str, default_units: dict[str, openff.units.Unit]):
    """A decorator used to flag a function as being able to convert a parameter handlers
    parameters into tensors.

    Args:
        type_: The type of parameter handler that the decorated function can convert.
        default_units: The default units of each parameter in the handler.
    """

    def parameter_converter_inner(func):
        if type_ in _CONVERTERS:
            raise KeyError(f"A {type_} converter is already registered.")

        _CONVERTERS[type_] = func
        _DEFAULT_UNITS[type_] = default_units

        return func

    return parameter_converter_inner


def _get_value(
    potential: openff.interchange.components.potentials.Potential,
    parameter: str,
    default_units: dict[str, openff.units.Unit],
    default_value: openff.units.Quantity | None = None,
) -> float:
    """Returns the value of a parameter in its default units"""
    default_units = default_units[parameter]
    value = potential.parameters[parameter]
    return (value if value is not None else default_value).m_as(default_units)


def _handlers_to_potential(
    handlers: list[openff.interchange.smirnoff._base.SMIRNOFFCollection],
    handler_type: str,
    parameter_cols: tuple[str, ...],
    attribute_cols: tuple[str, ...] | None,
) -> TensorPotential:
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
                _get_value(
                    parameters_by_key[parameter_key],
                    column,
                    _DEFAULT_UNITS[handler_type],
                )
                for column in parameter_cols
            ]
            for parameter_key in parameter_keys
        ]
    )

    attributes = None

    if attribute_cols is not None:
        attributes_by_column = {
            column: openff.units.Quantity(getattr(handler, column))
            for column in attribute_cols
            for handler in handlers
        }
        attributes_by_column = {
            k: v.m_as(_DEFAULT_UNITS[handler_type][k])
            for k, v in attributes_by_column.items()
        }
        attributes = torch.tensor(
            [attributes_by_column[column] for column in attribute_cols]
        )

    potential = TensorPotential(
        type=handler_type,
        fn=potential_fn,
        parameters=parameters,
        parameter_keys=parameter_keys,
        parameter_cols=parameter_cols,
        parameter_units=tuple(
            _DEFAULT_UNITS[handler_type][column] for column in parameter_cols
        ),
        attributes=attributes,
        attribute_cols=attribute_cols,
        attribute_units=None
        if attribute_cols is None
        else tuple(_DEFAULT_UNITS[handler_type][column] for column in attribute_cols),
    )
    return potential


def _convert_v_sites(
    handlers: list[_VSiteParameters],
    topologies: list[openff.toolkit.Topology],
) -> tuple[TensorVSites, list[VSiteMap | None]]:
    handler_types = {handler.type for handler in handlers}
    assert handler_types == {"VirtualSites"}, "invalid handler types found"

    assert all(
        isinstance(key, openff.interchange.models.VirtualSiteKey)
        for handler in handlers
        if handler is not None
        for key in handler.key_map
    ), "only v-site keys expected"

    # account for the fact the interchange doesn't track v-site parameter types
    parameter_key_to_type = {
        parameter_key: v_site_key.type
        for handler in handlers
        if handler is not None
        for v_site_key, parameter_key in handler.key_map.items()
    }
    parameters_by_key = {
        parameter_key: parameter
        for handler in handlers
        if handler is not None
        for parameter_key, parameter in handler.potentials.items()
    }
    parameter_keys = [*parameters_by_key]
    parameters = [
        [
            _get_value(
                parameters_by_key[parameter_key],
                column,
                _V_SITE_DEFAULT_UNITS,
                _V_SITE_DEFAULT_VALUES[column],
            )
            for column in ("distance", "inPlaneAngle", "outOfPlaneAngle")
        ]
        for parameter_key in parameter_keys
    ]

    v_site_frames = [
        smee.geometry.V_SITE_TYPE_TO_FRAME[parameter_key_to_type[parameter_key]]
        for parameter_key in parameter_keys
    ]
    v_sites = TensorVSites(
        keys=parameter_keys, weights=v_site_frames, parameters=torch.tensor(parameters)
    )

    v_site_maps = []

    for topology, handler in zip(topologies, handlers):
        if handler is None:
            v_site_maps.append(None)
            continue

        v_site_keys = sorted(
            handler.key_map,
            key=lambda k: handler.virtual_site_key_topology_index_map[k],
        )
        parameter_idxs = [
            parameter_keys.index(handler.key_map[v_site_key])
            for v_site_key in v_site_keys
        ]

        # check v-sites aren't interleaved between atoms, which isn't supported yet
        # TODO: interchange (#812) means we can't safely do this check currently
        # topology_indices = sorted(
        #     idx - topology.n_atoms
        #     for idx in handler.virtual_site_key_topology_index_map.values()
        # )
        # assert topology_indices == list(range(len(topology_indices)))

        v_site_map = VSiteMap(
            keys=v_site_keys,
            key_to_idx={key: i + topology.n_atoms for i, key in enumerate(v_site_keys)},
            parameter_idxs=torch.tensor(parameter_idxs),
        )
        v_site_maps.append(v_site_map)

    return v_sites, v_site_maps


def convert_handlers(
    handlers: list[openff.interchange.smirnoff._base.SMIRNOFFCollection],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[VSiteMap | None] | None = None,
):
    """Convert a set of SMIRNOFF parameter handlers into a set of tensor potentials.

    Args:
        handlers: The SMIRNOFF parameter handler collections for a set of interchange
            objects to convert.
        topologies: The topologies associated with each interchange object.
        v_site_maps: The v-site maps associated with each interchange object.

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

    importlib.import_module("smee.ff.nonbonded")
    importlib.import_module("smee.ff.valence")

    if handler_type not in _CONVERTERS:
        raise NotImplementedError(f"{handler_type} handlers is not yet supported.")

    converter = _CONVERTERS[handler_type]
    converter_spec = inspect.signature(converter)

    converter_kwargs = {}

    if "topologies" in converter_spec.parameters:
        converter_kwargs["topologies"] = topologies
    if "v_site_maps" in converter_spec.parameters:
        assert v_site_maps is not None, "v-site maps must be provided"
        converter_kwargs["v_site_maps"] = v_site_maps

    return converter(handlers, **converter_kwargs)


def convert_interchange(
    interchange: openff.interchange.Interchange | list[openff.interchange.Interchange],
) -> tuple[TensorForceField, list[TensorTopology]]:
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
        >>> tensor_ff, tensor_topologies = convert_interchange(interchanges)
    """
    interchanges = (
        [interchange]
        if isinstance(interchange, openff.interchange.Interchange)
        else interchange
    )
    topologies = []

    handler_types = {
        handler_type
        for interchange in interchanges
        for handler_type in interchange.collections
        if handler_type not in _IGNORED_HANDLERS
    }
    handlers_by_type = {handler_type: [] for handler_type in sorted(handler_types)}

    for interchange in interchanges:
        for handler_type in handlers_by_type:
            handler = (
                None
                if handler_type not in interchange.collections
                else interchange.collections[handler_type]
            )
            handlers_by_type[handler_type].append(handler)

        topologies.append(interchange.topology)

    v_sites, v_site_maps = None, [None] * len(topologies)

    if "VirtualSites" in handlers_by_type:
        v_sites, v_site_maps = _convert_v_sites(
            handlers_by_type["VirtualSites"], topologies
        )
        handlers_by_type.pop("VirtualSites")

    potentials, parameter_maps_by_handler = [], {}

    for handler_type, handlers in handlers_by_type.items():
        if (
            sum(len(handler.potentials) for handler in handlers if handler is not None)
            == 0
        ):
            continue

        potential, parameter_map = convert_handlers(handlers, topologies, v_site_maps)
        potentials.append(potential)

        parameter_maps_by_handler[handler_type] = parameter_map

    tensor_topologies = [
        TensorTopology(
            n_atoms=topologies[i].n_atoms,
            parameters={
                potential.type: parameter_maps_by_handler[potential.type][i]
                for potential in potentials
            },
            v_sites=v_site_maps[i],
        )
        for i in range(len(topologies))
    ]

    tensor_force_field = TensorForceField(potentials, v_sites)
    return tensor_force_field, tensor_topologies
