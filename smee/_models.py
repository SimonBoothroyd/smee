"""Tensor representations of force field parameters applied to molecules."""

import dataclasses
import typing

import openff.interchange.components.potentials
import openff.interchange.models
import openff.units
import torch

_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians


DeviceType = typing.Literal["cpu", "cuda"]
Precision = typing.Literal["single", "double"]


def _cast(
    tensor: torch.Tensor,
    device: DeviceType | None = None,
    precision: Precision | None = None,
) -> torch.Tensor:
    """Cast a tensor to the specified device."""

    if precision is not None:
        if tensor.dtype in (torch.float32, torch.float64):
            dtype = torch.float32 if precision == "single" else torch.float64
        elif tensor.dtype in (torch.int32, torch.int64):
            dtype = torch.int32 if precision == "single" else torch.int64
        else:
            raise NotImplementedError(f"cannot cast {tensor.dtype} to {precision}")
    else:
        dtype = None

    return tensor.to(device=device, dtype=dtype)


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

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "ValenceParameterMap":
        """Cast this object to the specified device."""
        return ValenceParameterMap(
            _cast(self.particle_idxs, device, precision),
            _cast(self.assignment_matrix, device, precision),
        )


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

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "NonbondedParameterMap":
        """Cast this object to the specified device."""
        return NonbondedParameterMap(
            _cast(self.assignment_matrix, device, precision),
            _cast(self.exclusions, device, precision),
            _cast(self.exclusion_scale_idxs, device, precision),
        )


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

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "VSiteMap":
        """Cast this object to the specified device."""
        return VSiteMap(
            self.keys, self.key_to_idx, _cast(self.parameter_idxs, device, precision)
        )


@dataclasses.dataclass
class TensorConstraints:
    """A tensor representation of a set of distance constraints between pairs of
    atoms."""

    idxs: torch.Tensor
    """The indices of the atoms involved in each constraint with
    ``shape=(n_constraints, 2)``"""
    distances: torch.Tensor
    """The distance [Å] between each pair of atoms with ``shape=(n_constraints,)``"""

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "TensorConstraints":
        """Cast this object to the specified device."""
        return TensorConstraints(
            _cast(self.idxs, device, precision),
            _cast(self.distances, device, precision),
        )


@dataclasses.dataclass
class TensorTopology:
    """A tensor representation of a molecular topology that has been assigned force
    field parameters."""

    atomic_nums: torch.Tensor
    """The atomic numbers of each atom in the topology with ``shape=(n_atoms,)``"""
    formal_charges: torch.Tensor
    """The formal charge of each atom in the topology with ``shape=(n_atoms,)``"""

    bond_idxs: torch.Tensor
    """The indices of the atoms involved in each bond with ``shape=(n_bonds, 2)``"""
    bond_orders: torch.Tensor
    """The bond orders of each bond with ``shape=(n_bonds,)``"""

    parameters: dict[str, ParameterMap]
    """The parameters that have been assigned to the topology."""
    v_sites: VSiteMap | None = None
    """The v-sites that have been assigned to the topology."""

    constraints: TensorConstraints | None = None
    """Distance constraints that should be applied **during MD simulations**. These
    will not be used outside of MD simulations."""

    residue_idxs: list[int] | None = None
    """The index of the residue that each atom in the topology belongs to with
    ``length=n_atoms``."""
    residue_ids: list[str] | None = None
    """The names of the residues that each atom belongs to with ``length=n_residues``.
    """

    chain_idxs: list[int] | None = None
    """The index of the chain that each atom in the topology belongs to with
    ``length=n_atoms``."""
    chain_ids: list[str] | None = None
    """The names of the chains that each atom belongs to with ``length=n_chains``."""

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the topology."""
        return len(self.atomic_nums)

    @property
    def n_bonds(self) -> int:
        """The number of bonds in the topology."""
        return len(self.bond_idxs)

    @property
    def n_residues(self) -> int:
        """The number of residues in the topology"""
        return 0 if self.residue_ids is None else len(self.residue_ids)

    @property
    def n_chains(self) -> int:
        """The number of chains in the topology"""
        return 0 if self.chain_ids is None else len(self.chain_ids)

    @property
    def n_v_sites(self) -> int:
        """The number of v-sites in the topology."""
        return 0 if self.v_sites is None else len(self.v_sites.parameter_idxs)

    @property
    def n_particles(self) -> int:
        """The number of atoms + v-sites in the topology."""
        return self.n_atoms + self.n_v_sites

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "TensorTopology":
        """Cast this object to the specified device."""
        return TensorTopology(
            self.atomic_nums,
            self.formal_charges,
            self.bond_idxs,
            self.bond_orders,
            {k: v.to(device, precision) for k, v in self.parameters.items()},
            None if self.v_sites is None else self.v_sites.to(device, precision),
            (
                None
                if self.constraints is None
                else self.constraints.to(device, precision)
            ),
            self.residue_idxs,
            self.residue_ids,
            self.chain_ids,
        )


@dataclasses.dataclass
class TensorSystem:
    """A tensor representation of a 'full' system."""

    topologies: list[TensorTopology]
    """The topologies of the individual molecules in the system."""
    n_copies: list[int]
    """The number of copies of each topology to include in the system."""

    is_periodic: bool
    """Whether the system is periodic or not."""

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the system."""
        return sum(
            topology.n_atoms * n_copies
            for topology, n_copies in zip(self.topologies, self.n_copies, strict=True)
        )

    @property
    def n_v_sites(self) -> int:
        """The number of v-sites in the system."""
        return sum(
            topology.n_v_sites * n_copies
            for topology, n_copies in zip(self.topologies, self.n_copies, strict=True)
        )

    @property
    def n_particles(self) -> int:
        """The number of atoms + v-sites in the system."""
        return self.n_atoms + self.n_v_sites

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "TensorSystem":
        """Cast this object to the specified device."""
        return TensorSystem(
            [topology.to(device, precision) for topology in self.topologies],
            self.n_copies,
            self.is_periodic,
        )


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
    ``shape=(n_attribute_cols,)``"""
    attribute_cols: tuple[str, ...] | None = None
    """The names of each column of ``attributes``."""
    attribute_units: tuple[openff.units.Unit, ...] | None = None
    """The units of each attribute in ``attributes``."""

    exceptions: dict[tuple[int, int], int] | None = None
    """A lookup for custom cross-interaction parameters that should override any mixing
    rules.

    Each key should correspond to the indices of the two parameters whose mixing rule
    should be overridden, and each value the index of the parameter that contains the
    'pre-mixed' parameter to use instead.

    For now, all exceptions are assumed to be symmetric, i.e. if (a, b) is an exception
    then (b, a) is also an exception, and so only one of the two should be defined.

    As a note of caution, not all potentials (e.g. common valence potentials) support
    such exceptions, and these are predominantly useful for non-bonded potentials.
    """

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "TensorPotential":
        """Cast this object to the specified device."""
        return TensorPotential(
            self.type,
            self.fn,
            _cast(self.parameters, device, precision),
            self.parameter_keys,
            self.parameter_cols,
            self.parameter_units,
            (
                None
                if self.attributes is None
                else _cast(self.attributes, device, precision)
            ),
            self.attribute_cols,
            self.attribute_units,
            self.exceptions,
        )


@dataclasses.dataclass
class TensorVSites:
    """A tensor representation of a set of virtual sites parameters."""

    @classmethod
    def default_units(cls) -> dict[str, openff.units.Unit]:
        """The default units of each v-site parameter."""
        return {
            "distance": _ANGSTROM,
            "inPlaneAngle": _RADIANS,
            "outOfPlaneAngle": _RADIANS,
        }

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
        return {**TensorVSites.default_units()}

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "TensorVSites":
        """Cast this object to the specified device."""
        return TensorVSites(
            self.keys,
            [_cast(weight, device, precision) for weight in self.weights],
            _cast(self.parameters, device, precision),
        )


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

    def to(
        self, device: DeviceType | None = None, precision: Precision | None = None
    ) -> "TensorForceField":
        """Cast this object to the specified device."""
        return TensorForceField(
            [potential.to(device, precision) for potential in self.potentials],
            None if self.v_sites is None else self.v_sites.to(device, precision),
        )


__all__ = [
    "ValenceParameterMap",
    "NonbondedParameterMap",
    "ParameterMap",
    "VSiteMap",
    "TensorConstraints",
    "TensorTopology",
    "TensorSystem",
    "TensorPotential",
    "TensorVSites",
    "TensorForceField",
]
