"""Tensor representations of force field parameters applied to molecules."""
import dataclasses
import typing

import openff.interchange.components.potentials
import openff.interchange.models
import openff.units
import torch

_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians


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
class TensorConstraints:
    """A tensor representation of a set of distance constraints between pairs of
    atoms."""

    idxs: torch.Tensor
    """The indices of the atoms involved in each constraint with
    ``shape=(n_constraints, 2)``"""
    distances: torch.Tensor
    """The distance [Å] between each pair of atoms with ``shape=(n_constraints,)``"""


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

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the topology."""
        return len(self.atomic_nums)

    @property
    def n_bonds(self) -> int:
        """The number of bonds in the topology."""
        return len(self.bond_idxs)


@dataclasses.dataclass
class TensorSystem:
    """A tensor representation of a 'full' system."""

    topologies: list[TensorTopology]
    """The topologies of the individual molecules in the system."""
    n_copies: list[int]
    """The number of copies of each topology to include in the system."""

    is_periodic: bool
    """Whether the system is periodic or not."""


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
    attribute_units: tuple[openff.units.Unit, ...] = None
    """The units of each attribute in ``attributes``."""


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
