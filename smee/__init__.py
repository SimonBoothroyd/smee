"""Differentiably evaluate energies of molecules using SMIRNOFF force fields"""

import importlib.metadata

from ._constants import CUTOFF_ATTRIBUTE, SWITCH_ATTRIBUTE, EnergyFn, PotentialType
from ._models import (
    NonbondedParameterMap,
    ParameterMap,
    TensorConstraints,
    TensorForceField,
    TensorPotential,
    TensorSystem,
    TensorTopology,
    TensorVSites,
    ValenceParameterMap,
    VSiteMap,
)
from .geometry import add_v_site_coords, compute_v_site_coords
from .potentials import compute_energy, compute_energy_potential

try:
    __version__ = importlib.metadata.version("smee")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "CUTOFF_ATTRIBUTE",
    "SWITCH_ATTRIBUTE",
    "EnergyFn",
    "PotentialType",
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
    "__version__",
    "add_v_site_coords",
    "compute_v_site_coords",
    "compute_energy",
    "compute_energy_potential",
]
