"""Compute differentiable ensemble averages using OpenMM and SMEE."""

from smee.mm._config import GenerateCoordsConfig, MinimizationConfig, SimulationConfig
from smee.mm._converters import convert_to_openmm_system, convert_to_openmm_topology
from smee.mm._mm import generate_system_coords, simulate
from smee.mm._ops import (
    GRADIENT_DELTA,
    GRADIENT_EXCLUDED_ATTRIBUTES,
    compute_ensemble_averages,
)

__all__ = [
    "GRADIENT_EXCLUDED_ATTRIBUTES",
    "GRADIENT_DELTA",
    "compute_ensemble_averages",
    "convert_to_openmm_system",
    "convert_to_openmm_topology",
    "generate_system_coords",
    "simulate",
    "GenerateCoordsConfig",
    "MinimizationConfig",
    "SimulationConfig",
]