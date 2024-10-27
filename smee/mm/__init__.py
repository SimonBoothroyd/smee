"""Compute differentiable ensemble averages using OpenMM and SMEE."""

from smee.mm._config import GenerateCoordsConfig, MinimizationConfig, SimulationConfig
from smee.mm._fe import generate_dg_solv_data
from smee.mm._mm import generate_system_coords, simulate
from smee.mm._ops import (
    NotEnoughSamplesError,
    compute_dg_solv,
    compute_ensemble_averages,
    reweight_dg_solv,
    reweight_ensemble_averages,
)
from smee.mm._reporters import TensorReporter, tensor_reporter, unpack_frames

__all__ = [
    "compute_dg_solv",
    "compute_ensemble_averages",
    "generate_dg_solv_data",
    "generate_system_coords",
    "reweight_dg_solv",
    "reweight_ensemble_averages",
    "simulate",
    "GenerateCoordsConfig",
    "MinimizationConfig",
    "NotEnoughSamplesError",
    "SimulationConfig",
    "TensorReporter",
    "tensor_reporter",
    "unpack_frames",
]
