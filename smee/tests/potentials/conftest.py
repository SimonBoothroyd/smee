import openff.interchange.models
import openff.units
import pytest
import torch

import smee.potentials
import smee.tests.utils
import smee.utils


@pytest.fixture()
def mock_lj_potential() -> smee.TensorPotential:
    return smee.TensorPotential(
        type="vdW",
        fn="LJ",
        parameters=torch.tensor([[0.1, 1.1], [0.2, 2.1], [0.3, 3.1]]),
        parameter_keys=[
            openff.interchange.models.PotentialKey(id="[#1:1]"),
            openff.interchange.models.PotentialKey(id="[#6:1]"),
            openff.interchange.models.PotentialKey(id="[#8:1]"),
        ],
        parameter_cols=("epsilon", "sigma"),
        parameter_units=(
            openff.units.unit.kilojoule_per_mole,
            openff.units.unit.angstrom,
        ),
        attributes=torch.tensor([0.0, 0.0, 0.5, 1.0, 9.0, 2.0]),
        attribute_cols=(
            "scale_12",
            "scale_13",
            "scale_14",
            "scale_15",
            "cutoff",
            "switch_width",
        ),
        attribute_units=(
            openff.units.unit.dimensionless,
            openff.units.unit.dimensionless,
            openff.units.unit.dimensionless,
            openff.units.unit.dimensionless,
            openff.units.unit.angstrom,
            openff.units.unit.angstrom,
        ),
    )


@pytest.fixture()
def mock_methane_top() -> smee.TensorTopology:
    methane_top = smee.tests.utils.topology_from_smiles("C")
    methane_top.parameters = {
        "vdW": smee.NonbondedParameterMap(
            assignment_matrix=torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ).to_sparse(),
            exclusions=torch.tensor(
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [1, 2],
                    [1, 3],
                    [1, 4],
                    [2, 3],
                    [2, 4],
                    [3, 4],
                ]
            ),
            exclusion_scale_idxs=torch.tensor([[0] * 4 + [1] * 6]),
        )
    }
    return methane_top


@pytest.fixture()
def mock_water_top() -> smee.TensorTopology:
    methane_top = smee.tests.utils.topology_from_smiles("O")
    methane_top.parameters = {
        "vdW": smee.NonbondedParameterMap(
            assignment_matrix=torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ).to_sparse(),
            exclusions=torch.tensor([[0, 1], [0, 2], [1, 2]]),
            exclusion_scale_idxs=torch.tensor([[0], [0], [1]]),
        )
    }
    return methane_top
