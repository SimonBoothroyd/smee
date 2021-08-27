import sys

import pytest
import torch
from openff.toolkit.topology import Molecule
from simtk import unit

from smirnoffee.geometry.internal import detect_internal_coordinates
from smirnoffee.tests.utilities.geometric import validate_internal_coordinates


def test_detect_internal_coordinates():

    conformer = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bond_indices = torch.tensor([[0, 1]])

    internal_coordinates = detect_internal_coordinates(conformer, bond_indices)

    assert {*internal_coordinates} == {"distances"}

    assert bond_indices.shape == internal_coordinates["distances"].shape
    assert torch.allclose(bond_indices, internal_coordinates["distances"])


@pytest.mark.parametrize(
    "smiles",
    ["C", "CC", "C#C", "CC#C", "CC#CC", "CCC#CCC", "CC#CC#CC", "CC(=O)c1ccc(cc1)C#N"],
)
def test_cartesian_to_internal(smiles):

    pytest.importorskip("geometric")

    if smiles == "CC#CC#CC" and (sys.platform == "linux" or sys.platform == "linux2"):

        pytest.skip(
            "This test currently fails on the CI as geomeTRIC fails to correctly "
            "detect all of the bonds."
        )

    molecule: Molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)

    conformer = torch.from_numpy(molecule.conformers[0].value_in_unit(unit.angstrom))

    validate_internal_coordinates(molecule, conformer, verbose=True)
