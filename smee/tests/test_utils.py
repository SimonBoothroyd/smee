import openff.interchange.models
import openff.toolkit
import torch

import smee
from smee.utils import find_exclusions, to_upper_tri_idx


def test_find_exclusions_simple():
    molecule = openff.toolkit.Molecule()
    for i in range(6):
        molecule.add_atom(6, 0, False)
    for i in range(5):
        molecule.add_bond(i, i + 1, 1, False)

    exclusions = find_exclusions(molecule.to_topology())
    assert exclusions == {
        (0, 1): "scale_12",
        (0, 2): "scale_13",
        (0, 3): "scale_14",
        (0, 4): "scale_15",
        (1, 2): "scale_12",
        (1, 3): "scale_13",
        (1, 4): "scale_14",
        (1, 5): "scale_15",
        (2, 3): "scale_12",
        (2, 4): "scale_13",
        (2, 5): "scale_14",
        (3, 4): "scale_12",
        (3, 5): "scale_13",
        (4, 5): "scale_12",
    }


def test_find_exclusions_rings():
    molecule = openff.toolkit.Molecule()
    for i in range(8):
        molecule.add_atom(6, 0, False)

    # para substituted 6-membered ring
    molecule.add_bond(0, 1, 1, False)
    for i in range(6):
        molecule.add_bond(i + 1, (i + 1) % 6 + 1, 1, False)
    molecule.add_bond(2, 7, 1, False)

    exclusions = find_exclusions(molecule.to_topology())
    assert exclusions == {
        (0, 1): "scale_12",
        (0, 2): "scale_13",
        (0, 6): "scale_13",
        (0, 3): "scale_14",
        (0, 5): "scale_14",
        (0, 7): "scale_14",
        (0, 4): "scale_15",
        (1, 2): "scale_12",
        (1, 6): "scale_12",
        (1, 3): "scale_13",
        (1, 5): "scale_13",
        (1, 7): "scale_13",
        (1, 4): "scale_14",
        (2, 3): "scale_12",
        (2, 6): "scale_13",
        (2, 4): "scale_13",
        (2, 5): "scale_14",
        (2, 7): "scale_12",
        (3, 4): "scale_12",
        (3, 5): "scale_13",
        (3, 6): "scale_14",
        (3, 7): "scale_13",
        (4, 5): "scale_12",
        (4, 6): "scale_13",
        (4, 7): "scale_14",
        (5, 6): "scale_12",
        (5, 7): "scale_15",
        (6, 7): "scale_14",
    }


def test_find_exclusions_dimer():
    molecule = openff.toolkit.Molecule()
    for i in range(3):
        molecule.add_atom(6, 0, False)

    molecule.add_bond(0, 1, 1, False)
    molecule.add_bond(1, 2, 1, False)

    topology = openff.toolkit.Topology()
    topology.add_molecule(molecule)
    topology.add_molecule(molecule)

    exclusions = find_exclusions(topology)
    assert exclusions == {
        (0, 1): "scale_12",
        (0, 2): "scale_13",
        (1, 2): "scale_12",
        (3, 4): "scale_12",
        (3, 5): "scale_13",
        (4, 5): "scale_12",
    }


def test_find_exclusions_v_sites():
    molecule = openff.toolkit.Molecule()
    for i in range(4):
        molecule.add_atom(6, 0, False)
    for i in range(3):
        molecule.add_bond(i, i + 1, 1, False)

    v_site_keys = [
        openff.interchange.models.VirtualSiteKey(
            orientation_atom_indices=(0, 1, 2),
            type="MonovalentLonePair",
            match="once",
            name="EP",
        ),
        openff.interchange.models.VirtualSiteKey(
            orientation_atom_indices=(3, 2, 1),
            type="MonovalentLonePair",
            match="once",
            name="EP",
        ),
    ]

    v_sites = smee.VSiteMap(
        keys=v_site_keys,
        key_to_idx={v_site_keys[0]: 4, v_site_keys[1]: 5},
        parameter_idxs=torch.zeros((2, 1)),
    )

    exclusions = find_exclusions(molecule.to_topology(), v_sites)
    assert exclusions == {
        (0, 1): "scale_12",
        (0, 2): "scale_13",
        (0, 3): "scale_14",
        (0, 5): "scale_14",
        (1, 2): "scale_12",
        (1, 3): "scale_13",
        (1, 5): "scale_13",
        (2, 3): "scale_12",
        (2, 5): "scale_12",
        (4, 0): "scale_12",
        (4, 1): "scale_12",
        (4, 2): "scale_13",
        (4, 3): "scale_14",
        (5, 3): "scale_12",
    }


def test_ones_like():
    expected_size = (4, 5)
    expected_type = torch.float16

    other = torch.tensor([1, 2, 3], dtype=expected_type, device="cpu")
    tensor = smee.utils.ones_like(expected_size, other)

    assert tensor.dtype == expected_type
    assert tensor.shape == expected_size
    assert torch.allclose(tensor, torch.tensor(1.0, dtype=expected_type))


def test_zeros_like():
    expected_size = (4, 5)
    expected_type = torch.float16

    other = torch.tensor([1, 2, 3], dtype=expected_type, device="cpu")
    tensor = smee.utils.zeros_like(expected_size, other)

    assert tensor.dtype == expected_type
    assert tensor.shape == expected_size
    assert torch.allclose(tensor, torch.tensor(0.0, dtype=expected_type))


def test_tensor_like():
    expected_type = torch.float16
    expected_data = [[3.0], [2.0], [1.0]]

    other = torch.tensor([], dtype=expected_type, device="cpu")
    tensor = smee.utils.tensor_like(expected_data, other)

    assert tensor.dtype == expected_type
    assert torch.allclose(tensor, torch.tensor(expected_data, dtype=expected_type))


def test_to_upper_tri_idx():
    i = torch.tensor([0, 1, 0])
    j = torch.tensor([1, 2, 2])

    idxs = to_upper_tri_idx(i, j, 3)

    expected_idxs = torch.tensor([0, 2, 1])
    assert idxs.shape == expected_idxs.shape
    assert torch.allclose(idxs, expected_idxs)
