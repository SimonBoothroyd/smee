import openff.interchange.models
import pytest
import torch

import smee.tests.utils
from smee._models import _cast


@pytest.mark.parametrize(
    "tensor, precision, expected_device, expected_dtype",
    [
        (torch.zeros(2, dtype=torch.float32), "single", "cpu", torch.float32),
        (torch.zeros(2, dtype=torch.float64), "single", "cpu", torch.float32),
        (torch.zeros(2, dtype=torch.float32), "double", "cpu", torch.float64),
        (torch.zeros(2, dtype=torch.float64), "double", "cpu", torch.float64),
        (torch.zeros(2, dtype=torch.int32), "single", "cpu", torch.int32),
        (torch.zeros(2, dtype=torch.int64), "single", "cpu", torch.int32),
        (torch.zeros(2, dtype=torch.int32), "double", "cpu", torch.int64),
        (torch.zeros(2, dtype=torch.int64), "double", "cpu", torch.int64),
    ],
)
def test_cast(tensor, precision, expected_device, expected_dtype):
    output = _cast(tensor, precision=precision)

    assert output.shape == tensor.shape
    assert output.device.type == expected_device
    assert output.dtype == expected_dtype


def _add_v_sites(topology: smee.TensorTopology):
    v_site_key = openff.interchange.models.VirtualSiteKey(
        orientation_atom_indices=(0,),
        type="BondCharge",
        name="EP",
        match="once",
    )
    topology.v_sites = smee.VSiteMap(
        keys=[v_site_key],
        key_to_idx={v_site_key: 6},
        parameter_idxs=torch.tensor([[0]]),
    )


class TestTensorTopology:
    def test_n_atoms(self):
        topology = smee.tests.utils.topology_from_smiles("CO")

        expected_n_atoms = 6
        assert topology.n_atoms == expected_n_atoms

    def test_n_bonds(self):
        topology = smee.tests.utils.topology_from_smiles("CO")

        expected_n_bonds = 5
        assert topology.n_bonds == expected_n_bonds

    def test_n_residues(self):
        topology = smee.tests.utils.topology_from_smiles("[Ar]")
        topology.residue_ids = None
        topology.residue_idxs = None
        assert topology.n_residues == 0

        topology.residue_ids = ["Ar"]
        topology.residue_idxs = [0]
        assert topology.n_residues == 1

    def test_n_chains(self) -> int:
        topology = smee.tests.utils.topology_from_smiles("[Ar]")
        topology.residue_ids = [0]
        topology.residue_idxs = ["UNK"]
        topology.chain_idxs = [0]
        topology.chain_ids = ["A"]
        assert topology.n_chains == 1

    def test_n_v_sites(self):
        topology = smee.tests.utils.topology_from_smiles("CO")

        expected_n_v_sites = 0
        assert topology.n_v_sites == expected_n_v_sites

        _add_v_sites(topology)

        expected_n_v_sites = 1
        assert topology.n_v_sites == expected_n_v_sites

    def test_n_particles(self):
        topology = smee.tests.utils.topology_from_smiles("CO")
        _add_v_sites(topology)

        expected_n_particles = 7
        assert topology.n_particles == expected_n_particles


class TestTensorSystem:
    def test_n_atoms(self):
        system = smee.TensorSystem(
            topologies=[
                smee.tests.utils.topology_from_smiles("CO"),
                smee.tests.utils.topology_from_smiles("O"),
            ],
            n_copies=[2, 5],
            is_periodic=True,
        )

        expected_n_atoms = 6 * 2 + 3 * 5
        assert system.n_atoms == expected_n_atoms

    def test_n_v_sites(self):
        system = smee.TensorSystem(
            topologies=[
                smee.tests.utils.topology_from_smiles("CO"),
                smee.tests.utils.topology_from_smiles("O"),
            ],
            n_copies=[2, 5],
            is_periodic=True,
        )

        expected_n_v_sites = 0 * 2 + 0 * 5
        assert system.n_v_sites == expected_n_v_sites

        _add_v_sites(system.topologies[0])

        expected_n_v_sites = 1 * 2 + 0 * 5
        assert system.n_v_sites == expected_n_v_sites

    def test_n_particles(self):
        system = smee.TensorSystem(
            topologies=[
                smee.tests.utils.topology_from_smiles("CO"),
                smee.tests.utils.topology_from_smiles("O"),
            ],
            n_copies=[2, 5],
            is_periodic=True,
        )
        _add_v_sites(system.topologies[0])

        expected_n_particles = (6 + 1) * 2 + 3 * 5
        assert system.n_particles == expected_n_particles
