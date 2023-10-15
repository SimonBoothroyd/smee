import openff.interchange.models
import torch

import smee.tests.utils


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
