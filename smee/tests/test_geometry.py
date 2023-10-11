import copy

import numpy
import openff.interchange.models
import openff.toolkit
import openmm
import pytest
import torch
import torch.autograd.functional
from openff.units import unit

import smee
import smee.converters
from smee.geometry import (
    V_SITE_TYPE_TO_FRAME,
    _build_v_site_coord_frames,
    _convert_v_site_coords,
    add_v_site_coords,
    compute_angles,
    compute_bond_vectors,
    compute_dihedrals,
    compute_v_site_coords,
)


@pytest.fixture
def v_site_force_field() -> openff.toolkit.ForceField:
    force_field = openff.toolkit.ForceField()

    force_field.get_parameter_handler("Electrostatics")

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[*:1]",
            "epsilon": 0.0 * unit.kilojoules / unit.mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )

    charge_handler = force_field.get_parameter_handler("LibraryCharges")
    charge_handler.add_parameter(
        {"smirks": "[*:1]", "charge1": 0.0 * openff.units.unit.e}
    )

    vsite_handler = force_field.get_parameter_handler("VirtualSites")

    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[H][#6:2]([H])=[#8:1]",
            "name": "EP",
            "type": "BondCharge",
            "distance": 7.0 * unit.angstrom,
            "match": "all_permutations",
            "charge_increment1": 0.2 * unit.elementary_charge,
            "charge_increment2": 0.1 * unit.elementary_charge,
            "sigma": 1.0 * unit.angstrom,
            "epsilon": 2.0 / 4.184 * unit.kilocalorie_per_mole,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#8:1]=[#6X3:2](-[#17])-[#1:3]",
            "name": "EP",
            "type": "MonovalentLonePair",
            "distance": 1.234 * unit.angstrom,
            "outOfPlaneAngle": 25.67 * unit.degrees,
            "inPlaneAngle": 134.0 * unit.degrees,
            "match": "all_permutations",
            "charge_increment1": 0.0 * unit.elementary_charge,
            "charge_increment2": 1.0552 * 0.5 * unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
            "sigma": 0.0 * unit.nanometers,
            "epsilon": 0.5 * unit.kilojoules_per_mole,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
            "name": "EP",
            "type": "DivalentLonePair",
            "distance": -3.21 * unit.nanometers,
            "outOfPlaneAngle": 37.43 * unit.degrees,
            "match": "all_permutations",
            "charge_increment1": 0.0 * unit.elementary_charge,
            "charge_increment2": 1.0552 * 0.5 * unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
            "sigma": 1.0 * unit.angstrom,
            "epsilon": 0.5 * unit.kilojoules_per_mole,
        }
    )
    vsite_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#1:2][#7:1]([#1:3])[#1:4]",
            "name": "EP",
            "type": "TrivalentLonePair",
            "distance": 0.5 * unit.nanometers,
            "match": "once",
            "charge_increment1": 0.2 * unit.elementary_charge,
            "charge_increment2": 0.0 * unit.elementary_charge,
            "charge_increment3": 0.0 * unit.elementary_charge,
            "charge_increment4": 0.0 * unit.elementary_charge,
            "sigma": 1.0 * unit.angstrom,
            "epsilon": 0.5 * unit.kilojoules_per_mole,
        }
    )
    return force_field


def _apply_v_site_force_field(
    molecule: openff.toolkit.Molecule,
    conformer: torch.Tensor,
    force_field: openff.toolkit.ForceField,
) -> openff.interchange.Interchange:
    force_field = copy.deepcopy(force_field)

    # newer interchange versions map to specific openmm v-site types. this requires
    # information about the geometry of the molecule / topology, which it looks up
    # directly from equilibrium bond lengths and constraint distances. this seems
    # like it could be problematic?
    constraint_handler = force_field.get_parameter_handler("Constraints")
    found_smirks = set()

    for bond in molecule.bonds:
        i, j = bond.atom1_index, bond.atom2_index

        molecule.properties["atom_map"] = {i: 0, j: 1}
        smirks = molecule.to_smiles(mapped=True)

        if smirks in found_smirks:
            continue

        found_smirks.add(smirks)

        distance = torch.norm(conformer[i, :] - conformer[j, :])

        constraint_handler.add_parameter(
            {"smirks": smirks, "distance": distance * unit.angstrom}
        )

    return openff.interchange.Interchange.from_smirnoff(
        force_field, molecule.to_topology()
    )


def compute_openmm_v_sites(
    conformer: torch.Tensor, interchange: openff.interchange.Interchange
) -> torch.Tensor:
    conformer = conformer.numpy()

    n_v_sites = len(interchange.collections["VirtualSites"].key_map)

    conformer = (
        numpy.vstack([conformer, numpy.zeros((n_v_sites, 3))]) * openmm.unit.angstrom
    )

    openmm_system = interchange.to_openmm()
    openmm_context = openmm.Context(
        openmm_system,
        openmm.VerletIntegrator(0.1),
        openmm.Platform.getPlatformByName("Reference"),
    )
    openmm_context.setPositions(conformer)
    openmm_context.computeVirtualSites()
    conformer_with_v_sites = openmm_context.getState(getPositions=True).getPositions(
        asNumpy=True
    )

    v_site_coords = conformer_with_v_sites[(len(conformer) - n_v_sites) :, :]
    return torch.tensor(v_site_coords.value_in_unit(openmm.unit.angstrom))


@pytest.mark.parametrize(
    "geometry_function", [compute_angles, compute_dihedrals, compute_bond_vectors]
)
def test_compute_geometry_no_atoms(geometry_function):
    valence_terms = geometry_function(torch.tensor([]), torch.tensor([]))

    if not isinstance(valence_terms, tuple):
        valence_terms = (valence_terms,)

    assert all(term.shape == torch.Size([0]) for term in valence_terms)


def test_compute_bond_vectors():
    conformer = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    atom_indices = torch.tensor([[2, 0], [0, 1]])

    bond_vectors, bond_norms = compute_bond_vectors(conformer, atom_indices)

    assert bond_vectors.shape == (2, 3)
    assert bond_norms.shape == (2,)

    assert torch.allclose(
        bond_vectors, torch.tensor([[0.0, -3.0, 0.0], [2.0, 0.0, 0.0]])
    )
    assert torch.allclose(bond_norms, torch.tensor([3.0, 2.0]))


def test_compute_bond_vectors_batched():
    conformer = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ]
    )

    atom_indices = torch.tensor([[2, 0], [0, 1]])

    bond_vectors, bond_norms = compute_bond_vectors(conformer, atom_indices)

    assert bond_vectors.shape == (2, 2, 3)
    assert bond_norms.shape == (2, 2)

    assert torch.allclose(
        bond_vectors,
        torch.tensor(
            [
                [[-4.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[-9.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ]
        ),
    )
    assert torch.allclose(bond_norms, torch.tensor([[4.0, 2.0], [9.0, 3.0]]))


def test_compute_angles():
    conformer = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    atom_indices = torch.tensor([[0, 1, 2], [1, 0, 2], [0, 1, 3]])

    angles = compute_angles(conformer, atom_indices)
    assert angles.shape == (3,)

    assert torch.allclose(angles, torch.tensor([numpy.pi / 2, numpy.pi / 4, numpy.pi]))

    # Make sure there are no singularities in the gradients.
    gradients = torch.autograd.functional.jacobian(
        lambda x: compute_angles(x, atom_indices), conformer
    )
    assert not torch.isnan(gradients).any() and not torch.isinf(gradients).any()


def test_compute_angles_batched():
    conformer = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]],
            [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    atom_indices = torch.tensor([[0, 1, 2]])

    angles = compute_angles(conformer, atom_indices)
    assert angles.shape == (2, 1)

    assert torch.allclose(angles, torch.tensor([[numpy.pi / 4], [numpy.pi / 2]]))


@pytest.mark.parametrize("phi_sign", [-1.0, 1.0])
def test_compute_dihedrals(phi_sign):
    conformer = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, phi_sign]]
    )
    atom_indices = torch.tensor([[0, 1, 2, 3]])

    dihedrals = compute_dihedrals(conformer, atom_indices)
    assert dihedrals.shape == (1,)

    assert torch.allclose(dihedrals, torch.tensor([phi_sign * numpy.pi / 4.0]))

    # Make sure there are no singularities in the gradients.
    gradients = torch.autograd.functional.jacobian(
        lambda x: compute_dihedrals(x, atom_indices), conformer
    )
    assert not torch.isnan(gradients).any() and not torch.isinf(gradients).any()


def test_compute_dihedrals_batched():
    conformer = torch.tensor(
        [
            [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]],
        ]
    )
    atom_indices = torch.tensor([[0, 1, 2, 3]])

    dihedrals = compute_dihedrals(conformer, atom_indices)
    assert dihedrals.shape == (2, 1)

    assert torch.allclose(dihedrals, torch.tensor([[0.0], [numpy.pi / 4.0]]))

    # Make sure there are no singularities in the gradients.
    gradients = torch.autograd.functional.jacobian(
        lambda x: compute_dihedrals(x, atom_indices), conformer
    )
    assert not torch.isnan(gradients).any() and not torch.isinf(gradients).any()


def test_build_v_site_coordinate_frames():
    conformer = torch.tensor(
        [
            [+1.0, +0.0, +0.0],
            [+0.0, +0.0, +0.0],
            [-1.0, +0.0, +1.0],
            [-1.0, +0.0, -1.0],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)

    force_field = smee.TensorForceField(
        potentials=[],
        v_sites=smee.TensorVSites(
            keys=[openff.interchange.models.PotentialKey(id="[O:1]=[C:2]-[H:3]")],
            weights=[V_SITE_TYPE_TO_FRAME["MonovalentLonePair"]],
            parameters=torch.Tensor([[1.0, 180.0, 45.0]]),
        ),
    )

    v_sites = [
        openff.interchange.models.VirtualSiteKey(
            orientation_atom_indices=atom_idxs,
            type="MonovalentLonePair",
            match="once",
            name="EP",
        )
        for atom_idxs in [(0, 1, 2), (0, 1, 3)]
    ]
    v_site_map = smee.VSiteMap(
        keys=v_sites,
        key_to_idx={v_sites[0]: 4, v_sites[1]: 5},
        parameter_idxs=torch.tensor([[0], [0]]),
    )

    actual_coord_frames = _build_v_site_coord_frames(v_site_map, conformer, force_field)

    expected_coord_frames = torch.tensor(
        [
            [[+1.0, +0.0, +0.0], [+1.0, +0.0, +0.0]],
            [[-1.0, +0.0, +0.0], [-1.0, +0.0, +0.0]],
            [[+0.0, +0.0, +1.0], [+0.0, +0.0, -1.0]],
            [[+0.0, +1.0, +0.0], [+0.0, -1.0, +0.0]],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)

    assert actual_coord_frames.shape == expected_coord_frames.shape
    assert torch.allclose(actual_coord_frames, expected_coord_frames)


def test_convert_v_site_coords():
    local_frame_coords = torch.tensor([[1.0, torch.pi / 4.0, torch.pi / 4.0]])
    local_coord_frames = torch.tensor(
        [
            [[+0.0, +0.0, +0.0]],
            [[+1.0, +0.0, +0.0]],
            [[+0.0, +1.0, +0.0]],
            [[+0.0, +0.0, +1.0]],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)

    actual_coords = _convert_v_site_coords(local_frame_coords, local_coord_frames)
    expected_coords = torch.tensor(
        [[0.5, 0.5, 1.0 / torch.sqrt(torch.tensor(2.0))]], dtype=torch.float64
    ).unsqueeze(0)

    assert actual_coords.shape == expected_coords.shape
    assert torch.allclose(actual_coords, expected_coords)


@pytest.mark.parametrize(
    "smiles",
    [
        "[H:3][C:2]([H:4])=[O:1]",
        "[Cl:3][C:2]([H:4])=[O:1]",
        "[H:2][O:1][H:3]",
        "[H:2][N:1]([H:3])[H:4]",
    ],
)
def test_compute_v_site_coords(smiles, v_site_force_field):
    molecule = openff.toolkit.Molecule.from_mapped_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)

    conformer = torch.tensor(molecule.conformers[0].m_as(unit.angstrom))

    interchange = _apply_v_site_force_field(molecule, conformer, v_site_force_field)
    force_field, [topology] = smee.converters.convert_interchange(interchange)

    assert topology.v_sites is not None
    assert len(topology.v_sites.keys) > 0

    openmm_v_site_coords = compute_openmm_v_sites(conformer, interchange)
    v_site_coords = compute_v_site_coords(topology.v_sites, conformer, force_field)

    assert v_site_coords.shape == openmm_v_site_coords.shape
    assert torch.allclose(v_site_coords, openmm_v_site_coords, atol=1e-5)


def test_compute_v_site_coords_batched(v_site_force_field):
    molecule = openff.toolkit.Molecule.from_mapped_smiles("[H:2][O:1][H:3]")

    conformers = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )

    interchange = _apply_v_site_force_field(molecule, conformers[0], v_site_force_field)
    force_field, [topology] = smee.converters.convert_interchange(interchange)

    assert topology.v_sites is not None
    assert len(topology.v_sites.keys) > 0

    openmm_v_site_coords_0 = compute_openmm_v_sites(conformers[0], interchange)

    interchange = _apply_v_site_force_field(molecule, conformers[1], v_site_force_field)
    openmm_v_site_coords_1 = compute_openmm_v_sites(conformers[1], interchange)

    openmm_v_site_coords = torch.stack([openmm_v_site_coords_0, openmm_v_site_coords_1])

    v_site_coords = compute_v_site_coords(topology.v_sites, conformers, force_field)

    assert v_site_coords.shape == openmm_v_site_coords.shape
    assert torch.allclose(v_site_coords, openmm_v_site_coords, atol=1e-5)


@pytest.mark.parametrize(
    "conformer, v_site_coords, cat_dim",
    [
        (torch.randn((5, 3)), torch.randn((3, 3)), 0),
        (torch.randn((4, 5, 3)), torch.randn((4, 3, 3)), 1),
    ],
)
def test_add_v_site_coords(conformer, v_site_coords, cat_dim, mocker):
    v_sites = mocker.MagicMock()
    force_field = mocker.MagicMock()

    mock_compute_coords_fn = mocker.patch(
        "smee.geometry.compute_v_site_coords", return_value=v_site_coords
    )

    coordinates = add_v_site_coords(v_sites, conformer, force_field)
    mock_compute_coords_fn.assert_called_once_with(v_sites, conformer, force_field)

    expected_coords = torch.cat([conformer, v_site_coords], dim=cat_dim)

    assert coordinates.shape == expected_coords.shape
    assert torch.allclose(coordinates, expected_coords)
