import openff.interchange
import openff.interchange.models
import openff.toolkit
import openff.units
import torch

import smirnoffee.ff
from smirnoffee.ff.nonbonded import convert_electrostatics, convert_vdw


def test_convert_electrostatics_am1bcc(ethanol, ethanol_interchange):
    charge_collection = ethanol_interchange.collections["Electrostatics"]

    potential, parameter_maps = convert_electrostatics(
        [charge_collection], [ethanol.to_topology()], [None]
    )

    assert potential.type == "Electrostatics"
    assert potential.fn == "coul"

    expected_attributes = torch.tensor([0.0, 0.0, 5.0 / 6.0, 1.0])
    assert torch.allclose(potential.attributes, expected_attributes)
    assert potential.attribute_cols == (
        "scale_12",
        "scale_13",
        "scale_14",
        "scale_15",
    )

    assert potential.parameter_cols == ("charge",)

    assert all(
        parameter_key.id == "[O:1]([C:3]([C:2]([H:5])([H:6])[H:7])([H:8])[H:9])[H:4]"
        for parameter_key in potential.parameter_keys
    )
    assert potential.parameters.shape == (9, 1)

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    assert parameter_map.assignment_matrix.shape == (ethanol.n_atoms, ethanol.n_atoms)
    assert torch.allclose(
        parameter_map.assignment_matrix.to_dense(), torch.eye(ethanol.n_atoms)
    )

    n_expected_exclusions = 36
    assert parameter_map.exclusions.shape == (n_expected_exclusions, 2)
    assert parameter_map.exclusion_scale_idxs.shape == (n_expected_exclusions, 1)


def test_convert_electrostatics_v_site():
    force_field = openff.toolkit.ForceField()
    force_field.get_parameter_handler("Electrostatics")
    force_field.get_parameter_handler("vdW")

    charge_handler = force_field.get_parameter_handler("LibraryCharges")
    charge_handler.add_parameter(
        {
            "smirks": "[Cl:1]-[H:2]",
            "charge1": -0.75 * openff.units.unit.e,
            "charge2": 0.25 * openff.units.unit.e,
        }
    )

    v_site_handler = force_field.get_parameter_handler("VirtualSites")
    v_site_handler.add_parameter(
        {
            "type": "BondCharge",
            "smirks": "[Cl:1]-[H:2]",
            "distance": 2.0 * openff.units.unit.angstrom,
            "match": "all_permutations",
            "charge_increment1": -0.25 * openff.units.unit.e,
            "charge_increment2": 0.5 * openff.units.unit.e,
        }
    )

    molecule = openff.toolkit.Molecule.from_mapped_smiles("[Cl:2]-[H:1]")

    interchange = openff.interchange.Interchange.from_smirnoff(
        force_field, molecule.to_topology(), allow_nonintegral_charges=True
    )
    charge_collection = interchange.collections["Electrostatics"]

    potential, parameter_maps = convert_electrostatics(
        [charge_collection],
        [molecule.to_topology()],
        [
            smirnoffee.ff.VSiteMap(
                keys=[*interchange.collections["VirtualSites"].key_map],
                key_to_idx=interchange.collections[
                    "VirtualSites"
                ].virtual_site_key_topology_index_map,
                parameter_idxs=torch.tensor([[0]]),
            )
        ],
    )

    assert potential.parameter_cols == ("charge",)
    expected_keys = [
        openff.interchange.models.PotentialKey(
            id="[Cl:1]-[H:2]", mult=0, associated_handler="LibraryCharges"
        ),
        openff.interchange.models.PotentialKey(
            id="[Cl:1]-[H:2]", mult=1, associated_handler="LibraryCharges"
        ),
        openff.interchange.models.PotentialKey(
            id="[Cl:1]-[H:2] EP all_permutations",
            mult=0,
            associated_handler="Electrostatics",
        ),
        openff.interchange.models.PotentialKey(
            id="[Cl:1]-[H:2] EP all_permutations",
            mult=1,
            associated_handler="Electrostatics",
        ),
    ]
    assert potential.parameter_keys == expected_keys
    assert potential.parameters.shape == (4, 1)

    expected_parameters = torch.tensor([[-0.75], [0.25], [-0.25], [0.5]])
    assert torch.allclose(potential.parameters.float(), expected_parameters)

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    n_particles = 3

    assert parameter_map.assignment_matrix.shape == (n_particles, len(expected_keys))
    expected_assignment_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(
        parameter_map.assignment_matrix.to_dense(), expected_assignment_matrix
    )

    n_expected_exclusions = 3
    assert parameter_map.exclusions.shape == (n_expected_exclusions, 2)
    assert parameter_map.exclusion_scale_idxs.shape == (n_expected_exclusions, 1)

    expected_exclusions = torch.tensor([[0, 1], [2, 1], [0, 2]])
    assert torch.allclose(parameter_map.exclusions, expected_exclusions)

    expected_scales = torch.zeros((n_expected_exclusions, 1), dtype=torch.long)
    assert torch.allclose(parameter_map.exclusion_scale_idxs, expected_scales)


def test_convert_vdw(ethanol, ethanol_interchange):
    vdw_collection = ethanol_interchange.collections["vdW"]

    potential, parameter_maps = convert_vdw(
        [vdw_collection], [ethanol.to_topology()], [None]
    )

    assert potential.type == "vdW"
    assert potential.fn == "4*epsilon*((sigma/r)**12-(sigma/r)**6)"