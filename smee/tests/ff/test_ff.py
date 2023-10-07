import openff.interchange.models
import openff.toolkit
import openff.units
import pytest
import torch

from smee.ff._ff import (
    _CONVERTERS,
    _DEFAULT_UNITS,
    VSiteMap,
    _convert_topology,
    convert_handlers,
    convert_interchange,
    parameter_converter,
)


def test_parameter_converter():
    parameter_converter("Dummy", {"parm-a": openff.units.unit.angstrom})(lambda x: None)
    assert "Dummy" in _CONVERTERS
    assert "parm-a" in _DEFAULT_UNITS["Dummy"]

    with pytest.raises(KeyError, match="A Dummy converter is already"):
        parameter_converter("Dummy", {})(lambda x: None)

    del _CONVERTERS["Dummy"]
    del _DEFAULT_UNITS["Dummy"]


def test_convert_handler(ethanol, ethanol_interchange, mocker):
    mock_result = mocker.MagicMock()

    mock_vectorize = mocker.patch(
        "smee.ff.nonbonded.convert_vdw",
        autospec=True,
        return_value=mock_result,
    )
    mocker.patch.dict(_CONVERTERS, {"vdW": mock_vectorize})

    handlers = [ethanol_interchange.collections["vdW"]]
    topologies = [ethanol.to_topology()]

    v_site = openff.interchange.models.VirtualSiteKey(
        orientation_atom_indices=(0, 1, 2),
        type="MonovalentLonePair",
        match="once",
        name="EP",
    )
    v_site_maps = [VSiteMap([v_site], {v_site: ethanol.n_atoms}, torch.tensor([[0]]))]

    result = convert_handlers(handlers, topologies, v_site_maps)

    mock_vectorize.assert_called_once_with(
        handlers, topologies=topologies, v_site_maps=v_site_maps
    )
    assert result == mock_result


def test_convert_topology(formaldehyde, mocker):
    parameters = mocker.MagicMock()
    v_sites = VSiteMap([], {}, torch.tensor([]))

    topology = _convert_topology(formaldehyde, parameters, v_sites)

    assert topology.n_atoms == 4
    assert topology.n_bonds == 3

    expected_atomic_nums = torch.tensor([6, 8, 1, 1])
    expected_formal_charges = torch.tensor([0, 0, 0, 0])

    expected_bond_idxs = torch.tensor([[0, 1], [0, 2], [0, 3]])
    expected_bond_orders = torch.tensor([2, 1, 1])

    assert topology.atomic_nums.shape == expected_atomic_nums.shape
    assert torch.allclose(topology.atomic_nums, expected_atomic_nums)
    assert topology.formal_charges.shape == expected_formal_charges.shape
    assert torch.allclose(topology.formal_charges, expected_formal_charges)

    assert topology.bond_idxs.shape == expected_bond_idxs.shape
    assert torch.allclose(topology.bond_idxs, expected_bond_idxs)
    assert topology.bond_orders.shape == expected_bond_orders.shape
    assert torch.allclose(topology.bond_orders, expected_bond_orders)

    assert topology.parameters == parameters
    assert topology.v_sites == v_sites


def test_convert_interchange():
    force_field = openff.toolkit.ForceField()
    force_field.get_parameter_handler("Electrostatics")
    force_field.get_parameter_handler("vdW")

    constraint_handler = force_field.get_parameter_handler("Constraints")
    constraint_handler.add_parameter(
        {"smirks": "[Cl:1]-[H:2]", "distance": 0.2 * openff.units.unit.nanometer}
    )

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

    tensor_force_field, tensor_topologies = convert_interchange(interchange)

    assert {*tensor_force_field.potentials_by_type} == {"vdW", "Electrostatics"}

    assert tensor_force_field.v_sites is not None
    assert len(tensor_force_field.v_sites.keys) == 1
    assert tensor_force_field.v_sites.keys[0].id == "[Cl:1]-[H:2] EP all_permutations"
    expected_parameters = torch.tensor([[2.0, torch.pi, 0.0]])
    assert torch.allclose(tensor_force_field.v_sites.parameters, expected_parameters)
    assert len(tensor_force_field.v_sites.weights) == 1

    assert len(tensor_topologies) == 1
    tensor_topology = tensor_topologies[0]

    assert len(tensor_topology.v_sites.keys) == 1
    assert tensor_topology.v_sites.keys[0].type == "BondCharge"
    assert tensor_topology.v_sites.keys[0].orientation_atom_indices == (1, 0)

    assert tensor_topology.constraints is not None
    expected_constraint_idxs = torch.tensor([[0, 1]])
    assert tensor_topology.constraints.idxs.shape == expected_constraint_idxs.shape
    assert torch.allclose(tensor_topology.constraints.idxs, expected_constraint_idxs)

    expected_constraint_distances = torch.tensor([2.0])
    assert (
        tensor_topology.constraints.distances.shape
        == expected_constraint_distances.shape
    )
    assert torch.allclose(
        tensor_topology.constraints.distances, expected_constraint_distances
    )


def test_convert_interchange_multiple(
    ethanol_conformer,
    ethanol_interchange,
    formaldehyde_conformer,
    formaldehyde_interchange,
):
    force_field, topologies = convert_interchange(
        [ethanol_interchange, formaldehyde_interchange]
    )
    assert len(topologies) == 2

    expected_potentials = {
        "Angles",
        "Bonds",
        "Electrostatics",
        "ImproperTorsions",
        "ProperTorsions",
        "vdW",
    }
    assert {*force_field.potentials_by_type} == expected_potentials

    expected_charge_keys = [
        openff.interchange.models.PotentialKey(
            id="[O:1]([C:3]([C:2]([H:5])([H:6])[H:7])([H:8])[H:9])[H:4]",
            mult=0,
            associated_handler="ToolkitAM1BCCHandler",
        ),
        openff.interchange.models.PotentialKey(
            id="[C:1](=[O:2])([H:3])[H:4]",
            mult=0,
            associated_handler="ToolkitAM1BCCHandler",
        ),
    ]
    assert all(
        key in force_field.potentials_by_type["Electrostatics"].parameter_keys
        for key in expected_charge_keys
    )

    expected_improper_keys = [
        openff.interchange.models.PotentialKey(
            id="[*:1]~[#6X3:2](~[*:3])~[*:4]",
            mult=0,
            associated_handler="ImproperTorsions",
        ),
    ]
    assert (
        force_field.potentials_by_type["ImproperTorsions"].parameter_keys
        == expected_improper_keys
    )
