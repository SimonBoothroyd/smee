import openff.units
import pytest
import torch

from smirnoffee.ff.smirnoff import (
    _HANDLER_CONVERTERS,
    _HANDLER_DEFAULT_UNITS,
    _convert_angles,
    _convert_bonds,
    _convert_electrostatics,
    _convert_impropers,
    _convert_propers,
    _convert_vdw,
    _handler_converter,
    convert_handlers,
)


def test_collection_vectorizer():
    _handler_converter("Dummy", {"parm-a": openff.units.unit.angstrom})(lambda x: None)
    assert "Dummy" in _HANDLER_CONVERTERS
    assert "parm-a" in _HANDLER_DEFAULT_UNITS["Dummy"]

    with pytest.raises(KeyError, match="A Dummy converter is already"):
        _handler_converter("Dummy", {})(lambda x: None)

    del _HANDLER_CONVERTERS["Dummy"]
    del _HANDLER_DEFAULT_UNITS["Dummy"]


def test_convert_bonds(ethanol, ethanol_interchange):
    bond_collection = ethanol_interchange.collections["Bonds"]

    potential, parameter_maps = _convert_bonds([bond_collection])

    assert potential.type == "Bonds"
    assert potential.fn == "k/2*(r-length)**2"

    assert potential.global_parameters is None
    assert potential.global_parameter_cols is None

    assert potential.parameter_cols == ("k", "length")

    parameter_keys = [key.id for key in potential.parameter_keys]
    expected_parameter_keys = [
        "[#6:1]-[#8:2]",
        "[#6X4:1]-[#1:2]",
        "[#6X4:1]-[#6X4:2]",
        "[#8:1]-[#1:2]",
    ]
    assert sorted(parameter_keys) == sorted(expected_parameter_keys)

    assert potential.parameters.shape == (4, 2)

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    assert len(parameter_map.parameter_idxs) == len(parameter_map.atom_idxs)

    actual_parameters = {
        tuple(atom_idxs.tolist()): parameter_keys[parameter_idx]
        for parameter_idx, atom_idxs in zip(
            parameter_map.parameter_idxs, parameter_map.atom_idxs
        )
    }
    expected_parameters = {
        (0, 2): "[#6:1]-[#8:2]",
        (0, 3): "[#8:1]-[#1:2]",
        (1, 2): "[#6X4:1]-[#6X4:2]",
        (1, 4): "[#6X4:1]-[#1:2]",
        (1, 5): "[#6X4:1]-[#1:2]",
        (1, 6): "[#6X4:1]-[#1:2]",
        (2, 7): "[#6X4:1]-[#1:2]",
        (2, 8): "[#6X4:1]-[#1:2]",
    }

    assert actual_parameters == expected_parameters


def test_convert_angles(ethanol, ethanol_interchange):
    angle_collection = ethanol_interchange.collections["Angles"]

    potential, parameter_maps = _convert_angles([angle_collection])

    assert potential.type == "Angles"
    assert potential.fn == "k/2*(theta-angle)**2"


def test_convert_propers(ethanol, ethanol_interchange):
    proper_collection = ethanol_interchange.collections["ProperTorsions"]

    potential, parameter_maps = _convert_propers([proper_collection])

    assert potential.type == "ProperTorsions"
    assert potential.fn == "k*(1+cos(periodicity*theta-phase))"

    hcco_smirks = "[#1:1]-[#6X4:2]-[#6X4:3]-[#8X2:4]"
    ccoh_smirks = "[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]"
    xcoh_smirks = "[*:1]-[#6X4:2]-[#8X2:3]-[#1:4]"
    hcch_smirks = "[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]"

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    assert len(parameter_map.parameter_idxs) == len(parameter_map.atom_idxs)

    actual_parameters = {
        (
            tuple(atom_idxs.tolist()),
            potential.parameter_keys[parameter_idx].id,
            potential.parameter_keys[parameter_idx].mult,
        )
        for parameter_idx, atom_idxs in zip(
            parameter_map.parameter_idxs, parameter_map.atom_idxs
        )
    }
    expected_parameters = {
        ((0, 2, 1, 4), hcco_smirks, 1),
        ((0, 2, 1, 4), hcco_smirks, 0),
        ((0, 2, 1, 5), hcco_smirks, 1),
        ((0, 2, 1, 5), hcco_smirks, 0),
        ((0, 2, 1, 6), hcco_smirks, 1),
        ((0, 2, 1, 6), hcco_smirks, 0),
        ((1, 2, 0, 3), ccoh_smirks, 1),
        ((1, 2, 0, 3), ccoh_smirks, 0),
        ((3, 0, 2, 7), xcoh_smirks, 0),
        ((3, 0, 2, 8), xcoh_smirks, 0),
        ((4, 1, 2, 7), hcch_smirks, 0),
        ((4, 1, 2, 8), hcch_smirks, 0),
        ((5, 1, 2, 7), hcch_smirks, 0),
        ((5, 1, 2, 8), hcch_smirks, 0),
        ((6, 1, 2, 7), hcch_smirks, 0),
        ((6, 1, 2, 8), hcch_smirks, 0),
    }
    assert actual_parameters == expected_parameters


def test_convert_impropers(formaldehyde, formaldehyde_interchange):
    improper_collection = formaldehyde_interchange.collections["ImproperTorsions"]

    potential, parameter_maps = _convert_impropers([improper_collection])

    assert potential.type == "ImproperTorsions"
    assert potential.fn == "k*(1+cos(periodicity*theta-phase))"


def test_convert_electrostatics(ethanol, ethanol_interchange):
    charge_collection = ethanol_interchange.collections["Electrostatics"]

    potential, parameter_maps = _convert_electrostatics(
        [charge_collection], [ethanol.to_topology()]
    )

    assert potential.type == "Electrostatics"
    assert potential.fn == "coul"

    expected_global_parameters = torch.tensor([0.0, 0.0, 5.0 / 6.0, 1.0])
    assert torch.allclose(potential.global_parameters, expected_global_parameters)
    assert potential.global_parameter_cols == (
        "scale_12",
        "scale_13",
        "scale_14",
        "scale_1n",
    )

    assert potential.parameter_cols == ("charge",)

    assert all(
        parameter_key.id == "[O:1]([C:3]([C:2]([H:5])([H:6])[H:7])([H:8])[H:9])[H:4]"
        for parameter_key in potential.parameter_keys
    )
    assert potential.parameters.shape == (9, 1)

    assert len(parameter_maps) == 1
    parameter_map = parameter_maps[0]

    assert len(parameter_map.parameter_idxs) == len(parameter_map.atom_idxs)


def test_convert_vdw(ethanol, ethanol_interchange):
    vdw_handler = ethanol_interchange.handlers["vdW"]

    potential, parameter_maps = _convert_vdw([vdw_handler], [ethanol.to_topology()])

    assert potential.type == "vdW"
    assert potential.fn == "4*epsilon*((sigma/r)**12-(sigma/r)**6)"


def test_convert_handler(ethanol, ethanol_interchange, mocker):
    mock_result = mocker.MagicMock()

    mock_vectorize = mocker.patch(
        "smirnoffee.ff.smirnoff._convert_vdw",
        autospec=True,
        return_value=mock_result,
    )
    mocker.patch.dict(_HANDLER_CONVERTERS, {"vdW": mock_vectorize})

    handlers = [ethanol_interchange.collections["vdW"]]
    topologies = [ethanol.to_topology()]

    result = convert_handlers(handlers, topologies)

    mock_vectorize.assert_called_once_with(handlers, topologies=topologies)
    assert result == mock_result
