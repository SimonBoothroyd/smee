import numpy
import pytest
from openff.system.components.smirnoff import SMIRNOFFConstraintHandler
from openff.system.models import PotentialKey

from smirnoffee.smirnoff import (
    _HANDLER_TO_VECTORIZER,
    handler_vectorizer,
    vectorize_angle_handler,
    vectorize_bond_handler,
    vectorize_electrostatics_handler,
    vectorize_handler,
    vectorize_improper_handler,
    vectorize_proper_handler,
    vectorize_system,
    vectorize_vdw_handler,
)


def test_handler_vectorizer_decorator():

    handler_vectorizer("DummyHandler")(lambda x: None)
    assert "DummyHandler" in _HANDLER_TO_VECTORIZER

    with pytest.raises(KeyError, match="A vectorizer for the DummyHandler"):
        handler_vectorizer("DummyHandler")(lambda x: None)

    del _HANDLER_TO_VECTORIZER["DummyHandler"]


def test_vectorize_handler_error():

    with pytest.raises(NotImplementedError, match="Vectorizing Constraints handlers"):
        vectorize_handler(SMIRNOFFConstraintHandler())


def test_vectorize_bond_handler(ethanol, ethanol_system):

    bond_handler = ethanol_system.handlers["Bonds"]

    bond_indices, parameters, parameter_ids = vectorize_bond_handler(bond_handler)

    expected_bond_indices = [
        sorted((bond.atom1_index, bond.atom2_index)) for bond in ethanol.bonds
    ]
    assert (
        sorted(sorted(bond_tuple) for bond_tuple in bond_indices)
        == expected_bond_indices
    )

    expected_parameter_ids = [
        (PotentialKey(id="[#6:1]-[#8:2]", mult=None), ("k", "length")),
        (PotentialKey(id="[#8:1]-[#1:2]", mult=None), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#6X4:2]", mult=None), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#1:2]", mult=None), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#1:2]", mult=None), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#1:2]", mult=None), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#1:2]", mult=None), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#1:2]", mult=None), ("k", "length")),
    ]
    assert parameter_ids == expected_parameter_ids
    assert parameters.shape == (ethanol.n_bonds, 2)


def test_vectorize_angle_handler(ethanol, ethanol_system):

    angle_handler = ethanol_system.handlers["Angles"]

    angle_indices, parameters, parameter_ids = vectorize_angle_handler(angle_handler)

    expected_angle_indices = sorted(
        tuple(atom.molecule_atom_index for atom in angle) for angle in ethanol.angles
    )
    assert (
        sorted(tuple(i.item() for i in angle_tuple) for angle_tuple in angle_indices)
        == expected_angle_indices
    )

    expected_parameter_ids = [
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[*:1]-[#8:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#1:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#1:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#1:3]", mult=None), ("k", "angle")),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#1:3]", mult=None), ("k", "angle")),
    ]

    assert parameter_ids == expected_parameter_ids
    assert parameters.shape == (ethanol.n_angles, 2)


def test_vectorize_proper_handler(ethanol, ethanol_system):

    proper_handler = ethanol_system.handlers["ProperTorsions"]

    proper_indices, parameters, parameter_ids = vectorize_proper_handler(proper_handler)

    expected_proper_indices = sorted(
        tuple(atom.molecule_atom_index for atom in proper) for proper in ethanol.propers
    )
    actual_proper_indices = sorted(
        {tuple(i.item() for i in proper_tuple) for proper_tuple in proper_indices}
    )

    assert actual_proper_indices == expected_proper_indices

    expected_attrs = ("k", "periodicity", "phase", "idivf")
    expected_parameter_ids = [
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#8X2:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#8X2:4]", mult=1), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#8X2:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#8X2:4]", mult=1), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#8X2:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#8X2:4]", mult=1), expected_attrs),
        (
            PotentialKey(id="[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]", mult=0),
            expected_attrs,
        ),
        (
            PotentialKey(id="[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]", mult=1),
            expected_attrs,
        ),
        (PotentialKey(id="[*:1]-[#6X4:2]-[#8X2:3]-[#1:4]", mult=0), expected_attrs),
        (PotentialKey(id="[*:1]-[#6X4:2]-[#8X2:3]-[#1:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]", mult=0), expected_attrs),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]", mult=0), expected_attrs),
    ]

    assert parameter_ids == expected_parameter_ids
    assert parameters.shape[1] == 4


def test_vectorize_improper_handler(formaldehyde, formaldehyde_system):

    improper_handler = formaldehyde_system.handlers["ImproperTorsions"]

    improper_indices, parameters, parameter_ids = vectorize_improper_handler(
        improper_handler
    )

    expected_improper_indices = sorted([(1, 0, 2, 3), (1, 2, 3, 0), (1, 3, 0, 2)])

    actual_improper_indices = sorted(
        tuple(i.item() for i in improper_tuple) for improper_tuple in improper_indices
    )

    assert actual_improper_indices == expected_improper_indices

    expected_parameter_ids = [
        (
            PotentialKey(id="[*:1]~[#6X3:2](~[*:3])~[*:4]", mult=0),
            ("k", "periodicity", "phase", "idivf"),
        )
    ] * 3

    assert parameter_ids == expected_parameter_ids
    assert parameters.shape == (3, 4)


def test_vectorize_electrostatics_handler(ethanol, ethanol_system):

    electrostatics_handler = ethanol_system.handlers["Electrostatics"]

    pair_indices, parameters, parameter_ids = vectorize_electrostatics_handler(
        electrostatics_handler, ethanol
    )

    expected_pair_indices = [
        (proper[0].molecule_atom_index, proper[3].molecule_atom_index)
        for proper in ethanol.propers
    ] + [(3, 4), (3, 5), (3, 6)]
    actual_pair_indices = [
        tuple(int(i) for i in pair_index) for pair_index in pair_indices
    ]

    assert actual_pair_indices == expected_pair_indices

    expected_parameter_ids = [
        (PotentialKey(id="[*:1]"), ("q1", "q2", "scale_14"))
    ] * ethanol.n_propers + [(PotentialKey(id="[*:1]"), ("q1", "q2", "scale_1n"))] * 3

    assert parameter_ids == expected_parameter_ids

    assert numpy.allclose(parameters[: ethanol.n_propers, 2].numpy(), 1.0 / 1.2)
    assert numpy.allclose(parameters[ethanol.n_propers :, 2].numpy(), 1.0)


def test_vectorize_vdw_handler(ethanol, ethanol_system):

    vdw_handler = ethanol_system.handlers["vdW"]

    pair_indices, parameters, parameter_ids = vectorize_vdw_handler(
        vdw_handler, ethanol
    )

    expected_pair_indices = [
        (proper[0].molecule_atom_index, proper[3].molecule_atom_index)
        for proper in ethanol.propers
    ] + [(3, 4), (3, 5), (3, 6)]
    actual_pair_indices = [
        tuple(int(i) for i in pair_index) for pair_index in pair_indices
    ]

    assert actual_pair_indices == expected_pair_indices

    expected_parameter_ids = [
        (PotentialKey(id="[*:1]"), ("epsilon", "sigma", "scale_14"))
    ] * ethanol.n_propers + [
        (PotentialKey(id="[*:1]"), ("epsilon", "sigma", "scale_1n"))
    ] * 3

    assert parameter_ids == expected_parameter_ids

    assert numpy.allclose(parameters[: ethanol.n_propers, 2].numpy(), 0.5)
    assert numpy.allclose(parameters[ethanol.n_propers :, 2].numpy(), 1.0)


def test_vectorize_handler(ethanol, ethanol_system, monkeypatch):

    bond_handler_triggered = False
    vdw_handler_triggered = False

    def _dummy_bond_vectorizer(handler):
        nonlocal bond_handler_triggered
        bond_handler_triggered = True

        assert handler.type == "Bonds"
        return (), (), ()

    def _dummy_vdw_vectorizer(handler, molecule):
        nonlocal vdw_handler_triggered
        vdw_handler_triggered = True

        assert handler.type == "vdW"
        assert molecule is not None
        return (), (), ()

    monkeypatch.setitem(_HANDLER_TO_VECTORIZER, "Bonds", _dummy_bond_vectorizer)
    monkeypatch.setitem(_HANDLER_TO_VECTORIZER, "vdW", _dummy_vdw_vectorizer)

    return_tuple = vectorize_handler(ethanol_system.handlers["Bonds"])
    assert bond_handler_triggered
    assert len(return_tuple) == 3

    with pytest.raises(TypeError, match="The `molecule` attribute must be provided"):
        vectorize_handler(ethanol_system.handlers["vdW"])

    return_tuple = vectorize_handler(ethanol_system.handlers["vdW"], ethanol)
    assert vdw_handler_triggered
    assert len(return_tuple) == 3


@pytest.mark.parametrize("molecule_name", ["ethanol", "formaldehyde"])
def test_vectorize_system(request, molecule_name):

    openff_system = request.getfixturevalue(f"{molecule_name}_system")

    vectorized_system = vectorize_system(openff_system)

    expected_keys = {
        ("Bonds", "k/2*(r-length)**2"),
        ("Angles", "k/2*(theta-angle)**2"),
        ("ProperTorsions", "k*(1+cos(periodicity*theta-phase))"),
        ("ImproperTorsions", "k*(1+cos(periodicity*theta-phase))"),
        ("vdW", "4*epsilon*((sigma/r)**12-(sigma/r)**6)"),
        ("Electrostatics", "coul"),
    }
    assert {*vectorized_system} == expected_keys
    assert all(len(value) == 3 for value in vectorized_system.values())
