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
    vectorize_improper_handler,
    vectorize_proper_handler,
    vectorize_valence_handler,
)


def test_handler_vectorizer_decorator():

    handler_vectorizer("DummyHandler")(lambda x: None)
    assert "DummyHandler" in _HANDLER_TO_VECTORIZER

    with pytest.raises(KeyError, match="A vectorizer for the DummyHandler"):
        handler_vectorizer("DummyHandler")(lambda x: None)

    del _HANDLER_TO_VECTORIZER["DummyHandler"]


def test_vectorize_handler_error():

    with pytest.raises(NotImplementedError, match="Vectorizing Constraints handlers"):
        vectorize_valence_handler(SMIRNOFFConstraintHandler())


@pytest.mark.parametrize(
    "vectorizer", [vectorize_valence_handler, vectorize_bond_handler]
)
def test_vectorize_bond_handler(ethanol, ethanol_system, vectorizer):

    bond_handler = ethanol_system.handlers["Bonds"]

    bond_indices, parameters, parameter_ids = vectorizer(bond_handler)

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


@pytest.mark.parametrize(
    "vectorizer", [vectorize_valence_handler, vectorize_angle_handler]
)
def test_vectorize_angle_handler(ethanol, ethanol_system, vectorizer):

    angle_handler = ethanol_system.handlers["Angles"]

    angle_indices, parameters, parameter_ids = vectorizer(angle_handler)

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


@pytest.mark.parametrize(
    "vectorizer", [vectorize_valence_handler, vectorize_proper_handler]
)
def test_vectorize_proper_handler(ethanol, ethanol_system, vectorizer):

    proper_handler = ethanol_system.handlers["ProperTorsions"]

    proper_indices, parameters, parameter_ids = vectorizer(proper_handler)

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


@pytest.mark.xfail(
    reason="`openff-system` incorrectly populates the improper slot maps"
)
@pytest.mark.parametrize(
    "vectorizer", [vectorize_valence_handler, vectorize_improper_handler]
)
def test_vectorize_improper_handler(formaldehyde, formaldehyde_system, vectorizer):

    improper_handler = formaldehyde_system.handlers["ImproperTorsions"]

    improper_indices, parameters, parameter_ids = vectorizer(improper_handler)

    expected_improper_indices = sorted(
        tuple(atom.molecule_atom_index for atom in improper)
        for improper in formaldehyde.impropers
    )
    assert (
        sorted(
            tuple(i.item() for i in improper_tuple)
            for improper_tuple in improper_indices
        )
        == expected_improper_indices
    )

    expected_parameter_ids = [
        (
            PotentialKey(id="'[*:1]~[#6X3:2](~[*:3])~[*:4]'", mult=0),
            ("k", "periodicity", "phase", "idivf"),
        )
    ]

    assert parameter_ids == expected_parameter_ids
    assert parameters.shape == (formaldehyde.n_impropers, 4)


@pytest.mark.parametrize("vectorizer", [vectorize_electrostatics_handler])
def test_vectorize_electrostatics_handler(ethanol, ethanol_system, vectorizer):

    electrostatics_handler = ethanol_system.handlers["Electrostatics"]

    pair_indices, parameters, parameter_ids = vectorizer(
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
