import pytest
from openff.system.components.smirnoff import SMIRNOFFConstraintHandler
from openff.system.models import PotentialKey

from smirnoffee.smirnoff import (
    _HANDLER_TO_VECTORIZER,
    handler_vectorizer,
    vectorize_angle_handler,
    vectorize_bond_handler,
    vectorize_handler,
    vectorize_improper_handler,
    vectorize_proper_handler,
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


@pytest.mark.parametrize("vectorizer", [vectorize_handler, vectorize_bond_handler])
def test_vectorize_bond_handler(ethanol, ethanol_system, vectorizer):

    bond_handler = ethanol_system.handlers["Bonds"]

    bond_indices, parameter_ids, parameters = vectorizer(bond_handler)

    expected_bond_indices = [
        sorted((bond.atom1_index, bond.atom2_index)) for bond in ethanol.bonds
    ]
    assert (
        sorted(sorted(bond_tuple) for bond_tuple in bond_indices)
        == expected_bond_indices
    )

    expected_parameter_ids = [
        (PotentialKey(id="[#6:1]-[#8:2]"), ("k", "length")),
        (PotentialKey(id="[#8:1]-[#1:2]"), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#1:2]"), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#1:2]"), ("k", "length")),
        (PotentialKey(id="[#6X4:1]-[#1:2]"), ("k", "length")),
    ]
    assert parameter_ids == expected_parameter_ids
    assert parameters.shape == (ethanol.n_bonds, 2)


@pytest.mark.parametrize("vectorizer", [vectorize_handler, vectorize_angle_handler])
def test_vectorize_angle_handler(ethanol, ethanol_system, vectorizer):

    angle_handler = ethanol_system.handlers["Angles"]

    angle_indices, parameter_ids, parameters = vectorizer(angle_handler)

    expected_angle_indices = sorted(
        tuple(atom.molecule_atom_index for atom in angle) for angle in ethanol.angles
    )
    assert (
        sorted(tuple(i.item() for i in angle_tuple) for angle_tuple in angle_indices)
        == expected_angle_indices
    )

    expected_parameter_ids = [
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]"), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]"), ("k", "angle")),
        (PotentialKey(id="[*:1]~[#6X4:2]-[*:3]"), ("k", "angle")),
        (PotentialKey(id="[*:1]-[#8:2]-[*:3]"), ("k", "angle")),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#1:3]"), ("k", "angle")),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#1:3]"), ("k", "angle")),
        (PotentialKey(id="[#1:1]-[#6X4:2]-[#1:3]"), ("k", "angle")),
    ]

    assert parameter_ids == expected_parameter_ids
    assert parameters.shape == (ethanol.n_angles, 2)


@pytest.mark.parametrize("vectorizer", [vectorize_handler, vectorize_proper_handler])
def test_vectorize_proper_handler(ethanol, ethanol_system, vectorizer):

    proper_handler = ethanol_system.handlers["ProperTorsions"]

    proper_indices, parameter_ids, parameters = vectorizer(proper_handler)

    expected_proper_indices = sorted(
        tuple(atom.molecule_atom_index for atom in proper) for proper in ethanol.propers
    )
    assert (
        sorted(tuple(i.item() for i in proper_tuple) for proper_tuple in proper_indices)
        == expected_proper_indices
    )

    expected_parameter_ids = [
        (
            PotentialKey(id="[*:1]-[#6X4:2]-[#8X2:3]-[#1:4]", mult=0),
            ("k", "periodicity", "phase", "idivf"),
        ),
        (
            PotentialKey(id="[*:1]-[#6X4:2]-[#8X2:3]-[#1:4]", mult=0),
            ("k", "periodicity", "phase", "idivf"),
        ),
        (
            PotentialKey(id="[*:1]-[#6X4:2]-[#8X2:3]-[#1:4]", mult=0),
            ("k", "periodicity", "phase", "idivf"),
        ),
    ]

    assert parameter_ids == expected_parameter_ids
    assert parameters.shape == (ethanol.n_propers, 4)


@pytest.mark.xfail(
    reason="`openff-system` incorrectly populates the improper slot maps"
)
@pytest.mark.parametrize("vectorizer", [vectorize_handler, vectorize_improper_handler])
def test_vectorize_improper_handler(formaldehyde, formaldehyde_system, vectorizer):

    improper_handler = formaldehyde_system.handlers["ImproperTorsions"]

    improper_indices, parameter_ids, parameters = vectorizer(improper_handler)

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
