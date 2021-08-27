from collections import defaultdict
from typing import Dict, Tuple

import numpy
import torch
from openff.toolkit.topology import Molecule

from smirnoffee.geometry.internal import cartesian_to_internal


def _order_key(key: Tuple[int, ...]) -> Tuple[int, ...]:
    return key if key[-1] > key[0] else tuple(reversed(key))


def geometric_internal_coordinates(
    molecule: Molecule, conformer: torch.Tensor
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Computes the internal coordinate representation of a molecule using ``geomeTRIC``
    and returns the output in tensor form.

    Args:
        molecule: The molecule to compute on.
        conformer: The cartesian coordinates of the molecule to convert with
            shape=(n_atoms, 3) and units of [Å].

    Returns:
        A dictionary storing the internal coordinates of the form
        ``ic[ic_type] = (atom_indices, values)`` where ``ic_type`` is a type of
        internal coordinate, ``atom_indices`` a 2D tensor of the atom indices involved
        in the internal coordinate, and values a tensor of the values of each internal
        coordinate of that type. Distances will have units of [Å] and angles units of
        [rad].
    """

    from geometric.internal import Angle, Dihedral, Distance, LinearAngle, OutOfPlane
    from geometric.internal import PrimitiveInternalCoordinates as GeometricPRIC
    from geometric.molecule import Molecule as GeometricMolecule

    geometric_molecule = GeometricMolecule()
    geometric_molecule.Data = {
        "resname": ["UNK"] * molecule.n_atoms,
        "resid": [0] * molecule.n_atoms,
        "elem": [atom.element.symbol for atom in molecule.atoms],
        "bonds": [(bond.atom1_index, bond.atom2_index) for bond in molecule.bonds],
        "name": molecule.name,
        "xyzs": [conformer.detach().numpy()],
    }

    expected_coordinates = {
        Distance: ("distances", ("a", "b")),
        Angle: ("angles", ("a", "b", "c")),
        LinearAngle: ("linear-angles", ("a", "b", "c", "axis")),
        OutOfPlane: ("out-of-plane-angles", ("a", "b", "c", "d")),
        Dihedral: ("dihedrals", ("a", "b", "c", "d")),
    }

    geometric_molecule.build_topology()

    geometric_coordinates = GeometricPRIC(geometric_molecule)

    coordinates_by_type = defaultdict(lambda: (list(), list()))

    for internal_coordinate in geometric_coordinates.Internals:

        internal_coordinate_type = type(internal_coordinate)

        if internal_coordinate_type not in expected_coordinates:
            continue

        internal_coordinate_name, index_attributes = expected_coordinates[
            internal_coordinate_type
        ]

        coordinates_by_type[internal_coordinate_name][0].append(
            [getattr(internal_coordinate, attr_name) for attr_name in index_attributes]
        )
        coordinates_by_type[internal_coordinate_name][1].append(
            internal_coordinate.value(conformer.detach().numpy())
        )

    return {
        key: (torch.tensor(indices), torch.tensor(values))
        for key, (indices, values) in coordinates_by_type.items()
    }


def validate_internal_coordinates(
    molecule: Molecule, conformer: torch.Tensor, verbose: bool = False
):
    """Compares the values of the primitive redundant internal coordinates computed
    by ``smirnoffee`` with those computed using ``geomeTRIC``.

    An assertion error will be raised if any differences are found.

    Args:
        molecule: The molecule of interest.
        conformer: The conformer of the molecule with shape=(n_atoms, 3) and units
            of [Å].
        verbose: Whether to print information about any differences.
    """

    if verbose:
        print("", flush=True)

    bond_indices = torch.tensor(
        [[bond.atom1_index, bond.atom2_index] for bond in molecule.bonds]
    )

    actual_internal_coordinates = cartesian_to_internal(
        conformer, bond_indices, coordinate_system="ric"
    )
    expected_internal_coordinates = geometric_internal_coordinates(molecule, conformer)

    for internal_coordinate_type in actual_internal_coordinates:

        if verbose:
            print(internal_coordinate_type.center(80, "="))

        actual_value_by_index = {
            _order_key(tuple(int(index) for index in indices)): float(value)
            for indices, value in zip(
                *actual_internal_coordinates[internal_coordinate_type]
            )
        }
        expected_value_by_index = {
            _order_key(tuple(int(index) for index in indices)): float(value)
            for indices, value in zip(
                *expected_internal_coordinates[internal_coordinate_type]
            )
        }

        if verbose:

            print("MISSING  ", {*expected_value_by_index} - {*actual_value_by_index})
            print("EXTRA    ", {*actual_value_by_index} - {*expected_value_by_index})

            print("", flush=True)

        assert {*expected_value_by_index} == {*actual_value_by_index}

        for key in {*expected_value_by_index}.union({*actual_value_by_index}):

            expected_value = expected_value_by_index.get(key, numpy.nan)
            actual_value = actual_value_by_index.get(key, numpy.nan)

            if verbose:
                print(key, f"{expected_value:.5f}", f"{actual_value:.5f}")

            assert numpy.isclose(expected_value, actual_value)

        if verbose:
            print("")
