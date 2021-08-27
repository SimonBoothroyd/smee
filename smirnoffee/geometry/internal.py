import abc
import itertools
from typing import Dict, Optional, Tuple

import networkx
import torch
from typing_extensions import Literal

from smirnoffee.geometry import (
    compute_angles,
    compute_bond_vectors,
    compute_dihedrals,
    compute_linear_displacement,
)

# The linear threshold is taken from `geometric ==0.9.7.2`
_LINEAR_THRESHOLD = 0.95

InternalCoordinateType = Literal[
    "distances", "angles", "linear-angles", "out-of-plane-angles", "dihedrals"
]


def _normal_vector(
    coordinates: torch.Tensor, angle_indices: torch.Tensor
) -> torch.Tensor:

    vector1 = coordinates[angle_indices[:, 0]] - coordinates[angle_indices[:, 1]]
    vector2 = coordinates[angle_indices[:, 2]] - coordinates[angle_indices[:, 1]]

    cross_product = torch.cross(vector1, vector2)

    return cross_product / torch.norm(cross_product, dim=-1).unsqueeze(-1)


class CoordinateSystem(abc.ABC):
    """The base for classes that interconvert between cartesian and internal coordinate
    systems."""

    @classmethod
    @abc.abstractmethod
    def detect_internal_coordinates(
        cls,
        coordinates: torch.Tensor,
        bond_indices: torch.Tensor,
    ) -> Dict[InternalCoordinateType, torch.Tensor]:
        """Determines the indices of the atoms involved in which type of internal
        coordinate.

        Args:
            coordinates: The coordinates of the molecules with shape=(n_atoms, 3) and
                units of [Å].
            bond_indices: The indices of the atoms involved in each bond with
                shape=(n_bonds, 2).

        Returns:
            A dictionary of the form ``indices[ic_type] = atom_indices`` where
            ``ic_type`` is a type of internal coordinate and ``atom_indices`` a 2D tensor
            of the atom indices involved in the internal coordinate.
        """
        raise NotImplementedError()

    @classmethod
    def cartesian_to_internal(
        cls,
        coordinates: torch.Tensor,
        bond_indices: Optional[torch.Tensor] = None,
        ic_indices: Optional[Dict[InternalCoordinateType, torch.Tensor]] = None,
    ):
        """Determines the indices of the atoms involved in which type of internal
        coordinate for a given internal coordinate system.

        Args:
            coordinates: The coordinates of the molecules with shape=(n_atoms, 3) and
                units of [Å].
            bond_indices: The indices of the atoms involved in each bond with
                shape=(n_bonds, 2). This argument is mutually exclusive with
                ``ic_indices``.
            ic_indices: The indices of atoms involved in each type of internal
                coordinates. See ``detect_internal_coordinates`` for more details. This
                argument is mutually exclusive with ``bond_indices``.

        Returns:
            A dictionary storing the internal coordinates of the form
            ``ic[ic_type] = (atom_indices, values)`` where ``ic_type`` is a type of
            internal coordinate, ``atom_indices`` a 2D tensor of the atom indices
            involved in the internal coordinate, and values a tensor of the values of
            each internal coordinate of that type. Distances will have units of [Å] and
            angles units of [rad].
        """

        assert (bond_indices is not None or ic_indices is not None) and (
            bond_indices is None or ic_indices is None
        ), "``bond_indices`` and ``ic_indices`` are mutually exclusive."

        if ic_indices is None:
            ic_indices = cls.detect_internal_coordinates(coordinates, bond_indices)

        ic_value_function = {
            "distances": lambda *args: compute_bond_vectors(*args)[1],
            "angles": compute_angles,
            "linear-angles": compute_linear_displacement,
            "out-of-plane-angles": compute_dihedrals,
            "dihedrals": compute_dihedrals,
        }

        return {
            ic_type: (
                ic_atom_indices,
                ic_value_function[ic_type](coordinates, ic_atom_indices),
            )
            for ic_type, ic_atom_indices in ic_indices.items()
        }


class RedundantInternalCoordinates(CoordinateSystem):
    """A class for interconverting between cartesian and reduced internal coordinates.

    Notes:
        * This class is heavily based off of the ``PrimitiveInternalCoordinates`` class
          in the ``geomTRIC`` package. See the main README for more information
          including license information.

    References:
        1. P. Pulay, G. Fogarasi, F. Pang, and J. E. Boggs J. Am. Chem. Soc. 1979, 101,
           10, 2550–2560
    """

    @classmethod
    def _detect_angles(
        cls, coordinates: torch.Tensor, graph: networkx.Graph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detects any non-linear and linear angle degrees within a graph representation
        of a molecule for a given conformer.

        Args:
            coordinates: The coordinates of the molecule with shape=(n_atoms, 3)
            graph: The associated molecule stored in a ``networkx`` graph object.

        Returns:
            The tuple of a tensor storing the indices of atoms that form any non-linear
            angles with shape=(n_angles, 3) and a tensor storing the indices of atoms
            that form any linear angles with shape=(n_terms, 4).
        """

        angle_indices = torch.tensor(
            [
                (a, b, c)
                for b in graph.nodes()
                for a in graph.neighbors(b)
                for c in graph.neighbors(b)
                if a < c
            ]
        )

        angles = compute_angles(coordinates, angle_indices)

        is_angle_linear = torch.abs(angles) >= torch.acos(
            torch.tensor(-_LINEAR_THRESHOLD)
        )

        linear_angle_indices = angle_indices[is_angle_linear]
        linear_angle_indices = torch.vstack(
            [
                torch.hstack(
                    [
                        linear_angle_indices,
                        torch.zeros(len(linear_angle_indices), 1, dtype=torch.int64),
                    ]
                ),
                torch.hstack(
                    [
                        linear_angle_indices,
                        torch.ones(len(linear_angle_indices), 1, dtype=torch.int64),
                    ]
                ),
            ]
        )

        angle_indices = angle_indices[~is_angle_linear]

        return angle_indices, linear_angle_indices

    @classmethod
    def _detect_out_of_plane_angles(
        cls,
        coordinates: torch.Tensor,
        graph: networkx.Graph,
        angle_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detects any non-linear and linear angle degrees within a graph representation
        of a molecule for a given conformer.

        Args:
            coordinates: The coordinates of the molecule with shape=(n_atoms, 3)
            graph: The associated molecule stored in a ``networkx`` graph object.
            angle_indices: A tensor storing the indices of atoms that form any non-linear
                angles with shape=(n_angles, 3)

        Returns:
            The tuple of a tensor storing the indices of atoms that form any non-linear
            angles with shape=(n_angles, 3) and a tensor storing the indices of atoms
            that form any out of plane (i.e. improper) angles with shape=(n_terms, 4).
        """

        improper_tuples = [
            (a, b, c, d)
            for b in graph.nodes()
            for a in graph.neighbors(b)
            for c in graph.neighbors(b)
            for d in graph.neighbors(b)
            if a < c < d
        ]

        out_of_plane_tuples, angles_to_remove = [], set()

        for a, b, c, d in improper_tuples:

            improper_indices = torch.tensor(
                [
                    (b, i, j, k)
                    for i, j, k in sorted(itertools.permutations([a, c, d], 3))
                ]
            )

            angles_a = compute_angles(coordinates, improper_indices[:, (0, 1, 2)])
            angles_b = compute_angles(coordinates, improper_indices[:, (1, 2, 3)])

            is_out_of_plane = (
                (torch.abs(torch.cos(angles_a)) <= _LINEAR_THRESHOLD)
                & (torch.abs(torch.cos(angles_b)) <= _LINEAR_THRESHOLD)
                & (
                    torch.abs(
                        torch.sum(
                            _normal_vector(coordinates, improper_indices[:, (0, 1, 2)])
                            * _normal_vector(
                                coordinates, improper_indices[:, (1, 2, 3)]
                            ),
                            dim=-1,
                        )
                    )
                    > _LINEAR_THRESHOLD
                )
            )

            if not torch.any(is_out_of_plane):
                continue

            out_of_plane_tuples.append(improper_indices[0, :])

            angle_to_remove = tuple(int(i) for i in improper_indices[0, (1, 0, 2)])

            angles_to_remove.add(angle_to_remove)
            angles_to_remove.add(tuple(reversed(angle_to_remove)))

        if len(out_of_plane_tuples) == 0:
            return angle_indices, torch.tensor([], dtype=torch.int64)

        angle_mask = torch.tensor(
            [
                tuple(int(i) for i in row) not in angles_to_remove
                for row in angle_indices
            ]
        )

        out_of_plane_indices = torch.vstack(out_of_plane_tuples)
        angle_indices = angle_indices[angle_mask]

        return angle_indices, out_of_plane_indices

    @classmethod
    def _detect_dihedrals(
        cls,
        coordinates: torch.Tensor,
        graph: networkx.Graph,
        linear_angle_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Detects any dihedral degrees within a graph representation of a molecule for
        a given conformer.

        Args:
            coordinates: The coordinates of the molecule with shape=(n_atoms, 3)
            graph: The associated molecule stored in a ``networkx`` graph object.

        Returns:
            The tensor storing the indices of atoms that form any dihedrals with
            shape=(n_dihedrals, 3).
        """

        # Compute all 'standard' dihedrals excluding linear dihedrals.
        dihedral_indices = torch.tensor(
            [
                (a, b, c, d)
                for (b, c) in graph.edges()
                for a in graph.neighbors(b)
                for d in graph.neighbors(c)
                if a != c and d != b
            ]
        )

        if len(dihedral_indices) == 0:
            return torch.tensor([], dtype=torch.int64)

        # Also compute dihedrals where the central bond is actually a linear chain
        # of atoms rather than a single bond.
        graph = graph.copy()

        linear_chain_edges = set()

        for node in {int(i) for i in linear_angle_indices[:, 1]}:

            chain_edge = tuple(graph.neighbors(node))

            graph.add_edge(*chain_edge)
            graph.remove_node(node)

            linear_chain_edges.add(chain_edge)

        chain_dihedral_indices = torch.tensor(
            [
                (a, b, c, d)
                for (b, c) in graph.edges()
                for a in graph.neighbors(b)
                for d in graph.neighbors(c)
                if a != c
                and d != b
                and (a, b) not in linear_chain_edges
                and (b, a) not in linear_chain_edges
                and (c, d) not in linear_chain_edges
                and (d, c) not in linear_chain_edges
            ]
        )

        if len(chain_dihedral_indices) > 0:

            dihedral_indices = torch.unique(
                torch.vstack([dihedral_indices, chain_dihedral_indices]), dim=0
            )

        angles_a = compute_angles(coordinates, dihedral_indices[:, (0, 1, 2)])
        angles_b = compute_angles(coordinates, dihedral_indices[:, (1, 2, 3)])

        # Remove linear dihedrals
        dihedral_mask = (torch.abs(torch.cos(angles_a)) < _LINEAR_THRESHOLD) & (
            torch.abs(torch.cos(angles_b)) < _LINEAR_THRESHOLD
        )

        dihedral_indices = dihedral_indices[dihedral_mask]

        return dihedral_indices

    @classmethod
    def detect_internal_coordinates(
        cls,
        coordinates: torch.Tensor,
        bond_indices: torch.Tensor,
    ) -> Dict[InternalCoordinateType, torch.Tensor]:

        # Construct a graph of the molecule to make iterating over all angles, impropers
        # and dihedrals easier.
        graph = networkx.Graph(bond_indices.detach().numpy().tolist())

        (
            angle_indices,
            linear_angle_indices,
        ) = cls._detect_angles(coordinates, graph)

        (
            angle_indices,
            out_of_plane_indices,
        ) = cls._detect_out_of_plane_angles(coordinates, graph, angle_indices)

        dihedral_indices = cls._detect_dihedrals(
            coordinates, graph, linear_angle_indices
        )

        return_value = {
            "distances": bond_indices,
            "angles": angle_indices,
            "linear-angles": linear_angle_indices,
            "out-of-plane-angles": out_of_plane_indices,
            "dihedrals": dihedral_indices,
        }

        return {key: value for key, value in return_value.items() if len(value) > 0}


_COORDINATE_SYSTEMS = {"ric": RedundantInternalCoordinates}


def detect_internal_coordinates(
    coordinates: torch.Tensor,
    bond_indices: torch.Tensor,
    coordinate_system: Literal["ric"] = "ric",
) -> Dict[InternalCoordinateType, torch.Tensor]:
    """Determines the indices of the atoms involved in which type of internal coordinate
    for a given internal coordinate system.

    Args:
        coordinates: The coordinates of the molecules with shape=(n_atoms, 3) and
            units of [Å].
        bond_indices: The indices of the atoms involved in each bond with
            shape=(n_bonds, 2).
        coordinate_system: The internal coordinate system.

    Returns:
        A dictionary of the form ``indices[ic_type] = atom_indices`` where ``ic_type``
        is a type of internal coordinate and ``atom_indices`` a 2D tensor of the atom
        indices involved in the internal coordinate.
    """

    if coordinate_system.lower() not in _COORDINATE_SYSTEMS:
        raise NotImplementedError()

    return _COORDINATE_SYSTEMS[coordinate_system.lower()].detect_internal_coordinates(
        coordinates, bond_indices
    )


def cartesian_to_internal(
    coordinates: torch.Tensor,
    bond_indices: Optional[torch.Tensor] = None,
    ic_indices: Optional[Dict[InternalCoordinateType, torch.Tensor]] = None,
    coordinate_system: Literal["ric"] = "ric",
):
    """Determines the indices of the atoms involved in which type of internal coordinate
    for a given internal coordinate system.

    Args:
        coordinates: The coordinates of the molecules with shape=(n_atoms, 3) and
            units of [Å].
        bond_indices: The indices of the atoms involved in each bond with
            shape=(n_bonds, 2). This argument is mutually exclusive with ``ic_indices``.
        ic_indices: The indices of atoms involved in each type of internal coordinates.
            See ``detect_internal_coordinates`` for more details. This argument is
            mutually exclusive with ``bond_indices``.
        coordinate_system: The internal coordinate system.

    Returns:
        A dictionary of the form ``indices[ic_type] = atom_indices`` where ``ic_type``
        is a type of internal coordinate and ``atom_indices`` a 2D tensor of the atom
        indices involved in the internal coordinate.
    """

    if coordinate_system.lower() not in _COORDINATE_SYSTEMS:
        raise NotImplementedError()

    return _COORDINATE_SYSTEMS[coordinate_system.lower()].cartesian_to_internal(
        coordinates, bond_indices, ic_indices
    )
