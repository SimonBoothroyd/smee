import dataclasses

import openff.interchange.models
import openff.toolkit
import torch


@dataclasses.dataclass
class ParameterMap:
    """A maps between atom indices part of a particular interaction (e.g. torsion
    indices) and the corresponding parameter in a ``TensorPotential``"""

    atom_idxs: torch.Tensor
    """The indices of the atoms involved in an interaction with
    ``shape=(n_interactions, n_atoms)``
    """

    parameter_idxs: torch.Tensor
    """The indices of the parameters to use when computing the energy of an interaction
    with ``shape=(n_interactions, n_parameters)``, where ``n_parameters`` is ``1`` for
    valence interactions and ``2`` for non-bonded interactions.
    """
    global_parameter_idxs: torch.Tensor | None = None
    """The indices of the global parameters to use when computing the energy of an
    interaction with ``shape=(n_interactions, 1)``.
    """


AppliedParameters = dict[str, ParameterMap]


@dataclasses.dataclass
class TensorPotential:
    """A tensor representation of a SMIRNOFF parameter handler"""

    type: str
    """The type of handler associated with these parameters"""
    fn: str
    """The associated potential energy function"""

    parameters: torch.Tensor
    """The values of the parameters with ``shape=(n_parameters, n_parameter_cols)``"""
    parameter_keys: list[openff.interchange.models.PotentialKey]
    """Unique keys associated with each parameter with ``length=(n_parameters)``"""
    parameter_cols: tuple[str, ...]
    """The names of each column of ``parameters``."""

    global_parameters: torch.Tensor | None = None
    """The parameters defined on a handler such as 1-4 scaling factors with
    ``shape=(n_global_parameters, n_global_parameter_cols)``"""
    global_parameter_cols: tuple[str, ...] | None = None
    """The names of each column of ``global_parameters``."""


@dataclasses.dataclass
class TensorForceField:
    """A tensor representation of a SMIRNOFF force field."""

    potentials: list[TensorPotential]
    """The terms and associated parameters of the potential energy function."""

    @property
    def potentials_by_type(self) -> dict[str, TensorPotential]:
        potentials = {potential.type: potential for potential in self.potentials}
        assert len(potentials) == len(self.potentials), "duplicate potentials found"

        return potentials

    def update_parameters(
        self,
        parameters: torch.Tensor,
        parameter_keys: list[openff.interchange.models.PotentialKey],
    ):
        pass
