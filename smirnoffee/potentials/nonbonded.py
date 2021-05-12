from typing import List, Optional, Tuple

import torch
from openff.system.components.potentials import PotentialHandler
from openff.system.models import PotentialKey
from openff.toolkit.topology import Molecule
from openff.units import unit

from smirnoffee.potentials import _POTENTIAL_ENERGY_FUNCTIONS, potential_energy_function
from smirnoffee.smirnoff import vectorize_nonbonded_handler

_COULOMB_PRE_FACTOR_UNITS = unit.kilojoule / unit.mole * unit.angstrom / unit.e ** 2
_COULOMB_PRE_FACTOR = (
    (unit.avogadro_constant / (4.0 * unit.pi * unit.eps0))
    .to(_COULOMB_PRE_FACTOR_UNITS)
    .magnitude
)


@potential_energy_function("vdW", "4*epsilon*((sigma/r)**12-(sigma/r)**6)")
def evaluate_lj_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of the vdW interactions using the
    standard Lennard-Jones potential.

    Notes:
        * No cutoff will be applied - this is consistent with OpenFF toolkit
          using the OpenMM `NoCutoff` method when creating an OpenMM system for
          a molecule in vacuum.

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The pairs of atom indices of the atoms involved in each
            interaction with shape=(n_pairs, 2).
        parameters: A tensor with shape=(n_pairs, 3) where there first and second column
            contains the epsilon [kJ / mol] and sigma [A] values of each pair
            respectively, and the third column a scale factor.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    directions = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]]
    distances_sqr = (directions * directions).sum(dim=1)

    # TODO: Do you get any noticeable speed-up for x2 -> x4 -> x6 vs pow?
    inverse_distances_6 = torch.rsqrt(distances_sqr) ** 6
    inverse_distances_12 = inverse_distances_6 * inverse_distances_6

    sigma_6 = parameters[:, 1] ** 6
    sigma_12 = sigma_6 * sigma_6

    return (
        4.0
        * parameters[:, 0]
        * parameters[:, 2]
        * (sigma_12 * inverse_distances_12 - sigma_6 * inverse_distances_6)
    ).sum()


@potential_energy_function("Electrostatics", "coul")
def evaluate_coulomb_energy(
    conformer: torch.Tensor,
    atom_indices: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] of the electrostatic interactions
    using the standard Coulomb potential.

    Notes:
        * No cutoff will be applied - this is consistent with OpenFF toolkit
          using the OpenMM `NoCutoff` method when creating an OpenMM system for
          a molecule in vacuum.

    Args:
        conformer: The conformer to evaluate the potential at.
        atom_indices: The pairs of atom indices of the atoms involved in each
            interaction with shape=(n_pairs, 2).
        parameters: A tensor with shape=(n_pairs, 2) where there first column
            contains the charge [e] on the first atom, the second column the charge [e]
            on the second atom, and the third column a scale factor.

    Returns:
        The evaluated potential energy [kJ / mol].
    """

    directions = conformer[atom_indices[:, 1]] - conformer[atom_indices[:, 0]]
    distances_sqr = (directions * directions).sum(dim=1)

    inverse_distances = torch.rsqrt(distances_sqr)

    return (
        _COULOMB_PRE_FACTOR
        * parameters[:, 0]
        * parameters[:, 1]
        * parameters[:, 2]
        * inverse_distances
    ).sum()


def evaluate_nonbonded_energy(
    nonbonded_handler: PotentialHandler,
    molecule: Molecule,
    conformer: torch.Tensor,
    parameter_delta: Optional[torch.Tensor] = None,
    parameter_delta_ids: Optional[List[Tuple[PotentialKey, str]]] = None,
) -> torch.Tensor:
    """Evaluates the potential energy [kJ / mol] contribution of a particular nonbonded
    potential handler for a given conformer.

    Args:
        nonbonded_handler: The nonbonded potential handler that encodes the potential
            energy function to evaluate.
        molecule: The molecule associated with the handler.
        conformer: The conformer to evaluate the potential at.
        parameter_delta: An optional tensor of values to perturb the assigned
            valence parameters by before evaluating the potential energy. If this
            option is specified then ``parameter_delta_ids`` must also be.
        parameter_delta_ids: An optional list of ids associated with the
            ``parameter_delta`` tensor which is used to identify which parameter
            delta matches which assigned parameter in the ``valence_handler``. If this
            option is specified then ``parameter_delta`` must also be.

    Returns:
        The potential energy of the conformer [kJ / mol].
    """

    if parameter_delta is not None or parameter_delta_ids is not None:
        raise NotImplementedError()

    indices, parameters, parameter_ids = vectorize_nonbonded_handler(
        nonbonded_handler, molecule
    )

    energy_expression = _POTENTIAL_ENERGY_FUNCTIONS[
        (nonbonded_handler.name, nonbonded_handler.expression)
    ]
    handler_energy = energy_expression(conformer, indices, parameters)

    return handler_energy
