import numpy
import torch
from openff.toolkit.typing.engines.smirnoff import ForceField

from smirnoffee.potentials.nonbonded import (
    _COULOMB_PRE_FACTOR,
    evaluate_coulomb_energy,
    evaluate_nonbonded_energy,
)
from smirnoffee.tests.utilities import (
    evaluate_openmm_energy,
    reduce_and_perturb_force_field,
)


def test_coulomb_pre_factor():

    # Compare against a value computed directly from C++ using the OpenMM 7.5.1
    # ONE_4PI_EPS0 define constant multiplied by 10 for nm -> A
    assert numpy.isclose(_COULOMB_PRE_FACTOR, 1389.3545764, atol=1.0e-7)


def test_evaluate_coulomb_energy_two_particle():

    scale_factor = 5.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atom_indices = torch.tensor([[0, 1]])

    parameters = torch.tensor([[0.25, 0.75, scale_factor]])

    actual_energy = evaluate_coulomb_energy(conformer, atom_indices, parameters)
    expected_energy = _COULOMB_PRE_FACTOR * 0.25 * 0.75 * 5.0 / 2.0

    assert torch.isclose(torch.tensor(expected_energy), actual_energy)


def test_evaluate_coulomb_energy_three_particle():

    scale_factor = 5.0

    conformer = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    atom_indices = torch.tensor([[0, 2], [0, 1], [1, 2]])

    parameters = torch.tensor(
        [
            [0.25, 0.75, scale_factor],
            [0.25, 0.50, scale_factor],
            [0.50, 0.75, scale_factor],
        ]
    )

    actual_energy = evaluate_coulomb_energy(conformer, atom_indices, parameters)

    expected_energy = torch.tensor(
        _COULOMB_PRE_FACTOR
        * 5.0
        * (0.25 * 0.75 / 4.0 + 0.25 * 0.50 / 3.0 + 0.50 * 0.75 / 5.0),
    )

    assert torch.isclose(expected_energy, actual_energy)


def test_evaluate_coulomb_energy_ethanol(
    default_force_field,
    ethanol,
    ethanol_conformer,
):

    force_field = reduce_and_perturb_force_field(
        ForceField("openff-1.0.0.offxml"), "ToolkitAM1BCC"
    )

    openff_system = force_field.create_openff_system(ethanol.to_topology())

    openff_energy = evaluate_nonbonded_energy(
        openff_system.handlers["Electrostatics"], ethanol, ethanol_conformer
    )

    expected_energy = evaluate_openmm_energy(
        ethanol, ethanol_conformer.numpy(), force_field
    )

    assert numpy.isclose(expected_energy, openff_energy.detach().numpy())
