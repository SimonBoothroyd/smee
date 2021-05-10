import numpy
import pytest
import torch
from openff.system.models import PotentialKey
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import unit

from smirnoffee.potentials import (
    _add_parameter_delta,
    _evaluate_cosine_torsion_energy,
    evaluate_cosine_improper_torsion_energy,
    evaluate_cosine_proper_torsion_energy,
    evaluate_handler_energy,
    evaluate_harmonic_angle_energy,
    evaluate_harmonic_bond_energy,
)
from smirnoffee.tests.utilities import (
    evaluate_openmm_energy,
    reduce_and_perturb_force_field,
)


def test_add_parameter_delta():

    parameters = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    parameter_ids = [("a", ("i", "j")), ("b", ("i", "j"))]

    delta = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True)
    delta_ids = [
        ("b", "i"),
        ("c", "k"),
        ("a", "j"),
        ("a", "i"),
        ("b", "j"),
        ("c", "l"),
    ]

    new_parameters = _add_parameter_delta(parameters, parameter_ids, delta, delta_ids)
    assert parameters.shape == new_parameters.shape

    expected_parameters = torch.tensor([[6.0, 6.0], [5.0, 10.0]])
    assert torch.allclose(new_parameters, expected_parameters)

    (new_parameters ** 2).sum().backward()

    expected_gradient = torch.tensor(
        [
            2.0 * (parameters[1, 0] + delta[0]),
            0.0,
            2.0 * (parameters[0, 1] + delta[2]),
            2.0 * (parameters[0, 0] + delta[3]),
            2.0 * (parameters[1, 1] + delta[4]),
            0.0,
        ]
    )
    assert torch.allclose(delta.grad, expected_gradient)


def test_evaluate_harmonic_bond_energy():

    conformer = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atom_indices = torch.tensor([[0, 1], [0, 2]])
    parameters = torch.tensor([[2.0, 0.95], [0.5, 1.01]], requires_grad=True)

    energy = evaluate_harmonic_bond_energy(conformer, atom_indices, parameters)
    energy.backward()

    assert torch.isclose(energy, torch.tensor(1.0 * 0.05 ** 2 + 0.25 * 0.01 ** 2))
    assert not torch.allclose(parameters.grad, torch.tensor(0.0))


def test_evaluate_harmonic_angle_energy():

    conformer = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atom_indices = torch.tensor([[0, 1, 2]])
    parameters = torch.tensor([[2.0, 92.5]], requires_grad=True)

    energy = evaluate_harmonic_angle_energy(conformer, atom_indices, parameters)
    energy.backward()

    expected_energy = 0.5 * parameters[0, 0] * (90.0 - parameters[0, 1]) ** 2
    expected_gradient = torch.tensor(
        [
            0.5 * (90.0 - parameters[0, 1]) ** 2,
            parameters[0, 0] * (parameters[0, 1] - 90.0),
        ]
    )

    assert torch.isclose(energy, expected_energy)
    assert torch.allclose(parameters.grad, expected_gradient)


@pytest.mark.parametrize(
    "energy_function",
    [
        _evaluate_cosine_torsion_energy,
        evaluate_cosine_proper_torsion_energy,
        evaluate_cosine_improper_torsion_energy,
    ],
)
@pytest.mark.parametrize("phi_sign", [-1.0, 1.0])
def test_evaluate_cosine_torsion_energy(energy_function, phi_sign):

    conformer = torch.tensor(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, phi_sign]]
    )
    atom_indices = torch.tensor([[0, 1, 2, 3]])
    parameters = torch.tensor([[2.0, 2.0, 20.0, 1.5]], requires_grad=True)

    energy = energy_function(conformer, atom_indices, parameters)
    energy.backward()

    expected_energy = (
        parameters[0, 0]
        / parameters[0, 3]
        * (
            1.0
            + torch.cos(
                torch.tensor(
                    [
                        parameters[0, 1] * torch.deg2rad(torch.tensor(phi_sign * 45.0))
                        - torch.deg2rad(parameters[0, 2])
                    ]
                )
            )
        )
    )

    assert torch.isclose(energy, expected_energy)


@pytest.mark.parametrize(
    "handler, delta, delta_ids, force_field, perturbed_force_field",
    [
        (
            "Bonds",
            None,
            None,
            reduce_and_perturb_force_field(ForceField("openff-1.0.0.offxml"), "Bonds"),
            reduce_and_perturb_force_field(ForceField("openff-1.0.0.offxml"), "Bonds"),
        ),
        (
            "Bonds",
            torch.tensor([0.1, 0.02], requires_grad=True),
            [
                (PotentialKey(id="[#6:1]-[#7:2]"), "k"),
                (PotentialKey(id="[#8:1]-[#1:2]"), "length"),
            ],
            reduce_and_perturb_force_field(ForceField("openff-1.0.0.offxml"), "Bonds"),
            reduce_and_perturb_force_field(
                ForceField("openff-1.0.0.offxml"),
                "Bonds",
                {("[#8:1]-[#1:2]", "length"): 0.02 * unit.angstrom},
            ),
        ),
        (
            "Angles",
            None,
            None,
            reduce_and_perturb_force_field(ForceField("openff-1.0.0.offxml"), "Angles"),
            reduce_and_perturb_force_field(ForceField("openff-1.0.0.offxml"), "Angles"),
        ),
        (
            "ProperTorsions",
            None,
            None,
            reduce_and_perturb_force_field(
                ForceField("openff-1.0.0.offxml"), "ProperTorsions"
            ),
            reduce_and_perturb_force_field(
                ForceField("openff-1.0.0.offxml"), "ProperTorsions"
            ),
        ),
        (
            "ImproperTorsions",
            None,
            None,
            reduce_and_perturb_force_field(
                ForceField("openff-1.0.0.offxml"), "ImproperTorsions"
            ),
            reduce_and_perturb_force_field(
                ForceField("openff-1.0.0.offxml"), "ImproperTorsions"
            ),
        ),
    ],
)
def test_evaluate_handler_energy(
    handler,
    delta,
    delta_ids,
    force_field,
    perturbed_force_field,
    ethanol,
    ethanol_conformer,
):

    openff_system = force_field.create_openff_system(ethanol.to_topology())

    openff_energy = evaluate_handler_energy(
        openff_system.handlers[handler], ethanol_conformer, delta, delta_ids
    )

    expected_energy = evaluate_openmm_energy(
        ethanol, ethanol_conformer.numpy(), perturbed_force_field
    )

    assert numpy.isclose(expected_energy, openff_energy.detach().numpy())

    if delta is not None and delta.requires_grad is not None:

        openff_energy.backward()
        assert not numpy.allclose(delta.grad, torch.zeros_like(delta.grad))
