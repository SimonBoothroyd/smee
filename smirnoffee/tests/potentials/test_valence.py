import numpy
import pytest
import torch
from openff.system.models import PotentialKey
from simtk import unit

from smirnoffee.potentials.valence import (
    _evaluate_cosine_torsion_energy,
    add_parameter_delta,
    evaluate_cosine_improper_torsion_energy,
    evaluate_cosine_proper_torsion_energy,
    evaluate_harmonic_angle_energy,
    evaluate_harmonic_bond_energy,
    evaluate_valence_energy,
)
from smirnoffee.smirnoff import _get_parameter_value
from smirnoffee.tests.utilities import (
    evaluate_openmm_energy,
    reduce_and_perturb_force_field,
)

_DEFAULT_UNITS = {
    "Bonds": {
        "k": unit.kilojoules / unit.mole / unit.angstrom ** 2,
        "length": unit.angstrom,
    },
    "Angles": {
        "k": unit.kilojoules / unit.mole / unit.degree ** 2,
        "angle": unit.degree,
    },
    "ProperTorsions": {
        "k": unit.kilojoules / unit.mole,
        "periodicity": unit.dimensionless,
        "phase": unit.degree,
        "idivf": unit.dimensionless,
    },
    "ImproperTorsions": {
        "k": unit.kilojoules / unit.mole,
        "periodicity": unit.dimensionless,
        "phase": unit.degree,
        "idivf": unit.dimensionless,
    },
    "vdW": {
        "epsilon": unit.kilojoules / unit.mole,
        "sigma": unit.angstrom,
    },
}


def test_add_parameter_delta():

    parameters = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    parameter_ids = [
        (PotentialKey(id="a"), ("i", "j")),
        (PotentialKey(id="b"), ("i", "j")),
    ]

    delta = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True)
    delta_ids = [
        (PotentialKey(id="b"), "i"),
        (PotentialKey(id="c"), "k"),
        (PotentialKey(id="a"), "j"),
        (PotentialKey(id="a"), "i"),
        (PotentialKey(id="b"), "j"),
        (PotentialKey(id="c"), "l"),
    ]

    new_parameters = add_parameter_delta(parameters, parameter_ids, delta, delta_ids)
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
    "molecule_name",
    ["ethanol", "formaldehyde"],
)
@pytest.mark.parametrize(
    "handler",
    ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"],
)
def test_evaluate_handler_energy(request, handler, molecule_name, default_force_field):

    molecule = request.getfixturevalue(molecule_name)
    conformer = request.getfixturevalue(f"{molecule_name}_conformer")

    force_field = reduce_and_perturb_force_field(default_force_field, handler)

    openff_system = force_field.create_openff_system(molecule.to_topology())

    openff_energy = evaluate_valence_energy(openff_system.handlers[handler], conformer)
    expected_energy = evaluate_openmm_energy(molecule, conformer.numpy(), force_field)

    assert numpy.isclose(expected_energy, openff_energy.numpy())


@pytest.mark.parametrize(
    "molecule_name",
    ["ethanol", "formaldehyde"],
)
@pytest.mark.parametrize(
    "handler",
    ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"],
)
def test_evaluate_handler_energy_delta(
    request,
    handler,
    molecule_name,
    default_force_field,
):

    molecule = request.getfixturevalue(molecule_name)
    conformer = request.getfixturevalue(f"{molecule_name}_conformer")

    force_field = reduce_and_perturb_force_field(default_force_field, handler)
    openff_system = force_field.create_openff_system(molecule.to_topology())

    deltas = {
        (key, parameter): _get_parameter_value(potential, handler, parameter) * 0.01
        for key, potential in openff_system.handlers[handler].potentials.items()
        for parameter in potential.parameters
        if parameter not in ["periodicity", "idivf"]
    }

    if len(deltas) == 0:
        pytest.skip(f"{molecule_name} has no {handler} parameters")

    perturbed_force_field = reduce_and_perturb_force_field(
        default_force_field,
        handler,
        {key: value * _DEFAULT_UNITS[handler][key[1]] for key, value in deltas.items()},
    )
    expected_energy = evaluate_openmm_energy(
        molecule, conformer.numpy(), perturbed_force_field
    )

    delta_ids, delta_values = zip(*deltas.items())
    delta_values = torch.tensor(delta_values, requires_grad=True)

    openff_energy = evaluate_valence_energy(
        openff_system.handlers[handler], conformer, delta_values, delta_ids
    )

    assert numpy.isclose(expected_energy, openff_energy.detach().numpy())

    openff_energy.backward()
    assert not numpy.allclose(delta_values.grad, torch.zeros_like(delta_values.grad))
