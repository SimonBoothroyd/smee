import copy

import numpy
import pytest
import torch
from openff.system.components.system import System
from openff.system.models import PotentialKey
from openff.toolkit.topology import Molecule
from simtk import unit

from smirnoffee.exceptions import MissingArgumentsError
from smirnoffee.potentials import _POTENTIAL_ENERGY_FUNCTIONS, potential_energy_function
from smirnoffee.potentials.potentials import (
    add_parameter_delta,
    evaluate_handler_energy,
    evaluate_system_energy,
)
from smirnoffee.smirnoff import _get_parameter_value
from smirnoffee.tests.utilities import (
    evaluate_openmm_energy,
    reduce_and_perturb_force_field,
)

_DEFAULT_SIMTK_UNITS = {
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


def test_potential_energy_function_decorator():

    potential_energy_function("DummyHandler", "x")(lambda x: None)
    assert ("DummyHandler", "x") in _POTENTIAL_ENERGY_FUNCTIONS

    with pytest.raises(KeyError, match="A potential energy function is already"):
        potential_energy_function("DummyHandler", "x")(lambda x: None)

    del _POTENTIAL_ENERGY_FUNCTIONS[("DummyHandler", "x")]


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


@pytest.mark.parametrize("energy_function", [*_POTENTIAL_ENERGY_FUNCTIONS.values()])
def test_potential_energy_functions_no_atoms(energy_function):
    """Tests that each of the registered potential energy functions can handle the
    case of no matched atoms."""

    actual_energy = energy_function(
        torch.tensor([]), torch.tensor([]), torch.tensor([])
    )
    expected_energy = torch.zeros(1)

    assert torch.isclose(actual_energy, expected_energy)


@pytest.mark.parametrize(
    "molecule_name",
    ["ethanol", "formaldehyde"],
)
@pytest.mark.parametrize(
    "handler",
    ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions", "vdW", "Electrostatics"],
)
def test_evaluate_handler_energy(request, handler, molecule_name, default_force_field):

    molecule = request.getfixturevalue(molecule_name)
    conformer = request.getfixturevalue(f"{molecule_name}_conformer")

    force_field = reduce_and_perturb_force_field(default_force_field, handler)

    openff_system = System.from_smirnoff(force_field, molecule.to_topology())

    openff_energy = evaluate_handler_energy(
        openff_system.handlers[handler], molecule, conformer
    )
    expected_energy = evaluate_openmm_energy(molecule, conformer.numpy(), force_field)

    if handler == "ImproperTorsions":
        assert numpy.isclose(expected_energy, openff_energy.numpy(), atol=1e-6)
    else:
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
    openff_system = System.from_smirnoff(force_field, molecule.to_topology())

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
        {
            key: value * _DEFAULT_SIMTK_UNITS[handler][key[1]]
            for key, value in deltas.items()
        },
    )
    expected_energy = evaluate_openmm_energy(
        molecule, conformer.numpy(), perturbed_force_field
    )

    delta_ids, delta_values = zip(*deltas.items())
    delta_values = torch.tensor(delta_values, requires_grad=True)

    openff_energy = evaluate_handler_energy(
        openff_system.handlers[handler], molecule, conformer, delta_values, delta_ids
    )

    if handler == "ImproperTorsions":
        assert numpy.isclose(expected_energy, openff_energy.detach().numpy(), atol=1e-4)
    else:
        assert numpy.isclose(expected_energy, openff_energy.numpy())

    openff_energy.backward()
    assert not numpy.allclose(delta_values.grad, torch.zeros_like(delta_values.grad))


def test_evaluate_handler_energy_missing_delta(
    ethanol_system, ethanol, ethanol_conformer
):

    with pytest.raises(MissingArgumentsError):

        evaluate_handler_energy(
            ethanol_system.handlers["Bonds"], ethanol, ethanol_conformer, torch.zeros(1)
        )

    with pytest.raises(MissingArgumentsError):

        evaluate_handler_energy(
            ethanol_system.handlers["Bonds"],
            ethanol,
            ethanol_conformer,
            parameter_delta_ids=[],
        )


@pytest.mark.parametrize("molecule_name", ["ethanol", "formaldehyde"])
def test_evaluate_system_energy(request, default_force_field, molecule_name):

    molecule = request.getfixturevalue(f"{molecule_name}")
    conformer = request.getfixturevalue(f"{molecule_name}_conformer")

    openff_system = request.getfixturevalue(f"{molecule_name}_system")

    openff_energy = evaluate_system_energy(openff_system, conformer)

    expected_energy = evaluate_openmm_energy(
        molecule, conformer.numpy(), default_force_field
    )

    assert numpy.isclose(expected_energy, openff_energy.detach().numpy())


@pytest.mark.parametrize("molecule_name", ["ethanol", "formaldehyde"])
def test_evaluate_system_energy_delta(
    request,
    molecule_name,
    default_force_field,
):

    molecule = request.getfixturevalue(molecule_name)
    conformer = request.getfixturevalue(f"{molecule_name}_conformer")

    openff_system = request.getfixturevalue(f"{molecule_name}_system")

    deltas = {
        (handler, key, parameter): _get_parameter_value(potential, handler, parameter)
        * 0.01
        for handler in ["Bonds", "Angles"]
        for key, potential in openff_system.handlers[handler].potentials.items()
        for parameter in potential.parameters
        if parameter not in ["periodicity", "idivf"]
    }

    perturbed_force_field = copy.deepcopy(default_force_field)

    for (handler, smirks, attribute), delta in deltas.items():

        delta = delta * _DEFAULT_SIMTK_UNITS[handler][attribute]

        if smirks.mult is not None:
            attribute = f"{attribute}{smirks.mult + 1}"

        smirks = smirks.id

        parameter = perturbed_force_field.get_parameter_handler(handler).parameters[
            smirks
        ]
        setattr(parameter, attribute, getattr(parameter, attribute) + delta)

    expected_energy = evaluate_openmm_energy(
        molecule, conformer.numpy(), perturbed_force_field
    )

    delta_ids, delta_values = zip(*deltas.items())
    delta_values = torch.tensor(delta_values, requires_grad=True)

    openff_energy = evaluate_system_energy(
        openff_system, conformer, delta_values, delta_ids
    )

    assert numpy.isclose(expected_energy, openff_energy.detach().numpy())

    openff_energy.backward()
    assert not numpy.allclose(delta_values.grad, torch.zeros_like(delta_values.grad))


def test_evaluate_system_missing_delta(ethanol_system, ethanol_conformer):

    with pytest.raises(MissingArgumentsError):

        evaluate_system_energy(
            ethanol_system, ethanol_conformer, parameter_delta=torch.zeros(1)
        )

    with pytest.raises(MissingArgumentsError):
        evaluate_system_energy(
            ethanol_system, ethanol_conformer, parameter_delta_ids=[]
        )
