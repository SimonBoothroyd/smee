import copy
from typing import Dict, Tuple

import numpy
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import unit


def reduce_and_perturb_force_field(
    force_field: ForceField,
    handler: str,
    deltas: Dict[Tuple[str, str], unit.Quantity] = None,
) -> ForceField:

    deltas = deltas if deltas is not None else {}

    force_field = copy.deepcopy(force_field)

    for handler_name in force_field.registered_parameter_handlers:

        if handler_name == handler:
            continue

        force_field.deregister_parameter_handler(handler_name)

    if handler != "vdW":

        vdw_handler = force_field.get_parameter_handler("vdW")
        vdw_handler.add_parameter(
            {
                "smirks": "[*:1]",
                "epsilon": 0.0 * unit.kilojoules_per_mole,
                "sigma": 1.0 * unit.angstrom,
            }
        )

    for (smirks, attribute), delta in deltas.items():

        parameter = force_field.get_parameter_handler(handler).parameters[smirks]
        setattr(parameter, attribute, getattr(parameter, attribute) + delta)

    return force_field


def evaluate_openmm_energy(
    molecule: Molecule, conformer: numpy.ndarray, force_field: ForceField
) -> float:
    """Evaluate the potential energy of a molecule in a specified conformer using a
    specified force field.

    Args:
        molecule: The molecule whose energy should be evaluated
        conformer: The conformer [Angstrom] of the molecule.
        force_field: The force field defining the potential energy function.

    Returns:
        The energy in units of [kJ / mol].
    """

    from simtk import openmm
    from simtk import unit as simtk_unit

    openmm_system = force_field.create_openmm_system(molecule.to_topology())

    integrator = openmm.VerletIntegrator(0.1 * simtk_unit.femtoseconds)

    openmm_platform = openmm.Platform.getPlatformByName("Reference")
    openmm_context = openmm.Context(openmm_system, integrator, openmm_platform)

    openmm_context.setPositions(
        (conformer * simtk_unit.angstrom).value_in_unit(simtk_unit.nanometers)
    )

    state = openmm_context.getState(getEnergy=True)

    potential_energy = state.getPotentialEnergy()

    return potential_energy.value_in_unit(simtk_unit.kilojoules_per_mole)
