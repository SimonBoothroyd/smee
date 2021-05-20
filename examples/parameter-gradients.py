import torch
from openff.system.components.system import System
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import unit

from smirnoffee.potentials.potentials import evaluate_system_energy


def main():

    # Load in a paracetamol molecule and generate a conformer for it.
    molecule: Molecule = Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")
    molecule.generate_conformers(n_conformers=1)
    molecule.to_file("initial.xyz", "XYZ")

    conformer = torch.tensor(molecule.conformers[0].value_in_unit(unit.angstrom)) * 1.10

    # Parameterize the molecule
    openff_system = System.from_smirnoff(
        ForceField("openff_unconstrained-1.0.0.offxml"), molecule.to_topology()
    )

    # Specify the parameters that we want to differentiate respect to, as well as a
    # tensor that the computed gradients will be attached to.
    parameter_ids = [
        (handler_type, potential_key, attribute)
        for handler_type, handler in openff_system.handlers.items()
        # Computing the gradients w.r.t. nonbonded handlers is not yet supported.
        if handler_type not in ["Electrostatics", "vdW", "ToolkitAM1BCC"]
        for potential_key, potential in handler.potentials.items()
        for attribute in potential.parameters
    ]
    parameter_delta = torch.zeros(len(parameter_ids), requires_grad=True)

    # Compute the energies and backpropagate to get gradient of the energy with respect
    # to the parameters specified above.
    energy = evaluate_system_energy(
        openff_system,
        conformer,
        parameter_delta=parameter_delta,
        parameter_delta_ids=parameter_ids
    )
    energy.backward()

    # Print the gradients.
    for parameter_id, gradient in zip(parameter_ids, parameter_delta.grad.numpy()):
        print(f"{parameter_id} - {gradient}")


if __name__ == "__main__":
    main()
