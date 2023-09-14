import openff.interchange
import openff.toolkit
import torch
from openff.units import unit

from smirnoffee.ff import convert_interchange
from smirnoffee.potentials import compute_energy


def main():
    # Load in a paracetamol molecule and generate a conformer for it.
    molecule = openff.toolkit.Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")
    molecule.generate_conformers(n_conformers=1)

    conformer = torch.tensor(molecule.conformers[0].m_as(unit.angstrom))

    # Parameterize the molecule
    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff_unconstrained-2.0.0.offxml"),
        molecule.to_topology(),
    )
    force_field, [topology] = convert_interchange(interchange)

    # Specify that we want to compute the gradient of the energy with respect to the
    # vdW parameters.
    vdw_potential = force_field.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True

    # Compute the energies and backpropagate to get gradient of the energy.
    energy = compute_energy(topology.parameters, conformer, force_field)
    energy.backward()

    # Print the gradients.
    for parameter_key, gradient in zip(
        vdw_potential.parameter_keys, vdw_potential.parameters.grad.numpy()
    ):
        parameter_cols = vdw_potential.parameter_cols

        parameter_grads = ", ".join(
            f"dU/d{parameter_col} = {parameter_grad: 8.3f}"
            for parameter_col, parameter_grad in zip(parameter_cols, gradient)
        )
        print(f"{parameter_key.id.ljust(15)} - {parameter_grads}")


if __name__ == "__main__":
    main()
