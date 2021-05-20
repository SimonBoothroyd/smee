import torch
from openff.system.components.system import System
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from simtk import unit
from torch import optim

from smirnoffee.potentials.potentials import evaluate_system_energy


def main():

    # Load in a paracetamol molecules, generate a conformer for it, and perturb the
    # conformer to ensure it needs minimization.
    molecule: Molecule = Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")
    molecule.generate_conformers(n_conformers=1)
    molecule.to_file("initial.xyz", "XYZ")

    conformer = torch.tensor(molecule.conformers[0].value_in_unit(unit.angstrom)) * 1.10
    conformer.requires_grad = True

    # Parameterize the molecule
    openff_system = System.from_smirnoff(
        ForceField("openff_unconstrained-1.0.0.offxml"), molecule.to_topology()
    )

    # Minimize the conformer
    optimizer = optim.Adam([conformer], lr=0.02)

    for epoch in range(75):

        energy = evaluate_system_energy(openff_system, conformer)
        energy.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}: E={energy.item()} kJ / mol")

    # Save the final conformer
    molecule._conformers = [conformer.detach().numpy() * unit.angstrom]
    molecule.to_file("final.xyz", "XYZ")


if __name__ == "__main__":
    main()
