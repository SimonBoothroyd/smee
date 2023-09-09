import openff.interchange
import openff.toolkit
import openff.units
import torch
import torch.optim

from smirnoffee.ff.smirnoff import convert_interchange
from smirnoffee.potentials import evaluate_energy


def main():
    # Load in a paracetamol molecule, generate a conformer for it, and perturb the
    # conformer to ensure it needs minimization.
    molecule = openff.toolkit.Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")
    molecule.generate_conformers(n_conformers=1)
    molecule.to_file("initial.xyz", "XYZ")

    conformer = (
        torch.tensor(molecule.conformers[0].m_as(openff.units.unit.angstrom)) * 1.10
    )
    conformer.requires_grad = True

    # Parameterize the molecule using OpenFF Interchange
    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff_unconstrained-1.0.0.offxml"),
        molecule.to_topology(),
    )

    # Convert the interchange object into a pytorch tensor representation
    force_field, [applied_parameters] = convert_interchange(interchange)

    # Minimize the conformer
    optimizer = torch.optim.Adam([conformer], lr=0.02)

    for epoch in range(75):
        energy = evaluate_energy(applied_parameters, conformer, force_field)
        energy.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}: E={energy.item()} kJ / mol")

    # Save the final conformer
    molecule._conformers = [conformer.detach().numpy() * openff.units.unit.angstrom]
    molecule.to_file("final.xyz", "XYZ")


if __name__ == "__main__":
    main()
