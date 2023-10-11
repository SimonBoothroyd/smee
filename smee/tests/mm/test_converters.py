import numpy.random
import openff.interchange
import openff.toolkit
import openff.units
import openmm

import smee
from smee.mm._converters import convert_to_openmm_system, convert_to_openmm_topology


def _compute_energy(
    system: openmm.System,
    coords: openmm.unit.Quantity,
    box_vectors: openmm.unit.Quantity | None,
) -> float:
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )

    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)

    context.setPositions(coords)

    state = context.getState(getEnergy=True)

    return state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalories_per_mole)


def _compare_smee_and_interchange(
    tensor_ff: smee.ff.TensorForceField,
    tensor_system: smee.ff.TensorSystem,
    interchange: openff.interchange.Interchange,
    coords: openmm.unit.Quantity,
    box_vectors: openmm.unit.Quantity | None,
):
    system_smee = convert_to_openmm_system(tensor_ff, tensor_system)
    assert isinstance(system_smee, openmm.System)
    system_interchange = interchange.to_openmm(True, True)

    coords += (numpy.random.randn(*coords.shape) * 0.1) * openmm.unit.angstrom

    energy_smee = _compute_energy(system_smee, coords, box_vectors)
    energy_interchange = _compute_energy(system_interchange, coords, box_vectors)

    assert numpy.isclose(energy_smee, energy_interchange)


def test_convert_to_openmm_system_vacuum():
    # carbonic acid has impropers, 1-5 interactions so should test most convertors
    mol = openff.toolkit.Molecule.from_smiles("OC(=O)O")
    mol.generate_conformers(n_conformers=1)

    coords = mol.conformers[0].m_as(openff.units.unit.angstrom)
    coords = coords * openmm.unit.angstrom

    interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff-2.0.0.offxml"), mol.to_topology()
    )

    tensor_ff, [tensor_top] = smee.convert_interchange(interchange)

    _compare_smee_and_interchange(tensor_ff, tensor_top, interchange, coords, None)


def test_convert_to_openmm_system_periodic():
    ff = openff.toolkit.ForceField("openff-2.0.0.offxml")
    top = openff.toolkit.Topology()

    interchanges = []

    n_copies_per_mol = [5, 5]

    # carbonic acid has impropers, 1-5 interactions so should test most convertors
    for smiles, n_copies in zip(["OC(=O)O", "O"], n_copies_per_mol):
        mol = openff.toolkit.Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)

        interchange = openff.interchange.Interchange.from_smirnoff(
            ff, mol.to_topology()
        )
        interchanges.append(interchange)

        for _ in range(n_copies):
            top.add_molecule(mol)

    tensor_ff, tensor_tops = smee.convert_interchange(interchanges)
    tensor_system = smee.ff.TensorSystem(tensor_tops, n_copies_per_mol, True)

    coords, _ = smee.mm.generate_system_coords(
        tensor_system, smee.mm.GenerateCoordsConfig()
    )
    box_vectors = numpy.eye(3) * 20.0 * openmm.unit.angstrom

    top.box_vectors = box_vectors

    interchange_top = openff.interchange.Interchange.from_smirnoff(ff, top)

    _compare_smee_and_interchange(
        tensor_ff, tensor_system, interchange_top, coords, box_vectors
    )


def test_convert_to_openmm_topology():
    formaldehyde_interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff-2.0.0.offxml"),
        openff.toolkit.Molecule.from_smiles("C=O").to_topology(),
    )
    water_interchange = openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff-2.0.0.offxml"),
        openff.toolkit.Molecule.from_smiles("O").to_topology(),
    )

    tensor_ff, [methane_top, water_top] = smee.convert_interchange(
        [formaldehyde_interchange, water_interchange]
    )
    tensor_system = smee.ff.TensorSystem([methane_top, water_top], [1, 2], True)

    openmm_topology = convert_to_openmm_topology(tensor_system)

    assert openmm_topology.getNumChains() == 2
    assert openmm_topology.getNumResidues() == 3  # 1 methane, 2 water

    residue_names = [residue.name for residue in openmm_topology.residues()]
    assert residue_names == ["UNK", "WAT", "WAT"]

    atom_names = [atom.name for atom in openmm_topology.atoms()]
    expected_atom_names = [
        "C",
        "O",
        "H1",
        "H2",
        "O",
        "H1",
        "H2",
        "O",
        "H1",
        "H2",
    ]
    assert atom_names == expected_atom_names

    bond_idxs = [
        (bond.atom1.index, bond.atom2.index, bond.order)
        for bond in openmm_topology.bonds()
    ]
    expected_bond_idxs = [
        (0, 1, 2),
        (0, 2, 1),
        (0, 3, 1),
        (4, 5, 1),
        (4, 6, 1),
        (7, 8, 1),
        (7, 9, 1),
    ]
    assert bond_idxs == expected_bond_idxs
