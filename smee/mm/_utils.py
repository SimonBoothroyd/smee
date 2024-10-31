from rdkit import Chem
from rdkit.Chem import AllChem

import smee


def topology_to_rdkit(topology: smee.TensorTopology) -> Chem.Mol:
    """Convert a topology to an RDKit molecule."""
    mol = Chem.RWMol()

    for atomic_num, formal_charge in zip(
        topology.atomic_nums, topology.formal_charges, strict=True
    ):
        atom = Chem.Atom(int(atomic_num))
        atom.SetFormalCharge(int(formal_charge))
        mol.AddAtom(atom)

    for bond_idxs, bond_order in zip(
        topology.bond_idxs, topology.bond_orders, strict=True
    ):
        idx_a, idx_b = int(bond_idxs[0]), int(bond_idxs[1])
        mol.AddBond(idx_a, idx_b, Chem.BondType(bond_order))

    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache()

    AllChem.EmbedMolecule(mol)

    return mol
