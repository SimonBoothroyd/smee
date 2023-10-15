import torch
from rdkit import Chem

import smee


def topology_from_smiles(smiles: str) -> smee.TensorTopology:
    """Creates a topology with no parameters from a SMILES string.

    Args:
        smiles: The SMILES string.

    Returns:
        The topology.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    return smee.TensorTopology(
        atomic_nums=torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
        formal_charges=torch.tensor(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        ),
        bond_idxs=torch.tensor(
            [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
        ),
        bond_orders=torch.tensor(
            [int(bond.GetBondTypeAsDouble()) for bond in mol.GetBonds()]
        ),
        parameters={},
        v_sites=None,
        constraints=None,
    )
