import openff.interchange
import openff.toolkit
import torch
from rdkit import Chem

import smee
import smee.converters


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


def system_from_smiles(
    smiles: list[str],
    n_copies: list[int],
    force_field: openff.toolkit.ForceField | None = None,
) -> tuple[smee.TensorSystem, smee.TensorForceField]:
    """Creates a system from a list of SMILES strings.

    Args:
        smiles: The list of SMILES strings.
        n_copies: The number of copies of each molecule.

    Returns:
        The system and force field.
    """
    force_field = (
        force_field
        if force_field is not None
        else openff.toolkit.ForceField("openff-2.0.0.offxml")
    )

    interchanges = [
        openff.interchange.Interchange.from_smirnoff(
            force_field,
            openff.toolkit.Molecule.from_smiles(pattern).to_topology(),
        )
        for pattern in smiles
    ]

    tensor_ff, tensor_tops = smee.converters.convert_interchange(interchanges)

    return smee.TensorSystem(tensor_tops, n_copies, is_periodic=True), tensor_ff
