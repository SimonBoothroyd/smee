import typing

import openff.interchange
import openff.toolkit
import openff.units
import torch
from rdkit import Chem

import smee
import smee.converters
import smee.potentials.nonbonded
import smee.utils

LJParam = typing.NamedTuple("LJParam", [("eps", float), ("sig", float)])


def convert_lj_to_dexp(potential: smee.TensorPotential):
    potential.fn = smee.potentials.nonbonded.DEXP_POTENTIAL

    parameter_cols = [*potential.parameter_cols]
    sigma_idx = potential.parameter_cols.index("sigma")

    sigma = potential.parameters[:, sigma_idx]
    r_min = 2 ** (1 / 6) * sigma

    potential.parameters[:, sigma_idx] = r_min

    parameter_cols[sigma_idx] = "r_min"
    potential.parameter_cols = tuple(parameter_cols)

    potential.attribute_cols = (*potential.attribute_cols, "alpha", "beta")
    potential.attributes = torch.cat([potential.attributes, torch.tensor([16.5, 5.0])])

    return potential


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


def _parameter_key_to_idx(potential: smee.TensorPotential, key: str):
    return next(iter(i for i, k in enumerate(potential.parameter_keys) if k.id == key))


def system_with_exceptions() -> (
    tuple[smee.TensorSystem, smee.TensorPotential, dict[str, LJParam]]
):
    ff = openff.toolkit.ForceField()

    kcal = openff.units.unit.kilocalorie / openff.units.unit.mole
    ang = openff.units.unit.angstrom

    eps_h, sig_h = 0.123, 1.234
    eps_o, sig_o = 0.234, 2.345
    eps_a, sig_a = 0.345, 3.456

    eps_hh, sig_hh = 0.3, 1.4
    eps_ah, sig_ah = 0.4, 2.5

    lj_handler = ff.get_parameter_handler("vdW")
    lj_handler.add_parameter(
        {"smirks": "[O:1]", "epsilon": eps_o * kcal, "sigma": sig_o * ang}
    )
    lj_handler.add_parameter(
        {"smirks": "[H:1]", "epsilon": eps_h * kcal, "sigma": sig_h * ang}
    )
    lj_handler.add_parameter(
        {"smirks": "[Ar:1]", "epsilon": eps_a * kcal, "sigma": sig_a * ang}
    )

    system, tensor_ff = smee.tests.utils.system_from_smiles(["O", "[Ar]"], [1, 1], ff)
    system.is_periodic = False

    lj_potential = tensor_ff.potentials_by_type["vdW"]
    lj_potential.attributes[lj_potential.attribute_cols.index("scale_12")] = 1.0
    lj_potential.attributes[lj_potential.attribute_cols.index("scale_13")] = 1.0

    parameter_idx_h = _parameter_key_to_idx(lj_potential, "[H:1]")
    parameter_idx_a = _parameter_key_to_idx(lj_potential, "[Ar:1]")

    assert lj_potential.parameter_cols[0] == "epsilon"

    lj_potential.parameters = torch.vstack(
        [lj_potential.parameters, torch.tensor([[eps_ah, sig_ah], [eps_hh, sig_hh]])]
    )
    lj_potential.parameter_keys = [*lj_potential.parameter_keys, "ar-h", "h-h"]

    lj_potential.exceptions = {
        (parameter_idx_a, parameter_idx_h): 3,
        (parameter_idx_h, parameter_idx_h): 4,
    }

    for top in system.topologies:
        assignment_matrix = smee.utils.zeros_like(
            (top.n_particles, len(lj_potential.parameters)),
            top.parameters["vdW"].assignment_matrix,
        )
        assignment_matrix[:, :3] = top.parameters["vdW"].assignment_matrix.to_dense()

        top.parameters["vdW"].assignment_matrix = assignment_matrix.to_sparse()

    params = {
        "oo": LJParam(eps_o, sig_o),
        "hh": LJParam(eps_hh, sig_hh),
        "aa": LJParam(eps_a, sig_a),
        "ah": LJParam(eps_ah, sig_ah),
        "oh": LJParam((eps_o * eps_h) ** 0.5, 0.5 * (sig_o + sig_h)),
        "oa": LJParam((eps_a * eps_o) ** 0.5, 0.5 * (sig_a + sig_o)),
    }

    for k, v in [*params.items()]:
        params[k[::-1]] = v

    return system, lj_potential, params
