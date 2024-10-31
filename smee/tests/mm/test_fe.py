import pathlib
import pickle

import openff.interchange
import openff.toolkit
import pytest
import torch

import smee.converters
import smee.mm


def load_systems(solute: str, solvent: str):
    ff_off = openff.toolkit.ForceField("openff-2.0.0.offxml")

    solute_inter = openff.interchange.Interchange.from_smirnoff(
        ff_off,
        openff.toolkit.Molecule.from_smiles(solute).to_topology(),
    )
    solvent_inter = openff.interchange.Interchange.from_smirnoff(
        ff_off,
        openff.toolkit.Molecule.from_smiles(solvent).to_topology(),
    )
    solvent_inter.to_openmm_system()

    ff, (top_solute, top_solvent) = smee.converters.convert_interchange(
        [solute_inter, solvent_inter]
    )

    return top_solute, top_solvent, ff


@pytest.mark.fe
def test_fe_ops():
    top_solute, top_solvent, ff = load_systems("CCO", "O")

    output_dir = pathlib.Path("CCO")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "ff.pkl").write_bytes(pickle.dumps(ff))

    result = smee.mm.generate_dg_solv_data(
        top_solute, top_solvent, ff, output_dir=output_dir
    )

    pathlib.Path("generate_dg.pkl").write_bytes(pickle.dumps(result))

    params = ff.potentials_by_type["Electrostatics"].parameters
    params.requires_grad_(True)

    dg = smee.mm.compute_dg_solv(ff, output_dir)
    dg_dtheta = torch.autograd.grad(dg, params)[0]

    pathlib.Path("compute_dg.pkl").write_bytes(pickle.dumps(dg))
    pathlib.Path("compute_dg_dtheta.pkl").write_bytes(pickle.dumps(dg_dtheta))

    dg, n_eff = smee.mm.reweight_dg_solv(ff, output_dir, dg)
    dg_dtheta = torch.autograd.grad(dg, params)[0]

    pathlib.Path("reweight_dg.pkl").write_bytes(pickle.dumps(dg))
    pathlib.Path("reweight_dg_dtheta.pkl").write_bytes(pickle.dumps(dg_dtheta))
