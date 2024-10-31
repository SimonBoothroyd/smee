import pathlib

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
def test_fe_ops(tmp_cwd):
    # taken from a run on commit 7915d1e323318d2314a8b0322e7f44968c660c21
    expected_dg = torch.tensor(-3.8262).double()
    expected_dg_dtheta = torch.tensor(
        [
            [1.0288e01],
            [1.3976e01],
            [2.6423e01],
            [9.1453e00],
            [9.0158e00],
            [9.5534e00],
            [1.0414e01],
            [1.1257e01],
            [-4.0618e00],
            [5.0233e03],
            [-1.3574e03],
        ]
    ).double()

    top_solute, top_solvent, ff = load_systems("CCO", "O")

    output_dir = pathlib.Path("CCO")
    output_dir.mkdir(parents=True, exist_ok=True)

    smee.mm.generate_dg_solv_data(top_solute, top_solvent, ff, output_dir=output_dir)

    params = ff.potentials_by_type["Electrostatics"].parameters
    params.requires_grad_(True)

    dg = smee.mm.compute_dg_solv(ff, output_dir)
    dg_dtheta = torch.autograd.grad(dg, params)[0]

    assert dg == pytest.approx(expected_dg, abs=0.5)
    assert dg_dtheta == pytest.approx(expected_dg_dtheta, rel=1.1)

    dg, n_eff = smee.mm.reweight_dg_solv(ff, output_dir, dg)
    dg_dtheta = torch.autograd.grad(dg, params)[0]

    assert dg == pytest.approx(expected_dg, abs=0.5)
    assert dg_dtheta == pytest.approx(expected_dg_dtheta, rel=1.1)
