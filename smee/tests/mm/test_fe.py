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
    # taken from a run on commit ec3d272b466f761ed838e16a5ba7b97ceadc463b
    expected_dg = torch.tensor(-3.8262).double()
    expected_dg_dtheta = torch.tensor(
        [
            [10.2679],
            [13.3933],
            [25.3670],
            [9.3747],
            [9.3279],
            [9.1520],
            [10.5614],
            [9.6908],
            [-4.4326],
            [-17.3971],
            [-38.5407],
        ]
    ).double()

    top_solute, top_solvent, ff = load_systems("CCO", "O")

    output_dir = pathlib.Path("CCO")
    output_dir.mkdir(parents=True, exist_ok=True)

    smee.mm.generate_dg_solv_data(
        top_solute, None, top_solvent, ff, output_dir=output_dir
    )

    params = ff.potentials_by_type["Electrostatics"].parameters
    params.requires_grad_(True)

    dg_comp = smee.mm.compute_dg_solv(top_solute, None, top_solvent, ff, output_dir)
    dg_comp_dtheta = torch.autograd.grad(dg_comp, params)[0]

    print("dg COMP", dg_comp, flush=True)
    print("dg_dtheta COMP", dg_comp_dtheta, dg_comp, flush=True)

    dg_rw, n_eff = smee.mm.reweight_dg_solv(
        top_solute, None, top_solvent, ff, output_dir, dg_comp
    )
    dg_rw_dtheta = torch.autograd.grad(dg_rw, params)[0]

    print("dg REWEIGHT", dg_rw, flush=True)
    print("dg_dtheta REWEIGHT", dg_rw_dtheta, dg_rw, flush=True)

    assert dg_comp.detach() == pytest.approx(expected_dg.detach(), abs=0.5)
    assert dg_comp_dtheta.detach() == pytest.approx(expected_dg_dtheta, rel=1.1)

    assert dg_rw.detach() == pytest.approx(expected_dg.detach(), abs=0.5)
    assert dg_rw_dtheta.detach() == pytest.approx(expected_dg_dtheta, rel=1.1)
