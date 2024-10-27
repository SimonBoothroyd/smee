"""Compute ddG from the output of ``femto`` / ``absolv``"""

import logging
import pathlib
import pickle
import typing

import mdtraj
import numpy
import openff.toolkit
import openmm.unit
import torch

import smee
import smee.converters
import smee.mm._utils

if typing.TYPE_CHECKING:
    import absolv.config

_LOGGER = logging.getLogger(__name__)

_NM_TO_ANGSTROM = 10.0


def generate_dg_solv_data(
    solute: smee.TensorTopology,
    solvent: smee.TensorTopology,
    force_field: smee.TensorForceField,
    temperature: openmm.unit.Quantity = 298.15 * openmm.unit.kelvin,
    pressure: openmm.unit.Quantity = 1.0 * openmm.unit.atmosphere,
    vacuum_protocol: typing.Optional["absolv.config.EquilibriumProtocol"] = None,
    solvent_protocol: typing.Optional["absolv.config.EquilibriumProtocol"] = None,
    n_solvent: int = 216,
    output_dir: pathlib.Path | None = None,
):
    """Run a solvation free energy calculation using ``absolv``, and saves the output
    such that a differentiable free energy can be computed.

    Args:
        solute: The solute topology.
        solvent: The solvent topology.
        force_field: The force field to parameterize the system with.
        temperature: The temperature to simulate at.
        pressure: The pressure to simulate at.
        vacuum_protocol: The protocol to use for the vacuum phase.
        solvent_protocol: The protocol to use for the solvent phase.
        n_solvent: The number of solvent molecules to use.
        output_dir: The directory to write the output FEP data to.
    """
    import absolv.config
    import absolv.runner
    import femto.md.config

    output_dir = pathlib.Path.cwd() if output_dir is None else output_dir

    if vacuum_protocol is None:
        vacuum_protocol = absolv.config.EquilibriumProtocol(
            production_protocol=absolv.config.HREMDProtocol(
                n_steps_per_cycle=500,
                n_cycles=2000,
                integrator=femto.md.config.LangevinIntegrator(
                    timestep=1.0 * openmm.unit.femtosecond
                ),
            ),
            lambda_sterics=absolv.config.DEFAULT_LAMBDA_STERICS_VACUUM,
            lambda_electrostatics=absolv.config.DEFAULT_LAMBDA_ELECTROSTATICS_VACUUM,
        )
    if solvent_protocol is None:
        solvent_protocol = absolv.config.EquilibriumProtocol(
            production_protocol=absolv.config.HREMDProtocol(
                n_steps_per_cycle=500,
                n_cycles=1000,
                integrator=femto.md.config.LangevinIntegrator(
                    timestep=4.0 * openmm.unit.femtosecond
                ),
            ),
            lambda_sterics=absolv.config.DEFAULT_LAMBDA_STERICS_SOLVENT,
            lambda_electrostatics=absolv.config.DEFAULT_LAMBDA_ELECTROSTATICS_SOLVENT,
        )

    config = absolv.config.Config(
        temperature=temperature,
        pressure=pressure,
        alchemical_protocol_a=vacuum_protocol,
        alchemical_protocol_b=solvent_protocol,
    )

    solute_mol = openff.toolkit.Molecule.from_rdkit(
        smee.mm._utils.topology_to_rdkit(solute),
        allow_undefined_stereo=True,
    )
    solvent_mol = openff.toolkit.Molecule.from_rdkit(
        smee.mm._utils.topology_to_rdkit(solvent),
        allow_undefined_stereo=True,
    )

    system_config = absolv.config.System(
        solutes={solute_mol.to_smiles(mapped=True): 1},
        solvent_a=None,
        solvent_b={solvent_mol.to_smiles(mapped=True): n_solvent},
    )

    topologies = {
        "solvent-a": smee.TensorSystem([solute], [1], is_periodic=False),
        "solvent-b": smee.TensorSystem(
            [solute, solvent], [1, n_solvent], is_periodic=True
        ),
    }
    pressures = {
        "solvent-a": None,
        "solvent-b": pressure.value_in_unit(openmm.unit.atmosphere),
    }

    for phase, topology in topologies.items():
        state = {
            "system": topology,
            "temperature": temperature.value_in_unit(openmm.unit.kelvin),
            "pressure": pressures[phase],
        }

        (output_dir / phase).mkdir(exist_ok=True, parents=True)
        (output_dir / phase / "system.pkl").write_bytes(pickle.dumps(state))

    def _parameterize(
        top, coords, phase: typing.Literal["solvent-a", "solvent-b"]
    ) -> openmm.System:
        return smee.converters.convert_to_openmm_system(force_field, topologies[phase])

    prepared_system_a, prepared_system_b = absolv.runner.setup(
        system_config, config, _parameterize
    )
    absolv.runner.run_eq(
        config, prepared_system_a, prepared_system_b, "CUDA", output_dir
    )


def _uncorrelated_frames(length: int, g: float) -> list[int]:
    """Return the indices of frames that are un-correlated.

    Args:
        length: The total number of correlated frames.
        g: The statistical inefficiency of the data.

    Returns:
        The indices of un-correlated frames.
    """
    indices = []
    n = 0

    while int(round(n * g)) < length:
        t = int(round(n * g))
        if n == 0 or t != indices[n - 1]:
            indices.append(t)
        n += 1

    return indices


def _load_trajectory(
    trajectory_dir: pathlib.Path,
    system: smee.TensorSystem,
    replica_to_state_idx: numpy.ndarray,
    state_idx: int = 0,
) -> tuple[numpy.ndarray, numpy.ndarray | None]:
    n_states = len(list(trajectory_dir.glob("r*.dcd")))

    topology_omm = smee.converters.convert_to_openmm_topology(system)
    topology_md = mdtraj.Topology.from_openmm(topology_omm)

    trajectories = [
        mdtraj.load(str(trajectory_dir / f"r{i}.dcd"), top=topology_md)
        for i in range(n_states)
    ]
    state_idxs = (replica_to_state_idx.reshape(-1, n_states).T == state_idx).argmax(
        axis=0
    )

    xyz = numpy.stack(
        [
            trajectories[traj_idx].xyz[frame_idx] * _NM_TO_ANGSTROM
            for frame_idx, traj_idx in enumerate(state_idxs)
        ]
    )

    if trajectories[0].unitcell_vectors is None:
        return xyz, None

    box = numpy.stack(
        [
            trajectories[traj_idx].unitcell_vectors[frame_idx] * _NM_TO_ANGSTROM
            for frame_idx, traj_idx in enumerate(state_idxs)
        ]
    )

    return xyz, box


def _load_samples(
    output_dir: pathlib.Path,
) -> tuple[
    smee.TensorSystem,
    float,
    float | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    import pyarrow
    import pymbar.timeseries

    state = pickle.loads((output_dir / "state.pkl").read_bytes())

    system = state["system"]

    temperature = state["temperature"] * openmm.unit.kelvin

    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)
    beta = beta.value_in_unit(openmm.unit.kilocalorie_per_mole**-1)

    pressure = None

    if state["pressure"] is not None:
        pressure = state["pressure"] * openmm.unit.atmosphere
        pressure = (pressure * openmm.unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**3
        )

    with pyarrow.OSFile(str(output_dir / "samples.arrow"), "rb") as file:
        with pyarrow.RecordBatchStreamReader(file) as reader:
            output_table = reader.read_all()

    replica_to_state_idx = numpy.hstack(
        [numpy.array(x) for x in output_table["replica_to_state_idx"].to_pylist()]
    )

    # group the data along axis 1 so that data sampled in the same state is grouped.
    # this will let us more easily de-correlate the data.
    u_kn = numpy.hstack([numpy.array(x) for x in output_table["u_kn"].to_pylist()])
    u_kn_per_k = [u_kn[:, replica_to_state_idx == i] for i in range(len(u_kn))]

    xyz_0, box_0 = _load_trajectory(
        output_dir / "trajectories", system, replica_to_state_idx
    )

    n_uncorrelated = u_kn.shape[1] // u_kn.shape[0]

    g = pymbar.timeseries.statistical_inefficiency_multiple(
        [
            u_kn_per_k[i][i, i * n_uncorrelated : (i + 1) * n_uncorrelated]
            for i in range(len(u_kn))
        ]
    )
    uncorrelated_frames = _uncorrelated_frames(n_uncorrelated, g)

    xyz_0 = xyz_0[uncorrelated_frames]
    box_0 = box_0[uncorrelated_frames] if box_0 is not None else None

    for state_idx, state_u_kn in enumerate(u_kn_per_k):
        u_kn_per_k[state_idx] = state_u_kn[:, uncorrelated_frames]

    u_kn = numpy.hstack(u_kn_per_k)
    n_k = numpy.array([len(uncorrelated_frames)] * u_kn.shape[0])

    return (
        system,
        beta,
        pressure,
        torch.tensor(u_kn),
        torch.tensor(n_k),
        torch.tensor(xyz_0),
        torch.tensor(box_0) if box_0 is not None else None,
    )


def _compute_energy(
    system: smee.TensorSystem,
    ff: smee.TensorForceField,
    xyz_0: torch.Tensor,
    box_0: torch.Tensor,
) -> torch.Tensor:
    if system.is_periodic:
        energy_per_frame = [
            smee.compute_energy(system, ff, c, b)
            for c, b in zip(xyz_0, box_0, strict=True)
        ]
        energy = torch.concat(energy_per_frame)
    else:
        energy = smee.compute_energy(system, ff, xyz_0, box_0)

    return energy


def compute_dg_and_grads(
    force_field: smee.TensorForceField,
    theta: tuple[torch.Tensor, ...],
    output_dir: pathlib.Path,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    import pymbar

    system, beta, _, u_kn, n_k, xyz_0, box_0 = _load_samples(output_dir)
    assert (box_0 is not None) == system.is_periodic

    mbar = pymbar.MBAR(u_kn.numpy(), n_k.numpy())

    f_i = mbar.compute_free_energy_differences()["Delta_f"][0, :]
    dg = (f_i[-1] - f_i[0]) / beta

    energy = _compute_energy(system, force_field, xyz_0, box_0)
    grads = torch.autograd.grad(energy.mean(), theta)

    return torch.tensor(dg), grads


def reweight_dg_and_grads(
    force_field: smee.TensorForceField,
    theta: tuple[torch.Tensor, ...],
    output_dir: pathlib.Path,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], int]:
    import pymbar

    system, beta, pressure, u_kn, n_k, xyz_0, box_0 = _load_samples(output_dir)
    assert (box_0 is not None) == system.is_periodic
    assert (box_0 is not None) == (pressure is not None)

    u_0_old = u_kn[0, : n_k[0]]
    energy_0 = _compute_energy(system, force_field, xyz_0, box_0)

    u_0_new = energy_0.detach().clone() * beta

    if pressure is not None:
        u_0_new += pressure * torch.det(torch.tensor(box_0)) * beta

    u_kn = numpy.stack([u_0_old.numpy(), u_0_new.numpy()])
    n_k = numpy.array([n_k[0], 0])

    mbar = pymbar.MBAR(u_kn, n_k)

    n_eff = mbar.compute_effective_sample_number()

    f_i = mbar.compute_free_energy_differences()["Delta_f"][0, :]
    dg = (f_i[-1] - f_i[0]) / beta

    weights = torch.tensor(mbar.W_nk[:, 1])
    grads = torch.autograd.grad((energy_0 * weights).sum(), theta)

    return torch.tensor(dg), grads, n_eff
