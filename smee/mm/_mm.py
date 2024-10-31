"""Run MM simulations using OpenMM."""

import copy
import functools
import logging
import pathlib
import subprocess
import tempfile
import typing

import numpy
import openmm.app
import openmm.unit
import torch

import smee
import smee.converters
import smee.mm._config
import smee.mm._reporters
import smee.mm._utils

_LOGGER = logging.getLogger("smee.mm")


class PACKMOLRuntimeError(RuntimeError):
    """An error raised when PACKMOL fails to execute / converge for some reason."""


def _apply_hmr(
    system_openmm: openmm.System,
    system_smee: smee.TensorSystem,
    hydrogen_mass: openmm.unit.Quantity = 1.5 * openmm.unit.amu,
):
    """Apply HMR to a system.

    Args:
        system_openmm: The OpenMM system to modify.
        system_smee: The SMEE system that defines the system topology.
        hydrogen_mass: The mass to use for hydrogen atoms.
    """

    idx_offset = 0

    for topology, n_copies in zip(
        system_smee.topologies, system_smee.n_copies, strict=True
    ):
        for _ in range(n_copies):
            for idx_a, idx_b in topology.bond_idxs:
                if topology.atomic_nums[idx_a] == 1:
                    (idx_a, idx_b) = (idx_b, idx_a)

                if topology.atomic_nums[idx_b] != 1:
                    continue
                if topology.atomic_nums[idx_a] == 1:
                    continue

                elements = sorted(topology.atomic_nums.tolist())

                if elements == [1, 1, 8]:
                    continue

                mass_delta = hydrogen_mass - system_openmm.getParticleMass(
                    idx_b + idx_offset
                )

                system_openmm.setParticleMass(idx_b + idx_offset, hydrogen_mass)
                system_openmm.setParticleMass(
                    idx_a + idx_offset,
                    system_openmm.getParticleMass(idx_a) - mass_delta,
                )

            idx_offset += topology.n_particles


def _topology_to_xyz(
    topology: smee.TensorTopology, force_field: smee.TensorForceField | None
) -> str:
    """Convert a topology to an RDKit molecule."""
    mol = smee.mm._utils.topology_to_rdkit(topology)

    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = torch.tensor(mol.GetConformer().GetPositions())

    if topology.n_v_sites > 0:
        assert force_field is not None, "v-sites require a force field"
        coords = coords.to(force_field.v_sites.weights[0].dtype)

        coords = smee.geometry.add_v_site_coords(topology.v_sites, coords, force_field)
        elements += ["X"] * topology.n_v_sites

    return "\n".join(
        [
            f"{len(elements)}",
            "",
            *[
                f"{element} {x:f} {y:f} {z:f}"
                for (element, (x, y, z)) in zip(elements, coords, strict=True)
            ],
        ]
    )


def _approximate_box_size(
    system: smee.TensorSystem,
    config: "smee.mm.GenerateCoordsConfig",
) -> openmm.unit.Quantity:
    """Generate an approximate box size based on the number and molecular weight of
    the molecules present, and a target density for the final system.

    Args:
        system: The system to generate the box size for.
        config: Configuration of how to generate the system coordinates.

    Returns:
        The approximate box size.
    """

    sum_fn = functools.partial(sum, start=0.0 * openmm.unit.dalton)

    weight = sum_fn(
        sum_fn(
            openmm.app.Element.getByAtomicNumber(int(atomic_num)).mass
            for atomic_num in topology.atomic_nums
        )
        * n_copies
        for topology, n_copies in zip(system.topologies, system.n_copies, strict=True)
    )

    volume = weight / openmm.unit.AVOGADRO_CONSTANT_NA / config.target_density
    volume = volume.in_units_of(openmm.unit.angstrom**3)
    return volume ** (1.0 / 3.0) * config.scale_factor


def _generate_packmol_input(
    n_copies: list[int],
    box_size: openmm.unit.Quantity,
    config: "smee.mm.GenerateCoordsConfig",
) -> str:
    """Construct the PACKMOL input file.

    Args:
        n_copies: The number of copies of each molecule in the system.
        box_size: The approximate box size to pack the molecules into.
        config: Configuration of how to generate the system coordinates.

    Returns:
        The string contents of the PACKMOL input file.
    """

    box_size = box_size.value_in_unit(openmm.unit.angstrom)
    tolerance = config.tolerance.value_in_unit(openmm.unit.angstrom)

    return "\n".join(
        [
            f"tolerance {tolerance:f}",
            "filetype xyz",
            "output output.xyz",
            *([] if config.seed is None else [f"seed {config.seed}"]),
            *[
                f"structure {i}.xyz\n"
                f"  number {n}\n"
                f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}\n"
                "end structure"
                for i, n in enumerate(n_copies)
            ],
        ]
    )


def generate_system_coords(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField | None,
    config: typing.Optional["smee.mm.GenerateCoordsConfig"] = None,
) -> tuple[openmm.unit.Quantity, openmm.unit.Quantity]:
    """Generate coordinates for a system of molecules using PACKMOL.

    Args:
        system: The system to generate coordinates for.
        force_field: The force field that describes the geometry of any virtual sites.
        config: Configuration of how to generate the system coordinates.

    Raises:
        * PACKMOLRuntimeError

    Returns:
        The coordinates with ``shape=(n_atoms, 3)`` and box vectors with
        ``shape=(3, 3)``
    """

    system = system.to("cpu")
    force_field = None if force_field is None else force_field.to("cpu")

    config = config if config is not None else smee.mm.GenerateCoordsConfig()

    box_size = _approximate_box_size(system, config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)

        for i, topology in enumerate(system.topologies):
            xyz = _topology_to_xyz(topology, force_field)
            (tmp_dir / f"{i}.xyz").write_text(xyz)

        input_file = tmp_dir / "input.txt"
        input_file.write_text(
            _generate_packmol_input(system.n_copies, box_size, config)
        )

        with input_file.open("r") as file:
            result = subprocess.run(
                "packmol", stdin=file, capture_output=True, text=True, cwd=tmp_dir
            )

        if result.returncode != 0 or not result.stdout.find("Success!") > 0:
            raise PACKMOLRuntimeError(result.stdout)

        output_lines = (tmp_dir / "output.xyz").read_text().splitlines()

    coordinates = (
        numpy.array(
            [
                [float(coordinate) for coordinate in coordinate_line.split()[1:]]
                for coordinate_line in output_lines[2:]
                if len(coordinate_line) > 0
            ]
        )
        * openmm.unit.angstrom
    )

    box_vectors = numpy.eye(3) * (box_size + config.padding)
    return coordinates, box_vectors


def _get_state_log(state: openmm.State) -> str:
    """Convert an OpenMM state to a string representation for logging."""

    energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

    box_vectors = state.getPeriodicBoxVectors()
    volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]
    volume = volume.value_in_unit(openmm.unit.angstrom**3)

    return f"energy={energy:.4f} kcal / mol volume={volume:.4f} Å^3"


def _get_platform(is_periodic: bool) -> openmm.Platform:
    """Returns the fastest OpenMM platform available.

    Notes:
        If the system is not periodic, it is assumed that a molecule (or dimer) in
        vacuum is being simulated and the Reference platform will be used.
    """

    if not is_periodic:
        return openmm.Platform.getPlatformByName("Reference")

    platforms = [
        openmm.Platform.getPlatform(i) for i in range(openmm.Platform.getNumPlatforms())
    ]
    return max(platforms, key=lambda platform: platform.getSpeed())


def _energy_minimize(
    omm_system: openmm.System,
    state: openmm.State | tuple[openmm.unit.Quantity, openmm.unit.Quantity],
    platform: openmm.Platform,
    config: "smee.mm.MinimizationConfig",
) -> openmm.State:
    omm_system = copy.deepcopy(omm_system)

    integrator = openmm.VerletIntegrator(0.0001)
    context = openmm.Context(omm_system, integrator, platform)

    if isinstance(state, openmm.State):
        context.setState(state)
    else:
        coords, box_vectors = state
        if box_vectors is not None:
            context.setPeriodicBoxVectors(*box_vectors)
        context.setPositions(coords)

    openmm.LocalEnergyMinimizer.minimize(
        context,
        config.tolerance.value_in_unit_system(openmm.unit.md_unit_system),
        config.max_iterations,
    )
    return context.getState(getEnergy=True, getPositions=True)


def _run_simulation(
    omm_system: openmm.System,
    omm_topology: openmm.app.Topology,
    state: openmm.State | tuple[openmm.unit.Quantity, openmm.unit.Quantity],
    platform: openmm.Platform,
    config: "smee.mm.SimulationConfig",
    reporters: list[typing.Any] | None = None,
) -> openmm.State:
    reporters = [] if reporters is None else reporters

    omm_system = copy.deepcopy(omm_system)

    if config.pressure is not None:
        barostat = openmm.MonteCarloBarostat(config.pressure, config.temperature)
        omm_system.addForce(barostat)

    integrator = openmm.LangevinIntegrator(
        config.temperature, config.friction_coeff, config.timestep
    )

    simulation = openmm.app.Simulation(omm_topology, omm_system, integrator, platform)

    if isinstance(state, openmm.State):
        simulation.context.setState(state)
    else:
        coords, box_vectors = state
        if box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(coords)

    simulation.reporters.extend(reporters)

    simulation.step(config.n_steps)

    return simulation.context.getState(getEnergy=True, getPositions=True)


def simulate(
    system: smee.TensorSystem | smee.TensorTopology,
    force_field: smee.TensorForceField,
    coords: openmm.unit.Quantity,
    box_vectors: openmm.unit.Quantity | None,
    equilibrate_configs: list[
        typing.Union["smee.mm.MinimizationConfig", "smee.mm.SimulationConfig"]
    ],
    production_config: "smee.mm.SimulationConfig",
    production_reporters: list[typing.Any] | None = None,
    apply_hmr: bool = False,
) -> openmm.State:
    """Simulate a SMEE system of molecules or topology.

    Args:
        system: The system / topology to simulate.
        force_field: The force field to simulate with.
        coords: The coordinates [Å] to use for the simulation. This should be
            a unit wrapped numpy array with ``shape=(n_atoms, 3)``.
        box_vectors: The box vectors [Å] to use for the simulation if periodic. This
            should be a unit wrapped numpy array with ``shape=(3, 3)``.
        equilibrate_configs: A list of configurations defining the steps to run for
            equilibration. No data will be stored from these simulations.
        production_config: The configuration defining the production simulation to run.
        production_reporters: A list of additional OpenMM reporters to use for the
            production simulation.
        apply_hmr: Whether to apply Hydrogen Mass Repartitioning to the system prior
            to simulation.
    """

    assert isinstance(coords.value_in_unit(openmm.unit.angstrom), numpy.ndarray)
    assert isinstance(box_vectors.value_in_unit(openmm.unit.angstrom), numpy.ndarray)

    force_field = force_field.to("cpu")

    system: smee.TensorSystem = (
        system
        if isinstance(system, smee.TensorSystem)
        else smee.TensorSystem([system], [1], False)
    ).to("cpu")

    requires_pbc = any(
        config.pressure is not None
        for config in equilibrate_configs + [production_config]
        if isinstance(config, smee.mm.SimulationConfig)
    )

    if not system.is_periodic and requires_pbc:
        raise ValueError("pressure cannot be specified for a non-periodic system")

    platform = _get_platform(system.is_periodic)

    omm_state = coords, box_vectors

    omm_system = smee.converters.convert_to_openmm_system(force_field, system)
    omm_topology = smee.converters.convert_to_openmm_topology(system)

    if apply_hmr:
        _apply_hmr(omm_system, system)

    for i, config in enumerate(equilibrate_configs):
        _LOGGER.info(f"running equilibration step {i + 1} / {len(equilibrate_configs)}")

        if isinstance(config, smee.mm.MinimizationConfig):
            omm_state = _energy_minimize(omm_system, omm_state, platform, config)

        elif isinstance(config, smee.mm.SimulationConfig):
            omm_state = _run_simulation(
                omm_system, omm_topology, omm_state, platform, config, None
            )
        else:
            raise NotImplementedError

        _LOGGER.info(_get_state_log(omm_state))

    _LOGGER.info("running production simulation")
    omm_state = _run_simulation(
        omm_system,
        omm_topology,
        omm_state,
        platform,
        production_config,
        production_reporters,
    )
    _LOGGER.info(_get_state_log(omm_state))

    return omm_state
