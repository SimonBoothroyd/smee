"""Convert tensor representations into OpenMM systems."""

import collections
import copy
import itertools
import math
import typing

import openmm
import openmm.app
import torch

import smee
import smee.utils

_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_ANGSTROM = openmm.unit.angstrom
_RADIANS = openmm.unit.radians

_ANGSTROM_TO_NM = 1.0 / 10.0

_INTRA_SCALE_VAR = "scale_excl"

_T = typing.TypeVar("_T", bound=openmm.NonbondedForce | openmm.CustomNonbondedForce)


def _create_nonbonded_force(
    potential: smee.TensorPotential,
    system: smee.TensorSystem,
    cls: typing.Type[_T] = openmm.NonbondedForce,
) -> _T:
    if cls == openmm.NonbondedForce:
        force = openmm.NonbondedForce()
        force.setUseDispersionCorrection(system.is_periodic)
        force.setEwaldErrorTolerance(1.0e-4)  # TODO: interchange hardcoded value
    elif cls == openmm.CustomNonbondedForce:
        force = openmm.CustomNonbondedForce("")
        force.setUseLongRangeCorrection(system.is_periodic)
    else:
        raise NotImplementedError(f"unsupported force class {cls}")

    cutoff_idx = potential.attribute_cols.index("cutoff")
    switch_idx = (
        None
        if "switch_width" not in potential.attribute_cols
        else potential.attribute_cols.index("switch_width")
    )

    if not system.is_periodic:
        force.setNonbondedMethod(cls.NoCutoff)
    else:
        cutoff = float(potential.attributes[cutoff_idx]) * _ANGSTROM

        method = (
            openmm.NonbondedForce.PME
            if cls == openmm.NonbondedForce
            else openmm.CustomNonbondedForce.CutoffPeriodic
        )

        force.setNonbondedMethod(method)
        force.setCutoffDistance(cutoff)

        if switch_idx is not None:
            switch_width = float(potential.attributes[switch_idx]) * _ANGSTROM
            switch_distance = cutoff - switch_width

            if switch_distance > 0.0 * _ANGSTROM:
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(switch_distance)

    return force


def _build_vdw_lookup(
    potential: smee.TensorPotential,
    mixing_fn: dict[str, typing.Callable[[float, float], float]],
) -> dict[str, list[float]]:
    """Build the ``n_param x n_param`` vdW parameter lookup table containing
    parameters for all interactions.
    """

    n_params = len(potential.parameters)
    n_params_sqr = n_params * n_params

    parameter_col_to_idx = {col: i for i, col in enumerate(potential.parameter_cols)}
    parameter_lookup = {col: [None] * n_params_sqr for col in potential.parameter_cols}

    exceptions = {
        **potential.exceptions,
        **{(j, i): idx for (i, j), idx in potential.exceptions.items()},
    }

    for i, j in itertools.product(range(n_params), range(n_params)):
        if (i, j) in exceptions:
            parameters = {
                col: potential.parameters[exceptions[i, j], col_idx]
                for col, col_idx in parameter_col_to_idx.items()
            }
        else:
            parameters = {
                col: mixing_fn[col](
                    potential.parameters[i, col_idx],
                    potential.parameters[j, col_idx],
                )
                for col, col_idx in parameter_col_to_idx.items()
            }

        unit_conversion = {
            col: (1.0 * potential.parameter_units[col_idx])
            .to_openmm()
            .value_in_unit_system(openmm.unit.md_unit_system)
            for col, col_idx in parameter_col_to_idx.items()
        }

        for col, col_idx in parameter_col_to_idx.items():
            parameter_lookup[col][i + j * n_params] = float(
                parameters[col] * unit_conversion[col]
            )

    return parameter_lookup


def _prepend_scale_to_energy_fn(fn: str, scale_var: str = _INTRA_SCALE_VAR) -> str:
    assert scale_var not in fn, f"scale variable {scale_var} already in energy fn"

    fn_split = fn.split(";")
    assert "=" not in fn_split[0], "energy function missing a return value"

    fn_split[0] = f"{scale_var}*({fn_split[0]})"
    return ";".join(fn_split)


def _convert_custom_vdw_potential(
    potential: smee.TensorPotential,
    system: smee.TensorSystem,
    energy_fn: str,
    mixing_fn: dict[str, typing.Callable[[float, float], float]],
) -> list[openmm.CustomNonbondedForce | openmm.CustomBondForce]:
    assert potential.exceptions is not None, "missing exceptions"

    parameter_lookup = _build_vdw_lookup(potential, mixing_fn)
    n_params = len(potential.parameters)

    lookup_fn = " ".join(
        f"{col}={col}_lookup(param_idx1, param_idx2);" for col in parameter_lookup
    )

    inter_force = _create_nonbonded_force(
        potential, system, openmm.CustomNonbondedForce
    )
    inter_force.setEnergyFunction(energy_fn + lookup_fn)
    inter_force.addPerParticleParameter("param_idx")

    for col, vals in parameter_lookup.items():
        inter_force.addTabulatedFunction(
            f"{col}_lookup",
            openmm.Discrete2DFunction(n_params, n_params, vals),
        )

    intra_force_energy_fn = _prepend_scale_to_energy_fn(energy_fn, _INTRA_SCALE_VAR)
    intra_force = openmm.CustomBondForce(intra_force_energy_fn)

    for col in parameter_lookup:
        intra_force.addPerBondParameter(col)

    intra_force.addPerBondParameter(_INTRA_SCALE_VAR)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_map = topology.parameters[potential.type]

        assignment_dense = parameter_map.assignment_matrix.to_dense()
        assigned_idxs = assignment_dense.argmax(axis=-1)

        if not (assignment_dense.abs().sum(axis=-1) == 1).all():
            raise NotImplementedError(
                f"exceptions can only be used when each particle is assigned exactly "
                f"one {potential.type} parameter"
            )

        for _ in range(n_copies):
            for idx in assigned_idxs:
                inter_force.addParticle([int(idx)])

            for index, (i, j) in enumerate(parameter_map.exclusions):
                inter_force.addExclusion(int(i + idx_offset), int(j + idx_offset))

                scale = potential.attributes[parameter_map.exclusion_scale_idxs[index]]

                if scale <= 0.0:
                    continue

                intra_parameters = [
                    vals[assigned_idxs[i] + assigned_idxs[j] * n_params]
                    for col, vals in parameter_lookup.items()
                ]
                intra_parameters.append(scale)

                intra_force.addBond(
                    int(i + idx_offset), int(j + idx_offset), intra_parameters
                )

            idx_offset += topology.n_particles

    return [inter_force, intra_force]


def _convert_lj_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.NonbondedForce | list[openmm.CustomNonbondedForce | openmm.CustomBondForce]:
    """Convert a Lennard-Jones potential to an OpenMM force.

    If the potential has custom mixing rules (i.e. exceptions), the interactions will
    be split into an inter- and intra-molecular force.
    """
    mixing_fn = {
        "epsilon": lambda x, y: (x * y) ** 0.5,
        "sigma": lambda x, y: (x + y) * 0.5,
    }

    if potential.exceptions is not None:
        return _convert_custom_vdw_potential(
            potential,
            system,
            energy_fn="4*epsilon*x6*(x6 - 1.0);x6=x4*x2;x4=x2*x2;x2=x*x;x=sigma/r;",
            mixing_fn=mixing_fn,
        )

    force = _create_nonbonded_force(potential, system)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters

        for _ in range(n_copies):
            for epsilon, sigma in parameters:
                force.addParticle(0.0, sigma * _ANGSTROM, epsilon * _KCAL_PER_MOL)

            for index, (i, j) in enumerate(parameter_map.exclusions):
                eps_i, sig_i = parameters[i, :]
                eps_j, sig_j = parameters[j, :]

                eps = mixing_fn["epsilon"](eps_i, eps_j)
                sig = mixing_fn["sigma"](sig_i, sig_j)

                scale = potential.attributes[parameter_map.exclusion_scale_idxs[index]]

                force.addException(
                    i + idx_offset,
                    j + idx_offset,
                    0.0,
                    sig * _ANGSTROM,
                    eps * _KCAL_PER_MOL * scale,
                )

            idx_offset += topology.n_particles

    return force


def _convert_dexp_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> tuple[openmm.CustomNonbondedForce, openmm.CustomBondForce]:
    if potential.exceptions is not None:
        raise NotImplementedError("exceptions not supported")

    energy_fn = (
        "CombinedEpsilon*RepulsionExp-CombinedEpsilon*AttractionExp;"
        "CombinedEpsilon=epsilon1*epsilon2;"
        "RepulsionExp=beta/(alpha-beta)*exp(alpha*(1-ExpDistance));"
        "AttractionExp=alpha/(alpha-beta)*exp(beta*(1-ExpDistance));"
        "ExpDistance=r/CombinedR;"
        "CombinedR=r_min1+r_min2;"
    )
    energy_fn_scaled_lines = energy_fn.split(";")
    energy_fn_scaled_lines[0] = f"scale_excl*({energy_fn_scaled_lines[0]})"
    energy_fn_scaled = ";".join(energy_fn_scaled_lines)

    assert potential.parameter_cols == ("epsilon", "r_min")

    transform = {"epsilon": lambda x: math.sqrt(x), "r_min": lambda x: x * 0.5}

    def add_globals(f):
        alpha_idx = potential.attribute_cols.index("alpha")
        alpha = float(potential.attributes[alpha_idx])
        f.addGlobalParameter("alpha", alpha)
        beta_idx = potential.attribute_cols.index("beta")
        beta = float(potential.attributes[beta_idx])
        f.addGlobalParameter("beta", beta)

    def parameter_to_openmm(p):
        return [
            transform[col_name](
                (float(col) * unit)
                .to_openmm()
                .value_in_unit_system(openmm.unit.md_unit_system)
            )
            for col, col_name, unit in zip(
                p, potential.parameter_cols, potential.parameter_units, strict=True
            )
        ]

    force_vdw = _create_nonbonded_force(potential, system, openmm.CustomNonbondedForce)
    force_vdw.setEnergyFunction(energy_fn)
    add_globals(force_vdw)

    for col in potential.parameter_cols:
        force_vdw.addPerParticleParameter(col)

    force_excl = openmm.CustomBondForce(energy_fn_scaled)
    force_excl.setUsesPeriodicBoundaryConditions(system.is_periodic)
    add_globals(force_excl)

    force_excl.addPerBondParameter("scale_excl")
    for i in (1, 2):
        for col in potential.parameter_cols:
            force_excl.addPerBondParameter(f"{col}{i}")

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters

        for _ in range(n_copies):
            for parameter in parameters:
                force_vdw.addParticle(parameter_to_openmm(parameter))

            for index, (i, j) in enumerate(parameter_map.exclusions):
                eps_i, r_min_i = parameter_to_openmm(parameters[i, :])
                eps_j, r_min_j = parameter_to_openmm(parameters[j, :])

                scale = potential.attributes[parameter_map.exclusion_scale_idxs[index]]

                force_vdw.addExclusion(i + idx_offset, j + idx_offset)

                if torch.isclose(scale, smee.utils.tensor_like(0.0, scale)):
                    continue

                force_excl.addBond(
                    i + idx_offset,
                    j + idx_offset,
                    [float(scale), eps_i, r_min_i, eps_j, r_min_j],
                )

            idx_offset += topology.n_particles

    return force_vdw, force_excl


def _convert_vdw_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> list[openmm.Force]:
    import smee.potentials.nonbonded

    if potential.fn == smee.potentials.nonbonded.LJ_POTENTIAL:
        forces = _convert_lj_potential(potential, system)
        return [forces] if not isinstance(forces, list) else forces
    elif potential.fn == smee.potentials.nonbonded.DEXP_POTENTIAL:
        return [*_convert_dexp_potential(potential, system)]

    raise NotImplementedError(f"unsupported potential function {potential.fn}")


def _convert_electrostatics_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.NonbondedForce:
    force = _create_nonbonded_force(potential, system)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters

        for _ in range(n_copies):
            for charge in parameters:
                force.addParticle(
                    charge.detach() * openmm.unit.elementary_charge,
                    1.0 * _ANGSTROM,
                    0.0 * _KCAL_PER_MOL,
                )

            for index, (i, j) in enumerate(parameter_map.exclusions):
                q_i, q_j = parameters[i], parameters[j]
                q = q_i * q_j

                scale = potential.attributes[parameter_map.exclusion_scale_idxs[index]]

                force.addException(
                    i + idx_offset,
                    j + idx_offset,
                    scale * q,
                    1.0,
                    0.0,
                )

            idx_offset += topology.n_particles

    return force


def _convert_bond_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.HarmonicBondForce:
    force = openmm.HarmonicBondForce()

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameters = (
            topology.parameters[potential.type].assignment_matrix @ potential.parameters
        )

        for _ in range(n_copies):
            atom_idxs = topology.parameters[potential.type].particle_idxs + idx_offset

            for (i, j), (constant, length) in zip(atom_idxs, parameters):
                force.addBond(
                    i,
                    j,
                    length * _ANGSTROM,
                    constant * _KCAL_PER_MOL / _ANGSTROM**2,
                )

            idx_offset += topology.n_particles

    return force


def _convert_angle_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.HarmonicAngleForce:
    force = openmm.HarmonicAngleForce()

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameters = (
            topology.parameters[potential.type].assignment_matrix @ potential.parameters
        )

        for _ in range(n_copies):
            atom_idxs = topology.parameters[potential.type].particle_idxs + idx_offset

            for (i, j, k), (constant, angle) in zip(atom_idxs, parameters):
                force.addAngle(
                    i,
                    j,
                    k,
                    angle * _RADIANS,
                    constant * _KCAL_PER_MOL / _RADIANS**2,
                )

            idx_offset += topology.n_particles

    return force


def _convert_torsion_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.PeriodicTorsionForce:
    force = openmm.PeriodicTorsionForce()

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameters = (
            topology.parameters[potential.type].assignment_matrix @ potential.parameters
        )

        for _ in range(n_copies):
            atom_idxs = topology.parameters[potential.type].particle_idxs + idx_offset

            for (i, j, k, l), (constant, periodicity, phase, idivf) in zip(
                atom_idxs, parameters
            ):
                force.addTorsion(
                    i,
                    j,
                    k,
                    l,
                    int(periodicity),
                    phase * _RADIANS,
                    constant / idivf * _KCAL_PER_MOL,
                )

            idx_offset += topology.n_particles

    return force


def _combine_nonbonded(
    vdw_force: openmm.NonbondedForce, electrostatic_force: openmm.NonbondedForce
) -> openmm.NonbondedForce:
    assert vdw_force.getNumParticles() == electrostatic_force.getNumParticles()
    assert vdw_force.getNumExceptions() == electrostatic_force.getNumExceptions()
    assert vdw_force.getNonbondedMethod() == electrostatic_force.getNonbondedMethod()
    assert vdw_force.getCutoffDistance() == electrostatic_force.getCutoffDistance()

    force = copy.deepcopy(vdw_force)
    force.setEwaldErrorTolerance(electrostatic_force.getEwaldErrorTolerance())

    for i in range(force.getNumParticles()):
        charge, _, _ = electrostatic_force.getParticleParameters(i)
        _, sigma, epsilon = vdw_force.getParticleParameters(i)

        force.setParticleParameters(i, charge, sigma, epsilon)

    vdw_exceptions, electrostatic_exceptions = {}, {}

    for index in range(vdw_force.getNumExceptions()):
        i, j, *values = vdw_force.getExceptionParameters(index)
        vdw_exceptions[(i, j)] = (index, *values)

    for index in range(electrostatic_force.getNumExceptions()):
        i, j, *values = electrostatic_force.getExceptionParameters(index)
        electrostatic_exceptions[(i, j)] = values

    for (i, j), (charge_prod, _, _) in electrostatic_exceptions.items():
        index, _, sigma, epsilon = vdw_exceptions[(i, j)]
        force.setExceptionParameters(index, i, j, charge_prod, sigma, epsilon)

    return force


def create_openmm_system(
    system: smee.TensorSystem, v_sites: smee.TensorVSites | None
) -> openmm.System:
    v_sites = None if v_sites is None else v_sites.to("cpu")
    system = system.to("cpu")

    omm_system = openmm.System()

    for topology, n_copies in zip(system.topologies, system.n_copies):
        for _ in range(n_copies):
            start_idx = omm_system.getNumParticles()

            for atomic_num in topology.atomic_nums:
                mass = openmm.app.Element.getByAtomicNumber(int(atomic_num)).mass
                omm_system.addParticle(mass)

            if topology.v_sites is None:
                continue

            for _ in range(topology.n_v_sites):
                omm_system.addParticle(0.0)

            for key, parameter_idx in zip(
                topology.v_sites.keys, topology.v_sites.parameter_idxs
            ):
                system_idx = start_idx + topology.v_sites.key_to_idx[key]
                assert system_idx >= start_idx

                parent_idxs = [i + start_idx for i in key.orientation_atom_indices]

                local_frame_coords = smee.geometry.polar_to_cartesian_coords(
                    v_sites.parameters[[parameter_idx], :]
                )
                origin, x_dir, y_dir = v_sites.weights[parameter_idx]

                v_site = openmm.LocalCoordinatesSite(
                    parent_idxs,
                    origin.numpy(),
                    x_dir.numpy(),
                    y_dir.numpy(),
                    local_frame_coords.numpy().flatten() * _ANGSTROM_TO_NM,
                )

                omm_system.setVirtualSite(system_idx, v_site)

    return omm_system


def _apply_constraints(omm_system: openmm.System, system: smee.TensorSystem):
    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        if topology.constraints is None:
            continue

        for _ in range(n_copies):
            atom_idxs = topology.constraints.idxs + idx_offset

            for (i, j), distance in zip(atom_idxs, topology.constraints.distances):
                omm_system.addConstraint(i, j, distance * _ANGSTROM)

            idx_offset += topology.n_particles


def convert_to_openmm_force(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> list[openmm.Force]:
    potential = potential.to("cpu")
    system = system.to("cpu")

    if potential.exceptions is not None and potential.type != "vdW":
        raise NotImplementedError("exceptions are only supported for vdW potentials")

    if potential.type == "Electrostatics":
        return [_convert_electrostatics_potential(potential, system)]
    elif potential.type == "vdW":
        return _convert_vdw_potential(potential, system)
    elif potential.type == "Bonds":
        return [_convert_bond_potential(potential, system)]
    elif potential.type == "Angles":
        return [_convert_angle_potential(potential, system)]
    elif potential.type == "ProperTorsions":
        return [_convert_torsion_potential(potential, system)]
    elif potential.type == "ImproperTorsions":
        return [_convert_torsion_potential(potential, system)]

    raise NotImplementedError(f"unsupported potential type {potential.type}")


def convert_to_openmm_system(
    force_field: smee.TensorForceField,
    system: smee.TensorSystem | smee.TensorTopology,
) -> openmm.System:
    """Convert a SMEE force field and system / topology into an OpenMM system.

    Args:
        force_field: The force field parameters.
        system: The system / topology to convert.

    Returns:
        The OpenMM system.
    """

    system: smee.TensorSystem = (
        system
        if isinstance(system, smee.TensorSystem)
        else smee.TensorSystem([system], [1], False)
    )

    force_field = force_field.to("cpu")
    system = system.to("cpu")

    omm_forces = {
        potential_type: convert_to_openmm_force(potential, system)
        for potential_type, potential in force_field.potentials_by_type.items()
    }
    omm_system = create_openmm_system(system, force_field.v_sites)

    if (
        "Electrostatics" in omm_forces
        and "vdW" in omm_forces
        and len(omm_forces["vdW"]) == 1
        and isinstance(omm_forces["vdW"][0], openmm.NonbondedForce)
    ):
        (electrostatic_force,) = omm_forces.pop("Electrostatics")
        (vdw_force,) = omm_forces.pop("vdW")

        nonbonded_force = _combine_nonbonded(vdw_force, electrostatic_force)
        omm_system.addForce(nonbonded_force)

    for forces in omm_forces.values():
        for force in forces:
            omm_system.addForce(force)

    _apply_constraints(omm_system, system)

    return omm_system


def convert_to_openmm_topology(system: smee.TensorSystem) -> openmm.app.Topology:
    """Convert a SMEE system to an OpenMM topology."""
    omm_topology = openmm.app.Topology()

    for topology, n_copies in zip(system.topologies, system.n_copies):
        chain = omm_topology.addChain()

        is_water = topology.n_atoms == 3 and sorted(
            int(v) for v in topology.atomic_nums
        ) == [1, 1, 8]

        residue_name = "WAT" if is_water else "UNK"

        for _ in range(n_copies):
            residue = omm_topology.addResidue(residue_name, chain)
            element_counter = collections.defaultdict(int)

            atoms = {}

            for i, atomic_num in enumerate(topology.atomic_nums):
                element = openmm.app.Element.getByAtomicNumber(int(atomic_num))
                element_counter[element.symbol] += 1

                name = element.symbol + (
                    ""
                    if element_counter[element.symbol] == 1 and element.symbol != "H"
                    else f"{element_counter[element.symbol]}"
                )
                atoms[i] = omm_topology.addAtom(name, element, residue)

            for i in range(topology.n_v_sites):
                omm_topology.addAtom(
                    "X", openmm.app.Element.getByAtomicNumber(82), residue
                )

            for bond_idxs, bond_order in zip(topology.bond_idxs, topology.bond_orders):
                idx_a, idx_b = int(bond_idxs[0]), int(bond_idxs[1])

                bond_order = int(bond_order)
                bond_type = {
                    1: openmm.app.Single,
                    2: openmm.app.Double,
                    3: openmm.app.Triple,
                }[bond_order]

                omm_topology.addBond(atoms[idx_a], atoms[idx_b], bond_type, bond_order)

    return omm_topology
