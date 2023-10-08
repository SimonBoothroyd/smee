"""Convert tensor representations into OpenMM systems."""
import copy

import openmm
import openmm.app

import smee.ff

_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_ANGSTROM = openmm.unit.angstrom
_RADIANS = openmm.unit.radians


def _create_nonbonded_force(
    potential: smee.ff.TensorPotential, system: smee.ff.TensorSystem
) -> openmm.NonbondedForce:
    force = openmm.NonbondedForce()
    force.setUseDispersionCorrection(system.is_periodic)

    cutoff_idx = potential.attribute_cols.index("cutoff")

    if not system.is_periodic:
        force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    else:
        force.setNonbondedMethod(openmm.NonbondedForce.PME)
        force.setCutoffDistance(potential.attributes[cutoff_idx] * _ANGSTROM)

    return force


def _convert_lj_potential(
    potential: smee.ff.TensorPotential, system: smee.ff.TensorSystem
) -> openmm.NonbondedForce:
    force = _create_nonbonded_force(potential, system)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters

        for _ in range(n_copies):
            for epsilon, sigma in parameters:
                force.addParticle(
                    0.0,
                    sigma * _ANGSTROM,
                    epsilon * _KCAL_PER_MOL,
                )

            for index, (i, j) in enumerate(parameter_map.exclusions):
                eps_i, sigma_i = parameters[i, :]
                eps_j, sigma_j = parameters[j, :]

                eps = (eps_i * eps_j) ** 0.5
                sigma = (sigma_i + sigma_j) * 0.5

                scale = potential.attributes[parameter_map.exclusion_scale_idxs[index]]

                force.addException(
                    i + idx_offset,
                    j + idx_offset,
                    0.0,
                    sigma * _ANGSTROM,
                    scale * eps * _KCAL_PER_MOL,
                )

            idx_offset += topology.n_atoms

    return force


def _convert_electrostatics_potential(
    potential: smee.ff.TensorPotential, system: smee.ff.TensorSystem
) -> openmm.NonbondedForce:
    force = _create_nonbonded_force(potential, system)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters

        for _ in range(n_copies):
            for charge in parameters:
                force.addParticle(
                    charge * openmm.unit.elementary_charge,
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

            idx_offset += topology.n_atoms

        return force


def _convert_bond_potential(
    potential: smee.ff.TensorPotential, system: smee.ff.TensorSystem
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

            idx_offset += topology.n_atoms

    return force


def _convert_angle_potential(
    potential: smee.ff.TensorPotential, system: smee.ff.TensorSystem
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

            idx_offset += topology.n_atoms

    return force


def _convert_torsion_potential(
    potential: smee.ff.TensorPotential, system: smee.ff.TensorSystem
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
                    periodicity,
                    phase * _RADIANS,
                    constant / idivf * _KCAL_PER_MOL,
                )

            idx_offset += topology.n_atoms

    return force


def _combine_nonbonded(
    vdw_force: openmm.NonbondedForce | None,
    electrostatic_force: openmm.NonbondedForce | None,
) -> openmm.NonbondedForce:
    if electrostatic_force is None:
        return vdw_force
    if vdw_force is None:
        return electrostatic_force

    assert vdw_force.getNumParticles() == electrostatic_force.getNumParticles()
    assert vdw_force.getNumExceptions() == electrostatic_force.getNumExceptions()
    assert vdw_force.getNonbondedMethod() == electrostatic_force.getNonbondedMethod()
    assert vdw_force.getCutoffDistance() == electrostatic_force.getCutoffDistance()

    force = copy.deepcopy(vdw_force)

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


def create_openmm_system(system: smee.ff.TensorSystem) -> openmm.System:
    omm_system = openmm.System()

    for topology, n_copies in zip(system.topologies, system.n_copies):
        for _ in range(n_copies):
            for atomic_num in topology.atomic_nums:
                mass = openmm.app.Element.getByAtomicNumber(int(atomic_num)).mass
                omm_system.addParticle(mass)

    return omm_system


def _apply_constraints(omm_system: openmm.System, system: smee.ff.TensorSystem):
    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        if topology.constraints is None:
            continue

        for _ in range(n_copies):
            atom_idxs = topology.constraints.idxs + idx_offset

            for (i, j), distance in zip(atom_idxs, topology.constraints.distances):
                omm_system.addConstraint(i, j, distance * _ANGSTROM)

            idx_offset += topology.n_atoms


def convert_potential_to_force(
    potential: smee.ff.TensorPotential, system: smee.ff.TensorSystem
) -> openmm.Force:
    if potential.type == "Electrostatics":
        return _convert_electrostatics_potential(potential, system)
    if potential.type == "vdW":
        return _convert_lj_potential(potential, system)
    if potential.type == "Bonds":
        force = _convert_bond_potential(potential, system)
    elif potential.type == "Angles":
        force = _convert_angle_potential(potential, system)
    elif potential.type == "ProperTorsions":
        force = _convert_torsion_potential(potential, system)
    elif potential.type == "ImproperTorsions":
        force = _convert_torsion_potential(potential, system)
    else:
        raise NotImplementedError(f"unsupported potential type {potential.type}")

    return force


def convert_to_openmm(
    force_field: smee.ff.TensorForceField,
    system: smee.ff.TensorSystem | smee.ff.TensorTopology,
) -> openmm.System:
    """Convert a SMEE force field and system / topology into an OpenMM system.

    Args:
        force_field: The force field parameters.
        system: The system / topology to convert.

    Returns:
        The OpenMM system.
    """

    system: smee.ff.TensorSystem = (
        system
        if isinstance(system, smee.ff.TensorSystem)
        else smee.ff.TensorSystem([system], [1], False)
    )

    omm_forces = {
        potential_type: convert_potential_to_force(potential, system)
        for potential_type, potential in force_field.potentials_by_type.items()
    }
    omm_system = create_openmm_system(system)

    if "Electrostatics" in omm_forces and "vdW" in omm_forces:
        electrostatic_force = omm_forces.pop("Electrostatics")
        vdw_force = omm_forces.pop("vdW")

        nonbonded_force = _combine_nonbonded(vdw_force, electrostatic_force)
        omm_system.addForce(nonbonded_force)

    for force in omm_forces.values():
        omm_system.addForce(force)

    _apply_constraints(omm_system, system)

    return omm_system
