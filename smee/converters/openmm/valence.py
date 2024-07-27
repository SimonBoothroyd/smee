"""Convert valence potentials to OpenMM forces."""

import openmm
import openmm.app

import smee

_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_ANGSTROM = openmm.unit.angstrom
_RADIANS = openmm.unit.radians


@smee.converters.openmm.potential_converter(
    smee.PotentialType.BONDS, smee.EnergyFn.BOND_HARMONIC
)
def convert_bond_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.HarmonicBondForce:
    """Convert a harmonic bond potential to a corresponding OpenMM force."""
    force = openmm.HarmonicBondForce()

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameters = (
            topology.parameters[potential.type].assignment_matrix @ potential.parameters
        ).detach()

        for _ in range(n_copies):
            atom_idxs = topology.parameters[potential.type].particle_idxs + idx_offset

            for (i, j), (constant, length) in zip(atom_idxs, parameters, strict=True):
                force.addBond(
                    i,
                    j,
                    length * _ANGSTROM,
                    constant * _KCAL_PER_MOL / _ANGSTROM**2,
                )

            idx_offset += topology.n_particles

    return force


@smee.converters.openmm.potential_converter(
    smee.PotentialType.ANGLES, smee.EnergyFn.ANGLE_HARMONIC
)
def _convert_angle_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.HarmonicAngleForce:
    """Convert a harmonic angle potential to a corresponding OpenMM force."""
    force = openmm.HarmonicAngleForce()

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameters = (
            topology.parameters[potential.type].assignment_matrix @ potential.parameters
        ).detach()

        for _ in range(n_copies):
            atom_idxs = topology.parameters[potential.type].particle_idxs + idx_offset

            for (i, j, k), (constant, angle) in zip(atom_idxs, parameters, strict=True):
                force.addAngle(
                    i,
                    j,
                    k,
                    angle * _RADIANS,
                    constant * _KCAL_PER_MOL / _RADIANS**2,
                )

            idx_offset += topology.n_particles

    return force


@smee.converters.openmm.potential_converter(
    smee.PotentialType.PROPER_TORSIONS, smee.EnergyFn.TORSION_COSINE
)
@smee.converters.openmm.potential_converter(
    smee.PotentialType.IMPROPER_TORSIONS, smee.EnergyFn.TORSION_COSINE
)
def convert_torsion_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.PeriodicTorsionForce:
    """Convert a torsion potential to a corresponding OpenMM force."""
    force = openmm.PeriodicTorsionForce()

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameters = (
            topology.parameters[potential.type].assignment_matrix @ potential.parameters
        ).detach()

        for _ in range(n_copies):
            atom_idxs = topology.parameters[potential.type].particle_idxs + idx_offset

            for (idx_i, idx_j, idx_k, idx_l), (
                constant,
                periodicity,
                phase,
                idivf,
            ) in zip(atom_idxs, parameters, strict=True):
                force.addTorsion(
                    idx_i,
                    idx_j,
                    idx_k,
                    idx_l,
                    int(periodicity),
                    phase * _RADIANS,
                    constant / idivf * _KCAL_PER_MOL,
                )

            idx_offset += topology.n_particles

    return force
