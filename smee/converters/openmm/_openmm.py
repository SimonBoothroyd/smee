import collections
import copy
import importlib
import typing

import openmm
import openmm.app

import smee

_ANGSTROM = openmm.unit.angstrom
_ANGSTROM_TO_NM = 1.0 / 10.0


_CONVERTER_FUNCTIONS: dict[
    tuple[str, str],
    typing.Callable[
        [smee.TensorPotential, smee.TensorSystem], openmm.Force | list[openmm.Force]
    ],
] = {}


def potential_converter(handler_type: str, energy_expression: str):
    """A decorator used to flag a function as being able to convert a tensor potential
    of a given type and energy function to an OpenMM force.
    """

    def _openmm_converter_inner(func):
        if (handler_type, energy_expression) in _CONVERTER_FUNCTIONS:
            raise KeyError(
                f"An OpenMM converter function is already defined for "
                f"handler={handler_type} fn={energy_expression}."
            )

        _CONVERTER_FUNCTIONS[(str(handler_type), str(energy_expression))] = func
        return func

    return _openmm_converter_inner


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
    """Create an empty OpenMM system from a ``smee`` system."""
    v_sites = None if v_sites is None else v_sites.to("cpu")
    system = system.to("cpu")

    omm_system = openmm.System()

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
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
                topology.v_sites.keys, topology.v_sites.parameter_idxs, strict=True
            ):
                system_idx = start_idx + topology.v_sites.key_to_idx[key]
                assert system_idx >= start_idx

                parent_idxs = [i + start_idx for i in key.orientation_atom_indices]

                local_frame_coords = smee.geometry.polar_to_cartesian_coords(
                    v_sites.parameters[[parameter_idx], :].detach()
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

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        if topology.constraints is None:
            continue

        for _ in range(n_copies):
            atom_idxs = topology.constraints.idxs + idx_offset

            for (i, j), distance in zip(
                atom_idxs, topology.constraints.distances, strict=True
            ):
                omm_system.addConstraint(i, j, distance * _ANGSTROM)

            idx_offset += topology.n_particles


def convert_to_openmm_force(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> list[openmm.Force]:
    """Convert a ``smee`` potential to OpenMM forces.

    Some potentials may return multiple forces, e.g. a vdW potential may return one
    force containing intermolecular interactions and another containing intramolecular
    interactions.

    See Also:
        potential_converter: for how to define a converter function.

    Args:
        potential: The potential to convert.
        system: The system to convert.

    Returns:
        The OpenMM force(s).
    """
    # register the built-in converter functions
    importlib.import_module("smee.converters.openmm.nonbonded")
    importlib.import_module("smee.converters.openmm.valence")

    potential = potential.to("cpu")
    system = system.to("cpu")

    if potential.exceptions is not None and potential.type != "vdW":
        raise NotImplementedError("exceptions are only supported for vdW potentials")

    converter_key = (str(potential.type), str(potential.fn))

    if converter_key not in _CONVERTER_FUNCTIONS:
        raise NotImplementedError(
            f"cannot convert type={potential.type} fn={potential.fn} to an OpenMM force"
        )

    forces = _CONVERTER_FUNCTIONS[converter_key](potential, system)
    return forces if isinstance(forces, (list, tuple)) else [forces]


def convert_to_openmm_system(
    force_field: smee.TensorForceField,
    system: smee.TensorSystem | smee.TensorTopology,
) -> openmm.System:
    """Convert a ``smee`` force field and system / topology into an OpenMM system.

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


def convert_to_openmm_topology(
    system: smee.TensorSystem | smee.TensorTopology,
) -> openmm.app.Topology:
    """Convert a ``smee`` system to an OpenMM topology.

    Notes:
        Virtual sites are given the name "X{i}".

    Args:
        system: The system to convert.

    Returns:
        The OpenMM topology.
    """
    system: smee.TensorSystem = (
        system
        if isinstance(system, smee.TensorSystem)
        else smee.TensorSystem([system], [1], False)
    )

    omm_topology = openmm.app.Topology()

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        chain = omm_topology.addChain()

        is_water = topology.n_atoms == 3 and sorted(
            int(v) for v in topology.atomic_nums
        ) == [1, 1, 8]

        residue_name = "HOH" if is_water else "UNK"

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
                omm_topology.addAtom(f"X{i + 1}", None, residue)

            for bond_idxs, bond_order in zip(
                topology.bond_idxs, topology.bond_orders, strict=True
            ):
                idx_a, idx_b = int(bond_idxs[0]), int(bond_idxs[1])

                bond_order = int(bond_order)
                bond_type = {
                    1: openmm.app.Single,
                    2: openmm.app.Double,
                    3: openmm.app.Triple,
                }[bond_order]

                omm_topology.addBond(atoms[idx_a], atoms[idx_b], bond_type, bond_order)

    return omm_topology
