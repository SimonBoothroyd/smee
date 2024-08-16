"""Convert non-bonded potentials to OpenMM forces."""

import itertools
import re
import typing

import openmm
import torch

import smee
import smee.converters.openmm
import smee.potentials.nonbonded
import smee.utils

_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_ANGSTROM = openmm.unit.angstrom

_INTRA_SCALE_VAR = "scale_excl"
"""The variable name used to scale the 1-n intramolecular interactions."""

_T = typing.TypeVar("_T", bound=openmm.NonbondedForce | openmm.CustomNonbondedForce)


def _create_nonbonded_force(
    potential: smee.TensorPotential,
    system: smee.TensorSystem,
    cls: typing.Type[_T] = openmm.NonbondedForce,
) -> _T:
    """Create a non-bonded force for a given potential and system, making sure to set
    the appropriate method and cutoffs."""
    if cls == openmm.NonbondedForce:
        force = openmm.NonbondedForce()
        force.setUseDispersionCorrection(system.is_periodic)
        force.setEwaldErrorTolerance(1.0e-4)  # TODO: interchange hardcoded value
    elif cls == openmm.CustomNonbondedForce:
        force = openmm.CustomNonbondedForce("")
        force.setUseLongRangeCorrection(system.is_periodic)
    else:
        raise NotImplementedError(f"unsupported force class {cls}")

    cutoff_idx = potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)
    switch_idx = (
        None
        if smee.SWITCH_ATTRIBUTE not in potential.attribute_cols
        else potential.attribute_cols.index(smee.SWITCH_ATTRIBUTE)
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


def _eval_mixing_fn(
    potential: smee.TensorPotential,
    mixing_fn: dict[str, str],
    param_1: torch.Tensor,
    param_2: torch.Tensor,
) -> dict[str, float]:
    import symengine

    values = {}

    for col in mixing_fn:
        col_idx = potential.parameter_cols.index(col)

        fn = symengine.sympify(mixing_fn[col].strip().strip(";").replace("^", "**"))

        values[col] = fn.subs(
            {
                symengine.Symbol(f"{col}1"): float(param_1[col_idx]),
                symengine.Symbol(f"{col}2"): float(param_2[col_idx]),
            }
        )

    return values


def _build_vdw_lookup(
    potential: smee.TensorPotential,
    mixing_fn: dict[str, str],
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
                col: potential.parameters[exceptions[i, j], col_idx].detach()
                for col, col_idx in parameter_col_to_idx.items()
            }
        else:
            parameters = _eval_mixing_fn(
                potential,
                mixing_fn,
                potential.parameters[i].detach(),
                potential.parameters[j].detach(),
            )

        unit_conversion = {
            col: (1.0 * potential.parameter_units[col_idx])
            .to_openmm()
            .value_in_unit_system(openmm.unit.md_unit_system)
            for col, col_idx in parameter_col_to_idx.items()
        }

        for col in parameter_col_to_idx:
            parameter_lookup[col][i + j * n_params] = float(
                parameters[col] * unit_conversion[col]
            )

    return parameter_lookup


def _prepend_scale_to_energy_fn(fn: str, scale_var: str = _INTRA_SCALE_VAR) -> str:
    """Prepend a scale variable to the return value of an energy function."""
    assert scale_var not in fn, f"1-n scale variable {scale_var} already in energy fn"

    fn_split = fn.split(";")
    assert "=" not in fn_split[0], "energy function missing a return value"

    fn_split[0] = f"{scale_var}*({fn_split[0]})"
    return ";".join(fn_split)


def _detect_parameters(
    potential: smee.TensorPotential, energy_fn: str, mixing_fn: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Detect the required parameters and attributes for a given energy function
    and associated mixing rules."""
    import symengine

    energy_fn = energy_fn.strip().strip(";")

    assigned_vars = set("r")
    free_vars = set()

    for line in reversed(energy_fn.split(";")):  # OMM parses from the end
        if "=" in line:
            assigned_var, line = line.split("=", 1)
            assigned_vars.add(assigned_var.strip())

        parsed_fn = symengine.sympify(line)
        free_vars.update({str(x) for x in parsed_fn.free_symbols} - assigned_vars)

    for assigned_var, fn in mixing_fn.items():
        fn = fn.strip().strip(";")

        assert len(fn.split(";")) == 1, "mixing functions must be single line"
        assert "=" not in fn, "mixing functions must not have an assignment"

        assigned_vars.add(assigned_var.strip())

        parsed_fn = symengine.sympify(fn)
        free_vars.update({str(x) for x in parsed_fn.free_symbols} - assigned_vars)

    free_vars -= assigned_vars

    parameter_vars = set(potential.parameter_cols)
    attribute_vars = set(potential.attribute_cols)

    overlapping_vars = set.intersection(parameter_vars, attribute_vars)
    assert len(overlapping_vars) == 0, "parameters and attributes must be unique"

    free_parameters = {v[:-1] for v in free_vars if v[-1] in {"1", "2"}}
    free_attributes = {v for v in free_vars if v[:-1] not in free_parameters}

    required_parameters = free_parameters.intersection(parameter_vars)
    required_attributes = free_attributes.intersection(attribute_vars)

    missing_parameters = required_parameters - parameter_vars
    assert len(missing_parameters) == 0, f"missing parameters: {missing_parameters}"

    missing_attributes = required_attributes - attribute_vars
    assert len(missing_attributes) == 0, f"missing attributes: {missing_attributes}"

    return sorted(required_parameters), sorted(required_attributes)


def _extract_parameters(
    potential: smee.TensorPotential, parameter: torch.Tensor, cols: list[str]
) -> list[float]:
    """Extract the values of a subset of parameters from a parameter tensor."""

    values = []

    for col in cols:
        col_idx = potential.parameter_cols.index(col)

        unit_conversion = (
            (1.0 * potential.parameter_units[col_idx])
            .to_openmm()
            .value_in_unit_system(openmm.unit.md_unit_system)
        )

        values.append(parameter[col_idx] * unit_conversion)

    return values


def _add_parameters_to_vdw_without_lookup(
    potential: smee.TensorPotential,
    system: smee.TensorSystem,
    energy_fn: str,
    mixing_fn: dict[str, str],
    inter_force: openmm.CustomNonbondedForce,
    intra_force: openmm.CustomBondForce,
    used_parameters: list[str],
):
    """Add parameters to a vdW force directly, i.e. without using a lookup table."""

    energy_fn = ";".join(
        [energy_fn.strip().strip(";")]
        + [f"{var}={rule.strip().strip(';')}" for var, rule in mixing_fn.items()]
    )

    inter_force_energy_fn = energy_fn
    intra_force_energy_fn = _prepend_scale_to_energy_fn(energy_fn, _INTRA_SCALE_VAR)

    inter_force.setEnergyFunction(inter_force_energy_fn)
    intra_force.setEnergyFunction(intra_force_energy_fn)

    for param in used_parameters:
        inter_force.addPerParticleParameter(param)

    for i in (1, 2):
        for param in used_parameters:
            intra_force.addPerBondParameter(f"{param}{i}")

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters.detach()

        for _ in range(n_copies):
            for parameter in parameters:
                values = _extract_parameters(potential, parameter, used_parameters)
                inter_force.addParticle(values)

            for index, (i, j) in enumerate(parameter_map.exclusions):
                values_i = _extract_parameters(
                    potential, parameters[i, :], used_parameters
                )
                values_j = _extract_parameters(
                    potential, parameters[j, :], used_parameters
                )

                scale = potential.attributes[parameter_map.exclusion_scale_idxs[index]]

                inter_force.addExclusion(i + idx_offset, j + idx_offset)

                if torch.isclose(scale, smee.utils.tensor_like(0.0, scale)):
                    continue

                intra_force.addBond(
                    i + idx_offset, j + idx_offset, [float(scale), *values_i, *values_j]
                )

            idx_offset += topology.n_particles


def _add_parameters_to_vdw_with_lookup(
    potential: smee.TensorPotential,
    system: smee.TensorSystem,
    energy_fn: str,
    mixing_fn: dict[str, str],
    inter_force: openmm.CustomNonbondedForce,
    intra_force: openmm.CustomBondForce,
):
    """Add parameters to a vdW force, explicitly defining all pairwise parameters
    using a lookup table."""
    n_params = len(potential.parameters)

    parameter_lookup = _build_vdw_lookup(potential, mixing_fn)

    inter_force_energy_fn = energy_fn + " ".join(
        f"{col}={col}_lookup(param_idx1, param_idx2);" for col in parameter_lookup
    )
    inter_force.setEnergyFunction(inter_force_energy_fn)
    inter_force.addPerParticleParameter("param_idx")

    for col, vals in parameter_lookup.items():
        lookup_table = openmm.Discrete2DFunction(n_params, n_params, vals)
        inter_force.addTabulatedFunction(f"{col}_lookup", lookup_table)

    for col in parameter_lookup:
        intra_force.addPerBondParameter(col)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
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

                if torch.isclose(scale, torch.zeros_like(scale)):
                    continue

                intra_parameters = [scale] + [
                    vals[assigned_idxs[i] + assigned_idxs[j] * n_params]
                    for col, vals in parameter_lookup.items()
                ]
                intra_force.addBond(
                    int(i + idx_offset), int(j + idx_offset), intra_parameters
                )

            idx_offset += topology.n_particles


def convert_custom_vdw_potential(
    potential: smee.TensorPotential,
    system: smee.TensorSystem,
    energy_fn: str,
    mixing_fn: dict[str, str],
) -> tuple[openmm.CustomNonbondedForce, openmm.CustomBondForce]:
    """Converts an arbitrary vdW potential to OpenMM forces.

    The intermolecular interactions are described by a custom nonbonded force, while the
    intramolecular interactions are described by a custom bond force.

    If the potential has custom mixing rules (i.e. exceptions), a lookup table will be
    used to store the parameters. Otherwise, the mixing rules will be applied directly
    in the energy function.

    Args:
        potential: The potential to convert.
        system: The system the potential belongs to.
        energy_fn: The energy function of the potential, written in OpenMM's custom
            energy function syntax.
        mixing_fn: A dictionary of mixing rules for each parameter of the potential.
            The keys are the parameter names, and the values are the mixing rules.

            The mixing rules should be a single expression that can be evaluated using
            OpenMM's energy function syntax, and should not contain any assignments.

    Examples:
        For a Lennard-Jones potential using Lorentz-Berthelot mixing rules:

        >>> energy_fn = "4*epsilon*x6*(x6 - 1.0);x6=x4*x2;x4=x2*x2;x2=x*x;x=sigma/r;"
        >>> mixing_fn = {
        ...     "epsilon": "sqrt(epsilon1 * epsilon2)",
        ...     "sigma": "0.5 * (sigma1 + sigma2)",
        ... }
    """
    energy_fn = re.sub(r"\s+", "", energy_fn)
    mixing_fn = {k: re.sub(r"\s+", "", v) for k, v in mixing_fn.items()}

    used_parameters, used_attributes = _detect_parameters(
        potential, energy_fn, mixing_fn
    )
    requires_lookup = potential.exceptions is not None

    inter_force = _create_nonbonded_force(
        potential, system, openmm.CustomNonbondedForce
    )
    inter_force.setEnergyFunction(energy_fn)

    intra_force = openmm.CustomBondForce(
        _prepend_scale_to_energy_fn(energy_fn, _INTRA_SCALE_VAR)
    )
    intra_force.addPerBondParameter(_INTRA_SCALE_VAR)
    intra_force.setUsesPeriodicBoundaryConditions(system.is_periodic)

    for force in [inter_force, intra_force]:
        for attr in used_attributes:
            attr_unit = potential.attribute_units[potential.attribute_cols.index(attr)]
            attr_conv = (
                (1.0 * attr_unit)
                .to_openmm()
                .value_in_unit_system(openmm.unit.md_unit_system)
            )
            attr_idx = potential.attribute_cols.index(attr)
            attr_val = float(potential.attributes[attr_idx]) * attr_conv

            force.addGlobalParameter(attr, attr_val)

    if requires_lookup:
        _add_parameters_to_vdw_with_lookup(
            potential, system, energy_fn, mixing_fn, inter_force, intra_force
        )
    else:
        _add_parameters_to_vdw_without_lookup(
            potential,
            system,
            energy_fn,
            mixing_fn,
            inter_force,
            intra_force,
            used_parameters,
        )

    return inter_force, intra_force


@smee.converters.openmm.potential_converter(
    smee.PotentialType.VDW, smee.EnergyFn.VDW_LJ
)
def convert_lj_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.NonbondedForce | list[openmm.CustomNonbondedForce | openmm.CustomBondForce]:
    """Convert a Lennard-Jones potential to an OpenMM force.

    If the potential has custom mixing rules (i.e. exceptions), the interactions will
    be split into an inter- and intra-molecular force.
    """
    energy_fn = "4*epsilon*x6*(x6 - 1.0);x6=x4*x2;x4=x2*x2;x2=x*x;x=sigma/r;"
    mixing_fn = {
        "epsilon": "sqrt(epsilon1 * epsilon2)",
        "sigma": "0.5 * (sigma1 + sigma2)",
    }

    if potential.exceptions is not None:
        return list(
            convert_custom_vdw_potential(potential, system, energy_fn, mixing_fn)
        )

    force = _create_nonbonded_force(potential, system)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters.detach()

        for _ in range(n_copies):
            for epsilon, sigma in parameters:
                force.addParticle(0.0, sigma * _ANGSTROM, epsilon * _KCAL_PER_MOL)

            for index, (i, j) in enumerate(parameter_map.exclusions):
                scale = potential.attributes[parameter_map.exclusion_scale_idxs[index]]

                eps_i, sig_i = parameters[i, :]
                eps_j, sig_j = parameters[j, :]

                eps, sig = smee.potentials.nonbonded.lorentz_berthelot(
                    eps_i, eps_j, sig_i, sig_j
                )

                force.addException(
                    i + idx_offset,
                    j + idx_offset,
                    0.0,
                    float(sig) * _ANGSTROM,
                    float(eps * scale) * _KCAL_PER_MOL,
                )

            idx_offset += topology.n_particles

    return force


@smee.converters.openmm.potential_converter(
    smee.PotentialType.VDW, smee.EnergyFn.VDW_DEXP
)
def convert_dexp_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> tuple[openmm.CustomNonbondedForce, openmm.CustomBondForce]:
    """Convert a DEXP potential to OpenMM forces.

    The intermolcular interactions are described by a custom nonbonded force, while the
    intramolecular interactions are described by a custom bond force.

    If the potential has custom mixing rules (i.e. exceptions), a lookup table will be
    used to store the parameters. Otherwise, the mixing rules will be applied directly
    in the energy function.
    """
    energy_fn = (
        "epsilon * (repulsion - attraction);"
        "repulsion  = beta  / (alpha - beta) * exp(alpha * (1 - x));"
        "attraction = alpha / (alpha - beta) * exp(beta  * (1 - x));"
        "x = r / r_min;"
    )
    mixing_fn = {
        "epsilon": "sqrt(epsilon1 * epsilon2)",
        "r_min": "0.5 * (r_min1 + r_min2)",
    }

    return convert_custom_vdw_potential(potential, system, energy_fn, mixing_fn)


@smee.converters.openmm.potential_converter(
    smee.PotentialType.VDW, smee.EnergyFn.VDW_DAMPEDEXP6810
)
def convert_dampedexp6810_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> tuple[openmm.CustomNonbondedForce, openmm.CustomBondForce]:
    """Convert a DampedExp6810 potential to OpenMM forces.

    The intermolcular interactions are described by a custom nonbonded force, while the
    intramolecular interactions are described by a custom bond force.

    If the potential has custom mixing rules (i.e. exceptions), a lookup table will be
    used to store the parameters. Otherwise, the mixing rules will be applied directly
    in the energy function.
    """
    energy_fn = (
        "repulsion - ttdamp6*c6*invR^6 - ttdamp8*c8*invR^8 - ttdamp10*c10*invR^10;"
        "repulsion = force_at_zero*invbeta*exp(-beta*(r-rho));"
        "ttdamp10 = select(expbr, 1.0 - expbr * ttdamp10Sum, 1);"
        "ttdamp8 = select(expbr, 1.0 - expbr * ttdamp8Sum, 1);"
        "ttdamp6 = select(expbr, 1.0 - expbr * ttdamp6Sum, 1);"
        "ttdamp10Sum = ttdamp8Sum + br^9/362880 + br^10/3628800;"
        "ttdamp8Sum = ttdamp6Sum + br^7/5040 + br^8/40320;"
        "ttdamp6Sum = 1.0 + br + br^2/2 + br^3/6 + br^4/24 + br^5/120 + br^6/720;"
        "expbr = exp(-br);"
        "br = beta*r;"
        "invR = 1.0/r;"
        "invbeta = 1.0/beta;"
    )
    mixing_fn = {
        "beta": "2.0 * beta1 * beta2 / (beta1 + beta2)",
        "rho": "0.5 * (rho1 + rho2)",
        "c6": "sqrt(c61*c62)",
        "c8": "sqrt(c81*c82)",
        "c10": "sqrt(c101*c102)",
    }

    return convert_custom_vdw_potential(potential, system, energy_fn, mixing_fn)


@smee.converters.openmm.potential_converter(
    smee.PotentialType.ELECTROSTATICS, smee.EnergyFn.POLARIZATION
)
def convert_multipole_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.AmoebaMultipoleForce:
    """Convert a Multipole potential to OpenMM forces."""

    thole = 0.39
    cutoff_idx = potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)
    cutoff = float(potential.attributes[cutoff_idx]) * _ANGSTROM

    force: openmm.AmoebaMultipoleForce = openmm.AmoebaMultipoleForce()

    if system.is_periodic:
        force.setNonbondedMethod(openmm.AmoebaMultipoleForce.PME)
    else:
        force.setNonbondedMethod(openmm.AmoebaMultipoleForce.NoCutoff)
    force.setPolarizationType(openmm.AmoebaMultipoleForce.Mutual)
    force.setCutoffDistance(cutoff)
    force.setEwaldErrorTolerance(0.0001)
    force.setMutualInducedTargetEpsilon(0.00001)
    force.setMutualInducedMaxIterations(60)
    force.setExtrapolationCoefficients([-0.154, 0.017, 0.658, 0.474])

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters
        parameters = parameters.detach().tolist()

        for _ in range(n_copies):
            for _ in range(topology.n_particles):
                force.addMultipole(
                    0,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    openmm.AmoebaMultipoleForce.NoAxisType,
                    -1,
                    -1,
                    -1,
                    thole,
                    0,
                    0,
                )

            for idx, parameter in enumerate(parameters):
                omm_idx = idx % topology.n_particles + idx_offset
                omm_params = force.getMultipoleParameters(omm_idx)
                if idx // topology.n_atoms == 0:
                    omm_params[0] = parameter[0] * openmm.unit.elementary_charge
                else:
                    omm_params[8] = (parameter[1] / 1000) ** (1 / 6)
                    omm_params[9] = parameter[1] * _ANGSTROM**3
                force.setMultipoleParameters(omm_idx, *omm_params)

            covalent_maps = {}

            for i, j in parameter_map.exclusions:
                i = int(i) + idx_offset
                j = int(j) + idx_offset
                if i in covalent_maps.keys():
                    covalent_maps[i].append(j)
                else:
                    covalent_maps[i] = [j]
                if j in covalent_maps.keys():
                    covalent_maps[j].append(i)
                else:
                    covalent_maps[j] = [i]

            for i in covalent_maps.keys():
                force.setCovalentMap(
                    i, openmm.AmoebaMultipoleForce.Covalent12, covalent_maps[i]
                )
                force.setCovalentMap(
                    i,
                    openmm.AmoebaMultipoleForce.PolarizationCovalent11,
                    covalent_maps[i],
                )

            idx_offset += topology.n_particles

    return force


@smee.converters.openmm.potential_converter(
    smee.PotentialType.ELECTROSTATICS, smee.EnergyFn.COULOMB
)
def convert_coulomb_potential(
    potential: smee.TensorPotential, system: smee.TensorSystem
) -> openmm.NonbondedForce:
    """Convert a Coulomb potential to an OpenMM force."""
    force = _create_nonbonded_force(potential, system)

    idx_offset = 0

    for topology, n_copies in zip(system.topologies, system.n_copies, strict=True):
        parameter_map = topology.parameters[potential.type]
        parameters = parameter_map.assignment_matrix @ potential.parameters.detach()

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
