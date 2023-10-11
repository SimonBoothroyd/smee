"""Custom PyTorch ops for computing ensemble averages."""
import copy
import functools
import typing

import numpy
import openmm
import openmm.app
import openmm.unit
import torch

import smee.ff
import smee.mm._converters
import smee.mm._reporters

GRADIENT_DELTA = 1.0e-3
"""The finite difference step size to use when computing gradients."""


GRADIENT_EXCLUDED_ATTRIBUTES = ("cutoff", "switch_width", "scale_12", "scale_13")
"""Attributes whose gradients will not be computed and will always be zero."""


class _EnsembleAverageKwargs(typing.TypedDict):
    """The keyword arguments passed to the custom PyTorch op for computing ensemble
    averages."""

    force_field: smee.ff.TensorForceField
    parameter_lookup: dict[str, int]
    attribute_lookup: dict[str, int]

    system: smee.ff.TensorSystem

    coords_config: "smee.mm.GenerateCoordsConfig"
    equilibrate_configs: list[
        typing.Union["smee.mm.MinimizationConfig", "smee.mm.SimulationConfig"]
    ]
    production_config: "smee.mm.SimulationConfig"
    production_report_interval: int
    production_reporters: list[typing.Any]


class EnsembleAverages(typing.TypedDict):
    """Ensemble averages computed over an MD trajectory.

    The potential energy is in units of kcal/mol, the volume is in units of Å^3, the
    density is in units of g/mL, and the enthalpy is in units of kcal/mol.
    """

    potential_energy: torch.Tensor
    volume: torch.Tensor
    density: torch.Tensor
    enthalpy: torch.Tensor | None


def _pack_force_field(
    force_field: smee.ff.TensorForceField,
) -> tuple[tuple[torch.Tensor, ...], dict[str, int], dict[str, int]]:
    """Pack a SMEE force field into a tuple that can be consumed by a custom PyTorch op.

    Notes:
        Currently custom PyTorch functions don't try to find tensors in lists and dicts,
        so we have to flatten the force field into an args tuple.

    Returns:
        A tuple of tensors containing the force field parameters and attributes, and
        lookup tables mapping potential types to their corresponding parameters and
        attributes in the tuple.
    """
    potential_types = [potential.type for potential in force_field.potentials]

    parameters = [potential.parameters for potential in force_field.potentials]
    attributes = [potential.attributes for potential in force_field.potentials]

    tensors = tuple(parameters + attributes)

    parameter_lookup = {
        potential_type: i for i, potential_type in enumerate(potential_types)
    }
    attribute_lookup = {
        potential_type: i + len(parameters)
        for i, potential_type in enumerate(potential_types)
    }

    return tensors, parameter_lookup, attribute_lookup


def _unpack_force_field(
    tensors: tuple[torch.Tensor, ...],
    parameter_lookup: dict[str, int],
    attribute_lookup: dict[str, int],
    force_field: smee.ff.TensorForceField,
) -> smee.ff.TensorForceField:
    """Unpack a SMEE force field from its packed tensor and lookup representation.

    Args:
        tensors: The tuple of force field parameter and attribute tensors.
        parameter_lookup: The lookup mapping potential types to the corresponding
            parameter tensor in ``tensors``.
        attribute_lookup: The lookup mapping potential types to the corresponding
            attribute tensor in ``tensors``.
        force_field: The original force field that was packed.

    Returns:
        The unpacked force field.
    """
    potentials = []

    for original_potential in force_field.potentials:
        parameters = tensors[parameter_lookup[original_potential.type]]
        attributes = tensors[attribute_lookup[original_potential.type]]

        potential = smee.ff.TensorPotential(
            type=original_potential.type,
            fn=original_potential.fn,
            parameters=parameters,
            parameter_keys=original_potential.parameter_keys,
            parameter_cols=original_potential.parameter_cols,
            parameter_units=original_potential.parameter_units,
            attributes=attributes,
            attribute_cols=original_potential.attribute_cols,
            attribute_units=original_potential.attribute_units,
        )
        potentials.append(potential)

    return smee.ff.TensorForceField(potentials, force_field.v_sites)


def _compute_energy(
    system: smee.ff.TensorSystem,
    potential: smee.ff.TensorPotential,
    coords: numpy.ndarray,
    box_vectors: numpy.ndarray,
) -> torch.Tensor:
    """Compute the energy of an OpenMM system.

    Args:
        system: The system to evaluate.
        coords: The coordinates [Å] of the system.
        box_vectors: The box vectors [Å] of the system.

    Returns:
        The energy [kcal/mol] of the system.
    """
    omm_force = smee.mm._converters.convert_potential_to_force(potential, system)

    omm_system = smee.mm._converters.create_openmm_system(system)
    omm_system.addForce(omm_force)

    integrator = openmm.VerletIntegrator(0.0001)
    context = openmm.Context(omm_system, integrator)

    energies = []

    for coords_i, box_vectors_i in zip(coords, box_vectors):
        context.setPeriodicBoxVectors(*box_vectors_i * openmm.unit.angstrom)
        context.setPositions(coords_i * openmm.unit.angstrom)

        energy = context.getState(getEnergy=True).getPotentialEnergy()
        energies.append(energy.value_in_unit(openmm.unit.kilocalorie_per_mole))

    return torch.tensor(energies)


def _compute_du_d_theta_parameter(
    system: smee.ff.TensorSystem,
    potential_0: smee.ff.TensorPotential,
    energy_0: torch.Tensor,
    coords: numpy.ndarray,
    box_vectors: numpy.ndarray,
) -> torch.Tensor:
    """Computes the gradients of the potential energy with respect to a potential's
    parameters using the forward finite difference method.

    Args:
        system: The system being evaluated.
        potential_0: The potential to compute the gradients for.
        energy_0: The potential energy [kcal / mol] of the system with ``potential_0``.
        coords: The coordinates [Å] of the system.
        box_vectors: The box vectors [Å] of the system.

    Returns:
        The gradients of the potential energy with respect to the parameters with
        ``shape=(n_parameters, n_parameter_cols, n_frames)``.
    """
    du_d_theta = torch.zeros((*potential_0.parameters.shape, *energy_0.shape))

    # only compute gradients for parameters that are actually used
    parameter_idxs = {
        i.item()
        for topology in system.topologies
        for i in topology.parameters[potential_0.type]
        .assignment_matrix.indices()[1, :]
        .unique()
    }

    for i in parameter_idxs:
        for j in range(potential_0.parameters.shape[1]):
            potential_1 = copy.deepcopy(potential_0)
            potential_1.parameters[i, j] += GRADIENT_DELTA

            energy_1 = _compute_energy(system, potential_1, coords, box_vectors)
            du_d_theta[i, j] = (energy_1 - energy_0) / GRADIENT_DELTA

    return du_d_theta


def _compute_du_d_theta_attribute(
    system: smee.ff.TensorSystem,
    potential_0: smee.ff.TensorPotential,
    energy_0: torch.Tensor,
    coords: numpy.ndarray,
    box_vectors: numpy.ndarray,
) -> torch.Tensor:
    """Computes the gradients of the potential energy with respect to a potential's
    attributes using the forward finite difference method.

    Notes:
        Attributes listed in ``GRADIENT_EXCLUDED_ATTRIBUTES`` will not have their
        gradients computed and will always be zero.

    Args:
        system: The system being evaluated.
        potential_0: The potential to compute the gradients for.
        energy_0: The potential energy [kcal / mol] of the system with ``potential_0``.
        coords: The coordinates [Å] of the system.
        box_vectors: The box vectors [Å] of the system.

    Returns:
        The gradients of the potential energy with respect to the attributes with
        ``shape=(1, n_attribute_cols, n_frames)``.
    """
    du_d_theta = torch.zeros((1, len(potential_0.attributes), *energy_0.shape))

    attribute_idxs = [
        i
        for i, col in enumerate(potential_0.attribute_cols)
        if col not in GRADIENT_EXCLUDED_ATTRIBUTES
    ]

    for i in attribute_idxs:
        potential_1 = copy.deepcopy(potential_0)
        potential_1.attributes[i] += GRADIENT_DELTA

        energy_1 = _compute_energy(system, potential_1, coords, box_vectors)
        du_d_theta[0, i] = (energy_1 - energy_0) / GRADIENT_DELTA

    return du_d_theta


def _compute_du_d_theta(
    theta: tuple[torch.Tensor, ...], ctx: typing.Any
) -> list[torch.Tensor | None]:
    """Computes the gradients of the potential energy with respect to each parameter
    and attribute tensor in ``theta`` using the forward finite difference method.

    Args:
        theta: The set of 'packed' parameter and attribute tensors.
        ctx: The context object passed to the custom PyTorch op.

    Returns:
        The gradients of the potential energy with respect to each theta. Gradients
        w.r.t. parameters will have ``shape=(n_parameters, n_parameter_cols, n_frames)``
        while gradients w.r.t. attributes will have
        ``shape=(1, n_attribute_cols, n_frames)``.
    """
    system = ctx.kwargs["system"]

    force_field = _unpack_force_field(
        theta,
        ctx.kwargs["parameter_lookup"],
        ctx.kwargs["attribute_lookup"],
        ctx.kwargs["force_field"],
    )
    potential_types = {*ctx.kwargs["parameter_lookup"], *ctx.kwargs["attribute_lookup"]}

    du_d_theta = [None] * len(theta)

    for potential_type in potential_types:
        parameter_idx = ctx.kwargs["parameter_lookup"][potential_type]
        attribute_idx = ctx.kwargs["attribute_lookup"][potential_type]

        # need to shift by 1 as kwargs is actually the first input
        if not (
            ctx.needs_input_grad[parameter_idx + 1]
            or ctx.needs_input_grad[attribute_idx + 1]
        ):
            continue

        potential_0 = force_field.potentials_by_type[potential_type]
        energy_0 = _compute_energy(system, potential_0, ctx.coords, ctx.box_vectors)

        if ctx.needs_input_grad[parameter_idx + 1]:
            du_d_theta[parameter_idx] = _compute_du_d_theta_parameter(
                system, potential_0, energy_0, ctx.coords, ctx.box_vectors
            )

        if ctx.needs_input_grad[attribute_idx + 1]:
            du_d_theta[attribute_idx] = _compute_du_d_theta_attribute(
                system, potential_0, energy_0, ctx.coords, ctx.box_vectors
            )

    return du_d_theta


class _EnsembleAverageOp(torch.autograd.Function):
    """A custom PyTorch op for computing ensemble averages over MD trajectories."""

    @staticmethod
    def _create_reporter(kwargs: _EnsembleAverageKwargs):
        """Creates an OpenMM reporter that stores coords and values in memory"""

        system = kwargs["system"]

        sum_fn = functools.partial(sum, start=0.0 * openmm.unit.dalton)

        total_mass = sum_fn(
            sum_fn(
                openmm.app.Element.getByAtomicNumber(int(atomic_num)).mass
                for atomic_num in topology.atomic_nums
            )
            * n_copies
            for topology, n_copies in zip(system.topologies, system.n_copies)
        )

        return smee.mm._reporters.TensorReporter(
            kwargs["production_report_interval"],
            total_mass,
            kwargs["production_config"].pressure,
        )

    @staticmethod
    def forward(ctx, kwargs: _EnsembleAverageKwargs, *tensors: torch.Tensor):
        force_field = _unpack_force_field(
            tensors,
            kwargs["parameter_lookup"],
            kwargs["attribute_lookup"],
            kwargs["force_field"],
        )

        reporter = _EnsembleAverageOp._create_reporter(kwargs)

        smee.mm.simulate(
            kwargs["system"],
            force_field,
            kwargs["coords_config"],
            kwargs["equilibrate_configs"],
            kwargs["production_config"],
            [reporter, *kwargs["production_reporters"]],
        )

        outputs = torch.stack(reporter.values)
        avg_outputs = outputs.mean(dim=0)

        ctx.coords = numpy.ascontiguousarray(numpy.stack(reporter.coords))
        ctx.box_vectors = numpy.ascontiguousarray(numpy.stack(reporter.box_vectors))
        ctx.kwargs = kwargs
        ctx.save_for_backward(*tensors, outputs)

        return tuple(avg_outputs[i] for i in range(len(avg_outputs)))

    @staticmethod
    def backward(ctx, *grad_outputs):
        temperature = ctx.kwargs["production_config"].temperature
        beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)
        beta = beta.value_in_unit(openmm.unit.kilocalorie_per_mole**-1)

        outputs = ctx.saved_tensors[-1]
        avg_outputs = outputs.mean(dim=0)

        # we need to return one extra 'gradient' for kwargs.
        theta = ctx.saved_tensors[:-1]
        grads = [None] + [None] * len(theta)

        du_d_theta = _compute_du_d_theta(theta, ctx)

        for i in range(len(du_d_theta)):
            if du_d_theta[i] is None:
                continue

            avg_du_d_theta_i = du_d_theta[i].mean(dim=-1)

            avg_d_output_d_theta_i = torch.stack(
                [
                    avg_du_d_theta_i,  # potential energy
                    torch.zeros_like(avg_du_d_theta_i),  # volume
                    torch.zeros_like(avg_du_d_theta_i),  # density
                ]
                + ([avg_du_d_theta_i] if len(avg_outputs) == 4 else []),  # enthalpy
                dim=-1,
            )
            avg_output_du_d_theta_i = torch.mean(
                du_d_theta[i][:, :, None, :] * outputs.T[None, None, :, :], dim=-1
            )
            avg_du_d_theta_i_avg_output = (
                avg_du_d_theta_i[:, :, None] * avg_outputs[None, None, :]
            )
            d_avg_output_d_theta_i = avg_d_output_d_theta_i - beta * (
                avg_output_du_d_theta_i - avg_du_d_theta_i_avg_output
            )

            grads[i + 1] = d_avg_output_d_theta_i @ torch.stack(grad_outputs)

        return tuple(grads)


def compute_ensemble_averages(
    system: smee.ff.TensorSystem,
    force_field: smee.ff.TensorForceField,
    coords_config: "smee.mm.GenerateCoordsConfig",
    equilibrate_configs: list[
        typing.Union["smee.mm.MinimizationConfig", "smee.mm.SimulationConfig"]
    ],
    production_config: "smee.mm.SimulationConfig",
    production_report_interval: int,
    production_reporters: list[typing.Any] | None = None,
) -> EnsembleAverages:
    """Compute ensemble average of the potential energy, volume, density,
    and enthalpy (if running NPT) over an MD trajectory.

    Args:
        system: The system to simulate.
        force_field: The force field to use.
        coords_config: The configuration defining how to generate the system
            coordinates.
        equilibrate_configs: A list of configurations defining the steps to run for
            equilibration. No data will be stored from these simulations.
        production_config: The configuration defining the production simulation to run.
        production_report_interval: The interval at which to store data
            (coords, box vectors, energy, etc) from the production simulation
        production_reporters: A list of additional OpenMM reporters to use for the
            production simulation.

    Returns:
        A dictionary containing the ensemble averages of the potential energy [kcal/mol],
        volume [Å^3], density [g/mL], and enthalpy [kcal/mol] (if running NPT).
    """
    tensors, parameter_lookup, attribute_lookup = _pack_force_field(force_field)

    production_reporters = (
        production_reporters if production_reporters is not None else []
    )

    kwargs: _EnsembleAverageKwargs = {
        "force_field": force_field,
        "parameter_lookup": parameter_lookup,
        "attribute_lookup": attribute_lookup,
        "system": system,
        "coords_config": coords_config,
        "equilibrate_configs": equilibrate_configs,
        "production_config": production_config,
        "production_report_interval": production_report_interval,
        "production_reporters": production_reporters,
    }

    avg_outputs = _EnsembleAverageOp.apply(kwargs, *tensors)

    avg_outputs_dict: EnsembleAverages = {
        "potential_energy": avg_outputs[0],
        "volume": avg_outputs[1],
        "density": avg_outputs[2],
        "enthalpy": None,
    }

    if len(avg_outputs) == 4:
        avg_outputs_dict["enthalpy"] = avg_outputs[3]

    return avg_outputs_dict
