"""Custom PyTorch ops for computing ensemble averages."""

import pathlib
import typing

import openmm.unit
import torch

import smee
import smee.converters
import smee.converters.openmm
import smee.utils

_DENSITY_CONVERSION = 1.0e24 / openmm.unit.AVOGADRO_CONSTANT_NA.value_in_unit(
    openmm.unit.mole**-1
)
"""Convert from g / mol / Å**3 to g / mL"""


class NotEnoughSamplesError(ValueError):
    """An error raised when an ensemble average is attempted with too few samples."""


class _EnsembleAverageKwargs(typing.TypedDict):
    """The keyword arguments passed to the custom PyTorch op for computing ensemble
    averages."""

    force_field: smee.TensorForceField
    parameter_lookup: dict[str, int]
    attribute_lookup: dict[str, int]
    has_v_sites: bool

    system: smee.TensorSystem

    frames_path: pathlib.Path

    beta: float
    pressure: float | None


class _ReweightAverageKwargs(_EnsembleAverageKwargs):
    """The keyword arguments passed to the custom PyTorch op for computing re-weighted
    ensemble averages."""

    min_samples: int


def _pack_force_field(
    force_field: smee.TensorForceField,
) -> tuple[tuple[torch.Tensor, ...], dict[str, int], dict[str, int], bool]:
    """Pack a SMEE force field into a tuple that can be consumed by a custom PyTorch op.

    Notes:
        Currently custom PyTorch functions don't try to find tensors in lists and dicts,
        so we have to flatten the force field into an args tuple.

    Returns:
        A tuple of tensors containing the force field parameters and attributes,
        lookup tables mapping potential types to their corresponding parameters and
        attributes in the tuple, and a flag indicating if v-sites are present.
    """
    potential_types = [potential.type for potential in force_field.potentials]

    parameters = [potential.parameters for potential in force_field.potentials]
    attributes = [potential.attributes for potential in force_field.potentials]

    has_v_sites = force_field.v_sites is not None
    v_sites = [force_field.v_sites.parameters] if has_v_sites else []

    tensors = tuple(parameters + attributes + v_sites)

    parameter_lookup = {
        potential_type: i for i, potential_type in enumerate(potential_types)
    }
    attribute_lookup = {
        potential_type: i + len(parameters)
        for i, potential_type in enumerate(potential_types)
    }

    return tensors, parameter_lookup, attribute_lookup, has_v_sites


def _unpack_force_field(
    tensors: tuple[torch.Tensor, ...],
    parameter_lookup: dict[str, int],
    attribute_lookup: dict[str, int],
    has_v_sites: bool,
    force_field: smee.TensorForceField,
) -> smee.TensorForceField:
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

        potential = smee.TensorPotential(
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

    v_sites = None

    if has_v_sites:
        v_sites = smee.TensorVSites(
            force_field.v_sites.keys, force_field.v_sites.weights, tensors[-1]
        )

    return smee.TensorForceField(potentials, v_sites)


def _compute_mass(system: smee.TensorSystem) -> float:
    """Compute the total mass of a system."""

    def _get_mass(v: int) -> float:
        return openmm.app.Element.getByAtomicNumber(int(v)).mass.value_in_unit(
            openmm.unit.daltons
        )

    return sum(
        sum(_get_mass(atomic_num) for atomic_num in topology.atomic_nums) * n_copies
        for topology, n_copies in zip(system.topologies, system.n_copies, strict=True)
    )


def _compute_frame_observables(
    system: smee.TensorSystem,
    box_vectors: torch.Tensor,
    potential_energy: float,
    kinetic_energy: float,
    beta: float,
    pressure: float | None,
) -> dict[str, float]:
    """Compute observables for a given frame in a trajectory.

    If the system is periodic, the potential energy, volume and density will be
    returned. If the pressure is not None, the enthalpy will be returned as well. If
    the system is not periodic, only the potential energy will be returned.

    Args:
        system: The system that was simulated.
        box_vectors: The box vectors [Å] of this frame.
        potential_energy: The potential energy [kcal / mol] of this frame.
        kinetic_energy: The kinetic energy [kcal / mol] of this frame.
        beta: The inverse temperature [mol / kcal].
        pressure: The pressure [kcal / mol / Å^3] if NPT.

    Returns:
        The observables for this frame.
    """

    values = {
        "potential_energy": potential_energy,
        "potential_energy^2": potential_energy**2,
    }
    reduced_potential = beta * potential_energy

    if not system.is_periodic:
        values["reduced_potential"] = reduced_potential
        return values

    volume = torch.det(box_vectors)
    values.update({"volume": volume, "volume^2": volume**2})

    total_mass = _compute_mass(system)
    values["density"] = total_mass / volume * _DENSITY_CONVERSION

    if pressure is not None:
        pv_term = volume * pressure

        values["enthalpy"] = potential_energy + kinetic_energy + pv_term
        values["enthalpy^2"] = values["enthalpy"] ** 2

        reduced_potential += beta * pv_term

        values["enthalpy_volume"] = values["enthalpy"] * values["volume"]

    values["reduced_potential"] = reduced_potential

    return values


def _compute_observables(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    frames_file: typing.BinaryIO,
    theta: tuple[torch.Tensor],
    beta: float,
    pressure: float | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor, list[torch.Tensor]]:
    """Computes the standard set of 'observables', and the gradient of the potential
    energy with respect to ``theta`` over a given trajectory.

    Notes:
        If the system is periodic, the energy, volume, and density will be returned
        as well as the enthaloy if it was an NPT simulation. Otherwise, only the
        potential energy will be returned.

    Args:
        system: The system that was simulated.
        force_field: The force field to evaluate energies with.
        frames_file: The file containing the trajectory.
        theta: The parameters to compute the gradient with respect to.
        beta: The inverse temperature [mol / kcal].
        pressure: The pressure [kcal / mol / Å^3] if NPT.

    Returns:
        The observables at each frame, the columns of the observable tensor, the
        reduced potential energy at each frame, and the gradients of the potential
        energy with respect to each tensor in theta with
        ``shape=(n_parameters, n_parameter_cols)``.
    """

    needs_grad = [i for i, v in enumerate(theta) if v is not None and v.requires_grad]
    du_d_theta = [None if i not in needs_grad else [] for i in range(len(theta))]

    reduced_potentials = []

    values = []
    columns = None

    for coords, box_vectors, _, kinetic in smee.mm._reporters.unpack_frames(
        frames_file
    ):
        coords = coords.to(theta[0].device)
        box_vectors = box_vectors.to(theta[0].device)

        with torch.enable_grad():
            potential = smee.compute_energy(system, force_field, coords, box_vectors)

        du_d_theta_subset = []

        if len(needs_grad) > 0:
            du_d_theta_subset = torch.autograd.grad(
                potential,
                [theta[i] for i in needs_grad],
                [smee.utils.ones_like(1, potential)],
                retain_graph=False,
                allow_unused=True,
            )

        for idx, i in enumerate(needs_grad):
            du_d_theta[i].append(du_d_theta_subset[idx].float())

        frame = _compute_frame_observables(
            system, box_vectors, potential.detach(), kinetic, beta, pressure
        )

        reduced_potentials.append(frame.pop("reduced_potential"))

        if columns is None:
            columns = [*frame]

        values.append(torch.tensor([frame[c] for c in columns]))

    values = torch.stack(values).to(theta[0].device)
    reduced_potentials = smee.utils.tensor_like(reduced_potentials, theta[0])

    return (
        values,
        columns,
        reduced_potentials,
        [v if v is None else torch.stack(v, dim=-1) for v in du_d_theta],
    )


class _EnsembleAverageOp(torch.autograd.Function):
    """A custom PyTorch op for computing ensemble averages over MD trajectories."""

    @staticmethod
    def forward(ctx, kwargs: _EnsembleAverageKwargs, *theta: torch.Tensor):
        force_field = _unpack_force_field(
            theta,
            kwargs["parameter_lookup"],
            kwargs["attribute_lookup"],
            kwargs["has_v_sites"],
            kwargs["force_field"],
        )
        system = kwargs["system"]

        with kwargs["frames_path"].open("rb") as file:
            values, columns, _, du_d_theta = _compute_observables(
                system, force_field, file, theta, kwargs["beta"], kwargs["pressure"]
            )

        avg_values = values.mean(dim=0)
        avg_stds = [*values.std(dim=0)]

        ctx.beta = kwargs["beta"]
        ctx.n_theta = len(theta)
        ctx.columns = columns
        ctx.save_for_backward(*theta, *du_d_theta, values, avg_values)

        ctx.mark_non_differentiable(*avg_stds)

        return *avg_values, *avg_stds, tuple(columns)

    @staticmethod
    def backward(ctx, *grad_outputs):
        theta = ctx.saved_tensors[: ctx.n_theta]
        du_d_theta = ctx.saved_tensors[ctx.n_theta : 2 * ctx.n_theta]

        # attributes are flat, so we need the extra dim
        du_d_theta = [
            None if v is None else (v if v.ndim == 3 else v.unsqueeze(0))
            for v in du_d_theta
        ]

        grad_outputs = torch.stack(grad_outputs[: (len(grad_outputs) - 1) // 2])

        values = ctx.saved_tensors[-2]
        avg_values = ctx.saved_tensors[-1]

        grads = [None] * len(theta)

        energy = values[:, ctx.columns.index("potential_energy")]
        volume = (
            None
            if "volume" not in ctx.columns
            else values[:, ctx.columns.index("volume")]
        )
        enthalpy = (
            None
            if "enthalpy" not in ctx.columns
            else values[:, ctx.columns.index("enthalpy")]
        )

        for i in range(len(du_d_theta)):
            if du_d_theta[i] is None:
                continue

            avg_du_d_theta_i = du_d_theta[i].mean(dim=-1)

            avg_d_output_d_theta_i = {
                "potential_energy": avg_du_d_theta_i,
                "potential_energy^2": (2 * energy * du_d_theta[i]).mean(dim=-1),
                "volume": torch.zeros_like(avg_du_d_theta_i),
                "volume^2": torch.zeros_like(avg_du_d_theta_i),
                "density": torch.zeros_like(avg_du_d_theta_i),
                "enthalpy": avg_du_d_theta_i,
                "enthalpy^2": (
                    None
                    if enthalpy is None
                    else (2 * enthalpy * du_d_theta[i]).mean(dim=-1)
                ),
                "enthalpy_volume": (
                    None if volume is None else (volume * du_d_theta[i]).mean(dim=-1)
                ),
            }

            avg_d_output_d_theta_i = torch.stack(
                [avg_d_output_d_theta_i[column] for column in ctx.columns], dim=-1
            )
            avg_output_du_d_theta_i = torch.mean(
                du_d_theta[i][:, :, None, :] * values.T[None, None, :, :], dim=-1
            )
            avg_du_d_theta_i_avg_output = (
                avg_du_d_theta_i[:, :, None] * avg_values[None, None, :]
            )
            d_avg_output_d_theta_i = avg_d_output_d_theta_i - ctx.beta * (
                avg_output_du_d_theta_i - avg_du_d_theta_i_avg_output
            )

            grads[i] = d_avg_output_d_theta_i @ grad_outputs

        grads = [
            None if v is None else (v if t.ndim == 2 else v.squeeze(0))
            for t, v in zip(theta, grads, strict=True)
        ]

        # we need to return one extra 'gradient' for kwargs.
        return tuple([None] + grads)


class _ReweightAverageOp(torch.autograd.Function):
    """A custom PyTorch op for computing ensemble averages over MD trajectories."""

    @staticmethod
    def forward(ctx, kwargs: _ReweightAverageKwargs, *theta: torch.Tensor):
        force_field = _unpack_force_field(
            theta,
            kwargs["parameter_lookup"],
            kwargs["attribute_lookup"],
            kwargs["has_v_sites"],
            kwargs["force_field"],
        )
        system = kwargs["system"]

        with kwargs["frames_path"].open("rb") as file:
            values, columns, reduced_pot, du_d_theta = _compute_observables(
                system, force_field, file, theta, kwargs["beta"], kwargs["pressure"]
            )

        with kwargs["frames_path"].open("rb") as file:
            reduced_pot_0 = smee.utils.tensor_like(
                [v for _, _, v, _ in smee.mm._reporters.unpack_frames(file)],
                reduced_pot,
            )

        delta = (reduced_pot_0 - reduced_pot).double()

        ln_weights = delta - torch.logsumexp(delta, dim=0)
        weights = torch.exp(ln_weights)

        n_effective = torch.exp(-torch.sum(weights * ln_weights, dim=0))

        if n_effective < kwargs["min_samples"]:
            raise NotEnoughSamplesError

        avg_values = (weights[:, None] * values).sum(dim=0)

        ctx.beta = kwargs["beta"]
        ctx.n_theta = len(theta)
        ctx.columns = columns
        ctx.save_for_backward(*theta, *du_d_theta, delta, weights, values)

        return *avg_values, columns

    @staticmethod
    def backward(ctx, *grad_outputs):
        theta = ctx.saved_tensors[: ctx.n_theta]

        du_d_theta = ctx.saved_tensors[ctx.n_theta : 2 * ctx.n_theta]
        d_reduced_d_theta = [None if du is None else ctx.beta * du for du in du_d_theta]

        values = ctx.saved_tensors[-1]
        weights = ctx.saved_tensors[-2]
        delta = ctx.saved_tensors[-3]

        grads = [None] * len(theta)

        energy = values[:, ctx.columns.index("potential_energy")]
        volume = (
            None
            if "volume" not in ctx.columns
            else values[:, ctx.columns.index("volume")]
        )
        enthalpy = (
            None
            if "enthalpy" not in ctx.columns
            else values[:, ctx.columns.index("enthalpy")]
        )

        for i in range(len(du_d_theta)):
            if du_d_theta[i] is None:
                continue

            log_exp_sum, log_exp_sign = smee.utils.logsumexp(
                delta[None, None, :], -1, b=d_reduced_d_theta[i], return_sign=True
            )
            avg_d_reduced_d_theta_i = log_exp_sign * torch.exp(
                log_exp_sum - torch.logsumexp(delta, 0)
            )

            d_ln_weight_d_theta_i = (
                -d_reduced_d_theta[i] + avg_d_reduced_d_theta_i[:, :, None]
            )
            d_weight_d_theta_i = weights[None, None, :] * d_ln_weight_d_theta_i

            d_output_d_theta_i = {
                "potential_energy": du_d_theta[i],
                "potential_energy^2": 2 * energy * du_d_theta[i],
                "volume": torch.zeros_like(du_d_theta[i]),
                "volume^2": torch.zeros_like(du_d_theta[i]),
                "density": torch.zeros_like(du_d_theta[i]),
                "enthalpy": du_d_theta[i],
                "enthalpy^2": (
                    None if enthalpy is None else 2 * enthalpy * du_d_theta[i]
                ),
                "enthalpy_volume": (None if volume is None else volume * du_d_theta[i]),
            }
            d_output_d_theta_i = torch.stack(
                [d_output_d_theta_i[column] for column in ctx.columns], dim=-1
            )

            grads[i] = (
                d_weight_d_theta_i[:, :, :, None] * values[None, None, :, :]
                + weights[None, None, :, None] * d_output_d_theta_i
            ).sum(-2) @ torch.stack(grad_outputs[:-1])

        # we need to return one extra 'gradient' for kwargs.
        return tuple([None] + grads + [None])


def compute_ensemble_averages(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    frames_path: pathlib.Path,
    temperature: openmm.unit.Quantity,
    pressure: openmm.unit.Quantity | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute ensemble average of the potential energy, volume, density,
    and enthalpy (if running NPT) over an MD trajectory.

    Args:
        system: The system to simulate.
        force_field: The force field to use.
        frames_path: The path to the trajectory to compute the average over.
        temperature: The temperature that the trajectory was simulated at.
        pressure: The pressure that the trajectory was simulated at.

    Returns:
        A dictionary containing the ensemble averages of the potential energy
        [kcal/mol], volume [Å^3], density [g/mL], and enthalpy [kcal/mol],
        and a dictionary containing their standard deviations.
    """
    tensors, parameter_lookup, attribute_lookup, has_v_sites = _pack_force_field(
        force_field
    )

    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)
    beta = beta.value_in_unit(openmm.unit.kilocalorie_per_mole**-1)

    if pressure is not None:
        pressure = (pressure * openmm.unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**3
        )

    kwargs: _EnsembleAverageKwargs = {
        "force_field": force_field,
        "parameter_lookup": parameter_lookup,
        "attribute_lookup": attribute_lookup,
        "has_v_sites": has_v_sites,
        "system": system,
        "frames_path": frames_path,
        "beta": beta,
        "pressure": pressure,
    }

    *avg_outputs, columns = _EnsembleAverageOp.apply(kwargs, *tensors)

    avg_values = avg_outputs[: len(avg_outputs) // 2]
    avg_std = avg_outputs[len(avg_outputs) // 2 :]

    return (
        {column: avg for avg, column in zip(avg_values, columns, strict=True)},
        {column: avg for avg, column in zip(avg_std, columns, strict=True)},
    )


def reweight_ensemble_averages(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    frames_path: pathlib.Path,
    temperature: openmm.unit.Quantity,
    pressure: openmm.unit.Quantity | None,
    min_samples: int = 50,
) -> dict[str, torch.Tensor]:
    """Compute the ensemble average of the potential energy, volume, density,
    and enthalpy (if running NPT) by re-weighting an existing MD trajectory.

    Args:
        system: The system that was simulated.
        force_field: The new force field to use.
        frames_path: The path to the trajectory to compute the average over.
        temperature: The temperature that the trajectory was simulated at.
        pressure: The pressure that the trajectory was simulated at.
        min_samples: The minimum number of samples required to compute the average.

    Raises:
        NotEnoughSamplesError: If the number of effective samples is less than
            ``min_samples``.

    Returns:
        A dictionary containing the ensemble averages of the potential energy
        [kcal/mol], volume [Å^3], density [g/mL], and enthalpy [kcal/mol].
    """
    tensors, parameter_lookup, attribute_lookup, has_v_sites = _pack_force_field(
        force_field
    )

    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)
    beta = beta.value_in_unit(openmm.unit.kilocalorie_per_mole**-1)

    if pressure is not None:
        pressure = (pressure * openmm.unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**3
        )

    kwargs: _ReweightAverageKwargs = {
        "force_field": force_field,
        "parameter_lookup": parameter_lookup,
        "attribute_lookup": attribute_lookup,
        "has_v_sites": has_v_sites,
        "system": system,
        "frames_path": frames_path,
        "beta": beta,
        "pressure": pressure,
        "min_samples": min_samples,
    }

    *avg_outputs, columns = _ReweightAverageOp.apply(kwargs, *tensors)
    return {column: avg for avg, column in zip(avg_outputs, columns, strict=True)}


class _ComputeDGSolv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwargs, *theta: torch.Tensor):
        from smee.mm._fe import compute_dg_and_grads

        force_field = _unpack_force_field(
            theta,
            kwargs["parameter_lookup"],
            kwargs["attribute_lookup"],
            kwargs["has_v_sites"],
            kwargs["force_field"],
        )

        needs_grad = [
            i for i, v in enumerate(theta) if v is not None and v.requires_grad
        ]
        theta_grad = tuple(theta[i] for i in needs_grad)

        dg_a, dg_d_theta_a = compute_dg_and_grads(
            kwargs["solute"],
            kwargs["solvent_a"],
            force_field,
            theta_grad,
            kwargs["fep_dir"] / "solvent-a",
        )
        dg_b, dg_d_theta_b = compute_dg_and_grads(
            kwargs["solute"],
            kwargs["solvent_b"],
            force_field,
            theta_grad,
            kwargs["fep_dir"] / "solvent-b",
        )

        dg = dg_a - dg_b
        dg_d_theta = [None] * len(theta)

        for grad_idx, orig_idx in enumerate(needs_grad):
            dg_d_theta[orig_idx] = dg_d_theta_b[grad_idx] - dg_d_theta_a[grad_idx]

        ctx.save_for_backward(*dg_d_theta)

        return dg

    @staticmethod
    def backward(ctx, *grad_outputs):
        dg_d_theta_0 = ctx.saved_tensors

        grads = [None if v is None else v * grad_outputs[0] for v in dg_d_theta_0]
        return tuple([None] + grads)


class _ReweightDGSolv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwargs, *theta: torch.Tensor):
        from smee.mm._fe import reweight_dg_and_grads

        force_field = _unpack_force_field(
            theta,
            kwargs["parameter_lookup"],
            kwargs["attribute_lookup"],
            kwargs["has_v_sites"],
            kwargs["force_field"],
        )

        dg_0 = kwargs["dg_0"]

        needs_grad = [
            i for i, v in enumerate(theta) if v is not None and v.requires_grad
        ]
        theta_grad = tuple(theta[i] for i in needs_grad)

        # new FF G - old FF G
        dg_a, dg_d_theta_a, n_effective_a = reweight_dg_and_grads(
            kwargs["solute"],
            kwargs["solvent_a"],
            force_field,
            theta_grad,
            kwargs["fep_dir"] / "solvent-a",
        )
        dg_b, dg_d_theta_b, n_effective_b = reweight_dg_and_grads(
            kwargs["solute"],
            kwargs["solvent_b"],
            force_field,
            theta_grad,
            kwargs["fep_dir"] / "solvent-b",
        )

        dg = -dg_a + dg_0 + dg_b
        dg_d_theta = [None] * len(theta)

        for grad_idx, orig_idx in enumerate(needs_grad):
            dg_d_theta[orig_idx] = dg_d_theta_b[grad_idx] - dg_d_theta_a[grad_idx]

        ctx.save_for_backward(*dg_d_theta)

        return dg, min(n_effective_a, n_effective_b)

    @staticmethod
    def backward(ctx, *grad_outputs):
        dg_d_theta_0 = ctx.saved_tensors

        grads = [None if v is None else v * grad_outputs[0] for v in dg_d_theta_0]
        return tuple([None] + grads)


def compute_dg_solv(
    solute: smee.TensorTopology,
    solvent_a: smee.TensorTopology | None,
    solvent_b: smee.TensorTopology | None,
    force_field: smee.TensorForceField,
    fep_dir: pathlib.Path,
) -> torch.Tensor:
    """Computes ∆G_solv from existing FEP data.

    Notes:
        It is assumed that FEP data was generated using the same force field as
        ``force_field``, and using ``generate_dg_solv_data``. No attempt is made to
        validate this assumption, so proceed with extreme caution.

    Args:
        solute: The solute topology.
        solvent_a: The topology of the solvent in phase A.
        solvent_b: The topology of the solvent in phase B.
        force_field: The force field used to generate the FEP data.
        fep_dir: The directory containing the FEP data.

    Returns:
        ∆G_solv [kcal/mol].
    """

    tensors, parameter_lookup, attribute_lookup, has_v_sites = _pack_force_field(
        force_field
    )

    kwargs = {
        "solute": solute,
        "solvent_a": solvent_a,
        "solvent_b": solvent_b,
        "force_field": force_field,
        "parameter_lookup": parameter_lookup,
        "attribute_lookup": attribute_lookup,
        "has_v_sites": has_v_sites,
        "fep_dir": fep_dir,
    }
    return _ComputeDGSolv.apply(kwargs, *tensors)


def reweight_dg_solv(
    solute: smee.TensorTopology,
    solvent_a: smee.TensorTopology | None,
    solvent_b: smee.TensorTopology | None,
    force_field: smee.TensorForceField,
    fep_dir: pathlib.Path,
    dg_0: torch.Tensor,
    min_samples: int = 50,
) -> tuple[torch.Tensor, float]:
    """Computes ∆G_solv by re-weighting existing FEP data.

    Notes:
        It is assumed that FEP data was generated using ``generate_dg_solv_data``.

    Args:
        solute: The solute topology.
        solvent_a: The topology of the solvent in phase A.
        solvent_b: The topology of the solvent in phase B.
        force_field: The force field to reweight to.
        fep_dir: The directory containing the FEP data.
        dg_0: ∆G_solv [kcal/mol] computed with the force field used to generate the
            FEP data.
        min_samples: The minimum number of effective samples required to re-weight.

    Raises:
        NotEnoughSamplesError: If the number of effective samples is less than
            ``min_samples``.

    Returns:
        The re-weighted ∆G_solv [kcal/mol], and the minimum number of effective samples
        between the two phases.
    """
    tensors, parameter_lookup, attribute_lookup, has_v_sites = _pack_force_field(
        force_field
    )

    kwargs = {
        "solute": solute,
        "solvent_a": solvent_a,
        "solvent_b": solvent_b,
        "force_field": force_field,
        "parameter_lookup": parameter_lookup,
        "attribute_lookup": attribute_lookup,
        "has_v_sites": has_v_sites,
        "fep_dir": fep_dir,
        "dg_0": dg_0,
    }

    dg, n_eff = _ReweightDGSolv.apply(kwargs, *tensors)

    if n_eff < min_samples:
        raise NotEnoughSamplesError

    return dg, n_eff
