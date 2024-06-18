"""Convert SMIRNOFF non-bonded parameters into tensors."""

import collections
import copy
import typing

import openff.interchange.components.potentials
import openff.interchange.models
import openff.toolkit
import openff.units
import torch

import smee
import smee.utils

if typing.TYPE_CHECKING:
    import smirnoff_plugins.collections.nonbonded

_UNITLESS = openff.units.unit.dimensionless
_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians
_KCAL_PER_MOL = openff.units.unit.kilocalories / openff.units.unit.mole
_ELEMENTARY_CHARGE = openff.units.unit.elementary_charge


def convert_nonbonded_handlers(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFCollection],
    handler_type: str,
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
    parameter_cols: tuple[str, ...],
    attribute_cols: tuple[str, ...] | None = None,
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    """Convert a list of SMIRNOFF non-bonded handlers into a tensor potential and
    associated parameter maps.

    Notes:
        This function assumes that all parameters come from the same force field

    Args:
        handlers: The list of SMIRNOFF non-bonded handlers to convert.
        handler_type: The type of non-bonded handler being converted.
        topologies: The topologies associated with each handler.
        v_site_maps: The virtual site maps associated with each handler.
        parameter_cols: The ordering of the parameter array columns.
        attribute_cols: The handler attributes to include in the potential *in addition*
            to the intra-molecular scaling factors.

    Returns:
        The potential containing tensors of the parameter values, and a list of
        parameter maps which map the parameters to the interactions they apply to.
    """
    attribute_cols = attribute_cols if attribute_cols is not None else []

    assert len(topologies) == len(handlers), "topologies and handlers must match"
    assert len(v_site_maps) == len(handlers), "v-site maps and handlers must match"

    potential = smee.converters.openff._openff._handlers_to_potential(
        handlers,
        handler_type,
        parameter_cols,
        ("scale_12", "scale_13", "scale_14", "scale_15", *attribute_cols),
    )

    parameter_key_to_idx = {
        parameter_key: i for i, parameter_key in enumerate(potential.parameter_keys)
    }
    attribute_to_idx = {column: i for i, column in enumerate(potential.attribute_cols)}

    parameter_maps = []

    for handler, topology, v_site_map in zip(handlers, topologies, v_site_maps):
        assignment_map = collections.defaultdict(lambda: collections.defaultdict(float))

        n_particles = topology.n_atoms + (
            0 if v_site_map is None else len(v_site_map.keys)
        )

        for topology_key, parameter_key in handler.key_map.items():
            if isinstance(topology_key, openff.interchange.models.VirtualSiteKey):
                continue

            atom_idx = topology_key.atom_indices[0]
            assignment_map[atom_idx][parameter_key_to_idx[parameter_key]] += 1.0

        for topology_key, parameter_key in handler.key_map.items():
            if not isinstance(topology_key, openff.interchange.models.VirtualSiteKey):
                continue

            v_site_idx = v_site_map.key_to_idx[topology_key]

            if parameter_key.associated_handler != "Electrostatics":
                assignment_map[v_site_idx][parameter_key_to_idx[parameter_key]] += 1.0
            else:
                for i, atom_idx in enumerate(topology_key.orientation_atom_indices):
                    mult_key = copy.deepcopy(parameter_key)
                    mult_key.mult = i

                    assignment_map[atom_idx][parameter_key_to_idx[mult_key]] += 1.0
                    assignment_map[v_site_idx][parameter_key_to_idx[mult_key]] += -1.0

        assignment_matrix = torch.zeros(
            (n_particles, len(potential.parameters)), dtype=torch.float64
        )

        for particle_idx in assignment_map:
            for parameter_idx, count in assignment_map[particle_idx].items():
                assignment_matrix[particle_idx, parameter_idx] = count

        exclusion_to_scale = smee.utils.find_exclusions(topology, v_site_map)
        exclusions = torch.tensor([*exclusion_to_scale])
        exclusion_scale_idxs = torch.tensor(
            [[attribute_to_idx[scale]] for scale in exclusion_to_scale.values()],
            dtype=torch.int64,
        )

        parameter_map = smee.NonbondedParameterMap(
            assignment_matrix=assignment_matrix.to_sparse(),
            exclusions=exclusions,
            exclusion_scale_idxs=exclusion_scale_idxs,
        )
        parameter_maps.append(parameter_map)

    return potential, parameter_maps


@smee.converters.smirnoff_parameter_converter(
    "vdW",
    {
        "epsilon": _KCAL_PER_MOL,
        "sigma": _ANGSTROM,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
        "cutoff": _ANGSTROM,
        "switch_width": _ANGSTROM,
    },
)
def convert_vdw(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFvdWCollection],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    mixing_rules = {handler.mixing_rule for handler in handlers}
    assert len(mixing_rules) == 1, "multiple mixing rules found"
    mixing_rule = next(iter(mixing_rules))

    if mixing_rule != "lorentz-berthelot":
        raise NotImplementedError("only Lorentz-Berthelot mixing rules are supported.")

    return convert_nonbonded_handlers(
        handlers,
        smee.PotentialType.VDW,
        topologies,
        v_site_maps,
        ("epsilon", "sigma"),
        ("cutoff", "switch_width"),
    )


@smee.converters.smirnoff_parameter_converter(
    "DoubleExponential",
    {
        "epsilon": _KCAL_PER_MOL,
        "r_min": _ANGSTROM,
        "alpha": _UNITLESS,
        "beta": _UNITLESS,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
        "cutoff": _ANGSTROM,
        "switch_width": _ANGSTROM,
    },
)
def convert_dexp(
    handlers: list[
        "smirnoff_plugins.collections.nonbonded.SMIRNOFFDoubleExponentialCollection"
    ],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    import smee.potentials.nonbonded

    (
        potential,
        parameter_maps,
    ) = smee.converters.openff.nonbonded.convert_nonbonded_handlers(
        handlers,
        "DoubleExponential",
        topologies,
        v_site_maps,
        ("epsilon", "r_min"),
        ("cutoff", "switch_width", "alpha", "beta"),
    )
    potential.type = smee.PotentialType.VDW
    potential.fn = smee.EnergyFn.VDW_DEXP

    return potential, parameter_maps


@smee.converters.smirnoff_parameter_converter(
    "DampedExp6810",
    {
        "rho": _ANGSTROM,
        "beta": _ANGSTROM**-1,
        "c6": _KCAL_PER_MOL * _ANGSTROM**6,
        "c8": _KCAL_PER_MOL * _ANGSTROM**8,
        "c10": _KCAL_PER_MOL * _ANGSTROM**10,
        "force_at_zero": _KCAL_PER_MOL * _ANGSTROM**-1,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
        "cutoff": _ANGSTROM,
        "switch_width": _ANGSTROM,
    },
)
def convert_dampedexp6810(
    handlers: list[
        "smirnoff_plugins.collections.nonbonded.SMIRNOFFDampedExp6810Collection"
    ],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    import smee.potentials.nonbonded

    (
        potential,
        parameter_maps,
    ) = smee.converters.openff.nonbonded.convert_nonbonded_handlers(
        handlers,
        "DampedExp6810",
        topologies,
        v_site_maps,
        ("rho", "beta", "c6", "c8", "c10"),
        ("cutoff", "switch_width", "force_at_zero"),
    )
    potential.type = smee.PotentialType.VDW
    potential.fn = smee.EnergyFn.VDW_DAMPEDEXP6810

    return potential, parameter_maps


def _make_v_site_electrostatics_compatible(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFElectrostaticsCollection],
):
    """Attempts to make electrostatic potentials associated with virtual sites more
    consistent with other parameters so that they can be more easily converted to
    tensors.

    Args:
        handlers: The list of SMIRNOFF electrostatic handlers to make compatible.
    """
    for handler_idx, handler in enumerate(handlers):
        if not any(
            key.associated_handler == "Electrostatics" for key in handler.potentials
        ):
            continue

        handler = copy.deepcopy(handler)
        potentials = {}

        for key, potential in handler.potentials.items():
            # for some reason interchange lists this as electrostatics and not v-sites
            if key.associated_handler != "Electrostatics":
                potentials[key] = potential
                continue

            assert key.mult is None

            for i in range(len(potential.parameters["charge_increments"])):
                mult_key = copy.deepcopy(key)
                mult_key.mult = i

                mult_potential = copy.deepcopy(potential)
                mult_potential.parameters = {
                    "charge": potential.parameters["charge_increments"][i]
                }
                assert mult_key not in potentials
                potentials[mult_key] = mult_potential

        handler.potentials = potentials
        handlers[handler_idx] = handler


@smee.converters.smirnoff_parameter_converter(
    "Electrostatics",
    {
        "charge": _ELEMENTARY_CHARGE,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
        "cutoff": _ANGSTROM,
    },
)
def convert_electrostatics(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFElectrostaticsCollection],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    handlers = [*handlers]
    _make_v_site_electrostatics_compatible(handlers)

    return convert_nonbonded_handlers(
        handlers, "Electrostatics", topologies, v_site_maps, ("charge",), ("cutoff",)
    )
