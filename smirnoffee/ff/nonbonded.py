"""Convert SMIRNOFF non-bonded parameters into tensors."""
import collections
import copy

import openff.interchange.components.potentials
import openff.interchange.models
import openff.interchange.smirnoff._base
import openff.interchange.smirnoff._nonbonded
import openff.toolkit
import openff.units
import torch

import smirnoffee.ff
import smirnoffee.utils

_VDWParameters = openff.interchange.smirnoff._nonbonded.SMIRNOFFvdWCollection
_ElectrostaticParameters = (
    openff.interchange.smirnoff._nonbonded.SMIRNOFFElectrostaticsCollection
)

_UNITLESS = openff.units.unit.dimensionless
_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians
_KCAL_PER_MOL = openff.units.unit.kilocalories / openff.units.unit.mole
_ELEMENTARY_CHARGE = openff.units.unit.elementary_charge


def convert_nonbonded_handlers(
    handlers: list[openff.interchange.smirnoff._base.SMIRNOFFCollection],
    handler_type: str,
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smirnoffee.ff.VSiteMap | None],
    parameter_cols: tuple[str, ...],
) -> tuple[smirnoffee.ff.TensorPotential, list[smirnoffee.ff.NonbondedParameterMap]]:
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

    Returns:
        The potential containing tensors of the parameter values, and a list of
        parameter maps which map the parameters to the interactions they apply to.
    """
    assert len(topologies) == len(handlers), "topologies and handlers must match"
    assert len(v_site_maps) == len(handlers), "v-site maps and handlers must match"

    potential = smirnoffee.ff._ff._handlers_to_potential(
        handlers, handler_type, parameter_cols, ("scale_13", "scale_14", "scale_15")
    )

    potential.attribute_cols = ("scale_12", *potential.attribute_cols)
    potential.attributes = torch.cat([torch.tensor([0.0]), potential.attributes])

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

                    assignment_map[atom_idx][parameter_key_to_idx[mult_key]] += -1.0
                    assignment_map[v_site_idx][parameter_key_to_idx[mult_key]] += 1.0

        assignment_matrix = torch.zeros((n_particles, len(potential.parameters)))

        for particle_idx in assignment_map:
            for parameter_idx, count in assignment_map[particle_idx].items():
                assignment_matrix[particle_idx, parameter_idx] = count

        exclusion_to_scale = smirnoffee.utils.find_exclusions(topology, v_site_map)
        exclusions = torch.tensor([*exclusion_to_scale])
        exclusion_scale_idxs = torch.tensor(
            [[attribute_to_idx[scale]] for scale in exclusion_to_scale.values()]
        )

        parameter_map = smirnoffee.ff.NonbondedParameterMap(
            assignment_matrix=assignment_matrix.float().to_sparse(),
            exclusions=exclusions,
            exclusion_scale_idxs=exclusion_scale_idxs,
        )
        parameter_maps.append(parameter_map)

    return potential, parameter_maps


@smirnoffee.ff.parameter_converter(
    "vdW",
    {
        "epsilon": _KCAL_PER_MOL,
        "sigma": _ANGSTROM,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
    },
)
def convert_vdw(
    handlers: list[_VDWParameters],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smirnoffee.ff.VSiteMap | None],
) -> tuple[smirnoffee.ff.TensorPotential, list[smirnoffee.ff.NonbondedParameterMap]]:
    mixing_rules = {handler.mixing_rule for handler in handlers}
    assert len(mixing_rules) == 1, "multiple mixing rules found"
    mixing_rule = next(iter(mixing_rules))

    if mixing_rule != "lorentz-berthelot":
        raise NotImplementedError("only Lorentz-Berthelot mixing rules are supported.")

    return convert_nonbonded_handlers(
        handlers, "vdW", topologies, v_site_maps, ("epsilon", "sigma")
    )


def _make_v_site_electrostatics_compatible(handlers: list[_ElectrostaticParameters]):
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


@smirnoffee.ff.parameter_converter(
    "Electrostatics",
    {
        "charge": _ELEMENTARY_CHARGE,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
    },
)
def convert_electrostatics(
    handlers: list[_ElectrostaticParameters],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smirnoffee.ff.VSiteMap | None],
) -> tuple[smirnoffee.ff.TensorPotential, list[smirnoffee.ff.NonbondedParameterMap]]:
    handlers = [*handlers]
    _make_v_site_electrostatics_compatible(handlers)

    return convert_nonbonded_handlers(
        handlers, "Electrostatics", topologies, v_site_maps, ("charge",)
    )
