"""Convert SMIRNOFF valence parameters into tensors."""
import openff.interchange.smirnoff
import openff.units
import torch

import smee

_UNITLESS = openff.units.unit.dimensionless
_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians
_KCAL_PER_MOL = openff.units.unit.kilocalories / openff.units.unit.mole


def convert_valence_handlers(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFCollection],
    handler_type: str,
    parameter_cols: tuple[str, ...],
) -> tuple[smee.TensorPotential, list[smee.ValenceParameterMap]]:
    """Convert a list of SMIRNOFF valence handlers into a tensor potential and
    associated parameter maps.

    Notes:
        This function assumes that all parameters come from the same force field

    Args:
        handlers: The list of SMIRNOFF valence handlers to convert.
        handler_type: The type of valence handler being converted.
        parameter_cols: The ordering of the parameter array columns.

    Returns:
        The potential containing tensors of the parameter values, and a list of
        parameter maps which map the parameters to the interactions they apply to.
    """
    potential = smee.converters.openff._openff._handlers_to_potential(
        handlers, handler_type, parameter_cols, None
    )

    parameter_key_to_idx = {
        parameter_key: i for i, parameter_key in enumerate(potential.parameter_keys)
    }
    parameter_maps = []

    for handler in handlers:
        particle_idxs = [topology_key.atom_indices for topology_key in handler.key_map]

        assignment_matrix = torch.zeros(
            (len(particle_idxs), len(potential.parameters)), dtype=torch.float64
        )

        for i, parameter_key in enumerate(handler.key_map.values()):
            assignment_matrix[i, parameter_key_to_idx[parameter_key]] += 1.0

        parameter_map = smee.ValenceParameterMap(
            torch.tensor(particle_idxs), assignment_matrix.to_sparse()
        )
        parameter_maps.append(parameter_map)

    return potential, parameter_maps


@smee.converters.smirnoff_parameter_converter(
    "Bonds", {"k": _KCAL_PER_MOL / _ANGSTROM**2, "length": _ANGSTROM}
)
def convert_bonds(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFBondCollection],
) -> tuple[smee.TensorPotential, list[smee.ValenceParameterMap]]:
    return convert_valence_handlers(handlers, "Bonds", ("k", "length"))


@smee.converters.smirnoff_parameter_converter(
    "Angles", {"k": _KCAL_PER_MOL / _RADIANS**2, "angle": _RADIANS}
)
def convert_angles(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFAngleCollection],
) -> tuple[smee.TensorPotential, list[smee.ValenceParameterMap]]:
    return convert_valence_handlers(handlers, "Angles", ("k", "angle"))


@smee.converters.smirnoff_parameter_converter(
    "ProperTorsions",
    {
        "k": _KCAL_PER_MOL,
        "periodicity": _UNITLESS,
        "phase": _RADIANS,
        "idivf": _UNITLESS,
    },
)
def convert_propers(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFProperTorsionCollection],
) -> tuple[smee.TensorPotential, list[smee.ValenceParameterMap]]:
    return convert_valence_handlers(
        handlers, "ProperTorsions", ("k", "periodicity", "phase", "idivf")
    )


@smee.converters.smirnoff_parameter_converter(
    "ImproperTorsions",
    {
        "k": _KCAL_PER_MOL,
        "periodicity": _UNITLESS,
        "phase": _RADIANS,
        "idivf": _UNITLESS,
    },
)
def convert_impropers(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFImproperTorsionCollection],
) -> tuple[smee.TensorPotential, list[smee.ValenceParameterMap]]:
    return convert_valence_handlers(
        handlers, "ImproperTorsions", ("k", "periodicity", "phase", "idivf")
    )
