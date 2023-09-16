"""Convert SMIRNOFF valence parameters into tensors."""
import openff.interchange.smirnoff._base
import openff.interchange.smirnoff._valence
import openff.units
import torch

import smee.ff

_BondParameters = openff.interchange.smirnoff._base.SMIRNOFFCollection
_AngleParameters = openff.interchange.smirnoff._base.SMIRNOFFCollection
_ProperTorsionParameters = openff.interchange.smirnoff._base.SMIRNOFFCollection
_ImproperTorsionParameters = openff.interchange.smirnoff._base.SMIRNOFFCollection

_UNITLESS = openff.units.unit.dimensionless
_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians
_KCAL_PER_MOL = openff.units.unit.kilocalories / openff.units.unit.mole


def convert_valence_handlers(
    handlers: list[_BondParameters],
    handler_type: str,
    parameter_cols: tuple[str, ...],
) -> tuple[smee.ff.TensorPotential, list[smee.ff.ValenceParameterMap]]:
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
    potential = smee.ff._ff._handlers_to_potential(
        handlers, handler_type, parameter_cols, None
    )

    parameter_key_to_idx = {
        parameter_key: i for i, parameter_key in enumerate(potential.parameter_keys)
    }
    parameter_maps = []

    for handler in handlers:
        particle_idxs = [topology_key.atom_indices for topology_key in handler.key_map]

        assignment_matrix = torch.zeros((len(particle_idxs), len(potential.parameters)))

        for i, parameter_key in enumerate(handler.key_map.values()):
            assignment_matrix[i, parameter_key_to_idx[parameter_key]] += 1.0

        parameter_map = smee.ff.ValenceParameterMap(
            torch.tensor(particle_idxs), assignment_matrix.float().to_sparse()
        )
        parameter_maps.append(parameter_map)

    return potential, parameter_maps


@smee.ff.parameter_converter(
    "Bonds", {"k": _KCAL_PER_MOL / _ANGSTROM**2, "length": _ANGSTROM}
)
def convert_bonds(
    handlers: list[openff.interchange.smirnoff._valence.SMIRNOFFBondCollection],
) -> tuple[smee.ff.TensorPotential, list[smee.ff.ValenceParameterMap]]:
    return convert_valence_handlers(handlers, "Bonds", ("k", "length"))


@smee.ff.parameter_converter(
    "Angles", {"k": _KCAL_PER_MOL / _RADIANS**2, "angle": _RADIANS}
)
def convert_angles(
    handlers: list[_AngleParameters],
) -> tuple[smee.ff.TensorPotential, list[smee.ff.ValenceParameterMap]]:
    return convert_valence_handlers(handlers, "Angles", ("k", "angle"))


@smee.ff.parameter_converter(
    "ProperTorsions",
    {
        "k": _KCAL_PER_MOL,
        "periodicity": _UNITLESS,
        "phase": _RADIANS,
        "idivf": _UNITLESS,
    },
)
def convert_propers(
    handlers: list[_ProperTorsionParameters],
) -> tuple[smee.ff.TensorPotential, list[smee.ff.ValenceParameterMap]]:
    return convert_valence_handlers(
        handlers, "ProperTorsions", ("k", "periodicity", "phase", "idivf")
    )


@smee.ff.parameter_converter(
    "ImproperTorsions",
    {
        "k": _KCAL_PER_MOL,
        "periodicity": _UNITLESS,
        "phase": _RADIANS,
        "idivf": _UNITLESS,
    },
)
def convert_impropers(
    handlers: list[_ImproperTorsionParameters],
) -> tuple[smee.ff.TensorPotential, list[smee.ff.ValenceParameterMap]]:
    return convert_valence_handlers(
        handlers, "ImproperTorsions", ("k", "periodicity", "phase", "idivf")
    )
