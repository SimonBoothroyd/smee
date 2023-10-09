"""Configuration from MM simulations."""
import functools
import typing

import openff.units
import openmm.unit
import pydantic
import pydantic_core

KCAL_PER_MOL = openmm.unit.kilocalories_per_mole
ANGSTROM = openmm.unit.angstrom
GRAMS_PER_ML = openmm.unit.grams / openmm.unit.milliliters


def _quantity_validator(
    value: str | openmm.unit.Quantity | openff.units.unit.Quantity,
    expected_units: openmm.unit.Unit,
) -> openmm.unit.Quantity:
    if isinstance(value, str):
        value = openff.units.Quantity(value)
    if isinstance(value, openff.units.Quantity):
        value = openff.units.openmm.to_openmm(value)

    assert isinstance(value, openmm.unit.Quantity), f"invalid type - {type(value)}"

    try:
        return value.in_units_of(expected_units)
    except TypeError:
        raise ValueError(f"invalid units {value.unit} - expected {expected_units}")


def _quantity_serializer(value: openmm.unit.Quantity) -> str:
    unit_str = openff.units.openmm.openmm_unit_to_string(value.unit)
    return f"{value.value_in_unit(value.unit):.8f} {unit_str}"


class _OpenMMQuantityAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: typing.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        from_value_schema = pydantic_core.core_schema.no_info_plain_validator_function(
            lambda x: x
        )

        return pydantic_core.core_schema.json_or_python_schema(
            json_schema=from_value_schema,
            python_schema=from_value_schema,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                _quantity_serializer
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: pydantic_core.core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(pydantic_core.core_schema.str_schema())


class _OpenMMQuantityMeta(type):
    def __getitem__(cls, item: openmm.unit.Unit):
        validator = functools.partial(_quantity_validator, expected_units=item)
        return typing.Annotated[
            openmm.unit.Quantity,
            _OpenMMQuantityAnnotation,
            pydantic.BeforeValidator(validator),
        ]


class OpenMMQuantity(openmm.unit.Quantity, metaclass=_OpenMMQuantityMeta):
    """A pydantic safe OpenMM quantity type validates unit compatibility."""


class GenerateCoordsConfig(pydantic.BaseModel):
    """Configure how coordinates should be generated for a system using PACKMOL."""

    target_density: OpenMMQuantity[GRAMS_PER_ML] = pydantic.Field(
        0.95 * GRAMS_PER_ML,
        description="Target mass density for final system with units compatible with "
        "g / mL.",
    )

    scale_factor: float = pydantic.Field(
        1.1,
        description="The amount to scale the approximate box size by to help alleviate "
        "issues with packing larger molecules.",
    )
    padding: OpenMMQuantity[openmm.unit.angstrom] = pydantic.Field(
        2.0 * openmm.unit.angstrom,
        description="The amount of padding to add to the final box size to help "
        "alleviate PBC issues.",
    )

    tolerance: OpenMMQuantity[openmm.unit.angstrom] = pydantic.Field(
        2.0 * openmm.unit.angstrom,
        description="The minimum spacing between molecules during packing.",
    )

    seed: int | None = pydantic.Field(
        None, description="The random seed to use when generating the coordinates."
    )


class MinimizationConfig(pydantic.BaseModel):
    """Configure how a system should be energy minimized."""

    tolerance: OpenMMQuantity[KCAL_PER_MOL / ANGSTROM] = pydantic.Field(
        10.0 * KCAL_PER_MOL / ANGSTROM,
        description="Minimization will be halted once the root-mean-square value of "
        "all force components reaches this tolerance.",
    )
    max_iterations: int = pydantic.Field(
        0,
        description="The maximum number of iterations to perform. If 0, minimization "
        "will continue until the tolerance is met.",
    )


class SimulationConfig(pydantic.BaseModel):
    temperature: OpenMMQuantity[openmm.unit.kelvin] = pydantic.Field(
        ...,
        description="The temperature to simulate at.",
    )
    pressure: OpenMMQuantity[openmm.unit.atmospheres] | None = pydantic.Field(
        ...,
        description="The pressure to simulate at, or none to run in NVT.",
    )

    n_steps: int = pydantic.Field(
        ..., description="The number of steps to simulate for."
    )

    timestep: OpenMMQuantity[openmm.unit.femtoseconds] = pydantic.Field(
        2.0 * openmm.unit.femtoseconds,
        description="The timestep to use during the simulation.",
    )
    friction_coeff: OpenMMQuantity[1.0 / openmm.unit.picoseconds] = pydantic.Field(
        1.0 / openmm.unit.picoseconds,
        description="The integrator friction coefficient.",
    )
