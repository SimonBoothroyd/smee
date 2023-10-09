import numpy
import openff.units
import openmm.unit
import pydantic
import pytest

from smee.mm._config import OpenMMQuantity


@pytest.mark.parametrize(
    "input_value",
    [
        1.0 * openmm.unit.angstrom,
        0.1 * openmm.unit.nanometers,
        "1.0 angstrom",
        1.0 * openff.units.unit.angstrom,
    ],
)
def test_openmm_unit_type(input_value):
    class MockModel(pydantic.BaseModel):
        value: OpenMMQuantity[openmm.unit.angstrom]

    x = MockModel(value=input_value)

    assert isinstance(x.value, openmm.unit.Quantity)
    assert x.value.unit == openmm.unit.angstrom
    assert numpy.isclose(x.value.value_in_unit(openmm.unit.angstrom), 1.0)

    model_json = x.model_dump_json()
    assert model_json == '{"value":"1.00000000 angstrom"}'

    model_schema = x.model_json_schema()
    assert model_schema == {
        "properties": {"value": {"title": "Value", "type": "string"}},
        "required": ["value"],
        "title": "MockModel",
        "type": "object",
    }

    y = MockModel.model_validate_json(model_json)
    assert isinstance(y.value, openmm.unit.Quantity)
    assert y.value.unit == openmm.unit.angstrom
    assert numpy.isclose(y.value.value_in_unit(openmm.unit.angstrom), 1.0)


def test_openmm_unit_type_incompatible():
    class MockModel(pydantic.BaseModel):
        value: OpenMMQuantity[openmm.unit.angstrom]

    with pytest.raises(
        pydantic.ValidationError,
        match="invalid units kilocalorie/mole - expected angstrom",
    ):
        MockModel(value=1.0 * openmm.unit.kilocalories_per_mole)
