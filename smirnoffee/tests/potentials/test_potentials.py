import pytest

from smirnoffee.potentials import _POTENTIAL_ENERGY_FUNCTIONS, potential_energy_function


def test_potential_energy_function_decorator():

    potential_energy_function("DummyHandler", "x")(lambda x: None)
    assert ("DummyHandler", "x") in _POTENTIAL_ENERGY_FUNCTIONS

    with pytest.raises(KeyError, match="A potential energy function is already"):
        potential_energy_function("DummyHandler", "x")(lambda x: None)

    del _POTENTIAL_ENERGY_FUNCTIONS[("DummyHandler", "x")]
