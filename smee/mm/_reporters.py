"""OpenMM simulation reporters"""
import math

import numpy
import openmm.app
import openmm.unit
import torch

_ANGSTROM = openmm.unit.angstrom
_KCAL_PER_MOL = openmm.unit.kilocalories_per_mole
_GRAM_PER_ML = openmm.unit.gram / openmm.unit.item / openmm.unit.milliliter

TENSOR_REPORTER_COLUMNS = ["potential_energy", "volume", "density", "enthalpy"]
TENSOR_REPORTER_UNITS = [_KCAL_PER_MOL, _ANGSTROM**3, _GRAM_PER_ML, _KCAL_PER_MOL]


class TensorReporter:
    """A reporter which stores coords, box vectors, and stats in memory."""

    @property
    def coords(self) -> list[numpy.ndarray]:
        """A list to store the coordinates of each reported frame in, where coordinates
        are stored as numpy arrays with ``shape=(n_atoms, 3)`` and units of angstroms.
        """
        return self._coords

    @property
    def box_vectors(self) -> list[numpy.ndarray]:
        """A list to store the box vectors of each reported frame in, where box vectors
        are stored as numpy arrays with ``shape=(3, 3)`` and units of angstroms.
        """
        return self._box_vectors

    @property
    def values(self) -> list[torch.Tensor]:
        """A list to store the tensor values of each reported frame in. See
        ``TENSOR_REPORTER_COLUMNS`` for the order of the values in each tensor.
        """
        return self._values

    def __init__(
        self,
        report_interval: int,
        total_mass: openmm.unit.Quantity,
        pressure: openmm.unit.Quantity | None,
    ):
        """

        Args:
            report_interval: The interval (in steps) at which to store the tensor
                values.
            total_mass: The total mass of the system.
            pressure: The pressure of the system. If none, no enthalpy will be reported.
        """
        self._report_interval = report_interval

        self._total_mass = total_mass
        self._pressure = pressure

        self._coords = []
        self._box_vectors = []
        self._values = []

    def describeNextReport(self, simulation: openmm.app.Simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        # requires - positions, velocities, forces, energies?
        return (steps, True, False, False, True)

    def report(self, simulation: openmm.app.Simulation, state: openmm.State):
        potential_energy = state.getPotentialEnergy()
        kinetic_energy = state.getKineticEnergy()

        total_energy = potential_energy + kinetic_energy

        if math.isnan(total_energy.value_in_unit(_KCAL_PER_MOL)):
            raise ValueError("total energy is nan")
        if math.isinf(total_energy.value_in_unit(_KCAL_PER_MOL)):
            raise ValueError("total energy is infinite")

        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]

        density = self._total_mass / volume

        values = [
            potential_energy.value_in_unit(_KCAL_PER_MOL),
            volume.value_in_unit(openmm.unit.angstrom**3),
            density.value_in_unit(_GRAM_PER_ML),
        ]

        if self._pressure is not None:
            pv_term = volume * self._pressure * openmm.unit.AVOGADRO_CONSTANT_NA

            enthalpy = total_energy + pv_term
            values.append(enthalpy.value_in_unit(_KCAL_PER_MOL))

        self._values.append(torch.tensor(values))

        coords = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
        coords = coords.astype(numpy.float32)
        box_vectors = box_vectors.value_in_unit(openmm.unit.angstrom)
        box_vectors = box_vectors.astype(numpy.float32)

        self._coords.append(coords)
        self._box_vectors.append(box_vectors)
