import math

import numpy
import openmm.app
import openmm.unit
import torch

_KCAL_PER_MOL = openmm.unit.kilocalories_per_mole
_GRAM_PER_ML = openmm.unit.gram / openmm.unit.item / openmm.unit.milliliter

TENSOR_REPORTER_COLUMNS = ["potential_energy", "volume", "density", "enthalpy"]


class TensorReporter:
    def __init__(
        self,
        report_interval: int,
        coords: list[numpy.ndarray],
        box_vectors: list[numpy.ndarray],
        values: list[torch.Tensor],
        total_mass: openmm.unit.Quantity | None,
        pressure: openmm.unit.Quantity | None,
    ):
        self._report_interval = report_interval

        self._total_mass = total_mass
        self._pressure = pressure

        self._coords = coords
        self._box_vectors = box_vectors
        self._values = values

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
