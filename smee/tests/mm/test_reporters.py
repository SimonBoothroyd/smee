import numpy
import openmm.unit
import pytest
import torch

from smee.mm._reporters import TensorReporter


class TestTensorReporter:
    def test_describe_next(self, mocker):
        simulation = mocker.MagicMock()
        simulation.currentStep = 5

        reporter = TensorReporter(
            report_interval=2, total_mass=1.0 * openmm.unit.daltons, pressure=None
        )

        assert reporter.describeNextReport(simulation) == (1, True, False, False, True)

    def test_report(sef, mocker):
        total_mass = 5.0 * openmm.unit.daltons
        pressure = 1.0 * openmm.unit.atmospheres

        reporter = TensorReporter(
            report_interval=1, total_mass=total_mass, pressure=pressure
        )

        expected_potential = 1.0 * openmm.unit.kilocalories_per_mole
        expected_kinetic = 2.0 * openmm.unit.kilocalories_per_mole
        expected_total = expected_potential + expected_kinetic

        expected_box_vectors = numpy.eye(3) * 3.0
        expected_coords = numpy.ones((1, 3))

        expected_volume = 27.0 * openmm.unit.angstrom**3
        expected_density = total_mass / expected_volume

        expected_enthalpy = (
            expected_total
            + pressure * expected_volume * openmm.unit.AVOGADRO_CONSTANT_NA
        )

        mock_state = mocker.MagicMock()
        mock_state.getPotentialEnergy.return_value = expected_potential
        mock_state.getKineticEnergy.return_value = expected_kinetic
        mock_state.getPeriodicBoxVectors.return_value = (
            expected_box_vectors * openmm.unit.angstrom
        )
        mock_state.getPositions.return_value = expected_coords * openmm.unit.angstrom

        reporter.report(None, mock_state)

        assert reporter.coords == [pytest.approx(expected_coords)]
        assert reporter.box_vectors == [pytest.approx(expected_box_vectors)]

        expected_values = torch.tensor(
            [
                expected_potential.value_in_unit(openmm.unit.kilocalories_per_mole),
                expected_volume.value_in_unit(openmm.unit.angstrom**3),
                expected_density.value_in_unit(
                    openmm.unit.gram / openmm.unit.item / openmm.unit.milliliter
                ),
                expected_enthalpy.value_in_unit(openmm.unit.kilocalories_per_mole),
            ]
        )
        assert reporter.values == [pytest.approx(expected_values)]
