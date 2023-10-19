import numpy
import openmm.unit
import pytest

from smee.mm._reporters import TensorReporter, unpack_frames


class TestTensorReporter:
    def test_describe_next(self, mocker):
        simulation = mocker.MagicMock()
        simulation.currentStep = 5

        reporter = TensorReporter(mocker.MagicMock(), 2)
        assert reporter.describeNextReport(simulation) == (1, True, False, False, True)

    def test_report(self, tmp_path, mocker):
        expected_potential = 1.0 * openmm.unit.kilocalories_per_mole
        expected_kinetic = 2.0 * openmm.unit.kilojoules_per_mole

        expected_box_vectors = numpy.eye(3) * 3.0
        expected_coords = numpy.ones((1, 3))

        mock_state = mocker.MagicMock()
        mock_state.getPotentialEnergy.return_value = expected_potential
        mock_state.getKineticEnergy.return_value = expected_kinetic
        mock_state.getPeriodicBoxVectors.return_value = (
            expected_box_vectors * openmm.unit.angstrom
        )
        mock_state.getPositions.return_value = expected_coords * openmm.unit.angstrom

        expected_output_path = tmp_path / "output.msgpack"

        with expected_output_path.open("wb") as file:
            reporter = TensorReporter(file, 1)
            reporter.report(None, mock_state)

        with expected_output_path.open("rb") as file:
            frames = [*unpack_frames(file)]

        assert len(frames) == 1
        coords, box_vectors, kinetic = frames[0]

        assert coords == pytest.approx(expected_coords)
        assert box_vectors == pytest.approx(expected_box_vectors)
        assert kinetic == pytest.approx(
            expected_kinetic.value_in_unit(openmm.unit.kilocalories_per_mole)
        )

    @pytest.mark.parametrize(
        "potential, contains", [(numpy.nan, "nan"), (numpy.inf, "inf")]
    )
    def test_report_energy_check(self, potential, contains, mocker):
        potential = potential * openmm.unit.kilocalories_per_mole
        kinetic = 2.0 * openmm.unit.kilocalories_per_mole

        mock_state = mocker.MagicMock()
        mock_state.getPotentialEnergy.return_value = potential
        mock_state.getKineticEnergy.return_value = kinetic

        with pytest.raises(ValueError, match=f"total energy is {contains}"):
            reporter = TensorReporter(mocker.MagicMock(), 1)
            reporter.report(None, mock_state)
