"""OpenMM simulation reporters"""
import math
import typing

import msgpack
import numpy
import openmm.app
import openmm.unit
import torch

_ANGSTROM = openmm.unit.angstrom
_KCAL_PER_MOL = openmm.unit.kilocalories_per_mole


def _encoder(obj, chain=None):
    """msgpack encoder for tensors"""
    if isinstance(obj, torch.Tensor):
        assert obj.dtype == torch.float32
        return {b"torch": True, b"shape": obj.shape, b"data": obj.numpy().tobytes()}
    else:
        return obj if chain is None else chain(obj)


def _decoder(obj, chain=None):
    """msgpack decoder for tensors"""
    try:
        if b"torch" in obj:
            array = numpy.ndarray(
                buffer=obj[b"data"], dtype=numpy.float32, shape=obj[b"shape"]
            )
            return torch.from_numpy(array.copy())
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)


class TensorReporter:
    """A reporter which stores coords, box vectors, and kinetic energy using msgpack."""

    def __init__(self, output_file: typing.BinaryIO, report_interval: int):
        """

        Args:
            output_file: The file to write the frames to.
            report_interval: The interval (in steps) at which to write frames.
        """
        self._output_file = output_file
        self._report_interval = report_interval

    def describeNextReport(self, simulation: openmm.app.Simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        # requires - positions, velocities, forces, energies?
        return steps, True, False, False, True

    def report(self, simulation: openmm.app.Simulation, state: openmm.State):
        potential_energy = state.getPotentialEnergy()
        kinetic_energy = state.getKineticEnergy()

        total_energy = potential_energy + kinetic_energy

        if math.isnan(total_energy.value_in_unit(_KCAL_PER_MOL)):
            raise ValueError("total energy is nan")
        if math.isinf(total_energy.value_in_unit(_KCAL_PER_MOL)):
            raise ValueError("total energy is infinite")

        coords = state.getPositions(asNumpy=True).value_in_unit(_ANGSTROM)
        coords = torch.from_numpy(coords).float()
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(_ANGSTROM)
        box_vectors = torch.from_numpy(box_vectors).float()

        frame = (coords, box_vectors, kinetic_energy.value_in_unit(_KCAL_PER_MOL))
        self._output_file.write(msgpack.dumps(frame, default=_encoder))


def unpack_frames(
    file: typing.BinaryIO,
) -> typing.Generator[tuple[torch.Tensor, torch.Tensor, float], None, None]:
    """Unpack frames saved by a ``TensorReporter``."""

    unpacker = msgpack.Unpacker(file, object_hook=_decoder)

    for frame in unpacker:
        yield frame
