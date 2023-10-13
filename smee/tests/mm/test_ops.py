import numpy
import openff.interchange.models
import openff.units
import openmm.unit
import torch

import smee
from smee.mm._ops import (
    _compute_du_d_theta,
    _compute_du_d_theta_attribute,
    _pack_force_field,
    _unpack_force_field,
    compute_ensemble_averages,
)


def _mock_potential(type_, parameters, attributes) -> smee.TensorPotential:
    return smee.TensorPotential(
        type_,
        f"{type_}-fn",
        parameters,
        [
            openff.interchange.models.PotentialKey(id=f"[#{i}:1]")
            for i in range(len(parameters))
        ],
        tuple(f"param{i}" for i in range(parameters.shape[1])),
        tuple(openff.units.unit.angstrom for _ in range(parameters.shape[1])),
        attributes,
        tuple(f"attr-{i}" for i in range(len(attributes))),
        tuple(openff.units.unit.angstrom for _ in range(len(attributes))),
    )


def test_pack_unpack_force_field():
    parameters_a = torch.randn(4).reshape(2, 2)
    attributes_a = torch.randn(2)

    parameters_b = torch.randn(6).reshape(2, 3)
    attributes_b = torch.randn(3)

    force_field = smee.TensorForceField(
        [
            _mock_potential("vdw", parameters_a, attributes_a),
            _mock_potential("bond", parameters_b, attributes_b),
        ],
        None,
    )

    tensors, param_lookup, attr_lookup = _pack_force_field(force_field)

    expected_tensors = (parameters_a, parameters_b, attributes_a, attributes_b)
    assert tensors == expected_tensors

    expected_param_lookup = {"vdw": 0, "bond": 1}
    assert param_lookup == expected_param_lookup
    expected_attr_lookup = {"vdw": 2, "bond": 3}
    assert attr_lookup == expected_attr_lookup

    updated_tensors = tuple(v + 1.0 for v in tensors)

    unpacked_force_field = _unpack_force_field(
        updated_tensors, param_lookup, attr_lookup, force_field
    )

    assert len(unpacked_force_field.potentials) == 2

    expected_params = updated_tensors[:2]
    expected_attrs = updated_tensors[2:]

    for i, (original, unpacked) in enumerate(
        zip(force_field.potentials, unpacked_force_field.potentials)
    ):
        assert original.type == unpacked.type
        assert original.fn == unpacked.fn

        assert original.parameters.shape == unpacked.parameters.shape
        assert torch.allclose(original.parameters, tensors[i])
        assert torch.allclose(unpacked.parameters, expected_params[i])

        assert original.parameter_keys == unpacked.parameter_keys
        assert original.parameter_cols == unpacked.parameter_cols
        assert original.parameter_units == unpacked.parameter_units

        assert original.attributes.shape == unpacked.attributes.shape
        assert torch.allclose(original.attributes, tensors[i + 2])
        assert torch.allclose(unpacked.attributes, expected_attrs[i])

        assert original.attribute_cols == unpacked.attribute_cols
        assert original.attribute_units == unpacked.attribute_units


def test_compute_du_d_theta_parameter(mocker, mock_argon_tensors, mock_argon_params):
    eps, sig = mock_argon_params

    eps = eps.m_as("kilocalorie / mole")
    sig = sig.m_as("angstrom")

    tensor_ff, tensor_top = mock_argon_tensors
    tensor_system = smee.TensorSystem([tensor_top], [2], False)

    theta = (
        tensor_ff.potentials_by_type["vdW"].parameters,
        tensor_ff.potentials_by_type["vdW"].attributes,
    )

    coords = numpy.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]
    )
    box_vectors = numpy.stack([numpy.eye(3) * 20.0] * 3)

    ctx = mocker.MagicMock()
    ctx.kwargs = {
        "force_field": tensor_ff,
        "parameter_lookup": {"vdW": 0},
        "attribute_lookup": {"vdW": 1},
        "system": tensor_system,
    }
    ctx.coords = coords
    ctx.box_vectors = box_vectors
    ctx.needs_input_grad = [False, True, False]

    distances = numpy.linalg.norm(coords[:, 1] - coords[:, 0], axis=-1)

    expected_du_d_eps = 4.0 * ((sig / distances) ** 12 - (sig / distances) ** 6)
    expected_du_d_sig = (
        eps * (sig**5) * (48.0 * (sig**6) - 24.0 * distances**6)
    ) / (distances**12)

    du_d_theta = _compute_du_d_theta(theta, ctx)
    assert len(du_d_theta) == 2
    assert du_d_theta[1] is None

    assert numpy.allclose(
        du_d_theta[0][0, 0, :].numpy(), expected_du_d_eps, atol=1.0e-4
    )
    assert numpy.allclose(
        du_d_theta[0][0, 1, :].numpy(), expected_du_d_sig, atol=1.0e-4
    )


def test_compute_du_d_theta_attribute(mocker):
    attr_a, attr_b = 1.0, 2.0
    const_a, const_b = 3.0, 4.0

    tensor_potential = smee.TensorPotential(
        "bond",
        "bond-fn",
        parameters=torch.zeros((0, 0)),
        parameter_keys=[],
        parameter_cols=tuple(),
        parameter_units=tuple(),
        attributes=torch.tensor([attr_a, attr_b]),
        attribute_cols=("attr-a", "attr-b"),
        attribute_units=(openff.units.unit.angstrom, openff.units.unit.angstrom),
    )
    tensor_top = smee.TensorTopology(
        atomic_nums=torch.tensor([17, 17]),
        formal_charges=torch.tensor([0, 0]),
        bond_idxs=torch.tensor([[0, 1]]),
        bond_orders=torch.tensor([1]),
        parameters={
            "MOCK": smee.ValenceParameterMap(
                particle_idxs=torch.zeros((0, 0), dtype=torch.int64),
                assignment_matrix=torch.zeros((0, 0), dtype=torch.int64).to_sparse(),
            )
        },
        v_sites=None,
        constraints=None,
    )
    tensor_system = smee.TensorSystem([tensor_top], [1], False)

    def create_mock_force(*args):
        mock_force = openmm.CustomBondForce(f"{const_a} * attr_a + {const_b} * attr_b")
        mock_force.addPerBondParameter("attr_a")
        mock_force.addPerBondParameter("attr_b")
        mock_force.addBond(0, 1, [*args[0].attributes])
        return mock_force

    mocker.patch(
        "smee.converters.convert_to_openmm_force",
        autospec=True,
        side_effect=create_mock_force,
    )

    coords = numpy.zeros((1, 2, 3))
    box_vectors = numpy.stack([numpy.eye(3) * 20.0])

    kj_to_kcal = 4.184

    energy_0 = torch.tensor([const_a * attr_a + const_b * attr_b]) / kj_to_kcal

    du_d_theta = (
        _compute_du_d_theta_attribute(
            tensor_system, tensor_potential, energy_0, coords, box_vectors
        )
        * kj_to_kcal
    )

    assert du_d_theta.shape == (1, 2, 1)
    expected_du_d_theta = torch.tensor([[[const_a], [const_b]]])

    assert torch.allclose(du_d_theta, expected_du_d_theta, atol=1.0e-3)


def test_compute_ensemble_averages(mocker, mock_argon_tensors):
    tensor_ff, tensor_top = mock_argon_tensors
    tensor_system = smee.TensorSystem([tensor_top], [2], False)

    mock_outputs = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 6.0, 20.0])]
    mock_du_d_theta = (torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]), None)

    def populate_reporter(*args, **__):
        reporter = args[-1][0]
        reporter._values = mock_outputs
        reporter._coords = [numpy.zeros((1, 3))] * 2
        reporter._box_vectors = [numpy.eye(3)] * 2

    mock_simulate_fn = mocker.patch(
        "smee.mm.simulate",
        autospec=True,
        side_effect=populate_reporter,
    )
    mock_du_d_theta_fn = mocker.patch(
        "smee.mm._ops._compute_du_d_theta", autospec=True, return_value=mock_du_d_theta
    )

    temperature = 86.0 * openmm.unit.kelvin

    tensor_ff.potentials_by_type["vdW"].parameters.requires_grad = True

    ensemble_avgs = compute_ensemble_averages(
        tensor_system,
        tensor_ff,
        smee.mm.GenerateCoordsConfig(),
        [],
        smee.mm.SimulationConfig(temperature=temperature, pressure=None, n_steps=2),
        1,
    )

    assert mock_simulate_fn.call_count == 1
    assert mock_du_d_theta_fn.call_count == 0

    ensemble_avgs["potential_energy"].backward(retain_graph=True)
    energy_grad = tensor_ff.potentials_by_type["vdW"].parameters.grad
    tensor_ff.potentials_by_type["vdW"].parameters.grad = None

    mock_simulate_fn.assert_called_once()
    mock_du_d_theta_fn.assert_called_once()

    ensemble_avgs["volume"].backward(retain_graph=True)
    volume_grad = tensor_ff.potentials_by_type["vdW"].parameters.grad
    tensor_ff.potentials_by_type["vdW"].parameters.grad = None

    mock_simulate_fn.assert_called_once()
    assert mock_du_d_theta_fn.call_count == 2

    ensemble_avgs["density"].backward(retain_graph=False)
    density_grad = tensor_ff.potentials_by_type["vdW"].parameters.grad
    tensor_ff.potentials_by_type["vdW"].parameters.grad = None

    mock_simulate_fn.assert_called_once()
    assert mock_du_d_theta_fn.call_count == 3

    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)
    beta = beta.value_in_unit(openmm.unit.kilocalorie_per_mole**-1)

    mock_outputs = torch.stack(mock_outputs)

    energy, volume, density = mock_outputs[:, 0], mock_outputs[:, 1], mock_outputs[:, 2]
    du_d_eps = mock_du_d_theta[0][0, 0, :]

    expected_d_avg_energy_d_eps = du_d_eps.mean(-1) - beta * (
        (energy * du_d_eps).mean(-1) - energy.mean() * du_d_eps.mean(-1)
    )
    expected_d_avg_volume_d_eps = -beta * (
        (volume * du_d_eps).mean(-1) - volume.mean() * du_d_eps.mean(-1)
    )
    expected_d_avg_density_d_eps = -beta * (
        (density * du_d_eps).mean(-1) - density.mean() * du_d_eps.mean(-1)
    )

    assert torch.isclose(expected_d_avg_energy_d_eps.double(), energy_grad[0, 0])
    assert torch.isclose(expected_d_avg_volume_d_eps.double(), volume_grad[0, 0])
    assert torch.isclose(expected_d_avg_density_d_eps.double(), density_grad[0, 0])
