import logging

import pytest
import torch

from smee.optimizers._lm import (
    LevenbergMarquardt,
    _damping_factor_loss_fn,
    _solver,
    _step,
)


def test_solver():
    gradient = torch.tensor([0.5, -0.3, 0.7])
    hessian = torch.tensor(
        [
            [2.0, 0.5, 0.3],
            [0.5, 1.8, 0.2],
            [0.3, 0.2, 1.5],
        ]
    )

    damping_factor = torch.tensor(1.2)

    dx, solution = _solver(damping_factor, gradient, hessian)

    # computed using ForceBalance 1.9.3
    expected_dx = torch.tensor([-0.24833229, 0.27860679, -0.44235173])
    expected_solution = torch.tensor(-0.26539651272205717)

    assert dx.shape == expected_dx.shape
    assert torch.allclose(dx, expected_dx)

    assert solution.shape == expected_solution.shape
    assert torch.allclose(solution, expected_solution)


def test_step():
    gradient = torch.tensor([0.5, -0.3, 0.7])
    hessian = torch.tensor(
        [
            [2.0, 0.5, 0.3],
            [0.5, 1.8, 0.2],
            [0.3, 0.2, 1.5],
        ]
    )

    expected_trust_radius = torch.tensor(0.123)

    dx, solution, adjusted = _step(
        gradient, hessian, trust_radius=expected_trust_radius
    )
    assert isinstance(dx, torch.Tensor)
    assert dx.shape == gradient.shape

    assert isinstance(solution, torch.Tensor)
    assert solution.shape == torch.Size([])

    assert torch.isclose(torch.norm(dx), torch.tensor(expected_trust_radius))
    assert adjusted is True


def test_step_sd(caplog):
    gradient = torch.tensor([1.0, 1.0])
    hessian = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    with caplog.at_level(logging.INFO):
        _ = _step(gradient, hessian, trust_radius=1.0)

    # TODO: not 100% sure on what cases LPW was trying to correct for here,
    #       for now settle to double check the SD is applied.
    assert "hessian has a small or negative eigenvalue" in caplog.text


def test_damping_factor_loss_fn(mocker):
    dx = torch.tensor([3.0, 4.0, 0.0])
    dx_norm = torch.linalg.norm(dx)

    damping_factor = mocker.Mock()
    gradient = mocker.Mock()
    hessian = mocker.Mock()

    solver_fn = mocker.patch(
        "smee.optimizers._lm._solver", autospec=True, return_value=(dx, 0.0)
    )

    trust_radius = 12

    difference = _damping_factor_loss_fn(
        damping_factor, gradient, hessian, trust_radius
    )

    solver_fn.assert_called_once_with(damping_factor, gradient, hessian)

    expected_difference = (dx_norm - trust_radius) ** 2
    assert torch.isclose(difference, expected_difference)


def test_levenberg_marquardt_adaptive(mocker, caplog):
    """Make sure the trust radius is adjusted correctly based on the loss."""

    mock_dx_traj = [
        (
            torch.tensor([10.0, 20]),
            torch.tensor(-100.0),
            False,
        ),
        (
            torch.tensor([0.1, 0.2]),
            torch.tensor(-0.5),
            False,
        ),
        (
            torch.tensor([0.05, 0.01]),
            torch.tensor(-2.0),
            True,
        ),
    ]
    mock_step_fn = mocker.patch(
        "smee.optimizers._lm._step", autospec=True, side_effect=mock_dx_traj
    )

    mock_loss_traj = [
        (
            torch.tensor(0.0),
            torch.tensor([1.0, 2.0]),
            torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        ),
        (
            torch.tensor(150.0),
            torch.tensor([7.0, 8.0]),
            torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
        ),
        (
            torch.tensor(-0.1),
            torch.tensor([13.0, 14.0]),
            torch.tensor([[15.0, 16.0], [17.0, 18.0]]),
        ),
        (
            torch.tensor(-2.1),
            torch.tensor([19.0, 20.0]),
            torch.tensor([[21.0, 22.0], [23.0, 24.0]]),
        ),
    ]
    expected_loss_traj = [
        (
            pytest.approx(mock_loss_traj[i][1]),
            pytest.approx(mock_loss_traj[i][2]),
            mocker.ANY,
        )
        for i in [0, 0, 2]
    ]

    x_traj = []

    def mock_loss_fn(_x):
        x_traj.append(_x.clone())
        return mock_loss_traj.pop(0)

    x = torch.tensor([0.0, 0.0])

    optimizer = LevenbergMarquardt()

    with caplog.at_level(logging.INFO):
        for _ in range(3):
            optimizer.step(x, mock_loss_fn)

    expected_x_traj = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([10.0, 20.0]),
        # previous step should have been rejected
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.15, 0.21]),
    ]
    assert x.shape == expected_x_traj[-1].shape
    assert torch.allclose(x, expected_x_traj[-1])

    trust_radius_messages = [m for m in caplog.messages if "trust radius" in m]

    expected_messages = [
        "reducing trust radius to",
        "low quality step detected - reducing trust radius to",
        "updating trust radius to",
    ]
    assert len(trust_radius_messages) == len(expected_messages)

    for message, expected in zip(trust_radius_messages, expected_messages):
        assert message.startswith(expected)

    # mock_step_fn.assert_has_calls(expected_loss_traj, any_order=False)
    mock_step_fn_calls = [call.args for call in mock_step_fn.call_args_list]

    assert mock_step_fn_calls == expected_loss_traj


def test_levenberg_marquardt():
    expected = torch.tensor([5.0, 3.0, 2.0])

    x_ref = torch.linspace(-2.0, 2.0, 100)
    y_ref = expected[0] * x_ref**2 + expected[1] * x_ref + expected[2]

    theta = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

    def loss_fn(_theta: torch.Tensor) -> torch.Tensor:
        y = _theta[0] * x_ref**2 + _theta[1] * x_ref + _theta[2]
        return torch.sum((y - y_ref) ** 2)

    @torch.enable_grad()
    def target_fn(
        _theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss = loss_fn(_theta)
        (grad,) = torch.autograd.grad(loss, _theta, torch.tensor(1.0))
        hess = torch.autograd.functional.hessian(loss_fn, _theta)

        return loss.detach(), grad.detach(), hess.detach()

    optimizer = LevenbergMarquardt()

    for _ in range(15):
        optimizer.step(theta, target_fn)

    assert theta.shape == expected.shape
    assert torch.allclose(theta, expected)
