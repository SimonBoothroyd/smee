import torch

from smee.optimize import (
    LevenbergMarquardtConfig,
    _levenberg_marquardt_solver,
    _levenberg_marquardt_step,
    optimize,
)


def test_levenberg_marquardt_solver():
    gradient = torch.tensor([0.5, -0.3, 0.7])
    hessian = torch.tensor(
        [
            [2.0, 0.5, 0.3],
            [0.5, 1.8, 0.2],
            [0.3, 0.2, 1.5],
        ]
    )

    damping_factor = torch.tensor(1.2)

    dx, solution = _levenberg_marquardt_solver(damping_factor, gradient, hessian)

    # computed using ForceBalance 1.9.3
    expected_dx = torch.tensor([-0.24833229, 0.27860679, -0.44235173])
    expected_solution = torch.tensor(-0.26539651272205717)

    assert dx.shape == expected_dx.shape
    assert torch.allclose(dx, expected_dx)

    assert solution.shape == expected_solution.shape
    assert torch.allclose(solution, expected_solution)


def test_levenberg_marquardt_step():
    gradient = torch.tensor([0.5, -0.3, 0.7])
    hessian = torch.tensor(
        [
            [2.0, 0.5, 0.3],
            [0.5, 1.8, 0.2],
            [0.3, 0.2, 1.5],
        ]
    )

    expected_trust_radius = 0.123

    dx, solution, adjusted = _levenberg_marquardt_step(
        gradient, hessian, trust_radius=expected_trust_radius
    )
    assert isinstance(dx, torch.Tensor)
    assert dx.shape == gradient.shape

    assert isinstance(solution, torch.Tensor)
    assert solution.shape == torch.Size([])

    assert torch.isclose(torch.norm(dx), torch.tensor(expected_trust_radius))
    assert adjusted is True


def test_optimize():
    expected = torch.tensor([5.0, 3.0, 2.0])

    x_ref = torch.linspace(-2.0, 2.0, 100)
    y_ref = expected[0] * x_ref**2 + expected[1] * x_ref + expected[2]

    theta_0 = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

    def loss_fn(theta: torch.Tensor) -> torch.Tensor:
        y = theta[0] * x_ref**2 + theta[1] * x_ref + theta[2]
        return torch.sum((y - y_ref) ** 2)

    def target_fn(
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            loss = loss_fn(theta)
            (grad,) = torch.autograd.grad(loss, theta, torch.tensor(1.0))
            hess = torch.autograd.functional.hessian(loss_fn, theta)

        return loss.detach(), grad.detach(), hess.detach()

    x_final = optimize(theta_0, target_fn, LevenbergMarquardtConfig())

    assert x_final.shape == expected.shape
    assert torch.allclose(x_final, expected)
