import torch

from smee.optimize import _levenberg_marquardt_solver, _levenberg_marquardt_step


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
