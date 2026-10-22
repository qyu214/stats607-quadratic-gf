import numpy as np
from src.dgps import make_diagonal_model
from src.methods import loss_and_gradient, solve_population_gradient_flow


def test_gradient_matches_finite_difference():
    lam, B = make_diagonal_model(3, 0.4, 1.0)
    W = np.random.default_rng(7).normal(size=(3, 3)) * 0.2
    _, analytic = loss_and_gradient(W, lam, B)
    eps = 1e-6
    numeric = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wa, Wb = W.copy(), W.copy()
            Wa[i, j] += eps
            Wb[i, j] -= eps
            numeric[i, j] = (loss_and_gradient(Wa, lam, B)[0] -
                             loss_and_gradient(Wb, lam, B)[0]) / (2 * eps)
    assert np.allclose(analytic, numeric, rtol=2e-5, atol=1e-7)


def test_gradient_flow_risk_nonincreasing():
    lam, B = make_diagonal_model(4, 0.8, 1.0)
    W0 = np.random.default_rng(8).normal(size=(4, 4)) * 0.15
    times = np.linspace(0, 8, 40)
    _, risks = solve_population_gradient_flow(W0, lam, B, times)
    assert np.all(np.diff(risks) <= 2e-7)
