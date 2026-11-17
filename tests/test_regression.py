"""Check that the low-rank implementation agrees with the original solver."""

import json
import numpy as np
import pytest

from src.dgps import make_diagonal_model, initialize_student, initialize_bias
from src.simulation import build_times
from src.methods import population_risk_and_gradients, solve_paper_population_gf
from src.methods_optimized import low_rank_field, solve_paper_population_gf_optimized


def initial_state(d, width):
    rng = np.random.default_rng(
        np.random.SeedSequence([20261019, d, width, 0])
    )
    W0 = initialize_student(rng, d, width, "paper")
    b0 = initialize_bias(rng, "normal")
    return W0, b0


@pytest.mark.parametrize("d,width,beta", [
    (32, 8, 0.10),
    (32, 8, 0.35),
    (32, 16, 0.35),
    (32, 16, 0.60),
    (256, 32, 0.35),
    (256, 64, 0.35),
    (256, 128, 0.35),
])
def test_low_rank_field(d, width, beta):
    alpha = 0.7 - beta
    lam, B = make_diagonal_model(d, beta, alpha)
    W, b = initial_state(d, width)

    _, grad, e, _ = population_risk_and_gradients(W, b, lam, B)

    teacher_weight = lam**2 * np.diag(B)
    dW, e_new = low_rank_field(W, b, lam, teacher_weight)

    np.testing.assert_allclose(dW, -width * grad, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(e_new, e, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("config_path,width", [
    ("config.json", 16),
    ("configs/d256_width_pilot.json", 64),
])
def test_full_trajectory(config_path, width):
    with open(config_path) as f:
        cfg = json.load(f)

    d = cfg["dimension"]
    beta = 0.35
    alpha = cfg["alpha_plus_beta"] - beta

    W0, b0 = initial_state(d, width)
    lam, B = make_diagonal_model(d, beta, alpha)
    times = build_times(cfg)

    args = (W0, b0, lam, B, times)
    opts = dict(delta=cfg["delta"], rtol=cfg["rtol"], atol=cfg["atol"])

    t_old, risk_old, e_old = solve_paper_population_gf(*args, **opts)
    t_new, risk_new, e_new = solve_paper_population_gf_optimized(*args, **opts)

    np.testing.assert_array_equal(t_old, t_new)
    np.testing.assert_allclose(risk_old, risk_new, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(e_old, e_new, rtol=1e-6, atol=1e-7)
