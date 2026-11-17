"""Check numerical agreement for a sensitive GF trajectory."""

import json
import numpy as np

from src.dgps import (
    make_diagonal_model,
    initialize_student,
    initialize_bias,
)
from src.simulation import build_times
from src.methods import solve_paper_population_gf
from src.methods_optimized import solve_paper_population_gf_optimized


def test_width128_tight_tolerance():
    with open("configs/d256_width_pilot.json") as f:
        cfg = json.load(f)

    d, width, rep = 256, 128, 0
    beta = 0.35
    alpha = cfg["alpha_plus_beta"] - beta

    rng = np.random.default_rng(
        np.random.SeedSequence([cfg["master_seed"], d, width, rep])
    )

    W0 = initialize_student(rng, d, width, "paper")
    b0 = initialize_bias(rng, "normal")

    lam, B = make_diagonal_model(d, beta, alpha)
    times = build_times(cfg)

    opts = dict(delta=cfg["delta"], rtol=1e-8, atol=1e-10)
    args = (W0, b0, lam, B, times)

    _, risk_old, e_old = solve_paper_population_gf(*args, **opts)
    _, risk_new, e_new = solve_paper_population_gf_optimized(
        *args, **opts
    )

    np.testing.assert_allclose(
        risk_old, risk_new, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        e_old, e_new, rtol=1e-8, atol=1e-9
    )
