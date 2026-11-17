import json
import numpy as np
import pandas as pd

from src.dgps import make_diagonal_model, initialize_student, initialize_bias
from src.simulation import build_times
from src.methods import solve_paper_population_gf
from src.methods_optimized import solve_paper_population_gf_optimized

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

for label, rtol, atol in [
    ("Default", 1e-6, 1e-8),
    ("Tight", 1e-8, 1e-10),
]:
    opts = dict(delta=cfg["delta"], rtol=rtol, atol=atol)

    t, risk_old, e_old = solve_paper_population_gf(
        W0, b0, lam, B, times, **opts
    )
    _, risk_new, e_new = solve_paper_population_gf_optimized(
        W0, b0, lam, B, times, **opts
    )

    j = np.argmin(np.abs(t - 146.115010))

    print(f"\n{label} tolerances:")
    print(f"  Max risk difference: {np.max(np.abs(risk_old-risk_new)):.3e}")
    print(f"  Max e difference:    {np.max(np.abs(e_old-e_new)):.3e}")
    print(f"  At t={t[j]:.6f}:")
    print(f"    Original e:  {e_old[j]:.12e}")
    print(f"    Optimized e: {e_new[j]:.12e}")
    print(f"    Difference:  {abs(e_old[j]-e_new[j]):.3e}")

    if label == "Default":
        paths = [
            ("Original CSV", "results/width256/raw/trajectories.csv", e_old),
            ("Optimized CSV",
             "results/performance/parallel_width256/trajectories.csv", e_new),
        ]

        for name, path, recalculated in paths:
            df = pd.read_csv(path)
            subset = df[
                (df.student_width == width)
                & (df.replicate == rep)
                & (df.beta == beta)
            ].sort_values("time")

            saved = subset.bias_mismatch.to_numpy()
            assert len(saved) == len(recalculated)

            print(
                f"  {name} vs rerun: "
                f"{np.max(np.abs(saved-recalculated)):.3e}"
            )
