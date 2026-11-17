"""Compare the original and low-rank GF solvers."""

import csv
import json
import time
from pathlib import Path
from statistics import median

import numpy as np

from src.dgps import make_diagonal_model, initialize_student, initialize_bias
from src.simulation import build_times
from src.methods import solve_paper_population_gf
from src.methods_optimized import solve_paper_population_gf_optimized


solvers = {
    "baseline": solve_paper_population_gf,
    "optimized": solve_paper_population_gf_optimized,
}

cases = [
    ("config.json", 16),
    ("configs/d256_width_pilot.json", 64),
]

rows = []

for config_path, width in cases:
    with open(config_path) as f:
        cfg = json.load(f)

    d = cfg["dimension"]
    beta = 0.35
    alpha = cfg["alpha_plus_beta"] - beta

    rng = np.random.default_rng(
        np.random.SeedSequence([cfg["master_seed"], d, width, 0])
    )
    W0 = initialize_student(rng, d, width, "paper")
    b0 = initialize_bias(rng, "normal")

    lam, B = make_diagonal_model(d, beta, alpha)
    times = build_times(cfg)
    opts = dict(delta=cfg["delta"], rtol=cfg["rtol"], atol=cfg["atol"])

    # Warm up both implementations.
    for solver in solvers.values():
        solver(W0, b0, lam, B, times, **opts)

    timings = {name: [] for name in solvers}

    for rep in range(5):
        order = list(solvers)
        if rep % 2:
            order.reverse()

        for name in order:
            start = time.perf_counter()
            solvers[name](W0, b0, lam, B, times, **opts)
            elapsed = time.perf_counter() - start

            timings[name].append(elapsed)
            rows.append([d, width, name, rep, elapsed])

    old = median(timings["baseline"])
    new = median(timings["optimized"])

    print(f"d={d}, width={width}")
    print(f"  Baseline:  {old:.4f} s")
    print(f"  Optimized: {new:.4f} s")
    print(f"  Speedup:   {old / new:.2f}x")

out = Path("results/performance/algorithmic_benchmark.csv")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["d", "width", "method", "rep", "seconds"])
    writer.writerows(rows)

print(f"Saved: {out}")
