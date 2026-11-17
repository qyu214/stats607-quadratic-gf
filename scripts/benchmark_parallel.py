"""Benchmark reproducible parallel execution of independent GF trajectories."""

import argparse
import csv
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np

from src.dgps import initialize_bias, initialize_student, make_diagonal_model
from src.methods_optimized import solve_paper_population_gf_optimized
from src.simulation import build_times


def run_job(job):
    cfg, width, rep = job
    d = int(cfg["dimension"])
    rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg["master_seed"]), d, width, rep])
    )
    W0 = initialize_student(rng, d, width, cfg.get("initialization_std", "paper"))
    b0 = initialize_bias(rng, cfg.get("bias_initialization", "normal"))
    times = build_times(cfg)
    rows = []
    for beta in cfg["covariance_exponents"]:
        alpha = (
            float(cfg["alpha_plus_beta"]) - float(beta)
            if "alpha_plus_beta" in cfg
            else float(cfg["teacher_exponent"])
        )
        lam, B = make_diagonal_model(d, float(beta), alpha, cfg.get("teacher_rank"))
        t, risks, mismatch = solve_paper_population_gf_optimized(
            W0, b0, lam, B, times,
            delta=float(cfg.get("delta", 0.05)),
            rtol=float(cfg.get("rtol", 1e-6)),
            atol=float(cfg.get("atol", 1e-8)),
        )
        for ti, ri, ei in zip(t, risks, mismatch):
            rows.append((d, width, rep, float(beta), alpha, float(ti), float(ri), float(ei)))
    return rows


def run_study(cfg, workers):
    jobs = [
        (cfg, int(width), rep)
        for width in cfg["student_widths"]
        for rep in range(int(cfg["n_replicates"]))
    ]
    if workers == 1:
        batches = [run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            # map preserves task order even though workers finish out of order.
            batches = list(pool.map(run_job, jobs))
    return [row for batch in batches for row in batch]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d256_width_pilot.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if args.repeats < 1 or not args.workers or min(args.workers) < 1:
        parser.error("repeats and worker counts must be positive")
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    output = Path("results/performance")
    output.mkdir(parents=True, exist_ok=True)
    reference = None
    raw = []
    medians = {}

    for workers in args.workers:
        durations = []
        for rep in range(args.repeats):
            start = time.perf_counter()
            rows = run_study(cfg, workers)
            elapsed = time.perf_counter() - start
            # Compare all time points, not just final risks.
            array = np.asarray(rows, dtype=float)
            if reference is None:
                reference = array
            else:
                np.testing.assert_allclose(
                    array, reference, rtol=1e-9, atol=1e-10,
                    err_msg=f"parallel results changed with {workers} workers",
                )
            durations.append(elapsed)
            raw.append((workers, rep, elapsed))
        medians[workers] = median(durations)
        print(f"workers={workers}: median {medians[workers]:.3f}s "
              f"(checked {len(rows)} timepoint rows)", flush=True)

    serial = medians.get(1)
    if serial is not None:
        for workers in args.workers:
            print(f"workers={workers}: speedup {serial / medians[workers]:.2f}x", flush=True)

    with (output / "parallel_benchmark.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workers", "repeat", "seconds"])
        writer.writerows(raw)
    print(f"Saved: {output / 'parallel_benchmark.csv'}", flush=True)


if __name__ == "__main__":
    main()
