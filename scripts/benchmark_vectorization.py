"""Compare dense-loop, low-rank-loop and batched risk post-processing.

Run from the project root:
    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
      python3 -m scripts.benchmark_vectorization
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from src.dgps import make_diagonal_model, initialize_student, initialize_bias
from src.methods import population_risk_and_gradients
from src.methods_optimized import low_rank_field
from src.simulation import build_times


def dense_loop(states, lam, B, d, width):
    n = states.shape[1]
    risks, mismatch = np.empty(n), np.empty(n)
    for j in range(n):
        W = states[:-1, j].reshape(d, width)
        b = float(states[-1, j])
        risks[j], _, _, mismatch[j] = population_risk_and_gradients(W, b, lam, B)
    return risks, mismatch


def lowrank_loop(states, lam, B, d, width):
    n = states.shape[1]
    risks, mismatch = np.empty(n), np.empty(n)
    c = lam**2 * np.diag(B)
    constant = np.sum((lam * np.diag(B))**2)
    for j in range(n):
        W = states[:-1, j].reshape(d, width)
        row_squares = np.sum(W * W, axis=1)
        gram = W.T @ (lam[:, None] * W)
        e = float(states[-1, j] - np.dot(lam, row_squares) / width)
        risks[j] = (np.sum(gram * gram) / width**2
                    - 2 * np.dot(c, row_squares) / width
                    + constant + 0.5 * e**2)
        mismatch[j] = e
    return risks, mismatch


def batched(states, lam, B, d, width, batch_size=16):
    n = states.shape[1]
    risks, mismatch = np.empty(n), np.empty(n)
    c = lam**2 * np.diag(B)
    constant = np.sum((lam * np.diag(B))**2)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        W = states[:-1, start:stop].T.reshape(stop - start, d, width)
        row_squares = np.sum(W * W, axis=2)
        gram = np.matmul(np.swapaxes(W, 1, 2), lam[None, :, None] * W)
        e = states[-1, start:stop] - (row_squares @ lam) / width
        risks[start:stop] = (np.sum(gram * gram, axis=(1, 2)) / width**2
                             - 2 * (row_squares @ c) / width
                             + constant + 0.5 * e**2)
        mismatch[start:stop] = e
    return risks, mismatch


def get_states(config_path, width):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    d = int(cfg["dimension"])
    beta = 0.35
    alpha = float(cfg["alpha_plus_beta"]) - beta
    lam, B = make_diagonal_model(d, beta, alpha)
    rng = np.random.default_rng(np.random.SeedSequence([
        int(cfg["master_seed"]), d, width, 0]))
    W0 = initialize_student(rng, d, width, "paper")
    b0 = initialize_bias(rng, "normal")
    teacher_weight = lam**2 * np.diag(B)
    times = build_times(cfg)

    def rhs(t, state):
        W = state[:-1].reshape(d, width)
        dW, e = low_rank_field(W, float(state[-1]), lam, teacher_weight)
        return np.r_[dW.ravel(), -e / float(cfg["delta"])]

    sol = solve_ivp(rhs, (times[0], times[-1]),
                    np.r_[W0.ravel(), b0], t_eval=times,
                    rtol=float(cfg["rtol"]), atol=float(cfg["atol"]),
                    method="RK45")
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y, lam, B, d


def main():
    out = Path("results/performance")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_path, width in [("config.json", 16),
                               ("configs/d256_width_pilot.json", 64)]:
        states, lam, B, d = get_states(config_path, width)
        funcs = {
            "dense_loop": lambda: dense_loop(states, lam, B, d, width),
            "lowrank_loop": lambda: lowrank_loop(states, lam, B, d, width),
            "batched": lambda: batched(states, lam, B, d, width),
        }
        reference_risk, reference_e = funcs["dense_loop"]()
        for name, func in funcs.items():
            risk, e = func()
            np.testing.assert_allclose(risk, reference_risk, atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(e, reference_e, atol=1e-8, rtol=1e-8)
        timings = {name: [] for name in funcs}
        for rep in range(7):
            names = list(funcs)
            names = names[rep % len(names):] + names[:rep % len(names)]
            for name in names:
                start = time.perf_counter()
                funcs[name]()
                elapsed = time.perf_counter() - start
                timings[name].append(elapsed)
                rows.append({"d": d, "width": width, "method": name,
                             "rep": rep, "seconds": elapsed})
        med = {name: float(np.median(v)) for name, v in timings.items()}
        print(f"d={d}, width={width}, saved points={states.shape[1]}")
        for name, seconds in med.items():
            print(f"  {name:13s}: {seconds:.6f} s")
        print(f"  batched vs lowrank loop: {med['lowrank_loop']/med['batched']:.2f}x")
        print(f"  batched vs dense loop:   {med['dense_loop']/med['batched']:.2f}x")

    filename = out / "vectorization_benchmark.csv"
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
