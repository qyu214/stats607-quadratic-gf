"""Runtime scaling of the original and low-rank population GF solvers."""

import argparse
import csv
import time
import timeit
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dgps import make_diagonal_model, initialize_student, initialize_bias
from src.methods import population_risk_and_gradients, solve_paper_population_gf
from src.methods_optimized import low_rank_field, solve_paper_population_gf_optimized


def benchmark_rhs(func, repeats):
    """Time one RHS call; batch calls to reduce timer noise."""
    for _ in range(3):
        func()
    start = time.perf_counter()
    for _ in range(20):
        func()
    per_call = (time.perf_counter() - start) / 20
    number = max(10, min(10000, int(0.12 / max(per_call, 1e-9))))
    return [t / number for t in timeit.repeat(func, repeat=repeats, number=number)]


def make_case(d, width):
    beta = alpha = 0.35
    delta = 0.05
    lam, B = make_diagonal_model(d, beta, alpha)
    rng = np.random.default_rng(np.random.SeedSequence([20261019, d, width, 0]))
    W0 = initialize_student(rng, d, width, "paper")
    b0 = initialize_bias(rng, "normal")
    state = np.r_[W0.ravel(), b0]
    teacher_weight = lam**2 * np.diag(B)
    times = np.r_[0.0, np.geomspace(0.02, 80.0, 50)]
    opts = {"delta": delta, "rtol": 1e-6, "atol": 1e-8}

    def rhs_original():
        W = state[:-1].reshape(d, width)
        b = float(state[-1])
        _, grad, e, _ = population_risk_and_gradients(W, b, lam, B)
        return np.r_[(-width * grad).ravel(), -e / delta]

    def rhs_optimized():
        W = state[:-1].reshape(d, width)
        b = float(state[-1])
        dW, e = low_rank_field(W, b, lam, teacher_weight)
        return np.r_[dW.ravel(), -e / delta]

    solvers = {
        "original": lambda: solve_paper_population_gf(W0, b0, lam, B, times, **opts),
        "low_rank": lambda: solve_paper_population_gf_optimized(W0, b0, lam, B, times, **opts),
    }

    np.testing.assert_allclose(rhs_original(), rhs_optimized(), rtol=1e-11, atol=1e-12)
    # Run both once to check the resulting risk curves and warm up each version.
    t0, risk0, e0 = solvers["original"]()
    t1, risk1, e1 = solvers["low_rank"]()
    np.testing.assert_array_equal(t0, t1)
    np.testing.assert_allclose(risk0, risk1, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(e0, e1, rtol=1e-6, atol=1e-7)

    return {"original": rhs_original, "low_rank": rhs_optimized}, solvers


def draw_plot(summary, design_names, metric, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (design, title) in zip(axes, design_names):
        rows = [x for x in summary if x["design"] == design]
        for version, label in [("original", "Original"), ("low_rank", "Low-rank")]:
            ds = np.array([x["d"] for x in rows], dtype=float)
            ts = np.array([x[f"{metric}_{version}_s"] for x in rows], dtype=float)
            slope = np.polyfit(np.log(ds), np.log(ts), 1)[0]
            ax.loglog(ds, ts, marker="o", label=f"{label} (slope {slope:.2f})")
        ax.set_title(title)
        ax.set_xlabel("Dimension d")
        ax.set_ylabel("Time (seconds)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of timed repeats per solver (default: 5)")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    designs = [
        ("fixed_width", "Fixed width (r_s = 16)", lambda d: 16),
        ("proportional", "Proportional width (r_s = d/4)", lambda d: d // 4),
    ]
    dimensions = (32, 64, 128, 256)
    outdir = Path("results/performance")
    outdir.mkdir(parents=True, exist_ok=True)
    raw, summary = [], []

    for design, title, get_width in designs:
        for d in dimensions:
            width = get_width(d)
            print(f"{design}: d={d}, width={width}", flush=True)
            rhs_funcs, solvers = make_case(d, width)
            timing = {}
            for version in ("original", "low_rank"):
                timing[("rhs", version)] = benchmark_rhs(rhs_funcs[version], args.repeats)

            # Alternate order to avoid always benchmarking the same solver first.
            timing[("ode", "original")] = []
            timing[("ode", "low_rank")] = []
            for rep in range(args.repeats):
                order = ("original", "low_rank") if rep % 2 == 0 else ("low_rank", "original")
                for version in order:
                    start = time.perf_counter()
                    solvers[version]()
                    timing[("ode", version)].append(time.perf_counter() - start)

            row = {"design": design, "d": d, "width": width}
            for metric in ("rhs", "ode"):
                for version in ("original", "low_rank"):
                    values = timing[(metric, version)]
                    row[f"{metric}_{version}_s"] = float(np.median(values))
                    for rep, seconds in enumerate(values):
                        raw.append({"design": design, "d": d, "width": width,
                                    "metric": metric, "version": version,
                                    "rep": rep, "seconds": seconds})
                row[f"{metric}_speedup"] = row[f"{metric}_original_s"] / row[f"{metric}_low_rank_s"]
            summary.append(row)
            print(f"  RHS: {row['rhs_speedup']:.2f}x; full ODE: {row['ode_speedup']:.2f}x", flush=True)

    raw_file = outdir / "complexity_raw.csv"
    with raw_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        writer.writeheader()
        writer.writerows(raw)

    summary_file = outdir / "complexity_summary.csv"
    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    names = [(design, title) for design, title, _ in designs]
    for metric in ("rhs", "ode"):
        draw_plot(summary, names, metric, outdir / f"complexity_{metric}.png")

    print("\nEmpirical log-log slopes (fit over these four dimensions):")
    for design, _, _ in designs:
        rows = [x for x in summary if x["design"] == design]
        ds = np.array([x["d"] for x in rows], dtype=float)
        for metric in ("rhs", "ode"):
            estimates = []
            for version in ("original", "low_rank"):
                ts = np.array([x[f"{metric}_{version}_s"] for x in rows])
                estimates.append(np.polyfit(np.log(ds), np.log(ts), 1)[0])
            print(f"  {design}, {metric}: original {estimates[0]:.2f}, low-rank {estimates[1]:.2f}")
    print(f"Saved: {raw_file}, {summary_file}, complexity_rhs.png, complexity_ode.png")


if __name__ == "__main__":
    main()
