"""Compare the complete Project 2 workflow across three implementations.

V0: original dense GF, serial
V1: low-rank GF, serial
V2: low-rank GF, parallel

Each version runs all three original configurations, including analysis/plots.
Outputs are isolated under results/performance/full_study/; Project 2 results
are not overwritten. Invoke from the repository root via `python -m scripts...`.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ROOT = Path("results/performance")
STUDIES = (
    ("main", "config.json", 4500),
    ("width256", "configs/d256_width_pilot.json", 1515),
    ("fixed_alpha", "configs/fixed_alpha_sweep.json", 2250),
)
KEYS = ["study_id", "dimension", "beta", "alpha", "student_width", "replicate", "time"]
VALUES = ["population_risk", "bias_mismatch"]
VERSIONS = ("V0", "V1", "V2")


def command(cmd, log_path, env):
    """Run one workflow stage, record wall time and keep its console log."""
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}\nSee {log_path}")
    return elapsed


def run_study(version, study, cfg_path, expected_rows, repeat, workers, env):
    dest = ROOT / "full_study" / version / f"run_{repeat:02d}" / study
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / "trajectories.csv"
    log = dest / "commands.log"
    py = sys.executable
    if version == "V0":
        simulation = [py, "-m", "src.simulation", "--config", cfg_path, "--output", str(raw)]
    else:
        simulation = [
            py, "-m", "scripts.run_optimized_study", "--config", cfg_path,
            "--workers", str(1 if version == "V1" else workers),
            "--output", str(raw),
        ]
    sim_s = command(simulation, log, env)
    post_s = 0.0

    if study in ("main", "fixed_alpha"):
        summary = dest / "summary.csv"
        endpoints = dest / "endpoints.csv"
        post_s += command([
            py, "-m", "src.analyze", "--raw", str(raw),
            "--output", str(summary), "--endpoints", str(endpoints),
        ], log, env)
        module = "src.figures" if study == "main" else "src.fixed_alpha_figures"
        cmd = [py, "-m", module, "--summary", str(summary),
               "--endpoints", str(endpoints), "--out", str(dest / "figures")]
        if study == "main":
            cmd.extend(["--raw", str(raw)])
        post_s += command(cmd, log, env)
    else:
        # Original width-follow-up module produces summary, checkpoints and figures.
        post_s += command([
            py, "-m", "src.width_pilot_figures", "--raw", str(raw),
            "--out", str(dest),
        ], log, env)

    with raw.open("r", encoding="utf-8") as f:
        nrows = sum(1 for _ in f) - 1
    if nrows != expected_rows:
        raise RuntimeError(f"{version}/{study}: expected {expected_rows} rows, got {nrows}")
    print(f"  {study:<11} simulation {sim_s:7.2f}s  analysis/figures {post_s:7.2f}s  rows {nrows}", flush=True)
    return sim_s, post_s


def compare_results(reference, candidate, study, label):
    left = pd.read_csv(reference).sort_values(KEYS).reset_index(drop=True)
    right = pd.read_csv(candidate).sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[KEYS], right[KEYS], check_dtype=False)
    if not np.isfinite(left[VALUES].to_numpy()).all() or not np.isfinite(right[VALUES].to_numpy()).all():
        raise ValueError(f"Non-finite numerical output in {study}: {label}")
    results = []
    for metric in VALUES:
        a = left[metric].to_numpy(float)
        b = right[metric].to_numpy(float)
        diff = np.abs(a - b)
        # Diagnostic threshold; not a claim that every element passes.
        tol = 1e-7 + 1e-6 * np.abs(a)
        flags = int(np.sum(diff > tol))
        results.append((study, label, metric, len(diff), float(np.max(diff)), flags))
        print(f"  {study:<11} {label:<8} {metric:<16} max |diff|={diff.max():.3e}; flags={flags}")
        # Worker counts should not change any computed trajectory for the optimized solver.
        if label == "V1 vs V2":
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
    return results


def make_figure(medians):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["V0 Original", "V1 Low-rank", "V2 Low-rank + parallel"]
    values = [medians[v] for v in VERSIONS]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(labels, values)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{value:.2f}s", ha="center", va="bottom")
    ax.set(ylabel="Full study wall time (s)", title="Project 2 workflow: 3 studies, analysis and figures")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output = ROOT / "full_study_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 2:
        parser.error("--repeats must be positive and --workers must be at least 2")

    required = [Path(p) for _, p, _ in STUDIES] + [
        Path("src/methods.py"), Path("src/methods_optimized.py"),
        Path("scripts/run_optimized_study.py"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        parser.error("Run from repository root. Missing: " + ", ".join(missing))

    ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"

    rows = []
    totals = {v: [] for v in VERSIONS}
    # Rotate order each repetition to reduce effects of run order and warm caches.
    for rep in range(1, args.repeats + 1):
        order = list(VERSIONS)
        order = order[(rep - 1) % len(order):] + order[:(rep - 1) % len(order)]
        for version in order:
            print(f"\nRepeat {rep}/{args.repeats}: {version}", flush=True)
            total_start = time.perf_counter()
            for study, cfg, expected in STUDIES:
                sim_s, post_s = run_study(version, study, cfg, expected, rep, args.workers, env)
                rows.append((rep, version, study, sim_s, post_s, sim_s + post_s))
            total = time.perf_counter() - total_start
            totals[version].append(total)
            print(f"  FULL STUDY {total:.3f}s", flush=True)
            rows.append((rep, version, "FULL_STUDY", "", "", total))

    raw_output = ROOT / "full_study_benchmark.csv"
    with raw_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["repeat", "version", "study", "simulation_seconds", "postprocessing_seconds", "total_seconds"])
        writer.writerows(rows)

    medians = {v: median(totals[v]) for v in VERSIONS}
    print("\nWhole-study medians (same 3 experiments, analysis and figures):")
    for version in VERSIONS:
        print(f"  {version}: {medians[version]:.3f}s; V0 / {version} = {medians['V0']/medians[version]:.2f}x", flush=True)
    summary_output = ROOT / "full_study_summary.csv"
    with summary_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "median_full_study_seconds", "speedup_vs_original"])
        writer.writerows((v, medians[v], medians["V0"] / medians[v]) for v in VERSIONS)

    print("\nChecking saved trajectories from first repetition:")
    checks = []
    for study, _, _ in STUDIES:
        def filepath(v):
            return ROOT / "full_study" / v / "run_01" / study / "trajectories.csv"
        checks.extend(compare_results(filepath("V0"), filepath("V1"), study, "V0 vs V1"))
        checks.extend(compare_results(filepath("V0"), filepath("V2"), study, "V0 vs V2"))
        checks.extend(compare_results(filepath("V1"), filepath("V2"), study, "V1 vs V2"))
    comparison_output = ROOT / "full_study_numerical_comparison.csv"
    with comparison_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study", "comparison", "metric", "rows", "max_abs_difference", "flags_at_diagnostic_tolerance"])
        writer.writerows(checks)
    print("\nNote: V0/V1 differences in near-zero bias mismatch at default RK45 tolerances are diagnostics,")
    print("not automatically treated as equality; the tight-tolerance investigation is separate.")
    make_figure(medians)
    print("Saved", raw_output, summary_output, comparison_output, sep="\n  ")
    print("No files under original results/raw/, results/width256/ or results/fixed_alpha/ were changed.")


if __name__ == "__main__":
    main()
