from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import warnings

import numpy as np

# Allow `python3 scripts/stability_check.py` from the repository root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dgps import make_diagonal_model, initialize_student, initialize_bias
from src.methods import solve_paper_population_gf


@dataclass(frozen=True)
class Case:
    study: str
    d: int
    width: int
    alpha: float
    beta: float
    end_time: float
    n_times: int


# Every distinct Project 2 scenario: six main, two additional fixed-alpha,
# and three d=256 width scenarios.  The shared fixed-alpha condition is
# already covered by the main study.
CASES = (
    Case("main", 32, 8, 0.60, 0.10, 80, 75),
    Case("main", 32, 8, 0.35, 0.35, 80, 75),
    Case("main", 32, 8, 0.10, 0.60, 80, 75),
    Case("main", 32, 16, 0.60, 0.10, 80, 75),
    Case("main", 32, 16, 0.35, 0.35, 80, 75),
    Case("main", 32, 16, 0.10, 0.60, 80, 75),
    Case("fixed_alpha", 32, 16, 0.35, 0.25, 80, 75),
    Case("fixed_alpha", 32, 16, 0.35, 0.55, 80, 75),
    Case("width256", 256, 32, 0.35, 0.35, 160, 101),
    Case("width256", 256, 64, 0.35, 0.35, 160, 101),
    Case("width256", 256, 128, 0.35, 0.35, 160, 101),
)

# These are two representative cases, not all cases.
TIGHT_CASES = {("main", 32, 16, 0.35), ("width256", 256, 64, 0.35)}

COLUMNS = (
    "study", "d", "width", "alpha", "beta", "replicate", "T",
    "status", "elapsed_seconds", "warning_count", "warning_details",
    "nonfinite_count", "max_risk_increase", "monotonicity_violations",
    "rank_floor", "minimum_risk_minus_floor", "floor_violations",
    "tight_max_risk_difference", "tight_max_mismatch_difference", "error",
)


def solve_and_capture(W0, b0, lam, B, times, *, rtol, atol):
    """Run the unchanged baseline solver, capturing all Python warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(over="warn", divide="warn", invalid="warn", under="ignore"):
            t, risk, mismatch = solve_paper_population_gf(
                W0, b0, lam, B, times,
                delta=0.05, rtol=rtol, atol=atol,
            )
    descriptions = [f"{w.category.__name__}: {w.message}" for w in caught]
    return t, risk, mismatch, descriptions


def run_case(case: Case, rep: int, *, tight: bool) -> dict:
    row = {name: "" for name in COLUMNS}
    row.update(study=case.study, d=case.d, width=case.width,
               alpha=case.alpha, beta=case.beta, replicate=rep, T=case.end_time)
    rng = np.random.default_rng(np.random.SeedSequence(
        [20261019, case.d, case.width, rep]
    ))
    W0 = initialize_student(rng, case.d, case.width, "paper")
    b0 = initialize_bias(rng, "normal")
    lam, B = make_diagonal_model(case.d, case.beta, case.alpha)
    times = np.r_[0.0, np.geomspace(0.02, case.end_time, case.n_times - 1)]

    start = time.perf_counter()
    try:
        t, risk, mismatch, warn = solve_and_capture(
            W0, b0, lam, B, times, rtol=1e-6, atol=1e-8,
        )
        row["warning_count"] = len(warn)
        row["warning_details"] = " | ".join(warn)
        if len(t) != len(times) or not np.allclose(t, times, rtol=0, atol=1e-12):
            raise ValueError("Solver did not return the requested time grid")

        nonfinite = int((~np.isfinite(risk)).sum() + (~np.isfinite(mismatch)).sum())
        row["nonfinite_count"] = nonfinite
        if nonfinite:
            row["status"] = "FAIL (nonfinite)"
        else:
            # The baseline teacher spectrum is ordered, so the rank-r floor
            # consists of the tail of Sigma * B (B is diagonal).
            floor = float(np.sum(np.square(lam[case.width:] * np.diag(B)[case.width:])))
            drisk = np.diff(risk)
            scale = max(1.0, float(abs(risk[0])))
            # Floating-point / ODE tolerance: diagnostic, not an exact identity.
            tol = 1e-8 + 2e-6 * scale
            row["max_risk_increase"] = float(max(0.0, np.max(drisk)))
            row["monotonicity_violations"] = int(np.count_nonzero(drisk > tol))
            row["rank_floor"] = floor
            row["minimum_risk_minus_floor"] = float(np.min(risk - floor))
            row["floor_violations"] = int(np.count_nonzero(risk < floor - tol))
            row["status"] = (
                "PASS" if not warn and not row["monotonicity_violations"]
                and not row["floor_violations"] else "CHECK"
            )

            if tight and rep == 0 and (
                case.study, case.d, case.width, case.beta
            ) in TIGHT_CASES:
                _, risk_tight, mismatch_tight, tight_warn = solve_and_capture(
                    W0, b0, lam, B, times, rtol=1e-8, atol=1e-10,
                )
                row["tight_max_risk_difference"] = float(np.max(np.abs(risk - risk_tight)))
                row["tight_max_mismatch_difference"] = float(
                    np.max(np.abs(mismatch - mismatch_tight))
                )
                if tight_warn:
                    row["warning_count"] += len(tight_warn)
                    row["warning_details"] += " | tight: " + " | ".join(tight_warn)
                    row["status"] = "CHECK"
    except Exception as exc:
        row["status"] = "FAIL (exception)"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["elapsed_seconds"] = round(time.perf_counter() - start, 4)
    return row


def fmt(value, decimals=3):
    return f"{value:.{decimals}g}" if isinstance(value, (float, np.floating)) else str(value)


def write_summary(path: Path, rows: list[dict], *, tight: bool, reps: int) -> None:
    n_fail = sum(str(row["status"]).startswith("FAIL") for row in rows)
    n_check = sum(row["status"] == "CHECK" for row in rows)
    n_warn = sum(int(row["warning_count"] or 0) for row in rows)
    max_increase = max((float(row["max_risk_increase"])
                        for row in rows if row["max_risk_increase"] != ""), default=float("nan"))
    min_gap = min((float(row["minimum_risk_minus_floor"])
                   for row in rows if row["minimum_risk_minus_floor"] != ""), default=float("nan"))
    lines = [
        "# Baseline stability check", "",
        "Original Project 2 solver: RK45, rtol=1e-6, atol=1e-8.",
        f"Tested {len(rows)} trajectories covering {len(CASES)} parameter conditions "
        f"({reps} initialization(s) per condition).", "",
        f"- Solver/finite-value failures: **{n_fail}**",
        f"- Cases requiring review (warnings or risk diagnostics): **{n_check}**",
        f"- Captured Python warning messages: **{n_warn}**",
        f"- Largest positive change between adjacent saved risks: **{max_increase:.6g}**",
        f"- Smallest observed risk minus theoretical floor: **{min_gap:.6g}**", "",
        "| Study | d | Width | beta | Rep | Status | Warnings | Risk increases* | Floor violations* | Tight risk diff |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['study']} | {r['d']} | {r['width']} | {r['beta']} | {r['replicate']} "
            f"| {r['status']} | {r['warning_count']} | {r['monotonicity_violations']} "
            f"| {r['floor_violations']} | {fmt(r['tight_max_risk_difference'])} |"
        )
    lines.extend([
        "", "*For the two risk diagnostics, a step is flagged only when its size exceeds "
        "1e-8 + 2e-6 * max(1, |initial risk|). The raw changes are in the CSV.",
        "", "The script checks finite risk and centering-mismatch values at saved time points. "
        "The original solver raises an exception on unsuccessful integration; this script "
        "does not inspect every intermediate internal solver state.",
        "", "Warning counts reflect captured Python warnings for these runs, not all possible "
        "numerical conditions. A small difference from the tighter-tolerance solution is "
        "a sensitivity diagnostic, not a proof of solution accuracy.",
    ])
    if tight:
        lines.append("\nTight tolerance (rtol=1e-8, atol=1e-10) was tested for one "
                     "d=32/width=16 and one d=256/width=64 trajectory.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/performance/stability_baseline.csv")
    parser.add_argument("--summary", default="results/performance/stability_baseline_summary.md")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--tight", action="store_true", help="Recompute two cases with stricter RK45 tolerances")
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    out = Path(args.output)
    summary = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASES:
        for rep in range(args.replicates):
            result = run_case(case, rep, tight=args.tight)
            rows.append(result)
            print(f"{case.study:11s} d={case.d:3d} width={case.width:3d} "
                  f"beta={case.beta:.2f} rep={rep}: {result['status']} "
                  f"(warnings={result['warning_count'] or 0}, "
                  f"{result['elapsed_seconds']:.2f}s)", flush=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(summary, rows, tight=args.tight, reps=args.replicates)
    n_bad = sum(r["status"] != "PASS" for r in rows)
    print(f"\nFinished: {len(rows)} cases, {n_bad} flagged.\nResults: {out}\nSummary: {summary}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
