# Project 3 — Baseline Performance

## Scope and preservation

This document records performance of the **unchanged Project 2 implementation** on 2026-11-14, before Project 3 optimization. The reference implementation is the repository's original `main` snapshot (commit `95593a993815e8147d0fb53ab1284c3731964964`), also preserved in the `project2-baseline` branch.

The original simulation uses SciPy `solve_ivp` with RK45 and tolerances `rtol=1e-6`, `atol=1e-8`. Three studies are launched by `make all`: the main `d=32` study, the `d=256` width follow-up, and the fixed-alpha `d=32` follow-up. There are 105 executed trajectories, including 10 reused conditions across the main and fixed-alpha studies.

## Whole-study runtime (baseline)

Measured locally on a Mac using a terminal, after `make test` passed (14 tests):

```shell
{ time make all; } 2>&1 | tee results/performance/baseline.log
```

| Component | Reported elapsed time |
| --- | ---: |
| Main study (`d=32`, 60 trajectories) | 5.08 s |
| Width follow-up (`d=256`, 15 trajectories) | 33.67 s |
| Fixed-alpha follow-up (`d=32`, 30 trajectories) | 2.60 s |
| **Full `make all` including analysis and figures** | **47.615 s** |

The three simulation-stage timings add to 41.35 s; the full wall-clock time also includes startup, analysis, plotting, and command overhead. The `d=256` stage accounts for approximately 70.7% of the measured whole-study runtime. These are initial single-run measurements, not repeated benchmark estimates.

## Representative CPU profiling

To avoid profiling the entire study, the original `d=256` experiment was run with `r_s=64`, one random initialization, `alpha=beta=0.35`, `T=160`, and the original RK45 tolerances:

```shell
python3 -m cProfile -o results/performance/baseline.prof -m src.simulation \
  --config results/performance/profile_config.json \
  --output results/performance/profile_trajectories.csv

python3 -c 'import pstats; pstats.Stats("results/performance/baseline.prof").strip_dirs().sort_stats("cumtime").print_stats(25)'
```

The profiled process took **2.932 s** including module imports. The `simulate` function took **2.245 s** and `solve_ivp` took **2.225 s** in cumulative profiled time.

| Function | Calls | Cumulative profiled time |
| --- | ---: | ---: |
| `solve_paper_population_gf` | 1 | 2.244 s |
| `solve_ivp` | 1 | 2.225 s |
| `rhs` | 9,398 | 2.023 s |
| `population_risk_and_gradients` | 9,499 | 1.944 s |

The risk-and-gradient routine accounted for approximately **86.6% of simulation time** (1.944 / 2.245). Its 9,499 invocations comprise 9,398 RHS evaluations and 101 post-solve risk evaluations at requested times. The RK45 solver completed 1,481 steps in this profile.

**Identified bottleneck:** Each `population_risk_and_gradients` call constructs `G = W @ W.T / r_s` and a dense `d x d` weighted residual, then multiplies by `W`. The RHS also computes a risk value that the integrator does not need. This motivates a mathematically equivalent low-rank reformulation and separation of gradient evaluation from risk evaluation.

Profiler timings include instrumentation overhead and must not be used as the speed denominator for final benchmarks.

## Computational complexity (baseline)

For `W` of shape `d x r_s`:

- Forming `W @ W.T`: `O(d^2 r_s)` arithmetic, `O(d^2)` temporary storage.
- Weighting the dense residual: `O(d^2)` arithmetic and storage.
- Multiplying the weighted residual by `W`: `O(d^2 r_s)` arithmetic.
- Remaining scalar/vector operations: at most `O(d r_s + d^2)` per evaluation.

Thus, for this implementation, one population risk/gradient evaluation costs **`O(d^2 r_s)`** for `r_s >= 1`, with a dense **`O(d^2)`** intermediate. For `N_f` RHS evaluations and `N_o` recorded time points, the leading compute work is `O((N_f+N_o)d^2 r_s)`. The full solver runtime also depends on how `N_f` varies with dimension, time horizon, and tolerance, and on numerical-library performance.

An empirical dimension-scaling benchmark will be added using the same experimental settings for baseline and optimized versions.

## Numerical warnings and convergence issues

- **Observed in the provided local run:** `make test` passed all 14 tests; the full `make all` completed all 105 trajectories and produced outputs; the profiled single trajectory completed. No warnings or exceptions were visible in the supplied console output.
- **Limitation:** These commands did not systematically count or classify numerical warnings. Therefore, it is **not** yet justified to claim a zero failure rate under all conditions.
- **Planned check:** run a dedicated stability-check across representative dimensions, widths, and solver tolerances; record warnings, non-finite results, solver failures, and discrepancies in the monotone risk and rank-floor diagnostics, with counts and test conditions.

## Evidence and reproducibility

Local evidence was generated under `results/performance/`:

- `baseline.log` — full-study console output and wall-clock timing
- `profile_config.json` — one-replicate profiling condition
- `baseline.prof` — binary `cProfile` output
- `profile_summary.txt` — human-readable top cumulative-time functions
- `profile_trajectories.csv` — profiled run's risk trajectory

These outputs are local files unless explicitly committed. Do not mix baseline profiler timings with unprofiled benchmark timings. The Project 3 optimization report will separately quantify measured improvements, numerical agreement, and trade-offs.
