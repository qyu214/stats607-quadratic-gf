# Baseline Performance

I used the Project 2 code as the baseline, before making any optimizations. A copy is kept in the `project2-baseline` branch. The simulations use SciPy's RK45 solver with `rtol=1e-6` and `atol=1e-8`.

## 1. Runtime

I ran `make all` on my Mac and timed the complete workflow, including simulation, analysis, and plotting. The existing tests also passed (`14 passed`).

| Experiment | Setting | Trajectories | Simulation time |
| --- | --- | ---: | ---: |
| Main | `d=32`, widths 8 and 16, three values of beta, `T=80` | 60 | 5.08 s |
| Width follow-up | `d=256`, widths 32, 64, and 128, `T=160` | 15 | 33.67 s |
| Fixed-alpha follow-up | `d=32`, width 16, three values of beta, `T=80` | 30 | 2.60 s |
| **Full `make all`** | **Including analysis and figures** | **105** | **47.615 s** |

The simulations themselves took 41.35 s. The `d=256` experiment accounted for about 71% of the total runtime, despite having only 15 trajectories. Ten trajectories were repeated across the main and fixed-alpha studies. These timings are from one run, not repeated benchmarks.

Command: `{ time make all; } 2>&1 | tee results/performance/baseline.log`.

## 2. Runtime profiling

I profiled one `d=256`, `r_s=64` trajectory with `alpha=beta=0.35` and `T=160`, using the same RK45 settings.

| Function | Calls | Cumulative time |
| --- | ---: | ---: |
| `solve_paper_population_gf` | 1 | 2.244 s |
| `solve_ivp` | 1 | 2.225 s |
| `rhs` | 9,398 | 2.023 s |
| `population_risk_and_gradients` | 9,499 | 1.944 s |

The simulation took 2.245 s under `cProfile`; about **86.6%** was spent in `population_risk_and_gradients`. The solver called it 9,398 times, with another 101 calls to evaluate the risk at saved time points.

Each call forms `W @ W.T / r_s` and a dense weighted residual matrix, then multiplies that matrix by `W`. The RHS also calculates the risk even though the solver only needs gradients. I will focus on these calculations in the optimized version. Since profiling adds overhead, I will use separate timings to measure speedup.

Profiler summary: `results/performance/profile_summary.txt`. The raw `baseline.prof` file can be regenerated with `make profile`.

## 3. Computational complexity

For a student matrix $W \in \mathbb{R}^{d\times r_s}$, forming $WW^\top$ costs $O(d^2r_s)$. Forming and weighting the dense residual costs $O(d^2)$, and multiplying it by $W$ costs another $O(d^2r_s)$. Thus, each risk/gradient evaluation costs **$O(d^2r_s)$**, with **$O(d^2)$** intermediate storage.

The initial $W$ takes $O(dr_s)$ work to generate, while constructing the dense diagonal teacher matrix takes $O(d^2)$. If the solver makes $N_f$ RHS evaluations and the risk is evaluated at $N_t$ saved times, the main cost is $O((N_f+N_t)d^2r_s)$. The number of evaluations can itself depend on the dimension, time interval, and solver tolerances. For fixed $r_s$, the per-evaluation cost is quadratic in $d$; for $r_s\propto d$, it is cubic.

## 4. Memory profiling

I used macOS `/usr/bin/time -l` on the same `d=256`, `r_s=64`, `T=160` trajectory.

| Metric | Baseline |
| --- | ---: |
| Maximum resident set size | 131.1 MiB |
| Peak memory footprint | 104.7 MiB |

These are measurements for the whole Python process, including SciPy and other libraries. The log is in `results/performance/memory_baseline.log`. I will use the same setup for the optimized version.

## 5. Numerical stability

I checked 11 parameter settings: six from the main experiment, two additional fixed-alpha settings, and three widths at `d=256`. Each used one initialization. The script checked solver failures, Python warnings, non-finite risk or bias-mismatch values, risk increases, and violations of the theoretical rank floor.

All **11/11** runs passed. There were **0** solver failures, **0** captured warnings, **0** non-finite outputs, and **0** flagged risk increases or floor violations. The largest positive change between saved risks was 0, and the smallest observed risk-minus-floor gap was `2.00e-5`. The full 105-trajectory `make all` run also finished without visible warnings or exceptions.

For two cases, I repeated the calculation with tighter RK45 tolerances (`rtol=1e-8`, `atol=1e-10`). The maximum saved-risk differences were `1.48e-6` (`d=32`, `r_s=16`) and `6.64e-6` (`d=256`, `r_s=64`). These are tolerance-sensitivity comparisons, not exact error bounds. The risk and floor checks flag differences only above `1e-8 + 2e-6 * max(1, |initial risk|)`; the checks concern saved outputs rather than every internal solver state.

Results: `results/performance/stability_baseline.csv` and `results/performance/stability_baseline_summary.md`.
