# Baseline Performance

I used the original Project 2 code as the baseline, before changing the gradient-flow implementation. The code is preserved in the `project2-baseline` branch (commit `95593a993815e8147d0fb53ab1284c3731964964`). All runs below use RK45 with `rtol=1e-6` and `atol=1e-8`.

## 1. Runtime

I ran the full Project 2 workflow on my Mac:

```bash
{ time make all; } 2>&1 | tee results/performance/baseline.log
```

| Experiment | Settings | Trajectories | Simulation time |
| --- | --- | ---: | ---: |
| Main | `d=32`, widths 8 and 16, three values of `beta`, `T=80` | 60 | 5.08 s |
| Width follow-up | `d=256`, widths 32, 64 and 128, `T=160` | 15 | 33.67 s |
| Fixed-alpha follow-up | `d=32`, width 16, three values of `beta`, `T=80` | 30 | 2.60 s |
| **Full `make all`** | **Simulation, analysis and figures** | **105** | **47.615 s total** |

The simulation stages took 41.35 s in total; the remaining time was spent on startup, analysis, plotting and other overhead. The `d=256` experiment alone took 33.67 s, about 71% of the full runtime. The 105 trajectories include 10 repeated conditions shared by the main and fixed-alpha experiments. This is one baseline timing run, not an average over repeated timings. The existing test suite passed (14 tests).

## 2. Profiling

I used `cProfile` on one representative trajectory: `d=256`, `r_s=64`, `alpha=beta=0.35`, `T=160`, one initialization, with the same RK45 tolerances.

```bash
python3 -m cProfile -o results/performance/baseline.prof -m src.simulation \
  --config results/performance/profile_config.json \
  --output results/performance/profile_trajectories.csv
```

| Function | Calls | Cumulative time |
| --- | ---: | ---: |
| `solve_paper_population_gf` | 1 | 2.244 s |
| `solve_ivp` | 1 | 2.225 s |
| `rhs` | 9,398 | 2.023 s |
| `population_risk_and_gradients` | 9,499 | 1.944 s |

The simulation took 2.245 s under the profiler (2.932 s including imports). About **86.6%** of the simulation time was spent in `population_risk_and_gradients`. It was called 9,398 times by the ODE solver and another 101 times to evaluate risk at the saved time points.

This is the main bottleneck. Each call forms `G = W @ W.T / r_s` and a weighted `d x d` residual matrix. The RHS also computes the risk even though the ODE solver only needs gradients. These are the operations I will focus on optimizing. The profiled times include profiling overhead, so I will use separate, unprofiled runs for speed comparisons.

## 3. Computational complexity

Let `W` have shape `d x r_s`. In the current implementation:

- `W @ W.T` costs `O(d^2 r_s)`.
- Multiplying the weighted `d x d` residual by `W` also costs `O(d^2 r_s)`.
- Forming and weighting the residual costs `O(d^2)`.

Therefore, **one risk/gradient evaluation costs `O(d^2 r_s)`**, with `O(d^2)` temporary matrix storage. If the solver uses `N_f` RHS evaluations and saves `N_t` time points, the main computation is approximately `O((N_f+N_t)d^2 r_s)`. The actual runtime also depends on the number of solver evaluations, which can change with the dimension and numerical tolerances.

This is a theoretical complexity estimate for the original algorithm. I will compare the scaling empirically in the optimization benchmarks.

## 4. Numerical warnings and convergence

No numerical warnings, exceptions or solver failures appeared in the console output for the **105 trajectories** in `make all` (the conditions in the first table), or for the **one profiled trajectory** at `d=256`, `r_s=64`, `T=160`. The existing 14 tests also passed.

Thus the *observed* warning/failure counts were **0/105** for the full study and **0/1** for the profiling run. These counts are based on console output and successful completion, not on a separate automated warning log. They do not establish that the numerical error is negligible. I will check additional conditions, solver tolerances and risk behavior in `make stability-check`.

The local baseline evidence is saved under `results/performance/`: `baseline.log`, `profile_config.json`, `baseline.prof`, `profile_summary.txt` and `profile_trajectories.csv`.
