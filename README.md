# Quadratic Network Gradient-Flow Simulation

STATS 607 — Project 2 (Simulation Study) and Project 3 (Performance Optimization)

This project studies how covariance decay and student width affect learning in a quadratic teacher–student network. I simulate population gradient flow and compare population excess risk with the best rank-constrained approximation risk. The setup is based on my research, using population gradient flow rather than online SGD.

## Setup

The project uses Python 3.13. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
```

## Project 2: Simulation Study

Run the original simulation study with:

```bash
make all
```

This runs three experiments, saves the trajectories and summary tables, and generates figures.

| Experiment | Setup | Runs |
|---|---|---:|
| Main | $d=32$, $\alpha+\beta=0.7$, $\beta\in\{0.1,0.35,0.6\}$, widths 8 and 16 | 60 |
| Width follow-up | $d=256$, $\alpha=\beta=0.35$, widths 32, 64, and 128 | 15 |
| Fixed-$\alpha$ follow-up | $d=32$, $\alpha=0.35$, $\beta\in\{0.25,0.35,0.55\}$, width 16 | 30 |

The main and fixed-$\alpha$ experiments share ten trajectories, so the 105 executed trajectories are not all independent.

See [ADEMP.md](ADEMP.md) for the simulation design and [ANALYSIS.md](ANALYSIS.md) for the results and discussion.

The results and figures are saved under `results/`, with the two follow-up experiments in `results/width256/` and `results/fixed_alpha/`.

## Project 3: Performance Optimization

For Project 3, I optimized the gradient-flow simulation in two ways:

- **Low-rank computation:** Rewrote the gradient-flow calculations to avoid constructing large $d\times d$ intermediate matrices.
- **Parallelization:** Used multiple processes to run independent simulation trajectories in parallel.

The original solver is in `src/methods.py`, and the low-rank implementation is in `src/methods_optimized.py`.

For a quick check of the optimized solver, run `make demo` (one small trajectory).

The performance experiments can be run with:

```bash
make profile
make complexity
make benchmark
make parallel
make stability-check
make test
```

The full-study benchmark compares the original serial implementation, the low-rank serial implementation, and the low-rank implementation with four workers. The median runtimes were 46.678, 23.051, and 13.834 seconds, respectively, giving an overall speedup of 3.37×.

Benchmark data and figures are saved under `results/performance/`.

See [BASELINE.md](docs/BASELINE.md) for the original performance measurements and [OPTIMIZATION.md](docs/OPTIMIZATION.md) for the optimizations, benchmarks, and numerical checks.


## AI Assistance

I used ChatGPT to help implement and debug the simulation and optimization code, develop the testing and benchmarking workflow, and revise the documentation.