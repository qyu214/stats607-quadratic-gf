# Quadratic Network Gradient-Flow Simulation

STATS 607 — Project 2 (Option B: original simulation study)

This project studies how covariance decay and student width affect learning in a quadratic teacher–student network. I simulate population gradient flow and compare population excess risk with the best rank-constrained approximation risk. The setup comes from my research, but this is not a reproduction of the original paper's online-SGD experiments.

## Setup and running

The project uses Python 3.13. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
make all
```

`make all` runs the three experiments, writes raw trajectories and summary tables, and generates figures. To run either follow-up separately:

```bash
bash scripts/run_d256_width.sh
bash scripts/run_fixed_alpha.sh
```

In my local run, the simulation stages took 4.91 seconds, 33.37 seconds, and 2.62 seconds, respectively (about 41 seconds combined). Data processing and plotting take additional time.

## Experiments

| Experiment | Setup | Runs |
|---|---|---:|
| Main | $d=32$, $\alpha+\beta=0.7$, $\beta\in\{0.1,0.35,0.6\}$, widths 8 and 16 | 60 |
| Width follow-up | $d=256$, $\alpha=\beta=0.35$, widths 32, 64, and 128 | 15 |
| Fixed-$\alpha$ follow-up | $d=32$, $\alpha=0.35$, $\beta\in\{0.25,0.35,0.55\}$, width 16 | 30 |

The main and fixed-$\alpha$ studies share ten baseline trajectories, so the 105 executed trajectories are not 105 independent runs. See [ADEMP.md](ADEMP.md) for the model, parameter table, and numerical method.

## Results

The main data, summaries, and figures are under `results/`. The width follow-up is saved under `results/width256/`, and the fixed-$\alpha$ follow-up under `results/fixed_alpha/`. The main figures include `results/figures/gf_risk_facets.pdf` and `results/figures/diagnostic_bias_tracking.pdf`.

In the settings studied, wider students achieved lower absolute risk but sometimes had larger gaps above their own rank floors at the chosen final time. See [ANALYSIS.md](ANALYSIS.md) for the results, design choices, and limitations.

## AI Assistance

I used ChatGPT to help implement and debug the simulation code, set up the experimental workflow, and draft and revise the documentation. I ran the simulations, examined the results, and made the final decisions about the experimental design and interpretation.