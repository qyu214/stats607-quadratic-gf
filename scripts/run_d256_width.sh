#!/usr/bin/env bash
# Run independently; do not overwrite the main Project 2 results.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
"$PYTHON" -m src.simulation --config configs/d256_width_pilot.json --output results/width256/raw/trajectories.csv
"$PYTHON" -m src.width_pilot_figures --raw results/width256/raw/trajectories.csv --out results/width256
