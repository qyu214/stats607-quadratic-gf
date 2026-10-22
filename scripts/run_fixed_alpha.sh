#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
mkdir -p results/fixed_alpha/raw results/fixed_alpha/figures
"$PYTHON" -m src.simulation --config configs/fixed_alpha_sweep.json --output results/fixed_alpha/raw/trajectories.csv
"$PYTHON" -m src.analyze --raw results/fixed_alpha/raw/trajectories.csv --output results/fixed_alpha/summary.csv --endpoints results/fixed_alpha/endpoints.csv
"$PYTHON" -m src.fixed_alpha_figures --summary results/fixed_alpha/summary.csv --endpoints results/fixed_alpha/endpoints.csv --out results/fixed_alpha/figures
