PYTHON ?= python3
.PHONY: all simulate analyze figures clean test

all: simulate analyze figures width256 fixed-alpha

simulate:
	$(PYTHON) -m src.simulation --config config.json --output results/raw/trajectories.csv

analyze:
	$(PYTHON) -m src.analyze --raw results/raw/trajectories.csv --output results/summary.csv --endpoints results/endpoints.csv

figures:
	$(PYTHON) -m src.figures --raw results/raw/trajectories.csv --summary results/summary.csv --endpoints results/endpoints.csv --out results/figures

test:
	$(PYTHON) -m pytest -q

clean:
	rm -f results/raw/trajectories.csv results/raw/run_metadata.json results/summary.csv results/endpoints.csv
	rm -f results/figures/*.png results/figures/*.pdf
	rm -f results/width256/raw/*.csv results/width256/raw/*.json
	rm -f results/width256/summary.csv results/width256/checkpoints.csv
	rm -f results/width256/figures/*.png results/width256/figures/*.pdf
	rm -f results/fixed_alpha/raw/*.csv results/fixed_alpha/raw/*.json
	rm -f results/fixed_alpha/summary.csv results/fixed_alpha/endpoints.csv
	rm -f results/fixed_alpha/figures/*.png results/fixed_alpha/figures/*.pdf

.PHONY: width256

width256:
	bash scripts/run_d256_width.sh

.PHONY: fixed-alpha

fixed-alpha:
	bash scripts/run_fixed_alpha.sh

# Project 3 performance targets
WORKERS ?= 4
REPEATS ?= 3
THREAD_ENV = OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.PHONY: profile complexity benchmark parallel stability-check

profile:
	mkdir -p results/performance
	$(PYTHON) -c 'import json; from pathlib import Path; c=json.loads(Path("configs/d256_width_pilot.json").read_text()); c["student_widths"]=[64]; c["n_replicates"]=1; Path("results/performance/profile_config.json").write_text(json.dumps(c,indent=2))'
	$(THREAD_ENV) $(PYTHON) -m cProfile -o results/performance/baseline.prof -m src.simulation --config results/performance/profile_config.json --output results/performance/profile_trajectories.csv
	$(THREAD_ENV) $(PYTHON) -m cProfile -o results/performance/optimized.prof -m scripts.memory_algorithmic optimized

complexity:
	$(THREAD_ENV) $(PYTHON) -m scripts.benchmark_complexity

benchmark:
	$(THREAD_ENV) $(PYTHON) -m scripts.benchmark_full_study --repeats $(REPEATS) --workers $(WORKERS)

parallel:
	$(THREAD_ENV) $(PYTHON) -m scripts.run_optimized_study --config configs/d256_width_pilot.json --workers $(WORKERS) --output results/performance/parallel_width256/trajectories.csv

stability-check:
	$(PYTHON) -m scripts.stability_check --tight --output results/performance/stability_baseline.csv --summary results/performance/stability_baseline_summary.md
	$(PYTHON) -m scripts.stability_check_optimized --tight --output results/performance/stability_optimized.csv --summary results/performance/stability_optimized_summary.md

# A quick, one-trajectory smoke test. Does not rerun the full benchmark.
.PHONY: demo
demo:
	$(THREAD_ENV) $(PYTHON) -m scripts.run_optimized_study --config configs/demo.json --workers 1 --output results/performance/demo/trajectories.csv
	$(PYTHON) -c 'import pandas as pd; p="results/performance/demo/trajectories.csv"; d=pd.read_csv(p); assert len(d)==21 and d.population_risk.notna().all(); print(f"Demo OK: {len(d)} saved time points; final risk = {d.population_risk.iloc[-1]:.6g}")'
