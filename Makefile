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
.PHONY: width256

width256:
	bash scripts/run_d256_width.sh

.PHONY: fixed-alpha

fixed-alpha:
	bash scripts/run_fixed_alpha.sh
