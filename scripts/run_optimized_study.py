"""Run a full Project 2 study with the low-rank GF solver, serial or parallel.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from scripts.benchmark_parallel import run_study

COLUMNS = [
    "dimension", "student_width", "replicate", "beta", "alpha", "time",
    "population_risk", "bias_mismatch",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Project 2 JSON configuration")
    parser.add_argument("--output", required=True, help="Output trajectories CSV")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    start = time.perf_counter()
    records = run_study(cfg, workers=args.workers)
    duration = time.perf_counter() - start
    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    df.insert(0, "study_id", cfg["study_id"])
    df["alpha"] = df["alpha"].round(10)
    # Keep the original src.simulation CSV column order.
    df = df[[
        "study_id", "dimension", "beta", "alpha", "student_width",
        "replicate", "time", "population_risk", "bias_mismatch",
    ]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    metadata = {
        "config": cfg,
        "solver": "low_rank_population_gf",
        "workers": args.workers,
        "elapsed_simulation_seconds": duration,
        "rng": "SeedSequence(master_seed, dimension, student_width, replicate)",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }
    with (output.parent / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(df)} timepoint rows in {output}; workers={args.workers}, "
          f"runtime={duration:.3f}s")


if __name__ == "__main__":
    main()
