"""Run either GF solver for memory measurements."""

import argparse
import sys
from src import simulation
from src.methods_optimized import solve_paper_population_gf_optimized

parser = argparse.ArgumentParser()
parser.add_argument("method", choices=["baseline", "optimized"])
args = parser.parse_args()

if args.method == "optimized":
    simulation.solve_paper_population_gf = solve_paper_population_gf_optimized

output = f"results/performance/memory_compare/{args.method}/trajectories.csv"

sys.argv = [
    "simulation",
    "--config", "results/performance/profile_config.json",
    "--output", output,
]
simulation.main()
