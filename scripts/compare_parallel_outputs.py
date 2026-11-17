"""Compare saved original and low-rank parallel trajectories."""
from pathlib import Path

import numpy as np
import pandas as pd

original_path = Path("results/width256/raw/trajectories.csv")
optimized_path = Path("results/performance/parallel_width256/trajectories.csv")
output_path = Path("results/performance/parallel_comparison.csv")

keys = ["study_id", "dimension", "beta", "alpha", "student_width", "replicate", "time"]
old = pd.read_csv(original_path).sort_values(keys).reset_index(drop=True)
new = pd.read_csv(optimized_path).sort_values(keys).reset_index(drop=True)
pd.testing.assert_frame_equal(old[keys], new[keys])

report = old[keys].copy()
for column in ("population_risk", "bias_mismatch"):
    report[f"original_{column}"] = old[column]
    report[f"optimized_{column}"] = new[column]
    report[f"abs_diff_{column}"] = np.abs(old[column] - new[column])

risk_diff = report["abs_diff_population_risk"].to_numpy()
e_diff = report["abs_diff_bias_mismatch"].to_numpy()
risk_tolerance = 1e-7 + 1e-6 * np.abs(new["population_risk"].to_numpy())
e_tolerance = 1e-7 + 1e-6 * np.abs(new["bias_mismatch"].to_numpy())
risk_flags = risk_diff > risk_tolerance
e_flags = e_diff > e_tolerance

print(f"Matched rows: {len(report)}")
print(f"Population risk: max abs diff = {risk_diff.max():.3e}; flags = {risk_flags.sum()}")
print(f"Bias mismatch:   max abs diff = {e_diff.max():.3e}; flags = {e_flags.sum()}")
print("\nLargest bias mismatch discrepancies:")
columns = ["student_width", "replicate", "beta", "time",
           "original_bias_mismatch", "optimized_bias_mismatch", "abs_diff_bias_mismatch"]
print(report.nlargest(8, "abs_diff_bias_mismatch")[columns].to_string(index=False))

output_path.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(output_path, index=False)
print(f"\nSaved detailed comparison: {output_path}")
