"""Plot parallel speedup from the benchmark results."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.read_csv("results/performance/parallel_benchmark.csv")

times = data.groupby("workers")["seconds"].median().sort_index()
workers = times.index.to_numpy()
speedup = times.loc[1] / times.to_numpy()

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.plot(workers, speedup, "o-", linewidth=2, label="Measured")
ax.plot(workers, workers, "--", label="Ideal linear speedup")

ax.set_xlabel("Number of workers")
ax.set_ylabel("Speedup")
ax.set_title("Parallel Speedup (d=256, 15 trajectories)")
ax.set_xticks(workers)
ax.set_ylim(0, max(workers) * 1.1)
ax.grid(alpha=0.2)
ax.legend()

fig.tight_layout()

out = Path("results/performance/parallel_speedup.png")
fig.savefig(out, dpi=180)
plt.close(fig)

print(f"Saved {out}")
