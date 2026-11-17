"""Parallel execution should not change seeded GF results."""
import numpy as np

from scripts.benchmark_parallel import run_study


def test_parallel_reproducibility():
    cfg = {
        "dimension": 16,
        "student_widths": [4],
        "covariance_exponents": [0.2, 0.35],
        "alpha_plus_beta": 0.7,
        "n_replicates": 2,
        "master_seed": 20261019,
        "initialization_std": "paper",
        "bias_initialization": "normal",
        "delta": 0.05,
        "time_grid": "log",
        "time_start": 0.02,
        "time_max": 5.0,
        "n_time_points": 12,
        "rtol": 1e-6,
        "atol": 1e-8,
    }
    serial = np.asarray(run_study(cfg, workers=1), dtype=float)
    parallel = np.asarray(run_study(cfg, workers=2), dtype=float)
    assert serial.shape == parallel.shape == (48, 8)
    np.testing.assert_array_equal(serial[:, :6], parallel[:, :6])
    np.testing.assert_allclose(serial[:, 6:], parallel[:, 6:], rtol=1e-12, atol=1e-12)
