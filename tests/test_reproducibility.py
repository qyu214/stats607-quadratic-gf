from src.simulation import simulate


def test_simulation_reproducible_with_same_seed():
    config = {"study_id": "test", "dimension": 3, "student_widths": [3],
              "covariance_exponents": [0.5], "teacher_exponent": 1.0,
              "initialization_std": 0.15, "n_replicates": 2,
              "master_seed": 34, "time_max": 2.0,
              "n_time_points": 5, "rtol": 1e-7, "atol": 1e-9}
    a, b = simulate(config), simulate(config)
    assert a.equals(b)
