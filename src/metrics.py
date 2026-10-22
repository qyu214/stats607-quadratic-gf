"""Analytical approximation floor and Monte Carlo performance summaries.

Risk convention is the paper's main-text excess population risk:
  ||S* - M(W)||_F^2 + 0.5 (b-tr(M(W)))^2.
When teacher/covariance commute and the teacher matrix is PSD,
  inf_{rank(W)<=r_s, b} R(W,b) = sum_{i>r_s} eig_i(S*)^2.
The floor is an *oracle representational lower bound*, not a claim GF finds it.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def rank_risk_floor(d: int, beta: float, alpha: float,
                    student_width: int, teacher_rank: int | None = None) -> float:
    """Closed-form global minimum over rank<=r_s PSD student matrices and free b."""
    if d <= 0 or student_width <= 0:
        raise ValueError("d and student width must be positive")
    if alpha < 0 or beta < 0:
        raise ValueError("Spectral exponents must be nonnegative")
    r = d if teacher_rank is None else int(teacher_rank)
    if r <= 0 or r > d:
        raise ValueError("teacher_rank must be in [1,d]")
    ii = np.arange(1, d + 1, dtype=float)
    vals = np.zeros(d)
    vals[:r] = ii[:r] ** (-alpha-beta)
    vals /= np.linalg.norm(vals)
    return float(np.sum(np.sort(vals**2)[::-1][student_width:]))


def _add_mc_intervals(summary: pd.DataFrame) -> pd.DataFrame:
    """Student-t uncertainty intervals for the mean across independent initializations."""
    n = summary['n'].to_numpy()
    crit = student_t.ppf(0.975, np.maximum(n - 1, 1))
    crit = np.where(n > 1, crit, np.nan)
    summary['ci95_low'] = summary['mean_risk'] - crit * summary['mcse_risk']
    summary['ci95_high'] = summary['mean_risk'] + crit * summary['mcse_risk']
    return summary


def summarize_trajectories(raw_path, output_path):
    """Summarize each (scenario,time) from raw runs, retaining MC uncertainty."""
    df = pd.read_csv(raw_path)
    gcols = ['dimension', 'alpha', 'beta', 'student_width', 'time']
    summary = df.groupby(gcols, as_index=False).agg(
        mean_risk=('population_risk', 'mean'),
        sd_risk=('population_risk', 'std'),
        n=('population_risk', 'size'),
        mean_abs_bias=('bias_mismatch', lambda s: float(np.mean(np.abs(s)))))
    summary['mcse_risk'] = summary['sd_risk'] / np.sqrt(summary['n'])
    # The input grid uses teacher_rank=d. If this changes, document and
    # pass the appropriate teacher rank in future experiment variants.
    summary['rank_floor'] = [
        rank_risk_floor(int(d), float(beta), float(alpha), int(rs))
        for d, alpha, beta, rs in zip(summary.dimension, summary.alpha,
                                     summary.beta, summary.student_width)
    ]
    summary['mean_gap_to_floor'] = summary['mean_risk'] - summary['rank_floor']
    summary = _add_mc_intervals(summary)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def summarize_endpoints(raw_path, output_path, *, target_time: float = 10.0):
    """Per-scenario final risk and risk at t=10, with uncertainty from seeds.

    Uses *linear interpolation in time within each run* to define R(10).
    Final_time is the maximal simulated time; no assumption of convergence.
    """
    raw = pd.read_csv(raw_path)
    keys = ['dimension', 'alpha', 'beta', 'student_width']
    one_run = []
    for (d, alpha, beta, rs, rep), group in raw.groupby(keys + ['replicate'], sort=True):
        group = group.sort_values('time')
        times = group.time.to_numpy()
        risks = group.population_risk.to_numpy()
        if not (times[0] <= target_time <= times[-1]):
            raise ValueError(f'target_time={target_time} outside grid [{times[0]}, {times[-1]}]')
        floor = rank_risk_floor(int(d), float(beta), float(alpha), int(rs))
        one_run.append({'dimension': int(d), 'alpha': float(alpha),
                        'beta': float(beta), 'student_width': int(rs),
                        'replicate': int(rep), 'final_time': float(times[-1]),
                        'risk_at_t10': float(np.interp(target_time, times, risks)),
                        'final_risk': float(risks[-1]),
                        'final_gap_to_floor': float(risks[-1]-floor),
                        'rank_floor': floor})
    run_table = pd.DataFrame(one_run)
    endpoints = run_table.groupby(keys, as_index=False).agg(
        n=('replicate','size'), final_time=('final_time','first'),
        mean_risk_at_t10=('risk_at_t10','mean'),
        mean_final_risk=('final_risk','mean'),
        mean_final_gap=('final_gap_to_floor','mean'),
        sd_final_gap=('final_gap_to_floor','std'),
        rank_floor=('rank_floor','first'))
    endpoints['mcse_final_gap'] = endpoints['sd_final_gap'] / np.sqrt(endpoints['n'])
    crit = student_t.ppf(0.975, np.maximum(endpoints['n']-1,1))
    endpoints['ci95_gap_low'] = endpoints['mean_final_gap'] - crit*endpoints['mcse_final_gap']
    endpoints['ci95_gap_high'] = endpoints['mean_final_gap'] + crit*endpoints['mcse_final_gap']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    endpoints.to_csv(output_path, index=False)
    return endpoints
