"""Run a small Figure-1(a)-style *population GF* experiment, not online SGD."""
import argparse
import json
from pathlib import Path
import time
import platform
import scipy
import numpy as np
import pandas as pd
from .dgps import make_diagonal_model, initialize_student, initialize_bias
from .methods import solve_paper_population_gf


def build_times(config):
    n = int(config['n_time_points'])
    if config.get('time_grid', 'linear') == 'log':
        return np.r_[0., np.geomspace(float(config.get('time_start', 0.02)),
                                       float(config['time_max']), n - 1)]
    return np.linspace(0., float(config['time_max']), n)


def simulate(config: dict):
    d = int(config['dimension'])
    widths = list(map(int, config['student_widths']))
    betas = list(map(float, config['covariance_exponents']))
    n_rep = int(config['n_replicates'])
    times = build_times(config)
    total = n_rep * len(widths) * len(betas)
    rows = []
    completed = 0
    for rs in widths:
        for rep in range(n_rep):
            # Paired across beta: same random W0 and b0 for each (width,rep).
            rng = np.random.default_rng(np.random.SeedSequence([
                int(config['master_seed']), d, rs, rep]))
            W0 = initialize_student(rng, d, rs,
                                    config.get('initialization_std', 'paper'))
            b0 = initialize_bias(rng, config.get('bias_initialization', 'normal'))
            for beta in betas:
                alpha = (float(config['alpha_plus_beta']) - beta
                         if 'alpha_plus_beta' in config
                         else float(config['teacher_exponent']))
                if alpha < 0:
                    raise ValueError('Negative teacher exponent: alpha_plus_beta must exceed beta')
                lam, B = make_diagonal_model(d, beta, alpha, config.get('teacher_rank'))
                t, risks, biases = solve_paper_population_gf(
                    W0, b0, lam, B, times,
                    delta=float(config.get('delta', 0.05)),
                    rtol=float(config.get('rtol', 1e-6)),
                    atol=float(config.get('atol', 1e-8)))
                for t_i, risk_i, e_i in zip(t, risks, biases):
                    rows.append({'study_id': config['study_id'], 'dimension': d,
                                 'beta': beta, 'alpha': round(alpha, 10), 'student_width': rs,
                                 'replicate': rep, 'time': float(t_i),
                                 'population_risk': float(risk_i),
                                 'bias_mismatch': float(e_i)})
                completed += 1
                print(f'Completed {completed}/{total}: beta={beta}, alpha={alpha:.2f}, '
                      f'r_s={rs}, rep={rep}', flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output', default='results/raw/trajectories.csv')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = json.load(f)
    start = time.perf_counter()
    df = simulate(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    elapsed = time.perf_counter() - start
    with (output.parent / 'run_metadata.json').open('w', encoding='utf-8') as f:
        json.dump({'config': config, 'elapsed_simulation_seconds': elapsed,
                   'rng': 'SeedSequence(master_seed, dimension, student_width, replicate)',
                   'comparison': 'paired W0/b0 across beta at fixed dimension, width, replicate',
                   'gf_convention': 'Main paper Eq (2.2) and (GF); zero observation noise',
                   'environment': {'python': platform.python_version(),
                                   'numpy': np.__version__,
                                   'scipy': scipy.__version__,
                                   'pandas': pd.__version__}},
                  f, indent=2)
    print(f'Saved {len(df)} timepoint records in {output}; runtime {elapsed:.2f}s')


if __name__ == '__main__':
    main()
