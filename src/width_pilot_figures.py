"""Analyze the exploratory d=256 width sweep without changing main Project 2 results.

Usage:
  python -m src.width_pilot_figures \
    --raw results/width256/raw/trajectories.csv \
    --out results/width256

The time course is an observed finite-dimensional population-GF simulation,
not an estimate of the paper's online-SGD asymptotic exponent.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .metrics import rank_risk_floor


def summarize_width_trial(raw: pd.DataFrame, checkpoints=(80.0, 160.0)):
    required = {'dimension', 'alpha', 'beta', 'student_width', 'replicate',
                'time', 'population_risk'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing columns: {required-set(raw.columns)}')
    if raw.empty:
        raise ValueError('No data')
    unique_dims = raw.dimension.unique()
    unique_betas = raw.beta.unique()
    if len(unique_dims) != 1 or len(unique_betas) != 1:
        raise ValueError('This figure assumes one dimension and one beta')
    d = int(unique_dims[0]); beta = float(unique_betas[0])
    raw = raw.copy()
    raw['rank_floor'] = [rank_risk_floor(int(di), float(bi), float(a), int(rs))
                         for di, bi, a, rs in zip(raw.dimension, raw.beta, raw.alpha, raw.student_width)]
    raw['gap'] = raw.population_risk - raw.rank_floor
    if (raw['gap'] < -1e-7).any():
        raise ValueError('Risk below representational rank floor: check normalization')

    per_run = []
    for (rs, rep), group in raw.groupby(['student_width','replicate']):
        g = group.sort_values('time')
        t_grid = g.time.to_numpy(float)
        risk_grid = g.population_risk.to_numpy(float)
        if np.any(np.diff(t_grid) <= 0):
            raise ValueError('Time grid must increase')
        floor = float(g.rank_floor.iloc[0])
        for t in checkpoints:
            if float(t) < t_grid[0] or float(t) > t_grid[-1]:
                raise ValueError(f'Checkpoint {t} outside sampled range')
            risk = float(np.interp(float(t), t_grid, risk_grid))
            per_run.append(dict(d=d, beta=beta, student_width=int(rs),
                                replicate=int(rep), time=float(t),
                                risk=risk, rank_floor=floor, gap=risk-floor))
    cp = pd.DataFrame(per_run)
    keys = ['student_width','time']
    curve = raw.groupby(keys, as_index=False).agg(
        mean_risk=('population_risk','mean'),
        sd_risk=('population_risk','std'),
        mean_gap=('gap','mean'),sd_gap=('gap','std'),
        n=('replicate','count'),rank_floor=('rank_floor','first'))
    ends = cp.groupby(keys, as_index=False).agg(
        mean_risk=('risk','mean'),mean_gap=('gap','mean'),sd_gap=('gap','std'),
        n=('replicate','count'),rank_floor=('rank_floor','first'))
    for table in (curve, ends):
        table['mcse_gap'] = table.sd_gap / np.sqrt(table.n)
        critical = student_t.ppf(.975, table.n - 1)
        table['ci_low_gap'] = table.mean_gap - critical * table.mcse_gap
        table['ci_high_gap'] = table.mean_gap + critical * table.mcse_gap
    return raw, curve, ends


def save_figure(fig, output_stem: Path):
    import matplotlib.pyplot as plt
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def generate_figures(raw, curve, ends, out_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    widths = sorted(raw.student_width.unique())
    nrep = int(curve.n.max())
    colors = {width:plt.get_cmap('tab10')(i) for i,width in enumerate(widths)}
    d = int(raw.dimension.iloc[0]); beta = float(raw.beta.iloc[0])
    # Absolute risk, together with oracle representational floors.
    fig, ax=plt.subplots(figsize=(8.8, 5.0))
    for width in widths:
        g=curve[(curve.student_width==width)&(curve.time>0)].sort_values('time')
        c=colors[width]
        ax.plot(g.time,g.mean_risk,lw=2.3,color=c,label=fr'$r_s={width}$')
        floor=float(g.rank_floor.iloc[0])
        ax.axhline(floor,ls=':',lw=1.2,color=c)
        if nrep>1:
            rawgroup=raw[(raw.student_width==width)&(raw.time>0)]
            for _,rep_data in rawgroup.groupby('replicate'):
                ax.plot(rep_data.sort_values('time').time,
                        rep_data.sort_values('time').population_risk,
                        color=c,lw=.65,alpha=.15)
    ax.set(xscale='log',yscale='log',xlabel='Gradient-flow time',
           ylabel='Population excess risk',
           title=fr'Width sensitivity in population GF ($d={d}$, $\beta={beta:g}$, {nrep} seeds)')
    ax.grid(alpha=.13,which='major')
    ax.legend(frameon=False,title='Solid: mean; dotted: rank floor')
    fig.tight_layout()
    save_figure(fig, out_dir/'risk_vs_time')
    # Gap to own floor, not raw risks, isolates finite-time excess over width floor.
    fig,ax=plt.subplots(figsize=(8.8,5.0))
    for width in widths:
        g=curve[(curve.student_width==width)&(curve.time>0)].sort_values('time')
        t=g.time.to_numpy(float); gap=g.mean_gap.to_numpy(float)
        c=colors[width]
        ax.plot(t, np.maximum(gap,1e-12), lw=2.3,color=c,label=fr'$r_s={width}$')
        if nrep>1:
            low=g.ci_low_gap.to_numpy(float)
            high=g.ci_high_gap.to_numpy(float)
            ax.fill_between(t, np.maximum(low,1e-12),np.maximum(high,1e-12),
                            alpha=.14,color=c,linewidth=0)
    ax.set(xscale='log',yscale='log',xlabel='Gradient-flow time',
           ylabel='Mean risk minus width-specific rank floor',
           title=fr'Finite-time gap ($d={d}$, $\beta={beta:g}$, {nrep} seeds)')
    ax.legend(frameon=False)
    ax.grid(alpha=.13,which='major')
    if nrep>1:
        ax.text(.015,.02,'Shading: pointwise Student-t intervals (exploratory)',
                transform=ax.transAxes,fontsize=8,alpha=.72)
    fig.tight_layout()
    save_figure(fig,out_dir/'gap_vs_time')
    # Transparent exact checkpoint table visual.
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    times=sorted(ends.time.unique())
    for j,t in enumerate(times):
        s=ends[ends.time==t].sort_values('student_width')
        x=s.student_width.to_numpy(float); y=s.mean_gap.to_numpy(float)
        if nrep>1:
            lo=s.ci_low_gap.to_numpy(float); hi=s.ci_high_gap.to_numpy(float)
            er=np.array([np.maximum(0,y-np.maximum(0,lo)),np.maximum(0,hi-y)])
            ax.errorbar(x + (j-.5)*2, y, yerr=er, fmt='o-',capsize=3,
                        label=fr'$t={t:g}$')
        else:
            ax.plot(x,y,'o-',label=fr'$t={t:g}$')
    ax.set(xlabel='Student width',ylabel='Mean risk minus rank floor',
           title=fr'Width comparison at fixed times ($d={d}$, {nrep} seeds)')
    ax.set_xticks(widths)
    ax.grid(alpha=.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig,out_dir/'gap_checkpoints')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',default='results/width256/raw/trajectories.csv')
    p.add_argument('--out',default='results/width256')
    args=p.parse_args()
    raw, curve, ends=summarize_width_trial(pd.read_csv(args.raw))
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    curve.to_csv(out/'summary.csv',index=False)
    ends.to_csv(out/'checkpoints.csv',index=False)
    generate_figures(raw,curve,ends,out/'figures')
    print(ends[['student_width','time','n','mean_risk','rank_floor','mean_gap','mcse_gap']]
          .to_string(index=False,float_format=lambda x:f'{x:.6g}'))
    print('Saved plots and summary to:',out)


if __name__=='__main__':
    main()
