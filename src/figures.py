"""Reproducible primary + endpoint figures and two implementation diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(out / (stem + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_figures(raw_csv, summary_csv, endpoints_csv, figures_dir):
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_csv)
    summary = pd.read_csv(summary_csv)
    endpoints = pd.read_csv(endpoints_csv)
    widths = sorted(summary.student_width.unique())
    betas = sorted(summary.beta.unique())
    d = int(summary.dimension.iloc[0])
    n_init = int(summary.n.iloc[0])
    a_plus_b = float(summary.alpha.iloc[0] + summary.beta.iloc[0])
    colors = {beta: plt.get_cmap('viridis')(0.15 + 0.75*i/max(1,len(betas)-1))
              for i,beta in enumerate(betas)}

    # Publication-quality small multiples: compare betas at fixed width.
    fig, axes = plt.subplots(1, len(widths), figsize=(10.8, 4.3), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, width in zip(axes, widths):
        for beta in betas:
            grp = summary[(summary.beta==beta)&(summary.student_width==width)&
                          (summary.time>0)].sort_values('time')
            t = grp.time.to_numpy(); y = grp.mean_risk.to_numpy()
            ax.plot(t, y, lw=2, color=colors[beta], label=fr'$\beta={beta:g}$')
            # Confidence intervals are MC uncertainty from randomized initialization,
            # not uncertainty about the ODE integration scheme.
            low = grp.ci95_low.to_numpy(); high = grp.ci95_high.to_numpy()
            if np.isfinite(low).all() and np.isfinite(high).all():
                floor = float(grp.rank_floor.iloc[0])
                ax.fill_between(t, np.maximum(low,floor), high,
                                color=colors[beta], alpha=0.12, lw=0)
        floor = float(endpoints[endpoints.student_width == width].rank_floor.iloc[0])
        ax.axhline(floor, color='black', linestyle=':', lw=1.3, label='rank floor')
        ax.set(xscale='log', yscale='log', title=fr'$r_s={width}$',
               xlabel='Gradient-flow time')
        ax.grid(alpha=0.14, which='major')
    axes[0].set_ylabel('Population excess risk')
    axes[-1].legend(frameon=False, fontsize=9, loc='upper right')
    fig.suptitle(fr'Two-timescale population GF ($d={d}$, $\alpha+\beta={a_plus_b:.2g}$; {n_init} initializations)', fontsize=12)
    save(fig, out, 'gf_risk_facets')

    # Original overlay maintained for easy comparison with earlier pilot screenshots.
    styles = ['-', '--', ':', '-.']
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for beta in betas:
        for j,width in enumerate(widths):
            grp=summary[(summary.beta==beta)&(summary.student_width==width)&
                        (summary.time>0)].sort_values('time')
            ax.plot(grp.time, grp.mean_risk, linestyle=styles[j%len(styles)],
                    color=colors[beta], lw=1.8,
                    label=fr'$\beta={beta:g}$, $r_s={width}$')
    ax.set(xscale='log',yscale='log',xlabel='Gradient-flow time (not FLOPs)',
           ylabel='Population excess risk',title='GF analogue of paper Figure 1(a)')
    ax.legend(ncol=2, fontsize=9, frameon=False)
    ax.grid(alpha=0.15, which='major')
    save(fig,out,'pilot_risk_comparison')

    # Quantitative extension: at t=80, compare how much risk remains above rank floor.
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for j,width in enumerate(widths):
        grp=endpoints[endpoints.student_width==width].sort_values('beta')
        x=grp.beta.to_numpy(float)
        y=grp.mean_final_gap.to_numpy(float)
        lo=grp.ci95_gap_low.to_numpy(float)
        hi=grp.ci95_gap_high.to_numpy(float)
        # Nonnegative risks justify clipping plotted lower intervals at 0.
        err=np.vstack([np.maximum(y-np.maximum(lo,0),0), np.maximum(hi-y,0)])
        ax.errorbar(x+(j-(len(widths)-1)/2)*0.012, y, yerr=err,
                    color=plt.get_cmap('tab10')(j), linestyle='-', marker='o',
                    linewidth=1.7, capsize=3, label=fr'$r_s={width}$')
    end_time=float(endpoints.final_time.iloc[0])
    ax.axhline(0, color='black', linestyle=':', lw=1)
    ax.set(xlabel=r'Covariance decay $\beta$',
           ylabel=r'Mean population risk minus rank floor',
           title=fr'Remaining risk gap at $t={end_time:g}$ ($d={d}$, {n_init} initializations)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.14)
    save(fig,out,'final_gap_by_beta')

    # Random-initialization diagnostic on first scenario.
    beta0, width0 = float(betas[0]), int(widths[0])
    sub = raw[(raw.beta==beta0)&(raw.student_width==width0)&(raw.time>0)]
    fig,ax=plt.subplots(figsize=(7,4.4))
    for rep,grp in sub.groupby('replicate'):
        grp=grp.sort_values('time')
        ax.plot(grp.time,grp.population_risk, lw=1,alpha=0.67)
    floor=float(endpoints[endpoints.student_width==width0].rank_floor.iloc[0])
    ax.axhline(floor,c='black',ls=':',lw=1.3,label='rank floor')
    ax.set(xscale='log',yscale='log',xlabel='Gradient-flow time',
           ylabel='Population excess risk',
           title=fr'Independent initializations ($\beta={beta0:g}$, $r_s={width0}$)')
    ax.legend(frameon=False,fontsize=9)
    save(fig,out,'diagnostic_replicates')

    # Bias-tracking diagnostic from the same scenario.
    rep0=int(raw.replicate.min())
    sub=raw[(raw.beta==beta0)&(raw.student_width==width0)&
            (raw.replicate==rep0)&(raw.time>0)].sort_values('time')
    fig,ax=plt.subplots(figsize=(7,4.4))
    ax.plot(sub.time,np.maximum(np.abs(sub.bias_mismatch),1e-15),lw=1.7)
    ax.set(xscale='log',yscale='log',xlabel='Gradient-flow time',
           ylabel=r'$|b-\mathrm{tr}(\Sigma WW^\top/r_s)|$',
           title='Two-timescale bias tracking (one initialization)')
    ax.grid(alpha=0.14, which='major')
    save(fig,out,'diagnostic_bias_tracking')
    print(f'Saved 5 figure pairs (.png/.pdf) to {out}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',default='results/raw/trajectories.csv')
    p.add_argument('--summary',default='results/summary.csv')
    p.add_argument('--endpoints',default='results/endpoints.csv')
    p.add_argument('--out',default='results/figures')
    args=p.parse_args()
    make_figures(args.raw,args.summary,args.endpoints,args.out)


if __name__=='__main__':
    main()
