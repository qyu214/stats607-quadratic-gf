"""Plot a fixed-alpha covariance sweep, with a *separate oracle floor for each beta*.

Unlike src.figures (the main-study plotting code), this module never assumes
that alpha+beta is constant across conditions.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot(summary_csv: str, endpoints_csv: str, out_dir: str) -> None:
    s = pd.read_csv(summary_csv)
    e = pd.read_csv(endpoints_csv)
    out = Path(out_dir)
    alphas = np.unique(np.round(s.alpha.to_numpy(dtype=float), 10))
    widths = np.unique(s.student_width)
    ds = np.unique(s.dimension)
    if len(alphas) != 1 or len(widths) != 1 or len(ds) != 1:
        raise ValueError('This follow-up expects a single fixed alpha, width, and d')
    alpha = float(alphas[0]); width = int(widths[0]); d = int(ds[0])
    betas = sorted(s.beta.unique())
    colors = {b: plt.get_cmap('viridis')(0.15 + 0.75*i/max(1,len(betas)-1))
              for i,b in enumerate(betas)}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharex=True)
    for beta in betas:
        grp = s[(s.beta == beta) & (s.time > 0)].sort_values('time')
        t=grp.time.to_numpy(float)
        floor=float(grp.rank_floor.iloc[0])
        color=colors[beta]
        label=fr'$\beta={beta:g}$'
        mu=grp.mean_risk.to_numpy(float)
        lo=grp.ci95_low.to_numpy(float)
        hi=grp.ci95_high.to_numpy(float)
        axes[0].plot(t, mu, color=color, linewidth=2, label=label)
        axes[0].fill_between(t, np.maximum(lo, floor), np.maximum(hi, floor),
                             color=color, alpha=0.12, linewidth=0)
        axes[0].axhline(floor, color=color, alpha=0.6, linestyle=':', linewidth=1.3)
        axes[1].plot(t, mu-floor, color=color, linewidth=2, label=label)
        axes[1].fill_between(t, np.maximum(lo-floor,0), np.maximum(hi-floor,0),
                             color=color, alpha=0.12, linewidth=0)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Gradient-flow time')
        ax.grid(alpha=0.16)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Mean population excess risk')
    axes[0].set_title('Risk (dotted: beta-specific rank floors)')
    axes[1].set_ylabel('Mean risk minus beta-specific rank floor')
    axes[1].set_title('Finite-time gap to representational optimum')
    axes[0].legend(frameon=False)
    fig.suptitle(fr'Fixed $\alpha={alpha:g}$, $d={d}$, $r_s={width}$; Monte Carlo pointwise 95% intervals')
    _save(fig, out, 'fixed_alpha_risk_and_gap')

    e=e.sort_values('beta')
    bs=e.beta.to_numpy(float)
    ys=e.mean_final_gap.to_numpy(float)
    los=np.maximum(e.ci95_gap_low.to_numpy(float),0.)
    his=np.maximum(e.ci95_gap_high.to_numpy(float),ys)
    fig, axes=plt.subplots(1,2,figsize=(11.5,4.1))
    a,b=axes
    a.plot(bs,e.mean_final_risk.to_numpy(float),'o-',label='Final mean risk')
    a.plot(bs,e.rank_floor.to_numpy(float),'s--',label='Beta-specific rank floor')
    a.set(ylabel='Population excess risk',xlabel=r'Covariance exponent $\beta$',
          title=fr'Absolute risk and oracle floor at $t={float(e.final_time.iloc[0]):g}$')
    a.legend(frameon=False)
    err=np.vstack([np.maximum(ys-los,0),np.maximum(his-ys,0)])
    b.errorbar(bs,ys,yerr=err,fmt='o-',capsize=3)
    b.set(xlabel=r'Covariance exponent $\beta$',ylabel='Final mean risk minus rank floor',
          title='Gap (pointwise approximate 95% MC intervals)')
    for ax in axes: ax.grid(alpha=0.16)
    _save(fig,out,'fixed_alpha_endpoints')
    print(f'Saved 2 figure pairs (.png/.pdf) to {out}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--summary',default='results/fixed_alpha/summary.csv')
    p.add_argument('--endpoints',default='results/fixed_alpha/endpoints.csv')
    p.add_argument('--out',default='results/fixed_alpha/figures')
    args=p.parse_args()
    plot(args.summary,args.endpoints,args.out)

if __name__ == '__main__':
    main()
