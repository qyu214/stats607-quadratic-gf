"""Checks for the d=256 exploratory study's reporting helper."""
import numpy as np
import pandas as pd
from src.metrics import rank_risk_floor
from src.width_pilot_figures import summarize_width_trial


def test_rank_floor_monotone_in_width():
    f=[rank_risk_floor(256,.35,.35,k) for k in [32,64,128]]
    assert f[0]>f[1]>f[2]>0


def test_width_summary_uses_individual_interpolation():
    rows=[]
    for rs in [2,3]:
        floor=rank_risk_floor(5,.35,.35,rs)
        for rep in [0,1]:
            for t in [0.,10.,20.]:
                risk=floor+1.+rep*.2-.01*t
                rows.append(dict(dimension=5,alpha=.35,beta=.35,
                                 student_width=rs,replicate=rep,time=t,
                                 population_risk=risk))
    _, curves, cp = summarize_width_trial(pd.DataFrame(rows),checkpoints=(5.,15.))
    assert len(cp)==4
    assert (cp['n']==2).all()
    assert np.allclose(cp[cp.time==5].mean_gap,1.05)
    assert (curves['mean_gap']>=0).all()
