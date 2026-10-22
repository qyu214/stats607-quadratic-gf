import numpy as np
import pandas as pd
from src.metrics import rank_risk_floor, summarize_endpoints


def test_rank_floor_is_same_beta_when_alpha_plus_beta_fixed():
    for rs in [8, 16]:
        observed = [rank_risk_floor(32, beta, 0.7-beta, rs)
                    for beta in [0.1,0.35,0.6]]
        assert np.allclose(observed, observed[0], atol=1e-14)


def test_rank_floor_monotone_in_width_and_zero_for_full_rank():
    f8 = rank_risk_floor(32, 0.1, 0.6, 8)
    f16 = rank_risk_floor(32, 0.1, 0.6, 16)
    f32 = rank_risk_floor(32, 0.1, 0.6, 32)
    assert 0 <= f32 < f16 < f8 < 1
    assert np.isclose(f32, 0)


def test_endpoints_use_last_time_and_interpolate(tmp_path):
    rows=[]
    for rep in range(3):
        for t,risk in [(0.,0.9), (5.,0.5), (15.,0.3), (20.,0.2)]:
            rows.append({'dimension':4,'alpha':0.6,'beta':0.1,
                         'student_width':2,'replicate':rep,'time':t,
                         'population_risk':risk+rep*0.01})
    source=tmp_path/'traj.csv';dest=tmp_path/'endpoints.csv'
    pd.DataFrame(rows).to_csv(source,index=False)
    out=summarize_endpoints(source,dest)
    assert len(out)==1 and int(out.iloc[0].n)==3
    assert np.isclose(out.iloc[0].mean_risk_at_t10, 0.41)
    assert np.isclose(out.iloc[0].mean_final_risk, 0.21)
    assert dest.exists()
