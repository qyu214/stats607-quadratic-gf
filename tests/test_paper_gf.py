"""Checks for exact-gradient two-timescale main-text GF."""
import numpy as np
from src.dgps import make_diagonal_model, initialize_student
from src.methods import population_risk_and_gradients, solve_paper_population_gf


def test_paper_gradient_finite_difference():
    lam,B=make_diagonal_model(3,0.35,0.35)
    W=np.random.default_rng(3).normal(size=(3,2))*0.3
    b=0.2
    R,gW,gb,e=population_risk_and_gradients(W,b,lam,B)
    eps=1e-6
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wa,Wb=W.copy(),W.copy()
            Wa[i,j]+=eps;Wb[i,j]-=eps
            Ra=population_risk_and_gradients(Wa,b,lam,B)[0]
            Rb=population_risk_and_gradients(Wb,b,lam,B)[0]
            assert np.isclose((Ra-Rb)/(2*eps),gW[i,j],atol=5e-7,rtol=1e-5)
    Rp=population_risk_and_gradients(W,b+eps,lam,B)[0]
    Rm=population_risk_and_gradients(W,b-eps,lam,B)[0]
    assert np.isclose((Rp-Rm)/(2*eps),gb,atol=5e-7)
    assert np.isclose(gb,e)


def test_paper_risk_nonincreasing():
    d, rs = 4,3
    lam,B=make_diagonal_model(d,0.35,0.35)
    W0=initialize_student(np.random.default_rng(5),d,rs)
    t=np.linspace(0,10,31)
    _, risk, mismatch=solve_paper_population_gf(W0,0.6,lam,B,t,delta=0.05)
    assert np.all(np.diff(risk)<1e-5)
    assert np.isfinite(mismatch).all()


def test_paper_initialization_scale():
    d,rs=40,10000
    W0=initialize_student(np.random.default_rng(8),d,rs)
    assert np.isclose(np.var(W0),1/d,rtol=0.02)


def test_alpha_beta_coupled_normalization():
    d=20
    for beta in [0.1,0.35,0.6]:
        alpha=0.7-beta
        lam,B=make_diagonal_model(d,beta,alpha)
        assert np.isclose(np.linalg.norm(lam * np.diag(B)),1)
