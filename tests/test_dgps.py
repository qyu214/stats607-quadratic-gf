import numpy as np
from src.dgps import make_diagonal_model, sample_covariates


def test_model_dimensions_and_normalization():
    lam, B = make_diagonal_model(8, 0.7, 1.0)
    assert lam.shape == (8,) and B.shape == (8, 8)
    assert np.all(lam > 0)
    # B = T/||T Sigma||_F, with diagonal T and Sigma.
    assert np.isclose(np.linalg.norm(np.diag(B) * lam), 1.0)


def test_sample_covariance_matches_population():
    lam, _ = make_diagonal_model(4, 0.5, 1.0)
    X = sample_covariates(np.random.default_rng(42), 18000, lam)
    emp = np.cov(X, rowvar=False)
    assert np.allclose(np.diag(emp), lam, atol=0.045)
