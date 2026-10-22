"""Spectrum and random initialization for a Gaussian quadratic teacher-student model.

Public-paper mathematical model. Full (not diagonal) student initializations.
"""
import numpy as np


def make_diagonal_model(d: int, beta: float, teacher_exponent: float, teacher_rank=None):
    """Sigma = diag(i**-beta), T=diag(i**-alpha, i<=r), B=T/||T Sigma||_F.

    The noise-free teacher is y=x.T@B@x-tr(Sigma@B).
    Parameters are separate so alpha may change with beta (alpha+beta=0.7).
    """
    if d < 1 or beta < 0 or teacher_exponent < 0:
        raise ValueError('d must be positive and exponents nonnegative')
    r = d if teacher_rank is None else int(teacher_rank)
    if not 1 <= r <= d:
        raise ValueError('teacher_rank must be between 1 and d')
    i = np.arange(1, d + 1, dtype=float)
    lam = i ** (-float(beta))
    signal = np.zeros(d)
    signal[:r] = i[:r] ** (-float(teacher_exponent))
    Zr = np.linalg.norm(lam * signal)
    B = np.diag(signal / Zr)
    return lam, B


def sample_covariates(rng: np.random.Generator, n: int, lam: np.ndarray):
    return rng.standard_normal((n, len(lam))) * np.sqrt(lam)


def initialize_student(rng: np.random.Generator, d: int, r_s: int, std=None):
    if r_s < 1 or d < 1:
        raise ValueError('d and r_s must be positive')
    if std is None or std == 'paper':
        std = d ** (-0.5)
    if not isinstance(std, (float, int)) or std <= 0:
        raise ValueError('initialization std must be positive')
    return rng.normal(0., float(std), (d, r_s))


def initialize_bias(rng: np.random.Generator, mode='normal'):
    """Paper main text: b0~N(0,1); Appendix uses b0=0 for its proof."""
    if mode == 'normal':
        return float(rng.normal())
    if mode == 'zero':
        return 0.0
    raise ValueError('bias initialization must be normal or zero')
