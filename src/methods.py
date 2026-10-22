"""Population gradient flow for the *full* student matrix and trainable bias.

For the main-text (2.2) risk normalization (without irreducible noise):
  R(W,b) = ||Sigma^(1/2)(WW.T/r_s - B)Sigma^(1/2)||_F^2
           + 1/2 (b - tr(Sigma WW.T/r_s))^2.
Main-text (GF): W'=-r_s * grad_W R, b'=-grad_b R/delta.
The Appendix B.2 convention uses 1/4 this risk, hence 1/4 these rates.
"""
import numpy as np
from scipy.integrate import solve_ivp


def loss_and_gradient(W: np.ndarray, lam: np.ndarray, B: np.ndarray):
    """Legacy centered half-squared risk & gradient; kept for pilot tests only."""
    d, r_s = W.shape
    if len(lam) != d or B.shape != (d, d):
        raise ValueError('Shape mismatch')
    G_minus_B = W @ W.T / r_s - B
    weighted = lam[:, None] * G_minus_B * lam[None, :]
    return float(np.sum(G_minus_B * weighted)), (4 / r_s) * weighted @ W


def population_risk_and_gradients(W: np.ndarray, b: float, lam: np.ndarray, B: np.ndarray):
    """Main-text risk, gradients wrt W and b, and centering mismatch e."""
    d, r_s = W.shape
    if lam.shape != (d,) or B.shape != (d, d):
        raise ValueError('Shape mismatch')
    G = (W @ W.T) / r_s
    delta_matrix = G - B
    weighted = lam[:, None] * delta_matrix * lam[None, :]
    e = float(b - np.dot(lam, np.diag(G)))
    risk = float(np.sum(delta_matrix * weighted) + 0.5 * e * e)
    grad_W = (4 / r_s) * (weighted @ W) - (2 * e / r_s) * (lam[:, None] * W)
    return risk, grad_W, e, e


def solve_paper_population_gf(W0, b0, lam, B, times, *, delta=0.05,
                              rtol=1e-6, atol=1e-8):
    """Solve main-text two-timescale full-matrix GF, return times, risks and e(t)."""
    times = np.asarray(times, float)
    if delta <= 0 or len(times) < 2 or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid time grid or delta')
    d, r_s = W0.shape

    def rhs(t, state):
        W = state[:-1].reshape(d, r_s)
        b = float(state[-1])
        _, grad_W, grad_b, _ = population_risk_and_gradients(W, b, lam, B)
        return np.r_[(-r_s * grad_W).ravel(), -grad_b / delta]

    state0 = np.r_[W0.ravel(), float(b0)]
    sol = solve_ivp(rhs, (times[0], times[-1]), state0,
                    t_eval=times, rtol=rtol, atol=atol, method='RK45')
    if not sol.success:
        raise RuntimeError(sol.message)
    risk = np.empty(len(sol.t))
    mismatch = np.empty(len(sol.t))
    for i in range(len(sol.t)):
        W = sol.y[:-1, i].reshape(d, r_s)
        b = float(sol.y[-1, i])
        risk[i], _, _, mismatch[i] = population_risk_and_gradients(W, b, lam, B)
    return sol.t, risk, mismatch


def solve_population_gradient_flow(W0, lam, B, times, *, rtol=1e-7, atol=1e-9):
    """Legacy centered-flow pilot implementation for backwards compatibility/tests."""
    times = np.asarray(times, float)
    if len(times) < 2 or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid time grid')
    d, r_s = W0.shape
    def rhs(t, wflat):
        _, grad = loss_and_gradient(wflat.reshape(d, r_s), lam, B)
        return -grad.ravel()
    sol = solve_ivp(rhs, (times[0], times[-1]), W0.ravel(), t_eval=times,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    risks = []
    for k in range(len(sol.t)):
        L, _ = loss_and_gradient(sol.y[:, k].reshape(d, r_s), lam, B)
        risks.append(2 * L)
    return sol.t, np.asarray(risks)
