"""Low-rank implementation of population gradient flow."""

import numpy as np
from scipy.integrate import solve_ivp
from .methods import population_risk_and_gradients


def low_rank_field(W, b, lam, teacher_weight):
    r_s = W.shape[1]

    SW = lam[:, None] * W
    gram = W.T @ SW
    e = b - np.sum(W * SW) / r_s

    dW = (
        -(4.0 / r_s) * (SW @ gram)
        + 4.0 * teacher_weight[:, None] * W
        + 2.0 * e * SW
    )

    return dW, e


def solve_paper_population_gf_optimized(
    W0, b0, lam, B, times, *,
    delta=0.05, rtol=1e-6, atol=1e-8
):
    times = np.asarray(times, float)
    if delta <= 0 or len(times) < 2 or times[0] < 0 \
            or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid time grid or delta")

    d, r_s = W0.shape
    teacher_weight = lam**2 * np.diag(B)

    def rhs(t, state):
        W = state[:-1].reshape(d, r_s)
        b = float(state[-1])
        dW, e = low_rank_field(W, b, lam, teacher_weight)
        return np.r_[dW.ravel(), -e / delta]

    state0 = np.r_[W0.ravel(), float(b0)]
    sol = solve_ivp(
        rhs, (times[0], times[-1]), state0,
        t_eval=times, rtol=rtol, atol=atol, method="RK45"
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    risks = np.empty(len(sol.t))
    mismatch = np.empty(len(sol.t))

    for i in range(len(sol.t)):
        W = sol.y[:-1, i].reshape(d, r_s)
        b = float(sol.y[-1, i])
        risks[i], _, _, mismatch[i] = (
            population_risk_and_gradients(W, b, lam, B)
        )

    return sol.t, risks, mismatch
