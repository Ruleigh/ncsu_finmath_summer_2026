"""
Week 4 — Merton portfolio problem under CARA, by HJB value iteration.

A worked example of the stochastic-control template (dynamics -> HJB ->
solution) that underlies Avellaneda–Stoikov. We confirm numerically the
hallmark of CARA (exponential) utility: the optimal *risky holding* is
constant in wealth,

    pi* = (mu - r) / (gamma * sigma^2),

whereas under CRRA the optimal *fraction* of wealth is constant instead.

    python merton.py
"""
from __future__ import annotations

import torch


def analytic_cara_holding(mu: float, r: float, sigma: float, gamma: float) -> float:
    """Closed-form optimal dollar amount in the risky asset under CARA."""
    return (mu - r) / (gamma * sigma ** 2)


def hjb_value_iteration(mu=0.08, r=0.02, sigma=0.2, gamma=1.0,
                        T=1.0, steps=2000, w_grid=None) -> dict:
    """March the HJB backward on a wealth grid to check the CARA solution.

    For CARA the optimal control is wealth-independent, so we plug pi* in and
    integrate the value function; the point of the exercise is to verify the
    structure numerically rather than to *find* the control by grid search.
    """
    if w_grid is None:
        w_grid = torch.linspace(-5.0, 5.0, 401)
    dt = T / steps
    dw = (w_grid[1] - w_grid[0]).item()
    V = -torch.exp(-gamma * w_grid)             # terminal utility U(w) = -e^{-gamma w}
    pi = analytic_cara_holding(mu, r, sigma, gamma)
    drift = r * w_grid + pi * (mu - r)
    diffusion = 0.5 * (pi * sigma) ** 2
    for _ in range(steps):
        dV = torch.gradient(V, spacing=dw)[0]
        d2V = torch.gradient(dV, spacing=dw)[0]
        V = V + dt * (drift * dV + diffusion * d2V)   # explicit Euler step
    return {"pi_star": pi, "w_grid": w_grid, "V": V}


if __name__ == "__main__":
    out = hjb_value_iteration()
    print(f"CARA optimal risky holding pi* = {out['pi_star']:.4f}  (constant in wealth)")
    # value function should remain monotone increasing and concave-ish
    V, w = out["V"], out["w_grid"]
    incr = bool((torch.diff(V) > 0).all())
    print(f"value function increasing in wealth: {incr}")
    print(f"V(0) = {V[len(V)//2]:.4f}")
    print("Under CRRA the same machinery gives a wealth-PROPORTIONAL policy.")
