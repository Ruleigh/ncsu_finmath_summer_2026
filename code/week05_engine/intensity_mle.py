"""
Week 5 — Maximum-likelihood calibration of the fill intensity (Milestone 1).

Fills arrive as a Poisson process whose rate depends on the quote distance:

    lambda(delta) = A * exp(-kappa * delta)

Given observed fills at distances ``delta_i`` over exposure times ``dt_i``,
the Poisson negative log-likelihood is

    -log L = sum_i [ lambda(delta_i) * dt_i - log lambda(delta_i) ]

We minimize it with PyTorch autograd (Adam). We optimize log(A) so A stays
positive. A chi-square style goodness-of-fit compares observed vs. expected
fill counts per distance bucket.

    python intensity_mle.py
"""
from __future__ import annotations

import torch


def simulate_fills(A=140.0, kappa=1.5, n=20000, dt=1e-3, seed=0):
    """Synthetic fills: quote distances and a fill/no-fill indicator per step."""
    g = torch.Generator().manual_seed(seed)
    delta = torch.rand(n, generator=g) * 2.0          # quote distances in [0,2]
    lam = A * torch.exp(-kappa * delta)
    filled = (torch.rand(n, generator=g) < (lam * dt).clamp(max=1.0)).float()
    return delta, filled, torch.full((n,), dt)


def calibrate(delta, filled, dt, steps=6000, lr=0.1):
    """Return fitted (A, kappa) by maximum likelihood.

    Both parameters are optimized in log-space (A = exp(logA),
    kappa = exp(logk)) so they stay positive and the gradient never gets
    clipped to zero -- a plain ``clamp`` would stall the optimizer.
    """
    logA = torch.zeros((), requires_grad=True)
    logk = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([logA, logk], lr=lr)
    for _ in range(steps):
        lam = torch.exp(logA) * torch.exp(-torch.exp(logk) * delta)
        # Poisson NLL using fill indicators as event counts in [t, t+dt)
        nll = (lam * dt - filled * torch.log(lam + 1e-12)).sum()
        opt.zero_grad()
        nll.backward()
        opt.step()
    return torch.exp(logA).item(), torch.exp(logk).item()


def chi_square(delta, filled, dt, A, kappa, bins=10):
    """Pearson chi-square of observed vs. expected fills per distance bucket."""
    edges = torch.linspace(0, float(delta.max()), bins + 1)
    chi2 = 0.0
    for b in range(bins):
        mask = (delta >= edges[b]) & (delta < edges[b + 1])
        if mask.sum() == 0:
            continue
        observed = filled[mask].sum()
        expected = (A * torch.exp(-kappa * delta[mask]) * dt[mask]).sum()
        if expected > 0:
            chi2 += float((observed - expected) ** 2 / expected)
    return chi2


if __name__ == "__main__":
    true_A, true_kappa = 140.0, 1.5
    delta, filled, dt = simulate_fills(true_A, true_kappa)
    A_hat, kappa_hat = calibrate(delta, filled, dt)
    print(f"true   A={true_A:.1f}  kappa={true_kappa:.2f}")
    print(f"fitted A={A_hat:7.1f}  kappa={kappa_hat:5.2f}")
    print(f"chi-square goodness-of-fit: {chi_square(delta, filled, dt, A_hat, kappa_hat):.2f}")
    print("On Coincall data: delta_i = |fill price - mid at fill|; bucket and fit.")
