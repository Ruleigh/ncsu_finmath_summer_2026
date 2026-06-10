"""
Week 3 — The Avellaneda–Stoikov quoting engine (closed form) in PyTorch.

Implements the asymptotic (long-horizon) Avellaneda–Stoikov quotes and a
simple one-day simulation comparing the optimal inventory-aware policy to a
symmetric policy that ignores inventory.

Key formulas (derived in lecture; see the Week 3 deck for the "why"):

    reservation price   r(q) = S - q * gamma * sigma^2 * (T - t)
    optimal half-spread d*   = (1/gamma) * ln(1 + gamma/kappa)
                                + (gamma * sigma^2 / 2) * (T - t)

with mid-price an arithmetic Brownian motion (dS = sigma dW) and fill
intensity lambda(delta) = A * exp(-kappa * delta).

    python as_engine.py
"""
from __future__ import annotations

import torch


def as_quotes(S: torch.Tensor, q: torch.Tensor, gamma: float, sigma: float,
              kappa: float, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymptotic Avellaneda–Stoikov bid/ask. Vectorized over inventory ``q``."""
    reservation = S - q * gamma * sigma ** 2 * tau
    half_spread = (1.0 / gamma) * torch.log1p(torch.tensor(gamma / kappa)) \
        + 0.5 * gamma * sigma ** 2 * tau
    return reservation - half_spread, reservation + half_spread


def simulate_day(gamma: float = 0.1, sigma: float = 2.0, kappa: float = 1.5,
                 A: float = 140.0, T: float = 1.0, steps: int = 2000,
                 symmetric: bool = False, seed: int = 0) -> dict:
    """Simulate one trading day; return P&L / inventory diagnostics.

    A fill on a side occurs in ``dt`` with probability ~ lambda(delta)*dt,
    where delta is that side's quote distance from mid. On a fill the inventory
    moves by one unit and cash changes by the executed price.
    """
    g = torch.Generator().manual_seed(seed)
    dt = T / steps
    S = torch.tensor(100.0)
    q = torch.tensor(0.0)      # inventory
    cash = torch.tensor(0.0)
    sigma_step = sigma * (dt ** 0.5)
    for i in range(steps):
        tau = T - i * dt
        if symmetric:
            base = (1.0 / gamma) * torch.log1p(torch.tensor(gamma / kappa))
            bid, ask = S - base, S + base
        else:
            bid, ask = as_quotes(S, q, gamma, sigma, kappa, tau)
        d_bid = (S - bid).clamp(min=0.0)
        d_ask = (ask - S).clamp(min=0.0)
        # fill probabilities this step
        p_bid = float(A * torch.exp(-kappa * d_bid) * dt)
        p_ask = float(A * torch.exp(-kappa * d_ask) * dt)
        if torch.rand(1, generator=g).item() < min(p_bid, 1.0):   # we buy at bid
            q = q + 1.0
            cash = cash - bid
        if torch.rand(1, generator=g).item() < min(p_ask, 1.0):   # we sell at ask
            q = q - 1.0
            cash = cash + ask
        S = S + sigma_step * torch.randn(1, generator=g).squeeze()
    pnl = cash + q * S                      # mark inventory to current mid
    return {"pnl": float(pnl), "inventory": float(q), "mid": float(S)}


if __name__ == "__main__":
    runs = 200
    opt = [simulate_day(symmetric=False, seed=s)["pnl"] for s in range(runs)]
    sym = [simulate_day(symmetric=True, seed=s)["pnl"] for s in range(runs)]
    opt_t, sym_t = torch.tensor(opt), torch.tensor(sym)
    print("Avellaneda–Stoikov closed-form quotes (inventory -5..5):")
    q = torch.arange(-5, 6).float()
    bid, ask = as_quotes(torch.tensor(100.0), q, 0.1, 2.0, 1.5, 1.0)
    for qi, bi, ai in zip(q.tolist(), bid.tolist(), ask.tolist()):
        print(f"  q={qi:+.0f}  bid={bi:7.3f}  ask={ai:7.3f}")
    print(f"\nOptimal  P&L: mean={opt_t.mean():.2f}  std={opt_t.std():.2f}")
    print(f"Symmetric P&L: mean={sym_t.mean():.2f}  std={sym_t.std():.2f}")
    print("Optimal policy mean-reverts inventory -> lower P&L variance.")
