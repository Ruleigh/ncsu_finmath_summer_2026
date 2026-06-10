"""
Week 5 — Minimal event-driven backtest harness (Milestone 1 skeleton).

A deterministic, seedable backtest of the Avellaneda–Stoikov engine against a
synthetic mid-price path with probabilistic fills. Reports the standard
diagnostics and a basic P&L attribution (spread capture vs. inventory mark).
Replace ``synthetic_path`` with a Coincall replay to grade Milestone 1.

    python backtest.py
"""
from __future__ import annotations

import numpy as np


def as_quotes(S, q, gamma, sigma, kappa, tau):
    """Closed-form Avellaneda–Stoikov bid/ask (numpy, scalar)."""
    reservation = S - q * gamma * sigma ** 2 * tau
    half = (1.0 / gamma) * np.log1p(gamma / kappa) + 0.5 * gamma * sigma ** 2 * tau
    return reservation - half, reservation + half


def synthetic_path(steps=5000, sigma=2.0, dt=1.0 / 5000, seed=0):
    """Arithmetic-Brownian-motion mid path."""
    rng = np.random.default_rng(seed)
    incr = rng.normal(0, sigma * np.sqrt(dt), steps)
    return 100.0 + np.cumsum(incr)


def run_backtest(gamma=0.1, sigma=2.0, kappa=1.5, A=140.0, seed=0):
    """Event-driven loop; returns metrics and attribution streams."""
    path = synthetic_path(sigma=sigma, seed=seed)
    rng = np.random.default_rng(seed + 1)              # separate stream for fills
    steps = len(path)
    dt = 1.0 / steps
    q = 0.0
    cash = 0.0
    spread_pnl = 0.0                                    # realized half-spread captured
    daily_marks = []
    for i, S in enumerate(path):
        tau = max(1.0 - i * dt, 1e-6)
        bid, ask = as_quotes(S, q, gamma, sigma, kappa, tau)
        d_bid, d_ask = max(S - bid, 0.0), max(ask - S, 0.0)
        if rng.random() < min(A * np.exp(-kappa * d_bid) * dt, 1.0):
            q += 1.0; cash -= bid; spread_pnl += (S - bid)
        if rng.random() < min(A * np.exp(-kappa * d_ask) * dt, 1.0):
            q -= 1.0; cash += ask; spread_pnl += (ask - S)
        daily_marks.append(cash + q * S)
    equity = np.asarray(daily_marks)
    rets = np.diff(equity)
    total = float(equity[-1])
    inventory_pnl = total - spread_pnl                 # residual = inventory mark drift
    metrics = {
        "total_pnl": total,
        "spread_pnl": spread_pnl,
        "inventory_pnl": inventory_pnl,
        "final_inventory": q,
        "pnl_vol": float(rets.std() * np.sqrt(steps)),
        "sharpe": float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(steps)),
        "max_drawdown": float((equity - np.maximum.accumulate(equity)).min()),
    }
    return metrics


if __name__ == "__main__":
    m = run_backtest()
    for k, v in m.items():
        print(f"{k:16s}: {v:.4f}")
    # reproducibility check: same seed -> identical results
    assert run_backtest(seed=0) == m, "backtest must be deterministic under a seed"
    print("deterministic replay: OK (identical results on re-run)")
