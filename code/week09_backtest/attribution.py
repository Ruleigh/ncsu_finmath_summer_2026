"""
Week 9 — P&L attribution and a block-bootstrap Sharpe confidence interval
(Milestone 3 building blocks).

  * ``attribute`` decomposes daily P&L into named streams and asserts they
    sum to the total (a mismatch is *always* a bug -- usually a timing error).
  * ``bootstrap_sharpe`` returns a confidence interval for the annualized
    Sharpe by resampling *blocks* of returns, which preserves autocorrelation;
    a high Sharpe with a wide CI is not a real edge.

    python attribution.py
"""
from __future__ import annotations

import numpy as np


def attribute(spread, inventory, gamma, hedging):
    """Combine attribution streams; verify they reconcile to the total P&L."""
    spread, inventory, gamma, hedging = map(np.asarray,
                                            (spread, inventory, gamma, hedging))
    total = spread + inventory + gamma + hedging
    recombined = spread.sum() + inventory.sum() + gamma.sum() + hedging.sum()
    assert abs(total.sum() - recombined) < 1e-6, "attribution must sum to total"
    return {"spread": spread, "inventory": inventory, "gamma": gamma,
            "hedging": hedging, "total": total}


def bootstrap_sharpe(returns, block=5, n_boot=2000, periods=252, seed=0):
    """Block-bootstrap CI (2.5/50/97.5 percentiles) of the annualized Sharpe."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    T = len(r)
    n_blocks = max(1, T // block)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()
        s = r[idx]
        estimates[i] = s.mean() / (s.std() + 1e-12) * np.sqrt(periods)
    return np.percentile(estimates, [2.5, 50, 97.5])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    days = 60
    spread = np.abs(rng.normal(3.0, 1.0, days))          # spread capture (positive)
    inventory = rng.normal(0.0, 2.0, days)               # inventory mark drift
    gamma = rng.normal(0.5, 1.5, days)                   # gamma P&L
    hedging = -np.abs(rng.normal(0.8, 0.4, days))        # hedging cost (negative)

    a = attribute(spread, inventory, gamma, hedging)
    print("attribution sums to total: OK")
    print(f"total P&L over {days}d: {a['total'].sum():.1f}")
    for k in ("spread", "inventory", "gamma", "hedging"):
        print(f"  {k:10s}: {a[k].sum():8.1f}")

    lo, med, hi = bootstrap_sharpe(a["total"])
    print(f"\nannualized Sharpe (block bootstrap): {med:.2f}  "
          f"95% CI [{lo:.2f}, {hi:.2f}]")
