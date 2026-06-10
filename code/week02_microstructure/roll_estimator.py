"""
Week 2 — The Roll (1984) effective-spread estimator.

Roll's insight: in an efficient market with a constant effective spread ``s``,
bid-ask bounce induces *negative* serial covariance in consecutive price
changes, with cov(dP_t, dP_{t-1}) = -s^2 / 4. Hence

    s_hat = 2 * sqrt( -cov(dP_t, dP_{t-1}) )            (when the cov is < 0)

The estimator recovers the *unobserved* effective spread from trade prices
alone; it typically differs from the quoted spread, and that gap is the
interesting finding (adverse selection + price impact), not a bug.

Runs on a synthetic trade-price series:

    python roll_estimator.py
"""
from __future__ import annotations

import numpy as np


def roll_spread(prices: np.ndarray) -> float:
    """Roll effective-spread estimate from a series of trade prices."""
    dp = np.diff(np.asarray(prices, dtype=float))
    # lag-1 autocovariance of price changes
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    return 2.0 * np.sqrt(-cov) if cov < 0 else 0.0


def simulate_trades(n: int = 20000, true_spread: float = 0.10,
                    sigma: float = 0.02, seed: int = 0) -> np.ndarray:
    """Efficient price + bid-ask bounce -> observed trade prices."""
    rng = np.random.default_rng(seed)
    efficient = 100.0 + np.cumsum(rng.normal(0, sigma, n))   # random-walk value
    direction = rng.choice([-1.0, 1.0], size=n)              # buyer/seller initiated
    return efficient + direction * (true_spread / 2.0)       # observed prints


if __name__ == "__main__":
    true_spread = 0.10
    prices = simulate_trades(true_spread=true_spread)
    est = roll_spread(prices)
    print(f"true effective spread:  {true_spread:.4f}")
    print(f"Roll estimate:          {est:.4f}")
    print(f"relative error:         {abs(est - true_spread) / true_spread:6.2%}")
    print("Note: on real Coincall data the Roll estimate will differ from the")
    print("quoted spread; the difference reflects adverse selection / impact.")
