"""
Week 8 — Multi-Greek inventory and a threshold delta hedge.

Options market making is Avellaneda–Stoikov with a *vector-valued* inventory:
the book's risk is the aggregated Greek vector q = (delta, gamma, vega, ...),
penalized by a quadratic form (1/2) q^T Sigma q. Here we:

  1. aggregate per-contract Greeks into the book's inventory vector,
  2. run a threshold delta-hedger against a synthetic price path, and
  3. numerically illustrate the gamma–theta–variance identity
        dPi = (1/2) * Gamma * S^2 * (realized_var - implied_var) * dt
     which explains most of a delta-hedged book's P&L.

    python multi_greek_hedge.py
"""
from __future__ import annotations

import torch


def aggregate_greeks(positions: torch.Tensor, greeks: torch.Tensor) -> torch.Tensor:
    """positions: (n,) contract quantities; greeks: (n, k) per-contract Greeks.

    Returns the book's aggregated Greek vector (k,).
    """
    return (positions.unsqueeze(1) * greeks).sum(dim=0)


def threshold_hedge(net_delta: float, perp_position: float,
                    threshold: float = 0.1) -> tuple[float, float]:
    """Rebalance the perp hedge only when |net delta| breaches the threshold.

    Returns (new_perp_position, hedge_trade).
    """
    if abs(net_delta) > threshold:
        trade = -net_delta                       # offset the directional risk
        return perp_position + trade, trade
    return perp_position, 0.0


def gamma_theta_pnl(gamma: float, S: float, realized_var: float,
                    implied_var: float, dt: float) -> float:
    """One step of the gamma–theta–variance identity for a hedged book."""
    return 0.5 * gamma * S ** 2 * (realized_var - implied_var) * dt


def simulate_hedged_book(steps=2000, gamma=0.02, implied_vol=0.6,
                         realized_vol=0.7, S0=100.0, seed=0) -> dict:
    """Accumulate the gamma–theta P&L of a continuously delta-hedged position."""
    g = torch.Generator().manual_seed(seed)
    dt = 1.0 / steps
    S = S0
    pnl = 0.0
    iv2, rv2 = implied_vol ** 2, realized_vol ** 2
    for _ in range(steps):
        pnl += gamma_theta_pnl(gamma, S, rv2, iv2, dt)
        S = S * (1.0 + realized_vol * dt ** 0.5 * float(torch.randn(1, generator=g)))
    return {"pnl": pnl, "final_S": S,
            "sign_matches_var_gap": (pnl > 0) == (realized_vol > implied_vol)}


if __name__ == "__main__":
    # 1) aggregate Greeks of a small book: long 2 calls, short 1 call
    positions = torch.tensor([2.0, -1.0])
    #            [delta, gamma, vega, theta, rho]
    greeks = torch.tensor([[0.55, 0.02, 0.30, -0.05, 0.10],
                           [0.40, 0.03, 0.28, -0.04, 0.09]])
    q = aggregate_greeks(positions, greeks)
    print("book Greek vector [delta,gamma,vega,theta,rho]:",
          [round(float(x), 3) for x in q])

    # 2) hedge if net delta exceeds 0.1
    new_perp, trade = threshold_hedge(float(q[0]), perp_position=0.0)
    print(f"net delta={float(q[0]):.3f} -> hedge trade={trade:.3f}, perp={new_perp:.3f}")

    # 3) gamma–theta–variance identity: long gamma earns when realized > implied
    res = simulate_hedged_book(realized_vol=0.7, implied_vol=0.6)
    print(f"hedged-book P&L (realized>implied): {res['pnl']:.2f} "
          f"(sign matches variance gap: {res['sign_matches_var_gap']})")
