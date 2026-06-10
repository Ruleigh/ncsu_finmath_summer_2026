"""
Week 10 — Structured-product payoffs (Fenni Kang / Coincall guest session)
and a sketch of multi-asset (BTC + ETH) inventory risk.

Corporate structured products generate the flow that lands on a market
maker's book. We implement the canonical payoffs at expiry:

  * collar       : long underlying + long put(K_put) + short call(K_call)
                   -> fixes a price band (floored downside, capped upside);
  * accumulator  : client buys a fixed quantity while spot is in a range,
                   with an accelerated buy below a strike (stylized here).

We also show how a two-asset Greek inventory is penalized through a
correlated covariance matrix Sigma (the multi-asset extension).

    python structured_products.py
"""
from __future__ import annotations

import numpy as np


def collar_payoff(S, spot=100.0, put_K=90.0, call_K=110.0):
    """Payoff of a collar around a long underlying position."""
    S = np.asarray(S, dtype=float)
    return (S - spot) + np.maximum(put_K - S, 0.0) - np.maximum(S - call_K, 0.0)


def accumulator_payoff(S, strike=100.0, ko=115.0, leverage=2.0):
    """Stylized accumulator: buy 1x in range, 2x (accelerated) below strike."""
    S = np.asarray(S, dtype=float)
    qty = np.where(S < strike, leverage, 1.0)            # accelerate below strike
    qty = np.where(S > ko, 0.0, qty)                     # knocked out above KO
    return qty * (S - strike)


def multi_asset_risk(greeks, Sigma):
    """Quadratic inventory risk (1/2) q^T Sigma q for a multi-asset Greek vector."""
    q = np.asarray(greeks, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    return 0.5 * float(q @ Sigma @ q)


if __name__ == "__main__":
    S = np.linspace(60, 140, 9)
    print("Collar payoff (floored below 90, capped above 110):")
    for s, p in zip(S, collar_payoff(S)):
        print(f"  S={s:5.0f}  payoff={p:6.1f}")

    print("\nAccumulator payoff (accelerated buy below 100, KO at 115):")
    for s, p in zip(S, accumulator_payoff(S)):
        print(f"  S={s:5.0f}  payoff={p:6.1f}")

    # multi-asset: BTC & ETH delta exposures with 0.8 correlation
    greeks = [1.5, -0.8]                                  # net delta in BTC, ETH units
    vol_btc, vol_eth, rho = 0.6, 0.7, 0.8
    Sigma = [[vol_btc ** 2, rho * vol_btc * vol_eth],
             [rho * vol_btc * vol_eth, vol_eth ** 2]]
    print(f"\nmulti-asset inventory risk (1/2)q^T Sigma q = "
          f"{multi_asset_risk(greeks, Sigma):.4f}")
    print("Correlation couples BTC/ETH risk -- a diagonal penalty would understate it.")
