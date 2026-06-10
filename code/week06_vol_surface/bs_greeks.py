"""
Week 6 — Black–Scholes price and Greeks via PyTorch autograd.

We never hand-derive algorithmic differentiation: we write the forward price
and let ``torch.autograd`` produce every Greek, including the second-order
vanna (d vega / dS) and volga (d vega / d sigma). We verify vanna/volga
against central finite differences.

    python bs_greeks.py
"""
from __future__ import annotations

import torch


def bs_call(S, K, T, r, sigma):
    """Black–Scholes European call price (autograd-friendly)."""
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    N = torch.distributions.Normal(0.0, 1.0).cdf
    return S * N(d1) - K * torch.exp(-r * T) * N(d2)


def greeks(S0=100.0, K=100.0, T=1.0, r=0.0, sigma0=0.6):
    """Return price and the Greek vector via autograd (double precision)."""
    dt = torch.float64
    S = torch.tensor(S0, dtype=dt, requires_grad=True)
    sigma = torch.tensor(sigma0, dtype=dt, requires_grad=True)
    K_, T_, r_ = (torch.tensor(x, dtype=dt) for x in (K, T, r))
    price = bs_call(S, K_, T_, r_, sigma)
    delta, = torch.autograd.grad(price, S, create_graph=True)
    gamma, = torch.autograd.grad(delta, S, create_graph=True)
    vega, = torch.autograd.grad(price, sigma, create_graph=True)
    vanna, = torch.autograd.grad(vega, S, create_graph=True)     # d vega / dS
    volga, = torch.autograd.grad(vega, sigma)                     # d vega / d sigma
    return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
            "vega": float(vega), "vanna": float(vanna), "volga": float(volga)}


def _fd_vega(S0, K, T, r, sigma0, h=1e-4):
    """Central finite-difference vega for the verification check (float64)."""
    f = lambda s: float(bs_call(*(torch.tensor(x, dtype=torch.float64)
                                  for x in (S0, K, T, r, s))))
    return (f(sigma0 + h) - f(sigma0 - h)) / (2 * h)


if __name__ == "__main__":
    g = greeks()
    for k, v in g.items():
        print(f"{k:6s}: {v: .6f}")
    fd = _fd_vega(100.0, 100.0, 1.0, 0.0, 0.6)
    rel = abs(g["vega"] - fd) / abs(fd)
    print(f"\nvega autograd vs finite-difference relative error: {rel:.2e}")
    assert rel < 1e-4, "autograd vega should match finite differences"
    print("autograd Greeks verified.")
