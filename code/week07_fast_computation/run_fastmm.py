"""
Week 7 -- exercise the C++ `fastmm` module from Python (Milestone 2).

Build the module first (see cpp/CMakeLists.txt):

    cd cpp
    cmake -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
    cmake --build build
    cd ..
    PYTHONPATH=cpp/build python run_fastmm.py

The script prices a small surface with the C++ COS pricer and validates it
against a pure-Python Black-Scholes reference (the "autograd is the oracle,
C++ is the speed" principle from lecture). If the module is not built yet it
explains how to build it and falls back to the Python reference.
"""
from __future__ import annotations

import math
import sys


def bs_call_ref(S, K, T, r, sigma):
    """Pure-Python analytic Black-Scholes call (the reference)."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    ncdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)


def main() -> int:
    try:
        import fastmm
    except ImportError:
        print("The 'fastmm' module is not built yet. Build it with:")
        print("  cd cpp && cmake -B build -Dpybind11_DIR=$(python -m pybind11 "
              "--cmakedir) && cmake --build build")
        print("  cd .. && PYTHONPATH=cpp/build python run_fastmm.py")
        print("\nShowing the Python reference prices instead:")
        for K in (80, 90, 100, 110, 120):
            print(f"  K={K:3d}  ref={bs_call_ref(100, K, 1.0, 0.0, 0.6):.6f}")
        return 0

    S, T, r, sigma = 100.0, 1.0, 0.0, 0.6
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    cos_prices = fastmm.price_surface(strikes, S, T, r, sigma)

    print(f"{'strike':>8} {'C++ COS':>14} {'py reference':>14} {'abs.err':>10}")
    max_err = 0.0
    for K, c in zip(strikes, cos_prices):
        ref = bs_call_ref(S, K, T, r, sigma)
        err = abs(c - ref)
        max_err = max(max_err, err)
        print(f"{K:8.1f} {c:14.8f} {ref:14.8f} {err:10.2e}")
    print(f"max abs error vs reference: {max_err:.2e}")

    iv = fastmm.implied_vol(fastmm.bs_call(S, 100.0, T, r, 0.55), S, 100.0, T, r)
    print(f"implied_vol round trip: input=0.5500 recovered={iv:.4f}")
    assert max_err < 1e-4, "COS prices must match the reference within 1e-4"
    print("C++/pybind11 module validated against the Python reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
