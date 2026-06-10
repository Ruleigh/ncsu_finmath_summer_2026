// Week 7 -- standalone demo for the COS pricer (no Python needed).
//
// Builds to `cos_demo`. It prices a few options with the COS method, compares
// against the analytic Black-Scholes price (self-check), and round-trips an
// implied volatility.
#include "cos.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const double S = 100.0, T = 1.0, r = 0.0, sigma = 0.6;  // crypto-like IV
    const std::vector<double> strikes = {80, 90, 100, 110, 120};

    std::printf("%-8s %-14s %-14s %-12s\n", "strike", "COS", "analytic", "abs.err");
    double max_err = 0.0;
    for (double K : strikes) {
        const double c = cos_call(S, K, T, r, sigma);
        const double a = bs_call_analytic(S, K, T, r, sigma);
        const double err = std::fabs(c - a);
        max_err = std::max(max_err, err);
        std::printf("%-8.1f %-14.8f %-14.8f %-12.2e\n", K, c, a, err);
    }
    std::printf("max |COS - analytic| = %.2e\n", max_err);

    // implied-vol round trip
    const double price = bs_call_analytic(S, 100.0, T, r, 0.55);
    const double iv = implied_vol(price, S, 100.0, T, r);
    std::printf("implied_vol round trip: input=0.5500  recovered=%.4f\n", iv);

    return (max_err < 1e-4) ? 0 : 1;  // nonzero exit signals a regression
}
