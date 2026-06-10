// Week 7 -- Fast option pricing in C++ (Fang-Oosterlee COS method).
//
// Declarations for a Black-Scholes COS European-call pricer, the analytic
// Black-Scholes price (used as a self-check), and a robust implied-vol solver.
// These are pure C++ (no Python dependency); bindings.cpp wraps them with
// pybind11 so they are callable from Python as the `fastmm` module.
#pragma once
#include <vector>

// Analytic Black-Scholes European call (closed form; the COS self-check).
double bs_call_analytic(double S, double K, double T, double r, double sigma);

// Fang-Oosterlee COS European call. `Ncos` is the number of cosine terms;
// 256 reaches ~1e-8 for Black-Scholes. The truncation range is derived from
// the cumulants of the log-return (not guessed).
double cos_call(double S, double K, double T, double r, double sigma,
                int Ncos = 256);

// Implied volatility by a safeguarded Newton iteration on the analytic price.
double implied_vol(double price, double S, double K, double T, double r,
                   double guess = 0.5);

// Vectorized surface revaluation (one COS price per strike).
std::vector<double> price_surface(const std::vector<double>& strikes, double S,
                                  double T, double r, double sigma,
                                  int Ncos = 256);
