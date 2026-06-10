// Week 7 -- COS pricer / implied-vol implementation. See cos.hpp.
#include "cos.hpp"

#include <cmath>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Standard normal CDF via erfc (no external deps).
double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Fang-Oosterlee chi_k(c,d): cosine coefficient of e^y on [c,d].
double chi(int k, double c, double d, double a, double b) {
    const double w = k * M_PI / (b - a);
    const double t1 = std::cos(w * (d - a)) * std::exp(d)
                    - std::cos(w * (c - a)) * std::exp(c);
    const double t2 = w * (std::sin(w * (d - a)) * std::exp(d)
                         - std::sin(w * (c - a)) * std::exp(c));
    return (t1 + t2) / (1.0 + w * w);
}

// Fang-Oosterlee psi_k(c,d): cosine coefficient of 1 on [c,d].
double psi(int k, double c, double d, double a, double b) {
    if (k == 0) return d - c;
    const double w = k * M_PI / (b - a);
    return (std::sin(w * (d - a)) - std::sin(w * (c - a))) / w;
}

}  // namespace

double bs_call_analytic(double S, double K, double T, double r, double sigma) {
    const double srt = sigma * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / srt;
    const double d2 = d1 - srt;
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double cos_call(double S, double K, double T, double r, double sigma, int Ncos) {
    // Cumulants of the log-return r_T ~ Normal((r-0.5 sig^2)T, sig^2 T).
    const double c1 = (r - 0.5 * sigma * sigma) * T;
    const double c2 = sigma * sigma * T;
    const double L = 10.0;                      // truncation width in std devs
    const double a = c1 - L * std::sqrt(c2);
    const double b = c1 + L * std::sqrt(c2);
    const double x = std::log(S / K);           // scaled log-spot
    const std::complex<double> I(0.0, 1.0);

    double sum = 0.0;
    for (int k = 0; k < Ncos; ++k) {
        const double w = k * M_PI / (b - a);
        // Characteristic function of the log-return evaluated at u = w.
        const std::complex<double> phi = std::exp(I * w * c1 - 0.5 * c2 * w * w);
        // Payoff cosine coefficient for a call: U_k = 2/(b-a) (chi - psi) on [0,b].
        const double Uk = (2.0 / (b - a)) * (chi(k, 0.0, b, a, b)
                                           - psi(k, 0.0, b, a, b));
        double re = std::real(phi * std::exp(I * w * (x - a)) * Uk);
        if (k == 0) re *= 0.5;                  // first term carries weight 1/2
        sum += re;
    }
    return K * std::exp(-r * T) * sum;
}

double implied_vol(double price, double S, double K, double T, double r,
                   double guess) {
    double sigma = guess;
    for (int it = 0; it < 100; ++it) {
        const double diff = bs_call_analytic(S, K, T, r, sigma) - price;
        if (std::fabs(diff) < 1e-10) break;
        const double srt = sigma * std::sqrt(T);
        const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / srt;
        const double vega = S * std::sqrt(T)
                          * std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);
        if (vega < 1e-12) break;                // safeguard against flat vega
        sigma -= diff / vega;
        if (sigma < 1e-6) sigma = 1e-6;         // keep positive
    }
    return sigma;
}

std::vector<double> price_surface(const std::vector<double>& strikes, double S,
                                  double T, double r, double sigma, int Ncos) {
    std::vector<double> out(strikes.size());
    for (std::size_t j = 0; j < strikes.size(); ++j)
        out[j] = cos_call(S, strikes[j], T, r, sigma, Ncos);
    return out;
}
