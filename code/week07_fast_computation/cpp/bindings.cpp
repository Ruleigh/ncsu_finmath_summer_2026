// Week 7 -- pybind11 bindings: expose the C++ COS pricer to Python as `fastmm`.
//
// Demonstrates the binder pattern taught in lecture: a clean module surface,
// keyword arguments with defaults, and releasing the GIL around the C++ hot
// loop so other Python threads can run while we price a whole surface.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // automatic std::vector <-> Python list conversion

#include "cos.hpp"

namespace py = pybind11;

// Surface revaluation with the GIL released for the duration of the C++ loop.
static std::vector<double> price_surface_nogil(const std::vector<double>& strikes,
                                               double S, double T, double r,
                                               double sigma, int Ncos) {
    std::vector<double> out;
    {
        py::gil_scoped_release release;        // let other Python threads run
        out = price_surface(strikes, S, T, r, sigma, Ncos);
    }
    return out;
}

PYBIND11_MODULE(fastmm, m) {
    m.doc() = "Fast option pricing (Fang-Oosterlee COS) implemented in C++ "
              "and exposed via pybind11.";

    m.def("bs_call", &bs_call_analytic,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          "Analytic Black-Scholes European call price.");

    m.def("cos_call", &cos_call,
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          py::arg("Ncos") = 256,
          "COS European call price.");

    m.def("implied_vol", &implied_vol,
          py::arg("price"), py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("guess") = 0.5,
          "Implied volatility by safeguarded Newton iteration.");

    m.def("price_surface", &price_surface_nogil,
          py::arg("strikes"), py::arg("S"), py::arg("T"), py::arg("r"),
          py::arg("sigma"), py::arg("Ncos") = 256,
          "Vectorized COS prices for many strikes (GIL released).");
}
