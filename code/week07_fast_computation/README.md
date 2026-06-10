# Week 7 — Fast Computation (C++ + pybind11)

A Fang–Oosterlee **COS** European-call pricer and a safeguarded-Newton
**implied-vol** solver, written in C++ and exposed to Python as `fastmm`.

## Files

- `cpp/cos.hpp`, `cpp/cos.cpp` — the pricer / implied-vol (pure C++).
- `cpp/main.cpp` — standalone demo (`cos_demo`); checks COS vs. analytic BS.
- `cpp/bindings.cpp` — pybind11 module surface (releases the GIL).
- `cpp/CMakeLists.txt` — builds both targets.
- `run_fastmm.py` — calls `fastmm` from Python and validates against a
  pure-Python reference.

## Build

```sh
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build
```

This produces:
- `build/cos_demo` — run `./build/cos_demo` (exit code 0 means COS matches
  the analytic price to < 1e-4).
- `build/fastmm.*.so` — the Python module.

## Use from Python

```sh
cd ..
PYTHONPATH=cpp/build python run_fastmm.py
```

## Why this split

In production a market maker prices hundreds of contracts many times per
second; a Python loop is too slow. The numerical kernel lives in C++ (cheap,
cache-friendly, GIL released), while Python orchestrates and **PyTorch
autograd provides the *reference* Greeks** — we do not hand-roll adjoint AD in
C++. `pybind11` is the thin, header-only bridge between the two.

> Replace the synthetic inputs with the calibrated Coincall surface
> (`coincall_calibrated_svi_surfaces_*.parquet`) and validate against
> `coincall_reference_prices_*.parquet`; confirm endpoints at
> <https://docs.coincall.com/>.
