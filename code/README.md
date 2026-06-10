# Course Code — Optimal Market Making for Cryptocurrency Options

Runnable, well-documented starter code for every week of the program.

- **Python** (problem sets): each script runs out-of-the-box on **synthetic
  data**, so you can execute it before you have any Coincall data, then swap
  in real data where the comments indicate.
- **C++** (Week 7, fast computation): a Fang–Oosterlee **COS** pricer and a
  robust **implied-vol** solver, built with **CMake** and exposed to Python
  via **pybind11**.
- **Coincall**: `common/coincall_client.py` is a documented signed-REST
  client (endpoints cross-checked against <https://docs.coincall.com/>).

## Layout

```
code/
  common/coincall_client.py        # signed REST + market-data helpers
  week02_microstructure/           # order book, Roll estimator
  week03_avellaneda_stoikov/       # closed-form quotes + simulation
  week04_control/                  # Merton HJB (CARA) value iteration
  week05_engine/                   # intensity MLE + event-driven backtest
  week06_vol_surface/              # BS Greeks via autograd, SVI calibration
  week07_fast_computation/cpp/     # C++ COS pricer + pybind11 module (CMake)
  week08_hedging/                  # multi-Greek inventory + delta hedge
  week09_backtest/                 # P&L attribution + bootstrap Sharpe
  week10_advanced/                 # structured-product payoffs, multi-asset
```

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run any Python example

```sh
python week03_avellaneda_stoikov/as_engine.py
python week06_vol_surface/bs_greeks.py
# ... every week*/ *.py file has a __main__ demo on synthetic data
```

## Build the Week 7 C++ / pybind11 module

```sh
cd week07_fast_computation/cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build
# standalone demo:
./build/cos_demo
# python module (import from the build dir, or copy the .so next to your code):
PYTHONPATH=build python -c "import fastmm; print(fastmm.bs_call(100,100,1,0,0.6))"
```

## Using real Coincall data

Set your credentials, then call the client:

```sh
export COINCALL_API_KEY=...      # from your Coincall account
export COINCALL_API_SECRET=...
python common/coincall_client.py     # lists BTC option instruments
```

Public market-data endpoints often work without signing; account endpoints
require the HMAC-SHA256 signature implemented in the client. **Always
re-confirm exact paths and field names against <https://docs.coincall.com/>**
— the API evolves.
