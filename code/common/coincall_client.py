"""
Coincall REST API client (documented starter).

This is a *minimal, readable* client for the Coincall exchange REST API,
intended for the NC State market-making program. It implements the signed
request scheme and thin wrappers around the market-data endpoints you use
across the weeks.

Authentication (per https://docs.coincall.com/):
  - Header ``X-CC-APIKEY``  : your API key
  - Header ``sign``         : HMAC-SHA256 of the canonical payload, UPPER hex
  - Header ``ts``           : current Unix time in milliseconds
  - Header ``X-REQ-TS-DIFF``: request validity window in ms (e.g. 5000)
  The canonical payload sorts the query parameters alphabetically, then
  appends ``uuid``, ``ts`` and ``x-req-ts-diff``, and is signed with your
  API secret.

IMPORTANT
  * Public market-data endpoints frequently work *without* a signature; the
    signed path here is required for account/private endpoints and is shown
    so you understand the scheme.
  * Endpoint paths and response field names are taken from the docs as of
    2026 and **must be re-confirmed against the live documentation** before
    you rely on them — the API evolves.

Usage:
    export COINCALL_API_KEY=...      export COINCALL_API_SECRET=...
    python coincall_client.py        # prints the BTC option instruments
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid
from typing import Any, Dict, Optional

import requests

BASE_URL = "https://api.coincall.com"
DEFAULT_TIMEOUT = 10.0          # seconds
DEFAULT_WINDOW_MS = "5000"      # X-REQ-TS-DIFF


class CoincallClient:
    """A tiny signed-REST client. One instance per API key.

    Parameters
    ----------
    api_key, api_secret:
        Your Coincall credentials. May be empty for public endpoints.
    base_url:
        Override for testing or for a regional host.
    """

    def __init__(self, api_key: str = "", api_secret: str = "",
                 base_url: str = BASE_URL) -> None:
        self.api_key = api_key or os.environ.get("COINCALL_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("COINCALL_API_SECRET", "")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    # -- signing -----------------------------------------------------------
    def _headers(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Build the signed headers for a request with query ``params``."""
        ts = str(int(time.time() * 1000))
        # canonical payload: params sorted by key, then uuid/ts/window appended
        canonical = "&".join(f"{k}={params[k]}" for k in sorted(params))
        canonical += f"&uuid={uuid.uuid4().hex}&ts={ts}&x-req-ts-diff={DEFAULT_WINDOW_MS}"
        signature = hmac.new(self.api_secret.encode(), canonical.encode(),
                             hashlib.sha256).hexdigest().upper()
        return {
            "X-CC-APIKEY": self.api_key,
            "sign": signature,
            "ts": ts,
            "X-REQ-TS-DIFF": DEFAULT_WINDOW_MS,
        }

    # -- core request ------------------------------------------------------
    def get(self, path: str, params: Optional[Dict[str, Any]] = None,
            signed: bool = True) -> Any:
        """GET ``path`` and return parsed JSON.

        Set ``signed=False`` for public market-data endpoints.
        """
        params = params or {}
        headers = self._headers(params) if signed else {}
        url = self.base_url + path
        resp = self.session.get(url, params=params, headers=headers,
                                timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    # -- market-data convenience wrappers ----------------------------------
    # Paths per docs.coincall.com (confirm before relying on them).
    def option_instruments(self, base_currency: str = "BTC") -> Any:
        """All listed option instruments for a base currency (BTC, ETH...)."""
        return self.get(f"/open/option/getInstruments/{base_currency}")

    def option_chain(self, index: str = "BTC") -> Any:
        """Option chain for an underlying index (mark, IV and Greeks per strike)."""
        return self.get(f"/open/option/get/v1/{index}")

    def option_orderbook(self, symbol: str) -> Any:
        """100-level L2 order book for a single option ``symbol``."""
        return self.get(f"/open/option/order/orderbook/v1/{symbol}")

    def option_detail(self, symbol: str) -> Any:
        """Mark price, implied vol and Greeks (delta/gamma/vega/theta/rho)."""
        return self.get(f"/open/option/detail/v1/{symbol}")

    def option_last_trades(self, symbol: str) -> Any:
        """Most recent trades for an option ``symbol``."""
        return self.get(f"/open/option/trade/lasttrade/v1/{symbol}")

    def futures_orderbook(self, symbol: str) -> Any:
        """L2 order book for a perpetual/futures ``symbol`` (the hedge venue)."""
        return self.get(f"/open/futures/order/orderbook/v1/{symbol}")

    def funding_rate(self, symbol: str = "BTCUSD") -> Any:
        """Current perpetual funding rate (needed for hedge-cost accounting)."""
        return self.get("/open/public/fundingRate/v1", {"symbol": symbol},
                        signed=False)

    def klines(self, option_name: str, period: str = "h1",
               start: Optional[int] = None, end: Optional[int] = None) -> Any:
        """Historical candlesticks. ``period`` in m1,m5,m15,m30,h1,h4,d1,..."""
        params: Dict[str, Any] = {"period": period}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        return self.get(f"/open/option/market/kline/history/v1/{option_name}",
                        params, signed=False)


def best_bid_ask(orderbook_json: Any) -> tuple[float, float]:
    """Extract (best_bid, best_ask) from an order-book response.

    The exact JSON shape varies; this expects a ``data`` object with ``bids``
    and ``asks`` arrays of ``[price, size]``. Adjust to the live schema.
    """
    data = orderbook_json["data"] if "data" in orderbook_json else orderbook_json
    bids = sorted((float(p) for p, *_ in data["bids"]), reverse=True)
    asks = sorted(float(p) for p, *_ in data["asks"])
    return bids[0], asks[0]


if __name__ == "__main__":
    client = CoincallClient()
    if not client.api_key:
        print("No COINCALL_API_KEY set. Public endpoints may still work; "
              "set credentials to exercise signed endpoints.")
    try:
        instruments = client.option_instruments("BTC")
        # Print a compact preview without assuming the exact schema.
        preview = str(instruments)
        print("BTC option instruments (preview):", preview[:400])
    except Exception as exc:  # network / auth / schema differences
        print("Request failed (expected offline or without keys):", repr(exc))
        print("This is fine for local development — the week scripts run on "
              "synthetic data and do not require network access.")
