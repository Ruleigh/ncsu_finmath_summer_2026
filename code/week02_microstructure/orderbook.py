"""
Week 2 — Limit order book reconstruction (L2).

Builds and maintains an aggregated (L2) order book from a stream of events
(new / cancel / modify / trade) and exposes mid-price and spread. Coincall
provides L2 (aggregated depth) data only — there is no per-order (L3) feed —
so we track size per price level, not individual queue position.

Runs out-of-the-box on a synthetic event stream:

    python orderbook.py

To use real Coincall data, replace ``synthetic_events`` with events parsed
from the provided Parquet snapshots, or poll
``CoincallClient.option_orderbook(symbol)`` (see ../common/coincall_client.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple


@dataclass
class Event:
    """A single order-book event."""
    kind: str          # "new" | "cancel" | "modify" | "trade"
    side: str          # "bid" | "ask"
    price: float
    size: float        # absolute size (new/modify) or size removed (cancel/trade)


@dataclass
class L2Book:
    """Aggregated limit order book keyed by price level."""
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)

    def _book(self, side: str) -> Dict[float, float]:
        return self.bids if side == "bid" else self.asks

    def apply(self, ev: Event) -> None:
        """Apply one event, keeping the book consistent."""
        book = self._book(ev.side)
        if ev.kind in ("new", "modify"):
            book[ev.price] = ev.size
        elif ev.kind in ("cancel", "trade"):
            remaining = book.get(ev.price, 0.0) - ev.size
            if remaining > 1e-12:
                book[ev.price] = remaining
            else:
                # IMPORTANT: drop empty levels, else best bid/ask go stale
                book.pop(ev.price, None)

    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return None if bb is None or ba is None else 0.5 * (bb + ba)

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return None if bb is None or ba is None else ba - bb


def synthetic_events(n: int = 2000, seed: int = 0) -> Iterable[Event]:
    """Generate a coherent synthetic event stream around a drifting mid.

    The generator tracks resting sizes so it can emit exact cancels for levels
    that the drifting mid would cross -- this keeps the book *uncrossed*
    (best bid < mid < best ask), as a real feed is.
    """
    rng = random.Random(seed)
    mid = 100.0
    bid_sz: Dict[float, float] = {}
    ask_sz: Dict[float, float] = {}

    def place(side: str, price: float, size: float) -> Event:
        (bid_sz if side == "bid" else ask_sz)[price] = size
        return Event("new", side, price, size)

    for i in range(1, 6):                              # seed an initial book
        yield place("bid", round(mid - 0.1 * i, 2), rng.uniform(1, 10))
        yield place("ask", round(mid + 0.1 * i, 2), rng.uniform(1, 10))

    for _ in range(n):
        mid += rng.gauss(0, 0.02)                      # random-walk mid
        # cancel (full size) any level the mid has crossed
        for p in [p for p in bid_sz if p >= mid]:
            yield Event("cancel", "bid", p, bid_sz.pop(p))
        for p in [p for p in ask_sz if p <= mid]:
            yield Event("cancel", "ask", p, ask_sz.pop(p))
        # post a fresh level strictly on the correct side of mid
        off = 0.1 * rng.randint(1, 5)
        if rng.random() < 0.5:
            p = round(mid - off, 2)
            if p < mid:
                yield place("bid", p, rng.uniform(0.5, 5))
        else:
            p = round(mid + off, 2)
            if p > mid:
                yield place("ask", p, rng.uniform(0.5, 5))


def run_stream(events: Iterable[Event]) -> Tuple[L2Book, list]:
    """Replay events; return the final book and a (mid, spread) time series."""
    book = L2Book()
    series = []
    for ev in events:
        book.apply(ev)
        m, s = book.mid(), book.spread()
        if m is not None:
            series.append((m, s))
    return book, series


if __name__ == "__main__":
    book, series = run_stream(synthetic_events())
    mids = [m for m, _ in series]
    spreads = [s for _, s in series if s is not None]
    print(f"events processed, samples: {len(series)}")
    print(f"final mid:    {book.mid():.3f}")
    print(f"final spread: {book.spread():.3f}")
    print(f"mean mid:     {sum(mids)/len(mids):.3f}")
    print(f"mean spread:  {sum(spreads)/len(spreads):.4f}")
    # To dump for plotting:  csv.writer(...).writerows(series)
