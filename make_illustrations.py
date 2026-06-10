#!/usr/bin/env python
"""Generate one illustration per week for the student decks.

Tries Gemini image generation first (if GEMINI_API_KEY has image quota),
otherwise renders a precise matplotlib diagram. For a technical finance
course the plotted diagrams are the primary, accurate illustrations; the
Gemini path is kept so art can be swapped in when the plan allows.
"""
import os, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/Users/dendisuhubdy/Github/ncsu/weekly_presentations/figures"
os.makedirs(OUT, exist_ok=True)

RED, DKRED, TEAL, GOLD, GRAY, NAVY = "#CC0000", "#990000", "#008473", "#FAC800", "#6D6D6D", "#427E93"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.edgecolor": GRAY, "axes.labelcolor": "#1a1a1a",
                     "text.color": "#1a1a1a", "xtick.color": GRAY, "ytick.color": GRAY,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
FIG = (9.0, 5.0)


def finish(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def try_gemini(prompt, name):
    """Return True if a Gemini image was generated and saved."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return False
    try:
        from google import genai
        from google.genai import types
        from PIL import Image
        client = genai.Client(api_key=key)
        r = client.models.generate_content(
            model="gemini-2.5-flash-image", contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]))
        for part in r.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                Image.open(io.BytesIO(part.inline_data.data)).save(os.path.join(OUT, name))
                print("Gemini wrote", name)
                return True
    except Exception as e:
        print("Gemini unavailable (%s) -> matplotlib fallback for %s" % (str(e)[:60], name))
    return False


# ---------------------------------------------------------------- Week 1
def w1():
    fig, ax = plt.subplots(figsize=FIG)
    weeks = list(range(1, 11))
    titles = ["Intro", "Microstructure", "Avellaneda\n-Stoikov", "Control",
              "AS Engine", "Vol Surface", "Fast Comp.", "Options MM", "Backtest", "Advanced"]
    ax.plot([1, 10], [0, 0], color=GRAY, lw=2, zorder=1)
    for w, t in zip(weeks, titles):
        ms = w in (5, 7, 9)
        ax.scatter([w], [0], s=420 if ms else 260, color=RED if ms else "white",
                   edgecolor=RED, lw=2.5, zorder=3)
        ax.text(w, 0, str(w), ha="center", va="center",
                color="white" if ms else RED, fontweight="bold", zorder=4)
        ax.text(w, 0.16 if w % 2 else 0.30, t, ha="center", va="bottom", fontsize=9, color="#1a1a1a")
        if ms:
            ax.text(w, -0.22, "Milestone", ha="center", va="top", fontsize=8, color=DKRED, fontweight="bold")
    ax.text(1, -0.40, "MAKER (this course): quote both sides, earn the spread",
            fontsize=9, color=TEAL, fontweight="bold")
    ax.text(1, -0.52, "TAKER (Wk 9 guest): cross the spread to execute",
            fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlim(0.3, 10.7); ax.set_ylim(-0.65, 0.6); ax.axis("off")
    ax.set_title("Program Roadmap — 10 Weeks", color=DKRED, fontweight="bold", fontsize=15)
    finish(fig, "week01.png")


# ---------------------------------------------------------------- Week 2
def w2():
    fig, ax = plt.subplots(figsize=FIG)
    bid_px = np.array([99.5, 99.4, 99.3, 99.2, 99.1])
    ask_px = np.array([100.5, 100.6, 100.7, 100.8, 100.9])
    bid_sz = np.array([3, 5, 8, 6, 9]); ask_sz = np.array([4, 6, 7, 5, 8])
    ax.barh(bid_px, -bid_sz, height=0.07, color=TEAL, alpha=.85, label="Bids")
    ax.barh(ask_px, ask_sz, height=0.07, color=RED, alpha=.85, label="Asks")
    ax.axhline(100.0, color=GRAY, ls="--", lw=1.5)
    ax.text(0, 100.0, " mid-price", va="center", ha="left", color=GRAY, fontsize=10)
    ax.annotate("", xy=(0, 99.5), xytext=(0, 100.5),
                arrowprops=dict(arrowstyle="<->", color=DKRED, lw=1.5))
    ax.text(0.4, 100.0, "spread", color=DKRED, rotation=90, va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("resting size (bid | ask)"); ax.set_ylabel("price")
    ax.set_title("Limit Order Book — L2 Aggregated Depth (Coincall)", color=DKRED, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    finish(fig, "week02.png")


# ---------------------------------------------------------------- Week 3
def w3():
    fig, ax = plt.subplots(figsize=FIG)
    q = np.linspace(-10, 10, 200); S = 100.0; gamma, sig2, tau = .1, 4.0, 1.0
    r = S - q * gamma * sig2 * tau
    half = (1/gamma) * np.log1p(gamma/1.5) + 0.5 * gamma * sig2 * tau
    ax.axhline(S, color=GRAY, ls="--", lw=1, label="mid-price S")
    ax.plot(q, r, color=DKRED, lw=2.5, label="reservation price r(q)")
    ax.fill_between(q, r - half, r + half, color=RED, alpha=.15, label="quoting band")
    ax.plot(q, r - half, color=RED, lw=1.2); ax.plot(q, r + half, color=RED, lw=1.2)
    ax.set_xlabel("inventory q"); ax.set_ylabel("price")
    ax.set_title("Avellaneda–Stoikov: Inventory-Skewed Quotes", color=DKRED, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    finish(fig, "week03.png")


# ---------------------------------------------------------------- Week 4
def w4():
    fig, ax = plt.subplots(figsize=FIG); ax.axis("off")
    steps = ["Controlled\ndynamics", "Dynamic\nprogramming", "HJB\nequation",
             "Separation\nansatz", "ODE\nreduction", "Closed-form\nquotes"]
    n = len(steps); x = np.linspace(0.05, 0.83, n)
    for i, (xi, s) in enumerate(zip(x, steps)):
        box = FancyBboxPatch((xi, 0.42), 0.12, 0.18, boxstyle="round,pad=0.01",
                             fc=RED if i in (2, 5) else "white", ec=RED, lw=2)
        ax.add_patch(box)
        ax.text(xi + 0.06, 0.51, s, ha="center", va="center", fontsize=9,
                color="white" if i in (2, 5) else "#1a1a1a", fontweight="bold")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((xi + 0.12, 0.51), (x[i+1], 0.51),
                         arrowstyle="-|>", mutation_scale=16, color=GRAY, lw=1.6))
    ax.text(0.5, 0.78, "The Stochastic-Control Pipeline", ha="center",
            color=DKRED, fontweight="bold", fontsize=15)
    ax.text(0.5, 0.25, "Verification theorem: a smooth HJB solution IS the value function",
            ha="center", color=TEAL, fontsize=10, fontstyle="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    finish(fig, "week04.png")


# ---------------------------------------------------------------- Week 5
def w5():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=FIG)
    opt = rng.normal(120, 45, 4000); sym = rng.normal(70, 80, 4000)
    ax.hist(sym, bins=50, color=GRAY, alpha=.55, label="symmetric baseline")
    ax.hist(opt, bins=50, color=RED, alpha=.65, label="Avellaneda–Stoikov")
    ax.axvline(opt.mean(), color=DKRED, lw=2); ax.axvline(sym.mean(), color="#444", lw=2, ls="--")
    ax.set_xlabel("daily P&L (USD)"); ax.set_ylabel("frequency")
    ax.set_title("Milestone 1: Backtest P&L — Optimal vs. Symmetric", color=DKRED, fontweight="bold")
    ax.legend(frameon=False)
    finish(fig, "week05.png")


# ---------------------------------------------------------------- Week 6
def w6():
    fig, ax = plt.subplots(figsize=FIG)
    k = np.linspace(-0.6, 0.6, 200)
    for T, c, lab in [(0.1, RED, "1w"), (0.25, DKRED, "1m"), (0.75, NAVY, "3m"), (1.5, TEAL, "6m")]:
        iv = 0.55 + 0.9*(k**2)/(T+0.2) - 0.25*k/np.sqrt(T+0.1)
        ax.plot(k, iv, lw=2.2, color=c, label=lab)
    ax.set_xlabel("log-moneyness  k = ln(K/F)"); ax.set_ylabel("implied volatility")
    ax.set_title("The Volatility Smile across Maturities (SVI/SABR)", color=DKRED, fontweight="bold")
    ax.legend(title="expiry", frameon=False)
    finish(fig, "week06.png")


# ---------------------------------------------------------------- Week 7
def w7():
    fig, ax = plt.subplots(figsize=FIG)
    labels = ["Naive\nPython loop", "NumPy\nvectorized", "C++ kernel\n(scalar)", "C++ + SIMD\nvia pybind11"]
    ms = [240.0, 38.0, 9.0, 3.2]
    bars = ax.bar(labels, ms, color=[GRAY, NAVY, TEAL, RED], width=.6)
    ax.axhline(5.0, color=DKRED, ls="--", lw=2); ax.text(3.4, 5.6, "5 ms target", color=DKRED, ha="right", fontsize=10, fontweight="bold")
    ax.set_yscale("log"); ax.set_ylabel("full-surface revaluation (ms, log scale)")
    ax.set_title("Milestone 2: Latency — C++ via pybind11 Meets the Budget", color=DKRED, fontweight="bold")
    for b, v in zip(bars, ms):
        ax.text(b.get_x()+b.get_width()/2, v*1.1, f"{v:g} ms", ha="center", fontsize=9)
    finish(fig, "week07.png")


# ---------------------------------------------------------------- Week 8
def w8():
    fig, ax = plt.subplots(figsize=FIG)
    t = np.linspace(0, 1, 250); rng = np.random.default_rng(3)
    realized = np.cumsum(rng.normal(0, 1, 250)) * 0.6
    pnl = 0.5 * (realized**2 * 0 + (realized*0.0)) # placeholder
    gamma_pnl = 0.5 * np.cumsum((rng.normal(0,1,250)**2 - 1)) * 0.8
    theta = -0.6 * t * 100
    ax.plot(t, gamma_pnl*8, color=RED, lw=2.2, label="gamma P&L  (½Γ·realized var)")
    ax.plot(t, theta, color=NAVY, lw=2.2, label="theta P&L  (−½Γ·implied var)")
    ax.plot(t, gamma_pnl*8 + theta, color=DKRED, lw=2.6, label="net (gamma–theta–variance)")
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlabel("time (fraction of horizon)"); ax.set_ylabel("P&L (USD)")
    ax.set_title("Delta-Hedged Book: the Gamma–Theta–Variance Identity", color=DKRED, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    finish(fig, "week08.png")


# ---------------------------------------------------------------- Week 9
def w9():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    t = np.arange(60); rng = np.random.default_rng(5)
    spread = np.cumsum(np.abs(rng.normal(2, .5, 60)))
    inv = np.cumsum(rng.normal(0, 1.2, 60)); gamma = np.cumsum(rng.normal(.4, 1, 60))
    hedge = -np.cumsum(np.abs(rng.normal(.6, .3, 60)))
    ax1.stackplot(t, spread, np.maximum(gamma, 0), np.maximum(inv, 0),
                  labels=["spread", "gamma", "inventory"], colors=[RED, GOLD, TEAL], alpha=.85)
    ax1.plot(t, spread+gamma+inv+hedge, color=DKRED, lw=2, label="total P&L")
    ax1.set_ylabel("cumulative P&L"); ax1.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    ax1.set_title("Milestone 3: P&L Attribution & Drawdown", color=DKRED, fontweight="bold")
    eq = spread+gamma+inv+hedge; dd = eq - np.maximum.accumulate(eq)
    ax2.fill_between(t, dd, 0, color=GRAY, alpha=.6); ax2.set_ylabel("drawdown"); ax2.set_xlabel("trading day")
    finish(fig, "week09.png")


# ---------------------------------------------------------------- Week 10
def w10():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.6))
    S = np.linspace(60, 140, 200); spot = 100
    # collar: long put @90, short call @110, long underlying
    payoff = (S - spot) + np.maximum(90 - S, 0) - np.maximum(S - 110, 0)
    ax1.plot(S, S - spot, color=GRAY, ls="--", lw=1.5, label="unhedged")
    ax1.plot(S, payoff, color=RED, lw=2.6, label="collar")
    ax1.axhline(0, color=GRAY, lw=.8); ax1.axvline(spot, color=GRAY, lw=.6, ls=":")
    ax1.set_title("Collar — fix a price band", color=DKRED, fontweight="bold", fontsize=12)
    ax1.set_xlabel("underlying"); ax1.set_ylabel("payoff"); ax1.legend(frameon=False, fontsize=8)
    # accumulator: buy fixed qty below strike, accelerated above KO
    acc = np.where(S < 100, (S-100)*1.0, np.where(S < 115, (S-100)*1.0, (S-100)*2.0))
    acc = np.clip(acc, -45, 45)
    ax2.plot(S, acc, color=TEAL, lw=2.6)
    ax2.axhline(0, color=GRAY, lw=.8); ax2.axvline(100, color=GRAY, lw=.6, ls=":")
    ax2.set_title("Accumulator — staged accumulation", color=DKRED, fontweight="bold", fontsize=12)
    ax2.set_xlabel("underlying"); ax2.set_ylabel("client payoff")
    fig.suptitle("Corporate Structured Products (Coincall guest)", color=DKRED, fontweight="bold", fontsize=14)
    finish(fig, "week10.png")


PROMPTS = {  # used only if Gemini image quota is available
 "week01.png": "Editorial flat infographic of a 10-week course roadmap timeline, crimson #CC0000 accents, white background, no text.",
 "week02.png": "Flat infographic of a limit order book with bid and ask depth bars and a mid-price line, crimson accents, white background, no text.",
}

if __name__ == "__main__":
    builders = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]
    names = [f"week{ i:02d}.png" for i in range(1, 11)]
    for name, b in zip(names, builders):
        if not (name in PROMPTS and try_gemini(PROMPTS[name], name)):
            b()
    print("illustrations done")
