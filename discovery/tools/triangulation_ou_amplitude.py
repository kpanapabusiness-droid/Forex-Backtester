"""Triangulation residual SECOND MOMENT — OU mean-reversion amplitude vs cost across
resolution (M1 -> M15 -> H1 -> H4). EXPERIMENT / OBSERVATION tool (BUILT).

Arc 3005 killed the cross-rate triangulation residual at the **level** (mean ~ 0,
fwd-convergence corr ~ 0.01) at **H4 only**, but reported a non-trivial residual std
(1.2-1.6 bp at H4) and explicitly flagged sub-H4 as "out of apparatus scope, a different
cost regime." Arcs 1027 / 2023 then closed the residual's *first moment*
driver-shock-conditionally. The residual's **variance / OU amplitude at finer resolution**
(strategist MENU item L1, `DISCOVERY_DIRECTION.md`) is the one named explore-now thread
never measured.

This tool measures, per triangle and per resolution:

  - the residual series r (bp) via the BUILT `triangle_log_residual_bp` (arc 1027),
    recomputed at each resolution by resampling each leg's bid/ask close (last) to the rule;
  - the OU fit as a discrete AR(1): r_t = phi * r_{t-1} + eps; half-life = -ln2 / ln(phi)
    bars (and wall-clock), amplitude sigma = std(r);
  - the single-cross FundedNext round-trip capture cost (bp): 1.5x spread + slippage
    (0.5 pip/fill x 2 fills) + commission ($5/lot RT ~= 0.5 bp), from the cross's own bars;
  - the fraction of bars where |r| exceeds the capture cost (and half-cost), and the
    fraction where a 2-sigma -> 0 convergence trade nets positive after the round-trip cost.

CHARACTERIZATION ONLY: never realizes P&L, never scores a trade, never touches the gate or
spends OOS. A pre-pool observation. The falsifier (DISCOVERY_DIRECTION L1): if at EVERY
resolution down to M1 the fraction of bars where the OU amplitude exceeds the single-cross
capture cost stays below ~5%, the triangulation thread is fully closed (level AND variance).

Created by: arc 1054 (chat 1000-1999), L1 triangulation second-moment / OU amplitude.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.data.histdata_loader import load_m1
from discovery.tools.triangulation_residual import triangle_log_residual_bp

HISTDATA_ROOT = r"C:\Users\panap\histdata_backup"
CACHE_ROOT = "data/cache"

# (cross, legA, legB, op).  cross = legA `op` legB.
TRIANGLES: list[tuple[str, str, str, str]] = [
    ("EURJPY", "EURUSD", "USDJPY", "mul"),
    ("GBPJPY", "GBPUSD", "USDJPY", "mul"),
    ("EURGBP", "EURUSD", "GBPUSD", "div"),
]

# pandas resample rules + bars-per-rule wall-clock minutes (for half-life in time)
RESOLUTIONS: list[tuple[str, str, int]] = [
    ("M1", "1min", 1),
    ("M15", "15min", 15),
    ("H1", "1h", 60),
    ("H4", "4h", 240),
]

# FundedNext cost knobs
SPREAD_MULT = 1.5           # 1.5x quoted spread
SLIP_PIPS_PER_FILL = 0.5    # 0.5 pip / fill
N_FILLS_RT = 2              # entry + exit (no partial on a convergence leg)
COMMISSION_BP_RT = 0.5      # $5 / 100k-lot round-turn ~= 0.5 bp


def _resample_legs(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample one leg's bid/ask close + spread to `rule` (bar close = last)."""
    cols = ["close_bid", "close_ask", "spread_close"]
    out = df[cols].resample(rule, label="right", closed="right").last().dropna()
    return out


def _pip_bp(cross: str, price: float) -> float:
    """One pip in bp of price for the cross at `price`."""
    pip = 0.01 if cross.endswith("JPY") else 0.0001
    return 1e4 * pip / price


def _capture_cost_bp(cross_res: pd.DataFrame, cross: str) -> float:
    """Single-cross FundedNext round-trip capture cost in bp (median over the window)."""
    mid = (cross_res["close_bid"] + cross_res["close_ask"]) / 2.0
    spread_bp = 1e4 * (cross_res["spread_close"] / mid)
    price = float(mid.median())
    spread_cost = SPREAD_MULT * float(spread_bp.median())
    slip_cost = SLIP_PIPS_PER_FILL * N_FILLS_RT * _pip_bp(cross, price)
    return spread_cost + slip_cost + COMMISSION_BP_RT


def _ou_fit(r: pd.Series) -> tuple[float, float]:
    """AR(1) phi (OLS, mean-removed) and half-life in bars. Returns (phi, half_life_bars)."""
    x = r.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    if len(x) < 100:
        return float("nan"), float("nan")
    x0, x1 = x[:-1], x[1:]
    phi = float(np.dot(x0, x1) / np.dot(x0, x0))
    if not (0.0 < phi < 1.0):
        hl = float("inf") if phi >= 1.0 else 0.0
    else:
        hl = -np.log(2.0) / np.log(phi)
    return phi, hl


@dataclass
class ResRow:
    cross: str
    res: str
    n: int
    sigma_bp: float
    p95_abs_bp: float
    phi: float
    half_life_bars: float
    half_life_min: float
    cost_bp: float
    frac_abs_gt_cost: float
    frac_abs_gt_halfcost: float
    frac_2sig_net_pos: float


def profile_triangle(
    cross: str, legA: str, legB: str, op: str,
    *, window=("2010-01-01", "2020-12-31"),
) -> list[ResRow]:
    """Per-resolution OU-amplitude-vs-cost profile for one triangle (IS window)."""
    raw = {p: load_m1(p, HISTDATA_ROOT, CACHE_ROOT) for p in (cross, legA, legB)}
    lo, hi = pd.Timestamp(window[0], tz="UTC"), pd.Timestamp(window[1], tz="UTC")
    raw = {p: d.loc[(d.index >= lo) & (d.index <= hi)] for p, d in raw.items()}
    rows: list[ResRow] = []
    for res, rule, minutes in RESOLUTIONS:
        legs = {p: _resample_legs(d, rule) for p, d in raw.items()}
        r = triangle_log_residual_bp(legs[cross], legs[legA], legs[legB], op=op).dropna()
        if len(r) < 200:
            continue
        sigma = float(r.std())
        p95 = float(r.abs().quantile(0.95))
        phi, hl_bars = _ou_fit(r)
        cost = _capture_cost_bp(legs[cross], cross)
        abs_r = r.abs()
        frac_cost = float((abs_r > cost).mean())
        frac_half = float((abs_r > cost / 2.0).mean())
        # 2-sigma convergence trade: enter when |r| > 2*sigma, gross capture ~ |r| (reverts
        # to ~0), net = |r| - cost. Fraction of ALL bars where that nets positive.
        entry = abs_r > 2.0 * sigma
        net_pos = entry & ((abs_r - cost) > 0.0)
        frac_net = float(net_pos.mean())
        rows.append(ResRow(cross, res, len(r), round(sigma, 4), round(p95, 4),
                           round(phi, 4), round(hl_bars, 2), round(hl_bars * minutes, 1),
                           round(cost, 4), round(frac_cost, 5), round(frac_half, 5),
                           round(frac_net, 6)))
    return rows


def main() -> None:
    print("L1 — triangulation residual OU amplitude vs single-cross cost (IS 2010-2020)\n")
    hdr = (f"{'cross':7} {'res':4} {'n':>9} {'sigma':>7} {'p95|r|':>7} {'phi':>6} "
           f"{'HL_bar':>7} {'HL_min':>8} {'cost':>6} {'%>cost':>7} {'%>hcost':>8} {'%2s_net+':>8}")
    print(hdr)
    print("-" * len(hdr))
    for cross, legA, legB, op in TRIANGLES:
        for row in profile_triangle(cross, legA, legB, op):
            print(f"{row.cross:7} {row.res:4} {row.n:>9,} {row.sigma_bp:>7.3f} "
                  f"{row.p95_abs_bp:>7.3f} {row.phi:>6.3f} {row.half_life_bars:>7.2f} "
                  f"{row.half_life_min:>8.1f} {row.cost_bp:>6.3f} "
                  f"{100*row.frac_abs_gt_cost:>6.3f}% {100*row.frac_abs_gt_halfcost:>7.3f}% "
                  f"{100*row.frac_2sig_net_pos:>7.4f}%")
        print()


if __name__ == "__main__":
    main()
