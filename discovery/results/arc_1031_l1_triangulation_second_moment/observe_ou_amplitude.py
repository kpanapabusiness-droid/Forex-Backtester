"""Arc 1031 — L1: triangulation residual SECOND moment (OU amplitude) at finer resolution.

The last named-live explore-now MENU thread. Arc 3005 killed the triangulation residual LEVEL
(median ~0, fwd-convergence corr ~0.01) unconditionally at H4; arcs 1027/2023 killed it
driver-shock-CONDITIONALLY at H4. All three closed only the FIRST moment. Arc 3005 explicitly
flagged the residual's non-trivial std (1.23-1.58 bp, |resid|>spread on 1.3-6.1% of H4 bars) and
M1/sub-H4 as "out of apparatus scope, a different cost regime" — that variance dimension is virgin.

L1 hypothesis: the residual is a mean-reverting OU process around 0 (its mean is arb-pinned, but
its AMPLITUDE spikes when one leg's quote is stale). Harvest the amplitude (a convergence trade),
not the direction. Falsifiable: as resolution rises H4->H1->M15(->M1), the OU half-life should
shorten; find a resolution where half-life < ~5 bars AND the amplitude exceeds the single-cross
ROUND-TURN cost on > ~5% of bars. FALSIFIER: if at EVERY resolution the frac(|resid| > rt_cost)
stays below ~5%, the triangulation thread is FULLY closed (level AND variance).

CHARACTERIZATION ONLY (no engine; one-leg cross trade, the menu notes it reintroduces directional
exposure between entry and convergence). Honest FundedNext round-turn cost charged vs the amplitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.triangulation_residual import triangle_log_residual_bp

HISTDATA = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"

# triangles: cross = legA op legB.  JPY crosses (mul) + EURGBP (div) — the 3005/1027 set.
TRIANGLES = [
    ("EURJPY", "EURUSD", "USDJPY", "mul"),
    ("GBPJPY", "GBPUSD", "USDJPY", "mul"),
    ("AUDJPY", "AUDUSD", "USDJPY", "mul"),
    ("EURGBP", "EURUSD", "GBPUSD", "div"),
]
PAIRS = sorted({p for t in TRIANGLES for p in t[:3]})
TFS = ["H4", "H1", "M15"]

# FundedNext one-leg ROUND-TURN cost (bp): 1.5x spread + 0.5pip/fill x2 slippage + $5/lot RT commission.
SLIP_PIP_PER_FILL, N_FILLS, COMM_BP = 0.5, 2, 0.5  # comm $5/lot on $100k = 5e-5 = 0.5 bp


def pip_size(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


def cross_spread_and_cost_bp(cross_df, cross_pair):
    """Median cross spread (bp) and FundedNext one-leg round-turn cost (bp)."""
    mid = (cross_df["close_bid"] + cross_df["close_ask"]) / 2.0
    spread_bp = (1e4 * (cross_df["close_ask"] - cross_df["close_bid"]) / mid).replace([np.inf, -np.inf], np.nan).dropna()
    med_spread = float(spread_bp.median())
    slip_bp = SLIP_PIP_PER_FILL * N_FILLS * (pip_size(cross_pair) / float(mid.median())) * 1e4
    rt_cost = 1.5 * med_spread + slip_bp + COMM_BP
    return med_spread, slip_bp, rt_cost


def ou_half_life_bars(resid):
    """AR(1) half-life in bars: r_t = phi*r_{t-1} + eps; half-life = -ln2/ln(phi) (NaN if phi<=0 or >=1)."""
    r = resid.dropna().values
    if len(r) < 200:
        return np.nan, np.nan
    x, y = r[:-1], r[1:]
    phi = float(np.dot(x, y) / np.dot(x, x))
    if not (0 < phi < 1):
        return phi, np.nan
    return phi, -np.log(2) / np.log(phi)


def main():
    print(f"loading panels for {PAIRS} at {TFS} ...")
    panels = {}
    for tf in TFS:
        panels[tf] = Panel.from_pairs(PAIRS, tf=tf, histdata_root=HISTDATA, cache_root=CACHE,
                                      boundary_convention="5ers_eet")
        print(f"  {tf} loaded")

    rows = []
    for tf in TFS:
        pn = panels[tf]
        for cross, legA, legB, op in TRIANGLES:
            if not all(p in pn.pairs for p in (cross, legA, legB)):
                print(f"  skip {cross} @ {tf} (missing pair)")
                continue
            cdf, adf, bdf = pn.pair_dfs[cross], pn.pair_dfs[legA], pn.pair_dfs[legB]
            resid = triangle_log_residual_bp(cdf, adf, bdf, op=op)
            if len(resid) < 200:
                continue
            sigma = float(resid.std())
            med_abs = float(resid.abs().median())
            phi, hl = ou_half_life_bars(resid)
            med_spread, slip_bp, rt_cost = cross_spread_and_cost_bp(cdf, cross)
            # the ceiling on tradeable opportunity: a convergence from residual R nets at most |R|-cost,
            # so |resid| must exceed the round-turn cost for a trade to even be theoretically profitable.
            frac_gt_cost = float((resid.abs() > rt_cost).mean())
            frac_gt_spread = float((resid.abs() > med_spread).mean())  # reproduce 3005's looser lens
            # a |z|>2 entry sits at ~2*sigma; its full-convergence harvest is ~2*sigma gross
            harvest_z2 = 2.0 * sigma
            rows.append({
                "tf": tf, "cross": cross, "n": len(resid),
                "sigma_bp": round(sigma, 3), "med_abs_bp": round(med_abs, 3),
                "AR1_phi": round(phi, 4), "half_life_bars": round(hl, 2) if np.isfinite(hl) else np.nan,
                "med_spread_bp": round(med_spread, 3), "rt_cost_bp": round(rt_cost, 3),
                "harvest_z2_bp": round(harvest_z2, 3), "net_z2_bp": round(harvest_z2 - rt_cost, 3),
                "frac|r|>cost": round(frac_gt_cost, 4), "frac|r|>spread": round(frac_gt_spread, 4),
            })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n" + "=" * 100)
    print("L1 — triangulation residual second moment (OU amplitude) vs one-leg round-turn cost")
    print("=" * 100)
    print(df.to_string(index=False))
    print("\n--- per-TF summary (mean across triangles) ---")
    print(df.groupby("tf", sort=False).agg(
        sigma_bp=("sigma_bp", "mean"), half_life_bars=("half_life_bars", "mean"),
        rt_cost_bp=("rt_cost_bp", "mean"), net_z2_bp=("net_z2_bp", "mean"),
        frac_gt_cost=("frac|r|>cost", "mean"),
    ).to_string())
    print("\nFALSIFIER: if frac|r|>cost stays < ~0.05 at every TF AND net_z2_bp <= 0, triangulation is "
          "FULLY closed (level [3005/1027/2023] AND variance [this arc]).")


if __name__ == "__main__":
    main()
