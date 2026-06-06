"""Arc 1031 L1 — M1 addendum (extend the H4/H1/M15 trend to the menu's explicit M1 falsifier).

Same metrics as observe_ou_amplitude.py, M1 only (the cache base, no aggregation). If the
frac|r|>cost keeps falling and the OU amplitude keeps shrinking at M1, the triangulation door
is closed at every resolution the menu named (down to M1), level AND variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.triangulation_residual import triangle_log_residual_bp

HISTDATA = r"C:\Users\panap\histdata_backup"
TRIANGLES = [
    ("EURJPY", "EURUSD", "USDJPY", "mul"),
    ("GBPJPY", "GBPUSD", "USDJPY", "mul"),
    ("AUDJPY", "AUDUSD", "USDJPY", "mul"),
    ("EURGBP", "EURUSD", "GBPUSD", "div"),
]
PAIRS = sorted({p for t in TRIANGLES for p in t[:3]})
SLIP_PIP_PER_FILL, N_FILLS, COMM_BP = 0.5, 2, 0.5


def pip_size(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


def main():
    print(f"loading M5 panel for {PAIRS} ...")
    pn = Panel.from_pairs(PAIRS, tf="M5", histdata_root=HISTDATA, cache_root="data/cache",
                          boundary_convention="5ers_eet")
    rows = []
    for cross, legA, legB, op in TRIANGLES:
        cdf = pn.pair_dfs[cross]
        resid = triangle_log_residual_bp(cdf, pn.pair_dfs[legA], pn.pair_dfs[legB], op=op)
        if len(resid) < 200:
            continue
        sigma = float(resid.std())
        r = resid.dropna().values
        x, y = r[:-1], r[1:]
        phi = float(np.dot(x, y) / np.dot(x, x))
        hl = (-np.log(2) / np.log(phi)) if 0 < phi < 1 else np.nan
        mid = (cdf["close_bid"] + cdf["close_ask"]) / 2.0
        spread_bp = (1e4 * (cdf["close_ask"] - cdf["close_bid"]) / mid).replace([np.inf, -np.inf], np.nan).dropna()
        med_spread = float(spread_bp.median())
        slip_bp = SLIP_PIP_PER_FILL * N_FILLS * (pip_size(cross) / float(mid.median())) * 1e4
        rt_cost = 1.5 * med_spread + slip_bp + COMM_BP
        rows.append({
            "tf": "M5", "cross": cross, "n": len(resid), "sigma_bp": round(sigma, 3),
            "AR1_phi": round(phi, 4), "half_life_bars": round(hl, 2) if np.isfinite(hl) else np.nan,
            "rt_cost_bp": round(rt_cost, 3), "harvest_z2_bp": round(2 * sigma, 3),
            "net_z2_bp": round(2 * sigma - rt_cost, 3),
            "frac|r|>cost": round(float((resid.abs() > rt_cost).mean()), 4),
            "frac|r|>spread": round(float((resid.abs() > med_spread).mean()), 4),
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(df.to_string(index=False))
    print(f"\nM5 means: sigma={df.sigma_bp.mean():.3f} bp  half_life={df.half_life_bars.mean():.2f} bars  "
          f"rt_cost={df.rt_cost_bp.mean():.3f} bp  net_z2={df.net_z2_bp.mean():.3f} bp  "
          f"frac|r|>cost={df['frac|r|>cost'].mean():.4f}")


if __name__ == "__main__":
    main()
