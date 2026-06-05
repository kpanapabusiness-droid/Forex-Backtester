"""Arc 1027 — M1: driver-shock-CONDITIONAL cross-rate triangulation residual.

Attacks arc 3005's UNCONDITIONAL/H4-only closure: condition the triangular residual on a
large D1 driver-leg shock and ask whether the dependent quoted cross re-prices with a lag
at the next H4 bar. Cheap OBSERVATION (no pool/engine) — decisive either way.

Triangles (driver = XXXUSD, cross = XXXJPY = XXXUSD*USDJPY):
  EUR/GBP/AUD/NZD vs USD/JPY. IS window 2010-2020 (develop on IS; OOS untouched).

Falsifiable prediction (DIRECTION.md M1): corr(driver_shock, next_bar_cross_residual) > 0.05
  (materially above arc 3005's unconditional ~0.01). Falsifier: stays <= ~0.05 -> dead.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.triangulation_residual import (
    d1_driver_shock,
    shock_available_at_next_day,
    triangle_log_residual_bp,
)

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
SHOCK_Z = 1.5

TRIANGLES = [  # (driver XXXUSD, second leg USDJPY, quoted cross XXXJPY)
    ("EURUSD", "USDJPY", "EURJPY"),
    ("GBPUSD", "USDJPY", "GBPJPY"),
    ("AUDUSD", "USDJPY", "AUDJPY"),
    ("NZDUSD", "USDJPY", "NZDJPY"),
]
H4_PAIRS = sorted({p for tri in TRIANGLES for p in tri})
D1_DRIVERS = sorted({tri[0] for tri in TRIANGLES})


def _clip_is(df):
    return df[(df.index >= IS_START) & (df.index <= IS_END)]


def main():
    print("Loading H4 panel:", H4_PAIRS)
    h4 = Panel.from_pairs(H4_PAIRS, tf="H4", histdata_root=HIST, cache_root=CACHE,
                          boundary_convention="5ers_eet")
    print("Loading D1 panel:", D1_DRIVERS)
    d1 = Panel.from_pairs(D1_DRIVERS, tf="D1", histdata_root=HIST, cache_root=CACHE,
                          boundary_convention="5ers_eet")

    rows = []
    for driver, legB, cross in TRIANGLES:
        cdf = _clip_is(h4.pair_dfs[cross])
        adf = _clip_is(h4.pair_dfs[driver])
        bdf = _clip_is(h4.pair_dfs[legB])
        resid = triangle_log_residual_bp(cdf, adf, bdf, op="mul")  # H4 residual bp

        # D1 driver shock, re-keyed to the first H4 bar after the D1 close
        z_d1 = d1_driver_shock(_clip_is(d1.pair_dfs[driver]))
        z_avail = shock_available_at_next_day(z_d1)

        # align shock (at next-day first H4 label) with the residual at that bar
        df = pd.DataFrame({"z": z_avail}).join(pd.DataFrame({"r": resid}), how="inner").dropna()
        # forward convergence: cross mid-close move over next k H4 bars at the dislocation bar
        cross_mid = pd.Series(((cdf["close_bid"] + cdf["close_ask"]) / 2).values, index=cdf.index)
        for k in (6,):
            fwd = (np.log(cross_mid).shift(-k) - np.log(cross_mid)) * 1e4  # bp
            df[f"fwd{k}"] = fwd.reindex(df.index)

        # unconditional baseline over ALL first-bars-of-day (3005 anchor + corr)
        uncond_corr = df["z"].corr(df["r"])
        # conditional on a large driver shock
        sh = df[df["z"].abs() > SHOCK_Z].copy()
        cond_corr = sh["z"].corr(sh["r"]) if len(sh) > 5 else np.nan
        # convergence: does -residual predict the forward cross move (mean reversion)?
        conv_uncond = (-df["r"]).corr(df["fwd6"])
        conv_cond = (-sh["r"]).corr(sh["fwd6"]) if len(sh) > 5 else np.nan

        rows.append({
            "triangle": f"{cross}={driver}x{legB}",
            "n_firstbar": len(df),
            "n_shock": len(sh),
            "resid_med_bp": float(np.median(df["r"])),
            "resid_std_bp": float(df["r"].std()),
            "shock_resid_med_bp": float(np.median(sh["r"])) if len(sh) else np.nan,
            "shock_resid_std_bp": float(sh["r"].std()) if len(sh) else np.nan,
            "corr_uncond": float(uncond_corr),
            "corr_shock": float(cond_corr),
            "conv_corr_uncond": float(conv_uncond),
            "conv_corr_shock": float(conv_cond),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== M1 driver-shock-conditional triangulation residual (IS 2010-2020) ===")
    print(out.to_string(index=False))

    print("\n--- Pooled across triangles (shock bars) ---")
    # pooled correlation across all triangles' shock bars (z-standardized r per triangle)
    pooled = []
    for driver, legB, cross in TRIANGLES:
        cdf = _clip_is(h4.pair_dfs[cross]); adf = _clip_is(h4.pair_dfs[driver]); bdf = _clip_is(h4.pair_dfs[legB])
        resid = triangle_log_residual_bp(cdf, adf, bdf, op="mul")
        z_avail = shock_available_at_next_day(d1_driver_shock(_clip_is(d1.pair_dfs[driver])))
        d = pd.DataFrame({"z": z_avail}).join(pd.DataFrame({"r": resid}), how="inner").dropna()
        d = d[d["z"].abs() > SHOCK_Z]
        pooled.append(d)
    P = pd.concat(pooled, ignore_index=True)
    print(f"pooled shock bars n={len(P)}  corr(z, next-bar resid) = {P['z'].corr(P['r']):.4f}")
    print(f"  |resid| at shock bars: median={P['r'].abs().median():.3f} bp  mean={P['r'].abs().mean():.3f} bp")
    # what fraction of shock bars have |resid| > a ~0.8 bp single-cross-spread cost floor?
    print(f"  frac |resid|>0.8bp (single-cross spread) = {(P['r'].abs()>0.8).mean():.3f}")

    # within-triangle-standardized pooled corr (fair pooling if a same-signed effect existed)
    zs, rs = [], []
    for d in pooled:
        if len(d) > 5:
            zs.append((d["z"] - d["z"].mean()) / d["z"].std())
            rs.append((d["r"] - d["r"].mean()) / d["r"].std())
    Z, R = pd.concat(zs, ignore_index=True), pd.concat(rs, ignore_index=True)
    print(f"  within-triangle-standardized pooled corr = {Z.corr(R):.4f}")
    signs = [np.sign(d["z"].corr(d["r"])) for d in pooled if len(d) > 5]
    print(f"  per-triangle corr signs = {signs}  (consistent lag would be all-same-sign)")


if __name__ == "__main__":
    main()
