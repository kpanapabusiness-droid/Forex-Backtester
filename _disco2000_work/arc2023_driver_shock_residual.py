"""arc 2023 — M1 (DISCOVERY_DIRECTION menu): driver-shock-CONDITIONAL cross-timeframe triangulation residual.

Arc 3005 killed the cross-rate triangulation residual but ONLY unconditionally and ONLY at H4
(fwd-convergence corr ~0.01, residual std 1.23-1.58 bp). The menu's freshest prior: a cross-instrument
lead-lag linked by the triangular identity was never tested. EURJPY == EURUSD * USDJPY is an identity; a
large directional shock in a DRIVER leg (EURUSD) forces the dependent cross to move. If the quoted cross
re-prices with a LAG, the next bar's identity residual should continue in the shock direction even though
the same-bar residual is ~0 on average.

EX-ANTE construction (all H4, one timeframe — the "D1 driver shock" is proxied by a 6-H4-bar [~1 day]
driver return measured at bar t-1, predicting bar t; no lookahead, no D1/H4 alignment ambiguity):

  triangles (clean XXXUSD drivers): EUR(EURUSD,USDJPY->EURJPY), GBP(GBPUSD,USDJPY->GBPJPY), AUD(AUDUSD,USDJPY->AUDJPY)
  log mid returns r_cross, r_driver, r_other(=USDJPY)
  driver_shock[t-1] = (6-bar driver log return) / rolling-std(6-bar return, 250), flagged |.|>1.5
  TWO residual measures at the NEXT bar t:
    (b) PURE identity dislocation (extends arc 3005):  idr[t] = r_cross[t] - r_driver[t] - r_other[t]  (~0 by identity)
    (a) menu-literal "not explained by contemporaneous USDJPY": resid_o[t] = r_cross[t] - beta*r_other[t]
        (NOTE: this conflates driver MOMENTUM (closed ground) with dislocation; (b) is the clean test)
  DECISIVE: corr(driver_shock[t-1], idr[t]) within the shock subset.  >0.05 -> real lag dislocation -> build the leg.
            ~0 (like 3005's 0.01) -> efficient -> KILL (extends 3005 to the driver-shock-conditional case).

CHARACTERIZATION ONLY (gross, no engine, no cost realized). A cheap-kill observation per the menu
("treat long-shots as cheap observations; a death at the cheap-obs stage is a high-value closure").
IS window only (2010-2020); OOS untouched.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from core.sim.panel import Panel

BACKUP = r"C:\Users\panap\histdata_backup"
TRIANGLES = [  # (driver, other, cross)
    ("EURUSD", "USDJPY", "EURJPY"),
    ("GBPUSD", "USDJPY", "GBPJPY"),
    ("AUDUSD", "USDJPY", "AUDJPY"),
]
ALLPAIRS = sorted({p for tr in TRIANGLES for p in tr})


def mid_logret(df):
    mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    r = np.full(len(mid), np.nan)
    r[1:] = np.log(mid[1:] / mid[:-1])
    return pd.Series(r, index=df.index)


def spread_bp(df):
    mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    return pd.Series((df["close_ask"].to_numpy(float) - df["close_bid"].to_numpy(float)) / mid * 1e4, index=df.index)


def main():
    panel = Panel.from_pairs(ALLPAIRS, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    IS0, IS1 = pd.Timestamp(date(2010, 1, 1), tz="UTC"), pd.Timestamp(date(2020, 12, 31), tz="UTC")

    print("=== arc 2023 M1: driver-shock-conditional triangulation residual (H4, IS 2010-2020) ===")
    print("   baseline (arc 3005, unconditional H4): convergence corr ~0.01, residual std 1.23-1.58 bp\n")
    all_idr_shock, all_drv_shock = [], []
    for driver, other, cross in TRIANGLES:
        dfc, dfd, dfo = panel.pair_dfs[cross], panel.pair_dfs[driver], panel.pair_dfs[other]
        idx = dfc.index.intersection(dfd.index).intersection(dfo.index)
        idx = idx[(idx >= IS0) & (idx <= IS1)]
        rc = mid_logret(dfc).reindex(idx)
        rd = mid_logret(dfd).reindex(idx)
        ro = mid_logret(dfo).reindex(idx)

        # (b) identity residual return (~0 by identity); arc-3005 residual in return space
        idr = (rc - rd - ro)
        # (a) menu-literal: cross move not explained by contemporaneous USDJPY (beta from full IS OLS)
        m = rc.notna() & ro.notna()
        beta = np.polyfit(ro[m], rc[m], 1)[0]
        resid_o = rc - beta * ro

        # driver shock at t-1: 6-bar driver return / rolling std(6-bar ret, 250); ex-ante (shift1 into t)
        drv6 = rd.rolling(6).sum()
        sd = drv6.rolling(250).std()
        shock_mag = (drv6 / sd)
        shock_prev = shock_mag.shift(1)                 # known at t-1, predicts t
        is_shock = shock_prev.abs() > 1.5

        # next-bar residuals correlated with the prior driver shock
        df = pd.DataFrame({"drv": shock_prev, "idr": idr, "resid_o": resid_o, "shock": is_shock}).dropna()
        sub = df[df["shock"]]
        unc_corr_idr = df["drv"].corr(df["idr"])
        sh_corr_idr = sub["drv"].corr(sub["idr"]) if len(sub) > 30 else np.nan
        sh_corr_ro = sub["drv"].corr(sub["resid_o"]) if len(sub) > 30 else np.nan
        # directional: does idr continue in the shock sign? mean(sign(drv)*idr) in bp
        cont_bp = float((np.sign(sub["drv"]) * sub["idr"]).mean() * 1e4)
        idr_std_bp = float(df["idr"].std() * 1e4)
        sp = float(spread_bp(dfc).reindex(idx).median())

        print(f"  {cross} (driver {driver}):  n_shocks={len(sub)}  idr_std={idr_std_bp:.2f}bp  cross_spread~{sp:.2f}bp")
        print(f"    corr(driver_shock, next-bar IDENTITY residual):  uncond={unc_corr_idr:+.3f}   SHOCK={sh_corr_idr:+.3f}   <-- decisive (>0.05 = lag)")
        print(f"    corr(driver_shock, next-bar 'not-expl-by-USDJPY' residual) SHOCK={sh_corr_ro:+.3f}  (conflates driver momentum)")
        print(f"    directional continuation: mean(sign(shock)*next-bar idr) = {cont_bp:+.3f} bp  (vs cross spread {sp:.2f}bp)")
        all_idr_shock.append(sub["idr"].to_numpy()); all_drv_shock.append(sub["drv"].to_numpy())

    # pooled
    di = np.concatenate(all_drv_shock); ii = np.concatenate(all_idr_shock)
    pooled = np.corrcoef(di, ii)[0, 1]
    cont = float((np.sign(di) * ii).mean() * 1e4)
    print(f"\n  POOLED (3 triangles): n_shocks={len(di)}  corr(driver_shock, next-bar idr)={pooled:+.3f}  directional={cont:+.3f}bp")
    print(f"  VERDICT GUIDE: |corr|<=~0.05 AND |directional|<cross-spread  ->  efficient, KILL (extends arc 3005 conditionally).")


if __name__ == "__main__":
    main()
