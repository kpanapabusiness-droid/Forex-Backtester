"""Arc 1065 — lens-A: the CORRELATION-vs-SIGN crux of me_long's strong-USD losses.

THE THREAD (council-surfaced, logged-not-executed; arc 1063 forward thread (a); the LAST open
thread in the 1000s range after arc 1064 closed lens-B). The 1063 council CHALLENGED its own
premise: "2015/2018 is a SIGN conflict, not co-located noise -> a symmetric veto can't make a year
prefer the leg it rejects." lens-A SETTLES that empirically.

THE QUESTION (decisive, two honest outcomes). me_long (1011 / the honest deploy object 1046) buys a
USD major that fell >=1 ATR into month-end. Its KNOWN IS hole is the contiguous 2014/2015/2016
strong-USD-bull block (arc 1012/1059); it carries a real POSITIVE 2018 via directional WMR (arc
2017/2018). lens-A asks: is the strong-USD LOSS a hedgeable COMMON-USD-FACTOR component (then a
beta-sized USD-basket hedge -- shorts now enabled PR#273 -- could strip it), OR an irreducible
IDIOSYNCRATIC-to-the-regime / sign-conflict effect (then no symmetric factor hedge rescues it)?

This is DISTINCT from:
  - arc 1059 (USD-breadth FILTER): conditioned ENTRY (drops systematic trades -> loses count AND the
    +2018 help). lens-A keeps ALL trades and subtracts the beta*factor component (a HEDGE, not a
    filter) -> isolates pure idiosyncratic alpha.
  - arc 2018 (cross-sectional / fully USD-neutral): neutralizes the WHOLE common move (long-laggard/
    short-leader). lens-A removes only the BETA-sized common-factor exposure, keeping each pair's full
    idiosyncratic reversion -> the least-aggressive hedge that could still strip the regime risk.

DECISIVE TEST (obs-level, gross, no engine -- the engine + doubled-cost only matter IF gross PROCEEDs):
  (1) Decompose me_long's per-month-end basket P&L (mean fwd_drift of fires) into a common-USD-factor
      component (regress on f = the month's common USD move) + an idiosyncratic residual (alpha).
  (2) Cross-pair co-movement of simultaneously-held fires by year (factor-wide => high co-movement).
  (3) THE CRUX: per-year de-factored alpha (y - beta*f). If alpha is positive AND all-folds-stable
      INCLUDING the 2014-16 block -> the loss WAS a hedgeable common factor -> PROCEED (engine+cost).
      If alpha is ALSO negative in 2014-16 -> the reversion genuinely fails when the move is a trend,
      hedge or not -> SIGN-CONFLICT / irreducible -> lens-A CLOSES (KILL), confirms operator path-A.

Reuses arc 1059's build_common_usd_move (canonical _month_end_into_move) + BUILT observe_long_capture.
IS 2010-2020, D1, 7 USD majors. No null/council; OOS untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.month_end_signals import MonthEndReversionLongSignal
from discovery.tools.observe_long_capture import observe_long_capture

# reuse arc 1059's common-USD-factor builder verbatim (canonical _month_end_into_move under the hood)
from discovery._disco1_work.arc1059_melong_usd_breadth import build_common_usd_move, USD_FACTOR

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
BLOCK = [2014, 2015, 2016]  # me_long's strong-USD-bull IS hole


def _ols(x: np.ndarray, y: np.ndarray):
    """Simple OLS y = a + b x. Returns (alpha, beta, r2)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def main():
    panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    allme, common = build_common_usd_move(panel)

    long_sig = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)
    ev = long_sig.evaluate({"D1": panel})
    fires = {p: ev.per_pair[p].signal_mask.to_numpy(bool) for p in panel.pairs}
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=2, warmup=30,
                               restrict=fires, direction="long")
    obs = obs[(obs["signal_time"] >= IS_START) & (obs["signal_time"] <= IS_END)].copy()
    obs["ym"] = obs["signal_time"].dt.to_period("M")
    obs["year"] = obs["signal_time"].dt.year
    obs = obs.merge(allme[["pair", "signal_time", "into_atr", "usd_move"]],
                    on=["pair", "signal_time"], how="left")
    obs = obs.merge(common[["common_usd_move", "breadth"]], left_on="ym", right_index=True, how="left")
    obs = obs.dropna(subset=["common_usd_move", "fwd_drift_atr"]).copy()

    # A me_long LONG on pair p has directional USD exposure = +USD_FACTOR[p]:
    #   long USDXXX (+1) gains when USD RISES (f>0); long XXXUSD (-1) gains when USD FALLS (f<0).
    # The position's signed exposure to f (common USD move, + = USD strength):
    obs["pos_usd_beta_sign"] = obs["pair"].map(USD_FACTOR)             # +1 (USDXXX) / -1 (XXXUSD)
    # the factor's contribution to THIS position's directional outcome, in factor units:
    obs["factor_signal"] = obs["pos_usd_beta_sign"] * obs["common_usd_move"]

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"=== arc 1065 lens-A: factor-vs-sign crux of me_long | IS 2010-2020 D1 | n_fires={len(obs)} ===")
    base = obs["fwd_drift_atr"]
    print(f"unconditional me_long: drift_mean {base.mean():+.4f}  drift_med {base.median():+.4f}  "
          f"capture {obs['capture'].mean():.4f}")

    # ---- (1) POOLED factor regression: per-fire drift ~ factor_signal -----------------------------
    a, b, r2 = _ols(obs["factor_signal"].to_numpy(), obs["fwd_drift_atr"].to_numpy())
    corr = np.corrcoef(obs["factor_signal"], obs["fwd_drift_atr"])[0, 1]
    print("\n-- (1) POOLED per-fire regression  drift ~ alpha + beta*factor_signal --")
    print(f"   alpha(idiosyncratic) {a:+.4f}   beta(factor load) {b:+.4f}   R2 {r2:.4f}   corr {corr:+.4f}")
    print("   (beta>0 + high R2 => loss is a COMMON-USD-FACTOR exposure = hedgeable;")
    print("    beta~0 / low R2  => outcome is idiosyncratic to the position, factor hedge inert)")

    # de-factored (hedged) per-fire outcome: subtract the fitted common-factor component
    obs["alpha_hedged"] = obs["fwd_drift_atr"] - b * obs["factor_signal"]

    # ---- (2) cross-pair co-movement of simultaneously-held fires, by year -------------------------
    print("\n-- (2) cross-pair co-movement of same-month fires (factor-wide => low within-month dispersion of sign) --")
    rows = []
    for yr, g in obs.groupby("year"):
        # within each month-end, do the held positions share outcome sign? (factor-wide => yes)
        frac_same = []
        for _ym, gm in g.groupby("ym"):
            if len(gm) >= 2:
                s = np.sign(gm["fwd_drift_atr"])
                frac_same.append(max((s > 0).mean(), (s < 0).mean()))  # 1.0 = all same sign
        rows.append({"year": yr, "n_fires": len(g),
                     "mean_drift": g["fwd_drift_atr"].mean(),
                     "n_multi_months": len(frac_same),
                     "within_month_sign_concord": np.mean(frac_same) if frac_same else np.nan})
    comove = pd.DataFrame(rows).set_index("year")
    print(comove.to_string(float_format=lambda v: f"{v:+.3f}"))

    # ---- (3) THE CRUX: per-year RAW vs FACTOR-PREDICTED vs HEDGED-ALPHA ---------------------------
    print("\n-- (3) CRUX: per-year mean drift  RAW  vs  factor-component(beta*signal)  vs  HEDGED alpha --")
    per = obs.groupby("year").agg(
        n=("fwd_drift_atr", "size"),
        raw=("fwd_drift_atr", "mean"),
        factor_comp=("factor_signal", lambda s: b * s.mean()),
        hedged_alpha=("alpha_hedged", "mean"),
        hedged_alpha_med=("alpha_hedged", "median"),
    )
    print(per.to_string(float_format=lambda v: f"{v:+.4f}"))
    raw = per["raw"]; hed = per["hedged_alpha"]
    print(f"\n   RAW         : neg-years {int((raw < 0).sum())}/{raw.notna().sum()}  "
          f"mean {raw.mean():+.4f}  worst-yr {raw.min():+.4f}")
    print(f"   HEDGED alpha: neg-years {int((hed < 0).sum())}/{hed.notna().sum()}  "
          f"mean {hed.mean():+.4f}  worst-yr {hed.min():+.4f}")
    print(f"   2014-16 block:  RAW {raw.reindex(BLOCK).mean():+.4f}   HEDGED {hed.reindex(BLOCK).mean():+.4f}")
    print(f"   2018 (the +help):  RAW {raw.get(2018, float('nan')):+.4f}   "
          f"HEDGED {hed.get(2018, float('nan')):+.4f}")

    # ---- verdict logic ----------------------------------------------------------------------------
    block_raw = raw.reindex(BLOCK).mean()
    block_hed = hed.reindex(BLOCK).mean()
    hed_neg = int((hed < 0).sum())
    print("\n=== lens-A READ ===")
    if block_hed > 0 and hed_neg == 0:
        print("  HEDGE RESCUES the block (de-factored alpha all-folds-positive) -> PROCEED to engine+cost"
              " (hedge doubles FundedNext cost; net-of-cost is the real gate, cf. arc 2010/2018).")
    elif block_hed > block_raw + 0.05:
        print("  HEDGE MITIGATES the block but does NOT flip all-folds-positive -> the loss is PARTLY"
              " common-factor but the residual reversion ALSO fails in strong-USD -> mixed; report.")
    else:
        print("  HEDGE does NOT help the block (de-factored alpha still negative in 2014-16) -> the loss"
              " is IDIOSYNCRATIC-to-the-regime, NOT a hedgeable common factor -> lens-A CLOSES: the"
              " 2015/2018 wall is a SIGN conflict (council pre-emption CONFIRMED), not co-located noise.")


if __name__ == "__main__":
    main()
