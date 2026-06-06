"""Arc 1061 — JAPANESE FISCAL-YEAR-END (March 31) JPY REPATRIATION seasonal.

THE IDEA (a documented institutional flow the corpus has NOT touched).
Japan's fiscal year ends March 31. Japanese institutions (life insurers, corporates, pensions)
repatriate foreign earnings before the books close -> sell foreign currency, BUY JPY -> JPY
appreciates -> JPY-quoted crosses (XXXJPY: USDJPY, EURJPY, ...) tend to FALL into late March.
This is a cited, recurring FX seasonal (the "fiscal year-end repatriation" / "Japanese exporter
flow") distinct from the corpus's already-mapped GENERIC monthly month-end (me_long/me_short,
arcs 1011/1019) and turn-of-month USD drift (1005). The honest, falsifiable claim: SHORT XXXJPY in
the final ~week of March has a real, March-SPECIFIC directional edge.

WHY this is NOT closed ground (§5a) and NOT a re-grind:
  - It is a CALENDAR-GATED, currency-SPECIFIC flow with a documented mechanism (Japanese fiscal
    year-end), keyed to ONE month (March) on JPY-quote pairs only -- not a shallow directional cut
    and not the generic month-end the corpus mapped (which fires EVERY month on USD majors).
  - The decisive control is MARCH-vs-OTHER-MONTHS in the SAME late-month window: if the edge exists
    in every month equally it is just month-end (covered, shares the reversion tail); March-SPECIFIC
    excess is the novel claim.
  - The mechanism is independent of the binding 2015/2018 folds (no outcome-aware fold selection;
    §1/§4 clean) -- it earns a fresh test on its own documented because.

HYPOTHESIS (falsifiable, mechanistic):
  (A) SHORT XXXJPY in the final week of March (day-of-month >= 24, causal) has honest take-the-loss
      capture > 0.50 and POSITIVE short-drift (price falls) -- repatriation JPY-buying;
  (B) the effect is MARCH-SPECIFIC: March late-window short capture/drift EXCEEDS the same late-window
      in non-March months (else it is generic month-end, already mapped);
  (C) it is ROBUST across years, not 1-2-year-carried (thin-tail tell, arcs 2011/2063).
KILL if (A) is coin-flip-or-adverse, OR (B) shows no March excess (generic month-end), OR (C) the
mean is carried by a couple of years. PROCEED to honest-engine triage only if March-specific,
robust, and not thin.

OBSERVATION cheap-kill (§5d): honest take-the-loss capture + fwd-drift (BUILT observe_long_capture,
direction-aware), IS 2010-2020, D1, JPY-quote crosses. No engine / null / council here; OOS untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture

HIST = r"C:\Users\panap\histdata_backup"
CACHE = "data/cache"
# JPY-quote crosses (JPY is bought in repatriation -> these fall). USD majors set has only USDJPY.
PAIRS = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]
IS_START, IS_END = pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")
STRONG_USD_FOLDS = [2014, 2015, 2016, 2018]  # the reversion book's wall (context only; NOT a selector)


def late_month_mask(idx: pd.DatetimeIndex, *, month: int | None, day_from: int) -> np.ndarray:
    """Causal mask: True on bars with calendar day-of-month >= day_from (and, if `month` given,
    in that calendar month). Day-of-month is known AT the bar -> no lookahead (unlike 'last N
    trading days', which needs future bars to know the month's trading-day count)."""
    dom = idx.day.values
    m = dom >= day_from
    if month is not None:
        m = m & (idx.month.values == month)
    return m


def observe(panel, *, direction: str, month: int | None, day_from: int, hold: int, drift_bars: int):
    restrict = {p: late_month_mask(panel.pair_dfs[p].index, month=month, day_from=day_from) for p in PAIRS}
    obs = observe_long_capture(panel, sl_mult=2.0, hold=hold, drift_bars=drift_bars, warmup=60,
                               restrict=restrict, direction=direction)
    obs = obs[(obs["signal_time"] >= IS_START) & (obs["signal_time"] <= IS_END)].copy()
    obs["year"] = obs["signal_time"].dt.year
    return obs


def summary(obs, *, label: str, per_year: bool = False, per_pair: bool = False):
    if len(obs) == 0:
        print(f"\n{label}: n=0 (no fires)"); return None
    print(f"\n{'='*84}\n{label}\n  n={len(obs)} | capture {obs['capture'].mean():.4f} | "
          f"drift_mean {obs['fwd_drift_atr'].mean():+.4f} | drift_med {obs['fwd_drift_atr'].median():+.4f}")
    if per_year:
        yr = obs.groupby("year").agg(n=("capture", "size"), cap=("capture", "mean"),
                                     drift=("fwd_drift_atr", "mean"))
        print(yr.to_string())
        pos = int((yr["drift"] > 0).sum())
        print(f"  per-year drift: pos-years {pos}/{len(yr)}  worst {yr['drift'].min():+.4f}  "
              f"best {yr['drift'].max():+.4f}  mean-of-yr {yr['drift'].mean():+.4f}")
        strong = yr.reindex(STRONG_USD_FOLDS)["drift"]
        print(f"  (context) strong-USD folds {STRONG_USD_FOLDS} drift: "
              f"{[f'{v:+.3f}' if pd.notna(v) else 'NA' for v in strong]}")
    if per_pair:
        pp = obs.groupby("pair").agg(n=("capture", "size"), cap=("capture", "mean"),
                                     drift=("fwd_drift_atr", "mean"))
        print(pp.to_string())
        print(f"  per-pair: {int((pp['drift']>0).sum())}/{len(pp)} pairs positive drift")
    return obs


def main():
    panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=HIST, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    pd.set_option("display.width", 200, "display.max_columns", 30)

    print("### arc 1061: JAPANESE FISCAL-YEAR-END (Mar 31) JPY REPATRIATION ###")
    print("IS 2010-2020, D1, 7 JPY-quote crosses. SHORT XXXJPY in late March (repatriation JPY-buying).")
    print("Honest take-the-loss capture + fwd-drift; OOS untouched. KILL unless March-SPECIFIC + robust.")

    # ---- (A) the claim: SHORT XXXJPY, last week of March (day>=24) ----
    for dfrom in (24, 20):
        mar_s = observe(panel, direction="short", month=3, day_from=dfrom, hold=8, drift_bars=5)
        summary(mar_s, label=f"(A) MARCH late-window SHORT  | day>={dfrom} | hold8 drift5",
                per_year=True, per_pair=True)

        # ---- (B) the decisive control: same late-window in NON-March months ----
        non_mar_s = observe(panel, direction="short", month=None, day_from=dfrom, hold=8, drift_bars=5)
        # remove March rows to get the pure non-March control
        non_mar_s = non_mar_s[non_mar_s["signal_time"].dt.month != 3].copy()
        summary(non_mar_s, label=f"(B) NON-MARCH late-window SHORT (control) | day>={dfrom}")
        if len(mar_s) and len(non_mar_s):
            print(f"  >>> MARCH-EXCESS (short): capture {mar_s['capture'].mean()-non_mar_s['capture'].mean():+.4f} | "
                  f"drift {mar_s['fwd_drift_atr'].mean()-non_mar_s['fwd_drift_atr'].mean():+.4f} "
                  f"(positive => a real March-specific repatriation edge; ~0 => generic month-end)")

    # ---- (D) direction check: is it actually LONG (April outward-flow / pre-hedge reversal)? ----
    mar_l = observe(panel, direction="long", month=3, day_from=24, hold=8, drift_bars=5)
    summary(mar_l, label="(D) MARCH late-window LONG (direction control) | day>=24", per_year=True)

    # ---- (E) wider / earlier windows + longer hold (repatriation is a multi-day flow) ----
    for (m, dfrom, h, d) in [(3, 24, 12, 10), (3, 20, 12, 10)]:
        o = observe(panel, direction="short", month=m, day_from=dfrom, hold=h, drift_bars=d)
        summary(o, label=f"(E) MARCH SHORT robustness | day>={dfrom} | hold{h} drift{d}", per_year=True)


if __name__ == "__main__":
    main()
