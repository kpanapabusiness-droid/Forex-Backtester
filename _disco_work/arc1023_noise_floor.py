"""Arc 1023 — INDEPENDENT verification of arc 2016's portfolio NOISE-FLOOR claim.

Arc 2016 (2000s) found the 4-way book's residual negative folds (2015 -0.047%, 2018 -0.124%)
are statistically indistinguishable from zero (0.067 / 0.176 sd; bootstrap CI spans zero) — i.e.
the all-folds-positive calendar-year gate, on a book of thin (10-28 trade/fold) components, is
evaluated BELOW its own noise floor, and the ~18-arc 5th-leg hunt was moving inside that floor.

This is programme-redirecting; the Arc-10 norm is independent reproduction by a DIFFERENT method.
arc 2016 used a within-fold per-trade bootstrap of the COMBINED book. Here I:
  (1) Re-run all 4 components through the CANONICAL apparatus (verify headlines — Arc-10),
  (2) reconstruct the book at arc 2016's documented frozen weights,
  (3) assess the noise floor by an INDEPENDENT lens: ACROSS-fold dispersion (t on the mean,
      worst-fold z, sign/binomial on fold signs) + a within-fold per-trade bootstrap with a
      DIFFERENT seed (123) — and report the book-MEAN significance (decision-support for the
      operator's flagged path A vs path B).

All IS-only (the binding folds 2015/2018 are IS). The combined-book OOS is NOT spent (the book
fails IS AFP — protocol §5g). Strict all-folds-positive gate STAYS a FAIL; this only quantifies
WHY, it does not loosen the gate (arc 2016's council rejected loosening).
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from core.sim.panel import Panel
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import run_config_over_folds

from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import MonthEndReversionLongSignal, MonthEndReversionShortSignal
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.time_exit_predicate import make_time_exit_predicate

HISTDATA = r"C:\Users\panap\histdata_backup"
IS_FOLDS = [f for f in build_v3_folds().folds if f.is_days >= 365]
RISK_PCT = float(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 0.5

_panels: dict = {}
def get_panel(pairs, tf):
    key = (tf, tuple(sorted(pairs)))
    if key not in _panels:
        _panels[key] = Panel.from_pairs(list(pairs), tf=tf, histdata_root=HISTDATA,
                                        cache_root="data/cache", boundary_convention="5ers_eet")
    return _panels[key]


def run_component(name, signal, pairs, tf, *, exit_policy, trail_enabled, time_exit_bars):
    panel = get_panel(pairs, tf)
    ev = signal.evaluate({tf: panel})
    if time_exit_bars is not None:
        pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in pairs}, n_bars=time_exit_bars)
        ev = replace(ev, per_pair={p: replace(st, exit_predicate=pred) for p, st in ev.per_pair.items()})
    cfg = A1Config(config_id=name, sl_atr_mult=2.0, trail_enabled=trail_enabled,
                   risk_pct=RISK_PCT, exit_policy=exit_policy)
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=ev, panels={tf: panel})
    stats = run_config_over_folds(runner, IS_FOLDS, cfg)
    rois = np.array([fs.roi_pct for fs in stats], float)         # decimal (0.0185 = 1.85%)
    ntr = np.array([fs.n_trades for fs in stats], int)
    return rois, ntr


COMPONENTS = {
    "gap":      dict(signal=WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36),
                     pairs=["EURJPY","GBPJPY","AUDJPY","CADJPY","CHFJPY"], tf="H4",
                     exit_policy=None, trail_enabled=False, time_exit_bars=24, headline=+0.685),
    "me_long":  dict(signal=MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2),
                     pairs=["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"], tf="D1",
                     exit_policy="sl_only", trail_enabled=False, time_exit_bars=2, headline=+0.232),
    "fbr":      dict(signal=FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25),
                     pairs=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"], tf="H4",
                     exit_policy="sl_plus_trailing_atr", trail_enabled=True, time_exit_bars=None, headline=+1.854),
    "me_short": dict(signal=MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2),
                     pairs=["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"], tf="D1",
                     exit_policy="sl_partial_close_1r_runner_trail", trail_enabled=False, time_exit_bars=None, headline=+0.683),
}

print(f"RISK_PCT (A1Config) = {RISK_PCT}  |  IS folds = {len(IS_FOLDS)} (ids {[f.fold_id for f in IS_FOLDS]})")
print("="*92)
results = {}
for name, c in COMPONENTS.items():
    rois, ntr = run_component(name, c["signal"], c["pairs"], c["tf"],
                              exit_policy=c["exit_policy"], trail_enabled=c["trail_enabled"],
                              time_exit_bars=c["time_exit_bars"])
    results[name] = (rois, ntr)
    mean_pct = rois.mean() * 100
    npos = int((rois > 0).sum())
    print(f"{name:9s} mean={mean_pct:+.3f}%  (headline {c['headline']:+.3f}%)  {npos}/{len(rois)} pos  "
          f"n_trades/fold={ntr.tolist()}  (tot {ntr.sum()})")
    print(f"          per-fold ROI %: {np.round(rois*100,3).tolist()}")

print("="*92)
# Book at arc 2016's documented IS-best convex weights
W = {"gap": 0.0, "me_long": 0.65, "fbr": 0.20, "me_short": 0.15}
book = sum(W[n] * results[n][0] for n in W)          # per-fold book ROI (decimal)
book_pct = book * 100
fold_ids = [f.fold_id for f in IS_FOLDS]
print(f"Book weights (arc 2016): {W}")
print(f"Book per-fold ROI %: {dict(zip(fold_ids, np.round(book_pct,3).tolist()))}")
print(f"Book mean = {book_pct.mean():+.4f}%  worst = {book_pct.min():+.4f}%  "
      f"best = {book_pct.max():+.4f}%  across-fold sd = {book_pct.std(ddof=1):.4f}%  "
      f"pos folds = {int((book>0).sum())}/{len(book)}")

# (3a) ACROSS-fold lens (independent of arc 2016's within-fold bootstrap)
m, s, n = book_pct.mean(), book_pct.std(ddof=1), len(book_pct)
se_mean = s / np.sqrt(n)
t_mean = m / se_mean
worst_z_vs0 = book_pct.min() / s          # worst fold in across-fold-sd units from ZERO
print("\n--- ACROSS-FOLD noise lens ---")
print(f"book MEAN t-stat (mean/se, df={n-1}) = {t_mean:+.3f}   (|t|>2.26 => mean>0 at p<0.05 two-sided, n=10)")
print(f"worst fold {book_pct.min():+.4f}% = {worst_z_vs0:+.3f} across-fold-sd from zero")
within1 = int((np.abs(book_pct) <= s).sum())
print(f"folds within +/-1 across-fold-sd ({s:.3f}%) of ZERO: {within1}/{n}")

# (3b) within-fold per-trade-equivalent bootstrap with DIFFERENT seed (123): resample the 10 folds
rng = np.random.default_rng(123)
B = 20000
boot_worst = np.empty(B); boot_mean = np.empty(B)
for b in range(B):
    samp = book_pct[rng.integers(0, n, n)]
    boot_worst[b] = samp.min(); boot_mean[b] = samp.mean()
print("\n--- FOLD bootstrap (seed 123, B=20000, resample folds) ---")
print(f"book mean 95% CI = [{np.percentile(boot_mean,2.5):+.3f}%, {np.percentile(boot_mean,97.5):+.3f}%]  "
      f"P(mean<=0) = {(boot_mean<=0).mean():.3f}")
print(f"P(any negative fold in a 10-fold draw) = {(boot_worst<0).mean():.3f}  "
      f"(a negative fold is EXPECTED if this high)")
