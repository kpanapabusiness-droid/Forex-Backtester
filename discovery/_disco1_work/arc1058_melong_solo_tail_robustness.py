"""arc 1058 — TAIL-ROBUSTNESS of me_long-SOLO (the operator's honest OOS deploy object).

Extends arc 1057 from the 2-leg book to the SINGLE-leg object the operator would
actually deploy. Arc 1046/1053 established that under honest frozen §5f exits the book
collapses to me_long-SOLO out-of-sample (fbr mean-negative OOS, gap & me_short honest-
negative IS+OOS, arc 2044), so me_long-solo (committed sl_only / SL2.0 / 2-bar time
exit, D1 USD majors) is the REAL honest deploy object. Arc 1057 showed the 2-leg book's
mean is fbr-runner-tail-FRAGILE (q95-winsorize or removing 1 fbr runner kills the
cluster-bootstrap significance). The unanswered fork for the operator's path-A call:
is me_long-solo's OWN mean tail-robust, or also a few-trade artifact?

because: me_long is a high-win-rate SHORT-horizon (2-bar) mean-reversion fade with a
non-convex exit (sl_only, no runner) — structurally the LEAST tail-carried of the 4
legs (1057 already showed fbr's top-5 = 127% of its net P&L; me_long's tail in $ is
tiny, max $580 vs fbr $2,513). If me_long-solo's mean SURVIVES winsorization where
fbr's didn't → the operator has a robust-but-low-Calmar deploy object distinct from
the tail-fragile {me_long,fbr} vehicle (sharpens the 1053 3-corner frontier). If it
ALSO collapses → NO deploy corner has a certifiable mean (uniform "edge too thin").

OBJECT: me_long-solo, committed honest exit (sl_only, SL2.0, 2-bar time-exit on the
signal — the all-folds-selected cell in arcs 1053/1056). Two windows:
  - IS: 8 evaluable folds 2013-2020 (parallel to 1057; the 10 v3 IS folds, warmup
    2011/2012 dropped for apples-to-apples with the nested-selection convention — but
    me_long's exit is FIXED so all 10 are valid; report both 8-fold and 10-fold).
  - OOS: per-year holdout via build_oos_year_folds(2021) — MEASURE-ONCE characterization
    of an ALREADY-SPENT series (1046/1053/2055 read it; a tail-robustness re-read adds
    NO new selection, §4: measuring OOS is fine, OPTIMIZING to it is contamination).

METHOD (CALLS canonical; experiment side = winsorization + bootstrap arithmetic on
per-position NET P&L only; never realizes P&L, never touches the gate, never SELECTS
on OOS): score me_long over each window's folds, capture per-fold per-position NET pnl
(gross - apply_cost_model), build the per-fold ROI series, then tail concentration +
winsorize (q99/97.5/95/90) + leave-top-N + cluster bootstrap (the correct SE-of-mean).
No new BUILT tool (cf. 1057/2016).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.runners._fold_stats_helpers import slice_equity_to_oos
from core.sim.costs.model import CostModel, apply_cost_model
from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import build_oos_year_folds
from discovery.tools.month_end_signals import MonthEndReversionLongSignal
from discovery.tools.time_exit_predicate import make_time_exit_predicate

HISTDATA_ROOT = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
CACHE_ROOT = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
SB = 100_000.0
SEED, N_BOOT = 42, 10_000


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach_te(sig: SignalEvaluation, panel: Panel, n: int) -> SignalEvaluation:
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(st, exit_predicate=pred) for p, st in sig.per_pair.items()})


def _cell_pnl(runner, fold, cfg, fn_cost):
    """Return (per-position NET pnl realized in the fold's OOS window, OOS-start denom, gate roi)."""
    fs = runner(fold, cfg)
    rr = runner.last_result.run_result
    costed = apply_cost_model(rr, fn_cost)
    oos_start = pd.Timestamp(fold.oos_start, tz="UTC")
    oos_end = pd.Timestamp(fold.oos_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    bd = costed.breakdown
    if len(bd):
        ex = pd.to_datetime(bd["final_exit_time"], utc=True)
        m = (ex >= oos_start) & (ex <= oos_end)
        net = bd.loc[m, "net_pnl"].to_numpy(dtype=float)
    else:
        net = np.empty(0, dtype=float)
    oos_eq = slice_equity_to_oos(costed.net_equity, fold)
    denom = float(oos_eq.iloc[0]) if len(oos_eq) else SB
    return net, denom, fs.roi_pct


def _cluster_boot(series, seed=SEED, n=N_BOOT):
    series = np.asarray(series, dtype=float)
    pt = float(series.mean()) * 100.0
    rng = np.random.default_rng(seed)
    fm = np.array([rng.choice(series, size=len(series), replace=True).mean() for _ in range(n)])
    ci = np.percentile(fm, [2.5, 97.5]) * 100.0
    p_neg = float((fm < 0).mean())
    t = pt / (fm.std() * 100.0) if fm.std() > 0 else float("nan")
    sig = "SIG+" if ci[0] > 0 else ("SIG-" if ci[1] < 0 else "~0")
    return pt, ci, p_neg, t, sig


def analyze(tag, cells, years):
    """cells: list of (net_pnl_array, denom) per fold; years: matching year labels."""
    base = np.array([0.0 if len(p) == 0 else p.sum() / d for (p, d) in cells])
    pooled = np.concatenate([p for (p, _) in cells]) if cells else np.empty(0)
    npos = len(pooled)
    tot = pooled.sum()
    s = np.sort(pooled)[::-1]
    print(f"\n{'='*86}\n[{tag}]  n_folds={len(cells)}  n_pos={npos}  Σnet=${tot:+,.0f}  "
          f"max=${(s[0] if npos else 0):+,.0f}")
    if npos:
        for k in (1, 3, 5):
            print(f"    top-{k} = {s[:k].sum()/tot*100:6.1f}% of Σnet", end="")
        print()

    def report(name, series):
        pt, ci, p_neg, t, sig = _cluster_boot(series)
        wi = int(np.argmin(series))
        print(f"    {name:30s} mean {pt:+.4f}%/yr  pos {int((series>0).sum())}/{len(series)}  "
              f"worst {series.min()*100:+.3f}%({years[wi]})  cluster t={t:+.2f} "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] P(<0)={p_neg:.3f} [{sig}]")

    report("BASELINE", base)
    # winsorize the pooled positive tail
    pos = pooled[pooled > 0]
    for q in (99.0, 97.5, 95.0, 90.0):
        cap = np.percentile(pos, q) if len(pos) else np.inf
        wser = np.array([0.0 if len(p) == 0 else np.minimum(p, cap).sum() / d for (p, d) in cells])
        report(f"winsorize q{q:g} (cap ${cap:,.0f})", wser)
    # leave-top-N (across all positions of this object)
    cat = []
    for fi, (p, d) in enumerate(cells):
        for j, v in enumerate(p):
            cat.append((v, fi, j))
    cat.sort(key=lambda r: r[0], reverse=True)
    for topn in (1, 2, 3, 5):
        drop = {(fi, j) for (_, fi, j) in cat[:topn]}
        who = ", ".join(f"{years[fi]}=${v:,.0f}" for (v, fi, j) in cat[:topn])
        lser = np.array([
            (lambda keep: 0.0 if len(keep) == 0 else keep.sum() / d)(
                np.array([v for j, v in enumerate(p) if (fi, j) not in drop], dtype=float))
            for fi, (p, d) in enumerate(cells)])
        report(f"drop top-{topn} [{who}]", lser)


def main():
    print("ARC 1058 — TAIL-ROBUSTNESS of me_long-SOLO (the honest OOS deploy object, arc 1046/1053)")
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}")
    print("me_long: MonthEndReversionLongSignal(1.0,2) D1 USD, sl_only, SL2.0, 2-bar time-exit (committed honest cell)")
    print("scoring canonical (A1->MultiPairBacktester, FundedNext, risk 0.005 linear)\n")
    d1 = _load(USD, "D1")
    sig = _attach_te(MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1}), d1, 2)
    cfg = A1Config(config_id="melong_solo_sl_only_sl2", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_only")
    runner = ArcFoldRunner(A1Architecture(), sig, {"D1": d1})
    fn_cost = CostModel.fundednext()

    # ── IS folds (10 v3 IS folds 2011-2020; me_long exit is FIXED so all are valid) ──
    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    is_years = [f.oos_start.year for f in is_folds]
    _is = [_cell_pnl(runner, f, cfg, fn_cost) for f in is_folds]
    is_cells = [(net, denom) for (net, denom, _) in _is]
    is_rois = [roi for (_, _, roi) in _is]
    # reproduce-anchor print
    print("IS per-fold gate ROI (me_long-solo, all 10 v3 folds):")
    print("  years " + str(is_years))
    print("  roi%  " + str([round(r*100, 2) for r in is_rois]))
    analyze("IS 10-fold 2011-2020", is_cells, is_years)
    # 8-fold (drop warmup 2011/2012) — apples-to-apples with arc 1057's evaluable set
    analyze("IS 8-fold 2013-2020 (1057-parallel)", is_cells[2:], is_years[2:])

    # ── OOS per-year folds (measure-once characterization of an ALREADY-SPENT series) ──
    oos_folds = build_oos_year_folds(start_year=2021)
    oos_years = [f.oos_start.year for f in oos_folds]
    _oos = [_cell_pnl(runner, f, cfg, fn_cost) for f in oos_folds]
    oos_cells = [(net, denom) for (net, denom, _) in _oos]
    oos_rois = [roi for (_, _, roi) in _oos]
    print("\nOOS per-year gate ROI (me_long-solo; measure-once char., already-spent series 1046/1053/2055):")
    print("  years " + str(oos_years))
    print("  roi%  " + str([round(r*100, 2) for r in oos_rois]))
    analyze("OOS per-year 2021+", oos_cells, oos_years)


if __name__ == "__main__":
    main()
