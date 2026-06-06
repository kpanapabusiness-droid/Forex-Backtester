"""arc 2063 — TAIL-DEPENDENCE of the {me_long, fbr} deploy vehicle (runner winsorization).

Resolves the explicitly-owed thread from arc 1056 (handoff pointer), measured at the
deploy object arcs 2059/2060 identified as the corpus-best OOS vehicle: the 2-leg
{me_long, fbr} book. 2059 found its +0.697%/yr OOS mean is doubly fat-tail-fragile at
the YEAR level (ex-2024 -> +0.31%/yr; ex-both-tail-years ~= +0.06%/yr flat), the +1.87
being fbr's 2024 runner tail. This is the rigorous TRADE-level version: cap each
position's realized net R at +K R (GEOMETRY-ONLY upside winsorization — the engine
already took every loss SL-first; losses are UNTOUCHED), and re-measure the deploy
mean + per-year series + cluster-bootstrap significance.

DECISION QUESTION (path-A gate-governance, the only live lever): is the deploy mean
BROAD-BASED (survives a +2R cap -> deployment-robust) or fbr-runner-TAIL-LUCK
(collapses -> fragile)?

METHOD (CALLS canonical; experiment-side = winsorization arithmetic only, never
realizes P&L):
 1. Build me_long + fbr via the committed/frozen deploy configs (build_component,
    IDENTICAL to subset_deploy_profile / validate_4way_book) -> canonical
    ArcFoldRunner -> A1 -> MultiPairBacktester. me_long = sl_only + 2-bar TE; fbr =
    sl_plus_trailing_atr double-trail (the convex-runner source).
 2. Per fold (= per year) capture per-POSITION NET pnl (apply_cost_model chokepoint,
    positions whose final_exit_time in the fold OOS window) + denom (OOS-start net
    equity) -- the arc-1056 unit (reproduces gate roi_pct to <=0.02pp pre-exposure-cap).
 3. REPRODUCTION ANCHOR (Arc-10 guard): uncapped per-year leg ROIs must match the
    committed 2056/2059 numbers byte-close before any cap is trusted.
 4. R-unit = risk_pct * SB = 0.005 * 100_000 = $500 (linear risk; cross-checked vs the
    empirical -1R me_long sl_only loss cluster). Winsorize upside: net_capped =
    min(net_pnl, K*R) for K in {2,3,4}. Recompute per-year leg + book ROI (equal + RP
    frozen-IS weights), deploy mean / neg-count / worst, fbr tail-fraction, and the
    year-level (cluster) bootstrap P(mean<0) + 95% CI, uncapped vs capped.

OOS = measure-once CHARACTERIZATION off the already-spent frozen series (arc
2055/2056/2059 precedent) -- nothing selected on OOS (Sec 4). Single-use diagnostic.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture
from core.runners.arc_fold_runner import ArcFoldRunner
from core.runners._fold_stats_helpers import slice_equity_to_oos
from core.sim.costs.model import CostModel, apply_cost_model
from core.wfo.discovery_measure import build_oos_year_folds
from core.wfo.folds import build_v3_folds
from discovery.tools.combine_fold_roi import fit_weights
from discovery.tools.solo_deploy_profile import build_component

HISTDATA_ROOT = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
CACHE_ROOT = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
SB = 100_000.0
RISK_PCT = 0.005
R_DOLLARS = RISK_PCT * SB          # $500 — linear risk per trade (arc 1056)
NAMES = ("me_long", "fbr")
SEED, N_BOOT = 42, 10_000
CAPS = (2.0, 3.0, 4.0)

# Committed reproduction anchors (byte-for-byte from arc 2056 Tier-2).
ANCHOR_IS = {
    "me_long": [0.40, 0.29, 0.96, -0.23, -1.14, -0.51, 0.34, 0.90, 1.16, 0.15],
    "fbr":     [7.55, 3.05, 0.91, 0.19, 3.17, 2.55, 1.23, -4.20, 0.05, 4.03],
}
ANCHOR_OOS = {
    "me_long": [-0.63, 0.55, 0.18, 0.54, 2.26, 0.02],
    "fbr":     [1.63, -1.39, 1.10, 7.61, -2.76, -0.57],
}


def _folds(window):
    if window == "oos":
        return list(build_oos_year_folds(start_year=2021))
    return [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]


def capture_leg(name, window):
    """Return {year: (net_pnl_array, denom)} for one component over one window."""
    sig, panels, cfg, _mark = build_component(name, HISTDATA_ROOT, CACHE_ROOT, BOUNDARY)
    runner = ArcFoldRunner(A1Architecture(), sig, panels)
    fn_cost = CostModel.fundednext()
    out = {}
    for fold in _folds(window):
        runner(fold, cfg)
        costed = apply_cost_model(runner.last_result.run_result, fn_cost)
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
        out[fold.oos_start.year] = (net, denom)
    return out


def leg_year_roi(cell, cap=None):
    """Per-year ROI % for one leg; cap = upside winsorization multiple K (None = uncapped)."""
    roi = {}
    for yr, (net, denom) in cell.items():
        x = net if cap is None else np.minimum(net, cap * R_DOLLARS)
        roi[yr] = float(x.sum()) / denom * 100.0
    return roi


def book_series(legs_cell, weights, cap=None):
    years = sorted(next(iter(legs_cell.values())))
    rois = {n: leg_year_roi(legs_cell[n], cap) for n in NAMES}
    book = [sum(weights[i] * rois[NAMES[i]][yr] for i in range(len(NAMES))) for yr in years]
    return years, book


def cluster_boot(series):
    arr = np.array(series) / 100.0
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(N_BOOT)])
    ci = np.percentile(means, [2.5, 97.5]) * 100.0
    p_neg = float((means < 0).mean())
    return ci, p_neg


def main():
    print("ARC 2063 — tail winsorization of the {me_long, fbr} deploy vehicle")
    print(f"R = risk_pct*SB = {R_DOLLARS:.0f}  caps K={CAPS}  (upside-only; losses UNTOUCHED)\n")

    for window, anchor in (("is", ANCHOR_IS), ("oos", ANCHOR_OOS)):
        print("=" * 84)
        print(f"WINDOW = {window.upper()}")
        print("=" * 84)
        cells = {n: capture_leg(n, window) for n in NAMES}
        years = sorted(next(iter(cells.values())))

        # (3) reproduction anchor
        print("\n[anchor] uncapped per-year leg ROI vs committed 2056 numbers:")
        maxdev = 0.0
        for n in NAMES:
            roi = leg_year_roi(cells[n])
            got = [round(roi[y], 2) for y in years]
            dev = max(abs(roi[years[i]] - anchor[n][i]) for i in range(len(years)))
            maxdev = max(maxdev, dev)
            print(f"  {n:8s} got {got}")
            print(f"  {n:8s} exp {anchor[n]}   max|dev|={dev:.3f}pp")
        print(f"  -> anchor max|dev| = {maxdev:.3f}pp  ({'OK' if maxdev < 0.05 else 'CHECK'})")

        # R-unit empirical cross-check (me_long sl_only clean -1R losers)
        allnet = np.concatenate([cells["me_long"][y][0] for y in years])
        losers = allnet[allnet < 0]
        if len(losers):
            mode_loss = float(np.median(losers))
            print(f"\n[R-check] me_long net losers: median {mode_loss:+.1f}  (canonical -1R ~= {-R_DOLLARS:.0f})")

        # fbr tail structure
        fbr_net = np.concatenate([cells["fbr"][y][0] for y in years])
        print(f"\n[fbr tail] n_pos={int((fbr_net>0).sum())}/{len(fbr_net)} positions; "
              f"net P&L sum {fbr_net.sum():+.0f}")
        for K in CAPS:
            above = fbr_net[fbr_net > K * R_DOLLARS]
            excess = float((above - K * R_DOLLARS).sum())
            tot_pos = float(fbr_net[fbr_net > 0].sum())
            print(f"   K={K:.0f}R: {len(above):3d} positions exceed +{K:.0f}R; "
                  f"excess-above-cap {excess:+.0f} = {excess/tot_pos*100:5.1f}% of fbr gross-positive P&L")

        # weights frozen on IS
        if window == "is":
            comp_is = [[leg_year_roi(cells[n])[y] / 100.0 for y in years] for n in NAMES]
            w_rp = fit_weights(comp_is, mode="risk_parity")
            globals()["_W_RP"] = w_rp
        w_rp = globals().get("_W_RP", [0.5, 0.5])
        weightings = [("equal", [0.5, 0.5]), ("RP(frozen-IS)", list(w_rp))]
        print(f"\nRP frozen-IS weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, w_rp)))

        for wlabel, w in weightings:
            print(f"\n--- weighting [{wlabel}] ---")
            yrs, base = book_series(cells, w, cap=None)
            ci0, pneg0 = cluster_boot(base)
            print(f"  UNCAPPED  mean {np.mean(base):+.4f}%/yr  neg {sum(r<0 for r in base)}/{len(base)}  "
                  f"worst {min(base):+.3f}%  per-yr {[round(r,2) for r in base]}")
            print(f"            cluster-boot 95% CI [{ci0[0]:+.3f}, {ci0[1]:+.3f}]  P(mean<0)={pneg0:.3f}")
            for K in CAPS:
                _, cap = book_series(cells, w, cap=K)
                ciK, pnegK = cluster_boot(cap)
                surv = np.mean(cap) / np.mean(base) * 100.0 if np.mean(base) != 0 else float("nan")
                print(f"  +{K:.0f}R cap  mean {np.mean(cap):+.4f}%/yr  ({surv:5.1f}% of uncapped survives)  "
                      f"neg {sum(r<0 for r in cap)}/{len(cap)}  worst {min(cap):+.3f}%")
                print(f"            cluster-boot 95% CI [{ciK[0]:+.3f}, {ciK[1]:+.3f}]  P(mean<0)={pnegK:.3f}")
        print()


if __name__ == "__main__":
    main()
