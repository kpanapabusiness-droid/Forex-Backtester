"""arc 2022 — DIAGNOSTIC: is arc 2019's terminal claim ("no 5th leg can make the 4-way book AFP")
correct when tested with a REAL, TARGETED +2015/+2016 candidate?

Context (the gap this fills):
  - arc 2019 (2000s) concluded the 4-way book is a sound ~3-bet mean-positive PORTFOLIO whose
    all-folds-positive (AFP) failure is a gate-resolution artifact, and that "a 5th decorrelated
    REVERSION leg cannot make the book AFP" -> the lever is the operator's gate-governance call.
    That claim was argued QUALITATIVELY (no diversification deficit) and only for REVERSION legs.
  - arc 3019 (3000s) found the corpus's STRONGEST +2015/+2016 candidate ever — forward-confirmed
    extreme-shock CONTINUATION (NOT reversion): honest-engine IS tp_3r 9/10, +2015 AND +2016 (the
    exact 4-way blockers), beats the fair null. It was KILLED on its OWN one-shot OOS (epoch-dependent:
    2/6, loses to null) BEFORE ever being combined with the 4-way book.

So the decisive feasibility question was NEVER answered: IF a +2015/+2016 leg with 3019's IS profile
existed (and survived OOS), would the 5-way book be all-folds-positive on IS? This arc answers it by
COMBINING the 4 frozen components with 3019's actual IS per-fold ROI and running the convex search.

  - 5-way IS-AFP achievable  -> arc 2019's terminal claim is too strong for a TARGETED (non-reversion)
    leg; the route is NOT terminal — the ONLY barrier is finding a +2015/+2016 leg that ALSO survives
    OOS (3019's profile is the existence proof of sufficiency; 3019 itself is OOS-dead + cap-clustered).
    => REOPEN the hunt with a sharpened, validated spec.
  - 5-way NOT IS-AFP even with 3019 -> even the strongest targeted candidate cannot lift the book;
    arc 2019's terminal claim is robustly confirmed with a number. => terminal; operator-gate is the lever.

Plus a feasibility frontier: the minimum per-fold magnitude an idealized 5th leg must add to the
blocker folds to clear AFP, vs. the per-fold noise floor (arc 2016/2023 ~0.7% sd) and vs. 3019's actual.

DIAGNOSTIC. CALLS canonical measurement only. No gate loosened. No OOS spent (the book fails IS AFP;
3019's OOS was already spent + recorded by arc 3019 — not re-touched here). Risk = 0.005 FRACTION
(arc-1024 confirmed deployable 0.5%, linear regime). Reuses arc 2015/2021 frozen configs EXACTLY.
"""
from __future__ import annotations

import dataclasses
import itertools
import numpy as np

from core.sim.panel import Panel
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import run_config_over_folds

from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import MonthEndReversionLongSignal, MonthEndReversionShortSignal
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.shock_continuation_signals import ShockContinuationSignal
from discovery.tools.time_exit_predicate import make_time_exit_predicate
from discovery.tools.combine_fold_roi import combine_fold_rois, fit_weights, rois_from_fold_stats

BACKUP = r"C:\Users\panap\histdata_backup"
JPY = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
USD = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
FOLD_YEAR = {fid: 2011 + (fid - 2) for fid in range(2, 12)}   # fid 2..11 -> 2011..2020
YEARS = [FOLD_YEAR[2 + i] for i in range(10)]


def inject_time_exit(eval_, panel, pairs, n_bars):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in pairs}, n_bars=n_bars)
    new_pp = {p: dataclasses.replace(st, exit_predicate=pred) for p, st in eval_.per_pair.items()}
    return dataclasses.replace(eval_, per_pair=new_pp)


def run(eval_, panel, tf, cfg, folds):
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=eval_, panels={tf: panel})
    return run_config_over_folds(runner, folds, cfg)


def main():
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    pj = Panel.from_pairs(JPY, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pu4 = Panel.from_pairs(USD, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pud = Panel.from_pairs(USD, tf="D1", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")

    # --- the 4 frozen components (EXACT arc-2015/2021 committed configs) ---
    ge = inject_time_exit(WeekendGapFillLongSignal(0.5, gap_hours=36).evaluate({"H4": pj}), pj, JPY, 24)
    gstats = run(ge, pj, "H4", A1Config(config_id="gap", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    le = inject_time_exit(MonthEndReversionLongSignal(1.0, 2).evaluate({"D1": pud}), pud, USD, 2)
    lstats = run(le, pud, "D1", A1Config(config_id="me_long", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    fstats = run(FailedBreakdownReclaimLongSignal(40, 1.25).evaluate({"H4": pu4}), pu4, "H4", A1Config(config_id="fbr", exit_policy="sl_plus_trailing_atr", sl_atr_mult=2.0, trail_enabled=True), is_folds)
    sstats = run(MonthEndReversionShortSignal(1.0, 2).evaluate({"D1": pud}), pud, "D1", A1Config(config_id="me_short", exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, trail_enabled=True), is_folds)

    # --- the 3019 shock-continuation leg: long (up-shocks) + short (down-shocks), tp_3r, summed per-fold ROI ---
    sc_l = run(ShockContinuationSignal(3.0, "long", confirm=True).evaluate({"H4": pu4}), pu4, "H4",
               A1Config(config_id="shock_long", exit_policy="sl_plus_tp_3r", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    sc_s = run(ShockContinuationSignal(3.0, "short", confirm=True).evaluate({"H4": pu4}), pu4, "H4",
               A1Config(config_id="shock_short", exit_policy="sl_plus_tp_3r", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    shock_roi = np.array(rois_from_fold_stats(sc_l)) + np.array(rois_from_fold_stats(sc_s))

    # scale/linearity + cap-breach check (resolve the vs-3019 magnitude question)
    print("=== shock leg breach/scale check (is 0.005 the LINEAR cap~0-breach regime?) ===")
    nL = sum(fs.n_trades for fs in sc_l); nS = sum(fs.n_trades for fs in sc_s)
    brL = sum(getattr(fs, "days_breaching_daily_5pct", 0) for fs in sc_l)
    brS = sum(getattr(fs, "days_breaching_daily_5pct", 0) for fs in sc_s)
    print(f"  n_trades long={nL} short={nS} total={nL+nS}  (3019 reported total n=287)")
    print(f"  daily-5%-DD breaches: long={brL} short={brS}  (cap~0 breaches => linear regime; >>0 => cap binds/distorts)")
    sc_l2 = run(ShockContinuationSignal(3.0, "long", confirm=True).evaluate({"H4": pu4}), pu4, "H4",
                A1Config(config_id="shock_long_lo", exit_policy="sl_plus_tp_3r", sl_atr_mult=2.0, trail_enabled=False, risk_pct=0.0005), is_folds)
    r_lo = np.array(rois_from_fold_stats(sc_l2)); r_hi = np.array(rois_from_fold_stats(sc_l))
    ratio = r_hi.sum() / r_lo.sum() if r_lo.sum() != 0 else float("nan")
    print(f"  long-only ROI sum at risk 0.005 = {r_hi.sum()*100:+.3f}%, at 0.0005 = {r_lo.sum()*100:+.3f}%, ratio={ratio:.1f} (=10.0 => perfectly linear)")

    comps = {"gap": gstats, "me_long": lstats, "fbr": fstats, "me_short": sstats}
    R = {k: np.array(rois_from_fold_stats(s)) for k, s in comps.items()}
    R["shock"] = shock_roi

    print("=== component headlines (VERIFY: gap +0.685 me_long +0.232 fbr +1.854 me_short +0.683) ===")
    print(f"  years order: {YEARS}")
    for k in ["gap", "me_long", "fbr", "me_short"]:
        by = {YEARS[i]: R[k][i] for i in range(10)}
        print(f"  {k:9s} mean={R[k].mean()*100:+6.3f}%  2015={by[2015]*100:+.2f} 2016={by[2016]*100:+.2f} 2018={by[2018]*100:+.2f}  neg={int((R[k]<0).sum())}/10")
    by = {YEARS[i]: shock_roi[i] for i in range(10)}
    print(f"  {'shock':9s} mean={shock_roi.mean()*100:+6.3f}%  2015={by[2015]*100:+.2f} 2016={by[2016]*100:+.2f} 2018={by[2018]*100:+.2f}  neg={int((shock_roi<0).sum())}/10   (VERIFY 3019 tp_3r: mean +0.034 2015 +0.037 2016 +0.109 2018 +0.012 9/10)")

    # --- correlation of shock vs the 4 (is it a genuinely decorrelated 5th bet?) ---
    print("\n=== shock per-fold ROI correlation vs the 4 frozen components ===")
    for k in ["gap", "me_long", "fbr", "me_short"]:
        print(f"  corr(shock, {k:8s}) = {np.corrcoef(shock_roi, R[k])[0,1]:+.3f}")

    # --- convex search helper ---
    def convex(parts, step=0.05):
        names = list(parts)
        Rs = [R[p] for p in names]
        g = [i * step for i in range(int(round(1/step)) + 1)]
        best = None; n_pass = 0; n_tot = 0
        # iterate simplex over len(parts) dims
        def rec(prefix, remaining_dims, remaining_mass):
            nonlocal best, n_pass, n_tot
            if remaining_dims == 1:
                w = prefix + [remaining_mass]
                if w[-1] < -1e-9: return
                n_tot += 1
                comb = sum(wi * Ri for wi, Ri in zip(w, Rs))
                mn = comb.min()
                if mn > 0: n_pass += 1
                if best is None or mn > best[0]:
                    best = (mn, list(w), comb.copy())
                return
            for v in g:
                if v <= remaining_mass + 1e-9:
                    rec(prefix + [v], remaining_dims - 1, remaining_mass - v)
        rec([], len(names), 1.0)
        return names, best, n_pass, n_tot

    for parts in (["gap", "me_long", "fbr", "me_short"],
                  ["gap", "me_long", "fbr", "me_short", "shock"]):
        names, best, n_pass, n_tot = convex(parts)
        mn, w, comb = best
        neg = [YEARS[i] for i, r in enumerate(comb) if r <= 0]
        print(f"\n=== {len(parts)}-WAY CONVEX SEARCH (step 0.05): {n_pass}/{n_tot} all-folds-positive ===")
        print(f"  best max-min worst-fold = {mn*100:+.3f}%  at w={dict(zip(names, [round(float(x),3) for x in w]))}")
        print(f"  mean={comb.mean()*100:+.3f}%  binding(<=0) folds={neg}")
        print(f"  per-year: " + " ".join(f"{YEARS[i]}:{r*100:+.2f}" for i, r in enumerate(comb)))

    # --- risk-parity 5-way (the weight a thin/high-vol leg actually gets) ---
    rl5 = [R[p] for p in ["gap", "me_long", "fbr", "me_short", "shock"]]
    w_rp = fit_weights(rl5, "risk_parity")
    comb_rp = np.array(combine_fold_rois(rl5, w_rp).combined_roi)
    print("\n=== 5-WAY HONEST WEIGHTS (arc-1021 rule: AFP under risk-parity AND equal, not just optimized) ===")
    for mode in ("risk_parity", "equal"):
        w = fit_weights(rl5, mode)
        comb = np.array(combine_fold_rois(rl5, w).combined_roi)
        neg = [YEARS[i] for i, r in enumerate(comb) if r <= 0]
        print(f"  {mode:11s} w={dict(zip(['gap','me_long','fbr','me_short','shock'], [round(float(x),3) for x in w]))}")
        print(f"     AFP={all(comb>0)} worst={comb.min()*100:+.3f}% mean={comb.mean()*100:+.3f}% neg={int((comb<0).sum())}/10 residual-neg-folds={neg}")
        print(f"     per-year: " + " ".join(f"{YEARS[i]}:{r*100:+.2f}" for i, r in enumerate(comb)))
    print(f"  shock per-fold ROI sd = {R['shock'].std(ddof=1)*100:.3f}%  (vs gap {R['gap'].std(ddof=1)*100:.2f} me_long {R['me_long'].std(ddof=1)*100:.2f} fbr {R['fbr'].std(ddof=1)*100:.2f} me_short {R['me_short'].std(ddof=1)*100:.2f}) -> high sd => low risk-parity weight")

    # --- FEASIBILITY FRONTIER: idealized 5th leg, positive ONLY in the 4-way blocker folds ---
    # take the 4-way best-convex book; for an idealized leg that contributes +x to each blocker fold and 0
    # elsewhere, at weight w5, find the minimum (w5, x) that clears AFP without flipping a good fold.
    names4, best4, _, _ = convex(["gap", "me_long", "fbr", "me_short"])
    _, w4, comb4 = best4
    blockers = [i for i, r in enumerate(comb4) if r <= 0]
    print("\n=== FEASIBILITY FRONTIER (idealized targeted leg vs the 4-way best-convex book) ===")
    print(f"  4-way blocker folds: {[YEARS[i] for i in blockers]}  (deficits: " +
          ", ".join(f"{YEARS[i]}:{comb4[i]*100:+.3f}%" for i in blockers) + ")")
    # idealized leg: +x in blockers, 0 else. 5-way fold f = (1-w5)*comb4[f] + w5*leg[f].
    # need (1-w5)*comb4[f] + w5*x > 0 for blockers  AND (1-w5)*comb4[g] > 0 for good folds (leg=0 there) -> always true.
    for w5 in (0.05, 0.10, 0.15, 0.20, 0.30):
        # required x per blocker: x > -(1-w5)/w5 * comb4[f]  (comb4[f] <= 0)
        req = max(-(1 - w5) / w5 * comb4[f] for f in blockers)
        print(f"  at w5={w5:.2f}: idealized leg needs +{req*100:.3f}% in EACH blocker fold (and >=0 elsewhere) to clear AFP")
    print("  -> compare required-x to the per-fold NOISE FLOOR (arc 2016/2023 ~0.70% sd): a thin leg whose")
    print("     blocker-fold ROI is within +/-0.70% CANNOT robustly deliver +req there.")


if __name__ == "__main__":
    main()
