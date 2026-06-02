"""Supplementary Stage-B evaluation of the A1 PRIMARY hypothesis configs.

The dispatch's primary hypothesis is A1 full-pool + asymmetric runner-trail exit
(the Arc 10 winning frame). In the main run ALL 24 A1 configs were screened out at
Stage-A triage (negative worst-fold ROI, DD@0.40% > 8%), so none reached the
top-3 Stage-B gate. To document the primary hypothesis with a real 11-fold gate
(not just a 3-fold triage), this runs the 6 A1 full-pool `sl_partial_close_1r_runner_trail`
configs (SL {2.5,3.0,3.5} × exp {2,unlimited}) — the exact Arc 10 extractor — through
the full 11-fold + holdout WFO + Amendment-3 gate. Reset-floor sizing, r_base 0.5%.

A1 full-pool needs no classifier/feature matrix, so this is fast (panels + pool only).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool  # noqa: E402
from core.architectures.a1_system_level_filter import A1Architecture, A1Config, A1RunContext  # noqa: E402
from core.determinism import seed_everything  # noqa: E402
from core.runners._fold_stats_helpers import compute_per_day_max_dd  # noqa: E402
from core.strategies.liquidity_sweep_reclaim_long.signal_module import LiquiditySweepReclaimLongSignal  # noqa: E402
from core.wfo.amended_gates import classify_amended_fold_stats, compute_scaling_factors  # noqa: E402
from core.wfo.chained_dd import compute_chained_max_dd_from_continuous_equity, stitch_per_fold_oos_equity  # noqa: E402
from core.wfo.folds import build_v3_folds  # noqa: E402

from scripts.arc_7.run_arc_7 import PAIRS_28, _build_panel_5ers_eet  # noqa: E402
from scripts.arc_7_v3_0_exit_extraction.run import (  # noqa: E402
    R_BASE, STARTING_BALANCE, OPERATING_R, _run_config_over_folds, _exp_str,
)

log = logging.getLogger("a1_primary_stageb")
SLS = (2.5, 3.0, 3.5)
EXPS = (2, None)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    seed_everything(42)
    t0 = time.perf_counter()
    sibling = REPO_ROOT.parent / "nice-mirzakhani-baa9e5"
    out = REPO_ROOT / "results" / "arc_7_v3.0_exit_extraction"
    s5 = out / "step_5"; s5.mkdir(parents=True, exist_ok=True)

    log.info("Building panels + pool …")
    ph4 = _build_panel_5ers_eet(list(PAIRS_28), "H4", histdata_root=sibling / "data" / "histdata", cache_root=sibling / "data" / "cache")
    pd1 = _build_panel_5ers_eet(list(PAIRS_28), "D1", histdata_root=sibling / "data" / "histdata", cache_root=sibling / "data" / "cache")
    pw1 = _build_panel_5ers_eet(list(PAIRS_28), "W1", histdata_root=sibling / "data" / "histdata", cache_root=sibling / "data" / "cache")
    object.__setattr__(ph4, "aux", {"d1": pd1, "w1": pw1})
    panels = {"H4": ph4, "D1": pd1, "W1": pw1}
    pool = build_arc_pool(LiquiditySweepReclaimLongSignal(), panels, ArcPoolConfig(
        arc_name="arc_7_v3.0_exit_extraction", sl_atr_mult=2.0, hold_bars=240, risk_pct=R_BASE,
        window_start=pd.Timestamp("2010-01-01").date(), window_end=pd.Timestamp("2025-12-31").date()))
    log.info("pool %d sha=%s", len(pool.trades), pool.pool_sha256[:16])

    wfo = build_v3_folds(holdout_end=pd.Timestamp("2025-12-31").date())
    search_folds = list(wfo.folds)
    folds_b = search_folds + ([wfo.holdout] if wfo.holdout else [])
    ctx = A1RunContext()

    rows, per_fold = [], []
    for sl in SLS:
        for cap in EXPS:
            cid = f"A1::full::sl_partial_close_1r_runner_trail::sl{sl:.1f}::exp{_exp_str(cap)}"
            log.info("Stage-B(A1 primary) %s", cid)
            cfg = A1Config(config_id=cid, sl_atr_mult=sl, trail_enabled=False, risk_pct=R_BASE,
                           starting_balance=STARTING_BALANCE, max_concurrent_per_currency=cap,
                           max_concurrent_per_pair=1, max_concurrent_total=None,
                           exit_policy="sl_partial_close_1r_runner_trail")
            res = _run_config_over_folds(A1Architecture(), cfg, ctx, pool.signal_evaluation, panels, folds_b)
            search_stats = [res[f.fold_id][0] for f in search_folds]
            holdout = res[wfo.holdout.fold_id][0] if wfo.holdout else None
            eqs = [res[f.fold_id][1] for f in folds_b if len(res[f.fold_id][1]) > 0]
            chained_eq = stitch_per_fold_oos_equity(eqs, starting_balance=STARTING_BALANCE) if eqs else pd.Series(dtype=float)
            chained_dd = compute_chained_max_dd_from_continuous_equity(chained_eq) if len(chained_eq) else 0.0
            per_day = compute_per_day_max_dd(chained_eq, pair_set=",".join(PAIRS_28)) if len(chained_eq) else pd.DataFrame()
            worst_dd_base = max((s.max_dd_pct for s in search_stats), default=0.0)
            _ = compute_scaling_factors(worst_dd_base, r_base=R_BASE)
            gate = classify_amended_fold_stats(folds=search_stats, chained_max_dd_base_pct=chained_dd,
                                               per_day_max_dd_df=per_day if not per_day.empty else None,
                                               holdout_stats_at_r_safe=None, holdout_stats_at_r_hard=None,
                                               sizing_convention="reset_floor", accept_equity_pct=False, r_base=R_BASE)
            for f in folds_b:
                s = res[f.fold_id][0]
                per_fold.append(dict(config_id=cid, fold_id=f.fold_id,
                                     segment=("holdout" if (wfo.holdout and f.fold_id == wfo.holdout.fold_id) else "search"),
                                     calendar_year=(2009 + f.fold_id if f.fold_id <= 11 else 2021),
                                     n_trades=s.n_trades, roi_pct=s.roi_pct, max_dd_base_pct=s.max_dd_pct,
                                     dd_at_0p40_pct=s.max_dd_pct * (OPERATING_R / R_BASE),
                                     days_breaching_daily_5pct=s.days_breaching_daily_5pct, roi_dd_ratio=s.roi_dd_ratio))
            rows.append(dict(config_id=cid, verdict=gate.verdict.value,
                             primary_failure_mode=getattr(gate.primary_failure_mode, "value", str(gate.primary_failure_mode)),
                             worst_fold_ratio=gate.worst_fold_ratio, mean_fold_ratio=gate.mean_fold_ratio,
                             worst_fold_roi_base_pct=gate.worst_fold_roi_base_pct,
                             worst_fold_dd_base_pct=gate.worst_fold_dd_base_pct,
                             worst_fold_dd_at_0p40_pct=gate.worst_fold_dd_base_pct * (OPERATING_R / R_BASE),
                             n_negative_folds=gate.n_negative_folds, min_trades_per_fold=gate.min_trades_per_fold,
                             chained_max_dd_base_pct=chained_dd, k_safe=gate.k_safe, r_safe_pct=gate.r_safe_pct,
                             scalable_to_safe=gate.scalable_to_safe,
                             worst_fold_roi_at_r_safe_pct=gate.worst_fold_roi_at_r_safe_pct,
                             chained_max_dd_at_r_safe_pct=gate.chained_max_dd_at_r_safe_pct,
                             daily_dd_breaches_at_r_safe=gate.daily_dd_breaches_at_r_safe,
                             holdout_roi_pct=(holdout.roi_pct if holdout else None),
                             holdout_dd_base_pct=(holdout.max_dd_pct if holdout else None),
                             holdout_n_trades=(holdout.n_trades if holdout else None)))
            log.info("  -> %s worst_ratio=%.2f worst_roi=%.2f%% worst_dd_base=%.2f%% chained_dd=%.2f%% neg_folds=%d min_n=%d",
                     gate.verdict.value, gate.worst_fold_ratio, gate.worst_fold_roi_base_pct * 100,
                     gate.worst_fold_dd_base_pct * 100, chained_dd * 100, gate.n_negative_folds, gate.min_trades_per_fold)

    df = pd.DataFrame(rows).sort_values("worst_fold_ratio", ascending=False)
    df.to_csv(s5 / "a1_primary_stageb.csv", index=False, lineterminator="\n")
    pd.DataFrame(per_fold).to_csv(s5 / "a1_primary_per_fold.csv", index=False, lineterminator="\n")
    (s5 / "a1_primary_stageb.json").write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    log.info("DONE in %.1fs — all A1 verdicts: %s", time.perf_counter() - t0, list(df["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
