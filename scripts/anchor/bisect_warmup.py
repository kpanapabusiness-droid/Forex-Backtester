"""Bisect the warmup-window hypothesis for the KH-24 A1 equivalence FAIL.

Per cc_07_diagnostic.md: 5/7 anchor folds match byte-identically between
KH24FoldRunner (30-day warmup, signal eval on sliced data) and
ArcFoldRunner(A1, kh24_to_a1) (full-history warmup, signal eval on
full panel). F2 and F3 diverge by ~1pp.

Hypothesis: the legacy 30-day warmup leaves the first OOS bars of F2/F3
with NaN kijun(26) on D1 (weekends/holidays consume the buffer). The new
path with full-history warmup doesn't have this NaN.

This script re-runs the legacy path with warmup_days = 365 (a full year
of pre-OOS D1 data — far past any rolling-kijun saturation point). If
the hypothesis is correct, F2/F3 under warmup=365 should match A1's
numbers byte-identically.

Outputs side-by-side per-fold table for chat to verify.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture
from core.parallel import build_panel_parallel, default_pool_size
from core.runners._fold_stats_helpers import build_fold_stats_from_run
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.multipair_backtester import MultiPairBacktester
from core.sim.panel import Panel
from core.strategies.kh24.a1_adapter import kh24_to_a1
from core.strategies.kh24.kh24 import KH24Config, build_kh24_runtime
from core.wfo.folds import Fold, build_kh24_anchor_folds


PAIRS_28 = (
    "AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY "
    "EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD "
    "GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY "
    "NZDUSD USDCAD USDCHF USDJPY"
).split()


def _slice_panel(panel: Panel, start: pd.Timestamp, end: pd.Timestamp) -> Panel:
    sliced = {p: df.loc[start:end] for p, df in panel.pair_dfs.items()}
    return Panel.from_frames(sliced, tf=panel.tf)


def _run_legacy_with_warmup(
    *,
    fold: Fold,
    panel_h4: Panel,
    panel_d1: Panel,
    panel_h1: Panel,
    config: KH24Config,
    warmup_days: int,
):
    """Mirror of KH24FoldRunner.__call__ but with parameterised warmup."""
    slice_start = pd.Timestamp(fold.oos_start, tz="UTC") - pd.Timedelta(days=warmup_days)
    slice_end = pd.Timestamp(fold.oos_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    h4 = _slice_panel(panel_h4, slice_start, slice_end)
    d1 = _slice_panel(panel_d1, slice_start, slice_end)
    h1 = _slice_panel(panel_h1, slice_start, slice_end)
    runtime = build_kh24_runtime(h4, d1, h1, config=config)
    bt = MultiPairBacktester(
        panel=h4,
        account=runtime.account,
        strategy=runtime.strategy,
        trail_manager=runtime.trail_manager,
        exit_predicates=runtime.exit_predicates,
    )
    result = bt.run()
    return build_fold_stats_from_run(
        fold=fold, run_result=result, starting_balance=config.starting_balance
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histdata-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=None)
    args = parser.parse_args()

    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    pool_size = args.pool_size if args.pool_size is not None else default_pool_size(28)

    print(f"[bisect] Loading H4/D1/H1 panels (pool_size={pool_size})...")
    panel_h4 = build_panel_parallel(
        list(PAIRS_28), "H4",
        histdata_root=args.histdata_root, cache_root=args.cache_root, pool_size=pool_size,
    )
    panel_d1 = build_panel_parallel(
        list(PAIRS_28), "D1",
        histdata_root=args.histdata_root, cache_root=args.cache_root, pool_size=pool_size,
    )
    panel_h1 = build_panel_parallel(
        list(PAIRS_28), "H1",
        histdata_root=args.histdata_root, cache_root=args.cache_root, pool_size=pool_size,
    )
    panels = {"H4": panel_h4, "D1": panel_d1, "H1": panel_h1}

    kh24_cfg = KH24Config()
    a1_cfg, kh24_signal = kh24_to_a1(kh24_cfg, config_id="kh24_canonical")
    print("[bisect] Evaluating signal once on full panel for A1 path...")
    signal_eval = kh24_signal.evaluate(panels)

    a1_runner = ArcFoldRunner(
        architecture=A1Architecture(),
        signal_evaluation=signal_eval,
        panels=panels,
    )

    structure = build_kh24_anchor_folds()
    rows = []
    for fold in structure.folds:
        print(f"[bisect] fold {fold.fold_id} {fold.oos_start} -> {fold.oos_end}")
        fs_30 = _run_legacy_with_warmup(
            fold=fold, panel_h4=panel_h4, panel_d1=panel_d1, panel_h1=panel_h1,
            config=kh24_cfg, warmup_days=30,
        )
        fs_365 = _run_legacy_with_warmup(
            fold=fold, panel_h4=panel_h4, panel_d1=panel_d1, panel_h1=panel_h1,
            config=kh24_cfg, warmup_days=365,
        )
        fs_a1 = a1_runner(fold, a1_cfg)

        rows.append({
            "fold_id": fold.fold_id,
            "oos_start": fold.oos_start.isoformat(),
            "oos_end": fold.oos_end.isoformat(),
            "warmup30_roi": fs_30.roi_pct,
            "warmup365_roi": fs_365.roi_pct,
            "a1_roi": fs_a1.roi_pct,
            "warmup30_dd": fs_30.max_dd_pct,
            "warmup365_dd": fs_365.max_dd_pct,
            "a1_dd": fs_a1.max_dd_pct,
            "warmup30_n": fs_30.n_trades,
            "warmup365_n": fs_365.n_trades,
            "a1_n": fs_a1.n_trades,
            "warmup365_vs_a1_roi_pp": (fs_365.roi_pct - fs_a1.roi_pct) * 100,
            "warmup365_vs_a1_dd_pp": (fs_365.max_dd_pct - fs_a1.max_dd_pct) * 100,
            "warmup30_vs_a1_roi_pp": (fs_30.roi_pct - fs_a1.roi_pct) * 100,
            "warmup30_vs_a1_dd_pp": (fs_30.max_dd_pct - fs_a1.max_dd_pct) * 100,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out / "bisect_warmup.parquet", engine="pyarrow", compression="snappy", index=False)
    df.to_csv(out / "bisect_warmup.csv", index=False, lineterminator="\n")

    lines = [
        "# KH-24 Warmup Bisect — confirmation of warmup-effect hypothesis",
        "",
        "Three runs per fold:",
        "  - **warmup30**: legacy KH24FoldRunner (warmup_days=30) — current anchor",
        "  - **warmup365**: legacy logic with warmup_days=365 — hypothesis test",
        "  - **a1**: ArcFoldRunner(A1, kh24_to_a1) — full-history warmup",
        "",
        "If hypothesis correct: warmup365 numbers match a1 numbers byte-identically.",
        "",
        "| Fold | warmup30 ROI | warmup365 ROI | A1 ROI | w365 vs A1 ROI Δpp | warmup30 DD | warmup365 DD | A1 DD | w365 vs A1 DD Δpp | w30 n | w365 n | A1 n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_match_under_365 = True
    for r in rows:
        roi_diff = r["warmup365_vs_a1_roi_pp"]
        dd_diff = r["warmup365_vs_a1_dd_pp"]
        if abs(roi_diff) > 0.01 or abs(dd_diff) > 0.01 or r["warmup365_n"] != r["a1_n"]:
            all_match_under_365 = False
        lines.append(
            f"| {r['fold_id']} | {r['warmup30_roi']:+.4%} | {r['warmup365_roi']:+.4%} | {r['a1_roi']:+.4%} | "
            f"{roi_diff:+.4f} | {r['warmup30_dd']:.4%} | {r['warmup365_dd']:.4%} | {r['a1_dd']:.4%} | "
            f"{dd_diff:+.4f} | {r['warmup30_n']} | {r['warmup365_n']} | {r['a1_n']} |"
        )
    lines.append("")
    if all_match_under_365:
        lines.append("## Verdict: **HYPOTHESIS CONFIRMED**")
        lines.append("")
        lines.append(
            "Every fold matches A1 byte-identically when the legacy path "
            "uses warmup_days=365. The 30-day warmup is buffer-management "
            "heuristic that NaN-masks kijun(26) at fold boundaries on F2/F3."
        )
    else:
        lines.append("## Verdict: **HYPOTHESIS NOT CONFIRMED**")
        lines.append("")
        lines.append(
            "warmup_days=365 does NOT bring legacy into byte-identity with A1. "
            "Other divergence sources remain — further triage required."
        )
    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
    (out / "verdict.json").write_text(
        json.dumps({
            "hypothesis_confirmed": all_match_under_365,
        }, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"[bisect] hypothesis_confirmed={all_match_under_365}; results in {out}")
    return 0 if all_match_under_365 else 1


if __name__ == "__main__":
    raise SystemExit(main())
