"""Chat-runnable: confirm KH24FoldRunner == ArcFoldRunner(A1, kh24_to_a1(...))
on real HistData.

This is the full-data equivalence check the dispatch's PR 7.D + 7.F
anchor checkpoints depend on. CC's session has no HistData layer
locally, so this script exists for chat to run on the workstation
where the M1 layer is available.

Usage::

    py -m scripts.anchor.check_a1_equivalence \\
        --histdata-root C:/Users/panap/Documents/Forex-Backtester/data/histdata \\
        --cache-root    C:/Users/panap/AppData/Local/Temp/pr_e2_cache \\
        --out-root      results/anchor_kh24_a1_check \\
        --tolerance-pp  0.5

The script:
  1. Builds H4/D1/H1 panels for the 28 FX universe
  2. Runs each KH-24 anchor fold via KH24FoldRunner (legacy path)
  3. Runs the same fold via ArcFoldRunner(A1, kh24_to_a1)
  4. Compares per-fold ROI / DD / n_trades
  5. PASSES iff every fold matches within ±tolerance_pp ROI / ±1pp DD

Output: a side-by-side comparison parquet + summary.md + a PASS/FAIL
verdict file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture
from core.parallel import build_panel_parallel, default_pool_size
from core.runners.arc_fold_runner import ArcFoldRunner
from core.strategies.kh24.a1_adapter import kh24_to_a1
from core.strategies.kh24.kh24 import KH24Config
from core.wfo.fold_runner import KH24FoldRunner
from core.wfo.folds import build_kh24_anchor_folds


PAIRS_28 = (
    "AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY "
    "EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD "
    "GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY "
    "NZDUSD USDCAD USDCHF USDJPY"
).split()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histdata-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--tolerance-pp", type=float, default=0.5,
                        help="±N percentage points ROI tolerance (default 0.5)")
    parser.add_argument("--pool-size", type=int, default=None)
    args = parser.parse_args()

    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)

    pool_size = args.pool_size if args.pool_size is not None else default_pool_size(28)

    print(f"[a1_equiv] Loading H4/D1/H1 panels (pool_size={pool_size})...")
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

    structure = build_kh24_anchor_folds()
    kh24_cfg = KH24Config()
    a1_cfg, kh24_signal = kh24_to_a1(kh24_cfg, config_id="kh24_canonical")

    print("[a1_equiv] Evaluating signal once for the new path...")
    signal_eval = kh24_signal.evaluate(panels)

    legacy = KH24FoldRunner(panel_h4=panel_h4, panel_d1=panel_d1, panel_h1=panel_h1)
    new = ArcFoldRunner(
        architecture=A1Architecture(),
        signal_evaluation=signal_eval,
        panels=panels,
    )

    rows = []
    fail = False
    for fold in structure.folds:
        print(f"[a1_equiv] fold {fold.fold_id} {fold.oos_start} → {fold.oos_end} ...")
        fs_legacy = legacy(fold, kh24_cfg)
        fs_new = new(fold, a1_cfg)
        roi_diff_pp = (fs_new.roi_pct - fs_legacy.roi_pct) * 100
        dd_diff_pp = (fs_new.max_dd_pct - fs_legacy.max_dd_pct) * 100
        trades_diff = fs_new.n_trades - fs_legacy.n_trades
        within_roi = abs(roi_diff_pp) <= args.tolerance_pp
        within_dd = abs(dd_diff_pp) <= 1.0
        fold_ok = within_roi and within_dd
        if not fold_ok:
            fail = True
        rows.append({
            "fold_id": fold.fold_id,
            "oos_start": fold.oos_start.isoformat(),
            "oos_end": fold.oos_end.isoformat(),
            "legacy_roi_pct": fs_legacy.roi_pct,
            "new_roi_pct": fs_new.roi_pct,
            "roi_diff_pp": roi_diff_pp,
            "legacy_max_dd_pct": fs_legacy.max_dd_pct,
            "new_max_dd_pct": fs_new.max_dd_pct,
            "dd_diff_pp": dd_diff_pp,
            "legacy_n_trades": fs_legacy.n_trades,
            "new_n_trades": fs_new.n_trades,
            "trades_diff": trades_diff,
            "within_tolerance": fold_ok,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out / "a1_equivalence.parquet", engine="pyarrow", compression="snappy", index=False)
    df.to_csv(out / "a1_equivalence.csv", index=False, lineterminator="\n")

    lines = [
        "# KH-24 A1 Equivalence Check",
        "",
        f"Tolerance: ±{args.tolerance_pp:.2f}pp ROI / ±1pp DD per fold.",
        "",
        "| Fold | Legacy ROI | New ROI | Δpp | Legacy DD | New DD | Δpp | Legacy n | New n | OK? |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fold_id']} | {r['legacy_roi_pct']:+.4%} | {r['new_roi_pct']:+.4%} | "
            f"{r['roi_diff_pp']:+.2f} | {r['legacy_max_dd_pct']:.4%} | "
            f"{r['new_max_dd_pct']:.4%} | {r['dd_diff_pp']:+.2f} | "
            f"{r['legacy_n_trades']} | {r['new_n_trades']} | "
            f"{'✓' if r['within_tolerance'] else '✗'} |"
        )
    lines.append("")
    verdict = "PASS" if not fail else "FAIL"
    lines.append(f"## Verdict: **{verdict}**")
    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
    (out / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "tolerance_pp": args.tolerance_pp}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[a1_equiv] Verdict: {verdict}; results in {out}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
