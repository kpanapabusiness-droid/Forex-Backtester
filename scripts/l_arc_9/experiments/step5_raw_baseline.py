"""Arc 9 Step 5 RAW BASELINE - full pool, no filter, default exit (SL=2.0xATR + 240-bar time-stop).

Held-open experiment under v2.3. The third reference point in the Arc 9 picture:

  oracle  (cluster 0, §11 exit)     -> done: 7/7 pass-deployable
  raw     (full pool, default exit) -> THIS RUN
  cal'd   (full pool, calibrated D1) -> pending

The default exit policy (SL=2.0xATR + 240-bar time-stop) is exactly the
exit policy that produced trades_all.csv at Step 1. There is no
re-simulation needed: final_r per trade is already canonical in
`results/l_arc_9/step1_verbatim/trades_all.csv`. This script re-accounts
those trades into the 7-fold WFO structure (from configs/wfo_kh24.yaml)
and applies §10 pass-deployable / pass-viable gates.

Reuses fold accounting (compute_fold_metrics, full_data_equity) from
scripts.l_arc_9.experiments.step5_validation so the comparison to the
oracle run is methodologically identical.

Outputs:
  results/l_arc_9/experiments/step5_raw_baseline/
    raw_baseline_trades.csv          trade_id, pair, entry_time, fold, final_r
    per_fold_metrics.csv             fold-level ROI, DD, n_trades
    full_data_metrics.json           full-data ROI, DD
    determinism_check.json           2-run sha256 comparison
    oracle_comparison.csv            side-by-side w/ Step 5 validation oracle metrics
    STEP5_RAW_BASELINE_RESULT.md     report
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_9.experiments.step5_validation import (  # noqa: E402
    STARTING_BALANCE,
    RISK_PCT,
    PASS_DEPLOYABLE,
    PASS_VIABLE,
    _compute_fold_metrics,
    _full_data_equity,
    evaluate_gates,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out_dir: Path, kh24_cfg_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_kh24 = yaml.safe_load(kh24_cfg_path.read_text(encoding="utf-8"))
    folds_yaml = cfg_kh24["wfo"]["folds"]
    fold_oos: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = [
        (int(f["fold"]), pd.Timestamp(f["oos_start"]), pd.Timestamp(f["oos_end"]))
        for f in folds_yaml
    ]
    fold_start_end_only: List[Tuple[pd.Timestamp, pd.Timestamp]] = [(s, e) for _, s, e in fold_oos]

    # Step 1 trades_all.csv carries final_r computed under SL=2.0xATR +
    # 240-bar time-stop exit policy, which is EXACTLY the default exit
    # policy this dispatch asks for. No re-simulation.
    trades = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["signal_bar_time"] = pd.to_datetime(trades["signal_bar_time"])
    n_total = int(len(trades))
    print(f"[raw baseline] full Arc 9 trade pool: {n_total} trades")

    # Stamp fold assignment for the persisted raw_baseline_trades.csv.
    def fold_of(ts: pd.Timestamp) -> int:
        for fid, s, e in fold_oos:
            if s <= ts < e:
                return fid
        return -1
    trades["fold"] = trades["entry_time"].apply(fold_of)

    keep_cols = [
        "trade_id", "pair", "signal_bar_time", "entry_time",
        "exit_time", "exit_reason", "bars_held", "final_r",
        "spread_pips_used", "spread_pips_exit", "fold",
    ]
    trades_keep = trades[keep_cols].sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
    trades_keep.to_csv(out_dir / "raw_baseline_trades.csv", index=False,
                       float_format="%.10g", lineterminator="\n")

    # Per-fold metrics. Trades whose entry_time falls in OOS [start, end).
    fold_rows: List[Dict[str, Any]] = []
    for (fold_id, s, e) in fold_oos:
        mask = (trades_keep["entry_time"] >= s) & (trades_keep["entry_time"] < e)
        sub = trades_keep[mask].reset_index(drop=True)
        fmetrics = _compute_fold_metrics(sub, s, e)
        fmetrics["fold"] = fold_id
        fmetrics["oos_start"] = s.strftime("%Y-%m-%d")
        fmetrics["oos_end"] = e.strftime("%Y-%m-%d")
        fold_rows.append(fmetrics)
    fold_df = pd.DataFrame(fold_rows)[[
        "fold", "oos_start", "oos_end", "n_trades", "final_r_mean",
        "final_r_sign_positive", "fold_roi_pct", "annualised_roi_pct",
        "max_dd_pct", "ending_equity",
    ]]
    fold_df.to_csv(out_dir / "per_fold_metrics.csv", index=False,
                   float_format="%.10g", lineterminator="\n")

    # Full-data accounting.
    in_window = trades_keep[
        (trades_keep["entry_time"] >= fold_start_end_only[0][0]) &
        (trades_keep["entry_time"] < fold_start_end_only[-1][1])
    ]
    full_m = _full_data_equity(in_window, fold_start_end_only)
    (out_dir / "full_data_metrics.json").write_text(
        json.dumps(full_m, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return trades_keep, fold_df, full_m, fold_start_end_only


def write_report(
    fold_df: pd.DataFrame, full_m: Dict[str, Any], gates: Dict[str, Any],
    oracle_fold_df: pd.DataFrame, oracle_full: Dict[str, Any], oracle_gates: Dict[str, Any],
    out_path: Path, out_dir: Path,
) -> None:
    if gates["pass_deployable"]:
        headline = "PASS-DEPLOYABLE"
    elif gates["pass_viable"]:
        headline = "PASS-VIABLE"
    else:
        headline = "FAIL"
    s = gates["summary"]
    os_ = oracle_gates["summary"]

    # Persist oracle_comparison.csv
    cmp_rows = [
        {"metric": "worst_fold_annualised_roi_pct", "raw_baseline": s["worst_fold_ann_roi_pct"], "oracle": os_["worst_fold_ann_roi_pct"], "gap": os_["worst_fold_ann_roi_pct"] - s["worst_fold_ann_roi_pct"]},
        {"metric": "mean_fold_annualised_roi_pct", "raw_baseline": s["mean_fold_ann_roi_pct"], "oracle": os_["mean_fold_ann_roi_pct"], "gap": os_["mean_fold_ann_roi_pct"] - s["mean_fold_ann_roi_pct"]},
        {"metric": "worst_fold_max_dd_pct", "raw_baseline": s["worst_fold_max_dd_pct"], "oracle": os_["worst_fold_max_dd_pct"], "gap": os_["worst_fold_max_dd_pct"] - s["worst_fold_max_dd_pct"]},
        {"metric": "min_trades_per_fold", "raw_baseline": s["min_trades_per_fold"], "oracle": os_["min_trades_per_fold"], "gap": os_["min_trades_per_fold"] - s["min_trades_per_fold"]},
        {"metric": "all_folds_positive", "raw_baseline": int(bool(s["all_folds_positive"])), "oracle": int(bool(os_["all_folds_positive"])), "gap": "-"},
        {"metric": "full_data_annualised_roi_pct", "raw_baseline": s["full_data_ann_roi_pct"], "oracle": os_["full_data_ann_roi_pct"], "gap": os_["full_data_ann_roi_pct"] - s["full_data_ann_roi_pct"]},
        {"metric": "full_data_max_dd_pct", "raw_baseline": s["full_data_max_dd_pct"], "oracle": os_["full_data_max_dd_pct"], "gap": os_["full_data_max_dd_pct"] - s["full_data_max_dd_pct"]},
    ]
    pd.DataFrame(cmp_rows).to_csv(out_dir / "oracle_comparison.csv", index=False,
                                  float_format="%.10g", lineterminator="\n")

    lines: List[str] = []
    lines.append("# Arc 9 Step 5 Raw Baseline - full pool, no filter, default exit")
    lines.append("")
    lines.append("> Held-open experiment under v2.3 §10. Establishes the raw signal floor —")
    lines.append("> deployable economics of the IB-trend signal as bare entry rule + default")
    lines.append("> protocol exit (SL=2.0×ATR + 240-bar time-stop). Third reference point")
    lines.append("> in the Arc 9 picture (oracle Step 5 + calibration recovery already done).")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"**{headline}** — worst-fold annualised ROI {s['worst_fold_ann_roi_pct']:+.2f}%, "
                 f"mean fold annualised ROI {s['mean_fold_ann_roi_pct']:+.2f}%, "
                 f"worst-fold max DD {s['worst_fold_max_dd_pct']:.2f}%.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Trade pool: 2,153 trades — the full Arc 9 Step 1 output. No clustering, no admission filter.")
    lines.append("- Entry: every signal taken at signal-bar-close → next-bar open (long, +S/2 fill).")
    lines.append("- SL: entry − 2.0×ATR(14)_4H at signal bar (anchored to entry_fill).")
    lines.append("- Time exit: bar entry + 240 (40 calendar days at 4H), fill at open of bar 240.")
    lines.append("- No MFE-lock, no trail, no §11 archetype routing.")
    lines.append("- Provenance: `final_r` per trade is read directly from `results/l_arc_9/step1_verbatim/trades_all.csv`")
    lines.append("  (Step 1 already executed under exactly this exit policy + live-execution semantics; no re-simulation needed).")
    lines.append("- Folds: 7 OOS windows from `configs/wfo_kh24.yaml` (anchor); same fold structure as the oracle Step 5 run.")
    lines.append("- Risk per trade: 0.5% compounded from $10k starting balance (same as oracle run).")
    lines.append("- Per-bar spread: real MT5 spread, floor only when raw spread = 0 (SPREAD_SEMANTICS_LOCK).")
    lines.append("")
    lines.append("## Per-fold table")
    lines.append("")
    lines.append("| Fold | OOS window | Trades | Final R mean | Sign | Compounded ROI (%) | Annualised ROI (%) | Max DD (%) | Ending equity |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in fold_df.iterrows():
        sign_str = "+" if r["final_r_sign_positive"] else "-"
        lines.append(
            f"| F{int(r['fold'])} | {r['oos_start']} → {r['oos_end']} | "
            f"{int(r['n_trades'])} | {r['final_r_mean']:+.3f} | {sign_str} | "
            f"{r['fold_roi_pct']:+.2f} | {r['annualised_roi_pct']:+.2f} | "
            f"{r['max_dd_pct']:.2f} | ${r['ending_equity']:,.2f} |"
        )
    lines.append("")
    lines.append("## Full-data (compounded across folds in time order)")
    lines.append("")
    lines.append(f"- n trades: {full_m['n_trades']}")
    lines.append(f"- Compounded ROI: {full_m['full_data_roi_pct']:+.2f}%")
    lines.append(f"- Annualised ROI: {full_m['full_data_annualised_roi_pct']:+.2f}%")
    lines.append(f"- Max DD: {full_m['full_data_max_dd_pct']:.2f}%")
    lines.append(f"- Ending equity: ${full_m['ending_equity']:,.2f}")
    lines.append("")
    lines.append("## Gates evaluated (§10 pass-deployable + pass-viable)")
    lines.append("")
    lines.append("### Pass-deployable")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Pass? |")
    lines.append("|---|---|---|---|")
    pdc = gates["pd_checks"]
    lines.append(f"| Worst-fold annualised ROI | ≥ {PASS_DEPLOYABLE['worst_fold_roi_min_pct_annualised']}% | {pdc['worst_fold_ann_roi'][0]:+.2f}% | {'PASS' if pdc['worst_fold_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Mean fold annualised ROI | ≥ {PASS_DEPLOYABLE['mean_fold_roi_min_pct_annualised']}% | {pdc['mean_fold_ann_roi'][0]:+.2f}% | {'PASS' if pdc['mean_fold_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Worst-fold max DD | ≤ {PASS_DEPLOYABLE['worst_fold_dd_max_pct']}% | {pdc['worst_fold_dd_pct'][0]:.2f}% | {'PASS' if pdc['worst_fold_dd_pct'][2] else 'FAIL'} |")
    lines.append(f"| All folds positive | required | {pdc['all_folds_positive'][0]} | {'PASS' if pdc['all_folds_positive'][2] else 'FAIL'} |")
    lines.append(f"| Trade count per fold | ≥ {PASS_DEPLOYABLE['trade_count_per_fold_min']} | min {pdc['trade_count_per_fold_min'][0]} | {'PASS' if pdc['trade_count_per_fold_min'][2] else 'FAIL'} |")
    lines.append(f"| Full-data annualised ROI | ≥ {PASS_DEPLOYABLE['full_data_roi_min_pct_annualised']}% | {pdc['full_data_ann_roi'][0]:+.2f}% | {'PASS' if pdc['full_data_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Full-data max DD | ≤ {PASS_DEPLOYABLE['full_data_dd_max_pct']}% | {pdc['full_data_dd_pct'][0]:.2f}% | {'PASS' if pdc['full_data_dd_pct'][2] else 'FAIL'} |")
    lines.append(f"| **Overall pass-deployable** | all 7 | - | **{'PASS' if gates['pass_deployable'] else 'FAIL'}** |")
    lines.append("")
    lines.append("### Pass-viable")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Pass? |")
    lines.append("|---|---|---|---|")
    pvc = gates["pv_checks"]
    lines.append(f"| Worst-fold annualised ROI | > {PASS_VIABLE['worst_fold_roi_min_pct_annualised']}% (positive) | {pvc['worst_fold_ann_roi'][0]:+.2f}% | {'PASS' if pvc['worst_fold_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Mean fold annualised ROI | ≥ {PASS_VIABLE['mean_fold_roi_min_pct_annualised']}% | {pvc['mean_fold_ann_roi'][0]:+.2f}% | {'PASS' if pvc['mean_fold_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Worst-fold max DD | ≤ {PASS_VIABLE['worst_fold_dd_max_pct']}% | {pvc['worst_fold_dd_pct'][0]:.2f}% | {'PASS' if pvc['worst_fold_dd_pct'][2] else 'FAIL'} |")
    lines.append(f"| All folds positive | required | {pvc['all_folds_positive'][0]} | {'PASS' if pvc['all_folds_positive'][2] else 'FAIL'} |")
    lines.append(f"| Trade count per fold | ≥ {PASS_VIABLE['trade_count_per_fold_min']} | min {pvc['trade_count_per_fold_min'][0]} | {'PASS' if pvc['trade_count_per_fold_min'][2] else 'FAIL'} |")
    lines.append(f"| Full-data annualised ROI | ≥ {PASS_VIABLE['full_data_roi_min_pct_annualised']}% | {pvc['full_data_ann_roi'][0]:+.2f}% | {'PASS' if pvc['full_data_ann_roi'][2] else 'FAIL'} |")
    lines.append(f"| Full-data max DD | ≤ {PASS_VIABLE['full_data_dd_max_pct']}% | {pvc['full_data_dd_pct'][0]:.2f}% | {'PASS' if pvc['full_data_dd_pct'][2] else 'FAIL'} |")
    lines.append(f"| **Overall pass-viable** | all 7 | - | **{'PASS' if gates['pass_viable'] else 'FAIL'}** |")
    lines.append("")
    lines.append("## Comparison to oracle Step 5 (cluster 0 only, §11 Stepwise exit)")
    lines.append("")
    lines.append("| Metric | Raw baseline (full pool, default exit) | Oracle (cluster 0 only, §11 exit) | Gap (oracle - raw) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Worst-fold ann ROI | {s['worst_fold_ann_roi_pct']:+.2f}% | {os_['worst_fold_ann_roi_pct']:+.2f}% | {os_['worst_fold_ann_roi_pct'] - s['worst_fold_ann_roi_pct']:+.2f}pp |")
    lines.append(f"| Mean fold ann ROI | {s['mean_fold_ann_roi_pct']:+.2f}% | {os_['mean_fold_ann_roi_pct']:+.2f}% | {os_['mean_fold_ann_roi_pct'] - s['mean_fold_ann_roi_pct']:+.2f}pp |")
    lines.append(f"| Worst-fold DD | {s['worst_fold_max_dd_pct']:.2f}% | {os_['worst_fold_max_dd_pct']:.2f}% | {os_['worst_fold_max_dd_pct'] - s['worst_fold_max_dd_pct']:+.2f}pp |")
    lines.append(f"| Full-data ann ROI | {s['full_data_ann_roi_pct']:+.2f}% | {os_['full_data_ann_roi_pct']:+.2f}% | {os_['full_data_ann_roi_pct'] - s['full_data_ann_roi_pct']:+.2f}pp |")
    lines.append(f"| Full-data DD | {s['full_data_max_dd_pct']:.2f}% | {os_['full_data_max_dd_pct']:.2f}% | {os_['full_data_max_dd_pct'] - s['full_data_max_dd_pct']:+.2f}pp |")
    lines.append(f"| Trades total | {int(fold_df['n_trades'].sum())} | {int(oracle_fold_df['n_trades'].sum())} | — |")
    lines.append(f"| Pass-deployable? | {'YES' if gates['pass_deployable'] else 'NO'} | {'YES' if oracle_gates['pass_deployable'] else 'NO'} | — |")
    lines.append("")
    # Interpretation written manually after run; placeholder block.
    lines.append("## Interpretation")
    lines.append("")
    lines.append("_See manually-curated interpretation appended below; script renders headline + tables only._")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 9 Step 5 raw baseline - full pool, no filter, default exit.")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_raw_baseline")
    p.add_argument("--kh24-cfg", type=Path, default=_REPO_ROOT / "configs" / "wfo_kh24.yaml")
    p.add_argument("--verify-determinism", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trades_keep, fold_df, full_m, fold_se = run(args.out_dir, args.kh24_cfg)
    gates = evaluate_gates(fold_df, full_m)

    if args.verify_determinism:
        scratch = args.out_dir / "_determinism_scratch"
        scratch.mkdir(exist_ok=True)
        run(scratch, args.kh24_cfg)
        sha_trades_1 = _sha256_file(args.out_dir / "raw_baseline_trades.csv")
        sha_trades_2 = _sha256_file(scratch / "raw_baseline_trades.csv")
        sha_folds_1 = _sha256_file(args.out_dir / "per_fold_metrics.csv")
        sha_folds_2 = _sha256_file(scratch / "per_fold_metrics.csv")
        det = {
            "trades_run1_sha256": sha_trades_1, "trades_run2_sha256": sha_trades_2,
            "folds_run1_sha256": sha_folds_1, "folds_run2_sha256": sha_folds_2,
            "byte_identical": bool(sha_trades_1 == sha_trades_2 and sha_folds_1 == sha_folds_2),
        }
        (args.out_dir / "determinism_check.json").write_text(
            json.dumps(det, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for f in scratch.iterdir():
            f.unlink()
        scratch.rmdir()
        print(f"[raw baseline] determinism: {'PASS' if det['byte_identical'] else 'FAIL'}")

    # Load oracle Step 5 results for the comparison block.
    oracle_dir = _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_validation"
    oracle_fold_df = pd.read_csv(oracle_dir / "per_fold_metrics.csv")
    oracle_full = json.loads((oracle_dir / "full_data_metrics.json").read_text(encoding="utf-8"))
    oracle_gates = evaluate_gates(oracle_fold_df, oracle_full)

    write_report(
        fold_df, full_m, gates, oracle_fold_df, oracle_full, oracle_gates,
        args.out_dir / "STEP5_RAW_BASELINE_RESULT.md", args.out_dir,
    )

    s = gates["summary"]
    head = "PASS-DEPLOYABLE" if gates["pass_deployable"] else "PASS-VIABLE" if gates["pass_viable"] else "FAIL"
    print(f"[raw baseline] HEADLINE: {head}")
    print(f"  worst-fold ann ROI: {s['worst_fold_ann_roi_pct']:+.2f}%  "
          f"mean-fold ann ROI: {s['mean_fold_ann_roi_pct']:+.2f}%  "
          f"worst-fold DD: {s['worst_fold_max_dd_pct']:.2f}%")
    print(f"  full-data ann ROI: {s['full_data_ann_roi_pct']:+.2f}%  "
          f"full-data DD: {s['full_data_max_dd_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
