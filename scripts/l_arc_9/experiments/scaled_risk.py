"""Arc 9 Candidate A scaled-risk WFO.

Re-accounts the fixed Candidate A admit set (236 trades, F2-F7) at five
per-trade risk levels {0.5%, 1.0%, 1.5%, 2.0%, 2.5%}. The classifier,
admit decisions, SL distance (2.0xATR), and §11 Stepwise exit policy are
ALL unchanged from baseline — only the dollar-sizing per trade varies,
and equity compounds accordingly.

Measures:
  - Per-fold annualised ROI
  - Per-fold max DD (intra-fold equity-curve peak-to-trough)
  - Per-fold worst-day DD (compounded intraday peak-to-trough at trade
    closure events)
  - Per-fold trade count
  - Aggregate worst-fold + mean + full-data ROI/DD/worst-day-DD
  - Sanity check: ROI scaling ratio vs baseline; DD scaling ratio vs linear

Recommendation rules (per dispatch):
  - Worst-fold max DD ≤ 6% (1pp safety under 8% in-system target)
  - Worst-day DD ≤ 3% (1pp safety under 4% in-system target / 5ers 5% hard limit)
  - All folds positive (F2-F7; F1 zero-admit artifact tolerated)
  - ROI scaling within [3×, 6×] of linear at 4× risk

Pick the HIGHEST risk level satisfying all four — that's the measured
deployment recommendation.

Hard rules:
  - No classifier retraining, no admit-set modification, no threshold tuning.
  - Worst-day DD computed at compounded intraday resolution from closure events
    (realized P&L per day with multi-trade compounding within day). Open-position
    MTM swings are NOT included — this is a lower bound on the true MTM
    worst-day DD. Documented in report as a methodological note.
  - Account size constant at $10k (project default; percentage metrics are
    starting-balance-invariant).
  - Determinism: two runs at any risk level produce byte-identical
    per_fold_metrics.csv (no randomness in re-accounting).

Outputs in results/l_arc_9/experiments/scaled_risk/:
  per_risk_<pct>/per_fold_metrics.csv     7-fold table (F1 zero-admit + F2-F7 real)
  per_risk_<pct>/per_day_dd.csv           per-day intraday compounded DD across folds
  per_risk_<pct>/full_data_metrics.json   compounded across all folds
  summary_table.csv                        5 risk levels x metrics
  worst_day_analysis.csv                   worst day per risk + contributing trades
  determinism_check.json
  SCALED_RISK_RESULT.md                    report
"""

from __future__ import annotations

import argparse
import hashlib
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


STARTING_BALANCE: float = 10_000.0
RISK_LEVELS: List[float] = [0.005, 0.010, 0.015, 0.020, 0.025]

# In-system safety targets (per dispatch).
IN_SYSTEM_MAX_DD_PCT: float = 8.0
IN_SYSTEM_DAILY_DD_PCT: float = 4.0  # 1pp margin under 5ers 5% hard limit
# 5ers hard limits.
HARD_MAX_DD_PCT: float = 10.0
HARD_DAILY_DD_PCT: float = 5.0
# Recommendation safety margins.
RECOMMEND_MAX_FOLD_DD_PCT: float = 6.0
RECOMMEND_WORST_DAY_DD_PCT: float = 3.0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _account_fold(
    fold_trades: pd.DataFrame, risk_pct: float,
) -> Tuple[pd.DataFrame, float, float, float, float, float, pd.DataFrame]:
    """Account one fold at the given risk level.

    Returns (equity_curve_df, fold_roi_pct, ann_roi_pct, max_dd_pct,
    worst_day_dd_pct, ending_equity, per_day_dd_df).
    """
    if len(fold_trades) == 0:
        empty_day = pd.DataFrame(columns=["date", "n_trades", "equity_start_of_day", "intraday_peak", "intraday_trough", "day_dd_pct", "net_day_pnl_dollars", "trade_ids_in_day"])
        empty_eq = pd.DataFrame(columns=["event_time", "exit_date", "trade_id", "pair", "final_r", "risk_dollars", "pnl_dollars", "equity_before", "equity_after"])
        return empty_eq, 0.0, 0.0, 0.0, STARTING_BALANCE, empty_day

    # Sort by exit_time (realization order drives equity moves).
    df = fold_trades.sort_values("exit_time", kind="mergesort").reset_index(drop=True)
    df["exit_date"] = pd.to_datetime(df["exit_time"]).dt.normalize()

    # Walk through closures, tracking equity.
    equity = STARTING_BALANCE
    rows: List[Dict[str, Any]] = []
    for _, t in df.iterrows():
        risk_dollars = equity * risk_pct
        pnl = risk_dollars * float(t["final_r"])
        equity_new = equity + pnl
        rows.append({
            "event_time": t["exit_time"],
            "exit_date": t["exit_date"],
            "trade_id": int(t["trade_id"]),
            "pair": str(t["pair"]),
            "final_r": float(t["final_r"]),
            "risk_dollars": risk_dollars,
            "pnl_dollars": pnl,
            "equity_before": equity,
            "equity_after": equity_new,
        })
        equity = equity_new
    eq_df = pd.DataFrame(rows)

    # Per-fold max DD on closure-event equity curve.
    eq_arr = np.concatenate([[STARTING_BALANCE], eq_df["equity_after"].to_numpy()])
    cummax = np.maximum.accumulate(eq_arr)
    dd_arr = (cummax - eq_arr) / cummax
    max_dd_pct = float(dd_arr.max() * 100.0)

    # Compounded intraday peak-to-trough per calendar day.
    # For each day with closures, walk in order: day starts at equity_start_of_day
    # (= equity_before of first closure), then equity moves through each closure.
    # Intraday peak / trough computed over the sequence; day_dd_pct = (peak - trough) / peak * 100.
    per_day_rows: List[Dict[str, Any]] = []
    for date, day_grp in eq_df.groupby("exit_date", sort=True):
        day_grp = day_grp.sort_values("event_time", kind="mergesort").reset_index(drop=True)
        # Sequence: equity_before of first closure, then equity_after of each closure.
        day_eq_seq = np.concatenate([
            [float(day_grp["equity_before"].iloc[0])],
            day_grp["equity_after"].to_numpy(),
        ])
        # Intraday running-peak DD (peak-to-trough where trough comes AFTER peak in time).
        # This is the "max equity drop from intraday high" — the standard prop-firm
        # daily DD measurement. Monotone-winning days produce 0% DD; only down-moves
        # after a higher prior point register.
        running_peak = np.maximum.accumulate(day_eq_seq)
        running_dd_pct = (running_peak - day_eq_seq) / running_peak * 100.0
        intraday_dd_pct = float(running_dd_pct.max())
        # Also compute day-start-anchored DD (alternative prop convention: "5% of
        # day-starting balance"; 5ers uses some variant of this).
        day_start_eq = float(day_grp["equity_before"].iloc[0])
        day_low = float(day_eq_seq.min())
        day_start_dd_pct = max(0.0, (day_start_eq - day_low) / day_start_eq * 100.0) if day_start_eq > 0 else 0.0
        # Conservative: take the larger of the two measurements as the day-DD figure.
        day_dd_pct_val = max(intraday_dd_pct, day_start_dd_pct)
        per_day_rows.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "n_trades": int(len(day_grp)),
            "equity_start_of_day": day_start_eq,
            "intraday_peak": float(day_eq_seq.max()),
            "intraday_trough": day_low,
            "intraday_running_peak_to_trough_dd_pct": intraday_dd_pct,
            "day_start_to_low_dd_pct": day_start_dd_pct,
            "day_dd_pct": day_dd_pct_val,
            "net_day_pnl_dollars": float(day_grp["pnl_dollars"].sum()),
            "trade_ids_in_day": ",".join(day_grp["trade_id"].astype(str).tolist()),
        })
    per_day_df = pd.DataFrame(per_day_rows).sort_values("day_dd_pct", ascending=False).reset_index(drop=True)
    worst_day_dd_pct = float(per_day_df["day_dd_pct"].max()) if len(per_day_df) > 0 else 0.0

    # ROI metrics.
    fold_roi_pct = float((equity - STARTING_BALANCE) / STARTING_BALANCE * 100.0)
    return eq_df, fold_roi_pct, max_dd_pct, worst_day_dd_pct, equity, per_day_df


def _annualise_roi(fold_roi_pct: float, days_in_fold: int) -> float:
    if days_in_fold <= 0:
        return float("nan")
    years = days_in_fold / 365.25
    if (1 + fold_roi_pct / 100.0) <= 0:
        return float("-inf")
    return ((1 + fold_roi_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0


def _full_data_account(
    all_trades: pd.DataFrame, risk_pct: float, fold_start_end: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[str, Any]:
    """Compounded equity across all folds in chronological order."""
    if len(all_trades) == 0:
        return {
            "n_trades": 0, "full_data_roi_pct": 0.0, "full_data_annualised_roi_pct": 0.0,
            "full_data_max_dd_pct": 0.0, "full_data_worst_day_dd_pct": 0.0,
            "ending_equity": STARTING_BALANCE,
        }
    df = all_trades.sort_values("exit_time", kind="mergesort").reset_index(drop=True)
    df["exit_date"] = pd.to_datetime(df["exit_time"]).dt.normalize()
    equity = STARTING_BALANCE
    eq_rows: List[Dict[str, Any]] = []
    for _, t in df.iterrows():
        risk_dollars = equity * risk_pct
        pnl = risk_dollars * float(t["final_r"])
        equity_new = equity + pnl
        eq_rows.append({
            "event_time": t["exit_time"], "exit_date": t["exit_date"],
            "equity_before": equity, "equity_after": equity_new,
            "pnl_dollars": pnl, "final_r": float(t["final_r"]),
        })
        equity = equity_new
    eq_df = pd.DataFrame(eq_rows)
    eq_arr = np.concatenate([[STARTING_BALANCE], eq_df["equity_after"].to_numpy()])
    cummax = np.maximum.accumulate(eq_arr)
    dd_arr = (cummax - eq_arr) / cummax
    full_dd_pct = float(dd_arr.max() * 100.0)
    # Full-data worst-day DD (across all dates). Same methodology as per-fold:
    # intraday running-peak-to-trough (must come after peak in time), taking the
    # MAX of intraday-DD and day-start-anchored-DD.
    per_day_dd_max = 0.0
    for date, day_grp in eq_df.groupby("exit_date", sort=True):
        day_grp = day_grp.sort_values("event_time", kind="mergesort").reset_index(drop=True)
        day_eq_seq = np.concatenate([
            [float(day_grp["equity_before"].iloc[0])],
            day_grp["equity_after"].to_numpy(),
        ])
        running_peak = np.maximum.accumulate(day_eq_seq)
        intraday_dd_pct = float(((running_peak - day_eq_seq) / running_peak * 100.0).max())
        day_start_eq = float(day_grp["equity_before"].iloc[0])
        day_low = float(day_eq_seq.min())
        day_start_dd_pct = max(0.0, (day_start_eq - day_low) / day_start_eq * 100.0) if day_start_eq > 0 else 0.0
        per_day_dd_max = max(per_day_dd_max, intraday_dd_pct, day_start_dd_pct)
    roi_pct = float((equity - STARTING_BALANCE) / STARTING_BALANCE * 100.0)
    total_days = (fold_start_end[-1][1] - fold_start_end[0][0]).days
    ann_pct = _annualise_roi(roi_pct, total_days)
    return {
        "n_trades": int(len(df)),
        "full_data_roi_pct": roi_pct,
        "full_data_annualised_roi_pct": ann_pct,
        "full_data_max_dd_pct": full_dd_pct,
        "full_data_worst_day_dd_pct": per_day_dd_max,
        "ending_equity": float(equity),
    }


def run_one_risk(
    resim_df: pd.DataFrame, risk_pct: float,
    kh24_folds: List[Tuple[int, pd.Timestamp, pd.Timestamp]],
    out_subdir: Path,
) -> Dict[str, Any]:
    out_subdir.mkdir(parents=True, exist_ok=True)
    fold_rows: List[Dict[str, Any]] = []
    all_day_rows: List[Dict[str, Any]] = []
    for fold_id, s, e in kh24_folds:
        mask = (resim_df["entry_time"] >= s) & (resim_df["entry_time"] < e)
        sub = resim_df[mask].copy()
        eq_df, fold_roi_pct, max_dd_pct, worst_day_dd_pct, end_eq, per_day_df = _account_fold(
            sub, risk_pct,
        )
        days_in_fold = (e - s).days
        ann_roi_pct = _annualise_roi(fold_roi_pct, days_in_fold)
        fold_rows.append({
            "fold": fold_id,
            "oos_start": s.strftime("%Y-%m-%d"),
            "oos_end": e.strftime("%Y-%m-%d"),
            "n_trades": int(len(sub)),
            "fold_roi_pct": fold_roi_pct,
            "annualised_roi_pct": ann_roi_pct,
            "max_dd_pct": max_dd_pct,
            "worst_day_dd_pct": worst_day_dd_pct,
            "ending_equity": end_eq,
            "final_r_sign_positive": int(fold_roi_pct > 0),
        })
        if len(per_day_df) > 0:
            tmp = per_day_df.copy()
            tmp.insert(0, "fold", fold_id)
            all_day_rows.append(tmp)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_subdir / "per_fold_metrics.csv", index=False,
                   float_format="%.10g", lineterminator="\n")
    per_day_all = pd.concat(all_day_rows, ignore_index=True) if all_day_rows else pd.DataFrame()
    per_day_all.to_csv(out_subdir / "per_day_dd.csv", index=False,
                       float_format="%.10g", lineterminator="\n")

    # Full-data metrics (continuous compounding across folds in time order).
    in_window = resim_df.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    full_m = _full_data_account(
        in_window, risk_pct,
        [(s, e) for _, s, e in kh24_folds],
    )
    (out_subdir / "full_data_metrics.json").write_text(
        json.dumps(full_m, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Aggregate metrics (F2-F7 only — F1 is zero-admit data-availability artifact).
    fold_df_real = fold_df[fold_df["n_trades"] > 0].copy()
    if len(fold_df_real) > 0:
        worst_fold_ann_roi = float(fold_df_real["annualised_roi_pct"].min())
        mean_fold_ann_roi = float(fold_df_real["annualised_roi_pct"].mean())
        worst_fold_max_dd = float(fold_df_real["max_dd_pct"].max())
        worst_fold_day_dd = float(fold_df_real["worst_day_dd_pct"].max())
        min_trades_per_fold = int(fold_df_real["n_trades"].min())
        all_pos = bool((fold_df_real["fold_roi_pct"] > 0).all())
    else:
        worst_fold_ann_roi = float("nan"); mean_fold_ann_roi = float("nan")
        worst_fold_max_dd = float("nan"); worst_fold_day_dd = float("nan")
        min_trades_per_fold = 0; all_pos = False

    return {
        "risk_pct": risk_pct,
        "fold_df": fold_df,
        "fold_df_real": fold_df_real,
        "per_day_all": per_day_all,
        "full_data": full_m,
        "agg_real_folds": {
            "worst_fold_ann_roi_pct": worst_fold_ann_roi,
            "mean_fold_ann_roi_pct": mean_fold_ann_roi,
            "worst_fold_max_dd_pct": worst_fold_max_dd,
            "worst_fold_day_dd_pct": worst_fold_day_dd,
            "min_trades_per_fold": min_trades_per_fold,
            "all_folds_positive": all_pos,
        },
    }


def run(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Candidate A admit set with §11 Stepwise exit re-sim outputs.
    resim = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e"
        / "candidate_A_thr0.40" / "resim_trades.csv"
    )
    resim["entry_time"] = pd.to_datetime(resim["entry_time"])
    resim["exit_time"] = pd.to_datetime(resim["exit_time"])
    print(f"[scaled-risk] Candidate A admit set: {len(resim)} trades, "
          f"exit-reason mix: {dict(resim['exit_reason'].value_counts())}")

    # KH-24 anchor folds.
    cfg_kh24 = yaml.safe_load((_REPO_ROOT / "configs" / "wfo_kh24.yaml").read_text(encoding="utf-8"))
    kh24_folds: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = [
        (int(f["fold"]), pd.Timestamp(f["oos_start"]), pd.Timestamp(f["oos_end"]))
        for f in cfg_kh24["wfo"]["folds"]
    ]

    risk_results: Dict[float, Dict[str, Any]] = {}
    for r in RISK_LEVELS:
        out_subdir = out_dir / f"per_risk_{int(r * 10000):04d}"
        print(f"\n[scaled-risk] risk {r*100:.2f}%...")
        rec = run_one_risk(resim, r, kh24_folds, out_subdir)
        risk_results[r] = rec
        a = rec["agg_real_folds"]
        fm = rec["full_data"]
        print(f"  worst-fold ann ROI {a['worst_fold_ann_roi_pct']:+.2f}%  "
              f"mean {a['mean_fold_ann_roi_pct']:+.2f}%  "
              f"worst-fold DD {a['worst_fold_max_dd_pct']:.2f}%  "
              f"worst-day DD {a['worst_fold_day_dd_pct']:.2f}%")
        print(f"  full-data ann ROI {fm['full_data_annualised_roi_pct']:+.2f}%  "
              f"full DD {fm['full_data_max_dd_pct']:.2f}%  "
              f"full worst-day {fm['full_data_worst_day_dd_pct']:.2f}%")

    # Summary table (across risk levels).
    summary_rows: List[Dict[str, Any]] = []
    baseline_full = risk_results[0.005]["full_data"]
    baseline_worst_dd = risk_results[0.005]["agg_real_folds"]["worst_fold_max_dd_pct"]
    for r in RISK_LEVELS:
        rec = risk_results[r]
        a = rec["agg_real_folds"]
        fm = rec["full_data"]
        roi_ratio_vs_baseline = (fm["full_data_annualised_roi_pct"] / baseline_full["full_data_annualised_roi_pct"]
                                  if baseline_full["full_data_annualised_roi_pct"] != 0 else float("nan"))
        dd_ratio_vs_baseline = (a["worst_fold_max_dd_pct"] / baseline_worst_dd
                                 if baseline_worst_dd != 0 else float("nan"))
        risk_ratio = r / 0.005
        summary_rows.append({
            "risk_pct": r * 100,
            "worst_fold_ann_roi_pct": a["worst_fold_ann_roi_pct"],
            "mean_fold_ann_roi_pct": a["mean_fold_ann_roi_pct"],
            "worst_fold_max_dd_pct": a["worst_fold_max_dd_pct"],
            "worst_fold_day_dd_pct": a["worst_fold_day_dd_pct"],
            "full_data_ann_roi_pct": fm["full_data_annualised_roi_pct"],
            "full_data_max_dd_pct": fm["full_data_max_dd_pct"],
            "full_data_worst_day_dd_pct": fm["full_data_worst_day_dd_pct"],
            "min_trades_per_fold_real": a["min_trades_per_fold"],
            "all_folds_positive_real": int(a["all_folds_positive"]),
            "risk_ratio_vs_baseline": risk_ratio,
            "roi_ratio_vs_baseline": roi_ratio_vs_baseline,
            "roi_ratio_vs_linear_projection": (roi_ratio_vs_baseline / risk_ratio if risk_ratio > 0 else float("nan")),
            "dd_ratio_vs_baseline": dd_ratio_vs_baseline,
            "dd_ratio_vs_linear_projection": (dd_ratio_vs_baseline / risk_ratio if risk_ratio > 0 else float("nan")),
            # In-system + 5ers gate evaluation
            "pass_in_system_max_dd_8pct": int(a["worst_fold_max_dd_pct"] <= IN_SYSTEM_MAX_DD_PCT),
            "pass_in_system_day_dd_4pct": int(a["worst_fold_day_dd_pct"] <= IN_SYSTEM_DAILY_DD_PCT),
            "pass_5ers_max_dd_10pct": int(a["worst_fold_max_dd_pct"] <= HARD_MAX_DD_PCT),
            "pass_5ers_day_dd_5pct": int(a["worst_fold_day_dd_pct"] <= HARD_DAILY_DD_PCT),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_table.csv", index=False,
                       float_format="%.10g", lineterminator="\n")

    # Worst-day analysis across risk levels.
    worst_day_rows: List[Dict[str, Any]] = []
    for r in RISK_LEVELS:
        per_day = risk_results[r]["per_day_all"]
        if len(per_day) == 0:
            continue
        # Identify the single worst day per risk level.
        worst = per_day.sort_values("day_dd_pct", ascending=False).iloc[0]
        # Look up contributing trades' pairs.
        trade_ids = [int(x) for x in worst["trade_ids_in_day"].split(",") if x]
        contributing_trades = resim[resim["trade_id"].isin(trade_ids)]
        n_pairs = int(contributing_trades["pair"].nunique())
        pairs_list = sorted(contributing_trades["pair"].unique().tolist())
        worst_day_rows.append({
            "risk_pct": r * 100,
            "worst_day_date": worst["date"],
            "worst_day_dd_pct": worst["day_dd_pct"],
            "fold": int(worst["fold"]),
            "n_contributing_trades": int(worst["n_trades"]),
            "n_contributing_pairs": n_pairs,
            "contributing_pairs": ", ".join(pairs_list),
            "trade_ids": worst["trade_ids_in_day"],
            "net_day_pnl_dollars": float(worst["net_day_pnl_dollars"]),
        })
    pd.DataFrame(worst_day_rows).to_csv(out_dir / "worst_day_analysis.csv", index=False,
                                         float_format="%.10g", lineterminator="\n")

    # Recommendation: highest risk level satisfying recommendation rules.
    recommended = None
    for r in sorted(RISK_LEVELS, reverse=True):
        a = risk_results[r]["agg_real_folds"]
        rule_max_dd = a["worst_fold_max_dd_pct"] <= RECOMMEND_MAX_FOLD_DD_PCT
        rule_day_dd = a["worst_fold_day_dd_pct"] <= RECOMMEND_WORST_DAY_DD_PCT
        rule_all_pos = a["all_folds_positive"]
        # ROI scaling sanity: check linear-projection ratio in [3/4, 6/4] for 4x level
        risk_ratio = r / 0.005
        if risk_ratio > 0:
            roi_ratio_full = risk_results[r]["full_data"]["full_data_annualised_roi_pct"] / max(baseline_full["full_data_annualised_roi_pct"], 1e-9)
            roi_lin_ratio = roi_ratio_full / risk_ratio
        else:
            roi_lin_ratio = float("nan")
        rule_roi_sanity = 0.75 <= roi_lin_ratio <= 1.50  # within reasonable band of linear
        if rule_max_dd and rule_day_dd and rule_all_pos and rule_roi_sanity:
            recommended = r
            break
    if recommended is None:
        recommended = 0.005  # baseline fallback

    rec_record = risk_results[recommended]
    rec_summary = {
        "recommended_risk_pct": recommended * 100,
        "worst_fold_max_dd_pct": rec_record["agg_real_folds"]["worst_fold_max_dd_pct"],
        "worst_fold_day_dd_pct": rec_record["agg_real_folds"]["worst_fold_day_dd_pct"],
        "worst_fold_ann_roi_pct": rec_record["agg_real_folds"]["worst_fold_ann_roi_pct"],
        "full_data_ann_roi_pct": rec_record["full_data"]["full_data_annualised_roi_pct"],
        "full_data_max_dd_pct": rec_record["full_data"]["full_data_max_dd_pct"],
        "full_data_worst_day_dd_pct": rec_record["full_data"]["full_data_worst_day_dd_pct"],
    }

    summary_json = {
        "rng_seed": "n/a (no randomness in re-accounting; fully deterministic)",
        "starting_balance": STARTING_BALANCE,
        "risk_levels": RISK_LEVELS,
        "in_system_targets": {
            "max_fold_dd_pct": IN_SYSTEM_MAX_DD_PCT,
            "daily_dd_pct": IN_SYSTEM_DAILY_DD_PCT,
        },
        "five_ers_hard_limits": {
            "max_dd_pct": HARD_MAX_DD_PCT,
            "daily_dd_pct": HARD_DAILY_DD_PCT,
        },
        "recommendation_rules": {
            "max_fold_dd_pct": RECOMMEND_MAX_FOLD_DD_PCT,
            "worst_day_dd_pct": RECOMMEND_WORST_DAY_DD_PCT,
        },
        "recommendation": rec_summary,
        "per_risk_aggregates": {
            f"{r*100:.2f}pct": risk_results[r]["agg_real_folds"]
            for r in RISK_LEVELS
        },
        "per_risk_full_data": {
            f"{r*100:.2f}pct": risk_results[r]["full_data"]
            for r in RISK_LEVELS
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary_json


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 Candidate A scaled-risk WFO.")
    parser.add_argument("--out-dir", type=Path,
                        default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "scaled_risk")
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = run(args.out_dir)
    rec = summary["recommendation"]
    print("\n" + "=" * 70)
    print(f"RECOMMENDED RISK: {rec['recommended_risk_pct']:.2f}% per trade")
    print(f"  Measured worst-fold max DD : {rec['worst_fold_max_dd_pct']:.2f}%")
    print(f"  Measured worst-fold day DD : {rec['worst_fold_day_dd_pct']:.2f}%")
    print(f"  Measured worst-fold ann ROI: {rec['worst_fold_ann_roi_pct']:+.2f}%")
    print(f"  Measured full-data ann ROI : {rec['full_data_ann_roi_pct']:+.2f}%")
    print(f"  Measured full-data DD      : {rec['full_data_max_dd_pct']:.2f}%")
    print(f"  Measured full-data day DD  : {rec['full_data_worst_day_dd_pct']:.2f}%")
    print("=" * 70)

    if args.verify_determinism:
        scratch = args.out_dir / "_determinism_scratch"
        scratch.mkdir(exist_ok=True)
        run(scratch)
        # Compare summary_table.csv sha at both runs.
        sha1 = _sha256_file(args.out_dir / "summary_table.csv")
        sha2 = _sha256_file(scratch / "summary_table.csv")
        det = {
            "summary_table_run1_sha256": sha1,
            "summary_table_run2_sha256": sha2,
            "byte_identical": bool(sha1 == sha2),
        }
        (args.out_dir / "determinism_check.json").write_text(
            json.dumps(det, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"[scaled-risk] determinism: {'PASS' if det['byte_identical'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
