"""Arc 2 counterfactual quick-test — TP1R variant.

Single-variant counterfactual:
TP1R = SL at -2 ATR (baseline), Fixed TP at +1R (= +2 ATR mid-relative),
       time exit at k=120. Sequential check per bar: SL -> TP -> time exit.

Reuses the variant-simulation engine from
`scripts/lchar/arc2_counterfactual_sweep_round_1_v2.py` (variant_TP with
T_TP_R = 1.0). BL row pulled from Round 1's variant_summary_pooled.csv and
variant_summary_per_fold.csv (frozen reference).

Inputs (locked, sha256-verified):
- results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv
- results/l6/arc2/characterisation/v1_2_full/trade_index.csv
- results/l6/arc2/trades_all.csv

Outputs (results/l6/arc2/characterisation/extended/counterfactuals/tp1r_quick/):
- variant_summary.csv     (2 rows: BL, TP1R)
- per_fold_summary.csv    (14 rows: 7 folds x 2 variants)
- variant_trades.csv      (3993 rows, TP1R only)
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lchar.arc2_counterfactual_sweep_round_1_v2 import (  # noqa: E402
    TIME_HORIZON_DEFAULT,
    _pip_size,
    variant_TP,
)

LOCKED_INPUT_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv": "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/v1_2_full/trade_index.csv": "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/trades_all.csv": "47fccbfe4dffa6577a6000b0c16c2ebb9597dcf76523ff2b8084631b19836b3c",
}

ROUND_1_DIR = REPO_ROOT / "results/l6/arc2/characterisation/extended/counterfactuals/round_1"
OUT_DIR = REPO_ROOT / "results/l6/arc2/characterisation/extended/counterfactuals/tp1r_quick"

BL_EXPECTED_MEAN_R = -0.0192
BL_EXPECTED_SL_RATE = 0.760


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_inputs() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel, expected in LOCKED_INPUT_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"Input sha256 mismatch on {rel}:\n  expected: {expected}\n  observed: {actual}"
            )
        out[rel] = actual
    return out


def _run_tp1r(
    *, per_bar_csv: Path, trade_index_csv: Path, trades_all_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("  Loading trade_index.csv + trades_all.csv...", flush=True)
    ti = pd.read_csv(trade_index_csv)
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ta = pd.read_csv(trades_all_csv)
    ta["signal_bar_ts"] = pd.to_datetime(ta["signal_bar_ts"])

    ti_full = ti.merge(
        ta[["pair", "signal_bar_ts", "spread_pips_entry", "spread_pips_exit"]],
        on=["pair", "signal_bar_ts"],
        how="left",
        validate="one_to_one",
    )
    if ti_full[["spread_pips_entry", "spread_pips_exit"]].isna().any().any():
        raise RuntimeError("null sp_entry/sp_exit after merge")
    ti_full = ti_full.sort_values("trade_id").reset_index(drop=True)

    print("  Pre-computing per-trade values...", flush=True)
    per_trade: Dict[int, Dict[str, Any]] = {}
    for _, row in ti_full.iterrows():
        tid = int(row["trade_id"])
        pair = row["pair"]
        pip = _pip_size(pair)
        sp_entry = float(row["spread_pips_entry"])
        sp_exit = float(row["spread_pips_exit"])
        atr = float(row["atr_1h_wilder_at_signal"])
        entry_fill = float(row["entry_price"])
        per_trade[tid] = {
            "pair": pair,
            "atr": atr,
            "entry_fill": entry_fill,
            "sp_entry_pips": sp_entry,
            "sp_exit_pips": sp_exit,
            "pip": pip,
            "entry_fill_offset_atr": sp_entry * pip / (2 * atr),
            "baseline_spread_cost_r": (sp_entry + sp_exit) * pip / (4 * atr),
            "exit_half_spread_r_base": sp_exit * pip / (4 * atr),
            "entry_mid": entry_fill - sp_entry * pip / 2,
            "fold_id": int(row["fold_id"]),
            "signal_bar_ts": row["signal_bar_ts"].strftime("%Y-%m-%dT%H:%M:%S"),
        }

    print("  Loading per_bar_paths.csv (118 MB)...", flush=True)
    pb = pd.read_csv(per_bar_csv)
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids_arr = pb["trade_id"].to_numpy(dtype=np.int64)
    n_trades = int(ti["trade_id"].max()) + 1
    starts = np.searchsorted(tids_arr, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids_arr, np.arange(n_trades), side="right")

    rmae_all = pb["running_mae_atr"].to_numpy(dtype=np.float64)
    rmfe_all = pb["running_mfe_atr"].to_numpy(dtype=np.float64)
    bl_all = pb["bar_low_atr"].to_numpy(dtype=np.float64)
    bh_all = pb["bar_high_atr"].to_numpy(dtype=np.float64)
    bc_all = pb["bar_close_atr"].to_numpy(dtype=np.float64)
    nbo_all = pb["next_bar_open_atr"].to_numpy(dtype=np.float64)
    hnb_all = pb["has_next_bar"].to_numpy(dtype=bool)

    out_tid = np.empty(n_trades, dtype=np.int64)
    out_pair = np.empty(n_trades, dtype=object)
    out_sigts = np.empty(n_trades, dtype=object)
    out_fold = np.empty(n_trades, dtype=np.int64)
    out_reason = np.empty(n_trades, dtype=object)
    out_exitbar = np.empty(n_trades, dtype=np.int64)
    out_exitlvl = np.empty(n_trades, dtype=np.float64)
    out_gross = np.empty(n_trades, dtype=np.float64)
    out_spread = np.empty(n_trades, dtype=np.float64)
    out_net = np.empty(n_trades, dtype=np.float64)

    print(f"  Computing TP1R for {n_trades} trades...", flush=True)
    t0 = time.time()
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        bavail = e - s
        rmae = rmae_all[s:e]
        rmfe = rmfe_all[s:e]
        bl_ = bl_all[s:e]
        bh_ = bh_all[s:e]
        bc_ = bc_all[s:e]
        nbo = nbo_all[s:e]
        hnb = hnb_all[s:e]
        T = per_trade[tid]
        r = variant_TP(
            rmae,
            rmfe,
            bl_,
            bh_,
            bc_,
            nbo,
            hnb,
            bavail,
            T,
            T_TP_R=1.0,
            time_horizon=TIME_HORIZON_DEFAULT,
        )
        out_tid[tid] = tid
        out_pair[tid] = T["pair"]
        out_sigts[tid] = T["signal_bar_ts"]
        out_fold[tid] = T["fold_id"]
        out_reason[tid] = r["exit_reason"]
        out_exitbar[tid] = r["exit_bar"]
        out_exitlvl[tid] = r["exit_level_atr_fill"]
        out_gross[tid] = r["gross_R"]
        out_spread[tid] = r["spread_cost_R"]
        out_net[tid] = r["net_R"]
    print(f"    done in {time.time() - t0:.1f}s", flush=True)

    vt = pd.DataFrame(
        {
            "variant_id": "TP1R",
            "trade_id": out_tid,
            "pair": out_pair,
            "signal_bar_ts": out_sigts,
            "fold_id": out_fold,
            "exit_reason_variant": out_reason,
            "exit_bar": out_exitbar,
            "exit_level_atr_fill": out_exitlvl,
            "gross_R": out_gross,
            "spread_cost_R": out_spread,
            "net_R": out_net,
        }
    )
    vt = vt.sort_values("trade_id").reset_index(drop=True)
    return vt, ti_full


def _aggregate_pooled_tp1r(vt: pd.DataFrame) -> Dict[str, float]:
    n = len(vt)
    net = vt["net_R"].to_numpy(dtype=np.float64)
    rc = vt["exit_reason_variant"].value_counts().to_dict()
    return {
        "n": n,
        "mean_R": float(np.mean(net)),
        "median_R": float(np.median(net)),
        "std_R": float(np.std(net, ddof=1)),
        "sl_rate": rc.get("stop_loss", 0) / n,
        "tp_rate": rc.get("fixed_tp", 0) / n,
        "te_rate": rc.get("time_exit", 0) / n,
        "data_end_rate": rc.get("data_end", 0) / n,
        "mean_spread_cost_R": float(vt["spread_cost_R"].mean()),
    }


def _aggregate_per_fold_tp1r(vt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sorted_vt = vt.sort_values(["fold_id", "signal_bar_ts", "trade_id"])
    folds = sorted(vt["fold_id"].unique().tolist())
    for fid in folds:
        sub = sorted_vt[sorted_vt["fold_id"] == fid]
        n = len(sub)
        net = sub["net_R"].to_numpy(dtype=np.float64)
        cum = np.cumsum(net)
        run_max = np.maximum.accumulate(cum)
        dd = run_max - cum
        mdd = float(np.max(dd))
        rc = sub["exit_reason_variant"].value_counts().to_dict()
        rows.append(
            {
                "variant_id": "TP1R",
                "fold_id": int(fid),
                "n": n,
                "mean_R": float(np.mean(net)),
                "total_R": float(np.sum(net)),
                "max_DD_R": mdd,
                "sl_rate": rc.get("stop_loss", 0) / n,
                "tp_rate": rc.get("fixed_tp", 0) / n,
                "te_rate": rc.get("time_exit", 0) / n,
            }
        )
    return pd.DataFrame(rows)


def _pull_bl_from_round1() -> tuple[Dict[str, float], pd.DataFrame]:
    pooled_p = ROUND_1_DIR / "variant_summary_pooled.csv"
    per_fold_p = ROUND_1_DIR / "variant_summary_per_fold.csv"
    pooled_r1 = pd.read_csv(pooled_p)
    per_fold_r1 = pd.read_csv(per_fold_p)
    bl_row = pooled_r1[pooled_r1["variant_id"] == "BL"].iloc[0]

    if not (abs(float(bl_row["mean_R"]) - BL_EXPECTED_MEAN_R) < 5e-4):
        raise RuntimeError(
            f"BL mean_R mismatch vs spec: expected ~{BL_EXPECTED_MEAN_R}, "
            f"observed {float(bl_row['mean_R']):.6f}"
        )
    if not (abs(float(bl_row["sl_rate"]) - BL_EXPECTED_SL_RATE) < 5e-4):
        raise RuntimeError(
            f"BL sl_rate mismatch vs spec: expected ~{BL_EXPECTED_SL_RATE}, "
            f"observed {float(bl_row['sl_rate']):.6f}"
        )

    bl_pooled = {
        "n": int(bl_row["n_trades"]),
        "mean_R": float(bl_row["mean_R"]),
        "median_R": float(bl_row["median_R"]),
        "std_R": float(bl_row["std_R"]),
        "sl_rate": float(bl_row["sl_rate"]),
        "tp_rate": float(bl_row["tp_exit_rate"]),
        "te_rate": float(bl_row["time_exit_rate"]),
        "data_end_rate": float(bl_row["data_end_rate"]),
        "mean_spread_cost_R": float(bl_row["mean_spread_cost_R"]),
    }

    bl_pf = per_fold_r1[per_fold_r1["variant_id"] == "BL"].copy()
    bl_pf_out = pd.DataFrame(
        {
            "variant_id": "BL",
            "fold_id": bl_pf["fold_id"].astype(int).values,
            "n": bl_pf["n"].astype(int).values,
            "mean_R": bl_pf["mean_R"].astype(float).values,
            "total_R": bl_pf["total_R"].astype(float).values,
            "max_DD_R": bl_pf["max_DD_R"].astype(float).values,
            "sl_rate": bl_pf["sl_rate"].astype(float).values,
            "tp_rate": bl_pf["tp_exit_rate"].astype(float).values,
            "te_rate": bl_pf["time_exit_rate"].astype(float).values,
        }
    )
    return bl_pooled, bl_pf_out


def main() -> int:
    print("=" * 60)
    print("Arc 2 counterfactual quick test — TP1R")
    print("=" * 60)
    t_start = time.time()

    print("\n[Validation] Verifying input sha256s...")
    input_shas = _verify_inputs()
    for k in input_shas:
        print(f"  OK {k}")

    print("\n[Validation] Pulling BL row from Round 1 pooled/per_fold...")
    bl_pooled, bl_pf = _pull_bl_from_round1()
    print(f"  BL mean_R = {bl_pooled['mean_R']:+.10f} (matches spec ~{BL_EXPECTED_MEAN_R})")
    print(f"  BL sl_rate = {bl_pooled['sl_rate']:.6f} (matches spec ~{BL_EXPECTED_SL_RATE})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[Output] {OUT_DIR}")

    print("\n[Compute] Running TP1R variant...")
    per_bar_csv = REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv"
    trade_index_csv = REPO_ROOT / "results/l6/arc2/characterisation/v1_2_full/trade_index.csv"
    trades_all_csv = REPO_ROOT / "results/l6/arc2/trades_all.csv"
    vt, ti_full = _run_tp1r(
        per_bar_csv=per_bar_csv, trade_index_csv=trade_index_csv, trades_all_csv=trades_all_csv
    )

    # Validation: no NaN/inf in net_R
    nan_inf = int((~np.isfinite(vt["net_R"].to_numpy(dtype=np.float64))).sum())
    if nan_inf > 0:
        raise RuntimeError(f"NaN/inf in TP1R net_R: {nan_inf} rows")

    # Validation: exit-reason rates sum to ~1
    n = len(vt)
    rc = vt["exit_reason_variant"].value_counts().to_dict()
    rate_sum = sum(rc.values()) / n
    if abs(rate_sum - 1.0) > 1e-9:
        raise RuntimeError(f"Exit reason rates sum to {rate_sum}, expected 1.0")

    tp1r_pooled = _aggregate_pooled_tp1r(vt)
    tp1r_pf = _aggregate_per_fold_tp1r(vt)

    # Validation: TP1R SL rate < BL SL rate
    if not (tp1r_pooled["sl_rate"] < bl_pooled["sl_rate"]):
        raise RuntimeError(
            f"TP1R sl_rate ({tp1r_pooled['sl_rate']:.4f}) not < "
            f"BL sl_rate ({bl_pooled['sl_rate']:.4f})"
        )

    # Build variant_summary.csv
    bl_lift = 0.0
    tp1r_lift = tp1r_pooled["mean_R"] - bl_pooled["mean_R"]
    summary_rows = [
        {
            "variant_id": "BL",
            "n": bl_pooled["n"],
            "mean_R": bl_pooled["mean_R"],
            "median_R": bl_pooled["median_R"],
            "std_R": bl_pooled["std_R"],
            "sl_rate": bl_pooled["sl_rate"],
            "tp_rate": bl_pooled["tp_rate"],
            "te_rate": bl_pooled["te_rate"],
            "mean_spread_cost_R": bl_pooled["mean_spread_cost_R"],
            "lift_vs_BL": bl_lift,
        },
        {
            "variant_id": "TP1R",
            "n": tp1r_pooled["n"],
            "mean_R": tp1r_pooled["mean_R"],
            "median_R": tp1r_pooled["median_R"],
            "std_R": tp1r_pooled["std_R"],
            "sl_rate": tp1r_pooled["sl_rate"],
            "tp_rate": tp1r_pooled["tp_rate"],
            "te_rate": tp1r_pooled["te_rate"],
            "mean_spread_cost_R": tp1r_pooled["mean_spread_cost_R"],
            "lift_vs_BL": tp1r_lift,
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    # Build per_fold_summary.csv (BL then TP1R, fold 1..7 each)
    per_fold_combined = pd.concat([bl_pf, tp1r_pf], ignore_index=True)

    # Write outputs
    summary_p = OUT_DIR / "variant_summary.csv"
    per_fold_p = OUT_DIR / "per_fold_summary.csv"
    vt_p = OUT_DIR / "variant_trades.csv"

    summary_df.to_csv(summary_p, index=False, lineterminator="\n", float_format="%.10g")
    per_fold_combined.to_csv(per_fold_p, index=False, lineterminator="\n", float_format="%.10g")
    vt.to_csv(vt_p, index=False, lineterminator="\n", float_format="%.10g")

    # Output sha256s
    output_shas = {
        "variant_summary.csv": _sha256_file(summary_p),
        "per_fold_summary.csv": _sha256_file(per_fold_p),
        "variant_trades.csv": _sha256_file(vt_p),
    }

    print("\n[Output] sha256 manifest:")
    print("  Inputs:")
    for k, v in input_shas.items():
        print(f"    {k}\n      {v}")
    print("  Outputs:")
    for k, v in output_shas.items():
        print(f"    {k}\n      {v}")

    # Print headline table
    print("\n[Headline] Pooled comparison:")
    print("  variant  pooled_R   median_R   sl_rate  tp_rate  te_rate  lift_vs_BL")
    print(
        f"  BL       {bl_pooled['mean_R']:+.4f}    {bl_pooled['median_R']:+.4f}    "
        f"{bl_pooled['sl_rate']:.4f}   -        {bl_pooled['te_rate']:.4f}   +0.0000"
    )
    print(
        f"  TP1R     {tp1r_pooled['mean_R']:+.4f}    {tp1r_pooled['median_R']:+.4f}    "
        f"{tp1r_pooled['sl_rate']:.4f}   {tp1r_pooled['tp_rate']:.4f}   "
        f"{tp1r_pooled['te_rate']:.4f}   {tp1r_lift:+.4f}"
    )

    print("\n[Headline] TP1R per-fold:")
    print("  fold  n    mean_R    max_DD_R   sl_rate  tp_rate  te_rate")
    for _, row in tp1r_pf.iterrows():
        print(
            f"  {int(row['fold_id'])}     {int(row['n']):<4d} "
            f"{row['mean_R']:+.4f}   {row['max_DD_R']:7.4f}   "
            f"{row['sl_rate']:.4f}   {row['tp_rate']:.4f}   {row['te_rate']:.4f}"
        )
    worst_fold = tp1r_pf.sort_values("mean_R").iloc[0]
    worst_dd = tp1r_pf.sort_values("max_DD_R", ascending=False).iloc[0]
    print(f"  worst-fold mean_R: fold {int(worst_fold['fold_id'])} = {worst_fold['mean_R']:+.4f}")
    print(f"  worst-fold max_DD: fold {int(worst_dd['fold_id'])} = {worst_dd['max_DD_R']:.4f}")

    el = time.time() - t_start
    print(f"\n[Wallclock] {el:.1f}s")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
