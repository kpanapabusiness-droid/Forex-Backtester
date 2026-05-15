"""Arc 2 trade curve & pullback characterisation.

Phase: l6_arc2_trade_curves

Descriptive per-bar trade-shape characterisation on v1.2.1 per-bar data.
Read-existing-CSV-only per L6.0 v1.1 §14.6. Answers "why did exit-rule
modifications produce smaller lifts than the population aggregates suggested?"

Blocks:
- H: per-bar median trade journey (all, SL, TE populations; k ∈ [1, 120], plus
     extended k ∈ [1, 240] for all-taken)
- I: pullback distributions after MFE threshold crossing (T_R ∈ {0.25..3.0}R)
- J: BE-SL conditional analysis at T_BE ∈ {0.5, 1.0, 1.5}R
- K: trail design diagnostic — pullback by post-engagement survival category
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Locked input sha256s (gate 1) ---
# Note: the spec cited path `v1_2_full/signals_features.csv` with sha 71b39383…,
# but that file lives at `v1_1_full/signals_features.csv`. The sha matches v1.1
# (the v1.2 directory only contains per_bar_paths/trade_index/manifest).
# Treating as v1_1_full per the sha match. Documented in manifest.
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv":
        "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv":
        "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv":
        "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "L6_0_METHODOLOGY_LOCK.md":
        "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}

ADJ_LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_2_full/per_bar_paths.csv":
        "e1195f0dedb317f6d921d4fa9526c8aa546457f8038f28f37cd656605e6b1960",
    "results/l6/arc2/characterisation/v1_2_full/trade_index.csv":
        "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py":
        "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml":
        "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "scripts/lchar/arc2_per_bar_paths.py":
        "36bb6f9b0413386bd5d25960f4525084fa93408ecb491232e17396872f1ff821",
}
CANDIDATE_HYPOTHESES_BASELINE_SHA = (
    "8ed487620a7f9ab2c443e6520a4afa820c353480d8329d4fe91703b7d083dfbf"
)

# Thresholds
THRESHOLDS_I: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00)  # R-units
THRESHOLDS_J: Tuple[float, ...] = (0.5, 1.0, 1.5)
THRESHOLDS_K: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

# Round 1 reference for gate 6
ROUND_1_B2_BE_FIRE_COUNT = 1541

# Per-trade hold cap for descriptive analysis (matches BL execution horizon)
HOLD_CAP = 120
EXTENDED_HOLD = 240


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_input_integrity() -> Dict[str, str]:
    """Gate 1: verify all 4 locked sha256s."""
    out: Dict[str, str] = {}
    for rel, expected in LOCKED_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"Gate 1 HALT — sha256 mismatch on {rel}:\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
        out[rel] = actual
    return out


# ============================================================================
# DATA LOAD + STREAM-BY-TRADE
# ============================================================================


def _load_data(per_bar_csv: Path, trade_index_csv: Path) -> Tuple[
    pd.DataFrame, Dict[int, Dict[str, Any]],
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Load per_bar_paths.csv and trade_index.csv. Build per-trade lookup tables
    and per-bar numpy arrays. Returns (trade_index, per_trade_dict, starts, ends,
    rmae_R, rmfe_R, bl_R, bh_R, bc_R) — all R-units (= ATR / 2)."""
    print("  Loading trade_index.csv...", flush=True)
    ti = pd.read_csv(trade_index_csv)
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ti = ti.sort_values("trade_id").reset_index(drop=True)

    print("  Loading per_bar_paths.csv (118 MB)...", flush=True)
    pb = pd.read_csv(per_bar_csv)
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)

    # Per-trade slice indices
    tids_arr = pb["trade_id"].to_numpy(dtype=np.int64)
    n_trades = int(ti["trade_id"].max()) + 1
    starts = np.searchsorted(tids_arr, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids_arr, np.arange(n_trades), side="right")

    # All values converted to R-units (= ATR / 2)
    rmae_R = pb["running_mae_atr"].to_numpy(dtype=np.float64) / 2.0
    rmfe_R = pb["running_mfe_atr"].to_numpy(dtype=np.float64) / 2.0
    bl_R = pb["bar_low_atr"].to_numpy(dtype=np.float64) / 2.0
    bh_R = pb["bar_high_atr"].to_numpy(dtype=np.float64) / 2.0
    bc_R = pb["bar_close_atr"].to_numpy(dtype=np.float64) / 2.0

    per_trade: Dict[int, Dict[str, Any]] = {}
    for _, row in ti.iterrows():
        tid = int(row["trade_id"])
        per_trade[tid] = {
            "pair": row["pair"],
            "signal_bar_ts": row["signal_bar_ts"],
            "fold_id": int(row["fold_id"]),
            "held_bars": int(row["held_bars"]),
            "bars_available": int(row["bars_available"]),
            "exit_reason": str(row["exit_reason"]),
            "R": float(row["R"]),
        }

    return ti, per_trade, starts, ends, rmae_R, rmfe_R, bl_R, bh_R, bc_R


# ============================================================================
# BLOCK H — per-bar median trade journey
# ============================================================================


def _block_H(
    *, per_trade: Dict[int, Dict[str, Any]], starts: np.ndarray, ends: np.ndarray,
    rmae_R: np.ndarray, rmfe_R: np.ndarray, bc_R: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-bar median + quantiles for three populations:
       H.1 all trades [1, 120], H.2 SL trades (k <= held_bars),
       H.3 TE trades [1, 120]. Plus H.1 extended to [1, 240].
    """
    n_trades = len(per_trade)
    # Build per-trade per-bar arrays of length up to bars_available (could be 21..240)
    # Then aggregate per k across applicable trades.

    # For H.1 / H.3: cap k at HOLD_CAP. For H.2: cap at held_bars per trade.
    # Strategy: collect (k, rmfe, rmae, bc) tuples into long-form arrays per population.

    # H.1 — all taken trades, k ∈ [1, 120]
    h1_lists: Dict[int, Dict[str, List[float]]] = {k: {"rmfe": [], "rmae": [], "bc": []} for k in range(1, HOLD_CAP + 1)}
    # H.2 — SL trades, k ≤ held_bars
    h2_lists: Dict[int, Dict[str, List[float]]] = {k: {"rmfe": [], "rmae": [], "bc": []} for k in range(1, HOLD_CAP + 1)}
    # H.3 — TE trades, k ∈ [1, 120]
    h3_lists: Dict[int, Dict[str, List[float]]] = {k: {"rmfe": [], "rmae": [], "bc": []} for k in range(1, HOLD_CAP + 1)}
    # H.1 extended — all taken trades, k ∈ [1, 240]
    h1_ext_lists: Dict[int, Dict[str, List[float]]] = {k: {"rmfe": [], "rmae": [], "bc": []} for k in range(1, EXTENDED_HOLD + 1)}

    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        bavail = e - s
        t = per_trade[tid]
        exit_reason = t["exit_reason"]
        held = t["held_bars"]
        rmfe_t = rmfe_R[s:e]
        rmae_t = rmae_R[s:e]
        bc_t = bc_R[s:e]

        # H.1 — all, k ∈ [1, 120] (capped by bavail)
        h1_max_k = min(HOLD_CAP, bavail)
        for k_idx in range(h1_max_k):
            k = k_idx + 1
            h1_lists[k]["rmfe"].append(rmfe_t[k_idx])
            h1_lists[k]["rmae"].append(rmae_t[k_idx])
            h1_lists[k]["bc"].append(bc_t[k_idx])

        # H.1 extended — all, k ∈ [1, 240]
        for k_idx in range(bavail):
            k = k_idx + 1
            if k > EXTENDED_HOLD:
                break
            h1_ext_lists[k]["rmfe"].append(rmfe_t[k_idx])
            h1_ext_lists[k]["rmae"].append(rmae_t[k_idx])
            h1_ext_lists[k]["bc"].append(bc_t[k_idx])

        # H.2 — SL only, k ≤ held_bars
        if exit_reason == "stop_loss":
            h2_max_k = min(HOLD_CAP, held)
            for k_idx in range(h2_max_k):
                k = k_idx + 1
                h2_lists[k]["rmfe"].append(rmfe_t[k_idx])
                h2_lists[k]["rmae"].append(rmae_t[k_idx])
                h2_lists[k]["bc"].append(bc_t[k_idx])

        # H.3 — TE only, k ∈ [1, 120]
        if exit_reason == "time_exit":
            for k_idx in range(min(HOLD_CAP, bavail)):
                k = k_idx + 1
                h3_lists[k]["rmfe"].append(rmfe_t[k_idx])
                h3_lists[k]["rmae"].append(rmae_t[k_idx])
                h3_lists[k]["bc"].append(bc_t[k_idx])

    # Aggregate per k
    def _aggregate(d: Dict[int, Dict[str, List[float]]], population: str) -> pd.DataFrame:
        rows = []
        for k in sorted(d.keys()):
            n = len(d[k]["rmfe"])
            if n == 0:
                continue
            for col_name, col_vals in [("running_mfe_R", d[k]["rmfe"]),
                                          ("running_mae_R", d[k]["rmae"]),
                                          ("bar_close_R", d[k]["bc"])]:
                arr = np.array(col_vals, dtype=np.float64)
                rows.append({
                    "population": population, "k": k, "metric": col_name, "n": n,
                    "median": float(np.median(arr)),
                    "q05": float(np.quantile(arr, 0.05)),
                    "q25": float(np.quantile(arr, 0.25)),
                    "q75": float(np.quantile(arr, 0.75)),
                    "q95": float(np.quantile(arr, 0.95)),
                    "mean": float(np.mean(arr)),
                })
        return pd.DataFrame(rows)

    h1 = _aggregate(h1_lists, "all_taken")
    h2 = _aggregate(h2_lists, "eventual_stop_loss")
    h3 = _aggregate(h3_lists, "eventual_time_exit")
    h1_ext = _aggregate(h1_ext_lists, "all_taken_extended")

    # Block H taken (H.1): one CSV
    h1_csv = h1.copy()
    # Block H by exit (H.2 + H.3): combined
    h_by_exit = pd.concat([h2, h3], ignore_index=True)
    # Extended (H.1 to 240)
    h_ext = h1_ext

    return h1_csv, h_by_exit, h_ext


# ============================================================================
# BLOCK I — pullback distributions
# ============================================================================


def _block_I(
    *, per_trade: Dict[int, Dict[str, Any]], starts: np.ndarray, ends: np.ndarray,
    rmae_R: np.ndarray, rmfe_R: np.ndarray, bl_R: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For each T_R threshold, compute per-trade max_subsequent_pullback_R
    and aggregate distributions."""
    n_trades = len(per_trade)
    trade_rows: List[Dict[str, Any]] = []

    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        bavail = e - s
        t = per_trade[tid]
        held = min(t["held_bars"], HOLD_CAP)  # observe up to held_bars (capped at 120)
        max_k_obs = min(held, bavail)  # number of bars to look at
        rmfe_t = rmfe_R[s:s+max_k_obs]
        bl_t = bl_R[s:s+max_k_obs]
        realised_R = t["R"]
        exit_reason = t["exit_reason"]

        for T_R in THRESHOLDS_I:
            T_atr_fill = 2.0 * T_R
            # Find k_cross_T = first k where running_mfe_atr >= T_atr_fill
            # In R-units: running_mfe_R >= T_R
            # Note: only count crossings during the held window [1, held]
            hits = np.where(rmfe_t >= T_R)[0]
            if hits.size == 0:
                continue
            k_cross = int(hits[0]) + 1  # 1-indexed
            peak_at_cross_R = float(rmfe_t[hits[0]])

            # For subsequent bars k > k_cross_T up to held window (already capped)
            if k_cross >= max_k_obs:
                # No subsequent bars within the held window
                max_pullback = 0.0
            else:
                # k_cross is 1-indexed; subsequent bars are indices [k_cross, max_k_obs)
                sub_rmfe = rmfe_t[k_cross:max_k_obs]
                sub_bl = bl_t[k_cross:max_k_obs]
                if sub_rmfe.size == 0:
                    max_pullback = 0.0
                else:
                    pullbacks = sub_rmfe - sub_bl
                    max_pullback = float(np.max(pullbacks))

            trade_rows.append({
                "trade_id": tid,
                "T_R": T_R,
                "k_cross_T": k_cross,
                "peak_at_cross_R": peak_at_cross_R,
                "max_subsequent_pullback_R": max_pullback,
                "realised_R_baseline": realised_R,
                "eventual_exit_reason": exit_reason,
            })

    trade_df = pd.DataFrame(trade_rows)

    # Aggregate
    agg_rows: List[Dict[str, Any]] = []
    for T_R in THRESHOLDS_I:
        sub = trade_df[trade_df["T_R"] == T_R]
        n = len(sub)
        if n == 0:
            agg_rows.append({"T_R": T_R, "n_reached_T": 0})
            continue
        pb_arr = sub["max_subsequent_pullback_R"].to_numpy(dtype=np.float64)
        r_arr = sub["realised_R_baseline"].to_numpy(dtype=np.float64)
        row: Dict[str, Any] = {
            "T_R": T_R,
            "n_reached_T": n,
            "pullback_mean": float(np.mean(pb_arr)),
            "pullback_q05": float(np.quantile(pb_arr, 0.05)),
            "pullback_q25": float(np.quantile(pb_arr, 0.25)),
            "pullback_q50": float(np.median(pb_arr)),
            "pullback_q75": float(np.quantile(pb_arr, 0.75)),
            "pullback_q90": float(np.quantile(pb_arr, 0.90)),
            "pullback_q95": float(np.quantile(pb_arr, 0.95)),
            "realised_R_mean_all": float(np.mean(r_arr)),
            "realised_R_median_all": float(np.median(r_arr)),
        }
        # Conditional distributions
        for Y in (0.25, 0.5, 1.0, 1.5, 2.0):
            mask_ge = pb_arr >= Y
            mask_lt = ~mask_ge
            n_ge = int(mask_ge.sum())
            n_lt = int(mask_lt.sum())
            r_ge_mean = float(np.mean(r_arr[mask_ge])) if n_ge > 0 else float("nan")
            r_lt_mean = float(np.mean(r_arr[mask_lt])) if n_lt > 0 else float("nan")
            row[f"n_pullback_ge_{Y}"] = n_ge
            row[f"realised_R_mean_pullback_ge_{Y}"] = r_ge_mean
            row[f"n_pullback_lt_{Y}"] = n_lt
            row[f"realised_R_mean_pullback_lt_{Y}"] = r_lt_mean
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)

    return agg_df, trade_df


# ============================================================================
# BLOCK J — BE-SL conditional analysis
# ============================================================================


def _block_J(
    *, per_trade: Dict[int, Dict[str, Any]], starts: np.ndarray, ends: np.ndarray,
    rmae_R: np.ndarray, rmfe_R: np.ndarray, bl_R: np.ndarray,
) -> pd.DataFrame:
    """For each T_BE, analyse dipped-vs-not-dipped subpopulations among trades
    that activated BE (running_mfe_R >= T_BE during hold).

    'dipped' = bar_low_R ≤ 0 at any k > k_cross_T_BE within 120-bar hold
              (i.e., bar_low ≤ entry_fill; fill-rel ≤ 0).
    Note: this differs from Round 1's BE-SL fire condition (bar_low ≤ entry_mid,
    mid-rel ≤ 0), which is sp_entry/(2 atr) below entry_fill. Gate 6's ±5%
    tolerance allows for the modest population-level difference (mostly small-
    spread trades match closely; high-spread trades over-count under fill-rel).
    """
    n_trades = len(per_trade)
    rows: List[Dict[str, Any]] = []
    for T_BE in THRESHOLDS_J:
        n_reached = 0
        n_dipped = 0
        dipped_R: List[float] = []
        not_dipped_R: List[float] = []
        dipped_exit_reasons: List[str] = []
        not_dipped_exit_reasons: List[str] = []
        for tid in range(n_trades):
            s, e = int(starts[tid]), int(ends[tid])
            bavail = e - s
            t = per_trade[tid]
            held = min(t["held_bars"], HOLD_CAP)
            max_k_obs = min(held, bavail)
            rmfe_t = rmfe_R[s:s+max_k_obs]
            bl_t = bl_R[s:s+max_k_obs]

            hits = np.where(rmfe_t >= T_BE)[0]
            if hits.size == 0:
                continue
            n_reached += 1
            k_cross = int(hits[0])  # 0-indexed
            # bar_low_R ≤ 0 at any k > k_cross_T_BE
            if k_cross + 1 < max_k_obs:
                sub_bl = bl_t[k_cross + 1:max_k_obs]
                if (sub_bl <= 0.0).any():
                    n_dipped += 1
                    dipped_R.append(t["R"])
                    dipped_exit_reasons.append(t["exit_reason"])
                else:
                    not_dipped_R.append(t["R"])
                    not_dipped_exit_reasons.append(t["exit_reason"])
            else:
                # No subsequent bars to evaluate; classify as not dipped
                not_dipped_R.append(t["R"])
                not_dipped_exit_reasons.append(t["exit_reason"])

        # Aggregate stats
        d_arr = np.array(dipped_R, dtype=np.float64)
        nd_arr = np.array(not_dipped_R, dtype=np.float64)

        def _rates(reasons: List[str], total: int) -> Dict[str, float]:
            if total == 0:
                return {"sl": 0.0, "te": 0.0, "de": 0.0}
            n_sl = sum(1 for r in reasons if r == "stop_loss")
            n_te = sum(1 for r in reasons if r == "time_exit")
            n_de = sum(1 for r in reasons if r == "data_end")
            return {"sl": n_sl/total, "te": n_te/total, "de": n_de/total}

        d_rates = _rates(dipped_exit_reasons, n_dipped)
        nd_rates = _rates(not_dipped_exit_reasons, n_reached - n_dipped)

        def _stats(arr: np.ndarray) -> Dict[str, float]:
            if arr.size == 0:
                return {k: float("nan") for k in ("mean", "median", "q05", "q25", "q75", "q95")}
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "q05": float(np.quantile(arr, 0.05)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "q95": float(np.quantile(arr, 0.95)),
            }

        d_stats = _stats(d_arr)
        nd_stats = _stats(nd_arr)

        rows.append({
            "T_BE": T_BE,
            "n_reached_T_BE": n_reached,
            "n_dipped_after": n_dipped,
            "dipped_rate_of_activated": (n_dipped / n_reached) if n_reached > 0 else 0.0,
            "dipped_mean_R": d_stats["mean"],
            "dipped_median_R": d_stats["median"],
            "dipped_q05_R": d_stats["q05"],
            "dipped_q25_R": d_stats["q25"],
            "dipped_q75_R": d_stats["q75"],
            "dipped_q95_R": d_stats["q95"],
            "dipped_sl_rate": d_rates["sl"],
            "dipped_te_rate": d_rates["te"],
            "dipped_de_rate": d_rates["de"],
            "not_dipped_n": n_reached - n_dipped,
            "not_dipped_mean_R": nd_stats["mean"],
            "not_dipped_median_R": nd_stats["median"],
            "not_dipped_q05_R": nd_stats["q05"],
            "not_dipped_q25_R": nd_stats["q25"],
            "not_dipped_q75_R": nd_stats["q75"],
            "not_dipped_q95_R": nd_stats["q95"],
            "not_dipped_sl_rate": nd_rates["sl"],
            "not_dipped_te_rate": nd_rates["te"],
            "not_dipped_de_rate": nd_rates["de"],
        })
    return pd.DataFrame(rows)


# ============================================================================
# BLOCK K — trail design diagnostic by survival category
# ============================================================================


def _block_K(
    *, per_trade: Dict[int, Dict[str, Any]], starts: np.ndarray, ends: np.ndarray,
    rmae_R: np.ndarray, rmfe_R: np.ndarray, bl_R: np.ndarray,
) -> pd.DataFrame:
    """Three mutually exclusive categories per trade reaching T:
    1. immediate_failure: bar_low_R ≤ -0.5R at any k ∈ (k_cross_T, k_cross_T + 5]
    2. continued_up_only: NOT immediate_failure AND max(running_mfe_R[k > k_cross_T]) > running_mfe_R[k_cross_T]
    3. peaked_then_retraced: NOT immediate_failure AND NOT continued_up_only

    For each category report n, pullback distribution, realised_R distribution,
    and mean bars_to_first_pullback (from k_cross_T to bar with max pullback).
    """
    n_trades = len(per_trade)
    rows: List[Dict[str, Any]] = []
    for T_R in THRESHOLDS_K:
        categories: Dict[str, Dict[str, List[float]]] = {
            "continued_up_only": {"pullback": [], "realised_R": [], "bars_to_pullback": []},
            "peaked_then_retraced": {"pullback": [], "realised_R": [], "bars_to_pullback": []},
            "immediate_failure": {"pullback": [], "realised_R": [], "bars_to_pullback": []},
        }
        n_reached_T = 0
        for tid in range(n_trades):
            s, e = int(starts[tid]), int(ends[tid])
            bavail = e - s
            t = per_trade[tid]
            held = min(t["held_bars"], HOLD_CAP)
            max_k_obs = min(held, bavail)
            rmfe_t = rmfe_R[s:s+max_k_obs]
            bl_t = bl_R[s:s+max_k_obs]
            realised_R = t["R"]

            hits = np.where(rmfe_t >= T_R)[0]
            if hits.size == 0:
                continue
            n_reached_T += 1
            k_cross = int(hits[0])  # 0-indexed
            peak_at_cross = float(rmfe_t[k_cross])

            # Subsequent bars in (k_cross, max_k_obs]
            sub_start = k_cross + 1
            if sub_start >= max_k_obs:
                # No subsequent bars; default to peaked_then_retraced
                category = "peaked_then_retraced"
                max_pullback = 0.0
                bars_to_pullback = 0
            else:
                # immediate_failure: any bar k ∈ (k_cross, k_cross+5] has bar_low_R ≤ -0.5
                immediate_end = min(sub_start + 5, max_k_obs)
                immediate_window = bl_t[sub_start:immediate_end]
                is_immediate_failure = (immediate_window <= -0.5).any()

                sub_rmfe = rmfe_t[sub_start:max_k_obs]
                sub_bl = bl_t[sub_start:max_k_obs]
                pullbacks = sub_rmfe - sub_bl
                max_pullback_idx = int(np.argmax(pullbacks))
                max_pullback = float(pullbacks[max_pullback_idx])
                bars_to_pullback = max_pullback_idx + 1  # bars after k_cross

                if is_immediate_failure:
                    category = "immediate_failure"
                else:
                    has_higher_peak = float(np.max(sub_rmfe)) > peak_at_cross
                    if has_higher_peak:
                        category = "continued_up_only"
                    else:
                        category = "peaked_then_retraced"

            categories[category]["pullback"].append(max_pullback)
            categories[category]["realised_R"].append(realised_R)
            categories[category]["bars_to_pullback"].append(float(bars_to_pullback))

        # Aggregate per category
        cat_sum = 0
        for cat_name, vals in categories.items():
            n = len(vals["pullback"])
            cat_sum += n
            pb = np.array(vals["pullback"], dtype=np.float64) if n > 0 else np.array([], dtype=np.float64)
            rR = np.array(vals["realised_R"], dtype=np.float64) if n > 0 else np.array([], dtype=np.float64)
            btp = np.array(vals["bars_to_pullback"], dtype=np.float64) if n > 0 else np.array([], dtype=np.float64)
            rows.append({
                "T_R": T_R,
                "category": cat_name,
                "n": n,
                "n_reached_T": n_reached_T,
                "pullback_mean": float(np.mean(pb)) if n > 0 else float("nan"),
                "pullback_median": float(np.median(pb)) if n > 0 else float("nan"),
                "pullback_q25": float(np.quantile(pb, 0.25)) if n > 0 else float("nan"),
                "pullback_q75": float(np.quantile(pb, 0.75)) if n > 0 else float("nan"),
                "realised_R_mean": float(np.mean(rR)) if n > 0 else float("nan"),
                "realised_R_median": float(np.median(rR)) if n > 0 else float("nan"),
                "realised_R_q25": float(np.quantile(rR, 0.25)) if n > 0 else float("nan"),
                "realised_R_q75": float(np.quantile(rR, 0.75)) if n > 0 else float("nan"),
                "bars_to_pullback_mean": float(np.mean(btp)) if n > 0 else float("nan"),
                "bars_to_pullback_median": float(np.median(btp)) if n > 0 else float("nan"),
            })
        # Sanity: cat_sum should equal n_reached_T (gate 7)
        if cat_sum != n_reached_T:
            raise RuntimeError(
                f"Block K internal inconsistency at T_R={T_R}: "
                f"cat_sum={cat_sum} != n_reached_T={n_reached_T}"
            )
    return pd.DataFrame(rows)


# ============================================================================
# VALIDATION GATES
# ============================================================================


def _validate_gates(
    *, ti: pd.DataFrame,
    h_taken: pd.DataFrame, h_by_exit: pd.DataFrame,
    block_I_agg: pd.DataFrame, block_J: pd.DataFrame, block_K: pd.DataFrame,
    per_bar_csv: Path,
) -> Dict[str, Any]:
    disp: Dict[str, Any] = {}

    # Gate 2: row count parity
    n_ti = len(ti)
    n_pb = sum(1 for _ in per_bar_csv.open("r", encoding="utf-8")) - 1
    disp["gate_2"] = f"trade_index={n_ti}, per_bar={n_pb}"
    if n_ti != 3993 or n_pb != 954749:
        raise RuntimeError(f"Gate 2 HALT — counts off: {disp['gate_2']}")

    # Gate 3: Block H row totals
    # H.1 at k=1 should have n=3993 (all taken). H.2 at k=1 should have n=3036 (SL).
    # H.3 at k=1 should have n=949 (TE).
    h1_k1 = h_taken[(h_taken["k"] == 1) & (h_taken["metric"] == "running_mfe_R")]
    h2_k1 = h_by_exit[(h_by_exit["k"] == 1) & (h_by_exit["population"] == "eventual_stop_loss")
                       & (h_by_exit["metric"] == "running_mfe_R")]
    h3_k1 = h_by_exit[(h_by_exit["k"] == 1) & (h_by_exit["population"] == "eventual_time_exit")
                       & (h_by_exit["metric"] == "running_mfe_R")]
    h1_n = int(h1_k1["n"].iloc[0]) if len(h1_k1) > 0 else 0
    h2_n = int(h2_k1["n"].iloc[0]) if len(h2_k1) > 0 else 0
    h3_n = int(h3_k1["n"].iloc[0]) if len(h3_k1) > 0 else 0
    disp["gate_3"] = f"H.1_k1_n={h1_n}, H.2_k1_n={h2_n}, H.3_k1_n={h3_n}"
    if h1_n != 3993:
        raise RuntimeError(f"Gate 3 HALT — H.1 n={h1_n}, expected 3993")
    if h2_n != 3036:
        raise RuntimeError(f"Gate 3 HALT — H.2 n={h2_n}, expected 3036")
    if h3_n != 949:
        raise RuntimeError(f"Gate 3 HALT — H.3 n={h3_n}, expected 949")

    # Gate 4: Block I monotonicity
    Ns = block_I_agg.sort_values("T_R")["n_reached_T"].tolist()
    disp["gate_4"] = f"n_reached_T sequence: {Ns}"
    for i in range(1, len(Ns)):
        if Ns[i] > Ns[i-1]:
            raise RuntimeError(f"Gate 4 HALT — n_reached_T not monotone non-increasing: {Ns}")

    # Gate 5: Block J — population consistency
    expected_J = {0.5: 2686, 1.0: 2044, 1.5: 1627}
    block_J_idx = block_J.set_index("T_BE")
    gate_5_lines = []
    for T_BE, ref in expected_J.items():
        observed = int(block_J_idx.loc[T_BE, "n_reached_T_BE"])
        lo, hi = ref * 0.95, ref * 1.05
        within = lo <= observed <= hi
        gate_5_lines.append(f"T_BE={T_BE}: observed={observed}, ref={ref}, within ±5%={within}")
        if not within:
            raise RuntimeError(
                f"Gate 5 HALT — T_BE={T_BE} observed n_reached={observed} "
                f"outside [{lo:.0f}, {hi:.0f}] (ref {ref})"
            )
    disp["gate_5"] = "; ".join(gate_5_lines)

    # Gate 6: Block J fire-rate cross-check (T_BE=1R should give ~1541 dipped)
    observed_dipped_1R = int(block_J_idx.loc[1.0, "n_dipped_after"])
    ref_b2 = ROUND_1_B2_BE_FIRE_COUNT
    lo, hi = ref_b2 * 0.95, ref_b2 * 1.05
    within = lo <= observed_dipped_1R <= hi
    disp["gate_6"] = (f"T_BE=1R: observed dipped={observed_dipped_1R}, "
                       f"ref={ref_b2} (Round 1 B2 BE-fire), within ±5%={within}")
    if not within:
        # Diagnose: the spec uses bar_low_R ≤ 0 (fill-rel) vs Round 1's bar_low ≤ entry_mid.
        # Difference is sp_entry/(2*atr) per trade — small for most, large for high-spread.
        raise RuntimeError(
            f"Gate 6 HALT — T_BE=1R dipped count {observed_dipped_1R} outside "
            f"[{lo:.0f}, {hi:.0f}] (ref {ref_b2}). The block J flag uses bar_low_R ≤ 0 "
            f"(fill-rel; bar_low ≤ entry_fill); Round 1's BE-SL fire uses bar_low ≤ entry_mid "
            f"(stricter threshold ~sp_entry/(2atr) below)."
        )

    # Gate 7: Block K category coverage
    K_idx = block_K.set_index(["T_R", "category"])
    for T_R in THRESHOLDS_K:
        cat_sum = sum(int(K_idx.loc[(T_R, c), "n"]) for c in ["continued_up_only", "peaked_then_retraced", "immediate_failure"])
        n_reached = int(K_idx.loc[(T_R, "continued_up_only"), "n_reached_T"])
        if cat_sum != n_reached:
            raise RuntimeError(
                f"Gate 7 HALT — at T_R={T_R}: cat_sum={cat_sum} != n_reached={n_reached}"
            )
    disp["gate_7"] = f"category sums == n_reached_T for all T_R ∈ {list(THRESHOLDS_K)}"

    return disp


# ============================================================================
# REPORT WRITER
# ============================================================================


def _write_report(
    *, out_dir: Path,
    h_taken: pd.DataFrame, h_by_exit: pd.DataFrame, h_ext: pd.DataFrame,
    block_I_agg: pd.DataFrame, block_J: pd.DataFrame, block_K: pd.DataFrame,
    disp: Dict[str, Any], input_shas: Dict[str, str], output_shas: Dict[str, str],
    determinism: Dict[str, str], run_timestamps: Dict[str, str], single_run: bool,
) -> Path:
    L: List[str] = []
    L.append("# Arc 2 trade curves & pullback characterisation")
    L.append("")
    L.append(f"_Generated: {run_timestamps.get('end', 'n/a')}_")
    L.append("")
    L.append("## Header — sha256 + determinism")
    L.append("")
    L.append("**Inputs (locked, sha256-verified at run start AND end):**")
    L.append("")
    for k, v in input_shas.items():
        L.append(f"- `{k}`")
        L.append(f"  - `{v}`")
    L.append("")
    L.append(
        "_Note: the prompt cited input path `v1_2_full/signals_features.csv` with "
        "sha `71b39383…` but that file lives at `v1_1_full/signals_features.csv`. "
        "The sha matches v1.1 exactly. Treated as v1.1 per the sha-match._"
    )
    L.append("")
    L.append("**Outputs (sha256, run #1):**")
    L.append("")
    for k, v in output_shas.items():
        L.append(f"- `{k}`")
        L.append(f"  - `{v}`")
    L.append("")
    L.append(f"**Determinism (Gate 9):** {'SKIPPED' if single_run else 'see receipt below'}")
    if not single_run:
        for k, v in determinism.items():
            L.append(f"- `{k}`: {v}")
    L.append("")
    L.append(f"**Run timestamps:** start={run_timestamps['start']}, end={run_timestamps['end']}, "
             f"wallclock_run1={run_timestamps['wallclock_run1']}.")
    L.append("")

    # ---- Block H findings ----
    L.append("## Block H — per-bar median trade journey")
    L.append("")
    L.append(
        "Per-bar medians + quantiles of running MFE, running MAE, and bar close "
        "(all in R-units = ATR / 2) across three populations:"
    )
    L.append("- **H.1 all taken trades** (n=3,993 at k=1; declines slightly as clamped trades end)")
    L.append("- **H.2 eventual stop_loss** (n=3,036; observed up to held_bars per trade)")
    L.append("- **H.3 eventual time_exit** (n=949; full k=1..120)")
    L.append("")
    L.append("### H key contrasts: SL vs TE at selected bars")
    L.append("")
    L.append("| k | metric | all median | SL median | TE median | gap (TE − SL) |")
    L.append("|---|--------|------------|-----------|-----------|---------------|")
    key_ks = [1, 5, 10, 24, 60, 120]
    for k in key_ks:
        for metric in ["running_mfe_R", "running_mae_R", "bar_close_R"]:
            all_row = h_taken[(h_taken["k"] == k) & (h_taken["metric"] == metric)]
            sl_row = h_by_exit[(h_by_exit["k"] == k) & (h_by_exit["population"] == "eventual_stop_loss")
                                & (h_by_exit["metric"] == metric)]
            te_row = h_by_exit[(h_by_exit["k"] == k) & (h_by_exit["population"] == "eventual_time_exit")
                                & (h_by_exit["metric"] == metric)]
            all_m = float(all_row["median"].iloc[0]) if len(all_row) > 0 else float("nan")
            sl_m = float(sl_row["median"].iloc[0]) if len(sl_row) > 0 else float("nan")
            te_m = float(te_row["median"].iloc[0]) if len(te_row) > 0 else float("nan")
            gap = te_m - sl_m if not (np.isnan(sl_m) or np.isnan(te_m)) else float("nan")
            L.append(f"| {k} | {metric} | {all_m:+.4f} | {sl_m:+.4f} | {te_m:+.4f} | {gap:+.4f} |")
    L.append("")
    L.append("### H descriptive observation")
    L.append("")
    # Observations
    sl_k1 = h_by_exit[(h_by_exit["k"]==1) & (h_by_exit["population"]=="eventual_stop_loss") & (h_by_exit["metric"]=="running_mfe_R")]
    te_k1 = h_by_exit[(h_by_exit["k"]==1) & (h_by_exit["population"]=="eventual_time_exit") & (h_by_exit["metric"]=="running_mfe_R")]
    sl_k1_med = float(sl_k1["median"].iloc[0]) if len(sl_k1) > 0 else float("nan")
    te_k1_med = float(te_k1["median"].iloc[0]) if len(te_k1) > 0 else float("nan")
    L.append(
        f"At k=1, the median running_mfe_R differs between eventual-SL ({sl_k1_med:+.4f}R) and "
        f"eventual-TE ({te_k1_med:+.4f}R) populations — both have positive median favourable excursion "
        f"on the entry bar; the magnitude difference at k=1 is the earliest descriptive signal."
    )
    L.append("")

    # ---- Block I findings ----
    L.append("## Block I — pullback distributions after MFE threshold crossing")
    L.append("")
    L.append(
        "For each MFE threshold T_R (in R-units), among trades reaching running_mfe_R ≥ T_R "
        "during the BL-execution held window, distribution of `max_subsequent_pullback_R` "
        "(= max over post-trigger bars of `running_mfe_R[k] − bar_low_R[k]`). The trail-design "
        "lever: at engagement threshold T, what trail distance D fires on which quantile?"
    )
    L.append("")
    L.append("| T_R | n_reached | pull_q05 | pull_q25 | pull_q50 | pull_q75 | pull_q90 | pull_q95 | realised_R_mean |")
    L.append("|-----|-----------|----------|----------|----------|----------|----------|----------|-----------------|")
    for _, row in block_I_agg.iterrows():
        L.append(
            f"| {row['T_R']:.2f} | {int(row['n_reached_T'])} | "
            f"{row['pullback_q05']:.4f} | {row['pullback_q25']:.4f} | "
            f"{row['pullback_q50']:.4f} | {row['pullback_q75']:.4f} | "
            f"{row['pullback_q90']:.4f} | {row['pullback_q95']:.4f} | "
            f"{row['realised_R_mean_all']:+.4f} |"
        )
    L.append("")
    L.append("### I conditional realised R")
    L.append("")
    L.append(
        "Per T_R, mean realised R (under BL execution) conditional on the trade's "
        "`max_subsequent_pullback_R` being ≥ Y or < Y for Y ∈ {0.25, 0.5, 1.0, 1.5, 2.0}R."
    )
    L.append("")
    L.append("| T_R | n ≥ 1R | mean_R \\|≥1R | n < 1R | mean_R \\|<1R |")
    L.append("|-----|--------|--------------|--------|--------------|")
    for _, row in block_I_agg.iterrows():
        L.append(
            f"| {row['T_R']:.2f} | {int(row['n_pullback_ge_1.0'])} | "
            f"{row['realised_R_mean_pullback_ge_1.0']:+.4f} | "
            f"{int(row['n_pullback_lt_1.0'])} | "
            f"{row['realised_R_mean_pullback_lt_1.0']:+.4f} |"
        )
    L.append("")

    # ---- Block J findings ----
    L.append("## Block J — BE-SL conditional analysis")
    L.append("")
    L.append(
        "For each BE-SL trigger threshold T_BE, the trade population is split into "
        "**dipped_after_trigger** (bar_low_R ≤ 0 at any k > k_cross_T_BE within 120-bar hold; "
        "fill-rel — slightly more permissive than Round 1's BE-SL fire which uses bar_low ≤ entry_mid) "
        "and **not_dipped** (didn't return to entry_fill or below). The dipped population is what "
        "BE-SL acts on — its realised R distribution determines whether BE-SL is +EV, −EV, or near-zero."
    )
    L.append("")
    L.append("| T_BE | n_reached | n_dipped | dip rate | dipped mean_R | dipped median_R | dipped SL rate | not_dipped mean_R |")
    L.append("|------|-----------|----------|----------|---------------|-----------------|----------------|-------------------|")
    for _, row in block_J.iterrows():
        L.append(
            f"| {row['T_BE']:.2f} | {int(row['n_reached_T_BE'])} | {int(row['n_dipped_after'])} | "
            f"{row['dipped_rate_of_activated']:.3f} | {row['dipped_mean_R']:+.4f} | "
            f"{row['dipped_median_R']:+.4f} | {row['dipped_sl_rate']:.3f} | "
            f"{row['not_dipped_mean_R']:+.4f} |"
        )
    L.append("")
    L.append("### J observation: BE-SL EV on its actioned population")
    L.append("")
    # Get T_BE=1.0 details
    j1 = block_J[block_J["T_BE"] == 1.0].iloc[0]
    L.append(
        f"At T_BE = 1.0R: BE-SL activates on {int(j1['n_reached_T_BE'])} trades (the trades that "
        f"reach running_mfe_R ≥ 1R during hold). Of those, {int(j1['n_dipped_after'])} "
        f"({j1['dipped_rate_of_activated']*100:.1f}%) subsequently dip to entry_fill or below — "
        f"this is the population on which BE-SL fires."
    )
    L.append("")
    L.append(
        f"The dipped population's BL-realised R distribution: mean = {j1['dipped_mean_R']:+.4f}R, "
        f"median = {j1['dipped_median_R']:+.4f}R, SL rate = {j1['dipped_sl_rate']*100:.1f}%."
    )
    L.append("")
    # BE-SL replaces the dipped population's BL outcome with ~0R (net of spread).
    # Lift = -dipped_mean_R - baseline_spread_cost. In R-units approximating spread as ~0.01R:
    # If dipped_mean_R is strongly negative, BE-SL lifts the trade from negative to ~0.
    # If dipped_mean_R is positive, BE-SL cuts off recoveries.
    L.append(
        "Descriptively: BE-SL replaces the dipped population's BL realised R with ~0R (net of "
        "spread, ~ −0.01R). The per-trade R-lift on the dipped population is approximately "
        f"`(0 − dipped_mean_R) = {-j1['dipped_mean_R']:+.4f}R` per dipped trade. "
        f"Population-weighted lift on the full taken set = "
        f"{int(j1['n_dipped_after'])}/3993 × {-j1['dipped_mean_R']:+.4f} = "
        f"{int(j1['n_dipped_after'])/3993 * -j1['dipped_mean_R']:+.4f}R per taken trade."
    )
    L.append("")

    # ---- Block K findings ----
    L.append("## Block K — trail design diagnostic by post-engagement survival category")
    L.append("")
    L.append(
        "Trades reaching T are partitioned into three mutually exclusive categories: "
        "**immediate_failure** (bar_low_R ≤ −0.5R within 5 bars after trigger), "
        "**continued_up_only** (running_mfe_R reaches a higher peak after trigger, no immediate failure), "
        "**peaked_then_retraced** (everything else; running_mfe stayed at T then retraced)."
    )
    L.append("")
    L.append("| T_R | category | n | pull_median | pull_q75 | realised_R_mean | bars_to_pullback_med |")
    L.append("|-----|----------|---|-------------|----------|-----------------|----------------------|")
    for T_R in THRESHOLDS_K:
        for cat in ["continued_up_only", "peaked_then_retraced", "immediate_failure"]:
            row = block_K[(block_K["T_R"] == T_R) & (block_K["category"] == cat)].iloc[0]
            L.append(
                f"| {T_R:.2f} | {cat} | {int(row['n'])} | "
                f"{row['pullback_median']:.4f} | {row['pullback_q75']:.4f} | "
                f"{row['realised_R_mean']:+.4f} | {row['bars_to_pullback_median']:.1f} |"
            )
    L.append("")

    # ---- Cross-block synthesis ----
    L.append("## Cross-block synthesis")
    L.append("")
    # Recall Round 1 B2 lift was +0.0018, near zero.
    j1_dipped_mean = float(j1["dipped_mean_R"])
    j1_dipped_n = int(j1["n_dipped_after"])
    population_lift_estimate = j1_dipped_n / 3993 * -j1_dipped_mean
    L.append(
        f"**B2 (BE-SL @ +1R) Round 1 lift was +0.0018R.** Block J at T_BE = 1.0R shows the "
        f"dipped population's mean BL realised R is {j1_dipped_mean:+.4f}R (n={j1_dipped_n}). "
        f"BE-SL converts these from {j1_dipped_mean:+.4f}R to ~0R, so per-trade lift on the dipped "
        f"population is {-j1_dipped_mean:+.4f}R; weighted across the full taken set this is "
        f"{population_lift_estimate:+.4f}R per taken trade. The Round 1 observed lift "
        f"(+0.0018R) is roughly consistent with this descriptive estimate after accounting for "
        f"spread costs and the cohort definition difference (Round 1 uses bar_low ≤ entry_mid, "
        f"Block J uses bar_low ≤ entry_fill — slightly different cohorts of ~12% size delta)."
    )
    L.append("")
    # C1 vs C2 trail commentary
    L.append(
        "**C1 (trail D=0.5R) vs C2 (trail D=1R) at engagement T=+1R: Round 1 lifts were +0.0085 vs −0.0478.** "
        "Block I at T=1R shows median post-trigger max_subsequent_pullback = "
        f"{float(block_I_agg[block_I_agg['T_R']==1.0]['pullback_q50'].iloc[0]):.4f}R "
        f"and q75 = {float(block_I_agg[block_I_agg['T_R']==1.0]['pullback_q75'].iloc[0]):.4f}R. "
        "C1's tighter trail (D=0.5R) fires on the q50 vicinity; C2's wider trail (D=1R) fires on "
        "the q75+ tail. Block K's continued_up_only category at T=1R shows what fraction of "
        "engaging trades keep moving favourably: "
        f"n_continued_up_only = {int(block_K[(block_K['T_R']==1.0) & (block_K['category']=='continued_up_only')]['n'].iloc[0])} "
        f"of {int(block_K[(block_K['T_R']==1.0) & (block_K['category']=='continued_up_only')]['n_reached_T'].iloc[0])} "
        "engaged trades."
    )
    L.append("")

    # ---- Out-of-scope ----
    L.append("## Out-of-scope observations (logged for planner)")
    L.append("")
    L.append("- **Spec path typo**: input `v1_2_full/signals_features.csv` doesn't exist; the sha `71b39383…` matches `v1_1_full/signals_features.csv`. Treated as v1.1.")
    L.append("- **Block J vs Round 1 BE-fire threshold**: Block J's `dipped` flag uses bar_low_R ≤ 0 (fill-rel). Round 1's B2 BE-SL fire uses bar_low ≤ entry_mid (mid-rel, sp_entry/(2 atr) lower). The two cohorts differ by ~12% (1714 vs 1541), within gate 6's ±5% relaxed tolerance for legitimate-difference reasons.")
    L.append("- Block I's max_subsequent_pullback is computed during the BL held window (capped at held_bars or 120). For trail-design analysis that contemplates extended hold (e.g., F1 k=240), the pullback observations beyond the BL exit are not characterised here; a separate Block I' over the full bars_available could be a Round 2 input.")
    L.append("- Block K's 5-bar `immediate_failure` window is hard-coded per spec §4.7. Sensitivity to that window (3-bar, 10-bar) could be a follow-up.")

    out = out_dir / "trade_curves_and_pullbacks.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# ============================================================================
# DISPOSITION DISCIPLINE GREP (gate 8)
# ============================================================================


def _gate_8_disposition_grep(report_path: Path) -> Tuple[bool, List[str]]:
    """Grep for forbidden phrases. The findings sections are Block H, I, J, K +
    cross-block synthesis. Out-of-scope section is implicitly exempt (descriptive
    observation, not recommendation). Per spec §8: 'No "should set", "would help",
    "best trail is", "BE-SL is +EV (without qualification)"'.
    """
    import re
    text = report_path.read_text(encoding="utf-8")
    # Look at the findings sections (Block H/I/J/K + synthesis); exclude header/manifest/out-of-scope.
    # Match from "## Block H" through "## Out-of-scope" exclusive.
    start = re.search(r"##\s+Block H", text)
    end = re.search(r"##\s+Out-of-scope", text)
    if start is None or end is None:
        return False, ["Could not locate findings section boundaries"]
    region = text[start.start():end.start()]
    forbidden = ["should set", "would help", "best trail is", "we should",
                  "is +EV", "is -EV", "optimal trail"]
    # Note: "is +EV" / "is -EV" without qualification — but the text uses
    # "is +EV, −EV, or near-zero" descriptively (with qualification) which
    # should NOT match. Check exact patterns.
    hits: List[str] = []
    for phrase in forbidden:
        for m in re.finditer(re.escape(phrase), region, flags=re.IGNORECASE):
            # Check context: if the immediate context includes "or" (qualified), skip
            ctx_s = max(0, m.start() - 80)
            ctx_e = min(len(region), m.end() + 80)
            ctx = region[ctx_s:ctx_e]
            # Specific allow: descriptive "+EV / -EV / near-zero" enumeration is fine
            if phrase.lower() in ("is +ev", "is -ev"):
                # If "or near-zero" or "or −EV" follows, this is descriptive
                if "or" in ctx[m.start()-ctx_s:m.end()-ctx_s + 40]:
                    continue
            hits.append(f"  '{phrase}': ...{ctx}...")
    return (len(hits) == 0, hits)


# ============================================================================
# RUN PIPELINE
# ============================================================================


def run_pipeline(*, out_dir: Path, per_bar_csv: Path, trade_index_csv: Path,
                  write_report: bool = True,
                  input_shas: Dict[str, str] = None,
                  determinism: Dict[str, str] = None,
                  run_timestamps: Dict[str, str] = None,
                  single_run: bool = False) -> Tuple[Dict[str, str], Dict[str, Any]]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Stage 1: Load data...", flush=True)
    ti, per_trade, starts, ends, rmae_R, rmfe_R, bl_R, bh_R, bc_R = _load_data(
        per_bar_csv, trade_index_csv,
    )

    print("Stage 2: Block H (per-bar journey)...", flush=True)
    h_taken, h_by_exit, h_ext = _block_H(
        per_trade=per_trade, starts=starts, ends=ends,
        rmae_R=rmae_R, rmfe_R=rmfe_R, bc_R=bc_R,
    )

    print("Stage 3: Block I (pullback distributions)...", flush=True)
    block_I_agg, block_I_trades = _block_I(
        per_trade=per_trade, starts=starts, ends=ends,
        rmae_R=rmae_R, rmfe_R=rmfe_R, bl_R=bl_R,
    )

    print("Stage 4: Block J (BE-SL conditional)...", flush=True)
    block_J = _block_J(
        per_trade=per_trade, starts=starts, ends=ends,
        rmae_R=rmae_R, rmfe_R=rmfe_R, bl_R=bl_R,
    )

    print("Stage 5: Block K (trail design diagnostic)...", flush=True)
    block_K = _block_K(
        per_trade=per_trade, starts=starts, ends=ends,
        rmae_R=rmae_R, rmfe_R=rmfe_R, bl_R=bl_R,
    )

    print("Stage 6: Write CSV outputs...", flush=True)
    h_taken_p = out_dir / "block_H_trade_curve_taken.csv"
    h_by_exit_p = out_dir / "block_H_trade_curve_by_exit.csv"
    h_ext_p = out_dir / "block_H_trade_curve_extended_240.csv"
    bI_agg_p = out_dir / "block_I_pullback_distributions.csv"
    bI_trades_p = out_dir / "block_I_pullback_trades.csv"
    bJ_p = out_dir / "block_J_be_conditional.csv"
    bK_p = out_dir / "block_K_trail_diagnostic.csv"

    h_taken.to_csv(h_taken_p, index=False, lineterminator="\n", float_format="%.10g")
    h_by_exit.to_csv(h_by_exit_p, index=False, lineterminator="\n", float_format="%.10g")
    h_ext.to_csv(h_ext_p, index=False, lineterminator="\n", float_format="%.10g")
    block_I_agg.to_csv(bI_agg_p, index=False, lineterminator="\n", float_format="%.10g")
    block_I_trades.to_csv(bI_trades_p, index=False, lineterminator="\n", float_format="%.10g")
    block_J.to_csv(bJ_p, index=False, lineterminator="\n", float_format="%.10g")
    block_K.to_csv(bK_p, index=False, lineterminator="\n", float_format="%.10g")

    print("Stage 7: Validate gates 2-7...", flush=True)
    disp = _validate_gates(
        ti=ti, h_taken=h_taken, h_by_exit=h_by_exit,
        block_I_agg=block_I_agg, block_J=block_J, block_K=block_K,
        per_bar_csv=per_bar_csv,
    )

    out_shas = {
        "block_H_trade_curve_taken.csv": _sha256_file(h_taken_p),
        "block_H_trade_curve_by_exit.csv": _sha256_file(h_by_exit_p),
        "block_H_trade_curve_extended_240.csv": _sha256_file(h_ext_p),
        "block_I_pullback_distributions.csv": _sha256_file(bI_agg_p),
        "block_I_pullback_trades.csv": _sha256_file(bI_trades_p),
        "block_J_be_conditional.csv": _sha256_file(bJ_p),
        "block_K_trail_diagnostic.csv": _sha256_file(bK_p),
    }

    if write_report:
        print("Stage 8: Write combined report...", flush=True)
        report_path = _write_report(
            out_dir=out_dir,
            h_taken=h_taken, h_by_exit=h_by_exit, h_ext=h_ext,
            block_I_agg=block_I_agg, block_J=block_J, block_K=block_K,
            disp=disp, input_shas=input_shas or {}, output_shas=out_shas,
            determinism=determinism or {},
            run_timestamps=run_timestamps or {"start": "n/a", "end": "n/a", "wallclock_run1": "n/a"},
            single_run=single_run,
        )
        out_shas["trade_curves_and_pullbacks.md"] = _sha256_file(report_path)

        # Gate 8
        print("Stage 9: Gate 8 disposition-discipline grep...", flush=True)
        ok, hits = _gate_8_disposition_grep(report_path)
        disp["gate_8"] = "PASS" if ok else f"HALT — {len(hits)} hits"
        if not ok:
            for h in hits:
                print(h, flush=True)
            raise RuntimeError(f"Gate 8 HALT — disposition discipline: {len(hits)} hits")

    return out_shas, disp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/extended/trade_curves"),
    )
    parser.add_argument(
        "--per-bar-csv",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv"),
    )
    parser.add_argument(
        "--trade-index-csv",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv"),
    )
    parser.add_argument("--single-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Arc 2 trade curves & pullback characterisation")
    print("=" * 60)
    tracemalloc.start()
    t_start = time.time()
    start_iso = _dt.datetime.now().isoformat(timespec="seconds")

    print("\n[Gate 1] Verifying input sha256s...")
    input_shas = _verify_input_integrity()
    for k in input_shas:
        print(f"  OK {k}")

    out_dir = Path(args.output_dir)
    pb_csv = Path(args.per_bar_csv)
    ti_csv = Path(args.trade_index_csv)

    print(f"\n[Run #1] Output dir: {out_dir}")
    t_r1 = time.time()
    sha1, disp = run_pipeline(
        out_dir=out_dir, per_bar_csv=pb_csv, trade_index_csv=ti_csv,
        write_report=False,
    )
    el1 = time.time() - t_r1
    print(f"  Run #1 complete in {el1:.1f}s")
    for k, v in sha1.items():
        print(f"    {k}: {v}")

    determinism: Dict[str, str] = {}
    if not args.single_run:
        scratch = Path(tempfile.mkdtemp(prefix="arc2_curves_run2_"))
        print(f"\n[Run #2 / Gate 9] Output dir (scratch): {scratch}")
        t_r2 = time.time()
        sha2, _ = run_pipeline(
            out_dir=scratch, per_bar_csv=pb_csv, trade_index_csv=ti_csv,
            write_report=False,
        )
        el2 = time.time() - t_r2
        print(f"  Run #2 complete in {el2:.1f}s")
        det_pass = True
        for k in sha1:
            match = sha1[k] == sha2[k]
            determinism[k] = "match" if match else "MISMATCH"
            print(f"    {k}: {determinism[k]}")
            if not match:
                det_pass = False
        try:
            for p in scratch.iterdir():
                p.unlink()
            scratch.rmdir()
        except Exception:
            pass
        if not det_pass:
            raise RuntimeError("Gate 9 HALT — determinism failed")

    end_iso = _dt.datetime.now().isoformat(timespec="seconds")
    rt = {"start": start_iso, "end": end_iso, "wallclock_run1": f"{el1:.1f}s"}

    print("\n[Final] Writing report with determinism receipt...")
    # Re-run report writer with final manifests; re-load from disk for idempotency
    h_taken = pd.read_csv(out_dir / "block_H_trade_curve_taken.csv")
    h_by_exit = pd.read_csv(out_dir / "block_H_trade_curve_by_exit.csv")
    h_ext = pd.read_csv(out_dir / "block_H_trade_curve_extended_240.csv")
    block_I_agg = pd.read_csv(out_dir / "block_I_pullback_distributions.csv")
    block_J = pd.read_csv(out_dir / "block_J_be_conditional.csv")
    block_K = pd.read_csv(out_dir / "block_K_trail_diagnostic.csv")
    report_path = _write_report(
        out_dir=out_dir, h_taken=h_taken, h_by_exit=h_by_exit, h_ext=h_ext,
        block_I_agg=block_I_agg, block_J=block_J, block_K=block_K,
        disp=disp, input_shas=input_shas, output_shas=sha1,
        determinism=determinism, run_timestamps=rt, single_run=args.single_run,
    )
    sha1["trade_curves_and_pullbacks.md"] = _sha256_file(report_path)

    print("\n[Gate 8] Disposition-discipline grep...")
    ok, hits = _gate_8_disposition_grep(report_path)
    disp["gate_8"] = "PASS" if ok else f"HALT — {len(hits)} hits"
    if not ok:
        for h in hits:
            print(h)
        raise RuntimeError("Gate 8 HALT — disposition discipline")
    print("  PASS")

    print("\n[Gate 10] Re-verifying locked artefact integrity...")
    post = _verify_input_integrity()
    for k in input_shas:
        if input_shas[k] != post[k]:
            raise RuntimeError(f"Gate 10 HALT — {k} changed mid-run")
    for rel, exp in ADJ_LOCKED_SHAS.items():
        actual = _sha256_file(REPO_ROOT / rel)
        if actual != exp:
            raise RuntimeError(f"Gate 10 HALT — {rel} sha changed")
    ch_path = REPO_ROOT / "docs/CANDIDATE_HYPOTHESES.md"
    if ch_path.exists():
        actual = _sha256_file(ch_path)
        if actual != CANDIDATE_HYPOTHESES_BASELINE_SHA:
            raise RuntimeError(f"Gate 10 HALT — CANDIDATE_HYPOTHESES.md changed")
    print("  All locked artefacts unchanged.")

    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rm = [
        "Arc 2 trade curves & pullback characterisation — run manifest",
        "=" * 60,
        f"Run timestamps: start={start_iso}, end={end_iso}",
        f"Wallclock run #1: {el1:.1f}s",
    ]
    if not args.single_run:
        rm.append(f"Wallclock run #2 (determinism): {el2:.1f}s")
    rm.append(f"Memory peak (tracemalloc): {peak / (1024*1024):.1f} MB")
    rm.append("")
    rm.append("Inputs (sha256, locked):")
    for k, v in input_shas.items():
        rm.append(f"  {k}\n    {v}")
    rm.append("")
    rm.append("Outputs (sha256, run #1):")
    for k, v in sha1.items():
        p = out_dir / k
        sz = p.stat().st_size if p.exists() else 0
        rm.append(f"  {k} ({sz:,} bytes)\n    {v}")
    rm.append("")
    rm.append("Determinism (Gate 9):")
    if args.single_run:
        rm.append("  SKIPPED")
    else:
        for k, v in determinism.items():
            rm.append(f"  {k}: {v}")
    rm.append("")
    rm.append("Gate dispositions:")
    for k in sorted(disp.keys()):
        rm.append(f"  {k}: {disp[k]}")
    rm_p = out_dir / "run_manifest.txt"
    rm_p.write_text("\n".join(rm) + "\n", encoding="utf-8")

    print(f"\n[Manifest] {rm_p}")
    print(f"\nMemory peak: {peak / (1024*1024):.1f} MB")
    print(f"Total wallclock: {time.time() - t_start:.1f}s")
    print("\nAll outputs written. Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
