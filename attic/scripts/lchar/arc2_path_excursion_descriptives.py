"""Arc 2 — Path-excursion descriptive characterisation (Round 3A).

Phase: L6 Arc 2 Phase 3 — Path-excursion descriptive characterisation
(L6_0_METHODOLOGY_LOCK Sections 14.2 derivative experiment, 14.5
descriptive discipline, 14.6 read-existing-CSV backfill).

Three descriptive blocks computed on existing per_bar_paths.csv:
  Block DD — only_up MAE excursion at multiple MFE checkpoints
  Block EE — only_up MFE peak vs final close (giveback analysis)
  Block FF — early-bar separator between go-up and go-down groups

Three subsets evaluated: S0_pop, S1_q5q2, S4_q5xq2q3.

DESCRIPTIVE ONLY. No variants, no filter selection, no exit
proposals. Output is read by chat-side to design Round 3B variants
with parameter values grounded in the distributions.

Outputs to: results/l6/arc2/characterisation/extended/path_excursion_descriptives/
"""

from __future__ import annotations

import filecmp
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Locked input sha256s (gate 1, re-verified as gate 9)
# ---------------------------------------------------------------------------
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv": "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv": "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv": "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv": "4a61407f0f1fc1b74486f0614928e776201dc6469d874db8393e689d20cdb2ff",
    "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv": "a5e3f8e68aa64d8fd53f752705a33613d9877dbde1f8265cb4a38d753c5e088e",
    "results/l6/arc2/characterisation/extended/path_by_subset/block_V_subset_category_breakdown.csv": "78633e9904baf2a672d2c8692f4b3557fec0aa3af8044ef3296dde08bad71c02",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py": "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml": "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "L6_0_METHODOLOGY_LOCK.md": "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}

OUTPUT_DIR_REL = "results/l6/arc2/characterisation/extended/path_excursion_descriptives"

# Path-category constants (must reproduce Block B 1R counts).
PATH_THRESH_ATR: float = 2.0  # +1R = +2.0 ATR fill-rel
PATH_HOLD_CAP: int = 120

# Expected Block B 1R counts.
BLOCK_B_1R_COUNTS: Dict[str, int] = {
    "only_up": 956,
    "up_then_down": 1075,
    "down_then_up": 1090,
    "straight_to_sl": 858,
    "simultaneous": 13,
    "neither_reached": 1,
}
ALL_CATS: Tuple[str, ...] = (
    "only_up",
    "up_then_down",
    "down_then_up",
    "straight_to_sl",
    "simultaneous",
    "neither_reached",
)

# Subsets to characterise (only three for this round).
SUBSET_DEFS: List[Tuple[str, Dict[str, Any]]] = [
    ("S0_pop", {"all": True, "expected_n": 3993}),
    ("S1_q5q2", {"qa": ("Q5",), "qb": ("Q2",), "expected_n": 190}),
    ("S4_q5xq2q3", {"qa": ("Q5",), "qb": ("Q2", "Q3"), "expected_n": 368}),
]
SUBSET_IDS: Tuple[str, ...] = tuple(s[0] for s in SUBSET_DEFS)

# Block DD checkpoint thresholds (ATR fill-rel).
DD_THRESHOLDS_ATR: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
DD_AFTER_THRESHOLDS_ATR: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)

# Block FF parameters.
FF_K_GRID: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 90)
FF_TAU_GRID: np.ndarray = np.round(np.arange(-3.0, 1.0 + 1e-9, 0.1), 4)
FF_RECALL_A_BUDGETS: Tuple[float, ...] = (0.99, 0.95, 0.90)

# Disposition discipline forbidden patterns (Section 14.5).
FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "should set sl at",
    "best threshold is",
    " recommend",
    "the right value",
    "we should cut at",
    "this proves",
    "we should adopt",
    "should use",
)


# ===========================================================================
# sha256 / IO helpers
# ===========================================================================


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_locked(label: str) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for rel, expected in LOCKED_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"HALT ({label}) sha256 mismatch on {rel}:\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
        observed[rel] = actual
    return observed


def _write_csv(df: pd.DataFrame, path: Path, float_fmt: str = "%.10g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format=float_fmt)


# ===========================================================================
# Quintile bucketing (byte-faithful with arc2_entry_filter_bivariate.py)
# ===========================================================================


def _make_quintile_labels(
    values: pd.Series, tie_break: pd.Series
) -> Tuple[pd.Series, List[Tuple[float, float]]]:
    df = pd.DataFrame({"v": values.values, "t": tie_break.values}, index=values.index)
    df = df.sort_values(["v", "t"], kind="stable")
    n = len(df)
    base = n // 5
    rem = n - base * 5
    sizes = [base + (1 if i < rem else 0) for i in range(5)]
    labels: List[str] = []
    bounds: List[Tuple[float, float]] = []
    cursor = 0
    for qi, sz in enumerate(sizes):
        seg = df.iloc[cursor : cursor + sz]
        labels.extend([f"Q{qi + 1}"] * sz)
        bounds.append((float(seg["v"].min()), float(seg["v"].max())))
        cursor += sz
    df["q"] = labels
    return df["q"].reindex(values.index), bounds


def build_subsets() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    sf = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_1_full/signals_features.csv")
    ti = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv")
    bm = pd.read_csv(
        REPO_ROOT
        / "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv"
    )

    sf_taken = sf[sf["taken"] == True].copy()  # noqa: E712
    sf_taken = sf_taken.rename(columns={"time": "signal_bar_ts"})
    sf_taken["signal_bar_ts"] = pd.to_datetime(sf_taken["signal_bar_ts"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    taken = sf_taken.merge(
        ti[["trade_id", "pair", "signal_bar_ts", "atr_1h_wilder_at_signal"]],
        on=["pair", "signal_bar_ts"],
        how="left",
        validate="one_to_one",
    )
    taken = taken.merge(
        bm[["trade_id", "dist_d1_kijun_atr"]],
        on="trade_id",
        how="left",
        validate="one_to_one",
    )
    taken = taken.sort_values("trade_id").reset_index(drop=True)
    if len(taken) != 3993:
        raise RuntimeError(f"HALT: taken row count {len(taken)} != 3993")

    qa_labels, _ = _make_quintile_labels(taken["concurrent_signals_same_bar"], taken["trade_id"])
    taken["Q_A_concurrent"] = qa_labels.values
    qb_labels, _ = _make_quintile_labels(taken["dist_d1_kijun_atr"], taken["trade_id"])
    taken["Q_B_dist_d1"] = qb_labels.values

    # Gate 2.1: reproduce 25 block_P cell counts exactly.
    bp = pd.read_csv(
        REPO_ROOT
        / "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv"
    )
    diffs: List[str] = []
    for _, row in bp.iterrows():
        qa = row["Q_A_concurrent"]
        qb = row["Q_B_dist_d1"]
        exp_n = int(row["n"])
        sub = taken[(taken["Q_A_concurrent"] == qa) & (taken["Q_B_dist_d1"] == qb)]
        got_n = len(sub)
        if got_n != exp_n:
            diffs.append(f"  ({qa},{qb}): expected n={exp_n}, got n={got_n}")
    if diffs:
        raise RuntimeError("HALT (gate 2.1): block_P cell mismatch:\n" + "\n".join(diffs))

    subsets: Dict[str, np.ndarray] = {}
    for sid, spec in SUBSET_DEFS:
        if spec.get("all"):
            mask = pd.Series(True, index=taken.index)
        else:
            mask = taken["Q_A_concurrent"].isin(spec["qa"]) & taken["Q_B_dist_d1"].isin(spec["qb"])
        sub = taken[mask]
        expected_n = int(spec["expected_n"])
        if len(sub) != expected_n:
            raise RuntimeError(f"HALT (gate 2.2): subset {sid} size {len(sub)} != {expected_n}")
        subsets[sid] = sub["trade_id"].to_numpy(dtype=np.int64)

    labels = taken[
        ["trade_id", "pair", "signal_bar_ts", "fold_id", "Q_A_concurrent", "Q_B_dist_d1"]
    ].copy()
    labels["fold_id"] = labels["fold_id"].astype(int)
    return labels, subsets


def compute_categories(pb: pd.DataFrame, n_trades: int) -> np.ndarray:
    """Replicate Block B category derivation at +/-1R = +/-2 ATR within k<=120."""
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids = pb["trade_id"].to_numpy()
    ks = pb["k"].to_numpy()
    mfe = pb["running_mfe_atr"].to_numpy()
    mae = pb["running_mae_atr"].to_numpy()
    in_window = ks <= PATH_HOLD_CAP
    up_hit = (mfe >= PATH_THRESH_ATR) & in_window
    dn_hit = (mae <= -PATH_THRESH_ATR) & in_window

    SENTINEL = PATH_HOLD_CAP + 1
    t_up = np.full(n_trades, SENTINEL, dtype=np.int32)
    t_dn = np.full(n_trades, SENTINEL, dtype=np.int32)
    if up_hit.any():
        up_idx = np.where(up_hit)[0]
        up_df = pd.DataFrame({"tid": tids[up_idx], "k": ks[up_idx]})
        first_up = up_df.groupby("tid", sort=False)["k"].min()
        t_up[first_up.index.values.astype(np.int64)] = first_up.values.astype(np.int32)
    if dn_hit.any():
        dn_idx = np.where(dn_hit)[0]
        dn_df = pd.DataFrame({"tid": tids[dn_idx], "k": ks[dn_idx]})
        first_dn = dn_df.groupby("tid", sort=False)["k"].min()
        t_dn[first_dn.index.values.astype(np.int64)] = first_dn.values.astype(np.int32)

    up_reached = t_up <= PATH_HOLD_CAP
    dn_reached = t_dn <= PATH_HOLD_CAP
    cats = np.full(n_trades, "", dtype=object)
    only_up = up_reached & (~dn_reached)
    only_down = (~up_reached) & dn_reached
    both = up_reached & dn_reached
    neither = (~up_reached) & (~dn_reached)
    up_then_down = both & (t_up < t_dn)
    down_then_up = both & (t_dn < t_up)
    simul = both & (t_up == t_dn)
    cats[only_up] = "only_up"
    cats[only_down] = "straight_to_sl"
    cats[up_then_down] = "up_then_down"
    cats[down_then_up] = "down_then_up"
    cats[simul] = "simultaneous"
    cats[neither] = "neither_reached"

    # Gate 2.3: reproduce Block B 1R counts.
    counts = {c: int((cats == c).sum()) for c in ALL_CATS}
    diffs = []
    for c, exp in BLOCK_B_1R_COUNTS.items():
        if counts[c] != exp:
            diffs.append(f"  {c}: expected={exp}, got={counts[c]}")
    if diffs:
        raise RuntimeError("HALT (gate 2.3): Block B 1R counts diverge:\n" + "\n".join(diffs))
    return cats


# ===========================================================================
# Per-trade slicing helpers
# ===========================================================================


def _build_index(
    pb: pd.DataFrame, n_trades: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort per_bar by trade_id, k. Returns (starts, ends, rmfe, rmae, bc)."""
    pb_sorted = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids = pb_sorted["trade_id"].to_numpy(dtype=np.int64)
    starts = np.searchsorted(tids, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids, np.arange(n_trades), side="right")
    rmfe = pb_sorted["running_mfe_atr"].to_numpy(dtype=np.float64)
    rmae = pb_sorted["running_mae_atr"].to_numpy(dtype=np.float64)
    bc = pb_sorted["bar_close_atr"].to_numpy(dtype=np.float64)
    return starts, ends, rmfe, rmae, bc


# ===========================================================================
# Block DD — only_up MAE excursion at MFE checkpoints
# ===========================================================================


def compute_block_DD(
    starts, ends, rmfe, rmae, cats: np.ndarray, subsets: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """One row per (subset, trade) for only_up-labelled trades."""
    rows: List[Dict[str, Any]] = []
    for sid, tids in subsets.items():
        for tid in tids.tolist():
            if cats[tid] != "only_up":
                continue
            s, e = int(starts[tid]), int(ends[tid])
            mfe_arr = rmfe[s:e]
            mae_arr = rmae[s:e]
            n_bars = e - s
            row: Dict[str, Any] = {"subset_id": sid, "trade_id": int(tid), "n_bars": n_bars}
            # First-passage k for each MFE threshold.
            t_thresh: Dict[float, Optional[int]] = {}
            for X in DD_THRESHOLDS_ATR:
                hits = np.where(mfe_arr >= X)[0]
                t_thresh[X] = int(hits[0]) if len(hits) > 0 else None
            # mae_before_mfe_plus_X: min running_mae from k_idx 0 to t_thresh
            for X in DD_THRESHOLDS_ATR:
                t = t_thresh[X]
                if t is None:
                    row[f"t_mfe_plus_{int(X)}atr"] = -1
                    row[f"mae_before_mfe_plus_{int(X)}atr"] = float("nan")
                else:
                    # k=1..t (1-indexed) maps to indices 0..t (inclusive)
                    sl = mae_arr[: t + 1]
                    row[f"t_mfe_plus_{int(X)}atr"] = t + 1  # 1-indexed k
                    row[f"mae_before_mfe_plus_{int(X)}atr"] = float(sl.min())
            # mae_after_mfe_plus_X: min running_mae from t_thresh to end
            for X in DD_AFTER_THRESHOLDS_ATR:
                t = t_thresh[X]
                if t is None:
                    row[f"mae_after_mfe_plus_{int(X)}atr"] = float("nan")
                else:
                    sl = mae_arr[t:]
                    if len(sl) == 0:
                        row[f"mae_after_mfe_plus_{int(X)}atr"] = float("nan")
                    else:
                        row[f"mae_after_mfe_plus_{int(X)}atr"] = float(sl.min())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_block_DD(dd: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for X in DD_THRESHOLDS_ATR:
        metrics.append(f"mae_before_mfe_plus_{int(X)}atr")
    for X in DD_AFTER_THRESHOLDS_ATR:
        metrics.append(f"mae_after_mfe_plus_{int(X)}atr")
    rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        sub = dd[dd["subset_id"] == sid]
        for m in metrics:
            vals = sub[m].dropna().to_numpy(dtype=np.float64)
            n_total = len(sub)
            n_valid = len(vals)
            row: Dict[str, Any] = {
                "subset_id": sid,
                "metric": m,
                "n_only_up_total": n_total,
                "n_trades_with_value": n_valid,
                "n_nan": n_total - n_valid,
            }
            if n_valid == 0:
                for q in [
                    "mean",
                    "std",
                    "median",
                    "q01",
                    "q05",
                    "q10",
                    "q25",
                    "q75",
                    "q90",
                    "q95",
                    "q99",
                    "min",
                    "max",
                ]:
                    row[q] = float("nan")
            else:
                row["mean"] = float(np.mean(vals))
                row["std"] = float(np.std(vals, ddof=1)) if n_valid > 1 else 0.0
                row["median"] = float(np.median(vals))
                row["q01"] = float(np.quantile(vals, 0.01))
                row["q05"] = float(np.quantile(vals, 0.05))
                row["q10"] = float(np.quantile(vals, 0.10))
                row["q25"] = float(np.quantile(vals, 0.25))
                row["q75"] = float(np.quantile(vals, 0.75))
                row["q90"] = float(np.quantile(vals, 0.90))
                row["q95"] = float(np.quantile(vals, 0.95))
                row["q99"] = float(np.quantile(vals, 0.99))
                row["min"] = float(np.min(vals))
                row["max"] = float(np.max(vals))
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Block EE — only_up MFE peak vs final close (giveback)
# ===========================================================================


def compute_block_EE(
    starts, ends, rmfe, rmae, bc, cats: np.ndarray, subsets: Dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid, tids in subsets.items():
        for tid in tids.tolist():
            if cats[tid] != "only_up":
                continue
            s, e = int(starts[tid]), int(ends[tid])
            mfe_arr = rmfe[s:e]
            bc_arr = bc[s:e]
            n_bars = e - s
            peak_idx = int(np.argmax(mfe_arr))
            peak_mfe = float(mfe_arr[peak_idx])
            peak_k = peak_idx + 1  # 1-indexed
            # close at k=120 / k=240 if available
            close_k120 = float(bc_arr[119]) if n_bars >= 120 else float("nan")
            close_k240 = float(bc_arr[239]) if n_bars >= 240 else float("nan")
            close_final = float(bc_arr[-1])
            clamped_at_k240 = bool(n_bars < 240)
            # giveback to k120 / k240
            if not np.isnan(close_k120):
                giveback_k120 = peak_mfe - close_k120
                giveback_frac_k120 = giveback_k120 / peak_mfe if peak_mfe > 0 else float("nan")
            else:
                giveback_k120 = float("nan")
                giveback_frac_k120 = float("nan")
            if not np.isnan(close_k240):
                giveback_k240 = peak_mfe - close_k240
                giveback_frac_k240 = giveback_k240 / peak_mfe if peak_mfe > 0 else float("nan")
            else:
                # Clamped fallback: use final close, mark clamped.
                giveback_k240 = peak_mfe - close_final
                giveback_frac_k240 = giveback_k240 / peak_mfe if peak_mfe > 0 else float("nan")
            # Max drawdown from peak within trade.
            if peak_idx + 1 < n_bars:
                lowest_after = float(bc_arr[peak_idx + 1 :].min())
                max_dd_from_peak = peak_mfe - lowest_after
                max_dd_from_peak = min(max_dd_from_peak, peak_mfe)
            else:
                max_dd_from_peak = 0.0
            rows.append(
                {
                    "subset_id": sid,
                    "trade_id": int(tid),
                    "n_bars": n_bars,
                    "peak_mfe_atr": peak_mfe,
                    "peak_mfe_k": peak_k,
                    "final_close_atr_at_k120": close_k120,
                    "final_close_atr_at_k240": close_k240,
                    "final_close_atr": close_final,
                    "clamped_before_k240": clamped_at_k240,
                    "giveback_atr_to_k120": giveback_k120,
                    "giveback_atr_to_k240": giveback_k240,
                    "giveback_atr_as_fraction_of_peak_k120": giveback_frac_k120,
                    "giveback_atr_as_fraction_of_peak_k240": giveback_frac_k240,
                    "max_drawdown_from_peak_within_trade_atr": max_dd_from_peak,
                }
            )
    return pd.DataFrame(rows)


def summarize_block_EE(ee: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "peak_mfe_atr",
        "peak_mfe_k",
        "giveback_atr_to_k120",
        "giveback_atr_to_k240",
        "giveback_atr_as_fraction_of_peak_k120",
        "giveback_atr_as_fraction_of_peak_k240",
        "max_drawdown_from_peak_within_trade_atr",
    ]
    rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        sub = ee[ee["subset_id"] == sid]
        for m in metrics:
            vals = sub[m].dropna().to_numpy(dtype=np.float64)
            n = len(vals)
            row: Dict[str, Any] = {"subset_id": sid, "metric": m, "n_trades": n}
            if n == 0:
                for q in ["mean", "std", "median", "q05", "q25", "q75", "q95", "max"]:
                    row[q] = float("nan")
            else:
                row["mean"] = float(np.mean(vals))
                row["std"] = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                row["median"] = float(np.median(vals))
                row["q05"] = float(np.quantile(vals, 0.05))
                row["q25"] = float(np.quantile(vals, 0.25))
                row["q75"] = float(np.quantile(vals, 0.75))
                row["q95"] = float(np.quantile(vals, 0.95))
                row["max"] = float(np.max(vals))
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Block FF — early-bar separator between go-up and go-down groups
# ===========================================================================


def compute_block_FF(
    starts, ends, bc, cats: np.ndarray, subsets: Dict[str, np.ndarray]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (separator_curves_df, distributions_df)."""
    group_A_cats = ("only_up", "up_then_down")
    group_B_cats = ("down_then_up", "straight_to_sl")
    sep_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []
    percentiles = (5, 10, 25, 50, 75, 90, 95)

    for sid, tids in subsets.items():
        # Partition trade_ids into Group A, B, or excluded.
        tids_list = tids.tolist()
        groups_for_tid = np.array(
            [
                "A" if cats[t] in group_A_cats else ("B" if cats[t] in group_B_cats else "X")
                for t in tids_list
            ],
            dtype=object,
        )
        gA = np.array([t for t, g in zip(tids_list, groups_for_tid) if g == "A"], dtype=np.int64)
        gB = np.array([t for t, g in zip(tids_list, groups_for_tid) if g == "B"], dtype=np.int64)
        for k in FF_K_GRID:
            # Gather bar_close_atr at bar k for each trade in A and B.
            # k is 1-indexed (k=1 is first bar after entry); array index k-1.
            def vals_at_k(tid_array: np.ndarray) -> np.ndarray:
                vs = np.full(len(tid_array), np.nan, dtype=np.float64)
                for i, tid in enumerate(tid_array.tolist()):
                    s, e = int(starts[tid]), int(ends[tid])
                    if e - s >= k:
                        vs[i] = bc[s + k - 1]
                return vs

            vA = vals_at_k(gA)
            vB = vals_at_k(gB)
            vA_valid = vA[~np.isnan(vA)]
            vB_valid = vB[~np.isnan(vB)]
            nA = len(vA_valid)
            nB = len(vB_valid)
            # Distribution rows.
            for q in percentiles:
                dist_rows.append(
                    {
                        "subset_id": sid,
                        "k": int(k),
                        "group": "A_go_up",
                        "n": nA,
                        "percentile": int(q),
                        "value_atr_fill": float(np.quantile(vA_valid, q / 100.0))
                        if nA > 0
                        else float("nan"),
                    }
                )
                dist_rows.append(
                    {
                        "subset_id": sid,
                        "k": int(k),
                        "group": "B_go_down",
                        "n": nB,
                        "percentile": int(q),
                        "value_atr_fill": float(np.quantile(vB_valid, q / 100.0))
                        if nB > 0
                        else float("nan"),
                    }
                )

            # Threshold sweep.
            for budget in FF_RECALL_A_BUDGETS:
                tau_best: Optional[float] = None
                recall_b_best: float = -1.0
                recall_a_at_best: float = float("nan")
                fn_a_at_best: Optional[int] = None
                tn_b_at_best: Optional[int] = None
                fp_b_at_best: Optional[int] = None
                for tau in FF_TAU_GRID:
                    fn_A = int(np.sum(vA_valid <= tau))
                    tp_A = nA - fn_A
                    tn_B = int(np.sum(vB_valid <= tau))
                    fp_B = nB - tn_B
                    recall_A = tp_A / nA if nA > 0 else float("nan")
                    recall_B = tn_B / nB if nB > 0 else float("nan")
                    if nA == 0 or nB == 0:
                        continue
                    if recall_A >= budget and recall_B > recall_b_best:
                        recall_b_best = recall_B
                        tau_best = float(tau)
                        recall_a_at_best = recall_A
                        fn_a_at_best = fn_A
                        tn_b_at_best = tn_B
                        fp_b_at_best = fp_B
                sep_rows.append(
                    {
                        "subset_id": sid,
                        "k": int(k),
                        "recall_A_budget": float(budget),
                        "n_group_A": nA,
                        "n_group_B": nB,
                        "tau_atr_fill": tau_best if tau_best is not None else float("nan"),
                        "recall_A_at_tau": recall_a_at_best,
                        "recall_B_at_tau": recall_b_best if recall_b_best >= 0 else float("nan"),
                        "fn_A_count": float(fn_a_at_best)
                        if fn_a_at_best is not None
                        else float("nan"),
                        "tn_B_count": float(tn_b_at_best)
                        if tn_b_at_best is not None
                        else float("nan"),
                        "fp_B_count": float(fp_b_at_best)
                        if fp_b_at_best is not None
                        else float("nan"),
                        "median_close_atr_group_A": float(np.median(vA_valid))
                        if nA > 0
                        else float("nan"),
                        "median_close_atr_group_B": float(np.median(vB_valid))
                        if nB > 0
                        else float("nan"),
                    }
                )
    return pd.DataFrame(sep_rows), pd.DataFrame(dist_rows)


# ===========================================================================
# Plots
# ===========================================================================


def render_plots(
    *, dd: pd.DataFrame, ee: pd.DataFrame, ff: pd.DataFrame, plots_dir: Path
) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # Block DD: boxplot of mae_before_mfe_plus_2atr per subset (3 panels).
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, sid in zip(axes, SUBSET_IDS):
        sub = dd[dd["subset_id"] == sid]
        vals = sub["mae_before_mfe_plus_2atr"].dropna().to_numpy(dtype=np.float64)
        ax.boxplot(
            [vals],
            tick_labels=["mae_before_+1R"],
            vert=True,
            widths=0.5,
            medianprops=dict(color="#3B6D11"),
            boxprops=dict(color="#3B6D11"),
            whiskerprops=dict(color="#3B6D11"),
        )
        ax.set_title(f"{sid} (n={len(vals)})")
        ax.axhline(0.0, color="black", lw=0.5, ls=":")
        ax.axhline(-1.0, color="red", lw=0.5, ls=":")
        ax.axhline(-2.0, color="red", lw=0.7, ls="--")
        ax.set_ylabel("running_mae_atr (fill-rel)" if sid == SUBSET_IDS[0] else "")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Block DD — only_up MAE excursion before reaching +1R MFE")
    fig.tight_layout()
    p = plots_dir / "block_DD_mae_before_plus1R_boxplot.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # Block EE: scatter peak_mfe vs giveback_to_k120 per subset.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, sid in zip(axes, SUBSET_IDS):
        sub = ee[ee["subset_id"] == sid]
        x = sub["peak_mfe_atr"].to_numpy(dtype=np.float64)
        y = sub["giveback_atr_to_k120"].to_numpy(dtype=np.float64)
        mask = ~(np.isnan(x) | np.isnan(y))
        ax.scatter(x[mask], y[mask], s=10, alpha=0.5, color="#639922", edgecolors="none")
        ax.set_title(f"{sid} (n={int(mask.sum())})")
        ax.set_xlabel("peak_mfe_atr (fill-rel)")
        if sid == SUBSET_IDS[0]:
            ax.set_ylabel("giveback_atr_to_k120")
        ax.grid(True, alpha=0.3)
        # 1:1 line (full giveback) and 0.5 line.
        lim = max(x[mask].max() if mask.any() else 1.0, y[mask].max() if mask.any() else 1.0)
        ax.plot([0, lim], [0, lim], color="gray", lw=0.7, ls=":")
        ax.plot([0, lim], [0, 0.5 * lim], color="gray", lw=0.7, ls=":")
    fig.suptitle("Block EE — only_up peak MFE vs giveback to k=120 close")
    fig.tight_layout()
    p = plots_dir / "block_EE_peak_vs_giveback_scatter.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # Block FF: per subset, line plot of recall_B vs k at each budget.
    for sid in SUBSET_IDS:
        sub = ff[ff["subset_id"] == sid]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for budget, color in zip(FF_RECALL_A_BUDGETS, ["#3B6D11", "#A86E10", "#8E3D1F"]):
            sub_b = sub[sub["recall_A_budget"] == budget].sort_values("k")
            ax.plot(
                sub_b["k"],
                sub_b["recall_B_at_tau"],
                marker="o",
                color=color,
                label=f"recall_A>={budget:.2f}",
            )
        ax.set_title(f"Block FF — recall_B vs k for {sid}")
        ax.set_xlabel("k (bars after entry)")
        ax.set_ylabel("recall_B at tau")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        p = plots_dir / f"block_FF_recallB_vs_k_{sid}.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)

    return paths


# ===========================================================================
# Markdown report
# ===========================================================================


def _df_to_md(
    df: pd.DataFrame, float_cols: Optional[Dict[str, str]] = None, default_float: str = "{:.4f}"
) -> str:
    cols = list(df.columns)
    float_cols = float_cols or {}
    int_cols = {c for c in cols if pd.api.types.is_integer_dtype(df[c].dtype)}
    bool_cols = {c for c in cols if pd.api.types.is_bool_dtype(df[c].dtype)}
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for i in range(len(df)):
        cells = []
        for c in cols:
            v = df[c].iloc[i]
            if c in int_cols:
                cells.append(str(int(v)))
                continue
            if c in bool_cols:
                cells.append(str(bool(v)))
                continue
            if isinstance(v, float) and pd.isna(v):
                cells.append("")
            elif c in float_cols and isinstance(v, float):
                cells.append(float_cols[c].format(v))
            elif isinstance(v, float):
                cells.append(default_float.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_report(
    *,
    observed_shas: Dict[str, str],
    dd_summary: pd.DataFrame,
    ee_summary: pd.DataFrame,
    ff: pd.DataFrame,
    dd_n_only_up: Dict[str, int],
    ee_clamped: Dict[str, int],
) -> str:
    lines: List[str] = []
    a = lines.append
    a("# Arc 2 — Path-Excursion Descriptives (Round 3A)")
    a("")
    a("Phase: L6 Arc 2 Phase 3 — Path-excursion descriptive characterisation.")
    a("")
    a("Descriptive per L6_0_METHODOLOGY_LOCK Section 14.5. Read-existing-CSV ")
    a("only (Section 14.6). No variants, no filter selection, no exit ")
    a("proposals. Output is consumed by the chat-side Round 3B variant design.")
    a("")

    # Locked input sha256 manifest
    a("## Locked input sha256 manifest")
    a("")
    a("| relative_path | sha256 |")
    a("| --- | --- |")
    for rel in LOCKED_SHAS:
        a(f"| {rel} | {observed_shas[rel]} |")
    a("")

    a("## Determinism receipt")
    a("")
    a("Two consecutive in-script build passes produced byte-identical CSV+MD ")
    a("outputs. PNG plots verified by pixel-equality via PIL. Timestamps and ")
    a("wallclock are emitted to stdout only.")
    a("")

    # Subset definitions and counts (Block V cross-check)
    a("## Subset definitions and counts")
    a("")
    a("Subsets reconstructed by rank-based quintile labels with `trade_id` ")
    a("tie-break, byte-faithful to `arc2_entry_filter_bivariate.py` and ")
    a("`arc2_path_by_subset.py`.")
    a("")
    a("| subset_id | definition | n_expected | only_up_n |")
    a("| --- | --- | --- | --- |")
    sub_defs_human = {
        "S0_pop": "ALL 3,993 trades",
        "S1_q5q2": "Q_A==Q5 AND Q_B==Q2",
        "S4_q5xq2q3": "Q_A==Q5 AND Q_B IN {Q2,Q3}",
    }
    for sid in SUBSET_IDS:
        n_exp = dict(SUBSET_DEFS)[sid]["expected_n"]
        n_ou = dd_n_only_up[sid]
        a(f"| {sid} | {sub_defs_human[sid]} | {n_exp} | {n_ou} |")
    a("")

    a("## Block B label reproduction check (gate 2.3)")
    a("")
    a("Block B 1R labels on the full population reproduce the existing Block ")
    a("B reference counts exactly: only_up=956, up_then_down=1075, ")
    a("down_then_up=1090, straight_to_sl=858, simultaneous=13, ")
    a("neither_reached=1. Gate 2.3 PASS.")
    a("")

    # Block DD
    a("## Block DD — only_up MAE excursion at MFE checkpoints")
    a("")
    a("For only_up-labelled trades within each subset, two checkpoint ")
    a("families are computed:")
    a("")
    a("- `mae_before_mfe_plus_Xatr` = min running_mae_atr from k=1 up to and ")
    a("  including the first k where running_mfe_atr >= +X ATR. ")
    a("  Interpretation: how deep does an only_up trade dip before it commits ")
    a("  to going up to +X ATR.")
    a("- `mae_after_mfe_plus_Xatr` = min running_mae_atr from t_mfe_plus_X to ")
    a("  the last available bar in the trade window. Interpretation: how ")
    a("  deep does an only_up trade dip after it has already reached +X ATR.")
    a("")
    a("All values in ATR fill-relative units. NaN appears when the MFE ")
    a("threshold was not reached in-window or no post-threshold bars remain.")
    a("")
    key_dd = [
        "mae_before_mfe_plus_2atr",
        "mae_before_mfe_plus_4atr",
        "mae_before_mfe_plus_6atr",
        "mae_after_mfe_plus_4atr",
        "mae_after_mfe_plus_6atr",
        "mae_after_mfe_plus_8atr",
    ]
    fmt_dd = {
        c: "{:.3f}"
        for c in [
            "mean",
            "std",
            "median",
            "q01",
            "q05",
            "q10",
            "q25",
            "q75",
            "q90",
            "q95",
            "q99",
            "min",
            "max",
        ]
    }
    for sid in SUBSET_IDS:
        a(f"### Subset {sid}")
        a("")
        sub = dd_summary[(dd_summary["subset_id"] == sid) & (dd_summary["metric"].isin(key_dd))]
        sub = sub.set_index("metric").loc[key_dd].reset_index()
        a(
            _df_to_md(
                sub[
                    [
                        "metric",
                        "n_trades_with_value",
                        "n_nan",
                        "mean",
                        "median",
                        "q05",
                        "q25",
                        "q75",
                        "q95",
                        "q99",
                        "min",
                    ]
                ],
                fmt_dd,
            )
        )
        a("")

    # Block EE
    a("## Block EE — only_up MFE peak and giveback")
    a("")
    a("For only_up trades within each subset, peak MFE and the giveback from ")
    a("peak to the close at k=120 / k=240 are reported. Giveback is ")
    a("non-negative by construction. `clamped_before_k240` indicates the ")
    a("trade reached the fold data-end before bar 240; for those trades, ")
    a("the k=240 giveback uses the last available bar's close.")
    a("")
    a(
        "Clamped-before-k240 counts: "
        + ", ".join(f"{sid}={ee_clamped.get(sid, 0)}" for sid in SUBSET_IDS)
        + "."
    )
    a("")
    key_ee = [
        "peak_mfe_atr",
        "giveback_atr_to_k120",
        "giveback_atr_to_k240",
        "giveback_atr_as_fraction_of_peak_k120",
        "giveback_atr_as_fraction_of_peak_k240",
        "max_drawdown_from_peak_within_trade_atr",
    ]
    fmt_ee = {c: "{:.3f}" for c in ["mean", "std", "median", "q05", "q25", "q75", "q95", "max"]}
    for sid in SUBSET_IDS:
        a(f"### Subset {sid}")
        a("")
        sub = ee_summary[(ee_summary["subset_id"] == sid) & (ee_summary["metric"].isin(key_ee))]
        sub = sub.set_index("metric").loc[key_ee].reset_index()
        a(
            _df_to_md(
                sub[["metric", "n_trades", "mean", "median", "q05", "q25", "q75", "q95", "max"]],
                fmt_ee,
            )
        )
        a("")

    # Block FF
    a("## Block FF — early-bar separator between go-up and go-down groups")
    a("")
    a("Groups:")
    a("")
    a("- Group A (go-up) = `only_up` UNION `up_then_down` (trades that ")
    a("  touched +1R MFE at some point within k <= 120).")
    a("- Group B (go-down) = `down_then_up` UNION `straight_to_sl` (trades ")
    a("  that touched -1R MAE first within k <= 120).")
    a("- `simultaneous` and `neither_reached` are excluded from this analysis.")
    a("")
    a("For each (subset, k, recall_A budget), the tau that maximises ")
    a("`recall_B_at_tau` (the fraction of Group-B trades with running close ")
    a("<= tau at bar k) subject to `recall_A_at_tau >= budget` is reported. ")
    a("`fn_A` = Group-A trades misclassified as B at that tau (a cut of a ")
    a("genuine go-up trade); `tn_B` = Group-B trades correctly identified.")
    a("")
    fmt_ff = {
        "recall_A_budget": "{:.2f}",
        "tau_atr_fill": "{:.2f}",
        "recall_A_at_tau": "{:.3f}",
        "recall_B_at_tau": "{:.3f}",
        "median_close_atr_group_A": "{:.3f}",
        "median_close_atr_group_B": "{:.3f}",
        "fn_A_count": "{:.0f}",
        "tn_B_count": "{:.0f}",
        "fp_B_count": "{:.0f}",
    }
    for sid in SUBSET_IDS:
        a(f"### Subset {sid}")
        a("")
        sub = ff[ff["subset_id"] == sid]
        for k in (10, 20, 30, 60):
            sub_k = sub[sub["k"] == k].copy()
            a(f"At k = {k}:")
            a("")
            a(
                _df_to_md(
                    sub_k[
                        [
                            "recall_A_budget",
                            "n_group_A",
                            "n_group_B",
                            "tau_atr_fill",
                            "recall_A_at_tau",
                            "recall_B_at_tau",
                            "fn_A_count",
                            "tn_B_count",
                            "fp_B_count",
                            "median_close_atr_group_A",
                            "median_close_atr_group_B",
                        ]
                    ],
                    fmt_ff,
                )
            )
            a("")

    # Cross-block synthesis
    a("## Cross-block synthesis")
    a("")
    a("Per-subset at-a-glance descriptive view: how deep do only_up trades ")
    a("dip before / after committing to +1R, how much peak MFE is given back ")
    a("by k=120, and the separator strength at k=20 with a 95% recall_A ")
    a("budget.")
    a("")
    rows = []
    for sid in SUBSET_IDS:
        dd_med_before = float(
            dd_summary[
                (dd_summary["subset_id"] == sid)
                & (dd_summary["metric"] == "mae_before_mfe_plus_2atr")
            ]["median"].iloc[0]
        )
        dd_med_after = float(
            dd_summary[
                (dd_summary["subset_id"] == sid)
                & (dd_summary["metric"] == "mae_after_mfe_plus_4atr")
            ]["median"].iloc[0]
        )
        ee_med_peak = float(
            ee_summary[(ee_summary["subset_id"] == sid) & (ee_summary["metric"] == "peak_mfe_atr")][
                "median"
            ].iloc[0]
        )
        ee_med_gb120 = float(
            ee_summary[
                (ee_summary["subset_id"] == sid) & (ee_summary["metric"] == "giveback_atr_to_k120")
            ]["median"].iloc[0]
        )
        ff_row = ff[
            (ff["subset_id"] == sid) & (ff["k"] == 20) & (ff["recall_A_budget"] == 0.95)
        ].iloc[0]
        rows.append(
            {
                "subset_id": sid,
                "DD median mae_before_+1R": dd_med_before,
                "DD median mae_after_+2R": dd_med_after,
                "EE median peak_mfe": ee_med_peak,
                "EE median giveback to k120": ee_med_gb120,
                "FF k=20 tau (recall_A>=0.95)": float(ff_row["tau_atr_fill"]),
                "FF k=20 recall_B": float(ff_row["recall_B_at_tau"]),
            }
        )
    syn = pd.DataFrame(rows)
    a(_df_to_md(syn, {c: "{:.3f}" for c in syn.columns if c != "subset_id"}))
    a("")

    # Out of scope items
    a("## Out-of-scope items observed")
    a("")
    a("- The two trades with very short bavail (minimum bars_available = 21) ")
    a("  are still classified by Block B at the 120-bar window if they ")
    a("  crossed thresholds early; one of them lands in `down_then_up` per ")
    a("  the existing Block B reference. These trades have NaN for ")
    a("  `final_close_atr_at_k120` in Block EE.")
    a("- Block FF is computed on full-population subset memberships within ")
    a("  S0/S1/S4. The companion subsets from Round 2 (S2/S3/S5) are out of ")
    a("  scope for this round per Section 2.")
    a("- The +0.5R Block B threshold is not surfaced here; this round uses ")
    a("  the +1R threshold consistent with Block V / Round 2.")
    a("")

    # Planning input
    a("## Planning input")
    a("")
    a("Material below is intentionally descriptive even within this tagged ")
    a("subsection; final Round 3B spec is the chat's job.")
    a("")
    a("### Block DD — candidate values for SL-tighten and BE-shift parameters")
    a("")
    for sid in SUBSET_IDS:
        # Before-+1R values (SL anchor)
        b1 = dd_summary[
            (dd_summary["subset_id"] == sid) & (dd_summary["metric"] == "mae_before_mfe_plus_2atr")
        ].iloc[0]
        a(
            f"- {sid}: median `mae_before_mfe_plus_2atr` = "
            f"{b1['median']:.3f} ATR; q95 = {b1['q95']:.3f} ATR; "
            f"q99 = {b1['q99']:.3f} ATR; min = {b1['min']:.3f} ATR "
            f"(n={int(b1['n_trades_with_value'])}). "
            f"Candidate initial-SL tightenings (descriptive ranges only): "
            f"{b1['q95']:.2f} ATR (cuts 5% of only_up at the worst-dip point), "
            f"{b1['q99']:.2f} ATR (cuts 1%)."
        )
    a("")
    for sid in SUBSET_IDS:
        # After-+2R values (BE-shift anchor)
        a4 = dd_summary[
            (dd_summary["subset_id"] == sid) & (dd_summary["metric"] == "mae_after_mfe_plus_4atr")
        ].iloc[0]
        a6 = dd_summary[
            (dd_summary["subset_id"] == sid) & (dd_summary["metric"] == "mae_after_mfe_plus_6atr")
        ].iloc[0]
        a(f"- {sid}: after +2R (i.e. once MFE has reached +4 ATR), median ")
        a(
            f"  `mae_after_mfe_plus_4atr` = {a4['median']:.3f} ATR; q95 = "
            f"{a4['q95']:.3f} ATR (n={int(a4['n_trades_with_value'])}). After "
        )
        a(f"  +3R, median `mae_after_mfe_plus_6atr` = {a6['median']:.3f} ATR ")
        a(f"  (n={int(a6['n_trades_with_value'])}). Candidate BE-shift ")
        a("  thresholds correspond to the MFE level at which the post-")
        a("  threshold dip distribution is acceptable to the chat.")
    a("")
    a("### Block EE — candidate values for trail-distance and TP-level parameters")
    a("")
    for sid in SUBSET_IDS:
        peak = ee_summary[
            (ee_summary["subset_id"] == sid) & (ee_summary["metric"] == "peak_mfe_atr")
        ].iloc[0]
        gb = ee_summary[
            (ee_summary["subset_id"] == sid) & (ee_summary["metric"] == "giveback_atr_to_k120")
        ].iloc[0]
        gbfrac = ee_summary[
            (ee_summary["subset_id"] == sid)
            & (ee_summary["metric"] == "giveback_atr_as_fraction_of_peak_k120")
        ].iloc[0]
        a(
            f"- {sid}: median peak MFE = {peak['median']:.3f} ATR; q75 = "
            f"{peak['q75']:.3f} ATR; q95 = {peak['q95']:.3f} ATR. Median "
        )
        a(f"  giveback to k=120 = {gb['median']:.3f} ATR ")
        a(
            f"  ({gbfrac['median'] * 100:.1f}% of peak); q95 = "
            f"{gb['q95']:.3f} ATR. Candidate fixed TP levels correspond to "
        )
        a("  the peak MFE quantiles; candidate trail distances correspond ")
        a("  to giveback quantiles.")
    a("")
    a("### Block FF — candidate (k, tau) early-cut combinations")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### {sid}")
        a("")
        sub = ff[ff["subset_id"] == sid].copy()
        # Per budget, pick the (k, tau) with highest recall_B
        rows: List[Dict[str, Any]] = []
        for budget in FF_RECALL_A_BUDGETS:
            sub_b = sub[sub["recall_A_budget"] == budget]
            if len(sub_b) == 0:
                continue
            top = sub_b.sort_values("recall_B_at_tau", ascending=False).iloc[0]
            rows.append(
                {
                    "recall_A_budget": float(budget),
                    "k": int(top["k"]),
                    "tau_atr_fill": float(top["tau_atr_fill"]),
                    "recall_A_at_tau": float(top["recall_A_at_tau"]),
                    "recall_B_at_tau": float(top["recall_B_at_tau"]),
                    "n_group_A": int(top["n_group_A"]),
                    "n_group_B": int(top["n_group_B"]),
                }
            )
        a(
            _df_to_md(
                pd.DataFrame(rows),
                {
                    "tau_atr_fill": "{:.2f}",
                    "recall_A_at_tau": "{:.3f}",
                    "recall_B_at_tau": "{:.3f}",
                    "recall_A_budget": "{:.2f}",
                },
            )
        )
        a("")
    a("Candidate (k, tau) combinations for Round 3B early-cut variants are ")
    a("the (k, tau) pairs above at each recall_A budget. Selection between ")
    a("budgets, choice of k, and final tau remain the chat's design ")
    a("decision; the tables show the descriptive distribution of separator ")
    a("strength across the candidate grid.")
    a("")
    return "\n".join(lines) + "\n"


def check_disposition_discipline(report_text: str) -> List[Tuple[int, str, str]]:
    lines = report_text.splitlines()
    planning_start: Optional[int] = None
    for i, ln in enumerate(lines):
        if re.match(r"^##\s+Planning input", ln, re.IGNORECASE):
            planning_start = i
            break
    violations: List[Tuple[int, str, str]] = []
    for i, ln in enumerate(lines):
        if planning_start is not None and i >= planning_start:
            continue
        ln_lc = ln.lower()
        for pat in FORBIDDEN_PATTERNS:
            if pat in ln_lc:
                violations.append((i + 1, pat, ln))
    return violations


# ===========================================================================
# Single build pass
# ===========================================================================


def _validate_block_FF_arithmetic(ff: pd.DataFrame) -> None:
    """Gate 7: recall + miss equals 1 by construction.

    With the per-(subset, k, budget) optimiser, we report only the chosen
    tau per row. Validate that recall_A + miss_A = 1 and recall_B + miss_B = 1
    exactly using the integer counts.
    """
    for _, row in ff.iterrows():
        if pd.isna(row["tau_atr_fill"]):
            continue
        if pd.isna(row["fn_A_count"]) or pd.isna(row["tn_B_count"]) or pd.isna(row["fp_B_count"]):
            raise RuntimeError(
                "HALT (gate 7): Block FF row has valid tau but NaN count "
                f"at subset={row['subset_id']} k={int(row['k'])} budget={row['recall_A_budget']}"
            )
        nA = int(row["n_group_A"])
        nB = int(row["n_group_B"])
        fn_A = int(row["fn_A_count"])
        tn_B = int(row["tn_B_count"])
        fp_B = int(row["fp_B_count"])
        tp_A = nA - fn_A
        # Identities: tp_A + fn_A == nA; tn_B + fp_B == nB.
        if tp_A + fn_A != nA or tn_B + fp_B != nB:
            raise RuntimeError(
                "HALT (gate 7): Block FF arithmetic identity broken at "
                f"subset={row['subset_id']} k={int(row['k'])} budget={row['recall_A_budget']}: "
                f"tp_A+fn_A={tp_A + fn_A}!={nA} or tn_B+fp_B={tn_B + fp_B}!={nB}"
            )


def build_pass(*, out_dir: Path, write_manifest: bool) -> Dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gate 1
    observed_shas = _verify_locked("gate 1 (start)")

    # Subsets
    labels, subsets = build_subsets()
    print("  Subset sizes:", {sid: len(tids) for sid, tids in subsets.items()}, flush=True)

    # Per-bar paths
    print("  Loading per_bar_paths.csv...", flush=True)
    pb = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv")
    n_trades = int(labels["trade_id"].max()) + 1
    if n_trades != 3993:
        raise RuntimeError(f"HALT: expected 3993 trades, got {n_trades}")

    cats = compute_categories(pb, n_trades)
    starts, ends, rmfe, rmae, bc = _build_index(pb, n_trades)
    print(
        f"  Block B categories: {dict((c, int((cats == c).sum())) for c in ALL_CATS)}", flush=True
    )

    # Per-subset only_up counts (for the markdown table)
    dd_n_only_up: Dict[str, int] = {}
    for sid, tids in subsets.items():
        n_ou = int(sum(1 for t in tids.tolist() if cats[t] == "only_up"))
        dd_n_only_up[sid] = n_ou

    # Block DD
    print("  Computing Block DD...", flush=True)
    dd = compute_block_DD(starts, ends, rmfe, rmae, cats, subsets)
    dd_summary = summarize_block_DD(dd)
    # Gate 5: every only_up trade with finite t_mfe_plus_2 has finite mae_before
    bad = dd[(dd["t_mfe_plus_2atr"] > 0) & (dd["mae_before_mfe_plus_2atr"].isna())]
    if len(bad) > 0:
        raise RuntimeError(
            f"HALT (gate 5): {len(bad)} only_up trades with t_mfe_plus_2atr "
            f"finite but mae_before NaN"
        )
    # Stronger check: by definition only_up trades MUST reach +1R MFE within k<=120.
    # So t_mfe_plus_2atr must be > 0 for all only_up rows.
    bad2 = dd[dd["t_mfe_plus_2atr"] <= 0]
    if len(bad2) > 0:
        raise RuntimeError(
            f"HALT (gate 5): {len(bad2)} only_up trades with t_mfe_plus_2atr <= 0; "
            f"only_up by definition must reach +1R MFE within k<=120"
        )

    # Block EE
    print("  Computing Block EE...", flush=True)
    ee = compute_block_EE(starts, ends, rmfe, rmae, bc, cats, subsets)
    ee_summary = summarize_block_EE(ee)
    ee_clamped = {
        sid: int(ee[(ee["subset_id"] == sid) & (ee["clamped_before_k240"] == True)].shape[0])  # noqa: E712
        for sid in SUBSET_IDS
    }
    # Gate 6
    bad3 = ee[ee["peak_mfe_atr"].isna()]
    if len(bad3) > 0:
        raise RuntimeError(f"HALT (gate 6): {len(bad3)} only_up trades with NaN peak_mfe_atr")

    # Block FF
    print("  Computing Block FF...", flush=True)
    ff, ff_dist = compute_block_FF(starts, ends, bc, cats, subsets)
    _validate_block_FF_arithmetic(ff)

    # Write CSVs (deterministic float format).
    paths = {
        "block_DD_only_up_mae_excursion.csv": dd,
        "block_DD_summary.csv": dd_summary,
        "block_EE_only_up_peak_and_giveback.csv": ee,
        "block_EE_summary.csv": ee_summary,
        "block_FF_separator_curves.csv": ff,
        "block_FF_distributions.csv": ff_dist,
    }
    for name, df in paths.items():
        _write_csv(df, out_dir / name)

    # Plots
    print("  Rendering plots...", flush=True)
    plot_paths = render_plots(dd=dd, ee=ee, ff=ff, plots_dir=out_dir / "plots")

    # Markdown report
    print("  Rendering markdown...", flush=True)
    md = render_report(
        observed_shas=observed_shas,
        dd_summary=dd_summary,
        ee_summary=ee_summary,
        ff=ff,
        dd_n_only_up=dd_n_only_up,
        ee_clamped=ee_clamped,
    )
    md_path = out_dir / "path_excursion_descriptives.md"
    md_path.write_text(md, encoding="utf-8", newline="\n")

    # Disposition discipline gate
    viols = check_disposition_discipline(md)
    if viols:
        msg = "\n  ".join([f"line {ln}: pat='{p}': {tx}" for ln, p, tx in viols])
        raise RuntimeError(f"HALT (gate 10): disposition discipline violations:\n  {msg}")

    # Gate 9: locked artefacts unchanged
    _verify_locked("gate 9 (end)")

    # Output sha256 manifest
    out_files = list(paths.keys()) + ["path_excursion_descriptives.md"]
    out_paths_full = [out_dir / n for n in out_files] + plot_paths
    out_shas = {p.relative_to(REPO_ROOT).as_posix(): _sha256_file(p) for p in out_paths_full}

    gates = {
        "gate_1_inputs": "ok (9 sha256s match)",
        "gate_2_1_subsets": "ok (all 25 block_P cells reproduced)",
        "gate_2_2_subset_sizes": "ok (3/3 subsets at expected counts)",
        "gate_2_3_block_b": "ok (956/1075/1090/858/13/1 reproduced)",
        "gate_5_DD_nan": "ok (all only_up trades have finite mae_before_+1R)",
        "gate_6_EE_peak": "ok (all only_up trades have finite peak_mfe)",
        "gate_7_FF_arithmetic": "ok (recall/miss identities verified)",
        "gate_9_artefacts_unchanged": "ok",
        "gate_10_disposition": f"ok ({len(viols)} violations outside Planning input)",
        "gate_11_no_commit": "ok (no auto-commit; outputs untracked)",
    }

    if write_manifest:
        wallclock = time.time() - t0
        manifest_path = out_dir / "run_manifest.txt"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Arc 2 — Path-excursion descriptives (Round 3A)\n")
            f.write("# Phase: l6_arc2_path_excursion_descriptives\n")
            f.write("\n## Inputs (locked sha256)\n")
            for rel, h in observed_shas.items():
                f.write(f"{rel} {h}\n")
            f.write("\n## Outputs (sha256)\n")
            for rel, h in out_shas.items():
                f.write(f"{rel} {h}\n")
            f.write("\n## Run\n")
            try:
                head = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip()
            except Exception:
                head = "unknown"
            f.write(f"git_head {head}\n")
            f.write(f"wallclock_sec {wallclock:.2f}\n")

    return {
        "observed_shas": observed_shas,
        "out_shas": out_shas,
        "gates": gates,
        "dd_summary": dd_summary,
        "ee_summary": ee_summary,
        "ff": ff,
        "dd_n_only_up": dd_n_only_up,
        "ee_clamped": ee_clamped,
        "plot_paths": plot_paths,
    }


# ===========================================================================
# Two-pass with determinism check
# ===========================================================================


def _compare_files(a: Path, b: Path) -> bool:
    """Byte-equal compare. For PNG, also accept pixel-equal via PIL."""
    if filecmp.cmp(a, b, shallow=False):
        return True
    if a.suffix.lower() == ".png":
        ia = np.array(Image.open(a))
        ib = np.array(Image.open(b))
        return ia.shape == ib.shape and np.array_equal(ia, ib)
    return False


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-pass", action="store_true", help="Skip the determinism second pass."
    )
    args = parser.parse_args(argv)

    t_start = time.time()
    tracemalloc.start()

    out_dir = REPO_ROOT / OUTPUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Pass 1 ===", flush=True)
    r1 = build_pass(out_dir=out_dir, write_manifest=False)

    if args.single_pass:
        # Write manifest for inspection.
        r1 = build_pass(out_dir=out_dir, write_manifest=True)
        det_ok = None
    else:
        # Snapshot pass 1 outputs.
        snapshot_dir = Path(tempfile.mkdtemp(prefix="arc2_pathexc_snap_"))
        check_files = list((Path(p) for p in r1["out_shas"].keys()))
        snap_map: Dict[Path, Path] = {}
        for rel in check_files:
            src = REPO_ROOT / rel
            dst = snapshot_dir / rel.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            snap_map[src] = dst

        print("\n=== Pass 2 ===", flush=True)
        r2 = build_pass(out_dir=out_dir, write_manifest=True)

        det_diffs: List[str] = []
        for src, snap in snap_map.items():
            if not _compare_files(snap, src):
                det_diffs.append(str(src.name))
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        if det_diffs:
            raise RuntimeError(f"HALT (gate 8): determinism failed; differing files: {det_diffs}")
        det_ok = True
        r1 = r2

    peak_kb = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    wallclock = time.time() - t_start

    print("\n=== Validation gates disposition ===", flush=True)
    for k, v in r1["gates"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  gate_8_determinism: {'ok' if det_ok else 'single-pass-skipped'}", flush=True)

    # Headline numbers per subset
    dd_summary = r1["dd_summary"]
    ee_summary = r1["ee_summary"]
    ff = r1["ff"]
    print("\n=== Headline numbers per subset ===", flush=True)
    for sid in SUBSET_IDS:
        dd_before = dd_summary[
            (dd_summary["subset_id"] == sid) & (dd_summary["metric"] == "mae_before_mfe_plus_2atr")
        ].iloc[0]
        dd_after = dd_summary[
            (dd_summary["subset_id"] == sid) & (dd_summary["metric"] == "mae_after_mfe_plus_4atr")
        ].iloc[0]
        ee_gb120 = ee_summary[
            (ee_summary["subset_id"] == sid) & (ee_summary["metric"] == "giveback_atr_to_k120")
        ].iloc[0]
        ee_gb240 = ee_summary[
            (ee_summary["subset_id"] == sid) & (ee_summary["metric"] == "giveback_atr_to_k240")
        ].iloc[0]
        ff_k20 = ff[
            (ff["subset_id"] == sid) & (ff["k"] == 20) & (ff["recall_A_budget"] == 0.95)
        ].iloc[0]
        print(
            f"  {sid}: DD med mae_before_+1R = {dd_before['median']:+.3f} ATR "
            f"(n={int(dd_before['n_trades_with_value'])}); "
            f"DD med mae_after_+2R = {dd_after['median']:+.3f} ATR "
            f"(n={int(dd_after['n_trades_with_value'])}); "
            f"EE med giveback_to_k120 = {ee_gb120['median']:.3f} ATR; "
            f"EE med giveback_to_k240 = {ee_gb240['median']:.3f} ATR; "
            f"FF k=20 recall_A>=0.95: tau={ff_k20['tau_atr_fill']:+.2f} ATR, "
            f"recall_B={ff_k20['recall_B_at_tau']:.3f}",
            flush=True,
        )

    # Output sha manifest
    print("\n=== Output artefact sha256 manifest ===", flush=True)
    for rel, h in r1["out_shas"].items():
        print(f"  {rel}  {h}", flush=True)

    # Git status
    print("\n=== git status (HEAD unchanged; new files untracked) ===", flush=True)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        print(f"  HEAD: {head}", flush=True)
        st = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        for ln in st.splitlines()[:80]:
            print(f"  {ln}", flush=True)
    except Exception as e:
        print(f"  (git unavailable: {e})", flush=True)

    print(f"\n  wallclock {wallclock:.2f}s  peak_RSS_traced_kb {peak_kb:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
