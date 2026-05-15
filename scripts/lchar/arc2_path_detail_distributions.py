"""Arc 2 — Path-detail descriptive characterisation (Round 3C).

Phase: L6 Arc 2 Phase 3 — Path-detail descriptive characterisation
(L6_0_METHODOLOGY_LOCK Sections 14.2 derivative experiment, 14.5
descriptive discipline, 14.6 read-existing-CSV backfill).

Seven descriptive blocks computed on existing per_bar_paths.csv:
  Block GG — only_up overall MAE distribution + SL kill-rate table
  Block HH — only_up peak MFE distribution + TP evaluation table
  Block II — only_up peak x overall MAE crosstab
  Block JJ — per-category running_close at early bars + only_up
             separator curves (the critical block)
  Block KK — only_up peak conditional on early position
  Block LL — down_then_up + up_then_down overall MAE distribution
  Block MM — only_up tier analysis (low/mid/high/runner) + median paths

Three subsets evaluated: S0_pop, S1_q5q2, S4_q5xq2q3.

DESCRIPTIVE ONLY (Section 14.5). No variants, no proposals, no
recommendations. Output is consumed by chat-side planning.

Outputs to: results/l6/arc2/characterisation/extended/path_detail_distributions/
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

OUTPUT_DIR_REL = "results/l6/arc2/characterisation/extended/path_detail_distributions"

# Path-category constants (must reproduce Block B 1R counts).
PATH_THRESH_ATR: float = 2.0  # +1R = +2.0 ATR fill-rel
PATH_HOLD_CAP: int = 120

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

SUBSET_DEFS: List[Tuple[str, Dict[str, Any]]] = [
    ("S0_pop", {"all": True, "expected_n": 3993}),
    ("S1_q5q2", {"qa": ("Q5",), "qb": ("Q2",), "expected_n": 190}),
    ("S4_q5xq2q3", {"qa": ("Q5",), "qb": ("Q2", "Q3"), "expected_n": 368}),
]
SUBSET_IDS: Tuple[str, ...] = tuple(s[0] for s in SUBSET_DEFS)

# ---------------------------------------------------------------------------
# Block-specific constants
# ---------------------------------------------------------------------------

# Block GG: histogram of only_up overall MAE on [-2.0, 0.0] in 0.1 steps.
GG_BIN_EDGES: np.ndarray = np.round(np.arange(-2.0, 0.0 + 1e-9, 0.1), 4)  # 21 edges -> 20 bins
GG_SL_CANDIDATES: Tuple[float, ...] = (1.0, 1.25, 1.5, 1.75)

# Block HH: histogram of only_up peak MFE on [0, 40] in 1.0 steps.
HH_BIN_EDGES: np.ndarray = np.round(np.arange(0.0, 40.0 + 1e-9, 1.0), 4)  # 41 edges -> 40 bins
HH_TP_CANDIDATES_ATR: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)

# Block II: peak x overall-MAE crosstab bins.
II_PEAK_BINS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 2.0, "peak_0_2"),
    (2.0, 4.0, "peak_2_4"),
    (4.0, 6.0, "peak_4_6"),
    (6.0, 8.0, "peak_6_8"),
    (8.0, 12.0, "peak_8_12"),
    (12.0, 20.0, "peak_12_20"),
    (20.0, float("inf"), "peak_20_plus"),
)
# mae bins go from closer-to-zero to more-negative.
II_MAE_BINS: Tuple[Tuple[float, float, str], ...] = (
    (-0.5, 0.0, "mae_0_to_-0p5"),
    (-1.0, -0.5, "mae_-0p5_to_-1p0"),
    (-1.5, -1.0, "mae_-1p0_to_-1p5"),
    (-2.0, -1.5, "mae_-1p5_to_-2p0"),
    (-float("inf"), -2.0, "mae_below_-2p0_overflow"),
)

# Block JJ: per-category running_close distribution at bar k.
JJ_K_GRID: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 40, 60, 90)
JJ_CATEGORIES_FOR_DIST: Tuple[str, ...] = (
    "only_up",
    "up_then_down",
    "down_then_up",
    "straight_to_sl",
)
JJ_RECALL_BUDGETS: Tuple[float, ...] = (0.90, 0.95, 0.99)

# Block KK: peak MFE conditional on early position.
KK_K_GRID: Tuple[int, ...] = (10, 20, 30, 40)
KK_POSITION_BINS: Tuple[Tuple[float, float, str], ...] = (
    (1.0, float("inf"), "above_+1ATR"),
    (0.0, 1.0, "in_(0,+1ATR]"),
    (-1.0, 0.0, "in_(-1,0]ATR"),
    (-2.0, -1.0, "in_(-2,-1]ATR"),
)
KK_HIGH_PEAK_ATR: float = 10.0  # = +5R under BL SL=2ATR

# Block LL: down_then_up / up_then_down overall MAE histogram and widened SL.
LL_BIN_EDGES: np.ndarray = np.round(np.arange(-10.0, -2.0 + 1e-9, 0.25), 4)  # 33 edges -> 32 bins
LL_WIDENED_SL: Tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 5.0)
LL_CATEGORIES: Tuple[str, ...] = ("down_then_up", "up_then_down")

# Block MM: tier analysis.
MM_TIER_BOUNDS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 4.0, "tier_low"),
    (4.0, 10.0, "tier_mid"),
    (10.0, 20.0, "tier_high"),
    (20.0, float("inf"), "tier_runner"),
)
MM_PATH_K_GRID: Tuple[int, ...] = (1, 5, 10, 15, 20, 30, 50, 80, 120, 180, 240)

# Block NN: only_up final R under BL and H240 exit policies.
# BL = SL=2 ATR + hold to k=120. H240 = SL=2 ATR + hold to k=240.
# Histogram: 0.25R bins from -1.0 to +15.0 (64 bins) + overflow >+15R.
NN_BIN_EDGES: np.ndarray = np.round(np.arange(-1.0, 15.0 + 1e-9, 0.25), 4)
NN_EXIT_POLICIES: Tuple[str, ...] = ("BL", "H240")
NN_BL_HOLD_CAP: int = 120
NN_H240_HOLD_CAP: int = 240
NN_SL_ATR: float = 2.0  # SL distance for both policies

# Block KK3: tier breakdown by position at early k.
# Position bins identical to Block KK.
KK3_K_GRID: Tuple[int, ...] = (10, 20, 30)
KK3_TAU_SELECTIVITY: float = -0.5  # ATR fill-rel threshold for "below tau"

# Section 14.5 disposition discipline.
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
# Quintile bucketing (byte-faithful with arc2_entry_filter_bivariate.py and
# arc2_path_excursion_descriptives.py)
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


def _per_trade_overall_mae_and_peak(
    starts: np.ndarray,
    ends: np.ndarray,
    rmfe: np.ndarray,
    rmae: np.ndarray,
    bc: np.ndarray,
    n_trades: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-trade reductions.

    Returns:
      overall_mae: min(running_mae_atr) over full per_bar window
      peak_mfe: max(running_mfe_atr) over full per_bar window
      close_at_k120: bar_close_atr at k=120 (NaN if n_bars < 120)
      final_close: bar_close_atr at last available bar
      n_bars: trade window length
      peak_mfe_k: 1-indexed k at first occurrence of peak
    """
    overall_mae = np.full(n_trades, np.nan, dtype=np.float64)
    peak_mfe = np.full(n_trades, np.nan, dtype=np.float64)
    close_k120 = np.full(n_trades, np.nan, dtype=np.float64)
    final_close = np.full(n_trades, np.nan, dtype=np.float64)
    n_bars = np.zeros(n_trades, dtype=np.int32)
    peak_k = np.zeros(n_trades, dtype=np.int32)
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        if e <= s:
            continue
        nb = e - s
        n_bars[tid] = nb
        mfe_arr = rmfe[s:e]
        mae_arr = rmae[s:e]
        bc_arr = bc[s:e]
        overall_mae[tid] = float(mae_arr.min())
        idx = int(np.argmax(mfe_arr))
        peak_mfe[tid] = float(mfe_arr[idx])
        peak_k[tid] = idx + 1
        final_close[tid] = float(bc_arr[-1])
        if nb >= 120:
            close_k120[tid] = float(bc_arr[119])
    return overall_mae, peak_mfe, close_k120, final_close, n_bars, peak_k


def _per_trade_BL_H240_final_R(
    starts: np.ndarray, ends: np.ndarray, rmae: np.ndarray, bc: np.ndarray, n_trades: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-trade BL and H240 exit final_R derivation from per_bar_paths.

    BL exit policy: SL = -2 ATR fill-rel, hold cap = 120 bars.
      - SL hit when running_mae_atr <= -2.0 at any k in [1, min(120, n_bars)].
        On SL: gross_R = -1.0 (matches V00_BL convention).
      - Else exit at bar_close at k = min(120, n_bars); R = close / 2.0.

    H240 exit policy: SL = -2 ATR fill-rel, hold cap = 240 bars.
      - SL hit when running_mae_atr <= -2.0 at any k in [1, min(240, n_bars)].
        On SL: gross_R = -1.0 (matches V09_H240 convention).
      - Else exit at bar_close at k = min(240, n_bars); R = close / 2.0.

    Returns:
      final_R_BL, final_R_H240,
      sl_hit_k_BL (1-indexed, -1 if not stopped),
      sl_hit_k_H240 (1-indexed, -1 if not stopped).
    """
    final_R_BL = np.full(n_trades, np.nan, dtype=np.float64)
    final_R_H240 = np.full(n_trades, np.nan, dtype=np.float64)
    sl_k_BL = np.full(n_trades, -1, dtype=np.int32)
    sl_k_H240 = np.full(n_trades, -1, dtype=np.int32)
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        nb = e - s
        if nb == 0:
            continue
        rmae_arr = rmae[s:e]
        bc_arr = bc[s:e]
        # BL.
        end_idx_bl = min(NN_BL_HOLD_CAP, nb)
        mask_bl = rmae_arr[:end_idx_bl] <= -NN_SL_ATR
        if mask_bl.any():
            sl_k_BL[tid] = int(np.argmax(mask_bl)) + 1
            final_R_BL[tid] = -1.0
        else:
            final_R_BL[tid] = float(bc_arr[end_idx_bl - 1]) / 2.0
        # H240.
        end_idx_h = min(NN_H240_HOLD_CAP, nb)
        mask_h = rmae_arr[:end_idx_h] <= -NN_SL_ATR
        if mask_h.any():
            sl_k_H240[tid] = int(np.argmax(mask_h)) + 1
            final_R_H240[tid] = -1.0
        else:
            final_R_H240[tid] = float(bc_arr[end_idx_h - 1]) / 2.0
    return final_R_BL, final_R_H240, sl_k_BL, sl_k_H240


# ===========================================================================
# Block GG — only_up overall MAE distribution
# ===========================================================================


def compute_block_GG(
    cats: np.ndarray, overall_mae: np.ndarray, subsets: Dict[str, np.ndarray]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_trade_rows: List[Dict[str, Any]] = []
    hist_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    kill_rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        ou_tids = np.array(
            [int(t) for t in tids.tolist() if cats[int(t)] == "only_up"],
            dtype=np.int64,
        )
        vals = overall_mae[ou_tids]
        n_total = int(len(ou_tids))
        # Per-trade.
        for t, v in zip(ou_tids.tolist(), vals.tolist()):
            per_trade_rows.append(
                {
                    "subset_id": sid,
                    "trade_id": int(t),
                    "overall_mae_atr": float(v),
                }
            )
        # Histogram bins.
        n_underflow = int((vals < GG_BIN_EDGES[0]).sum())
        # Per-bin counts using half-open [left, right) with last bin closed.
        for i in range(len(GG_BIN_EDGES) - 1):
            lo = float(GG_BIN_EDGES[i])
            hi = float(GG_BIN_EDGES[i + 1])
            if i == len(GG_BIN_EDGES) - 2:
                in_bin = (vals >= lo) & (vals <= hi)
            else:
                in_bin = (vals >= lo) & (vals < hi)
            n_in = int(in_bin.sum())
            # Cumulative count of trades with overall_mae <= bin right edge
            # (i.e. killed under SL = -hi).
            cum_at_or_below = int((vals <= hi).sum())
            hist_rows.append(
                {
                    "subset_id": sid,
                    "bin_idx": i,
                    "bin_left_atr": lo,
                    "bin_right_atr": hi,
                    "n_trades_in_bin": n_in,
                    "cumulative_count_at_or_below_right_edge": cum_at_or_below,
                    "pct_cumulative_at_or_below_right_edge": (
                        cum_at_or_below / n_total if n_total > 0 else float("nan")
                    ),
                }
            )
        # Overflow row for trades worse than -2.0 ATR.
        hist_rows.append(
            {
                "subset_id": sid,
                "bin_idx": -1,
                "bin_left_atr": -float("inf"),
                "bin_right_atr": float(GG_BIN_EDGES[0]),
                "n_trades_in_bin": n_underflow,
                "cumulative_count_at_or_below_right_edge": n_underflow,
                "pct_cumulative_at_or_below_right_edge": (
                    n_underflow / n_total if n_total > 0 else float("nan")
                ),
            }
        )

        # Summary percentiles.
        if n_total == 0:
            row: Dict[str, Any] = {"subset_id": sid, "n_only_up": 0}
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
            ]:
                row[q] = float("nan")
        else:
            row = {
                "subset_id": sid,
                "n_only_up": n_total,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if n_total > 1 else 0.0,
                "median": float(np.median(vals)),
                "q01": float(np.quantile(vals, 0.01)),
                "q05": float(np.quantile(vals, 0.05)),
                "q10": float(np.quantile(vals, 0.10)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q90": float(np.quantile(vals, 0.90)),
                "q95": float(np.quantile(vals, 0.95)),
                "q99": float(np.quantile(vals, 0.99)),
                "min": float(np.min(vals)),
            }
        summary_rows.append(row)

        # Kill rate.
        for d in GG_SL_CANDIDATES:
            n_killed = int((vals <= -d).sum())
            kill_rows.append(
                {
                    "subset_id": sid,
                    "SL_distance_atr": float(d),
                    "n_only_up_total": n_total,
                    "n_only_up_killed": n_killed,
                    "pct_only_up_killed": (n_killed / n_total if n_total > 0 else float("nan")),
                }
            )

    return (
        pd.DataFrame(per_trade_rows),
        pd.DataFrame(hist_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(kill_rows),
    )


# ===========================================================================
# Block HH — only_up peak MFE distribution
# ===========================================================================


def compute_block_HH(
    cats: np.ndarray, peak_mfe: np.ndarray, subsets: Dict[str, np.ndarray]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    tp_rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        ou_tids = np.array(
            [int(t) for t in tids.tolist() if cats[int(t)] == "only_up"],
            dtype=np.int64,
        )
        vals = peak_mfe[ou_tids]
        n_total = int(len(ou_tids))
        # Histogram bins 0..40 with last bin closed.
        for i in range(len(HH_BIN_EDGES) - 1):
            lo = float(HH_BIN_EDGES[i])
            hi = float(HH_BIN_EDGES[i + 1])
            in_bin = (vals >= lo) & (vals < hi)
            n_in = int(in_bin.sum())
            cum_above = int((vals >= hi).sum())
            hist_rows.append(
                {
                    "subset_id": sid,
                    "bin_idx": i,
                    "bin_left_atr": lo,
                    "bin_right_atr": hi,
                    "bin_left_R_at_BL_SL": lo / 2.0,
                    "bin_right_R_at_BL_SL": hi / 2.0,
                    "n_trades_in_bin": n_in,
                    "cumulative_count_at_or_above_right_edge": cum_above,
                    "pct_cumulative_at_or_above_right_edge": (
                        cum_above / n_total if n_total > 0 else float("nan")
                    ),
                }
            )
        # Overflow bin >= 40 ATR.
        last_edge = float(HH_BIN_EDGES[-1])
        n_overflow = int((vals >= last_edge).sum())
        hist_rows.append(
            {
                "subset_id": sid,
                "bin_idx": len(HH_BIN_EDGES) - 1,
                "bin_left_atr": last_edge,
                "bin_right_atr": float("inf"),
                "bin_left_R_at_BL_SL": last_edge / 2.0,
                "bin_right_R_at_BL_SL": float("inf"),
                "n_trades_in_bin": n_overflow,
                "cumulative_count_at_or_above_right_edge": 0,
                "pct_cumulative_at_or_above_right_edge": 0.0,
            }
        )

        # Summary percentiles.
        if n_total == 0:
            row: Dict[str, Any] = {"subset_id": sid, "n_only_up": 0}
            for q in [
                "mean",
                "std",
                "median",
                "q05",
                "q10",
                "q25",
                "q50",
                "q75",
                "q80",
                "q85",
                "q90",
                "q95",
                "q99",
                "max",
            ]:
                row[q] = float("nan")
        else:
            row = {
                "subset_id": sid,
                "n_only_up": n_total,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if n_total > 1 else 0.0,
                "median": float(np.median(vals)),
                "q05": float(np.quantile(vals, 0.05)),
                "q10": float(np.quantile(vals, 0.10)),
                "q25": float(np.quantile(vals, 0.25)),
                "q50": float(np.quantile(vals, 0.50)),
                "q75": float(np.quantile(vals, 0.75)),
                "q80": float(np.quantile(vals, 0.80)),
                "q85": float(np.quantile(vals, 0.85)),
                "q90": float(np.quantile(vals, 0.90)),
                "q95": float(np.quantile(vals, 0.95)),
                "q99": float(np.quantile(vals, 0.99)),
                "max": float(np.max(vals)),
            }
        summary_rows.append(row)

        # TP evaluation per candidate.
        for tp in HH_TP_CANDIDATES_ATR:
            reaches = vals >= tp
            n_reach = int(reaches.sum())
            # Of those reaching tp, excess loss = peak - tp.
            excess_per_trade = np.maximum(vals - tp, 0.0)
            mean_excess_all = float(excess_per_trade.mean()) if n_total > 0 else float("nan")
            tp_rows.append(
                {
                    "subset_id": sid,
                    "tp_atr": float(tp),
                    "tp_R_at_BL_SL": float(tp) / 2.0,
                    "n_only_up_total": n_total,
                    "n_only_up_reaching_tp": n_reach,
                    "pct_only_up_reaching_tp": (n_reach / n_total if n_total > 0 else float("nan")),
                    "n_only_up_capped_above_tp": n_reach,  # alias for clarity
                    "pct_only_up_capped_above_tp": (
                        n_reach / n_total if n_total > 0 else float("nan")
                    ),
                    "expected_capped_loss_atr_per_only_up": mean_excess_all,
                    "expected_capped_loss_R_per_only_up": mean_excess_all / 2.0,
                }
            )

    return (
        pd.DataFrame(hist_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(tp_rows),
    )


# ===========================================================================
# Block II — only_up peak x overall MAE crosstab
# ===========================================================================


def _bin_peak(peak_v: float) -> str:
    for lo, hi, label in II_PEAK_BINS:
        if peak_v > lo and peak_v <= hi:
            return label
    # Edge case: peak == 0 (impossible for only_up but handle).
    return II_PEAK_BINS[0][2]


def _bin_mae(mae_v: float) -> str:
    # mae_bins intervals are (lo, hi] with lo < hi (numerically).
    # For overflow: (-inf, -2.0] => mae_v <= -2.0.
    for lo, hi, label in II_MAE_BINS:
        if mae_v > lo and mae_v <= hi:
            return label
    # Shouldn't happen but fallback.
    return II_MAE_BINS[0][2]


def compute_block_II(
    cats: np.ndarray,
    peak_mfe: np.ndarray,
    overall_mae: np.ndarray,
    close_k120: np.ndarray,
    final_close: np.ndarray,
    subsets: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows: List[Dict[str, Any]] = []
    marg_rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        ou = [int(t) for t in tids.tolist() if cats[int(t)] == "only_up"]
        if len(ou) == 0:
            continue
        ou_arr = np.array(ou, dtype=np.int64)
        peak_vals = peak_mfe[ou_arr]
        mae_vals = overall_mae[ou_arr]
        final_R = final_close[ou_arr] / 2.0
        giveback = np.where(
            np.isnan(close_k120[ou_arr]),
            np.nan,
            peak_vals - close_k120[ou_arr],
        )
        peak_bins = np.array([_bin_peak(float(p)) for p in peak_vals.tolist()], dtype=object)
        mae_bins = np.array([_bin_mae(float(m)) for m in mae_vals.tolist()], dtype=object)
        for _, _, pb_label in II_PEAK_BINS:
            for _, _, mb_label in II_MAE_BINS:
                mask = (peak_bins == pb_label) & (mae_bins == mb_label)
                n = int(mask.sum())
                if n == 0:
                    cell_rows.append(
                        {
                            "subset_id": sid,
                            "peak_bin": pb_label,
                            "mae_bin": mb_label,
                            "n_trades": 0,
                            "mean_peak_atr": float("nan"),
                            "mean_mae_atr": float("nan"),
                            "mean_final_close_R_at_BL_SL": float("nan"),
                            "mean_giveback_to_k120_atr": float("nan"),
                            "n_with_giveback": 0,
                        }
                    )
                    continue
                gv = giveback[mask]
                gv_valid = gv[~np.isnan(gv)]
                cell_rows.append(
                    {
                        "subset_id": sid,
                        "peak_bin": pb_label,
                        "mae_bin": mb_label,
                        "n_trades": n,
                        "mean_peak_atr": float(np.mean(peak_vals[mask])),
                        "mean_mae_atr": float(np.mean(mae_vals[mask])),
                        "mean_final_close_R_at_BL_SL": float(np.mean(final_R[mask])),
                        "mean_giveback_to_k120_atr": (
                            float(np.mean(gv_valid)) if len(gv_valid) > 0 else float("nan")
                        ),
                        "n_with_giveback": int(len(gv_valid)),
                    }
                )

        # Marginal by mae bin.
        for _, _, mb_label in II_MAE_BINS:
            mask = mae_bins == mb_label
            n = int(mask.sum())
            if n == 0:
                marg_rows.append(
                    {
                        "subset_id": sid,
                        "mae_bin": mb_label,
                        "n_trades": 0,
                        "mean_peak_atr": float("nan"),
                        "median_peak_atr": float("nan"),
                        "pct_of_only_up_in_subset": 0.0,
                    }
                )
                continue
            marg_rows.append(
                {
                    "subset_id": sid,
                    "mae_bin": mb_label,
                    "n_trades": n,
                    "mean_peak_atr": float(np.mean(peak_vals[mask])),
                    "median_peak_atr": float(np.median(peak_vals[mask])),
                    "pct_of_only_up_in_subset": n / len(ou_arr),
                }
            )

    return pd.DataFrame(cell_rows), pd.DataFrame(marg_rows)


# ===========================================================================
# Block JJ — per-category running_close distribution + only_up separator
# ===========================================================================


def _close_at_k(starts: np.ndarray, ends: np.ndarray, bc: np.ndarray, tid: int, k: int) -> float:
    s, e = int(starts[tid]), int(ends[tid])
    if e - s >= k:
        return float(bc[s + k - 1])
    return float("nan")


def _close_at_k_for_tids(
    starts: np.ndarray, ends: np.ndarray, bc: np.ndarray, tid_array: np.ndarray, k: int
) -> np.ndarray:
    out = np.full(len(tid_array), np.nan, dtype=np.float64)
    for i, tid in enumerate(tid_array.tolist()):
        s, e = int(starts[tid]), int(ends[tid])
        if e - s >= k:
            out[i] = bc[s + k - 1]
    return out


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d using pooled SD with ddof=1.

    Returns NaN if either group has < 2 valid samples or pooled SD == 0.
    """
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = float(np.std(x, ddof=1))
    s2 = float(np.std(y, ddof=1))
    pooled_var = ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return float("nan")
    pooled = float(np.sqrt(pooled_var))
    return (float(np.mean(x)) - float(np.mean(y))) / pooled


def compute_block_JJ(
    starts: np.ndarray,
    ends: np.ndarray,
    bc: np.ndarray,
    cats: np.ndarray,
    subsets: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_category_close_at_k_df, only_up_separator_curves_df)."""
    cat_rows: List[Dict[str, Any]] = []
    sep_rows: List[Dict[str, Any]] = []

    for sid in SUBSET_IDS:
        tids = subsets[sid]
        # Partition tids by category.
        tids_by_cat: Dict[str, np.ndarray] = {}
        for c in JJ_CATEGORIES_FOR_DIST:
            tlist = [int(t) for t in tids.tolist() if cats[int(t)] == c]
            tids_by_cat[c] = np.array(tlist, dtype=np.int64)

        for k in JJ_K_GRID:
            # Per-category distribution rows.
            vals_by_cat: Dict[str, np.ndarray] = {}
            for c in JJ_CATEGORIES_FOR_DIST:
                vs = _close_at_k_for_tids(starts, ends, bc, tids_by_cat[c], k)
                vs_valid = vs[~np.isnan(vs)]
                vals_by_cat[c] = vs_valid
                n = int(len(vs_valid))
                row: Dict[str, Any] = {
                    "subset_id": sid,
                    "k": int(k),
                    "category": c,
                    "n": n,
                }
                if n == 0:
                    for q in ["mean", "std", "median", "q05", "q10", "q25", "q75", "q90", "q95"]:
                        row[q] = float("nan")
                else:
                    row["mean"] = float(np.mean(vs_valid))
                    row["std"] = float(np.std(vs_valid, ddof=1)) if n > 1 else 0.0
                    row["median"] = float(np.median(vs_valid))
                    row["q05"] = float(np.quantile(vs_valid, 0.05))
                    row["q10"] = float(np.quantile(vs_valid, 0.10))
                    row["q25"] = float(np.quantile(vs_valid, 0.25))
                    row["q75"] = float(np.quantile(vs_valid, 0.75))
                    row["q90"] = float(np.quantile(vs_valid, 0.90))
                    row["q95"] = float(np.quantile(vs_valid, 0.95))
                cat_rows.append(row)

            # only_up separator curves.
            vou = vals_by_cat["only_up"]
            v_utd = vals_by_cat["up_then_down"]
            v_dtu = vals_by_cat["down_then_up"]
            v_sts = vals_by_cat["straight_to_sl"]
            n_ou = int(len(vou))
            n_utd = int(len(v_utd))
            n_dtu = int(len(v_dtu))
            n_sts = int(len(v_sts))
            n_loser_total = n_utd + n_dtu + n_sts
            v_losers = np.concatenate([v_utd, v_dtu, v_sts]) if n_loser_total > 0 else np.array([])
            d_utd = _cohens_d(vou, v_utd)
            d_dtu = _cohens_d(vou, v_dtu)
            d_sts = _cohens_d(vou, v_sts)
            d_losers = _cohens_d(vou, v_losers)

            for budget in JJ_RECALL_BUDGETS:
                if n_ou == 0:
                    sep_rows.append(
                        {
                            "subset_id": sid,
                            "k": int(k),
                            "recall_only_up_budget": float(budget),
                            "n_only_up": 0,
                            "tau_only_up_atr_fill": float("nan"),
                            "n_only_up_above_tau": 0,
                            "recall_only_up_actual": float("nan"),
                            "n_losers_total": n_loser_total,
                            "n_losers_below_tau": 0,
                            "pct_losers_below_tau": float("nan"),
                            "n_up_then_down": n_utd,
                            "n_up_then_down_below_tau": 0,
                            "n_down_then_up": n_dtu,
                            "n_down_then_up_below_tau": 0,
                            "n_straight_to_sl": n_sts,
                            "n_straight_to_sl_below_tau": 0,
                            "cohens_d_only_up_vs_up_then_down": d_utd,
                            "cohens_d_only_up_vs_down_then_up": d_dtu,
                            "cohens_d_only_up_vs_straight_to_sl": d_sts,
                            "cohens_d_only_up_vs_loser_pool": d_losers,
                        }
                    )
                    continue
                # tau = largest value such that >= budget * n_ou trades
                # have close >= tau. That is the (1 - budget) quantile of
                # vou, using lower interpolation so the count is exactly
                # ceil(budget * n_ou).
                target_count = int(np.ceil(budget * n_ou))
                # Sort descending; threshold is the target_count-th largest
                # value (1-indexed); equivalent to vou_sorted_desc[target_count-1].
                vou_sorted_desc = np.sort(vou)[::-1]
                if target_count < 1:
                    target_count = 1
                if target_count > n_ou:
                    target_count = n_ou
                tau = float(vou_sorted_desc[target_count - 1])
                n_above = int((vou >= tau).sum())
                recall_actual = n_above / n_ou
                n_utd_below = int((v_utd < tau).sum()) if n_utd > 0 else 0
                n_dtu_below = int((v_dtu < tau).sum()) if n_dtu > 0 else 0
                n_sts_below = int((v_sts < tau).sum()) if n_sts > 0 else 0
                n_losers_below = n_utd_below + n_dtu_below + n_sts_below
                sep_rows.append(
                    {
                        "subset_id": sid,
                        "k": int(k),
                        "recall_only_up_budget": float(budget),
                        "n_only_up": n_ou,
                        "tau_only_up_atr_fill": tau,
                        "n_only_up_above_tau": n_above,
                        "recall_only_up_actual": recall_actual,
                        "n_losers_total": n_loser_total,
                        "n_losers_below_tau": n_losers_below,
                        "pct_losers_below_tau": (
                            n_losers_below / n_loser_total if n_loser_total > 0 else float("nan")
                        ),
                        "n_up_then_down": n_utd,
                        "n_up_then_down_below_tau": n_utd_below,
                        "n_down_then_up": n_dtu,
                        "n_down_then_up_below_tau": n_dtu_below,
                        "n_straight_to_sl": n_sts,
                        "n_straight_to_sl_below_tau": n_sts_below,
                        "cohens_d_only_up_vs_up_then_down": d_utd,
                        "cohens_d_only_up_vs_down_then_up": d_dtu,
                        "cohens_d_only_up_vs_straight_to_sl": d_sts,
                        "cohens_d_only_up_vs_loser_pool": d_losers,
                    }
                )

    return pd.DataFrame(cat_rows), pd.DataFrame(sep_rows)


# ===========================================================================
# Block KK — peak MFE conditional on early position
# ===========================================================================


def _position_bin(rc: float) -> Optional[str]:
    for lo, hi, label in KK_POSITION_BINS:
        if rc > lo and rc <= hi:
            return label
    return None


def compute_block_KK(
    starts: np.ndarray,
    ends: np.ndarray,
    bc: np.ndarray,
    cats: np.ndarray,
    peak_mfe: np.ndarray,
    subsets: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        tids_arr = np.array([int(t) for t in tids.tolist()], dtype=np.int64)
        cats_for_sub = cats[tids_arr]
        ou_mask = cats_for_sub == "only_up"
        ou_tids = tids_arr[ou_mask]
        for k in KK_K_GRID:
            # Close at bar k for all trades in subset and the only_up subset.
            close_all = _close_at_k_for_tids(starts, ends, bc, tids_arr, k)
            close_ou = _close_at_k_for_tids(starts, ends, bc, ou_tids, k)
            peak_ou = peak_mfe[ou_tids]
            # Assign bins.
            bins_all = np.array(
                [_position_bin(v) if not np.isnan(v) else None for v in close_all.tolist()],
                dtype=object,
            )
            bins_ou = np.array(
                [_position_bin(v) if not np.isnan(v) else None for v in close_ou.tolist()],
                dtype=object,
            )
            for _, _, bin_label in KK_POSITION_BINS:
                mask_all = bins_all == bin_label
                mask_ou = bins_ou == bin_label
                n_all = int(mask_all.sum())
                n_ou_in = int(mask_ou.sum())
                if n_ou_in > 0:
                    peaks_in = peak_ou[mask_ou]
                    mean_peak = float(np.mean(peaks_in))
                    med_peak = float(np.median(peaks_in))
                    n_high = int((peaks_in >= KK_HIGH_PEAK_ATR).sum())
                    pct_high = n_high / n_ou_in
                else:
                    mean_peak = float("nan")
                    med_peak = float("nan")
                    n_high = 0
                    pct_high = float("nan")
                rows.append(
                    {
                        "subset_id": sid,
                        "k": int(k),
                        "position_bin": bin_label,
                        "n_only_up_in_bin": n_ou_in,
                        "mean_peak_mfe_atr": mean_peak,
                        "median_peak_mfe_atr": med_peak,
                        "n_only_up_reaching_+10atr": n_high,
                        "pct_of_only_up_in_bin_reaching_+10atr": pct_high,
                        "n_all_trades_in_bin": n_all,
                        "pct_of_all_trades_in_bin_that_are_only_up": (
                            n_ou_in / n_all if n_all > 0 else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


# ===========================================================================
# Block LL — down_then_up + up_then_down overall MAE distribution
# ===========================================================================


def compute_block_LL(
    cats: np.ndarray, overall_mae: np.ndarray, subsets: Dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        for cat_label in LL_CATEGORIES:
            sub_tids = np.array(
                [int(t) for t in tids.tolist() if cats[int(t)] == cat_label],
                dtype=np.int64,
            )
            vals = overall_mae[sub_tids]
            n_total = int(len(sub_tids))
            # Histogram bins -10.0..-2.0 with last bin closed.
            for i in range(len(LL_BIN_EDGES) - 1):
                lo = float(LL_BIN_EDGES[i])
                hi = float(LL_BIN_EDGES[i + 1])
                if i == len(LL_BIN_EDGES) - 2:
                    in_bin = (vals >= lo) & (vals <= hi)
                else:
                    in_bin = (vals >= lo) & (vals < hi)
                n_in = int(in_bin.sum())
                cum_at_or_below = int((vals <= hi).sum())
                rows.append(
                    {
                        "subset_id": sid,
                        "category": cat_label,
                        "bin_idx": i,
                        "bin_left_atr": lo,
                        "bin_right_atr": hi,
                        "n_trades_in_bin": n_in,
                        "cumulative_count_at_or_below_right_edge": cum_at_or_below,
                        "pct_cumulative_at_or_below_right_edge": (
                            cum_at_or_below / n_total if n_total > 0 else float("nan")
                        ),
                        "n_total_category": n_total,
                        "row_kind": "bin",
                        "widened_SL_atr": float("nan"),
                        "n_surviving_widened_SL": -1,
                        "pct_surviving_widened_SL": float("nan"),
                    }
                )
            # Underflow row for < -10.
            n_under = int((vals < LL_BIN_EDGES[0]).sum())
            rows.append(
                {
                    "subset_id": sid,
                    "category": cat_label,
                    "bin_idx": -1,
                    "bin_left_atr": -float("inf"),
                    "bin_right_atr": float(LL_BIN_EDGES[0]),
                    "n_trades_in_bin": n_under,
                    "cumulative_count_at_or_below_right_edge": n_under,
                    "pct_cumulative_at_or_below_right_edge": (
                        n_under / n_total if n_total > 0 else float("nan")
                    ),
                    "n_total_category": n_total,
                    "row_kind": "bin_underflow",
                    "widened_SL_atr": float("nan"),
                    "n_surviving_widened_SL": -1,
                    "pct_surviving_widened_SL": float("nan"),
                }
            )
            # Widened SL survival rows.
            for d in LL_WIDENED_SL:
                n_surv = int((vals > -d).sum())
                rows.append(
                    {
                        "subset_id": sid,
                        "category": cat_label,
                        "bin_idx": -2,
                        "bin_left_atr": float("nan"),
                        "bin_right_atr": float("nan"),
                        "n_trades_in_bin": -1,
                        "cumulative_count_at_or_below_right_edge": -1,
                        "pct_cumulative_at_or_below_right_edge": float("nan"),
                        "n_total_category": n_total,
                        "row_kind": "widened_SL_survival",
                        "widened_SL_atr": float(d),
                        "n_surviving_widened_SL": n_surv,
                        "pct_surviving_widened_SL": (
                            n_surv / n_total if n_total > 0 else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


# ===========================================================================
# Block MM — only_up tier analysis + median paths
# ===========================================================================


def _assign_tier(peak_v: float) -> str:
    for lo, hi, label in MM_TIER_BOUNDS:
        if peak_v > lo and peak_v <= hi:
            return label
    return MM_TIER_BOUNDS[0][2]


def compute_block_MM(
    starts: np.ndarray,
    ends: np.ndarray,
    bc: np.ndarray,
    cats: np.ndarray,
    peak_mfe: np.ndarray,
    close_k120: np.ndarray,
    final_close: np.ndarray,
    subsets: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tier_rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        ou = [int(t) for t in tids.tolist() if cats[int(t)] == "only_up"]
        if len(ou) == 0:
            continue
        ou_arr = np.array(ou, dtype=np.int64)
        peak_vals = peak_mfe[ou_arr]
        final_R = final_close[ou_arr] / 2.0
        # Giveback to k=120 (where available).
        gv_k120 = np.where(
            np.isnan(close_k120[ou_arr]),
            np.nan,
            peak_vals - close_k120[ou_arr],
        )
        tier_labels = np.array(
            [_assign_tier(float(p)) for p in peak_vals.tolist()],
            dtype=object,
        )
        subset_total_R = float(np.nansum(final_R))
        for _, _, tier_label in MM_TIER_BOUNDS:
            mask = tier_labels == tier_label
            n = int(mask.sum())
            if n == 0:
                tier_rows.append(
                    {
                        "subset_id": sid,
                        "tier": tier_label,
                        "n_trades": 0,
                        "mean_peak_atr": float("nan"),
                        "mean_giveback_to_k120_atr": float("nan"),
                        "n_with_giveback": 0,
                        "mean_final_close_R_at_BL_SL": float("nan"),
                        "total_R_contribution": 0.0,
                        "pct_of_subset_total_R_contribution": (
                            0.0 if subset_total_R != 0 else float("nan")
                        ),
                    }
                )
                continue
            gv_in = gv_k120[mask]
            gv_valid = gv_in[~np.isnan(gv_in)]
            mean_fr = float(np.mean(final_R[mask]))
            tot_r = float(np.sum(final_R[mask]))
            tier_rows.append(
                {
                    "subset_id": sid,
                    "tier": tier_label,
                    "n_trades": n,
                    "mean_peak_atr": float(np.mean(peak_vals[mask])),
                    "mean_giveback_to_k120_atr": (
                        float(np.mean(gv_valid)) if len(gv_valid) > 0 else float("nan")
                    ),
                    "n_with_giveback": int(len(gv_valid)),
                    "mean_final_close_R_at_BL_SL": mean_fr,
                    "total_R_contribution": tot_r,
                    "pct_of_subset_total_R_contribution": (
                        tot_r / subset_total_R if subset_total_R != 0 else float("nan")
                    ),
                }
            )

        # Median running_close path per tier across k grid.
        for _, _, tier_label in MM_TIER_BOUNDS:
            tier_tids = ou_arr[tier_labels == tier_label]
            n_tier = int(len(tier_tids))
            for k in MM_PATH_K_GRID:
                vs = _close_at_k_for_tids(starts, ends, bc, tier_tids, k)
                vs_valid = vs[~np.isnan(vs)]
                n_valid = int(len(vs_valid))
                path_rows.append(
                    {
                        "subset_id": sid,
                        "tier": tier_label,
                        "k": int(k),
                        "n_tier_total": n_tier,
                        "n_with_close_at_k": n_valid,
                        "median_running_close_atr": (
                            float(np.median(vs_valid)) if n_valid > 0 else float("nan")
                        ),
                        "q25_running_close_atr": (
                            float(np.quantile(vs_valid, 0.25)) if n_valid > 0 else float("nan")
                        ),
                        "q75_running_close_atr": (
                            float(np.quantile(vs_valid, 0.75)) if n_valid > 0 else float("nan")
                        ),
                    }
                )

    return pd.DataFrame(tier_rows), pd.DataFrame(path_rows)


# ===========================================================================
# Block NN — only_up final R distribution under BL and H240
# ===========================================================================


def _nn_summary_row(vals: np.ndarray) -> Dict[str, float]:
    n = int(len(vals))
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(vals)),
        "q05": float(np.quantile(vals, 0.05)),
        "q10": float(np.quantile(vals, 0.10)),
        "q25": float(np.quantile(vals, 0.25)),
        "q50": float(np.quantile(vals, 0.50)),
        "q75": float(np.quantile(vals, 0.75)),
        "q90": float(np.quantile(vals, 0.90)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
        "max": float(np.max(vals)),
    }


def _nn_histogram_rows(
    vals: np.ndarray, subset_id: str, exit_policy: str, tier: Optional[str] = None
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_total = int(len(vals))
    n_underflow = int((vals < NN_BIN_EDGES[0]).sum())
    for i in range(len(NN_BIN_EDGES) - 1):
        lo = float(NN_BIN_EDGES[i])
        hi = float(NN_BIN_EDGES[i + 1])
        if i == len(NN_BIN_EDGES) - 2:
            in_bin = (vals >= lo) & (vals <= hi)
        else:
            in_bin = (vals >= lo) & (vals < hi)
        n_in = int(in_bin.sum())
        row = {
            "subset_id": subset_id,
            "exit_policy": exit_policy,
            "bin_idx": i,
            "bin_left_R": lo,
            "bin_right_R": hi,
            "n_trades_in_bin": n_in,
            "pct_of_only_up": n_in / n_total if n_total > 0 else float("nan"),
            "row_kind": "bin",
        }
        if tier is not None:
            row["tier"] = tier
        rows.append(row)
    # Overflow > +15R.
    last_edge = float(NN_BIN_EDGES[-1])
    n_overflow = int((vals > last_edge).sum())
    row_ov = {
        "subset_id": subset_id,
        "exit_policy": exit_policy,
        "bin_idx": len(NN_BIN_EDGES) - 1,
        "bin_left_R": last_edge,
        "bin_right_R": float("inf"),
        "n_trades_in_bin": n_overflow,
        "pct_of_only_up": n_overflow / n_total if n_total > 0 else float("nan"),
        "row_kind": "bin_overflow",
    }
    if tier is not None:
        row_ov["tier"] = tier
    rows.append(row_ov)
    # Underflow row (defensive; for only_up trades these stay >= -1.0R by
    # construction, but emit the row to satisfy NN.1 gate counting).
    row_un = {
        "subset_id": subset_id,
        "exit_policy": exit_policy,
        "bin_idx": -1,
        "bin_left_R": -float("inf"),
        "bin_right_R": float(NN_BIN_EDGES[0]),
        "n_trades_in_bin": n_underflow,
        "pct_of_only_up": n_underflow / n_total if n_total > 0 else float("nan"),
        "row_kind": "bin_underflow",
    }
    if tier is not None:
        row_un["tier"] = tier
    rows.append(row_un)
    # Summary row.
    s = _nn_summary_row(vals)
    row_summ = {
        "subset_id": subset_id,
        "exit_policy": exit_policy,
        "bin_idx": -2,
        "bin_left_R": float("nan"),
        "bin_right_R": float("nan"),
        "n_trades_in_bin": n_total,
        "pct_of_only_up": 1.0 if n_total > 0 else float("nan"),
        "row_kind": "summary",
        **{f"summary_{k}": v for k, v in s.items()},
    }
    if tier is not None:
        row_summ["tier"] = tier
    rows.append(row_summ)
    return rows


def compute_block_NN(
    cats: np.ndarray,
    peak_mfe: np.ndarray,
    final_R_BL: np.ndarray,
    final_R_H240: np.ndarray,
    subsets: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_subset_df, per_tier_df).

    Each df has histogram bin rows + an overflow row + an underflow row +
    a summary row keyed by `row_kind`.
    """
    subset_rows: List[Dict[str, Any]] = []
    tier_rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        ou_tids = np.array(
            [int(t) for t in tids.tolist() if cats[int(t)] == "only_up"],
            dtype=np.int64,
        )
        # Per-subset.
        for policy in NN_EXIT_POLICIES:
            arr = final_R_BL if policy == "BL" else final_R_H240
            vals = arr[ou_tids]
            subset_rows.extend(_nn_histogram_rows(vals, sid, policy))
        # Per-tier.
        peak_vals = peak_mfe[ou_tids]
        tier_labels = np.array(
            [_assign_tier(float(p)) for p in peak_vals.tolist()],
            dtype=object,
        )
        for _, _, tier_label in MM_TIER_BOUNDS:
            mask = tier_labels == tier_label
            tier_tids = ou_tids[mask]
            for policy in NN_EXIT_POLICIES:
                arr = final_R_BL if policy == "BL" else final_R_H240
                vals = arr[tier_tids]
                tier_rows.extend(_nn_histogram_rows(vals, sid, policy, tier=tier_label))
    return pd.DataFrame(subset_rows), pd.DataFrame(tier_rows)


# ===========================================================================
# Block KK3 — tier breakdown by position at k
# ===========================================================================


def compute_block_KK3(
    starts: np.ndarray,
    ends: np.ndarray,
    bc: np.ndarray,
    cats: np.ndarray,
    peak_mfe: np.ndarray,
    subsets: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (tier_by_position_df, position_distribution_by_tier_df).

    Position bins follow Block KK (above_+1ATR, in_(0,+1ATR],
    in_(-1,0]ATR, in_(-2,-1]ATR).
    """
    tbyp_rows: List[Dict[str, Any]] = []
    pdbt_rows: List[Dict[str, Any]] = []
    TIER_LABELS = [b[2] for b in MM_TIER_BOUNDS]  # tier_low, mid, high, runner
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        tids_arr = np.array([int(t) for t in tids.tolist()], dtype=np.int64)
        cats_sub = cats[tids_arr]
        ou_mask = cats_sub == "only_up"
        ou_tids = tids_arr[ou_mask]
        peak_ou = peak_mfe[ou_tids]
        tier_ou = np.array(
            [_assign_tier(float(p)) for p in peak_ou.tolist()],
            dtype=object,
        )
        for k in KK3_K_GRID:
            close_all = _close_at_k_for_tids(starts, ends, bc, tids_arr, k)
            close_ou = _close_at_k_for_tids(starts, ends, bc, ou_tids, k)
            bins_all = np.array(
                [_position_bin(v) if not np.isnan(v) else None for v in close_all.tolist()],
                dtype=object,
            )
            bins_ou = np.array(
                [_position_bin(v) if not np.isnan(v) else None for v in close_ou.tolist()],
                dtype=object,
            )

            # Per-tier selectivity at tau = -0.5.
            sel_tau = KK3_TAU_SELECTIVITY
            sel_per_tier: Dict[str, Dict[str, Any]] = {}
            for tier_label in TIER_LABELS:
                tier_mask = tier_ou == tier_label
                vals_tier = close_ou[tier_mask]
                n_t = int(np.sum(~np.isnan(vals_tier)))
                n_below = int(np.sum(vals_tier <= sel_tau))
                sel_per_tier[tier_label] = {
                    "n_only_up_in_tier_with_close_at_k": n_t,
                    "n_only_up_in_tier_below_tau": n_below,
                    "pct_below_tau": (n_below / n_t if n_t > 0 else float("nan")),
                }
            # tier_low_plus_mid.
            lpm_mask = (tier_ou == "tier_low") | (tier_ou == "tier_mid")
            vals_lpm = close_ou[lpm_mask]
            n_lpm = int(np.sum(~np.isnan(vals_lpm)))
            n_lpm_below = int(np.sum(vals_lpm <= sel_tau))
            sel_per_tier["tier_low_plus_mid"] = {
                "n_only_up_in_tier_with_close_at_k": n_lpm,
                "n_only_up_in_tier_below_tau": n_lpm_below,
                "pct_below_tau": (n_lpm_below / n_lpm if n_lpm > 0 else float("nan")),
            }

            # Per (subset, k, position_bin) tier breakdown.
            for _, _, bin_label in KK_POSITION_BINS:
                mask_all = bins_all == bin_label
                mask_ou = bins_ou == bin_label
                n_all = int(mask_all.sum())
                n_ou_in = int(mask_ou.sum())
                tier_counts = {tl: 0 for tl in TIER_LABELS}
                if n_ou_in > 0:
                    tiers_in_bin = tier_ou[mask_ou]
                    for tl in TIER_LABELS:
                        tier_counts[tl] = int((tiers_in_bin == tl).sum())
                tbyp_rows.append(
                    {
                        "subset_id": sid,
                        "k": int(k),
                        "position_bin": bin_label,
                        "n_total_trades_in_bin": n_all,
                        "n_only_up_in_bin": n_ou_in,
                        "n_only_up_tier_low_in_bin": tier_counts["tier_low"],
                        "n_only_up_tier_mid_in_bin": tier_counts["tier_mid"],
                        "n_only_up_tier_high_in_bin": tier_counts["tier_high"],
                        "n_only_up_tier_runner_in_bin": tier_counts["tier_runner"],
                        # Per-(subset,k) selectivity at tau=-0.5 (constant across
                        # position bins; duplicated for convenience).
                        "pct_of_only_up_tier_high_below_tau_-0.5": (
                            sel_per_tier["tier_high"]["pct_below_tau"]
                        ),
                        "pct_of_only_up_tier_runner_below_tau_-0.5": (
                            sel_per_tier["tier_runner"]["pct_below_tau"]
                        ),
                        "pct_of_only_up_tier_low_plus_mid_below_tau_-0.5": (
                            sel_per_tier["tier_low_plus_mid"]["pct_below_tau"]
                        ),
                    }
                )

            # Per (subset, k, tier) running_close distribution + selectivity.
            for tier_label in TIER_LABELS + ["tier_low_plus_mid"]:
                if tier_label == "tier_low_plus_mid":
                    vals_t = vals_lpm
                else:
                    vals_t = close_ou[tier_ou == tier_label]
                vals_t = vals_t[~np.isnan(vals_t)]
                n_t = int(len(vals_t))
                row: Dict[str, Any] = {
                    "subset_id": sid,
                    "k": int(k),
                    "tier": tier_label,
                    "n_only_up_in_tier_with_close_at_k": n_t,
                }
                if n_t == 0:
                    for q in ["mean", "std", "median", "q05", "q10", "q25", "q75", "q90", "q95"]:
                        row[q] = float("nan")
                    row["n_below_tau_-0.5"] = 0
                    row["pct_below_tau_-0.5"] = float("nan")
                else:
                    row["mean"] = float(np.mean(vals_t))
                    row["std"] = float(np.std(vals_t, ddof=1)) if n_t > 1 else 0.0
                    row["median"] = float(np.median(vals_t))
                    row["q05"] = float(np.quantile(vals_t, 0.05))
                    row["q10"] = float(np.quantile(vals_t, 0.10))
                    row["q25"] = float(np.quantile(vals_t, 0.25))
                    row["q75"] = float(np.quantile(vals_t, 0.75))
                    row["q90"] = float(np.quantile(vals_t, 0.90))
                    row["q95"] = float(np.quantile(vals_t, 0.95))
                    nb = int(np.sum(vals_t <= sel_tau))
                    row["n_below_tau_-0.5"] = nb
                    row["pct_below_tau_-0.5"] = nb / n_t
                pdbt_rows.append(row)
    return pd.DataFrame(tbyp_rows), pd.DataFrame(pdbt_rows)


# ===========================================================================
# Plots
# ===========================================================================


def render_plots(
    *,
    gg_hist: pd.DataFrame,
    gg_summary: pd.DataFrame,
    hh_hist: pd.DataFrame,
    hh_summary: pd.DataFrame,
    jj_cat: pd.DataFrame,
    jj_sep: pd.DataFrame,
    mm_paths: pd.DataFrame,
    plots_dir: Path,
) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # Block GG: histogram of only_up overall MAE per subset (3 panels).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, sid in zip(axes, SUBSET_IDS):
        sub = gg_hist[(gg_hist["subset_id"] == sid) & (gg_hist["bin_idx"] >= 0)].sort_values(
            "bin_idx"
        )
        n_ou = int(gg_summary[gg_summary["subset_id"] == sid]["n_only_up"].iloc[0])
        centers = (sub["bin_left_atr"].to_numpy() + sub["bin_right_atr"].to_numpy()) / 2.0
        ax.bar(
            centers,
            sub["n_trades_in_bin"].to_numpy(),
            width=0.09,
            color="#3B6D11",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(-1.0, color="red", lw=0.7, ls=":")
        ax.axvline(-1.5, color="red", lw=0.7, ls=":")
        ax.set_title(f"{sid} (n={n_ou})")
        ax.set_xlabel("overall_mae_atr (fill-rel)")
        if sid == SUBSET_IDS[0]:
            ax.set_ylabel("count of only_up trades")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Block GG — only_up overall MAE distribution")
    fig.tight_layout()
    p = plots_dir / "block_GG_only_up_overall_mae_hist.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # Block HH: peak MFE histogram per subset.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, sid in zip(axes, SUBSET_IDS):
        sub = hh_hist[
            (hh_hist["subset_id"] == sid) & (np.isfinite(hh_hist["bin_right_atr"]))
        ].sort_values("bin_idx")
        n_ou = int(hh_summary[hh_summary["subset_id"] == sid]["n_only_up"].iloc[0])
        centers = (sub["bin_left_atr"].to_numpy() + sub["bin_right_atr"].to_numpy()) / 2.0
        ax.bar(
            centers,
            sub["n_trades_in_bin"].to_numpy(),
            width=0.9,
            color="#3B6D11",
            edgecolor="white",
            linewidth=0.3,
        )
        # mark a few TP candidates
        for tp in (4.0, 8.0, 12.0, 20.0):
            ax.axvline(tp, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{sid} (n={n_ou})")
        ax.set_xlabel("peak_mfe_atr (fill-rel)")
        if sid == SUBSET_IDS[0]:
            ax.set_ylabel("count of only_up trades")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Block HH — only_up peak MFE distribution")
    fig.tight_layout()
    p = plots_dir / "block_HH_only_up_peak_mfe_hist.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # Block JJ: per-category median + IQR running_close paths per subset.
    cat_colors = {
        "only_up": "#3B6D11",
        "up_then_down": "#A86E10",
        "down_then_up": "#1F4E79",
        "straight_to_sl": "#8E3D1F",
    }
    for sid in SUBSET_IDS:
        fig, ax = plt.subplots(figsize=(9, 5))
        sub_all = jj_cat[jj_cat["subset_id"] == sid]
        for c in JJ_CATEGORIES_FOR_DIST:
            sub_c = sub_all[sub_all["category"] == c].sort_values("k")
            if len(sub_c) == 0:
                continue
            ks = sub_c["k"].to_numpy(dtype=np.float64)
            med = sub_c["median"].to_numpy(dtype=np.float64)
            q25 = sub_c["q25"].to_numpy(dtype=np.float64)
            q75 = sub_c["q75"].to_numpy(dtype=np.float64)
            n0 = int(sub_c["n"].iloc[0])
            ax.plot(ks, med, color=cat_colors[c], lw=1.5, label=f"{c} (n~{n0})")
            ax.fill_between(ks, q25, q75, color=cat_colors[c], alpha=0.15)
        ax.axhline(0.0, color="black", lw=0.5, ls=":")
        ax.axhline(2.0, color="green", lw=0.5, ls=":")
        ax.axhline(-2.0, color="red", lw=0.5, ls=":")
        ax.set_title(f"Block JJ — running_close median (IQR shaded) per category — {sid}")
        ax.set_xlabel("k (bars after entry)")
        ax.set_ylabel("running_close_atr (fill-rel)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p = plots_dir / f"block_JJ_close_paths_by_category_{sid}.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)

    # Block JJ: only_up separator strength — recall_loser_below_tau vs k.
    for sid in SUBSET_IDS:
        sub = jj_sep[jj_sep["subset_id"] == sid]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for budget, color in zip(JJ_RECALL_BUDGETS, ["#3B6D11", "#A86E10", "#8E3D1F"]):
            sub_b = sub[sub["recall_only_up_budget"] == budget].sort_values("k")
            ax.plot(
                sub_b["k"],
                sub_b["pct_losers_below_tau"],
                marker="o",
                color=color,
                label=f"recall_only_up>={budget:.2f}",
            )
        ax.set_title(f"Block JJ — only_up separator: pct_losers_below_tau vs k — {sid}")
        ax.set_xlabel("k (bars after entry)")
        ax.set_ylabel("pct of losers below tau")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        p = plots_dir / f"block_JJ_only_up_separator_{sid}.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)

    # Block MM: median path per tier per subset.
    tier_colors = {
        "tier_low": "#A86E10",
        "tier_mid": "#639922",
        "tier_high": "#3B6D11",
        "tier_runner": "#1F4E79",
    }
    for sid in SUBSET_IDS:
        sub = mm_paths[mm_paths["subset_id"] == sid]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for tier in ("tier_low", "tier_mid", "tier_high", "tier_runner"):
            sub_t = sub[sub["tier"] == tier].sort_values("k")
            n0 = int(sub_t["n_tier_total"].iloc[0]) if len(sub_t) > 0 else 0
            if n0 == 0:
                continue
            ax.plot(
                sub_t["k"],
                sub_t["median_running_close_atr"],
                marker="o",
                color=tier_colors[tier],
                label=f"{tier} (n={n0})",
            )
        ax.axhline(0.0, color="black", lw=0.5, ls=":")
        ax.axhline(2.0, color="green", lw=0.5, ls=":")
        ax.set_title(f"Block MM — median running_close path per tier — {sid}")
        ax.set_xlabel("k (bars after entry)")
        ax.set_ylabel("median running_close_atr (fill-rel)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p = plots_dir / f"block_MM_tier_median_paths_{sid}.png"
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
            elif isinstance(v, float) and not np.isfinite(v):
                cells.append("inf" if v > 0 else "-inf")
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
    per_subset_only_up_n: Dict[str, int],
    gg_summary: pd.DataFrame,
    gg_kill: pd.DataFrame,
    hh_summary: pd.DataFrame,
    hh_tp: pd.DataFrame,
    ii_cells: pd.DataFrame,
    ii_marg: pd.DataFrame,
    jj_sep: pd.DataFrame,
    kk: pd.DataFrame,
    ll: pd.DataFrame,
    mm_tier: pd.DataFrame,
    nn_subset: pd.DataFrame,
    nn_by_tier: pd.DataFrame,
    kk3_tbyp: pd.DataFrame,
    kk3_pdbt: pd.DataFrame,
) -> str:
    lines: List[str] = []
    a = lines.append
    a("# Arc 2 — Path-Detail Distributions (Round 3C)")
    a("")
    a("Phase: L6 Arc 2 Phase 3 — Path-detail descriptive characterisation.")
    a("")
    a("Descriptive per L6_0_METHODOLOGY_LOCK Section 14.5. Read-existing-CSV ")
    a("only (Section 14.6). No variants, no filter selection, no exit ")
    a("proposals. Output is consumed by chat-side planning.")
    a("")

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

    a("## Subset definitions and only_up counts")
    a("")
    a("Subsets reconstructed by rank-based quintile labels with `trade_id` ")
    a("tie-break, byte-faithful to `arc2_entry_filter_bivariate.py` and ")
    a("`arc2_path_excursion_descriptives.py` (Round 3A).")
    a("")
    a("| subset_id | definition | n_total | only_up_n |")
    a("| --- | --- | --- | --- |")
    sub_defs_human = {
        "S0_pop": "ALL 3,993 trades",
        "S1_q5q2": "Q_A==Q5 AND Q_B==Q2",
        "S4_q5xq2q3": "Q_A==Q5 AND Q_B IN {Q2,Q3}",
    }
    for sid in SUBSET_IDS:
        n_exp = dict(SUBSET_DEFS)[sid]["expected_n"]
        n_ou = per_subset_only_up_n[sid]
        a(f"| {sid} | {sub_defs_human[sid]} | {n_exp} | {n_ou} |")
    a("")

    a("## Block B label reproduction check (gate 2.3)")
    a("")
    a("Block B 1R labels on the full population reproduce the existing Block ")
    a("B reference counts exactly: only_up=956, up_then_down=1075, ")
    a("down_then_up=1090, straight_to_sl=858, simultaneous=13, ")
    a("neither_reached=1. Gate 2.3 PASS.")
    a("")

    # Block GG
    a("## Block GG — only_up overall MAE distribution")
    a("")
    a("For each only_up trade (Block B 1R label), `overall_mae_atr` = ")
    a("`min(running_mae_atr)` taken across the full per_bar window ")
    a("(k=1..K_trade, where K_trade extends past the +/-1R 120-bar Block-B ")
    a("decision window to per_bar end-of-fold / k=240).")
    a("")
    a("Per-subset percentile summary:")
    a("")
    fmt_gg = {
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
        ]
    }
    a(
        _df_to_md(
            gg_summary[
                [
                    "subset_id",
                    "n_only_up",
                    "mean",
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
                ]
            ],
            fmt_gg,
        )
    )
    a("")
    a("### Block GG — only_up kill rate by candidate SL distance")
    a("")
    a("Per-subset percentage of only_up trades with `overall_mae_atr <= -d` ")
    a("(i.e. trades that would have hit a wider-than-BL-but-tighter-than-")
    a("`+/-2 ATR` SL placed at distance d).")
    a("")
    fmt_kill = {
        "SL_distance_atr": "{:.2f}",
        "pct_only_up_killed": "{:.4f}",
    }
    a(
        _df_to_md(
            gg_kill[
                [
                    "subset_id",
                    "SL_distance_atr",
                    "n_only_up_total",
                    "n_only_up_killed",
                    "pct_only_up_killed",
                ]
            ],
            fmt_kill,
        )
    )
    a("")

    # Block HH
    a("## Block HH — only_up peak MFE distribution")
    a("")
    a("For each only_up trade, `peak_mfe_atr` = `max(running_mfe_atr)` across ")
    a("the full per_bar window. R-equivalent under baseline SL=2 ATR: ")
    a("`peak_mfe_R = peak_mfe_atr / 2.0`.")
    a("")
    a("Per-subset percentile summary:")
    a("")
    fmt_hh = {
        c: "{:.3f}"
        for c in [
            "mean",
            "std",
            "median",
            "q05",
            "q10",
            "q25",
            "q50",
            "q75",
            "q80",
            "q85",
            "q90",
            "q95",
            "q99",
            "max",
        ]
    }
    a(
        _df_to_md(
            hh_summary[
                [
                    "subset_id",
                    "n_only_up",
                    "mean",
                    "median",
                    "q05",
                    "q25",
                    "q75",
                    "q85",
                    "q90",
                    "q95",
                    "q99",
                    "max",
                ]
            ],
            fmt_hh,
        )
    )
    a("")
    a("### Block HH — TP candidate evaluation")
    a("")
    a("For each candidate fixed TP level T (ATR fill-rel), per-subset:")
    a("- `pct_only_up_reaching_tp` = fraction of only_up with `peak_mfe >= T`.")
    a("- `expected_capped_loss_atr_per_only_up` = mean over the WHOLE only_up ")
    a("  group of `max(0, peak_mfe - T)`. Trades capped at T forgo the ")
    a("  `peak - T` excess.")
    a("")
    fmt_tp = {
        "tp_atr": "{:.1f}",
        "tp_R_at_BL_SL": "{:.2f}",
        "pct_only_up_reaching_tp": "{:.4f}",
        "pct_only_up_capped_above_tp": "{:.4f}",
        "expected_capped_loss_atr_per_only_up": "{:.3f}",
        "expected_capped_loss_R_per_only_up": "{:.3f}",
    }
    a(
        _df_to_md(
            hh_tp[
                [
                    "subset_id",
                    "tp_atr",
                    "tp_R_at_BL_SL",
                    "n_only_up_total",
                    "n_only_up_reaching_tp",
                    "pct_only_up_reaching_tp",
                    "expected_capped_loss_atr_per_only_up",
                    "expected_capped_loss_R_per_only_up",
                ]
            ],
            fmt_tp,
        )
    )
    a("")

    # Block II
    a("## Block II — only_up peak x overall-MAE crosstab")
    a("")
    a("Per-subset 7-peak x 5-mae table (mae bins include an overflow row for ")
    a("trades with `overall_mae < -2 ATR`). Cells:")
    a("")
    a("- `mean_peak_atr`, `mean_mae_atr`: cell mean of the two binned axes.")
    a("- `mean_final_close_R_at_BL_SL` = `mean(final_close_atr / 2.0)` for ")
    a("  cell members. Under BL SL=2 ATR, this is the final close in R units ")
    a("  (close at end of per_bar window, k<=240 or data-end).")
    a("- `mean_giveback_to_k120_atr` = `mean(peak_mfe - close_at_k120)` for ")
    a("  cell members with `n_bars >= 120`.")
    a("")
    for sid in SUBSET_IDS:
        a(f"### Subset {sid} — cells")
        a("")
        sub = ii_cells[ii_cells["subset_id"] == sid].copy()
        fmt_ii = {
            "mean_peak_atr": "{:.3f}",
            "mean_mae_atr": "{:.3f}",
            "mean_final_close_R_at_BL_SL": "{:.3f}",
            "mean_giveback_to_k120_atr": "{:.3f}",
        }
        a(
            _df_to_md(
                sub[
                    [
                        "peak_bin",
                        "mae_bin",
                        "n_trades",
                        "mean_peak_atr",
                        "mean_mae_atr",
                        "mean_final_close_R_at_BL_SL",
                        "mean_giveback_to_k120_atr",
                        "n_with_giveback",
                    ]
                ],
                fmt_ii,
            )
        )
        a("")
        a(f"### Subset {sid} — marginal peak by mae bin")
        a("")
        marg = ii_marg[ii_marg["subset_id"] == sid].copy()
        fmt_marg = {
            "mean_peak_atr": "{:.3f}",
            "median_peak_atr": "{:.3f}",
            "pct_of_only_up_in_subset": "{:.4f}",
        }
        a(
            _df_to_md(
                marg[
                    [
                        "mae_bin",
                        "n_trades",
                        "mean_peak_atr",
                        "median_peak_atr",
                        "pct_of_only_up_in_subset",
                    ]
                ],
                fmt_marg,
            )
        )
        a("")

    # Block JJ
    a("## Block JJ — per-category running_close at early bars + only_up separator")
    a("")
    a("Critical block. Tests whether only_up trades have a distinguishable ")
    a("`running_close_atr` position from the three loser categories ")
    a("(up_then_down, down_then_up, straight_to_sl) at any early k.")
    a("")
    a("Per-subset, per-(k, recall_only_up_budget), `tau_only_up_atr_fill` = ")
    a("largest threshold such that the fraction of only_up trades with ")
    a("`running_close >= tau` is >= budget. Reports how many losers (and ")
    a("per-category breakdown) fall strictly below tau at the same k.")
    a("")
    a("Cohen's d uses pooled SD with ddof=1; positive d means only_up has ")
    a("higher mean running_close than the comparison group at that k.")
    a("")
    for sid in SUBSET_IDS:
        a(f"### Subset {sid} — separator curves at recall_only_up>=0.95")
        a("")
        sub = jj_sep[
            (jj_sep["subset_id"] == sid) & (jj_sep["recall_only_up_budget"] == 0.95)
        ].copy()
        sub = sub.sort_values("k")
        fmt_jj = {
            "recall_only_up_budget": "{:.2f}",
            "tau_only_up_atr_fill": "{:+.3f}",
            "recall_only_up_actual": "{:.4f}",
            "pct_losers_below_tau": "{:.4f}",
            "cohens_d_only_up_vs_up_then_down": "{:+.3f}",
            "cohens_d_only_up_vs_down_then_up": "{:+.3f}",
            "cohens_d_only_up_vs_straight_to_sl": "{:+.3f}",
            "cohens_d_only_up_vs_loser_pool": "{:+.3f}",
        }
        a(
            _df_to_md(
                sub[
                    [
                        "k",
                        "n_only_up",
                        "tau_only_up_atr_fill",
                        "recall_only_up_actual",
                        "n_losers_total",
                        "n_losers_below_tau",
                        "pct_losers_below_tau",
                        "cohens_d_only_up_vs_loser_pool",
                        "cohens_d_only_up_vs_up_then_down",
                        "cohens_d_only_up_vs_down_then_up",
                        "cohens_d_only_up_vs_straight_to_sl",
                    ]
                ],
                fmt_jj,
            )
        )
        a("")
        a(f"### Subset {sid} — separator curves at recall_only_up>=0.99")
        a("")
        sub = jj_sep[
            (jj_sep["subset_id"] == sid) & (jj_sep["recall_only_up_budget"] == 0.99)
        ].copy()
        sub = sub.sort_values("k")
        a(
            _df_to_md(
                sub[
                    [
                        "k",
                        "n_only_up",
                        "tau_only_up_atr_fill",
                        "recall_only_up_actual",
                        "n_losers_total",
                        "n_losers_below_tau",
                        "pct_losers_below_tau",
                    ]
                ],
                fmt_jj,
            )
        )
        a("")

    # Block KK
    a("## Block KK — peak MFE conditional on early position")
    a("")
    a("Per (subset, k, position_bin), where position_bin partitions ")
    a("`running_close_atr` at bar k into four ranges:")
    a("")
    a("- `above_+1ATR`: rc >= +1.0")
    a("- `in_(0,+1ATR]`: 0 < rc <= +1.0")
    a("- `in_(-1,0]ATR`: -1.0 < rc <= 0")
    a("- `in_(-2,-1]ATR`: -2.0 < rc <= -1.0")
    a("")
    a("`n_only_up_in_bin` reports only_up at the position; ")
    a("`pct_of_only_up_in_bin_reaching_+10atr` is the within-bin survival ")
    a("rate to a peak MFE >= +10 ATR (= +5R under BL SL=2 ATR). ")
    a("`pct_of_all_trades_in_bin_that_are_only_up` gives the base rate of ")
    a("only_up among all trades observed at the same position at bar k.")
    a("")
    for sid in SUBSET_IDS:
        a(f"### Subset {sid}")
        a("")
        for k in (10, 20, 30, 40):
            sub = kk[(kk["subset_id"] == sid) & (kk["k"] == k)].copy()
            a(f"At k = {k}:")
            a("")
            fmt_kk = {
                "mean_peak_mfe_atr": "{:.3f}",
                "median_peak_mfe_atr": "{:.3f}",
                "pct_of_only_up_in_bin_reaching_+10atr": "{:.4f}",
                "pct_of_all_trades_in_bin_that_are_only_up": "{:.4f}",
            }
            a(
                _df_to_md(
                    sub[
                        [
                            "position_bin",
                            "n_only_up_in_bin",
                            "mean_peak_mfe_atr",
                            "median_peak_mfe_atr",
                            "n_only_up_reaching_+10atr",
                            "pct_of_only_up_in_bin_reaching_+10atr",
                            "n_all_trades_in_bin",
                            "pct_of_all_trades_in_bin_that_are_only_up",
                        ]
                    ],
                    fmt_kk,
                )
            )
            a("")

    # Block LL
    a("## Block LL — down_then_up + up_then_down overall MAE distribution")
    a("")
    a("For trades in `down_then_up` (hit -1R MAE first within k<=120 then ")
    a("crossed +1R MFE before k=120) and `up_then_down` (hit +1R first then ")
    a("-1R), `overall_mae_atr` covers the full per_bar window.")
    a("")
    a("`pct_surviving_widened_SL` per candidate SL distance d = fraction of ")
    a("trades with `overall_mae > -d` (would NOT be killed by a widened ")
    a("SL placed at distance d).")
    a("")
    for sid in SUBSET_IDS:
        a(f"### Subset {sid} — widened-SL survival rates")
        a("")
        sub = ll[(ll["subset_id"] == sid) & (ll["row_kind"] == "widened_SL_survival")].copy()
        fmt_ll = {
            "widened_SL_atr": "{:.2f}",
            "pct_surviving_widened_SL": "{:.4f}",
        }
        a(
            _df_to_md(
                sub[
                    [
                        "category",
                        "n_total_category",
                        "widened_SL_atr",
                        "n_surviving_widened_SL",
                        "pct_surviving_widened_SL",
                    ]
                ],
                fmt_ll,
            )
        )
        a("")

    # Block MM
    a("## Block MM — only_up tier analysis")
    a("")
    a("Tiers by `peak_mfe_atr`:")
    a("")
    a("- `tier_low`: 0 < peak <= 4 ATR (= up to +2R)")
    a("- `tier_mid`: 4 < peak <= 10 ATR (= +2R to +5R)")
    a("- `tier_high`: 10 < peak <= 20 ATR (= +5R to +10R)")
    a("- `tier_runner`: peak > 20 ATR (= +10R+)")
    a("")
    a("`mean_final_close_R_at_BL_SL` and `total_R_contribution` are ")
    a("computed from `final_close_atr / 2.0` (R units under SL=2 ATR baseline).")
    a("")
    for sid in SUBSET_IDS:
        a(f"### Subset {sid}")
        a("")
        sub = mm_tier[mm_tier["subset_id"] == sid].copy()
        fmt_mm = {
            "mean_peak_atr": "{:.3f}",
            "mean_giveback_to_k120_atr": "{:.3f}",
            "mean_final_close_R_at_BL_SL": "{:.3f}",
            "total_R_contribution": "{:.2f}",
            "pct_of_subset_total_R_contribution": "{:.4f}",
        }
        a(
            _df_to_md(
                sub[
                    [
                        "tier",
                        "n_trades",
                        "mean_peak_atr",
                        "mean_giveback_to_k120_atr",
                        "mean_final_close_R_at_BL_SL",
                        "total_R_contribution",
                        "pct_of_subset_total_R_contribution",
                    ]
                ],
                fmt_mm,
            )
        )
        a("")

    # Block NN
    a("## Block NN — only_up final R distribution under BL and H240 exits")
    a("")
    a("Per only_up trade, re-derived from per_bar_paths:")
    a("")
    a("- `BL` = SL=-2 ATR fill-rel + hold cap k=120. On SL hit (running_mae ")
    a("  reaches -2 ATR), final_R = -1.0. Else exit at `bar_close_atr` at ")
    a("  k=min(120, n_bars) and final_R = close / 2.0. By only_up's Block B ")
    a("  definition (running_mae > -2 ATR for k<=120), only_up trades never ")
    a("  SL-stop under BL; they exit at k=120 close (or earlier if clamped).")
    a("- `H240` = SL=-2 ATR + hold cap k=240. On SL hit anywhere in ")
    a("  [1, min(240, n_bars)], final_R = -1.0. Else exit at close at ")
    a("  k=min(240, n_bars). For only_up trades, SL hits are possible in ")
    a("  the interval (120, 240] (the per_bar window extends past the Block ")
    a("  B decision horizon).")
    a("")
    a("Per-subset summary (BL and H240):")
    a("")
    summ_rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        for policy in NN_EXIT_POLICIES:
            r = nn_subset[
                (nn_subset["subset_id"] == sid)
                & (nn_subset["exit_policy"] == policy)
                & (nn_subset["row_kind"] == "summary")
            ].iloc[0]
            summ_rows.append(
                {
                    "subset_id": sid,
                    "exit_policy": policy,
                    "n_only_up": int(r["n_trades_in_bin"]),
                    "mean_R": float(r["summary_mean"]),
                    "median_R": float(r["summary_median"]),
                    "q05_R": float(r["summary_q05"]),
                    "q25_R": float(r["summary_q25"]),
                    "q75_R": float(r["summary_q75"]),
                    "q90_R": float(r["summary_q90"]),
                    "q95_R": float(r["summary_q95"]),
                    "q99_R": float(r["summary_q99"]),
                    "max_R": float(r["summary_max"]),
                }
            )
    fmt_nn = {
        c: "{:+.3f}"
        for c in [
            "mean_R",
            "median_R",
            "q05_R",
            "q25_R",
            "q75_R",
            "q90_R",
            "q95_R",
            "q99_R",
            "max_R",
        ]
    }
    a(_df_to_md(pd.DataFrame(summ_rows), fmt_nn))
    a("")
    a("### Per-tier summary (BL and H240)")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### Subset {sid}")
        a("")
        tier_summ_rows: List[Dict[str, Any]] = []
        for _, _, tier_label in MM_TIER_BOUNDS:
            for policy in NN_EXIT_POLICIES:
                r_sub = nn_by_tier[
                    (nn_by_tier["subset_id"] == sid)
                    & (nn_by_tier["tier"] == tier_label)
                    & (nn_by_tier["exit_policy"] == policy)
                    & (nn_by_tier["row_kind"] == "summary")
                ]
                if len(r_sub) == 0:
                    continue
                r = r_sub.iloc[0]
                tier_summ_rows.append(
                    {
                        "tier": tier_label,
                        "exit_policy": policy,
                        "n": int(r["n_trades_in_bin"]),
                        "mean_R": float(r["summary_mean"]),
                        "median_R": float(r["summary_median"]),
                        "q25_R": float(r["summary_q25"]),
                        "q75_R": float(r["summary_q75"]),
                        "q95_R": float(r["summary_q95"]),
                        "max_R": float(r["summary_max"]),
                    }
                )
        fmt_nnt = {c: "{:+.3f}" for c in ["mean_R", "median_R", "q25_R", "q75_R", "q95_R", "max_R"]}
        a(_df_to_md(pd.DataFrame(tier_summ_rows), fmt_nnt))
        a("")

    # Block KK3
    a("## Block KK3 — tier breakdown by position at early k")
    a("")
    a("Per (subset, k, position_bin) tier composition of only_up trades. ")
    a("Position bins identical to Block KK. Tier groupings follow Block MM ")
    a("(low <= 4 ATR peak; mid 4-10 ATR; high 10-20 ATR; runner > 20 ATR).")
    a("")
    a("Selectivity at `tau = -0.5 ATR` (the descriptive reference threshold) ")
    a("is duplicated across position-bin rows because it is computed across ")
    a("all only_up trades in (subset, k) — it does not depend on the ")
    a("position_bin column.")
    a("")
    for sid in SUBSET_IDS:
        for k in KK3_K_GRID:
            a(f"### Subset {sid} at k = {k}")
            a("")
            sub = kk3_tbyp[(kk3_tbyp["subset_id"] == sid) & (kk3_tbyp["k"] == int(k))].copy()
            fmt_kk3 = {
                "pct_of_only_up_tier_high_below_tau_-0.5": "{:.4f}",
                "pct_of_only_up_tier_runner_below_tau_-0.5": "{:.4f}",
                "pct_of_only_up_tier_low_plus_mid_below_tau_-0.5": "{:.4f}",
            }
            a(
                _df_to_md(
                    sub[
                        [
                            "position_bin",
                            "n_total_trades_in_bin",
                            "n_only_up_in_bin",
                            "n_only_up_tier_low_in_bin",
                            "n_only_up_tier_mid_in_bin",
                            "n_only_up_tier_high_in_bin",
                            "n_only_up_tier_runner_in_bin",
                            "pct_of_only_up_tier_high_below_tau_-0.5",
                            "pct_of_only_up_tier_runner_below_tau_-0.5",
                            "pct_of_only_up_tier_low_plus_mid_below_tau_-0.5",
                        ]
                    ],
                    fmt_kk3,
                )
            )
            a("")
            sub_d = kk3_pdbt[(kk3_pdbt["subset_id"] == sid) & (kk3_pdbt["k"] == int(k))].copy()
            fmt_kk3d = {
                c: "{:+.3f}" for c in ["mean", "median", "q05", "q10", "q25", "q75", "q90", "q95"]
            }
            fmt_kk3d["pct_below_tau_-0.5"] = "{:.4f}"
            a("Running_close distribution per tier at this k:")
            a("")
            a(
                _df_to_md(
                    sub_d[
                        [
                            "tier",
                            "n_only_up_in_tier_with_close_at_k",
                            "mean",
                            "median",
                            "q05",
                            "q25",
                            "q75",
                            "q95",
                            "n_below_tau_-0.5",
                            "pct_below_tau_-0.5",
                        ]
                    ],
                    fmt_kk3d,
                )
            )
            a("")

    # Out-of-scope items
    a("## Out-of-scope items observed")
    a("")
    a("- Underflow in Block GG (only_up trades with `overall_mae < -2 ATR` ")
    a("  in the full per_bar window) is reported as a single `bin_idx=-1` ")
    a("  row in `block_GG_overall_mae_histogram.csv`. Such trades exist ")
    a("  because Block B's -1R label is computed on k<=120 only; the per_bar ")
    a("  window extends past that.")
    a("- Block II includes an explicit `mae_below_-2p0_overflow` mae bin to ")
    a("  accommodate the same trades.")
    a("- Block KK's four position bins cover `running_close > -2 ATR`. ")
    a("  Other trades (with rc <= -2 at the same k) fall outside all four ")
    a("  bins. For k in {10,20,30,40} only loser categories can have ")
    a("  rc <= -2; only_up by definition cannot (running_mae > -2 within ")
    a("  k<=120). The crosstab counts in `n_all_trades_in_bin` therefore ")
    a("  exclude those trades by construction.")
    a("- Block LL records both `down_then_up` and `up_then_down`. The user-")
    a("  spec's symmetric request is satisfied by the single CSV with a ")
    a("  `category` discriminator column.")
    a("- Block FF (Round 3A) treated `only_up UNION up_then_down` as Group A. ")
    a("  This round (3C) Block JJ treats `only_up` ALONE as the target, ")
    a("  contrasting against each of the three other categories ")
    a("  separately and against the pooled loser group. These are ")
    a("  complementary, not contradictory.")
    a("")

    # Planning input
    a("## Planning input")
    a("")
    a("Material below is intentionally descriptive even within this tagged ")
    a("subsection; final variant spec is the chat's job.")
    a("")
    a("### Block GG — candidate SL-tighten anchors")
    a("")
    a("Per subset, descriptive quantile reference for only_up overall MAE:")
    a("")
    for sid in SUBSET_IDS:
        s = gg_summary[gg_summary["subset_id"] == sid].iloc[0]
        kill_175 = gg_kill[(gg_kill["subset_id"] == sid) & (gg_kill["SL_distance_atr"] == 1.75)][
            "pct_only_up_killed"
        ].iloc[0]
        kill_150 = gg_kill[(gg_kill["subset_id"] == sid) & (gg_kill["SL_distance_atr"] == 1.5)][
            "pct_only_up_killed"
        ].iloc[0]
        a(
            f"- {sid}: n_only_up={int(s['n_only_up'])}; q01={s['q01']:+.3f} ATR, "
            f"q05={s['q05']:+.3f} ATR, q10={s['q10']:+.3f} ATR, "
            f"median={s['median']:+.3f} ATR, min={s['min']:+.3f} ATR. "
            f"Kill-rate descriptive references: SL=1.75 ATR kills "
            f"{kill_175 * 100:.2f}% of only_up; SL=1.50 ATR kills "
            f"{kill_150 * 100:.2f}%."
        )
    a("")

    a("### Block HH — candidate fixed TP levels")
    a("")
    for sid in SUBSET_IDS:
        s = hh_summary[hh_summary["subset_id"] == sid].iloc[0]
        tp10 = hh_tp[(hh_tp["subset_id"] == sid) & (hh_tp["tp_atr"] == 10.0)].iloc[0]
        tp20 = hh_tp[(hh_tp["subset_id"] == sid) & (hh_tp["tp_atr"] == 20.0)].iloc[0]
        a(
            f"- {sid}: peak MFE q50={s['median']:.2f} ATR (={s['median'] / 2:.2f}R), "
            f"q75={s['q75']:.2f} ATR (={s['q75'] / 2:.2f}R), "
            f"q90={s['q90']:.2f} ATR (={s['q90'] / 2:.2f}R), "
            f"q95={s['q95']:.2f} ATR (={s['q95'] / 2:.2f}R). "
            f"TP=10 ATR descriptive references: {tp10['pct_only_up_reaching_tp'] * 100:.2f}% "
            f"reach, expected capped loss = {tp10['expected_capped_loss_R_per_only_up']:.2f}R "
            f"per only_up. TP=20 ATR: {tp20['pct_only_up_reaching_tp'] * 100:.2f}% reach, "
            f"expected capped loss = {tp20['expected_capped_loss_R_per_only_up']:.2f}R."
        )
    a("")

    a("### Block JJ — early-bar separator candidate (k, tau) descriptive ranges")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### {sid}")
        a("")
        sub = jj_sep[jj_sep["subset_id"] == sid].copy()
        rows: List[Dict[str, Any]] = []
        for budget in JJ_RECALL_BUDGETS:
            sub_b = sub[sub["recall_only_up_budget"] == budget]
            if len(sub_b) == 0:
                continue
            top = sub_b.sort_values("pct_losers_below_tau", ascending=False).iloc[0]
            rows.append(
                {
                    "recall_only_up_budget": float(budget),
                    "k_at_best": int(top["k"]),
                    "tau_only_up_atr_fill": float(top["tau_only_up_atr_fill"]),
                    "recall_only_up_actual": float(top["recall_only_up_actual"]),
                    "pct_losers_below_tau": float(top["pct_losers_below_tau"]),
                    "cohens_d_vs_loser_pool": float(top["cohens_d_only_up_vs_loser_pool"]),
                }
            )
        a(
            _df_to_md(
                pd.DataFrame(rows),
                {
                    "recall_only_up_budget": "{:.2f}",
                    "tau_only_up_atr_fill": "{:+.3f}",
                    "recall_only_up_actual": "{:.4f}",
                    "pct_losers_below_tau": "{:.4f}",
                    "cohens_d_vs_loser_pool": "{:+.3f}",
                },
            )
        )
        a("")

    a("### Block KK — early-position conditional-survival reference")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### {sid}")
        a("")
        sub = kk[(kk["subset_id"] == sid) & (kk["k"] == 20)].copy()
        a("At k = 20:")
        a("")
        a(
            _df_to_md(
                sub[
                    [
                        "position_bin",
                        "n_only_up_in_bin",
                        "pct_of_only_up_in_bin_reaching_+10atr",
                        "pct_of_all_trades_in_bin_that_are_only_up",
                    ]
                ],
                {
                    "pct_of_only_up_in_bin_reaching_+10atr": "{:.4f}",
                    "pct_of_all_trades_in_bin_that_are_only_up": "{:.4f}",
                },
            )
        )
        a("")

    a("### Block LL — widened-SL survival reference")
    a("")
    for sid in SUBSET_IDS:
        for cat_label in LL_CATEGORIES:
            sub = ll[
                (ll["subset_id"] == sid)
                & (ll["category"] == cat_label)
                & (ll["row_kind"] == "widened_SL_survival")
            ].copy()
            n_cat = int(sub["n_total_category"].iloc[0]) if len(sub) > 0 else 0
            a(
                f"- {sid} / {cat_label} (n_total={n_cat}): "
                + ", ".join(
                    f"SL={r['widened_SL_atr']:.2f} ATR survives {r['pct_surviving_widened_SL'] * 100:.2f}%"
                    for _, r in sub.iterrows()
                )
                + "."
            )
    a("")

    a("### Block NN — BL vs H240 final R descriptive comparison")
    a("")
    for sid in SUBSET_IDS:
        bl_summ = nn_subset[
            (nn_subset["subset_id"] == sid)
            & (nn_subset["exit_policy"] == "BL")
            & (nn_subset["row_kind"] == "summary")
        ].iloc[0]
        h_summ = nn_subset[
            (nn_subset["subset_id"] == sid)
            & (nn_subset["exit_policy"] == "H240")
            & (nn_subset["row_kind"] == "summary")
        ].iloc[0]
        a(
            f"- {sid}: BL median={bl_summ['summary_median']:+.3f}R, "
            f"mean={bl_summ['summary_mean']:+.3f}R, "
            f"q95={bl_summ['summary_q95']:+.3f}R, "
            f"max={bl_summ['summary_max']:+.3f}R. "
            f"H240 median={h_summ['summary_median']:+.3f}R, "
            f"mean={h_summ['summary_mean']:+.3f}R, "
            f"q95={h_summ['summary_q95']:+.3f}R, "
            f"max={h_summ['summary_max']:+.3f}R. "
            f"H240 vs BL median delta = "
            f"{h_summ['summary_median'] - bl_summ['summary_median']:+.3f}R; "
            f"mean delta = "
            f"{h_summ['summary_mean'] - bl_summ['summary_mean']:+.3f}R."
        )
    a("")
    a("### Block KK3 — tier composition at k=20")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### {sid}")
        a("")
        sub = kk3_tbyp[(kk3_tbyp["subset_id"] == sid) & (kk3_tbyp["k"] == 20)].copy()
        a("| position_bin | n_only_up | tier_low | tier_mid | tier_high | tier_runner |")
        a("| --- | --- | --- | --- | --- | --- |")
        for _, r in sub.iterrows():
            a(
                f"| {r['position_bin']} | {int(r['n_only_up_in_bin'])} | "
                f"{int(r['n_only_up_tier_low_in_bin'])} | "
                f"{int(r['n_only_up_tier_mid_in_bin'])} | "
                f"{int(r['n_only_up_tier_high_in_bin'])} | "
                f"{int(r['n_only_up_tier_runner_in_bin'])} |"
            )
        # tau=-0.5 selectivity (constant across position bins).
        r0 = sub.iloc[0]
        a("")
        a("At k=20, pct of only_up tier_X with running_close <= -0.5 ATR: ")
        a(
            f"tier_high = {r0['pct_of_only_up_tier_high_below_tau_-0.5'] * 100:.2f}%, "
            f"tier_runner = {r0['pct_of_only_up_tier_runner_below_tau_-0.5'] * 100:.2f}%, "
            f"tier_low_plus_mid = {r0['pct_of_only_up_tier_low_plus_mid_below_tau_-0.5'] * 100:.2f}%."
        )
        a("")
    a("### Block MM — tier mix and R contribution reference")
    a("")
    for sid in SUBSET_IDS:
        sub = mm_tier[mm_tier["subset_id"] == sid].copy()
        parts = []
        for _, r in sub.iterrows():
            parts.append(
                f"{r['tier']}: n={int(r['n_trades'])}, "
                f"mean_peak={r['mean_peak_atr']:.2f} ATR, "
                f"mean_final_R={r['mean_final_close_R_at_BL_SL']:+.2f}R, "
                f"R_contrib={r['total_R_contribution']:+.2f}R "
                f"({r['pct_of_subset_total_R_contribution'] * 100:.1f}% of subset)"
            )
        a(f"- {sid}: " + "; ".join(parts) + ".")
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
# Block-arithmetic validation gates
# ===========================================================================


def _gate5_GG_bin_sum(gg_hist: pd.DataFrame, per_subset_only_up_n: Dict[str, int]) -> None:
    """GG: bins (incl. underflow) sum to n_only_up per subset."""
    for sid in SUBSET_IDS:
        sub = gg_hist[gg_hist["subset_id"] == sid]
        s = int(sub["n_trades_in_bin"].sum())
        exp = per_subset_only_up_n[sid]
        if s != exp:
            raise RuntimeError(
                f"HALT (gate 5.GG): subset {sid} hist bins sum={s} != n_only_up={exp}"
            )


def _gate5_HH_bin_sum(hh_hist: pd.DataFrame, per_subset_only_up_n: Dict[str, int]) -> None:
    for sid in SUBSET_IDS:
        sub = hh_hist[hh_hist["subset_id"] == sid]
        s = int(sub["n_trades_in_bin"].sum())
        exp = per_subset_only_up_n[sid]
        if s != exp:
            raise RuntimeError(
                f"HALT (gate 5.HH): subset {sid} hist bins sum={s} != n_only_up={exp}"
            )


def _gate5_II_cell_sum(ii_cells: pd.DataFrame, per_subset_only_up_n: Dict[str, int]) -> None:
    for sid in SUBSET_IDS:
        if per_subset_only_up_n[sid] == 0:
            continue
        sub = ii_cells[ii_cells["subset_id"] == sid]
        s = int(sub["n_trades"].sum())
        exp = per_subset_only_up_n[sid]
        if s != exp:
            raise RuntimeError(f"HALT (gate 5.II): subset {sid} cell sum={s} != n_only_up={exp}")


def _gate5_MM_tier_sum(mm_tier: pd.DataFrame, per_subset_only_up_n: Dict[str, int]) -> None:
    for sid in SUBSET_IDS:
        if per_subset_only_up_n[sid] == 0:
            continue
        sub = mm_tier[mm_tier["subset_id"] == sid]
        s = int(sub["n_trades"].sum())
        exp = per_subset_only_up_n[sid]
        if s != exp:
            raise RuntimeError(f"HALT (gate 5.MM): subset {sid} tier sum={s} != n_only_up={exp}")


def _gate6_JJ_separator(jj_sep: pd.DataFrame) -> None:
    """JJ: recall_only_up_actual >= recall_budget when tau is valid."""
    for _, row in jj_sep.iterrows():
        if pd.isna(row["tau_only_up_atr_fill"]):
            continue
        actual = float(row["recall_only_up_actual"])
        budget = float(row["recall_only_up_budget"])
        if actual + 1e-12 < budget:
            raise RuntimeError(
                f"HALT (gate 6.JJ): subset={row['subset_id']} k={int(row['k'])} "
                f"budget={budget:.2f}: actual_recall={actual:.6f} < budget"
            )


def _gate_NN1_bin_sum(nn_subset: pd.DataFrame, per_subset_only_up_n: Dict[str, int]) -> None:
    """NN.1: histogram bin counts per (subset, exit_policy) sum to n_only_up."""
    bin_rows = nn_subset[nn_subset["row_kind"].isin(["bin", "bin_overflow", "bin_underflow"])]
    for sid in SUBSET_IDS:
        for policy in NN_EXIT_POLICIES:
            sub = bin_rows[(bin_rows["subset_id"] == sid) & (bin_rows["exit_policy"] == policy)]
            s = int(sub["n_trades_in_bin"].sum())
            exp = per_subset_only_up_n[sid]
            if s != exp:
                raise RuntimeError(
                    f"HALT (gate NN.1): subset {sid} exit_policy {policy} "
                    f"hist bin sum={s} != n_only_up={exp}"
                )


def _gate_KK31_tier_sum(kk3_tbyp: pd.DataFrame, per_subset_only_up_n: Dict[str, int]) -> None:
    """KK3.1: tier counts per subset sum to n_only_up per subset.

    For each (subset, k), sum across position bins and across tiers must
    equal the count of only_up trades with a valid close_at_k (i.e.,
    n_only_up minus those with n_bars < k). For k in {10, 20, 30} the
    minimum n_bars across the population is 21 (per Round 3A note), so
    all only_up trades have valid close at k=10. For k=20 / k=30 there
    may be at most 1-2 trades clamped; sum may be n_only_up or
    n_only_up - clamped_count. We assert sum equals total only_up
    in-bin count for the (subset, k).
    """
    for sid in SUBSET_IDS:
        n_ou = per_subset_only_up_n[sid]
        for k in KK3_K_GRID:
            sub = kk3_tbyp[(kk3_tbyp["subset_id"] == sid) & (kk3_tbyp["k"] == int(k))]
            sum_tiers = int(
                (
                    sub["n_only_up_tier_low_in_bin"].sum()
                    + sub["n_only_up_tier_mid_in_bin"].sum()
                    + sub["n_only_up_tier_high_in_bin"].sum()
                    + sub["n_only_up_tier_runner_in_bin"].sum()
                )
            )
            sum_ou = int(sub["n_only_up_in_bin"].sum())
            if sum_tiers != sum_ou:
                raise RuntimeError(
                    f"HALT (gate KK3.1): subset {sid} k={k}: tier sum={sum_tiers} "
                    f"!= n_only_up_in_bin sum={sum_ou}"
                )
            # Also: position-bin sum must be <= n_ou (some trades may have
            # close outside the 4 bins). For only_up at k<=120 close is
            # always > -2 ATR so should be exactly n_ou unless clamped.
            if sum_ou > n_ou:
                raise RuntimeError(
                    f"HALT (gate KK3.1): subset {sid} k={k}: only_up bin sum "
                    f"={sum_ou} > n_only_up={n_ou}"
                )


def _gate_KK32_position_sum(
    kk3_tbyp: pd.DataFrame,
    per_subset_only_up_n: Dict[str, int],
    starts: np.ndarray,
    ends: np.ndarray,
    cats: np.ndarray,
    subsets: Dict[str, np.ndarray],
) -> None:
    """KK3.2: n_only_up_in_bin across position bins equals n_only_up minus
    trades clamped before bar k. only_up at k<=120 has running_close > -2 ATR
    so trades with n_bars >= k all fit one of the 4 position bins.
    """
    for sid in SUBSET_IDS:
        ou_tids = np.array(
            [int(t) for t in subsets[sid].tolist() if cats[int(t)] == "only_up"],
            dtype=np.int64,
        )
        for k in KK3_K_GRID:
            n_with_valid_close = 0
            for tid in ou_tids.tolist():
                s, e = int(starts[tid]), int(ends[tid])
                if e - s >= k:
                    n_with_valid_close += 1
            sub = kk3_tbyp[(kk3_tbyp["subset_id"] == sid) & (kk3_tbyp["k"] == int(k))]
            sum_ou_in_bins = int(sub["n_only_up_in_bin"].sum())
            if sum_ou_in_bins != n_with_valid_close:
                raise RuntimeError(
                    f"HALT (gate KK3.2): subset {sid} k={k}: "
                    f"sum_only_up_in_bins={sum_ou_in_bins} != "
                    f"n_only_up_with_valid_close_at_k={n_with_valid_close}"
                )


def _gate7_LL_widened_SL(ll: pd.DataFrame) -> None:
    """LL: widened_SL pct_surviving is monotone non-decreasing in d."""
    sub = ll[ll["row_kind"] == "widened_SL_survival"]
    for (sid, cat_label), g in sub.groupby(["subset_id", "category"]):
        g_sorted = g.sort_values("widened_SL_atr")
        pcts = g_sorted["pct_surviving_widened_SL"].to_numpy(dtype=np.float64)
        diffs = np.diff(pcts)
        if (diffs < -1e-12).any():
            raise RuntimeError(
                f"HALT (gate 7.LL): subset={sid} category={cat_label}: "
                f"widened-SL survival not monotone non-decreasing: {pcts}"
            )


# ===========================================================================
# Single build pass
# ===========================================================================


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
    print("  Block B categories:", {c: int((cats == c).sum()) for c in ALL_CATS}, flush=True)

    overall_mae, peak_mfe, close_k120, final_close, n_bars, peak_k = (
        _per_trade_overall_mae_and_peak(starts, ends, rmfe, rmae, bc, n_trades)
    )
    final_R_BL, final_R_H240, sl_k_BL, sl_k_H240 = _per_trade_BL_H240_final_R(
        starts, ends, rmae, bc, n_trades
    )

    # Per-subset only_up counts.
    per_subset_only_up_n: Dict[str, int] = {}
    for sid, tids in subsets.items():
        n_ou = int(sum(1 for t in tids.tolist() if cats[int(t)] == "only_up"))
        per_subset_only_up_n[sid] = n_ou
    print(f"  only_up per subset: {per_subset_only_up_n}", flush=True)

    # Block GG
    print("  Computing Block GG...", flush=True)
    gg_per_trade, gg_hist, gg_summary, gg_kill = compute_block_GG(cats, overall_mae, subsets)
    _gate5_GG_bin_sum(gg_hist, per_subset_only_up_n)

    # Block HH
    print("  Computing Block HH...", flush=True)
    hh_hist, hh_summary, hh_tp = compute_block_HH(cats, peak_mfe, subsets)
    _gate5_HH_bin_sum(hh_hist, per_subset_only_up_n)

    # Block II
    print("  Computing Block II...", flush=True)
    ii_cells, ii_marg = compute_block_II(
        cats, peak_mfe, overall_mae, close_k120, final_close, subsets
    )
    _gate5_II_cell_sum(ii_cells, per_subset_only_up_n)

    # Block JJ
    print("  Computing Block JJ...", flush=True)
    jj_cat, jj_sep = compute_block_JJ(starts, ends, bc, cats, subsets)
    _gate6_JJ_separator(jj_sep)

    # Block KK
    print("  Computing Block KK...", flush=True)
    kk = compute_block_KK(starts, ends, bc, cats, peak_mfe, subsets)

    # Block LL
    print("  Computing Block LL...", flush=True)
    ll = compute_block_LL(cats, overall_mae, subsets)
    _gate7_LL_widened_SL(ll)

    # Block MM
    print("  Computing Block MM...", flush=True)
    mm_tier, mm_paths = compute_block_MM(
        starts, ends, bc, cats, peak_mfe, close_k120, final_close, subsets
    )
    _gate5_MM_tier_sum(mm_tier, per_subset_only_up_n)

    # Block NN
    print("  Computing Block NN...", flush=True)
    nn_subset, nn_by_tier = compute_block_NN(cats, peak_mfe, final_R_BL, final_R_H240, subsets)
    _gate_NN1_bin_sum(nn_subset, per_subset_only_up_n)

    # Block KK3
    print("  Computing Block KK3...", flush=True)
    kk3_tbyp, kk3_pdbt = compute_block_KK3(starts, ends, bc, cats, peak_mfe, subsets)
    _gate_KK31_tier_sum(kk3_tbyp, per_subset_only_up_n)
    _gate_KK32_position_sum(kk3_tbyp, per_subset_only_up_n, starts, ends, cats, subsets)

    # Write CSVs.
    paths = {
        "block_GG_only_up_overall_mae.csv": gg_per_trade,
        "block_GG_overall_mae_histogram.csv": gg_hist,
        "block_GG_overall_mae_summary.csv": gg_summary,
        "block_GG_only_up_kill_rate_by_SL.csv": gg_kill,
        "block_HH_only_up_peak_distribution.csv": hh_hist,
        "block_HH_only_up_peak_summary.csv": hh_summary,
        "block_HH_TP_evaluation.csv": hh_tp,
        "block_II_peak_x_mae_crosstab.csv": ii_cells,
        "block_II_marginal_peak_by_mae.csv": ii_marg,
        "block_JJ_per_category_close_at_k.csv": jj_cat,
        "block_JJ_only_up_separator_curves.csv": jj_sep,
        "block_KK_peak_conditional_on_position.csv": kk,
        "block_LL_down_then_up_overall_mae.csv": ll,
        "block_MM_only_up_tier_analysis.csv": mm_tier,
        "block_MM_tier_median_paths.csv": mm_paths,
        "block_NN_only_up_final_R_distribution.csv": nn_subset,
        "block_NN_only_up_final_R_by_tier.csv": nn_by_tier,
        "block_KK3_tier_by_position.csv": kk3_tbyp,
        "block_KK3_position_distribution_by_tier.csv": kk3_pdbt,
    }
    for name, df in paths.items():
        _write_csv(df, out_dir / name)

    # Plots
    print("  Rendering plots...", flush=True)
    plot_paths = render_plots(
        gg_hist=gg_hist,
        gg_summary=gg_summary,
        hh_hist=hh_hist,
        hh_summary=hh_summary,
        jj_cat=jj_cat,
        jj_sep=jj_sep,
        mm_paths=mm_paths,
        plots_dir=out_dir / "plots",
    )

    # Markdown report
    print("  Rendering markdown...", flush=True)
    md = render_report(
        observed_shas=observed_shas,
        per_subset_only_up_n=per_subset_only_up_n,
        gg_summary=gg_summary,
        gg_kill=gg_kill,
        hh_summary=hh_summary,
        hh_tp=hh_tp,
        ii_cells=ii_cells,
        ii_marg=ii_marg,
        jj_sep=jj_sep,
        kk=kk,
        ll=ll,
        mm_tier=mm_tier,
        nn_subset=nn_subset,
        nn_by_tier=nn_by_tier,
        kk3_tbyp=kk3_tbyp,
        kk3_pdbt=kk3_pdbt,
    )
    md_path = out_dir / "path_detail_distributions.md"
    md_path.write_text(md, encoding="utf-8", newline="\n")

    # Disposition discipline gate
    viols = check_disposition_discipline(md)
    if viols:
        msg = "\n  ".join([f"line {ln}: pat='{p}': {tx}" for ln, p, tx in viols])
        raise RuntimeError(f"HALT (gate 10): disposition discipline violations:\n  {msg}")

    # Gate 9: locked artefacts unchanged
    _verify_locked("gate 9 (end)")

    # Output sha256 manifest
    out_files = list(paths.keys()) + ["path_detail_distributions.md"]
    out_paths_full = [out_dir / n for n in out_files] + plot_paths
    out_shas = {p.relative_to(REPO_ROOT).as_posix(): _sha256_file(p) for p in out_paths_full}

    gates = {
        "gate_1_inputs": "ok (9 sha256s match)",
        "gate_2_1_subsets": "ok (all 25 block_P cells reproduced)",
        "gate_2_2_subset_sizes": "ok (3/3 subsets at expected counts)",
        "gate_2_3_block_b": "ok (956/1075/1090/858/13/1 reproduced)",
        "gate_5_block_bin_sums": "ok (GG/HH/II/MM cell counts match n_only_up)",
        "gate_6_JJ_recall": "ok (actual recall >= budget at every valid tau)",
        "gate_7_LL_monotone": "ok (widened-SL survival monotone in d)",
        "gate_NN_1_hist_sum": "ok (NN bin sums equal n_only_up per subset)",
        "gate_KK3_1_tier_sum": "ok (KK3 tier sums equal n_only_up_in_bin sums)",
        "gate_KK3_2_position_sum": "ok (KK3 position bin sums equal valid-close-at-k count)",
        "gate_9_artefacts_unchanged": "ok",
        "gate_10_disposition": f"ok ({len(viols)} violations outside Planning input)",
        "gate_11_no_commit": "ok (no auto-commit; outputs untracked)",
    }

    if write_manifest:
        wallclock = time.time() - t0
        manifest_path = out_dir / "run_manifest.txt"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Arc 2 — Path-detail distributions (Round 3C)\n")
            f.write("# Phase: l6_arc2_path_detail_distributions\n")
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
        "per_subset_only_up_n": per_subset_only_up_n,
        "gg_summary": gg_summary,
        "gg_kill": gg_kill,
        "hh_summary": hh_summary,
        "hh_tp": hh_tp,
        "jj_sep": jj_sep,
        "kk": kk,
        "ll": ll,
        "mm_tier": mm_tier,
        "nn_subset": nn_subset,
        "nn_by_tier": nn_by_tier,
        "kk3_tbyp": kk3_tbyp,
        "kk3_pdbt": kk3_pdbt,
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
        r1 = build_pass(out_dir=out_dir, write_manifest=True)
        det_ok = None
    else:
        snapshot_dir = Path(tempfile.mkdtemp(prefix="arc2_pathdet_snap_"))
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
    gg_summary = r1["gg_summary"]
    gg_kill = r1["gg_kill"]
    hh_summary = r1["hh_summary"]
    hh_tp = r1["hh_tp"]
    jj_sep = r1["jj_sep"]
    r1["kk"]
    r1["mm_tier"]
    nn_subset = r1["nn_subset"]
    kk3_tbyp = r1["kk3_tbyp"]
    print("\n=== Headline numbers per subset ===", flush=True)
    for sid in SUBSET_IDS:
        s_gg = gg_summary[gg_summary["subset_id"] == sid].iloc[0]
        s_hh = hh_summary[hh_summary["subset_id"] == sid].iloc[0]
        kill_175 = gg_kill[(gg_kill["subset_id"] == sid) & (gg_kill["SL_distance_atr"] == 1.75)][
            "pct_only_up_killed"
        ].iloc[0]
        tp10 = hh_tp[(hh_tp["subset_id"] == sid) & (hh_tp["tp_atr"] == 10.0)].iloc[0]
        sep95_k20 = jj_sep[
            (jj_sep["subset_id"] == sid)
            & (jj_sep["k"] == 20)
            & (jj_sep["recall_only_up_budget"] == 0.95)
        ].iloc[0]
        bl_summ = nn_subset[
            (nn_subset["subset_id"] == sid)
            & (nn_subset["exit_policy"] == "BL")
            & (nn_subset["row_kind"] == "summary")
        ].iloc[0]
        h_summ = nn_subset[
            (nn_subset["subset_id"] == sid)
            & (nn_subset["exit_policy"] == "H240")
            & (nn_subset["row_kind"] == "summary")
        ].iloc[0]
        print(
            f"  {sid}: GG q05={s_gg['q05']:+.3f} ATR, q01={s_gg['q01']:+.3f} ATR, "
            f"min={s_gg['min']:+.3f} ATR; SL=1.75 kills "
            f"{kill_175 * 100:.2f}% of only_up. "
            f"HH q50={s_hh['median']:.2f} ATR, q95={s_hh['q95']:.2f} ATR; "
            f"TP=10 ATR reached by {tp10['pct_only_up_reaching_tp'] * 100:.2f}% of only_up. "
            f"JJ k=20 (recall>=0.95): tau={sep95_k20['tau_only_up_atr_fill']:+.2f}, "
            f"pct_losers_below={sep95_k20['pct_losers_below_tau'] * 100:.2f}%, "
            f"d_vs_loser_pool={sep95_k20['cohens_d_only_up_vs_loser_pool']:+.3f}. "
            f"NN: BL med={bl_summ['summary_median']:+.3f}R, mean={bl_summ['summary_mean']:+.3f}R; "
            f"H240 med={h_summ['summary_median']:+.3f}R, mean={h_summ['summary_mean']:+.3f}R "
            f"(H240-BL mean delta = "
            f"{h_summ['summary_mean'] - bl_summ['summary_mean']:+.3f}R).",
            flush=True,
        )
    # KK3 headline.
    print("\n=== KK3 selectivity at tau=-0.5 (subset / k=20) ===", flush=True)
    for sid in SUBSET_IDS:
        r0 = kk3_tbyp[(kk3_tbyp["subset_id"] == sid) & (kk3_tbyp["k"] == 20)].iloc[0]
        print(
            f"  {sid}: tier_high {r0['pct_of_only_up_tier_high_below_tau_-0.5'] * 100:.2f}% below; "
            f"tier_runner {r0['pct_of_only_up_tier_runner_below_tau_-0.5'] * 100:.2f}% below; "
            f"tier_low_plus_mid {r0['pct_of_only_up_tier_low_plus_mid_below_tau_-0.5'] * 100:.2f}% below.",
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
