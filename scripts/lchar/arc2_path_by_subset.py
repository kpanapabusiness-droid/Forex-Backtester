"""Arc 2 — Path-category breakdown across bivariate-filtered subsets.

Phase: L6 Arc 2 Phase 3 — block_V / block_W / block_X / block_Y.

Descriptive (§14.5). Read-existing-CSV (§14.6). Path categories are clean
labels used as evaluation buckets only (§14).

For each filtered subset (bivariate quintile cells from block_P), compute the
Block B path-category distribution, per-category mean R, per-category
contribution to the subset's pooled mean R, per-bar median paths, and a
mix-shift / per-category-shift decomposition vs the population baseline.

Outputs to: results/l6/arc2/characterisation/extended/path_by_subset/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import platform
import re
import shutil
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
from matplotlib.collections import LineCollection  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Locked input sha256s (gate 1, re-verified at gate 8)
# ---------------------------------------------------------------------------
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv":
        "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv":
        "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv":
        "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv":
        "4a61407f0f1fc1b74486f0614928e776201dc6469d874db8393e689d20cdb2ff",
    "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv":
        "a5e3f8e68aa64d8fd53f752705a33613d9877dbde1f8265cb4a38d753c5e088e",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py":
        "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml":
        "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "L6_0_METHODOLOGY_LOCK.md":
        "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}

OUTPUT_DIR_REL = "results/l6/arc2/characterisation/extended/path_by_subset"

# Population baselines (from block_P, established by prior phases).
BASELINE_MEAN_R: float = -0.0193

# Block B reference counts at 1R (must reproduce exactly at gate 3.1).
BLOCK_B_1R_COUNTS: Dict[str, int] = {
    "only_up": 956,
    "up_then_down": 1075,
    "down_then_up": 1090,
    "straight_to_sl": 858,
    "simultaneous": 13,
    "neither_reached": 1,
}

# Spec section 3 text says `running_mfe_atr >= +1.0` but Gate 3.1 mandates the
# Block B 1R counts (956/1075/1090/858/13/1). Block B at 1R uses ±2 ATR (= ±1R)
# and a 120-bar BL execution horizon. Treating spec's threshold as a typo for
# +2.0 ATR is the only resolution consistent with Gate 3.1.
PATH_THRESH_ATR: float = 2.0  # +1R = +2 ATR
PATH_HOLD_CAP: int = 120  # BL execution horizon
EXTENDED_HOLD: int = 240  # path observation horizon for visualisation

# Quintile expectations.
EXPECTED_Q_SIZES_BY_Q: Dict[str, int] = {
    "Q1": 799, "Q2": 799, "Q3": 799, "Q4": 798, "Q5": 798,
}
Q_ORDER: Tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Q5")

# Subsets to characterise.
SUBSET_DEFS: List[Tuple[str, Dict[str, Any]]] = [
    ("S0_pop", {"all": True, "expected_n": 3993}),
    ("S1_q5q2", {"qa": ("Q5",), "qb": ("Q2",), "expected_n": 190}),
    ("S2_q5q3", {"qa": ("Q5",), "qb": ("Q3",), "expected_n": 178}),
    ("S3_q4q2", {"qa": ("Q4",), "qb": ("Q2",), "expected_n": 151}),
    ("S4_q5xq2q3", {"qa": ("Q5",), "qb": ("Q2", "Q3"), "expected_n": 368}),
    ("S5_q4q5xq2q3",
     {"qa": ("Q4", "Q5"), "qb": ("Q2", "Q3"), "expected_n": 682}),
]

# Per-subset expected pooled mean R for cross-check vs block_P (§4 gate V.1).
EXPECTED_POOLED_MEAN_R: Dict[str, Optional[float]] = {
    "S0_pop": None,        # full population, no block_P cell ref
    "S1_q5q2": 0.4325631187,
    "S2_q5q3": 0.2212491288,
    "S3_q4q2": 0.278122118,
    "S4_q5xq2q3": None,    # union of cells; cross-check via internal additivity
    "S5_q4q5xq2q3": None,
}

# Per-bar median path k grid for Block W.
W_K_GRID: Tuple[int, ...] = (
    1, 2, 3, 5, 7, 10, 13, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120,
    140, 160, 180, 200, 220, 240,
)

# Plot constants.
PLOT_K_MAX = 240
PLOT_Y_MIN = -3.0
PLOT_Y_MAX = 8.0
PLOT_FIG_SIZE_2x2 = (14.0, 9.0)
PLOT_FIG_SIZE_OVERLAY = (14.0, 9.0)
PLOT_DPI = 100

# Category styling (matches plot_05_combined_2x2 in trade_paths_by_block_b).
CATEGORY_COLOURS: Dict[str, Tuple[str, str]] = {
    "only_up":         ("#639922", "#3B6D11"),  # green
    "up_then_down":    ("#EF9F27", "#A86E10"),  # amber
    "down_then_up":    ("#D85A30", "#8E3D1F"),  # coral
    "straight_to_sl":  ("#E24B4A", "#8A2828"),  # red
}
CATEGORY_PLOT_ORDER: Tuple[str, ...] = (
    "only_up", "up_then_down", "down_then_up", "straight_to_sl",
)
ALL_CATS: Tuple[str, ...] = (
    "only_up", "up_then_down", "down_then_up",
    "straight_to_sl", "simultaneous", "neither_reached",
)

# Disposition discipline §14.5 — forbidden patterns outside Planning input.
FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "should filter on",
    "best filter is",
    "we should exclude",
    "this filter would pass",
    " recommend",  # leading space to avoid matching "recommendation" in URLs
    "predicts",
    "the right exit rule is",
    "should exit at",
    "we should hold longer",
)

PLOT_THIN_THRESHOLD = 20  # below this skip median; 5..19 still show paths


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(label: str) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for rel, expected in LOCKED_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"HALT {label} - sha256 mismatch on {rel}:\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
        observed[rel] = actual
    return observed


# ---------------------------------------------------------------------------
# Quintile bucketing — rank-based, deterministic tie-break by trade_id.
# Matches arc2_entry_filter_bivariate._make_quintile_labels.
# ---------------------------------------------------------------------------
def make_quintile_labels(
    values: pd.Series, tie_break: pd.Series
) -> Tuple[pd.Series, List[Tuple[float, float]]]:
    df = pd.DataFrame({"v": values.values, "t": tie_break.values},
                      index=values.index)
    df = df.sort_values(["v", "t"], kind="stable")
    n = len(df)
    base = n // 5
    rem = n - base * 5
    sizes = [base + (1 if i < rem else 0) for i in range(5)]
    labels: List[str] = []
    bounds: List[Tuple[float, float]] = []
    cursor = 0
    for qi, sz in enumerate(sizes):
        seg = df.iloc[cursor:cursor + sz]
        labels.extend([f"Q{qi + 1}"] * sz)
        bounds.append((float(seg["v"].min()), float(seg["v"].max())))
        cursor += sz
    df["q"] = labels
    return df["q"].reindex(values.index), bounds


# ---------------------------------------------------------------------------
# Data loading + merge.
# ---------------------------------------------------------------------------
def load_and_join(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("  Loading signals_features.csv...", flush=True)
    sf = pd.read_csv(
        REPO_ROOT / "results/l6/arc2/characterisation/v1_1_full/signals_features.csv",
        usecols=["pair", "time", "taken", "concurrent_signals_same_bar"],
    )
    sf = sf[sf["taken"]].copy()
    sf["signal_bar_ts"] = pd.to_datetime(sf["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    sf = sf.drop(columns=["time", "taken"])

    if verbose:
        print("  Loading trade_index.csv...", flush=True)
    ti = pd.read_csv(
        REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv",
    )
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    if verbose:
        print("  Loading block_M_kijun_distances.csv...", flush=True)
    bm = pd.read_csv(
        REPO_ROOT / "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv",
        usecols=["trade_id", "dist_d1_kijun_atr"],
    )

    merged = ti.merge(sf, on=["pair", "signal_bar_ts"], how="left",
                       validate="one_to_one")
    merged = merged.merge(bm, on="trade_id", how="left", validate="one_to_one")
    merged = merged.sort_values("trade_id").reset_index(drop=True)

    if len(merged) != 3993:
        raise RuntimeError(
            f"HALT - merged taken row count {len(merged)} != 3993"
        )
    for col in ("concurrent_signals_same_bar", "dist_d1_kijun_atr"):
        if merged[col].isna().any():
            raise RuntimeError(f"HALT - {col} has NaN after merge")
    return merged


def assign_quintiles(taken: pd.DataFrame) -> pd.DataFrame:
    qa_labels, _ = make_quintile_labels(
        taken["concurrent_signals_same_bar"], taken["trade_id"]
    )
    qb_labels, _ = make_quintile_labels(
        taken["dist_d1_kijun_atr"], taken["trade_id"]
    )
    taken = taken.copy()
    taken["Q_A_concurrent"] = qa_labels
    taken["Q_B_dist_d1"] = qb_labels

    qa_counts = taken["Q_A_concurrent"].value_counts().to_dict()
    qb_counts = taken["Q_B_dist_d1"].value_counts().to_dict()
    for q, exp in EXPECTED_Q_SIZES_BY_Q.items():
        if qa_counts.get(q, 0) != exp:
            raise RuntimeError(
                f"HALT - Q_A_concurrent marginal {q} = {qa_counts.get(q,0)},"
                f" expected {exp}"
            )
        if qb_counts.get(q, 0) != exp:
            raise RuntimeError(
                f"HALT - Q_B_dist_d1 marginal {q} = {qb_counts.get(q,0)},"
                f" expected {exp}"
            )
    return taken


# ---------------------------------------------------------------------------
# Gate 2.1 — recover block_P 25-cell counts and verify byte-equal.
# ---------------------------------------------------------------------------
def gate_2_1_cell_counts(taken: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    block_p = pd.read_csv(
        REPO_ROOT / "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv"
    )
    expected = {
        (row["Q_A_concurrent"], row["Q_B_dist_d1"]): int(row["n"])
        for _, row in block_p.iterrows()
    }
    observed: Dict[Tuple[str, str], int] = {}
    diffs: List[str] = []
    for qa in Q_ORDER:
        for qb in Q_ORDER:
            n = int(
                ((taken["Q_A_concurrent"] == qa) &
                 (taken["Q_B_dist_d1"] == qb)).sum()
            )
            observed[(qa, qb)] = n
            exp_n = expected.get((qa, qb), -1)
            if n != exp_n:
                diffs.append(f"  ({qa},{qb}): observed={n}, block_P={exp_n}")
    if diffs:
        raise RuntimeError(
            "HALT (gate 2.1): bivariate cell counts diverge from block_P:\n"
            + "\n".join(diffs)
        )
    return observed


# ---------------------------------------------------------------------------
# Subset masks.
# ---------------------------------------------------------------------------
def subset_mask(taken: pd.DataFrame, defn: Dict[str, Any]) -> np.ndarray:
    if defn.get("all"):
        return np.ones(len(taken), dtype=bool)
    qa_set = set(defn["qa"])
    qb_set = set(defn["qb"])
    return (
        taken["Q_A_concurrent"].isin(qa_set).to_numpy()
        & taken["Q_B_dist_d1"].isin(qb_set).to_numpy()
    )


# ---------------------------------------------------------------------------
# Path-category assignment from per_bar_paths.csv.
# Logic: t_up = first k in [1, 120] where running_mfe_atr[k] >= +2.0
#        t_down = first k in [1, 120] where running_mae_atr[k] <= -2.0
# (±2 ATR = ±1R; 120 = BL execution horizon). Spec text says +1.0 ATR,
# but Gate 3.1 demands Block B 1R counts which are only reachable with +2.0
# ATR + 120 cap; documented at PATH_THRESH_ATR.
# ---------------------------------------------------------------------------
def assign_path_categories(
    per_bar: pd.DataFrame, n_trades: int
) -> np.ndarray:
    pb = per_bar.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids = pb["trade_id"].to_numpy()
    ks = pb["k"].to_numpy()
    mfe = pb["running_mfe_atr"].to_numpy()
    mae = pb["running_mae_atr"].to_numpy()
    # Cap at PATH_HOLD_CAP for first-crossing detection.
    in_window = ks <= PATH_HOLD_CAP
    up_hit = (mfe >= PATH_THRESH_ATR) & in_window
    dn_hit = (mae <= -PATH_THRESH_ATR) & in_window

    # Compute first-cross k per trade using groupby min over hits.
    SENTINEL = PATH_HOLD_CAP + 1  # 121 = never reached within cap
    t_up = np.full(n_trades, SENTINEL, dtype=np.int32)
    t_dn = np.full(n_trades, SENTINEL, dtype=np.int32)

    if up_hit.any():
        up_idx = np.where(up_hit)[0]
        # Use np.minimum.reduceat or just pandas-side groupby.
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
    return cats


def gate_3_1_block_b_reproduction(cats: np.ndarray) -> Dict[str, int]:
    counts = {c: int((cats == c).sum()) for c in ALL_CATS}
    diffs: List[str] = []
    for c, exp in BLOCK_B_1R_COUNTS.items():
        if counts[c] != exp:
            diffs.append(f"  {c}: observed={counts[c]}, expected={exp}")
    if diffs:
        raise RuntimeError(
            "HALT (gate 3.1): path-category counts diverge from existing "
            "Block B 1R report:\n" + "\n".join(diffs)
        )
    return counts


# ---------------------------------------------------------------------------
# Block V — Per-subset category breakdown.
# ---------------------------------------------------------------------------
def compute_block_V(
    taken: pd.DataFrame, cats: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    R = taken["R"].to_numpy(dtype=np.float64)
    pop_counts_by_cat = {c: int((cats == c).sum()) for c in ALL_CATS}
    pop_mean_R_by_cat: Dict[str, float] = {}
    for c in ALL_CATS:
        m = cats == c
        pop_mean_R_by_cat[c] = float(R[m].mean()) if m.any() else float("nan")
    pop_pct_by_cat = {c: pop_counts_by_cat[c] / 3993.0 for c in ALL_CATS}

    rows_V: List[Dict[str, Any]] = []
    rows_summary: List[Dict[str, Any]] = []

    for subset_id, defn in SUBSET_DEFS:
        mask = subset_mask(taken, defn)
        n_subset = int(mask.sum())
        R_sub = R[mask]
        cats_sub = cats[mask]
        pooled = float(R_sub.mean()) if n_subset else float("nan")
        contribs_sum = 0.0
        for c in ALL_CATS:
            cell_mask = cats_sub == c
            n_cat = int(cell_mask.sum())
            pct_of_subset = n_cat / n_subset if n_subset else float("nan")
            pop_n_cat = pop_counts_by_cat[c]
            pct_of_population = (
                n_cat / pop_n_cat if pop_n_cat else float("nan")
            )
            over_rep = (
                pct_of_subset / pop_pct_by_cat[c]
                if pop_pct_by_cat[c] > 0 else float("nan")
            )
            if n_cat > 0:
                mean_R_within = float(R_sub[cell_mask].mean())
                median_R_within = float(np.median(R_sub[cell_mask]))
            else:
                mean_R_within = float("nan")
                median_R_within = float("nan")
            contrib = (
                pct_of_subset * mean_R_within
                if (n_cat > 0 and n_subset > 0) else 0.0
            )
            contribs_sum += contrib
            rows_V.append({
                "subset_id": subset_id,
                "category": c,
                "n_cat": n_cat,
                "n_subset": n_subset,
                "pct_of_subset": pct_of_subset,
                "pct_of_population": pct_of_population,
                "over_representation": over_rep,
                "mean_R_within_cat": mean_R_within,
                "median_R_within_cat": median_R_within,
                "mean_R_within_cat_in_pop": pop_mean_R_by_cat[c],
                "mean_R_diff_vs_pop": (
                    mean_R_within - pop_mean_R_by_cat[c]
                    if n_cat > 0 else float("nan")
                ),
                "contribution_to_pooled_R": contrib,
            })

        expected_pooled = EXPECTED_POOLED_MEAN_R.get(subset_id)
        consistency_diff = (
            float(pooled - expected_pooled)
            if expected_pooled is not None else float("nan")
        )
        # Gate V.2 — additivity within 1e-9
        addit_err = float(contribs_sum - pooled)
        if abs(addit_err) > 1e-9:
            raise RuntimeError(
                f"HALT (gate V.2): subset {subset_id} contributions sum "
                f"{contribs_sum} != pooled {pooled} (err={addit_err:.3e})"
            )
        rows_summary.append({
            "subset_id": subset_id,
            "n_subset": n_subset,
            "pooled_mean_R": pooled,
            "pooled_mean_R_from_block_P": expected_pooled if expected_pooled is not None else float("nan"),
            "consistency_diff": consistency_diff,
            "contributions_sum_check": addit_err,
        })

        # Gate V.1 — for subsets with block_P reference, match within 1e-9
        if expected_pooled is not None and abs(consistency_diff) > 1e-9:
            raise RuntimeError(
                f"HALT (gate V.1): subset {subset_id} pooled mean R "
                f"{pooled} differs from block_P {expected_pooled} "
                f"(diff={consistency_diff:.3e})"
            )

    df_V = pd.DataFrame(rows_V)
    df_summary = pd.DataFrame(rows_summary)
    return df_V, df_summary


# ---------------------------------------------------------------------------
# Block Y — Mix-shift decomposition.
# ---------------------------------------------------------------------------
def compute_block_Y(taken: pd.DataFrame, cats: np.ndarray) -> pd.DataFrame:
    R = taken["R"].to_numpy(dtype=np.float64)
    pop_counts = {c: int((cats == c).sum()) for c in ALL_CATS}
    pop_pct = {c: pop_counts[c] / 3993.0 for c in ALL_CATS}
    pop_mean_R: Dict[str, float] = {}
    for c in ALL_CATS:
        m = cats == c
        pop_mean_R[c] = float(R[m].mean()) if m.any() else float("nan")
    pop_pooled = float(R.mean())  # ≈ -0.0193

    rows: List[Dict[str, Any]] = []
    for subset_id, defn in SUBSET_DEFS:
        if subset_id == "S0_pop":
            continue
        mask = subset_mask(taken, defn)
        n_sub = int(mask.sum())
        R_sub = R[mask]
        cats_sub = cats[mask]
        sub_pooled = float(R_sub.mean()) if n_sub else float("nan")
        sub_pct: Dict[str, float] = {}
        sub_mean_R: Dict[str, float] = {}
        for c in ALL_CATS:
            cm = cats_sub == c
            sub_pct[c] = int(cm.sum()) / n_sub if n_sub else 0.0
            sub_mean_R[c] = float(R_sub[cm].mean()) if cm.any() else float("nan")

        lift = sub_pooled - pop_pooled
        mix_shift = 0.0
        per_cat_shift = 0.0
        for c in ALL_CATS:
            if pop_counts[c] == 0:
                continue
            # Mix-shift: (pct_sub - pct_pop) * mean_R_pop
            mix_shift += (sub_pct[c] - pop_pct[c]) * pop_mean_R[c]
            # Per-category shift: pct_pop * (mean_R_sub - mean_R_pop)
            if not np.isnan(sub_mean_R[c]):
                per_cat_shift += pop_pct[c] * (sub_mean_R[c] - pop_mean_R[c])
        interaction = lift - mix_shift - per_cat_shift

        # Internal additivity check (gate Y).
        if abs(mix_shift + per_cat_shift + interaction - lift) > 1e-9:
            raise RuntimeError(
                f"HALT (gate Y): {subset_id} additivity "
                f"mix={mix_shift} + per_cat={per_cat_shift} + int={interaction}"
                f" != lift={lift}"
            )

        rows.append({
            "subset_id": subset_id,
            "n_subset": n_sub,
            "pooled_mean_R": sub_pooled,
            "pooled_mean_R_pop_baseline": pop_pooled,
            "lift_vs_baseline": lift,
            "mix_shift_component": mix_shift,
            "per_cat_shift_component": per_cat_shift,
            "interaction_component": interaction,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Block W — Per-subset × per-category × per-bar median path.
# Conversion: bar_close_atr / 2.0 -> R-units.
# Filter: only bars with has_next_bar == True.
# Restrict to bars in W_K_GRID (24 timestamps) for output CSV.
# ---------------------------------------------------------------------------
def compute_block_W(
    taken: pd.DataFrame, cats: np.ndarray, per_bar: pd.DataFrame
) -> pd.DataFrame:
    # Build per-trade close_R lookup arrays.
    tids = per_bar["trade_id"].to_numpy(dtype=np.int64)
    ks_arr = per_bar["k"].to_numpy(dtype=np.int32)
    bar_close_R = per_bar["bar_close_atr"].to_numpy(dtype=np.float64) / 2.0
    has_next = per_bar["has_next_bar"].to_numpy(dtype=bool)

    n_trades = int(taken["trade_id"].max()) + 1
    # Build dense (n_trades, 240) matrix of close_R, with NaN for unobserved/clamped bars.
    mat = np.full((n_trades, PLOT_K_MAX), np.nan, dtype=np.float64)
    mask_valid = (ks_arr >= 1) & (ks_arr <= PLOT_K_MAX) & has_next
    tt = tids[mask_valid]
    kk = ks_arr[mask_valid]
    cc = bar_close_R[mask_valid]
    mat[tt, kk - 1] = cc

    rows: List[Dict[str, Any]] = []
    for subset_id, defn in SUBSET_DEFS:
        mask = subset_mask(taken, defn)
        for c in CATEGORY_PLOT_ORDER:
            cm = mask & (cats == c)
            n_cat = int(cm.sum())
            if n_cat < PLOT_THIN_THRESHOLD:
                # Thin: emit a sentinel row noting skip
                rows.append({
                    "subset_id": subset_id,
                    "category": c,
                    "k": -1,
                    "n_at_bar_k": n_cat,
                    "median_close_R": float("nan"),
                    "q25_close_R": float("nan"),
                    "q75_close_R": float("nan"),
                    "thin_flag": "thin_skipped",
                })
                continue
            tids_in = np.where(cm)[0]
            sub_mat = mat[tids_in]  # (n_cat, 240)
            for k in W_K_GRID:
                col = sub_mat[:, k - 1]
                valid = col[~np.isnan(col)]
                n_at_k = int(valid.size)
                if n_at_k == 0:
                    med = q25 = q75 = float("nan")
                else:
                    med = float(np.median(valid))
                    q25 = float(np.quantile(valid, 0.25))
                    q75 = float(np.quantile(valid, 0.75))
                rows.append({
                    "subset_id": subset_id,
                    "category": c,
                    "k": int(k),
                    "n_at_bar_k": n_at_k,
                    "median_close_R": med,
                    "q25_close_R": q25,
                    "q75_close_R": q75,
                    "thin_flag": "",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Block X — Visualisation panels.
# Produces 10 PNGs total: 5 subsets (S1..S5) × {combined 2x2, overlay}.
# ---------------------------------------------------------------------------
def _build_close_matrix(
    per_bar: pd.DataFrame, n_trades: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns:
      mat (n_trades, PLOT_K_MAX) of bar_close_R (NaN if not has_next_bar).
      valid_mask (n_trades, PLOT_K_MAX) boolean.
    """
    tids = per_bar["trade_id"].to_numpy(dtype=np.int64)
    ks = per_bar["k"].to_numpy(dtype=np.int32)
    bcR = per_bar["bar_close_atr"].to_numpy(dtype=np.float64) / 2.0
    has_next = per_bar["has_next_bar"].to_numpy(dtype=bool)
    mat = np.full((n_trades, PLOT_K_MAX), np.nan, dtype=np.float64)
    in_range = (ks >= 1) & (ks <= PLOT_K_MAX) & has_next
    tt = tids[in_range]
    kk = ks[in_range]
    mat[tt, kk - 1] = bcR[in_range]
    return mat, ~np.isnan(mat)


def _style_axes(ax: matplotlib.axes.Axes, annotate_k120: bool = True) -> None:
    ax.set_xlim(1, PLOT_K_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_xticks(np.arange(0, PLOT_K_MAX + 1, 20))
    ax.set_xlabel("Bar number (k)")
    ax.set_ylabel("R-units from entry")
    ax.grid(True, color="lightgray", alpha=0.3, linewidth=0.5)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(120, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)


def _pick_alpha(n: int) -> float:
    if n >= 1000: return 0.04
    if n >= 500:  return 0.05
    if n >= 200:  return 0.06
    if n >= 100:  return 0.07
    return 0.08


def _category_segments(
    mat: np.ndarray, tids_in_cat: np.ndarray
) -> List[np.ndarray]:
    segs: List[np.ndarray] = []
    for tid in tids_in_cat:
        row = mat[tid]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        ks = np.where(valid)[0] + 1
        ys = row[valid]
        segs.append(np.column_stack([ks.astype(np.float64), ys]))
    return segs


def _plot_category_panel(
    ax: matplotlib.axes.Axes,
    cat: str, subset_id: str,
    tids_in_cat: np.ndarray,
    mat: np.ndarray,
    title_size: int = 12,
    annotate: bool = True,
) -> None:
    light, dark = CATEGORY_COLOURS[cat]
    n_cat = len(tids_in_cat)

    if n_cat == 0:
        _style_axes(ax)
        ax.set_title(f"{cat} (n=0) - Subset {subset_id} - Block B at 1R",
                     fontsize=title_size)
        return

    segs = _category_segments(mat, tids_in_cat)
    alpha = _pick_alpha(n_cat)
    if n_cat >= 5:
        lc = LineCollection(
            segs, colors=[light], linewidths=0.5, alpha=alpha,
            rasterized=True, zorder=1,
        )
        ax.add_collection(lc)

    if n_cat >= PLOT_THIN_THRESHOLD:
        # median + IQR shading
        sub_mat = mat[tids_in_cat]
        k_grid = np.arange(1, PLOT_K_MAX + 1)
        med = np.full(PLOT_K_MAX, np.nan)
        q25 = np.full(PLOT_K_MAX, np.nan)
        q75 = np.full(PLOT_K_MAX, np.nan)
        for k_idx in range(PLOT_K_MAX):
            col = sub_mat[:, k_idx]
            valid = col[~np.isnan(col)]
            if valid.size > 0:
                med[k_idx] = np.median(valid)
                q25[k_idx] = np.quantile(valid, 0.25)
                q75[k_idx] = np.quantile(valid, 0.75)
        valid_med = ~np.isnan(med)
        ax.fill_between(
            k_grid[valid_med], q25[valid_med], q75[valid_med],
            color=light, alpha=0.15, zorder=2, linewidth=0,
        )
        ax.plot(
            k_grid[valid_med], med[valid_med], color=dark,
            linewidth=2.5, alpha=1.0, zorder=3,
        )
    else:
        ax.text(
            120, (PLOT_Y_MAX + PLOT_Y_MIN) / 2.0,
            f"n={n_cat} - below visualisation threshold;\nmedian omitted",
            fontsize=11, color="dimgray", ha="center", va="center",
            style="italic",
        )

    _style_axes(ax)
    ax.set_title(
        f"{cat} (n={n_cat}) - Subset {subset_id} - Block B at 1R",
        fontsize=title_size,
    )
    if annotate:
        ax.text(
            122, PLOT_Y_MAX - 0.4,
            "k > 120 is post-exit forward observation",
            fontsize=8, color="dimgray", style="italic",
            verticalalignment="top",
        )


def _save_png_deterministic(fig, path: Path) -> None:
    # Matplotlib's PNG metadata includes a software string and creation time
    # by default. Strip them by passing a controlled metadata dict.
    fig.savefig(
        path, dpi=PLOT_DPI,
        metadata={"Software": None, "Creation Time": None},
    )


def render_subset_plots(
    out_dir: Path, taken: pd.DataFrame, cats: np.ndarray, mat: np.ndarray,
) -> Dict[str, str]:
    """Render 10 PNGs (5 subsets × 2 plots) into out_dir/plots/.
    Returns map plot_filename -> "ok"/"thin" status info."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, str] = {}

    for subset_id, defn in SUBSET_DEFS:
        if subset_id == "S0_pop":
            continue
        mask = subset_mask(taken, defn)
        # combined 2x2
        fig, axes = plt.subplots(
            2, 2, figsize=PLOT_FIG_SIZE_2x2, dpi=PLOT_DPI,
        )
        # ordering per spec: top-left only_up, top-right up_then_down,
        # bottom-left down_then_up, bottom-right straight_to_sl
        panel_map = {
            (0, 0): "only_up",
            (0, 1): "up_then_down",
            (1, 0): "down_then_up",
            (1, 1): "straight_to_sl",
        }
        for (i, j), cat in panel_map.items():
            ax = axes[i, j]
            tids_in_cat = np.where(mask & (cats == cat))[0]
            _plot_category_panel(
                ax, cat, subset_id, tids_in_cat, mat,
                title_size=11, annotate=False,
            )
        fig.suptitle(
            f"Block X - {subset_id} path categories (1R threshold)",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fname = f"{subset_id}_combined_2x2.png"
        _save_png_deterministic(fig, plots_dir / fname)
        plt.close(fig)
        status[fname] = "ok"
        print(f"    {fname}", flush=True)

        # overlay (all 4 cats on one axes)
        fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE_OVERLAY, dpi=PLOT_DPI)
        legend_handles = []
        legend_labels = []
        for cat in CATEGORY_PLOT_ORDER:
            tids_in_cat = np.where(mask & (cats == cat))[0]
            n_cat = len(tids_in_cat)
            light, dark = CATEGORY_COLOURS[cat]
            if n_cat == 0:
                continue
            segs = _category_segments(mat, tids_in_cat)
            alpha = _pick_alpha(n_cat) * 0.6  # dimmer for overlay clarity
            if n_cat >= 5:
                lc = LineCollection(
                    segs, colors=[light], linewidths=0.4, alpha=alpha,
                    rasterized=True, zorder=1,
                )
                ax.add_collection(lc)
            if n_cat >= PLOT_THIN_THRESHOLD:
                sub_mat = mat[tids_in_cat]
                k_grid = np.arange(1, PLOT_K_MAX + 1)
                med = np.full(PLOT_K_MAX, np.nan)
                for k_idx in range(PLOT_K_MAX):
                    col = sub_mat[:, k_idx]
                    valid = col[~np.isnan(col)]
                    if valid.size > 0:
                        med[k_idx] = np.median(valid)
                valid_med = ~np.isnan(med)
                line, = ax.plot(
                    k_grid[valid_med], med[valid_med], color=dark,
                    linewidth=2.5, zorder=3,
                    label=f"{cat} (n={n_cat})",
                )
                legend_handles.append(line)
                legend_labels.append(f"{cat} (n={n_cat})")
            else:
                from matplotlib.lines import Line2D
                line = Line2D([0], [0], color=dark, linewidth=2.5)
                legend_handles.append(line)
                legend_labels.append(f"{cat} (n={n_cat}, thin)")
        _style_axes(ax)
        ax.set_title(
            f"Block X overlay - {subset_id} (all 4 path categories, 1R)",
            fontsize=14,
        )
        ax.text(
            122, PLOT_Y_MAX - 0.4,
            "k > 120 is post-exit forward observation",
            fontsize=9, color="dimgray", style="italic",
        )
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="upper left",
                       fontsize=10, framealpha=0.9)
        fig.tight_layout()
        fname = f"{subset_id}_overlay.png"
        _save_png_deterministic(fig, plots_dir / fname)
        plt.close(fig)
        status[fname] = "ok"
        print(f"    {fname}", flush=True)

    return status


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------
def _df_to_md(df: pd.DataFrame, float_fmt: str = "{:.6f}") -> str:
    raw_cols = list(df.columns)
    header_cols = [str(c) for c in raw_cols]
    out = ["| " + " | ".join(header_cols) + " |",
           "| " + " | ".join(["---"] * len(header_cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in raw_cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("")
                else:
                    cells.append(float_fmt.format(v))
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(
    out_dir: Path,
    input_shas: Dict[str, str],
    cell_counts: Dict[Tuple[str, str], int],
    block_b_counts: Dict[str, int],
    df_V: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_W: pd.DataFrame,
    df_Y: pd.DataFrame,
    plot_status: Dict[str, str],
    determinism_status: str,
) -> Path:
    L: List[str] = []
    L.append("# Arc 2 - Path-category breakdown by subset (Phase 3)")
    L.append("")
    L.append(
        "Disposition: descriptive only (§14.5). Read-existing-CSV (§14.6)."
        " Path categories are evaluation buckets per Block B 1R; not used in"
        " signal-time decisions (§14)."
    )
    L.append("")

    L.append("## Locked input sha256 manifest")
    L.append("")
    for rel, sha in input_shas.items():
        L.append(f"- `{rel}` : `{sha}`")
    L.append("")

    L.append("## Determinism receipt")
    L.append("")
    L.append(f"- Gate 7 status: {determinism_status}")
    L.append("")

    L.append("## Subset definitions and counts (Block V summary)")
    L.append("")
    L.append(_df_to_md(df_summary, float_fmt="{:.10g}"))
    L.append("")

    L.append("## Block V - Per-subset category breakdown")
    L.append("")
    cols_order = [
        "category", "n_cat", "pct_of_subset", "pct_of_population",
        "over_representation", "mean_R_within_cat", "median_R_within_cat",
        "mean_R_within_cat_in_pop", "mean_R_diff_vs_pop",
        "contribution_to_pooled_R",
    ]
    for subset_id, _ in SUBSET_DEFS:
        sub = df_V[df_V["subset_id"] == subset_id].copy()
        sub["abs_contrib"] = sub["contribution_to_pooled_R"].abs()
        sub = sub.sort_values("abs_contrib", ascending=False).drop(
            columns="abs_contrib"
        )
        sub = sub[cols_order]
        L.append(f"### {subset_id}")
        L.append("")
        L.append(_df_to_md(sub, float_fmt="{:.6f}"))
        L.append("")

    L.append("## Block Y - Mix-shift decomposition (headline)")
    L.append("")
    L.append(_df_to_md(df_Y, float_fmt="{:.6f}"))
    L.append("")

    L.append("## Block W - Per-subset x per-category median paths (selected bars)")
    L.append("")
    L.append("Selected bars: k = 10, 20, 30, 60, 120.")
    L.append("")
    sel_ks = (10, 20, 30, 60, 120)
    for subset_id, _ in SUBSET_DEFS:
        sub = df_W[(df_W["subset_id"] == subset_id) & (df_W["k"].isin(sel_ks))]
        if sub.empty:
            L.append(f"### {subset_id}: (all categories thin or no data)")
            L.append("")
            continue
        # Pivot: rows = category, cols = k
        piv = sub.pivot_table(
            index="category", columns="k", values="median_close_R",
            aggfunc="first",
        )
        # Add pop reference row (S0_pop) for comparison
        pop_sub = df_W[(df_W["subset_id"] == "S0_pop")
                       & (df_W["k"].isin(sel_ks))]
        pop_piv = pop_sub.pivot_table(
            index="category", columns="k", values="median_close_R",
            aggfunc="first",
        )
        L.append(f"### {subset_id}")
        L.append("")
        L.append("Subset median_close_R at selected bars:")
        L.append("")
        L.append(_df_to_md(piv.reset_index(), float_fmt="{:.4f}"))
        L.append("")
        if subset_id != "S0_pop":
            L.append("Population median_close_R at selected bars (for comparison):")
            L.append("")
            L.append(_df_to_md(pop_piv.reset_index(), float_fmt="{:.4f}"))
            L.append("")

    L.append("## Block X - Plot index")
    L.append("")
    for fname in sorted(plot_status.keys()):
        L.append(f"- `plots/{fname}`")
    L.append("")

    L.append("## Out-of-scope items observed")
    L.append("")
    L.append("- The spec section 3 path-category text used a +1.0 ATR threshold,"
             " but Gate 3.1 mandates Block B 1R counts which are only reachable"
             " with a +2.0 ATR (= 1R) threshold and a 120-bar BL execution"
             " horizon. The +2.0/120 logic was used to satisfy Gate 3.1; the"
             " +1.0 wording is read as a typo. Documented at"
             " `PATH_THRESH_ATR` / `PATH_HOLD_CAP` in the script.")
    L.append("- The spec mentions a `running_close_atr` column; the actual"
             " per_bar_paths.csv column is `bar_close_atr` (already fill-relative,"
             " in ATR-units). Used as-is, divided by 2.0 to convert to R-units.")
    L.append("")

    L.append("## Planning input")
    L.append("")
    L.append("Per-subset mechanism notes (descriptive only):")
    L.append("")
    pop_pooled = float(df_summary.loc[
        df_summary["subset_id"] == "S0_pop", "pooled_mean_R"
    ].iloc[0])
    for subset_id, _ in SUBSET_DEFS:
        if subset_id == "S0_pop":
            continue
        sub = df_V[df_V["subset_id"] == subset_id].copy()
        sub["abs_contrib"] = sub["contribution_to_pooled_R"].abs()
        sub = sub.sort_values("abs_contrib", ascending=False)
        top = sub.iloc[0]
        y_row = df_Y[df_Y["subset_id"] == subset_id].iloc[0]
        mix_part = float(y_row["mix_shift_component"])
        per_cat_part = float(y_row["per_cat_shift_component"])
        lift = float(y_row["lift_vs_baseline"])
        if abs(mix_part) > abs(per_cat_part):
            channel = "mix-shift"
        else:
            channel = "per-category-shift"
        L.append(
            f"- **{subset_id}** (n={int(y_row['n_subset'])}, lift {lift:+.4f}R):"
            f" top contributing category is `{top['category']}` "
            f"(pct_of_subset={top['pct_of_subset']:.4f},"
            f" mean_R_within={top['mean_R_within_cat']:.4f},"
            f" contribution={top['contribution_to_pooled_R']:+.4f}R)."
            f" Decomposition: mix-shift={mix_part:+.4f},"
            f" per-cat-shift={per_cat_part:+.4f},"
            f" interaction={float(y_row['interaction_component']):+.4f}."
            f" Channel by magnitude: {channel}."
        )
        # Categories with mean_R_diff > 0.30R in magnitude
        big_shifts = sub[(sub["n_cat"] > 0) & (sub["mean_R_diff_vs_pop"].abs() > 0.30)]
        if not big_shifts.empty:
            entries = ", ".join(
                f"`{r['category']}` (diff {r['mean_R_diff_vs_pop']:+.3f}R, n={int(r['n_cat'])})"
                for _, r in big_shifts.iterrows()
            )
            L.append(f"  - Categories with |mean_R_diff_vs_pop| > 0.30R: {entries}.")
        else:
            L.append("  - No category in this subset has |mean_R_diff_vs_pop| > 0.30R.")
    L.append("")

    L.append("Cross-subset comparison (descriptive only):")
    L.append("")
    # Largest only_up over-representation
    onlyup_rep = df_V[(df_V["category"] == "only_up")
                       & (df_V["subset_id"] != "S0_pop")]
    onlyup_rep_sorted = onlyup_rep.sort_values(
        "over_representation", ascending=False
    )
    if not onlyup_rep_sorted.empty:
        top_ou = onlyup_rep_sorted.iloc[0]
        # population pct = pct_of_subset / over_representation (inverse of over_rep def)
        pop_pct_only_up = (
            top_ou["pct_of_subset"] / top_ou["over_representation"]
            if top_ou["over_representation"] > 0 else float("nan")
        )
        L.append(
            f"- Subset with largest `only_up` over-representation: "
            f"`{top_ou['subset_id']}` "
            f"(over_repr={top_ou['over_representation']:.4f},"
            f" pct_of_subset={top_ou['pct_of_subset']:.4f}"
            f" vs population pct = {pop_pct_only_up:.4f})."
        )
    # Smallest straight_to_sl share
    stsl_share = df_V[(df_V["category"] == "straight_to_sl")
                       & (df_V["subset_id"] != "S0_pop")]
    stsl_sorted = stsl_share.sort_values("pct_of_subset", ascending=True)
    if not stsl_sorted.empty:
        top_st = stsl_sorted.iloc[0]
        L.append(
            f"- Subset with smallest `straight_to_sl` share: "
            f"`{top_st['subset_id']}` "
            f"(pct_of_subset={top_st['pct_of_subset']:.4f},"
            f" over_repr={top_st['over_representation']:.4f})."
        )
    # Channel signature grouping
    chan_assignments: List[Tuple[str, str]] = []
    for _, yr in df_Y.iterrows():
        ch = ("mix-shift" if abs(yr["mix_shift_component"])
              > abs(yr["per_cat_shift_component"])
              else "per-category-shift")
        chan_assignments.append((str(yr["subset_id"]), ch))
    mix_subs = [s for s, c in chan_assignments if c == "mix-shift"]
    pcs_subs = [s for s, c in chan_assignments if c == "per-category-shift"]
    L.append(
        f"- Subsets where the lift signature is dominated by mix-shift: "
        f"{', '.join(f'`{s}`' for s in mix_subs) if mix_subs else 'none'}."
    )
    L.append(
        f"- Subsets where the lift signature is dominated by per-cat-shift: "
        f"{', '.join(f'`{s}`' for s in pcs_subs) if pcs_subs else 'none'}."
    )
    L.append("")
    L.append(f"_Population baseline pooled mean R = {pop_pooled:.6f}._")
    L.append("")

    p = out_dir / "path_by_subset.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Disposition discipline grep (gate 9).
# ---------------------------------------------------------------------------
def disposition_grep(md_path: Path) -> List[Tuple[int, str, str]]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
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


# ---------------------------------------------------------------------------
# Build pass — executes the full pipeline writing into `out_dir`.
# Returns dict mapping output file path -> sha256.
# ---------------------------------------------------------------------------
def build_pass(out_dir: Path, *, verbose: bool = True) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # gate 1 already verified by caller; re-verify at start of each pass.
    input_shas = verify_inputs("(pass start)")

    if verbose:
        print("  [Load] reading inputs...", flush=True)
    taken = load_and_join(verbose=verbose)

    if verbose:
        print("  [Quintiles] assigning Q_A / Q_B...", flush=True)
    taken = assign_quintiles(taken)

    if verbose:
        print("  [Gate 2.1] cell counts vs block_P...", flush=True)
    cell_counts = gate_2_1_cell_counts(taken)

    if verbose:
        print("  [Gate 2.2] subset sizes...", flush=True)
    for subset_id, defn in SUBSET_DEFS:
        m = subset_mask(taken, defn)
        n = int(m.sum())
        if n != defn["expected_n"]:
            raise RuntimeError(
                f"HALT (gate 2.2): {subset_id} n={n} != {defn['expected_n']}"
            )

    if verbose:
        print("  [Load] per_bar_paths.csv (~120 MB)...", flush=True)
    pb = pd.read_csv(
        REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv",
        usecols=[
            "trade_id", "k", "running_mfe_atr", "running_mae_atr",
            "bar_close_atr", "has_next_bar",
        ],
        dtype={
            "trade_id": np.int64, "k": np.int32,
            "running_mfe_atr": np.float64, "running_mae_atr": np.float64,
            "bar_close_atr": np.float64,
            "has_next_bar": bool,
        },
    )

    if verbose:
        print("  [Categorise] path-category assignment...", flush=True)
    n_trades = int(taken["trade_id"].max()) + 1
    cats = assign_path_categories(pb, n_trades)
    # cats has length n_trades. taken is indexed by trade_id 0..n_trades-1
    # and sorted by trade_id; aligns with cats.

    if verbose:
        print("  [Gate 3.1] Block B 1R reproduction...", flush=True)
    block_b_counts = gate_3_1_block_b_reproduction(cats)

    if verbose:
        print("  [Block V] subset category breakdown...", flush=True)
    df_V, df_summary = compute_block_V(taken, cats)

    if verbose:
        print("  [Block Y] mix-shift decomposition...", flush=True)
    df_Y = compute_block_Y(taken, cats)

    if verbose:
        print("  [Block W] per-bar median paths...", flush=True)
    df_W = compute_block_W(taken, cats, pb)

    # Build close matrix for plotting; free pb afterward.
    if verbose:
        print("  [Block X] rendering plots...", flush=True)
    mat, _ = _build_close_matrix(pb, n_trades)
    del pb

    plot_status = render_subset_plots(out_dir, taken, cats, mat)
    del mat

    # Write CSVs deterministically.
    if verbose:
        print("  [Write] CSVs...", flush=True)
    (out_dir / "block_V_subset_category_breakdown.csv").write_text("")
    df_V.to_csv(
        out_dir / "block_V_subset_category_breakdown.csv",
        index=False, lineterminator="\n", float_format="%.10g",
    )
    df_summary.to_csv(
        out_dir / "block_V_subset_summary.csv",
        index=False, lineterminator="\n", float_format="%.10g",
    )
    df_W.to_csv(
        out_dir / "block_W_per_subset_per_category_per_bar.csv",
        index=False, lineterminator="\n", float_format="%.10g",
    )
    df_Y.to_csv(
        out_dir / "block_Y_mix_shift_decomposition.csv",
        index=False, lineterminator="\n", float_format="%.10g",
    )

    # Write report.
    if verbose:
        print("  [Write] report...", flush=True)
    md_path = write_report(
        out_dir, input_shas, cell_counts, block_b_counts,
        df_V, df_summary, df_W, df_Y, plot_status,
        determinism_status="see manifest",
    )

    # Disposition discipline (gate 9).
    if verbose:
        print("  [Gate 9] disposition discipline grep...", flush=True)
    viol = disposition_grep(md_path)
    if viol:
        details = "\n".join(
            f"  L{ln}: pattern '{pat}' in: {txt}"
            for ln, pat, txt in viol
        )
        raise RuntimeError(
            "HALT (gate 9): forbidden patterns in path_by_subset.md outside"
            f" 'Planning input':\n{details}"
        )

    # Compute sha256s of all output files (excluding plots subdir for the
    # CSV+MD comparison; plots are handled separately at gate 7).
    out_shas: Dict[str, str] = {}
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            out_shas[f.name] = _sha256_file(f)
    plots_dir = out_dir / "plots"
    if plots_dir.exists():
        for f in sorted(plots_dir.iterdir()):
            if f.is_file():
                out_shas[f"plots/{f.name}"] = _sha256_file(f)
    return out_shas


# ---------------------------------------------------------------------------
# PNG pixel-array comparison fallback (matplotlib metadata may vary).
# ---------------------------------------------------------------------------
def _png_pixels_sha256(path: Path) -> str:
    """Return sha256 of raw RGBA pixel bytes for a PNG.

    Falls back from full-file sha256 (which includes metadata) to pixel-only
    comparison. matplotlib's PNG metadata may include creation timestamps even
    when we ask for them to be omitted; bytes-of-pixels is the principled
    deterministic check.
    """
    from PIL import Image  # type: ignore
    with Image.open(path) as im:
        im = im.convert("RGBA")
        buf = im.tobytes()
    return hashlib.sha256(buf).hexdigest()


# ---------------------------------------------------------------------------
# Manifest.
# ---------------------------------------------------------------------------
def write_manifest(
    out_dir: Path,
    input_shas: Dict[str, str],
    out_shas: Dict[str, str],
    determinism_summary: Dict[str, Any],
    git_head: str,
) -> Path:
    L: List[str] = []
    L.append("=" * 78)
    L.append("Arc 2 - Path-category breakdown by subset - run manifest")
    L.append("=" * 78)
    L.append("phase: l6_arc2_path_by_subset")
    L.append(f"python: {sys.version.split()[0]}")
    L.append(f"numpy: {np.__version__}")
    L.append(f"pandas: {pd.__version__}")
    L.append(f"matplotlib: {matplotlib.__version__}")
    L.append(f"git_head: {git_head}")
    L.append("")
    L.append("--- LOCKED INPUT SHA256s (gate 1 verified) ---")
    for rel, sha in input_shas.items():
        L.append(f"{sha}  {rel}")
    L.append("")
    L.append("--- DETERMINISM (gate 7) ---")
    for k, v in determinism_summary.items():
        L.append(f"{k}: {v}")
    L.append("")
    L.append("--- OUTPUT FILE SHA256s ---")
    for name, sha in out_shas.items():
        L.append(f"{sha}  {name}")
    p = out_dir / "run_manifest.txt"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / OUTPUT_DIR_REL,
    )
    parser.add_argument(
        "--single-pass", action="store_true",
        help="Skip the determinism re-run (for debug).",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    t0 = time.perf_counter()

    print("[Gate 1] Verifying locked input sha256s...", flush=True)
    input_shas = verify_inputs("(gate 1)")
    print(f"  OK - {len(input_shas)} sha256s match", flush=True)

    # --- Two-pass determinism per gate 7 ---
    determinism_summary: Dict[str, Any] = {}
    if not args.single_pass:
        print("[Determinism] running build pass 1 -> scratch_A...", flush=True)
        scratch_A = Path(tempfile.mkdtemp(prefix="path_by_subset_A_"))
        shas_A = build_pass(scratch_A, verbose=True)

        print("[Determinism] running build pass 2 -> scratch_B...", flush=True)
        scratch_B = Path(tempfile.mkdtemp(prefix="path_by_subset_B_"))
        shas_B = build_pass(scratch_B, verbose=False)

        print("[Gate 7] comparing pass A vs pass B...", flush=True)
        # CSV + MD must match byte-for-byte.
        nondet_csv_md: List[str] = []
        for k in shas_A:
            if k.endswith(".csv") or k.endswith(".md") or k == "run_manifest.txt":
                # manifest not yet written
                if k == "run_manifest.txt":
                    continue
                if shas_A[k] != shas_B.get(k):
                    nondet_csv_md.append(k)
        if nondet_csv_md:
            raise RuntimeError(
                "HALT (gate 7): non-deterministic CSV/MD files between two "
                "passes:\n  " + "\n  ".join(nondet_csv_md)
            )
        print(f"  CSV/MD: all {sum(1 for k in shas_A if k.endswith('.csv') or k.endswith('.md'))} files byte-identical", flush=True)

        # PNGs — full-file sha256 first, fall back to pixel comparison.
        png_keys = [k for k in shas_A if k.endswith(".png")]
        full_match = sum(1 for k in png_keys if shas_A[k] == shas_B.get(k))
        pixel_match = 0
        pixel_diff: List[str] = []
        if full_match < len(png_keys):
            for k in png_keys:
                if shas_A[k] == shas_B.get(k):
                    pixel_match += 1
                    continue
                pa = _png_pixels_sha256(scratch_A / k)
                pb = _png_pixels_sha256(scratch_B / k)
                if pa == pb:
                    pixel_match += 1
                else:
                    pixel_diff.append(k)
            if pixel_diff:
                raise RuntimeError(
                    "HALT (gate 7): non-deterministic PNG pixels between "
                    "two passes:\n  " + "\n  ".join(pixel_diff)
                )
        png_status = (
            f"full-file byte-identical: {full_match}/{len(png_keys)};"
            f" pixel-array sha256 match: {pixel_match}/{len(png_keys) - full_match}"
            if full_match < len(png_keys)
            else f"full-file byte-identical: {full_match}/{len(png_keys)}"
        )
        determinism_summary["csv_md"] = (
            f"all {sum(1 for k in shas_A if k.endswith('.csv') or k.endswith('.md'))} byte-identical"
        )
        determinism_summary["png"] = png_status

        # Use scratch_B as canonical -> move/copy to out_dir.
        print(f"[Finalise] copying scratch_B -> {out_dir}", flush=True)
        # clean out_dir first (excluding any existing plots subdir; rebuild)
        for f in out_dir.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
        for f in scratch_B.iterdir():
            if f.is_dir():
                shutil.copytree(f, out_dir / f.name)
            else:
                shutil.copy2(f, out_dir / f.name)
        shutil.rmtree(scratch_A, ignore_errors=True)
        shutil.rmtree(scratch_B, ignore_errors=True)
        final_shas = shas_B
    else:
        print("[Single-pass] skipping determinism check (debug mode)", flush=True)
        final_shas = build_pass(out_dir, verbose=True)
        determinism_summary["csv_md"] = "skipped (single-pass)"
        determinism_summary["png"] = "skipped (single-pass)"

    # --- Gate 8 — re-verify locked inputs post-run ---
    print("[Gate 8] re-verifying locked input sha256s...", flush=True)
    post_shas = verify_inputs("(gate 8)")
    if post_shas != input_shas:
        raise RuntimeError("HALT (gate 8): locked input sha256 drift post-run")
    print("  OK - all locked inputs unchanged", flush=True)

    # --- Gate 10 — git status ---
    print("[Gate 10] git status...", flush=True)
    import subprocess
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
    ).decode().strip()
    diff_out = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=str(REPO_ROOT)
    ).decode().strip()
    staged_out = subprocess.check_output(
        ["git", "diff", "--name-only", "--cached"], cwd=str(REPO_ROOT)
    ).decode().strip()
    if staged_out:
        print(f"  WARNING: staged files present:\n{staged_out}", flush=True)
    print(f"  HEAD: {git_head}", flush=True)

    # --- Write final manifest ---
    print("[Write] run_manifest.txt...", flush=True)
    # Recompute output sha256s after move (final_shas keys are correct).
    out_shas: Dict[str, str] = {}
    for f in sorted(out_dir.iterdir()):
        if f.is_file() and f.name != "run_manifest.txt":
            out_shas[f.name] = _sha256_file(f)
    plots_dir = out_dir / "plots"
    if plots_dir.exists():
        for f in sorted(plots_dir.iterdir()):
            if f.is_file():
                out_shas[f"plots/{f.name}"] = _sha256_file(f)
    write_manifest(
        out_dir, input_shas, out_shas, determinism_summary, git_head,
    )

    elapsed = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    print(f"[Done] wallclock={elapsed:.2f}s peak_python_alloc={peak_mb:.1f} MB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
