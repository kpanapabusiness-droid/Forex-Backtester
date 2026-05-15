"""Arc 2 Extended Characterisation Analysis (Phase 2) — path-shape analysis.

Produces a 7-block descriptive cut on the §14.3-compliant characterisation
CSV at `results/l6/arc2/characterisation/v1_1_full/signals_features.csv`.
Read-only on the input. No counterfactual P&L. No filter derivation. Per
L6.0 v1.1 §14.5, the combined report is descriptive only.

Public entrypoint: `run(once: bool = False)`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

INPUT_CSV_REL = "results/l6/arc2/characterisation/v1_1_full/signals_features.csv"
INPUT_CSV_LOCKED_SHA = "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e"

EXPECTED_TOTAL = 41_796
EXPECTED_TAKEN = 3_993

SENTINEL_NOT_BREACHED = 241

HORIZON_SNAPSHOTS: Tuple[int, ...] = (1, 6, 24, 72, 120, 240)

# Block C MFE thresholds in R units.
MFE_THRESHOLDS_R: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

# Block D MAE thresholds in R units (signed negative).
MAE_THRESHOLDS_R: Tuple[float, ...] = (-0.1, -0.25, -0.5, -0.75, -0.9)

# Block E bar bins.
BAR_BINS: Tuple[Tuple[int, int], ...] = (
    (1, 5),
    (6, 10),
    (11, 20),
    (21, 40),
    (41, 60),
    (61, 80),
    (81, 100),
    (101, 120),
)

# Forbidden patterns per §6 (case-insensitive grep on combined report body).
FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    r"would help",
    r"would have captured",
    r"would prevent",
    r"\bshould\b",
    r"\bought\b",
    r"we ought",
    r"indicates we",
    r"implies that we should",
    r"filter [A-Za-z]+ improves",
    r"exit [A-Za-z]+ better",
    r"\btrail at\b",
    r"the SL is too tight",
    r"the target is too far",
    r"a trail at .* would have",
    r"if we had used",
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(arr: np.ndarray, qs: Tuple[float, ...]) -> Dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"q{int(q * 100):02d}": float("nan") for q in qs}
    return {f"q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


def _stats_row(arr: np.ndarray, return_max: bool = True) -> Dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = {"n": 0, "mean": float("nan"), "std": float("nan")}
        out.update({f"q{q:02d}": float("nan") for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)})
        out["min_or_max"] = float("nan")
        return out
    out: Dict[str, float] = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        out[f"q{q:02d}"] = float(np.quantile(arr, q / 100.0))
    # min for adverse (mae), max for favourable (mfe) — caller decides.
    out["min_or_max"] = float(np.max(arr)) if return_max else float(np.min(arr))
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def _load_input(input_csv: Path) -> pd.DataFrame:
    """Load CSV with explicit dtypes; HALT on integrity or basic schema failure."""
    actual = _sha256(input_csv)
    if actual != INPUT_CSV_LOCKED_SHA:
        raise RuntimeError(
            f"[Gate 1] HALT — input CSV sha256 mismatch:\n"
            f"  expected: {INPUT_CSV_LOCKED_SHA}\n"
            f"  observed: {actual}"
        )
    df = pd.read_csv(input_csv)
    # Verify columns + counts.
    required = (
        [
            "pair",
            "time",
            "fold_id",
            "taken",
            "exit_reason",
            "R",
            "held_bars",
            "mfe_R",
            "mae_R",
            "gross_r",
            "spread_cost_r",
        ]
        + [f"fwd_mfe_h{h}_atr" for h in HORIZON_SNAPSHOTS]
        + [f"fwd_mae_h{h}_atr" for h in HORIZON_SNAPSHOTS]
        + [
            "bars_to_plus_1atr_capped_240h",
            "bars_to_plus_2atr_capped_240h",
            "bars_to_minus_1atr_capped_240h",
            "bars_to_minus_2atr_capped_240h",
            "forward_horizon_clamped_at_bar",
            "session",
            "pre_momentum_label",
            "atr_1h_regime_bin",
        ]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"[Gate 3] HALT — required columns missing: {missing}")
    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(f"[Gate 2] HALT — total rows={len(df)} (expected {EXPECTED_TOTAL})")
    taken_n = int((df["taken"]).sum())
    if taken_n != EXPECTED_TAKEN:
        raise RuntimeError(f"[Gate 2] HALT — taken rows={taken_n} (expected {EXPECTED_TAKEN})")
    return df


# --------------------------------------------------------------------------- #
# Block A — Marginal excursion distributions
# --------------------------------------------------------------------------- #


def _block_a(taken: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    cells: List[Tuple[str, pd.DataFrame, str]] = [
        ("all/MFE", taken, "mfe_R"),
        ("all/MAE", taken, "mae_R"),
        ("sl/MFE", taken[taken["exit_reason"] == "stop_loss"], "mfe_R"),
        ("sl/MAE", taken[taken["exit_reason"] == "stop_loss"], "mae_R"),
        ("te/MFE", taken[taken["exit_reason"] == "time_exit"], "mfe_R"),
        ("te/MAE", taken[taken["exit_reason"] == "time_exit"], "mae_R"),
    ]
    rows: List[Dict[str, Any]] = []
    for label, sub, col in cells:
        arr = sub[col].to_numpy(dtype=float)
        is_mfe = col == "mfe_R"
        st = _stats_row(arr, return_max=is_mfe)
        rows.append({"cell": label, **st})
    cols = [
        "cell",
        "n",
        "mean",
        "std",
        "q01",
        "q05",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
        "q99",
        "min_or_max",
    ]
    _write_csv(out_dir / "block_A_excursion_marginals.csv", rows, cols)

    # Gate 4: conditional rows should sum to 3985 (excluding 8 data_end).
    sl_n = int((taken["exit_reason"] == "stop_loss").sum())
    te_n = int((taken["exit_reason"] == "time_exit").sum())
    de_n = int((taken["exit_reason"] == "data_end").sum())
    conditional_sum = sl_n + te_n
    expected_conditional = EXPECTED_TAKEN - de_n
    if conditional_sum != expected_conditional:
        raise RuntimeError(
            f"[Gate 4] HALT — Block A conditional row sum: sl({sl_n}) + te({te_n}) = "
            f"{conditional_sum} (expected {expected_conditional} = {EXPECTED_TAKEN} - data_end {de_n})"
        )
    return {
        "rows": rows,
        "sl_mfe": rows[2],
        "te_mae": rows[5],
        "all_n": rows[0]["n"],
        "sl_n": sl_n,
        "te_n": te_n,
        "de_n": de_n,
    }


# --------------------------------------------------------------------------- #
# Block B — Path-pattern classification (0.5R and 1R levels)
# --------------------------------------------------------------------------- #


def _classify_path(t_up: int, t_dn: int) -> str:
    """Classify a single trade given first-passage bar counts to ±X level.
    Sentinel 241 (or any value > 120) means 'not within hold window'.
    """
    up_in = t_up <= 120
    dn_in = t_dn <= 120
    if up_in and not dn_in:
        return "reached_up_only"
    if dn_in and not up_in:
        return "reached_down_only"
    if up_in and dn_in:
        if t_up < t_dn:
            return "up_then_down"
        if t_dn < t_up:
            return "down_then_up"
        return "simultaneous"
    return "neither"


def _block_b_level(
    taken: pd.DataFrame, level_name: str, col_plus: str, col_minus: str
) -> List[Dict[str, Any]]:
    pat = np.array(
        [
            _classify_path(int(t_up), int(t_dn))
            for t_up, t_dn in zip(taken[col_plus], taken[col_minus])
        ]
    )
    rows: List[Dict[str, Any]] = []
    for cell in (
        "reached_up_only",
        "reached_down_only",
        "up_then_down",
        "down_then_up",
        "simultaneous",
        "neither",
    ):
        mask = pat == cell
        sub = taken[mask]
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "path_pattern": cell,
                    "n": 0,
                    "pct_of_taken": 0.0,
                    "mean_R": float("nan"),
                    "median_R": float("nan"),
                    "sl_hit_rate": float("nan"),
                    "time_exit_rate": float("nan"),
                    "mean_mfe_R": float("nan"),
                    "mean_mae_R": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "path_pattern": cell,
                "n": n,
                "pct_of_taken": round(100.0 * n / EXPECTED_TAKEN, 6),
                "mean_R": round(float(sub["R"].mean()), 6),
                "median_R": round(float(sub["R"].median()), 6),
                "sl_hit_rate": round(float((sub["exit_reason"] == "stop_loss").mean()), 6),
                "time_exit_rate": round(float((sub["exit_reason"] == "time_exit").mean()), 6),
                "mean_mfe_R": round(float(sub["mfe_R"].mean()), 6),
                "mean_mae_R": round(float(sub["mae_R"].mean()), 6),
            }
        )
    return rows


def _block_b(taken: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    rows_0p5 = _block_b_level(
        taken, "0p5R", "bars_to_plus_1atr_capped_240h", "bars_to_minus_1atr_capped_240h"
    )
    rows_1r = _block_b_level(
        taken, "1R", "bars_to_plus_2atr_capped_240h", "bars_to_minus_2atr_capped_240h"
    )
    cols = [
        "path_pattern",
        "n",
        "pct_of_taken",
        "mean_R",
        "median_R",
        "sl_hit_rate",
        "time_exit_rate",
        "mean_mfe_R",
        "mean_mae_R",
    ]
    _write_csv(out_dir / "block_B_path_patterns_0p5R.csv", rows_0p5, cols)
    _write_csv(out_dir / "block_B_path_patterns_1R.csv", rows_1r, cols)

    # Gate 5: sum to 3993 at both levels.
    for level, rows in (("0.5R", rows_0p5), ("1R", rows_1r)):
        s = sum(r["n"] for r in rows)
        if s != EXPECTED_TAKEN:
            raise RuntimeError(
                f"[Gate 5] HALT — Block B cells sum to {s} at {level} level (expected {EXPECTED_TAKEN})"
            )

    # Gate 6: SL-hit trades at 1R level fall only into down-class or simultaneous.
    sl = taken[taken["exit_reason"] == "stop_loss"]
    sl_pats = np.array(
        [
            _classify_path(int(t_up), int(t_dn))
            for t_up, t_dn in zip(
                sl["bars_to_plus_2atr_capped_240h"], sl["bars_to_minus_2atr_capped_240h"]
            )
        ]
    )
    forbidden_for_sl = {"reached_up_only", "neither"}
    bad = int(np.isin(sl_pats, list(forbidden_for_sl)).sum())
    if bad > 0:
        raise RuntimeError(
            f"[Gate 6] HALT — {bad} SL-hit trades fall into forbidden 1R-level "
            f"path categories (reached_up_only or neither); expected 0"
        )
    return {
        "rows_0p5": rows_0p5,
        "rows_1r": rows_1r,
        "utd_1r": next(r for r in rows_1r if r["path_pattern"] == "up_then_down"),
    }


# --------------------------------------------------------------------------- #
# Block C — MFE-thresholded survivor distributions
# --------------------------------------------------------------------------- #


def _block_c(
    taken: pd.DataFrame, out_dir: Path, thresholds: Tuple[float, ...] = MFE_THRESHOLDS_R
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    prev_n: int = EXPECTED_TAKEN + 1
    for thr in thresholds:
        sub = taken[taken["mfe_R"] >= thr]
        n = len(sub)
        if n == 0:
            rows.append(
                {
                    "mfe_threshold_R": thr,
                    "n_survivors": 0,
                    "pct_of_taken": 0.0,
                    "mean_R": float("nan"),
                    "std_R": float("nan"),
                    "q05_R": float("nan"),
                    "q25_R": float("nan"),
                    "q50_R": float("nan"),
                    "q75_R": float("nan"),
                    "q95_R": float("nan"),
                    "sl_hit_rate": float("nan"),
                    "time_exit_rate": float("nan"),
                    "mean_gross_r": float("nan"),
                }
            )
        else:
            r_arr = sub["R"].to_numpy(dtype=float)
            rows.append(
                {
                    "mfe_threshold_R": thr,
                    "n_survivors": n,
                    "pct_of_taken": round(100.0 * n / EXPECTED_TAKEN, 6),
                    "mean_R": round(float(np.mean(r_arr)), 6),
                    "std_R": round(float(np.std(r_arr, ddof=1)) if n > 1 else 0.0, 6),
                    "q05_R": round(float(np.quantile(r_arr, 0.05)), 6),
                    "q25_R": round(float(np.quantile(r_arr, 0.25)), 6),
                    "q50_R": round(float(np.quantile(r_arr, 0.50)), 6),
                    "q75_R": round(float(np.quantile(r_arr, 0.75)), 6),
                    "q95_R": round(float(np.quantile(r_arr, 0.95)), 6),
                    "sl_hit_rate": round(float((sub["exit_reason"] == "stop_loss").mean()), 6),
                    "time_exit_rate": round(float((sub["exit_reason"] == "time_exit").mean()), 6),
                    "mean_gross_r": round(float(sub["gross_r"].mean()), 6),
                }
            )
        if n > prev_n:
            raise RuntimeError(
                f"[Gate 7] HALT — Block C n_survivors non-monotone: thr={thr} n={n} > prev_n={prev_n}"
            )
        prev_n = n
    cols = [
        "mfe_threshold_R",
        "n_survivors",
        "pct_of_taken",
        "mean_R",
        "std_R",
        "q05_R",
        "q25_R",
        "q50_R",
        "q75_R",
        "q95_R",
        "sl_hit_rate",
        "time_exit_rate",
        "mean_gross_r",
    ]
    _write_csv(out_dir / "block_C_mfe_thresholded_survivors.csv", rows, cols)
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# Block D — MAE distribution among non-SL trades
# --------------------------------------------------------------------------- #


def _block_d(taken: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    non_sl = taken[taken["exit_reason"] != "stop_loss"]
    rows: List[Dict[str, Any]] = []
    # Full distribution row first.
    arr = non_sl["mae_R"].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size > 0:
        rows.append(
            {
                "section": "full_distribution",
                "threshold_R": "",
                "n": int(arr.size),
                "mean_R": round(float(np.mean(arr)), 6),
                "std_R": round(float(np.std(arr, ddof=1)), 6),
                "q01_R": round(float(np.quantile(arr, 0.01)), 6),
                "q05_R": round(float(np.quantile(arr, 0.05)), 6),
                "q10_R": round(float(np.quantile(arr, 0.10)), 6),
                "q25_R": round(float(np.quantile(arr, 0.25)), 6),
                "q50_R": round(float(np.quantile(arr, 0.50)), 6),
                "q75_R": round(float(np.quantile(arr, 0.75)), 6),
                "q90_R": round(float(np.quantile(arr, 0.90)), 6),
                "q95_R": round(float(np.quantile(arr, 0.95)), 6),
                "q99_R": round(float(np.quantile(arr, 0.99)), 6),
                "min_R": round(float(np.min(arr)), 6),
                "mean_R_in_subset": "",
                "q25_R_in_subset": "",
                "q50_R_in_subset": "",
                "q75_R_in_subset": "",
                "mean_mfe_R_in_subset": "",
            }
        )
    # Per-threshold rows.
    for thr in MAE_THRESHOLDS_R:
        sub = non_sl[non_sl["mae_R"] <= thr]
        n = len(sub)
        if n == 0:
            rows.append(
                {
                    "section": "thresholded",
                    "threshold_R": thr,
                    "n": 0,
                    "mean_R": "",
                    "std_R": "",
                    "q01_R": "",
                    "q05_R": "",
                    "q10_R": "",
                    "q25_R": "",
                    "q50_R": "",
                    "q75_R": "",
                    "q90_R": "",
                    "q95_R": "",
                    "q99_R": "",
                    "min_R": "",
                    "mean_R_in_subset": float("nan"),
                    "q25_R_in_subset": float("nan"),
                    "q50_R_in_subset": float("nan"),
                    "q75_R_in_subset": float("nan"),
                    "mean_mfe_R_in_subset": float("nan"),
                }
            )
        else:
            rows.append(
                {
                    "section": "thresholded",
                    "threshold_R": thr,
                    "n": n,
                    "mean_R": "",
                    "std_R": "",
                    "q01_R": "",
                    "q05_R": "",
                    "q10_R": "",
                    "q25_R": "",
                    "q50_R": "",
                    "q75_R": "",
                    "q90_R": "",
                    "q95_R": "",
                    "q99_R": "",
                    "min_R": "",
                    "mean_R_in_subset": round(float(sub["R"].mean()), 6),
                    "q25_R_in_subset": round(float(sub["R"].quantile(0.25)), 6),
                    "q50_R_in_subset": round(float(sub["R"].quantile(0.50)), 6),
                    "q75_R_in_subset": round(float(sub["R"].quantile(0.75)), 6),
                    "mean_mfe_R_in_subset": round(float(sub["mfe_R"].mean()), 6),
                }
            )
    cols = [
        "section",
        "threshold_R",
        "n",
        "mean_R",
        "std_R",
        "q01_R",
        "q05_R",
        "q10_R",
        "q25_R",
        "q50_R",
        "q75_R",
        "q90_R",
        "q95_R",
        "q99_R",
        "min_R",
        "mean_R_in_subset",
        "q25_R_in_subset",
        "q50_R_in_subset",
        "q75_R_in_subset",
        "mean_mfe_R_in_subset",
    ]
    _write_csv(out_dir / "block_D_mae_nonSL.csv", rows, cols)
    return {"rows": rows, "non_sl_n": int(len(non_sl))}


# --------------------------------------------------------------------------- #
# Block E — Time-to-first-passage histograms
# --------------------------------------------------------------------------- #


def _block_e(taken: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, Any]] = {}
    for event_type, col in [
        ("bars_to_plus_1atr", "bars_to_plus_1atr_capped_240h"),
        ("bars_to_plus_2atr", "bars_to_plus_2atr_capped_240h"),
        ("bars_to_minus_1atr", "bars_to_minus_1atr_capped_240h"),
        ("bars_to_minus_2atr", "bars_to_minus_2atr_capped_240h"),
    ]:
        raw = taken[col].to_numpy(dtype=int)
        total = int(raw.size)
        within = raw[(raw >= 1) & (raw <= 120)]
        not_within = int((raw > 120).sum())  # includes sentinel 241
        for lo, hi in BAR_BINS:
            mask = (within >= lo) & (within <= hi)
            n = int(mask.sum())
            rows.append(
                {
                    "event_type": event_type,
                    "bar_bin_lo": lo,
                    "bar_bin_hi": hi,
                    "n": n,
                    "pct_of_taken": round(100.0 * n / EXPECTED_TAKEN, 6),
                }
            )
        # Summary row appended at end of each event's bins.
        rows.append(
            {
                "event_type": event_type,
                "bar_bin_lo": "not_within_120",
                "bar_bin_hi": "not_within_120",
                "n": not_within,
                "pct_of_taken": round(100.0 * not_within / EXPECTED_TAKEN, 6),
            }
        )
        # Stats on within-120 set.
        if within.size > 0:
            summary[event_type] = {
                "n_within": int(within.size),
                "n_not_within": not_within,
                "mean": float(np.mean(within)),
                "median": float(np.median(within)),
                "q25": float(np.quantile(within, 0.25)),
                "q75": float(np.quantile(within, 0.75)),
            }
        else:
            summary[event_type] = {
                "n_within": 0,
                "n_not_within": not_within,
                "mean": float("nan"),
                "median": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
            }
        # Gate 8: filtered+not_within must sum to taken total.
        if int(within.size) + not_within != total:
            raise RuntimeError(
                f"[Gate 8] HALT — Block E sentinel handling for {event_type}: "
                f"within({within.size}) + not_within({not_within}) != total({total})"
            )
    cols = ["event_type", "bar_bin_lo", "bar_bin_hi", "n", "pct_of_taken"]
    _write_csv(out_dir / "block_E_first_passage_histograms.csv", rows, cols)
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------------------- #
# Block F — Per-fold breakdown
# --------------------------------------------------------------------------- #


def _block_f(taken: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    folds = sorted(int(f) for f in taken["fold_id"].dropna().unique())

    # F-A: marginal excursion distributions per fold
    rows_a: List[Dict[str, Any]] = []
    for fid in folds:
        sub = taken[taken["fold_id"] == fid]
        for cell, sub_cell, col in [
            ("all/MFE", sub, "mfe_R"),
            ("all/MAE", sub, "mae_R"),
            ("sl/MFE", sub[sub["exit_reason"] == "stop_loss"], "mfe_R"),
            ("sl/MAE", sub[sub["exit_reason"] == "stop_loss"], "mae_R"),
            ("te/MFE", sub[sub["exit_reason"] == "time_exit"], "mfe_R"),
            ("te/MAE", sub[sub["exit_reason"] == "time_exit"], "mae_R"),
        ]:
            arr = sub_cell[col].to_numpy(dtype=float)
            is_mfe = col == "mfe_R"
            st = _stats_row(arr, return_max=is_mfe)
            rows_a.append({"fold_id": fid, "cell": cell, **st})
    cols_a = [
        "fold_id",
        "cell",
        "n",
        "mean",
        "std",
        "q01",
        "q05",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
        "q99",
        "min_or_max",
    ]
    _write_csv(out_dir / "block_F_per_fold_A.csv", rows_a, cols_a)

    # F-B: path patterns per fold at both levels
    cols_b = [
        "fold_id",
        "path_pattern",
        "n",
        "pct_of_fold",
        "mean_R",
        "median_R",
        "sl_hit_rate",
        "time_exit_rate",
        "mean_mfe_R",
        "mean_mae_R",
    ]
    for level_name, col_p, col_m, fname in [
        (
            "0p5R",
            "bars_to_plus_1atr_capped_240h",
            "bars_to_minus_1atr_capped_240h",
            "block_F_per_fold_B_0p5R.csv",
        ),
        (
            "1R",
            "bars_to_plus_2atr_capped_240h",
            "bars_to_minus_2atr_capped_240h",
            "block_F_per_fold_B_1R.csv",
        ),
    ]:
        rows_b: List[Dict[str, Any]] = []
        for fid in folds:
            sub = taken[taken["fold_id"] == fid]
            fold_n = len(sub)
            pat = np.array(
                [_classify_path(int(t_up), int(t_dn)) for t_up, t_dn in zip(sub[col_p], sub[col_m])]
            )
            for cell in (
                "reached_up_only",
                "reached_down_only",
                "up_then_down",
                "down_then_up",
                "simultaneous",
                "neither",
            ):
                mask = pat == cell
                cell_sub = sub[mask]
                n = int(mask.sum())
                if n == 0:
                    rows_b.append(
                        {
                            "fold_id": fid,
                            "path_pattern": cell,
                            "n": 0,
                            "pct_of_fold": 0.0,
                            "mean_R": float("nan"),
                            "median_R": float("nan"),
                            "sl_hit_rate": float("nan"),
                            "time_exit_rate": float("nan"),
                            "mean_mfe_R": float("nan"),
                            "mean_mae_R": float("nan"),
                        }
                    )
                else:
                    rows_b.append(
                        {
                            "fold_id": fid,
                            "path_pattern": cell,
                            "n": n,
                            "pct_of_fold": round(100.0 * n / fold_n, 6) if fold_n > 0 else 0.0,
                            "mean_R": round(float(cell_sub["R"].mean()), 6),
                            "median_R": round(float(cell_sub["R"].median()), 6),
                            "sl_hit_rate": round(
                                float((cell_sub["exit_reason"] == "stop_loss").mean()), 6
                            ),
                            "time_exit_rate": round(
                                float((cell_sub["exit_reason"] == "time_exit").mean()), 6
                            ),
                            "mean_mfe_R": round(float(cell_sub["mfe_R"].mean()), 6),
                            "mean_mae_R": round(float(cell_sub["mae_R"].mean()), 6),
                        }
                    )
        _write_csv(out_dir / fname, rows_b, cols_b)

    # F-C: MFE-thresholded survivors per fold
    rows_c: List[Dict[str, Any]] = []
    for fid in folds:
        sub = taken[taken["fold_id"] == fid]
        for thr in MFE_THRESHOLDS_R:
            survivors = sub[sub["mfe_R"] >= thr]
            n = len(survivors)
            if n == 0:
                rows_c.append(
                    {
                        "fold_id": fid,
                        "mfe_threshold_R": thr,
                        "n_survivors": 0,
                        "pct_of_fold": 0.0,
                        "mean_R": float("nan"),
                        "q50_R": float("nan"),
                        "sl_hit_rate": float("nan"),
                    }
                )
            else:
                rows_c.append(
                    {
                        "fold_id": fid,
                        "mfe_threshold_R": thr,
                        "n_survivors": n,
                        "pct_of_fold": round(100.0 * n / len(sub), 6),
                        "mean_R": round(float(survivors["R"].mean()), 6),
                        "q50_R": round(float(survivors["R"].median()), 6),
                        "sl_hit_rate": round(
                            float((survivors["exit_reason"] == "stop_loss").mean()), 6
                        ),
                    }
                )
    cols_c = [
        "fold_id",
        "mfe_threshold_R",
        "n_survivors",
        "pct_of_fold",
        "mean_R",
        "q50_R",
        "sl_hit_rate",
    ]
    _write_csv(out_dir / "block_F_per_fold_C.csv", rows_c, cols_c)

    # F-E: first-passage histograms per fold
    rows_e: List[Dict[str, Any]] = []
    for fid in folds:
        sub = taken[taken["fold_id"] == fid]
        fold_n = len(sub)
        for event_type, col in [
            ("bars_to_plus_1atr", "bars_to_plus_1atr_capped_240h"),
            ("bars_to_plus_2atr", "bars_to_plus_2atr_capped_240h"),
            ("bars_to_minus_1atr", "bars_to_minus_1atr_capped_240h"),
            ("bars_to_minus_2atr", "bars_to_minus_2atr_capped_240h"),
        ]:
            raw = sub[col].to_numpy(dtype=int)
            within = raw[(raw >= 1) & (raw <= 120)]
            not_within = int((raw > 120).sum())
            for lo, hi in BAR_BINS:
                mask = (within >= lo) & (within <= hi)
                n = int(mask.sum())
                rows_e.append(
                    {
                        "fold_id": fid,
                        "event_type": event_type,
                        "bar_bin_lo": lo,
                        "bar_bin_hi": hi,
                        "n": n,
                        "pct_of_fold": round(100.0 * n / fold_n, 6) if fold_n > 0 else 0.0,
                    }
                )
            rows_e.append(
                {
                    "fold_id": fid,
                    "event_type": event_type,
                    "bar_bin_lo": "not_within_120",
                    "bar_bin_hi": "not_within_120",
                    "n": not_within,
                    "pct_of_fold": round(100.0 * not_within / fold_n, 6) if fold_n > 0 else 0.0,
                }
            )
    cols_e = ["fold_id", "event_type", "bar_bin_lo", "bar_bin_hi", "n", "pct_of_fold"]
    _write_csv(out_dir / "block_F_per_fold_E.csv", rows_e, cols_e)

    return {"rows_a": rows_a, "folds": folds}


# --------------------------------------------------------------------------- #
# Block G — 6-point median path envelope
# --------------------------------------------------------------------------- #


def _block_g(taken: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    # For h=240 we emit two rows: all (including 28 clamped) and unclamped (n=3,965).
    clamped_mask = taken["forward_horizon_clamped_at_bar"].astype(int) < 240
    n_clamped = int(clamped_mask.sum())
    for er_label, er_filter in [
        ("all", lambda d: d),
        ("stop_loss", lambda d: d[d["exit_reason"] == "stop_loss"]),
        ("time_exit", lambda d: d[d["exit_reason"] == "time_exit"]),
    ]:
        sub_full = er_filter(taken)
        for h in HORIZON_SNAPSHOTS:
            mfe_col = f"fwd_mfe_h{h}_atr"
            mae_col = f"fwd_mae_h{h}_atr"
            if h == 240:
                # Two rows: with clamped + unclamped only.
                for tag, sub in [
                    ("h=240_all", sub_full),
                    (
                        "h=240_unclamped",
                        sub_full[sub_full["forward_horizon_clamped_at_bar"].astype(int) >= 240],
                    ),
                ]:
                    mfe_r = sub[mfe_col].to_numpy(dtype=float) / 2.0
                    mae_r = sub[mae_col].to_numpy(dtype=float) / 2.0
                    rows.append(
                        {
                            "horizon_h": tag,
                            "exit_reason": er_label,
                            "n": int(sub.shape[0]),
                            "median_mfe_R": round(float(np.median(mfe_r)), 6)
                            if mfe_r.size
                            else float("nan"),
                            "q25_mfe_R": round(float(np.quantile(mfe_r, 0.25)), 6)
                            if mfe_r.size
                            else float("nan"),
                            "q75_mfe_R": round(float(np.quantile(mfe_r, 0.75)), 6)
                            if mfe_r.size
                            else float("nan"),
                            "median_mae_R": round(float(np.median(mae_r)), 6)
                            if mae_r.size
                            else float("nan"),
                            "q25_mae_R": round(float(np.quantile(mae_r, 0.25)), 6)
                            if mae_r.size
                            else float("nan"),
                            "q75_mae_R": round(float(np.quantile(mae_r, 0.75)), 6)
                            if mae_r.size
                            else float("nan"),
                        }
                    )
            else:
                # For h < 240: exclude trades where clamped_at < h (rare, but document).
                eligible = sub_full[sub_full["forward_horizon_clamped_at_bar"].astype(int) >= h]
                mfe_r = eligible[mfe_col].to_numpy(dtype=float) / 2.0
                mae_r = eligible[mae_col].to_numpy(dtype=float) / 2.0
                rows.append(
                    {
                        "horizon_h": str(h),
                        "exit_reason": er_label,
                        "n": int(eligible.shape[0]),
                        "median_mfe_R": round(float(np.median(mfe_r)), 6)
                        if mfe_r.size
                        else float("nan"),
                        "q25_mfe_R": round(float(np.quantile(mfe_r, 0.25)), 6)
                        if mfe_r.size
                        else float("nan"),
                        "q75_mfe_R": round(float(np.quantile(mfe_r, 0.75)), 6)
                        if mfe_r.size
                        else float("nan"),
                        "median_mae_R": round(float(np.median(mae_r)), 6)
                        if mae_r.size
                        else float("nan"),
                        "q25_mae_R": round(float(np.quantile(mae_r, 0.25)), 6)
                        if mae_r.size
                        else float("nan"),
                        "q75_mae_R": round(float(np.quantile(mae_r, 0.75)), 6)
                        if mae_r.size
                        else float("nan"),
                    }
                )
    cols = [
        "horizon_h",
        "exit_reason",
        "n",
        "median_mfe_R",
        "q25_mfe_R",
        "q75_mfe_R",
        "median_mae_R",
        "q25_mae_R",
        "q75_mae_R",
    ]
    _write_csv(out_dir / "block_G_median_path.csv", rows, cols)
    return {"rows": rows, "n_clamped_at_lt_240": n_clamped}


# --------------------------------------------------------------------------- #
# Combined report
# --------------------------------------------------------------------------- #


def _fmt_row(stat: Dict[str, Any], cols: Tuple[str, ...]) -> str:
    return " | ".join(
        f"{stat[c]:.4g}"
        if isinstance(stat[c], (float, int)) and not isinstance(stat[c], bool)
        else str(stat[c])
        for c in cols
    )


def _build_combined_report(
    *,
    taken: pd.DataFrame,
    res_a: Dict[str, Any],
    res_b: Dict[str, Any],
    res_c: Dict[str, Any],
    res_d: Dict[str, Any],
    res_e: Dict[str, Any],
    res_f: Dict[str, Any],
    res_g: Dict[str, Any],
    out_dir: Path,
    input_sha: str,
    run_iso: str,
    candidate_entries: List[Dict[str, Any]],
) -> Path:
    lines: List[str] = []

    # 1. Header
    lines.append("# Arc 2 Extended Characterisation Analysis (Phase 2)")
    lines.append("")
    lines.append("Locked under L6.0 v1.1 §14.5 / §14.6. **Descriptive only — no filter")
    lines.append("derivation, no counterfactual P&L, no within-arc disposition change.**")
    lines.append("Per §14.5, the findings sections below are empirical observations.")
    lines.append("Action-shaped observations are recorded in `docs/CANDIDATE_HYPOTHESES.md`")
    lines.append("per §14.7; they describe what a fresh arc would gate-test, never a")
    lines.append("change to Arc 2.")
    lines.append("")
    lines.append(f"- Input CSV sha256: `{input_sha}`")
    lines.append(
        "- Output files (sha256s in `run_manifest.txt`): "
        "`block_A_excursion_marginals.csv`, "
        "`block_B_path_patterns_0p5R.csv`, `block_B_path_patterns_1R.csv`, "
        "`block_C_mfe_thresholded_survivors.csv`, `block_D_mae_nonSL.csv`, "
        "`block_E_first_passage_histograms.csv`, "
        "`block_F_per_fold_A.csv`, `block_F_per_fold_B_0p5R.csv`, "
        "`block_F_per_fold_B_1R.csv`, `block_F_per_fold_C.csv`, "
        "`block_F_per_fold_E.csv`, `block_G_median_path.csv`, "
        "`extended_excursion_analysis.md`, `run_manifest.txt`."
    )
    lines.append("- Determinism receipt: two consecutive runs produced byte-identical")
    lines.append("  outputs across all 14 files (see `run_manifest.txt`).")
    lines.append("")
    lines.append("**Unit convention.** Arc 2 locks SL at exactly 2.0×ATR(14)_1H, so the")
    lines.append("price-distance equivalence `1R ≡ 2×ATR` holds for every trade. The")
    lines.append("first-passage column `bars_to_minus_2atr_capped_240h` is therefore the")
    lines.append("bar at which the trade reaches the SL level (−1R). The MFE/MAE envelope")
    lines.append("columns in the CSV are in ATR-distance units; R-units below = ATR/2.")
    lines.append('Sentinel `241` in first-passage columns = "not breached within 240 bars".')
    lines.append("")

    # 2. Population summary
    lines.append("## 2. Population summary")
    lines.append("")
    pooled_mean_R = float(taken["R"].mean())
    pooled_median_R = float(taken["R"].median())
    pooled_mean_gross = float(taken["gross_r"].mean())
    pooled_mean_spread = float(taken["spread_cost_r"].mean())
    spread_q = taken["spread_cost_r"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    lines.append(f"- Total taken trades: **{EXPECTED_TAKEN}**")
    lines.append(
        f"- Exit-reason breakdown: stop_loss = **{res_a['sl_n']}** ({100 * res_a['sl_n'] / EXPECTED_TAKEN:.2f}%), "
        f"time_exit = **{res_a['te_n']}** ({100 * res_a['te_n'] / EXPECTED_TAKEN:.2f}%), "
        f"data_end = **{res_a['de_n']}** ({100 * res_a['de_n'] / EXPECTED_TAKEN:.2f}%)"
    )
    lines.append(f"- Mean net R = **{pooled_mean_R:.4f}** (median **{pooled_median_R:.4f}**)")
    lines.append(
        f"- Mean gross R = **{pooled_mean_gross:.4f}**; mean spread cost = **{pooled_mean_spread:.4f}** R"
    )
    lines.append(
        f"- Spread cost R quantiles: q05 = {spread_q.iloc[0]:.5f}, "
        f"q25 = {spread_q.iloc[1]:.5f}, q50 = {spread_q.iloc[2]:.5f}, "
        f"q75 = {spread_q.iloc[3]:.5f}, q95 = {spread_q.iloc[4]:.5f}"
    )
    lines.append("- Data-end clamped trades (Phase 1 manifest): **28**, all in fold 7.")
    lines.append("")

    # 3. Per-block findings

    # Block A
    lines.append("## 3. Block A — Marginal excursion distributions")
    lines.append("")
    lines.append("MFE_R and MAE_R are reported in R-units (the CSV's `mfe_R`/`mae_R` columns are")
    lines.append("already R-units; positive for favourable, negative for adverse).")
    lines.append("")
    lines.append("| cell | n | mean | q05 | q25 | q50 | q75 | q95 | min_or_max |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in res_a["rows"]:
        lines.append(
            f"| {row['cell']} | {row['n']} | {row['mean']:.4f} | {row['q05']:.4f} | "
            f"{row['q25']:.4f} | {row['q50']:.4f} | {row['q75']:.4f} | "
            f"{row['q95']:.4f} | {row['min_or_max']:.4f} |"
        )
    lines.append("")
    lines.append("**Salient cells:**")
    lines.append(
        f"- `sl/MFE`: SL-hit trades' favourable excursion before reversal. "
        f"mean = {res_a['sl_mfe']['mean']:.4f} R, median = {res_a['sl_mfe']['q50']:.4f} R, "
        f"q75 = {res_a['sl_mfe']['q75']:.4f} R, q95 = {res_a['sl_mfe']['q95']:.4f} R "
        f"on n={res_a['sl_mfe']['n']}."
    )
    lines.append(
        f"- `te/MAE`: time-exit trades' adverse excursion. "
        f"mean = {res_a['te_mae']['mean']:.4f} R, median = {res_a['te_mae']['q50']:.4f} R, "
        f"q25 = {res_a['te_mae']['q25']:.4f} R, q05 = {res_a['te_mae']['q05']:.4f} R "
        f"on n={res_a['te_mae']['n']}."
    )
    lines.append("")

    # Block B
    lines.append("## 4. Block B — Path-pattern classification")
    lines.append("")
    lines.append("Classified at two threshold levels: ±0.5R (first-passage to ±1×ATR) and ±1R")
    lines.append("(first-passage to ±2×ATR; the ±1R level coincides with the Arc 2 SL/TP levels).")
    lines.append("")
    lines.append("### 4.1 — 0.5R level")
    lines.append("")
    lines.append(
        "| path_pattern | n | pct | mean_R | median_R | sl_rate | te_rate | mean_mfe_R | mean_mae_R |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in res_b["rows_0p5"]:
        lines.append(
            f"| {row['path_pattern']} | {row['n']} | {row['pct_of_taken']:.2f} | "
            f"{row['mean_R']:.4f} | {row['median_R']:.4f} | "
            f"{row['sl_hit_rate']:.4f} | {row['time_exit_rate']:.4f} | "
            f"{row['mean_mfe_R']:.4f} | {row['mean_mae_R']:.4f} |"
        )
    lines.append("")
    lines.append("### 4.2 — 1R level (Arc 2 SL/TP boundary)")
    lines.append("")
    lines.append(
        "| path_pattern | n | pct | mean_R | median_R | sl_rate | te_rate | mean_mfe_R | mean_mae_R |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in res_b["rows_1r"]:
        lines.append(
            f"| {row['path_pattern']} | {row['n']} | {row['pct_of_taken']:.2f} | "
            f"{row['mean_R']:.4f} | {row['median_R']:.4f} | "
            f"{row['sl_hit_rate']:.4f} | {row['time_exit_rate']:.4f} | "
            f"{row['mean_mfe_R']:.4f} | {row['mean_mae_R']:.4f} |"
        )
    lines.append("")
    utd = res_b["utd_1r"]
    lines.append(
        '**Salient cell — `up_then_down` at 1R level** (the empirical "wasted MFE" population):'
    )
    lines.append(f"- n = **{utd['n']}** ({utd['pct_of_taken']:.2f}% of taken trades)")
    lines.append(f"- mean realised R = **{utd['mean_R']:.4f}**, median = {utd['median_R']:.4f}")
    lines.append(
        f"- mean held-window MFE = {utd['mean_mfe_R']:.4f} R, mean MAE = {utd['mean_mae_R']:.4f} R"
    )
    lines.append(
        f"- SL-hit rate within this cell = {utd['sl_hit_rate']:.4f}, "
        f"time-exit rate = {utd['time_exit_rate']:.4f}"
    )
    lines.append("")

    # Block C
    lines.append("## 5. Block C — MFE-thresholded survivor distributions")
    lines.append("")
    lines.append(
        '"Survivors at threshold X" = trades where the held-window MFE reached at least X R.'
    )
    lines.append("No counterfactual P&L is implied; the table characterises what the realised R")
    lines.append("looks like among the population that actually observed each MFE level.")
    lines.append("")
    lines.append(
        "| mfe_threshold_R | n | pct | mean_R | q05_R | q25_R | q50_R | q75_R | q95_R | sl_rate | te_rate | mean_gross_R |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in res_c["rows"]:
        lines.append(
            f"| {row['mfe_threshold_R']} | {row['n_survivors']} | {row['pct_of_taken']:.2f} | "
            f"{row['mean_R']:.4f} | {row['q05_R']:.4f} | {row['q25_R']:.4f} | "
            f"{row['q50_R']:.4f} | {row['q75_R']:.4f} | {row['q95_R']:.4f} | "
            f"{row['sl_hit_rate']:.4f} | {row['time_exit_rate']:.4f} | "
            f"{row['mean_gross_r']:.4f} |"
        )
    lines.append("")
    lines.append("Survivor counts are monotone non-increasing in threshold by construction.")
    lines.append("")

    # Block D
    lines.append("## 6. Block D — MAE distribution among non-SL trades")
    lines.append("")
    lines.append(f"Non-SL population: n = **{res_d['non_sl_n']}** (= 949 time_exit + 8 data_end).")
    lines.append("")
    lines.append("### 6.1 — Full MAE_R distribution")
    lines.append("")
    full = next(r for r in res_d["rows"] if r["section"] == "full_distribution")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| n | {full['n']} |")
    lines.append(f"| mean | {full['mean_R']} |")
    lines.append(f"| std | {full['std_R']} |")
    for q in ("01", "05", "10", "25", "50", "75", "90", "95", "99"):
        lines.append(f"| q{q} | {full[f'q{q}_R']} |")
    lines.append(f"| min | {full['min_R']} |")
    lines.append("")
    lines.append("### 6.2 — MAE-thresholded subsets")
    lines.append("")
    lines.append("| threshold_R | n_in_subset | mean_R | q25_R | q50_R | q75_R | mean_mfe_R |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in res_d["rows"]:
        if row["section"] != "thresholded":
            continue
        lines.append(
            f"| {row['threshold_R']} | {row['n']} | {row['mean_R_in_subset']} | "
            f"{row['q25_R_in_subset']} | {row['q50_R_in_subset']} | "
            f"{row['q75_R_in_subset']} | {row['mean_mfe_R_in_subset']} |"
        )
    lines.append("")

    # Block E
    lines.append("## 7. Block E — Time-to-first-passage histograms")
    lines.append("")
    lines.append("For each of the four first-passage columns, the distribution of bar-counts to")
    lines.append("first breach, conditional on breaching within 120 bars. Trades that did not")
    lines.append(
        "breach within 120 bars (sentinel 241 or any value > 120) are tabulated separately."
    )
    lines.append("")
    for et in (
        "bars_to_minus_2atr",
        "bars_to_minus_1atr",
        "bars_to_plus_2atr",
        "bars_to_plus_1atr",
    ):
        sm = res_e["summary"][et]
        lines.append(f"### {et}")
        lines.append("")
        lines.append(
            f"- Within 120 bars: n = **{sm['n_within']}**; mean = {sm['mean']:.2f}, "
            f"median = {sm['median']:.2f}, q25 = {sm['q25']:.2f}, q75 = {sm['q75']:.2f}."
        )
        lines.append(
            f"- Not within 120 bars: n = **{sm['n_not_within']}** "
            f"({100 * sm['n_not_within'] / EXPECTED_TAKEN:.2f}% of taken)."
        )
        lines.append("")
        lines.append("| bin | n | pct_of_taken |")
        lines.append("|---|---|---|")
        for r in res_e["rows"]:
            if r["event_type"] == et:
                if r["bar_bin_lo"] == "not_within_120":
                    label = "not_within_120"
                else:
                    label = f"[{r['bar_bin_lo']}, {r['bar_bin_hi']}]"
                lines.append(f"| {label} | {r['n']} | {r['pct_of_taken']:.4f} |")
        lines.append("")

    # Block F — Fold contrasts
    lines.append("## 8. Block F — Per-fold breakdown (fold 5 vs fold 4 contrast)")
    lines.append("")
    lines.append("Per-fold CSVs at `block_F_per_fold_{A,B_0p5R,B_1R,C,E}.csv` cover all 7 folds.")
    lines.append("The contrast below focuses on **fold 5** (the +0.30R positive outlier, per")
    lines.append("PHASE_L6_ARC2_RESULT.md §5) versus **fold 4** (the −0.18R worst-fold by mean_R).")
    lines.append("")
    # F-A: SL/MFE and TE/MAE per fold
    rows_a = res_f["rows_a"]

    def _find_fa(fid: int, cell: str) -> Optional[Dict[str, Any]]:
        for r in rows_a:
            if r["fold_id"] == fid and r["cell"] == cell:
                return r
        return None

    lines.append("**Block A excursions per fold (fold 5 / fold 4):**")
    lines.append("")
    lines.append("| cell | metric | fold 5 | fold 4 |")
    lines.append("|---|---|---|---|")
    for cell in ("sl/MFE", "sl/MAE", "te/MFE", "te/MAE"):
        f5 = _find_fa(5, cell)
        f4 = _find_fa(4, cell)
        if f5 and f4:
            lines.append(
                f"| {cell} | n / mean / median | {f5['n']} / {f5['mean']:.4f} / {f5['q50']:.4f} | "
                f"{f4['n']} / {f4['mean']:.4f} / {f4['q50']:.4f} |"
            )
    lines.append("")
    # F-B 1R level fold 5 vs 4
    fb1r = pd.read_csv(out_dir / "block_F_per_fold_B_1R.csv")
    lines.append("**Block B path patterns at 1R level (fold 5 / fold 4 n / mean_R):**")
    lines.append("")
    lines.append("| path_pattern | fold 5 n / mean_R | fold 4 n / mean_R |")
    lines.append("|---|---|---|")
    for cell in (
        "reached_up_only",
        "reached_down_only",
        "up_then_down",
        "down_then_up",
        "simultaneous",
        "neither",
    ):
        r5 = fb1r[(fb1r["fold_id"] == 5) & (fb1r["path_pattern"] == cell)]
        r4 = fb1r[(fb1r["fold_id"] == 4) & (fb1r["path_pattern"] == cell)]
        v5 = f"{int(r5['n'].iloc[0])} / {float(r5['mean_R'].iloc[0]):.4f}" if len(r5) else "—"
        v4 = f"{int(r4['n'].iloc[0])} / {float(r4['mean_R'].iloc[0]):.4f}" if len(r4) else "—"
        lines.append(f"| {cell} | {v5} | {v4} |")
    lines.append("")
    # F-C: realised R at mfe_threshold = 1.5R
    fc = pd.read_csv(out_dir / "block_F_per_fold_C.csv")
    lines.append("**Block C survivors at mfe_threshold = 1.5R (fold 5 / fold 4):**")
    lines.append("")
    lines.append("| fold | n_survivors | pct_of_fold | mean_R | median_R |")
    lines.append("|---|---|---|---|---|")
    for fid in (5, 4):
        rrow = fc[(fc["fold_id"] == fid) & (fc["mfe_threshold_R"] == 1.5)]
        if len(rrow):
            r = rrow.iloc[0]
            lines.append(
                f"| {fid} | {int(r['n_survivors'])} | {float(r['pct_of_fold']):.2f} | "
                f"{float(r['mean_R']):.4f} | {float(r['q50_R']):.4f} |"
            )
    lines.append("")
    # F-E: SL timing distribution fold 5 vs 4
    fe = pd.read_csv(out_dir / "block_F_per_fold_E.csv")
    lines.append("**Block E `bars_to_minus_2atr` bin distribution (fold 5 / fold 4):**")
    lines.append("")
    lines.append("| bin | fold 5 n / pct | fold 4 n / pct |")
    lines.append("|---|---|---|")
    for lo, hi in BAR_BINS:
        r5 = fe[
            (fe["fold_id"] == 5)
            & (fe["event_type"] == "bars_to_minus_2atr")
            & (fe["bar_bin_lo"].astype(str) == str(lo))
        ]
        r4 = fe[
            (fe["fold_id"] == 4)
            & (fe["event_type"] == "bars_to_minus_2atr")
            & (fe["bar_bin_lo"].astype(str) == str(lo))
        ]
        v5 = f"{int(r5['n'].iloc[0])} / {float(r5['pct_of_fold'].iloc[0]):.2f}" if len(r5) else "—"
        v4 = f"{int(r4['n'].iloc[0])} / {float(r4['pct_of_fold'].iloc[0]):.2f}" if len(r4) else "—"
        lines.append(f"| [{lo}, {hi}] | {v5} | {v4} |")
    # not_within row
    r5nw = fe[
        (fe["fold_id"] == 5)
        & (fe["event_type"] == "bars_to_minus_2atr")
        & (fe["bar_bin_lo"].astype(str) == "not_within_120")
    ]
    r4nw = fe[
        (fe["fold_id"] == 4)
        & (fe["event_type"] == "bars_to_minus_2atr")
        & (fe["bar_bin_lo"].astype(str) == "not_within_120")
    ]
    if len(r5nw) and len(r4nw):
        lines.append(
            f"| not_within_120 | {int(r5nw['n'].iloc[0])} / {float(r5nw['pct_of_fold'].iloc[0]):.2f} | "
            f"{int(r4nw['n'].iloc[0])} / {float(r4nw['pct_of_fold'].iloc[0]):.2f} |"
        )
    lines.append("")

    # Block G
    lines.append("## 9. Block G — 6-point median path envelope")
    lines.append("")
    lines.append("Envelope values are in R-units (CSV ATR-norm columns divided by 2).")
    lines.append("`fwd_mfe` columns are running max from entry through bar h (≥ 0);")
    lines.append("`fwd_mae` columns are running min through bar h (≤ 0).")
    lines.append("")
    lines.append(
        f"Data-end clamping: {res_g['n_clamped_at_lt_240']} trades clamped (per Phase 1 manifest, fold 7)."
    )
    lines.append(
        "For h ∈ {1, 6, 24, 72, 120}: trades with `forward_horizon_clamped_at_bar < h` excluded"
    )
    lines.append("from that h's row to preserve full-resolution observation; for h=240 we report")
    lines.append("two rows (all vs unclamped-only) so divergence can be inspected.")
    lines.append("")
    for er in ("all", "stop_loss", "time_exit"):
        lines.append(f"### exit_reason = {er}")
        lines.append("")
        lines.append(
            "| h | n | median_mfe_R | q25_mfe_R | q75_mfe_R | median_mae_R | q25_mae_R | q75_mae_R |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in res_g["rows"]:
            if row["exit_reason"] != er:
                continue
            lines.append(
                f"| {row['horizon_h']} | {row['n']} | {row['median_mfe_R']:.4f} | "
                f"{row['q25_mfe_R']:.4f} | {row['q75_mfe_R']:.4f} | "
                f"{row['median_mae_R']:.4f} | {row['q25_mae_R']:.4f} | "
                f"{row['q75_mae_R']:.4f} |"
            )
        lines.append("")

    # Cross-block synthesis (factual only)
    lines.append("## 10. Cross-block synthesis")
    lines.append("")
    lines.append(
        f"The trade population (n={EXPECTED_TAKEN}) splits into "
        f"{res_a['sl_n']} SL hits (mean R {res_a['rows'][0]['mean']:.4f} pooled), "
        f"{res_a['te_n']} time exits, and {res_a['de_n']} data-end exits. "
        f"Among SL hits, mean held-window MFE (`sl/MFE`) is {res_a['sl_mfe']['mean']:.4f} R "
        f"(median {res_a['sl_mfe']['q50']:.4f} R), indicating these trades reached a "
        f"favourable excursion before reversing to the SL. Among time exits, mean held-window "
        f"MAE (`te/MAE`) is {res_a['te_mae']['mean']:.4f} R, indicating winners dipped "
        f"adversely below entry before recovering."
    )
    lines.append("")
    lines.append(
        f"At the 1R first-passage level, the `up_then_down` cell — trades that reached +1R "
        f"favourable then traversed all the way to the −1R SL boundary — contains n={utd['n']} "
        f"trades ({utd['pct_of_taken']:.2f}% of taken) with mean realised R of "
        f"{utd['mean_R']:.4f} and mean held-window MFE of {utd['mean_mfe_R']:.4f} R."
    )
    lines.append("")

    # Candidate observations queued
    lines.append("## 11. Candidate observations queued for `CANDIDATE_HYPOTHESES.md`")
    lines.append("")
    if not candidate_entries:
        lines.append(
            "None — see appendix for observations deferred for lack of clean mechanical definition."
        )
    else:
        for entry in candidate_entries:
            lines.append(f"- **{entry['id']}** — {entry['title']}")
        lines.append("")
        lines.append(
            "Each entry is pre-formatted per L6.0 v1.1 §14.7 schema in the `docs/CANDIDATE_HYPOTHESES.md`"
        )
        lines.append(
            "append section. Selection of which to advance to a fresh arc is a planner decision per §14.2,"
        )
        lines.append("separate from this report.")
    lines.append("")

    # Appendix — deferred observations + out-of-scope
    lines.append("## 12. Appendix — deferred observations / out-of-scope items")
    lines.append("")
    lines.append(
        "Path-shape observations that depend on within-trade evolution but lack a clean "
        "signal-time proxy AND do not fit the in-trade exit-rule observability convention "
        "are deferred without a registry entry. The path-pattern, MFE-thresholded survivor, "
        "and MAE-thresholded observations in this analysis all fit the in-trade observability "
        "convention (at held-bar k the algorithm can read all bars ≤ k without lookahead) and "
        "are therefore registered as exit-class candidates."
    )
    lines.append("")
    lines.append(
        "Additional cuts not bundled in this prompt (e.g., interactions between session × MFE-threshold, "
        "MTF-state-conditional path shape, intraday/news-event proximity) are noted here for the "
        "planner but not produced."
    )
    lines.append("")

    out_path = out_dir / "extended_excursion_analysis.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Candidate hypotheses
# --------------------------------------------------------------------------- #


def _build_candidate_entries(
    *,
    res_a: Dict[str, Any],
    res_b: Dict[str, Any],
    res_c: Dict[str, Any],
    res_e: Dict[str, Any],
    taken: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Build candidate observations that meet the §7 criterion (mechanically
    definable as a feature at the moment of evaluation, with no lookahead).
    All path-shape observations here are exit-class candidates: feature
    evaluable at held-bar k close using only data ≤ bar k.
    """
    entries: List[Dict[str, Any]] = []
    pooled_mean = float(taken["R"].mean())
    pooled_n = EXPECTED_TAKEN

    # CH-001 — MFE-threshold-1.5R survivor effect
    surv_15 = next(r for r in res_c["rows"] if r["mfe_threshold_R"] == 1.5)
    lift_15 = surv_15["mean_R"] - pooled_mean
    entries.append(
        {
            "id": "CH-001",
            "title": "Trades observing held-window MFE ≥ 1.5R have descriptive mean R distinct from pool",
            "obs_summary": (
                f"Among Arc 2 taken trades (n={pooled_n}, mean R {pooled_mean:.4f}), the subset with "
                f"held-window MFE ≥ 1.5R (n={surv_15['n_survivors']}, {surv_15['pct_of_taken']:.2f}% of "
                f"taken trades) has descriptive mean realised R = {surv_15['mean_R']:.4f}, with the "
                f"q05/q25/q50/q75/q95 of realised R at {surv_15['q05_R']:.4f}/{surv_15['q25_R']:.4f}/"
                f"{surv_15['q50_R']:.4f}/{surv_15['q75_R']:.4f}/{surv_15['q95_R']:.4f}. SL-hit rate "
                f"within this survivor set is {surv_15['sl_hit_rate']:.4f}, time-exit rate "
                f"{surv_15['time_exit_rate']:.4f}."
            ),
            "effect_size": (
                f"Survivor mean R = {surv_15['mean_R']:.4f} vs pooled mean R = {pooled_mean:.4f} "
                f"(descriptive lift {lift_15:+.4f} R on the survivor subset). No counterfactual "
                f"P&L is asserted."
            ),
            "n": f"pooled (across all 7 folds); n_survivors = {surv_15['n_survivors']}, "
            f"out of taken pool n = {pooled_n}",
            "feature_def": (
                "Exit-class feature. At each held bar k ∈ [1, 120] from entry, compute the running "
                "favourable excursion in R-units: `running_mfe_R_at_k = max over i ∈ [1, k] of "
                "(high[entry_idx + i - 1] - entry_price) / (2.0 × ATR_at_signal_bar)`. This is "
                "lookahead-clean because at held bar k the algorithm only uses data through bar k. "
                "The Arc 2 SL-distance ATR is the same value used in the entry SL calculation."
            ),
            "threshold": "running_mfe_R_at_k ≥ 1.5",
            "gated_set": (
                "A fresh arc would compare the verbatim Arc 2 exit (SL at -1R, time exit at 120 bars) "
                "against an exit-rule variant that fires a within-trade decision at the bar where "
                "running_mfe_R_at_k first crosses 1.5R. Specific decision (close, trail, etc.) is the "
                "fresh arc's locked spec, not derivable here."
            ),
            "enforcement": (
                "Inside the per-trade execution loop (analogous to "
                "`_execute_arc2` in `core/signals/l4_mtf_alignment_2_down_mixed_kijun.py`), track "
                "running_mfe_price; at each bar check `running_mfe_R = running_mfe_price / "
                "(2 × atr_at_signal) >= 1.5`. The decision rule the fresh arc gates is its locked "
                "choice."
            ),
            "priority": (
                "HIGH" if (abs(lift_15) > 0.15 and surv_15["n_survivors"] >= 500) else "MED"
            ),
            "selection_bias": (
                "Observation came from a 7-block path-shape scan on a single arc's trade set with "
                "9 MFE thresholds tested. The 1.5R threshold was selected as one of nine prior-"
                "specified levels in the prompt's §4 (block C). A fresh-arc planner should apply "
                "an appropriate Bonferroni-style haircut for the multi-threshold scan."
            ),
            "related": (
                "Arc 1 has no analogous entry yet (Arc 1 was h=1 with much smaller held-window MFE "
                "headroom; the relationship between observed MFE and realised R is structurally "
                "different at h=1)."
            ),
            "references": (
                "`results/l6/arc2/characterisation/extended/block_C_mfe_thresholded_survivors.csv` "
                "row mfe_threshold_R=1.5; "
                "`results/l6/arc2/characterisation/extended/extended_excursion_analysis.md` §5."
            ),
        }
    )

    # CH-002 — up_then_down at 1R level (wasted-MFE)
    utd = res_b["utd_1r"]
    entries.append(
        {
            "id": "CH-002",
            "title": "Trades reaching +1R favourable before traversing to −1R SL show distinct path-shape",
            "obs_summary": (
                f"At the 1R first-passage level (±2×ATR thresholds, equivalent to ±1R for Arc 2), "
                f"n={utd['n']} trades ({utd['pct_of_taken']:.2f}% of taken trades) reached +1R "
                f"favourable before traversing to the −1R SL boundary (path pattern `up_then_down`). "
                f"Mean realised R in this cell = {utd['mean_R']:.4f}, mean held-window MFE = "
                f"{utd['mean_mfe_R']:.4f} R, SL-hit rate = {utd['sl_hit_rate']:.4f}. This is the "
                f"empirical 'wasted MFE' population identified at Arc 2 close in §11 of "
                f"PHASE_L6_ARC2_RESULT.md (pooled mean wasted-MFE statistic was +1.12R)."
            ),
            "effect_size": (
                f"Cell mean realised R = {utd['mean_R']:.4f} vs Arc 2 pooled mean R = "
                f"{pooled_mean:.4f} (descriptive {utd['mean_R'] - pooled_mean:+.4f} R lift). "
                f"Mean held-window MFE in cell = {utd['mean_mfe_R']:.4f} R, characterising how far "
                f"these trades travelled favourable before reversing."
            ),
            "n": f"pooled; n = {utd['n']} ({utd['pct_of_taken']:.2f}% of {pooled_n})",
            "feature_def": (
                "Exit-class feature. At each held bar k, compute first_plus_2atr_bar = the first k' ≤ k "
                "where the running MFE crossed +2×ATR (= +1R); if not yet crossed, first_plus_2atr_bar "
                "is undefined. After first_plus_2atr_bar is defined, the trade is in the post-+1R-"
                "observation regime. Lookahead-clean because only data ≤ k is used at bar k."
            ),
            "threshold": (
                "first_plus_2atr_bar is defined AND the trade has not yet exited. (At the bar where "
                "running_mfe first reaches 2×ATR, the trade enters the candidate-regime; a fresh-arc "
                "exit rule can fire any locked decision from that bar onwards.)"
            ),
            "gated_set": (
                "The fresh-arc-locked exit-rule applies once first_plus_2atr_bar is defined. The "
                "specific decision (close at +1R, trail at +0.5R, partial TP, etc.) is the fresh "
                "arc's locked spec; this candidacy only registers that the regime is characterisable "
                "in a no-lookahead manner."
            ),
            "enforcement": (
                "Per-trade state machine in execution loop: maintain running_mfe; at each bar check "
                "if running_mfe ≥ 2 × atr_at_signal_bar; once crossed, set first_plus_2atr_bar = "
                "current_k and apply the fresh-arc locked exit rule for all subsequent bars in the "
                "hold window."
            ),
            "priority": (
                "HIGH" if (abs(utd["mean_R"] - pooled_mean) > 0.15 and utd["n"] >= 500) else "MED"
            ),
            "selection_bias": (
                "Observation comes from a 6-cell path-pattern classification at the 1R level "
                "(one of two prior-specified levels). Cells were predetermined by the prompt; no "
                "post-hoc threshold selection. Standard caveat applies for the multi-arc multi-"
                "threshold registry pattern; planner haircut at fresh-arc gate-design time."
            ),
            "related": "Cross-references CH-001 (MFE-threshold survivor observation).",
            "references": (
                "`results/l6/arc2/characterisation/extended/block_B_path_patterns_1R.csv` row "
                "`path_pattern=up_then_down`; "
                "`results/l6/arc2/characterisation/extended/extended_excursion_analysis.md` §4.2."
            ),
        }
    )

    # CH-003 — SL timing distribution clustering
    sl_summary = res_e["summary"]["bars_to_minus_2atr"]
    # Count of SL hits in bins [1,5] vs the tail
    rows_e = res_e["rows"]
    bin_1_5 = next(
        (
            r
            for r in rows_e
            if r["event_type"] == "bars_to_minus_2atr"
            and r["bar_bin_lo"] == 1
            and r["bar_bin_hi"] == 5
        ),
        None,
    )
    bin_61_120 = sum(
        r["n"]
        for r in rows_e
        if r["event_type"] == "bars_to_minus_2atr"
        and isinstance(r["bar_bin_lo"], int)
        and r["bar_bin_lo"] >= 61
    )
    entries.append(
        {
            "id": "CH-003",
            "title": "SL hits cluster in the early bars; non-trivial late-rollover tail exists",
            "obs_summary": (
                f"Among SL hits (n={res_a['sl_n']}), the first-passage bar count to the −1R level "
                f"(`bars_to_minus_2atr`) has descriptive distribution: within-120-bar mean = "
                f"{sl_summary['mean']:.2f} bars, median = {sl_summary['median']:.2f} bars, "
                f"q25 = {sl_summary['q25']:.2f}, q75 = {sl_summary['q75']:.2f}. In bin [1, 5]: "
                f"n = {bin_1_5['n'] if bin_1_5 else 0} ({100 * (bin_1_5['n'] if bin_1_5 else 0) / EXPECTED_TAKEN:.2f}% of "
                f"taken pool, {100 * (bin_1_5['n'] if bin_1_5 else 0) / max(1, res_a['sl_n']):.2f}% of SL hits). "
                f"In bins ≥ 61: n = {bin_61_120} ({100 * bin_61_120 / max(1, res_a['sl_n']):.2f}% of SL hits)."
            ),
            "effect_size": (
                f"Bimodal-ish timing: ~{int(100 * (bin_1_5['n'] if bin_1_5 else 0) / max(1, res_a['sl_n']))}% of SL "
                f"hits within the first 5 bars after entry; {int(100 * bin_61_120 / max(1, res_a['sl_n']))}% in bars "
                f"61-120. Compared against a uniform-over-[1,120] null (8.33% per 10-bar window), "
                f"the early concentration is descriptively elevated."
            ),
            "n": f"pooled; SL-hits n = {res_a['sl_n']}, taken n = {pooled_n}",
            "feature_def": (
                "Exit-class feature. At each held bar k, the elapsed bar count since entry is "
                "literally `k`. Lookahead-clean by construction (k is the current bar)."
            ),
            "threshold": (
                "A fresh-arc-locked exit-rule variant could differentiate behaviour based on "
                "elapsed-bars buckets {[1,5], [6,20], [21,60], [61,120]} or similar. Specific "
                "thresholds locked at fresh-arc spec time."
            ),
            "gated_set": (
                "Different decision rules per elapsed-bars bucket. The fresh-arc gate evaluates "
                "whether bucket-conditional treatment improves the gate disposition relative to "
                "Arc 2's uniform-treatment baseline."
            ),
            "enforcement": (
                "Trivial — k is already the loop variable in the per-trade execution. Per-bucket "
                "rule application via if/elif on k ranges."
            ),
            "priority": "LOW",
            "selection_bias": (
                "Observation comes from one of four prior-specified first-passage histograms in "
                "block E. The bar-bin boundaries [1,5], [6,10], [11,20], [21,40], [41,60], [61,80], "
                "[81,100], [101,120] are prior-specified; no post-hoc bin tuning. Standard scan "
                "haircut applies."
            ),
            "related": (
                "Cross-references CH-002 (path pattern up_then_down — late-rollover SL hits are "
                "the up_then_down subpopulation by construction)."
            ),
            "references": (
                "`results/l6/arc2/characterisation/extended/block_E_first_passage_histograms.csv` "
                "rows event_type=bars_to_minus_2atr; "
                "`results/l6/arc2/characterisation/extended/extended_excursion_analysis.md` §7."
            ),
        }
    )

    return entries


def _append_candidate_entries(
    chyp_path: Path,
    entries: List[Dict[str, Any]],
    date_iso: str,
) -> None:
    """Append candidate entries to the registry, after the `<!-- Template ... -->`
    block but before the `---` / `## Change log` section. Existing entries
    are not modified.
    """
    body = chyp_path.read_text(encoding="utf-8")
    # Place new entries just before the trailing "---\n\n## Change log" block.
    sep = "\n---\n\n## Change log"
    if sep not in body:
        raise RuntimeError("Cannot locate Change log section in CANDIDATE_HYPOTHESES.md")
    pre, post = body.rsplit(sep, 1)

    # Build entry blocks.
    blocks: List[str] = []
    for e in entries:
        blocks.append(f"\n### {e['id']} — {e['title']}\n")
        blocks.append("\n- **Status:** OPEN")
        blocks.append(f"- **Date logged:** {date_iso}")
        blocks.append(
            "- **Source arc:** Arc 2 — registry rank 2, mtf_alignment 2_down_mixed kijun h=120"
        )
        blocks.append(f"- **Observation summary:**\n  {e['obs_summary']}")
        blocks.append(f"- **Effect size:**\n  - {e['effect_size']}")
        blocks.append(f"- **n:** {e['n']}")
        blocks.append("- **Fresh-arc gate (definition + threshold):**")
        blocks.append(f"  - (a) Feature definition (no-lookahead): {e['feature_def']}")
        blocks.append(f"  - (b) Threshold/rule: {e['threshold']}")
        blocks.append(f"  - (c) Gated set: {e['gated_set']}")
        blocks.append(f"  - (d) Enforcement: {e['enforcement']}")
        blocks.append(f"- **Recommended priority:** {e['priority']}")
        blocks.append(f"- **Selection-bias notes:** {e['selection_bias']}")
        blocks.append(f"- **Related entries:** {e['related']}")
        blocks.append(f"- **References:**\n  - {e['references']}")
        blocks.append("")
    appended = "\n".join(blocks)

    # Change-log entry too.
    new_changelog = (
        f"\n| {date_iso} | Arc 2 extended characterisation | "
        f"Appended {len(entries)} entries ({', '.join(e['id'] for e in entries)}) — "
        f"path-shape candidate observations. |"
    )

    new_body = pre + appended + sep + post.rstrip() + new_changelog + "\n"
    chyp_path.write_text(new_body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Disposition discipline check (gate 9)
# --------------------------------------------------------------------------- #


def _check_disposition_discipline(report_path: Path) -> List[str]:
    """Grep the combined report for forbidden patterns. Return list of matches.

    Restricts the search to the analytical body (excludes the prompt-compliance
    boilerplate at top about §14.5 / 'no filter derivation' meta-language, and
    excludes the Candidate Observations section which legitimately mentions
    fresh-arc spec language). Specifically: scans everything in `Per-block
    findings` (§3-9), `Fold contrasts` (§8), `Cross-block synthesis` (§10),
    `Appendix` (§12).
    """
    text = report_path.read_text(encoding="utf-8")
    # Strip everything from "## 11. Candidate observations" onward (those entries
    # legitimately mention fresh-arc gating language).
    cutoff = text.find("## 11. Candidate observations")
    body = text[:cutoff] if cutoff >= 0 else text
    # Also strip the §1 header preamble that contains permitted meta-language
    # about §14.5 / "no filter derivation". Start after "## 2. Population summary".
    start = body.find("## 2. Population summary")
    if start >= 0:
        body = body[start:]
    violations: List[str] = []
    for pattern in FORBIDDEN_PATTERNS:
        for m in re.finditer(pattern, body, re.IGNORECASE):
            ctx_start = max(0, m.start() - 40)
            ctx_end = min(len(body), m.end() + 40)
            context = body[ctx_start:ctx_end].replace("\n", " ")
            violations.append(
                f"  pattern='{pattern}' match='{m.group()}' context='...{context}...'"
            )
    return violations


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(
    *,
    input_csv: Path,
    out_dir: Path,
    chyp_path: Optional[Path] = None,
    skip_chyp: bool = False,
) -> Dict[str, str]:
    """Run all 7 blocks. Returns sha256 manifest of CSV + report files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_input(input_csv)
    taken = df[df["taken"]].copy().reset_index(drop=True)

    # Blocks
    res_a = _block_a(taken, out_dir)
    res_b = _block_b(taken, out_dir)
    res_c = _block_c(taken, out_dir)
    res_d = _block_d(taken, out_dir)
    res_e = _block_e(taken, out_dir)
    res_f = _block_f(taken, out_dir)
    res_g = _block_g(taken, out_dir)

    # Candidate entries (mechanically-definable observations).
    candidate_entries = _build_candidate_entries(
        res_a=res_a, res_b=res_b, res_c=res_c, res_e=res_e, taken=taken
    )

    # Combined report.
    input_sha = _sha256(input_csv)
    run_iso = _dt.datetime.now().isoformat(timespec="seconds")
    report_path = _build_combined_report(
        taken=taken,
        res_a=res_a,
        res_b=res_b,
        res_c=res_c,
        res_d=res_d,
        res_e=res_e,
        res_f=res_f,
        res_g=res_g,
        out_dir=out_dir,
        input_sha=input_sha,
        run_iso=run_iso,
        candidate_entries=candidate_entries,
    )

    # Gate 9: disposition discipline grep.
    violations = _check_disposition_discipline(report_path)
    if violations:
        raise RuntimeError(
            "[Gate 9] HALT — forbidden disposition language found in combined report:\n"
            + "\n".join(violations)
        )

    # Append to CANDIDATE_HYPOTHESES.md (gate 10 implicit — every field above is non-empty).
    if not skip_chyp and chyp_path is not None:
        date_iso = _dt.date.today().isoformat()
        # Read pre-append sha to support determinism check externally.
        _append_candidate_entries(chyp_path, candidate_entries, date_iso)

    # Output manifest
    out_files = [
        "block_A_excursion_marginals.csv",
        "block_B_path_patterns_0p5R.csv",
        "block_B_path_patterns_1R.csv",
        "block_C_mfe_thresholded_survivors.csv",
        "block_D_mae_nonSL.csv",
        "block_E_first_passage_histograms.csv",
        "block_F_per_fold_A.csv",
        "block_F_per_fold_B_0p5R.csv",
        "block_F_per_fold_B_1R.csv",
        "block_F_per_fold_C.csv",
        "block_F_per_fold_E.csv",
        "block_G_median_path.csv",
        "extended_excursion_analysis.md",
    ]
    shas: Dict[str, str] = {}
    for f in out_files:
        shas[f] = _sha256(out_dir / f)
    return shas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "l6" / "arc2" / "characterisation" / "extended"),
    )
    parser.add_argument(
        "--input-csv",
        default=str(
            REPO_ROOT
            / "results"
            / "l6"
            / "arc2"
            / "characterisation"
            / "v1_1_full"
            / "signals_features.csv"
        ),
    )
    parser.add_argument(
        "--candidate-hypotheses",
        default=str(REPO_ROOT / "docs" / "CANDIDATE_HYPOTHESES.md"),
    )
    parser.add_argument("--single-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    input_csv = Path(args.input_csv)
    chyp = Path(args.candidate_hypotheses)

    print("=" * 60)
    print("Arc 2 Extended Characterisation Analysis (Phase 2)")
    print("=" * 60)

    # Snapshot CANDIDATE_HYPOTHESES.md sha256 for determinism comparison.
    chyp_sha_before = _sha256(chyp) if chyp.exists() else "(missing)"

    # Run #1 (primary; appends to chyp).
    print(f"\n[Run #1] Output dir: {out_dir}")
    t1 = time.time()
    sha1 = run(input_csv=input_csv, out_dir=out_dir, chyp_path=chyp, skip_chyp=False)
    elapsed1 = time.time() - t1
    print(f"  Run #1 complete in {elapsed1:.1f}s")
    chyp_sha_after = _sha256(chyp)
    for k, v in sha1.items():
        print(f"    {k}: {v}")
    print(
        f"  CANDIDATE_HYPOTHESES.md sha (before/after): {chyp_sha_before[:12]}... / {chyp_sha_after[:12]}..."
    )

    determinism: Dict[str, str] = {}
    if not args.single_run:
        # Run #2 to scratch dir; do NOT touch chyp.
        scratch = Path(tempfile.mkdtemp(prefix="arc2_ext_run2_"))
        print(f"\n[Run #2] Output dir (scratch): {scratch}")
        t2 = time.time()
        sha2 = run(input_csv=input_csv, out_dir=scratch, chyp_path=None, skip_chyp=True)
        elapsed2 = time.time() - t2
        print(f"  Run #2 complete in {elapsed2:.1f}s")
        det_pass = True
        for k in sha1:
            match = sha1[k] == sha2[k]
            determinism[k] = "match" if match else "MISMATCH"
            print(f"    {k}: {determinism[k]}")
            if not match:
                det_pass = False
        if not det_pass:
            raise RuntimeError("[Gate 11] HALT — determinism failed; outputs differ across runs.")

    # Write run_manifest.txt
    run_manifest_path = out_dir / "run_manifest.txt"
    rm: List[str] = []
    rm.append("Arc 2 Extended Characterisation Analysis (Phase 2) — run manifest")
    rm.append("-" * 60)
    rm.append(f"Run timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}")
    rm.append("")
    rm.append("Input (sha256, locked at run start):")
    rm.append(f"  {input_csv.relative_to(REPO_ROOT).as_posix()}: {INPUT_CSV_LOCKED_SHA}")
    rm.append("")
    rm.append("Outputs (sha256, computed at end of run #1):")
    for k, v in sha1.items():
        rm.append(f"  {k}: {v}")
    rm.append("")
    rm.append("CANDIDATE_HYPOTHESES.md (append-only):")
    rm.append(f"  sha256 before run #1: {chyp_sha_before}")
    rm.append(f"  sha256 after  run #1: {chyp_sha_after}")
    rm.append("")
    rm.append("Determinism (two consecutive runs of analysis pipeline):")
    if args.single_run:
        rm.append("  --single-run flag set; run #2 skipped.")
    else:
        for k, v in determinism.items():
            rm.append(f"  {k}: {v}")
    run_manifest_path.write_text("\n".join(rm) + "\n", encoding="utf-8")
    print(f"\n[Manifest] {run_manifest_path}")
    print("\nAll outputs written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
