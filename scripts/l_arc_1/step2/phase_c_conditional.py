# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""Phase C — conditional breakdowns (op spec §5.9) and 2D joint distributions (§5.10).

Per-stratum aggregates of pool-level metrics:
  - pair (28 per-pair tables), fold (1..7), exit_reason, session, DOW, hour,
    vol_regime, pre_momentum_bin, trigger_magnitude_decile, hour_in_d1_bar,
    hour_in_4h_bar, first_bar_direction, first_bar_range_bin.

Sample-size discipline (op spec §5.9): cells with n<30 flagged; n<10 pooled
into `_insufficient_n` aggregate row.

2D joint distributions written as binned-count heatmaps (op spec §11.2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_1.step2._io import STEP2_DIR

COND_DIR = STEP2_DIR / "conditional_breakdowns"
JOINT_DIR = STEP2_DIR / "joint_distributions"

# Metrics to aggregate per stratum
AGG_METRICS = [
    "net_r",
    "gross_r",
    "mfe_held_atr",
    "mae_held_atr",
    "fwd_logret_h24",
    "fwd_mfe_h24_atr",
    "fwd_mae_h24_atr",
    "fwd_logret_h120",
    "fwd_mfe_h120_atr",
    "fwd_mae_h120_atr",
    "fwd_mfe_h240_atr",
    "fwd_mae_h240_atr",
    "race_bars_plus1_minus_minus1",
    "concurrent_signals_within_3h",
]


def _stratum_summary(values: pd.Series) -> dict:
    finite = values.dropna()
    n = len(finite)
    out = {
        "n": int(n),
        "mean": float(finite.mean()) if n else np.nan,
        "std": float(finite.std(ddof=1)) if n >= 2 else np.nan,
        "min": float(finite.min()) if n else np.nan,
        "p5": float(finite.quantile(0.05)) if n else np.nan,
        "p25": float(finite.quantile(0.25)) if n else np.nan,
        "p50": float(finite.median()) if n else np.nan,
        "p75": float(finite.quantile(0.75)) if n else np.nan,
        "p95": float(finite.quantile(0.95)) if n else np.nan,
        "max": float(finite.max()) if n else np.nan,
    }
    return out


def _emit_strat(f: pd.DataFrame, strat_col: str, sub: str) -> None:
    out_dir = COND_DIR / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pool with n<10 into insufficient_n
    counts = f.groupby(strat_col).size()
    insufficient = counts[counts < 10].index.tolist()
    f2 = f.copy()
    if insufficient:
        f2[strat_col] = f2[strat_col].astype(object)
        f2.loc[f2[strat_col].isin(insufficient), strat_col] = "_insufficient_n"

    for metric in AGG_METRICS:
        if metric not in f2.columns:
            continue
        rows = []
        for level, sub_df in f2.groupby(strat_col, dropna=False):
            d = _stratum_summary(sub_df[metric])
            d[strat_col] = str(level)
            d["flagged_n_lt_30"] = d["n"] < 30
            rows.append(d)
        cols = [
            strat_col,
            "n",
            "flagged_n_lt_30",
            "mean",
            "std",
            "min",
            "p5",
            "p25",
            "p50",
            "p75",
            "p95",
            "max",
        ]
        pd.DataFrame(rows)[cols].to_csv(
            out_dir / f"{metric}.csv",
            index=False,
            lineterminator="\n",
        )


def run_phase_c() -> None:
    print("[Phase C] reading signals_features.csv...")
    f = pd.read_csv(STEP2_DIR / "signals_features.csv")

    # Hour-in-bar conditionals + others
    strata = [
        ("pair", "pair"),
        ("fold_id", "fold"),
        ("exit_reason", "exit_reason"),
        ("session", "session"),
        ("day_of_week", "day_of_week"),
        ("hour_utc", "hour_of_day"),
        ("vol_regime", "vol_regime"),
        ("pre_momentum_bin", "pre_momentum"),
        ("trigger_magnitude_decile", "trigger_decile"),
        ("hour_in_d1_bar", "hour_in_d1_bar"),
        ("hour_in_4h_bar", "hour_in_4h_bar"),
        ("first_bar_direction", "first_bar_direction"),
        ("first_bar_range_bin", "first_bar_range"),
    ]
    for col, sub in strata:
        if col not in f.columns:
            continue
        print(f"[Phase C]   stratifying by {col}...")
        _emit_strat(f, col, sub)

    # 2D joint distributions (binned counts) — op spec §5.10
    print("[Phase C] 2D heatmaps...")
    JOINT_DIR.mkdir(parents=True, exist_ok=True)

    def _binify(s: pd.Series, n_bins: int):
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 1 or n_bins <= 1:
                # Degenerate; treat as one category
                return s.astype(str)
            lo = np.nanpercentile(s, 1)
            hi = np.nanpercentile(s, 99)
            if hi <= lo:
                return s.astype(str)
            edges = np.linspace(lo, hi, n_bins + 1)
            edges = np.unique(edges)
            if len(edges) < 2:
                return s.astype(str)
            return pd.cut(s, edges, include_lowest=True, duplicates="drop").astype(str)
        return s.astype(str)

    def heatmap(x_col, y_col, x_bins, y_bins, out_name):
        if x_col not in f.columns or y_col not in f.columns:
            return
        xb = _binify(f[x_col], x_bins)
        yb = _binify(f[y_col], y_bins)
        ct = pd.crosstab(xb, yb)
        ct.to_csv(JOINT_DIR / f"{out_name}.csv", lineterminator="\n")

    # Spec list — heatmaps. Use mfe_held_atr (== fwd_mfe_h1_atr) and so on.
    heatmap("mfe_held_atr", "mae_held_atr", 30, 30, "mfe_held__mae_held")
    heatmap("mfe_held_atr", "bars_held", 30, 5, "mfe_held__bars_held")
    heatmap("mae_held_atr", "bars_held", 30, 5, "mae_held__bars_held")
    heatmap("fwd_mfe_h24_atr", "fwd_time_to_peak_mfe", 30, 30, "fwd_mfe_h24__time_to_peak_mfe")
    heatmap("mfe_held_atr", "exit_reason", 30, 1, "mfe_held__exit_reason")
    heatmap("net_r", "bars_held", 30, 5, "net_r__bars_held")
    heatmap("net_r", "fwd_oscillation_count", 30, 20, "net_r__fwd_oscillation_count")
    heatmap("first_bar_direction", "net_r", 1, 30, "first_bar_direction__net_r")
    heatmap("mfe_sequence_class_fwd_h24", "net_r", 1, 30, "mfe_sequence_class_fwd_h24__net_r")
    heatmap("mfe_sequence_class_fwd_h120", "net_r", 1, 30, "mfe_sequence_class_fwd_h120__net_r")
    # Additional informative heatmaps
    heatmap("concurrent_signals_within_3h", "net_r", 20, 30, "concurrent_signals_within_3h__net_r")
    heatmap(
        "concurrent_signals_within_3h",
        "fwd_mae_h24_atr",
        20,
        30,
        "concurrent_signals_within_3h__fwd_mae_h24",
    )
    heatmap("hour_in_d1_bar", "net_r", 24, 30, "hour_in_d1_bar__net_r")
    heatmap("vol_regime", "fwd_mfe_h24_atr", 1, 30, "vol_regime__fwd_mfe_h24")

    print("[Phase C] done.")


if __name__ == "__main__":
    run_phase_c()
