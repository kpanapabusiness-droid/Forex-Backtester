# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""Shared utilities for L Arc 1 Step 3 extractability scripts.

Determinism: all random seeds derive from BASE_SEED. All sorts are explicit.
All file writes go through write_csv() and write_text() which use a fixed
formatting/float precision so re-runs are byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths --------------------------------------------------------------

REPO = Path(__file__).resolve().parents[3]
STEP2_DIR = REPO / "results" / "l_arc_1" / "step2_descriptive"
SIGNALS_CSV = STEP2_DIR / "signals_features.csv"
PATHS_CSV = STEP2_DIR / "trade_paths.csv"
FWD_CTX = STEP2_DIR / "forward_context_evolution"
OUT_DIR = REPO / "results" / "l_arc_1" / "step3_extractability"
STRAT_DIR = OUT_DIR / "stratifications"

# --- seeds --------------------------------------------------------------

BASE_SEED = 1234
N_PERMUTATIONS = 1000  # v1.1 target; floor 500 per Amendment 11 / op spec §6.7
WARD_SUBSAMPLE = 10_000  # subsample size for hierarchical ward (memory)
N_FOLDS = 7

# --- whitelists ---------------------------------------------------------

# Identity columns (never used as predictors)
IDENTITY_COLS = (
    "trade_id",
    "pair",
    "fold_id",
    "signal_bar_ts",
    "entry_bar_ts",
    "exit_bar_ts",
    "direction",
)

# Trade outcome columns — excluded from predictor sets
OUTCOME_COLS = (
    "net_r",
    "gross_r",
    "spread_cost_R",
    "mfe_R",
    "mae_R",
    "bars_held",
    "exit_reason",
    "spread_pips_entry",
    "spread_pips_exit",
    "spread_floored",
    "sl_distance_atr",
    "sl_distance_price",
    "mfe_held_atr",
    "mae_held_atr",
    "peak_to_final_r_ratio",
    "mfe_to_mae_ratio_held",
    "mfe_sequence_class_held",
    "time_to_peak_mfe_held",
    "time_to_trough_mae_held",
)

# Bars observed only after entry / first-bar columns — NOT signal-time
# but available at t=1 for 3d.
FIRST_BAR_COLS = ("first_bar_direction", "first_bar_range_atr", "first_bar_range_bin")

# Numeric signal-time predictors (computable at bar N close)
SIGNAL_TIME_NUMERIC = (
    "signal_bar_open",
    "signal_bar_close",
    "signal_bar_high",
    "signal_bar_low",
    "signal_bar_log_return",
    "signal_bar_abs_log_return",
    "signal_threshold_q90",
    "trigger_excess",
    "trigger_ratio",
    "signal_bar_volume",
    "signal_bar_volume_nan",
    "atr_at_signal_1h",
    "atr_baseline_1h_200",
    "atr_ratio_to_baseline",
    "cum_logret_1h_6",
    "cum_logret_1h_3",
    "dist_close_to_high30_atr",
    "dist_close_to_low30_atr",
    "hour_utc",
    "day_of_week",
    "hour_in_4h_bar",
    "bars_to_next_4h_close",
    "hour_in_d1_bar",
    "bars_to_next_d1_close",
    "concurrent_signals_same_bar",
    "concurrent_signals_within_3h",
    "currency_basket_3h_USD",
    "currency_basket_3h_EUR",
    "currency_basket_3h_JPY",
    "currency_basket_3h_GBP",
    "trade_overlap_at_execution_time",
    "sequential_same_pair_density_24h",
    "trigger_magnitude_decile",
)

# Categorical signal-time predictors (one-hot encoded)
SIGNAL_TIME_CATEGORICAL = ("pair", "session", "vol_regime", "pre_momentum_bin")

# Forward-context observation features (per t-slice, in forward_context_evolution/t{t}.csv)
FWD_CTX_NUMERIC = (
    "atr_regime_ratio",
    "broker_spread_pips_raw",
    "broker_spread_pips_floored",
    "cross_pair_dispersion_proxy",
    "basket_cum_logret_USD",
    "basket_cum_logret_EUR",
    "basket_cum_logret_JPY",
    "basket_cum_logret_GBP",
)

HELD_BAR_TS = (1, 3, 5, 10, 20)

# --- v1.1 clustering feature subset (Amendment 4) -----------------------
# h=1 trades have degenerate held window. The 12-feature v1.0 set is
# extended by three path-geometry features (Amendment 4):
#   fwd_realized_range_atr, fwd_fraction_time_above_entry,
#   fwd_max_consecutive_directional_bars.
# Plus race_bars_plus1_minus_minus1 (this prompt's clarification of v1.0
# "race condition" inclusion).
# peak_to_final_r_ratio_fwd_h240 derived in load_signals (small-return
# approximation; documented in §schema_notes).
CLUSTER_FEATURES_NUMERIC = (
    "mfe_held_atr",  # = fwd_mfe_h1_atr
    "mae_held_atr",
    "fwd_mfe_h24_atr",
    "fwd_mae_h24_atr",
    "fwd_mfe_h120_atr",
    "fwd_mae_h120_atr",
    "race_bars_plus1_minus_minus1",
    "fwd_time_to_peak_mfe",
    "fwd_time_to_trough_mae",
    "peak_to_final_r_ratio_fwd_h240",  # derived in load
    "fwd_oscillation_count",
    "fwd_monotonicity_ratio",
    # v1.1 Amendment 4 additions:
    "fwd_realized_range_atr",
    "fwd_fraction_time_above_entry",
    "fwd_max_consecutive_directional_bars",
)
CLUSTER_FEATURES_CATEGORICAL = ("mfe_sequence_class_fwd_h24", "mfe_sequence_class_fwd_h120")

# v1.1 Amendment 6: K range extended to 2..8
K_VALUES = (2, 3, 4, 5, 6, 7, 8)
ALGOS = ("kmeans", "hierarchical")  # HDBSCAN handled as separate single column

# v1.1 Amendment 3: HDBSCAN added
HDBSCAN_MIN_CLUSTER_SIZE_FRAC = 0.05  # 5% of pool
HDBSCAN_MIN_SAMPLES = 50

# Predictor scan scope.
# Phase C (signal-time) runs all 4 models for every (algorithm, K, target_cluster)
# pair. Per Amendment 12 there is exactly ONE target cluster per (algorithm, K).
# Phase D (held-bar) runs all 4 models for K=2 + HDBSCAN target clusters only
# (3 targets × 5 t-slices × 4 models = 60 cells). Higher-K targets are
# computationally infeasible at full GB hyperparameters within session budget;
# K=2 is the canonical binary good/bad axis (op spec §6.1). Documented in
# PHASE_L_ARC_1_STEP3.md §schema_notes_step3.
K_PREDICTOR_SCAN = (2, 3, 4, 5, 6, 7, 8)
K_PREDICTOR_SCAN_3D = (2,)
INCLUDE_HDBSCAN_IN_PHASE_D = True

# v1.1 Amendment 7: differentiated cluster-size floors
SIZE_FLOOR_FILTER_TO = 0.15
SIZE_FLOOR_FILTER_OUT = 0.0
SIZE_FLOOR_EXIT_OR_DELAYED_ENTRY = 0.05

# v1.1 Amendment 5: PCA pre-check
PCA_REDUNDANCY_THRESHOLD = 0.85  # |Pearson| > 0.85 flagged candidate-redundant


# v1.1 Amendment 2: phase G/H eligibility — top-K-by-raw-AUC per target,
# K = max(20, 0.5 * n_features_scanned)
def phase_gh_top_k(n_features_scanned: int) -> int:
    return max(20, n_features_scanned // 2)


# v1.1 Amendment 1: family-level calibration check
# Family = §5.15 cross-pair / portfolio features (8 members)
CALIBRATION_FAMILY_V11 = (
    "concurrent_signals_same_bar",
    "concurrent_signals_within_3h",
    "currency_basket_3h_USD",
    "currency_basket_3h_EUR",
    "currency_basket_3h_JPY",
    "currency_basket_3h_GBP",
    "trade_overlap_at_execution_time",
    "sequential_same_pair_density_24h",
)
CALIBRATION_FAMILY_HISTORICAL_CARRIER = "concurrent_signals_within_3h"

# v1.1 Amendment 9: partial AUC at worst-decile cutoff
PARTIAL_AUC_DECILE_CUTOFF = 0.10

# v1.1 Amendment 11: permutation seeds use hashlib.sha256, not Python hash()
import hashlib as _hashlib


def perm_seed_for_feature(fname: str, base_offset: int = 0) -> int:
    h = _hashlib.sha256(fname.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) + base_offset) & 0xFFFFFFFF


# v1.1: permutation count floor 500, target 1000 per amendment doc
N_PERMUTATIONS_TARGET = 1000
N_PERMUTATIONS_FLOOR = 500

# Effect size thresholds (operational spec §8)
ES_THRESHOLDS = {
    "delta_median_fwd_mfe_h24": 0.10,  # ATR-norm, or >= 0.4 * pool_std
    "delta_median_fwd_mfe_h24_stdfrac": 0.40,
    "delta_median_fwd_mfe_to_mae_ratio_h24": 0.25,
    "delta_race_condition_median": 5.0,  # bars
    "delta_p_reach_plus1atr_240": 0.10,
}

# BH tier thresholds (op spec §6.7)
BH_TIER_1 = 0.05
BH_TIER_2 = 0.20

# Phase H filter dry-run: top-K predictors by AUC
FILTER_DRY_RUN_TOP_K = 10


# --- IO helpers ---------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path, float_format: str = "%.10g") -> str:
    """Deterministic CSV write. Returns sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort columns: do NOT reorder (preserve insertion). Just write.
    df.to_csv(path, index=False, float_format=float_format, lineterminator="\n")
    return sha256_file(path)


def write_text(text: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return sha256_file(path)


def make_rng(seed_offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + seed_offset)


def fmt_dist_stats(values: np.ndarray, label_prefix: str = "") -> dict[str, float]:
    """Per op spec §11.1: full distribution stats."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {
            f"{label_prefix}{k}": float("nan")
            for k in (
                "n",
                "mean",
                "std",
                "skew",
                "kurt",
                "min",
                "p1",
                "p5",
                "p10",
                "p20",
                "p30",
                "p40",
                "p50",
                "p60",
                "p70",
                "p80",
                "p90",
                "p95",
                "p99",
                "max",
            )
        }
    from scipy import stats as ss

    pcts = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)
    quants = np.percentile(v, pcts)
    out = {
        f"{label_prefix}n": int(v.size),
        f"{label_prefix}mean": float(np.mean(v)),
        f"{label_prefix}std": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        f"{label_prefix}skew": float(ss.skew(v)) if v.size > 2 else float("nan"),
        f"{label_prefix}kurt": float(ss.kurtosis(v)) if v.size > 3 else float("nan"),
        f"{label_prefix}min": float(np.min(v)),
    }
    for p, q in zip(pcts, quants):
        out[f"{label_prefix}p{p}"] = float(q)
    out[f"{label_prefix}max"] = float(np.max(v))
    return out


def append_manifest(entries: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cur = json.load(f)
    else:
        cur = {}
    cur.update(entries)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=2, sort_keys=True)
        f.write("\n")
