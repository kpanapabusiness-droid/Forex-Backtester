"""Shared utilities for L Arc 2 Step 3 extractability scripts.

Arc 2 differs from arc 1:
- h=120 horizon (vs h=1) — held-window features REAL, not forward proxies.
- Clustering uses held-window 9 + Amendment 4 three = 12 features (per task spec).
- `mfe_sequence_class_held` ordinal-encoded (MAE_first=0, simultaneous_bar=1, MFE_first=2)
  per task spec, NOT one-hot.
- Amendment 4 three features ALREADY in signals_features.csv (computed in step 2).
- No calibration check (v1.1 Amendment 1 second clause — no known-prior carrier feature).
- Phase D held-bar features include trade_paths.csv bar-by-bar observations
  (open/high/low/close/cum_logret_from_entry/mfe_to_date_atr/mae_to_date_atr)
  PLUS held_bar_evolution/t{t}.csv context.
- Signal-time predictor set includes arc-2 NEW pre-signal context (cum_logret_1h_24/72/168,
  vol_realized_1h_24h) and bin categoricals (cum_logret_1h_24_bin, cum_logret_1h_168_bin).

Determinism: all random seeds derive from BASE_SEED. All sorts explicit.
Hash-based feature seeds per Amendment 11.
"""

from __future__ import annotations

import hashlib
import hashlib as _hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths --------------------------------------------------------------

REPO = Path(__file__).resolve().parents[3]
STEP2_DIR = REPO / "results" / "l_arc_2" / "step2_descriptive"
SIGNALS_CSV = STEP2_DIR / "signals_features.csv"
PATHS_CSV = STEP2_DIR / "trade_paths.csv"
HELD_CTX = STEP2_DIR / "held_bar_evolution"
FWD_CTX = STEP2_DIR / "forward_context_evolution"
OUT_DIR = REPO / "results" / "l_arc_2" / "step3_extractability"
STRAT_DIR = OUT_DIR / "stratifications"

# --- seeds --------------------------------------------------------------

BASE_SEED = 1234
N_PERMUTATIONS = 1000
WARD_SUBSAMPLE = 10_000
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

# Trade outcome / held / forward / amendment-4 columns — excluded from predictor sets
OUTCOME_COLS = (
    "net_r",
    "gross_r",
    "spread_cost_R",
    "mfe_R",
    "mae_R",
    "bars_held",
    "exit_reason",
    "exit_reason_engine",
    "spread_pips_entry",
    "spread_pips_exit",
    "spread_floored",
    "sl_distance_atr",
    "sl_distance_price",
    "mfe_held_atr",
    "mae_held_atr",
    "peak_to_final_r_ratio",
    "mfe_to_mae_ratio_held",
    "r_given_back_from_peak",
    "mfe_sequence_class_held",
    "time_to_peak_mfe",
    "time_to_trough_mae",
    "time_from_peak_to_exit",
    "oscillation_count",
    "monotonicity_ratio",
    "max_consecutive_with",
    "max_consecutive_against",
    "acf1_returns_during_hold",
    "data_end_flag",
    "forward_window_bars_available",
    # Amendment 4 (forward-derived, NOT signal-time)
    "fwd_realized_range_atr",
    "fwd_fraction_time_above_entry",
    "fwd_max_consecutive_directional_bars",
)

# Numeric signal-time predictors (computable at bar N close).
# Arc 2 has NO signal_threshold_q90 / trigger_excess / trigger_ratio columns;
# adds the 4 NEW pre-signal context cols (cum_logret_1h_24/72/168, vol_realized_1h_24h)
# and vol_realized_1h_24h_decile.
SIGNAL_TIME_NUMERIC = (
    "signal_bar_open",
    "signal_bar_close",
    "signal_bar_high",
    "signal_bar_low",
    "signal_bar_log_return",
    "signal_bar_abs_log_return",
    "signal_bar_volume",
    "signal_bar_volume_nan",
    "atr_at_signal_1h",
    "atr_baseline_1h_200",
    "atr_ratio_to_baseline",
    "cum_logret_1h_3",
    "cum_logret_1h_6",
    "cum_logret_1h_24",
    "cum_logret_1h_72",
    "cum_logret_1h_168",
    "vol_realized_1h_24h",
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
    "vol_realized_1h_24h_decile",
)

# Categorical signal-time predictors (one-hot encoded). Arc 2 adds the 3 new bin
# categoricals (cum_logret_1h_6_bin already existed for arc 1; arc 2 also has 24 + 168).
SIGNAL_TIME_CATEGORICAL = (
    "pair",
    "session",
    "vol_regime",
    "pre_momentum_bin",
    "cum_logret_1h_6_bin",
    "cum_logret_1h_24_bin",
    "cum_logret_1h_168_bin",
)

# Forward / held-bar context numeric features (identical schema in both
# held_bar_evolution/ and forward_context_evolution/ per step 2 §6.14).
# Phase D uses held_bar_evolution per task spec.
HELD_CTX_NUMERIC = (
    "atr_regime_ratio",
    "broker_spread_pips_raw",
    "broker_spread_pips_floored",
    "cross_pair_dispersion_proxy",
    "basket_cum_logret_USD",
    "basket_cum_logret_EUR",
    "basket_cum_logret_JPY",
    "basket_cum_logret_GBP",
)

# Per-bar path columns from trade_paths.csv (available from bar_offset=0 onward).
PATH_NUMERIC_PER_BAR = (
    "open",
    "high",
    "low",
    "close",
    "cum_logret_from_entry",
    "mfe_to_date_atr",
    "mae_to_date_atr",
)

HELD_BAR_TS = (1, 3, 5, 10, 20)

# --- clustering feature subset (per task spec) ---------------------------
#
# Arc 2 has REAL held-window features (mean bars_held ~47, not degenerate).
# 12 features = legacy 9 (op spec §6.1) + Amendment 4 three.
# `mfe_sequence_class_held` is ORDINAL encoded per task spec:
#   MAE_first = 0, simultaneous_bar = 1, MFE_first = 2.
CLUSTER_FEATURES_NUMERIC = (
    "mfe_held_atr",
    "mae_held_atr",
    "bars_held",
    "time_to_peak_mfe",
    "time_to_trough_mae",
    "peak_to_final_r_ratio",
    "oscillation_count",
    "monotonicity_ratio",
    # Amendment 4 three (already in signals_features.csv on arc 2)
    "fwd_realized_range_atr",
    "fwd_fraction_time_above_entry",
    "fwd_max_consecutive_directional_bars",
)
# Ordinal encoded (per task spec)
CLUSTER_FEATURES_ORDINAL = {
    "mfe_sequence_class_held": {"MAE_first": 0, "simultaneous_bar": 1, "MFE_first": 2},
}

# v1.1 Amendment 6: K range extended to 2..8
K_VALUES = (2, 3, 4, 5, 6, 7, 8)
ALGOS = ("kmeans", "hierarchical")

# v1.1 Amendment 3: HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE_FRAC = 0.05
HDBSCAN_MIN_SAMPLES = 50

# Predictor scan scope.
# Phase C runs all 4 models per (algo, K, target) — Amendment 12 yields 1 target each.
# Phase D restricts to K=2 kmeans + K=2 hierarchical + HDBSCAN (if non-empty) for
# computational feasibility within session budget; documented in §schema_notes.
K_PREDICTOR_SCAN = (2, 3, 4, 5, 6, 7, 8)
K_PREDICTOR_SCAN_3D = (2,)
INCLUDE_HDBSCAN_IN_PHASE_D = True

# v1.1 Amendment 7: differentiated cluster-size floors
SIZE_FLOOR_FILTER_TO = 0.15
SIZE_FLOOR_FILTER_OUT = 0.0
SIZE_FLOOR_EXIT_OR_DELAYED_ENTRY = 0.05

# v1.1 Amendment 5: PCA pre-check
PCA_REDUNDANCY_THRESHOLD = 0.85


# v1.1 Amendment 2: Phase G/H eligibility — top-K by raw AUC per target.
def phase_gh_top_k(n_features_scanned: int) -> int:
    return max(20, n_features_scanned // 2)


# v1.1 Amendment 9: partial AUC at worst-decile cutoff
PARTIAL_AUC_DECILE_CUTOFF = 0.10

# Cross-pair / portfolio family (arc-open §4 cross-arc carry from arc 1).
# Reported regardless of tier outcome per arc-2 open doc.
CROSS_ARC_PORTFOLIO_FAMILY = (
    "concurrent_signals_same_bar",
    "concurrent_signals_within_3h",
    "currency_basket_3h_USD",
    "currency_basket_3h_EUR",
    "currency_basket_3h_JPY",
    "currency_basket_3h_GBP",
    "trade_overlap_at_execution_time",
    "sequential_same_pair_density_24h",
)

# Effect-size thresholds (op spec §8) — unchanged from arc 1
ES_THRESHOLDS = {
    "delta_median_fwd_mfe_h24": 0.10,
    "delta_median_fwd_mfe_h24_stdfrac": 0.40,
    "delta_median_fwd_mfe_to_mae_ratio_h24": 0.25,
    "delta_race_condition_median": 5.0,
    "delta_p_reach_plus1atr_240": 0.10,
}

BH_TIER_1 = 0.05
BH_TIER_2 = 0.20


# v1.1 Amendment 11: deterministic permutation seeds
def perm_seed_for_feature(fname: str, base_offset: int = 0) -> int:
    h = _hashlib.sha256(fname.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) + base_offset) & 0xFFFFFFFF


# --- IO helpers ---------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path, float_format: str = "%.10g") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format, lineterminator="\n")
    return sha256_file(path)


def write_text(text: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return sha256_file(path)


def make_rng(seed_offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + seed_offset)


def fmt_dist_stats(values: np.ndarray, label_prefix: str = "") -> dict:
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


def append_manifest(entries: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cur = json.load(f)
    else:
        cur = {}
    cur.update(entries)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
