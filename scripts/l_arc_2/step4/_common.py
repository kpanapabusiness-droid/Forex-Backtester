"""Shared paths, seeds, constants for L Arc 2 Step 4.

Resolutions per `results/l_arc_2/step4/open_questions.md`:
- Cluster column actual name: K2_kmeans (renamed to kmeans_K2_cluster_id internally).
- Sentinel -2 trades excluded from cluster-conditional candidates, retained for filters.
- Spread cost: use signals_features.spread_cost_R as structural per-trade approximation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
STEP2_DIR = REPO / "results" / "l_arc_2" / "step2_descriptive"
STEP3_DIR = REPO / "results" / "l_arc_2" / "step3_extractability"
OUT_DIR = REPO / "results" / "l_arc_2" / "step4"

SIGNALS_CSV = STEP2_DIR / "signals_features.csv"
PATHS_CSV = STEP2_DIR / "trade_paths.csv"
HELD_CTX = STEP2_DIR / "held_bar_evolution"
CLUSTER_CSV = STEP3_DIR / "cluster_assignments.csv"
SPREAD_FLOOR_YAML = REPO / "configs" / "spread_floors_5ers.yaml"

BASE_SEED = 1234
HGB_RANDOM_STATE = 0

# Cluster column rename
CLUSTER_COL_FILE = "K2_kmeans"
CLUSTER_COL_INTERNAL = "kmeans_K2_cluster_id"
CLUSTER_SENTINEL = -2

# Fold split
FIT_FOLDS = (1, 2, 3, 4, 5)
VALIDATE_FOLDS = (6, 7)
ALL_FOLDS = (1, 2, 3, 4, 5, 6, 7)

# t-sweep for cluster-predictor candidates
T_SWEEP = (1, 3, 5, 10)

# Slugs (11 candidates)
SLUGS = (
    "exit_cluster_cond_gb",
    "exit_cluster_cond_gb_h240",
    "delayed_entry_t_gb",
    "filter_jpy_pairs",
    "filter_basket_usd_above_p50",
    "filter_basket_eur_above_p50",
    "filter_basket_jpy_above_p50",
    "filter_basket_gbp_above_p50",
    "filter_atr_at_signal_above_p50",
    "filter_concurrent_signals_above_p75",
    "exit_only_unfiltered_h240",
)

# Mechanism class per slug
MECHANISM = {
    "exit_cluster_cond_gb": "exit",
    "exit_cluster_cond_gb_h240": "exit",
    "delayed_entry_t_gb": "delayed_entry",
    "filter_jpy_pairs": "filter",
    "filter_basket_usd_above_p50": "filter",
    "filter_basket_eur_above_p50": "filter",
    "filter_basket_jpy_above_p50": "filter",
    "filter_basket_gbp_above_p50": "filter",
    "filter_atr_at_signal_above_p50": "filter",
    "filter_concurrent_signals_above_p75": "filter",
    "exit_only_unfiltered_h240": "exit_only",
}

ROUTING = {
    "exit_cluster_cond_gb": "S6",
    "exit_cluster_cond_gb_h240": "S6",
    "delayed_entry_t_gb": "S5",
    "filter_jpy_pairs": "S5",
    "filter_basket_usd_above_p50": "S5",
    "filter_basket_eur_above_p50": "S5",
    "filter_basket_jpy_above_p50": "S5",
    "filter_basket_gbp_above_p50": "S5",
    "filter_atr_at_signal_above_p50": "S5",
    "filter_concurrent_signals_above_p75": "S5",
    "exit_only_unfiltered_h240": "S6",
}

# Horizon used in capture ratio + simulation per candidate
HORIZON_BARS = {
    "exit_cluster_cond_gb": 120,
    "exit_cluster_cond_gb_h240": 240,
    "delayed_entry_t_gb": 120,
    "filter_jpy_pairs": 120,
    "filter_basket_usd_above_p50": 120,
    "filter_basket_eur_above_p50": 120,
    "filter_basket_jpy_above_p50": 120,
    "filter_basket_gbp_above_p50": 120,
    "filter_atr_at_signal_above_p50": 120,
    "filter_concurrent_signals_above_p75": 120,
    "exit_only_unfiltered_h240": 240,
}

# Component table thresholds (per dispatch §6 + spec)
MFE_GEOMETRY_BAND = (0.85, 1.15)
RETAINED_PER_FOLD_FLOOR = 100
RETAINED_PER_FOLD_AUTO_DISQ = 50
AUC_NEAR_CHANCE_BAND = (0.45, 0.55)
AUC_STABILITY_CV_FLOOR = 0.30

# Pre-eval gate (cand 2)
CLUSTER0_CURVE_TOLERANCE_R = 0.05


# ---- IO helpers ----

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


def write_json(obj: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return sha256_file(path)


def candidate_dir(slug: str) -> Path:
    return OUT_DIR / slug
