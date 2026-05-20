"""Shared infrastructure for Arc 10 experimental Step 5 diagnostics.

All experiments must respect:
- No lookahead.
- Determinism (seed every stochastic component; this module uses RANDOM_STATE
  for any np.random.default_rng calls; classifiers reuse the Step 4 seed=42).
- D1 lag correctness (re-uses Arc 10 Step 4 classifier wiring which is audited).
- Walk-forward (TimeSeriesSplit) for any AUC measurement.

Diagnostic only — no deployment, no commission proposals, no queue mutation.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-use Arc 10 Step 4 wiring (deterministic seeds, feature builders, D1 cache).
from scripts.l_arc_10.step4_extractability import (  # noqa: E402
    PerPairCache,
    _build_pair_cache,
    _build_paths_index,
    compute_pipeline_d1_features,
    compute_pipeline_e_features,
    compute_success_labels,
)

RANDOM_STATE = 42
DATA_DIR_4H = "C:/Users/panap/Documents/Forex-Backtester/data/4hr"
DATA_DIR_D1 = "C:/Users/panap/Documents/Forex-Backtester/data/daily"

# Arc 10 c1 SL frame from Step 3 (capturability composite selection).
C1_SL_ATR_MULT = 3.0
C1_CLUSTER_ID = 1
ORIGINAL_SL_ATR_MULT = 2.0

# Step 4 model config (mirrors configs/l_arc_10/step4.yaml).
RF_CONFIG = dict(
    n_estimators=200,
    max_depth=8,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
N_SPLITS = 5
CLASS_WEIGHT_BALANCED_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Arc 10 c1 loaders
# ---------------------------------------------------------------------------


@dataclass
class C1Bundle:
    """All artefacts needed for Arc 10 c1 experiments."""

    trades: pd.DataFrame            # 228 rows, full schema
    paths: pd.DataFrame             # is_held=1 path rows
    y: np.ndarray                   # success label (final_r >= 1R at SL=3.0xATR)
    e_features: pd.DataFrame        # Pipeline E feature frame (entry_time ordered)
    d1_features: pd.DataFrame       # Pipeline D1 feature frame (entry_time ordered)
    pair_caches: Dict[str, PerPairCache]
    base_success: float


def load_arc10_c1_bundle(verbose: bool = False) -> C1Bundle:
    """Reconstruct the Arc 10 c1 (n=228) bundle from on-disk Step 1/2/3
    artefacts plus a fresh recompute of E + D1 features.

    Reuses Step 4's compute functions byte-equivalently — no feature drift.
    """
    s1_dir = _REPO_ROOT / "results" / "l_arc_10" / "step1_verbatim"
    s2_dir = _REPO_ROOT / "results" / "l_arc_10" / "step2"

    trades = pd.read_csv(
        s1_dir / "trades_all.csv",
        parse_dates=["signal_bar_time", "entry_time", "exit_time"],
    )
    paths = pd.read_csv(s1_dir / "trades_paths.csv")
    clusters = pd.read_csv(s2_dir / "clusters_K3.csv")
    c1_tids = sorted(
        clusters[clusters["cluster_id"] == C1_CLUSTER_ID]["trade_id"].astype(int).tolist()
    )

    trades_c1 = trades[trades["trade_id"].isin(c1_tids)].reset_index(drop=True)
    paths_c1 = paths[paths["trade_id"].isin(c1_tids)].reset_index(drop=True)

    # CRITICAL: compute features over the FULL trade pool so pair_id_int
    # encoding matches Step 4 exactly (alphabetical over all 28 pairs that
    # actually fired, not just the subset present in c1). Then filter.
    pairs = sorted(trades["pair"].astype(str).unique())
    pair_caches: Dict[str, PerPairCache] = {}
    for p in pairs:
        if verbose:
            print(f"  caching {p}", file=sys.stderr)
        pair_caches[p] = _build_pair_cache(p, DATA_DIR_4H, DATA_DIR_D1)

    if verbose:
        print(f"  computing pipeline E features (full pool, n={len(trades)})", file=sys.stderr)
    e_features_full = compute_pipeline_e_features(trades, pair_caches)

    if verbose:
        print("  computing pipeline D1 features (full pool)", file=sys.stderr)
    d1_features_full, _ = compute_pipeline_d1_features(trades, pair_caches)

    paths_index = _build_paths_index(paths_c1)
    y_by_tid = compute_success_labels(
        c1_tids, paths_index, C1_SL_ATR_MULT, ORIGINAL_SL_ATR_MULT
    )

    # Filter to c1, then sort by entry_time (mirrors Step 4 evaluate_pipeline).
    e_sub = (
        e_features_full[e_features_full["trade_id"].isin(c1_tids)]
        .sort_values("entry_time", kind="mergesort")
        .reset_index(drop=True)
    )
    d1_sub = d1_features_full[d1_features_full["trade_id"].isin(c1_tids)].copy()
    d1_sub = d1_sub.merge(e_sub[["trade_id", "entry_time"]], on="trade_id", how="left")
    d1_sub = d1_sub.sort_values("entry_time", kind="mergesort").reset_index(drop=True)

    y = np.array([y_by_tid[int(tid)] for tid in e_sub["trade_id"]], dtype=int)
    base = float(y.mean())

    return C1Bundle(
        trades=trades_c1,
        paths=paths_c1,
        y=y,
        e_features=e_sub,
        d1_features=d1_sub,
        pair_caches=pair_caches,
        base_success=base,
    )


# ---------------------------------------------------------------------------
# Walk-forward AUC (TimeSeriesSplit, ordered by entry_time)
# ---------------------------------------------------------------------------


def class_weight_used(y: np.ndarray) -> str:
    base = float(y.mean())
    minority = min(base, 1.0 - base)
    return "balanced" if minority < CLASS_WEIGHT_BALANCED_THRESHOLD else "none"


def wf_oof_preds(
    X: pd.DataFrame, y: np.ndarray, feature_cols: List[str], n_splits: int = N_SPLITS
) -> Tuple[np.ndarray, np.ndarray, List[float], List[Tuple[int, int]], List[int]]:
    """Walk-forward out-of-fold predictions + per-fold AUC.

    Returns:
        oof_p:  predicted P(positive) for each test sample, in test-fold order.
        oof_y:  ground-truth for the same.
        per_fold_auc: AUC per fold.
        fold_indices: list of (n_train, n_test) per fold.
        fold_membership: array length n with fold idx for the row (-1 if untested).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    cw = class_weight_used(y)
    kw = dict(RF_CONFIG)
    if cw == "balanced":
        kw["class_weight"] = "balanced"

    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof_p_list: List[np.ndarray] = []
    oof_y_list: List[np.ndarray] = []
    per_fold_auc: List[float] = []
    fold_sizes: List[Tuple[int, int]] = []
    fold_membership = np.full(len(y), -1, dtype=int)

    for fold_idx, (tr, te) in enumerate(splitter.split(X)):
        Xtr = X[feature_cols].iloc[tr]
        ytr = y[tr]
        Xte = X[feature_cols].iloc[te]
        yte = y[te]
        med = Xtr.median(numeric_only=True)
        Xtr_f = Xtr.fillna(med)
        Xte_f = Xte.fillna(med)
        clf = RandomForestClassifier(**kw)
        clf.fit(Xtr_f, ytr)
        p = clf.predict_proba(Xte_f)[:, 1]
        if len(set(yte.tolist())) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(yte, p))
        per_fold_auc.append(auc)
        oof_p_list.append(p)
        oof_y_list.append(yte)
        fold_sizes.append((int(len(tr)), int(len(te))))
        for j, idx in enumerate(te):
            fold_membership[int(idx)] = fold_idx

    oof_p = np.concatenate(oof_p_list) if oof_p_list else np.array([])
    oof_y = np.concatenate(oof_y_list) if oof_y_list else np.array([])
    return oof_p, oof_y, per_fold_auc, fold_sizes, fold_membership


def mean_auc_safe(per_fold_auc: List[float]) -> Tuple[float, float]:
    valid = [a for a in per_fold_auc if not math.isnan(a)]
    if not valid:
        return float("nan"), 0.0
    return float(np.mean(valid)), float(np.std(valid, ddof=1)) if len(valid) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Deterministic IO
# ---------------------------------------------------------------------------


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_g(x) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return ""
    except Exception:
        return str(x)
    return f"{xf:.10g}"
