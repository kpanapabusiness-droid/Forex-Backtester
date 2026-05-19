"""Arc 9 Candidate A — deterministic in-memory rebuild of the LGBM Pipeline E classifier.

Phase 1.1 of dispatch v3. Audit-8 pattern: reconstructs the classifier from
the feature matrix deterministically (no joblib reload). Verifies byte-identical
per-fold TimeSeriesSplit(5) AUCs against the locked reference values from
results/l_arc_9/experiments/pipeline_e_retry/per_fold_aucs.csv (cell
'lgbm_expanded_28').

Locked reference per-fold AUCs (lgbm_expanded_28):
  F1 0.8483796296  F2 0.7357456140  F3 0.7172630377  F4 0.7144016227  F5 0.7380414807
  mean 0.750766277  std 0.05558948303

Phase 1.1 gate: ALL five per-fold AUCs must match byte-identically (string equality
at the 10-digit precision written by pipeline_e_retry.py's float_format='%.10g').

This script also produces the FULL-DATA classifier (fit on all 2153-row clean
matrix) which Phase 1.2 exports to ONNX, and persists supporting artefacts:
  - rebuilt feature matrix sha256 (verify against pipeline_e_retry's matrix)
  - per-fold AUC reproduction table
  - full-data classifier feature importances
  - the in-memory classifier object (returned by rebuild_full_data_classifier())

Usage as a script:
    python scripts/deployment/arc9_canda_rebuild_classifier.py

Usage as a module (Phase 1.2 ONNX export):
    from scripts.deployment.arc9_canda_rebuild_classifier import (
        rebuild_full_data_classifier,
    )
    classifier, X_expanded, y, feature_names = rebuild_full_data_classifier()

Determinism:
  - All RNG via random_state=42 (per pipeline_e_retry LGBM_KW + SEED)
  - LGBM deterministic=True, force_row_wise=True
  - pandas sort_values(kind='mergesort') (stable sort)
  - to_csv(lineterminator='\n', float_format='%.10g')
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-use training-time feature pipeline + hyperparameters verbatim.
from scripts.l_arc_9.experiments.pipeline_e_retry import (  # noqa: E402
    BASELINE_16, D1_8, SESSION_4, EXPANDED_28, FORBIDDEN_LEAK_FEATURES,
    LGBM_KW, N_SPLITS, SEED,
    _attach_d1_features, _attach_session_features,
)

# Locked reference AUCs (lgbm_expanded_28 cell) from
# results/l_arc_9/experiments/pipeline_e_retry/per_fold_aucs.csv.
# These string values match what pipeline_e_retry.py writes via float_format='%.10g'.
REFERENCE_AUCS_LGBM_EXPANDED_28: Dict[int, str] = {
    1: "0.8483796296",
    2: "0.735745614",
    3: "0.7172630377",
    4: "0.7144016227",
    5: "0.7380414807",
}
REFERENCE_AUC_MEAN: str = "0.750766277"

# Reference feature_matrix.csv sha256 (computed once below if absent; written to
# rebuild_check.json for future audits).
PE_RETRY_DIR = _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "pipeline_e_retry"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_dir(kind: str) -> Path:
    """Resolve absolute data dir; mirror pipeline_e_retry's fallback."""
    main_repo = _REPO_ROOT.parent.parent.parent  # worktrees/<wt>/../../.. = repo root
    candidate = main_repo / "data" / kind
    if candidate.exists():
        return candidate
    fallback = Path("C:/Users/panap/Documents/Forex-Backtester/data") / kind
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"data dir for {kind} not found at {candidate} or {fallback}")


def _build_feature_matrix() -> pd.DataFrame:
    """Reconstruct the EXPANDED_28 feature matrix.

    Mirrors pipeline_e_retry.run() up to the post-clean, post-sort dataframe.
    """
    entry = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step4_extractability" / "entry_features.csv")
    forbidden = set(entry.columns) & FORBIDDEN_LEAK_FEATURES
    if forbidden:
        raise RuntimeError(f"path-shape features leaked into entry features: {forbidden}")
    for c in BASELINE_16:
        if c not in entry.columns:
            raise RuntimeError(f"missing baseline feature column: {c}")

    # Cluster 0 binary label.
    clusters = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step2_clustering" / "clusters_K3.csv")
    cid0 = set(clusters[clusters["cluster_id"] == 0]["trade_id"].astype(int))

    # ATR at 4H signal bar (for d1_atr_ratio_to_4h denominator).
    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    atr_4h_by_tid: Dict[int, float] = dict(zip(
        trades_all["trade_id"].astype(int),
        trades_all["atr14_at_signal"].astype(float),
    ))
    entry_time_by_tid: Dict[int, str] = dict(zip(
        trades_all["trade_id"].astype(int),
        trades_all["entry_time"].astype(str),
    ))

    data_d1_dir = _data_dir("daily")
    df = _attach_d1_features(entry, data_d1_dir, atr_4h_by_tid)
    df = _attach_session_features(df)
    df["entry_time"] = df["trade_id"].astype(int).map(entry_time_by_tid)
    df["y"] = df["trade_id"].astype(int).apply(lambda x: 1 if int(x) in cid0 else 0)

    feat_cols = EXPANDED_28
    if len(feat_cols) > 50:
        raise RuntimeError(f"feature count {len(feat_cols)} exceeds hard cap 50")
    df_clean = df.dropna(subset=feat_cols).reset_index(drop=True)
    df_clean["entry_time"] = pd.to_datetime(df_clean["entry_time"])
    df_clean = df_clean.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
    return df_clean


def _tss_cv_aucs(X: np.ndarray, y: np.ndarray) -> List[float]:
    """Compute TSS(5) per-fold AUCs with the locked LGBM_KW hyperparameters."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_aucs: List[float] = []
    for tr_idx, te_idx in tscv.split(X):
        mdl = lgb.LGBMClassifier(**LGBM_KW)
        mdl.fit(X[tr_idx], y[tr_idx])
        p = mdl.predict_proba(X[te_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[te_idx], p)))
    return fold_aucs


def rebuild_full_data_classifier() -> Tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Rebuild the full-data LGBM classifier deterministically.

    Returns:
      (classifier, X_expanded, y, feature_names, df_clean)
    """
    df_clean = _build_feature_matrix()
    X_expanded = df_clean[EXPANDED_28].to_numpy(dtype=float)
    y = df_clean["y"].to_numpy(dtype=int)
    mdl_full = lgb.LGBMClassifier(**LGBM_KW)
    mdl_full.fit(X_expanded, y)
    return mdl_full, X_expanded, y, list(EXPANDED_28), df_clean


def run_phase_1_1(out_dir: Path) -> Dict[str, object]:
    """Execute Phase 1.1: AUC parity gate + persist supporting artefacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[phase 1.1] building feature matrix (D1 lag merge_asof + session)...")
    df_clean = _build_feature_matrix()
    n_total = len(df_clean)
    n_pos = int(df_clean["y"].sum())
    print(f"[phase 1.1] matrix: n_total={n_total}, n_pos={n_pos}, n_features={len(EXPANDED_28)}")

    # Cross-check vs pipeline_e_retry's persisted feature_matrix.csv (informational).
    pe_fm_path = PE_RETRY_DIR / "feature_matrix.csv"
    if pe_fm_path.exists():
        pe_fm_sha = _sha256_file(pe_fm_path)
    else:
        pe_fm_sha = None

    # Persist rebuilt feature_matrix.csv with same float_format/lineterminator
    # as pipeline_e_retry for byte-identical check (extra determinism marker).
    keep_cols = ["trade_id", "pair", "signal_bar_time", "entry_time", "y"] + list(EXPANDED_28)
    rebuilt_fm_path = out_dir / "rebuilt_feature_matrix.csv"
    df_clean[keep_cols].to_csv(
        rebuilt_fm_path, index=False, float_format="%.10g", lineterminator="\n",
    )
    rebuilt_fm_sha = _sha256_file(rebuilt_fm_path)
    feature_matrix_match = (pe_fm_sha is not None and pe_fm_sha == rebuilt_fm_sha)
    print(f"[phase 1.1] feature_matrix sha256 match vs pipeline_e_retry: {feature_matrix_match}")
    if pe_fm_sha is not None:
        print(f"    pipeline_e_retry: {pe_fm_sha}")
        print(f"    rebuilt:          {rebuilt_fm_sha}")

    # Phase 1.1 GATE: per-fold AUC parity against locked reference values.
    print("[phase 1.1] running TSS(5) CV with locked LGBM_KW hyperparameters...")
    X_expanded = df_clean[EXPANDED_28].to_numpy(dtype=float)
    y = df_clean["y"].to_numpy(dtype=int)
    fold_aucs = _tss_cv_aucs(X_expanded, y)
    mean_auc = float(np.nanmean(fold_aucs))

    # Format each fold AUC with the same %.10g format pipeline_e_retry writes,
    # so string equality matches the values in per_fold_aucs.csv byte-for-byte.
    rebuilt_str = {i + 1: ("%.10g" % v) for i, v in enumerate(fold_aucs)}
    mean_str = "%.10g" % mean_auc

    auc_parity_rows = []
    all_match = True
    for fold in (1, 2, 3, 4, 5):
        ref = REFERENCE_AUCS_LGBM_EXPANDED_28[fold]
        got = rebuilt_str[fold]
        match = (ref == got)
        if not match:
            all_match = False
        auc_parity_rows.append({
            "fold": fold,
            "reference_auc": ref,
            "rebuilt_auc": got,
            "byte_identical": int(match),
        })
        print(f"    F{fold}: ref={ref:>14s}  rebuilt={got:>14s}  {'PASS' if match else 'FAIL'}")
    mean_match = (mean_str == REFERENCE_AUC_MEAN)
    auc_parity_rows.append({
        "fold": "mean",
        "reference_auc": REFERENCE_AUC_MEAN,
        "rebuilt_auc": mean_str,
        "byte_identical": int(mean_match),
    })
    if not mean_match:
        all_match = False
    print(f"    mean: ref={REFERENCE_AUC_MEAN:>14s}  rebuilt={mean_str:>14s}  {'PASS' if mean_match else 'FAIL'}")
    pd.DataFrame(auc_parity_rows).to_csv(
        out_dir / "auc_parity.csv", index=False, lineterminator="\n",
    )

    if not all_match:
        raise RuntimeError(
            "Phase 1.1 GATE FAILED: per-fold AUCs do not match locked reference values byte-identically. "
            "Audit-8 reproduction failed. See auc_parity.csv. Halt — do not proceed to ONNX export."
        )

    # Train full-data classifier (the one Phase 1.2 exports to ONNX).
    print("[phase 1.1] fitting full-data classifier...")
    mdl_full = lgb.LGBMClassifier(**LGBM_KW)
    mdl_full.fit(X_expanded, y)

    # Sanity: full-data predict_proba on all rows (for use by Phase 1.3 parity check).
    full_data_proba = mdl_full.predict_proba(X_expanded)
    print(f"[phase 1.1] full-data predict_proba shape: {full_data_proba.shape}")

    # Persist feature importances (informational).
    importances = pd.DataFrame({
        "feature": EXPANDED_28,
        "index": list(range(len(EXPANDED_28))),
        "importance_gain": mdl_full.booster_.feature_importance(importance_type="gain"),
        "importance_split": mdl_full.booster_.feature_importance(importance_type="split"),
    })
    importances.to_csv(
        out_dir / "rebuilt_feature_importances.csv",
        index=False, float_format="%.10g", lineterminator="\n",
    )

    # Persist full-data probabilities (used by Phase 1.3 parity-verify script).
    proba_df = df_clean[["trade_id", "pair", "signal_bar_time", "entry_time", "y"]].copy()
    proba_df["native_prob_negative"] = full_data_proba[:, 0]
    proba_df["native_prob_positive"] = full_data_proba[:, 1]
    proba_df.to_csv(
        out_dir / "native_full_data_probabilities.csv",
        index=False, float_format="%.10g", lineterminator="\n",
    )

    summary = {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_features": len(EXPANDED_28),
        "feature_order": list(EXPANDED_28),
        "rebuilt_feature_matrix_sha256": rebuilt_fm_sha,
        "pipeline_e_retry_feature_matrix_sha256": pe_fm_sha,
        "feature_matrix_sha256_match": bool(feature_matrix_match),
        "auc_parity_pass": bool(all_match),
        "rebuilt_fold_aucs_str": rebuilt_str,
        "rebuilt_mean_auc_str": mean_str,
        "reference_fold_aucs_str": REFERENCE_AUCS_LGBM_EXPANDED_28,
        "reference_mean_auc_str": REFERENCE_AUC_MEAN,
        "rng_seed": SEED,
        "lgbm_kw": {k: (v if not callable(v) else str(v)) for k, v in LGBM_KW.items()},
    }
    (out_dir / "rebuild_check.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[phase 1.1] PASS — gate satisfied; summary written to {out_dir / 'rebuild_check.json'}")
    return summary


def main() -> int:
    out_dir = _REPO_ROOT / "results" / "l_arc_9" / "deployment" / "rebuild"
    run_phase_1_1(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
