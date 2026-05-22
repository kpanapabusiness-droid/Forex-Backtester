"""CC-0 materialisation script for L Arc 2 step 5 delayed_entry_t_gb.

Lives under scripts/l_arc_2/step4/ because it re-uses the step 4 predictor
and orchestrator (run_delayed_entry from _actions). The CC-0 dispatch is
step-5 work but the compute surface is step-4 code.

Produces under results/l_arc_2/step5_recharacterisation/delayed_entry_t_gb/:
  - trades_post_mechanism_f2_f5.csv  (schema byte-identical to step 4's
    F6+F7 CSV; columns: trade_id, fold, pair, fire_bar, action_bar,
    exit_bar, exit_reason, net_r, gross_r, spread_cost_r, mfe_at_exit,
    mae_at_exit; sorted by trade_id)
  - predicted_mirror_f{2..5}.txt  (one trade_id per line per fold)
  - run_manifest.json  (mirrors step 4's manifest schema, with per-fold
    training/prediction sha256 + classifier hyperparameter hash)

Two top-level entries:
  - materialise() → produces all outputs (Task B/C/D)
  - lookahead_invariant_test() → perturbs trade_paths at bar_offset >= 2
    for sampled trades and confirms predictions byte-identical (Task F)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import _actions as A
from . import _common as C
from . import _data as D
from . import _predictor as P

REPO = C.REPO
OUT_DIR = REPO / "results" / "l_arc_2" / "step5_recharacterisation" / "delayed_entry_t_gb"
T_STAR = 1
SLUG = "delayed_entry_t_gb"
PREDICTOR_NAME = "fit_predict_cluster_anchored_expanding"

# Pinned commit per CC-0 dispatch / OPEN doc §1 reproducibility lock.
# HEAD may differ if unrelated commits land — the script-source pin is
# verified at blob level (see _verify_predictor_source_pin).
PREDICTOR_PIN_COMMIT = "fe345be58a8517189e898d0618266f083bda900c"
PINNED_SCRIPT_FILES = (
    "scripts/l_arc_2/step4/_predictor.py",
    "scripts/l_arc_2/step4/_common.py",
    "scripts/l_arc_2/step4/_data.py",
    "scripts/l_arc_2/step4/_actions.py",
    "scripts/l_arc_2/step4/_simulator.py",
    "scripts/l_arc_2/step4/run_step4.py",
)

CLASSIFIER_HYPERPARAMS = {
    "class": "sklearn.ensemble.HistGradientBoostingClassifier",
    "max_iter": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "random_state": C.HGB_RANDOM_STATE,
    "decision_threshold": 0.5,
}


# ============================================================
# Hashing helpers
# ============================================================


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    """Stable sha256 of a numpy array: shape + dtype + bytes."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _classifier_hp_hash() -> str:
    canonical = json.dumps(CLASSIFIER_HYPERPARAMS, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(canonical.encode())


def _git_commit() -> str:
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO))
    return out.decode().strip()


def _git_blob_at_commit(commit: str, path: str) -> str:
    """Return the git blob sha for `path` at `commit`. Raises if not present."""
    out = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=str(REPO),
    )
    return out.decode().strip()


def _git_blob_at_head(path: str) -> str:
    return _git_blob_at_commit("HEAD", path)


def _verify_predictor_source_pin() -> dict:
    """Verify the script-source pin: every pinned file at HEAD must have a
    blob hash byte-identical to the same file at the pinned commit. Returns
    a per-file verification block suitable for the manifest. If HEAD == pin
    commit, the check is trivially true.
    """
    head = _git_commit()
    per_file = {}
    all_match = True
    for path in PINNED_SCRIPT_FILES:
        head_blob = _git_blob_at_commit(head, path)
        pin_blob = _git_blob_at_commit(PREDICTOR_PIN_COMMIT, path)
        match = head_blob == pin_blob
        per_file[path] = {
            "blob_at_pin_commit": pin_blob,
            "blob_at_head": head_blob,
            "match": match,
        }
        if not match:
            all_match = False
    return {
        "pin_commit": PREDICTOR_PIN_COMMIT,
        "head_commit": head,
        "head_matches_pin_commit": head == PREDICTOR_PIN_COMMIT,
        "all_pinned_files_blob_match": all_match,
        "per_file": per_file,
    }


def _versions() -> dict:
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


# ============================================================
# Per-fold sha256 capture (mirrors _predictor.py training logic)
# ============================================================


def _capture_per_fold_shas(signals_df: pd.DataFrame, t: int) -> dict:
    """Re-execute the per-fold training/prediction set construction so we
    can hash the training-set and prediction-set arrays. Mirrors the logic
    in fit_predict_cluster_anchored_expanding's F2..F5 branch exactly.
    """
    df = signals_df.sort_values("trade_id").reset_index(drop=True)
    X_full, _ = P.build_t_matrix(df, t)

    bars_held = df["bars_held"].values
    cluster = df[C.CLUSTER_COL_INTERNAL].values
    fold = df["fold_id"].values

    active_mask = bars_held >= t
    valid_cluster_mask = cluster != C.CLUSTER_SENTINEL
    use_mask = active_mask & valid_cluster_mask

    per_fold = {}
    for f_target in (2, 3, 4, 5):
        train_folds = list(range(1, f_target))
        train_mask = use_mask & np.isin(fold, train_folds)
        test_mask = use_mask & (fold == f_target)
        X_train = X_full[train_mask]
        y_train = cluster[train_mask].astype(int)
        X_test = X_full[test_mask]
        per_fold[f"f{f_target}"] = {
            "train_X_sha256": _sha256_array(X_train),
            "train_y_sha256": _sha256_array(y_train),
            "train_n_rows": int(X_train.shape[0]),
            "train_n_features": int(X_train.shape[1]) if X_train.size else 0,
            "predict_X_sha256": _sha256_array(X_test),
            "predict_n_rows": int(X_test.shape[0]),
        }
    return per_fold


# ============================================================
# Materialisation (Task B/C/D)
# ============================================================


def _load_inputs():
    signals = D.load_signals().sort_values("trade_id").reset_index(drop=True)
    clusters = D.load_clusters().sort_values("trade_id").reset_index(drop=True)
    signals_clu = (
        signals.merge(clusters, on="trade_id", how="left")
        .sort_values("trade_id")
        .reset_index(drop=True)
    )
    paths_120 = D.load_paths_long(max_offset=120)
    return signals_clu, paths_120


def materialise() -> dict:
    """Produces all CC-0 outputs and returns a manifest dict."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify the predictor source pin before doing any compute.
    source_pin = _verify_predictor_source_pin()
    if not source_pin["all_pinned_files_blob_match"]:
        raise RuntimeError(
            f"Predictor source pin drift: not all pinned files match "
            f"commit {PREDICTOR_PIN_COMMIT}. See per_file detail: "
            f"{json.dumps(source_pin['per_file'], indent=2)}"
        )

    # Clear path cache to ensure deterministic re-execution
    D._load_paths_full.cache_clear()

    signals_clu, paths_120 = _load_inputs()

    # Capture per-fold training/prediction sha256s
    per_fold_shas = _capture_per_fold_shas(signals_clu, T_STAR)

    # Run the anchored expanding predictor (produces F2..F7 predictions)
    preds_all = P.fit_predict_cluster_anchored_expanding(signals_clu, T_STAR)

    # Filter to F2..F5
    preds_f2_f5 = (
        preds_all[preds_all["fold"].isin([2, 3, 4, 5])]
        .sort_values("trade_id")
        .reset_index(drop=True)
    )

    # Identify predicted-mirror trades per fold (predicted_cluster == 0)
    preds_mirror = preds_f2_f5[preds_f2_f5["predicted_cluster"] == 0]
    per_fold_counts = {}
    per_fold_mirror_files = {}
    for f in (2, 3, 4, 5):
        ids = preds_mirror.loc[preds_mirror["fold"] == f, "trade_id"].astype(int).tolist()
        per_fold_counts[f"f{f}"] = len(ids)
        # Write per-fold trade-id list
        path = OUT_DIR / f"predicted_mirror_f{f}.txt"
        # Sort ascending for determinism
        ids_sorted = sorted(ids)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for tid in ids_sorted:
                fh.write(f"{tid}\n")
        per_fold_mirror_files[f"f{f}"] = path

    # Run the action (delayed-entry simulator) on the F2..F5 predictions
    held_ctx = D.load_held_ctx(T_STAR)
    post = A.run_delayed_entry(signals_clu, paths_120, held_ctx, preds_f2_f5, T_STAR)

    # Write trades_post_mechanism_f2_f5.csv with the same conventions as step 4
    trades_path = OUT_DIR / "trades_post_mechanism_f2_f5.csv"
    post.to_csv(trades_path, index=False, float_format="%.10g", lineterminator="\n")

    # Compute all output sha256s
    output_shas = {
        "trades_post_mechanism_f2_f5.csv": _sha256_file(trades_path),
    }
    for f, path in per_fold_mirror_files.items():
        output_shas[f"predicted_mirror_{f}.txt"] = _sha256_file(path)

    manifest = {
        "step": "l_arc_2/step5_recharacterisation/delayed_entry_t_gb/CC-0",
        "predictor_function": PREDICTOR_NAME,
        "git_commit_head_at_run": _git_commit(),
        "predictor_source_pin_verification": source_pin,
        "versions": _versions(),
        "t_star": T_STAR,
        "folds_materialised": [2, 3, 4, 5],
        "f1_handling": "skipped (pre-F1 history empty for L Arc 2)",
        "f6_f7_handling": "not materialised here; already on disk at "
        "results/l_arc_2/step4/delayed_entry_t_gb/trades_post_mechanism.csv",
        "classifier_hyperparameters": CLASSIFIER_HYPERPARAMS,
        "classifier_hyperparameter_hash": _classifier_hp_hash(),
        "per_fold": per_fold_shas,
        "per_fold_predicted_mirror_counts": per_fold_counts,
        "post_mechanism_n_rows": int(len(post)),
        "outputs_sha256": output_shas,
        "inputs": {
            "signals_features": {
                "path": "results/l_arc_2/step2_descriptive/signals_features.csv",
                "sha256": _sha256_file(C.SIGNALS_CSV),
            },
            "cluster_assignments": {
                "path": "results/l_arc_2/step3_extractability/cluster_assignments.csv",
                "sha256": _sha256_file(C.CLUSTER_CSV),
            },
            "trade_paths": {
                "path": "results/l_arc_2/step2_descriptive/trade_paths.csv",
                "sha256": _sha256_file(C.PATHS_CSV),
            },
            "held_bar_evolution_t1": {
                "path": "results/l_arc_2/step2_descriptive/held_bar_evolution/t1.csv",
                "sha256": _sha256_file(C.HELD_CTX / "t1.csv"),
            },
            "spread_floors_5ers": {
                "path": "configs/spread_floors_5ers.yaml",
                "sha256": _sha256_file(C.SPREAD_FLOOR_YAML),
            },
        },
        # No timestamp_utc — would break two-run byte-identity. Temporal
        # anchoring is via git_commit_head_at_run + predictor_source_pin.
    }

    manifest_path = OUT_DIR / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    manifest["outputs_sha256"]["run_manifest.json"] = _sha256_file(manifest_path)

    # Re-write manifest including its own sha (so it's stable on second run
    # — actually no: including its own sha is self-referential. Skip; just
    # report the manifest sha separately for receipts.)

    return manifest


# ============================================================
# Lookahead invariant test (Task F)
# ============================================================


def _perturb_paths(
    paths_full: pd.DataFrame,
    sample_trade_ids: list[int],
    bar_offset_floor: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a copy of paths_full with OHLC columns randomly perturbed for
    the sampled trade_ids at bar_offset >= bar_offset_floor. Other rows
    unchanged.
    """
    perturbed = paths_full.copy()
    mask = (perturbed["trade_id"].isin(sample_trade_ids)) & (
        perturbed["bar_offset"] >= bar_offset_floor
    )
    n_cells = int(mask.sum())
    if n_cells == 0:
        return perturbed
    # Add large random noise so any feature dependency would show through
    for col in (
        "open",
        "high",
        "low",
        "close",
        "cum_logret_from_entry",
        "mfe_to_date_atr",
        "mae_to_date_atr",
    ):
        noise = rng.normal(loc=0.0, scale=10.0, size=n_cells)
        perturbed.loc[mask, col] = perturbed.loc[mask, col].values + noise
    return perturbed


def lookahead_invariant_test(n_sample: int = 100, seed: int = 17) -> dict:
    """Test that classifier predictions for F2..F5 trades are invariant to
    perturbations of trade_paths data at bar_offset > t* (i.e., beyond
    what the predictor reads).

    Returns dict with sampled/byte_identical/divergent counts and per-trade
    diff list (empty on PASS).
    """
    D._load_paths_full.cache_clear()

    signals_clu, _ = _load_inputs()

    # Baseline predictions for F2..F5
    preds_baseline = P.fit_predict_cluster_anchored_expanding(signals_clu, T_STAR)
    preds_baseline = preds_baseline[preds_baseline["fold"].isin([2, 3, 4, 5])]
    preds_baseline = preds_baseline.set_index("trade_id").sort_index()

    # Sample trades from the baseline prediction set (these are the trades
    # with active_mask & valid_cluster_mask in F2..F5)
    all_ids = preds_baseline.index.tolist()
    if not all_ids:
        return {
            "sampled": 0,
            "byte_identical": 0,
            "divergent": 0,
            "diffs": [],
            "note": "no F2..F5 trades to sample",
        }
    rng = np.random.default_rng(seed)
    sample_ids = list(rng.choice(all_ids, size=min(n_sample, len(all_ids)), replace=False))
    sample_ids_int = [int(x) for x in sample_ids]

    # Load paths and perturb at bar_offset >= 2 (predictor reads <= 1)
    paths_full = D._load_paths_full().copy()
    paths_perturbed = _perturb_paths(
        paths_full, sample_ids_int, bar_offset_floor=T_STAR + 1, rng=rng
    )

    # Monkey-patch the loader to return perturbed paths
    original_loader = D._load_paths_full
    D._load_paths_full = lambda: paths_perturbed.copy()
    # Also clear pivot caches if any
    try:
        preds_perturbed = P.fit_predict_cluster_anchored_expanding(signals_clu, T_STAR)
    finally:
        D._load_paths_full = original_loader
        D._load_paths_full.cache_clear()

    preds_perturbed = preds_perturbed[preds_perturbed["fold"].isin([2, 3, 4, 5])]
    preds_perturbed = preds_perturbed.set_index("trade_id").sort_index()

    # Compare predictions for sampled trades only (the others are unperturbed
    # but might shift due to z-score normalisation contamination — that's a
    # different invariant; this test scopes to the sampled trades)
    diffs = []
    byte_identical = 0
    for tid in sample_ids_int:
        if tid not in preds_baseline.index or tid not in preds_perturbed.index:
            diffs.append({"trade_id": tid, "reason": "missing from one set"})
            continue
        b = preds_baseline.loc[tid]
        p = preds_perturbed.loc[tid]
        # Byte-identity check on the predicted_cluster int
        if int(b["predicted_cluster"]) == int(p["predicted_cluster"]) and float(
            b["p_cluster_1"]
        ) == float(p["p_cluster_1"]):
            byte_identical += 1
        else:
            diffs.append(
                {
                    "trade_id": tid,
                    "baseline_p1": float(b["p_cluster_1"]),
                    "perturbed_p1": float(p["p_cluster_1"]),
                    "baseline_class": int(b["predicted_cluster"]),
                    "perturbed_class": int(p["predicted_cluster"]),
                }
            )

    return {
        "sampled": len(sample_ids_int),
        "byte_identical": byte_identical,
        "divergent": len(diffs),
        "diffs": diffs[:10],  # cap reported diffs at 10
        "perturbation_floor_bar_offset": T_STAR + 1,
    }


if __name__ == "__main__":
    m = materialise()
    print("Materialisation manifest summary:")
    print(
        json.dumps(
            {
                "git_commit": m["git_commit"],
                "per_fold_counts": m["per_fold_predicted_mirror_counts"],
                "total": m["post_mechanism_n_rows"],
                "outputs_sha256": m["outputs_sha256"],
            },
            indent=2,
        )
    )
