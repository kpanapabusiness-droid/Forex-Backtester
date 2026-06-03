"""v2.0 predictability investigation driver.

Three angles per target archetype + grouped target:
  A) RF on the 8-feature basic entry set (reused from v2.0 diagnostic)
  B) Logistic + RF on the expanded feature set
  C) RF on path-so-far features at t in {1, 3, 5, 10}

Determinism: pinned single-thread BLAS; random_state=42 throughout.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.v1_3_calibration.load_paths import load_paths  # noqa: E402
from scripts.v2_0_predictability import features_expanded as FE  # noqa: E402
from scripts.v2_0_predictability import features_t as FT  # noqa: E402
from scripts.v2_0_predictability import models as M  # noqa: E402
from scripts.v2_0_predictability import report as RP  # noqa: E402
from scripts.v2_0_predictability import targets as TG  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_ROOT = REPO_ROOT / "results" / "v2_0_diagnostic"
OUT_ROOT  = REPO_ROOT / "results" / "v2_0_predictability"

DATASETS = ("kh24", "arc2")
T_OBSERVE = (1, 3, 5, 10)

BASIC_FEATURES = (
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_to_atr_14",
    "ret_5bar_atr",
    "ret_20bar_atr",
    "pos_in_20bar_range",
    "rsi_14",
)

PATH_SO_FAR_FEATURES = (
    "close_r_at_t",
    "mfe_so_far_r_at_t",
    "mae_so_far_r_at_t",
    "bars_in_profit_at_t",
    "local_peaks_so_far_at_t",
    "monotonicity_so_far_at_t",
    "velocity_first_t",
)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def _load_basic_entry_features(dataset: str) -> pd.DataFrame:
    """Reuse the 8-feature entry-features CSV from v2.0 diagnostic."""
    return pd.read_csv(DIAG_ROOT / dataset / "entry_features_basic.csv")


def _load_cluster_assignments(dataset: str, k: int) -> pd.DataFrame:
    return pd.read_csv(DIAG_ROOT / dataset / f"clusters_K{k}.csv")


def _baseline_auc(dataset: str, k: int, archetype_id: int) -> float:
    """Read the logistic AUC from the diagnostic's predictability CSV."""
    p = pd.read_csv(DIAG_ROOT / dataset / f"predictability_K{k}.csv")
    sub = p[p["archetype_id"] == archetype_id]
    if len(sub) == 0:
        return float("nan")
    return float(sub["auc_mean"].iloc[0])


def _logistic_baseline_for_group(
    entry_features: pd.DataFrame, dataset: str, k: int, archetype_ids: list[int],
) -> tuple[float, float, list[float]]:
    """Compute logistic baseline AUC on basic features for a grouped target."""
    asgn = _load_cluster_assignments(dataset, k)
    asgn["trade_id"] = asgn["trade_id"].astype("string")
    ef = entry_features.copy()
    ef["trade_id"] = ef["trade_id"].astype("string")
    df = ef.merge(asgn, on="trade_id", how="inner").dropna(subset=list(BASIC_FEATURES))
    y = df["archetype_id"].isin(archetype_ids).astype(int).to_numpy()
    X = df[list(BASIC_FEATURES)].astype("float64").to_numpy()
    return M.fit_logistic(X, y)


def _build_targets_long(targets: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    """Long-form table of (dataset, K, target_id, archetype_ids list, target_kind).

    target_kind in {'archetype', 'group:<tag>'}
    """
    rows = []
    for _, r in targets.iterrows():
        rows.append({
            "dataset": r["dataset"],
            "K": int(r["K"]),
            "target_id": f"arch_{int(r['archetype_id'])}",
            "target_kind": "archetype",
            "archetype_ids": [int(r["archetype_id"])],
            "exit_family_tag": r["exit_family_tag"],
        })
    for _, r in grouped.iterrows():
        ids = [int(x) for x in str(r["archetype_ids_in_group"]).split(",")]
        if len(ids) < 2:
            continue  # singleton groups duplicate the archetype target
        rows.append({
            "dataset": r["dataset"],
            "K": int(r["K"]),
            "target_id": f"group_{r['exit_family_tag']}",
            "target_kind": "group",
            "archetype_ids": ids,
            "exit_family_tag": r["exit_family_tag"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Angle A — RF on the 8 basic features
# ----------------------------------------------------------------------

def angle_A(targets_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cache_basic_ef: dict[str, pd.DataFrame] = {}
    for ds in DATASETS:
        cache_basic_ef[ds] = _load_basic_entry_features(ds)

    for _, t in targets_long.iterrows():
        ds = t["dataset"]
        k = int(t["K"])
        ef = cache_basic_ef[ds].copy()
        ef["trade_id"] = ef["trade_id"].astype("string")
        asgn = _load_cluster_assignments(ds, k)
        asgn["trade_id"] = asgn["trade_id"].astype("string")
        df = ef.merge(asgn, on="trade_id", how="inner").dropna(subset=list(BASIC_FEATURES))
        y = df["archetype_id"].isin(t["archetype_ids"]).astype(int).to_numpy()
        X = df[list(BASIC_FEATURES)].astype("float64").to_numpy()

        rf_mean, rf_std, folds = M.fit_rf(X, y)
        if t["target_kind"] == "archetype":
            baseline = _baseline_auc(ds, k, t["archetype_ids"][0])
        else:
            baseline, _, _ = _logistic_baseline_for_group(cache_basic_ef[ds], ds, k, t["archetype_ids"])

        rows.append({
            "dataset": ds,
            "K": k,
            "target_id": t["target_id"],
            "target_kind": t["target_kind"],
            "archetype_ids_in_target": ",".join(str(i) for i in t["archetype_ids"]),
            "exit_family_tag": t["exit_family_tag"],
            "target_size": int(y.sum()),
            "n_total": int(len(y)),
            "auc_mean": rf_mean,
            "auc_std":  rf_std,
            "auc_fold_1": folds[0],
            "auc_fold_2": folds[1],
            "auc_fold_3": folds[2],
            "auc_fold_4": folds[3],
            "auc_fold_5": folds[4],
            "auc_logistic_baseline": baseline,
            "lift_vs_logistic": (rf_mean - baseline) if (not np.isnan(rf_mean) and not np.isnan(baseline)) else float("nan"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Angle B — expanded feature set
# ----------------------------------------------------------------------

def _build_expanded_ef(
    dataset: str, targets: pd.DataFrame, out_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute (or load) expanded feature set per dataset. Returns (frame, feature names)."""
    out_path = out_root / dataset / "entry_features_expanded.csv"
    if dataset == "kh24":
        base = _load_basic_entry_features("kh24").copy()
        base["trade_id"] = base["trade_id"].astype("string")
        ext = FE.kh24_expanded_features()
        ext["trade_id"] = ext["trade_id"].astype("string")
        # Merge; basic + extended (drop duplicate 'pair', 'entry_date' from ext)
        ext_cols = [c for c in ext.columns if c not in ("pair", "entry_date")]
        ef = base.merge(ext[ext_cols], on="trade_id", how="left", validate="one_to_one")
        feature_names = list(BASIC_FEATURES) + [c for c in ext.columns if c not in ("trade_id", "pair", "entry_date")]
        _write_csv(ef, out_path)
        return ef, feature_names

    # arc2
    base = _load_basic_entry_features("arc2").copy()
    base["trade_id"] = base["trade_id"].astype("string")
    arc2_ef, selected = FE.arc2_expanded_features(targets, top_k=30)
    arc2_ef["trade_id"] = arc2_ef["trade_id"].astype("string")
    ef = base.merge(arc2_ef, on="trade_id", how="left", validate="one_to_one")
    feature_names = list(BASIC_FEATURES) + selected
    _write_csv(ef, out_path)
    return ef, feature_names


def angle_B(targets_long: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    cache: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for ds in DATASETS:
        ef, feats = _build_expanded_ef(ds, targets, OUT_ROOT)
        cache[ds] = (ef, feats)

    rows = []
    for _, t in targets_long.iterrows():
        ds = t["dataset"]
        k = int(t["K"])
        ef, feats = cache[ds]
        ef = ef.copy()
        ef["trade_id"] = ef["trade_id"].astype("string")
        asgn = _load_cluster_assignments(ds, k)
        asgn["trade_id"] = asgn["trade_id"].astype("string")
        df = ef.merge(asgn, on="trade_id", how="inner")
        # Filter rows where any feature is NaN.
        df = df.dropna(subset=feats)
        y = df["archetype_id"].isin(t["archetype_ids"]).astype(int).to_numpy()
        X = df[feats].astype("float64").to_numpy()

        log_m, log_s, _ = M.fit_logistic(X, y)
        rf_m, rf_s, _ = M.fit_rf(X, y)

        if t["target_kind"] == "archetype":
            baseline = _baseline_auc(ds, k, t["archetype_ids"][0])
        else:
            baseline, _, _ = _logistic_baseline_for_group(_load_basic_entry_features(ds), ds, k, t["archetype_ids"])

        rows.append({
            "dataset": ds,
            "K": k,
            "target_id": t["target_id"],
            "target_kind": t["target_kind"],
            "archetype_ids_in_target": ",".join(str(i) for i in t["archetype_ids"]),
            "exit_family_tag": t["exit_family_tag"],
            "target_size": int(y.sum()),
            "n_total": int(len(y)),
            "n_features_used": int(len(feats)),
            "auc_logistic_expanded_mean": log_m,
            "auc_logistic_expanded_std":  log_s,
            "auc_rf_expanded_mean": rf_m,
            "auc_rf_expanded_std":  rf_s,
            "auc_logistic_baseline": baseline,
            "lift_logistic_expanded_vs_baseline": (log_m - baseline) if (not np.isnan(log_m) and not np.isnan(baseline)) else float("nan"),
            "lift_rf_expanded_vs_baseline":       (rf_m  - baseline) if (not np.isnan(rf_m)  and not np.isnan(baseline)) else float("nan"),
        })
    feature_lists = {ds: feats for ds, (_, feats) in cache.items()}
    return pd.DataFrame(rows), feature_lists


# ----------------------------------------------------------------------
# Angle C — t > 0 path-so-far
# ----------------------------------------------------------------------

def angle_C(targets_long: pd.DataFrame, angle_a_df: pd.DataFrame) -> pd.DataFrame:
    """RF AUC at t in {1,3,5,10} using 8 basic + 7 path-so-far features."""
    cache_basic_ef: dict[str, pd.DataFrame] = {ds: _load_basic_entry_features(ds) for ds in DATASETS}
    cache_pm: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {ds: load_paths(ds) for ds in DATASETS}
    cache_t_feats: dict[tuple[str, int], tuple[pd.DataFrame, int]] = {}

    for ds in DATASETS:
        paths, meta = cache_pm[ds]
        for t in T_OBSERVE:
            f, excl = FT.compute_path_so_far_features(paths, meta, t)
            cache_t_feats[(ds, t)] = (f, excl)

    rows = []
    for _, tg in targets_long.iterrows():
        ds = tg["dataset"]
        k = int(tg["K"])
        ef = cache_basic_ef[ds].copy()
        ef["trade_id"] = ef["trade_id"].astype("string")
        asgn = _load_cluster_assignments(ds, k)
        asgn["trade_id"] = asgn["trade_id"].astype("string")

        # Reference: RF AUC with basic 8 features only (Angle A) for this target
        ref = angle_a_df[
            (angle_a_df["dataset"] == ds) & (angle_a_df["K"] == k) & (angle_a_df["target_id"] == tg["target_id"])
        ]
        rf_basic_auc = float(ref["auc_mean"].iloc[0]) if len(ref) else float("nan")

        row = {
            "dataset": ds,
            "K": k,
            "target_id": tg["target_id"],
            "target_kind": tg["target_kind"],
            "archetype_ids_in_target": ",".join(str(i) for i in tg["archetype_ids"]),
            "exit_family_tag": tg["exit_family_tag"],
            "auc_rf_basic_only": rf_basic_auc,
        }
        for t in T_OBSERVE:
            ftf, excl = cache_t_feats[(ds, t)]
            ftf2 = ftf.copy()
            ftf2["trade_id"] = ftf2["trade_id"].astype("string")
            df = ef.merge(ftf2, on="trade_id", how="inner")
            df = df.merge(asgn, on="trade_id", how="inner")
            df = df.dropna(subset=list(BASIC_FEATURES) + list(PATH_SO_FAR_FEATURES))
            y = df["archetype_id"].isin(tg["archetype_ids"]).astype(int).to_numpy()
            X = df[list(BASIC_FEATURES) + list(PATH_SO_FAR_FEATURES)].astype("float64").to_numpy()
            m, s, _ = M.fit_rf(X, y)
            row[f"target_size_at_t{t}"] = int(y.sum())
            row[f"n_excluded_at_t{t}"]  = int(excl)
            row[f"auc_rf_mean_at_t{t}"] = m
            row[f"auc_rf_std_at_t{t}"]  = s
            row[f"lift_vs_entry_only_at_t{t}"] = (m - rf_basic_auc) if (not np.isnan(m) and not np.isnan(rf_basic_auc)) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run() -> None:
    print(f"[v2.0-predictability] writing to {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Step 1: targets
    targets_raw = TG.identify_targets()
    targets, grouping, grouped = TG.build_grouping(targets_raw)
    _write_csv(targets,  OUT_ROOT / "target_archetypes.csv")
    _write_csv(grouping, OUT_ROOT / "exit_family_grouping.csv")
    _write_csv(grouped,  OUT_ROOT / "grouped_targets.csv")
    print(f"  targets: {len(targets)} | groups: {len(grouped)}")

    targets_long = _build_targets_long(targets, grouped)
    _write_csv(
        targets_long.assign(archetype_ids=targets_long["archetype_ids"].apply(lambda L: ",".join(str(i) for i in L))),
        OUT_ROOT / "targets_long.csv",
    )

    # Angle A
    a = angle_A(targets_long)
    _write_csv(a, OUT_ROOT / "angle_A_rf_basic.csv")
    print(f"  Angle A: {len(a)} rows; max AUC = {a['auc_mean'].max():.4f}")

    # Angle B
    b, feature_lists = angle_B(targets_long, targets)
    _write_csv(b, OUT_ROOT / "angle_B_expanded.csv")
    for ds, feats in feature_lists.items():
        _write_csv(
            pd.DataFrame({"feature": feats}),
            OUT_ROOT / ds / "expanded_feature_list.csv",
        )
    print(f"  Angle B: {len(b)} rows; max RF AUC = {b['auc_rf_expanded_mean'].max():.4f}")

    # Angle C
    c = angle_C(targets_long, a)
    _write_csv(c, OUT_ROOT / "angle_C_t_observation.csv")
    print(f"  Angle C: {len(c)} rows; max AUC at any t = {max(c[[f'auc_rf_mean_at_t{t}' for t in T_OBSERVE]].max()):.4f}")

    # Report
    RP.write_report(OUT_ROOT, targets, grouping, grouped, a, b, c, feature_lists)
    print(f"[v2.0-predictability] report written: {OUT_ROOT / 'PREDICTABILITY_INVESTIGATION.md'}")


if __name__ == "__main__":
    run()
