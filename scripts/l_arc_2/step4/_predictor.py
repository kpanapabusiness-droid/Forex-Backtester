"""Cluster predictor for cluster-conditional candidates (1, 2, 3).

Builds held-bar feature matrix at bar t and trains a HistGradientBoosting
classifier (kmeans K=2) on F1..F5, predicts on F6..F7. Returns per-trade
predictions with trade_id, fold, true_cluster, p_cluster_1, predicted_cluster.

Pattern mirrors `scripts/l_arc_2/step3/_data.build_t_matrix`:
- Signal-time numeric (z-scored) + categorical (one-hot).
- First-bar numeric + categorical.
- Per-bar path observations 0..t (pivoted to wide, z-scored).
- Held-bar context @ t (z-scored).
- Fill NaN with median, then 0 for any remaining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import _common as C
from . import _data as D

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

SIGNAL_TIME_CATEGORICAL = (
    "pair",
    "session",
    "vol_regime",
    "pre_momentum_bin",
    "cum_logret_1h_6_bin",
    "cum_logret_1h_24_bin",
    "cum_logret_1h_168_bin",
)


def _onehot(series: pd.Series, prefix: str) -> pd.DataFrame:
    s = series.fillna("__nan__").astype(str)
    levels = sorted(s.unique().tolist())
    return pd.DataFrame({f"{prefix}__{lev}": (s == lev).astype(np.int8) for lev in levels})


def _zscore_block(num: pd.DataFrame) -> pd.DataFrame:
    """Median-fill, z-score, then fill 0 for any remaining NaN."""
    num = num.astype(float)
    num = num.fillna(num.median(numeric_only=True))
    mu = num.mean(axis=0)
    sd = num.std(axis=0, ddof=0).replace(0, 1.0)
    z = (num - mu) / sd
    return z.fillna(0.0)


def build_t_matrix(signals_df: pd.DataFrame, t: int):
    """Build feature matrix X (n_rows × d) at bar t for the full signal set.

    Returns (X np.float64 [n,d], feature_names list).
    """
    df = signals_df.reset_index(drop=True).copy()
    trade_ids = df["trade_id"].values

    # ---- Signal-time numeric ----
    num_st = df[list(SIGNAL_TIME_NUMERIC)].copy()
    num_st_z = _zscore_block(num_st)

    # ---- Signal-time categorical ----
    cat_pieces = []
    for c in SIGNAL_TIME_CATEGORICAL:
        cat_pieces.append(_onehot(df[c], c))

    # ---- First-bar numeric ----
    fb_num = df[["first_bar_range_atr"]].copy()
    fb_num_z = _zscore_block(fb_num)

    # ---- First-bar categorical ----
    fb_dir = _onehot(df["first_bar_direction"], "first_bar_direction")
    fb_bin = _onehot(df["first_bar_range_bin"], "first_bar_range_bin")

    # ---- Per-bar path obs 0..t ----
    paths_wide = D.load_paths_wide_to_t(t, trade_ids)
    paths_num = paths_wide.drop(columns=["trade_id"]).copy()
    paths_z = _zscore_block(paths_num)

    # ---- Held-bar context @ t ----
    held = D.load_held_ctx(t).set_index("trade_id").reindex(trade_ids).reset_index(drop=True)
    held_z = _zscore_block(held)

    parts = [
        num_st_z.reset_index(drop=True),
        *[c.reset_index(drop=True) for c in cat_pieces],
        fb_num_z.reset_index(drop=True),
        fb_dir.reset_index(drop=True),
        fb_bin.reset_index(drop=True),
        paths_z.reset_index(drop=True),
        held_z.reset_index(drop=True),
    ]
    X_df = pd.concat(parts, axis=1)
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float64)
    return X, feature_names


def fit_predict_cluster(signals_df: pd.DataFrame, t: int) -> pd.DataFrame:
    """Fit HGB classifier on F1..F5 valid-cluster active trades, predict on F6..F7.

    Active = bars_held >= t.
    Excludes sentinel kmeans_K2_cluster_id == -2 from BOTH fit and predict sets.

    Returns DataFrame with cols:
        trade_id, fold, true_cluster, p_cluster_1, predicted_cluster,
        bars_already_held_at_t, already_exited_before_t.
    Only rows that are "active at t AND valid cluster" are returned.
    """
    df = signals_df.reset_index(drop=True).copy()
    if C.CLUSTER_COL_INTERNAL not in df.columns:
        raise ValueError(f"signals_df missing {C.CLUSTER_COL_INTERNAL} — merge clusters first")

    # Build full-pool X (deterministic stable order = trade_id ascending)
    df = df.sort_values("trade_id").reset_index(drop=True)
    X_full, _ = build_t_matrix(df, t)

    bars_held = df["bars_held"].values
    cluster = df[C.CLUSTER_COL_INTERNAL].values
    fold = df["fold_id"].values

    active_mask = bars_held >= t
    valid_cluster_mask = cluster != C.CLUSTER_SENTINEL
    use_mask = active_mask & valid_cluster_mask

    fit_mask = use_mask & np.isin(fold, list(C.FIT_FOLDS))
    pred_mask = use_mask & np.isin(fold, list(C.VALIDATE_FOLDS))

    X_fit = X_full[fit_mask]
    y_fit = cluster[fit_mask].astype(int)

    if len(X_fit) == 0 or len(np.unique(y_fit)) < 2:
        # No training data — return empty predictions
        return pd.DataFrame(
            columns=[
                "trade_id",
                "fold",
                "true_cluster",
                "p_cluster_1",
                "predicted_cluster",
                "bars_already_held_at_t",
                "already_exited_before_t",
            ]
        )

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=C.HGB_RANDOM_STATE,
    )
    clf.fit(X_fit, y_fit)

    # Predict on F6/F7 active-valid trades
    X_pred = X_full[pred_mask]
    if len(X_pred) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "fold",
                "true_cluster",
                "p_cluster_1",
                "predicted_cluster",
                "bars_already_held_at_t",
                "already_exited_before_t",
            ]
        )

    proba = clf.predict_proba(X_pred)
    # Make sure class 1 column index is found robustly
    classes = clf.classes_
    if 1 in classes:
        c1_idx = int(np.where(classes == 1)[0][0])
        p1 = proba[:, c1_idx]
    else:
        p1 = np.zeros(len(X_pred))

    pred_cluster = (p1 >= 0.5).astype(int)

    out = pd.DataFrame(
        {
            "trade_id": df.loc[pred_mask, "trade_id"].values,
            "fold": fold[pred_mask],
            "true_cluster": cluster[pred_mask].astype(int),
            "p_cluster_1": p1,
            "predicted_cluster": pred_cluster,
            "bars_already_held_at_t": np.full(int(pred_mask.sum()), t, dtype=int),
            "already_exited_before_t": (bars_held[pred_mask] < t).astype(int),
        }
    )
    return out.sort_values("trade_id").reset_index(drop=True)


def fit_predict_cluster_all_folds(signals_df: pd.DataFrame, t: int) -> pd.DataFrame:
    """For internal t-selection: also predict on F1..F5 with out-of-fold scheme.

    Train one model per held-out fold f∈F1..F5 (LOFO within F1..F5), predict on f.
    This avoids in-sample contamination during t-selection.
    Plus the standard F6/F7 predictions from a F1..F5-fit model.
    """
    df = signals_df.reset_index(drop=True).sort_values("trade_id").reset_index(drop=True)
    if C.CLUSTER_COL_INTERNAL not in df.columns:
        raise ValueError(f"signals_df missing {C.CLUSTER_COL_INTERNAL}")

    X_full, _ = build_t_matrix(df, t)
    bars_held = df["bars_held"].values
    cluster = df[C.CLUSTER_COL_INTERNAL].values
    fold = df["fold_id"].values

    active_mask = bars_held >= t
    valid_cluster_mask = cluster != C.CLUSTER_SENTINEL
    use_mask = active_mask & valid_cluster_mask

    rows = []

    # ---- OOF predictions within F1..F5 (LOFO) ----
    fit_folds = list(C.FIT_FOLDS)
    for f in fit_folds:
        train_folds = [x for x in fit_folds if x != f]
        train_mask = use_mask & np.isin(fold, train_folds)
        test_mask = use_mask & (fold == f)
        if not train_mask.any() or not test_mask.any():
            continue
        y_train = cluster[train_mask].astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=C.HGB_RANDOM_STATE,
        )
        clf.fit(X_full[train_mask], y_train)
        proba = clf.predict_proba(X_full[test_mask])
        classes = clf.classes_
        if 1 in classes:
            c1_idx = int(np.where(classes == 1)[0][0])
            p1 = proba[:, c1_idx]
        else:
            p1 = np.zeros(int(test_mask.sum()))
        pred = (p1 >= 0.5).astype(int)
        sub = pd.DataFrame(
            {
                "trade_id": df.loc[test_mask, "trade_id"].values,
                "fold": fold[test_mask],
                "true_cluster": cluster[test_mask].astype(int),
                "p_cluster_1": p1,
                "predicted_cluster": pred,
                "bars_already_held_at_t": np.full(int(test_mask.sum()), t, dtype=int),
                "already_exited_before_t": (bars_held[test_mask] < t).astype(int),
            }
        )
        rows.append(sub)

    # ---- F6/F7 from F1..F5-fit model ----
    fit_all_mask = use_mask & np.isin(fold, fit_folds)
    val_mask = use_mask & np.isin(fold, list(C.VALIDATE_FOLDS))
    if fit_all_mask.any() and val_mask.any():
        y_fit = cluster[fit_all_mask].astype(int)
        if len(np.unique(y_fit)) >= 2:
            clf = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=C.HGB_RANDOM_STATE,
            )
            clf.fit(X_full[fit_all_mask], y_fit)
            proba = clf.predict_proba(X_full[val_mask])
            classes = clf.classes_
            if 1 in classes:
                c1_idx = int(np.where(classes == 1)[0][0])
                p1 = proba[:, c1_idx]
            else:
                p1 = np.zeros(int(val_mask.sum()))
            pred = (p1 >= 0.5).astype(int)
            sub = pd.DataFrame(
                {
                    "trade_id": df.loc[val_mask, "trade_id"].values,
                    "fold": fold[val_mask],
                    "true_cluster": cluster[val_mask].astype(int),
                    "p_cluster_1": p1,
                    "predicted_cluster": pred,
                    "bars_already_held_at_t": np.full(int(val_mask.sum()), t, dtype=int),
                    "already_exited_before_t": (bars_held[val_mask] < t).astype(int),
                }
            )
            rows.append(sub)

    if not rows:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "fold",
                "true_cluster",
                "p_cluster_1",
                "predicted_cluster",
                "bars_already_held_at_t",
                "already_exited_before_t",
            ]
        )
    return pd.concat(rows, axis=0, ignore_index=True).sort_values("trade_id").reset_index(drop=True)


def fit_predict_cluster_anchored_expanding(signals_df: pd.DataFrame, t: int) -> pd.DataFrame:
    """Anchored expanding gb cluster predictor for step 5 F2..F5 materialisation.

    Per-fold training scheme:
        F1: skipped (no rows emitted) — pre-F1 history is empty in L Arc 2
            (signals_features.csv fold_id range is exactly 1..7), so the
            anchored-expanding training set for F1 has no examples.
        F2: train on F1                  -> predict F2
        F3: train on F1+F2               -> predict F3
        F4: train on F1+F2+F3            -> predict F4
        F5: train on F1+F2+F3+F4         -> predict F5
        F6,F7: single pooled F1..F5 fit  -> predict F6+F7 from one model
               (byte-identical to fit_predict_cluster's F6/F7 branch).

    The F2..F5 branch is the new piece; F6/F7 mirrors fit_predict_cluster
    exactly so that step 4's published trades_post_mechanism.csv F6+F7 rows
    re-emerge byte-identical when the F6/F7 predictions are passed through
    run_delayed_entry.

    Determinism requirements (preserved from fit_predict_cluster):
    - X_full built ONCE via build_t_matrix on the full sorted pool before
      any per-fold masking. Z-score normalisation, median fill, and one-hot
      level enumeration must see the full pool. Splitting X_full construction
      per-fold would change normalisation and break byte identity.
    - F6/F7 predictions come from a single predict_proba call covering both
      folds combined (not two separate calls).
    - Hyperparameters and random_state pinned from C.HGB_RANDOM_STATE.
    - Output sorted by trade_id ascending.
    """
    df = signals_df.reset_index(drop=True).copy()
    if C.CLUSTER_COL_INTERNAL not in df.columns:
        raise ValueError(f"signals_df missing {C.CLUSTER_COL_INTERNAL} — merge clusters first")

    df = df.sort_values("trade_id").reset_index(drop=True)
    X_full, _ = build_t_matrix(df, t)

    bars_held = df["bars_held"].values
    cluster = df[C.CLUSTER_COL_INTERNAL].values
    fold = df["fold_id"].values

    active_mask = bars_held >= t
    valid_cluster_mask = cluster != C.CLUSTER_SENTINEL
    use_mask = active_mask & valid_cluster_mask

    rows = []

    # ---- F2..F5 anchored expanding ----
    # F1 deliberately skipped (pre-F1 history empty in L Arc 2).
    for f_target in (2, 3, 4, 5):
        train_folds = list(range(1, f_target))  # F2 trains on [1]; F5 trains on [1,2,3,4]
        train_mask = use_mask & np.isin(fold, train_folds)
        test_mask = use_mask & (fold == f_target)
        if not train_mask.any() or not test_mask.any():
            continue
        y_train = cluster[train_mask].astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=C.HGB_RANDOM_STATE,
        )
        clf.fit(X_full[train_mask], y_train)
        proba = clf.predict_proba(X_full[test_mask])
        classes = clf.classes_
        if 1 in classes:
            c1_idx = int(np.where(classes == 1)[0][0])
            p1 = proba[:, c1_idx]
        else:
            p1 = np.zeros(int(test_mask.sum()))
        pred = (p1 >= 0.5).astype(int)
        sub = pd.DataFrame(
            {
                "trade_id": df.loc[test_mask, "trade_id"].values,
                "fold": fold[test_mask],
                "true_cluster": cluster[test_mask].astype(int),
                "p_cluster_1": p1,
                "predicted_cluster": pred,
                "bars_already_held_at_t": np.full(int(test_mask.sum()), t, dtype=int),
                "already_exited_before_t": (bars_held[test_mask] < t).astype(int),
            }
        )
        rows.append(sub)

    # ---- F6/F7 from pooled F1..F5 fit (mirrors fit_predict_cluster) ----
    fit_mask = use_mask & np.isin(fold, list(C.FIT_FOLDS))
    pred_mask = use_mask & np.isin(fold, list(C.VALIDATE_FOLDS))
    if fit_mask.any() and pred_mask.any():
        y_fit = cluster[fit_mask].astype(int)
        if len(np.unique(y_fit)) >= 2:
            clf = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=C.HGB_RANDOM_STATE,
            )
            clf.fit(X_full[fit_mask], y_fit)
            proba = clf.predict_proba(X_full[pred_mask])
            classes = clf.classes_
            if 1 in classes:
                c1_idx = int(np.where(classes == 1)[0][0])
                p1 = proba[:, c1_idx]
            else:
                p1 = np.zeros(int(pred_mask.sum()))
            pred = (p1 >= 0.5).astype(int)
            sub = pd.DataFrame(
                {
                    "trade_id": df.loc[pred_mask, "trade_id"].values,
                    "fold": fold[pred_mask],
                    "true_cluster": cluster[pred_mask].astype(int),
                    "p_cluster_1": p1,
                    "predicted_cluster": pred,
                    "bars_already_held_at_t": np.full(int(pred_mask.sum()), t, dtype=int),
                    "already_exited_before_t": (bars_held[pred_mask] < t).astype(int),
                }
            )
            rows.append(sub)

    if not rows:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "fold",
                "true_cluster",
                "p_cluster_1",
                "predicted_cluster",
                "bars_already_held_at_t",
                "already_exited_before_t",
            ]
        )
    return pd.concat(rows, axis=0, ignore_index=True).sort_values("trade_id").reset_index(drop=True)
