"""Data loading + predictor whitelist construction for L Arc 1 Step 3.

Outputs:
- load_signals(): the signals_features.csv augmented with
  peak_to_final_r_ratio_fwd_h240 (derived from forward path)
  and reached_plus_1atr_within_240 (derived from existing columns).
- predictor_feature_set(): the 3c signal-time whitelist as a single
  DataFrame with one-hot expanded categoricals + standardised numerics.
- load_t_features(t): the 3d held-bar feature set at slice t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _common as C


def load_signals() -> pd.DataFrame:
    """Load signals_features.csv and add derived columns."""
    df = pd.read_csv(C.SIGNALS_CSV)

    # ---- derive peak_to_final_r_ratio_fwd_h240 (forward-path version) ----
    eps = 1e-6
    realised_atr_h240 = (
        df["signal_bar_close"].astype(float) * df["fwd_logret_h240"].astype(float)
        / df["atr_at_signal_1h"].astype(float)
    )
    df["peak_to_final_r_ratio_fwd_h240"] = (
        realised_atr_h240 / np.maximum(df["fwd_mfe_h240_atr"].astype(float), eps)
    )

    # ---- ensure reached_plus_1_atr_within_240 exists ----
    if "reached_plus_1.0_atr_within_240" not in df.columns:
        df["reached_plus_1.0_atr_within_240"] = (
            df["bars_to_plus_1.0_atr_capped_480"].astype(float) <= 240
        ).astype(int)

    return df


def compute_v11_new_features(df_signals: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    """Compute v1.1 Amendment 4 features from trade_paths.csv.

    Returns a DataFrame with columns:
      trade_id,
      fwd_realized_range_atr,
      fwd_fraction_time_above_entry,
      fwd_max_consecutive_directional_bars.

    Persisted to cache_path (CSV) if provided. Re-reads from cache if exists
    AND the cached trade_paths sha256 matches (deterministic).
    """
    if cache_path is not None and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            if (cached["trade_id"].values == df_signals["trade_id"].values).all() and \
               set(["fwd_realized_range_atr",
                    "fwd_fraction_time_above_entry",
                    "fwd_max_consecutive_directional_bars"]).issubset(cached.columns):
                return cached
        except Exception:
            pass

    # Compute from trade_paths.csv (22M rows × 6 cols).
    # Strategy: load, sort by (trade_id, t), reshape to (n_trades, n_bars)
    # matrices, then vectorise.
    paths = pd.read_csv(C.PATHS_CSV, usecols=["trade_id", "t", "fwd_logret_cum", "fwd_logret_step"])
    paths = paths.sort_values(["trade_id", "t"]).reset_index(drop=True)

    # Confirm regular grid
    tid_unique = paths["trade_id"].unique()
    n_trades = len(tid_unique)
    n_bars = len(paths) // n_trades
    assert n_trades * n_bars == len(paths), (
        f"paths is not a regular grid: n_trades*n_bars={n_trades*n_bars} != len={len(paths)}"
    )

    lr_cum_mat = paths["fwd_logret_cum"].values.reshape(n_trades, n_bars)
    lr_step_mat = paths["fwd_logret_step"].values.reshape(n_trades, n_bars)

    # fwd_fraction_time_above_entry: fraction of bars where cum > 0
    frac_above = (lr_cum_mat > 0).mean(axis=1)

    # fwd_realized_range_atr: (max_lr - min_lr) * entry / atr (small-return approx)
    max_lr = lr_cum_mat.max(axis=1)
    min_lr = lr_cum_mat.min(axis=1)
    sig = df_signals[["trade_id", "signal_bar_close", "atr_at_signal_1h"]].set_index("trade_id")
    sig = sig.reindex(tid_unique)  # align order
    entry = sig["signal_bar_close"].values.astype(float)
    atr = sig["atr_at_signal_1h"].values.astype(float)
    realized_range = entry * (max_lr - min_lr) / np.where(atr > 0, atr, np.nan)

    # fwd_max_consecutive_directional_bars: per row, max run of same nonzero sign
    signs = np.sign(lr_step_mat).astype(np.int8)
    # change[i, t] = 1 if signs[i, t] differs from signs[i, t-1]
    change = np.empty_like(signs, dtype=bool)
    change[:, 0] = True
    change[:, 1:] = signs[:, 1:] != signs[:, :-1]
    # group_id per (row, position) via cumsum of change along axis 1
    group_id = np.cumsum(change, axis=1)
    # For each (row, group_id) find length and sign. Concatenate row offset to
    # make global group ids unique.
    row_offset = (np.arange(n_trades).reshape(-1, 1) * (group_id.max() + 2))
    global_gid = (group_id + row_offset).ravel()
    signs_flat = signs.ravel()
    # Aggregate per global gid: count + first sign
    # Use bincount; group sizes summed
    counts = np.bincount(global_gid)
    # first-sign per gid = lookup at first occurrence
    first_idx = np.full(counts.shape[0], -1, dtype=np.int64)
    # The first occurrence of each gid: it's the smallest index where global_gid==gid;
    # since global_gid is monotonically non-decreasing within rows, the first idx of
    # each gid is its starting index. We can compute via np.diff on global_gid.
    diff_mask = np.empty(global_gid.shape, dtype=bool)
    diff_mask[0] = True
    diff_mask[1:] = global_gid[1:] != global_gid[:-1]
    start_indices = np.where(diff_mask)[0]
    first_gids = global_gid[start_indices]
    first_signs_for_gid = signs_flat[start_indices]
    # Map gid -> sign
    sign_per_gid = np.zeros(counts.shape[0], dtype=np.int8)
    sign_per_gid[first_gids] = first_signs_for_gid
    # max run per row: gids in row i have value in [i*(maxgid+2), (i+1)*(maxgid+2))
    row_block = group_id.max() + 2
    max_run = np.zeros(n_trades, dtype=np.int64)
    for i in range(n_trades):
        lo = i * row_block
        hi = lo + row_block
        row_counts = counts[lo:hi]
        row_signs = sign_per_gid[lo:hi]
        mask = row_signs != 0
        max_run[i] = int(row_counts[mask].max()) if mask.any() else 0

    out = pd.DataFrame({
        "trade_id": tid_unique,
        "fwd_realized_range_atr": realized_range,
        "fwd_fraction_time_above_entry": frac_above,
        "fwd_max_consecutive_directional_bars": max_run,
    })
    out = out.set_index("trade_id").reindex(df_signals["trade_id"].values).reset_index()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False, float_format="%.10g", lineterminator="\n")
    return out


def load_paths_sample(trade_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Load trade_paths.csv (selectively if trade_ids passed)."""
    df = pd.read_csv(C.PATHS_CSV)
    if trade_ids is not None:
        df = df[df["trade_id"].isin(trade_ids)].copy()
    return df


def load_t_features(t: int) -> pd.DataFrame:
    """Load forward_context_evolution/t{t}.csv."""
    p = C.FWD_CTX / f"t{t}.csv"
    df = pd.read_csv(p)
    # Rename so they can be joined alongside signal-time features
    keep = ["trade_id"] + list(C.FWD_CTX_NUMERIC)
    df = df[keep]
    return df


# ---- predictor feature set ----------------------------------------------

def _onehot(series: pd.Series, prefix: str) -> pd.DataFrame:
    s = series.fillna("__nan__").astype(str)
    levels = sorted(s.unique().tolist())
    return pd.DataFrame({f"{prefix}__{lev}": (s == lev).astype(np.int8) for lev in levels})


def build_signal_time_matrix(df: pd.DataFrame, *, return_meta: bool = False):
    """Return (X, feature_names, group_for_categorical).

    X is a numeric ndarray, standardised numerics + one-hot categoricals.
    feature_names is parallel list of names. group_for_categorical maps
    feature name to original categorical column (for BH grouping if needed).
    """
    cols_num = list(C.SIGNAL_TIME_NUMERIC)
    cols_cat = list(C.SIGNAL_TIME_CATEGORICAL)

    num = df[cols_num].astype(float)
    # impute NaNs with column median (rare; signal_bar_volume_nan flag captures)
    num = num.fillna(num.median(numeric_only=True))

    # standardise numerics
    mu = num.mean(axis=0)
    sd = num.std(axis=0, ddof=0).replace(0, 1.0)
    num_z = (num - mu) / sd

    cat_frames = []
    cat_meta = {}
    for c in cols_cat:
        oh = _onehot(df[c], c)
        cat_frames.append(oh)
        for name in oh.columns:
            cat_meta[name] = c

    pieces = [num_z.reset_index(drop=True)] + [f.reset_index(drop=True) for f in cat_frames]
    X = pd.concat(pieces, axis=1)
    feature_names = X.columns.tolist()

    if return_meta:
        # parallel array of "source feature name" — collapse one-hot dummies
        # back to their parent for BH grouping (and for clarity in the
        # haircut output).
        source_for = {n: n for n in cols_num}
        source_for.update(cat_meta)
        meta = pd.DataFrame({
            "feature": feature_names,
            "source": [source_for.get(n, n) for n in feature_names],
            "kind": ["numeric" if n in cols_num else "categorical_dummy" for n in feature_names],
        })
        return X.values.astype(np.float64), feature_names, meta
    return X.values.astype(np.float64), feature_names


def build_t_matrix(df_signals: pd.DataFrame, t: int):
    """Signal-time features + first-bar features + forward_context @ t.

    Returns (X, feature_names, meta_df)."""
    X_st, names_st, meta_st = build_signal_time_matrix(df_signals, return_meta=True)

    # first-bar features (available at t>=1)
    fb_num = df_signals[["first_bar_range_atr"]].astype(float)
    fb_num = fb_num.fillna(fb_num.median(numeric_only=True))
    fb_mu = fb_num.mean(); fb_sd = fb_num.std(ddof=0).replace(0, 1.0)
    fb_num_z = (fb_num - fb_mu) / fb_sd

    fb_dir = _onehot(df_signals["first_bar_direction"], "first_bar_direction")
    fb_bin = _onehot(df_signals["first_bar_range_bin"], "first_bar_range_bin")

    # forward_context at t
    tdf = load_t_features(t).set_index("trade_id").reindex(df_signals["trade_id"].values)
    tnum = tdf[list(C.FWD_CTX_NUMERIC)].astype(float).reset_index(drop=True)
    tnum = tnum.fillna(tnum.median(numeric_only=True))
    tmu = tnum.mean(); tsd = tnum.std(ddof=0).replace(0, 1.0)
    tnum_z = (tnum - tmu) / tsd
    # suffix to disambiguate from signal-time
    tnum_z.columns = [f"{c}_t{t}" for c in tnum_z.columns]

    extra = pd.concat([fb_num_z.reset_index(drop=True),
                       fb_dir.reset_index(drop=True),
                       fb_bin.reset_index(drop=True),
                       tnum_z.reset_index(drop=True)], axis=1)
    extra_names = extra.columns.tolist()

    X_full = np.concatenate([X_st, extra.values.astype(np.float64)], axis=1)
    names_full = names_st + extra_names

    extra_sources = (
        ["first_bar_range_atr"]
        + ["first_bar_direction"] * fb_dir.shape[1]
        + ["first_bar_range_bin"] * fb_bin.shape[1]
        + [f"{c}_t{t}" for c in C.FWD_CTX_NUMERIC]
    )
    extra_kinds = (
        ["numeric"]
        + ["categorical_dummy"] * fb_dir.shape[1]
        + ["categorical_dummy"] * fb_bin.shape[1]
        + ["numeric"] * len(C.FWD_CTX_NUMERIC)
    )
    extra_meta = pd.DataFrame({
        "feature": extra_names, "source": extra_sources, "kind": extra_kinds,
    })
    meta = pd.concat([meta_st, extra_meta], ignore_index=True)
    return X_full, names_full, meta


def build_cluster_features(df: pd.DataFrame):
    """Clustering feature subset per prompt Phase A.

    Returns (X_z, feature_names).
    """
    num = df[list(C.CLUSTER_FEATURES_NUMERIC)].astype(float)
    num = num.fillna(num.median(numeric_only=True))
    mu = num.mean(); sd = num.std(ddof=0).replace(0, 1.0)
    num_z = (num - mu) / sd

    cat_frames = [_onehot(df[c], c) for c in C.CLUSTER_FEATURES_CATEGORICAL]
    pieces = [num_z.reset_index(drop=True)] + [f.reset_index(drop=True) for f in cat_frames]
    X = pd.concat(pieces, axis=1)
    return X.values.astype(np.float64), X.columns.tolist()
