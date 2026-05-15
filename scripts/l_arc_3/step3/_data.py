"""Data loading + predictor whitelist construction for L Arc 3 Step 3.

Arc 3-specific notes:
- 12 clustering features per task spec (held-window 9 + Amendment 4 three).
- `mfe_sequence_class_held` ORDINAL encoded (MAE_first=0, simultaneous_bar=1, MFE_first=2).
- Amendment 4 features are ALREADY in signals_features.csv (pre-computed in step 2).
- Phase D held-bar features include trade_paths bar-by-bar observations
  PLUS held_bar_evolution/t{t}.csv context.
"""

# ruff: noqa: E402, E701, E702, F841, I001, F401
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _common as C


def load_signals() -> pd.DataFrame:
    """Load signals_features.csv and add minimal derived columns."""
    df = pd.read_csv(C.SIGNALS_CSV)
    # Ensure reached_plus_1.0_atr_within_240 — derive from the 480-capped col on arc 3
    if "reached_plus_1.0_atr_within_240" not in df.columns:
        df["reached_plus_1.0_atr_within_240"] = (
            df["bars_to_plus_1.0_atr_capped_480"].astype(float) <= 240
        ).astype(int)
    return df


def load_t_held_features(t: int) -> pd.DataFrame:
    """Load held_bar_evolution/t{t}.csv (per task spec for Phase D)."""
    p = C.HELD_CTX / f"t{t}.csv"
    df = pd.read_csv(p)
    keep = ["trade_id"] + list(C.HELD_CTX_NUMERIC)
    return df[keep]


def load_paths_to_t(t: int, all_trade_ids: np.ndarray) -> pd.DataFrame:
    """Load trade_paths.csv bar-by-bar observations from bar_offset 0..t.

    Returns a wide-format DataFrame keyed by trade_id with columns:
      <feature>_b{offset} for offset in 0..t, feature in PATH_NUMERIC_PER_BAR.
    Missing rows (e.g., bar_offset > bars_held for some trades) yield NaN.
    """
    # Read only needed columns and rows for efficiency.
    cols = ["trade_id", "bar_offset"] + list(C.PATH_NUMERIC_PER_BAR)
    df = pd.read_csv(C.PATHS_CSV, usecols=cols)
    df = df[df["bar_offset"] <= t].copy()

    # Wide-format reshape per trade × bar
    pieces = []
    for off in range(t + 1):
        sub = df[df["bar_offset"] == off].set_index("trade_id")[list(C.PATH_NUMERIC_PER_BAR)]
        sub = sub.add_suffix(f"_b{off}")
        pieces.append(sub)
    wide = pd.concat(pieces, axis=1)
    wide = wide.reindex(all_trade_ids)
    wide = wide.reset_index().rename(columns={"index": "trade_id"})
    return wide


# ---- predictor feature set ---------------------------------------------


def _onehot(series: pd.Series, prefix: str) -> pd.DataFrame:
    s = series.fillna("__nan__").astype(str)
    levels = sorted(s.unique().tolist())
    return pd.DataFrame({f"{prefix}__{lev}": (s == lev).astype(np.int8) for lev in levels})


def build_signal_time_matrix(df: pd.DataFrame, *, return_meta: bool = False):
    """Standardised numerics + one-hot categoricals.

    Returns (X, feature_names, meta_df) when return_meta=True.
    """
    cols_num = list(C.SIGNAL_TIME_NUMERIC)
    cols_cat = list(C.SIGNAL_TIME_CATEGORICAL)

    num = df[cols_num].astype(float)
    num = num.fillna(num.median(numeric_only=True))
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
        source_for = {n: n for n in cols_num}
        source_for.update(cat_meta)
        meta = pd.DataFrame(
            {
                "feature": feature_names,
                "source": [source_for.get(n, n) for n in feature_names],
                "kind": [
                    "numeric" if n in cols_num else "categorical_dummy" for n in feature_names
                ],
            }
        )
        return X.values.astype(np.float64), feature_names, meta
    return X.values.astype(np.float64), feature_names


def build_t_matrix(df_signals: pd.DataFrame, t: int):
    """Signal-time + first-bar + trade_paths bar 0..t + held_bar_evolution @ t."""
    X_st, names_st, meta_st = build_signal_time_matrix(df_signals, return_meta=True)

    # First-bar features
    fb_num = df_signals[["first_bar_range_atr"]].astype(float)
    fb_num = fb_num.fillna(fb_num.median(numeric_only=True))
    fb_mu = fb_num.mean()
    fb_sd = fb_num.std(ddof=0).replace(0, 1.0)
    fb_num_z = (fb_num - fb_mu) / fb_sd
    fb_dir = _onehot(df_signals["first_bar_direction"], "first_bar_direction")
    fb_bin = _onehot(df_signals["first_bar_range_bin"], "first_bar_range_bin")

    # Bar-by-bar path observations (bars 0..t)
    paths_wide = load_paths_to_t(t, df_signals["trade_id"].values)
    paths_num = paths_wide.drop(columns=["trade_id"]).astype(float)
    paths_num = paths_num.fillna(paths_num.median(numeric_only=True))
    pmu = paths_num.mean()
    psd = paths_num.std(ddof=0).replace(0, 1.0)
    paths_z = (paths_num - pmu) / psd.replace(0, 1.0)
    paths_z = paths_z.fillna(0.0)  # any leftover

    # Held-bar context @ t
    hdf = load_t_held_features(t).set_index("trade_id").reindex(df_signals["trade_id"].values)
    hnum = hdf[list(C.HELD_CTX_NUMERIC)].astype(float).reset_index(drop=True)
    hnum = hnum.fillna(hnum.median(numeric_only=True))
    hnum = hnum.fillna(0.0)  # in case the entire column is NaN at this t
    hmu = hnum.mean()
    hsd = hnum.std(ddof=0).replace(0, 1.0)
    hnum_z = (hnum - hmu) / hsd
    hnum_z.columns = [f"{c}_held_t{t}" for c in hnum_z.columns]

    extra = pd.concat(
        [
            fb_num_z.reset_index(drop=True),
            fb_dir.reset_index(drop=True),
            fb_bin.reset_index(drop=True),
            paths_z.reset_index(drop=True),
            hnum_z.reset_index(drop=True),
        ],
        axis=1,
    )
    extra_names = extra.columns.tolist()

    X_full = np.concatenate([X_st, extra.values.astype(np.float64)], axis=1)
    names_full = names_st + extra_names

    extra_sources = (
        ["first_bar_range_atr"]
        + ["first_bar_direction"] * fb_dir.shape[1]
        + ["first_bar_range_bin"] * fb_bin.shape[1]
        + paths_z.columns.tolist()
        + [f"{c}_held_t{t}" for c in C.HELD_CTX_NUMERIC]
    )
    extra_kinds = (
        ["numeric"]
        + ["categorical_dummy"] * fb_dir.shape[1]
        + ["categorical_dummy"] * fb_bin.shape[1]
        + ["numeric"] * paths_z.shape[1]
        + ["numeric"] * len(C.HELD_CTX_NUMERIC)
    )
    extra_meta = pd.DataFrame(
        {
            "feature": extra_names,
            "source": extra_sources,
            "kind": extra_kinds,
        }
    )
    meta = pd.concat([meta_st, extra_meta], ignore_index=True)
    return X_full, names_full, meta


def build_cluster_features(df: pd.DataFrame, return_drop_mask: bool = False):
    """Clustering feature subset per task spec (12 features).

    Numerics z-scored. Ordinal-encoded `mfe_sequence_class_held` per task spec.
    Returns (X_z, feature_names) by default; if return_drop_mask True, also
    returns a boolean array (n_rows,) of rows kept (no NaN in any clustering feature).
    """
    cols_num = list(C.CLUSTER_FEATURES_NUMERIC)

    num = df[cols_num].astype(float).copy()

    # Ordinal encoding for mfe_sequence_class_held
    ord_cols = []
    for col, mapping in C.CLUSTER_FEATURES_ORDINAL.items():
        ord_series = df[col].map(mapping)
        num[col] = ord_series.astype(float)
        ord_cols.append(col)

    # Drop rows with any NaN in clustering features (per task spec)
    keep_mask = ~num.isna().any(axis=1)

    # Standardise on the kept rows only
    sub = num[keep_mask]
    mu = sub.mean()
    sd = sub.std(ddof=0).replace(0, 1.0)
    num_z_full = ((num - mu) / sd).fillna(0.0)

    feature_names = num.columns.tolist()
    X = num_z_full.values.astype(np.float64)
    if return_drop_mask:
        return X, feature_names, keep_mask.values
    return X, feature_names
