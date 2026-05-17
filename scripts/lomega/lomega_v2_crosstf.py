"""Lomega v2 — cross-TF feature expansion (interpretation 3 test).

Tests whether the B1-B4 worst-fold AUC ceiling at 1H/4H is bound by feature
set thinness vs market-structure limitations. Two changes vs B1-B4:

  1. Cross-TF feature expansion — ~15 multi-TF features covering D1 vol
     regime / D1 swing structure / cross-TF trend agreement / multi-TF EMA
     stack / D1 momentum / weekly EMA slope.
  2. Relaxed label — keep mono_pre_peak >= 0.55 and mae_pre_peak_R > -1.0;
     drop reached_1R requirement; require at least one of {mfe_max_R >= 1.0,
     reached_0.5R_pre_peak}.

Run on 1H and 4H only (D1 skipped per dispatch scope). Fold dates and RF
hyperparameters identical to B1-B4 for comparability. Full 45-feature set
trained (NOT top-15 by univariate AUC).

Usage:
    py scripts/lomega/lomega_v2_crosstf.py --tf 1h
    py scripts/lomega/lomega_v2_crosstf.py --tf 4h
    py scripts/lomega/lomega_v2_crosstf.py --tf all
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# Reuse byte-identical helpers from B1-B4 — feature definitions for the
# 30 reused features and the existing cross-TF anchors must match exactly.
from scripts.lomega.lomega_b1_b4 import (  # noqa: E402
    PAIRS,
    USD_BASE,
    USD_QUOTE,
    START_DATE,
    END_DATE,
    RNG_SEED,
    load_pair,
    compute_atr,
    ema_adjust_false,
    rolling_rank_pct,
    rolling_std,
    sma,
    compute_pair_features,
    attach_d1_anchor,
    attach_h4_anchor,
    attach_dxy_proxy,
    cohen_d,
)


_REPO = Path(__file__).resolve().parent.parent.parent
_DATA = _REPO / "data"
_OUT = _REPO / "results" / "lomega" / "v2_crosstf"


# Per-TF config — forward windows + RF leaf size identical to B1-B4.
# D1 omitted: dispatch limits scope to 1H/4H.
TF_CONFIG = {
    "1h": {"folder": "1hr", "forward_window": 480, "rf_min_samples_leaf": 20},
    "4h": {"folder": "4hr", "forward_window": 240, "rf_min_samples_leaf": 20},
}


def load_weekly(pair: str) -> pd.DataFrame:
    """Load weekly bars (data/w1/<pair>.csv) for cross-TF anchoring."""
    fp = _DATA / "w1" / f"{pair}.csv"
    df = pd.read_csv(fp, usecols=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float64")
    return df


# ---------------------------------------------------------------- B1: relaxed labels

@dataclass
class RelaxedLabelArrays:
    clean_move_relaxed: np.ndarray
    clean_score_relaxed: np.ndarray
    mono_pre_peak: np.ndarray
    mfe_max_R: np.ndarray
    mae_pre_peak_R: np.ndarray
    reached_0_5R_pre_peak: np.ndarray
    reached_1R_pre_peak: np.ndarray
    peak_mfe_bar: np.ndarray
    has_label: np.ndarray


def compute_labels_relaxed(close, high, low, atr_14, time_vals, W: int) -> RelaxedLabelArrays:
    """Relaxed clean_move label.

    REQUIRED:
      mono_pre_peak >= 0.55  AND  mae_pre_peak_R > -1.0
    OR-condition (at least one):
      mfe_max_R >= 1.0  OR  reached_0.5R_pre_peak (== mfe_max_R >= 0.5)

    Graded clean_score_relaxed = int(mono>=0.55) + int(mae>-1) +
                                 int(mfe>=1.0) + int(mfe>=0.5).
    Range [0, 4]. Note dispatch wrote {0..3} but the four listed sub-
    conditions admit a max of 4 (with mfe>=1.0 strictly implying mfe>=0.5).
    """
    n = len(close)
    cm = np.zeros(n, dtype=bool)
    cs = np.zeros(n, dtype=np.int8)
    mono = np.zeros(n, dtype="float64")
    mfe_max = np.zeros(n, dtype="float64")
    mae_pp = np.zeros(n, dtype="float64")
    r05 = np.zeros(n, dtype=bool)
    r1 = np.zeros(n, dtype=bool)
    peak_bar = np.zeros(n, dtype=np.int32)
    has_label = np.zeros(n, dtype=bool)

    end_ts = END_DATE.to_numpy()
    for t in range(n):
        if t + W >= n:
            continue
        if time_vals[t + W] > end_ts:
            continue
        R = 2.0 * atr_14[t]
        if not np.isfinite(R) or R <= 0:
            continue

        c_t = close[t]
        fwd_high = high[t + 1 : t + 1 + W]
        fwd_low = low[t + 1 : t + 1 + W]
        fwd_close = close[t + 1 : t + 1 + W]

        mfe_path = (fwd_high - c_t) / R
        mae_path = (fwd_low - c_t) / R
        close_r = (fwd_close - c_t) / R

        mfe_so_far = np.maximum.accumulate(mfe_path)
        mae_so_far = np.minimum.accumulate(mae_path)

        pb = int(np.argmax(mfe_so_far))
        mfe_val = float(mfe_so_far[pb])
        mae_val = float(mae_so_far[pb])
        reached_05 = mfe_val >= 0.5
        reached_1 = mfe_val >= 1.0

        pre_close_r = close_r[: pb + 1]
        ip = pre_close_r[pre_close_r > 0]
        if ip.size > 1:
            mono_val = float(np.mean(ip[1:] >= ip[:-1]))
        else:
            mono_val = 0.0

        req_mono = mono_val >= 0.55
        req_mae = mae_val > -1.0
        cond_mfe1 = reached_1
        cond_mfe05 = reached_05

        passes = req_mono and req_mae and (cond_mfe1 or cond_mfe05)
        score = int(req_mono) + int(req_mae) + int(cond_mfe1) + int(cond_mfe05)

        has_label[t] = True
        cm[t] = passes
        cs[t] = score
        mono[t] = mono_val
        mfe_max[t] = mfe_val
        mae_pp[t] = mae_val
        r05[t] = reached_05
        r1[t] = reached_1
        peak_bar[t] = pb

    return RelaxedLabelArrays(
        clean_move_relaxed=cm,
        clean_score_relaxed=cs,
        mono_pre_peak=mono,
        mfe_max_R=mfe_max,
        mae_pre_peak_R=mae_pp,
        reached_0_5R_pre_peak=r05,
        reached_1R_pre_peak=r1,
        peak_mfe_bar=peak_bar,
        has_label=has_label,
    )


# ---------------------------------------------------------------- B2: cross-TF features

def _compute_d1_panel(d1_df: pd.DataFrame) -> pd.DataFrame:
    """Compute D1-derived features keyed by d1_date (the calendar date the
    D1 bar opens). Returned frame is sorted by d1_date and contains the
    full set of D1 features that will be anchored to lower-TF signal bars
    via the strict prior-calendar-day lag rule.
    """
    c = d1_df["close"].values.astype("float64")
    h = d1_df["high"].values.astype("float64")
    lo = d1_df["low"].values.astype("float64")

    atr14, _ = compute_atr(h, lo, c, 14)

    # D1 vol regime
    atr_pct20 = rolling_rank_pct(atr14, 20)
    atr_pct60 = rolling_rank_pct(atr14, 60)
    bar_range = h - lo
    rng_mean20 = sma(bar_range, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        range_compression20 = bar_range / rng_mean20
        atr_14_normalised = atr14 / c
    log_ret = np.zeros(len(c), dtype="float64")
    log_ret[1:] = np.log(c[1:] / c[:-1])
    realised_vol20 = rolling_std(log_ret, 20)

    # D1 swing structure (higher_highs / higher_lows over last 5 bars)
    hh_step = np.zeros(len(c), dtype=np.int8)
    hl_step = np.zeros(len(c), dtype=np.int8)
    hh_step[1:] = (h[1:] > h[:-1]).astype(np.int8)
    hl_step[1:] = (lo[1:] > lo[:-1]).astype(np.int8)
    hh_5 = pd.Series(hh_step).rolling(5, min_periods=5).sum().values
    hl_5 = pd.Series(hl_step).rolling(5, min_periods=5).sum().values

    # Swing state: priority-encoded categorical.
    #   4 up_strong   if hh>=4 and hl>=4
    #   0 down_strong if hh<=1 and hl<=1
    #   3 up_weak     if hh>=3 and hl>=3 (not strong)
    #   1 down_weak   if hh<=2 and hl<=2 (not strong-down)
    #   2 range       if 2<=hh<=3 and 2<=hl<=3
    #   5 mixed       if hh/hl disagree (e.g., hh=4 hl=2) — kept as
    #                  separate non-ordinal bucket so the row is not
    #                  dropped by the all-finite RF row filter. Documented
    #                  in the summary; the dispatch's literal 5-bucket
    #                  list does not name a mixed bin but ~8% of bars
    #                  fall in this gap.
    #   NaN if hh_5/hl_5 themselves NaN (warmup).
    swing_state = np.full(len(c), np.nan, dtype="float64")
    valid = ~np.isnan(hh_5)
    hh_v = np.where(valid, hh_5, np.nan)
    hl_v = np.where(valid, hl_5, np.nan)
    is_up_strong = (hh_v >= 4) & (hl_v >= 4)
    is_down_strong = (hh_v <= 1) & (hl_v <= 1)
    is_up_weak = (hh_v >= 3) & (hl_v >= 3) & ~is_up_strong
    is_down_weak = (hh_v <= 2) & (hl_v <= 2) & ~is_down_strong
    is_range = (hh_v >= 2) & (hh_v <= 3) & (hl_v >= 2) & (hl_v <= 3)
    swing_state = np.where(is_up_strong, 4.0, swing_state)
    swing_state = np.where(is_down_strong, 0.0, swing_state)
    swing_state = np.where(is_up_weak & np.isnan(swing_state), 3.0, swing_state)
    swing_state = np.where(is_down_weak & np.isnan(swing_state), 1.0, swing_state)
    swing_state = np.where(is_range & np.isnan(swing_state), 2.0, swing_state)
    swing_state = np.where(valid & np.isnan(swing_state), 5.0, swing_state)

    # D1 EMA stack — need EMA-20, EMA-50, EMA-200 levels
    ema20 = ema_adjust_false(c, 20)
    ema50 = ema_adjust_false(c, 50)
    ema200 = ema_adjust_false(c, 200)
    ema_stack_up = (ema20 > ema50) & (ema50 > ema200)

    # D1 momentum (atr-normalised price displacement)
    n = len(c)
    mom5 = np.full(n, np.nan)
    mom20 = np.full(n, np.nan)
    if n > 5:
        with np.errstate(divide="ignore", invalid="ignore"):
            mom5[5:] = (c[5:] - c[:-5]) / atr14[5:]
    if n > 20:
        with np.errstate(divide="ignore", invalid="ignore"):
            mom20[20:] = (c[20:] - c[:-20]) / atr14[20:]

    out = pd.DataFrame(
        {
            "d1_date": d1_df["time"].dt.normalize().values,
            "d1_atr_percentile_20": atr_pct20,
            "d1_atr_percentile_60": atr_pct60,
            "d1_range_compression_20": range_compression20,
            "d1_realised_vol_20": realised_vol20,
            "d1_atr_14_normalised": atr_14_normalised,
            "d1_higher_highs_5": hh_5,
            "d1_higher_lows_5": hl_5,
            "d1_swing_state": swing_state,
            "d1_ema_stack_up": ema_stack_up.astype("float64"),
            "d1_momentum_5_atr": mom5,
            "d1_momentum_20_atr": mom20,
        }
    )
    return out.sort_values("d1_date").reset_index(drop=True)


def attach_d1_extra(features_per_pair: dict, d1_data: dict) -> None:
    """Attach D1-derived extra features to lower-TF features via strict
    prior-calendar-day lag: for signal time T, use the largest D1 bar with
    d1_date < T.date().
    """
    extra_cols = [
        "d1_atr_percentile_20", "d1_atr_percentile_60",
        "d1_range_compression_20", "d1_realised_vol_20",
        "d1_atr_14_normalised", "d1_higher_highs_5",
        "d1_higher_lows_5", "d1_swing_state",
        "d1_ema_stack_up", "d1_momentum_5_atr", "d1_momentum_20_atr",
    ]
    for pair, feats in features_per_pair.items():
        d1_panel = _compute_d1_panel(d1_data[pair])
        target_dates = pd.to_datetime(feats["time"]).dt.normalize().values
        idx = np.searchsorted(d1_panel["d1_date"].values, target_dates, side="left") - 1
        valid = idx >= 0
        for col in extra_cols:
            arr = d1_panel[col].values
            out = np.full(len(feats), np.nan)
            out[valid] = arr[idx[valid]]
            feats[col] = out


def _compute_h4_panel(h4_df: pd.DataFrame) -> pd.DataFrame:
    """H4 EMA stack levels keyed by h4_time. Used to anchor h4_ema_stack_up
    onto 1H signal bars via "most recent CLOSED H4 bar" rule.
    """
    c = h4_df["close"].values.astype("float64")
    ema20 = ema_adjust_false(c, 20)
    ema50 = ema_adjust_false(c, 50)
    ema200 = ema_adjust_false(c, 200)
    stack_up = ((ema20 > ema50) & (ema50 > ema200)).astype("float64")
    return pd.DataFrame(
        {"h4_time": h4_df["time"].values, "h4_ema_stack_up": stack_up}
    ).sort_values("h4_time").reset_index(drop=True)


def attach_h4_extra_for_1h(features_per_pair: dict, h4_data: dict) -> None:
    """Attach h4_ema_stack_up onto 1H feature frames via the most recent
    CLOSED H4 bar rule (B closed at B+4h => use largest B with B + 4h <= T).
    """
    four_h = np.timedelta64(4, "h")
    for pair, feats in features_per_pair.items():
        h4_panel = _compute_h4_panel(h4_data[pair])
        h4_panel = h4_panel.dropna(subset=["h4_ema_stack_up"]).reset_index(drop=True)
        # Treat as valid only where EMA-200 has had 200 bars to converge,
        # i.e., wherever stack_up is finite (ema_adjust_false NaN until ready).
        target = pd.to_datetime(feats["time"]).values - four_h
        idx = np.searchsorted(h4_panel["h4_time"].values, target, side="right") - 1
        valid = idx >= 0
        out = np.full(len(feats), np.nan)
        out[valid] = h4_panel["h4_ema_stack_up"].values[idx[valid]]
        feats["h4_ema_stack_up"] = out


def attach_h4_extra_for_4h(features_per_pair: dict, h4_data: dict) -> None:
    """At 4H signal TF the signal bar IS the H4 bar, so use the SAME-TF EMA
    stack — but the existing compute_pair_features only emits slopes and
    pairwise booleans, not the level-stack triplet. Compute h4_ema_stack_up
    directly from each pair's 4H close at the signal bar.
    """
    for pair, feats in features_per_pair.items():
        df = h4_data[pair]
        c = df["close"].values.astype("float64")
        ema20 = ema_adjust_false(c, 20)
        ema50 = ema_adjust_false(c, 50)
        ema200 = ema_adjust_false(c, 200)
        stack_up = ((ema20 > ema50) & (ema50 > ema200)).astype("float64")
        # Align by time: features at the same h4_time should map directly.
        # df is the source of feats too — index alignment must match.
        time_to_stack = pd.DataFrame(
            {"time": df["time"].values, "h4_ema_stack_up": stack_up}
        )
        merged = feats[["time"]].merge(time_to_stack, on="time", how="left")
        feats["h4_ema_stack_up"] = merged["h4_ema_stack_up"].values


def attach_h1_anchor_for_4h(features_per_pair: dict, h1_data: dict) -> None:
    """Attach h1_ema_50_slope_at_entry to 4H feature frames via the most
    recent CLOSED 1H bar (B closes at B+1h => use largest B with B+1h <= T).
    Needed at 4H to feed cross_tf_trend_alignment over (D1, H4, H1).
    """
    one_h = np.timedelta64(1, "h")
    for pair, feats in features_per_pair.items():
        df = h1_data[pair]
        c = df["close"].values.astype("float64")
        ema50 = ema_adjust_false(c, 50)
        slope50 = np.full(len(df), np.nan)
        slope50[50:] = (ema50[50:] - ema50[:-50]) / ema50[:-50]
        panel = pd.DataFrame(
            {"h1_time": df["time"].values, "h1_ema_50_slope": slope50}
        ).dropna(subset=["h1_ema_50_slope"]).sort_values("h1_time").reset_index(drop=True)
        target = pd.to_datetime(feats["time"]).values - one_h
        idx = np.searchsorted(panel["h1_time"].values, target, side="right") - 1
        valid = idx >= 0
        out = np.full(len(feats), np.nan)
        out[valid] = panel["h1_ema_50_slope"].values[idx[valid]]
        feats["h1_ema_50_slope_at_entry"] = out


def attach_w_anchor(features_per_pair: dict, w_data: dict) -> None:
    """Attach w_ema_20_slope to lower-TF features via "most recent CLOSED W
    bar" rule. A W bar at timestamp B opens that week (Sunday) and closes 7
    days later. Use the largest B such that B + 7d <= T.
    """
    one_w = np.timedelta64(7, "D")
    for pair, feats in features_per_pair.items():
        df = w_data[pair]
        c = df["close"].values.astype("float64")
        ema20 = ema_adjust_false(c, 20)
        slope20 = np.full(len(df), np.nan)
        slope20[20:] = (ema20[20:] - ema20[:-20]) / ema20[:-20]
        panel = pd.DataFrame(
            {"w_time": df["time"].values, "w_ema_20_slope": slope20}
        ).dropna(subset=["w_ema_20_slope"]).sort_values("w_time").reset_index(drop=True)
        target = pd.to_datetime(feats["time"]).values - one_w
        idx = np.searchsorted(panel["w_time"].values, target, side="right") - 1
        valid = idx >= 0
        out = np.full(len(feats), np.nan)
        out[valid] = panel["w_ema_20_slope"].values[idx[valid]]
        feats["w_ema_20_slope"] = out


def compute_cross_tf_agreement(features_per_pair: dict, tf: str) -> None:
    """Derive cross_tf_trend_alignment (count of D1/H4/H1 slopes > 0),
    cross_tf_trend_strong_up (all three > 0), and d1_h4_slope_agreement
    (same sign D1 and H4 slopes). Mutates frames in place.

    Slope sources per signal TF:
      1H: D1 anchor, H4 anchor, signal-TF own EMA-50 slope (== H1)
      4H: D1 anchor, signal-TF own EMA-50 slope (== H4), H1 anchor
    """
    for pair, feats in features_per_pair.items():
        d1_slope = feats["d1_ema_50_slope_at_entry"].values
        if tf == "1h":
            h4_slope = feats["h4_ema_50_slope_at_entry"].values
            h1_slope = feats["ema_50_slope"].values
        elif tf == "4h":
            h4_slope = feats["ema_50_slope"].values
            h1_slope = feats["h1_ema_50_slope_at_entry"].values
        else:
            raise ValueError(f"Unsupported TF for cross-TF agreement: {tf}")

        with np.errstate(invalid="ignore"):
            d1_pos = (d1_slope > 0).astype("float64")
            h4_pos = (h4_slope > 0).astype("float64")
            h1_pos = (h1_slope > 0).astype("float64")

        # If any anchor is NaN, set the alignment count NaN as well to
        # avoid spurious zero counts on warmup rows. (Once 2020+ this is
        # essentially never hit.)
        nan_mask = (
            np.isnan(d1_slope) | np.isnan(h4_slope) | np.isnan(h1_slope)
        )
        alignment = d1_pos + h4_pos + h1_pos
        alignment = np.where(nan_mask, np.nan, alignment)
        strong_up = ((d1_slope > 0) & (h4_slope > 0) & (h1_slope > 0)).astype("float64")
        strong_up = np.where(nan_mask, np.nan, strong_up)
        d1h4_agree = (np.sign(d1_slope) == np.sign(h4_slope)).astype("float64")
        d1h4_nan = np.isnan(d1_slope) | np.isnan(h4_slope)
        d1h4_agree = np.where(d1h4_nan, np.nan, d1h4_agree)

        feats["cross_tf_trend_alignment"] = alignment
        feats["cross_tf_trend_strong_up"] = strong_up
        feats["d1_h4_slope_agreement"] = d1h4_agree


# ---------------------------------------------------------------- B3: univariate

def b3_univariate(features: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = labels.astype(int)
    for col in feature_cols:
        x = features[col].values
        mask = np.isfinite(x)
        xm = x[mask]
        ym = y[mask]
        if len(np.unique(ym)) < 2 or len(xm) < 50:
            rows.append({"feature": col, "auc": np.nan, "ks_stat": np.nan,
                         "ks_pvalue": np.nan, "cohen_d": np.nan,
                         "mean_pos": np.nan, "mean_neg": np.nan,
                         "n": int(len(xm))})
            continue
        try:
            lr = LogisticRegression(max_iter=200, solver="lbfgs")
            lr.fit(xm.reshape(-1, 1), ym)
            proba = lr.predict_proba(xm.reshape(-1, 1))[:, 1]
            auc = float(roc_auc_score(ym, proba))
        except Exception:
            auc = float("nan")
        pos = xm[ym == 1]
        neg = xm[ym == 0]
        if len(pos) > 0 and len(neg) > 0:
            ks = sps.ks_2samp(pos, neg)
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
            d = cohen_d(pos, neg)
            mp = float(np.mean(pos))
            mn = float(np.mean(neg))
        else:
            ks_stat = ks_p = d = mp = mn = float("nan")
        rows.append({"feature": col, "auc": auc, "ks_stat": ks_stat,
                     "ks_pvalue": ks_p, "cohen_d": d, "mean_pos": mp,
                     "mean_neg": mn, "n": int(len(xm))})
    out = pd.DataFrame(rows)
    out["auc_dist_from_0_5"] = (out["auc"] - 0.5).abs()
    out = out.sort_values("auc_dist_from_0_5", ascending=False).drop(columns="auc_dist_from_0_5")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- B4: full-feature multivariate

def b4_multivariate_full(features: pd.DataFrame, labels: np.ndarray,
                         feature_cols: list[str], pair_col: np.ndarray,
                         time_col: np.ndarray, n_splits: int = 7,
                         min_samples_leaf: int = 20):
    """Time-series CV (anchored expanding) on the FULL feature set — NOT
    top-15. Records per-fold per-feature importance for drift analysis.
    """
    X_raw = features[feature_cols].values
    y = labels.astype(int)
    mask = np.all(np.isfinite(X_raw), axis=1)
    X = X_raw[mask]
    y = y[mask]
    pairs = pair_col[mask]
    times = time_col[mask]

    order = np.argsort(times, kind="stable")
    X = X[order]
    y = y[order]
    pairs = pairs[order]
    times = times[order]

    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    fold_aucs = []
    per_fold_importances = {}
    pair_breakdown_per_fold = []

    for fold_idx, (tr, va) in enumerate(tss.split(X), start=1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            fold_rows.append({
                "fold_id": fold_idx,
                "fold_train_start": str(times[tr[0]]),
                "fold_train_end": str(times[tr[-1]]),
                "fold_val_start": str(times[va[0]]),
                "fold_val_end": str(times[va[-1]]),
                "train_n": int(len(tr)),
                "val_n": int(len(va)),
                "val_auc": float("nan"),
                "val_clean_rate": float(np.mean(y[va])) if len(va) else float("nan"),
                "feature_importances_top10": "",
            })
            continue
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=min_samples_leaf,
            random_state=RNG_SEED,
            n_jobs=-1,
        )
        rf.fit(X[tr], y[tr])
        proba = rf.predict_proba(X[va])[:, 1]
        auc = float(roc_auc_score(y[va], proba))
        fold_aucs.append(auc)
        imps = pd.Series(rf.feature_importances_, index=feature_cols)
        per_fold_importances[fold_idx] = imps
        top10 = "; ".join(f"{k}:{v:.4f}" for k, v in imps.sort_values(ascending=False).head(10).items())
        fold_rows.append({
            "fold_id": fold_idx,
            "fold_train_start": str(times[tr[0]]),
            "fold_train_end": str(times[tr[-1]]),
            "fold_val_start": str(times[va[0]]),
            "fold_val_end": str(times[va[-1]]),
            "train_n": int(len(tr)),
            "val_n": int(len(va)),
            "val_auc": auc,
            "val_clean_rate": float(np.mean(y[va])),
            "feature_importances_top10": top10,
        })
        pb = pd.DataFrame({"pair": pairs[va], "y": y[va], "proba": proba})
        pb_summary = pb.groupby("pair").agg(
            sum=("y", "sum"), count=("y", "count")
        ).reset_index()
        pb_summary["fold_id"] = fold_idx
        pair_breakdown_per_fold.append(pb_summary)

    rf_full = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=min_samples_leaf,
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    rf_full.fit(X, y)
    proba_full = rf_full.predict_proba(X)[:, 1]
    auc_full = float(roc_auc_score(y, proba_full))
    feat_imps_full = pd.Series(rf_full.feature_importances_, index=feature_cols)

    folds_df = pd.DataFrame(fold_rows)
    summary = {
        "n_splits": n_splits,
        "n_after_dropna": int(len(y)),
        "clean_rate": float(np.mean(y)),
        "mean_fold_auc": float(np.nanmean(fold_aucs)) if fold_aucs else float("nan"),
        "worst_fold_auc": float(np.nanmin(fold_aucs)) if fold_aucs else float("nan"),
        "fold_auc_stdev": float(np.nanstd(fold_aucs, ddof=0)) if fold_aucs else float("nan"),
        "full_data_auc": auc_full,
        "n_features": len(feature_cols),
    }
    pair_breakdown = (
        pd.concat(pair_breakdown_per_fold, ignore_index=True)
        if pair_breakdown_per_fold else pd.DataFrame()
    )
    return folds_df, summary, feat_imps_full.sort_values(ascending=False), pair_breakdown, per_fold_importances


def feature_importance_drift(per_fold: dict[int, pd.Series],
                             feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold importance wide, spearman between consecutive
    folds + F1<->F_last)."""
    if not per_fold:
        return pd.DataFrame(), pd.DataFrame()
    drift = pd.DataFrame({f"fold_{k}": v for k, v in per_fold.items()},
                         index=feature_cols).reset_index().rename(
        columns={"index": "feature"}
    )
    spearman_rows = []
    fold_ids = sorted(per_fold.keys())
    for a, b in zip(fold_ids[:-1], fold_ids[1:]):
        rho, p = sps.spearmanr(per_fold[a], per_fold[b])
        spearman_rows.append({"comparison": f"F{a}<->F{b}", "spearman_rho": float(rho), "p_value": float(p)})
    if len(fold_ids) >= 2:
        a, b = fold_ids[0], fold_ids[-1]
        rho, p = sps.spearmanr(per_fold[a], per_fold[b])
        spearman_rows.append({"comparison": f"F{a}<->F{b}", "spearman_rho": float(rho), "p_value": float(p)})
    spearman_df = pd.DataFrame(spearman_rows)
    return drift, spearman_df


# ---------------------------------------------------------------- spot-check

def spot_check_no_lookahead(rows_df: pd.DataFrame, pair_dfs: dict,
                            d1_data: dict, h4_data: dict | None,
                            w_data: dict, h1_data: dict | None,
                            tf: str, n: int = 10, seed: int = RNG_SEED) -> list[dict]:
    """Manually recompute key features at random labelled bars from raw
    OHLC <= t (signal TF + each cross-TF source), and compare to stored
    values. Cross-TF lag boundaries are sampled directly.
    """
    rng = np.random.default_rng(seed)
    checks = []
    idx = rng.choice(len(rows_df), size=min(n, len(rows_df)), replace=False)
    for i in idx:
        row = rows_df.iloc[int(i)]
        pair = row["pair"]
        t = pd.Timestamp(row["time"])
        df = pair_dfs[pair]
        loc = int(df.index[df["time"] == t][0])
        h = df["high"].values[: loc + 1]
        l = df["low"].values[: loc + 1]
        c = df["close"].values[: loc + 1]
        if loc + 1 < 14:
            continue
        atr_recomp = float(compute_atr(h, l, c, 14)[0][-1])

        # D1 anchor — strict prior calendar day
        d1_df = d1_data[pair]
        d1_dates = d1_df["time"].dt.normalize().values
        d1_idx = int(np.searchsorted(d1_dates, np.datetime64(t.normalize()), side="left") - 1)
        if d1_idx >= 0 and d1_idx >= 200:  # need EMA-200 warmup
            d1_c = d1_df["close"].values[: d1_idx + 1]
            d1_ema50 = ema_adjust_false(d1_c, 50)
            d1_ema50_slope = float((d1_ema50[-1] - d1_ema50[-51]) / d1_ema50[-51])
            # d1_ema_stack_up
            d1_ema20 = ema_adjust_false(d1_c, 20)
            d1_ema200 = ema_adjust_false(d1_c, 200)
            d1_stack = bool((d1_ema20[-1] > d1_ema50[-1]) and (d1_ema50[-1] > d1_ema200[-1]))
        else:
            d1_ema50_slope = float("nan")
            d1_stack = None

        # W anchor — most recent CLOSED weekly bar (B + 7d <= T)
        w_df = w_data[pair]
        w_times = w_df["time"].values
        target_w = np.datetime64(t) - np.timedelta64(7, "D")
        w_idx = int(np.searchsorted(w_times, target_w, side="right") - 1)
        if w_idx >= 0 and w_idx >= 40:
            w_c = w_df["close"].values[: w_idx + 1]
            w_ema20 = ema_adjust_false(w_c, 20)
            w_ema20_slope = float((w_ema20[-1] - w_ema20[-21]) / w_ema20[-21])
        else:
            w_ema20_slope = float("nan")

        # H4 anchor (for 1H signal TF) — most recent CLOSED H4 bar
        h4_stack_recomp = None
        if tf == "1h" and h4_data is not None:
            h4_df = h4_data[pair]
            h4_times = h4_df["time"].values
            target_h4 = np.datetime64(t) - np.timedelta64(4, "h")
            h4_idx = int(np.searchsorted(h4_times, target_h4, side="right") - 1)
            if h4_idx >= 200:
                h4_c = h4_df["close"].values[: h4_idx + 1]
                h4_ema20 = ema_adjust_false(h4_c, 20)
                h4_ema50 = ema_adjust_false(h4_c, 50)
                h4_ema200 = ema_adjust_false(h4_c, 200)
                h4_stack_recomp = bool(
                    (h4_ema20[-1] > h4_ema50[-1]) and (h4_ema50[-1] > h4_ema200[-1])
                )
        elif tf == "4h":
            # signal bar IS the H4 bar
            c4 = df["close"].values[: loc + 1]
            ema20 = ema_adjust_false(c4, 20)
            ema50 = ema_adjust_false(c4, 50)
            ema200 = ema_adjust_false(c4, 200)
            if len(ema200) >= 200:
                h4_stack_recomp = bool(
                    (ema20[-1] > ema50[-1]) and (ema50[-1] > ema200[-1])
                )

        # H1 anchor (for 4H signal TF) — most recent CLOSED H1 bar
        h1_slope_recomp = None
        if tf == "4h" and h1_data is not None:
            h1_df = h1_data[pair]
            h1_times = h1_df["time"].values
            target_h1 = np.datetime64(t) - np.timedelta64(1, "h")
            h1_idx = int(np.searchsorted(h1_times, target_h1, side="right") - 1)
            if h1_idx >= 100:
                h1_c = h1_df["close"].values[: h1_idx + 1]
                h1_ema50 = ema_adjust_false(h1_c, 50)
                h1_slope_recomp = float((h1_ema50[-1] - h1_ema50[-51]) / h1_ema50[-51])

        checks.append({
            "pair": pair,
            "time": str(t),
            "tf": tf,
            "stored_atr_14": float(row.get("atr_14", np.nan)),
            "recomputed_atr_14": atr_recomp,
            "stored_d1_ema_50_slope_at_entry": float(row.get("d1_ema_50_slope_at_entry", np.nan)),
            "recomputed_d1_ema_50_slope": d1_ema50_slope,
            "stored_d1_ema_stack_up": float(row.get("d1_ema_stack_up", np.nan)),
            "recomputed_d1_ema_stack_up": (None if d1_stack is None else float(d1_stack)),
            "stored_w_ema_20_slope": float(row.get("w_ema_20_slope", np.nan)),
            "recomputed_w_ema_20_slope": w_ema20_slope,
            "stored_h4_ema_stack_up": float(row.get("h4_ema_stack_up", np.nan)),
            "recomputed_h4_ema_stack_up": (None if h4_stack_recomp is None else float(h4_stack_recomp)),
            "stored_h1_ema_50_slope_at_entry": float(row.get("h1_ema_50_slope_at_entry", np.nan)) if "h1_ema_50_slope_at_entry" in row.index else None,
            "recomputed_h1_ema_50_slope": h1_slope_recomp,
        })
    return checks


# ---------------------------------------------------------------- pipeline

def run_timeframe(tf: str) -> dict:
    cfg = TF_CONFIG[tf]
    W = cfg["forward_window"]
    out_dir = _OUT / f"timeframe_{tf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tf}] loading data for {len(PAIRS)} pairs ...", flush=True)
    pair_dfs = {p: load_pair(tf, p) for p in PAIRS}
    d1_data = {p: load_pair("d1", p) for p in PAIRS}
    w_data = {p: load_weekly(p) for p in PAIRS}
    h4_data = None
    h1_data = None
    if tf == "1h":
        h4_data = {p: load_pair("4h", p) for p in PAIRS}
    elif tf == "4h":
        h4_data = pair_dfs  # signal TF IS H4
        h1_data = {p: load_pair("1h", p) for p in PAIRS}

    print(f"[{tf}] computing labels (relaxed) + base features per pair ...", flush=True)
    pair_label_frames = []
    pair_feature_frames = {}
    for pair in PAIRS:
        df = pair_dfs[pair]
        atr_14, _ = compute_atr(df["high"].values, df["low"].values,
                                df["close"].values, 14)
        labels = compute_labels_relaxed(
            df["close"].values, df["high"].values, df["low"].values,
            atr_14, df["time"].values, W,
        )
        feats = compute_pair_features(df, tf)
        feats["pair"] = pair
        label_frame = pd.DataFrame({
            "pair": pair,
            "time": df["time"].values,
            "clean_move_relaxed": labels.clean_move_relaxed,
            "clean_score_relaxed": labels.clean_score_relaxed,
            "mono_pre_peak": labels.mono_pre_peak,
            "mfe_max_R": labels.mfe_max_R,
            "mae_pre_peak_R": labels.mae_pre_peak_R,
            "reached_0_5R_pre_peak": labels.reached_0_5R_pre_peak,
            "reached_1R_pre_peak": labels.reached_1R_pre_peak,
            "peak_mfe_bar": labels.peak_mfe_bar,
            "has_label": labels.has_label,
        })
        pair_label_frames.append(label_frame)
        pair_feature_frames[pair] = feats

    print(f"[{tf}] attaching existing B1-B4 cross-TF anchors ...", flush=True)
    attach_d1_anchor(pair_feature_frames, d1_data)
    if tf == "1h" and h4_data is not None:
        attach_h4_anchor(pair_feature_frames, h4_data)

    print(f"[{tf}] attaching v2 cross-TF extras (D1 panel + H4 stack + H1/W anchors) ...", flush=True)
    attach_d1_extra(pair_feature_frames, d1_data)
    if tf == "1h":
        attach_h4_extra_for_1h(pair_feature_frames, h4_data)
    elif tf == "4h":
        attach_h4_extra_for_4h(pair_feature_frames, h4_data)
        attach_h1_anchor_for_4h(pair_feature_frames, h1_data)
    attach_w_anchor(pair_feature_frames, w_data)

    print(f"[{tf}] computing cross-TF agreement features ...", flush=True)
    compute_cross_tf_agreement(pair_feature_frames, tf)

    # DXY proxy on signal TF (reused B1-B4 attach)
    close_dict = {pair: pair_dfs[pair].set_index("time")["close"] for pair in PAIRS}
    close_panel = pd.DataFrame(close_dict)
    attach_dxy_proxy(pair_feature_frames, close_panel)

    # Concat + merge labels
    all_feats = pd.concat(pair_feature_frames.values(), ignore_index=True)
    all_labels = pd.concat(pair_label_frames, ignore_index=True)
    merged = all_feats.merge(all_labels, on=["pair", "time"], how="inner")
    merged = merged[merged["has_label"]].copy()
    merged = merged[(merged["time"] >= START_DATE) & (merged["time"] <= END_DATE)]
    merged = merged.sort_values(["time", "pair"]).reset_index(drop=True)
    print(f"[{tf}] labelled rows after date filter: {len(merged):,}", flush=True)

    label_cols = ["pair", "time", "clean_move_relaxed", "clean_score_relaxed",
                  "mono_pre_peak", "mfe_max_R", "mae_pre_peak_R",
                  "reached_0_5R_pre_peak", "reached_1R_pre_peak", "peak_mfe_bar"]
    merged[label_cols].to_csv(out_dir / "labels_relaxed.csv", index=False)

    feature_cols_all = [
        c for c in merged.columns
        if c not in (label_cols + ["has_label"])
    ]
    feature_cols_no_meta = [c for c in feature_cols_all if c not in ("pair", "time")]
    merged[["pair", "time"] + feature_cols_no_meta].to_csv(
        out_dir / "features_expanded.csv", index=False
    )

    # ----- B3 univariate
    print(f"[{tf}] B3 univariate over {len(feature_cols_no_meta)} features ...", flush=True)
    b3 = b3_univariate(merged, merged["clean_move_relaxed"].values, feature_cols_no_meta)
    b3.to_csv(out_dir / "b3_univariate_expanded.csv", index=False)

    # ----- B4 multivariate on FULL feature set
    print(f"[{tf}] B4 multivariate (FULL set, no top-15 filter) ...", flush=True)
    folds_df, summary, imps_full, pair_breakdown, per_fold_imps = b4_multivariate_full(
        merged, merged["clean_move_relaxed"].values, feature_cols_no_meta,
        merged["pair"].values, merged["time"].values,
        n_splits=7, min_samples_leaf=cfg["rf_min_samples_leaf"],
    )
    folds_df.to_csv(out_dir / "b4_multivariate_expanded.csv", index=False)
    if not pair_breakdown.empty:
        pair_breakdown.to_csv(out_dir / "b4_per_pair_per_fold.csv", index=False)
    imps_full.rename("importance").to_csv(
        out_dir / "b4_feature_importances_full.csv", header=True, index_label="feature"
    )
    with open(out_dir / "b4_summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write(f"feature_set_size: {len(feature_cols_no_meta)}\n")
        f.write(f"feature_cols: {feature_cols_no_meta}\n")

    # ----- importance drift
    print(f"[{tf}] cross-fold importance drift ...", flush=True)
    drift_df, spearman_df = feature_importance_drift(per_fold_imps, feature_cols_no_meta)
    drift_df.to_csv(out_dir / "feature_importance_drift.csv", index=False)
    spearman_df.to_csv(out_dir / "feature_importance_spearman.csv", index=False)

    # ----- label distribution
    print(f"[{tf}] label distribution ...", flush=True)
    base_rate = float(merged["clean_move_relaxed"].mean())
    score_dist = merged["clean_score_relaxed"].value_counts().sort_index().to_dict()
    pair_counts = merged.groupby("pair")["clean_move_relaxed"].agg(["sum", "count"]).reset_index()
    pair_counts["clean_rate"] = pair_counts["sum"] / pair_counts["count"]
    pair_counts = pair_counts.sort_values("sum", ascending=False)
    total_pos = int(merged["clean_move_relaxed"].sum())
    pair_counts["share_of_clean"] = pair_counts["sum"] / max(total_pos, 1)
    pair_flag = pair_counts[pair_counts["share_of_clean"] > 0.40]
    with open(out_dir / "label_distribution.txt", "w") as f:
        f.write(f"timeframe: {tf}\n")
        f.write(f"forward_window_bars: {W}\n")
        f.write(f"n_rows: {len(merged)}\n")
        f.write(f"clean_move_relaxed_count: {total_pos}\n")
        f.write(f"clean_move_relaxed_rate: {base_rate:.6f}\n")
        f.write(f"clean_score_relaxed_distribution: {score_dist}\n")
        f.write("per_pair_clean_counts:\n")
        f.write(pair_counts.to_string(index=False))
        f.write("\n")
        if not pair_flag.empty:
            f.write("\nWARNING: pair(s) contribute > 40% of relaxed clean labels:\n")
            f.write(pair_flag.to_string(index=False))
            f.write("\n")
        else:
            f.write("\nOK: no single pair contributes > 40% of relaxed clean labels.\n")
        # Flag if base rate outside 15-70% extreme range
        if base_rate < 0.15:
            f.write(f"\nFLAG: relaxed clean base rate {base_rate:.4f} < 0.15 — review label calibration.\n")
        elif base_rate > 0.70:
            f.write(f"\nFLAG: relaxed clean base rate {base_rate:.4f} > 0.70 — review label calibration.\n")
        else:
            f.write(f"\nOK: relaxed clean base rate {base_rate:.4f} within 15-70% range.\n")
        # Note expected band per dispatch
        if base_rate < 0.30 or base_rate > 0.50:
            f.write(f"NOTE: relaxed clean base rate {base_rate:.4f} outside dispatch's 30-50% expected band; not a blocker.\n")

    # ----- spot-check (10 bars; recompute cross-TF features by hand)
    print(f"[{tf}] no-lookahead spot-checks (10 bars) ...", flush=True)
    spot = spot_check_no_lookahead(merged, pair_dfs, d1_data, h4_data, w_data, h1_data,
                                   tf=tf, n=10)
    with open(out_dir / "lookahead_spot_check.txt", "w") as f:
        f.write(f"timeframe: {tf}\n")
        f.write("Per bar: compare stored cross-TF features vs manual recompute from raw OHLC<=t at each source TF.\n")
        f.write("Cross-TF lag rules:\n")
        f.write("  D1 anchor: largest D1 bar with d1_date < t.date() (strict prior calendar day)\n")
        f.write("  H4 anchor: largest H4 bar with h4_time + 4h <= t (most recent CLOSED H4)\n")
        f.write("  H1 anchor: largest H1 bar with h1_time + 1h <= t (most recent CLOSED H1, used at 4H signal TF)\n")
        f.write("  W anchor:  largest W bar with w_time + 7d <= t (most recent CLOSED W)\n\n")
        n_pass = 0
        n_total = 0
        for s in spot:
            f.write(str(s) + "\n")
            for stored_key, recomp_key in [
                ("stored_atr_14", "recomputed_atr_14"),
                ("stored_d1_ema_50_slope_at_entry", "recomputed_d1_ema_50_slope"),
                ("stored_d1_ema_stack_up", "recomputed_d1_ema_stack_up"),
                ("stored_w_ema_20_slope", "recomputed_w_ema_20_slope"),
                ("stored_h4_ema_stack_up", "recomputed_h4_ema_stack_up"),
            ]:
                sv = s[stored_key]
                rv = s[recomp_key]
                if rv is None:
                    continue
                n_total += 1
                if sv is None or (isinstance(sv, float) and np.isnan(sv) and np.isnan(rv)):
                    n_pass += 1
                    continue
                try:
                    if np.isfinite(sv) and np.isfinite(rv) and abs(sv - rv) < 1e-9:
                        n_pass += 1
                except Exception:
                    pass
        f.write(f"\nSpot-check: {n_pass}/{n_total} stored-vs-recomputed matches (tol 1e-9).\n")

    print(f"[{tf}] DONE — worst-fold AUC {summary['worst_fold_auc']:.4f} "
          f"mean {summary['mean_fold_auc']:.4f} clean-rate {base_rate:.4f} "
          f"n_features={summary['n_features']}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["1h", "4h", "all"], default="all")
    args = ap.parse_args()
    tfs = ["1h", "4h"] if args.tf == "all" else [args.tf]
    summaries = {}
    for tf in tfs:
        summaries[tf] = run_timeframe(tf)
    print("\n=== summary ===")
    for tf, s in summaries.items():
        print(f"  {tf}: worst_fold_auc={s['worst_fold_auc']:.4f} "
              f"mean={s['mean_fold_auc']:.4f} clean_rate={s['clean_rate']:.4f} "
              f"n_features={s['n_features']} n={s['n_after_dropna']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
