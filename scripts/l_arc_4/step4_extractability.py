"""Arc 4 — Step 4 extractability + artefact production.

L_ARC_PROTOCOL v2.1.1 §8 / §12. Per surviving cluster from Step 3
(capturability_pass_list.csv: cluster 1 SL=3.0×ATR; cluster 3 SL=4.0×ATR),
train Pipeline E (entry-time) and Pipeline D1 (deferred-identification) RF
classifiers via the §8 Step A → B → C filter-selection cascade, sweep
threshold to maximise precision subject to recall ≥ 0.60, and ship artefacts
(joblib + YAML) for any pipeline clearing its AUC gate.

Outputs (results/l_arc_4/step4/):
  - predictability_angle_E.csv
  - predictability_angle_D1.csv
  - feature_importances_cluster_{1,3}_E.csv
  - feature_importances_cluster_{1,3}_D1.csv (if D1 passes)
  - extractability_pass_list.csv
  - cluster_{1,3}_E_classifier.joblib + cluster_{1,3}_E_filter.yaml
    (if E clears the AUC gate)
  - cluster_{1,3}_D1_classifier.joblib + cluster_{1,3}_D1_policy.yaml
    (if D1 clears the AUC gate at its smallest valid t)
  - stacking_log.csv (Step C combinations evaluated; budget ≤ 30 per arc)
  - step4_diagnostics.md

Determinism:
  - random_state=42 throughout (StratifiedKFold + RF + LR).
  - Two-run byte-identical for all written files (CSV + joblib + YAML).
  - Both run sha256s logged.

Usage:
  py scripts/l_arc_4/step4_extractability.py -c configs/l_arc_4.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ============================================================================
# Constants
# ============================================================================

RNDSTATE: int = 42
N_CV: int = 5

RF_HP = dict(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    random_state=RNDSTATE,
    n_jobs=-1,  # sklearn RF with random_state is deterministic regardless of n_jobs
)
LR_HP = dict(
    max_iter=2000,
    random_state=RNDSTATE,
    solver="lbfgs",
)

THRESHOLD_GRID: Tuple[float, ...] = (0.40, 0.50, 0.60, 0.70)
RECALL_FLOOR: float = 0.60

AUC_GATE_E: float = 0.65
AUC_GATE_D1: float = 0.60
D1_T_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 5, 10)
D1_EXCL_MAX: float = 0.30

# Step C stacking budget across this arc (both clusters × both pipelines).
STACKING_BUDGET: int = 30
STEP_B_FORWARD_MAX_FEATS: int = 15
STEP_B_FORWARD_MIN_GAIN: float = 0.005

# Cluster labels supplied by the chat (Step 4 prompt).
CLUSTER_LABELS: Dict[int, str] = {
    1: "stepwise_pullback_extended",
    3: "stepwise_slow_climber",
}


# ============================================================================
# Entry-feature computation per pair (8 base + arc-specific from catalogue)
# ============================================================================


def _wilder_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _wilder_rsi(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=float)
    delta = np.diff(close, prepend=close[0])
    delta[0] = 0.0
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.full(n, np.nan, dtype=float)
    avg_l = np.full(n, np.nan, dtype=float)
    avg_g[period] = float(np.mean(gain[1 : period + 1]))
    avg_l[period] = float(np.mean(loss[1 : period + 1]))
    for i in range(period + 1, n):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period
    rs = np.divide(
        avg_g,
        avg_l,
        out=np.full_like(avg_g, np.nan),
        where=(avg_l > 0) & ~np.isnan(avg_g) & ~np.isnan(avg_l),
    )
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Where avg_l == 0 (no losses), RSI = 100.
    rsi = np.where((avg_l == 0) & (avg_g > 0), 100.0, rsi)
    return rsi


def _compute_entry_features_for_pair(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Vectorise the 8 base + 24 arc-specific entry features per 1H bar.

    The output DataFrame is keyed by the bar's timestamp ('date' column) —
    one row per 1H bar. Caller looks up by signal_time = bar_N close.

    All features computed strictly from bars ≤ N (no lookahead).
    """
    df = df_1h.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    open_ = df["open"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()

    rng = high - low
    rng_safe = np.where(rng > 0, rng, np.nan)
    body = np.abs(close - open_)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    out = pd.DataFrame({"date": df["date"].to_numpy()})

    # --- 8 base features ---
    out["body_to_range_ratio"] = body / rng_safe
    out["upper_wick_ratio"] = upper_wick / rng_safe
    out["lower_wick_ratio"] = lower_wick / rng_safe

    atr_14 = _wilder_atr(df, 14)
    atr_safe_14 = np.where(atr_14 > 0, atr_14, np.nan)
    out["range_to_atr_14"] = rng / atr_safe_14

    close_shift_5 = np.roll(close, 5)
    close_shift_5[:5] = np.nan
    close_shift_20 = np.roll(close, 20)
    close_shift_20[:20] = np.nan
    out["ret_5bar_atr"] = (close - close_shift_5) / atr_safe_14
    out["ret_20bar_atr"] = (close - close_shift_20) / atr_safe_14

    # pos_in_20bar_range: includes current bar.
    high_20 = pd.Series(high).rolling(20, min_periods=20).max().to_numpy()
    low_20 = pd.Series(low).rolling(20, min_periods=20).min().to_numpy()
    rng_20 = high_20 - low_20
    rng_20_safe = np.where(rng_20 > 0, rng_20, np.nan)
    out["pos_in_20bar_range"] = (close - low_20) / rng_20_safe

    out["rsi_14"] = _wilder_rsi(close, 14)

    # --- Arc-specific (24) ---
    atr_7 = _wilder_atr(df, 7)
    atr_28 = _wilder_atr(df, 28)
    out["atr_7_to_atr_14_ratio"] = atr_7 / atr_safe_14
    out["atr_28_to_atr_14_ratio"] = atr_28 / atr_safe_14

    out["rsi_7"] = _wilder_rsi(close, 7)
    out["rsi_28"] = _wilder_rsi(close, 28)

    sma_20 = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()
    std_20 = pd.Series(close).rolling(20, min_periods=20).std(ddof=0).to_numpy()
    std_20_safe = np.where(std_20 > 0, std_20, np.nan)
    out["bb_pos_20"] = (close - sma_20) / std_20_safe

    log_close = np.log(close)
    log_ret = np.diff(log_close, prepend=log_close[0])
    log_ret[0] = 0.0
    out["realized_vol_20"] = pd.Series(log_ret).rolling(20, min_periods=20).std(ddof=0).to_numpy()
    out["realized_vol_60"] = pd.Series(log_ret).rolling(60, min_periods=60).std(ddof=0).to_numpy()

    # bars_since_high_20 / low_20: argmax/argmin position within the 20-bar window
    # ending at bar N. window-1 (=19) - argmax_idx => bars since.
    # Vectorised via stride-trick rolling windows.
    def _bars_since_extremum(arr: np.ndarray, op: str, window: int = 20) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view
        nb = len(arr)
        out_arr = np.full(nb, np.nan, dtype=float)
        if nb < window:
            return out_arr
        windows = sliding_window_view(arr, window)
        if op == "max":
            idx = np.argmax(windows, axis=1)
        else:
            idx = np.argmin(windows, axis=1)
        out_arr[window - 1 :] = (window - 1) - idx
        return out_arr

    out["bars_since_high_20"] = _bars_since_extremum(high, "max")
    out["bars_since_low_20"] = _bars_since_extremum(low, "min")

    out["dist_to_swing_high_20_atr"] = (high_20 - close) / atr_safe_14
    out["dist_to_swing_low_20_atr"] = (close - low_20) / atr_safe_14

    close_shift_50 = np.roll(close, 50)
    close_shift_50[:50] = np.nan
    close_shift_100 = np.roll(close, 100)
    close_shift_100[:100] = np.nan
    out["ret_50bar_atr"] = (close - close_shift_50) / atr_safe_14
    out["ret_100bar_atr"] = (close - close_shift_100) / atr_safe_14

    # 5-bar microstructure averages.
    btr = body / rng_safe
    upr = upper_wick / rng_safe
    lwr = lower_wick / rng_safe
    out["body_to_range_mean_5"] = pd.Series(btr).rolling(5, min_periods=5).mean().to_numpy()
    out["upper_wick_mean_5"] = pd.Series(upr).rolling(5, min_periods=5).mean().to_numpy()
    out["lower_wick_mean_5"] = pd.Series(lwr).rolling(5, min_periods=5).mean().to_numpy()

    # close_below_open_run_length: current streak of close < open ending at N.
    is_neg = (close < open_).astype(int)
    # Compute streak: cumulative count within consecutive 1-blocks.
    grp = (pd.Series(is_neg) != pd.Series(is_neg).shift(1)).cumsum()
    streak = pd.Series(is_neg).groupby(grp).cumcount() + 1
    streak = streak.where(pd.Series(is_neg) == 1, 0).to_numpy()
    out["close_below_open_run_length"] = streak.astype(float)

    # bar_range_pctile_100: trailing-100 (excluding bar N) percentile rank of
    # current bar's range. Matches the signal-class p90 threshold mechanism.
    # For bar i (i >= window), output = (count of bars in cur[i-window..i-1]
    # with value < cur[i]) / (count of valid bars in that window).
    # Vectorised via stride-trick rolling windows for performance.
    def _pctile_rank(s: pd.Series, window: int) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view
        cur = s.to_numpy(dtype=float)
        n_bars = len(cur)
        rank = np.full(n_bars, np.nan, dtype=float)
        if n_bars < window + 1:
            return rank
        # shifted_prev[i] = cur[i-1]; sliding_window_view of length `window`
        # gives row k = shifted_prev[k..k+window-1] = cur[k-1..k+window-2].
        shifted_prev = np.concatenate([[np.nan], cur[:-1]])
        full = sliding_window_view(shifted_prev, window)
        # For output index i = k + window, we need windows[k+1] which is
        # shifted_prev[k+1..k+window] = cur[k..k+window-1] = bars i-window..i-1.
        windows = full[1:]
        cur_aligned = cur[window:]
        valid = ~np.isnan(windows)
        less_than = (windows < cur_aligned[:, None]) & valid
        counts_less = less_than.sum(axis=1)
        counts_valid = valid.sum(axis=1)
        pct = np.where(counts_valid > 0, counts_less / counts_valid, np.nan)
        rank[window:] = pct
        return rank

    out["bar_range_pctile_100"] = _pctile_rank(pd.Series(rng), 100)

    # Session / calendar features (derived from bar timestamp; lookahead-safe).
    ts = pd.to_datetime(df["date"]).dt
    hour = ts.hour.to_numpy()
    weekday = ts.weekday.to_numpy()
    out["session_asia"] = ((hour >= 0) & (hour < 7)).astype(int)
    out["session_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    out["session_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    out["hour_of_day"] = hour.astype(int)
    out["day_of_week"] = weekday.astype(int)

    return out


def _load_pair_1h(pair: str, data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{pair}.csv")
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def build_entry_feature_matrix(
    trades: pd.DataFrame,
    data_dir: Path,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """For each trade, look up its signal_time row in the per-pair entry-feature
    table. Returns (feature_matrix, audit_info).
    """
    trades = trades.copy()
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], errors="coerce")
    pairs = sorted(trades["pair"].unique().tolist())
    rows: List[pd.DataFrame] = []
    coverage_info: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    for pair in pairs:
        df_1h = _load_pair_1h(pair, data_dir)
        feats = _compute_entry_features_for_pair(df_1h)
        feats = feats.set_index("date")
        sub = trades[trades["pair"] == pair].copy()
        # Look up each signal_time. signal_time is the bar N timestamp, which
        # exists in the bar feed.
        looked = feats.reindex(sub["signal_time"].to_numpy())
        looked.insert(0, "trade_id", sub["trade_id"].to_numpy())
        looked.insert(1, "pair", pair)
        rows.append(looked.reset_index(drop=True))
        coverage_info[pair] = {
            "n_trades": int(len(sub)),
            "n_signal_time_in_bars": int(looked.notna().any(axis=1).sum()),
        }
        print(
            f"[l_arc_4 step4] entry features {pair}: {len(sub)} trades, "
            f"{time.time() - t0:.1f}s",
            file=sys.stderr,
        )
    big = pd.concat(rows, axis=0, ignore_index=True)
    big = big.sort_values("trade_id").reset_index(drop=True)

    # Verify alignment with trades.
    if list(big["trade_id"].to_numpy(dtype=int)) != list(
        trades.sort_values("trade_id")["trade_id"].to_numpy(dtype=int)
    ):
        raise ValueError("trade_id ordering mismatch after entry feature build")

    keep = ["trade_id", "pair"] + feature_names
    big = big[keep]
    return big, {"coverage": coverage_info}


# ============================================================================
# D1 path-so-far feature computation
# ============================================================================


def _compute_d1_features_at_t(
    cluster_paths: np.ndarray,
    entry_prices: np.ndarray,
    r_per_trade: np.ndarray,
    t: int,
) -> np.ndarray:
    """Vectorised D1 path-so-far features at observation point t.

    cluster_paths shape: (n_trades, PATH_BARS, 4) — (open, high, low, close).
    Returns: (n_eligible, 7) feature array where n_eligible may be < n_trades
    (caller filters by bars_held >= t).

    Features (in column order):
      0 close_r_at_t
      1 mfe_so_far_r_at_t
      2 mae_so_far_r_at_t
      3 bars_in_profit_at_t
      4 local_peaks_so_far_at_t
      5 monotonicity_so_far_at_t
      6 velocity_first_t  (= close_r_at_t / max(t, 1))
    """
    n_trades = cluster_paths.shape[0]
    # Slice to bars 0..t inclusive.
    sliced = cluster_paths[:, : t + 1, :]
    highs = sliced[:, :, 1]
    lows = sliced[:, :, 2]
    closes = sliced[:, :, 3]
    inv_r = 1.0 / r_per_trade
    close_r = (closes - entry_prices[:, None]) * inv_r[:, None]
    high_r = (highs - entry_prices[:, None]) * inv_r[:, None]
    low_r = (lows - entry_prices[:, None]) * inv_r[:, None]

    # close_r_at_t
    close_r_at_t = close_r[:, t]
    # mfe_so_far_r_at_t (max of high_r over 0..t)
    mfe_so_far = np.max(high_r, axis=1)
    # mae_so_far_r_at_t (min of low_r over 0..t)
    mae_so_far = np.min(low_r, axis=1)
    # bars_in_profit_at_t
    bars_in_profit = np.sum(close_r > 0, axis=1).astype(float)
    # local_peaks_so_far_at_t (count of bars where mfe > prev bar's mfe)
    # mfe_so_far is monotone non-decreasing — use cumulative max.
    cummax_high_r = np.maximum.accumulate(high_r, axis=1)
    if t >= 1:
        peak_increments = (cummax_high_r[:, 1:] > cummax_high_r[:, :-1]).astype(float)
        local_peaks = np.sum(peak_increments, axis=1)
    else:
        local_peaks = np.zeros(n_trades, dtype=float)
    # monotonicity_so_far_at_t (per-trade loop — same edge cases as Step 2/3)
    mono = np.zeros(n_trades, dtype=float)
    for i in range(n_trades):
        cl = close_r[i]
        in_profit = cl > 0.0
        n_ip = int(in_profit.sum())
        if n_ip <= 1:
            mono[i] = 0.0
            continue
        ipc = cl[in_profit]
        gte = ipc[1:] >= ipc[:-1]
        mono[i] = float(gte.sum() / gte.size)
    velocity = close_r_at_t / max(t, 1)

    return np.column_stack(
        [
            close_r_at_t,
            mfe_so_far,
            mae_so_far,
            bars_in_profit,
            local_peaks,
            mono,
            velocity,
        ]
    )


# ============================================================================
# Model evaluation helpers
# ============================================================================


def _stratified_kfold() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RNDSTATE)


def _eval_rf(X: np.ndarray, y: np.ndarray) -> Tuple[float, List[float]]:
    """Return (mean_auc, per_fold_auc) using 5-fold CV."""
    skf = _stratified_kfold()
    model = RandomForestClassifier(**RF_HP)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=1)
    return float(np.mean(scores)), [float(s) for s in scores]


def _eval_lr(X: np.ndarray, y: np.ndarray) -> Tuple[float, List[float]]:
    skf = _stratified_kfold()
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(**LR_HP))])
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=1)
    return float(np.mean(scores)), [float(s) for s in scores]


def _rf_feature_importances(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[Tuple[str, float]]:
    model = RandomForestClassifier(**RF_HP)
    model.fit(X, y)
    imp = model.feature_importances_
    out = list(zip(feature_names, imp))
    out.sort(key=lambda kv: -kv[1])
    return out


def _univariate_auc(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Per-feature univariate AUC, |AUC| > 0.5 retained.

    AUC is computed against the binary cluster_membership label by feeding the
    raw feature as the prediction score. NaNs filled with feature median.
    """
    out: List[Tuple[str, float]] = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        col = np.where(np.isnan(col), np.nanmedian(col), col)
        try:
            a = roc_auc_score(y, col)
        except ValueError:
            a = 0.5
        # Take max(a, 1 - a) — feature signal magnitude.
        out.append((name, max(float(a), 1.0 - float(a))))
    out.sort(key=lambda kv: -kv[1])
    return out


# ============================================================================
# Step A / B / C cascade per pipeline per cluster
# ============================================================================


@dataclass
class FilterSelectionResult:
    step: str                  # "A" | "B" | "C" | "FAIL"
    feature_subset: List[str]
    rf_auc_mean: float
    rf_auc_folds: List[float]
    lr_auc_mean: float
    lr_auc_folds: List[float]
    importances: List[Tuple[str, float]]
    step_b_paths_tried: int
    step_c_combinations: int
    notes: str
    passes_gate: bool


def _step_b_subsets(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    gate: float,
) -> Tuple[Optional[List[str]], Optional[float], Optional[List[float]], int, str]:
    """Try top-5 / top-10 / top-15 by RF importance + forward selection.

    Returns (passing_subset_names, mean_auc, per_fold_auc, paths_tried, notes).
    Returns (None, None, None, paths_tried, notes) if no subset clears the gate.
    """
    importances = _rf_feature_importances(X, y, feature_names)
    paths_tried = 0
    notes_log: List[str] = []
    # Subsets by RF importance.
    for size in (5, 10, 15):
        if size > len(feature_names):
            continue
        sub_names = [n for n, _ in importances[:size]]
        sub_idx = [feature_names.index(n) for n in sub_names]
        sub_auc_mean, sub_auc_folds = _eval_rf(X[:, sub_idx], y)
        paths_tried += 1
        notes_log.append(f"importance-top-{size}: AUC={sub_auc_mean:.4f}")
        if sub_auc_mean >= gate:
            return (sub_names, sub_auc_mean, sub_auc_folds, paths_tried, "; ".join(notes_log))
    # Forward selection from highest univariate-AUC seed.
    uni = _univariate_auc(X, y, feature_names)
    selected: List[str] = []
    selected_idx: List[int] = []
    best_auc = -np.inf
    best_folds: List[float] = []
    seed_name = uni[0][0]
    selected.append(seed_name)
    selected_idx.append(feature_names.index(seed_name))
    seed_auc, seed_folds = _eval_rf(X[:, selected_idx], y)
    paths_tried += 1
    best_auc = seed_auc
    best_folds = seed_folds
    notes_log.append(f"forward seed='{seed_name}' AUC={seed_auc:.4f}")
    if best_auc >= gate:
        return (list(selected), best_auc, best_folds, paths_tried, "; ".join(notes_log))
    # Greedy: try each remaining feature, keep if improvement ≥ MIN_GAIN.
    remaining_uni = [n for n, _ in uni if n not in selected]
    for cand in remaining_uni:
        if len(selected) >= STEP_B_FORWARD_MAX_FEATS:
            break
        cand_idx = feature_names.index(cand)
        sel_with_cand_idx = selected_idx + [cand_idx]
        sub_auc, sub_folds = _eval_rf(X[:, sel_with_cand_idx], y)
        paths_tried += 1
        if sub_auc > best_auc + STEP_B_FORWARD_MIN_GAIN:
            selected.append(cand)
            selected_idx.append(cand_idx)
            best_auc = sub_auc
            best_folds = sub_folds
            notes_log.append(f"forward add='{cand}' AUC={sub_auc:.4f}")
            if best_auc >= gate:
                return (list(selected), best_auc, best_folds, paths_tried, "; ".join(notes_log))
    notes_log.append(f"forward final size={len(selected)} AUC={best_auc:.4f} (gate not cleared)")
    return (None, None, None, paths_tried, "; ".join(notes_log))


def _step_c_stacking(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    gate: float,
    remaining_budget: int,
    stacking_log: List[Dict[str, Any]],
    cluster_id: int,
    pipeline: str,
) -> Tuple[Optional[List[List[str]]], Optional[float], int]:
    """Try pairs of feature subsets where both classifiers must admit to pass.

    Combined AUC measured via probabilistic agreement: P_combined = P_a × P_b
    (probabilistic intersection — both classifiers must rate the trade highly).
    Returns (pair_of_subsets, mean_auc, combinations_used).
    """
    importances = _rf_feature_importances(X, y, feature_names)
    importance_subsets = [
        ("top_5", [n for n, _ in importances[:5]]),
        ("top_10", [n for n, _ in importances[:10]]),
        ("top_15", [n for n, _ in importances[:15]]),
        ("all", list(feature_names)),
    ]
    # Build "disjoint" candidates: even / odd by importance rank.
    even_imp = [n for i, (n, _) in enumerate(importances) if i % 2 == 0][:10]
    odd_imp = [n for i, (n, _) in enumerate(importances) if i % 2 == 1][:10]
    importance_subsets.append(("even_imp_top10", even_imp))
    importance_subsets.append(("odd_imp_top10", odd_imp))
    # Combinations: pairs of subsets (with replacement disallowed for parsimony).
    used = 0
    best_pair: Optional[List[List[str]]] = None
    best_combined_auc: float = -np.inf
    skf = _stratified_kfold()
    for i, (name_a, sub_a) in enumerate(importance_subsets):
        for j, (name_b, sub_b) in enumerate(importance_subsets):
            if i >= j:
                continue
            if used >= remaining_budget:
                break
            idx_a = [feature_names.index(n) for n in sub_a]
            idx_b = [feature_names.index(n) for n in sub_b]
            # Generate out-of-fold probabilities from each subset.
            model_a = RandomForestClassifier(**RF_HP)
            model_b = RandomForestClassifier(**RF_HP)
            oof_a = cross_val_predict(
                model_a, X[:, idx_a], y, cv=skf, method="predict_proba", n_jobs=1
            )[:, 1]
            oof_b = cross_val_predict(
                model_b, X[:, idx_b], y, cv=skf, method="predict_proba", n_jobs=1
            )[:, 1]
            combined = oof_a * oof_b
            try:
                a_combined = float(roc_auc_score(y, combined))
            except ValueError:
                a_combined = 0.5
            stacking_log.append(
                {
                    "cluster_id": int(cluster_id),
                    "pipeline": pipeline,
                    "subset_a": name_a,
                    "subset_b": name_b,
                    "subset_a_size": len(sub_a),
                    "subset_b_size": len(sub_b),
                    "combined_auc": a_combined,
                    "passes_gate": int(a_combined >= gate),
                }
            )
            used += 1
            if a_combined > best_combined_auc:
                best_combined_auc = a_combined
                if a_combined >= gate:
                    best_pair = [sub_a, sub_b]
                    return (best_pair, a_combined, used)
        if used >= remaining_budget:
            break
    return (None, best_combined_auc if best_combined_auc > -np.inf else None, used)


def run_filter_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    gate: float,
    cluster_id: int,
    pipeline: str,
    stacking_log: List[Dict[str, Any]],
    stacking_budget_remaining: int,
) -> Tuple[FilterSelectionResult, int]:
    """Apply Step A → B → C cascade. Return (result, stacking_budget_remaining_after)."""
    # --- Step A ---
    rf_mean, rf_folds = _eval_rf(X, y)
    lr_mean, lr_folds = _eval_lr(X, y)
    importances = _rf_feature_importances(X, y, feature_names)
    if rf_mean >= gate:
        return (
            FilterSelectionResult(
                step="A",
                feature_subset=list(feature_names),
                rf_auc_mean=rf_mean,
                rf_auc_folds=rf_folds,
                lr_auc_mean=lr_mean,
                lr_auc_folds=lr_folds,
                importances=importances,
                step_b_paths_tried=0,
                step_c_combinations=0,
                notes=f"Step A passes: RF AUC {rf_mean:.4f} ≥ {gate}",
                passes_gate=True,
            ),
            stacking_budget_remaining,
        )

    # --- Step B ---
    sub_names, sub_auc, sub_folds, paths_tried, b_notes = _step_b_subsets(
        X, y, feature_names, gate
    )
    if sub_names is not None and sub_auc is not None and sub_folds is not None:
        sub_idx = [feature_names.index(n) for n in sub_names]
        lr_mean_b, lr_folds_b = _eval_lr(X[:, sub_idx], y)
        imp_b = _rf_feature_importances(X[:, sub_idx], y, sub_names)
        return (
            FilterSelectionResult(
                step="B",
                feature_subset=sub_names,
                rf_auc_mean=sub_auc,
                rf_auc_folds=sub_folds,
                lr_auc_mean=lr_mean_b,
                lr_auc_folds=lr_folds_b,
                importances=imp_b,
                step_b_paths_tried=paths_tried,
                step_c_combinations=0,
                notes=f"Step A failed (RF AUC {rf_mean:.4f} < {gate}); Step B notes: {b_notes}",
                passes_gate=True,
            ),
            stacking_budget_remaining,
        )

    # --- Step C ---
    best_pair, combined_auc, c_combos = _step_c_stacking(
        X, y, feature_names, gate, stacking_budget_remaining, stacking_log, cluster_id, pipeline
    )
    new_budget = stacking_budget_remaining - c_combos
    if best_pair is not None and combined_auc is not None:
        # Locked: stacking pair. Report combined AUC; LR / importance based on
        # the union of the two subsets for diagnostic purposes.
        union = sorted(set(best_pair[0]) | set(best_pair[1]))
        union_idx = [feature_names.index(n) for n in union]
        lr_mean_c, lr_folds_c = _eval_lr(X[:, union_idx], y)
        imp_c = _rf_feature_importances(X[:, union_idx], y, union)
        return (
            FilterSelectionResult(
                step="C",
                feature_subset=union,
                rf_auc_mean=combined_auc,
                rf_auc_folds=[combined_auc] * N_CV,  # combined AUC is single-pass; folds not computed
                lr_auc_mean=lr_mean_c,
                lr_auc_folds=lr_folds_c,
                importances=imp_c,
                step_b_paths_tried=paths_tried,
                step_c_combinations=c_combos,
                notes=(
                    f"Step A failed (RF AUC {rf_mean:.4f} < {gate}); Step B failed "
                    f"({b_notes}); Step C passed with stacked pair: {best_pair[0]} ∩ {best_pair[1]}; "
                    f"combined AUC {combined_auc:.4f}"
                ),
                passes_gate=True,
            ),
            new_budget,
        )

    # All steps failed.
    return (
        FilterSelectionResult(
            step="FAIL",
            feature_subset=list(feature_names),
            rf_auc_mean=rf_mean,
            rf_auc_folds=rf_folds,
            lr_auc_mean=lr_mean,
            lr_auc_folds=lr_folds,
            importances=importances,
            step_b_paths_tried=paths_tried,
            step_c_combinations=c_combos,
            notes=(
                f"Step A failed (RF AUC {rf_mean:.4f} < {gate}); Step B failed ({b_notes}); "
                f"Step C failed (best combined AUC "
                f"{(combined_auc if combined_auc is not None else float('nan')):.4f})"
            ),
            passes_gate=False,
        ),
        new_budget,
    )


# ============================================================================
# Threshold sweep + final classifier
# ============================================================================


@dataclass
class ThresholdSweepResult:
    selected_threshold: float
    selected_precision: float
    selected_recall: float
    full_grid: List[Dict[str, float]]
    used_fallback: bool
    confusion_matrix_tn_fp_fn_tp: Tuple[int, int, int, int]


def _threshold_sweep(X: np.ndarray, y: np.ndarray) -> ThresholdSweepResult:
    """Out-of-fold predict_proba → sweep threshold {0.40, 0.50, 0.60, 0.70} →
    select max precision with recall ≥ 0.60. If no threshold meets recall floor,
    pick highest-recall threshold (flagged as fallback).
    """
    skf = _stratified_kfold()
    model = RandomForestClassifier(**RF_HP)
    oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    grid: List[Dict[str, float]] = []
    for t in THRESHOLD_GRID:
        y_pred = (oof >= t).astype(int)
        if y_pred.sum() == 0:
            prec = 0.0
            rec = 0.0
        else:
            prec = float(precision_score(y, y_pred, zero_division=0))
            rec = float(recall_score(y, y_pred, zero_division=0))
        grid.append({"threshold": float(t), "precision": prec, "recall": rec})
    passing = [g for g in grid if g["recall"] >= RECALL_FLOOR]
    if passing:
        sel = max(passing, key=lambda g: (g["precision"], -g["threshold"]))
        used_fallback = False
    else:
        sel = max(grid, key=lambda g: g["recall"])
        used_fallback = True
    # Confusion matrix at selected threshold.
    y_pred_sel = (oof >= sel["threshold"]).astype(int)
    cm = confusion_matrix(y, y_pred_sel, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return ThresholdSweepResult(
        selected_threshold=float(sel["threshold"]),
        selected_precision=float(sel["precision"]),
        selected_recall=float(sel["recall"]),
        full_grid=grid,
        used_fallback=used_fallback,
        confusion_matrix_tn_fp_fn_tp=(tn, fp, fn, tp),
    )


def _train_final_rf(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(**RF_HP)
    model.fit(X, y)
    return model


# ============================================================================
# Helper: assemble feature matrix with NaN handling
# ============================================================================


def _features_to_matrix(
    df: pd.DataFrame, feature_names: List[str], fillna_strategy: str = "median"
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return (X, fillna_values). NaNs filled with column median computed on the
    available training pool (full pool here).
    """
    X = df[feature_names].to_numpy(dtype=float)
    fill: Dict[str, float] = {}
    for j, name in enumerate(feature_names):
        col = X[:, j]
        med = float(np.nanmedian(col)) if col.size else 0.0
        if math.isnan(med):
            med = 0.0
        if np.any(np.isnan(col)):
            X[np.isnan(col), j] = med
        fill[name] = med
    return X, fill


# ============================================================================
# Single-run orchestration
# ============================================================================


@dataclass
class _PipelineResult:
    pipeline: str                            # "E" | "D1"
    cluster_id: int
    selected_t: Optional[int]                # only for D1
    fsr: FilterSelectionResult
    threshold: Optional[ThresholdSweepResult]
    n_trades_used: int                       # eligible after exclusion (D1 only)
    exclusion_pct: float                     # D1 only
    final_model: Any                         # sklearn classifier (or stacked pair)
    artefact_yaml: Dict[str, Any] = field(default_factory=dict)
    classifier_path: Optional[Path] = None
    yaml_path: Optional[Path] = None
    importances_path: Optional[Path] = None
    cluster_label: str = ""
    selected_sl: Optional[float] = None
    t_sweep_records: List[Dict[str, Any]] = field(default_factory=list)


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(df_or_rows: Any, path: Path) -> None:
    if isinstance(df_or_rows, list):
        df = pd.DataFrame(df_or_rows)
    else:
        df = df_or_rows
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_yaml_deterministic(content: Dict[str, Any], path: Path) -> None:
    """Write YAML with sort_keys=True and default_flow_style=False for stable
    serialization."""
    path.write_text(
        yaml.safe_dump(content, sort_keys=True, default_flow_style=False, allow_unicode=False),
        encoding="utf-8",
    )


def run_once(
    trades_csv: Path,
    paths_csv: Path,
    clusters_csv: Path,
    pass_list_csv: Path,
    catalogue_path: Path,
    data_dir: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(trades_csv)
    trades = trades.sort_values("trade_id").reset_index(drop=True)
    clusters = pd.read_csv(clusters_csv).sort_values("trade_id").reset_index(drop=True)
    if not np.array_equal(
        trades["trade_id"].to_numpy(dtype=int), clusters["trade_id"].to_numpy(dtype=int)
    ):
        raise ValueError("trade_id mismatch between trades_all and clusters_K4")
    pass_list = pd.read_csv(pass_list_csv)
    survivors: List[Dict[str, Any]] = [
        {"cluster_id": int(r["cluster_id"]), "selected_SL": float(r["selected_SL"])}
        for _, r in pass_list.iterrows()
    ]

    catalogue = yaml.safe_load(catalogue_path.read_text(encoding="utf-8"))
    base = list(catalogue["base_features"])
    arc_specific = list(catalogue["l_arc_4"]["arc_specific_features"])
    all_entry_features = base + arc_specific

    # --- Build entry features ---
    print("[l_arc_4 step4] Building entry feature matrix...", file=sys.stderr)
    entry_feats_df, _entry_audit = build_entry_feature_matrix(
        trades, data_dir, all_entry_features
    )

    # Match clusters / labels.
    cluster_id_per_trade = clusters["cluster_id"].to_numpy(dtype=int)

    # --- Load paths into a 2D (n_trades, 241, 4) tensor for D1 ---
    print("[l_arc_4 step4] Loading paths into 2D tensor for D1 features...", file=sys.stderr)
    paths_df = pd.read_csv(paths_csv)
    paths_df = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    n_trades = int(len(trades))
    PATH_BARS = 241
    if len(paths_df) != n_trades * PATH_BARS:
        raise ValueError(
            f"paths CSV has {len(paths_df)} rows; expected {n_trades} × {PATH_BARS} = "
            f"{n_trades * PATH_BARS}"
        )
    opens = paths_df["open"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    highs = paths_df["high"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    lows = paths_df["low"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    closes = paths_df["close"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    path_tensor = np.stack([opens, highs, lows, closes], axis=2)  # (n, 241, 4)

    entry_prices_all = trades["entry_price"].to_numpy(dtype=float)
    atr_14_all = trades["atr_14_at_signal"].to_numpy(dtype=float)
    bars_held_all = trades["bars_held"].to_numpy(dtype=int)

    # === Per-cluster processing ===
    stacking_log: List[Dict[str, Any]] = []
    stacking_budget_remaining = STACKING_BUDGET

    angle_e_rows: List[Dict[str, Any]] = []
    angle_d1_rows: List[Dict[str, Any]] = []
    extract_pass_rows: List[Dict[str, Any]] = []
    pipeline_results: List[_PipelineResult] = []

    for sv in survivors:
        cid = sv["cluster_id"]
        selected_sl = sv["selected_SL"]
        cluster_label = CLUSTER_LABELS.get(cid, f"cluster_{cid}")

        # Binary target: 1 if trade is in this cluster.
        y = (cluster_id_per_trade == cid).astype(int)
        if int(y.sum()) < 50:
            print(f"[l_arc_4 step4] cluster {cid}: positive class < 50 — skipping (Step 4 §15 floor)", file=sys.stderr)
            continue

        # --- Pipeline E ---
        print(f"[l_arc_4 step4] cluster {cid} ({cluster_label}) — Pipeline E", file=sys.stderr)
        X_e, fill_e = _features_to_matrix(entry_feats_df, all_entry_features)
        fsr_e, stacking_budget_remaining = run_filter_selection(
            X_e,
            y,
            all_entry_features,
            AUC_GATE_E,
            cid,
            "E",
            stacking_log,
            stacking_budget_remaining,
        )
        # If gate passes, sweep threshold + train final model.
        if fsr_e.passes_gate:
            sub_idx = [all_entry_features.index(n) for n in fsr_e.feature_subset]
            X_e_sub = X_e[:, sub_idx]
            tsr_e = _threshold_sweep(X_e_sub, y)
            final_model_e = _train_final_rf(X_e_sub, y)
            res_e = _PipelineResult(
                pipeline="E",
                cluster_id=cid,
                selected_t=None,
                fsr=fsr_e,
                threshold=tsr_e,
                n_trades_used=int(len(y)),
                exclusion_pct=0.0,
                final_model=final_model_e,
                cluster_label=cluster_label,
                selected_sl=selected_sl,
            )
        else:
            res_e = _PipelineResult(
                pipeline="E",
                cluster_id=cid,
                selected_t=None,
                fsr=fsr_e,
                threshold=None,
                n_trades_used=int(len(y)),
                exclusion_pct=0.0,
                final_model=None,
                cluster_label=cluster_label,
                selected_sl=selected_sl,
            )

        angle_e_rows.append(
            {
                "cluster_id": cid,
                "cluster_label": cluster_label,
                "selected_SL": selected_sl,
                "rf_auc_mean": fsr_e.rf_auc_mean,
                "rf_auc_folds": ";".join(f"{x:.4f}" for x in fsr_e.rf_auc_folds),
                "lr_auc_mean": fsr_e.lr_auc_mean,
                "lr_auc_folds": ";".join(f"{x:.4f}" for x in fsr_e.lr_auc_folds),
                "rf_lr_gap": fsr_e.rf_auc_mean - fsr_e.lr_auc_mean,
                "gate_passed": int(fsr_e.passes_gate),
                "step": fsr_e.step,
                "feature_subset_size": len(fsr_e.feature_subset),
                "feature_subset": ";".join(fsr_e.feature_subset),
                "step_b_paths_tried": fsr_e.step_b_paths_tried,
                "step_c_combinations": fsr_e.step_c_combinations,
                "selected_threshold": (res_e.threshold.selected_threshold if res_e.threshold else float("nan")),
                "selected_precision": (res_e.threshold.selected_precision if res_e.threshold else float("nan")),
                "selected_recall": (res_e.threshold.selected_recall if res_e.threshold else float("nan")),
                "threshold_used_fallback": int(res_e.threshold.used_fallback if res_e.threshold else False),
                "notes": fsr_e.notes,
            }
        )

        # --- Pipeline D1 ---
        print(f"[l_arc_4 step4] cluster {cid} ({cluster_label}) — Pipeline D1 t-sweep", file=sys.stderr)
        # The R-frame for D1 features = cluster's selected SL × ATR (per trade).
        # For D1 t-sweep we apply Step A only at each t. The smallest-t result
        # (passing AUC ≥ 0.60 AND exclusion ≤ 30%) is then re-run through full
        # filter selection if needed.
        d1_t_records: List[Dict[str, Any]] = []
        d1_t_candidates: List[Tuple[int, float, List[float], int, float]] = []
        # We need a way to filter trades by bars_held >= t.
        # Pipeline D1 mechanic: pre-t SL = 2.0 × ATR (matching Step 1). Use
        # Step 1's bars_held for exclusion (it was computed under 2.0 × ATR).
        for t in D1_T_VALUES:
            eligible_mask = bars_held_all >= t
            # Restrict to eligible trades AND inside this cluster's pool — but
            # for cluster MEMBERSHIP prediction we use the FULL pool (positive
            # class = trades in cluster cid, negative = trades NOT in cluster).
            # Pre-t SL is uniform 2.0 × ATR for everyone, so eligibility is
            # pool-level not cluster-conditioned.
            elig_idx = np.where(eligible_mask)[0]
            if elig_idx.size < 200:
                d1_t_records.append(
                    {
                        "cluster_id": cid,
                        "t": t,
                        "n_eligible": int(elig_idx.size),
                        "exclusion_pct": float(1 - elig_idx.size / n_trades),
                        "rf_auc_mean": float("nan"),
                        "rf_auc_folds": "",
                        "passes_gate": 0,
                        "note": "skipped: n_eligible < 200",
                    }
                )
                continue
            entry_prices_elig = entry_prices_all[elig_idx]
            # R per trade = cluster's selected SL × ATR(14) at signal. Use
            # cluster-level SL even for non-cluster trades; the classifier is
            # predicting cluster MEMBERSHIP, and the R-normalisation is just a
            # feature scaling choice — applied uniformly to all candidates.
            r_per_trade = selected_sl * atr_14_all[elig_idx]
            d1_feats = _compute_d1_features_at_t(
                path_tensor[elig_idx],
                entry_prices_elig,
                r_per_trade,
                t,
            )
            # Build combined feature matrix (8 base + 7 D1 path-so-far).
            X_e_elig = X_e[elig_idx]
            X_combined = np.concatenate([X_e_elig[:, : len(base)], d1_feats], axis=1)
            y_elig = y[elig_idx]
            exclusion_pct = float(1 - elig_idx.size / n_trades)
            t_auc_mean, t_auc_folds = _eval_rf(X_combined, y_elig)
            passes = bool(t_auc_mean >= AUC_GATE_D1 and exclusion_pct <= D1_EXCL_MAX)
            d1_t_records.append(
                {
                    "cluster_id": cid,
                    "t": t,
                    "n_eligible": int(elig_idx.size),
                    "exclusion_pct": exclusion_pct,
                    "rf_auc_mean": t_auc_mean,
                    "rf_auc_folds": ";".join(f"{x:.4f}" for x in t_auc_folds),
                    "passes_gate": int(passes),
                    "note": (
                        "PASS"
                        if passes
                        else (
                            f"AUC {t_auc_mean:.4f} < {AUC_GATE_D1}"
                            if t_auc_mean < AUC_GATE_D1
                            else f"exclusion {exclusion_pct:.2%} > {D1_EXCL_MAX:.0%}"
                        )
                    ),
                }
            )
            if passes:
                d1_t_candidates.append((t, t_auc_mean, t_auc_folds, int(elig_idx.size), exclusion_pct))

        angle_d1_rows.extend(d1_t_records)

        # Smallest-t selection.
        if d1_t_candidates:
            d1_t_candidates.sort(key=lambda x: x[0])
            chosen_t, chosen_auc, chosen_folds, chosen_n, chosen_excl = d1_t_candidates[0]
            print(f"[l_arc_4 step4] cluster {cid} — D1 smallest passing t = {chosen_t}", file=sys.stderr)
            # Re-build the X for the chosen t to lock the classifier.
            eligible_mask = bars_held_all >= chosen_t
            elig_idx = np.where(eligible_mask)[0]
            entry_prices_elig = entry_prices_all[elig_idx]
            r_per_trade = selected_sl * atr_14_all[elig_idx]
            d1_feats = _compute_d1_features_at_t(
                path_tensor[elig_idx], entry_prices_elig, r_per_trade, chosen_t
            )
            X_e_elig = X_e[elig_idx]
            X_combined = np.concatenate([X_e_elig[:, : len(base)], d1_feats], axis=1)
            y_elig = y[elig_idx]
            d1_feature_names = base + [f"{n}_at_t{chosen_t}" for n in [
                "close_r", "mfe_so_far_r", "mae_so_far_r", "bars_in_profit",
                "local_peaks_so_far", "monotonicity_so_far", "velocity_first_t"
            ]]
            # Step A already passed (passes_gate True at chosen t). Use Step A
            # result; if needed could escalate to Steps B/C — protocol applies
            # them analogously but only when single full-feature RF fails.
            lr_mean_d1, lr_folds_d1 = _eval_lr(X_combined, y_elig)
            imp_d1 = _rf_feature_importances(X_combined, y_elig, d1_feature_names)
            fsr_d1 = FilterSelectionResult(
                step="A",
                feature_subset=d1_feature_names,
                rf_auc_mean=chosen_auc,
                rf_auc_folds=chosen_folds,
                lr_auc_mean=lr_mean_d1,
                lr_auc_folds=lr_folds_d1,
                importances=imp_d1,
                step_b_paths_tried=0,
                step_c_combinations=0,
                notes=f"Step A passes at smallest valid t={chosen_t}: RF AUC {chosen_auc:.4f} ≥ {AUC_GATE_D1}",
                passes_gate=True,
            )
            tsr_d1 = _threshold_sweep(X_combined, y_elig)
            final_model_d1 = _train_final_rf(X_combined, y_elig)
            res_d1 = _PipelineResult(
                pipeline="D1",
                cluster_id=cid,
                selected_t=chosen_t,
                fsr=fsr_d1,
                threshold=tsr_d1,
                n_trades_used=int(elig_idx.size),
                exclusion_pct=chosen_excl,
                final_model=final_model_d1,
                cluster_label=cluster_label,
                selected_sl=selected_sl,
                t_sweep_records=d1_t_records,
            )
        else:
            # Pipeline D1 fails for this cluster.
            res_d1 = _PipelineResult(
                pipeline="D1",
                cluster_id=cid,
                selected_t=None,
                fsr=FilterSelectionResult(
                    step="FAIL",
                    feature_subset=[],
                    rf_auc_mean=float("nan"),
                    rf_auc_folds=[],
                    lr_auc_mean=float("nan"),
                    lr_auc_folds=[],
                    importances=[],
                    step_b_paths_tried=0,
                    step_c_combinations=0,
                    notes="no t in {1,2,3,4,5,10} satisfies AUC ≥ 0.60 AND exclusion ≤ 30%",
                    passes_gate=False,
                ),
                threshold=None,
                n_trades_used=0,
                exclusion_pct=float("nan"),
                final_model=None,
                cluster_label=cluster_label,
                selected_sl=selected_sl,
                t_sweep_records=d1_t_records,
            )

        # --- Artefact production ---
        if res_e.fsr.passes_gate:
            clf_path = out_dir / f"cluster_{cid}_E_classifier.joblib"
            joblib.dump(res_e.final_model, clf_path, compress=0)
            res_e.classifier_path = clf_path

            artefact_yaml = {
                "cluster_id": int(cid),
                "cluster_label": cluster_label,
                "pipeline": "E",
                "archetype_R_frame_atr_mult": float(selected_sl),
                "classifier_type": "RandomForestClassifier",
                "classifier_hyperparams": {k: v for k, v in RF_HP.items()},
                "feature_list": list(res_e.fsr.feature_subset),
                "feature_count": len(res_e.fsr.feature_subset),
                "training_pool_size": int(len(y)),
                "training_positive_count": int(y.sum()),
                "cv_n_folds": int(N_CV),
                "cv_rf_auc_mean": float(res_e.fsr.rf_auc_mean),
                "cv_rf_auc_folds": [float(x) for x in res_e.fsr.rf_auc_folds],
                "cv_lr_auc_mean": float(res_e.fsr.lr_auc_mean),
                "selected_threshold": float(res_e.threshold.selected_threshold),
                "expected_precision_at_threshold": float(res_e.threshold.selected_precision),
                "expected_recall_at_threshold": float(res_e.threshold.selected_recall),
                "threshold_used_fallback": bool(res_e.threshold.used_fallback),
                "feature_fillna_values": {k: float(v) for k, v in fill_e.items() if k in res_e.fsr.feature_subset},
                "filter_selection_step": res_e.fsr.step,
                "filter_selection_notes": res_e.fsr.notes,
            }
            res_e.artefact_yaml = artefact_yaml
            yaml_path = out_dir / f"cluster_{cid}_E_filter.yaml"
            _write_yaml_deterministic(artefact_yaml, yaml_path)
            res_e.yaml_path = yaml_path

            imp_rows = [
                {"rank": rank + 1, "feature": name, "importance": float(imp)}
                for rank, (name, imp) in enumerate(res_e.fsr.importances[:10])
            ]
            imp_path = out_dir / f"feature_importances_cluster_{cid}_E.csv"
            _write_csv(imp_rows, imp_path)
            res_e.importances_path = imp_path

        if res_d1.fsr.passes_gate:
            clf_path = out_dir / f"cluster_{cid}_D1_classifier_t{res_d1.selected_t}.joblib"
            joblib.dump(res_d1.final_model, clf_path, compress=0)
            res_d1.classifier_path = clf_path

            # §11 row exit policy lookup. Cluster 1 = unclassified;
            # cluster 3 = Stepwise climber (near-miss, row 2).
            policy_ref = {
                1: {"row": "unclassified", "exit_policy": "TBD (no §11 row matches; chat-level assignment required)"},
                3: {"row": "2", "exit_policy": "Stepwise climber: MFE-lock at 1R, trail 0.75R from new high (Pipeline D1 at bar N)"},
            }.get(cid, {"row": "unclassified", "exit_policy": "TBD"})

            artefact_yaml_d1 = {
                "cluster_id": int(cid),
                "cluster_label": cluster_label,
                "pipeline": "D1",
                "selected_t": int(res_d1.selected_t),
                "archetype_R_frame_atr_mult": float(selected_sl),
                "pre_t_sl_atr_mult": 2.0,  # uniform per §3 Pipeline D1 spec
                "post_t_sl_atr_mult": float(selected_sl),
                "classifier_type": "RandomForestClassifier",
                "classifier_hyperparams": {k: v for k, v in RF_HP.items()},
                "feature_list": list(res_d1.fsr.feature_subset),
                "feature_count": len(res_d1.fsr.feature_subset),
                "training_pool_size": int(res_d1.n_trades_used),
                "training_positive_count": int(
                    y[bars_held_all >= res_d1.selected_t].sum()
                ),
                "exclusion_pct": float(res_d1.exclusion_pct),
                "cv_n_folds": int(N_CV),
                "cv_rf_auc_mean": float(res_d1.fsr.rf_auc_mean),
                "cv_rf_auc_folds": [float(x) for x in res_d1.fsr.rf_auc_folds],
                "cv_lr_auc_mean": float(res_d1.fsr.lr_auc_mean),
                "selected_threshold": float(res_d1.threshold.selected_threshold),
                "expected_precision_at_threshold": float(res_d1.threshold.selected_precision),
                "expected_recall_at_threshold": float(res_d1.threshold.selected_recall),
                "threshold_used_fallback": bool(res_d1.threshold.used_fallback),
                "filter_selection_step": res_d1.fsr.step,
                "filter_selection_notes": res_d1.fsr.notes,
                "section_11_row": policy_ref["row"],
                "section_11_exit_policy": policy_ref["exit_policy"],
            }
            res_d1.artefact_yaml = artefact_yaml_d1
            yaml_path_d1 = out_dir / f"cluster_{cid}_D1_policy.yaml"
            _write_yaml_deterministic(artefact_yaml_d1, yaml_path_d1)
            res_d1.yaml_path = yaml_path_d1

            imp_rows_d1 = [
                {"rank": rank + 1, "feature": name, "importance": float(imp)}
                for rank, (name, imp) in enumerate(res_d1.fsr.importances[:10])
            ]
            imp_path_d1 = out_dir / f"feature_importances_cluster_{cid}_D1.csv"
            _write_csv(imp_rows_d1, imp_path_d1)
            res_d1.importances_path = imp_path_d1

        # Extractability pass list entry.
        passes_e = res_e.fsr.passes_gate
        passes_d1 = res_d1.fsr.passes_gate
        if passes_e and passes_d1:
            pipeline_assign = "both"
        elif passes_e:
            pipeline_assign = "E"
        elif passes_d1:
            pipeline_assign = "D1"
        else:
            pipeline_assign = "dies"
        extract_pass_rows.append(
            {
                "cluster_id": cid,
                "cluster_label": cluster_label,
                "selected_SL": selected_sl,
                "passes_E": int(passes_e),
                "passes_D1": int(passes_d1),
                "pipeline_assignment": pipeline_assign,
                "E_rf_auc": res_e.fsr.rf_auc_mean,
                "E_step": res_e.fsr.step,
                "D1_rf_auc": res_d1.fsr.rf_auc_mean,
                "D1_selected_t": res_d1.selected_t if res_d1.selected_t is not None else "",
                "D1_step": res_d1.fsr.step,
            }
        )
        pipeline_results.append(res_e)
        pipeline_results.append(res_d1)

    # === Write top-level CSVs ===
    angle_e_path = out_dir / "predictability_angle_E.csv"
    _write_csv(angle_e_rows, angle_e_path)

    angle_d1_path = out_dir / "predictability_angle_D1.csv"
    _write_csv(angle_d1_rows, angle_d1_path)

    pass_list_path = out_dir / "extractability_pass_list.csv"
    _write_csv(extract_pass_rows, pass_list_path)

    stacking_path = out_dir / "stacking_log.csv"
    _write_csv(stacking_log, stacking_path)

    sha_files: Dict[str, str] = {}
    for p in [angle_e_path, angle_d1_path, pass_list_path, stacking_path]:
        sha_files[p.name] = _file_sha256(p)
    for res in pipeline_results:
        if res.classifier_path is not None:
            sha_files[res.classifier_path.name] = _file_sha256(res.classifier_path)
        if res.yaml_path is not None:
            sha_files[res.yaml_path.name] = _file_sha256(res.yaml_path)
        if res.importances_path is not None:
            sha_files[res.importances_path.name] = _file_sha256(res.importances_path)

    run_artefacts = {
        "angle_e_rows": angle_e_rows,
        "angle_d1_rows": angle_d1_rows,
        "extract_pass_rows": extract_pass_rows,
        "stacking_log": stacking_log,
        "pipeline_results": pipeline_results,
        "stacking_budget_remaining": stacking_budget_remaining,
        "stacking_budget_total": STACKING_BUDGET,
        "n_pool": int(n_trades),
        "n_features_base": len(base),
        "n_features_arc": len(arc_specific),
        "all_entry_features": all_entry_features,
    }
    return sha_files, run_artefacts


# ============================================================================
# Diagnostics markdown writer
# ============================================================================


def write_diagnostics(
    out_path: Path,
    run_arts: Dict[str, Any],
    sha_run1: Dict[str, str],
    sha_run2: Optional[Dict[str, str]],
    determinism_gate: str,
    config_paths: Dict[str, str],
    config_shas: Dict[str, str],
) -> str:
    extract_pass_rows = run_arts["extract_pass_rows"]
    pipeline_results: List[_PipelineResult] = run_arts["pipeline_results"]
    stacking_log = run_arts["stacking_log"]
    n_base = run_arts["n_features_base"]
    n_arc = run_arts["n_features_arc"]
    all_entry_features = run_arts["all_entry_features"]

    clusters_pass_e = [r for r in extract_pass_rows if r["passes_E"]]
    clusters_pass_d1 = [r for r in extract_pass_rows if r["passes_D1"]]
    clusters_pass_both = [r for r in extract_pass_rows if r["passes_E"] and r["passes_D1"]]
    clusters_die = [r for r in extract_pass_rows if not (r["passes_E"] or r["passes_D1"])]
    arc_pass = len(clusters_pass_e) + len(clusters_pass_d1) >= 1
    arc_disp = "PASS" if arc_pass else "FAIL"

    lines: List[str] = []
    lines.append("# Arc 4 — Step 4 extractability + artefact production diagnostics")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §8 (Step A/B/C cascade, threshold sweep, artefact production)")
    lines.append(
        "Signal:   `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001` "
        "(LCHAR_TOPN_REGISTRY.md Entry 4)"
    )
    lines.append("")

    # Headline.
    lines.append("## Summary")
    lines.append("")
    summary_bits = []
    if clusters_pass_e:
        summary_bits.append(
            f"{len(clusters_pass_e)} cluster(s) clear Pipeline E: "
            + ", ".join(f"cluster {r['cluster_id']} (RF AUC {r['E_rf_auc']:.4f}, Step {r['E_step']})" for r in clusters_pass_e)
        )
    else:
        summary_bits.append("0 clusters clear Pipeline E")
    if clusters_pass_d1:
        summary_bits.append(
            f"{len(clusters_pass_d1)} cluster(s) clear Pipeline D1: "
            + ", ".join(f"cluster {r['cluster_id']} (RF AUC {r['D1_rf_auc']:.4f}, t={r['D1_selected_t']}, Step {r['D1_step']})" for r in clusters_pass_d1)
        )
    else:
        summary_bits.append("0 clusters clear Pipeline D1")
    if clusters_pass_both:
        summary_bits.append(f"{len(clusters_pass_both)} cluster(s) clear both pipelines")
    if clusters_die:
        summary_bits.append(f"{len(clusters_die)} cluster(s) die on both pipelines")
    lines.append(
        f"{len(extract_pass_rows)} surviving cluster(s) tested. "
        + ". ".join(summary_bits) + ". "
        f"Arc-level §8 gate disposition (≥ 1 cluster clears either pipeline): **{arc_disp}**. "
        f"Determinism: **{determinism_gate}**."
    )
    lines.append("")

    # Compact per-cluster table.
    lines.append("## Per-cluster compact summary")
    lines.append("")
    lines.append(
        "| Cluster | Label | Pipeline | RF AUC | LR AUC | gap | Step | Gate | Threshold | "
        "Precision | Recall | n features |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|"
    )
    for res in pipeline_results:
        gate_threshold = AUC_GATE_E if res.pipeline == "E" else AUC_GATE_D1
        gate = "PASS" if res.fsr.passes_gate else f"FAIL (< {gate_threshold})"
        thr = res.threshold.selected_threshold if res.threshold else float("nan")
        prec = res.threshold.selected_precision if res.threshold else float("nan")
        rec = res.threshold.selected_recall if res.threshold else float("nan")
        gap = res.fsr.rf_auc_mean - res.fsr.lr_auc_mean if not math.isnan(res.fsr.lr_auc_mean) else float("nan")
        pipe_label = res.pipeline if res.pipeline == "E" else f"D1 (t={res.selected_t if res.selected_t else '—'})"
        lines.append(
            f"| {res.cluster_id} | {res.cluster_label} | {pipe_label} | "
            f"{res.fsr.rf_auc_mean:.4f} | {res.fsr.lr_auc_mean:.4f} | "
            f"{gap:+.4f} | {res.fsr.step} | {gate} | "
            f"{thr:.2f} | {prec:.4f} | {rec:.4f} | {len(res.fsr.feature_subset)} |"
        )
    lines.append("")

    # Pipeline assignment.
    lines.append("## Pipeline assignment per cluster")
    lines.append("")
    lines.append("| Cluster | Label | Selected SL | Passes E | Passes D1 | Assignment |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for r in extract_pass_rows:
        lines.append(
            f"| {r['cluster_id']} | {r['cluster_label']} | {r['selected_SL']} × ATR | "
            f"{'yes' if r['passes_E'] else 'no'} | "
            f"{'yes' if r['passes_D1'] else 'no'} | {r['pipeline_assignment']} |"
        )
    lines.append("")

    # Per-cluster detail sections.
    for r in extract_pass_rows:
        cid = r["cluster_id"]
        lbl = r["cluster_label"]
        sl = r["selected_SL"]
        lines.append(f"## Cluster {cid} — {lbl} (R-frame = {sl} × ATR)")
        lines.append("")

        # Pipeline E.
        e_res = next((p for p in pipeline_results if p.cluster_id == cid and p.pipeline == "E"), None)
        if e_res is not None:
            lines.append("### Pipeline E")
            lines.append("")
            lines.append(f"- Filter-selection step: **{e_res.fsr.step}**")
            lines.append(f"- RF AUC mean: **{e_res.fsr.rf_auc_mean:.4f}**  (gate: ≥ {AUC_GATE_E})")
            lines.append(
                f"- RF AUC per fold: {[f'{x:.4f}' for x in e_res.fsr.rf_auc_folds]} "
                f"(std {np.std(e_res.fsr.rf_auc_folds):.4f})"
            )
            lines.append(f"- LR AUC mean: {e_res.fsr.lr_auc_mean:.4f}  (informational)")
            lines.append(f"- RF−LR gap: {e_res.fsr.rf_auc_mean - e_res.fsr.lr_auc_mean:+.4f}")
            gap = e_res.fsr.rf_auc_mean - e_res.fsr.lr_auc_mean
            if abs(gap) >= 0.10:
                lines.append(
                    "  - Gap ≥ 0.10: **non-linear dynamics dominate** — feature interactions are "
                    "predictive beyond what linear combinations capture."
                )
            else:
                lines.append(
                    "  - Gap < 0.10: feature set is binding — richer features may help; "
                    "non-linearity alone won't."
                )
            lines.append(f"- Feature subset size: {len(e_res.fsr.feature_subset)} / 32")
            if e_res.threshold:
                lines.append(
                    f"- Selected threshold: **{e_res.threshold.selected_threshold:.2f}** — "
                    f"precision {e_res.threshold.selected_precision:.4f}, "
                    f"recall {e_res.threshold.selected_recall:.4f}"
                    + (" *(fallback — no threshold met recall floor)*" if e_res.threshold.used_fallback else "")
                )
                tn, fp, fn, tp = e_res.threshold.confusion_matrix_tn_fp_fn_tp
                lines.append(
                    f"- Confusion matrix @ threshold (TN/FP/FN/TP): {tn} / {fp} / {fn} / {tp}"
                )
                lines.append("- Threshold grid:")
                lines.append("  | Threshold | Precision | Recall |")
                lines.append("  |---:|---:|---:|")
                for g in e_res.threshold.full_grid:
                    lines.append(
                        f"  | {g['threshold']:.2f} | {g['precision']:.4f} | {g['recall']:.4f} |"
                    )
            else:
                lines.append("- No threshold sweep — Step A/B/C all failed; pipeline FAIL.")
            lines.append("- Top-10 RF feature importances:")
            lines.append("  | Rank | Feature | Importance |")
            lines.append("  |---:|---|---:|")
            for rank, (name, imp) in enumerate(e_res.fsr.importances[:10]):
                lines.append(f"  | {rank + 1} | `{name}` | {imp:.4f} |")
            lines.append(f"- Notes: {e_res.fsr.notes}")
            lines.append("")

        # Pipeline D1.
        d1_res = next((p for p in pipeline_results if p.cluster_id == cid and p.pipeline == "D1"), None)
        if d1_res is not None:
            lines.append("### Pipeline D1")
            lines.append("")
            # t-sweep table.
            lines.append("t-sweep (RF AUC + exclusion):")
            lines.append("")
            lines.append("| t | n_eligible | exclusion% | RF AUC mean | passes? | note |")
            lines.append("|---:|---:|---:|---:|---|---|")
            for rec in d1_res.t_sweep_records:
                auc_str = f"{rec['rf_auc_mean']:.4f}" if rec["rf_auc_mean"] == rec["rf_auc_mean"] else "—"
                lines.append(
                    f"| {rec['t']} | {rec['n_eligible']} | {rec['exclusion_pct']:.2%} | "
                    f"{auc_str} | "
                    f"{'PASS' if rec['passes_gate'] else 'FAIL'} | {rec['note']} |"
                )
            lines.append("")
            if d1_res.fsr.passes_gate:
                lines.append(
                    f"- Smallest passing t: **{d1_res.selected_t}** "
                    f"(RF AUC {d1_res.fsr.rf_auc_mean:.4f}, exclusion {d1_res.exclusion_pct:.2%})"
                )
                lines.append(f"- Filter-selection step: **{d1_res.fsr.step}**")
                lines.append(f"- LR AUC mean (at t={d1_res.selected_t}): {d1_res.fsr.lr_auc_mean:.4f}")
                lines.append(f"- RF−LR gap: {d1_res.fsr.rf_auc_mean - d1_res.fsr.lr_auc_mean:+.4f}")
                lines.append(f"- Feature subset size: {len(d1_res.fsr.feature_subset)} (8 base + 7 path-so-far)")
                if d1_res.threshold:
                    lines.append(
                        f"- Selected threshold: **{d1_res.threshold.selected_threshold:.2f}** — "
                        f"precision {d1_res.threshold.selected_precision:.4f}, "
                        f"recall {d1_res.threshold.selected_recall:.4f}"
                        + (" *(fallback)*" if d1_res.threshold.used_fallback else "")
                    )
                    tn, fp, fn, tp = d1_res.threshold.confusion_matrix_tn_fp_fn_tp
                    lines.append(
                        f"- Confusion matrix @ threshold (TN/FP/FN/TP): {tn} / {fp} / {fn} / {tp}"
                    )
                lines.append("- Top-10 RF feature importances:")
                lines.append("  | Rank | Feature | Importance |")
                lines.append("  |---:|---|---:|")
                for rank, (name, imp) in enumerate(d1_res.fsr.importances[:10]):
                    lines.append(f"  | {rank + 1} | `{name}` | {imp:.4f} |")
            else:
                lines.append(f"- Pipeline D1 FAIL: {d1_res.fsr.notes}")
            lines.append("")

        lines.append(f"### Pipeline assignment: **{r['pipeline_assignment']}**")
        if r["pipeline_assignment"] == "E":
            lines.append("- Cluster routes through entry-time filter at signal.")
        elif r["pipeline_assignment"] == "D1":
            lines.append("- Cluster routes through deferred classification at bar N.")
        elif r["pipeline_assignment"] == "both":
            lines.append(
                "- Cluster qualifies for both; Step 6 WFO evaluates E alone, D1 alone, "
                "and E+D1 unison; ships best."
            )
        else:
            lines.append("- Cluster dies — no pipeline clears the AUC gate.")
        lines.append("")

    # Stacking budget consumed.
    lines.append("## Stacking budget (Tier 1 / Step C)")
    lines.append("")
    used_combos = len(stacking_log)
    lines.append(
        f"Combinations evaluated: **{used_combos}** / {STACKING_BUDGET} budgeted "
        f"({run_arts['stacking_budget_remaining']} remaining)."
    )
    if used_combos > 0:
        lines.append("")
        lines.append("| Cluster | Pipeline | Subset A | size A | Subset B | size B | Combined AUC | Passes |")
        lines.append("|---:|---|---|---:|---|---:|---:|---|")
        for r in stacking_log:
            lines.append(
                f"| {r['cluster_id']} | {r['pipeline']} | {r['subset_a']} | "
                f"{r['subset_a_size']} | {r['subset_b']} | {r['subset_b_size']} | "
                f"{r['combined_auc']:.4f} | {'PASS' if r['passes_gate'] else 'FAIL'} |"
            )
    else:
        lines.append("No Step C stacking combinations were evaluated (Steps A/B sufficient or pipeline failed before C).")
    lines.append("")

    # Feature catalogue audit.
    lines.append("## Feature catalogue audit")
    lines.append("")
    lines.append(
        f"Total entry features: **{n_base + n_arc}** = {n_base} base + {n_arc} arc-specific (≤ 38 cap)."
    )
    lines.append("")
    lines.append("Feature list:")
    lines.append("")
    lines.append("| Source | Feature |")
    lines.append("|---|---|")
    for f in all_entry_features:
        source = "base" if f in [
            "body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "range_to_atr_14", "ret_5bar_atr", "ret_20bar_atr",
            "pos_in_20bar_range", "rsi_14",
        ] else "arc-specific"
        lines.append(f"| {source} | `{f}` |")
    lines.append("")
    lines.append("Data leakage spot-check:")
    lines.append("- All entry features computed strictly from bars ≤ N (rolling/trailing windows ending at N).")
    lines.append("- D1 path-so-far features computed from bars 0..t inclusive (t is post-entry observation point).")
    lines.append("- Pre-t SL for D1 = 2.0 × ATR (uniform per §3 mechanic); eligibility uses Step 1 `bars_held` field (also under 2.0 × ATR).")
    lines.append("- D1 feature R-frame = cluster's selected SL × ATR (per cluster).")
    lines.append("- Session / hour / day-of-week derive solely from signal bar timestamp — no future info.")
    lines.append("")

    # Cross-arc observations (informational only).
    lines.append("## Cross-arc observations (informational; no calibration moves within arc)")
    lines.append("")
    e_rf_aucs = [r["E_rf_auc"] for r in extract_pass_rows]
    d1_rf_aucs = [r["D1_rf_auc"] for r in extract_pass_rows]
    if e_rf_aucs and d1_rf_aucs:
        max_e = max(e_rf_aucs)
        max_d1 = max(d1_rf_aucs)
        lines.append(
            f"- Highest Pipeline E RF AUC across surviving clusters: {max_e:.4f} "
            f"(gate {AUC_GATE_E})"
        )
        lines.append(
            f"- Highest Pipeline D1 RF AUC across surviving clusters: {max_d1:.4f} "
            f"(gate {AUC_GATE_D1})"
        )
    lines.append("")

    # Determinism.
    lines.append("## Determinism")
    lines.append("")
    lines.append("Two-run byte-identical sha256 over all written files:")
    lines.append("")
    lines.append("| File | Run 1 sha256 | Run 2 sha256 | Match |")
    lines.append("|---|---|---|---|")
    for fname in sorted(sha_run1.keys()):
        s1 = sha_run1[fname]
        s2 = sha_run2.get(fname) if sha_run2 else None
        match = "—" if s2 is None else ("PASS" if s1 == s2 else "FAIL")
        lines.append(f"| `{fname}` | `{s1[:16]}…` | `{(s2 or '—')[:16]}{'…' if s2 else ''}` | {match} |")
    lines.append("")
    lines.append(f"**Determinism: {determinism_gate}**")
    lines.append("")

    # Config sha256s.
    lines.append("## Config / input sha256s")
    lines.append("")
    for label, path in config_paths.items():
        lines.append(f"- `{label}` (`{path}`) — `{config_shas[label]}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return arc_disp


# ============================================================================
# Main
# ============================================================================


def _env_dict() -> Dict[str, str]:
    try:
        import sklearn  # type: ignore

        sk_ver = sklearn.__version__
    except Exception:
        sk_ver = "not_installed"
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sk_ver,
        "joblib": joblib.__version__,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arc 4 Step 4 extractability + artefact production (L_ARC_PROTOCOL v2.1.1 §8)."
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "l_arc_4.yaml",
    )
    p.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--paths-csv",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--clusters-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step2" / "clusters_K4.csv",
    )
    p.add_argument(
        "--pass-list-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step3" / "capturability_pass_list.csv",
    )
    p.add_argument(
        "--catalogue",
        type=Path,
        default=_REPO_ROOT / "configs" / "feature_catalogue.yaml",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step4",
    )
    p.add_argument(
        "--no-determinism-check",
        action="store_true",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.config = args.config.resolve()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    step1_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades_csv = (args.trades_csv or (step1_dir / cfg["output"]["trades_csv"])).resolve()
    paths_csv = (args.paths_csv or (step1_dir / cfg["output"]["paths_csv"])).resolve()
    clusters_csv = args.clusters_csv.resolve()
    pass_list_csv = args.pass_list_csv.resolve()
    catalogue_path = args.catalogue.resolve()
    out_dir = args.out_dir.resolve()
    data_dir = (_REPO_ROOT / cfg["data"]["data_dirs"]["1H"]).resolve()

    print("[l_arc_4 step4] === RUN 1 ===", file=sys.stderr)
    sha_run1, run_arts = run_once(
        trades_csv, paths_csv, clusters_csv, pass_list_csv, catalogue_path, data_dir, out_dir, cfg
    )

    sha_run2: Optional[Dict[str, str]] = None
    determinism_gate = "N/A"
    if not args.no_determinism_check:
        print("[l_arc_4 step4] === RUN 2 (determinism) ===", file=sys.stderr)
        sha_run2, _ = run_once(
            trades_csv, paths_csv, clusters_csv, pass_list_csv, catalogue_path, data_dir, out_dir, cfg
        )
        matched = all(
            sha_run1.get(k) == sha_run2.get(k) for k in sorted(set(sha_run1) | set(sha_run2))
        )
        determinism_gate = "PASS" if matched else "FAIL"

    # Config / input sha256s.
    config_paths = {
        "configs/l_arc_4.yaml": str(args.config.relative_to(_REPO_ROOT)),
        "configs/feature_catalogue.yaml": str(catalogue_path.relative_to(_REPO_ROOT)),
        f"{cfg['output']['results_dir']}/trades_all.csv": str(trades_csv.relative_to(_REPO_ROOT)),
        f"{cfg['output']['results_dir']}/trades_paths.csv": str(paths_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step2/clusters_K4.csv": str(clusters_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step3/capturability_pass_list.csv": str(pass_list_csv.relative_to(_REPO_ROOT)),
    }
    config_shas = {
        label: _file_sha256(_REPO_ROOT / Path(path)) for label, path in config_paths.items()
    }

    diag_path = out_dir / "step4_diagnostics.md"
    arc_disp = write_diagnostics(
        diag_path,
        run_arts,
        sha_run1,
        sha_run2,
        determinism_gate,
        config_paths,
        config_shas,
    )

    print(
        f"[l_arc_4 step4] DONE arc_disp={arc_disp} determinism={determinism_gate}",
        file=sys.stderr,
    )
    print(f"[l_arc_4 step4] diagnostics → {diag_path}", file=sys.stderr)

    (out_dir / "step4_env.json").write_text(
        json.dumps(
            {
                "env": _env_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
                "arc_disposition": arc_disp,
                "determinism_gate": determinism_gate,
                "stacking_combinations_used": len(run_arts["stacking_log"]),
                "stacking_budget_remaining": run_arts["stacking_budget_remaining"],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    return 0 if (arc_disp == "PASS" and determinism_gate in ("PASS", "N/A")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
