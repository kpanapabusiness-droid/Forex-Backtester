"""Arc 8 — Step 4 extractability investigation + artefact production.

L_ARC_PROTOCOL v2.3 stack (base v2.1.2 + v2.2 + v2.3 amendments).
§8 + v2.2 §2 (Tier 2 lift cap ≤ 5) + v2.2 §3 (no max-F1 fallback at threshold
sweep) + v2.3 §4 (pre_t_sl_atr_multiplier in D1 policy YAML).

For each Step-3 surviving unit (c1, c3, agg_c1_c3 from
capturability_pass_list.csv):

1. Build success labels under unit's selected SL: success = 1 iff
   final_r ≥ 1.0 in the unit-SL R-frame (path re-evaluated via the
   step3 SL imposition logic).
2. Angle E (entry-time predictability):
   - Step A: full feature set (8 base + Arc 8 entry features ≤ 30 → ≤ 38 total).
     Train Logistic + RF, 5-fold TimeSeriesSplit CV ROC-AUC. If RF AUC ≥ 0.65 → lock.
   - Step B (if A fails): top-5/10/15 RF importance subsets + forward selection.
     If any subset RF AUC ≥ 0.65 → lock.
   - Step C (if A/B fail): stack 2 classifiers (RF + Logistic, intersection-mode).
     Budget ≤ 30 combinations across all archetypes this arc.
3. Angle D1 (deferred identification via bar-offset-t):
   - For t ∈ {1, 2, 3, 4, 5, 10}: compute path-so-far features at t
     (close_r_at_t, mfe_so_far_r_at_t, mae_so_far_r_at_t, bars_in_profit_at_t,
     local_peaks_so_far_at_t, monotonicity_so_far_at_t, velocity_first_t).
     Re-evaluate trade exit under unit SL; exclude trades that exit before t.
     Plus the 8 base entry features. RF only.
   - Smallest-t rule: smallest t with RF AUC ≥ 0.60 AND exclusion ≤ 30%.
4. Pipeline assignment per §8: E only / D1 only / both / dies.
5. Threshold sweep per v2.2 §3: {0.40, 0.50, 0.60, 0.70}. Max precision with
   recall ≥ 0.60. **No max-F1 fallback** — if no threshold satisfies recall ≥ 0.60
   the archetype FAILS Step 4 (§16a).
6. Train final classifier on full data; save artefacts:
   - archetype_<slug>_E_classifier.joblib + _E_filter.yaml (if E passes)
   - archetype_<slug>_D1_classifier.joblib + _D1_policy.yaml (if D1 passes)
     D1 policy YAML includes v2.3 §4 pre_t_sl_atr_multiplier.

Tier 2 lift: deferred — protocol allows ≤ 5 lift candidates per archetype
(v2.2 §2), evaluated at Step 5 WFO. This dispatch produces only Tier 1
(single-classifier baseline) artefacts. Tier 2 lift candidates can be added
in a follow-up dispatch within the v2.2 §2 cap.

Models: RandomForestClassifier(n_estimators=200, max_depth=8,
random_state=42, n_jobs=1) — n_jobs=1 for deterministic output. Logistic:
LogisticRegression(max_iter=1000, random_state=42). CV: 5-fold
TimeSeriesSplit.

Usage:
    py scripts/l_arc_8/step4_extractability.py -c configs/l_arc_8/step4.yaml
    py scripts/l_arc_8/step4_extractability.py -c configs/l_arc_8/step4.yaml --determinism
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the byte-identical SL imposition logic from Step 3.
from scripts.l_arc_8.step3_capturability import _eval_trade_at_sl  # noqa: E402

ATR_PERIOD = 14


# ============================================================
# Base entry features (8 base per §8 Angle E)
# ============================================================


def _wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    hi = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    cl = df["close"].astype(float).to_numpy()
    n = hi.size
    if n == 0:
        return np.array([], dtype=float)
    prev_cl = np.empty(n, dtype=float)
    prev_cl[0] = np.nan
    prev_cl[1:] = cl[:-1]
    tr = np.maximum.reduce([hi - lo, np.abs(hi - prev_cl), np.abs(lo - prev_cl)])
    tr[0] = hi[0] - lo[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.size
    out = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return out
    delta = np.diff(close)
    up = np.maximum(delta, 0.0)
    dn = np.maximum(-delta, 0.0)
    avg_up = float(up[:period].mean())
    avg_dn = float(dn[:period].mean())
    if avg_dn == 0.0:
        out[period] = 100.0
    else:
        rs = avg_up / avg_dn
        out[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        avg_up = (avg_up * (period - 1) + up[i - 1]) / period
        avg_dn = (avg_dn * (period - 1) + dn[i - 1]) / period
        if avg_dn == 0.0:
            out[i] = 100.0
        else:
            rs = avg_up / avg_dn
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


# ============================================================
# Per-pair data cache for base feature computation
# ============================================================


@dataclass
class _PairCache:
    df_4h: pd.DataFrame
    idx_by_ts: Dict[pd.Timestamp, int]
    atr14: np.ndarray
    rsi14: np.ndarray


def _load_pair_4h(pair: str, dir_4h: str) -> pd.DataFrame:
    p = _REPO_ROOT / dir_4h / f"{pair}.csv"
    df = pd.read_csv(p)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _build_pair_cache(pair: str, dir_4h: str) -> _PairCache:
    df = _load_pair_4h(pair, dir_4h)
    close = df["close"].astype(float).to_numpy()
    atr = _wilder_atr(df, ATR_PERIOD)
    rsi = _rsi(close, ATR_PERIOD)
    idx_by_ts = {pd.Timestamp(t): i for i, t in enumerate(df["date"].to_numpy())}
    return _PairCache(df_4h=df, idx_by_ts=idx_by_ts, atr14=atr, rsi14=rsi)


# ============================================================
# §8 Angle E base features (8) — universal, computed from 4H bars
# ============================================================


PIPELINE_E_BASE_FEATURES: List[str] = [
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_to_atr_14",
    "ret_5bar_atr",
    "ret_20bar_atr",
    "pos_in_20bar_range",
    "rsi_14",
]


# Arc 8 PR-HHHL-specific entry features (already in trades_all.csv).
PIPELINE_E_ARC8_FEATURES: List[str] = [
    "num_higher_highs",
    "num_higher_lows",
    "most_recent_sh_age",
    "most_recent_sl_age",
    "hh_range_atr",
    "hl_range_atr",
    "pullback_depth_atr",
    "trigger_body_atr",
    "trigger_close_pos",
    "trigger_break_size_atr",
]


def compute_base_e_features(
    trades_df: pd.DataFrame, pair_caches: Dict[str, _PairCache]
) -> pd.DataFrame:
    """Compute the 8 §8 base entry features for each trade."""
    rows: List[Dict[str, Any]] = []
    for _, t in trades_df.iterrows():
        pair = str(t["pair"])
        cache = pair_caches[pair]
        sig_ts = pd.Timestamp(t["signal_bar_time"])
        i = cache.idx_by_ts.get(sig_ts, -1)
        if i < 0:
            raise ValueError(f"signal_bar_time {sig_ts} missing in {pair} 4H data")
        o = float(cache.df_4h["open"].iloc[i])
        h = float(cache.df_4h["high"].iloc[i])
        lo = float(cache.df_4h["low"].iloc[i])
        c = float(cache.df_4h["close"].iloc[i])
        rng = h - lo
        atr_t = float(cache.atr14[i]) if not math.isnan(cache.atr14[i]) else float("nan")
        rsi_t = float(cache.rsi14[i]) if not math.isnan(cache.rsi14[i]) else float("nan")

        body_to_range = abs(c - o) / rng if rng > 0 else 0.0
        upper_wick = (h - max(o, c)) / rng if rng > 0 else 0.0
        lower_wick = (min(o, c) - lo) / rng if rng > 0 else 0.0
        range_to_atr = rng / atr_t if (atr_t > 0 and math.isfinite(atr_t)) else float("nan")

        # ret_5bar_atr = (close_t - close_{t-5}) / ATR_t
        def _ret(lookback: int) -> float:
            if i < lookback or atr_t <= 0 or not math.isfinite(atr_t):
                return float("nan")
            prev_close = float(cache.df_4h["close"].iloc[i - lookback])
            return (c - prev_close) / atr_t

        ret_5 = _ret(5)
        ret_20 = _ret(20)

        # pos_in_20bar_range = (close_t - low_20) / (high_20 - low_20)
        if i >= 20:
            window = cache.df_4h.iloc[i - 20: i + 1]
            wh = float(window["high"].max())
            wl = float(window["low"].min())
            pos20 = (c - wl) / (wh - wl) if (wh - wl) > 0 else float("nan")
        else:
            pos20 = float("nan")

        rows.append(
            {
                "trade_id": int(t["trade_id"]),
                "body_to_range_ratio": body_to_range,
                "upper_wick_ratio": upper_wick,
                "lower_wick_ratio": lower_wick,
                "range_to_atr_14": range_to_atr,
                "ret_5bar_atr": ret_5,
                "ret_20bar_atr": ret_20,
                "pos_in_20bar_range": pos20,
                "rsi_14": rsi_t,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# §8 Angle D1: path-so-far features at bar offset t
# ============================================================


def compute_d1_features_at_t(
    trades_df: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    t: int,
    unit_selected_sl: float,
    original_sl: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute Angle D1 path-so-far features at bar offset t.

    Returns (features_df, eligible_mask). eligible_mask is True for trades
    whose actual exit (under unit_selected_sl) is at bar offset >= t (i.e.,
    the trade is still open at t and can be classified there).

    Path-so-far features (§8 Angle D1):
      close_r_at_t        — close_r at bar offset t (in unit R-frame)
      mfe_so_far_r_at_t   — running max of high_r up to t (unit R)
      mae_so_far_r_at_t   — running min of low_r up to t (unit R)
      bars_in_profit_at_t — count of bars 0..t with close_r > 0 (unit R)
      local_peaks_so_far_at_t — count of bars 1..t with mfe_so_far > prev
      monotonicity_so_far_at_t — fraction of in-profit bars with non-decreasing close
      velocity_first_t    — mfe_so_far_at_t / max(t, 1)
    """
    feature_cols = [
        "close_r_at_t",
        "mfe_so_far_r_at_t",
        "mae_so_far_r_at_t",
        "bars_in_profit_at_t",
        "local_peaks_so_far_at_t",
        "monotonicity_so_far_at_t",
        "velocity_first_t",
    ]
    rows: List[Dict[str, Any]] = []
    eligible: List[bool] = []
    scale = original_sl / unit_selected_sl

    for _, tr in trades_df.iterrows():
        tid = int(tr["trade_id"])
        path = paths_index[tid]

        # Re-evaluate exit under unit's selected SL.
        eval_ = _eval_trade_at_sl(path, unit_selected_sl, original_sl)
        actual_exit_bar = eval_.truncated_at_bar  # bar_offset at which trade exited

        is_eligible = actual_exit_bar >= t
        eligible.append(is_eligible)

        if not is_eligible or t == 0:
            rows.append(
                {
                    "trade_id": tid,
                    **{c: float("nan") for c in feature_cols},
                }
            )
            continue

        # Slice path bars 0..t (inclusive).
        path_sorted = path.sort_values("bar_offset", kind="mergesort").reset_index(drop=True)
        slice_end = t + 1
        if slice_end > len(path_sorted):
            slice_end = len(path_sorted)
        seg = path_sorted.iloc[:slice_end]
        if len(seg) == 0:
            rows.append(
                {
                    "trade_id": tid,
                    **{c: float("nan") for c in feature_cols},
                }
            )
            continue

        close_orig = seg["close_r"].to_numpy(dtype=float)
        mfe_orig = seg["mfe_so_far_r"].to_numpy(dtype=float)
        mae_orig = seg["mae_so_far_r"].to_numpy(dtype=float)
        high_orig = seg["high_r"].to_numpy(dtype=float)
        low_orig = seg["low_r"].to_numpy(dtype=float)

        # Convert to unit R-frame (scale).
        close_new = close_orig * scale
        # mfe_so_far / mae_so_far in unit frame = scale × original, but we want
        # running max/min in NEW frame, which equals scale × running max/min
        # in original frame (monotone in scale > 0).
        mfe_new = mfe_orig * scale
        mae_new = mae_orig * scale
        high_orig * scale
        low_orig * scale

        # Features at bar t (last bar of slice).
        idx_t = len(seg) - 1
        close_r_at_t = float(close_new[idx_t])
        mfe_at_t = float(mfe_new[idx_t])
        mae_at_t = float(mae_new[idx_t])

        bars_in_profit = int(np.sum(close_new > 0))
        # local_peaks_so_far_at_t: bars 1..t where mfe_so_far[i] > mfe_so_far[i-1]
        local_peaks = (
            int(np.sum(mfe_new[1:] > mfe_new[:-1])) if mfe_new.size >= 2 else 0
        )
        # monotonicity_so_far: fraction of consecutive-in-profit-bar pairs with non-decrease
        in_profit = close_new[close_new > 0]
        if in_profit.size >= 2:
            mono = float(np.mean(in_profit[1:] >= in_profit[:-1]))
        else:
            mono = 0.0
        velocity = mfe_at_t / max(t, 1)

        rows.append(
            {
                "trade_id": tid,
                "close_r_at_t": close_r_at_t,
                "mfe_so_far_r_at_t": mfe_at_t,
                "mae_so_far_r_at_t": mae_at_t,
                "bars_in_profit_at_t": float(bars_in_profit),
                "local_peaks_so_far_at_t": float(local_peaks),
                "monotonicity_so_far_at_t": mono,
                "velocity_first_t": velocity,
            }
        )

    return pd.DataFrame(rows), np.array(eligible, dtype=bool)


# ============================================================
# Success labels — final_r >= 1.0 under unit SL
# ============================================================


def build_unit_labels(
    trades_df: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    unit_selected_sl: float,
    original_sl: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, tr in trades_df.iterrows():
        tid = int(tr["trade_id"])
        path = paths_index[tid]
        eval_ = _eval_trade_at_sl(path, unit_selected_sl, original_sl)
        rows.append(
            {
                "trade_id": tid,
                "entry_time": pd.Timestamp(tr["entry_time"]),
                "final_r_under_unit_sl": eval_.final_r_new,
                "actual_exit_bar_under_unit_sl": eval_.truncated_at_bar,
                "label_success_1R": 1 if eval_.final_r_new >= 1.0 else 0,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Classifier training + CV AUC
# ============================================================


def _make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=1
    )


def _make_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def _cv_auc_rf(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[float, List[float]]:
    """5-fold TimeSeriesSplit CV ROC-AUC for an RF classifier."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=n_splits)
    aucs: List[float] = []
    for fold, (tr_idx, te_idx) in enumerate(tss.split(X)):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            aucs.append(float("nan"))
            continue
        clf = _make_rf()
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        aucs.append(float(roc_auc_score(yte, prob)))
    valid = [a for a in aucs if not math.isnan(a)]
    return (float(np.mean(valid)) if valid else float("nan")), aucs


def _cv_auc_logistic(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[float, List[float]]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=n_splits)
    aucs: List[float] = []
    for fold, (tr_idx, te_idx) in enumerate(tss.split(X)):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            aucs.append(float("nan"))
            continue
        clf = _make_logistic()
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        aucs.append(float(roc_auc_score(yte, prob)))
    valid = [a for a in aucs if not math.isnan(a)]
    return (float(np.mean(valid)) if valid else float("nan")), aucs


# ============================================================
# Angle E gate: A → B → C
# ============================================================


@dataclass
class _AngleEResult:
    pass_gate: bool
    chosen_step: Optional[str]      # "A" | "B-top5" | "B-top10" | "B-top15" | "B-fwd" | "C-stack" | None
    feature_set: List[str]
    rf_auc: float
    logistic_auc: float
    per_fold_rf_aucs: List[float]
    notes: str


def _impute_nans(X_df: pd.DataFrame) -> np.ndarray:
    """Median-impute NaNs to keep RF/Logistic happy."""
    X = X_df.copy()
    for c in X.columns:
        v = X[c]
        if v.isna().any():
            med = float(v.median())
            X[c] = v.fillna(med if math.isfinite(med) else 0.0)
    return X.to_numpy(dtype=float)


def run_angle_e(
    feat_df: pd.DataFrame, y: np.ndarray, all_features: List[str], stack_budget_remaining: int
) -> Tuple[_AngleEResult, int]:
    """Run §8 Angle E A → B → C. Returns (result, updated_stack_budget).

    feat_df has trade_id-aligned rows ordered by entry_time ascending.
    """
    X_full = _impute_nans(feat_df[all_features])

    # Step A: full feature set RF.
    rf_auc_full, fold_aucs_full = _cv_auc_rf(X_full, y)
    log_auc_full, _ = _cv_auc_logistic(X_full, y)
    if rf_auc_full >= 0.65:
        return (
            _AngleEResult(
                pass_gate=True,
                chosen_step="A",
                feature_set=all_features,
                rf_auc=rf_auc_full,
                logistic_auc=log_auc_full,
                per_fold_rf_aucs=fold_aucs_full,
                notes=f"Step A pass: full {len(all_features)}-feature RF AUC {rf_auc_full:.4f} >= 0.65",
            ),
            stack_budget_remaining,
        )

    # Step B: top-k by RF importance + forward selection.
    rf_full = _make_rf()
    rf_full.fit(X_full, y)
    importances = pd.Series(rf_full.feature_importances_, index=all_features)
    imp_sorted = importances.sort_values(ascending=False).index.tolist()

    best_b: Optional[_AngleEResult] = None
    for k in (5, 10, 15):
        sub = imp_sorted[:k]
        X_sub = _impute_nans(feat_df[sub])
        rf_auc_sub, fold_aucs_sub = _cv_auc_rf(X_sub, y)
        log_auc_sub, _ = _cv_auc_logistic(X_sub, y)
        if rf_auc_sub >= 0.65:
            return (
                _AngleEResult(
                    pass_gate=True,
                    chosen_step=f"B-top{k}",
                    feature_set=sub,
                    rf_auc=rf_auc_sub,
                    logistic_auc=log_auc_sub,
                    per_fold_rf_aucs=fold_aucs_sub,
                    notes=f"Step B top-{k} pass: RF AUC {rf_auc_sub:.4f} >= 0.65",
                ),
                stack_budget_remaining,
            )
        if best_b is None or rf_auc_sub > best_b.rf_auc:
            best_b = _AngleEResult(
                pass_gate=False,
                chosen_step=f"B-top{k}",
                feature_set=sub,
                rf_auc=rf_auc_sub,
                logistic_auc=log_auc_sub,
                per_fold_rf_aucs=fold_aucs_sub,
                notes=f"Step B top-{k}: RF AUC {rf_auc_sub:.4f}",
            )

    # Forward selection: start with highest univariate-AUC feature, add greedily.
    univ_aucs: Dict[str, float] = {}
    for f in all_features:
        X_one = _impute_nans(feat_df[[f]])
        try:
            a, _ = _cv_auc_rf(X_one, y)
        except Exception:
            a = float("nan")
        univ_aucs[f] = a
    fwd_seed = max(univ_aucs.items(), key=lambda kv: (kv[1] if math.isfinite(kv[1]) else -1.0))[0]
    fwd_set: List[str] = [fwd_seed]
    fwd_best_auc = univ_aucs[fwd_seed]
    remaining = [f for f in all_features if f != fwd_seed]
    while remaining:
        best_add: Tuple[Optional[str], float] = (None, fwd_best_auc)
        for f in remaining:
            cand_set = fwd_set + [f]
            X_cand = _impute_nans(feat_df[cand_set])
            try:
                a, _ = _cv_auc_rf(X_cand, y)
            except Exception:
                a = float("nan")
            if math.isfinite(a) and a > best_add[1]:
                best_add = (f, a)
        if best_add[0] is None or best_add[1] <= fwd_best_auc + 1e-6:
            break
        fwd_set.append(best_add[0])
        remaining.remove(best_add[0])
        fwd_best_auc = best_add[1]
        if fwd_best_auc >= 0.65:
            break

    X_fwd = _impute_nans(feat_df[fwd_set])
    rf_auc_fwd, fold_aucs_fwd = _cv_auc_rf(X_fwd, y)
    log_auc_fwd, _ = _cv_auc_logistic(X_fwd, y)
    if rf_auc_fwd >= 0.65:
        return (
            _AngleEResult(
                pass_gate=True,
                chosen_step="B-fwd",
                feature_set=fwd_set,
                rf_auc=rf_auc_fwd,
                logistic_auc=log_auc_fwd,
                per_fold_rf_aucs=fold_aucs_fwd,
                notes=f"Step B forward-selection pass: {len(fwd_set)} features, RF AUC {rf_auc_fwd:.4f}",
            ),
            stack_budget_remaining,
        )
    if best_b is None or rf_auc_fwd > best_b.rf_auc:
        best_b = _AngleEResult(
            pass_gate=False,
            chosen_step="B-fwd",
            feature_set=fwd_set,
            rf_auc=rf_auc_fwd,
            logistic_auc=log_auc_fwd,
            per_fold_rf_aucs=fold_aucs_fwd,
            notes=f"Step B fwd-sel: {len(fwd_set)} features, RF AUC {rf_auc_fwd:.4f}",
        )

    # Step C: stack 2 classifiers, intersection. Budget ≤ 30 combinations.
    if stack_budget_remaining <= 0:
        return (
            _AngleEResult(
                pass_gate=False,
                chosen_step=None,
                feature_set=best_b.feature_set if best_b else all_features,
                rf_auc=best_b.rf_auc if best_b else rf_auc_full,
                logistic_auc=best_b.logistic_auc if best_b else log_auc_full,
                per_fold_rf_aucs=best_b.per_fold_rf_aucs if best_b else fold_aucs_full,
                notes="Step A and B failed; Step C budget exhausted at arc level — angle_E DIES",
            ),
            stack_budget_remaining,
        )

    # Heuristic 2-classifier stacks: top-5 (RF) ∩ logistic-top-5, and a few combos.
    # Each combination costs one budget unit.
    best_c: Optional[_AngleEResult] = None
    combos: List[Tuple[str, str, List[str], List[str]]] = [
        ("rf_top5", "logistic_top5", imp_sorted[:5], imp_sorted[:5]),
        ("rf_top10", "logistic_top10", imp_sorted[:10], imp_sorted[:10]),
        ("rf_top5", "logistic_top10", imp_sorted[:5], imp_sorted[:10]),
        ("rf_fwd", "logistic_fwd", fwd_set, fwd_set),
    ]
    used = 0
    for _name1, _name2, feats1, feats2 in combos:
        if stack_budget_remaining - used <= 0:
            break
        used += 1
        X1 = _impute_nans(feat_df[feats1])
        X2 = _impute_nans(feat_df[feats2])
        # Compute per-fold intersection AUC via CV.
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
        tss = TimeSeriesSplit(n_splits=5)
        stack_aucs: List[float] = []
        for tr_idx, te_idx in tss.split(X1):
            ytr, yte = y[tr_idx], y[te_idx]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                stack_aucs.append(float("nan"))
                continue
            rf = _make_rf()
            rf.fit(X1[tr_idx], ytr)
            p_rf = rf.predict_proba(X1[te_idx])[:, 1]
            lg = _make_logistic()
            lg.fit(X2[tr_idx], ytr)
            p_lg = lg.predict_proba(X2[te_idx])[:, 1]
            # Intersection-mode: min(p_rf, p_lg) as combined score.
            p_combined = np.minimum(p_rf, p_lg)
            try:
                stack_aucs.append(float(roc_auc_score(yte, p_combined)))
            except Exception:
                stack_aucs.append(float("nan"))
        valid = [a for a in stack_aucs if math.isfinite(a)]
        mean_auc = float(np.mean(valid)) if valid else float("nan")
        if mean_auc >= 0.65:
            return (
                _AngleEResult(
                    pass_gate=True,
                    chosen_step="C-stack",
                    feature_set=sorted(set(feats1) | set(feats2)),
                    rf_auc=mean_auc,
                    logistic_auc=float("nan"),
                    per_fold_rf_aucs=stack_aucs,
                    notes=f"Step C stack pass: feats1={len(feats1)}, feats2={len(feats2)} intersection AUC {mean_auc:.4f}",
                ),
                stack_budget_remaining - used,
            )
        if best_c is None or mean_auc > best_c.rf_auc:
            best_c = _AngleEResult(
                pass_gate=False,
                chosen_step="C-stack",
                feature_set=sorted(set(feats1) | set(feats2)),
                rf_auc=mean_auc,
                logistic_auc=float("nan"),
                per_fold_rf_aucs=stack_aucs,
                notes=f"Step C combo: feats1={len(feats1)}, feats2={len(feats2)}, intersection AUC {mean_auc:.4f}",
            )

    final_best = max(
        [r for r in [best_b, best_c] if r is not None],
        key=lambda r: r.rf_auc if math.isfinite(r.rf_auc) else -1.0,
        default=None,
    )
    if final_best is None:
        final_best = _AngleEResult(
            pass_gate=False,
            chosen_step=None,
            feature_set=all_features,
            rf_auc=rf_auc_full,
            logistic_auc=log_auc_full,
            per_fold_rf_aucs=fold_aucs_full,
            notes="Step A+B+C all fail; no best-of",
        )
    final_best.pass_gate = False
    final_best.notes += " — angle_E DIES (no step clears 0.65)"
    return final_best, stack_budget_remaining - used


# ============================================================
# Angle D1: bar-offset-t sweep + smallest-t rule
# ============================================================


@dataclass
class _AngleD1AtT:
    t: int
    n_eligible: int
    exclusion_pct: float
    rf_auc: float
    per_fold_aucs: List[float]


@dataclass
class _AngleD1Result:
    pass_gate: bool
    chosen_t: Optional[int]
    per_t: Dict[int, _AngleD1AtT]
    notes: str


D1_T_SWEEP: List[int] = [1, 2, 3, 4, 5, 10]


def run_angle_d1(
    trades_df: pd.DataFrame,
    paths_index: Dict[int, pd.DataFrame],
    base_e_df: pd.DataFrame,
    y_full: np.ndarray,
    unit_pool_size: int,
    unit_selected_sl: float,
    original_sl: float,
) -> _AngleD1Result:
    """Sweep t ∈ {1..5, 10}. RF only. Smallest-t rule.

    Per protocol §8 Angle D1: exclusion = fraction of *archetype pool*
    that exits before bar t under the unit's selected SL (i.e., trades
    whose actual exit bar < t). NOT fraction of the full Step 1 pool.
    """
    per_t: Dict[int, _AngleD1AtT] = {}
    for t in D1_T_SWEEP:
        d1_feat_df, eligible = compute_d1_features_at_t(
            trades_df, paths_index, t, unit_selected_sl, original_sl
        )
        n_elig = int(eligible.sum())
        if unit_pool_size == 0:
            exclusion_pct = 0.0
        else:
            # Exclusion = trades_exited_before_t / archetype_pool.
            exclusion_pct = float(1.0 - n_elig / unit_pool_size)

        if n_elig < 50 or len(np.unique(y_full[eligible])) < 2:
            per_t[t] = _AngleD1AtT(
                t=t,
                n_eligible=n_elig,
                exclusion_pct=exclusion_pct,
                rf_auc=float("nan"),
                per_fold_aucs=[],
            )
            continue

        # Assemble features: 8 base entry features + 7 path-so-far features.
        d1_path_cols = [
            "close_r_at_t",
            "mfe_so_far_r_at_t",
            "mae_so_far_r_at_t",
            "bars_in_profit_at_t",
            "local_peaks_so_far_at_t",
            "monotonicity_so_far_at_t",
            "velocity_first_t",
        ]
        feat_t = base_e_df.merge(d1_feat_df, on="trade_id").set_index("trade_id")
        feat_t = feat_t.loc[trades_df["trade_id"].to_numpy()].reset_index(drop=True)
        sub = feat_t[PIPELINE_E_BASE_FEATURES + d1_path_cols].iloc[eligible].copy()
        Xt = _impute_nans(sub)
        yt = y_full[eligible]
        rf_auc_t, fold_aucs_t = _cv_auc_rf(Xt, yt)
        per_t[t] = _AngleD1AtT(
            t=t,
            n_eligible=n_elig,
            exclusion_pct=exclusion_pct,
            rf_auc=rf_auc_t,
            per_fold_aucs=fold_aucs_t,
        )

    # Smallest-t rule: smallest t with AUC >= 0.60 AND exclusion <= 0.30.
    chosen: Optional[int] = None
    for t in D1_T_SWEEP:
        ev = per_t[t]
        if (
            math.isfinite(ev.rf_auc)
            and ev.rf_auc >= 0.60
            and ev.exclusion_pct <= 0.30
        ):
            chosen = t
            break

    return _AngleD1Result(
        pass_gate=chosen is not None,
        chosen_t=chosen,
        per_t=per_t,
        notes=(
            f"Chosen t={chosen} per smallest-t rule"
            if chosen is not None
            else "No t clears AUC>=0.60 AND exclusion<=0.30"
        ),
    )


# ============================================================
# Threshold sweep — v2.2 §3 (no max-F1 fallback)
# ============================================================


@dataclass
class _ThresholdResult:
    pass_gate: bool
    chosen_threshold: Optional[float]
    chosen_precision: float
    chosen_recall: float
    swept_threshold: List[Dict[str, Any]]
    notes: str


def threshold_sweep(
    X: np.ndarray, y: np.ndarray, model_factory, sweep: List[float] = (0.40, 0.50, 0.60, 0.70)
) -> _ThresholdResult:
    """Threshold sweep per v2.2 §3. Train on 80% time-prefix, evaluate on 20% holdout."""
    from sklearn.metrics import precision_score, recall_score

    n = len(y)
    cut = int(n * 0.8)
    if cut < 1 or n - cut < 1:
        return _ThresholdResult(
            pass_gate=False,
            chosen_threshold=None,
            chosen_precision=float("nan"),
            chosen_recall=float("nan"),
            swept_threshold=[],
            notes="Insufficient data for threshold sweep",
        )
    Xtr, ytr = X[:cut], y[:cut]
    Xte, yte = X[cut:], y[cut:]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return _ThresholdResult(
            pass_gate=False,
            chosen_threshold=None,
            chosen_precision=float("nan"),
            chosen_recall=float("nan"),
            swept_threshold=[],
            notes="Single-class train or test fold",
        )

    clf = model_factory()
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]

    rows: List[Dict[str, Any]] = []
    for thr in sweep:
        pred = (prob >= thr).astype(int)
        if pred.sum() == 0:
            p = 0.0
            r = 0.0
        else:
            p = float(precision_score(yte, pred, zero_division=0))
            r = float(recall_score(yte, pred, zero_division=0))
        rows.append({"threshold": thr, "precision": p, "recall": r, "admit_pct": float(pred.mean())})

    # v2.2 §3: pick max-precision threshold with recall >= 0.60. NO max-F1 fallback.
    valid = [r for r in rows if r["recall"] >= 0.60]
    if not valid:
        return _ThresholdResult(
            pass_gate=False,
            chosen_threshold=None,
            chosen_precision=max(r["precision"] for r in rows),
            chosen_recall=max(r["recall"] for r in rows),
            swept_threshold=rows,
            notes="v2.2 §3: no threshold satisfies recall >= 0.60 — archetype FAILS Step 4",
        )
    chosen = max(valid, key=lambda r: r["precision"])
    return _ThresholdResult(
        pass_gate=True,
        chosen_threshold=float(chosen["threshold"]),
        chosen_precision=float(chosen["precision"]),
        chosen_recall=float(chosen["recall"]),
        swept_threshold=rows,
        notes=f"Chosen threshold {chosen['threshold']:.2f} (precision {chosen['precision']:.4f}, recall {chosen['recall']:.4f})",
    )


# ============================================================
# Output writers
# ============================================================


def _fmt_g(x: Any) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return ""
    except Exception:
        return str(x)
    return f"{xf:.10g}"


def _slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("+", "and")
    )


def write_predictability_csv(out_path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt_g(r.get(c)) if isinstance(r.get(c), float) else (r.get(c, "") if r.get(c) is not None else "") for c in cols])


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Driver
# ============================================================


def _build_paths_index(paths_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    paths_sorted = paths_df.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    for tid, g in paths_sorted.groupby("trade_id", sort=True):
        out[int(tid)] = g.reset_index(drop=True)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 8 Step 4 — extractability.")
    ap.add_argument("-c", "--config", type=Path, default=Path("configs/l_arc_8/step4.yaml"))
    ap.add_argument("--determinism", action="store_true", help="run twice, compare hashes")
    args = ap.parse_args(argv)
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (_REPO_ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    in_cfg = cfg["input"]
    out_cfg = cfg["output"]
    step1_dir = _REPO_ROOT / in_cfg["step1_dir"]
    step3_dir = _REPO_ROOT / in_cfg["step3_dir"]
    out_dir = _REPO_ROOT / out_cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.read_csv(step1_dir / in_cfg["trades_csv"])
    paths_df = pd.read_csv(step1_dir / in_cfg["paths_csv"])
    pass_list = pd.read_csv(step3_dir / in_cfg["capturability_pass_list_csv"])
    clusters_df = pd.read_csv(_REPO_ROOT / in_cfg["step2_dir"] / in_cfg["clusters_csv"])

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["signal_bar_time"] = pd.to_datetime(trades_df["signal_bar_time"])
    len(trades_df)

    paths_index = _build_paths_index(paths_df)

    # Build base 4H features once for the entire pool (reused per unit).
    pairs = sorted(trades_df["pair"].unique())
    dir_4h = cfg["data"]["dir_4h"]
    pair_caches = {p: _build_pair_cache(p, dir_4h) for p in pairs}
    base_e_df = compute_base_e_features(trades_df, pair_caches)
    # Merge Arc 8 entry features (already in trades_all.csv).
    base_e_df = base_e_df.merge(
        trades_df[["trade_id"] + PIPELINE_E_ARC8_FEATURES], on="trade_id", how="left"
    )

    ALL_E_FEATURES = PIPELINE_E_BASE_FEATURES + PIPELINE_E_ARC8_FEATURES
    print(f"[l_arc_8 step4] Pipeline E feature count: {len(ALL_E_FEATURES)} (8 base + {len(PIPELINE_E_ARC8_FEATURES)} arc8)", file=sys.stderr)

    stack_budget = int(cfg.get("stack_budget", 30))
    original_sl = float(cfg.get("original_sl_atr_mult", 2.0))

    predictability_e_rows: List[Dict[str, Any]] = []
    predictability_d1_rows: List[Dict[str, Any]] = []
    extractability_pass_rows: List[Dict[str, Any]] = []

    for _, unit_row in pass_list.iterrows():
        unit_id = str(unit_row["unit_id"])
        unit_type = str(unit_row["type"])
        final_label = str(unit_row["final_archetype_label"])
        selected_sl = float(unit_row["selected_SL_atr_mult"])

        # Build the trade pool for this unit.
        if unit_type == "cluster":
            cid = int(unit_id[1:])  # "c1" → 1
            unit_trade_ids = clusters_df[clusters_df["cluster_id"] == cid]["trade_id"].tolist()
        elif unit_type == "aggregate":
            # Parse "agg_c1_c3" → [1, 3]
            cids = [int(p[1:]) for p in unit_id.replace("agg_", "").split("_")]
            unit_trade_ids = clusters_df[clusters_df["cluster_id"].isin(cids)]["trade_id"].tolist()
        else:
            print(f"[l_arc_8 step4] unknown unit type {unit_type}; skipping", file=sys.stderr)
            continue

        unit_trades = trades_df[trades_df["trade_id"].isin(unit_trade_ids)].copy()
        unit_trades = unit_trades.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
        n_unit = len(unit_trades)

        # Build labels under unit's selected SL.
        labels_df = build_unit_labels(unit_trades, paths_index, selected_sl, original_sl)
        labels_df = labels_df.set_index("trade_id").loc[unit_trades["trade_id"].to_numpy()].reset_index()
        y = labels_df["label_success_1R"].to_numpy(dtype=int)
        positive_rate = float(np.mean(y))

        # Get the base+arc8 entry features for these trades, ordered chronologically.
        unit_feat = base_e_df.merge(
            unit_trades[["trade_id", "entry_time"]].rename(columns={"entry_time": "_entry_time"}),
            on="trade_id",
            how="inner",
        )
        unit_feat = unit_feat.sort_values("_entry_time", kind="mergesort").reset_index(drop=True)

        print(
            f"[l_arc_8 step4] unit={unit_id} n={n_unit} pos_rate={positive_rate:.3f} "
            f"SL={selected_sl}×ATR final_label={final_label!r}",
            file=sys.stderr,
        )

        # Angle E.
        e_result, stack_budget = run_angle_e(unit_feat, y, ALL_E_FEATURES, stack_budget)
        predictability_e_rows.append(
            {
                "unit_id": unit_id,
                "type": unit_type,
                "final_archetype_label": final_label,
                "selected_SL_atr_mult": selected_sl,
                "n": n_unit,
                "positive_rate": positive_rate,
                "rf_auc": e_result.rf_auc,
                "logistic_auc": e_result.logistic_auc,
                "chosen_step": e_result.chosen_step or "",
                "feature_count": len(e_result.feature_set),
                "pass_065_gate": int(e_result.pass_gate),
                "per_fold_rf_aucs": ";".join(f"{a:.4f}" if math.isfinite(a) else "nan" for a in e_result.per_fold_rf_aucs),
                "notes": e_result.notes,
            }
        )

        # Angle D1. unit_pool_size = n_unit (denominator for exclusion_pct).
        d1_result = run_angle_d1(
            unit_trades, paths_index, base_e_df, y, n_unit, selected_sl, original_sl
        )
        for t in D1_T_SWEEP:
            ev = d1_result.per_t.get(t)
            predictability_d1_rows.append(
                {
                    "unit_id": unit_id,
                    "type": unit_type,
                    "final_archetype_label": final_label,
                    "selected_SL_atr_mult": selected_sl,
                    "t": t,
                    "n_eligible": ev.n_eligible if ev else 0,
                    "exclusion_pct": ev.exclusion_pct if ev else 1.0,
                    "rf_auc": ev.rf_auc if ev else float("nan"),
                    "is_chosen_t": 1 if d1_result.chosen_t == t else 0,
                    "per_fold_aucs": (
                        ";".join(f"{a:.4f}" if math.isfinite(a) else "nan" for a in ev.per_fold_aucs)
                        if ev else ""
                    ),
                }
            )

        # Threshold sweep + artefact production.
        pipelines_clearing: List[str] = []
        e_threshold_result: Optional[_ThresholdResult] = None
        d1_threshold_result: Optional[_ThresholdResult] = None

        if e_result.pass_gate:
            X_for_thresh = _impute_nans(unit_feat[e_result.feature_set])
            e_threshold_result = threshold_sweep(X_for_thresh, y, _make_rf)
            if e_threshold_result.pass_gate:
                pipelines_clearing.append("E")
                # Train final E classifier on full data and save.
                from joblib import dump
                final_clf = _make_rf()
                final_clf.fit(X_for_thresh, y)
                slug = _slug(final_label) if final_label else unit_id
                joblib_path = out_dir / f"archetype_{slug}_{unit_id}_E_classifier.joblib"
                dump(final_clf, joblib_path)
                filter_yaml_path = out_dir / f"archetype_{slug}_{unit_id}_E_filter.yaml"
                filter_yaml_path.write_text(
                    yaml.safe_dump(
                        {
                            "archetype_label": final_label,
                            "unit_id": unit_id,
                            "unit_type": unit_type,
                            "selected_SL_atr_mult": selected_sl,
                            "feature_set": e_result.feature_set,
                            "threshold": e_threshold_result.chosen_threshold,
                            "precision_holdout": e_threshold_result.chosen_precision,
                            "recall_holdout": e_threshold_result.chosen_recall,
                            "cv_rf_auc": e_result.rf_auc,
                            "chosen_step": e_result.chosen_step,
                            "classifier_file": joblib_path.name,
                            "model": "RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)",
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

        if d1_result.pass_gate:
            t_chosen = d1_result.chosen_t
            # Recompute features at chosen t for the eligible pool.
            d1_feat_t, eligible = compute_d1_features_at_t(
                unit_trades, paths_index, t_chosen, selected_sl, original_sl
            )
            d1_path_cols = [
                "close_r_at_t",
                "mfe_so_far_r_at_t",
                "mae_so_far_r_at_t",
                "bars_in_profit_at_t",
                "local_peaks_so_far_at_t",
                "monotonicity_so_far_at_t",
                "velocity_first_t",
            ]
            feat_t = base_e_df.merge(d1_feat_t, on="trade_id")
            feat_t = feat_t.set_index("trade_id").loc[unit_trades["trade_id"].to_numpy()].reset_index()
            sub = feat_t[PIPELINE_E_BASE_FEATURES + d1_path_cols].iloc[eligible]
            X_d1 = _impute_nans(sub)
            y_d1 = y[eligible]
            d1_threshold_result = threshold_sweep(X_d1, y_d1, _make_rf)
            if d1_threshold_result.pass_gate:
                pipelines_clearing.append("D1")
                from joblib import dump
                final_clf = _make_rf()
                final_clf.fit(X_d1, y_d1)
                slug = _slug(final_label) if final_label else unit_id
                joblib_path = out_dir / f"archetype_{slug}_{unit_id}_D1_classifier.joblib"
                dump(final_clf, joblib_path)
                policy_yaml_path = out_dir / f"archetype_{slug}_{unit_id}_D1_policy.yaml"
                policy_yaml_path.write_text(
                    yaml.safe_dump(
                        {
                            "archetype_label": final_label,
                            "unit_id": unit_id,
                            "unit_type": unit_type,
                            "selected_SL_atr_mult": selected_sl,
                            "pre_t_sl_atr_multiplier": selected_sl,  # v2.3 §4
                            "chosen_t": t_chosen,
                            "feature_set": PIPELINE_E_BASE_FEATURES + d1_path_cols,
                            "threshold": d1_threshold_result.chosen_threshold,
                            "precision_holdout": d1_threshold_result.chosen_precision,
                            "recall_holdout": d1_threshold_result.chosen_recall,
                            "cv_rf_auc_at_chosen_t": d1_result.per_t[t_chosen].rf_auc,
                            "exclusion_pct_at_chosen_t": d1_result.per_t[t_chosen].exclusion_pct,
                            "classifier_file": joblib_path.name,
                            "model": "RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)",
                            "exit_policy_row": "§11 row 5: V-shape recovery — after bar N confirms reversal, standard trail",
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

        # Update predictability rows with threshold-sweep outcome.
        predictability_e_rows[-1]["threshold_sweep_pass"] = (
            int(e_threshold_result.pass_gate) if e_threshold_result is not None else 0
        )
        predictability_e_rows[-1]["chosen_threshold"] = (
            e_threshold_result.chosen_threshold if e_threshold_result is not None else None
        )
        predictability_e_rows[-1]["threshold_sweep_notes"] = (
            e_threshold_result.notes if e_threshold_result is not None else "skipped — gate fail"
        )

        # Pipeline assignment per §8.
        if pipelines_clearing:
            assigned = "+".join(pipelines_clearing)
            extractability_pass_rows.append(
                {
                    "unit_id": unit_id,
                    "type": unit_type,
                    "final_archetype_label": final_label,
                    "selected_SL_atr_mult": selected_sl,
                    "pre_t_sl_atr_multiplier": selected_sl if "D1" in pipelines_clearing else "",
                    "pipeline_assignment": assigned,
                    "e_rf_auc": e_result.rf_auc,
                    "e_chosen_step": e_result.chosen_step or "",
                    "e_threshold": e_threshold_result.chosen_threshold if e_threshold_result and e_threshold_result.pass_gate else "",
                    "d1_chosen_t": d1_result.chosen_t if d1_result.chosen_t is not None else "",
                    "d1_rf_auc": (
                        d1_result.per_t[d1_result.chosen_t].rf_auc
                        if d1_result.chosen_t is not None
                        else ""
                    ),
                    "d1_threshold": (
                        d1_threshold_result.chosen_threshold
                        if d1_threshold_result and d1_threshold_result.pass_gate
                        else ""
                    ),
                }
            )

    # Write outputs.
    write_predictability_csv(
        out_dir / "predictability_angle_E.csv",
        predictability_e_rows,
        [
            "unit_id", "type", "final_archetype_label", "selected_SL_atr_mult",
            "n", "positive_rate", "rf_auc", "logistic_auc", "chosen_step",
            "feature_count", "pass_065_gate", "per_fold_rf_aucs",
            "threshold_sweep_pass", "chosen_threshold", "threshold_sweep_notes",
            "notes",
        ],
    )
    write_predictability_csv(
        out_dir / "predictability_angle_D1.csv",
        predictability_d1_rows,
        [
            "unit_id", "type", "final_archetype_label", "selected_SL_atr_mult",
            "t", "n_eligible", "exclusion_pct", "rf_auc", "is_chosen_t",
            "per_fold_aucs",
        ],
    )
    write_predictability_csv(
        out_dir / "extractability_pass_list.csv",
        extractability_pass_rows,
        [
            "unit_id", "type", "final_archetype_label", "selected_SL_atr_mult",
            "pre_t_sl_atr_multiplier", "pipeline_assignment",
            "e_rf_auc", "e_chosen_step", "e_threshold",
            "d1_chosen_t", "d1_rf_auc", "d1_threshold",
        ],
    )

    # Write summary.
    summary = {
        "phase": cfg.get("phase"),
        "protocol_version": "v2.3 stack (base v2.1.2 + v2.2 + v2.3)",
        "n_units_evaluated": int(len(pass_list)),
        "n_units_pass_step4": int(len(extractability_pass_rows)),
        "stack_budget_remaining": int(stack_budget),
        "sha256": {
            "predictability_angle_E.csv": _file_sha256(out_dir / "predictability_angle_E.csv"),
            "predictability_angle_D1.csv": _file_sha256(out_dir / "predictability_angle_D1.csv"),
            "extractability_pass_list.csv": _file_sha256(out_dir / "extractability_pass_list.csv"),
        },
    }
    (out_dir / "STEP4_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"[l_arc_8 step4] DONE. units_evaluated={summary['n_units_evaluated']} "
        f"units_pass={summary['n_units_pass_step4']} "
        f"stack_budget_left={summary['stack_budget_remaining']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
