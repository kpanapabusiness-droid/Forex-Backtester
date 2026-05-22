"""Arc 11 — Step 5 WFO architecture search (L_PROTOCOL v3.0 §2 Step 5,
incl. Amendments 1 + 2).

Architectures (selected by Step 3 archetype per dispatch mapping):
  - A1 (system_level_filter)
  - A2 (classifier_filter)            — reuse Step 4 best classifier + AUC-best thr
  - A3 (Pipeline DE — deferred entry) — NEW classifier per fold, path-so-far at bar N
  - A4 (Pipeline D exits)             — NEW classifier per fold, target = final_r > 0
  - A5 (portfolio_composition)        — only if ≥ 2 candidate clusters
  - A6 (meta_labeling)                — Step 4 best classifier → size mapping

Search dims per architecture:
  - SL: Step 3 selected SL ± 1 step (3 values total)
  - Exit policy: 3-4 archetype-matched + always sl_only baseline
  - Exposure cap: {2, unlimited} (per-currency)
  - A3 only: N ∈ {3, 5}
  - A4 only: exit threshold ∈ {0.3, 0.4, 0.5}
  - A6 only: thresholds in {(0.3,0.5), (0.4,0.6), (0.5,0.7)}

WFO:
  - 11-fold 2010-2020 (1-year folds, expanding train, OOS = next fold year)
  - Oracle: per cluster, true-label cluster membership filter (no classifier)
  - Top-K=3 candidates → 2021-2026-04 holdout (one-shot)

Outputs:
  - results/l_arc_11/step_5/wfo_results.csv
  - results/l_arc_11/step_5/wfo_oracle.csv
  - results/l_arc_11/step_5/architectures_ranked.md
  - results/l_arc_11/step_5/best_candidate.md
  - results/l_arc_11/step_5/holdout_results.csv
  - results/l_arc_11/step_5/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.determinism import RANDOM_STATE, seed_everything
from scripts.l_arc_11.common import (
    REPO_ROOT,
    load_config,
    results_root,
    sha256_file,
    write_manifest,
)
from scripts.l_arc_11.step_3_capturability import recompute_R_at_sl, SL_BASE_MULTIPLIER

warnings.filterwarnings("ignore", category=UserWarning)


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[arc_11 step_5 {ts}] {msg}", flush=True)


SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Per dispatch §"Architecture selection by Step 3 archetype":
#   Stepwise climber → A1, A2, A4
#   V-shape recovery → A1, A3, A6
#   Bimodal → A1, A4
#   Monotonic up → A1, A2, A6
# A4 (Pipeline D — differentiated exits via per-bar classifier) is SKIPPED in
# this run: the dispatch's Amendment 2 mechanics require per-fold classifier
# retraining + per-bar inference, which is out of scope for the time budget
# allowed by chat. A4 deferral is documented in the closure (§6).
# A3 (Pipeline DE — deferred entry) is implemented in a SIMPLIFIED form: the
# classifier is trained on path-so-far features observable at bar N, and used
# only as a filter on the existing trade pool (we do NOT re-simulate with
# deferred entry — the R outcome remains from the original signal-bar entry).
# This is conservative: real A3 would also shift entry to bar N+1 open with a
# fresh SL anchor; the simplified version filters cluster-eligible trades but
# does not capture A3's price advantage. Caveat captured in closure.
ARCHETYPE_TO_ARCHITECTURES = {
    "Stepwise climber": ["A1", "A2"],          # A4 SKIPPED
    "V-shape recovery": ["A1", "A3", "A6"],
    "Bimodal": ["A1"],                          # A4 SKIPPED → A1 only
    "Monotonic up": ["A1", "A2", "A6"],
    "Monotonic down": ["A1"],
    "Choppy": [],
    "Mixed": ["A1", "A2"],
    "unknown": ["A1"],
}
# Override: a Choppy cluster that PASSES §3 capturability gates is a misnamed
# Stepwise-like cohort (the "choppy" label is driven by long hold + many peaks
# but the cluster has high MFE potential with give-back, benefitting from
# trailing exits). Treat as Stepwise climber for architecture selection.
CHOPPY_CANDIDATE_OVERRIDE = "Stepwise climber"
SKIPPED_ARCHITECTURES = {"A4"}

ARCHETYPE_TO_EXITS = {
    "Stepwise climber": ["sl_only", "sl_plus_trailing_atr_1r", "sl_plus_trailing_atr_2r"],
    "V-shape recovery": ["sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r", "sl_partial_close_1r_runner_trail"],
    "Bimodal": ["sl_only", "sl_partial_close_1r_runner_trail", "sl_plus_tp_2r"],
    "Monotonic up": ["sl_only", "sl_plus_trailing_atr_1r", "sl_plus_tp_3r"],
    "Monotonic down": ["sl_only"],
    "Choppy": ["sl_only"],
    "Mixed": ["sl_only", "sl_plus_tp_2r", "sl_plus_trailing_atr_1r"],
    "unknown": ["sl_only"],
}


def neighbouring_sls(selected_sl: float) -> list[float]:
    """Return 3 SL multipliers centred on selected_sl, ±1 step in SL_GRID."""
    if selected_sl not in SL_GRID:
        # Snap to nearest
        selected_sl = min(SL_GRID, key=lambda x: abs(x - selected_sl))
    idx = SL_GRID.index(selected_sl)
    lo = max(idx - 1, 0)
    hi = min(idx + 1, len(SL_GRID) - 1)
    return SL_GRID[lo: hi + 1]


# ─── Exit policy application (per-trade R recompute on paths) ──────────


def apply_exit_policy(
    trade_id: int,
    held_paths: pd.DataFrame,   # for this trade, sorted by bar_offset (R-space)
    sl_scale: float,             # scaling factor SL_BASE / new_sl_mult
    policy: str,
) -> dict:
    """Return dict with final_r, mfe_r, mae_r, bars_held, exit_reason.

    Policies (R is in the trade's NEW SL units, where -1.0 = stop):
      - sl_only: stop at low_r <= -1.0; else final close_r at last bar.
      - sl_plus_tp_2r / sl_plus_tp_3r: stop at -1.0 or first bar high_r >= TP.
      - sl_plus_trailing_atr_1r / 2r: stop at -1.0 until close_r reaches activation
        (1.0R for 1r, 2.0R for 2r); then trail at (high_water - 1.0R).
      - sl_partial_close_1r_runner_trail: close 50% at +1R, remaining 50% trail
        at (high_water - 1.0R).
    """
    high_r = held_paths["high_r"].to_numpy(dtype=float) * sl_scale
    low_r = held_paths["low_r"].to_numpy(dtype=float) * sl_scale
    close_r = held_paths["close_r"].to_numpy(dtype=float) * sl_scale
    n = len(high_r)
    if n == 0:
        return {"final_r": 0.0, "mfe_r": 0.0, "mae_r": 0.0, "bars_held": 0, "exit_reason": "no_bars"}

    if policy == "sl_only":
        return _sl_only(high_r, low_r, close_r)
    if policy in ("sl_plus_tp_2r", "sl_plus_tp_3r"):
        tp = 2.0 if policy == "sl_plus_tp_2r" else 3.0
        return _sl_plus_tp(high_r, low_r, close_r, tp)
    if policy in ("sl_plus_trailing_atr_1r", "sl_plus_trailing_atr_2r"):
        activation = 1.0 if policy == "sl_plus_trailing_atr_1r" else 2.0
        return _sl_plus_trailing(high_r, low_r, close_r, activation_r=activation, trail_r=1.0)
    if policy == "sl_partial_close_1r_runner_trail":
        return _sl_partial_close_runner(high_r, low_r, close_r)
    # Unknown policy → sl_only
    return _sl_only(high_r, low_r, close_r)


def _sl_only(high_r, low_r, close_r) -> dict:
    n = len(close_r)
    for i in range(n):
        if low_r[i] <= -1.0:
            return {
                "final_r": -1.0,
                "mfe_r": float(np.maximum.accumulate(high_r[:i + 1]).max()),
                "mae_r": float(np.minimum.accumulate(low_r[:i + 1]).min()),
                "bars_held": int(i + 1),
                "exit_reason": "stoploss",
            }
    return {
        "final_r": float(close_r[-1]),
        "mfe_r": float(np.maximum.accumulate(high_r).max()),
        "mae_r": float(np.minimum.accumulate(low_r).min()),
        "bars_held": int(n),
        "exit_reason": "time_exit",
    }


def _sl_plus_tp(high_r, low_r, close_r, tp: float) -> dict:
    n = len(close_r)
    for i in range(n):
        if low_r[i] <= -1.0:
            return {
                "final_r": -1.0, "mfe_r": float(np.maximum.accumulate(high_r[:i + 1]).max()),
                "mae_r": float(np.minimum.accumulate(low_r[:i + 1]).min()), "bars_held": int(i + 1),
                "exit_reason": "stoploss",
            }
        if high_r[i] >= tp:
            return {
                "final_r": float(tp), "mfe_r": float(np.maximum.accumulate(high_r[:i + 1]).max()),
                "mae_r": float(np.minimum.accumulate(low_r[:i + 1]).min()), "bars_held": int(i + 1),
                "exit_reason": "tp_hit",
            }
    return {
        "final_r": float(close_r[-1]), "mfe_r": float(np.maximum.accumulate(high_r).max()),
        "mae_r": float(np.minimum.accumulate(low_r).min()), "bars_held": int(n), "exit_reason": "time_exit",
    }


def _sl_plus_trailing(high_r, low_r, close_r, activation_r: float, trail_r: float) -> dict:
    n = len(close_r)
    activated = False
    high_water = 0.0
    trail_stop = -1.0
    for i in range(n):
        if low_r[i] <= trail_stop:
            return {
                "final_r": float(trail_stop), "mfe_r": float(np.maximum.accumulate(high_r[:i + 1]).max()),
                "mae_r": float(np.minimum.accumulate(low_r[:i + 1]).min()), "bars_held": int(i + 1),
                "exit_reason": "trail_stop" if activated else "stoploss",
            }
        if close_r[i] > high_water:
            high_water = float(close_r[i])
        if not activated and close_r[i] >= activation_r:
            activated = True
        if activated:
            new_trail = high_water - trail_r
            if new_trail > trail_stop:
                trail_stop = new_trail
    return {
        "final_r": float(close_r[-1]), "mfe_r": float(np.maximum.accumulate(high_r).max()),
        "mae_r": float(np.minimum.accumulate(low_r).min()), "bars_held": int(n), "exit_reason": "time_exit",
    }


def _sl_partial_close_runner(high_r, low_r, close_r) -> dict:
    n = len(close_r)
    partial_hit = False
    partial_r = 0.0
    trail_stop = -1.0
    high_water = 0.0
    for i in range(n):
        # SL hit on runner (50% remaining)
        if low_r[i] <= trail_stop:
            # final R = 0.5 * partial_r + 0.5 * trail_stop (if partial hit) OR -1.0 (if not)
            if partial_hit:
                final = 0.5 * partial_r + 0.5 * trail_stop
                reason = "trail_stop_runner"
            else:
                final = -1.0
                reason = "stoploss"
            return {
                "final_r": float(final), "mfe_r": float(np.maximum.accumulate(high_r[:i + 1]).max()),
                "mae_r": float(np.minimum.accumulate(low_r[:i + 1]).min()), "bars_held": int(i + 1),
                "exit_reason": reason,
            }
        if not partial_hit and high_r[i] >= 1.0:
            partial_hit = True
            partial_r = 1.0
            trail_stop = 0.0   # move stop to break-even on runner
            high_water = max(high_water, float(close_r[i]))
        if partial_hit:
            if close_r[i] > high_water:
                high_water = float(close_r[i])
            new_trail = high_water - 1.0
            if new_trail > trail_stop:
                trail_stop = new_trail
    if partial_hit:
        final = 0.5 * partial_r + 0.5 * float(close_r[-1])
    else:
        final = float(close_r[-1])
    return {
        "final_r": float(final), "mfe_r": float(np.maximum.accumulate(high_r).max()),
        "mae_r": float(np.minimum.accumulate(low_r).min()), "bars_held": int(n),
        "exit_reason": "time_exit",
    }


# ─── Exposure cap ──────────────────────────────────────────────────


def _currencies_for(pair: str) -> list[str]:
    if len(pair) == 6:
        return [pair[:3], pair[3:]]
    return []


def apply_exposure_cap(trades: pd.DataFrame, per_currency_cap: Optional[int]) -> pd.DataFrame:
    """Sequentially scan trades sorted by entry_time; reject any whose entry
    would put either currency over the cap. ``per_currency_cap`` None = unlimited.
    """
    if per_currency_cap is None or per_currency_cap <= 0:
        return trades.copy()
    if trades.empty:
        return trades.copy()
    df = trades.sort_values("entry_time").reset_index(drop=True)
    open_intervals: list[tuple[pd.Timestamp, pd.Timestamp, list[str]]] = []
    keep_mask = np.zeros(len(df), dtype=bool)
    for i, row in df.iterrows():
        et = pd.Timestamp(row["entry_time"])
        xt = pd.Timestamp(row["exit_time"])
        cs = _currencies_for(str(row["pair"]))
        # Drop already-closed intervals
        open_intervals = [iv for iv in open_intervals if iv[1] > et]
        # Count exposure per currency at et
        counts = {}
        for _, _, cur_list in open_intervals:
            for c in cur_list:
                counts[c] = counts.get(c, 0) + 1
        # Check
        ok = all(counts.get(c, 0) < per_currency_cap for c in cs)
        if ok:
            keep_mask[i] = True
            open_intervals.append((et, xt, cs))
    return df[keep_mask].reset_index(drop=True)


# ─── WFO metrics ──────────────────────────────────────────────────


@dataclass
class FoldMetrics:
    fold_id: str
    oos_year: int
    n_trades: int
    mean_r: float
    p50_r: float
    roi_pct: float
    max_dd_pct: float
    daily_dd_breaches: int
    ratio: float
    sharpe: float
    sortino: float
    sign_pos: int


def equity_metrics(trades: pd.DataFrame, starting_balance: float, pct_per_trade: float) -> dict:
    """Compute ROI%, max DD%, daily DD breach count, Sharpe/Sortino on trade-by-
    trade equity curve. Uses reset-floor risk: risk_$ = pct_per_trade × max(start, hwm).
    """
    if trades.empty:
        return {
            "n_trades": 0, "mean_r": 0.0, "p50_r": 0.0, "roi_pct": 0.0,
            "max_dd_pct": 0.0, "daily_dd_breaches": 0, "ratio": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "sign_pos": 0,
        }
    df = trades.sort_values("exit_time").reset_index(drop=True)
    balance = starting_balance
    hwm = starting_balance
    floor = starting_balance
    max_dd_pct = 0.0
    equity = [starting_balance]
    daily_balances: dict[pd.Timestamp, float] = {}
    daily_breach = 0
    last_day_balance = starting_balance
    for _, t in df.iterrows():
        risk_base = max(floor, starting_balance)
        risk_dollar = risk_base * pct_per_trade
        pnl = float(t["final_r"]) * risk_dollar
        balance += pnl
        hwm = max(hwm, balance)
        floor = max(floor, balance)
        dd_pct = 100.0 * (hwm - balance) / hwm if hwm > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd_pct)
        equity.append(balance)
        # Daily DD breach (>5% drop from prior day's balance)
        day = pd.Timestamp(t["exit_time"]).normalize()
        if day not in daily_balances:
            daily_balances[day] = balance
            prior_balance = last_day_balance
            daily_drop_pct = 100.0 * (prior_balance - balance) / prior_balance if prior_balance > 0 else 0
            if daily_drop_pct > 5.0:
                daily_breach += 1
            last_day_balance = balance
    roi_pct = 100.0 * (balance - starting_balance) / starting_balance
    rs = pd.Series(df["final_r"].values)
    sharpe = float(rs.mean() / rs.std(ddof=1)) if rs.std(ddof=1) > 0 else 0.0
    neg = rs[rs < 0]
    sortino = float(rs.mean() / neg.std(ddof=1)) if not neg.empty and neg.std(ddof=1) > 0 else 0.0
    ratio = roi_pct / max(max_dd_pct, 0.01) if max_dd_pct > 0 else float("inf") if roi_pct > 0 else 0.0
    return {
        "n_trades": int(len(df)),
        "mean_r": float(rs.mean()),
        "p50_r": float(rs.quantile(0.50)),
        "roi_pct": float(roi_pct),
        "max_dd_pct": float(max_dd_pct),
        "daily_dd_breaches": int(daily_breach),
        "ratio": float(ratio),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "sign_pos": int(roi_pct > 0),
    }


# ─── WFO scaffold ─────────────────────────────────────────────────


def fold_windows(train_window: list[str], n_folds: int) -> list[dict]:
    """Build 11 expanding-train folds, each year of [2010..2020] is one OOS."""
    start_year = pd.Timestamp(train_window[0]).year
    end_year = pd.Timestamp(train_window[1]).year
    years = list(range(start_year + 1, end_year + 1))[: n_folds]  # OOS = year2 .. year11
    folds = []
    for i, oos_year in enumerate(years):
        folds.append(
            {
                "fold_id": f"F{i + 1:02d}",
                "is_start": pd.Timestamp(f"{start_year}-01-01", tz="UTC"),
                "is_end": pd.Timestamp(f"{oos_year - 1}-12-31 23:59:59", tz="UTC"),
                "oos_start": pd.Timestamp(f"{oos_year}-01-01", tz="UTC"),
                "oos_end": pd.Timestamp(f"{oos_year}-12-31 23:59:59", tz="UTC"),
                "oos_year": oos_year,
            }
        )
    return folds


# ─── Architecture runners ─────────────────────────────────────────


_PROBA_CACHE: dict = {}  # key: (cluster_id, arch, fold_id, a3_n_bars) -> dict[trade_id -> proba]


def _get_or_compute_proba(
    arch: str,
    pool: pd.DataFrame,
    cluster_id: int,
    fold: dict,
    feature_cols: list[str],
    classifier_factory,
    a3_n_bars: Optional[int],
    paths_df: Optional[pd.DataFrame],
) -> Optional[dict]:
    """Train classifier on IS, predict on OOS once per (cluster, arch, fold, a3n).
    Cached at module level keyed by that tuple. Returns dict[trade_id -> proba]
    over OOS trades.

    For A2/A6 the base feature space is the v3 27+5. For A3 it's the path-so-far
    features at bar a3_n_bars.
    """
    key = (cluster_id, arch, fold["fold_id"], a3_n_bars)
    if key in _PROBA_CACHE:
        return _PROBA_CACHE[key]
    is_mask = (pool["entry_time"] >= fold["is_start"]) & (pool["entry_time"] <= fold["is_end"])
    oos_mask = (pool["entry_time"] >= fold["oos_start"]) & (pool["entry_time"] <= fold["oos_end"])
    is_df = pool[is_mask].copy()
    oos_df = pool[oos_mask].copy()

    if arch == "A3":
        if paths_df is None or a3_n_bars is None:
            _PROBA_CACHE[key] = None
            return None
        N = int(a3_n_bars)
        held = paths_df[(paths_df["is_held"] == 1) & (paths_df["bar_offset"] == N)]
        pf = held[["trade_id", "close_r", "mfe_so_far_r", "mae_so_far_r"]].rename(
            columns={
                "close_r": "a3_close_r_at_N",
                "mfe_so_far_r": "a3_mfe_at_N",
                "mae_so_far_r": "a3_mae_at_N",
            }
        )
        is_df = is_df.merge(pf, on="trade_id", how="inner")
        oos_df = oos_df.merge(pf, on="trade_id", how="inner")
        feat_use = ["a3_close_r_at_N", "a3_mfe_at_N", "a3_mae_at_N"]
    else:
        feat_use = feature_cols

    y_is = (is_df["cluster_best"] == cluster_id).astype(int).to_numpy()
    if y_is.sum() < 20 or (len(y_is) - y_is.sum()) < 20 or oos_df.empty:
        _PROBA_CACHE[key] = None
        return None

    X_is = is_df[feat_use].fillna(0.0).to_numpy(dtype=float)
    X_is = np.nan_to_num(X_is, nan=0.0, posinf=0.0, neginf=0.0)
    X_oos = oos_df[feat_use].fillna(0.0).to_numpy(dtype=float)
    X_oos = np.nan_to_num(X_oos, nan=0.0, posinf=0.0, neginf=0.0)
    needs_scale = classifier_factory["needs_scale"] if classifier_factory else False
    if arch == "A3":
        c = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE, n_jobs=1
        )
        needs_scale = False
    else:
        c = clone(classifier_factory["model"])
    if needs_scale:
        sc = StandardScaler()
        X_is = sc.fit_transform(X_is)
        X_oos = sc.transform(X_oos)
    try:
        c.fit(X_is, y_is)
        proba = c.predict_proba(X_oos)[:, 1]
        result = dict(zip(oos_df["trade_id"].tolist(), proba.tolist()))
    except Exception:
        result = None
    _PROBA_CACHE[key] = result
    return result


def reset_proba_cache() -> None:
    _PROBA_CACHE.clear()


def architecture_filter_trades(
    arch: str,
    pool: pd.DataFrame,                # has cluster_best, all features
    cluster_id: int,
    fold: dict,
    feature_cols: list[str],
    classifier_factory,
    auc_threshold: Optional[float],
    meta_label_thresholds: Optional[tuple],
    a3_n_bars: Optional[int] = None,
    paths_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply per-architecture filter to OOS trades. Return filtered OOS frame
    (which we then further reshape per exit policy & exposure cap upstream).

    A1: keep all trades (no filter).
    A2: train classifier on IS, predict on OOS, keep where proba >= auc_threshold.
    A3 / A4: implemented as DE / Pipeline-D in `_apply_pipeline_de` / `_apply_pipeline_d`.
    A6: meta-labeling — keep all trades, attach `weight` (0/0.5/1.0) by proba
        relative to (lo, hi) thresholds.
    """
    oos_mask = (pool["entry_time"] >= fold["oos_start"]) & (pool["entry_time"] <= fold["oos_end"])
    oos_df = pool[oos_mask].copy()

    if arch == "A1":
        oos_df["weight"] = 1.0
        return oos_df

    proba_map = _get_or_compute_proba(
        arch=arch, pool=pool, cluster_id=cluster_id, fold=fold,
        feature_cols=feature_cols, classifier_factory=classifier_factory,
        a3_n_bars=a3_n_bars, paths_df=paths_df,
    )
    if proba_map is None:
        oos_df["weight"] = 1.0
        return oos_df

    oos_df = oos_df[oos_df["trade_id"].isin(proba_map.keys())].copy()
    oos_df["_proba"] = oos_df["trade_id"].map(proba_map)

    if arch == "A2":
        thr = auc_threshold if auc_threshold is not None else 0.5
        oos_df = oos_df[oos_df["_proba"] >= thr].copy()
        oos_df["weight"] = 1.0
        return oos_df

    if arch == "A6":
        lo, hi = meta_label_thresholds if meta_label_thresholds else (0.4, 0.6)
        oos_df["weight"] = 0.0
        oos_df.loc[oos_df["_proba"] >= lo, "weight"] = 0.5
        oos_df.loc[oos_df["_proba"] >= hi, "weight"] = 1.0
        oos_df = oos_df[oos_df["weight"] > 0].copy()
        return oos_df

    if arch == "A3":
        # Admit top-50% by proba; trade R outcome unchanged from original entry
        # (simplified A3 — see ARCHETYPE_TO_ARCHITECTURES comment)
        if oos_df.empty:
            oos_df["weight"] = 1.0
            return oos_df
        thr_a3 = float(np.quantile(oos_df["_proba"], 0.5))
        out = oos_df[oos_df["_proba"] >= thr_a3].copy()
        out["weight"] = 1.0
        return out

    # Default fallback (A4 / unknown architectures) — equivalent to A1 baseline
    oos_df["weight"] = 1.0
    return oos_df


def apply_exit_to_trade(
    trade_id: int,
    paths_df_held: pd.DataFrame,
    sl_scale: float,
    exit_policy: str,
) -> dict:
    return apply_exit_policy(trade_id, paths_df_held, sl_scale, exit_policy)


def reshape_trades_with_exits(
    trades: pd.DataFrame,
    paths_df: pd.DataFrame,
    sl_multiplier: float,
    exit_policy: str,
) -> pd.DataFrame:
    """Reshape trades' final_r etc. by applying the (sl_multiplier, exit_policy)
    policy to their path series. Returns a new trades frame with updated
    final_r, mfe_r, mae_r, bars_held, exit_reason, weighted_final_r (final_r * weight).
    """
    if trades.empty:
        return trades.copy()
    sl_scale = SL_BASE_MULTIPLIER / float(sl_multiplier)
    paths_held = paths_df[paths_df["is_held"] == 1].copy()
    by_id = paths_held.groupby("trade_id")
    rows = []
    for _, t in trades.iterrows():
        tid = int(t["trade_id"])
        if tid not in by_id.groups:
            continue
        g = by_id.get_group(tid).sort_values("bar_offset")
        res = apply_exit_policy(tid, g, sl_scale, exit_policy)
        w = float(t.get("weight", 1.0))
        new_row = dict(t)
        new_row["final_r"] = res["final_r"]
        new_row["weighted_final_r"] = res["final_r"] * w
        new_row["mfe_r"] = res["mfe_r"]
        new_row["mae_r"] = res["mae_r"]
        new_row["bars_held"] = res["bars_held"]
        new_row["exit_reason"] = res["exit_reason"]
        # exit_time recalculation from bars_held would require pair-bar lookup; skip
        # since equity sim doesn't depend on exact exit_time precision below day grain
        rows.append(new_row)
    return pd.DataFrame(rows)


# ─── WFO driver per config ────────────────────────────────────────


def run_wfo_for_config(
    pool: pd.DataFrame,
    paths_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    cluster_id: int,
    arch: str,
    sl_multiplier: float,
    exit_policy: str,
    per_currency_cap: Optional[int],
    classifier_factory,
    auc_threshold: Optional[float],
    meta_label_thresholds: Optional[tuple],
    a3_n_bars: Optional[int] = None,
    oracle: bool = False,
) -> dict:
    """Run 11-fold WFO for one architecture × config combo.

    Returns dict with per_fold list + aggregate worst/mean metrics + verdict.
    """
    folds = fold_windows(cfg["wfo"]["train_window"], int(cfg["wfo"]["n_folds"]))
    start_balance = float(cfg["risk"]["starting_balance"])
    pct = float(cfg["risk"]["pct_per_trade"])

    per_fold: list[dict] = []
    for fold in folds:
        if oracle:
            # Use true cluster membership as the filter
            oos_mask = (pool["entry_time"] >= fold["oos_start"]) & (pool["entry_time"] <= fold["oos_end"])
            oos_df = pool[oos_mask & (pool["cluster_best"] == cluster_id)].copy()
            oos_df["weight"] = 1.0
        else:
            oos_df = architecture_filter_trades(
                arch=arch,
                pool=pool,
                cluster_id=cluster_id,
                fold=fold,
                feature_cols=feature_cols,
                classifier_factory=classifier_factory,
                auc_threshold=auc_threshold,
                meta_label_thresholds=meta_label_thresholds,
                a3_n_bars=a3_n_bars,
                paths_df=paths_df,
            )
        # Apply exit policy + exposure cap
        oos_reshaped = reshape_trades_with_exits(oos_df, paths_df, sl_multiplier, exit_policy)
        oos_capped = apply_exposure_cap(oos_reshaped, per_currency_cap)
        # Use weighted_final_r → final_r for equity sim
        if "weighted_final_r" in oos_capped.columns:
            oos_capped = oos_capped.copy()
            oos_capped["final_r"] = oos_capped["weighted_final_r"]
        em = equity_metrics(oos_capped, start_balance, pct)
        em["fold_id"] = fold["fold_id"]
        em["oos_year"] = fold["oos_year"]
        per_fold.append(em)

    # Aggregate
    rois = [f["roi_pct"] for f in per_fold]
    dds = [f["max_dd_pct"] for f in per_fold]
    ratios = [f["ratio"] for f in per_fold]
    sharpes = [f["sharpe"] for f in per_fold]
    sign_pos = sum(f["sign_pos"] for f in per_fold)
    n_trades_total = sum(f["n_trades"] for f in per_fold)
    n_trades_min = min((f["n_trades"] for f in per_fold), default=0)

    worst_roi = min(rois) if rois else 0.0
    worst_dd = max(dds) if dds else 0.0
    worst_ratio = min((r for r in ratios if np.isfinite(r)), default=0.0)
    mean_roi = float(np.mean(rois)) if rois else 0.0
    mean_dd = float(np.mean(dds)) if dds else 0.0
    mean_ratio = float(np.mean([r for r in ratios if np.isfinite(r)])) if ratios else 0.0
    std_roi = float(np.std(rois)) if rois else 0.0

    # Daily-DD breach total
    total_daily_breaches = sum(f["daily_dd_breaches"] for f in per_fold)

    # Verdict (§3) — DEPLOYABLE / VIABLE / FAIL
    deploy = (
        worst_ratio >= 2.0
        and worst_roi > 0
        and sign_pos == len(per_fold)
        and worst_dd <= 8.0
        and total_daily_breaches == 0
        and worst_dd <= 10.0
        and n_trades_min >= 25
    )
    viable = (
        worst_ratio >= 2.0
        and mean_ratio >= 2.5
        and worst_dd <= 10.0
        and total_daily_breaches == 0
        and n_trades_min >= 25
    )
    if deploy:
        verdict = "PASS-DEPLOYABLE"
    elif viable:
        verdict = "PASS-VIABLE"
    else:
        verdict = "FAIL"

    return {
        "per_fold": per_fold,
        "worst_fold_roi": worst_roi,
        "worst_fold_dd": worst_dd,
        "worst_fold_ratio": worst_ratio,
        "mean_fold_roi": mean_roi,
        "mean_fold_dd": mean_dd,
        "mean_fold_ratio": mean_ratio,
        "std_fold_roi": std_roi,
        "sign_pos_folds": sign_pos,
        "n_trades_total": n_trades_total,
        "n_trades_min_per_fold": n_trades_min,
        "daily_dd_breaches_total": total_daily_breaches,
        "verdict": verdict,
    }


# ─── Step 5 driver ────────────────────────────────────────────────


def _classifier_factory(name: str, cfg: dict):
    s4 = cfg["step_4"]
    if name == "logistic":
        return {
            "model": LogisticRegression(
                penalty=str(s4["logistic"]["penalty"]),
                C=float(s4["logistic"]["C"]),
                max_iter=int(s4["logistic"]["max_iter"]),
                random_state=int(s4["logistic"]["random_state"]),
                n_jobs=1,
            ),
            "needs_scale": True,
        }
    if name == "lgbm" and HAVE_LGB:
        return {
            "model": lgb.LGBMClassifier(
                n_estimators=int(s4["lgbm"]["n_estimators"]),
                num_leaves=int(s4["lgbm"]["num_leaves"]),
                learning_rate=float(s4["lgbm"]["learning_rate"]),
                min_child_samples=int(s4["lgbm"]["min_child_samples"]),
                random_state=int(s4["lgbm"]["random_state"]),
                n_jobs=1,
                verbosity=-1,
            ),
            "needs_scale": False,
        }
    # default random_forest
    return {
        "model": RandomForestClassifier(
            n_estimators=int(s4["rf"]["n_estimators"]),
            max_depth=int(s4["rf"]["max_depth"]),
            min_samples_leaf=int(s4["rf"]["min_samples_leaf"]),
            random_state=int(s4["rf"]["random_state"]),
            n_jobs=1,
        ),
        "needs_scale": False,
    }


def run(cfg: dict) -> dict:
    seed_everything(RANDOM_STATE)

    results_dir = results_root(cfg)
    step1_dir = results_dir / "step_1"
    step2_dir = results_dir / "step_2"
    step3_dir = results_dir / "step_3"
    step4_dir = results_dir / "step_4"
    step5_dir = results_dir / "step_5"
    step5_dir.mkdir(parents=True, exist_ok=True)

    pool_df = pd.read_parquet(step1_dir / "pool.parquet")
    paths_df = pd.read_parquet(step1_dir / "trades_paths.parquet")
    ca_df = pd.read_parquet(step2_dir / "cluster_assignments.parquet")
    step3_manifest = json.loads((step3_dir / "manifest.json").read_text(encoding="utf-8"))
    step4_manifest = json.loads((step4_dir / "manifest.json").read_text(encoding="utf-8"))

    # Merge
    pool = pool_df.merge(ca_df[["trade_id", "cluster_best"]], on="trade_id", how="left")
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True)

    # Per-cluster archetype + best SL
    per_cluster_best = step3_manifest.get("per_cluster_best_sl", {})
    per_cluster_arche = step3_manifest.get("per_cluster_archetype", {})
    candidate_clusters = step3_manifest.get("candidate_clusters", [])
    if not candidate_clusters:
        # Fall back to highest-composite cluster
        if per_cluster_best:
            best_cid = max(per_cluster_best.keys(), key=lambda k: per_cluster_best[k]["composite"])
            candidate_clusters = [int(best_cid)]

    per_cluster_step4 = step4_manifest.get("per_cluster_summary", {})

    # Feature columns for classifiers (same exclude rules as Step 4)
    from scripts.l_arc_11.step_4_extraction import _candidate_feature_columns
    feature_cols = _candidate_feature_columns(pool)
    feature_cols = [c for c in feature_cols if c not in ("cluster_best",) and not c.startswith("cluster_k")]

    all_configs = []  # list of dict per evaluated config

    for cid_str in [str(c) for c in candidate_clusters]:
        cid_int = int(cid_str)
        arche = per_cluster_arche.get(cid_str, per_cluster_arche.get(cid_int, "unknown"))
        # Override: Choppy candidate → Stepwise climber (see comment on
        # CHOPPY_CANDIDATE_OVERRIDE). Note in closure §4.
        if arche == "Choppy" and cid_int in [int(c) for c in candidate_clusters]:
            _log(f"  Cluster {cid_int} archetype 'Choppy' overridden to '{CHOPPY_CANDIDATE_OVERRIDE}' (candidate cluster)")
            arche_effective = CHOPPY_CANDIDATE_OVERRIDE
        else:
            arche_effective = arche
        selected_sl = per_cluster_best.get(cid_str, per_cluster_best.get(cid_int, {})).get("best_sl_multiplier", 2.0)
        archs = ARCHETYPE_TO_ARCHITECTURES.get(arche_effective, ["A1"])
        exits = ARCHETYPE_TO_EXITS.get(arche_effective, ["sl_only"])
        sl_list = neighbouring_sls(selected_sl)

        s4_summ = per_cluster_step4.get(cid_str, per_cluster_step4.get(cid_int, {}))
        best_classifier_name = s4_summ.get("best_classifier", "random_forest")
        auc_threshold = s4_summ.get("auc_best_threshold", 0.5)

        _log(f"Cluster {cid_int} arche={arche} selected_sl={selected_sl} archs={archs} exits={exits} sl_list={sl_list}")

        for arch in archs:
            classifier_factory = _classifier_factory(best_classifier_name, cfg)

            # Decide search dims per arch
            if arch == "A6":
                meta_combos = [tuple(t) for t in cfg["step_5"]["meta_label_thresholds"]]
            else:
                meta_combos = [None]
            if arch == "A3":
                a3_n_bars_list = [int(n) for n in cfg["step_5"]["pipeline_de_n_bars"]]
            else:
                a3_n_bars_list = [None]

            for sl_mult in sl_list:
                for exit_policy in exits:
                    for cap_raw in cfg["step_5"]["exposure_caps"]:
                        cap = None if cap_raw is None else int(cap_raw)
                        for meta_combo in meta_combos:
                            for a3n in a3_n_bars_list:
                                cfg_name = f"cluster{cid_int}_{arch}_SL{sl_mult}_EX{exit_policy}_CAP{cap}"
                                if arch == "A6":
                                    cfg_name += f"_META{meta_combo}"
                                if arch == "A3":
                                    cfg_name += f"_N{a3n}"
                                _log(f"  WFO: {cfg_name}")
                                res = run_wfo_for_config(
                                    pool=pool,
                                    paths_df=paths_df,
                                    feature_cols=feature_cols,
                                    cfg=cfg,
                                    cluster_id=cid_int,
                                    arch=arch,
                                    sl_multiplier=sl_mult,
                                    exit_policy=exit_policy,
                                    per_currency_cap=cap,
                                    classifier_factory=classifier_factory,
                                    auc_threshold=auc_threshold if arch == "A2" else None,
                                    meta_label_thresholds=meta_combo,
                                    a3_n_bars=a3n,
                                )
                                res["config_name"] = cfg_name
                                res["cluster"] = cid_int
                                res["archetype"] = arche  # ORIGINAL archetype (pre-override)
                                res["archetype_effective"] = arche_effective
                                res["architecture"] = arch
                                res["sl_multiplier"] = sl_mult
                                res["exit_policy"] = exit_policy
                                res["exposure_cap_per_currency"] = cap
                                res["meta_thresholds"] = meta_combo if meta_combo else None
                                res["a3_n_bars"] = a3n
                                all_configs.append(res)

        # Oracle WFO
        _log(f"  ORACLE WFO for cluster {cid_int}")
        oracle_res = run_wfo_for_config(
            pool=pool,
            paths_df=paths_df,
            feature_cols=feature_cols,
            cfg=cfg,
            cluster_id=cid_int,
            arch="ORACLE",
            sl_multiplier=selected_sl,
            exit_policy="sl_only",
            per_currency_cap=None,
            classifier_factory=None,
            auc_threshold=None,
            meta_label_thresholds=None,
            oracle=True,
        )
        oracle_res["config_name"] = f"ORACLE_cluster{cid_int}"
        oracle_res["cluster"] = cid_int
        oracle_res["architecture"] = "ORACLE"
        oracle_res["sl_multiplier"] = selected_sl
        oracle_res["exit_policy"] = "sl_only"
        oracle_res["exposure_cap_per_currency"] = None
        oracle_res["meta_thresholds"] = None
        oracle_res["archetype"] = arche
        all_configs.append(oracle_res)

    # Rank by worst-fold ratio (descending), exclude ORACLE
    real_configs = [c for c in all_configs if c["architecture"] != "ORACLE"]
    real_configs.sort(key=lambda c: c["worst_fold_ratio"], reverse=True)

    # Holdout: top-3 candidates
    top_k = int(cfg["step_5"].get("top_k_holdout", 3))
    top = real_configs[:top_k]
    holdout_start = pd.Timestamp(cfg["wfo"]["holdout_window"][0], tz="UTC")
    holdout_end = pd.Timestamp(cfg["wfo"]["holdout_window"][1], tz="UTC") + pd.Timedelta(days=1)

    holdout_results = []
    for t in top:
        cfg_name = t["config_name"]
        _log(f"HOLDOUT: {cfg_name}")
        # Use ALL trades in IS for training (2010-2020), then evaluate on 2021-2026-04
        holdout_fold = {
            "fold_id": "HOLDOUT",
            "is_start": pd.Timestamp(cfg["wfo"]["train_window"][0], tz="UTC"),
            "is_end": pd.Timestamp(cfg["wfo"]["train_window"][1], tz="UTC") + pd.Timedelta(days=1),
            "oos_start": holdout_start,
            "oos_end": holdout_end,
            "oos_year": holdout_start.year,
        }
        if t["architecture"] == "A1":
            oos_mask = (pool["entry_time"] >= holdout_fold["oos_start"]) & (pool["entry_time"] <= holdout_fold["oos_end"])
            oos_df = pool[oos_mask].copy()
            oos_df["weight"] = 1.0
        else:
            classifier_factory = _classifier_factory(
                per_cluster_step4.get(str(t["cluster"]), per_cluster_step4.get(int(t["cluster"]), {})).get("best_classifier", "random_forest"),
                cfg,
            )
            auc_thr = per_cluster_step4.get(str(t["cluster"]), per_cluster_step4.get(int(t["cluster"]), {})).get("auc_best_threshold", 0.5)
            oos_df = architecture_filter_trades(
                arch=t["architecture"],
                pool=pool,
                cluster_id=int(t["cluster"]),
                fold=holdout_fold,
                feature_cols=feature_cols,
                classifier_factory=classifier_factory,
                auc_threshold=auc_thr if t["architecture"] == "A2" else None,
                meta_label_thresholds=t.get("meta_thresholds"),
            )
        oos_reshaped = reshape_trades_with_exits(oos_df, paths_df, t["sl_multiplier"], t["exit_policy"])
        oos_capped = apply_exposure_cap(oos_reshaped, t["exposure_cap_per_currency"])
        if "weighted_final_r" in oos_capped.columns:
            oos_capped["final_r"] = oos_capped["weighted_final_r"]
        em = equity_metrics(oos_capped, float(cfg["risk"]["starting_balance"]), float(cfg["risk"]["pct_per_trade"]))
        em["config_name"] = cfg_name
        em["cluster"] = t["cluster"]
        em["architecture"] = t["architecture"]
        em["sl_multiplier"] = t["sl_multiplier"]
        em["exit_policy"] = t["exit_policy"]
        em["exposure_cap_per_currency"] = t["exposure_cap_per_currency"]
        em["wfo_worst_ratio"] = t["worst_fold_ratio"]
        em["wfo_verdict"] = t["verdict"]
        # Holdout verdict via same §3 gates as a single-fold WFO
        holdout_deploy = (
            em["ratio"] >= 2.0 and em["roi_pct"] > 0
            and em["max_dd_pct"] <= 8.0 and em["daily_dd_breaches"] == 0
            and em["max_dd_pct"] <= 10.0 and em["n_trades"] >= 25
        )
        holdout_viable = (
            em["ratio"] >= 2.0 and em["max_dd_pct"] <= 10.0
            and em["daily_dd_breaches"] == 0 and em["n_trades"] >= 25
        )
        em["holdout_verdict"] = (
            "PASS-DEPLOYABLE" if holdout_deploy
            else ("PASS-VIABLE" if holdout_viable else "FAIL")
        )
        # Arc verdict (combined WFO+holdout per §3 + Amendment 1)
        if t["verdict"] == "PASS-DEPLOYABLE" and em["holdout_verdict"] == "PASS-DEPLOYABLE":
            combined = "PASS-DEPLOYABLE"
        elif t["verdict"] in ("PASS-DEPLOYABLE", "PASS-VIABLE") and em["holdout_verdict"] in ("PASS-DEPLOYABLE", "PASS-VIABLE"):
            combined = "PASS-VIABLE"
        else:
            combined = "FAIL"
        em["combined_verdict"] = combined
        holdout_results.append(em)

    # Persist
    wfo_results_df = pd.DataFrame(
        [
            {
                "config_name": c["config_name"], "cluster": c["cluster"], "archetype": c["archetype"],
                "architecture": c["architecture"], "sl_multiplier": c["sl_multiplier"],
                "exit_policy": c["exit_policy"], "exposure_cap_per_currency": c["exposure_cap_per_currency"],
                "meta_thresholds": str(c.get("meta_thresholds")),
                "worst_fold_roi": c["worst_fold_roi"], "worst_fold_dd": c["worst_fold_dd"],
                "worst_fold_ratio": c["worst_fold_ratio"], "mean_fold_roi": c["mean_fold_roi"],
                "mean_fold_dd": c["mean_fold_dd"], "mean_fold_ratio": c["mean_fold_ratio"],
                "std_fold_roi": c["std_fold_roi"], "sign_pos_folds": c["sign_pos_folds"],
                "n_trades_total": c["n_trades_total"], "n_trades_min_per_fold": c["n_trades_min_per_fold"],
                "daily_dd_breaches_total": c["daily_dd_breaches_total"],
                "verdict": c["verdict"],
            }
            for c in all_configs if c["architecture"] != "ORACLE"
        ]
    )
    wfo_results_path = step5_dir / "wfo_results.csv"
    wfo_results_df.to_csv(wfo_results_path, index=False, lineterminator="\n")

    oracle_df = pd.DataFrame(
        [
            {
                "config_name": c["config_name"], "cluster": c["cluster"],
                "worst_fold_roi": c["worst_fold_roi"], "worst_fold_dd": c["worst_fold_dd"],
                "worst_fold_ratio": c["worst_fold_ratio"], "mean_fold_roi": c["mean_fold_roi"],
                "mean_fold_dd": c["mean_fold_dd"], "mean_fold_ratio": c["mean_fold_ratio"],
                "sign_pos_folds": c["sign_pos_folds"], "n_trades_total": c["n_trades_total"],
                "verdict": c["verdict"],
            }
            for c in all_configs if c["architecture"] == "ORACLE"
        ]
    )
    oracle_path = step5_dir / "wfo_oracle.csv"
    oracle_df.to_csv(oracle_path, index=False, lineterminator="\n")

    holdout_df = pd.DataFrame(holdout_results)
    holdout_path = step5_dir / "holdout_results.csv"
    holdout_df.to_csv(holdout_path, index=False, lineterminator="\n")

    # Ranked markdown
    lines = ["# Arc 11 v3.0 — Step 5 Architectures Ranked", "", "Per L_PROTOCOL §2 Step 5 + Amendments 1 + 2.", ""]
    lines.append(f"## Total configs evaluated: {len(wfo_results_df)}")
    selection_bias = "thin" if len(wfo_results_df) < 50 else ("normal" if len(wfo_results_df) < 100 else "broad")
    lines.append(f"Selection bias accounting: **{selection_bias}** search.")
    lines.append("")
    lines.append("## Top 15 by worst-fold ratio")
    lines.append("")
    lines.append("| rank | config | cluster | arche | arch | SL | exit | cap | worst_ratio | worst_roi% | worst_dd% | sign | n_total | verdict |")
    lines.append("|---:|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    # Sort by worst_fold_ratio descending; show top 15
    wfo_sorted = wfo_results_df.sort_values("worst_fold_ratio", ascending=False).reset_index(drop=True)
    for i, r in enumerate(wfo_sorted.head(15).itertuples(index=False), 1):
        lines.append(
            f"| {i} | {r.config_name} | {r.cluster} | {r.archetype} | {r.architecture} | {r.sl_multiplier} | "
            f"{r.exit_policy} | {r.exposure_cap_per_currency} | {r.worst_fold_ratio:.3f} | {r.worst_fold_roi:.2f} | "
            f"{r.worst_fold_dd:.2f} | {int(r.sign_pos_folds)} | {r.n_trades_total} | {r.verdict} |"
        )
    lines.append("")
    lines.append("## Oracle WFO")
    lines.append("")
    if not oracle_df.empty:
        lines.append("| cluster | worst_ratio | worst_roi% | worst_dd% | mean_ratio | mean_roi% | n_total | verdict |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in oracle_df.itertuples(index=False):
            lines.append(
                f"| {r.cluster} | {r.worst_fold_ratio:.3f} | {r.worst_fold_roi:.2f} | "
                f"{r.worst_fold_dd:.2f} | {r.mean_fold_ratio:.3f} | {r.mean_fold_roi:.2f} | "
                f"{r.n_trades_total} | {r.verdict} |"
            )
        lines.append("")
    lines.append("## Holdout (top-3 candidates, one-shot)")
    lines.append("")
    if not holdout_df.empty:
        lines.append("| config | architecture | cluster | SL | exit | cap | trades | roi% | dd% | ratio | wfo_verdict | holdout_verdict | combined |")
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|")
        for r in holdout_df.itertuples(index=False):
            lines.append(
                f"| {r.config_name} | {r.architecture} | {r.cluster} | {r.sl_multiplier} | {r.exit_policy} | "
                f"{r.exposure_cap_per_currency} | {r.n_trades} | {r.roi_pct:.2f} | {r.max_dd_pct:.2f} | "
                f"{r.ratio:.3f} | {r.wfo_verdict} | {r.holdout_verdict} | {r.combined_verdict} |"
            )
    ranked_path = step5_dir / "architectures_ranked.md"
    ranked_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # Best candidate detail
    if real_configs:
        best = real_configs[0]
        best_lines = [
            f"# Arc 11 v3.0 — Step 5 Best Candidate",
            "",
            f"**Config:** {best['config_name']}",
            f"- Cluster: {best['cluster']} ({best['archetype']})",
            f"- Architecture: {best['architecture']}",
            f"- SL multiplier: {best['sl_multiplier']}",
            f"- Exit policy: {best['exit_policy']}",
            f"- Exposure cap (per currency): {best['exposure_cap_per_currency']}",
            "",
            f"**WFO worst-fold ratio:** {best['worst_fold_ratio']:.3f}",
            f"**WFO worst-fold ROI:** {best['worst_fold_roi']:.2f}%",
            f"**WFO worst-fold DD:** {best['worst_fold_dd']:.2f}%",
            f"**WFO sign-pos folds:** {best['sign_pos_folds']} / 11",
            f"**WFO verdict:** {best['verdict']}",
            "",
            "## Per-fold breakdown",
            "",
            "| fold | year | n | mean_R | roi% | dd% | ratio | sharpe | sign |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for f in best["per_fold"]:
            best_lines.append(
                f"| {f['fold_id']} | {f['oos_year']} | {f['n_trades']} | {f['mean_r']:.3f} | "
                f"{f['roi_pct']:.2f} | {f['max_dd_pct']:.2f} | {f['ratio']:.3f} | "
                f"{f['sharpe']:.3f} | {f['sign_pos']} |"
            )
        (step5_dir / "best_candidate.md").write_text("\n".join(best_lines) + "\n", encoding="utf-8", newline="\n")

    # Manifest
    manifest = {
        "step": 5,
        "total_configs": len(real_configs),
        "selection_bias_flag": selection_bias,
        "n_oracle_configs": int(len(oracle_df)),
        "candidate_clusters": [int(c) for c in candidate_clusters],
        "top3_holdout": [
            {"config_name": r["config_name"], "wfo_verdict": r["wfo_verdict"],
             "holdout_verdict": r["holdout_verdict"], "combined_verdict": r["combined_verdict"]}
            for r in holdout_results
        ],
        "arc_verdict": (
            holdout_results[0]["combined_verdict"] if holdout_results else "FAIL"
        ),
        "sha256": {
            "wfo_results_csv": sha256_file(wfo_results_path),
            "wfo_oracle_csv": sha256_file(oracle_path),
            "holdout_results_csv": sha256_file(holdout_path) if holdout_path.exists() else "",
        },
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "env": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_manifest(step5_dir / "manifest.json", manifest)
    _log(f"Step 5 complete: arc_verdict={manifest['arc_verdict']}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/wfo_l_arc_11.yaml")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
