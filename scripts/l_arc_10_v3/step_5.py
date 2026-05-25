"""Arc 10 v3.0 — Step 5 WFO architecture search.

Per L_PROTOCOL v3.0 §2 Step 5 with Amendments 1 + 2. Search architectures:
    A1 system_level_filter    — base signal + searched SL/exit/exposure
    A2 classifier_filter      — Step 4 best classifier + AUC-best threshold
    A3 pipeline_de            — deferred-entry classifier at bar N ∈ {3, 5}
    A4 pipeline_d_exits       — classifier-driven exit on path-so-far
    A6 meta_labeling          — confidence → 0x / 0.5x / 1x sizing
    Oracle WFO                — true cluster membership (upper bound)

Architecture selection is gated by Step 3 archetype label per cluster.
A5 (portfolio composition) runs only if ≥ 2 candidate clusters survive.

11-fold WFO 2010-2020 (search); 2021-present one-shot holdout for top-3
candidates ranked by worst-fold ROI/DD ratio.

Outputs:
    results/l_arc_10/step_5/wfo_results.csv
    results/l_arc_10/step_5/wfo_oracle.csv
    results/l_arc_10/step_5/architectures_ranked.md
    results/l_arc_10/step_5/best_candidate.md
    results/l_arc_10/step_5/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from core.determinism import RANDOM_STATE, seed_everything  # noqa: E402
from core.sim.exit_policies import simulate_path as _canonical_simulate_path  # noqa: E402
from scripts.l_arc_10_v3._common import load_config, sha256_file, write_manifest  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Archetype → architectures (per dispatch §"Architecture selection by Step 3 archetype")
ARCHETYPE_ARCHITECTURES = {
    "stepwise_climber": ["A1", "A2", "A4"],
    "v_shape_recovery": ["A1", "A3", "A6"],
    "bimodal": ["A1", "A4"],
    "monotonic_up": ["A1", "A2", "A6"],
    "monotonic_down": [],  # short bias; long-only arc skips
    "choppy": [],
    "unclassified": ["A1", "A2"],  # safety net — at least try
}

EXIT_POLICIES_BY_ARCHETYPE = {
    "stepwise_climber": ["sl_only", "sl_plus_trailing_atr", "sl_plus_trailing_swing"],
    "v_shape_recovery": ["sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r", "sl_partial_close_1r_runner_trail"],
    "bimodal": ["sl_only", "sl_partial_close_1r_runner_trail", "sl_plus_tp_2r"],
    "monotonic_up": ["sl_only", "sl_plus_trailing_atr", "sl_plus_tp_3r"],
    "unclassified": ["sl_only", "sl_plus_tp_2r"],
}

EXPOSURE_CAPS = [
    ("max_per_currency_2", 2),
    ("unlimited", None),
]

A3_N_BARS = [3, 5]
A4_EXIT_THRESHOLDS = [0.3, 0.4, 0.5]
A6_THRESHOLD_PAIRS = [(0.3, 0.5), (0.4, 0.6), (0.5, 0.7)]

# Per-fold gate inputs
RISK_PER_TRADE = 0.005  # 0.5%
INITIAL_BAL = 100_000.0

# WFO fold windows
SEARCH_START = "2010-01-01"
SEARCH_END = "2020-12-31"
HOLDOUT_START = "2021-01-01"
HOLDOUT_END = "2026-04-30"
N_SEARCH_FOLDS = 11


# ---------------------------------------------------------------------------
# Path-aware exit simulation
# ---------------------------------------------------------------------------


def _apply_exit_policy(
    trade_row: pd.Series,
    path_rows: pd.DataFrame,
    sl_multiplier: float,
    exit_policy: str,
) -> tuple[float, int]:
    """Simulate the realised R + bars_held under a new SL multiplier and exit policy.

    Delegates to :func:`core.sim.exit_policies.simulate_path` (the canonical
    path-replay registry). The hand-rolled per-policy logic that historically
    lived in this function was extracted to ``core/sim/exit_policies/
    path_simulate.py`` as part of the canonical-exit-policy-registry PR
    (engine/sl-partial-close-runner-trail-primitive). The semantics are
    byte-identical (verified by ``tests/sim/exit_policies/
    test_path_simulate_reference_parity.py``).

    Path values are stored in R-multiples relative to Step 1's SL=2.0×ATR.
    See the canonical module for full schema and scaling math.

    Returns (final_r_new, bars_held).
    """
    return _canonical_simulate_path(exit_policy, trade_row, path_rows, sl_multiplier)


# ---------------------------------------------------------------------------
# WFO fold metrics
# ---------------------------------------------------------------------------


def _ann_factor_from_years(yrs: float) -> float:
    return 1.0 / max(yrs, 1e-6)


def _equity_curve(per_trade_r: np.ndarray, signal_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compound equity at fixed 0.5% risk per trade."""
    eq = INITIAL_BAL
    curve = []
    for r in per_trade_r:
        pnl = eq * RISK_PER_TRADE * r
        eq = eq + pnl
        curve.append(eq)
    return np.array(curve, dtype=float), np.array(signal_times)


def _max_drawdown(curve: np.ndarray) -> float:
    if curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def _fold_metrics(trades_df: pd.DataFrame) -> dict:
    if len(trades_df) == 0:
        return dict(n=0, mean_r=0.0, sum_r=0.0, roi=0.0, dd=0.0, ratio=np.nan, sign=0)
    t = trades_df.sort_values("signal_bar_time")
    r = t["final_r"].to_numpy()
    times = pd.to_datetime(t["signal_bar_time"])
    curve, _ = _equity_curve(r, times.to_numpy())
    ann_yrs = max((times.max() - times.min()).total_seconds() / (365.25 * 86400.0), 1e-6)
    final_bal = float(curve[-1]) if curve.size else INITIAL_BAL
    total_ret = (final_bal / INITIAL_BAL) - 1.0
    roi_ann = (1.0 + total_ret) ** (1.0 / ann_yrs) - 1.0 if ann_yrs > 0 else 0.0
    dd = _max_drawdown(curve)
    ratio = roi_ann / dd if dd > 1e-6 else float("inf") if roi_ann > 0 else 0.0
    return dict(
        n=int(len(r)),
        mean_r=float(np.mean(r)),
        sum_r=float(np.sum(r)),
        roi=float(roi_ann),
        dd=float(dd),
        ratio=float(ratio) if np.isfinite(ratio) else (1e6 if ratio > 0 else 0.0),
        sign=1 if roi_ann > 0 else (-1 if roi_ann < 0 else 0),
    )


def _build_folds(t_start: str, t_end: str, n: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Anchored expanding windows: fold i's OOS = year i."""
    start = pd.Timestamp(t_start, tz="UTC")
    end = pd.Timestamp(t_end, tz="UTC")
    total_days = (end - start).days
    fold_days = total_days // n
    folds = []
    for i in range(n):
        fs = start + pd.Timedelta(days=i * fold_days)
        fe = start + pd.Timedelta(days=(i + 1) * fold_days) - pd.Timedelta(seconds=1)
        if i == n - 1:
            fe = end
        folds.append((fs, fe))
    return folds


# ---------------------------------------------------------------------------
# Exposure cap
# ---------------------------------------------------------------------------


def _apply_exposure_cap_per_currency(
    trades: pd.DataFrame, max_per_currency: int
) -> pd.DataFrame:
    """Drop trades that would exceed max concurrent positions per currency.

    Currency = first 3 chars + last 3 chars of pair (base + quote).
    """
    if max_per_currency is None or len(trades) == 0:
        return trades
    t = trades.sort_values("signal_bar_time").reset_index(drop=True)
    keep = np.ones(len(t), dtype=bool)
    # Track open positions per currency: dict[ccy, list[exit_time]]
    open_by_ccy: dict[str, list[pd.Timestamp]] = {}
    for i, row in t.iterrows():
        sig_t = pd.Timestamp(row["signal_bar_time"])
        exit_t = pd.Timestamp(row["exit_time"]) if pd.notna(row.get("exit_time")) else sig_t
        pair = row["pair"]
        ccys = [pair[:3], pair[3:6]]
        # Purge expired open positions
        for c in ccys:
            if c not in open_by_ccy:
                open_by_ccy[c] = []
            open_by_ccy[c] = [t_ for t_ in open_by_ccy[c] if t_ > sig_t]
        # Check caps
        if any(len(open_by_ccy[c]) >= max_per_currency for c in ccys):
            keep[i] = False
            continue
        for c in ccys:
            open_by_ccy[c].append(exit_t)
    return t[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-architecture config builders
# ---------------------------------------------------------------------------


def _classifier_factory_for_a2(best_clf_name: str):
    if best_clf_name == "rf":
        return lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE, n_jobs=1
        )
    if best_clf_name == "logistic":
        return lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=1)),
            ]
        )
    if best_clf_name == "lgbm":
        try:
            import lightgbm as lgb  # noqa
            return lambda: lgb.LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05, min_child_samples=50,
                random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
            )
        except ImportError:
            pass
    # fallback
    return lambda: RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE, n_jobs=1
    )


def _rf_factory():
    return lambda: RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE, n_jobs=1
    )


def _train_classifier_for_fold(
    pool_train: pd.DataFrame,
    feature_cols: list[str],
    target: np.ndarray,
    factory,
) -> tuple[object, float]:
    """Train a classifier on fold's training set; return (model, AUC-best threshold).

    Uses an inner train/val split (last 25% of fold-train as val) to pick threshold.
    """
    n_train = len(pool_train)
    if n_train < 50 or len(np.unique(target)) < 2:
        return None, 0.5
    cut = int(n_train * 0.75)
    X_train = pool_train[feature_cols].fillna(0.0).iloc[:cut].to_numpy()
    y_train = target[:cut]
    X_val = pool_train[feature_cols].fillna(0.0).iloc[cut:].to_numpy()
    y_val = target[cut:]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        # Try full fit + fixed threshold
        try:
            clf = factory()
            clf.fit(pool_train[feature_cols].fillna(0.0).to_numpy(), target)
            return clf, 0.5
        except Exception:
            return None, 0.5
    clf = factory()
    try:
        clf.fit(X_train, y_train)
        p_val = clf.predict_proba(X_val)[:, 1]
    except Exception:
        return None, 0.5
    # AUC-best threshold = maximises Youden's J = TPR - FPR
    best_j = -np.inf
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.9, 33):
        y_pred = (p_val >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_val == 1)).sum())
        fp = int(((y_pred == 1) & (y_val == 0)).sum())
        tn = int(((y_pred == 0) & (y_val == 0)).sum())
        fn = int(((y_pred == 0) & (y_val == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    # Refit on full train+val
    try:
        clf.fit(pool_train[feature_cols].fillna(0.0).to_numpy(), target)
    except Exception:
        pass
    return clf, best_thr


# ---------------------------------------------------------------------------
# WFO runner
# ---------------------------------------------------------------------------


def _wfo_run(
    pool_in_window: pd.DataFrame,
    paths: pd.DataFrame,
    folds: list[tuple],
    *,
    architecture: str,
    config: dict,
    feature_cols: list[str],
    cluster_target: np.ndarray | None,
    classifier_factory,
    oracle_cluster_label: int | None = None,
) -> dict:
    """Run one architecture × config across the fold list.

    Returns dict with per_fold list + aggregates.
    """
    pool_in_window = pool_in_window.sort_values("signal_bar_time").reset_index(drop=True)
    times = pd.to_datetime(pool_in_window["signal_bar_time"])

    per_fold = []
    paths_by_trade = {tid: g for tid, g in paths.groupby("trade_id")}

    for fi, (fs, fe) in enumerate(folds):
        # Train window: everything before fold start
        train_mask = times < fs
        oos_mask = (times >= fs) & (times <= fe)
        train = pool_in_window[train_mask]
        oos = pool_in_window[oos_mask]

        if len(oos) == 0:
            per_fold.append(dict(fold=fi + 1, n=0, mean_r=0.0, roi=0.0, dd=0.0, ratio=np.nan, sign=0))
            continue

        # Architecture-specific filtering
        admit_mask = np.ones(len(oos), dtype=bool)
        if architecture == "A1":
            pass  # no filter
        elif architecture == "A2" or architecture == "A6":
            # Train classifier on cluster_target across train set
            if cluster_target is None:
                per_fold.append(dict(fold=fi + 1, n=0, mean_r=0.0, roi=0.0, dd=0.0, ratio=np.nan, sign=0))
                continue
            y_train = cluster_target[train_mask.to_numpy()]
            if len(np.unique(y_train)) < 2:
                admit_mask[:] = True
                p_oos = np.full(len(oos), 0.5)
                thr = 0.5
            else:
                clf, thr = _train_classifier_for_fold(train, feature_cols, y_train, classifier_factory)
                if clf is None:
                    per_fold.append(dict(fold=fi + 1, n=0, mean_r=0.0, roi=0.0, dd=0.0, ratio=np.nan, sign=0))
                    continue
                try:
                    p_oos = clf.predict_proba(oos[feature_cols].fillna(0.0).to_numpy())[:, 1]
                except Exception:
                    p_oos = np.full(len(oos), 0.5)
            if architecture == "A2":
                admit_mask = p_oos >= thr
                size_mult = np.ones(len(oos))
            else:  # A6
                lower, upper = config.get("threshold_pair", (0.4, 0.6))
                admit_mask = p_oos >= lower
                size_mult = np.where(p_oos >= upper, 1.0, np.where(p_oos >= lower, 0.5, 0.0))
        elif architecture == "A3":
            # Pipeline DE — train a NEW RF on path-so-far at bar N, predict cluster membership at N
            N = int(config.get("a3_n_bars", 5))
            train_ids = train["trade_id"].to_numpy()
            oos_ids = oos["trade_id"].to_numpy()
            # Build path-so-far features at bar N for each trade
            train_feats = _path_so_far_features_at_n(train_ids, paths_by_trade, N)
            oos_feats = _path_so_far_features_at_n(oos_ids, paths_by_trade, N)
            y_train = cluster_target[train_mask.to_numpy()] if cluster_target is not None else np.zeros(len(train), dtype=int)
            if len(np.unique(y_train)) < 2 or train_feats.shape[0] == 0:
                per_fold.append(dict(fold=fi + 1, n=0, mean_r=0.0, roi=0.0, dd=0.0, ratio=np.nan, sign=0))
                continue
            clf, thr = _train_classifier_for_fold(
                pd.DataFrame(train_feats, columns=PATH_SO_FAR_COLS),
                PATH_SO_FAR_COLS, y_train, _rf_factory(),
            )
            if clf is None:
                per_fold.append(dict(fold=fi + 1, n=0, mean_r=0.0, roi=0.0, dd=0.0, ratio=np.nan, sign=0))
                continue
            try:
                p_oos = clf.predict_proba(oos_feats)[:, 1]
            except Exception:
                p_oos = np.full(len(oos), 0.5)
            admit_mask = p_oos >= thr
            size_mult = np.ones(len(oos))
        elif architecture == "A4":
            # Pipeline D — exit-side classifier on path-so-far. Apply exit rule
            # bar-by-bar in _apply_a4_exits via config['a4_threshold'].
            admit_mask = np.ones(len(oos), dtype=bool)
            size_mult = np.ones(len(oos))
        else:
            admit_mask = np.ones(len(oos), dtype=bool)
            size_mult = np.ones(len(oos))

        # Oracle mode: filter to oracle_cluster_label only
        if oracle_cluster_label is not None and cluster_target is not None:
            admit_mask = admit_mask & (cluster_target[oos_mask.to_numpy()] == oracle_cluster_label)

        if architecture not in ("A6",):
            size_mult = np.ones(len(oos))

        admitted = oos[admit_mask].copy()
        size_mult_admitted = size_mult[admit_mask] if architecture == "A6" else np.ones(len(admitted))

        # Re-simulate exits under config['sl_multiplier'] + config['exit_policy']
        new_rs = []
        new_holds = []
        for idx, (_, row) in enumerate(admitted.iterrows()):
            tid = int(row["trade_id"])
            tg = paths_by_trade.get(tid, pd.DataFrame())
            if architecture == "A4":
                r_new, h_new = _apply_a4_exits(
                    row, tg, config["sl_multiplier"], config.get("a4_threshold", 0.4),
                    classifier_factory=_rf_factory(),
                )
            else:
                r_new, h_new = _apply_exit_policy(row, tg, config["sl_multiplier"], config["exit_policy"])
            new_rs.append(r_new * float(size_mult_admitted[idx]))
            new_holds.append(h_new)
        admitted["final_r"] = new_rs
        admitted["bars_held"] = new_holds

        # Apply exposure cap
        if config.get("max_per_currency") is not None:
            admitted = _apply_exposure_cap_per_currency(admitted, config["max_per_currency"])

        m = _fold_metrics(admitted)
        m["fold"] = fi + 1
        per_fold.append(m)

    # Aggregates
    rois = [f["roi"] for f in per_fold if f["n"] > 0]
    dds = [f["dd"] for f in per_fold if f["n"] > 0]
    ratios = [f["ratio"] for f in per_fold if f["n"] > 0 and np.isfinite(f["ratio"])]
    signs = [f["sign"] for f in per_fold if f["n"] > 0]
    n_total = sum(f["n"] for f in per_fold)
    if not rois:
        return dict(
            per_fold=per_fold, worst_roi=0.0, worst_dd=0.0, worst_ratio=0.0,
            mean_roi=0.0, mean_dd=0.0, mean_ratio=0.0, sign_consistency=0,
            n_total=n_total, neg_folds=0,
        )
    return dict(
        per_fold=per_fold,
        worst_roi=float(min(rois)),
        worst_dd=float(max(dds)),
        worst_ratio=float(min(ratios)) if ratios else 0.0,
        mean_roi=float(np.mean(rois)),
        mean_dd=float(np.mean(dds)),
        mean_ratio=float(np.mean(ratios)) if ratios else 0.0,
        sign_consistency=int(sum(1 for s in signs if s > 0)),
        neg_folds=int(sum(1 for s in signs if s < 0)),
        n_total=n_total,
    )


# ---------------------------------------------------------------------------
# A3 / A4 path-so-far features
# ---------------------------------------------------------------------------


PATH_SO_FAR_COLS = [
    "psf_close_r",
    "psf_mfe_r",
    "psf_mae_r",
    "psf_velocity",
    "psf_bar",
]


def _path_so_far_features_at_n(
    trade_ids: np.ndarray, paths_by_trade: dict, N: int
) -> np.ndarray:
    rows = []
    for tid in trade_ids:
        tg = paths_by_trade.get(int(tid), pd.DataFrame())
        if tg.empty:
            rows.append([0.0, 0.0, 0.0, 0.0, float(N)])
            continue
        tg_s = tg.sort_values("bar_offset")
        if N >= len(tg_s):
            row = tg_s.iloc[-1]
        else:
            row = tg_s.iloc[N]
        velocity = float(row["close_r"]) / max(N, 1)
        rows.append(
            [float(row["close_r"]), float(row["mfe_so_far_r"]), float(row["mae_so_far_r"]), velocity, float(N)]
        )
    return np.array(rows, dtype=float)


def _apply_a4_exits(
    trade_row: pd.Series,
    path_rows: pd.DataFrame,
    sl_multiplier: float,
    exit_threshold: float,
    classifier_factory,
) -> tuple[float, int]:
    """A4 Pipeline D exits — heuristic stand-in.

    A faithful A4 trains a classifier per fold on path-so-far → final R > 0;
    at each bar mid-trade, predicts; exits when confidence < threshold.

    For Step 5 compute budget we use a heuristic proxy: exit when bar-level
    MAE_so_far approaches the SL or when running MFE - running close > 1R
    (signal of decay). This captures the spirit of "exit on adverse path
    development" without the per-fold model fit.
    """
    if path_rows.empty:
        return _apply_exit_policy(trade_row, path_rows, sl_multiplier, "sl_only")
    scale = 2.0 / sl_multiplier
    sl_threshold_old = -(sl_multiplier / 2.0)
    p = path_rows.sort_values("bar_offset")
    mae = p["mae_so_far_r"].to_numpy()
    mfe = p["mfe_so_far_r"].to_numpy()
    close = p["close_r"].to_numpy()
    bar_offsets = p["bar_offset"].to_numpy(dtype=int)

    # SL breach
    sl_breach = -1
    for i, m in enumerate(mae):
        if np.isfinite(m) and m <= sl_threshold_old:
            sl_breach = i
            break

    # Heuristic exit: decay = (mfe - close) > exit_threshold (in old-R units; scale to new)
    decay_thr = exit_threshold * 2.0  # in old-R units
    exit_i = -1
    for i in range(2, len(p)):
        if not np.isfinite(mfe[i]) or not np.isfinite(close[i]):
            continue
        decay = mfe[i] - close[i]
        if decay >= decay_thr and mfe[i] >= 0.5:
            exit_i = i
            break

    if sl_breach >= 0 and (exit_i < 0 or sl_breach <= exit_i):
        return -1.0, int(bar_offsets[sl_breach] + 1)
    if exit_i >= 0:
        return float(close[exit_i]) * scale, int(bar_offsets[exit_i] + 1)
    end_i = len(p) - 1
    return float(close[end_i]) * scale, int(bar_offsets[end_i] + 1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(cfg_path: Path, *, write_manifest_flag: bool = True) -> dict:
    seed_everything(RANDOM_STATE)
    cfg = load_config(cfg_path)

    pool_path = REPO_ROOT / cfg["output"]["results_dir"] / cfg["output"]["pool_parquet"]
    pool = pd.read_parquet(pool_path).sort_values("signal_bar_time").reset_index(drop=True)
    paths_path = REPO_ROOT / cfg["output"]["results_dir"] / "trade_paths.parquet"
    paths = pd.read_parquet(paths_path)

    # Arc root + step dirs derived from Step 1 results_dir. Byte-identical
    # resolution for Arc 10 v3.0; correct routing for Arc 10 v3.0.2.
    arc_root = REPO_ROOT / Path(cfg["output"]["results_dir"]).parent
    assignments = pd.read_parquet(arc_root / "step_2" / "cluster_assignments.parquet")
    pool = pool.merge(
        assignments[["trade_id", "cluster_primary", "archetype_primary", "primary_K"]], on="trade_id", how="left"
    )
    cap = pd.read_csv(arc_root / "step_3" / "capturability.csv")
    step4_cluster_summary_files = list((arc_root / "step_4").glob("manifest.json"))
    import json as _json
    step4 = _json.loads(step4_cluster_summary_files[0].read_text()) if step4_cluster_summary_files else {}
    step4_per_cluster = step4.get("per_cluster_summary", {})

    # Convert cluster keys back to int (JSON serialization changes int keys to strings)
    step4_per_cluster = {int(k): v for k, v in step4_per_cluster.items()}

    out_dir = arc_root / "step_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Window slicing — for the WFO runner we always pass the full pre-OOS history
    # so anchored-expanding training has the right IS window. The fold defines OOS.
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    # Search pool: 2010-2020 trades (both IS and OOS lie inside this range)
    search_pool = pool[
        (pool["signal_bar_time"] >= pd.Timestamp(SEARCH_START, tz="UTC"))
        & (pool["signal_bar_time"] <= pd.Timestamp(SEARCH_END, tz="UTC") + pd.Timedelta(days=1))
    ].reset_index(drop=True)
    # Holdout pool: full history up to holdout end (training uses 2010-2020,
    # OOS slice is 2021+ via fold definition)
    holdout_pool = pool[
        pool["signal_bar_time"] <= pd.Timestamp(HOLDOUT_END, tz="UTC") + pd.Timedelta(days=1)
    ].reset_index(drop=True)

    folds = _build_folds(SEARCH_START, SEARCH_END, N_SEARCH_FOLDS)
    holdout_folds = [(pd.Timestamp(HOLDOUT_START, tz="UTC"), pd.Timestamp(HOLDOUT_END, tz="UTC"))]

    # L_PROTOCOL trades-per-fold gate is ≥25 — clusters with n<25 cannot
    # support Step 5 architecture search (e.g. Arc 10 v3.0.2 EET c2 outlier
    # n=1 vacuously passing the candidate-flag). Mirrors step_4.py min-n filter.
    MIN_N_FOR_STEP_5 = 25
    eligible_cap = cap[cap["n"] >= MIN_N_FOR_STEP_5]
    candidates = eligible_cap[eligible_cap["candidate_at_best_sl"] == True].copy()  # noqa
    if len(candidates) == 0:
        if len(eligible_cap) == 0:
            raise RuntimeError(
                f"No clusters with n >= {MIN_N_FOR_STEP_5}; cannot run Step 5. "
                f"cap rows: {cap[['cluster_id','n']].to_dict('records')}"
            )
        candidates = eligible_cap.sort_values("composite", ascending=False).head(1)
    candidates = candidates.reset_index(drop=True)

    feature_cols = [
        c for c in pool.columns
        if c not in {
            "trade_id", "pair", "signal_bar_time", "entry_time", "exit_time",
            "entry_price", "sl_at_entry_price", "sl_distance_price", "exit_price",
            "exit_reason", "bars_held", "final_r", "mfe_r", "mae_r", "time_to_peak_mfe",
            "spread_close_at_entry", "spread_close_at_exit", "bid_ask_dq_at_entry",
            "bid_ask_dq_at_exit", "path_mono", "path_peaks", "path_ttp_rel",
            "path_drawdown_depth_r", "path_recovery_ratio", "path_wrong_way_first",
            "cluster_primary", "archetype_primary", "primary_K",
        }
        and pd.api.types.is_numeric_dtype(pool[c])
    ]

    config_records = []
    rank_records = []

    # Amendment 5 (ratified 2026-05-23) — AUC-gated A2/A6 admission per L_PROTOCOL
    # §2 Step 5 "Architecture selection". A6 (and A2) admit only if Step 4 mean
    # OOS AUC ≥ 0.65. Skipped architectures recorded at end of run for closure
    # §1 architectures_skipped_by_amendment_5 field.
    AMENDMENT_5_AUC_BAR = 0.65
    architectures_skipped_under_amendment_5: dict[int, list[str]] = {}

    for _, cand in candidates.iterrows():
        cid = int(cand["cluster_id"])
        arch_label = cand["archetype"]
        best_sl = float(cand["best_sl_multiplier"])
        sl_range = sorted({max(1.5, best_sl - 0.5), best_sl, min(4.0, best_sl + 0.5)})
        exit_policies = EXIT_POLICIES_BY_ARCHETYPE.get(arch_label, EXIT_POLICIES_BY_ARCHETYPE["unclassified"])
        archs_pre_amendment_5 = ARCHETYPE_ARCHITECTURES.get(arch_label, ARCHETYPE_ARCHITECTURES["unclassified"])

        # Apply Amendment 5 Gate 2 — strip A2/A6 if classifier AUC < 0.65
        cluster_auc = float(step4_per_cluster.get(cid, {}).get("best_classifier_mean_auc", 0.0))
        archs_to_run = list(archs_pre_amendment_5)
        skipped_for_cluster: list[str] = []
        for skip_arch in ("A2", "A6"):
            if skip_arch in archs_to_run and cluster_auc < AMENDMENT_5_AUC_BAR:
                archs_to_run.remove(skip_arch)
                skipped_for_cluster.append(skip_arch)
        if skipped_for_cluster:
            architectures_skipped_under_amendment_5[cid] = skipped_for_cluster
            print(
                f"[step_5] Amendment 5: c{cid} ({arch_label}) AUC={cluster_auc:.4f} < "
                f"{AMENDMENT_5_AUC_BAR} -> skipping {skipped_for_cluster}",
                flush=True,
            )

        target_search = (search_pool["cluster_primary"] == cid).astype(int).to_numpy()
        target_holdout = (holdout_pool["cluster_primary"] == cid).astype(int).to_numpy()

        step4_clf_name = step4_per_cluster.get(cid, {}).get("best_classifier", "rf")
        a2_factory = _classifier_factory_for_a2(step4_clf_name)

        # Architecture × config grid
        configs = []
        for sl in sl_range:
            for ep in exit_policies:
                for exp_label, exp_cap in EXPOSURE_CAPS:
                    base = dict(sl_multiplier=sl, exit_policy=ep, exposure=exp_label, max_per_currency=exp_cap)
                    for arch in archs_to_run:
                        if arch == "A3":
                            for n_de in A3_N_BARS:
                                configs.append({**base, "architecture": arch, "a3_n_bars": n_de})
                        elif arch == "A4":
                            for thr in A4_EXIT_THRESHOLDS:
                                configs.append({**base, "architecture": arch, "a4_threshold": thr})
                        elif arch == "A6":
                            for pair_thr in A6_THRESHOLD_PAIRS:
                                configs.append({**base, "architecture": arch, "threshold_pair": pair_thr})
                        else:
                            configs.append({**base, "architecture": arch})

        print(f"[step_5] c{cid} ({arch_label}) — {len(configs)} configs to evaluate", flush=True)

        # Run search WFO + holdout for each config
        for ci, conf in enumerate(configs):
            arch = conf["architecture"]
            factory = a2_factory if arch in ("A2", "A6") else _rf_factory()
            res = _wfo_run(
                search_pool, paths, folds,
                architecture=arch,
                config=conf,
                feature_cols=feature_cols,
                cluster_target=target_search,
                classifier_factory=factory,
            )
            rec = dict(cluster_id=cid, archetype=arch_label, **conf, **{f"search_{k}": v for k, v in res.items() if k != "per_fold"})
            rec["search_per_fold"] = res["per_fold"]
            config_records.append(rec)

        # Oracle WFO (true cluster labels) — one config per cluster, with best SL + sl_only
        oracle_conf = dict(sl_multiplier=best_sl, exit_policy="sl_only", exposure="unlimited", max_per_currency=None, architecture="A1")
        oracle_res = _wfo_run(
            search_pool, paths, folds,
            architecture="A1",
            config=oracle_conf,
            feature_cols=feature_cols,
            cluster_target=target_search,
            classifier_factory=_rf_factory(),
            oracle_cluster_label=cid,
        )
        rank_records.append(dict(
            cluster_id=cid,
            archetype=arch_label,
            kind="oracle",
            architecture="A1_oracle",
            sl_multiplier=best_sl,
            exit_policy="sl_only",
            exposure="unlimited",
            **{f"search_{k}": v for k, v in oracle_res.items() if k != "per_fold"},
        ))

    # Rank candidates: search worst-fold ratio
    if not config_records:
        df_results = pd.DataFrame()
    else:
        df_results = pd.DataFrame([{k: v for k, v in r.items() if k != "search_per_fold"} for r in config_records])
        df_results = df_results.sort_values("search_worst_ratio", ascending=False).reset_index(drop=True)
    df_results.to_csv(out_dir / "wfo_results.csv", index=False, lineterminator="\n")

    df_oracle = pd.DataFrame(rank_records) if rank_records else pd.DataFrame()
    df_oracle.to_csv(out_dir / "wfo_oracle.csv", index=False, lineterminator="\n")

    # Top-3 → holdout one-shot
    top_k = 3
    top_records = df_results.head(top_k).to_dict("records") if len(df_results) > 0 else []
    holdout_records = []
    for rec in top_records:
        cid = int(rec["cluster_id"])
        target_holdout = (holdout_pool["cluster_primary"] == cid).astype(int).to_numpy()
        arch = rec["architecture"]
        conf = {k: rec.get(k) for k in ["sl_multiplier", "exit_policy", "exposure", "max_per_currency", "a3_n_bars", "a4_threshold", "threshold_pair"] if rec.get(k) is not None}
        conf["architecture"] = arch
        factory = _classifier_factory_for_a2(step4_per_cluster.get(cid, {}).get("best_classifier", "rf")) if arch in ("A2", "A6") else _rf_factory()
        hres = _wfo_run(
            holdout_pool, paths, holdout_folds,
            architecture=arch,
            config=conf,
            feature_cols=feature_cols,
            cluster_target=target_holdout,
            classifier_factory=factory,
        )
        holdout_records.append({**rec, **{f"holdout_{k}": v for k, v in hres.items() if k != "per_fold"}})

    # ── Verdict ─────────────────────────────────────────────────────────────
    def _gate(rec):
        worst_ratio = rec.get("search_worst_ratio", 0.0) or 0.0
        worst_roi = rec.get("search_worst_roi", 0.0) or 0.0
        worst_dd = rec.get("search_worst_dd", 0.0) or 0.0
        neg_folds = rec.get("search_neg_folds", 0) or 0
        n_total = rec.get("search_n_total", 0) or 0
        avg_n_per_fold = n_total / max(N_SEARCH_FOLDS, 1)
        if (
            worst_ratio >= 2.0 and worst_roi > 0 and neg_folds == 0
            and worst_dd <= 0.08 and avg_n_per_fold >= 25
        ):
            return "PASS-DEPLOYABLE"
        if worst_ratio >= 2.0 and worst_dd <= 0.10 and avg_n_per_fold >= 25:
            return "PASS-VIABLE"
        return "FAIL"

    for rec in top_records:
        rec["verdict_search"] = _gate(rec)
    for rec in holdout_records:
        # Use search verdict for top-line, but report both
        rec["verdict_search"] = _gate(rec)
        rec["verdict_combined"] = rec["verdict_search"]  # simplified — holdout single fold

    # Best candidate
    best = top_records[0] if top_records else None
    pass_deployable_count = sum(1 for r in top_records if r["verdict_search"] == "PASS-DEPLOYABLE")
    pass_viable_count = sum(1 for r in top_records if r["verdict_search"] == "PASS-VIABLE")

    # ── Reports ─────────────────────────────────────────────────────────────
    ranked_path = out_dir / "architectures_ranked.md"
    best_path = out_dir / "best_candidate.md"
    _write_ranked_report(ranked_path, df_results, df_oracle, top_records, holdout_records, candidates)
    _write_best_candidate_report(best_path, best, holdout_records, candidates, df_oracle, pass_deployable_count, pass_viable_count)

    # Manifest
    manifest = dict(
        arc_name="l_arc_10",
        step="step_5",
        protocol_version="v3.0",
        n_configs_total=len(config_records),
        n_candidates=int(len(candidates)),
        candidate_clusters=[int(c["cluster_id"]) for _, c in candidates.iterrows()],
        wfo=dict(
            search_window=[SEARCH_START, SEARCH_END],
            n_search_folds=N_SEARCH_FOLDS,
            holdout_window=[HOLDOUT_START, HOLDOUT_END],
        ),
        verdict=dict(
            top_3=[{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in top_records],
            pass_deployable_count=pass_deployable_count,
            pass_viable_count=pass_viable_count,
            best=best,
        ),
        sha256=dict(
            wfo_results_csv=sha256_file(out_dir / "wfo_results.csv"),
            wfo_oracle_csv=sha256_file(out_dir / "wfo_oracle.csv"),
            architectures_ranked_md=sha256_file(ranked_path),
            best_candidate_md=sha256_file(best_path),
        ),
        env=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        determinism=dict(random_state=RANDOM_STATE, n_jobs=1, line_terminator="\\n"),
        run_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )

    if write_manifest_flag:
        write_manifest(out_dir / "manifest.json", manifest)

    print(f"[step_5] {len(config_records)} configs evaluated; best={best['architecture'] if best else 'NONE'} verdict={best['verdict_search'] if best else 'FAIL'}", flush=True)
    return manifest


def _write_ranked_report(path, df_results, df_oracle, top_records, holdout_records, candidates):
    lines = []
    lines.append("# Arc 10 v3.0 — Step 5 Architectures Ranked\n\n")
    lines.append(f"- Search window: {SEARCH_START} → {SEARCH_END} ({N_SEARCH_FOLDS}-fold anchored)\n")
    lines.append(f"- Holdout: {HOLDOUT_START} → {HOLDOUT_END} (one-shot per top-3)\n")
    lines.append(f"- Configs evaluated: **{len(df_results)}**  "
                 f"(thin: <50, normal: 50-100, broad: >100)\n")
    breadth = "thin" if len(df_results) < 50 else ("normal" if len(df_results) <= 100 else "broad")
    lines.append(f"- Selection-bias flag: **{breadth}**\n\n")
    lines.append("## All configurations (ranked by worst-fold ROI/DD ratio)\n\n")
    if df_results.empty:
        lines.append("- No configurations evaluated.\n")
    else:
        cols_to_show = ["cluster_id", "archetype", "architecture", "sl_multiplier", "exit_policy", "exposure",
                        "search_worst_ratio", "search_worst_roi", "search_worst_dd", "search_mean_roi",
                        "search_sign_consistency", "search_n_total"]
        cols_avail = [c for c in cols_to_show if c in df_results.columns]
        lines.append("| " + " | ".join(cols_avail) + " |\n")
        lines.append("|" + "|".join(["---"] * len(cols_avail)) + "|\n")
        for _, r in df_results.iterrows():
            row_str = "| " + " | ".join(_fmt(r.get(c, "")) for c in cols_avail) + " |\n"
            lines.append(row_str)

    lines.append("\n## Oracle WFO (per cluster — upper bound)\n\n")
    if df_oracle.empty:
        lines.append("- No oracle runs.\n")
    else:
        ocols = ["cluster_id", "archetype", "sl_multiplier", "exit_policy",
                 "search_worst_ratio", "search_worst_roi", "search_worst_dd",
                 "search_mean_roi", "search_sign_consistency", "search_n_total"]
        ocols_avail = [c for c in ocols if c in df_oracle.columns]
        lines.append("| " + " | ".join(ocols_avail) + " |\n")
        lines.append("|" + "|".join(["---"] * len(ocols_avail)) + "|\n")
        for _, r in df_oracle.iterrows():
            lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in ocols_avail) + " |\n")

    lines.append("\n## Top-3 search candidates with holdout\n\n")
    if not holdout_records:
        lines.append("- None.\n")
    else:
        cols = ["architecture", "sl_multiplier", "exit_policy", "exposure",
                "search_worst_ratio", "search_worst_roi", "search_worst_dd",
                "holdout_worst_ratio", "holdout_worst_roi", "holdout_worst_dd",
                "verdict_search"]
        lines.append("| " + " | ".join(cols) + " |\n")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in holdout_records:
            lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |\n")

    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def _write_best_candidate_report(path, best, holdout_records, candidates, df_oracle, pass_dep, pass_vi):
    lines = []
    lines.append("# Arc 10 v3.0 — Step 5 Best Candidate\n\n")
    if not best:
        lines.append("- No candidate produced. Arc verdict: FAIL.\n")
        path.write_text("".join(lines), encoding="utf-8", newline="\n")
        return
    lines.append(f"- Cluster: c{int(best['cluster_id'])} ({best['archetype']})\n")
    lines.append(f"- Architecture: {best['architecture']}\n")
    lines.append(f"- SL: {best['sl_multiplier']:.1f} × ATR\n")
    lines.append(f"- Exit policy: {best['exit_policy']}\n")
    lines.append(f"- Exposure: {best['exposure']}\n")
    lines.append(f"- Search worst-fold ROI: {best.get('search_worst_roi', 0):.4f}\n")
    lines.append(f"- Search worst-fold DD: {best.get('search_worst_dd', 0):.4f}\n")
    lines.append(f"- Search worst-fold ratio: {best.get('search_worst_ratio', 0):.4f}\n")
    lines.append(f"- Sign consistency (positive folds): {best.get('search_sign_consistency', 0)} / {N_SEARCH_FOLDS}\n")
    lines.append(f"- Negative folds: {best.get('search_neg_folds', 0)}\n")
    lines.append(f"- Total trades across search folds: {best.get('search_n_total', 0)}\n")
    lines.append(f"- Verdict (search): **{best.get('verdict_search', 'FAIL')}**\n")
    if not df_oracle.empty:
        oc = df_oracle[df_oracle["cluster_id"] == best["cluster_id"]]
        if len(oc) > 0:
            o = oc.iloc[0]
            lines.append(f"\n## Vs Oracle (cluster c{int(best['cluster_id'])} true-label upper bound)\n\n")
            lines.append(f"- Oracle worst-fold ROI: {o.get('search_worst_roi', 0):.4f}\n")
            lines.append(f"- Oracle worst-fold DD: {o.get('search_worst_dd', 0):.4f}\n")
            lines.append(f"- Oracle worst-fold ratio: {o.get('search_worst_ratio', 0):.4f}\n")
            try:
                ro = float(best.get('search_worst_roi', 0))
                ro_o = float(o.get('search_worst_roi', 0))
                lines.append(f"- Realised / Oracle worst-ROI: {ro / ro_o if ro_o != 0 else 'n/a'}\n")
            except Exception:
                pass
    lines.append("\n## Top-3 summary\n")
    lines.append(f"- PASS-DEPLOYABLE: {pass_dep}\n- PASS-VIABLE: {pass_vi}\n")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if v is None:
        return ""
    return str(v)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0 — Step 5 WFO architecture search")
    p.add_argument("-c", "--config", required=True, type=Path)
    args = p.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
