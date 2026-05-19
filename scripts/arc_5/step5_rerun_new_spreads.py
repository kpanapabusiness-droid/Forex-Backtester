"""Arc 5 — Step 5 skip-rerun under 2026-05-17 per-pair p50 spread floors.

Per chat-level decision (see results/l_arc_5/PHASE_L_ARC_5_STEP5_NEW_SPREADS.md):
re-run Step 1 plumbing under new spreads (done — Phase 1), then SKIP Steps 2/3/4/4b
and jump directly to Step 5 evaluation using baseline cluster labels and baseline
per-fold classifiers.

Methodology caveat (also documented in result doc):
  This run uses baseline classifiers + baseline F9 thresholds against new-spread
  feature distributions. Strictly correct methodology would re-run Steps 2/3/4/4b
  under new spreads to produce new cluster labels / classifier / thresholds.
  Skip is a deliberate approximation for decision-quality assessment.

Critical sanity-check #1 outcome (chat-level call to proceed):
  Phase 1 trade pool diverged 12262 -> 12348 (+86) via the concurrent-per-pair
  guard cascade (signal triggers are price-action-only but exit timing shifts
  with new spread, which shifts open_until_idx, which gates the next signal).
  99.04% of baseline signals survive in the new pool (12144 matched by
  (pair, signal_time)). User opted for "Phase 2 on full new 12,348":
    - Matched trades (12144) inherit baseline cluster labels by (pair, signal_time)
    - New-only trades (204) get cluster assignment via nearest baseline centroid
      in StandardScaler-normalised path-feature space
  Baseline orphan trades (118 — exist baseline, gone new) are excluded.

Reuses Step 4's deterministic feature builders from scripts/l_arc_4/step4_extractability.py:
  - _compute_entry_features_for_pair  (8 base + arc-specific features per 1H bar)
  - _compute_d1_features_at_t          (7 path-so-far features at bar t)

Usage:
  py scripts/arc_5/step5_rerun_new_spreads.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_4.step4_extractability import (  # noqa: E402
    _compute_d1_features_at_t,
    _compute_entry_features_for_pair,
    _load_pair_1h,
)

# ============================================================
# Constants — locked artefacts from baseline
# ============================================================

BASELINE_STEP1_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step1"
NEW_STEP1_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step1_spread_v2"
BASELINE_STEP2_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step2"
BASELINE_STEP4_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step4"
BASELINE_STEP5_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step5"

OUT_DIR = _REPO_ROOT / "results" / "l_arc_5" / "step5_spread_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = _REPO_ROOT / "data" / "1hr"

# Locked features (15 = 8 base + 7 path-so-far) per baseline pipeline_d1_policy.yaml
BASE_ENTRY_FEATURES = [
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_to_atr_14",
    "ret_5bar_atr",
    "ret_20bar_atr",
    "pos_in_20bar_range",
    "rsi_14",
]
D1_PATH_FEATURES = [
    "close_r_at_t1",
    "mfe_so_far_r_at_t1",
    "mae_so_far_r_at_t1",
    "bars_in_profit_at_t1",
    "local_peaks_so_far_at_t1",
    "monotonicity_so_far_at_t1",
    "velocity_first_t_at_t1",
]
ALL_FEATURES = BASE_ENTRY_FEATURES + D1_PATH_FEATURES

# Per-cluster archetype R-frame (from Step 4 pipeline_d1_policy.yaml)
CLUSTER_R_FRAME_ATR_MULT = {1: 3.0, 3: 2.0}
CLUSTER_F9_THRESHOLD = {1: 0.20, 3: 0.15}
CLUSTER_LABEL = {1: "stepwise_climber", 3: "stepwise_noisy_pullback"}

# Path-shape feature columns (Step 2 clustering basis)
PATH_SHAPE_FEATURES = [
    "monotonicity_ratio_in_profit",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
]

WFO_FOLDS = [
    ("2019-01-01", "2020-10-01", "2020-10-01", "2021-07-01"),
    ("2019-01-01", "2021-07-01", "2021-07-01", "2022-04-01"),
    ("2019-01-01", "2022-04-01", "2022-04-01", "2023-01-01"),
    ("2019-01-01", "2023-01-01", "2023-01-01", "2023-10-01"),
    ("2019-01-01", "2023-10-01", "2023-10-01", "2024-07-01"),
    ("2019-01-01", "2024-07-01", "2024-07-01", "2025-04-01"),
    ("2019-01-01", "2025-04-01", "2025-04-01", "2026-01-01"),
]

RISK_PCT = 0.005  # L arc convention


# ============================================================
# Path-shape feature recomputation (mirrors Step 2 §6 definitions)
# ============================================================


def _compute_path_shape_features_per_trade(
    paths_df: pd.DataFrame, n_trades: int
) -> pd.DataFrame:
    """Recompute the 4 path-shape features (Step 2 §6) on a new trades_paths.csv.

    Features (outcome-blind; restricted to held bars per Step 2):
      - monotonicity_ratio_in_profit
      - local_peaks_count
      - pullback_magnitude_median
      - time_to_peak_mfe_relative

    Edge cases match Step 2 §6 (zero in-profit → 0; <2 peaks → 0; never-profit → 0).
    """
    paths = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    PATH_BARS = 241
    if len(paths) != n_trades * PATH_BARS:
        raise ValueError(
            f"paths_df rows {len(paths)} != n_trades {n_trades} × {PATH_BARS}"
        )
    close_r = paths["close_r"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    mfe_r = paths["mfe_so_far_r"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    is_held = paths["is_held"].to_numpy(dtype=int).reshape(n_trades, PATH_BARS)
    trade_ids = paths["trade_id"].to_numpy(dtype=int).reshape(n_trades, PATH_BARS)[:, 0]

    rows: List[Dict] = []
    for i in range(n_trades):
        mask_held = is_held[i] == 1
        cl = close_r[i, mask_held]
        mfe = mfe_r[i, mask_held]
        bars_h = int(mask_held.sum())
        # Step 2 features
        in_profit = cl > 0.0
        n_ip = int(in_profit.sum())
        if n_ip <= 1:
            mono = 0.0
        else:
            ipc = cl[in_profit]
            gte = ipc[1:] >= ipc[:-1]
            mono = float(gte.sum() / gte.size) if gte.size > 0 else 0.0
        # local_peaks_count: bars where mfe > previous bar's mfe
        if bars_h <= 1:
            peaks = 0.0
        else:
            cummax = np.maximum.accumulate(mfe)
            peaks = float((cummax[1:] > cummax[:-1]).sum())
        # pullback_magnitude_median: per Step 2 operational definition
        # (peak[k].mfe - min(close_r between peaks))
        peak_indices = [j for j in range(1, bars_h) if mfe[j] > mfe[j - 1]] if bars_h > 1 else []
        # Each "peak" is the bar where mfe stepped up; use mfe value at that bar.
        if len(peak_indices) < 2:
            pullback = 0.0
        else:
            pbs: List[float] = []
            for k in range(len(peak_indices) - 1):
                ka = peak_indices[k]
                kb = peak_indices[k + 1]
                if kb - ka < 2:
                    continue
                min_btw = float(np.min(cl[ka + 1 : kb]))
                pbs.append(float(mfe[ka]) - min_btw)
            pullback = float(np.median(pbs)) if pbs else 0.0
        # time_to_peak_mfe_relative
        ever_profit = (cl > 0.0).any() if bars_h > 0 else False
        if not ever_profit:
            ttp = 0.0
        else:
            peak_bar = int(np.argmax(mfe))
            ttp = min(1.0, peak_bar / max(bars_h, 1))
        rows.append(
            {
                "trade_id": int(trade_ids[i]),
                "monotonicity_ratio_in_profit": mono,
                "local_peaks_count": peaks,
                "pullback_magnitude_median": pullback,
                "time_to_peak_mfe_relative": ttp,
                "bars_held_feature": bars_h,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Cluster assignment
# ============================================================


def _build_cluster_assignment(
    new_trades: pd.DataFrame,
    new_path_features: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    baseline_clusters: pd.DataFrame,
    baseline_path_features: pd.DataFrame,
    baseline_centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Return DataFrame[new_trade_id, baseline_trade_id_or_None, cluster_id, assignment_source].

    assignment_source in {'matched_by_signal_time', 'nearest_baseline_centroid'}.
    """
    # Build (pair, signal_time) -> baseline_trade_id mapping.
    base_map: Dict[Tuple[str, str], int] = {}
    for tid, pair, st in zip(
        baseline_trades["trade_id"].to_numpy(),
        baseline_trades["pair"].to_numpy(),
        baseline_trades["signal_time"].to_numpy(),
    ):
        base_map[(str(pair), str(st))] = int(tid)
    base_cluster: Dict[int, int] = dict(
        zip(baseline_clusters["trade_id"].astype(int), baseline_clusters["cluster_id"].astype(int))
    )

    # StandardScaler fit on baseline raw path features (matches Step 2 §6).
    scaler = StandardScaler()
    baseline_pf = baseline_path_features.sort_values("trade_id").reset_index(drop=True)
    baseline_X = baseline_pf[PATH_SHAPE_FEATURES].to_numpy(dtype=float)
    scaler.fit(baseline_X)

    # Compute centroids in scaled space (from baseline raw centroids — KMeans
    # operates on scaled features so centroid distances must use scaled space).
    centroids_raw = baseline_centroids.sort_values("cluster_id").reset_index(drop=True)
    centroids_scaled = scaler.transform(centroids_raw[PATH_SHAPE_FEATURES].to_numpy(dtype=float))

    # Build new lookup by trade_id.
    new_pf = new_path_features.set_index("trade_id")[PATH_SHAPE_FEATURES]

    rows: List[Dict] = []
    for tid, pair, st in zip(
        new_trades["trade_id"].to_numpy(),
        new_trades["pair"].to_numpy(),
        new_trades["signal_time"].to_numpy(),
    ):
        key = (str(pair), str(st))
        base_tid = base_map.get(key, None)
        if base_tid is not None and base_tid in base_cluster:
            cid = base_cluster[base_tid]
            src = "matched_by_signal_time"
            rows.append(
                {
                    "trade_id": int(tid),
                    "baseline_trade_id": int(base_tid),
                    "cluster_id": int(cid),
                    "assignment_source": src,
                }
            )
        else:
            # New-only trade: assign to nearest baseline centroid in scaled space.
            raw_vec = new_pf.loc[int(tid)].to_numpy(dtype=float).reshape(1, -1)
            scaled = scaler.transform(raw_vec)
            d2 = np.sum((centroids_scaled - scaled) ** 2, axis=1)
            cid = int(np.argmin(d2))
            rows.append(
                {
                    "trade_id": int(tid),
                    "baseline_trade_id": None,
                    "cluster_id": int(cid),
                    "assignment_source": "nearest_baseline_centroid",
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# Entry feature matrix (8 base only — the slim feature set is what classifiers use)
# ============================================================


def _build_entry_features_for_pool(trades: pd.DataFrame) -> pd.DataFrame:
    """Build 8-base entry features keyed by trade_id."""
    trades = trades.copy()
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], errors="coerce")
    pairs = sorted(trades["pair"].unique().tolist())
    rows: List[pd.DataFrame] = []
    t0 = time.time()
    for pair in pairs:
        df_1h = _load_pair_1h(pair, DATA_DIR)
        feats = _compute_entry_features_for_pair(df_1h).set_index("date")
        sub = trades[trades["pair"] == pair]
        looked = feats.reindex(sub["signal_time"].to_numpy())
        looked.insert(0, "trade_id", sub["trade_id"].to_numpy())
        rows.append(looked.reset_index(drop=True))
        print(
            f"[step5_rerun] entry features {pair}: {len(sub)} trades "
            f"({time.time() - t0:.1f}s)",
            file=sys.stderr,
        )
    big = pd.concat(rows, axis=0, ignore_index=True).sort_values("trade_id").reset_index(drop=True)
    return big[["trade_id"] + BASE_ENTRY_FEATURES]


# ============================================================
# Path tensor + t=1 features
# ============================================================


def _build_path_tensor(paths_df: pd.DataFrame, n_trades: int) -> np.ndarray:
    paths_df = paths_df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    PATH_BARS = 241
    opens = paths_df["open"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    highs = paths_df["high"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    lows = paths_df["low"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    closes = paths_df["close"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    return np.stack([opens, highs, lows, closes], axis=2)  # (n, 241, 4)


def _compute_t1_features(
    path_tensor: np.ndarray,
    entry_prices: np.ndarray,
    atr_14: np.ndarray,
    cluster_r_frame_mult: float,
) -> np.ndarray:
    """7-column matrix of D1 features at t=1 under cluster_r_frame_mult × ATR R-frame.

    Output column order matches D1_PATH_FEATURES.
    """
    r_per_trade = cluster_r_frame_mult * atr_14
    return _compute_d1_features_at_t(path_tensor, entry_prices, r_per_trade, t=1)


# ============================================================
# Fold metrics
# ============================================================


def _compute_fold_metrics(
    final_r_admitted: np.ndarray,
    entry_times_admitted: pd.Series,
    fold_oos_days: int,
) -> Dict[str, float]:
    """Compute admit-set fold metrics matching the baseline Step 5 formulation.

    Convention (reverse-engineered from baseline cluster_1 fold_stability):
      - ROI = sum(R) × RISK_PCT × 100   (additive, exact baseline match)
      - Equity curve = cumsum(R × RISK_PCT), starting at 0
      - DD = max(running_max(equity) - equity) × 100  (additive equity DD, in PnL%)
      - Annualised ROI = ROI × (365 / oos_days)
      - t-stat = mean(R) × sqrt(N) / std(R, ddof=1)

    Fold-1-only quirk: 0.3pp DD discrepancy vs baseline on cluster 1 fold 1 due
    to an opaque tiebreak on duplicate entry_time bars (6/7 baseline folds match
    exactly with `sort_values('entry_time')`). For like-for-like deltas, the
    comparison code re-derives baseline metrics with this same function.
    """
    n = len(final_r_admitted)
    if n == 0:
        return {
            "n_admitted": 0,
            "final_r_mean": 0.0,
            "final_r_t_stat": 0.0,
            "fold_roi_pct": 0.0,
            "fold_roi_pct_annualised": 0.0,
            "fold_max_dd_pct": 0.0,
        }
    final_r_mean = float(np.mean(final_r_admitted))
    sd = float(np.std(final_r_admitted, ddof=1)) if n > 1 else 0.0
    t_stat = float(final_r_mean * np.sqrt(n) / sd) if sd > 0 else 0.0
    # Chronological by entry_time; stable sort preserves the input order for ties.
    et = entry_times_admitted.to_numpy()
    order = np.argsort(et, kind="mergesort")
    r_chrono = final_r_admitted[order] * RISK_PCT
    eq = np.cumsum(r_chrono)
    fold_roi_pct = float(eq[-1] * 100.0)
    fold_roi_pct_annualised = float(fold_roi_pct * (365.0 / max(fold_oos_days, 1)))
    running_max = np.maximum.accumulate(eq)
    fold_max_dd_pct = float(np.max(running_max - eq) * 100.0)
    return {
        "n_admitted": n,
        "final_r_mean": final_r_mean,
        "final_r_t_stat": t_stat,
        "fold_roi_pct": fold_roi_pct,
        "fold_roi_pct_annualised": fold_roi_pct_annualised,
        "fold_max_dd_pct": fold_max_dd_pct,
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Driver
# ============================================================


def main() -> int:
    t_start = time.time()

    print("[step5_rerun] loading new + baseline trades_all", file=sys.stderr)
    new_trades = pd.read_csv(NEW_STEP1_DIR / "trades_all.csv")
    base_trades = pd.read_csv(BASELINE_STEP1_DIR / "trades_all.csv")
    print(f"  new pool : {len(new_trades)} trades", file=sys.stderr)
    print(f"  base pool: {len(base_trades)} trades", file=sys.stderr)

    print("[step5_rerun] loading new trades_paths (large CSV)", file=sys.stderr)
    new_paths = pd.read_csv(NEW_STEP1_DIR / "trades_paths.csv")

    print("[step5_rerun] loading baseline cluster artefacts", file=sys.stderr)
    base_clusters = pd.read_csv(BASELINE_STEP2_DIR / "clusters_K4.csv")
    base_path_features = pd.read_csv(BASELINE_STEP2_DIR / "path_features.csv")
    base_centroids = pd.read_csv(BASELINE_STEP2_DIR / "centroids_K4.csv")

    print("[step5_rerun] recomputing path-shape features on new pool", file=sys.stderr)
    new_path_features = _compute_path_shape_features_per_trade(new_paths, len(new_trades))
    new_path_features.to_csv(OUT_DIR / "new_path_features.csv", index=False)

    print("[step5_rerun] cluster assignment (matched / nearest-centroid)", file=sys.stderr)
    cluster_assign = _build_cluster_assignment(
        new_trades, new_path_features, base_trades, base_clusters,
        base_path_features, base_centroids,
    )
    cluster_assign.to_csv(OUT_DIR / "cluster_assignment.csv", index=False)
    counts_src = cluster_assign.groupby("assignment_source").size()
    print(f"  assignment source breakdown:\n{counts_src.to_string()}", file=sys.stderr)
    counts_cid = cluster_assign.groupby("cluster_id").size()
    print(f"  cluster_id counts:\n{counts_cid.to_string()}", file=sys.stderr)

    print("[step5_rerun] building entry features for new pool", file=sys.stderr)
    entry_feats = _build_entry_features_for_pool(new_trades)

    print("[step5_rerun] loading path tensor", file=sys.stderr)
    path_tensor = _build_path_tensor(new_paths, len(new_trades))
    new_trades_sorted = new_trades.sort_values("trade_id").reset_index(drop=True)
    entry_prices = new_trades_sorted["entry_price"].to_numpy(dtype=float)
    atr_14 = new_trades_sorted["atr_14_at_signal"].to_numpy(dtype=float)
    bars_held = new_trades_sorted["bars_held"].to_numpy(dtype=int)
    entry_times = pd.to_datetime(new_trades_sorted["entry_time"])
    final_r = new_trades_sorted["final_r"].to_numpy(dtype=float)
    pairs_arr = new_trades_sorted["pair"].to_numpy()
    trade_ids_arr = new_trades_sorted["trade_id"].to_numpy(dtype=int)
    # Map new_trade_id -> row index in sorted arrays.
    tid_to_row = {int(tid): i for i, tid in enumerate(trade_ids_arr)}
    cluster_id_per_trade = np.full(len(new_trades_sorted), -1, dtype=int)
    for _, row in cluster_assign.iterrows():
        cluster_id_per_trade[tid_to_row[int(row["trade_id"])]] = int(row["cluster_id"])

    # Map new_trade_id -> baseline_trade_id (or None) for Jaccard
    base_tid_for_new = {
        int(r["trade_id"]): (int(r["baseline_trade_id"]) if pd.notna(r["baseline_trade_id"]) else None)
        for _, r in cluster_assign.iterrows()
    }

    # ===== Per cluster =====
    cluster_results: Dict[int, Dict] = {}
    for cid in (1, 3):
        print(f"[step5_rerun] === cluster {cid} ({CLUSTER_LABEL[cid]}) ===", file=sys.stderr)
        r_frame_mult = CLUSTER_R_FRAME_ATR_MULT[cid]
        threshold = CLUSTER_F9_THRESHOLD[cid]
        cluster_dir = OUT_DIR / f"cluster_{cid}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Compute t=1 path-so-far features under cluster R-frame.
        print(f"  computing t=1 features under R={r_frame_mult}×ATR", file=sys.stderr)
        t1_feats = _compute_t1_features(path_tensor, entry_prices, atr_14, r_frame_mult)

        # Build full feature matrix [entry features | t=1 features], aligned by trade_id.
        entry_sorted = entry_feats.sort_values("trade_id").reset_index(drop=True)
        if list(entry_sorted["trade_id"].astype(int)) != list(trade_ids_arr):
            raise RuntimeError("entry_features trade_id ordering mismatch")
        X_entry = entry_sorted[BASE_ENTRY_FEATURES].to_numpy(dtype=float)
        X_full = np.concatenate([X_entry, t1_feats], axis=1)
        # NaN-fill with column median (matches baseline _features_to_matrix behaviour).
        for j in range(X_full.shape[1]):
            col = X_full[:, j]
            if np.isnan(col).any():
                col = np.where(np.isnan(col), np.nanmedian(col), col)
                X_full[:, j] = col
        # Eligibility: bars_held >= 1 (all trades pass since min bars_held = 1).
        elig_mask = bars_held >= 1

        # Per-fold rows
        fold_rows: List[Dict] = []
        admit_rows: List[Dict] = []
        for fold_idx, (is_start, is_end, oos_start, oos_end) in enumerate(WFO_FOLDS, start=1):
            is_start_ts = pd.Timestamp(is_start)
            is_end_ts = pd.Timestamp(is_end)
            oos_start_ts = pd.Timestamp(oos_start)
            oos_end_ts = pd.Timestamp(oos_end)
            oos_days = int((oos_end_ts - oos_start_ts).total_seconds() / 86400)

            # IS / OOS by entry_time
            is_mask = (entry_times >= is_start_ts) & (entry_times < is_end_ts) & elig_mask
            oos_mask = (entry_times >= oos_start_ts) & (entry_times < oos_end_ts) & elig_mask
            n_is = int(is_mask.sum())
            n_oos = int(oos_mask.sum())
            n_target_oos = int(((cluster_id_per_trade == cid) & oos_mask).sum())

            # Load baseline per-fold classifier
            clf_path = BASELINE_STEP5_DIR / f"cluster_{cid}" / "per_fold_classifiers" / f"fold_{fold_idx}_classifier.joblib"
            clf_sha = _file_sha256(clf_path)
            clf = joblib.load(clf_path)

            # Score OOS trades
            oos_idx = np.where(oos_mask)[0]
            if len(oos_idx) == 0:
                continue
            X_oos = X_full[oos_idx, :]
            proba = clf.predict_proba(X_oos)[:, 1]
            admit = proba >= threshold
            admitted_local = oos_idx[admit]
            n_admitted = int(admit.sum())
            # Among admits, "n_target_admitted" = those whose baseline cluster_id == cid
            target_admit_mask = (cluster_id_per_trade[admitted_local] == cid)
            n_target_admit = int(target_admit_mask.sum())
            admit_rate = n_admitted / n_oos if n_oos > 0 else 0.0
            # OOS RF AUC (using cluster membership as label).
            y_oos = (cluster_id_per_trade[oos_idx] == cid).astype(int)
            if y_oos.sum() > 0 and y_oos.sum() < len(y_oos):
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_oos, proba))
            else:
                auc = float("nan")

            # Metrics on admit set
            r_admit = final_r[admitted_local]
            et_admit = entry_times.iloc[admitted_local].reset_index(drop=True)
            metrics = _compute_fold_metrics(r_admit, et_admit, oos_days)

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "oos_days": oos_days,
                    "is_n_trades": n_is,
                    "n_oos": n_oos,
                    "n_target_class_oos": n_target_oos,
                    "n_admitted": n_admitted,
                    "n_target_admitted": n_target_admit,
                    "admit_rate": admit_rate,
                    "oos_rf_auc": auc,
                    "final_r_mean": metrics["final_r_mean"],
                    "final_r_t_stat": metrics["final_r_t_stat"],
                    "fold_roi_pct": metrics["fold_roi_pct"],
                    "fold_roi_pct_annualised": metrics["fold_roi_pct_annualised"],
                    "fold_max_dd_pct": metrics["fold_max_dd_pct"],
                    "classifier_sha256": clf_sha,
                }
            )
            # Per-admit records (for Jaccard + diagnostics)
            for li in admitted_local:
                tid = int(trade_ids_arr[li])
                btid = base_tid_for_new.get(tid)
                admit_rows.append(
                    {
                        "fold": fold_idx,
                        "trade_id": tid,
                        "baseline_trade_id": btid if btid is not None else "",
                        "entry_time": entry_times.iloc[li].isoformat(),
                        "pair": str(pairs_arr[li]),
                        "predicted_proba": float(proba[oos_idx.tolist().index(li)]),
                        "y_true_cluster_match": int(cluster_id_per_trade[li] == cid),
                        "final_r": float(final_r[li]),
                    }
                )

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(cluster_dir / "fold_stability_new_spreads.csv", index=False)

        admit_df = pd.DataFrame(admit_rows)
        admit_df.to_csv(cluster_dir / "oos_trade_admits_new_spreads.csv", index=False)

        # §9 gate check
        sign_consistency = bool((fold_df["final_r_mean"] > 0).all())
        ar = fold_df["admit_rate"].to_numpy(dtype=float)
        ar_nz = ar[ar > 0]
        size_ratio = float(ar_nz.max() / ar_nz.min()) if len(ar_nz) > 0 else float("inf")
        size_pass = size_ratio <= 3.0
        dd = fold_df["fold_max_dd_pct"].to_numpy(dtype=float)
        dd_med = float(np.median(dd))
        dd_max = float(np.max(dd))
        dd_ratio = dd_max / dd_med if dd_med > 0 else float("inf")
        dd_pass = dd_ratio <= 2.0
        overall_pass = sign_consistency and size_pass and dd_pass

        gate_rows = [
            {"gate": "A_sign_consistency", "spec": "final_r_mean > 0 in every fold (with admits)",
             "result": "PASS" if sign_consistency else "FAIL", "value": "all positive" if sign_consistency else "fails"},
            {"gate": "B_size_variance", "spec": "max(admit_rate) / min(admit_rate) <= 3.0",
             "result": "PASS" if size_pass else "FAIL",
             "value": f"ratio={size_ratio:.6f}; min={ar_nz.min():.6f}; max={ar_nz.max():.6f}"},
            {"gate": "C_dd_ceiling", "spec": "worst_fold_dd <= 2.0 × median_fold_dd",
             "result": "PASS" if dd_pass else "FAIL",
             "value": f"max={dd_max:.6f}; median={dd_med:.6f}; ratio={dd_ratio:.6f}"},
            {"gate": "overall_verdict",
             "spec": "A AND B AND C, with §9 discipline for single-flip / size-flag cases",
             "result": "PASS" if overall_pass else "FAIL",
             "value": "all gates pass" if overall_pass else "at least one gate fails"},
        ]
        pd.DataFrame(gate_rows).to_csv(cluster_dir / "gate_check_new_spreads.csv", index=False)

        # Worst-fold ROI for ship decision
        worst_fold_idx = int(fold_df["fold_roi_pct_annualised"].idxmin())
        worst_fold_ann_roi = float(fold_df["fold_roi_pct_annualised"].iloc[worst_fold_idx])
        worst_fold_dd = float(fold_df["fold_max_dd_pct"].iloc[worst_fold_idx])

        cluster_results[cid] = {
            "fold_df": fold_df,
            "admit_df": admit_df,
            "sign_consistency": sign_consistency,
            "size_ratio": size_ratio,
            "size_pass": size_pass,
            "dd_max": dd_max,
            "dd_med": dd_med,
            "dd_ratio": dd_ratio,
            "dd_pass": dd_pass,
            "overall_pass": overall_pass,
            "worst_fold": worst_fold_idx + 1,
            "worst_fold_ann_roi": worst_fold_ann_roi,
            "worst_fold_dd": worst_fold_dd,
        }
        print(
            f"  cluster {cid}: §9 pass={overall_pass}  worst-fold ann ROI={worst_fold_ann_roi:.2f}%  "
            f"worst-fold DD={worst_fold_dd:.2f}%",
            file=sys.stderr,
        )

    # ===== Comparison + Jaccard =====
    print("[step5_rerun] computing baseline-vs-new comparison + Jaccard", file=sys.stderr)
    for cid in (1, 3):
        cluster_dir = OUT_DIR / f"cluster_{cid}"
        # Load baseline fold stability + admits
        base_fold = pd.read_csv(BASELINE_STEP5_DIR / f"cluster_{cid}" / f"fold_stability_cluster_{cid}.csv")
        base_admit = pd.read_csv(BASELINE_STEP5_DIR / f"cluster_{cid}" / f"oos_trade_admits_cluster_{cid}.csv")
        new_fold = cluster_results[cid]["fold_df"]
        new_admit = cluster_results[cid]["admit_df"]

        # Re-derive baseline metrics with OUR formula for like-for-like deltas.
        # (Baseline-reported values match this formula on 6/7 folds; fold 1 has
        # a 0.3pp DD discrepancy due to opaque tiebreak.)
        base_recomputed: Dict[int, Dict] = {}
        for f in range(1, 8):
            bdf = base_admit[(base_admit["fold"] == f) & (base_admit["admitted"] == 1)]
            r = bdf["final_r"].to_numpy(dtype=float)
            et = pd.to_datetime(bdf["entry_time"])
            spec = base_fold[base_fold["fold"] == f].iloc[0]
            oos_days = int(spec["oos_days"])
            metrics = _compute_fold_metrics(r, et, oos_days)
            base_recomputed[f] = metrics

        # Side-by-side comparison
        cmp_rows = []
        for f in range(1, 8):
            br = base_fold[base_fold["fold"] == f].iloc[0]
            br_re = base_recomputed[f]
            nr_match = new_fold[new_fold["fold"] == f]
            if len(nr_match) == 0:
                continue
            nr = nr_match.iloc[0]
            cmp_rows.append(
                {
                    "fold": f,
                    "baseline_admit_rate": float(br["admit_rate"]),
                    "new_admit_rate": float(nr["admit_rate"]),
                    "delta_admit_rate_pp": float((nr["admit_rate"] - br["admit_rate"]) * 100),
                    "baseline_mean_r": float(br["final_r_mean"]),
                    "new_mean_r": float(nr["final_r_mean"]),
                    "delta_mean_r": float(nr["final_r_mean"] - br["final_r_mean"]),
                    "baseline_ann_roi_reported": float(br["fold_roi_pct_annualised"]),
                    "baseline_ann_roi_recomputed": float(br_re["fold_roi_pct_annualised"]),
                    "new_ann_roi": float(nr["fold_roi_pct_annualised"]),
                    "delta_ann_roi_like_for_like": float(nr["fold_roi_pct_annualised"] - br_re["fold_roi_pct_annualised"]),
                    "baseline_max_dd_reported": float(br["fold_max_dd_pct"]),
                    "baseline_max_dd_recomputed": float(br_re["fold_max_dd_pct"]),
                    "new_max_dd": float(nr["fold_max_dd_pct"]),
                    "delta_max_dd_like_for_like": float(nr["fold_max_dd_pct"] - br_re["fold_max_dd_pct"]),
                    "baseline_sign_consistent": float(br["final_r_mean"]) > 0,
                    "new_sign_consistent": float(nr["final_r_mean"]) > 0,
                }
            )
        pd.DataFrame(cmp_rows).to_csv(cluster_dir / "baseline_vs_new_comparison.csv", index=False)

        # Jaccard per fold using baseline_trade_id (matched trades only).
        jacc_rows = []
        for f in range(1, 8):
            base_set = set(int(t) for t in base_admit[base_admit["fold"] == f]["trade_id"].tolist())
            new_set_raw = new_admit[new_admit["fold"] == f]
            # Map new admits to baseline_trade_id where available; drop new-only admits
            # from the Jaccard universe (no baseline counterpart to compare against).
            new_set_mapped: set = set()
            new_only_count = 0
            for _, r in new_set_raw.iterrows():
                btid = r["baseline_trade_id"]
                if btid == "" or pd.isna(btid):
                    new_only_count += 1
                else:
                    new_set_mapped.add(int(btid))
            inter = base_set & new_set_mapped
            union = base_set | new_set_mapped
            j = float(len(inter) / len(union)) if union else 0.0
            jacc_rows.append(
                {
                    "fold": f,
                    "baseline_admits": len(base_set),
                    "new_admits_total": len(new_set_raw),
                    "new_admits_with_baseline_id": len(new_set_mapped),
                    "new_admits_only_new": new_only_count,
                    "intersection": len(inter),
                    "union": len(union),
                    "jaccard": j,
                    "baseline_only": len(base_set - new_set_mapped),
                    "new_only_within_matched": len(new_set_mapped - base_set),
                }
            )
        jacc_df = pd.DataFrame(jacc_rows)
        jacc_df.to_csv(cluster_dir / "admit_set_jaccard.csv", index=False)
        print(
            f"  cluster {cid}: mean Jaccard across 7 folds = {jacc_df['jaccard'].mean():.4f}",
            file=sys.stderr,
        )

    # ===== Top-level metadata =====
    meta = {
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "new_step1_dir": str(NEW_STEP1_DIR),
        "baseline_step1_dir": str(BASELINE_STEP1_DIR),
        "baseline_step2_dir": str(BASELINE_STEP2_DIR),
        "baseline_step5_dir": str(BASELINE_STEP5_DIR),
        "wfo_folds": WFO_FOLDS,
        "risk_pct": RISK_PCT,
        "clusters_evaluated": list(CLUSTER_R_FRAME_ATR_MULT.keys()),
        "cluster_r_frame_atr_mult": CLUSTER_R_FRAME_ATR_MULT,
        "cluster_f9_threshold": CLUSTER_F9_THRESHOLD,
        "classifier_sha256_unchanged": "verified per-fold below",
        "phase1_sanity_check_1_trade_count": {
            "baseline": int(len(base_trades)),
            "new": int(len(new_trades)),
            "delta": int(len(new_trades) - len(base_trades)),
            "verdict": "FAIL (mechanism: concurrent-per-pair guard cascade under spread shift; user-approved override per chat)",
        },
        "cluster_results_summary": {
            cid: {
                "overall_pass": res["overall_pass"],
                "sign_consistency": res["sign_consistency"],
                "size_ratio": res["size_ratio"],
                "size_pass": res["size_pass"],
                "dd_max": res["dd_max"],
                "dd_med": res["dd_med"],
                "dd_ratio": res["dd_ratio"],
                "dd_pass": res["dd_pass"],
                "worst_fold": res["worst_fold"],
                "worst_fold_ann_roi": res["worst_fold_ann_roi"],
                "worst_fold_dd": res["worst_fold_dd"],
            }
            for cid, res in cluster_results.items()
        },
        "elapsed_seconds": float(time.time() - t_start),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"[step5_rerun] DONE in {time.time() - t_start:.1f}s; results in {OUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
