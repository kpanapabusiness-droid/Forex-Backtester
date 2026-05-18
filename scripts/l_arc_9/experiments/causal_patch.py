"""Arc 9 causal patch + re-run — corrective dispatch.

External review of Arc 9 detected producer-level lookahead in two D1 swing
features used by the Candidate A / B LightGBM classifier (AUC 0.7508):

  - d1_bars_since_swing_low  (20% of classifier gain — top feature)
  - d1_bars_since_swing_high (7% of classifier gain — third)

Combined 27% of classifier gain came from leaked features. Cohort
definition (Step 3) is unaffected; classifier and all downstream economics
are invalidated pending this dispatch.

Root cause: `_d1_swing_high_low(d1, half=10)` in pipeline_e_retry.py uses
a two-sided ±10-bar window — swing at bar k uses bars [k-10, k+10]. The
producer suppressed the LAST 10 bars of the D1 frame (to handle the
unconfirmed-swing-at-end-of-history edge case) but did not suppress
per-signal: the swing flag at D1 row d depends on D1 bars [d-10, d+10],
which means at trade-signal time t with lookup_date d=normalize(t)-1day,
the flag at d uses bars from [d+1, d+10] — strictly FUTURE relative to
trade-time t.

Empirical: random sample of 50 EUR_USD signals in Arc 9 window showed
24% of (sample, feature) pairs differ between full-pipeline and
truncated-to-causal-data computations; median shift +25 bars.

Patch — Option A confirmed-swing with 10-day lag (dispatch-preferred):
  At date d, a swing at bar k is CONFIRMED iff:
    (a) k is a two-sided ±10 swing extremum (same definition as before)
    (b) k ≤ d - 10  (so 10 forward bars exist to verify)
  Feature `d1_bars_since_swing_low[d]` = d - max(k : confirmed_at_d(k)).
  Causally clean by construction.

Phases:
  1. Patched swing producer + verify on small sample
  2. Producer-level causal audit on ALL 8 D1 features (100 trades × 8 features)
  3. Re-run Pipeline E retry (4-cell 2×2: RF/LGBM × baseline/expanded), TSS-CV(5)
  4. Re-run Step 5 LGBM E WFO (conditional on patched AUC ≥ 0.65)
  5. Re-run scaled-risk (conditional on patched Candidate A clearing §10)
  6. Master report

All scripts read-only against the existing pipeline_e_retry / step5_lgbm_pipeline_e /
scaled_risk modules, with new patched feature engineering substituted.

Outputs in results/l_arc_9/experiments/causal_patch/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse exact hyperparams + feature catalogue from the original pipeline_e_retry.
from scripts.l_arc_9.experiments.pipeline_e_retry import (  # noqa: E402
    BASE_8,
    BASELINE_16,
    D1_8,
    EXPANDED_28,
    FORBIDDEN_LEAK_FEATURES,
    LGBM_KW,
    N_SPLITS,
    PIPELINE_E_AUC_FLOOR,
    RECALL_FLOOR,
    RF_KW,
    SESSION_4,
    _attach_session_features,
    _d1_swing_high_low,
    _kijun,
    _load_d1,
    _rsi,
    _wilder_atr_d1,
)

SWING_HALF = 10
SWING_CONFIRM_LAG = 10
AUDIT_SEED = 4242


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =====================================================================
# Phase 1 — Patched D1 feature frame
# =====================================================================


def _bars_since_confirmed_swing(is_swing: np.ndarray, lag: int = SWING_CONFIRM_LAG) -> np.ndarray:
    """Producer-causal version of _bars_since.

    `is_swing[k]` is a two-sided swing flag (computed with ±lag window). At each
    index d, a swing at index k is CONFIRMED iff k <= d - lag — only then have
    the lag forward bars existed to verify k as a swing extremum.

    Returns array out where out[d] = (d - k) for the most recent confirmed
    swing index k ≤ d - lag, or -1 if none.

    Causally clean by construction: out[d] depends only on is_swing[:d-lag+1],
    each of whose entries depends only on bars[:d].
    """
    n = is_swing.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    last_confirmed_idx = -1
    for d in range(n):
        cand = d - lag
        if cand >= 0 and is_swing[cand]:
            last_confirmed_idx = cand
        if last_confirmed_idx >= 0:
            out[d] = d - last_confirmed_idx
    return out


def _build_d1_feature_frame_patched(df_d1: pd.DataFrame) -> pd.DataFrame:
    """Patched D1 feature frame builder.

    Identical to pipeline_e_retry._build_d1_feature_frame for ALL D1 features
    EXCEPT d1_bars_since_swing_low / d1_bars_since_swing_high which use the
    Option A confirmed-swing-with-10-day-lag pattern.

    The other 6 D1 features (d1_atr14, d1_rsi_14, d1_kijun, d1_trend_state,
    d1_20d_high/low → d1_pos_in_20d_range, d1_close_above_kijun, d1_ret_5d_atr)
    are causally clean by construction (one-sided recursive smoothers / rolling
    window maxes / scalar comparisons over [k-N+1, k]) — verified at Phase 2.
    """
    d1 = df_d1[["date", "open", "high", "low", "close"]].copy()
    d1["d1_atr14"] = _wilder_atr_d1(d1, 14)
    d1["d1_rsi_14"] = _rsi(d1["close"].astype(float).to_numpy(), 14)
    d1["d1_kijun"] = _kijun(d1, 26)
    d1["d1_close_prev"] = d1["close"].shift(1)
    d1["d1_trend_state"] = ((d1["close"] > d1["open"]) & (d1["close"] > d1["d1_close_prev"])).astype(int)
    d1["d1_20d_high"] = d1["high"].rolling(20).max()
    d1["d1_20d_low"] = d1["low"].rolling(20).min()
    _rng = d1["d1_20d_high"] - d1["d1_20d_low"]
    d1["d1_pos_in_20d_range"] = np.where(_rng > 0, (d1["close"] - d1["d1_20d_low"]) / _rng, 0.5)
    d1["d1_close_5d_ago"] = d1["close"].shift(5)
    d1["d1_ret_5d_atr"] = np.where(
        (d1["d1_atr14"] > 0) & d1["d1_atr14"].notna(),
        (d1["close"] - d1["d1_close_5d_ago"]) / d1["d1_atr14"],
        np.nan,
    )
    d1["d1_close_above_kijun"] = (d1["close"] > d1["d1_kijun"]).astype(int)
    # --- PATCHED swing features: confirmed-swing with 10-day lag ---
    is_sw_high, is_sw_low = _d1_swing_high_low(d1, half=SWING_HALF)
    d1["d1_bars_since_swing_high"] = _bars_since_confirmed_swing(is_sw_high, lag=SWING_CONFIRM_LAG)
    d1["d1_bars_since_swing_low"] = _bars_since_confirmed_swing(is_sw_low, lag=SWING_CONFIRM_LAG)
    return d1


def _attach_d1_features_patched(
    trades: pd.DataFrame, data_d1_dir: Path, atr_4h_by_tid: Dict[int, float],
) -> pd.DataFrame:
    """Same merge_asof pattern as pipeline_e_retry._attach_d1_features, with
    _build_d1_feature_frame_patched substituted for the D1 frame builder.
    """
    trades = trades.copy()
    trades["signal_bar_time"] = pd.to_datetime(trades["signal_bar_time"])
    trades["lookup_date"] = trades["signal_bar_time"].dt.normalize() - pd.Timedelta(days=1)
    d1_cols_pre_ratio = [c for c in D1_8 if c != "d1_atr_ratio_to_4h"]
    out_rows: List[pd.DataFrame] = []
    pairs = sorted(trades["pair"].unique().tolist())
    for pair in pairs:
        sub = trades[trades["pair"] == pair].sort_values("lookup_date", kind="mergesort").reset_index(drop=True)
        df_d1 = _load_d1(pair, data_d1_dir)
        d1_feats = _build_d1_feature_frame_patched(df_d1)
        merge_cols = ["date", "d1_atr14"] + d1_cols_pre_ratio
        d1_lite = d1_feats[merge_cols].rename(columns={"date": "d1_date", "d1_atr14": "d1_atr14_lag1"})
        merged = pd.merge_asof(
            sub[["trade_id", "lookup_date"]], d1_lite,
            left_on="lookup_date", right_on="d1_date", direction="backward",
        )
        out_rows.append(merged)
    merged_all = pd.concat(out_rows, ignore_index=True)
    result = trades.merge(merged_all, on="trade_id", how="left")
    atr4h = result["trade_id"].astype(int).map(atr_4h_by_tid)
    result["d1_atr_ratio_to_4h"] = np.where(
        (atr4h > 0) & atr4h.notna() & result["d1_atr14_lag1"].notna(),
        result["d1_atr14_lag1"] / atr4h, np.nan,
    )
    return result


def _build_feature_matrix_patched(out_dir: Path) -> pd.DataFrame:
    """Patched mirror of step5_lgbm_pipeline_e._build_feature_matrix.

    Loads the same baseline 16-feature entry matrix, attaches PATCHED D1
    features (confirmed-swing-with-lag) and session features, attaches y label.
    Sorted by entry_time for TimeSeriesSplit chronological correctness.
    """
    entry = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step4_extractability" / "entry_features.csv"
    )
    forbidden = set(entry.columns) & FORBIDDEN_LEAK_FEATURES
    if forbidden:
        raise RuntimeError(f"path-shape features leaked into entry features: {forbidden}")
    for c in BASELINE_16:
        if c not in entry.columns:
            raise RuntimeError(f"missing baseline feature: {c}")

    clusters = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step2_clustering" / "clusters_K3.csv"
    )
    cid0 = set(clusters[clusters["cluster_id"] == 0]["trade_id"].astype(int))

    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    atr_4h_by_tid: Dict[int, float] = dict(zip(
        trades_all["trade_id"].astype(int),
        trades_all["atr14_at_signal"].astype(float),
    ))
    entry_time_by_tid: Dict[int, str] = dict(zip(
        trades_all["trade_id"].astype(int),
        trades_all["entry_time"].astype(str),
    ))

    data_d1_dir = Path("C:/Users/panap/Documents/Forex-Backtester/data/daily")
    df = _attach_d1_features_patched(entry, data_d1_dir, atr_4h_by_tid)
    df = _attach_session_features(df)
    df["entry_time"] = df["trade_id"].astype(int).map(entry_time_by_tid)
    df["y"] = df["trade_id"].astype(int).apply(lambda x: 1 if int(x) in cid0 else 0)

    df_clean = df.dropna(subset=EXPANDED_28).reset_index(drop=True)
    df_clean["entry_time"] = pd.to_datetime(df_clean["entry_time"])
    df_clean = df_clean.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
    return df_clean


# =====================================================================
# Phase 2 — Producer-level causal audit (NEW audit dimension)
# =====================================================================


def phase_2_producer_audit(out_dir: Path) -> Dict[str, Any]:
    """For each of 8 D1 features, sample 100 trades and verify that the
    feature value at the join row dated `d` equals the value computed from
    a truncated D1 history (bars ≤ d).

    Critical: stratify samples across head (first 30 D1 days in Arc 9 window),
    midpoint (most trades), and tail (last 30 D1 days).
    """
    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    trades_all["signal_bar_time"] = pd.to_datetime(trades_all["signal_bar_time"])
    trades_all["lookup_date"] = trades_all["signal_bar_time"].dt.normalize() - pd.Timedelta(days=1)

    rng = random.Random(AUDIT_SEED)
    # Stratify sampling: 70 midpoint, 15 tail, 15 head.
    arc_start = pd.Timestamp("2020-10-01")
    arc_end = pd.Timestamp("2026-01-01")
    head_window = trades_all[(trades_all["signal_bar_time"] >= arc_start) &
                              (trades_all["signal_bar_time"] < arc_start + pd.Timedelta(days=60))]
    tail_window = trades_all[(trades_all["signal_bar_time"] >= arc_end - pd.Timedelta(days=60)) &
                              (trades_all["signal_bar_time"] < arc_end)]
    mid_window = trades_all[(trades_all["signal_bar_time"] >= arc_start + pd.Timedelta(days=60)) &
                             (trades_all["signal_bar_time"] < arc_end - pd.Timedelta(days=60))]
    n_mid = 70
    n_head = 15
    n_tail = 15
    samples = []
    if len(mid_window) >= n_mid:
        samples += [mid_window.iloc[i] for i in rng.sample(range(len(mid_window)), n_mid)]
    if len(head_window) >= n_head:
        samples += [head_window.iloc[i] for i in rng.sample(range(len(head_window)), n_head)]
    if len(tail_window) >= n_tail:
        samples += [tail_window.iloc[i] for i in rng.sample(range(len(tail_window)), n_tail)]
    print(f"[phase 2] {len(samples)} stratified samples (mid={n_mid}, head={n_head}, tail={n_tail})")

    data_d1_dir = Path("C:/Users/panap/Documents/Forex-Backtester/data/daily")
    # Cache: per pair, build the full PATCHED D1 frame ONCE for lookup.
    pair_full_feats: Dict[str, pd.DataFrame] = {}
    pair_raw_d1: Dict[str, pd.DataFrame] = {}

    audit_rows: List[Dict[str, Any]] = []
    for s in samples:
        pair = str(s["pair"])
        lookup_date = s["lookup_date"]
        if pair not in pair_raw_d1:
            pair_raw_d1[pair] = _load_d1(pair, data_d1_dir)
            pair_full_feats[pair] = _build_d1_feature_frame_patched(pair_raw_d1[pair])
        d1_raw = pair_raw_d1[pair]
        d1_full_feats = pair_full_feats[pair]
        # Truncate D1 history to bars ≤ lookup_date (the actual data available at signal time).
        d1_trunc = d1_raw[d1_raw["date"] <= lookup_date].reset_index(drop=True)
        if len(d1_trunc) < 30:
            continue
        d1_trunc_feats = _build_d1_feature_frame_patched(d1_trunc)
        # Lookup row at lookup_date from FULL (merge_asof would resolve to ≤ lookup_date).
        full_row_q = d1_full_feats[d1_full_feats["date"] <= lookup_date]
        if full_row_q.empty:
            continue
        full_row = full_row_q.iloc[-1]
        # Lookup row at lookup_date from TRUNCATED (last row of truncated frame).
        trunc_row = d1_trunc_feats.iloc[-1]
        # Both rows should be the same calendar date.
        for fname in D1_8:
            if fname == "d1_atr_ratio_to_4h":
                continue  # computed post-merge from atr_4h, not in D1 frame
            full_val = full_row[fname] if fname in full_row.index else float("nan")
            trunc_val = trunc_row[fname] if fname in trunc_row.index else float("nan")
            try:
                fv = float(full_val)
                tv = float(trunc_val)
                if np.isnan(fv) and np.isnan(tv):
                    match = True
                elif np.isnan(fv) or np.isnan(tv):
                    match = False
                else:
                    match = abs(fv - tv) < 1e-6
            except Exception:
                match = (full_val == trunc_val)
            audit_rows.append({
                "pair": pair,
                "signal_bar_time": s["signal_bar_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "lookup_date": pd.Timestamp(lookup_date).strftime("%Y-%m-%d"),
                "d1_row_date_full": pd.Timestamp(full_row["date"]).strftime("%Y-%m-%d"),
                "d1_row_date_trunc": pd.Timestamp(trunc_row["date"]).strftime("%Y-%m-%d"),
                "feature": fname,
                "full_pipeline_value": full_val,
                "truncated_pipeline_value": trunc_val,
                "match": int(match),
            })
    df = pd.DataFrame(audit_rows)
    df.to_csv(out_dir / "phase_2_producer_audit.csv", index=False,
              float_format="%.10g", lineterminator="\n")
    # Per-feature verdict.
    feature_summary = (
        df.groupby("feature").agg(n=("match", "size"), n_match=("match", "sum")).reset_index()
    )
    feature_summary["match_pct"] = feature_summary["n_match"] / feature_summary["n"] * 100
    feature_summary["verdict"] = feature_summary.apply(
        lambda r: "PASS" if r["n_match"] == r["n"] else "FAIL", axis=1
    )
    feature_summary.to_csv(out_dir / "phase_2_per_feature_verdict.csv", index=False,
                            float_format="%.10g", lineterminator="\n")
    overall_pass = bool((feature_summary["verdict"] == "PASS").all())
    return {
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "n_samples_evaluated": int(len(samples)),
        "per_feature": feature_summary.to_dict("records"),
    }


# =====================================================================
# Phase 3 — Re-run Pipeline E retry with patched features
# =====================================================================


def _fit_and_oof_tss(X: np.ndarray, y: np.ndarray, model_kind: str) -> Tuple[float, np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_aucs: List[float] = []
    for tr_idx, te_idx in tscv.split(X):
        if model_kind == "rf":
            mdl = RandomForestClassifier(**RF_KW)
        elif model_kind == "lgbm":
            mdl = lgb.LGBMClassifier(**LGBM_KW)
        else:
            raise ValueError(model_kind)
        mdl.fit(X[tr_idx], y[tr_idx])
        p = mdl.predict_proba(X[te_idx])[:, 1]
        oof[te_idx] = p
        try:
            fold_aucs.append(float(roc_auc_score(y[te_idx], p)))
        except Exception:
            fold_aucs.append(float("nan"))
    return float(np.nanmean(fold_aucs)), np.array(fold_aucs), oof


def phase_3_pipeline_e_retry_patched(
    feat_matrix: pd.DataFrame, out_dir: Path,
) -> Dict[str, Any]:
    X_baseline = feat_matrix[BASELINE_16].to_numpy(dtype=float)
    X_expanded = feat_matrix[EXPANDED_28].to_numpy(dtype=float)
    y = feat_matrix["y"].to_numpy(dtype=int)
    print(f"[phase 3] feature matrix: n={len(feat_matrix)}, n_pos={int(y.sum())}, n_features expanded={len(EXPANDED_28)}")

    cells: Dict[str, Dict[str, Any]] = {}
    for cell_name, X, feature_set, kind in [
        ("rf_baseline_16",   X_baseline, BASELINE_16, "rf"),
        ("rf_expanded_28",   X_expanded, EXPANDED_28, "rf"),
        ("lgbm_baseline_16", X_baseline, BASELINE_16, "lgbm"),
        ("lgbm_expanded_28", X_expanded, EXPANDED_28, "lgbm"),
    ]:
        print(f"[phase 3] training {cell_name}...")
        mean_auc, fold_aucs, oof = _fit_and_oof_tss(X, y, kind)
        cells[cell_name] = {
            "kind": kind, "n_features": len(feature_set),
            "mean_auc": mean_auc, "fold_aucs": fold_aucs.tolist(),
        }
        print(f"  mean AUC {mean_auc:.4f}, folds {[round(a, 4) for a in fold_aucs.tolist()]}")

    # Per-fold AUCs table (patched).
    rows: List[Dict[str, Any]] = []
    for cn, info in cells.items():
        for i, auc in enumerate(info["fold_aucs"], start=1):
            rows.append({"cell": cn, "fold": i, "auc": auc})
        rows.append({"cell": cn, "fold": "mean", "auc": info["mean_auc"]})
        rows.append({"cell": cn, "fold": "std",
                     "auc": float(np.nanstd(info["fold_aucs"], ddof=1))})
    pd.DataFrame(rows).to_csv(out_dir / "phase_3_per_fold_aucs_patched.csv", index=False,
                               float_format="%.10g", lineterminator="\n")

    # Feature importances on LGBM expanded (patched).
    mdl_full = lgb.LGBMClassifier(**LGBM_KW)
    mdl_full.fit(X_expanded, y)
    imp = pd.DataFrame({
        "feature": EXPANDED_28,
        "importance_gain": mdl_full.booster_.feature_importance(importance_type="gain"),
        "importance_split": mdl_full.booster_.feature_importance(importance_type="split"),
    })
    imp["origin"] = imp["feature"].apply(
        lambda c: "D1" if c in D1_8 else ("session" if c in SESSION_4
                 else ("base" if c in BASE_8 else "arc_specific"))
    )
    imp = imp.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    imp.to_csv(out_dir / "phase_3_feature_importances_patched.csv", index=False,
               float_format="%.10g", lineterminator="\n")
    return cells


# =====================================================================
# Phase 4 — Re-run Step 5 LGBM E WFO with patched classifier
# =====================================================================


def _select_threshold(thr_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    cands = [r for r in thr_rows if r["recall"] >= RECALL_FLOOR]
    return max(cands, key=lambda r: r["precision"]) if cands else None


def phase_4_step5_wfo_patched(
    feat_matrix: pd.DataFrame, out_dir: Path,
) -> Dict[str, Any]:
    """Mirror of step5_lgbm_pipeline_e._wfo_for_threshold but with patched
    feature matrix. Same KH-24 anchored expanding training, same threshold
    candidates A (0.40) and B (0.05), same §11 Stepwise exit re-sim.
    """
    from core.spread_floor import STATE_CFG_KEY, load_spread_floor
    from scripts.l_arc_9.experiments.step5_validation import (
        STARTING_BALANCE as SV_STARTING_BALANCE,
    )
    from scripts.l_arc_9.experiments.step5_validation import (
        _compute_fold_metrics,
        _full_data_equity,
        _resimulate_trade,
        evaluate_gates,
    )

    cfg_kh24 = yaml.safe_load((_REPO_ROOT / "configs" / "wfo_kh24.yaml").read_text(encoding="utf-8"))
    kh24_folds: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = [
        (int(f["fold"]), pd.Timestamp(f["oos_start"]), pd.Timestamp(f["oos_end"]))
        for f in cfg_kh24["wfo"]["folds"]
    ]
    cfg_arc = yaml.safe_load((_REPO_ROOT / "configs" / "wfo_l_arc_9.yaml").read_text(encoding="utf-8"))
    spread_state = load_spread_floor(cfg_arc)
    cfg_arc[STATE_CFG_KEY] = spread_state
    cfg_arc.setdefault("spreads", {})
    cfg_arc["spreads"].setdefault("points_per_pip", float(spread_state.points_per_pip))

    # Per-pair raw 4H data cache (for trade re-sim).
    pairs = sorted(feat_matrix["pair"].unique().tolist())
    data_4h_path = cfg_arc["data"]["data_dirs"]["4H"]
    data_dir_4h = Path(data_4h_path) if Path(data_4h_path).is_absolute() else _REPO_ROOT / data_4h_path
    pair_cache: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df_raw = pd.read_csv(data_dir_4h / f"{pair}.csv")
        if "time" in df_raw.columns and "date" not in df_raw.columns:
            df_raw = df_raw.rename(columns={"time": "date"})
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        s_ts = pd.Timestamp(str(cfg_arc["data"]["date_start"]))
        e_ts = pd.Timestamp(str(cfg_arc["data"]["date_end"])) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        pair_cache[pair] = df_raw[(df_raw["date"] >= s_ts) & (df_raw["date"] <= e_ts)].reset_index(drop=True)

    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    trades_all["signal_bar_time"] = pd.to_datetime(trades_all["signal_bar_time"])
    trades_all["entry_time"] = pd.to_datetime(trades_all["entry_time"])
    tidx_to_resim_row = trades_all.set_index("trade_id")

    X_all = feat_matrix[EXPANDED_28].to_numpy(dtype=float)
    y_all = feat_matrix["y"].to_numpy(dtype=int)
    entry_time_arr = feat_matrix["entry_time"].to_numpy()

    def _run_wfo(threshold: float, cand_dir: Path):
        cand_dir.mkdir(parents=True, exist_ok=True)
        fold_rows: List[Dict[str, Any]] = []
        all_admitted_rows: List[Dict[str, Any]] = []
        all_resim_rows: List[Dict[str, Any]] = []
        for (fold_id, s, e) in kh24_folds:
            train_mask = entry_time_arr < np.datetime64(s)
            oos_mask = (entry_time_arr >= np.datetime64(s)) & (entry_time_arr < np.datetime64(e))
            n_train = int(train_mask.sum())
            n_oos = int(oos_mask.sum())
            n_train_pos = int(y_all[train_mask].sum())
            if n_train == 0 or n_train_pos < 10:
                fold_rows.append({
                    "fold": fold_id, "oos_start": s.strftime("%Y-%m-%d"),
                    "oos_end": e.strftime("%Y-%m-%d"),
                    "n_oos_signals": n_oos, "n_train": n_train, "n_train_pos": n_train_pos,
                    "n_admitted": 0, "n_admitted_cluster0_true_pos": 0, "admit_rate": 0.0,
                    "n_trades": 0, "final_r_mean": float("nan"), "final_r_sign_positive": 0,
                    "fold_roi_pct": 0.0, "annualised_roi_pct": 0.0, "max_dd_pct": 0.0,
                    "ending_equity": SV_STARTING_BALANCE,
                    "note": "no training data (Arc 9 data start coincides with F1 OOS start)",
                })
                continue
            mdl = lgb.LGBMClassifier(**LGBM_KW)
            mdl.fit(X_all[train_mask], y_all[train_mask])
            prob_oos = mdl.predict_proba(X_all[oos_mask])[:, 1]
            admit = prob_oos >= threshold
            admit_idx = np.where(oos_mask)[0][admit]
            admit_tids = feat_matrix["trade_id"].iloc[admit_idx].astype(int).tolist()
            admit_true_pos = int(y_all[admit_idx].sum())
            n_admitted = int(admit.sum())

            resim_list = []
            for tid in admit_tids:
                row = tidx_to_resim_row.loc[tid].copy()
                row["trade_id"] = tid
                row["pair"] = str(row["pair"])
                df_pair = pair_cache[row["pair"]]
                r = _resimulate_trade(row, df_pair, cfg_arc, spread_state)
                if r is None:
                    continue
                resim_list.append({
                    "trade_id": r.trade_id, "pair": r.pair,
                    "signal_bar_time": r.signal_bar_time, "entry_time": r.entry_time,
                    "exit_time": r.exit_time, "exit_reason": r.exit_reason,
                    "bars_held": r.bars_held,
                    "mfe_lock_active_at_exit": int(r.mfe_lock_active_at_exit),
                    "final_r": r.final_r, "mfe_r": r.mfe_r, "mae_r": r.mae_r,
                    "spread_pips_used": r.spread_pips_used,
                    "spread_pips_exit": r.spread_pips_exit, "fold": fold_id,
                })
                all_admitted_rows.append({"trade_id": tid, "fold": fold_id, "prob": float(prob_oos[admit][len(resim_list) - 1])})
            all_resim_rows.extend(resim_list)
            resim_df = pd.DataFrame(resim_list)
            if len(resim_df) == 0:
                fmetrics_sub = {
                    "n_trades": 0, "final_r_mean": float("nan"), "final_r_sign_positive": 0,
                    "fold_roi_pct": 0.0, "annualised_roi_pct": 0.0, "max_dd_pct": 0.0,
                    "ending_equity": SV_STARTING_BALANCE,
                }
            else:
                resim_df["entry_time"] = pd.to_datetime(resim_df["entry_time"])
                fmetrics_sub = _compute_fold_metrics(resim_df, s, e)
            fold_rows.append({
                "fold": fold_id, "oos_start": s.strftime("%Y-%m-%d"),
                "oos_end": e.strftime("%Y-%m-%d"), "n_oos_signals": n_oos,
                "n_train": n_train, "n_train_pos": n_train_pos,
                "n_admitted": n_admitted, "n_admitted_cluster0_true_pos": admit_true_pos,
                "admit_rate": float(n_admitted / max(n_oos, 1)), **fmetrics_sub, "note": "",
            })
        fold_df = pd.DataFrame(fold_rows)
        admitted_df = pd.DataFrame(all_admitted_rows)
        resim_df_all = pd.DataFrame(all_resim_rows)
        if len(resim_df_all) > 0:
            resim_df_all["entry_time"] = pd.to_datetime(resim_df_all["entry_time"])
            resim_df_all = resim_df_all.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
            in_window = resim_df_all[
                (resim_df_all["entry_time"] >= kh24_folds[0][1]) &
                (resim_df_all["entry_time"] < kh24_folds[-1][2])
            ]
            full_m = _full_data_equity(in_window, [(s_, e_) for _, s_, e_ in kh24_folds])
        else:
            full_m = {"n_trades": 0, "full_data_roi_pct": 0.0,
                      "full_data_annualised_roi_pct": 0.0,
                      "full_data_max_dd_pct": 0.0, "ending_equity": SV_STARTING_BALANCE}
        fold_df.to_csv(cand_dir / "per_fold_metrics.csv", index=False,
                       float_format="%.10g", lineterminator="\n")
        admitted_df.to_csv(cand_dir / "admitted_trades.csv", index=False,
                            float_format="%.10g", lineterminator="\n")
        resim_df_all.to_csv(cand_dir / "resim_trades.csv", index=False,
                             float_format="%.10g", lineterminator="\n")
        (cand_dir / "full_data_metrics.json").write_text(
            json.dumps(full_m, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        gates = evaluate_gates(fold_df, full_m)
        return {"fold_df": fold_df, "full_m": full_m, "gates": gates,
                "n_admitted": int(fold_df["n_admitted"].sum()),
                "resim_df": resim_df_all, "admitted_df": admitted_df}

    results: Dict[str, Any] = {}
    for cn, thr in [("A_thr0.40", 0.40), ("B_thr0.05", 0.05)]:
        print(f"[phase 4] running candidate {cn} (thr={thr})...")
        out_sub = out_dir / f"candidate_{cn}"
        results[cn] = _run_wfo(thr, out_sub)
        g = results[cn]["gates"]["summary"]
        print(f"  n_admitted: {results[cn]['n_admitted']}; "
              f"worst-fold ann ROI {g['worst_fold_ann_roi_pct']:+.2f}%; "
              f"worst-fold DD {g['worst_fold_max_dd_pct']:.2f}%; "
              f"pass-deployable {'YES' if results[cn]['gates']['pass_deployable'] else 'NO'}")
    return results


# =====================================================================
# Phase 5 — Re-run scaled-risk (conditional)
# =====================================================================


def phase_5_scaled_risk(candidate_a_resim: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    """Mirror scaled_risk methodology on the patched Candidate A admit set."""
    from scripts.l_arc_9.experiments.scaled_risk import (
        HARD_DAILY_DD_PCT,
        HARD_MAX_DD_PCT,
        IN_SYSTEM_DAILY_DD_PCT,
        IN_SYSTEM_MAX_DD_PCT,
        RECOMMEND_MAX_FOLD_DD_PCT,
        RECOMMEND_WORST_DAY_DD_PCT,
        _account_fold,
        _annualise_roi,
        _full_data_account,
    )
    from scripts.l_arc_9.experiments.scaled_risk import (
        RISK_LEVELS as SR_RISK_LEVELS,
    )
    cfg_kh24 = yaml.safe_load((_REPO_ROOT / "configs" / "wfo_kh24.yaml").read_text(encoding="utf-8"))
    kh24_folds: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = [
        (int(f["fold"]), pd.Timestamp(f["oos_start"]), pd.Timestamp(f["oos_end"]))
        for f in cfg_kh24["wfo"]["folds"]
    ]
    resim = candidate_a_resim.copy()
    resim["entry_time"] = pd.to_datetime(resim["entry_time"])
    resim["exit_time"] = pd.to_datetime(resim["exit_time"])

    summary_rows: List[Dict[str, Any]] = []
    per_risk_summary: Dict[float, Dict[str, Any]] = {}
    worst_day_rows: List[Dict[str, Any]] = []
    for r in SR_RISK_LEVELS:
        out_subdir = out_dir / f"per_risk_{int(r * 10000):04d}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        fold_rows: List[Dict[str, Any]] = []
        all_day_rows: List[pd.DataFrame] = []
        for fold_id, s, e in kh24_folds:
            mask = (resim["entry_time"] >= s) & (resim["entry_time"] < e)
            sub = resim[mask].copy()
            eq_df, fold_roi_pct, max_dd_pct, worst_day_dd_pct, end_eq, per_day_df = _account_fold(sub, r)
            days_in_fold = (e - s).days
            ann_roi = _annualise_roi(fold_roi_pct, days_in_fold)
            fold_rows.append({
                "fold": fold_id, "oos_start": s.strftime("%Y-%m-%d"),
                "oos_end": e.strftime("%Y-%m-%d"), "n_trades": int(len(sub)),
                "fold_roi_pct": fold_roi_pct, "annualised_roi_pct": ann_roi,
                "max_dd_pct": max_dd_pct, "worst_day_dd_pct": worst_day_dd_pct,
                "ending_equity": end_eq, "final_r_sign_positive": int(fold_roi_pct > 0),
            })
            if len(per_day_df) > 0:
                tmp = per_day_df.copy()
                tmp.insert(0, "fold", fold_id)
                all_day_rows.append(tmp)
        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(out_subdir / "per_fold_metrics.csv", index=False,
                       float_format="%.10g", lineterminator="\n")
        per_day_all = pd.concat(all_day_rows, ignore_index=True) if all_day_rows else pd.DataFrame()
        per_day_all.to_csv(out_subdir / "per_day_dd.csv", index=False,
                           float_format="%.10g", lineterminator="\n")
        in_window = resim.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
        full_m = _full_data_account(in_window, r, [(s, e) for _, s, e in kh24_folds])
        (out_subdir / "full_data_metrics.json").write_text(
            json.dumps(full_m, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        real = fold_df[fold_df["n_trades"] > 0]
        if len(real) > 0:
            worst_fold_ann_roi = float(real["annualised_roi_pct"].min())
            mean_fold_ann_roi = float(real["annualised_roi_pct"].mean())
            worst_fold_max_dd = float(real["max_dd_pct"].max())
            worst_fold_day_dd = float(real["worst_day_dd_pct"].max())
            all_pos = bool((real["fold_roi_pct"] > 0).all())
        else:
            worst_fold_ann_roi = float("nan")
            mean_fold_ann_roi = float("nan")
            worst_fold_max_dd = float("nan")
            worst_fold_day_dd = float("nan")
            all_pos = False
        per_risk_summary[r] = {
            "worst_fold_ann_roi_pct": worst_fold_ann_roi,
            "mean_fold_ann_roi_pct": mean_fold_ann_roi,
            "worst_fold_max_dd_pct": worst_fold_max_dd,
            "worst_fold_day_dd_pct": worst_fold_day_dd,
            "all_folds_positive": all_pos,
            "full_m": full_m, "per_day_all": per_day_all,
        }
        if abs(r - 0.005) < 1e-9:
            full_m["full_data_annualised_roi_pct"]
        summary_rows.append({
            "risk_pct": r * 100,
            "worst_fold_ann_roi_pct": worst_fold_ann_roi,
            "mean_fold_ann_roi_pct": mean_fold_ann_roi,
            "worst_fold_max_dd_pct": worst_fold_max_dd,
            "worst_fold_day_dd_pct": worst_fold_day_dd,
            "full_data_ann_roi_pct": full_m["full_data_annualised_roi_pct"],
            "full_data_max_dd_pct": full_m["full_data_max_dd_pct"],
            "full_data_worst_day_dd_pct": full_m["full_data_worst_day_dd_pct"],
            "all_folds_positive_real": int(all_pos),
            "pass_in_system_max_dd_8pct": int(worst_fold_max_dd <= IN_SYSTEM_MAX_DD_PCT),
            "pass_in_system_day_dd_4pct": int(worst_fold_day_dd <= IN_SYSTEM_DAILY_DD_PCT),
            "pass_5ers_max_dd_10pct": int(worst_fold_max_dd <= HARD_MAX_DD_PCT),
            "pass_5ers_day_dd_5pct": int(worst_fold_day_dd <= HARD_DAILY_DD_PCT),
        })
        if len(per_day_all) > 0:
            worst = per_day_all.sort_values("day_dd_pct", ascending=False).iloc[0]
            tids = [int(x) for x in worst["trade_ids_in_day"].split(",") if x]
            ctrades = resim[resim["trade_id"].isin(tids)]
            worst_day_rows.append({
                "risk_pct": r * 100,
                "worst_day_date": worst["date"], "worst_day_dd_pct": worst["day_dd_pct"],
                "fold": int(worst["fold"]), "n_contributing_trades": int(worst["n_trades"]),
                "n_contributing_pairs": int(ctrades["pair"].nunique()),
                "contributing_pairs": ", ".join(sorted(ctrades["pair"].unique().tolist())),
                "trade_ids": worst["trade_ids_in_day"],
                "net_day_pnl_dollars": float(worst["net_day_pnl_dollars"]),
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_table.csv", index=False,
                                       float_format="%.10g", lineterminator="\n")
    pd.DataFrame(worst_day_rows).to_csv(out_dir / "worst_day_analysis.csv", index=False,
                                         float_format="%.10g", lineterminator="\n")

    # Recommendation.
    recommended = None
    for r in sorted(SR_RISK_LEVELS, reverse=True):
        a = per_risk_summary[r]
        if (a["worst_fold_max_dd_pct"] <= RECOMMEND_MAX_FOLD_DD_PCT
                and a["worst_fold_day_dd_pct"] <= RECOMMEND_WORST_DAY_DD_PCT
                and a["all_folds_positive"]):
            recommended = r
            break
    if recommended is None:
        recommended = 0.005

    return {
        "summary_rows": summary_rows, "worst_day_rows": worst_day_rows,
        "per_risk_summary": per_risk_summary, "recommended_risk_pct": recommended * 100,
        "rec_metrics": {
            "worst_fold_max_dd_pct": per_risk_summary[recommended]["worst_fold_max_dd_pct"],
            "worst_fold_day_dd_pct": per_risk_summary[recommended]["worst_fold_day_dd_pct"],
            "worst_fold_ann_roi_pct": per_risk_summary[recommended]["worst_fold_ann_roi_pct"],
            "full_data_ann_roi_pct": per_risk_summary[recommended]["full_m"]["full_data_annualised_roi_pct"],
            "full_data_max_dd_pct": per_risk_summary[recommended]["full_m"]["full_data_max_dd_pct"],
            "full_data_worst_day_dd_pct": per_risk_summary[recommended]["full_m"]["full_data_worst_day_dd_pct"],
        },
    }


# =====================================================================
# Driver
# =====================================================================


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 causal patch + re-run dispatch.")
    parser.add_argument("--out-dir", type=Path,
                        default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "causal_patch")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}

    # Phase 1 — build patched feature matrix.
    print("\n[phase 1] building patched feature matrix...")
    feat_matrix = _build_feature_matrix_patched(args.out_dir)
    feat_matrix.to_csv(args.out_dir / "feature_matrix_patched.csv", index=False,
                       float_format="%.10g", lineterminator="\n")
    summary["phase_1"] = {
        "n_total": int(len(feat_matrix)), "n_pos": int(feat_matrix["y"].sum()),
        "n_features": len(EXPANDED_28),
    }

    # Phase 2 — producer-level causal audit.
    print("\n[phase 2] producer-level causal audit (100 samples × 8 D1 features)...")
    p2 = phase_2_producer_audit(args.out_dir)
    summary["phase_2"] = p2
    print(f"  Verdict: {p2['overall_verdict']}")
    for r in p2["per_feature"]:
        print(f"    {r['feature']}: {int(r['n_match'])}/{int(r['n'])} match ({r['match_pct']:.1f}%) -> {r['verdict']}")
    if p2["overall_verdict"] != "PASS":
        print("[phase 2] HALT — producer-level leak detected. Aborting phases 3-5.")
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return 1

    # Phase 3 — re-run Pipeline E retry (4 cells).
    print("\n[phase 3] re-running Pipeline E retry with patched features...")
    p3 = phase_3_pipeline_e_retry_patched(feat_matrix, args.out_dir)
    summary["phase_3"] = {cn: {"mean_auc": v["mean_auc"], "fold_aucs": v["fold_aucs"]}
                          for cn, v in p3.items()}
    lgbm_e_patched_auc = p3["lgbm_expanded_28"]["mean_auc"]

    # Conditional Phase 4.
    if lgbm_e_patched_auc < PIPELINE_E_AUC_FLOOR:
        print(f"\n[phase 4] SKIPPED — patched LGBM AUC {lgbm_e_patched_auc:.4f} < §8 gate {PIPELINE_E_AUC_FLOOR}")
        print("Arc 9 reverts to STEP_4_KILL with leaked-classifier results invalidated.")
        summary["phase_4"] = {"skipped": True, "reason": f"AUC {lgbm_e_patched_auc:.4f} < gate {PIPELINE_E_AUC_FLOOR}"}
        summary["headline"] = "STEP_4_KILL_AFTER_PATCH"
    else:
        print(f"\n[phase 4] AUC {lgbm_e_patched_auc:.4f} >= gate; running Step 5 LGBM E WFO...")
        p4 = phase_4_step5_wfo_patched(feat_matrix, args.out_dir)
        # Restricted §10 evaluation on F2-F7 (per prior Step 5 LGBM E precedent).
        def restricted_pass(fold_df, full_m):
            real = fold_df[fold_df["n_trades"] > 0]
            if len(real) == 0:
                return False
            return bool(
                real["annualised_roi_pct"].min() >= 5.0
                and real["annualised_roi_pct"].mean() >= 8.0
                and real["max_dd_pct"].max() <= 8.0
                and (real["fold_roi_pct"] > 0).all()
                and real["n_trades"].min() >= 15
                and full_m["full_data_annualised_roi_pct"] >= 5.0
                and full_m["full_data_max_dd_pct"] <= 10.0
            )
        cand_a_pass = restricted_pass(p4["A_thr0.40"]["fold_df"], p4["A_thr0.40"]["full_m"])
        cand_b_pass = restricted_pass(p4["B_thr0.05"]["fold_df"], p4["B_thr0.05"]["full_m"])
        summary["phase_4"] = {
            "candidate_A": {
                "n_admitted": p4["A_thr0.40"]["n_admitted"],
                "gates_summary": p4["A_thr0.40"]["gates"]["summary"],
                "pass_deployable_restricted_F2_F7": cand_a_pass,
            },
            "candidate_B": {
                "n_admitted": p4["B_thr0.05"]["n_admitted"],
                "gates_summary": p4["B_thr0.05"]["gates"]["summary"],
                "pass_deployable_restricted_F2_F7": cand_b_pass,
            },
        }
        # Conditional Phase 5.
        if cand_a_pass:
            print("\n[phase 5] patched Candidate A clears restricted §10; running scaled-risk...")
            scaled_out = args.out_dir / "scaled_risk"
            scaled_out.mkdir(exist_ok=True)
            p5 = phase_5_scaled_risk(p4["A_thr0.40"]["resim_df"], scaled_out)
            summary["phase_5"] = {
                "recommended_risk_pct": p5["recommended_risk_pct"],
                "rec_metrics": p5["rec_metrics"],
                "summary_rows": p5["summary_rows"],
                "worst_day_rows": p5["worst_day_rows"],
            }
            summary["headline"] = (
                f"PASS-DEPLOYABLE @ {p5['recommended_risk_pct']:.1f}% risk (patched)"
                if cand_a_pass else "FAIL_STEP_5_AFTER_PATCH"
            )
        else:
            print("\n[phase 5] SKIPPED — patched Candidate A fails restricted §10 pass-deployable")
            summary["phase_5"] = {"skipped": True, "reason": "Candidate A fails restricted §10"}
            summary["headline"] = "FAIL_STEP_5_AFTER_PATCH"

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print(f"HEADLINE: {summary.get('headline', 'unknown')}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
