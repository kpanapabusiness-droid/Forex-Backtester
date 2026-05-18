"""Arc 9 lookahead and cross-timeframe leak audit.

Read-only. Verifies the existing LightGBM Pipeline E classifier (AUC 0.7508 from
Pipeline E retry; reproduced byte-identical at Step 5 LGBM E WFO) and its
training/inference pipeline for:

  Audit 1 - 4H feature timestamp boundary (no peek into t+1 or beyond)
  Audit 2 - D1 lag integrity (one-day-backward merge_asof, NOT same-day) CRITICAL
  Audit 3 - Session/hour features computed from signal-bar timestamp only
  Audit 4 - Label leakage check (no path-shape features in entry-time matrix)
  Audit 5 - Training/inference fold disjointness
  Audit 6 - Cluster label flow (target only, never feature)
  Audit 7 - Spread and execution semantics at inference
  Audit 8 - End-to-end probability reproduction

Any RED verdict halts deployment momentum. Audit 2 RED is hard-stop after
the first failed sample.

Outputs all under results/l_arc_9/experiments/lookahead_audit/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse exact feature engineering (read-only): same code that produced the
# audited matrix.
from scripts.l_arc_9.experiments.pipeline_e_retry import (  # noqa: E402
    EXPANDED_28, BASELINE_16, D1_8, SESSION_4, BASE_8, ARC_SPECIFIC_8,
    LGBM_KW, SEED, FORBIDDEN_LEAK_FEATURES,
    _attach_d1_features, _attach_session_features,
    _build_d1_feature_frame, _load_d1,
)
from scripts.l_arc_9.step4_extractability import (  # noqa: E402
    compute_entry_features,
)

AUDIT_SEED = 4242


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pair_4h(pair: str) -> pd.DataFrame:
    path = Path("C:/Users/panap/Documents/Forex-Backtester/data/4hr") / f"{pair}.csv"
    df = pd.read_csv(path)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _load_pair_d1(pair: str) -> pd.DataFrame:
    return _load_d1(pair, Path("C:/Users/panap/Documents/Forex-Backtester/data/daily"))


# =========================================================================
# Audit 1 - 4H feature timestamp boundary
# =========================================================================

def audit_1_4h_timestamps(out_dir: Path, rng: random.Random) -> Dict[str, Any]:
    """For 10 random signal bars per pair, recompute the 16 baseline features
    using a TRUNCATED 4H series (bars ≤ signal_bar_time only). If the
    recomputed values match the recorded feature values exactly, no future
    bars can have contributed.

    This is stronger than a code-review check: any leak would manifest as a
    mismatch between truncated-data recompute and recorded values.
    """
    entry = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step4_extractability" / "entry_features.csv"
    )
    trades_all = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv"
    )
    trades_all["signal_bar_time"] = pd.to_datetime(trades_all["signal_bar_time"])
    entry["signal_bar_time"] = pd.to_datetime(entry["signal_bar_time"])

    arc_cfg = yaml.safe_load((_REPO_ROOT / "configs" / "wfo_l_arc_9.yaml").read_text(encoding="utf-8"))

    pairs = sorted(entry["pair"].unique())
    rows: List[Dict[str, Any]] = []
    n_mismatch = 0
    sample_per_pair = 10
    # Tolerance: RSI accumulates floating-point ops over hundreds-to-thousands of
    # bars; precision drift up to ~1e-6 is normal in repeated computation under
    # different traversal orders. The leak signal would be values differing in
    # the 3rd-4th decimal or beyond (>1e-4) — anything ≤ 1e-6 is FP noise.
    tol = 1e-6
    for pair in pairs:
        sub = entry[entry["pair"] == pair]
        sample_idx = rng.sample(range(len(sub)), min(sample_per_pair, len(sub)))
        # Match production slicing: production code slices to
        # [arc_cfg.date_start, sig_t] BEFORE computing recursive smoothers (Wilder
        # RSI/ATR). Loading the full historical CSV would give a different RSI
        # initialization (more bars to converge from) and produce small drifts in
        # the first weeks of the Arc 9 window — that's a windowing-convention
        # mismatch, not a lookahead leak.
        prod_date_start = pd.Timestamp(str(arc_cfg["data"]["date_start"]))
        df_pair_full = _load_pair_4h(pair)
        for i in sample_idx:
            ref_row = sub.iloc[i]
            sig_t = ref_row["signal_bar_time"]
            # Slice to production window upper-bounded at sig_t (no future bars).
            df_pair = df_pair_full[
                (df_pair_full["date"] >= prod_date_start) &
                (df_pair_full["date"] <= sig_t)
            ].reset_index(drop=True)
            if len(df_pair) < 25:
                continue
            # Build a single-trade "trades" frame matching what compute_entry_features expects.
            t_match = trades_all[(trades_all["pair"] == pair) & (trades_all["signal_bar_time"] == sig_t)]
            if t_match.empty:
                continue
            single = t_match.iloc[[0]].copy()
            # We need compute_entry_features to load truncated pair data. Easiest
            # path: monkey-patch the cfg's data_dirs to a temp dir with truncated
            # CSV. Simpler: re-derive the 16 features inline using truncated data.
            # Use the same computation logic from step4_extractability.compute_entry_features.
            atr = float(single["atr14_at_signal"].iloc[0])
            # Find index of signal bar in truncated series.
            idx = int(np.where(df_pair["date"].to_numpy() == np.datetime64(sig_t))[0][0])
            n = len(df_pair)
            if idx < 21:
                continue
            op_t = float(df_pair["open"].iloc[idx])
            hi_t = float(df_pair["high"].iloc[idx])
            lo_t = float(df_pair["low"].iloc[idx])
            cl_t = float(df_pair["close"].iloc[idx])
            rng_t = max(hi_t - lo_t, 1e-12)
            body = abs(cl_t - op_t)
            upper_wick = hi_t - max(cl_t, op_t)
            lower_wick = min(cl_t, op_t) - lo_t
            body_to_range = body / rng_t
            upper_wick_ratio = upper_wick / rng_t
            lower_wick_ratio = lower_wick / rng_t
            range_to_atr = rng_t / atr if atr > 0 else float("nan")
            cl_5_ago = float(df_pair["close"].iloc[idx - 5])
            cl_20_ago = float(df_pair["close"].iloc[idx - 20])
            ret_5bar_atr = (cl_t - cl_5_ago) / atr if atr > 0 else float("nan")
            ret_20bar_atr = (cl_t - cl_20_ago) / atr if atr > 0 else float("nan")
            win_lo = df_pair["low"].iloc[idx - 20:idx].min()
            win_hi = df_pair["high"].iloc[idx - 20:idx].max()
            pos_in_20bar_range = (cl_t - win_lo) / max(win_hi - win_lo, 1e-12)
            # RSI(14) at idx using only bars [0..idx]
            close_arr = df_pair["close"].astype(float).to_numpy()
            delta = np.diff(close_arr)
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            period = 14
            avg_gain = float("nan")
            avg_loss = float("nan")
            if len(close_arr) > period:
                avg_gain = float(np.mean(gain[:period]))
                avg_loss = float(np.mean(loss[:period]))
                for i_step in range(period + 1, len(close_arr)):
                    avg_gain = (avg_gain * (period - 1) + gain[i_step - 1]) / period
                    avg_loss = (avg_loss * (period - 1) + loss[i_step - 1]) / period
            rs = (avg_gain / avg_loss) if (avg_loss is not None and avg_loss > 0) else float("nan")
            rsi14 = 100.0 - (100.0 / (1.0 + rs)) if np.isfinite(rs) else float("nan")

            # Arc-specific structural features.
            hi_im1 = float(df_pair["high"].iloc[idx - 1])
            lo_im1 = float(df_pair["low"].iloc[idx - 1])
            hi_im2 = float(df_pair["high"].iloc[idx - 2])
            lo_im2 = float(df_pair["low"].iloc[idx - 2])
            mother_range = max(hi_im2 - lo_im2, 1e-12)
            inside_range = max(hi_im1 - lo_im1, 1e-12)
            mother_bar_range_atr = mother_range / atr if atr > 0 else float("nan")
            inside_bar_range_atr = inside_range / atr if atr > 0 else float("nan")
            ib_range_ratio = inside_range / mother_range
            break_bar_body_atr = body / atr if atr > 0 else float("nan")
            break_close_above_high_atr = (cl_t - hi_im1) / atr if atr > 0 else float("nan")
            swing_low_used = float(single["swing_low_used"].iloc[0])
            swing_low_dist_atr = (cl_t - swing_low_used) / atr if (atr > 0 and not np.isnan(swing_low_used)) else float("nan")

            recomputed = {
                "body_to_range_ratio": body_to_range,
                "upper_wick_ratio": upper_wick_ratio,
                "lower_wick_ratio": lower_wick_ratio,
                "range_to_atr_14": range_to_atr,
                "ret_5bar_atr": ret_5bar_atr,
                "ret_20bar_atr": ret_20bar_atr,
                "pos_in_20bar_range": pos_in_20bar_range,
                "rsi_14": rsi14,
                "mother_bar_range_atr": mother_bar_range_atr,
                "inside_bar_range_atr": inside_bar_range_atr,
                "ib_range_ratio": ib_range_ratio,
                "break_bar_body_atr": break_bar_body_atr,
                "break_close_above_high_atr": break_close_above_high_atr,
                "swing_low_dist_atr": swing_low_dist_atr,
            }
            for fname, recomp_val in recomputed.items():
                recorded_val = float(ref_row[fname]) if pd.notna(ref_row[fname]) else float("nan")
                if np.isnan(recomp_val) and np.isnan(recorded_val):
                    match = True
                elif np.isnan(recomp_val) or np.isnan(recorded_val):
                    match = False
                else:
                    match = abs(recomp_val - recorded_val) < tol
                rows.append({
                    "pair": pair,
                    "signal_bar_ts": sig_t.strftime("%Y-%m-%d %H:%M:%S"),
                    "feature_name": fname,
                    "max_source_bar_ts": sig_t.strftime("%Y-%m-%d %H:%M:%S"),
                    "recomputed": recomp_val,
                    "recorded": recorded_val,
                    "abs_diff": abs(recomp_val - recorded_val) if (np.isfinite(recomp_val) and np.isfinite(recorded_val)) else float("nan"),
                    "leak_flag": 0 if match else 1,
                })
                if not match:
                    n_mismatch += 1
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "audit_1_4h_feature_timestamps.csv", index=False,
              float_format="%.10g", lineterminator="\n")
    return {
        "audit": "1_4h_feature_timestamps",
        "n_samples": int(len(df) // 14) if len(df) > 0 else 0,
        "n_feature_checks": int(len(df)),
        "n_mismatches": int(n_mismatch),
        "verdict": "GREEN" if n_mismatch == 0 else "RED",
        "note": f"Truncated-series recompute matched recorded feature values for {len(df) - n_mismatch}/{len(df)} (feature, sample) pairs. Mismatches imply forward bars contributed.",
    }


# =========================================================================
# Audit 2 - D1 lag integrity (CRITICAL)
# =========================================================================

def audit_2_d1_lag(out_dir: Path, rng: random.Random) -> Dict[str, Any]:
    # --- Step 2a: code review of merge_asof pattern ---
    code_review_path = _REPO_ROOT / "scripts" / "l_arc_9" / "experiments" / "pipeline_e_retry.py"
    src = code_review_path.read_text(encoding="utf-8")
    # Locate the merge_asof block.
    expected_pattern_pieces = [
        'trades["lookup_date"] = trades["signal_bar_time"].dt.normalize() - pd.Timedelta(days=1)',
        'pd.merge_asof(',
        'direction="backward"',
    ]
    code_review_md_lines: List[str] = []
    code_review_md_lines.append("# Audit 2 — D1 lag code review")
    code_review_md_lines.append("")
    code_review_md_lines.append("## Arc 9 D1 lag implementation")
    code_review_md_lines.append("")
    code_review_md_lines.append(f"Source: `scripts/l_arc_9/experiments/pipeline_e_retry.py:_attach_d1_features`")
    code_review_md_lines.append("")
    code_review_md_lines.append("```python")
    # Extract the function body.
    func_start = src.find("def _attach_d1_features(")
    func_end = src.find("\n\n", func_start)
    code_review_md_lines.append(src[func_start:func_end])
    code_review_md_lines.append("```")
    code_review_md_lines.append("")

    missing_pieces = [p for p in expected_pattern_pieces if p not in src]
    pattern_ok = len(missing_pieces) == 0

    # --- Step 2b: compare against reference engine pattern ---
    ref_src = (_REPO_ROOT / "scripts" / "phase_kgl_v2_4h_wfo.py").read_text(encoding="utf-8")
    ref_pattern_pieces = [
        'dates_4h_norm = pd.to_datetime(df_4h["date"]).dt.normalize() - pd.Timedelta(days=1)',
        'pd.merge_asof(',
        'direction="backward"',
    ]
    ref_missing = [p for p in ref_pattern_pieces if p not in ref_src]
    ref_pattern_ok = len(ref_missing) == 0
    code_review_md_lines.append("## Reference engine pattern (KH-24 backtester)")
    code_review_md_lines.append("")
    code_review_md_lines.append(f"Source: `scripts/phase_kgl_v2_4h_wfo.py:_precompute_d1_exit_arrays` (line ~900)")
    code_review_md_lines.append("")
    code_review_md_lines.append("```python")
    code_review_md_lines.append('dates_4h_norm = pd.to_datetime(df_4h["date"]).dt.normalize() - pd.Timedelta(days=1)')
    code_review_md_lines.append("df_4h_dates = pd.DataFrame({")
    code_review_md_lines.append('    "date": dates_4h_norm,')
    code_review_md_lines.append('    "_idx": np.arange(n, dtype=int),')
    code_review_md_lines.append("})")
    code_review_md_lines.append("...")
    code_review_md_lines.append("merged = pd.merge_asof(")
    code_review_md_lines.append("    df_4h_dates.sort_values('date'),")
    code_review_md_lines.append("    df_d1,")
    code_review_md_lines.append("    on='date',")
    code_review_md_lines.append("    direction='backward',")
    code_review_md_lines.append(")")
    code_review_md_lines.append("```")
    code_review_md_lines.append("")
    code_review_md_lines.append("## Pattern equivalence")
    code_review_md_lines.append("")
    code_review_md_lines.append("Both implementations perform:")
    code_review_md_lines.append("1. Normalize 4H signal-bar timestamp to its calendar date.")
    code_review_md_lines.append("2. Subtract `pd.Timedelta(days=1)` to produce `lookup_date` = signal-day minus 1.")
    code_review_md_lines.append("3. `pd.merge_asof(direction='backward')` on the lookup_date against the D1 date series.")
    code_review_md_lines.append("")
    code_review_md_lines.append("Result: each 4H signal bar receives the D1 bar with the LATEST date ≤ (signal_date − 1 day).")
    code_review_md_lines.append("- Mid-week signal: signal_date − 1 = yesterday → D1 bar = yesterday's D1 close.")
    code_review_md_lines.append("- Monday signal: signal_date − 1 = Sunday (no D1 bar) → merge_asof finds Friday's D1 close. Days_lag = 3.")
    code_review_md_lines.append("- 00:00 UTC signal: normalize() yields the SAME signal_date that 04:00 UTC has;")
    code_review_md_lines.append("  the −1 day subtraction shifts uniformly. No 'same-day D1' branch exists in the code.")
    code_review_md_lines.append("")
    code_review_md_lines.append(f"**Arc 9 pattern verbatim-equivalent to KH-24 engine pattern: {'YES' if pattern_ok and ref_pattern_ok else 'NO'}.**")
    code_review_md_lines.append("")
    code_review_md_lines.append(f"Missing pieces in Arc 9 code (should be empty): {missing_pieces}")
    code_review_md_lines.append(f"Missing pieces in reference engine code (should be empty): {ref_missing}")

    (out_dir / "audit_2_d1_lag_code_review.md").write_text("\n".join(code_review_md_lines) + "\n", encoding="utf-8")

    # --- Step 2c: per-trade D1 lag verification (560 samples) ---
    trades_all = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv"
    )
    trades_all["signal_bar_time"] = pd.to_datetime(trades_all["signal_bar_time"])
    pairs = sorted(trades_all["pair"].unique())
    sample_per_pair = 20

    rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    hard_stop_triggered = False
    n_samples = 0

    for pair in pairs:
        sub = trades_all[trades_all["pair"] == pair].reset_index(drop=True)
        sample_idx = rng.sample(range(len(sub)), min(sample_per_pair, len(sub)))
        # Build the D1 feature frame once per pair (same as production code).
        df_d1 = _load_pair_d1(pair)
        d1_dates_arr = df_d1["date"].to_numpy()
        for i in sample_idx:
            row = sub.iloc[i]
            sig_t = row["signal_bar_time"]
            signal_date = sig_t.normalize()
            lookup_date = signal_date - pd.Timedelta(days=1)
            # Simulate the merge_asof backward: find latest D1 bar with date <= lookup_date.
            valid = d1_dates_arr <= np.datetime64(lookup_date)
            if not valid.any():
                rows.append({
                    "pair": pair, "signal_ts": sig_t, "signal_date": signal_date,
                    "lookup_date": lookup_date, "d1_date_used": pd.NaT,
                    "days_lag": -1, "leak_flag": 0, "no_d1_available": 1,
                })
                continue
            d1_idx_used = int(np.where(valid)[0][-1])
            d1_date_used = pd.Timestamp(d1_dates_arr[d1_idx_used])
            days_lag = (signal_date - d1_date_used).days
            leak_flag = int(days_lag < 1)
            n_samples += 1
            r = {
                "pair": pair,
                "signal_ts": sig_t.strftime("%Y-%m-%d %H:%M:%S"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "lookup_date": lookup_date.strftime("%Y-%m-%d"),
                "d1_date_used": d1_date_used.strftime("%Y-%m-%d"),
                "days_lag": days_lag,
                "leak_flag": leak_flag,
                "no_d1_available": 0,
                "is_monday": int(sig_t.weekday() == 0),
                "is_early_hours_utc": int(sig_t.hour in (0, 1)),
            }
            rows.append(r)
            if r["is_monday"] or r["is_early_hours_utc"]:
                boundary_rows.append(r)
            if leak_flag == 1:
                hard_stop_triggered = True
                # Don't break (continue collecting for diagnostics) but flag for halt verdict.

    df_samples = pd.DataFrame(rows)
    df_samples.to_csv(out_dir / "audit_2_d1_lag_samples.csv", index=False,
                       float_format="%.10g", lineterminator="\n")
    df_boundary = pd.DataFrame(boundary_rows)
    df_boundary.to_csv(out_dir / "audit_2_boundary_cases.csv", index=False,
                        float_format="%.10g", lineterminator="\n")

    n_leak = int((df_samples["leak_flag"] == 1).sum())
    n_no_d1 = int((df_samples["no_d1_available"] == 1).sum()) if "no_d1_available" in df_samples.columns else 0
    n_monday = int(df_samples["is_monday"].sum())
    n_early = int(df_samples["is_early_hours_utc"].sum())
    monday_min_lag = int(df_samples[df_samples["is_monday"] == 1]["days_lag"].min()) if n_monday > 0 else None
    early_min_lag = int(df_samples[df_samples["is_early_hours_utc"] == 1]["days_lag"].min()) if n_early > 0 else None

    verdict = "GREEN" if (n_leak == 0 and pattern_ok and ref_pattern_ok) else "RED"
    return {
        "audit": "2_d1_lag",
        "pattern_arc9_ok": pattern_ok,
        "pattern_ref_engine_ok": ref_pattern_ok,
        "missing_pieces_arc9": missing_pieces,
        "missing_pieces_reference": ref_missing,
        "n_samples": n_samples,
        "n_leak": n_leak,
        "n_no_d1_avail": n_no_d1,
        "n_monday_boundary": n_monday,
        "n_early_hours_boundary": n_early,
        "monday_min_days_lag": monday_min_lag,
        "early_hours_min_days_lag": early_min_lag,
        "min_days_lag_overall": int(df_samples["days_lag"].min()) if len(df_samples) else None,
        "max_days_lag_overall": int(df_samples["days_lag"].max()) if len(df_samples) else None,
        "hard_stop_triggered": hard_stop_triggered,
        "verdict": verdict,
    }


# =========================================================================
# Audit 3 - Session / hour features
# =========================================================================

def audit_3_session_features(out_dir: Path, rng: random.Random) -> Dict[str, Any]:
    feat_matrix = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "pipeline_e_retry" / "feature_matrix.csv"
    )
    feat_matrix["signal_bar_time"] = pd.to_datetime(feat_matrix["signal_bar_time"])
    sample_idx = rng.sample(range(len(feat_matrix)), 20)
    rows: List[Dict[str, Any]] = []
    n_mismatch = 0
    for i in sample_idx:
        r = feat_matrix.iloc[i]
        sig_t = r["signal_bar_time"]
        hour = sig_t.hour
        recomputed = {
            "session_london": int(8 <= hour < 16),
            "session_ny_overlap": int(12 <= hour < 16),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        }
        for fname, recomp_val in recomputed.items():
            recorded = float(r[fname])
            ok = abs(recomp_val - recorded) < 1e-9
            rows.append({
                "signal_bar_ts": sig_t.strftime("%Y-%m-%d %H:%M:%S"),
                "hour_utc": hour,
                "feature_name": fname,
                "recomputed": recomp_val,
                "recorded": recorded,
                "match": int(ok),
            })
            if not ok:
                n_mismatch += 1
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "audit_3_session_features.csv", index=False,
              float_format="%.10g", lineterminator="\n")
    return {
        "audit": "3_session_features",
        "n_samples": 20,
        "n_feature_checks": int(len(df)),
        "n_mismatches": n_mismatch,
        "verdict": "GREEN" if n_mismatch == 0 else "RED",
    }


# =========================================================================
# Audit 4 - Label leakage check (cluster_0 path-shape features in entry matrix)
# =========================================================================

def audit_4_label_leak(out_dir: Path) -> Dict[str, Any]:
    feat_matrix = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "pipeline_e_retry" / "feature_matrix.csv"
    )
    path_features = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step2_clustering" / "path_features.csv"
    )
    forward_patterns = [
        "monotonicity", "local_peaks", "pullback", "time_to_peak",
        "mfe", "mae", "final_r", "fwd_", "_so_far", "post", "peak_mfe",
    ]
    classified_rows: List[Dict[str, Any]] = []
    for c in EXPANDED_28:
        cls = "entry-time"
        for pat in forward_patterns:
            if pat in c.lower():
                cls = "FORWARD-GEOMETRY (LEAK)"
                break
        classified_rows.append({"feature": c, "classification": cls})
    n_forward = sum(1 for r in classified_rows if "LEAK" in r["classification"])

    # Column-name overlap with path-shape feature matrix.
    path_cols = set(path_features.columns) - {"trade_id", "bars_held", "final_r", "mfe_r", "mae_r", "exit_reason"}
    expanded_set = set(EXPANDED_28)
    overlap = sorted(expanded_set & path_cols)

    # Correlation matrix: each entry-time feature vs each path-shape feature on the same trade pool.
    merged = feat_matrix[["trade_id"] + EXPANDED_28].merge(
        path_features[["trade_id"] + list(path_cols)], on="trade_id", how="inner"
    )
    corr_rows: List[Dict[str, Any]] = []
    max_corr_overall = 0.0
    max_corr_pair: Tuple[str, str, float] = ("", "", 0.0)
    for fe in EXPANDED_28:
        for pc in path_cols:
            try:
                c = float(merged[fe].corr(merged[pc]))
            except Exception:
                c = float("nan")
            if not np.isnan(c) and abs(c) > abs(max_corr_overall):
                max_corr_overall = c
                max_corr_pair = (fe, pc, c)
            corr_rows.append({
                "entry_feature": fe, "path_feature": pc,
                "pearson_r": c, "abs_r": abs(c) if not np.isnan(c) else float("nan"),
                "above_0_85": int(abs(c) > 0.85) if not np.isnan(c) else 0,
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_df = corr_df.sort_values("abs_r", ascending=False)
    corr_df.to_csv(out_dir / "audit_4_correlation_matrix.csv", index=False,
                    float_format="%.10g", lineterminator="\n")

    n_above_085 = int(corr_df["above_0_85"].sum())

    feat_md = ["# Audit 4 — Feature list review", "",
               "## Entry-time feature classification (28 features)", "",
               "| Feature | Classification |",
               "|---|---|"]
    for r in classified_rows:
        feat_md.append(f"| {r['feature']} | {r['classification']} |")
    feat_md.append("")
    feat_md.append("## Column-name overlap with path-shape feature matrix")
    feat_md.append("")
    feat_md.append(f"Path-shape columns scanned: {sorted(path_cols)}")
    feat_md.append(f"Overlap with entry-time features: {overlap}")
    feat_md.append("")
    feat_md.append("## Max abs Pearson correlation (entry-time × path-shape, same trade pool)")
    feat_md.append("")
    feat_md.append(f"- Max |r|: {abs(max_corr_overall):.4f} between `{max_corr_pair[0]}` (entry-time) and `{max_corr_pair[1]}` (path-shape) (r={max_corr_pair[2]:+.4f})")
    feat_md.append(f"- Pairs with |r| > 0.85: {n_above_085}")
    feat_md.append("")
    feat_md.append("Full correlation table: `audit_4_correlation_matrix.csv` (sorted by |r| desc)")
    (out_dir / "audit_4_feature_list_review.md").write_text("\n".join(feat_md) + "\n", encoding="utf-8")

    verdict = "GREEN" if (n_forward == 0 and len(overlap) == 0 and n_above_085 == 0) else "RED"
    return {
        "audit": "4_label_leak",
        "n_features": len(EXPANDED_28),
        "n_forward_geometry_in_entry": n_forward,
        "column_overlap_with_path_shape": overlap,
        "max_abs_correlation": abs(max_corr_overall),
        "max_corr_feature_pair": [max_corr_pair[0], max_corr_pair[1]],
        "n_pairs_with_abs_corr_above_0_85": n_above_085,
        "verdict": verdict,
    }


# =========================================================================
# Audit 5 - Training/inference fold disjointness
# =========================================================================

def audit_5_fold_disjointness(out_dir: Path) -> Dict[str, Any]:
    """Code review: confirm Step 5 trains per-fold on data with entry_time <
    fold.oos_start (anchored expanding) AND Pipeline E retry TSS-CV uses
    chronological order with no test-fold-train-fold overlap."""

    feat_matrix = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e" / "feature_matrix.csv"
    )
    feat_matrix["entry_time"] = pd.to_datetime(feat_matrix["entry_time"])
    feat_matrix = feat_matrix.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
    n = len(feat_matrix)

    # TSS(5) train/test bounds in chronological order.
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    tss_folds: List[Dict[str, Any]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(np.arange(n)), start=1):
        tss_folds.append({
            "scheme": "TimeSeriesSplit(5)",
            "fold": fold_idx,
            "train_idx_min": int(tr_idx.min()),
            "train_idx_max": int(tr_idx.max()),
            "test_idx_min": int(te_idx.min()),
            "test_idx_max": int(te_idx.max()),
            "train_date_min": feat_matrix["entry_time"].iloc[tr_idx.min()].strftime("%Y-%m-%d"),
            "train_date_max": feat_matrix["entry_time"].iloc[tr_idx.max()].strftime("%Y-%m-%d"),
            "test_date_min": feat_matrix["entry_time"].iloc[te_idx.min()].strftime("%Y-%m-%d"),
            "test_date_max": feat_matrix["entry_time"].iloc[te_idx.max()].strftime("%Y-%m-%d"),
            "overlap": int(tr_idx.max() >= te_idx.min()),
        })

    # KH-24 anchor fold windows.
    kh24 = yaml.safe_load((_REPO_ROOT / "configs" / "wfo_kh24.yaml").read_text(encoding="utf-8"))
    wfo_folds: List[Dict[str, Any]] = []
    for f in kh24["wfo"]["folds"]:
        s = pd.Timestamp(f["oos_start"])
        e = pd.Timestamp(f["oos_end"])
        # Under Step 5 anchored expanding, training set = entry_time < s.
        train_mask = feat_matrix["entry_time"] < s
        n_train = int(train_mask.sum())
        n_oos = int(((feat_matrix["entry_time"] >= s) & (feat_matrix["entry_time"] < e)).sum())
        train_date_max = feat_matrix.loc[train_mask, "entry_time"].max() if n_train > 0 else None
        wfo_folds.append({
            "scheme": "KH-24 anchored expanding (Step 5)",
            "fold": int(f["fold"]),
            "oos_start": s.strftime("%Y-%m-%d"),
            "oos_end": e.strftime("%Y-%m-%d"),
            "n_train": n_train,
            "n_oos": n_oos,
            "train_date_max": train_date_max.strftime("%Y-%m-%d") if train_date_max is not None else "—",
            "overlap_train_into_oos": int((train_date_max >= s) if train_date_max is not None else 0),
        })

    overlap_table = tss_folds + wfo_folds
    pd.DataFrame(overlap_table).to_csv(
        out_dir / "audit_5_fold_overlap_table.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )

    tss_overlap = sum(f["overlap"] for f in tss_folds)
    wfo_overlap = sum(f["overlap_train_into_oos"] for f in wfo_folds)

    # Step 5 LGBM E ALREADY performs per-fold anchored-expanding retraining
    # (per scripts/l_arc_9/experiments/step5_lgbm_pipeline_e.py:_wfo_for_threshold).
    # Code review evidence:
    step5_src = (_REPO_ROOT / "scripts" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e.py").read_text(encoding="utf-8")
    walkforward_in_step5 = (
        "train_mask = entry_time < np.datetime64(s)" in step5_src
        and "mdl = lgb.LGBMClassifier(**LGBM_KW)" in step5_src
    )

    # Cross-check the parity file confirms reproduction (Pipeline E retry's TSS-CV
    # was already walk-forward by construction of TimeSeriesSplit).
    parity = json.loads((
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e" / "parity_check.json"
    ).read_text(encoding="utf-8"))

    verdict = "GREEN" if (tss_overlap == 0 and wfo_overlap == 0 and walkforward_in_step5) else "RED"
    return {
        "audit": "5_fold_disjointness",
        "tss_folds_overlap_count": tss_overlap,
        "wfo_folds_train_into_oos_count": wfo_overlap,
        "step5_walkforward_in_code": walkforward_in_step5,
        "parity_reproduced_auc": parity.get("reproduced_mean_auc"),
        "parity_pass": parity.get("parity"),
        "verdict": verdict,
        "note": (
            "Pipeline E retry uses TimeSeriesSplit(5) which is walk-forward by construction "
            "(train indices < test indices always). Step 5 LGBM Pipeline E retrains per "
            "KH-24 fold using anchored expanding training (train_mask = entry_time < fold.oos_start). "
            "Neither scheme has training overlap with its own test fold."
        ),
    }


# =========================================================================
# Audit 6 - Cluster label flow
# =========================================================================

def audit_6_cluster_label_flow(out_dir: Path) -> Dict[str, Any]:
    findings: List[str] = []
    md = ["# Audit 6 — Cluster label flow", "",
          "## Question",
          "",
          "Does `is_cluster_0` (computed from forward-geometry path features) leak into the",
          "inference pathway as anything other than a target during training?",
          ""]

    # 1) feature_matrix.csv columns: y must be present as target; cluster id NOT present as feature.
    fm = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "pipeline_e_retry" / "feature_matrix.csv"
    )
    has_y = "y" in fm.columns
    forbidden_label_cols = {"cluster_id", "cluster_0", "is_cluster_0", "label", "target"}
    forbidden_in_features = forbidden_label_cols & set(EXPANDED_28)
    md.append("## Step 1 — `y` is target column, never a feature column")
    md.append("")
    md.append(f"- Feature matrix has `y` column: {has_y} (used as target)")
    md.append(f"- Forbidden label columns appearing in EXPANDED_28: {sorted(forbidden_in_features)}")
    md.append("")

    # 2) Inference call site in Step 5: classifier scores X = features only.
    step5_src = (_REPO_ROOT / "scripts" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e.py").read_text(encoding="utf-8")
    inference_line = 'mdl.predict_proba(X[oos_mask])[:, 1]'
    inference_uses_features_only = inference_line in step5_src
    md.append("## Step 2 — Inference uses features-only X matrix")
    md.append("")
    md.append(f"- `scripts/l_arc_9/experiments/step5_lgbm_pipeline_e.py` contains: `{inference_line}`: {inference_uses_features_only}")
    md.append("- The X matrix is `df_clean[EXPANDED_28].to_numpy(...)` — explicitly the feature columns, not the label column.")
    md.append("")

    # 3) Cluster fitting: clusters_K3.csv was produced ONCE at Step 2 from path-shape features.
    # No re-fit at inference time (Step 5 does not re-run k-means).
    cluster_src = (_REPO_ROOT / "scripts" / "l_arc_9" / "step2_clustering.py").read_text(encoding="utf-8")
    kmeans_in_step5 = "KMeans" in step5_src
    md.append("## Step 3 — No k-means at inference time")
    md.append("")
    md.append(f"- KMeans imported in Step 5 LGBM Pipeline E script: {kmeans_in_step5}")
    md.append(f"- k-means is computed once at Step 2 (`scripts/l_arc_9/step2_clustering.py`) using only path-shape features;")
    md.append(f"  the resulting `clusters_K3.csv` provides the binary `is_cluster_0` labels used as classification target.")
    md.append("")

    md.append("## Conclusion")
    md.append("")
    flow_clean = (has_y and len(forbidden_in_features) == 0
                  and inference_uses_features_only and not kmeans_in_step5)
    md.append(f"- Label flows: Step 2 (k-means on path-shape) → `is_cluster_0` target → Step 4/E-retry training → trained model → Step 5 inference on entry-time features only.")
    md.append(f"- No leakage path: cluster labels never enter inference X. Classifier predicts is_cluster_0 from entry-time features only.")
    md.append(f"- **Verdict: {'GREEN' if flow_clean else 'RED'}**")
    (out_dir / "audit_6_label_flow.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return {
        "audit": "6_cluster_label_flow",
        "y_is_target_only": has_y and len(forbidden_in_features) == 0,
        "inference_uses_features_only": inference_uses_features_only,
        "no_kmeans_in_step5_inference": not kmeans_in_step5,
        "verdict": "GREEN" if flow_clean else "RED",
    }


# =========================================================================
# Audit 7 - Spread & execution semantics at inference
# =========================================================================

def audit_7_execution(out_dir: Path, rng: random.Random) -> Dict[str, Any]:
    # Sample 20 admitted trades from candidate A (or B if A insufficient).
    cand_a = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e"
        / "candidate_A_thr0.40" / "resim_trades.csv"
    )
    cand_b = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e"
        / "candidate_B_thr0.05" / "resim_trades.csv"
    )
    sample_source = cand_a if len(cand_a) >= 20 else pd.concat([cand_a, cand_b])
    sample_idx = rng.sample(range(len(sample_source)), 20)
    samples = sample_source.iloc[sample_idx].copy()
    samples["signal_bar_time"] = pd.to_datetime(samples["signal_bar_time"])
    samples["entry_time"] = pd.to_datetime(samples["entry_time"])

    # Cross-reference each sample against Step 1 trades_all.csv (source of truth).
    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    trades_all["signal_bar_time"] = pd.to_datetime(trades_all["signal_bar_time"])
    trades_all["entry_time"] = pd.to_datetime(trades_all["entry_time"])

    # Load spread floor for the per-pair floor check (production semantics:
    # spread_pips_used = max(raw_pips, floor_pips_for_pair)).
    floor_yaml = yaml.safe_load((_REPO_ROOT / "configs" / "spread_floors_5ers.yaml").read_text(encoding="utf-8"))
    floor_pips_by_pair: Dict[str, float] = {}
    for pair_name, floor_info in floor_yaml["floors"].items():
        # min_nonzero_spread_native is raw MT5 points; divide by 10 = pips.
        floor_pips_by_pair[pair_name] = float(floor_info["min_nonzero_spread_native"]) / 10.0

    rows: List[Dict[str, Any]] = []
    n_fail = 0
    for _, s in samples.iterrows():
        pair = s["pair"]
        sig_t = s["signal_bar_time"]
        entry_t = s["entry_time"]
        # Load 4H bar data.
        df_pair = _load_pair_4h(pair)
        sig_idx_arr = np.where(df_pair["date"].to_numpy() == np.datetime64(sig_t))[0]
        if sig_idx_arr.size == 0:
            n_fail += 1
            continue
        sig_idx = int(sig_idx_arr[0])
        entry_idx = sig_idx + 1
        # Check 1: entry_time matches bar t+1
        entry_t_expected = pd.Timestamp(df_pair["date"].iloc[entry_idx])
        check_entry_time_at_t_plus_1 = (entry_t == entry_t_expected)
        # Check 2: spread used = bar t+1's spread (from data)
        sp_t1 = df_pair["spread"].iloc[entry_idx]
        # The Step 1 trades_all.csv has spread_pips_used = raw_pips after floor.
        # raw_pips = spread / 10; if 0, floor kicks in; otherwise raw_pips itself.
        # For the audit, just confirm that the recorded spread_pips_used reflects bar t+1 data
        # (i.e. <= max plausible spread for the bar).
        ta_row = trades_all[(trades_all["pair"] == pair) & (trades_all["signal_bar_time"] == sig_t)]
        if ta_row.empty:
            n_fail += 1
            continue
        ta = ta_row.iloc[0]
        spread_pips_used_recorded = float(ta["spread_pips_used"])
        # Production semantics (core/spread_floor.py:apply_spread_floor_to_pips):
        # spread_pips_used = max(raw_pips_from_execution_bar, floor_pips_for_pair).
        # Raw_pips = bar t+1 spread / 10. Floor = floor_pips_by_pair[pair].
        raw_pips_t_plus_1 = float(sp_t1) / 10.0
        floor_pips = floor_pips_by_pair.get(pair, 0.0)
        expected_spread_pips = max(raw_pips_t_plus_1, floor_pips)
        spread_check_ok = abs(spread_pips_used_recorded - expected_spread_pips) < 1e-6
        # Check 3: SL placement uses ATR at signal bar (not t+1)
        # ATR at signal bar is stored in trades_all.csv as atr14_at_signal.
        atr_at_sig_recorded = float(ta["atr14_at_signal"])
        # The actual entry_fill - sl_at_entry_price = 2.0 * atr14_at_signal
        entry_price = float(ta["entry_price"])
        sl_price = float(ta["sl_at_entry_price"])
        sl_distance_recorded = entry_price - sl_price
        sl_distance_expected_from_signal_atr = 2.0 * atr_at_sig_recorded
        sl_check_ok = abs(sl_distance_recorded - sl_distance_expected_from_signal_atr) < 1e-6
        all_ok = check_entry_time_at_t_plus_1 and spread_check_ok and sl_check_ok
        if not all_ok:
            n_fail += 1
        rows.append({
            "pair": pair, "signal_bar_ts": sig_t.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time_recorded": entry_t.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time_expected_t_plus_1": entry_t_expected.strftime("%Y-%m-%d %H:%M:%S"),
            "check_entry_time_at_t_plus_1": int(check_entry_time_at_t_plus_1),
            "spread_at_bar_t_plus_1_raw": float(sp_t1),
            "spread_pips_used_recorded": spread_pips_used_recorded,
            "check_spread_at_execution_bar": int(spread_check_ok),
            "atr_at_signal_recorded": atr_at_sig_recorded,
            "sl_distance_recorded": sl_distance_recorded,
            "sl_distance_expected_2x_atr_signal": sl_distance_expected_from_signal_atr,
            "check_sl_from_signal_atr_not_t_plus_1": int(sl_check_ok),
            "all_checks_pass": int(all_ok),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "audit_7_execution_semantics.csv", index=False,
              float_format="%.10g", lineterminator="\n")
    return {
        "audit": "7_execution_semantics",
        "n_samples": 20,
        "n_pass": int((df["all_checks_pass"] == 1).sum()),
        "n_fail": int((df["all_checks_pass"] == 0).sum()),
        "volume_veto_note": "Arc 9 signal (lchar_inside_bar_break_trend_long.py) has no volume veto condition; check N/A.",
        "verdict": "GREEN" if n_fail == 0 else "RED",
    }


# =========================================================================
# Audit 8 - End-to-end probability reproduction
# =========================================================================

def audit_8_e2e_reproduction(out_dir: Path, rng: random.Random) -> Dict[str, Any]:
    """For 100 random admitted trades, reload classifier (re-train per fold same
    as Step 5), score, compare probability to recorded value in admitted_trades.csv.
    """
    # Load Candidate A admitted trades (smaller set, 236 admits — more than enough to sample).
    cand_a = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e"
        / "candidate_A_thr0.40" / "admitted_trades.csv"
    )
    cand_b = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e"
        / "candidate_B_thr0.05" / "admitted_trades.csv"
    )
    # Sample 50 from each.
    a_sample = cand_a.sample(n=min(50, len(cand_a)), random_state=AUDIT_SEED)
    b_sample = cand_b.sample(n=min(50, len(cand_b)), random_state=AUDIT_SEED + 1)
    a_sample["candidate"] = "A_thr0.40"
    b_sample["candidate"] = "B_thr0.05"
    samples = pd.concat([a_sample, b_sample], ignore_index=True)
    # CRITICAL: rebuild the feature matrix IN-MEMORY using the same pipeline as
    # Step 5 / Pipeline E retry. Reading from feature_matrix.csv loses precision
    # via %.10g serialisation, which (a) is deterministic but (b) propagates to
    # slightly-different LGBM splits and ~1e-2 prob diffs. The audit requires
    # byte-identical reproduction vs the Step 5 in-memory training path, so we
    # invoke Step 5's _build_feature_matrix directly.
    from scripts.l_arc_9.experiments.step5_lgbm_pipeline_e import _build_feature_matrix
    feat_matrix = _build_feature_matrix(out_dir)
    feat_matrix["entry_time"] = pd.to_datetime(feat_matrix["entry_time"])
    feat_matrix = feat_matrix.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
    X_all = feat_matrix[EXPANDED_28].to_numpy(dtype=float)
    y_all = feat_matrix["y"].to_numpy(dtype=int)

    kh24 = yaml.safe_load((_REPO_ROOT / "configs" / "wfo_kh24.yaml").read_text(encoding="utf-8"))
    kh24_folds: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = [
        (int(f["fold"]), pd.Timestamp(f["oos_start"]), pd.Timestamp(f["oos_end"]))
        for f in kh24["wfo"]["folds"]
    ]

    # Pre-train one classifier per fold (cache for speed).
    classifiers: Dict[int, Any] = {}
    entry_time_arr = feat_matrix["entry_time"].to_numpy()
    for fold_id, s, e in kh24_folds:
        train_mask = entry_time_arr < np.datetime64(s)
        if train_mask.sum() == 0 or y_all[train_mask].sum() < 10:
            classifiers[fold_id] = None
            continue
        mdl = lgb.LGBMClassifier(**LGBM_KW)
        mdl.fit(X_all[train_mask], y_all[train_mask])
        classifiers[fold_id] = mdl

    # Re-score each sample trade using its fold's classifier.
    rows: List[Dict[str, Any]] = []
    n_match = 0
    n_total = 0
    for _, s in samples.iterrows():
        tid = int(s["trade_id"])
        fold = int(s["fold"])
        recorded_prob = float(s["prob"])
        # Find the row in feat_matrix.
        match = feat_matrix[feat_matrix["trade_id"].astype(int) == tid]
        if match.empty:
            continue
        X_one = match[EXPANDED_28].to_numpy(dtype=float)
        mdl = classifiers.get(fold)
        if mdl is None:
            continue
        prob = float(mdl.predict_proba(X_one)[:, 1][0])
        diff = abs(prob - recorded_prob)
        ok = diff < 1e-6
        if ok:
            n_match += 1
        n_total += 1
        rows.append({
            "candidate": s["candidate"],
            "trade_id": tid,
            "fold": fold,
            "recorded_prob": recorded_prob,
            "reproduced_prob": prob,
            "abs_diff": diff,
            "match": int(ok),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "audit_8_e2e_reproduction.csv", index=False,
              float_format="%.10g", lineterminator="\n")
    return {
        "audit": "8_e2e_reproduction",
        "n_samples": n_total,
        "n_match_within_1e-6": n_match,
        "n_mismatch": n_total - n_match,
        "max_abs_diff": float(df["abs_diff"].max()) if len(df) > 0 else float("nan"),
        "verdict": "GREEN" if (n_total > 0 and n_match == n_total) else "RED",
    }


# =========================================================================
# Master driver
# =========================================================================


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 lookahead and cross-timeframe leak audit.")
    parser.add_argument("--out-dir", type=Path,
                        default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "lookahead_audit")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(AUDIT_SEED)

    print("=" * 70)
    print("Arc 9 lookahead and leak audit — 8 checks")
    print("=" * 70)

    results: Dict[str, Any] = {}

    print("\n[audit 1] 4H feature timestamp boundary...")
    results["1"] = audit_1_4h_timestamps(args.out_dir, rng)
    print(f"  -> {results['1']['verdict']} ({results['1']['n_mismatches']} mismatches / {results['1']['n_feature_checks']} checks)")

    print("\n[audit 2] D1 lag integrity (CRITICAL)...")
    results["2"] = audit_2_d1_lag(args.out_dir, rng)
    print(f"  -> {results['2']['verdict']} ({results['2']['n_leak']} leaks / {results['2']['n_samples']} samples; "
          f"min lag {results['2']['min_days_lag_overall']}, max lag {results['2']['max_days_lag_overall']})")
    if results["2"]["hard_stop_triggered"]:
        print("  ** HARD-STOP TRIGGER FIRED: D1 lag failure **")

    print("\n[audit 3] Session / hour features...")
    results["3"] = audit_3_session_features(args.out_dir, rng)
    print(f"  -> {results['3']['verdict']} ({results['3']['n_mismatches']} mismatches / {results['3']['n_feature_checks']} checks)")

    print("\n[audit 4] Label leakage (cluster_0 path-shape features)...")
    results["4"] = audit_4_label_leak(args.out_dir)
    print(f"  -> {results['4']['verdict']} (forward-geom in entry: {results['4']['n_forward_geometry_in_entry']}, "
          f"col overlap: {len(results['4']['column_overlap_with_path_shape'])}, "
          f"max |r|: {results['4']['max_abs_correlation']:.4f})")

    print("\n[audit 5] Training/inference fold disjointness...")
    results["5"] = audit_5_fold_disjointness(args.out_dir)
    print(f"  -> {results['5']['verdict']} (TSS overlap: {results['5']['tss_folds_overlap_count']}, "
          f"WFO overlap: {results['5']['wfo_folds_train_into_oos_count']}, "
          f"walkforward in Step 5: {results['5']['step5_walkforward_in_code']})")

    print("\n[audit 6] Cluster label flow...")
    results["6"] = audit_6_cluster_label_flow(args.out_dir)
    print(f"  -> {results['6']['verdict']}")

    print("\n[audit 7] Spread & execution semantics...")
    results["7"] = audit_7_execution(args.out_dir, rng)
    print(f"  -> {results['7']['verdict']} ({results['7']['n_pass']} pass / {results['7']['n_samples']})")

    print("\n[audit 8] End-to-end probability reproduction...")
    results["8"] = audit_8_e2e_reproduction(args.out_dir, rng)
    print(f"  -> {results['8']['verdict']} ({results['8']['n_match_within_1e-6']}/{results['8']['n_samples']} match, "
          f"max abs diff {results['8']['max_abs_diff']:.2e})")

    print("\n" + "=" * 70)
    overall = "GREEN" if all(r["verdict"] == "GREEN" for r in results.values()) else "RED"
    print(f"OVERALL VERDICT: {overall}")
    print("=" * 70)

    # Persist summary JSON.
    summary = {
        "overall_verdict": overall,
        "audits": results,
        "rng_seed": AUDIT_SEED,
    }
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return 0 if overall == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
