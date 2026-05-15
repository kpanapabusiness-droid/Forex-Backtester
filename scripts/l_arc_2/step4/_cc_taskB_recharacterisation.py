"""CC Task B: §9 re-characterisation (focused subset) for L Arc 2 step 5
delayed_entry_t_gb.

Lives under scripts/l_arc_2/step4/ alongside CC-0 / CC-1 / CC Task A.

Per planner authorisation, this is a FOCUSED-SUBSET replication of step
2 + step 3 templates on the post-mechanism trade sets — NOT a literal
full-template re-run of the existing step-2 / step-3 orchestrators
(which hardcode the full pool as input and the canonical pool output
directories, and would overwrite pool baselines if re-invoked as-is).

Approach: re-use step 3's library functions (`_data.build_cluster_features`,
`_clustering.fit_kmeans`, `_common.fmt_dist_stats`) so methodology stays
faithful to op spec §6.1 / §11.1. Step 2 marginals + conditionals are
computed inline from signals_features.csv since the pool's step-2
features are already on disk per trade.

Tasks B-I per dispatch:
  - B: step 2 focused subset on F6+F7 (43 trades)
  - C: step 2 focused subset on F2-F5 (157 trades)
  - D: step 3 focused subset on F6+F7 (cluster K=2 + effect sizes + univariate predictor scan)
  - E: step 3 focused subset on F2-F5
  - F: filter-intent check (lookup-based against original cluster_assignments.csv — no refit)
  - G: sub-archetype discovery (comparison of new K=2 vs original K=2 on F2-F5)
  - H: determinism receipt (two-run byte-identity)
  - I: execution-side lookahead test (N/A — no per-bar OHLC recomputation in this script)

Outputs under results/l_arc_2/step5_recharacterisation/delayed_entry_t_gb/:
  - step2_recharacterisation_{f2_f5, f6_f7}/* (marginals, forward, sequence,
    survival, asymmetry, per-pair/-fold/-session, exit_reason)
  - step3_recharacterisation_{f2_f5, f6_f7}/* (cluster_assignments_new,
    cluster_summary, cluster_effect_sizes, cluster_distributions,
    predictor_scan_univariate)
  - filter_intent_check.csv
  - subarchetype_discovery.csv
  - cc_taskB_run_manifest.json
  - cc_taskB_validation_receipts.json

Determinism: all seeds derive from C.BASE_SEED. All sorts explicit. No
wall-clock timestamps in manifest. Two-run byte-identity required (Task H).

Pin discipline: same approach as CC Task A — data sha256 pins are STRICT;
source-blob drift is documented but lint-only per PR #124 and authorised
by planner. Custom-input methodology blocker on step 2/3 orchestrators
flagged in the report; this script is the scope-bounded alternative
deliverable approved by planner.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.l_arc_2.step3 import _clustering as CL
from scripts.l_arc_2.step3 import _common as C3
from scripts.l_arc_2.step3 import _data as D3

REPO = C3.REPO
STEP4_OUT = REPO / "results" / "l_arc_2" / "step4" / "delayed_entry_t_gb"
STEP5_OUT = REPO / "results" / "l_arc_2" / "step5_recharacterisation" / "delayed_entry_t_gb"
STEP3_POOL_DIR = REPO / "results" / "l_arc_2" / "step3_extractability"

# Reproducibility pins — mirror CC-1 / CC Task A.
PREDICTOR_PIN_COMMIT = "fe345be58a8517189e898d0618266f083bda900c"
CC0_SCRIPT_PIN_COMMIT = "86215fc"
CC1_SCRIPT_PIN_COMMIT = "0224cd3"
PINNED_SCRIPT_FILES = (
    "scripts/l_arc_2/step4/_predictor.py",
    "scripts/l_arc_2/step4/_common.py",
    "scripts/l_arc_2/step4/_data.py",
    "scripts/l_arc_2/step4/_actions.py",
    "scripts/l_arc_2/step4/_simulator.py",
    "scripts/l_arc_2/step4/run_step4.py",
    "scripts/l_arc_2/step4/_cc0_materialise_f2_f5_delayed_entry.py",
    "scripts/l_arc_2/step4/_cc1_winner_path_diagnostic.py",
)
CC0_PINNED_FILE = "scripts/l_arc_2/step4/_cc0_materialise_f2_f5_delayed_entry.py"
CC1_PINNED_FILE = "scripts/l_arc_2/step4/_cc1_winner_path_diagnostic.py"

DATA_SHA256_PINS = {
    "trades_post_mechanism_f2_f5.csv":
        "e76bae339e456946c1674a90694e7906aca28596799746b21dcc405de69054be",
    "trades_post_mechanism.csv":
        "e909d2c5e95ddf7765b315541db41aacb48def517ab894be0a2e00371483b744",
    "winner_paths_bar_by_bar.csv":
        "1680e3ea642044b4a29aa7242e8009fddaf429872e61f8884c4dbe19b9556122",
    "loser_paths_bar_by_bar.csv":
        "121e6c51600c4f4b07b64ed1a97e9a63c6da8a5bfc4679c95ea741c1322e250f",
}

PR_124_DRIFT_CLASSIFICATION = {
    "scripts/l_arc_2/step4/_predictor.py":
        "lint_cleanup: removed 1 blank line; no semantic change",
    "scripts/l_arc_2/step4/_common.py":
        "lint_cleanup: removed unused `import numpy as np`; no semantic change",
    "scripts/l_arc_2/step4/_data.py":
        "lint_cleanup: removed 1 blank line; no semantic change",
    "scripts/l_arc_2/step4/_actions.py":
        "lint_cleanup: removed 2 unused imports + 1 unused local var; no semantic change",
    "scripts/l_arc_2/step4/_simulator.py":
        "lint_cleanup: removed 2 unused imports + blank lines; no semantic change",
    "scripts/l_arc_2/step4/run_step4.py":
        "lint_cleanup: alphabetised imports + 1 f-string with no substitutions converted to str; no semantic change",
    "scripts/l_arc_2/step4/_cc0_materialise_f2_f5_delayed_entry.py":
        "lint_cleanup: removed 1 blank line; no semantic change",
    "scripts/l_arc_2/step4/_cc1_winner_path_diagnostic.py":
        "lint_cleanup: removed 1 blank line; no semantic change",
}

# Step 2 marginal metrics.
#
# POST_MECHANISM_METRICS are computed from trades_post_mechanism CSV +
# CC-1 / CC Task A bar-by-bar held R derivations — i.e. delayed-entry-
# indexed under post-mechanism economics. These override any verbatim
# values that the same column name would have in signals_features.csv.
#
# VERBATIM_INFORMATIONAL_METRICS are step-2 path-complexity / sequence
# features computed from the verbatim trade (before the delayed-entry
# mechanism rewrote exit policy). They are NOT delayed-entry-indexed.
# Reported as informational with a verbatim_frame=True flag so the
# planner cannot misread them as post-mechanism numbers.
POST_MECHANISM_METRICS = (
    "net_r", "gross_r", "spread_cost_r",
    "mfe_atr_DE", "mae_atr_DE", "bars_held_DE",
)
VERBATIM_INFORMATIONAL_METRICS = (
    "peak_to_final_r_ratio", "mfe_to_mae_ratio_held", "r_given_back_from_peak",
    "time_to_peak_mfe", "time_to_trough_mae",
    "oscillation_count", "monotonicity_ratio",
)

# Step 2 forward-geometry metrics
STEP2_FORWARD_METRICS = (
    "fwd_mfe_h24_atr", "fwd_mae_h24_atr", "fwd_mfe_to_mae_ratio_h24",
    "fwd_mfe_h120_atr", "fwd_mae_h120_atr", "fwd_mfe_to_mae_ratio_h120",
    "race_bars_plus1_minus_minus1",
    "reached_plus_1.0_atr_within_240",
)

SURVIVAL_BARS = (1, 5, 10, 20, 50, 100, 120)

# Signal-time predictor whitelist for univariate AUC scan (subset of C3.SIGNAL_TIME_NUMERIC)
PREDICTOR_SCAN_FEATURES = (
    "atr_at_signal_1h", "atr_ratio_to_baseline",
    "cum_logret_1h_3", "cum_logret_1h_6", "cum_logret_1h_24",
    "cum_logret_1h_72", "cum_logret_1h_168",
    "vol_realized_1h_24h",
    "dist_close_to_high30_atr", "dist_close_to_low30_atr",
    "hour_utc", "day_of_week", "hour_in_4h_bar", "hour_in_d1_bar",
    "concurrent_signals_same_bar", "concurrent_signals_within_3h",
    "currency_basket_3h_USD", "currency_basket_3h_EUR",
    "currency_basket_3h_JPY", "currency_basket_3h_GBP",
    "trade_overlap_at_execution_time",
    "sequential_same_pair_density_24h",
    "trigger_magnitude_decile",
)


# ============================================================
# Pin verification
# ============================================================

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_head() -> str:
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO))
    return out.decode().strip()


def _git_blob_at_commit(commit: str, path: str) -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"], cwd=str(REPO),
    )
    return out.decode().strip()


def _git_blob_safe(commit: str, path: str) -> str | None:
    try:
        return _git_blob_at_commit(commit, path)
    except subprocess.CalledProcessError:
        return None


def _diff_w_shortstat(a: str, b: str, path: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "diff", "-w", "--shortstat", f"{a}..{b}", "--", path],
            cwd=str(REPO),
        )
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return None


def _verify_source_pin() -> dict:
    head = _git_commit_head()
    per_file = {}
    all_match = True
    all_lint = True
    for path in PINNED_SCRIPT_FILES:
        if path == CC0_PINNED_FILE:
            pin = CC0_SCRIPT_PIN_COMMIT
        elif path == CC1_PINNED_FILE:
            pin = CC1_SCRIPT_PIN_COMMIT
        else:
            pin = PREDICTOR_PIN_COMMIT
        head_blob = _git_blob_safe(head, path)
        pin_blob = _git_blob_safe(pin, path)
        match = head_blob is not None and pin_blob is not None and head_blob == pin_blob
        if not match:
            all_match = False
        diff_w_stat = None
        cls = None
        if not match and head_blob is not None and pin_blob is not None:
            diff_w_stat = _diff_w_shortstat(pin, head, path)
            cls = PR_124_DRIFT_CLASSIFICATION.get(path, "substantive_unknown")
            if not cls.startswith("lint_cleanup"):
                all_lint = False
        per_file[path] = {
            "pin_commit": pin,
            "blob_at_pin_commit": pin_blob,
            "blob_at_head": head_blob,
            "match": match,
            "diff_w_shortstat": diff_w_stat,
            "drift_classification": cls,
        }
    return {
        "head_commit": head,
        "all_pinned_files_blob_match": all_match,
        "all_drift_classified_as_lint_cleanup": all_lint,
        "drift_resolution_note": (
            "Lint-only drift across pinned step-4 / CC-0 / CC-1 source files "
            "introduced by PR #124, documented and authorised by planner. This "
            "dispatch invokes neither classifier nor simulator (only library "
            "functions from scripts/l_arc_2/step3); data sha256 pins are strict."
        ),
        "per_file": per_file,
    }


def _verify_data_pins() -> dict:
    paths = {
        "trades_post_mechanism_f2_f5.csv":
            STEP5_OUT / "trades_post_mechanism_f2_f5.csv",
        "trades_post_mechanism.csv":
            STEP4_OUT / "trades_post_mechanism.csv",
        "winner_paths_bar_by_bar.csv":
            STEP5_OUT / "winner_paths_bar_by_bar.csv",
        "loser_paths_bar_by_bar.csv":
            STEP5_OUT / "loser_paths_bar_by_bar.csv",
    }
    per_file = {}
    all_match = True
    for name, p in paths.items():
        actual = _sha256_file(p)
        expected = DATA_SHA256_PINS[name]
        match = actual == expected
        if not match:
            all_match = False
        per_file[name] = {"pinned_sha256": expected, "actual_sha256": actual, "match": match}
    return {"all_data_sha256_match": all_match, "per_file": per_file}


def _versions() -> dict:
    import sklearn
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


# ============================================================
# Step 2 focused subset
# ============================================================

def _marginal_stats(values: pd.Series) -> dict:
    arr = pd.to_numeric(values, errors="coerce").dropna().values.astype(float)
    if arr.size == 0:
        keys = ("n", "mean", "std", "min", "p1", "p5", "p10", "p25", "p50",
                "p75", "p90", "p95", "p99", "max")
        return {k: float("nan") for k in keys}
    pcts = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    q = np.percentile(arr, pcts)
    out = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(np.min(arr)),
    }
    for p, qv in zip(pcts, q):
        out[f"p{p}"] = float(qv)
    out["max"] = float(np.max(arr))
    return out


def _step2_marginals_table(sub_features: pd.DataFrame,
                            metrics: tuple[str, ...],
                            frame_label: str) -> pd.DataFrame:
    """Emit a marginals table over `metrics`. `frame_label` is recorded
    in a `frame` column (either "delayed_entry_indexed_post_mechanism"
    or "verbatim_step2_frame_informational") so the planner can read
    the methodology context from the output directly.
    """
    rows = []
    nan_template = {k: float("nan") for k in [
        "n", "mean", "std", "min", "p1", "p5", "p10", "p25", "p50",
        "p75", "p90", "p95", "p99", "max"]}
    for m in metrics:
        if m not in sub_features.columns:
            rows.append({"metric": m, "frame": frame_label, **nan_template})
            continue
        row = {"metric": m, "frame": frame_label, **_marginal_stats(sub_features[m])}
        rows.append(row)
    return pd.DataFrame(rows)


def _step2_sequence_class_counts(sub_features: pd.DataFrame) -> pd.DataFrame:
    """§5.4 sequence-class counts."""
    col = "mfe_sequence_class_held"
    if col not in sub_features.columns:
        return pd.DataFrame(columns=["category", "count", "fraction"])
    vc = sub_features[col].fillna("__nan__").value_counts().sort_index()
    total = int(vc.sum())
    return pd.DataFrame({
        "category": vc.index.astype(str).values,
        "count": vc.values.astype(int),
        "fraction": (vc.values.astype(float) / total) if total else vc.values,
    })


def _step2_survival_curve_DE(pm_frame: pd.DataFrame) -> pd.DataFrame:
    """§5.6 survival curve under POST-MECHANISM economics: fraction
    still open at t ∈ SURVIVAL_BARS using delayed-entry-indexed
    bars_held_DE.
    """
    if "bars_held_DE" not in pm_frame.columns:
        return pd.DataFrame()
    bh = pm_frame["bars_held_DE"].astype(float).dropna()
    n = int(len(bh))
    rows = []
    for t in SURVIVAL_BARS:
        still_open_mask = pm_frame["bars_held_DE"].astype(float) > t
        n_open = int(still_open_mask.sum())
        frac = (n_open / n) if n else float("nan")
        if "net_r" in pm_frame.columns and n_open > 0:
            r = pm_frame.loc[still_open_mask, "net_r"]
            mean_r = float(r.mean())
            win_pct = float((r > 0).mean())
        else:
            mean_r = float("nan")
            win_pct = float("nan")
        rows.append({
            "t_bar": t,
            "n_total": n,
            "n_still_open_at_t": n_open,
            "fraction_still_open": frac,
            "mean_net_r_cond_open": mean_r,
            "win_pct_cond_open": win_pct,
            "frame": "delayed_entry_indexed_post_mechanism",
        })
    return pd.DataFrame(rows)


def _step2_asymmetry_DE(pm_frame: pd.DataFrame) -> pd.DataFrame:
    """§5.8 winner / loser asymmetry under POST-MECHANISM economics
    (winner = post-mechanism net_r > 0).
    """
    if "net_r" not in pm_frame.columns:
        return pd.DataFrame()
    win = pm_frame[pm_frame["net_r"] > 0]
    los = pm_frame[pm_frame["net_r"] <= 0]

    def _agg(grp: pd.DataFrame, label: str) -> dict:
        n = int(len(grp))
        if n == 0:
            return {"group": label, "n": 0}
        return {
            "group": label,
            "n": n,
            "mean_r": float(grp["net_r"].mean()),
            "median_r": float(grp["net_r"].median()),
            "p5_r": float(np.percentile(grp["net_r"], 5)),
            "p95_r": float(np.percentile(grp["net_r"], 95)),
            "median_bars_held_DE": float(grp["bars_held_DE"].median())
                if "bars_held_DE" in grp.columns else float("nan"),
            "median_mfe_atr_DE": float(grp["mfe_atr_DE"].median())
                if "mfe_atr_DE" in grp.columns else float("nan"),
            "median_mae_atr_DE": float(grp["mae_atr_DE"].median())
                if "mae_atr_DE" in grp.columns else float("nan"),
            "frame": "delayed_entry_indexed_post_mechanism",
        }
    rows = [_agg(win, "winners"), _agg(los, "losers")]
    return pd.DataFrame(rows)


def _step2_conditional(sub_features: pd.DataFrame, by_col: str,
                       sort_key: str = "category") -> pd.DataFrame:
    """§5.9 conditional breakdown: net_r marginal by stratum. Adds
    sample-size flags per op spec §5.9 (n<30 flagged, n<10 pooled into
    'insufficient_n').
    """
    if by_col not in sub_features.columns or "net_r" not in sub_features.columns:
        return pd.DataFrame()
    rows = []
    pooled_below_10 = []
    for cat, grp in sub_features.groupby(by_col, sort=True):
        n = int(len(grp))
        if n < 10:
            pooled_below_10.append((str(cat), n, grp["net_r"]))
            continue
        rows.append({
            "category": str(cat),
            "n": n,
            "flag_n_lt_30": bool(n < 30),
            "mean_net_r": float(grp["net_r"].mean()),
            "median_net_r": float(grp["net_r"].median()),
            "p25_net_r": float(np.percentile(grp["net_r"], 25)),
            "p75_net_r": float(np.percentile(grp["net_r"], 75)),
            "win_pct": float((grp["net_r"] > 0).mean()),
        })
    if pooled_below_10:
        n_pool = sum(n for _, n, _ in pooled_below_10)
        all_r = pd.concat([r for _, _, r in pooled_below_10])
        rows.append({
            "category": "insufficient_n_pooled",
            "n": n_pool,
            "flag_n_lt_30": True,
            "mean_net_r": float(all_r.mean()) if n_pool else float("nan"),
            "median_net_r": float(all_r.median()) if n_pool else float("nan"),
            "p25_net_r": float(np.percentile(all_r, 25)) if n_pool else float("nan"),
            "p75_net_r": float(np.percentile(all_r, 75)) if n_pool else float("nan"),
            "win_pct": float((all_r > 0).mean()) if n_pool else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(sort_key, kind="stable").reset_index(drop=True)


def _step2_exit_reason(sub_features: pd.DataFrame) -> pd.DataFrame:
    if "exit_reason" not in sub_features.columns:
        return pd.DataFrame()
    vc = sub_features["exit_reason"].fillna("__nan__").value_counts().sort_index()
    total = int(vc.sum())
    return pd.DataFrame({
        "exit_reason": vc.index.astype(str).values,
        "count": vc.values.astype(int),
        "fraction": (vc.values.astype(float) / total) if total else vc.values,
    })


def _build_post_mechanism_frame(post_mech_trades: pd.DataFrame,
                                  bar_by_bar: pd.DataFrame,
                                  signals: pd.DataFrame) -> pd.DataFrame:
    """Build per-trade frame with POST-MECHANISM economics + verbatim
    informational features.

    Post-mechanism columns (delayed-entry-indexed, t* = 1, mirror CC-1):
      net_r, gross_r, spread_cost_r (from trades_post_mechanism CSV)
      mfe_atr_DE = max(max_held_R) × 2  (R → ATR units)
      mae_atr_DE = abs(min(min_held_R)) × 2
      bars_held_DE = action_bar - T_STAR + 1 = action_bar (T_STAR=1)
      exit_reason: post-mechanism (sl_hit_post_delayed_entry / time_exit_delayed)

    Verbatim columns merged from signals_features.csv (informational):
      pair, fold_id, session, hour_utc, day_of_week, hour_in_4h_bar, hour_in_d1_bar
      peak_to_final_r_ratio, mfe_to_mae_ratio_held, r_given_back_from_peak,
      time_to_peak_mfe, time_to_trough_mae, oscillation_count, monotonicity_ratio,
      All forward-geometry features (fwd_mfe_h*, fwd_mae_h*, race_*, reached_*)
      All signal-time predictors (atr_*, cum_logret_*, etc.)
    """
    pm = post_mech_trades.copy()
    # Drop `pair` and `fold` (both present in trades_post_mechanism CSV) so
    # they don't collide with `pair` and `fold_id` brought in via the
    # signals_features merge below. Both sources agree on trade-level
    # identity by construction (same trade_id), so dropping is safe.
    pm = pm.drop(columns=[c for c in ("pair", "fold") if c in pm.columns])

    # Derive delayed-entry-indexed MFE_DE / MAE_DE / bars_held_DE from
    # the bar-by-bar held R sequences (R units = SL multiples = 2 ATR).
    per_trade = bar_by_bar.groupby("trade_id").agg(
        max_held_R_max=("max_held_R", "max"),
        min_held_R_min=("min_held_R", "min"),
        bar_offset_max=("bar_offset", "max"),
        bar_offset_min=("bar_offset", "min"),
    ).reset_index()
    per_trade["mfe_atr_DE"] = per_trade["max_held_R_max"] * 2.0
    per_trade["mae_atr_DE"] = per_trade["min_held_R_min"].abs() * 2.0
    per_trade["bars_held_DE"] = (
        per_trade["bar_offset_max"] - per_trade["bar_offset_min"] + 1
    ).astype(int)

    pm = pm.merge(
        per_trade[["trade_id", "mfe_atr_DE", "mae_atr_DE", "bars_held_DE"]],
        on="trade_id", how="left",
    )

    # Merge verbatim-frame columns from signals_features
    verbatim_cols = [
        "trade_id",
        "pair", "fold_id", "session",
        "hour_utc", "day_of_week", "hour_in_4h_bar", "hour_in_d1_bar",
    ] + list(VERBATIM_INFORMATIONAL_METRICS) + list(STEP2_FORWARD_METRICS) \
        + list(PREDICTOR_SCAN_FEATURES) \
        + list(C3.CLUSTER_FEATURES_NUMERIC) + list(C3.CLUSTER_FEATURES_ORDINAL.keys())
    verbatim_cols = list(dict.fromkeys(verbatim_cols))  # dedup, preserve order
    verbatim_cols = [c for c in verbatim_cols if c in signals.columns]
    sig_sub = signals[verbatim_cols].copy()
    pm = pm.merge(sig_sub, on="trade_id", how="left")

    return pm


def _emit_step2_subset(pm_frame: pd.DataFrame, out_dir: Path) -> dict:
    """Write step 2 focused subset for one population. `pm_frame` is the
    merged frame from `_build_post_mechanism_frame` — post-mechanism
    economics columns + verbatim informational features.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    marg_pm = _step2_marginals_table(
        pm_frame, POST_MECHANISM_METRICS, "delayed_entry_indexed_post_mechanism"
    )
    marg_verb = _step2_marginals_table(
        pm_frame, VERBATIM_INFORMATIONAL_METRICS, "verbatim_step2_frame_informational"
    )
    marg = pd.concat([marg_pm, marg_verb], ignore_index=True)

    fwd = _step2_marginals_table(
        pm_frame, STEP2_FORWARD_METRICS, "verbatim_step2_frame_informational"
    )
    seq = _step2_sequence_class_counts(pm_frame)
    surv = _step2_survival_curve_DE(pm_frame)
    asym = _step2_asymmetry_DE(pm_frame)
    by_pair = _step2_conditional(pm_frame, "pair", sort_key="category")
    by_fold = _step2_conditional(pm_frame, "fold_id", sort_key="category")
    by_session = _step2_conditional(pm_frame, "session", sort_key="category")
    exit_r = _step2_exit_reason(pm_frame)

    for name, df in [
        ("marginals.csv", marg),
        ("forward_geometry.csv", fwd),
        ("sequence_class.csv", seq),
        ("survival.csv", surv),
        ("asymmetry.csv", asym),
        ("conditional_by_pair_net_r.csv", by_pair),
        ("conditional_by_fold_net_r.csv", by_fold),
        ("conditional_by_session_net_r.csv", by_session),
        ("exit_reason_counts.csv", exit_r),
    ]:
        p = out_dir / name
        df.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
        files[name] = _sha256_file(p)
    return files


# ============================================================
# Step 3 focused subset
# ============================================================

def _step3_cluster_and_describe(sub_features: pd.DataFrame, pool_features: pd.DataFrame,
                                  out_dir: Path) -> dict:
    """Cluster discovery K=2 + per-cluster effect sizes + univariate
    predictor scan (against original step 3 K2 label). Returns sha256 map.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    # Build clustering feature matrix on the subset, using step 3's exact builder.
    X_sub, feat_names, keep_mask_sub = D3.build_cluster_features(
        sub_features, return_drop_mask=True
    )
    X_sub_used = X_sub[keep_mask_sub]
    sub_kept = sub_features.iloc[np.flatnonzero(keep_mask_sub)].reset_index(drop=True)

    # Fit K=2 kmeans with same seed as step 3 (BASE_SEED + 2 = 1236).
    seed = C3.BASE_SEED + 2
    labels_sub = CL.fit_kmeans(X_sub_used, k=2, seed=seed)

    # ---- cluster_assignments_new.csv
    ca_new = pd.DataFrame({
        "trade_id": sub_kept["trade_id"].astype(int).values,
        "fold": sub_kept["fold_id"].astype(int).values if "fold_id" in sub_kept.columns else -1,
        "new_K2_kmeans": labels_sub.astype(int),
    }).sort_values("trade_id").reset_index(drop=True)
    p = out_dir / "cluster_assignments_new.csv"
    ca_new.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
    files["cluster_assignments_new.csv"] = _sha256_file(p)

    # ---- cluster_summary.csv
    n_used = len(labels_sub)
    rows = []
    for cid in (0, 1):
        mask = labels_sub == cid
        n = int(mask.sum())
        rows.append({
            "cluster_id": cid,
            "n": n,
            "fraction_of_subset_kept": (n / n_used) if n_used else float("nan"),
        })
    rows.append({
        "cluster_id": "dropped_nan",
        "n": int((~keep_mask_sub).sum()),
        "fraction_of_subset_kept": float("nan"),
    })
    sil = float("nan")
    if n_used >= 4 and len(set(labels_sub)) > 1:
        try:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(X_sub_used, labels_sub))
        except Exception:
            sil = float("nan")
    df_sum = pd.DataFrame(rows)
    df_sum["silhouette_overall"] = sil
    p = out_dir / "cluster_summary.csv"
    df_sum.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
    files["cluster_summary.csv"] = _sha256_file(p)

    # ---- per-cluster distributions (net_r, mfe, mae, bars_held + forward geom)
    dist_rows = []
    sub_kept2 = sub_kept.copy()
    sub_kept2["__new_K2"] = labels_sub
    for cid in (0, 1):
        grp = sub_kept2[sub_kept2["__new_K2"] == cid]
        for metric in ("net_r", "mfe_held_atr", "mae_held_atr", "bars_held",
                       "fwd_mfe_h24_atr", "fwd_mae_h24_atr",
                       "fwd_mfe_h120_atr", "fwd_mae_h120_atr"):
            if metric not in grp.columns:
                continue
            stats = _marginal_stats(grp[metric])
            dist_rows.append({"cluster_id": cid, "metric": metric, **stats})
    df_dist = pd.DataFrame(dist_rows)
    p = out_dir / "cluster_distributions.csv"
    df_dist.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
    files["cluster_distributions.csv"] = _sha256_file(p)

    # ---- cluster_effect_sizes.csv per §6.5 (vs pool baseline)
    pool_used = pool_features.dropna(subset=list(C3.CLUSTER_FEATURES_NUMERIC) +
                                       list(C3.CLUSTER_FEATURES_ORDINAL.keys()))
    es_metrics = {
        "fwd_mfe_h24_atr": pool_used["fwd_mfe_h24_atr"].median(),
        "fwd_mfe_h120_atr": pool_used["fwd_mfe_h120_atr"].median(),
        "fwd_mfe_to_mae_ratio_h24": pool_used["fwd_mfe_to_mae_ratio_h24"].median(),
        "race_bars_plus1_minus_minus1": pool_used["race_bars_plus1_minus_minus1"].median(),
    }
    pool_p1atr = pool_used["reached_plus_1.0_atr_within_240"].astype(float).mean() \
        if "reached_plus_1.0_atr_within_240" in pool_used.columns else float("nan")
    pool_std_mfe24 = float(pool_used["fwd_mfe_h24_atr"].std(ddof=1))

    es_rows = []
    for cid in (0, 1):
        grp = sub_kept2[sub_kept2["__new_K2"] == cid]
        n = int(len(grp))
        row = {
            "cluster_id": cid,
            "n": n,
            "frac_of_subset_kept": (n / n_used) if n_used else float("nan"),
            "pool_median_fwd_mfe_h24_atr": float(es_metrics["fwd_mfe_h24_atr"]),
            "cluster_median_fwd_mfe_h24_atr":
                float(grp["fwd_mfe_h24_atr"].median()) if "fwd_mfe_h24_atr" in grp.columns
                else float("nan"),
        }
        row["delta_median_fwd_mfe_h24_atr"] = (
            row["cluster_median_fwd_mfe_h24_atr"] - row["pool_median_fwd_mfe_h24_atr"]
        )
        row["delta_in_pool_stds"] = (
            row["delta_median_fwd_mfe_h24_atr"] / pool_std_mfe24
            if pool_std_mfe24 > 0 else float("nan")
        )
        row["pool_median_fwd_mfe_h120_atr"] = float(es_metrics["fwd_mfe_h120_atr"])
        row["cluster_median_fwd_mfe_h120_atr"] = (
            float(grp["fwd_mfe_h120_atr"].median())
            if "fwd_mfe_h120_atr" in grp.columns else float("nan")
        )
        row["delta_median_fwd_mfe_h120_atr"] = (
            row["cluster_median_fwd_mfe_h120_atr"] - row["pool_median_fwd_mfe_h120_atr"]
        )
        row["pool_median_fwd_mfe_to_mae_ratio_h24"] = float(es_metrics["fwd_mfe_to_mae_ratio_h24"])
        row["cluster_median_fwd_mfe_to_mae_ratio_h24"] = (
            float(grp["fwd_mfe_to_mae_ratio_h24"].median())
            if "fwd_mfe_to_mae_ratio_h24" in grp.columns else float("nan")
        )
        row["delta_median_fwd_mfe_to_mae_ratio_h24"] = (
            row["cluster_median_fwd_mfe_to_mae_ratio_h24"]
            - row["pool_median_fwd_mfe_to_mae_ratio_h24"]
        )
        row["pool_race_p50"] = float(es_metrics["race_bars_plus1_minus_minus1"])
        row["cluster_race_p50"] = (
            float(grp["race_bars_plus1_minus_minus1"].median())
            if "race_bars_plus1_minus_minus1" in grp.columns else float("nan")
        )
        row["delta_race_p50"] = row["cluster_race_p50"] - row["pool_race_p50"]
        row["pool_p_reach_plus1atr_240"] = float(pool_p1atr)
        row["cluster_p_reach_plus1atr_240"] = (
            float(grp["reached_plus_1.0_atr_within_240"].astype(float).mean())
            if "reached_plus_1.0_atr_within_240" in grp.columns else float("nan")
        )
        row["delta_p_reach_plus1atr_240"] = (
            row["cluster_p_reach_plus1atr_240"] - row["pool_p_reach_plus1atr_240"]
        )
        # Pass flags per §8 thresholds
        row["pass_delta_mfe24"] = bool(
            abs(row["delta_median_fwd_mfe_h24_atr"])
                >= C3.ES_THRESHOLDS["delta_median_fwd_mfe_h24"]
            or abs(row["delta_in_pool_stds"])
                >= C3.ES_THRESHOLDS["delta_median_fwd_mfe_h24_stdfrac"]
        )
        row["pass_delta_ratio"] = bool(
            abs(row["delta_median_fwd_mfe_to_mae_ratio_h24"])
                >= C3.ES_THRESHOLDS["delta_median_fwd_mfe_to_mae_ratio_h24"]
        )
        row["pass_delta_race"] = bool(
            abs(row["delta_race_p50"])
                >= C3.ES_THRESHOLDS["delta_race_condition_median"]
        )
        row["pass_delta_p1atr"] = bool(
            abs(row["delta_p_reach_plus1atr_240"])
                >= C3.ES_THRESHOLDS["delta_p_reach_plus1atr_240"]
        )
        row["pass_any"] = bool(
            row["pass_delta_mfe24"] or row["pass_delta_ratio"]
            or row["pass_delta_race"] or row["pass_delta_p1atr"]
        )
        es_rows.append(row)
    df_es = pd.DataFrame(es_rows)
    p = out_dir / "cluster_effect_sizes.csv"
    df_es.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
    files["cluster_effect_sizes.csv"] = _sha256_file(p)

    # ---- predictor_scan_univariate.csv: univariate AUC of each
    # PREDICTOR_SCAN_FEATURES feature against new K2 label.
    # Discriminate cluster 0 (positive) vs cluster 1.
    y = (labels_sub == 0).astype(int)
    pred_rows = []
    if len(set(y)) >= 2 and n_used >= 4:
        for feat in PREDICTOR_SCAN_FEATURES:
            if feat not in sub_kept.columns:
                continue
            x = pd.to_numeric(sub_kept[feat], errors="coerce").values
            mask = ~np.isnan(x)
            if mask.sum() < 4 or len(set(y[mask])) < 2:
                continue
            try:
                auc_raw = float(roc_auc_score(y[mask], x[mask]))
            except Exception:
                continue
            auc_sym = max(auc_raw, 1.0 - auc_raw)
            direction = "+x_predicts_cluster_0" if auc_raw >= 0.5 else "-x_predicts_cluster_0"
            pred_rows.append({
                "feature": feat,
                "n_non_nan": int(mask.sum()),
                "auc_raw": auc_raw,
                "auc_symmetric": auc_sym,
                "direction": direction,
            })
    df_pred = pd.DataFrame(pred_rows).sort_values(
        ["auc_symmetric", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    p = out_dir / "predictor_scan_univariate.csv"
    df_pred.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
    files["predictor_scan_univariate.csv"] = _sha256_file(p)

    return {"files": files, "labels_sub": labels_sub, "n_used": n_used,
            "keep_mask_sub": keep_mask_sub, "silhouette": sil}


# ============================================================
# Task F — filter-intent check (lookup-based)
# ============================================================

def _filter_intent_check(f2_f5_ids: np.ndarray, f6_f7_ids: np.ndarray,
                          cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    """Lookup-based projection onto original step 3 K2_kmeans labels.
    Reports cluster shares + shifts vs pool baseline.
    """
    ca = cluster_assignments[["trade_id", "K2_kmeans"]].copy()

    pool_total = int(len(ca))
    pool_non_sentinel = int((ca["K2_kmeans"] != -2).sum())
    pool_c0 = int((ca["K2_kmeans"] == 0).sum())
    pool_c1 = int((ca["K2_kmeans"] == 1).sum())
    pool_sent = int((ca["K2_kmeans"] == -2).sum())

    def _check(set_label: str, ids: np.ndarray):
        m = ca[ca["trade_id"].isin(ids)]
        n = int(len(m))
        c0 = int((m["K2_kmeans"] == 0).sum())
        c1 = int((m["K2_kmeans"] == 1).sum())
        sent = int((m["K2_kmeans"] == -2).sum())
        c0_share = (c0 / n) if n else float("nan")
        c1_share = (c1 / n) if n else float("nan")
        pool_c0_share_full = pool_c0 / pool_total
        pool_c0_share_nonsent = pool_c0 / pool_non_sentinel if pool_non_sentinel else float("nan")
        pool_c1_share_full = pool_c1 / pool_total
        pool_c1_share_nonsent = pool_c1 / pool_non_sentinel if pool_non_sentinel else float("nan")
        return [
            {
                "set": set_label,
                "cluster_id": 0,
                "n_in_set": n,
                "n_in_cluster": c0,
                "post_mechanism_share": c0_share,
                "pool_share_full_denom": pool_c0_share_full,
                "pool_share_non_sentinel_denom": pool_c0_share_nonsent,
                "absolute_shift_vs_pool_full": c0_share - pool_c0_share_full,
                "absolute_shift_vs_pool_non_sentinel":
                    c0_share - pool_c0_share_nonsent if pool_non_sentinel else float("nan"),
                "ratio_shift_vs_pool_full":
                    c0_share / pool_c0_share_full if pool_c0_share_full else float("nan"),
                "ratio_shift_vs_pool_non_sentinel":
                    c0_share / pool_c0_share_nonsent
                    if pool_c0_share_nonsent else float("nan"),
                "n_sentinel_in_set": sent,
            },
            {
                "set": set_label,
                "cluster_id": 1,
                "n_in_set": n,
                "n_in_cluster": c1,
                "post_mechanism_share": c1_share,
                "pool_share_full_denom": pool_c1_share_full,
                "pool_share_non_sentinel_denom": pool_c1_share_nonsent,
                "absolute_shift_vs_pool_full": c1_share - pool_c1_share_full,
                "absolute_shift_vs_pool_non_sentinel":
                    c1_share - pool_c1_share_nonsent if pool_non_sentinel else float("nan"),
                "ratio_shift_vs_pool_full":
                    c1_share / pool_c1_share_full if pool_c1_share_full else float("nan"),
                "ratio_shift_vs_pool_non_sentinel":
                    c1_share / pool_c1_share_nonsent
                    if pool_c1_share_nonsent else float("nan"),
                "n_sentinel_in_set": sent,
            },
        ]

    rows = []
    rows.extend(_check("F6+F7_post_mechanism", f6_f7_ids))
    rows.extend(_check("F2-F5_post_mechanism", f2_f5_ids))
    # Pool baseline row for reference
    rows.append({
        "set": "pool_baseline",
        "cluster_id": 0,
        "n_in_set": pool_total,
        "n_in_cluster": pool_c0,
        "post_mechanism_share": pool_c0 / pool_total,
        "pool_share_full_denom": pool_c0 / pool_total,
        "pool_share_non_sentinel_denom": pool_c0 / pool_non_sentinel,
        "absolute_shift_vs_pool_full": 0.0,
        "absolute_shift_vs_pool_non_sentinel": 0.0,
        "ratio_shift_vs_pool_full": 1.0,
        "ratio_shift_vs_pool_non_sentinel": 1.0,
        "n_sentinel_in_set": pool_sent,
    })
    rows.append({
        "set": "pool_baseline",
        "cluster_id": 1,
        "n_in_set": pool_total,
        "n_in_cluster": pool_c1,
        "post_mechanism_share": pool_c1 / pool_total,
        "pool_share_full_denom": pool_c1 / pool_total,
        "pool_share_non_sentinel_denom": pool_c1 / pool_non_sentinel,
        "absolute_shift_vs_pool_full": 0.0,
        "absolute_shift_vs_pool_non_sentinel": 0.0,
        "ratio_shift_vs_pool_full": 1.0,
        "ratio_shift_vs_pool_non_sentinel": 1.0,
        "n_sentinel_in_set": pool_sent,
    })
    return pd.DataFrame(rows)


# ============================================================
# Task G — sub-archetype discovery
# ============================================================

def _subarchetype_discovery(sub_kept: pd.DataFrame, new_labels: np.ndarray,
                             original_assignments: pd.DataFrame) -> pd.DataFrame:
    """For F2-F5 (kept post-mechanism trades): overlap matrix between
    new K=2 clusters and original step 3 K2_kmeans labels (lookup).
    """
    sub = sub_kept[["trade_id"]].copy()
    sub["new_K2"] = new_labels.astype(int)
    merged = sub.merge(
        original_assignments[["trade_id", "K2_kmeans"]].rename(
            columns={"K2_kmeans": "original_K2"}
        ),
        on="trade_id", how="left",
    )
    rows = []
    for new_c in (0, 1):
        for orig_c in (0, 1, -2):
            m = merged[(merged["new_K2"] == new_c) & (merged["original_K2"] == orig_c)]
            rows.append({
                "new_cluster_id": new_c,
                "original_K2_kmeans": orig_c,
                "n_overlap": int(len(m)),
            })
    cross = pd.DataFrame(rows)
    # Also: ARI between new and original (on common rows where both are 0/1)
    from sklearn.metrics import adjusted_rand_score
    valid = merged[merged["original_K2"].isin([0, 1])].copy()
    if len(valid) >= 4:
        ari = float(adjusted_rand_score(valid["original_K2"], valid["new_K2"]))
    else:
        ari = float("nan")
    cross["adjusted_rand_index_new_vs_original"] = ari
    return cross


# ============================================================
# Main
# ============================================================

def _load_inputs():
    # Use D3.load_signals() so derived columns (e.g.
    # reached_plus_1.0_atr_within_240) are present, matching step 3's
    # internal frame exactly.
    signals = D3.load_signals()
    cluster_assignments = pd.read_csv(STEP3_POOL_DIR / "cluster_assignments.csv")
    f2_f5_trades = pd.read_csv(STEP5_OUT / "trades_post_mechanism_f2_f5.csv")
    f6_f7_trades = pd.read_csv(STEP4_OUT / "trades_post_mechanism.csv")
    return signals, cluster_assignments, f2_f5_trades, f6_f7_trades


def main() -> dict:
    STEP5_OUT.mkdir(parents=True, exist_ok=True)

    source_pin = _verify_source_pin()
    data_pin = _verify_data_pins()
    if not data_pin["all_data_sha256_match"]:
        raise RuntimeError(
            f"Data sha256 pin drift:\n{json.dumps(data_pin['per_file'], indent=2)}"
        )

    signals, cluster_assignments, f2_f5_trades, f6_f7_trades = _load_inputs()

    f2_f5_ids = f2_f5_trades["trade_id"].astype(int).values
    f6_f7_ids = f6_f7_trades["trade_id"].astype(int).values

    sub_f2_f5 = signals[signals["trade_id"].isin(f2_f5_ids)].reset_index(drop=True)
    sub_f6_f7 = signals[signals["trade_id"].isin(f6_f7_ids)].reset_index(drop=True)

    # Load CC-1 / CC Task A bar-by-bar derivations (delayed-entry-indexed)
    bbb_winners = pd.read_csv(STEP5_OUT / "winner_paths_bar_by_bar.csv")
    bbb_losers = pd.read_csv(STEP5_OUT / "loser_paths_bar_by_bar.csv")
    bbb_all = pd.concat([bbb_winners, bbb_losers], ignore_index=True)

    # Build post-mechanism frames for step 2 (one per population)
    pm_f2_f5 = _build_post_mechanism_frame(f2_f5_trades, bbb_all, signals)
    pm_f6_f7 = _build_post_mechanism_frame(f6_f7_trades, bbb_all, signals)

    # Step 2 focused subset — uses POST-MECHANISM economics + verbatim
    # informational features. Path geometry (MFE, MAE, bars_held) is
    # delayed-entry-indexed per CC-1 / CC Task A conventions.
    step2_dir_f6_f7 = STEP5_OUT / "step2_recharacterisation_f6_f7"
    step2_dir_f2_f5 = STEP5_OUT / "step2_recharacterisation_f2_f5"
    files_step2_f6_f7 = _emit_step2_subset(pm_f6_f7, step2_dir_f6_f7)
    files_step2_f2_f5 = _emit_step2_subset(pm_f2_f5, step2_dir_f2_f5)

    # Step 3 focused subset
    step3_dir_f6_f7 = STEP5_OUT / "step3_recharacterisation_f6_f7"
    step3_dir_f2_f5 = STEP5_OUT / "step3_recharacterisation_f2_f5"
    step3_out_f6_f7 = _step3_cluster_and_describe(sub_f6_f7, signals, step3_dir_f6_f7)
    step3_out_f2_f5 = _step3_cluster_and_describe(sub_f2_f5, signals, step3_dir_f2_f5)

    # Task F filter-intent
    fic = _filter_intent_check(f2_f5_ids, f6_f7_ids, cluster_assignments)
    p = STEP5_OUT / "filter_intent_check.csv"
    fic.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
    fic_sha = _sha256_file(p)

    # Task G sub-archetype on F2-F5
    sub_kept_f2_f5 = sub_f2_f5.iloc[
        np.flatnonzero(step3_out_f2_f5["keep_mask_sub"])
    ].reset_index(drop=True)
    sub_arch = _subarchetype_discovery(
        sub_kept_f2_f5, step3_out_f2_f5["labels_sub"], cluster_assignments
    )
    p = STEP5_OUT / "subarchetype_discovery.csv"
    sub_arch.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
    sub_arch_sha = _sha256_file(p)

    # Manifest
    manifest = {
        "step": "l_arc_2/step5_recharacterisation/delayed_entry_t_gb/CC_taskB",
        "dispatch":
            "§9 step 5 re-characterisation (focused subset, planner-authorised "
            "scope alternative to literal full-template re-run)",
        "git_commit_head_at_run": _git_commit_head(),
        "predictor_source_pin_verification": source_pin,
        "data_sha256_pin_verification": data_pin,
        "versions": _versions(),
        "scope_note": (
            "Step 2 / step 3 orchestrators hardcode the full pool as input "
            "and the canonical pool output dirs as destination, and CANNOT "
            "be re-invoked on a custom trade-set without significant patches "
            "(would overwrite pool baselines). Per planner authorisation, "
            "this script delivers a FOCUSED SUBSET that mirrors op spec "
            "§5.1, §5.2, §5.4, §5.6, §5.8, §5.9 and §6.1, §6.5, §6.3 "
            "(univariate) using step 3's library functions for methodology "
            "faithfulness. Op spec §11.5 (no recommendations in step 2/3 "
            "reports) is honoured. Filter-intent (Task F) is lookup-based "
            "against cluster_assignments.csv — no kmeans refit, byte-faithful "
            "to original step 3 K2_kmeans assignments."
        ),
        "n_counts": {
            "pool_total": int(len(signals)),
            "f2_f5_post_mechanism": int(len(f2_f5_ids)),
            "f6_f7_post_mechanism": int(len(f6_f7_ids)),
            "f2_f5_kept_for_clustering": int(step3_out_f2_f5["n_used"]),
            "f6_f7_kept_for_clustering": int(step3_out_f6_f7["n_used"]),
        },
        "step3_silhouette": {
            "f2_f5_K2_new": step3_out_f2_f5["silhouette"],
            "f6_f7_K2_new": step3_out_f6_f7["silhouette"],
        },
        "inputs": {
            "signals_features.csv": {
                "path": "results/l_arc_2/step2_descriptive/signals_features.csv",
                "sha256": _sha256_file(C3.SIGNALS_CSV),
            },
            "cluster_assignments.csv": {
                "path": "results/l_arc_2/step3_extractability/cluster_assignments.csv",
                "sha256": _sha256_file(STEP3_POOL_DIR / "cluster_assignments.csv"),
            },
            "trades_post_mechanism_f2_f5.csv": {
                "path":
                    "results/l_arc_2/step5_recharacterisation/delayed_entry_t_gb/trades_post_mechanism_f2_f5.csv",
                "sha256": _sha256_file(STEP5_OUT / "trades_post_mechanism_f2_f5.csv"),
            },
            "trades_post_mechanism.csv": {
                "path":
                    "results/l_arc_2/step4/delayed_entry_t_gb/trades_post_mechanism.csv",
                "sha256": _sha256_file(STEP4_OUT / "trades_post_mechanism.csv"),
            },
            "winner_paths_bar_by_bar.csv": {
                "path":
                    "results/l_arc_2/step5_recharacterisation/delayed_entry_t_gb/winner_paths_bar_by_bar.csv",
                "sha256": _sha256_file(STEP5_OUT / "winner_paths_bar_by_bar.csv"),
            },
            "loser_paths_bar_by_bar.csv": {
                "path":
                    "results/l_arc_2/step5_recharacterisation/delayed_entry_t_gb/loser_paths_bar_by_bar.csv",
                "sha256": _sha256_file(STEP5_OUT / "loser_paths_bar_by_bar.csv"),
            },
        },
        "outputs_sha256": {
            **{f"step2_recharacterisation_f6_f7/{k}": v
                for k, v in files_step2_f6_f7.items()},
            **{f"step2_recharacterisation_f2_f5/{k}": v
                for k, v in files_step2_f2_f5.items()},
            **{f"step3_recharacterisation_f6_f7/{k}": v
                for k, v in step3_out_f6_f7["files"].items()},
            **{f"step3_recharacterisation_f2_f5/{k}": v
                for k, v in step3_out_f2_f5["files"].items()},
            "filter_intent_check.csv": fic_sha,
            "subarchetype_discovery.csv": sub_arch_sha,
        },
    }
    manifest_path = STEP5_OUT / "cc_taskB_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    manifest["outputs_sha256"]["cc_taskB_run_manifest.json"] = _sha256_file(manifest_path)
    return manifest


# ============================================================
# Determinism + lookahead
# ============================================================

def _all_output_paths() -> list[Path]:
    paths = []
    for sub in ("step2_recharacterisation_f6_f7", "step2_recharacterisation_f2_f5",
                "step3_recharacterisation_f6_f7", "step3_recharacterisation_f2_f5"):
        d = STEP5_OUT / sub
        if d.exists():
            for p in sorted(d.glob("*.csv")):
                paths.append(p)
    for fname in ("filter_intent_check.csv", "subarchetype_discovery.csv",
                  "cc_taskB_run_manifest.json"):
        p = STEP5_OUT / fname
        if p.exists():
            paths.append(p)
    return paths


def _snapshot_shas() -> dict:
    return {str(p.relative_to(STEP5_OUT)): _sha256_file(p) for p in _all_output_paths()}


def run_full_dispatch() -> dict:
    main()
    run1 = _snapshot_shas()
    main()
    run2 = _snapshot_shas()
    determinism_per_file = {}
    det_pass = True
    for k, sha1 in run1.items():
        sha2 = run2.get(k)
        match = sha1 == sha2
        if not match:
            det_pass = False
        determinism_per_file[k] = {"run1_sha256": sha1, "run2_sha256": sha2, "match": match}

    # Task I lookahead: N/A by construction — this script reads only
    # pre-computed signals_features.csv columns and original cluster
    # assignments; no per-bar OHLC recomputation. Recorded as N/A.
    lookahead_receipt = {
        "applicable": False,
        "reason": (
            "Task B/C use pre-computed signals_features.csv columns "
            "(forward-geometry, marginals — all at-signal or post-trade "
            "aggregates computed once in step 2 with lookahead test PASS at "
            "step 2 closure). Task D/E cluster on op-spec 12 features that "
            "are similarly pre-computed in step 2 (Amendment 4 forward "
            "features included). Task F is a pure lookup against "
            "cluster_assignments.csv. No per-bar OHLC perturbation surface "
            "in this dispatch — inherited PASS from step 2's lookahead "
            "test stands."
        ),
        "step2_lookahead_status_at_closure":
            "PASS (per results/l_arc_2/step2_descriptive/feature_lag_audit.txt; "
            "see step 2 phase doc)",
    }

    receipts = {
        "step":
            "l_arc_2/step5_recharacterisation/delayed_entry_t_gb/CC_taskB_validation",
        "git_commit_head_at_run": _git_commit_head(),
        "task_h_determinism": {"pass": det_pass, "per_file": determinism_per_file},
        "task_i_lookahead_invariant": lookahead_receipt,
    }
    receipts_path = STEP5_OUT / "cc_taskB_validation_receipts.json"
    with open(receipts_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(receipts, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    if not det_pass:
        for p in _all_output_paths():
            if p.exists() and p.name != "cc_taskB_validation_receipts.json":
                p.unlink()
        raise RuntimeError(
            "Task H determinism FAILED — outputs removed.\n"
            f"{json.dumps(receipts, indent=2, default=str)}"
        )
    return receipts


if __name__ == "__main__":
    receipts = run_full_dispatch()
    print("CC Task B summary:")
    print(json.dumps({
        "determinism_pass": receipts["task_h_determinism"]["pass"],
        "lookahead_status": receipts["task_i_lookahead_invariant"]["applicable"],
        "n_output_files":
            len(receipts["task_h_determinism"]["per_file"]),
    }, indent=2, default=str))
