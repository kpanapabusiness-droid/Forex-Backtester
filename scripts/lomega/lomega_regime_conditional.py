"""Lω regime-conditional dispatch — three binning variants, 1H + 4H.

Tests post-mortem (PR #134) interpretation 1 (regime-conditional non-stationarity).
For each (timeframe × binning_variant × bin × fold), fits the same top-15 RF as
B1-B4 on bin members only; reports per-bin worst-fold AUC against the gate
(worst-fold AUC ≥ 0.55 with ≥ 1000 training samples per fold).

Three binning variants per TF:
  A — vol-percentile (5 quintiles on atr_percentile_60, full-data quintile edges)
  B — D1 trend sign (d1_up / d1_flat / d1_down on d1_ema_50_slope_at_entry)
  C — year-quarter (calendar bins; drop < 5000 samples)

Hard constraints (per dispatch):
  - reuse B1-B4 features (no new) and labels (binary clean_move)
  - reuse top-15 features pinned in TOP15 below (matches lomega_postmortem)
  - same RF hyperparams (n_estimators=200, max_depth=8, min_samples_leaf=20,
    random_state=42)
  - same 7-fold scheme as B1-B4 (TimeSeriesSplit n_splits=7, anchored expanding)
  - bins frozen on full-data values; no mid-fold redefinition
  - determinism: byte-identical two-run on per-bin AUC tables

Fold-date interpretation note: dispatch says "identical fold dates" as the
hard constraint, but Variant C bins are each concentrated in a single quarter,
which makes the B1-B4 full-timeline fold boundaries produce empty val cells.
Uniform treatment across variants: TimeSeriesSplit(n_splits=7) is applied to
each bin's time-sorted subset (which is what B1-B4 itself uses on its own
time-sorted dataset). Documented in the summary.

Usage:
    py scripts/lomega/lomega_regime_conditional.py --tf 1h
    py scripts/lomega/lomega_regime_conditional.py --tf all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

_REPO = Path(__file__).resolve().parent.parent.parent
_B1B4_DIR = _REPO / "results" / "lomega" / "b1_b4_discovery"
_OUT = _REPO / "results" / "lomega" / "regime_conditional"

TF_LIST = ("1h", "4h")
RNG_SEED = 42
RF_KW = dict(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    random_state=RNG_SEED,
    n_jobs=-1,
)
N_SPLITS = 7
MIN_TRAIN_N = 1000
MIN_VAL_N = 200
GATE_AUC = 0.55
GATE_MIN_TRAIN_N = 1000
VARIANT_C_MIN_SAMPLES = 5000

# Top-15 features per TF — pinned from PR #133's univariate-AUC ranking.
# Matches lomega_postmortem.TOP15. Duplicated here so the script is
# self-contained (scripts/lomega/ is not a Python package).
TOP15 = {
    "1h": [
        "d1_ema_50_slope_at_entry", "atr_14", "atr_50", "atr_200",
        "d1_close_above_ema_50", "realised_vol_20", "ema_200_slope",
        "h4_ema_50_slope_at_entry", "atr_ratio_14_200", "atr_percentile_200",
        "hour_of_week", "ema_50_above_200", "atr_ratio_14_50",
        "day_of_week", "session_marker",
    ],
    "4h": [
        "atr_14", "atr_50", "atr_200", "ema_200_slope", "atr_ratio_14_200",
        "atr_percentile_200", "d1_close_above_ema_50",
        "d1_ema_50_slope_at_entry", "ema_50_above_200",
        "ema_spread_20_50_atr", "ema_50_slope", "atr_ratio_14_50",
        "ema_20_slope", "position_in_range_60", "atr_percentile_60",
    ],
}

# B1-B4 baseline worst-fold AUCs (from PR #134 LOMEGA_POSTMORTEM_SUMMARY.md §1).
B1B4_BASELINE_WORST_AUC = {"1h": 0.4841, "4h": 0.4733}


def load_merged(tf: str) -> pd.DataFrame:
    """Load labels.csv + features.csv joined on (pair, time), sorted."""
    tf_dir = _B1B4_DIR / f"timeframe_{tf}"
    labels_fp = tf_dir / "labels.csv"
    feats_fp = tf_dir / "features.csv"
    if not labels_fp.exists() or not feats_fp.exists():
        raise FileNotFoundError(
            f"Missing labels.csv / features.csv for tf={tf}. "
            f"Regenerate via: py scripts/lomega/lomega_b1_b4.py --tf {tf}"
        )
    labels = pd.read_csv(labels_fp, parse_dates=["time"])
    feats = pd.read_csv(feats_fp, parse_dates=["time"])
    merged = feats.merge(labels, on=["pair", "time"], how="inner")
    return merged.sort_values(["time", "pair"], kind="mergesort").reset_index(drop=True)


# ----------------------------------------------------------- bin assignment

def variant_a_vol_quintile(merged: pd.DataFrame) -> tuple[pd.Series, dict]:
    """5 bins on atr_percentile_60 quintiles, full-data quintile edges."""
    col = "atr_percentile_60"
    vals = merged[col].values
    finite = np.isfinite(vals)
    if not finite.any():
        raise RuntimeError(f"Variant A: no finite values in {col}")
    edges = np.quantile(vals[finite], [0.2, 0.4, 0.6, 0.8])
    bins = np.array(["missing"] * len(merged), dtype=object)
    bins[finite & (vals < edges[0])] = "vol_q1"
    bins[finite & (vals >= edges[0]) & (vals < edges[1])] = "vol_q2"
    bins[finite & (vals >= edges[1]) & (vals < edges[2])] = "vol_q3"
    bins[finite & (vals >= edges[2]) & (vals < edges[3])] = "vol_q4"
    bins[finite & (vals >= edges[3])] = "vol_q5"
    meta = {
        "variant": "vol_percentile",
        "feature": col,
        "edges_q20_q40_q60_q80": [float(e) for e in edges],
        "n_finite": int(finite.sum()),
        "n_missing": int((~finite).sum()),
    }
    return pd.Series(bins, index=merged.index, name="bin"), meta


def variant_b_d1_trend(merged: pd.DataFrame) -> tuple[pd.Series, dict]:
    """3 bins on d1_ema_50_slope_at_entry sign with ±0.001 deadband.
    If d1_flat < 5% of finite data, widen to ±0.002.
    """
    col = "d1_ema_50_slope_at_entry"
    vals = merged[col].values
    finite = np.isfinite(vals)
    n_finite = int(finite.sum())
    if n_finite == 0:
        raise RuntimeError(f"Variant B: no finite values in {col}")

    def assign(thresh):
        bins = np.array(["missing"] * len(merged), dtype=object)
        bins[finite & (vals > thresh)] = "d1_up"
        bins[finite & (vals < -thresh)] = "d1_down"
        bins[finite & (vals >= -thresh) & (vals <= thresh)] = "d1_flat"
        return bins

    threshold = 0.001
    bins = assign(threshold)
    flat_share = float((bins == "d1_flat").sum() / n_finite)
    widened = False
    if flat_share < 0.05:
        threshold = 0.002
        bins = assign(threshold)
        flat_share = float((bins == "d1_flat").sum() / n_finite)
        widened = True
    meta = {
        "variant": "d1_trend",
        "feature": col,
        "threshold": threshold,
        "widened_from_0.001_to_0.002": widened,
        "d1_flat_share": flat_share,
        "n_finite": n_finite,
        "n_missing": int((~finite).sum()),
    }
    return pd.Series(bins, index=merged.index, name="bin"), meta


def variant_c_year_quarter(merged: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Bin by YYYY-Qn from bar timestamp. Drop bins with < 5000 samples."""
    period = merged["time"].dt.to_period("Q")
    years = period.dt.year.astype(str)
    quarters = period.dt.quarter.astype(str)
    bins = (years + "-Q" + quarters).values
    counts = pd.Series(bins).value_counts().sort_index()
    keep = set(counts[counts >= VARIANT_C_MIN_SAMPLES].index.tolist())
    final = np.where(np.isin(bins, list(keep)), bins, "insufficient_data")
    dropped = {k: int(v) for k, v in counts.items() if k not in keep}
    meta = {
        "variant": "year_quarter",
        "min_samples_threshold": VARIANT_C_MIN_SAMPLES,
        "bins_kept": sorted(keep),
        "n_bins_kept": len(keep),
        "bins_dropped": dropped,
        "n_bins_dropped": len(dropped),
    }
    return pd.Series(final, index=merged.index, name="bin"), meta


# ----------------------------------------------------------- training per bin

def run_one_bin(merged_bin: pd.DataFrame, feat_cols: list[str]) -> list[dict]:
    """Fit RF with 7-fold anchored expanding on bin-only rows.

    Returns one row per fold with train_n, val_n, val_auc, base_rate, status.
    """
    X_raw = merged_bin[feat_cols].values
    finite_mask = np.all(np.isfinite(X_raw), axis=1)
    X = X_raw[finite_mask]
    y = merged_bin["clean_move"].astype(int).values[finite_mask]
    times = merged_bin["time"].values[finite_mask]

    if len(X) == 0:
        return []

    order = np.argsort(times, kind="stable")
    X = X[order]
    y = y[order]
    times = times[order]

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []
    for fold_id, (tr, va) in enumerate(tss.split(X), start=1):
        train_n = int(len(tr))
        val_n = int(len(va))
        base_rate_val = float(np.mean(y[va])) if val_n else float("nan")
        row = {
            "fold": fold_id,
            "train_start": str(pd.Timestamp(times[tr[0]])) if train_n else "",
            "train_end": str(pd.Timestamp(times[tr[-1]])) if train_n else "",
            "val_start": str(pd.Timestamp(times[va[0]])) if val_n else "",
            "val_end": str(pd.Timestamp(times[va[-1]])) if val_n else "",
            "train_n": train_n,
            "val_n": val_n,
            "base_rate_val": base_rate_val,
            "val_auc": float("nan"),
            "status": "pending",
        }
        if train_n < MIN_TRAIN_N or val_n < MIN_VAL_N:
            row["status"] = "insufficient_data"
            rows.append(row)
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            row["status"] = "degenerate_y"
            rows.append(row)
            continue
        rf = RandomForestClassifier(**RF_KW)
        rf.fit(X[tr], y[tr])
        proba = rf.predict_proba(X[va])[:, 1]
        row["val_auc"] = float(roc_auc_score(y[va], proba))
        row["status"] = "trained"
        rows.append(row)
    return rows


def evaluate_bin(per_fold_rows: list[dict]) -> dict:
    """Aggregate per-fold rows into bin-level summary + gate verdict.

    Gate: status == "trained" for all 7 folds AND worst_auc >= 0.55
    AND min_train_n >= 1000.
    Any fold with insufficient samples poisons the whole bin (per dispatch).
    """
    if not per_fold_rows:
        return {
            "status": "no_rows", "n_folds_trained": 0,
            "min_train_n": 0, "min_val_n": 0,
            "mean_auc": float("nan"), "worst_auc": float("nan"),
            "fold_stdev": float("nan"), "mean_base_rate_val": float("nan"),
            "gate_pass": False,
        }
    statuses = [r["status"] for r in per_fold_rows]
    aucs = [r["val_auc"] for r in per_fold_rows if r["status"] == "trained"]
    train_ns = [r["train_n"] for r in per_fold_rows]
    val_ns = [r["val_n"] for r in per_fold_rows]
    base_rates = [
        r["base_rate_val"] for r in per_fold_rows
        if r["val_n"] > 0 and np.isfinite(r["base_rate_val"])
    ]
    if any(s == "insufficient_data" for s in statuses):
        bin_status = "insufficient_data"
    elif any(s == "degenerate_y" for s in statuses):
        bin_status = "degenerate_y"
    elif len(aucs) < N_SPLITS:
        bin_status = "partial_train"
    else:
        bin_status = "trained"
    summary = {
        "status": bin_status,
        "n_folds_trained": len(aucs),
        "min_train_n": int(min(train_ns)),
        "min_val_n": int(min(val_ns)),
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "worst_auc": float(np.min(aucs)) if aucs else float("nan"),
        "fold_stdev": float(np.std(aucs, ddof=0)) if aucs else float("nan"),
        "mean_base_rate_val": float(np.mean(base_rates)) if base_rates else float("nan"),
    }
    summary["gate_pass"] = bool(
        bin_status == "trained"
        and summary["worst_auc"] >= GATE_AUC
        and summary["min_train_n"] >= GATE_MIN_TRAIN_N
    )
    return summary


# ----------------------------------------------------------- driver per variant

def run_variant(
    merged: pd.DataFrame, tf: str, variant_name: str, bin_fn,
    feat_cols: list[str], out_dir: Path,
) -> dict:
    print(f"  [{tf} / {variant_name}] computing bin membership ...", flush=True)
    bin_labels, meta = bin_fn(merged)
    out_dir.mkdir(parents=True, exist_ok=True)

    # bin_membership.csv: pair, time, bin
    pd.DataFrame({
        "pair": merged["pair"].values,
        "time": merged["time"].values,
        "bin": bin_labels.values,
    }).to_csv(out_dir / "bin_membership.csv", index=False)

    # Save bin meta as a sidecar txt (informational; not the headline output)
    with open(out_dir / "bin_definition.txt", "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    bin_names = sorted(
        b for b in pd.unique(bin_labels)
        if b not in ("missing", "insufficient_data")
    )

    per_bin_fold_rows: list[dict] = []
    bin_summaries: list[dict] = []
    for bin_name in bin_names:
        mask = (bin_labels == bin_name).values
        merged_bin = merged.loc[mask]
        n_rows = int(mask.sum())
        print(
            f"  [{tf} / {variant_name} / {bin_name}] rows={n_rows:,}",
            flush=True,
        )
        fold_rows = run_one_bin(merged_bin, feat_cols)
        for r in fold_rows:
            r2 = {"bin": bin_name, "n_bin_rows": n_rows, **r}
            per_bin_fold_rows.append(r2)
        bin_summary = evaluate_bin(fold_rows)
        bin_summary = {"bin": bin_name, "n_rows": n_rows, **bin_summary}
        bin_summaries.append(bin_summary)

    if per_bin_fold_rows:
        cols = [
            "bin", "n_bin_rows", "fold", "train_start", "train_end",
            "val_start", "val_end", "train_n", "val_n",
            "base_rate_val", "val_auc", "status",
        ]
        pd.DataFrame(per_bin_fold_rows)[cols].to_csv(
            out_dir / "per_bin_per_fold_auc.csv", index=False,
        )

    if bin_summaries:
        cols = [
            "bin", "n_rows", "status", "n_folds_trained",
            "min_train_n", "min_val_n",
            "mean_auc", "worst_auc", "fold_stdev",
            "mean_base_rate_val", "gate_pass",
        ]
        pd.DataFrame(bin_summaries)[cols].to_csv(
            out_dir / "variant_summary.csv", index=False,
        )

    any_pass = any(s["gate_pass"] for s in bin_summaries)
    trained_bins = [s for s in bin_summaries if s["status"] == "trained"]
    best_bin = max(
        trained_bins,
        key=lambda s: s["worst_auc"] if np.isfinite(s["worst_auc"]) else -np.inf,
        default=None,
    )
    return {
        "variant": variant_name,
        "any_bin_pass": any_pass,
        "best_bin": best_bin["bin"] if best_bin else "",
        "best_bin_worst_auc": (
            best_bin["worst_auc"] if best_bin else float("nan")
        ),
        "n_bins_total": len(bin_summaries),
        "n_bins_trained": len(trained_bins),
        "meta": meta,
    }


def run_tf(tf: str) -> dict:
    print(f"[{tf}] loading merged labels+features ...", flush=True)
    merged = load_merged(tf)
    print(f"[{tf}] rows: {len(merged):,}", flush=True)
    feat_cols = TOP15[tf]

    tf_out = _OUT / f"timeframe_{tf}"
    tf_out.mkdir(parents=True, exist_ok=True)

    variants = [
        ("variant_a_vol_percentile", variant_a_vol_quintile),
        ("variant_b_d1_trend", variant_b_d1_trend),
        ("variant_c_year_quarter", variant_c_year_quarter),
    ]

    rows = []
    for name, fn in variants:
        result = run_variant(merged, tf, name, fn, feat_cols, tf_out / name)
        rows.append({
            "variant": name,
            "best_bin": result["best_bin"],
            "best_bin_worst_auc": result["best_bin_worst_auc"],
            "n_bins_total": result["n_bins_total"],
            "n_bins_trained": result["n_bins_trained"],
            "gate_pass": result["any_bin_pass"],
            "b1_b4_baseline_worst_auc": B1B4_BASELINE_WORST_AUC[tf],
        })

    pd.DataFrame(rows).to_csv(tf_out / "timeframe_summary.csv", index=False)
    print(f"[{tf}] DONE", flush=True)
    return {r["variant"]: r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["1h", "4h", "all"], default="all")
    args = ap.parse_args()
    tfs = list(TF_LIST) if args.tf == "all" else [args.tf]
    for tf in tfs:
        run_tf(tf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
