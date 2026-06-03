"""Lomega B1-B4 post-mortem diagnostic.

Diagnostic-only — operates on the labels.csv + features.csv produced by
scripts/lomega/lomega_b1_b4.py (PR #133). Does NOT modify B4 model
choice, label, or features. The RF re-fit per fold is solely to extract
the per-pair validation AUC (Diagnostic 1) and the full top-15
importance vector per fold (Diagnostic 3) that were not persisted in
the original B4 run.

Diagnostics:
  1 — per-fold × per-pair validation AUC (refit RF with same hyperparams)
  2 — per-fold × per-pair clean_move base rate (label distribution shift)
  3 — feature-importance drift across folds (Spearman ρ + >50% shifts)
  4 — quarterly vol regime markers (ATR, ATR_percentile, clean rate, AUC)

Usage:
    py scripts/lomega/lomega_postmortem.py --tf 1h
    py scripts/lomega/lomega_postmortem.py --tf all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

_REPO = Path(__file__).resolve().parent.parent.parent
_IN = _REPO / "results" / "lomega" / "b1_b4_discovery"
_OUT = _REPO / "results" / "lomega" / "b1_b4_postmortem"

RNG_SEED = 42

# Top-15 features per TF — frozen from PR #133 b4_summary.txt (deterministic
# univariate-AUC ranking on the same data). Pinning here keeps the diagnostic
# byte-deterministic without re-running B3.
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
    "d1": [
        "atr_200", "atr_50", "atr_14", "adx_14", "di_minus_14",
        "ema_50_above_200", "atr_ratio_14_50", "atr_percentile_200",
        "di_plus_14", "dxy_proxy_ema_20_slope", "range_compression_60",
        "atr_ratio_14_200", "position_in_range_20", "body_size_atr",
        "ema_20_slope",
    ],
}


def load_merged(tf: str) -> pd.DataFrame:
    tf_dir = _IN / f"timeframe_{tf}"
    labels = pd.read_csv(tf_dir / "labels.csv", parse_dates=["time"])
    feats = pd.read_csv(tf_dir / "features.csv", parse_dates=["time"])
    merged = feats.merge(labels, on=["pair", "time"], how="inner")
    merged = merged.sort_values(["time", "pair"]).reset_index(drop=True)
    return merged


def prepare_xy(merged: pd.DataFrame, feature_cols: list[str]):
    X = merged[feature_cols].values
    y = merged["clean_move"].astype(int).values
    pairs = merged["pair"].values
    times = merged["time"].values
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    pairs = pairs[mask]
    times = times[mask]
    order = np.argsort(times, kind="stable")
    return X[order], y[order], pairs[order], times[order]


def diagnostic_1_and_3(tf: str, merged: pd.DataFrame, out_dir: Path) -> dict:
    """Refit per fold; extract per-pair val AUC + per-fold top-15 importance."""
    feat_cols = TOP15[tf]
    X, y, pairs, times = prepare_xy(merged, feat_cols)

    tss = TimeSeriesSplit(n_splits=7)
    pair_universe = sorted(np.unique(pairs).tolist())

    rows_per_pair = []
    importance_per_fold = {}
    fold_overall_auc = {}
    fold_date_table = []

    for fold_id, (tr_idx, va_idx) in enumerate(tss.split(X), start=1):
        train_start = pd.Timestamp(times[tr_idx[0]])
        train_end = pd.Timestamp(times[tr_idx[-1]])
        val_start = pd.Timestamp(times[va_idx[0]])
        val_end = pd.Timestamp(times[va_idx[-1]])
        fold_date_table.append({
            "fold_id": fold_id,
            "train_start": train_start, "train_end": train_end,
            "val_start": val_start, "val_end": val_end,
            "train_n": int(len(tr_idx)), "val_n": int(len(va_idx)),
        })

        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
            print(f"  [fold {fold_id}] degenerate y — skip", flush=True)
            continue

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            random_state=RNG_SEED, n_jobs=-1,
        )
        rf.fit(X[tr_idx], y[tr_idx])
        proba = rf.predict_proba(X[va_idx])[:, 1]
        overall_auc = float(roc_auc_score(y[va_idx], proba))
        fold_overall_auc[fold_id] = overall_auc
        importance_per_fold[fold_id] = pd.Series(
            rf.feature_importances_, index=feat_cols
        )

        va_pairs = pairs[va_idx]
        va_y = y[va_idx]
        for p in pair_universe:
            sel = va_pairs == p
            n_total = int(sel.sum())
            n_pos = int(va_y[sel].sum()) if n_total > 0 else 0
            if n_total == 0 or n_pos == 0 or n_pos == n_total:
                pair_auc = np.nan
            else:
                pair_auc = float(roc_auc_score(va_y[sel], proba[sel]))
            rows_per_pair.append({
                "fold_id": fold_id,
                "pair": p,
                "val_auc": pair_auc,
                "n_val": n_total,
                "n_pos": n_pos,
                "clean_rate": (n_pos / n_total) if n_total else np.nan,
            })
        print(
            f"  [fold {fold_id}] {train_start.date()}..{val_end.date()} "
            f"overall AUC {overall_auc:.4f} "
            f"({len(tr_idx):,} train, {len(va_idx):,} val)",
            flush=True,
        )

    # Save D1 outputs
    pair_df = pd.DataFrame(rows_per_pair)
    pair_df = pair_df.sort_values(["fold_id", "pair"]).reset_index(drop=True)

    # Pivot for readability: rows = pair, cols = fold AUC
    pivot_auc = pair_df.pivot(index="pair", columns="fold_id", values="val_auc")
    fold_cols_sorted = sorted([c for c in pivot_auc.columns])
    pivot_auc = pivot_auc[fold_cols_sorted]
    # Append a row with the overall (all-pairs) val AUC per fold
    overall_vals = [fold_overall_auc.get(f, np.nan) for f in fold_cols_sorted]
    overall_row = pd.DataFrame([overall_vals], index=["__OVERALL__"],
                               columns=fold_cols_sorted)
    pivot_auc_out = pd.concat([pivot_auc, overall_row])
    pair_df.to_csv(out_dir / "per_fold_per_pair_auc.csv", index=False)
    pivot_auc_out.to_csv(out_dir / "per_fold_per_pair_auc_pivot.csv",
                         index_label="pair")

    # Save D3 outputs
    if importance_per_fold:
        imp_df = pd.DataFrame(importance_per_fold).reindex(feat_cols)
        imp_df.columns = [f"fold_{f}" for f in imp_df.columns]
        imp_df.to_csv(out_dir / "feature_importance_drift.csv", index_label="feature")

        # Spearman rank correlations between consecutive folds + fold1 vs fold7
        ranks = imp_df.rank(ascending=False, method="min")
        spearman_rows = []
        fold_keys = list(imp_df.columns)
        for i in range(len(fold_keys) - 1):
            a, b = fold_keys[i], fold_keys[i + 1]
            rho, p = sps.spearmanr(ranks[a], ranks[b])
            spearman_rows.append({"pair": f"{a}_vs_{b}", "spearman_rho": float(rho), "p_value": float(p)})
        if len(fold_keys) >= 2:
            rho, p = sps.spearmanr(ranks[fold_keys[0]], ranks[fold_keys[-1]])
            spearman_rows.append({"pair": f"{fold_keys[0]}_vs_{fold_keys[-1]}",
                                  "spearman_rho": float(rho), "p_value": float(p)})
        pd.DataFrame(spearman_rows).to_csv(
            out_dir / "feature_importance_spearman.csv", index=False
        )

        # Flag features whose importance shifted >50% between any consecutive folds
        shift_rows = []
        for i in range(len(fold_keys) - 1):
            a, b = fold_keys[i], fold_keys[i + 1]
            for f in feat_cols:
                va = float(imp_df.loc[f, a])
                vb = float(imp_df.loc[f, b])
                if va == 0:
                    pct = np.nan if vb == 0 else float("inf")
                else:
                    pct = (vb - va) / va
                if np.isfinite(pct) and abs(pct) > 0.50:
                    shift_rows.append({
                        "from": a, "to": b, "feature": f,
                        "imp_from": va, "imp_to": vb, "rel_change": pct,
                    })
        pd.DataFrame(shift_rows).to_csv(
            out_dir / "feature_importance_shifts_gt50pct.csv", index=False
        )

    fold_date_df = pd.DataFrame(fold_date_table)
    fold_date_df.to_csv(out_dir / "fold_dates.csv", index=False)

    return {
        "fold_overall_auc": fold_overall_auc,
        "fold_dates": fold_date_table,
        "feat_cols": feat_cols,
    }


def diagnostic_2(tf: str, merged: pd.DataFrame, fold_dates: list,
                  out_dir: Path) -> None:
    """Per-fold per-pair clean_move base rate (val window only)."""
    rows = []
    for fd in fold_dates:
        fid = fd["fold_id"]
        vs, ve = fd["val_start"], fd["val_end"]
        sub = merged[(merged["time"] >= vs) & (merged["time"] <= ve)]
        for p, grp in sub.groupby("pair", sort=True):
            n = int(len(grp))
            pos = int(grp["clean_move"].sum())
            rows.append({
                "fold_id": fid, "pair": p,
                "val_start": str(vs.date()), "val_end": str(ve.date()),
                "n_val": n, "n_clean_move": pos,
                "clean_rate": (pos / n) if n else np.nan,
            })
    df = pd.DataFrame(rows).sort_values(["fold_id", "pair"]).reset_index(drop=True)
    df.to_csv(out_dir / "per_fold_per_pair_base_rate.csv", index=False)


def diagnostic_4(tf: str, merged: pd.DataFrame, fold_overall_auc: dict,
                  fold_dates: list, out_dir: Path) -> None:
    """Quarterly vol regime markers + AUC overlay (AUC per quarter = mean
    of overall fold AUCs whose val window overlaps the quarter)."""
    m = merged.copy()
    m["quarter"] = m["time"].dt.to_period("Q").astype(str)

    quarterly = (
        m.groupby("quarter")
        .agg(
            mean_atr_14=("atr_14", "mean"),
            mean_atr_percentile_60=("atr_percentile_60", "mean"),
            clean_move_rate=("clean_move", "mean"),
            n_rows=("clean_move", "size"),
        )
        .reset_index()
        .sort_values("quarter")
    )

    # AUC per quarter — average overall fold AUC across folds whose val window
    # touches the quarter; mark NaN if no fold-val overlap (e.g., pre-2020 train-only quarters).
    fold_period_intervals = []
    for fd in fold_dates:
        fid = fd["fold_id"]
        vs, ve = pd.Timestamp(fd["val_start"]), pd.Timestamp(fd["val_end"])
        fold_period_intervals.append((fid, vs, ve))

    def auc_for_quarter(q_str):
        q = pd.Period(q_str, freq="Q")
        q_start, q_end = q.start_time, q.end_time
        aucs = []
        for fid, vs, ve in fold_period_intervals:
            if vs <= q_end and ve >= q_start:
                a = fold_overall_auc.get(fid, np.nan)
                if np.isfinite(a):
                    aucs.append(a)
        return float(np.mean(aucs)) if aucs else np.nan

    quarterly["mean_fold_auc_overlap"] = quarterly["quarter"].apply(auc_for_quarter)
    quarterly.to_csv(out_dir / "vol_regime_quarterly.csv", index=False)


def run_tf(tf: str) -> dict:
    out_dir = _OUT / f"timeframe_{tf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{tf}] loading merged labels+features ...", flush=True)
    merged = load_merged(tf)
    print(f"[{tf}] rows: {len(merged):,}", flush=True)

    print(f"[{tf}] Diagnostic 1+3: refit per fold, per-pair AUC + importance ...",
          flush=True)
    d13 = diagnostic_1_and_3(tf, merged, out_dir)

    print(f"[{tf}] Diagnostic 2: per-fold per-pair base rate ...", flush=True)
    diagnostic_2(tf, merged, d13["fold_dates"], out_dir)

    print(f"[{tf}] Diagnostic 4: quarterly vol regime markers ...", flush=True)
    diagnostic_4(tf, merged, d13["fold_overall_auc"], d13["fold_dates"], out_dir)

    print(f"[{tf}] DONE", flush=True)
    return d13


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["1h", "4h", "d1", "all"], default="all")
    args = ap.parse_args()
    tfs = ["1h", "4h", "d1"] if args.tf == "all" else [args.tf]
    for tf in tfs:
        run_tf(tf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
