"""EXP-04 — Q2 2022 fold regime characterisation.

Question: does the AUC drop in the Q2 2022 fold correlate with a measurable,
entry-time-knowable regime feature?

Recap of fold-AUC pattern from Step 4 (reproduced from _common.py):
  fold 0: E 0.625, D1 0.669
  fold 1: E 0.670, D1 0.625
  fold 2: E 0.551, D1 0.518   <-- the drop
  fold 3: E 0.646, D1 0.461
  fold 4: E 0.656, D1 0.676

5-fold TimeSeriesSplit on entry-time-ordered c1 (n=228). Each test fold has
38 trades. The temporal split divides the 5y window into 5 chronological
chunks.

Method:
  1. Recover fold membership per trade from wf_oof_preds.
  2. Per fold, report:
       - calendar date range of test trades
       - regime descriptors computed at trade entry time (no lookahead):
           * atr14_4h_at_entry — entry-time ATR (pip-normalised)
           * atr14_d1_at_entry — entry-time D1 ATR
           * realised_vol_proxy — entry-bar ATR / 20-bar mean ATR (rolling)
           * d1_trend_strength — abs(ema20_d1_dist_atr) signed-trend magnitude
           * spread_pips_used — entry-bar spread (pips)
  3. Per fold, mean & std of each regime descriptor.
  4. Correlate fold-mean regime descriptor with fold-AUC across the 5 folds.
     Identify descriptors that separate fold 2 from the rest.
  5. Note any descriptor where fold 2 sits ≥ 1.5 σ from the cohort mean.

All regime descriptors are computed from the existing Pipeline E / D1 feature
frames (entry-time-only). Realised vol proxy uses pair_caches which Step 4
already builds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.experiments._common import (  # noqa: E402
    PIPELINE_D1_FEATURES,
    PIPELINE_E_FEATURES,
    load_arc10_c1_bundle,
    sha256_file,
    wf_oof_preds,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_10" / "experiments"
RAW_DIR = OUT_DIR / "raw"

DESCRIPTOR_FEATURES = [
    "atr14_4h_at_entry",
    "atr14_4h_at_entry_pct_of_close",
    "range_pct_atr_last_5_bars",
    "range_pct_atr_last_20_bars",
    "ema20_4h_dist_atr",
    "ema50_4h_dist_atr",
    "ema200_4h_dist_atr",
    "ema20_slope_4h",
    "ema50_slope_4h",
]


def main() -> int:
    print("[EXP-04] loading c1 bundle...", file=sys.stderr)
    b = load_arc10_c1_bundle()

    print("[EXP-04] computing fold membership for Pipeline E...", file=sys.stderr)
    op_E, oy_E, paf_E, fs_E, fm_E = wf_oof_preds(
        b.e_features, b.y, PIPELINE_E_FEATURES
    )
    print("[EXP-04] computing fold membership for Pipeline D1...", file=sys.stderr)
    op_D, oy_D, paf_D, fs_D, fm_D = wf_oof_preds(
        b.d1_features, b.y, PIPELINE_D1_FEATURES
    )

    # Merge fold membership onto the c1 entry-ordered feature frame.
    e_df = b.e_features.copy()
    e_df["fold_E"] = fm_E
    e_df["fold_D1"] = fm_D
    # E and D1 should split identically because both use entry_time ordering
    # and same TimeSeriesSplit config. Verify:
    assert np.array_equal(fm_E, fm_D), "E and D1 fold splits diverged — order bug"

    # Add D1 features needed for descriptors.
    d1_df = b.d1_features.copy()
    full = e_df.merge(
        d1_df[["trade_id", "atr14_d1_at_entry", "atr14_d1_at_entry_pct_of_close",
               "ema20_d1_dist_atr", "ema50_d1_dist_atr", "ema200_d1_dist_atr",
               "ema20_slope_d1", "ema50_slope_d1"]],
        on="trade_id",
        how="left",
    )

    # spread_pips from the original trades_all.
    sp = b.trades[["trade_id", "spread_pips_used"]]
    full = full.merge(sp, on="trade_id", how="left")

    # Per-fold summary.
    folds = sorted(int(f) for f in full["fold_E"].unique() if int(f) >= 0)
    fold_rows: List[Dict[str, object]] = []
    for fold_idx in folds:
        sub = full[full["fold_E"] == fold_idx]
        rec = {
            "fold": fold_idx,
            "n_test_trades": int(len(sub)),
            "n_train_trades": int(fs_E[fold_idx][0]),
            "entry_time_min": str(pd.to_datetime(sub["entry_time"]).min()),
            "entry_time_max": str(pd.to_datetime(sub["entry_time"]).max()),
            "auc_E": paf_E[fold_idx],
            "auc_D1": paf_D[fold_idx],
            "base_success": float(b.y[full["fold_E"] == fold_idx].mean()) if len(sub) else float("nan"),
        }
        for col in DESCRIPTOR_FEATURES + ["spread_pips_used", "atr14_d1_at_entry",
                                          "atr14_d1_at_entry_pct_of_close",
                                          "ema20_d1_dist_atr", "ema50_d1_dist_atr",
                                          "ema200_d1_dist_atr", "ema20_slope_d1", "ema50_slope_d1"]:
            if col in sub.columns:
                vals = sub[col].astype(float)
                rec[f"{col}__mean"] = float(vals.mean())
                rec[f"{col}__std"] = float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0
        fold_rows.append(rec)

    df_folds = pd.DataFrame(fold_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(RAW_DIR / "exp_04_fold_summary.csv", index=False, lineterminator="\n")

    # Correlation: fold-mean descriptor vs fold-AUC.
    descriptor_cols = [c for c in df_folds.columns if c.endswith("__mean") and c not in ("base_success__mean",)]
    auc_E_arr = df_folds["auc_E"].astype(float).to_numpy()
    auc_D1_arr = df_folds["auc_D1"].astype(float).to_numpy()
    corr_rows = []
    for col in descriptor_cols:
        vals = df_folds[col].astype(float).to_numpy()
        # Pearson correlation across n=5 folds. With n=5, anything < |0.9| is noise.
        # Useful for ranking, not statistical inference.
        valid = ~(np.isnan(vals) | np.isnan(auc_E_arr) | np.isnan(auc_D1_arr))
        if valid.sum() >= 3:
            pe = np.corrcoef(vals[valid], auc_E_arr[valid])[0, 1] if vals[valid].std() > 0 else 0.0
            pd1 = np.corrcoef(vals[valid], auc_D1_arr[valid])[0, 1] if vals[valid].std() > 0 else 0.0
        else:
            pe = pd1 = float("nan")
        # Fold-2 z-score within the 5-fold descriptor distribution.
        cohort_mean = float(np.mean(vals))
        cohort_std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        if cohort_std > 0 and not np.isnan(vals[2]):
            z_fold2 = (vals[2] - cohort_mean) / cohort_std
        else:
            z_fold2 = float("nan")
        corr_rows.append({
            "descriptor": col.replace("__mean", ""),
            "fold0": vals[0],
            "fold1": vals[1],
            "fold2": vals[2],
            "fold3": vals[3],
            "fold4": vals[4],
            "cohort_mean_5fold": cohort_mean,
            "cohort_std_5fold": cohort_std,
            "fold2_zscore": z_fold2,
            "pearson_corr_with_E_AUC": pe,
            "pearson_corr_with_D1_AUC": pd1,
        })
    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(RAW_DIR / "exp_04_regime_correlations.csv", index=False, lineterminator="\n")

    # Identify descriptors where fold 2 is >= 1.5 sigma from cohort mean.
    outlier = df_corr[df_corr["fold2_zscore"].abs() >= 1.5].copy()
    outlier_sorted = outlier.reindex(outlier["fold2_zscore"].abs().sort_values(ascending=False).index)

    md = []
    md.append("# EXP-04 — Q2 2022 fold regime characterisation")
    md.append("")
    md.append("**Status:** experimental (not a Step 5 gate).")
    md.append("")
    md.append("## Question")
    md.append("Does the AUC drop in fold 2 correlate with a measurable, entry-time-knowable regime descriptor?")
    md.append("")
    md.append("## Per-fold summary")
    md.append("")
    md.append("| fold | n_test | date range | base success | E AUC | D1 AUC |")
    md.append("|---:|---:|---|---:|---:|---:|")
    for r in fold_rows:
        md.append(
            f"| {r['fold']} | {r['n_test_trades']} | "
            f"{str(r['entry_time_min'])[:10]} → {str(r['entry_time_max'])[:10]} | "
            f"{r['base_success']:.4f} | {r['auc_E']:.4f} | {r['auc_D1']:.4f} |"
        )
    md.append("")
    md.append("## Fold-2 regime outliers (|z-score| ≥ 1.5 within 5-fold distribution)")
    md.append("")
    if len(outlier_sorted) == 0:
        md.append("_No descriptor places fold 2 ≥ 1.5σ from the cohort mean._")
    else:
        md.append("| descriptor | fold0 | fold1 | fold2 | fold3 | fold4 | fold2 z-score | corr w/ E AUC | corr w/ D1 AUC |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in outlier_sorted.iterrows():
            md.append(
                f"| {r['descriptor']} | {r['fold0']:.4g} | {r['fold1']:.4g} | {r['fold2']:.4g} | "
                f"{r['fold3']:.4g} | {r['fold4']:.4g} | {r['fold2_zscore']:+.2f} | "
                f"{r['pearson_corr_with_E_AUC']:+.3f} | {r['pearson_corr_with_D1_AUC']:+.3f} |"
            )
    md.append("")
    md.append("## All descriptors — sorted by correlation with E AUC")
    md.append("")
    df_corr_sorted_E = df_corr.reindex(
        df_corr["pearson_corr_with_E_AUC"].abs().sort_values(ascending=False).index
    )
    md.append("| descriptor | fold2 z | corr w/ E AUC | corr w/ D1 AUC |")
    md.append("|---|---:|---:|---:|")
    for _, r in df_corr_sorted_E.iterrows():
        md.append(
            f"| {r['descriptor']} | {r['fold2_zscore']:+.2f} | "
            f"{r['pearson_corr_with_E_AUC']:+.3f} | {r['pearson_corr_with_D1_AUC']:+.3f} |"
        )
    md.append("")
    md.append("## Caveats and interpretation")
    md.append("- n=5 folds is too few for statistical inference on correlations; Pearson values are ranking only.")
    md.append("- Fold size is the protocol-mandated 5-fold TimeSeriesSplit; cannot increase n without altering Step 4 wiring.")
    md.append("- A descriptor flagged as fold-2 outlier (|z| ≥ 1.5) AND with |corr| ≥ 0.5 to E AUC is the candidate ")
    md.append("  for cross-arc entry-time regime filtering (Open-04 informational support).")
    md.append("")
    if len(outlier_sorted) > 0:
        candidate_count = int(((outlier_sorted["pearson_corr_with_E_AUC"].abs() >= 0.5)).sum())
        if candidate_count > 0:
            top_cand = outlier_sorted[outlier_sorted["pearson_corr_with_E_AUC"].abs() >= 0.5].iloc[0]
            md.append(
                f"**Strongest candidate descriptor:** `{top_cand['descriptor']}` "
                f"(fold-2 z = {top_cand['fold2_zscore']:+.2f}, "
                f"corr w/ E AUC = {top_cand['pearson_corr_with_E_AUC']:+.3f})."
            )
        else:
            md.append("No descriptor combines outlier z-score with |corr w/ E AUC| ≥ 0.5.")
    else:
        md.append("No fold-2 outliers found; fold-2 AUC drop is not explained by these regime descriptors.")
    md.append("")
    md.append("## Artefacts")
    md.append(f"- `raw/exp_04_fold_summary.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_04_fold_summary.csv')[:16]}…`)")
    md.append(f"- `raw/exp_04_regime_correlations.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_04_regime_correlations.csv')[:16]}…`)")
    md.append("")

    (OUT_DIR / "EXP_04_q2_2022_regime.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[EXP-04] wrote {OUT_DIR / 'EXP_04_q2_2022_regime.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
