"""EXP-01 — Bootstrap AUC confidence intervals for Arc 10 c1.

Question: is c1 E AUC 0.6296 statistically distinguishable from the 0.65
threshold? Is D1 AUC 0.5897 distinguishable from 0.60?

Method:
  1. Compute walk-forward out-of-fold predictions (oof_p, oof_y) for both
     pipelines exactly as Step 4 did — 5-fold TimeSeriesSplit, seed=42.
     This yields one (p, y) pair per c1 trade with no in-sample leakage.
  2. Bootstrap-resample the (p, y) pairs n_resamples=2000 times. Each
     resample picks 228 (p, y) pairs WITH replacement and recomputes AUC.
     This is the standard "AUC sampling distribution" estimator and avoids
     re-fitting 2000 RFs (which would not change the qualitative result —
     model variance is dominated by fold split not bootstrap split).
  3. Report 95% CI, point estimate, and mass above threshold.
  4. Joint distribution: P(E_AUC>=0.65 OR D1_AUC>=0.60) over paired resamples
     using the same trade-row indices (so the c1 sample is shared).

All seeds explicit. Determinism re-run check via sha256.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.experiments._common import (  # noqa: E402
    PIPELINE_D1_FEATURES,
    PIPELINE_E_FEATURES,
    RANDOM_STATE,
    load_arc10_c1_bundle,
    mean_auc_safe,
    sha256_file,
    wf_oof_preds,
)

N_RESAMPLES = 2000
E_THRESHOLD = 0.65
D1_THRESHOLD = 0.60
OUT_DIR = _REPO_ROOT / "results" / "l_arc_10" / "experiments"
RAW_DIR = OUT_DIR / "raw"


def bootstrap_auc(
    p: np.ndarray, y: np.ndarray, n_resamples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (auc_distribution, index_matrix) where index_matrix[i] is the
    i-th resample's row indices (length n) so the same resample can be
    re-used for joint distribution computation."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(p)
    idx_matrix = rng.integers(0, n, size=(n_resamples, n))
    aucs = np.full(n_resamples, np.nan, dtype=float)
    for i in range(n_resamples):
        idx = idx_matrix[i]
        ys = y[idx]
        ps = p[idx]
        if len(set(ys.tolist())) < 2:
            continue
        aucs[i] = float(roc_auc_score(ys, ps))
    return aucs, idx_matrix


def main() -> int:
    print("[EXP-01] loading c1 bundle...", file=sys.stderr)
    b = load_arc10_c1_bundle()
    n = len(b.y)

    print("[EXP-01] walk-forward OOF preds for Pipeline E...", file=sys.stderr)
    oof_pE, oof_yE, paf_E, fs_E, fm_E = wf_oof_preds(
        b.e_features, b.y, PIPELINE_E_FEATURES
    )
    mean_E, std_E = mean_auc_safe(paf_E)
    print(f"  E mean AUC (point estimate): {mean_E:.4f}", file=sys.stderr)

    print("[EXP-01] walk-forward OOF preds for Pipeline D1...", file=sys.stderr)
    oof_pD1, oof_yD1, paf_D1, fs_D1, fm_D1 = wf_oof_preds(
        b.d1_features, b.y, PIPELINE_D1_FEATURES
    )
    mean_D1, std_D1 = mean_auc_safe(paf_D1)
    print(f"  D1 mean AUC (point estimate): {mean_D1:.4f}", file=sys.stderr)

    # Pool OOF predictions: each c1 trade has exactly one (p, y) once across
    # folds. The order from wf_oof_preds is fold-stacking order (fold 0 test
    # rows, then fold 1 test rows, ...). Length = sum of fold test sizes.
    n_oof = len(oof_pE)
    assert len(oof_pD1) == n_oof, "E and D1 OOF lengths must match"

    print(f"[EXP-01] bootstrapping {N_RESAMPLES} resamples on n_oof={n_oof}...",
          file=sys.stderr)
    aucs_E, idx_matrix = bootstrap_auc(oof_pE, oof_yE, N_RESAMPLES, seed=RANDOM_STATE)

    # Re-use idx_matrix for D1 so paired resamples share row sets.
    from sklearn.metrics import roc_auc_score
    aucs_D1 = np.full(N_RESAMPLES, np.nan, dtype=float)
    for i in range(N_RESAMPLES):
        idx = idx_matrix[i]
        ys = oof_yD1[idx]
        ps = oof_pD1[idx]
        if len(set(ys.tolist())) >= 2:
            aucs_D1[i] = float(roc_auc_score(ys, ps))

    valid_E = aucs_E[~np.isnan(aucs_E)]
    valid_D1 = aucs_D1[~np.isnan(aucs_D1)]

    def pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q))

    e_ci = (pct(valid_E, 2.5), pct(valid_E, 97.5))
    d1_ci = (pct(valid_D1, 2.5), pct(valid_D1, 97.5))
    mass_E = float(np.mean(valid_E >= E_THRESHOLD))
    mass_D1 = float(np.mean(valid_D1 >= D1_THRESHOLD))

    # Joint: P(either clears) over paired resamples (only resamples valid for both).
    paired_mask = (~np.isnan(aucs_E)) & (~np.isnan(aucs_D1))
    paired_E = aucs_E[paired_mask]
    paired_D1 = aucs_D1[paired_mask]
    joint_either = float(np.mean((paired_E >= E_THRESHOLD) | (paired_D1 >= D1_THRESHOLD)))
    joint_both = float(np.mean((paired_E >= E_THRESHOLD) & (paired_D1 >= D1_THRESHOLD)))

    # OOF point estimates (AUC computed once on the pooled OOF preds — note this
    # differs from "mean of per-fold AUC" because folds have different bases).
    oof_auc_E = float(roc_auc_score(oof_yE, oof_pE)) if len(set(oof_yE.tolist())) >= 2 else float("nan")
    oof_auc_D1 = float(roc_auc_score(oof_yD1, oof_pD1)) if len(set(oof_yD1.tolist())) >= 2 else float("nan")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df_E = pd.DataFrame({"resample_idx": np.arange(N_RESAMPLES), "auc": aucs_E})
    df_E.to_csv(RAW_DIR / "exp_01_bootstrap_auc_E.csv", index=False, lineterminator="\n")
    df_D1 = pd.DataFrame({"resample_idx": np.arange(N_RESAMPLES), "auc": aucs_D1})
    df_D1.to_csv(RAW_DIR / "exp_01_bootstrap_auc_D1.csv", index=False, lineterminator="\n")

    summary = {
        "n_trades": n,
        "n_oof_pairs": n_oof,
        "n_resamples": N_RESAMPLES,
        "random_state": RANDOM_STATE,
        "pipeline_E": {
            "mean_per_fold_auc": mean_E,
            "std_per_fold_auc": std_E,
            "per_fold_auc": [float(a) for a in paf_E],
            "oof_pooled_auc": oof_auc_E,
            "bootstrap_mean": float(np.mean(valid_E)),
            "bootstrap_median": float(np.median(valid_E)),
            "bootstrap_ci_95": [e_ci[0], e_ci[1]],
            "threshold": E_THRESHOLD,
            "mass_above_threshold": mass_E,
            "step4_reported_auc": 0.6296,
            "step4_margin": mean_E - E_THRESHOLD,
        },
        "pipeline_D1": {
            "mean_per_fold_auc": mean_D1,
            "std_per_fold_auc": std_D1,
            "per_fold_auc": [float(a) for a in paf_D1],
            "oof_pooled_auc": oof_auc_D1,
            "bootstrap_mean": float(np.mean(valid_D1)),
            "bootstrap_median": float(np.median(valid_D1)),
            "bootstrap_ci_95": [d1_ci[0], d1_ci[1]],
            "threshold": D1_THRESHOLD,
            "mass_above_threshold": mass_D1,
            "step4_reported_auc": 0.5897,
            "step4_margin": mean_D1 - D1_THRESHOLD,
        },
        "joint": {
            "p_either_clears": joint_either,
            "p_both_clear": joint_both,
            "n_paired_resamples": int(paired_mask.sum()),
        },
    }
    (RAW_DIR / "exp_01_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Markdown summary.
    md = []
    md.append("# EXP-01 — AUC bootstrap confidence intervals")
    md.append("")
    md.append("**Status:** experimental (not a Step 5 gate).")
    md.append("")
    md.append("## Question")
    md.append("Is c1 E AUC 0.6296 statistically distinguishable from the 0.65 threshold?")
    md.append("Is D1 AUC 0.5897 statistically distinguishable from 0.60?")
    md.append("")
    md.append("## Method")
    md.append(f"- c1 cohort: n={n} trades, base success rate {b.base_success:.4f}.")
    md.append(f"- Walk-forward 5-fold TimeSeriesSplit (matches Step 4); RF seed={RANDOM_STATE}.")
    md.append(f"- Pool out-of-fold predictions across folds: n_oof_pairs={n_oof}.")
    md.append(f"- Bootstrap-resample (p, y) pairs WITH replacement n={N_RESAMPLES} times,")
    md.append("  paired resamples (same indices for E and D1) so joint P is exact.")
    md.append("- Note: per-fold-AUC bootstrap is intentionally not done — re-fitting 2000")
    md.append("  RFs would add model-variance noise without changing the qualitative answer.")
    md.append("  AUC sampling-distribution bootstrap on OOF (p, y) is the standard estimator.")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("| Pipeline | Point estimate (mean per-fold) | OOF pooled AUC | Bootstrap mean | 95% CI | Threshold | P(AUC >= threshold) | Step 4 margin |")
    md.append("|---|---:|---:|---:|---|---:|---:|---:|")
    md.append(
        f"| E  | {mean_E:.4f} | {oof_auc_E:.4f} | {np.mean(valid_E):.4f} | "
        f"[{e_ci[0]:.4f}, {e_ci[1]:.4f}] | {E_THRESHOLD:.2f} | {mass_E:.4f} | {mean_E - E_THRESHOLD:+.4f} |"
    )
    md.append(
        f"| D1 | {mean_D1:.4f} | {oof_auc_D1:.4f} | {np.mean(valid_D1):.4f} | "
        f"[{d1_ci[0]:.4f}, {d1_ci[1]:.4f}] | {D1_THRESHOLD:.2f} | {mass_D1:.4f} | {mean_D1 - D1_THRESHOLD:+.4f} |"
    )
    md.append("")
    md.append("### Joint distribution (paired resamples)")
    md.append("")
    md.append(f"- P(E AUC >= {E_THRESHOLD} OR D1 AUC >= {D1_THRESHOLD}) = **{joint_either:.4f}**")
    md.append(f"- P(E AUC >= {E_THRESHOLD} AND D1 AUC >= {D1_THRESHOLD}) = {joint_both:.4f}")
    md.append(f"- n paired resamples: {int(paired_mask.sum())} of {N_RESAMPLES}")
    md.append("")
    md.append("## Interpretation")
    e_lower_above = e_ci[0] >= E_THRESHOLD
    e_upper_below = e_ci[1] < E_THRESHOLD
    d1_lower_above = d1_ci[0] >= D1_THRESHOLD
    d1_upper_below = d1_ci[1] < D1_THRESHOLD
    if e_upper_below:
        e_verdict = "Pipeline E is **statistically below** the 0.65 threshold (upper 95% CI bound < 0.65)."
    elif e_lower_above:
        e_verdict = "Pipeline E is **statistically above** the 0.65 threshold (lower 95% CI bound > 0.65)."
    else:
        e_verdict = f"Pipeline E is **inside the 95% CI band around the threshold** — P(AUC>=0.65) = {mass_E:.4f}; not statistically distinguishable from threshold."
    if d1_upper_below:
        d1_verdict = "Pipeline D1 is **statistically below** the 0.60 threshold (upper 95% CI bound < 0.60)."
    elif d1_lower_above:
        d1_verdict = "Pipeline D1 is **statistically above** the 0.60 threshold (lower 95% CI bound > 0.60)."
    else:
        d1_verdict = f"Pipeline D1 is **inside the 95% CI band around the threshold** — P(AUC>=0.60) = {mass_D1:.4f}; not statistically distinguishable from threshold."
    md.append(f"- {e_verdict}")
    md.append(f"- {d1_verdict}")
    md.append("")
    md.append("## Artefacts")
    md.append(f"- `raw/exp_01_bootstrap_auc_E.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_01_bootstrap_auc_E.csv')[:16]}…`)")
    md.append(f"- `raw/exp_01_bootstrap_auc_D1.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_01_bootstrap_auc_D1.csv')[:16]}…`)")
    md.append(f"- `raw/exp_01_summary.json` (sha256 `{sha256_file(RAW_DIR / 'exp_01_summary.json')[:16]}…`)")
    md.append("")

    out_md = OUT_DIR / "EXP_01_auc_bootstrap.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[EXP-01] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
