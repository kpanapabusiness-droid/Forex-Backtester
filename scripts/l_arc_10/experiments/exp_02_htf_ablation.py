"""EXP-02 — HTF feature ablation on Arc 10 c1.

Question: which of {L_1, L_0, age, slope} drives the +AUC lift over the
entry-time-only baseline?

Method:
  - Define HTF feature buckets:
      L_1   = {L1_to_atr_proximity, reject_buffer_atr}      # L_1 distance + reject buffer
      L_0   = {L1_minus_L0_atr}                              # adds L_0 via the delta
      age   = {L1_age_d1_bars, L0_age_d1_bars,
                L1_minus_L0_d1_bars}                         # age fields (incl. their delta)
      slope = {upper_fraction}                               # 4H signal-bar geometry (kept for completeness)
  - Variants:
      full           = full 25-feature Pipeline E
      no_L_1         = full minus L_1 bucket
      no_L_0         = full minus L_0 bucket
      no_age         = full minus age bucket
      no_slope       = full minus slope bucket
      base_no_HTF    = full minus all 4 buckets (entry-time generic + cyclic + pair_id only)

  Same procedure for D1: D1's PIPELINE_D1_FEATURES list doesn't include the HTF
  signal-bar features, so the ablation is meaningful only for Pipeline E.
  We report D1 baseline separately for context.

  5-fold TimeSeriesSplit walk-forward, RF seed=42.

Deliverable: per-feature-bucket attribution table ranked by AUC contribution
vs full set and vs entry-time-only baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.experiments._common import (  # noqa: E402
    PIPELINE_E_FEATURES,
    PIPELINE_D1_FEATURES,
    load_arc10_c1_bundle,
    wf_oof_preds,
    mean_auc_safe,
    sha256_file,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_10" / "experiments"
RAW_DIR = OUT_DIR / "raw"

# Feature buckets per the dispatch's {L_1, L_0, age, slope} grouping.
HTF_BUCKETS: Dict[str, List[str]] = {
    "L_1": ["L1_to_atr_proximity", "reject_buffer_atr"],
    "L_0": ["L1_minus_L0_atr"],
    "age": ["L1_age_d1_bars", "L0_age_d1_bars", "L1_minus_L0_d1_bars"],
    "slope": ["upper_fraction"],
}

E_THRESHOLD = 0.65
D1_THRESHOLD = 0.60


def main() -> int:
    print("[EXP-02] loading c1 bundle...", file=sys.stderr)
    b = load_arc10_c1_bundle()

    # Sanity: all bucket features present in PIPELINE_E_FEATURES.
    all_htf = sum(HTF_BUCKETS.values(), [])
    missing = [f for f in all_htf if f not in PIPELINE_E_FEATURES]
    assert not missing, f"missing HTF features in PIPELINE_E_FEATURES: {missing}"

    # Variant feature sets.
    variants: Dict[str, List[str]] = {"full_E": list(PIPELINE_E_FEATURES)}
    for bname, feats in HTF_BUCKETS.items():
        variants[f"no_{bname}"] = [f for f in PIPELINE_E_FEATURES if f not in feats]
    variants["base_no_HTF"] = [f for f in PIPELINE_E_FEATURES if f not in all_htf]

    rows: List[Dict[str, object]] = []
    per_fold_table: List[Dict[str, object]] = []
    for variant_name, feats in variants.items():
        print(f"[EXP-02] training E variant '{variant_name}' (n_feat={len(feats)})...",
              file=sys.stderr)
        _, _, paf, _, _ = wf_oof_preds(b.e_features, b.y, feats)
        m, s = mean_auc_safe(paf)
        rows.append({
            "variant": variant_name,
            "pipeline": "E",
            "n_features": len(feats),
            "mean_auc": m,
            "std_auc": s,
            "gate": E_THRESHOLD,
            "margin": m - E_THRESHOLD,
            "pass": m >= E_THRESHOLD,
        })
        for fold_idx, auc in enumerate(paf):
            per_fold_table.append({
                "variant": variant_name,
                "pipeline": "E",
                "fold": fold_idx,
                "auc": auc,
            })

    # Pipeline D1 baseline for context (HTF buckets don't apply).
    print("[EXP-02] training D1 baseline for context...", file=sys.stderr)
    _, _, paf_D1, _, _ = wf_oof_preds(b.d1_features, b.y, PIPELINE_D1_FEATURES)
    m_D1, s_D1 = mean_auc_safe(paf_D1)
    rows.append({
        "variant": "full_D1_reference",
        "pipeline": "D1",
        "n_features": len(PIPELINE_D1_FEATURES),
        "mean_auc": m_D1,
        "std_auc": s_D1,
        "gate": D1_THRESHOLD,
        "margin": m_D1 - D1_THRESHOLD,
        "pass": m_D1 >= D1_THRESHOLD,
    })
    for fold_idx, auc in enumerate(paf_D1):
        per_fold_table.append({
            "variant": "full_D1_reference",
            "pipeline": "D1",
            "fold": fold_idx,
            "auc": auc,
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "exp_02_ablation_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(per_fold_table).to_csv(
        RAW_DIR / "exp_02_per_fold_auc.csv", index=False, lineterminator="\n"
    )

    # Compute attribution deltas.
    full_e_auc = float(df[df["variant"] == "full_E"]["mean_auc"].iloc[0])
    base_e_auc = float(df[df["variant"] == "base_no_HTF"]["mean_auc"].iloc[0])
    total_htf_lift = full_e_auc - base_e_auc

    bucket_attribution: List[Dict[str, object]] = []
    for bname in HTF_BUCKETS.keys():
        no_b_auc = float(df[df["variant"] == f"no_{bname}"]["mean_auc"].iloc[0])
        # LOO attribution: AUC drop when removing the bucket from the full set.
        loo_drop = full_e_auc - no_b_auc
        bucket_attribution.append({
            "bucket": bname,
            "features": ",".join(HTF_BUCKETS[bname]),
            "auc_without_bucket": no_b_auc,
            "loo_drop_from_full": loo_drop,
            "loo_pct_of_total_htf_lift": (loo_drop / total_htf_lift) if total_htf_lift > 0 else float("nan"),
        })
    pd.DataFrame(bucket_attribution).to_csv(
        RAW_DIR / "exp_02_attribution.csv", index=False, lineterminator="\n"
    )

    # Markdown summary.
    md = []
    md.append("# EXP-02 — HTF feature ablation")
    md.append("")
    md.append("**Status:** experimental (not a Step 5 gate).")
    md.append("")
    md.append("## Question")
    md.append("Which of {L_1, L_0, age, slope} drives the AUC lift over the entry-time-only baseline?")
    md.append("")
    md.append("## Method")
    md.append(f"- c1 cohort: n={len(b.y)} trades, base success {b.base_success:.4f}.")
    md.append("- Walk-forward 5-fold TimeSeriesSplit, RF seed=42 (matches Step 4).")
    md.append("- HTF buckets:")
    for bname, feats in HTF_BUCKETS.items():
        md.append(f"  - **{bname}**: `{', '.join(feats)}`")
    md.append("- Variants: full E, full E minus each bucket (LOO), full E minus all HTF (base).")
    md.append("- Pipeline D1 reported for context — D1 feature set does not include the")
    md.append("  signal-bar HTF features so bucket ablation does not apply.")
    md.append("")
    md.append("## Variant AUCs")
    md.append("")
    md.append("| variant | pipeline | n_feat | mean AUC | std | gate | margin | pass |")
    md.append("|---|---|---:|---:|---:|---:|---:|:---:|")
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['pipeline']} | {r['n_features']} | "
            f"{r['mean_auc']:.4f} | {r['std_auc']:.4f} | {r['gate']:.2f} | "
            f"{r['margin']:+.4f} | {'YES' if r['pass'] else 'no'} |"
        )
    md.append("")
    md.append("## Attribution — Pipeline E HTF buckets")
    md.append("")
    md.append(f"- Full E mean AUC: **{full_e_auc:.4f}**")
    md.append(f"- Base (no HTF) E mean AUC: **{base_e_auc:.4f}**")
    md.append(f"- Total HTF lift: **{total_htf_lift:+.4f}**")
    md.append("")
    md.append("| bucket | features | AUC without bucket | LOO drop from full | % of total HTF lift |")
    md.append("|---|---|---:|---:|---:|")
    bucket_attribution_sorted = sorted(bucket_attribution, key=lambda r: -r["loo_drop_from_full"])
    for r in bucket_attribution_sorted:
        pct = r["loo_pct_of_total_htf_lift"]
        pct_str = f"{pct*100:.1f}%" if not (isinstance(pct, float) and np.isnan(pct)) else "n/a (no HTF lift)"
        md.append(
            f"| {r['bucket']} | `{r['features']}` | {r['auc_without_bucket']:.4f} | "
            f"{r['loo_drop_from_full']:+.4f} | {pct_str} |"
        )
    md.append("")
    md.append("## Interpretation")
    if total_htf_lift <= 0:
        md.append("- HTF features as a class do **not** add measurable lift over the entry-time-only baseline.")
        md.append("  Bucket attribution percentages are not meaningful when the total lift is non-positive.")
    else:
        top = bucket_attribution_sorted[0]
        top_pct = (top["loo_drop_from_full"] / total_htf_lift) * 100
        md.append(f"- HTF features collectively contribute {total_htf_lift:+.4f} to mean E AUC.")
        if top_pct > 50:
            md.append(
                f"- The `{top['bucket']}` bucket carries **{top_pct:.1f}%** of that lift — "
                f"candidate for cross-arc feature promotion if Arc 8/9/11 show a similar pattern."
            )
        else:
            md.append(
                f"- No single bucket carries >50% of the lift; the top bucket "
                f"(`{top['bucket']}`) contributes {top_pct:.1f}%."
            )
        # Flag any negative bucket.
        negatives = [r for r in bucket_attribution_sorted if r["loo_drop_from_full"] < 0]
        if negatives:
            md.append(
                f"- Note: {len(negatives)} bucket(s) ({', '.join(n['bucket'] for n in negatives)}) "
                f"show negative LOO drop — removing them improves AUC slightly. Likely noise on n=228."
            )
    md.append("")
    md.append("## Artefacts")
    md.append(f"- `raw/exp_02_ablation_summary.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_02_ablation_summary.csv')[:16]}…`)")
    md.append(f"- `raw/exp_02_per_fold_auc.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_02_per_fold_auc.csv')[:16]}…`)")
    md.append(f"- `raw/exp_02_attribution.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_02_attribution.csv')[:16]}…`)")
    md.append("")

    (OUT_DIR / "EXP_02_htf_ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[EXP-02] wrote {OUT_DIR / 'EXP_02_htf_ablation.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
