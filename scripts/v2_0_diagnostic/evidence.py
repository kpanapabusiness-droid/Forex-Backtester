"""v2.0 archetype diagnostic — Step 6: v2.0 evidence flag.

For every (dataset, K, archetype) tuple, evaluate against first-pass priors.

  clean        monotonicity_centroid >= 0.55
               AND local_peaks_centroid <= 4
               AND pullback_magnitude_centroid <= 0.8
  meaty        final_r_mean > 0
               AND frac_reach_1R >= 0.70
               AND fwd_mfe_h240_p50 >= 1.5
               AND frac_wrong_way <= 0.30
  predictable  auc_mean >= 0.60
  size_viable  size_fraction_of_pool >= 0.15
  not_overlap  does NOT majority-overlap (>=60%) with dual-gate-passing
               existing cluster (Arc 1 C0, Arc 2 C2). KH-24 -> always True.

qualifies_as_v2_0_evidence = all five TRUE.

Thresholds are PRIORS, not floors. The report lists tuples meeting them;
the chat may apply different thresholds. Output is the union of all
evaluated tuples — TRUE and FALSE — for transparency.
"""
from __future__ import annotations

import pandas as pd


def build_evidence(
    per_dataset: dict[str, dict],
) -> pd.DataFrame:
    """Assemble the v2.0 evidence flags CSV across all 15 runs.

    per_dataset[name] = {
        "K": {
            k: {
                "summary":          archetype-summary DataFrame,
                "predictability":   AUC DataFrame,
                "overlap_flags":    {(k, archetype_id): bool} or {} for KH-24
            } for k in K_VALUES
        }
    }
    """
    rows = []
    for name, payload in per_dataset.items():
        for k, run in payload["K"].items():
            summary = run["summary"]
            pred    = run["predictability"]
            overlap = run.get("overlap_flags", {})  # {} for KH-24
            for _, srow in summary.iterrows():
                arch = int(srow["archetype_id"])
                prow = pred[pred["archetype_id"] == arch]
                auc_mean = float(prow["auc_mean"].iloc[0]) if len(prow) else float("nan")

                clean = (
                    (srow["monotonicity_centroid"]       >= 0.55)
                    and (srow["local_peaks_centroid"]    <= 4)
                    and (srow["pullback_magnitude_centroid"] <= 0.8)
                )
                meaty = (
                    (srow["final_r_mean"]    > 0)
                    and (srow["frac_reach_1R"] >= 0.70)
                    and (srow["fwd_mfe_h240_p50"] >= 1.5)
                    and (srow["frac_wrong_way"] <= 0.30)
                )
                predictable = (auc_mean >= 0.60) if pd.notna(auc_mean) else False
                size_viable = (srow["size_fraction_of_pool"] >= 0.15)

                if name == "kh24":
                    not_overlap = True
                else:
                    not_overlap = not bool(overlap.get((k, arch), False))

                qualifies = bool(clean and meaty and predictable and size_viable and not_overlap)

                rows.append({
                    "dataset":       name,
                    "K":             k,
                    "archetype_id":  arch,
                    "clean":         bool(clean),
                    "meaty":         bool(meaty),
                    "predictable":   bool(predictable),
                    "size_viable":   bool(size_viable),
                    "not_overlap":   bool(not_overlap),
                    "qualifies_as_v2_0_evidence": qualifies,
                    # Echoed fields for inline reading:
                    "size_fraction_of_pool":         float(srow["size_fraction_of_pool"]),
                    "monotonicity_centroid":         float(srow["monotonicity_centroid"]),
                    "local_peaks_centroid":          float(srow["local_peaks_centroid"]),
                    "pullback_magnitude_centroid":   float(srow["pullback_magnitude_centroid"]),
                    "final_r_mean":                  float(srow["final_r_mean"]),
                    "frac_reach_1R":                 float(srow["frac_reach_1R"]),
                    "fwd_mfe_h240_p50":              float(srow["fwd_mfe_h240_p50"]),
                    "frac_wrong_way":                float(srow["frac_wrong_way"]),
                    "auc_mean":                      auc_mean,
                    "shape_tag":                     srow["shape_tag"],
                })
    return pd.DataFrame(rows)
