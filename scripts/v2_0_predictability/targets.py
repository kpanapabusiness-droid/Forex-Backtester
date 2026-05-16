"""Step 1: identify target archetypes + exit-family grouping.

Targets are archetypes from PR #129 that meet RELAXED clean+meaty+size
criteria (mono >= 0.50; final_r_mean > 0; frac_reach_1R >= 0.70;
size_fraction >= 0.10). Arc 1 excluded (96% bars_held=1).

Exit-family grouping rules — mechanical, no judgement:

  trail_compatible  monotonicity >= 0.55 AND local_peaks <= 4
                    AND time_to_peak_relative >= 0.5
  stepwise_lock     monotonicity >= 0.50 AND 5 <= local_peaks <= 15
                    AND time_to_peak_relative >= 0.5
                    AND pullback_magnitude <= 0.5
  early_peak_tp     time_to_peak_relative <= 0.3 AND fwd_mfe_p50 >= 1.0
  untradeable       pct_peak_and_collapse >= 0.50 OR frac_wrong_way >= 0.40
  mixed             falls between rules
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIAGNOSTIC_ROOT = REPO_ROOT / "results" / "v2_0_diagnostic"


def identify_targets() -> pd.DataFrame:
    """Read v2_0 evidence flags; return target archetypes meeting relaxed criteria.

    Joins back to per-archetype summaries (per dataset) for full centroid
    context needed downstream (exit-family classification, characterisation).
    """
    ev = pd.read_csv(DIAGNOSTIC_ROOT / "v2_0_evidence_flags.csv")
    mask = (
        (ev["dataset"] != "arc1")
        & (ev["monotonicity_centroid"] >= 0.50)
        & (ev["final_r_mean"] > 0)
        & (ev["frac_reach_1R"] >= 0.70)
        & (ev["size_fraction_of_pool"] >= 0.10)
    )
    tgt = ev[mask].copy()

    # Pull additional centroid fields not already in v2_0_evidence_flags.csv.
    # The evidence CSV already includes fwd_mfe_h240_p50, so only add
    # time_to_peak_relative_centroid + pct_peak_and_collapse here.
    extra_rows = []
    for (ds, k), sub in tgt.groupby(["dataset", "K"]):
        path = DIAGNOSTIC_ROOT / ds / f"archetype_summaries_K{int(k)}.csv"
        summ = pd.read_csv(path)
        merged = sub.merge(
            summ[[
                "archetype_id", "size_count",
                "time_to_peak_relative_centroid", "pct_peak_and_collapse",
            ]],
            on="archetype_id", how="left", validate="one_to_one",
        )
        extra_rows.append(merged)
    out = pd.concat(extra_rows, axis=0, ignore_index=True)
    return out.sort_values(["dataset", "K", "archetype_id"]).reset_index(drop=True)


def assign_exit_family(row: pd.Series) -> str:
    mono = float(row["monotonicity_centroid"])
    peaks = float(row["local_peaks_centroid"])
    pull = float(row["pullback_magnitude_centroid"])
    ttp = float(row["time_to_peak_relative_centroid"])
    fwd_p50 = float(row["fwd_mfe_h240_p50"])
    pct_pc = float(row["pct_peak_and_collapse"])
    wrong = float(row["frac_wrong_way"])

    if pct_pc >= 0.50 or wrong >= 0.40:
        return "untradeable"
    if mono >= 0.55 and peaks <= 4 and ttp >= 0.5:
        return "trail_compatible"
    if mono >= 0.50 and 5 <= peaks <= 15 and ttp >= 0.5 and pull <= 0.5:
        return "stepwise_lock"
    if ttp <= 0.3 and fwd_p50 >= 1.0:
        return "early_peak_tp"
    return "mixed"


def build_grouping(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (targets_tagged, exit_family_grouping, grouped_targets) DataFrames.

    targets_tagged has the original target fields + exit_family_tag column.
    """
    targets = targets.copy()
    targets["exit_family_tag"] = targets.apply(assign_exit_family, axis=1)
    targets["centroid_summary"] = targets.apply(
        lambda r: (
            f"mono={r['monotonicity_centroid']:.2f} "
            f"peaks={r['local_peaks_centroid']:.1f} "
            f"pull={r['pullback_magnitude_centroid']:.2f} "
            f"ttp={r['time_to_peak_relative_centroid']:.2f}"
        ), axis=1,
    )
    grouping = targets[[
        "dataset", "K", "archetype_id", "centroid_summary", "exit_family_tag",
        "size_count", "size_fraction_of_pool",
    ]].copy() if "size_count" in targets.columns else targets[[
        "dataset", "K", "archetype_id", "centroid_summary", "exit_family_tag",
        "size_fraction_of_pool",
    ]].copy()

    # Per (dataset, K, tag) where tag != untradeable/mixed: combine archetype ids
    rows = []
    for (ds, k, tag), sub in targets.groupby(["dataset", "K", "exit_family_tag"]):
        if tag in ("untradeable", "mixed"):
            continue
        ids = sorted(sub["archetype_id"].astype(int).tolist())
        rows.append({
            "dataset": ds,
            "K": int(k),
            "exit_family_tag": tag,
            "archetype_ids_in_group": ",".join(str(i) for i in ids),
            "n_archetypes_in_group": len(ids),
            "total_size_fraction": float(sub["size_fraction_of_pool"].sum()),
        })
    grouped = pd.DataFrame(rows).sort_values(
        ["dataset", "K", "exit_family_tag"]
    ).reset_index(drop=True)
    return targets, grouping, grouped
