"""v2.0 archetype diagnostic — Step 5: overlap with existing dual-gate clusters.

Arc 1 and Arc 2 only — KH-24 has no L-arc step-3 clusters.

For each K in {3, 4, 5, 6, 7}: confusion matrix path-shape archetype x
existing K3_kmeans cluster (the cluster column used in v1.3 calibration
and elsewhere in the L-arc pipeline).

Plus an overlap_summary.csv:
  - per existing dual-gate-passing cluster (Arc 1 C0, Arc 2 C2) at each K:
    which path-shape archetype contains its majority, and what fraction
  - per path-shape archetype (every K, every dataset): composition by
    existing cluster source, and the is_majority_from_passing_cluster flag.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Existing step-3 cluster column. K3_kmeans is the standard used in v1.3.
EXISTING_CLUSTER_COL = "K3_kmeans"

# Dual-gate-passing cluster id per dataset, per v1.3 calibration §3
# (high frac_reach_2R, low wrong_way, distinctly meaty).
DUAL_GATE_PASSING = {
    "arc1": 0,
    "arc2": 2,
}

OVERLAP_MAJORITY_THRESH = 0.60


def load_existing_clusters(dataset: str) -> pd.DataFrame:
    """Load existing K3_kmeans cluster assignments for dataset."""
    p = REPO_ROOT / "results" / f"l_arc_{dataset[-1]}" / "step3_extractability" / "cluster_assignments.csv"
    df = pd.read_csv(p, usecols=["trade_id", EXISTING_CLUSTER_COL])
    df["trade_id"] = df["trade_id"].astype("string")
    df = df.rename(columns={EXISTING_CLUSTER_COL: "existing_cluster"})
    return df


def confusion_matrix(
    assignments: pd.DataFrame, existing: pd.DataFrame
) -> pd.DataFrame:
    """Counts: rows = path-shape archetype, cols = existing cluster id."""
    merged = assignments.merge(existing, on="trade_id", how="inner", validate="one_to_one")
    return (
        merged.groupby(["archetype_id", "existing_cluster"])
        .size().unstack(fill_value=0)
        .sort_index().sort_index(axis=1)
    )


def overlap_summary(
    per_k_assignments: dict[int, pd.DataFrame],
    existing: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """One CSV combining forward + inverse overlap views across all K."""
    rows = []
    passing = DUAL_GATE_PASSING[dataset]
    existing_ids = sorted(existing["existing_cluster"].dropna().unique().tolist())

    for k, asgn in per_k_assignments.items():
        merged = asgn.merge(existing, on="trade_id", how="inner", validate="one_to_one")
        # Forward direction: per existing cluster (focus on passing), which path-shape arch holds majority
        for ex_id in existing_ids:
            ex_sub = merged[merged["existing_cluster"] == ex_id]
            if len(ex_sub) == 0:
                continue
            counts = ex_sub["archetype_id"].value_counts()
            top_arch = int(counts.idxmax())
            top_share = float(counts.iloc[0] / len(ex_sub))
            rows.append({
                "view": "forward",
                "K": k,
                "existing_cluster": int(ex_id),
                "is_dual_gate_passing": bool(ex_id == passing),
                "archetype_id": top_arch,
                "overlap_pct": top_share,
                "n_in_existing": int(len(ex_sub)),
                "pct_from_existing_0": float("nan"),
                "pct_from_existing_1": float("nan"),
                "pct_from_existing_2": float("nan"),
                "pct_from_existing_-2": float("nan"),
                "is_majority_from_passing_cluster": False,
            })
        # Inverse direction: per path-shape archetype, source decomposition
        for arch_id in sorted(merged["archetype_id"].unique().tolist()):
            arch_sub = merged[merged["archetype_id"] == arch_id]
            n = len(arch_sub)
            counts_norm = arch_sub["existing_cluster"].value_counts(normalize=True)
            comp = {f"pct_from_existing_{int(c)}": float(counts_norm.get(c, 0.0)) for c in existing_ids}
            is_majority_from_passing = bool(comp.get(f"pct_from_existing_{passing}", 0.0) >= OVERLAP_MAJORITY_THRESH)
            row = {
                "view": "inverse",
                "K": k,
                "existing_cluster": -999,
                "is_dual_gate_passing": False,
                "archetype_id": int(arch_id),
                "overlap_pct": float("nan"),
                "n_in_existing": int(n),
                "pct_from_existing_0": comp.get("pct_from_existing_0", 0.0),
                "pct_from_existing_1": comp.get("pct_from_existing_1", 0.0),
                "pct_from_existing_2": comp.get("pct_from_existing_2", 0.0),
                "pct_from_existing_-2": comp.get("pct_from_existing_-2", 0.0),
                "is_majority_from_passing_cluster": is_majority_from_passing,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def archetype_majority_overlap_flags(
    per_k_assignments: dict[int, pd.DataFrame],
    existing: pd.DataFrame,
    dataset: str,
) -> dict[tuple[int, int], bool]:
    """For each (K, archetype) return: does it majority-overlap (>=60%) the
    dual-gate-passing existing cluster? Used by Step 6's not_overlap criterion.
    """
    out: dict[tuple[int, int], bool] = {}
    passing = DUAL_GATE_PASSING[dataset]
    for k, asgn in per_k_assignments.items():
        merged = asgn.merge(existing, on="trade_id", how="inner", validate="one_to_one")
        for arch_id in sorted(merged["archetype_id"].unique().tolist()):
            arch_sub = merged[merged["archetype_id"] == arch_id]
            if len(arch_sub) == 0:
                out[(k, int(arch_id))] = False
                continue
            counts_norm = arch_sub["existing_cluster"].value_counts(normalize=True)
            pct_from_passing = float(counts_norm.get(passing, 0.0))
            out[(k, int(arch_id))] = bool(pct_from_passing >= OVERLAP_MAJORITY_THRESH)
    return out
