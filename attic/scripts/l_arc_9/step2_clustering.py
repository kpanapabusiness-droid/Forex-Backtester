"""Arc 9 - Step 2 path-shape clustering (L_ARC_PROTOCOL §6).

Reads results/l_arc_9/step1_verbatim/{trades_all,trades_paths}.csv.
Operates on held-window bars only (is_held=1) per the path-shape feature
definitions in §6.

Produces in results/l_arc_9/step2_clustering/:
    path_features.csv          per-trade four-feature row + bars_held + final_r
    silhouette_summary.csv     K, silhouette, sizes, gate result
    silhouette_K<k>.txt        per-K silhouette + size detail
    clusters_K<k>.csv          trade_id, cluster_id at K=k
    centroids_K<k>.csv         per-cluster centroid + size
    archetype_assignments.csv  per-cluster archetype-pattern match at the selected K
    STEP2_SUMMARY.md           verdict + chosen K + archetype routing

Gate (§6): silhouette >= 0.30, no cluster > 90%, all clusters >= 30.
K selection: highest silhouette satisfying gate; smaller K wins within 0.01.
Degenerate features: 2+ features with >80% at single value => halt (KILL).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


K_RANGE: List[int] = [3, 4, 5, 6, 7]
GATE_SILHOUETTE: float = 0.30
GATE_MAX_CLUSTER_FRAC: float = 0.90
GATE_MIN_CLUSTER_N: int = 30
K_TIE_TOLERANCE: float = 0.01
DEGEN_FRAC: float = 0.80
DEGEN_MAX_OK: int = 1  # >= 2 degenerate features => halt
PATH_FEATURE_COLS: List[str] = [
    "monotonicity_ratio_in_profit",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
]


def _monotonicity_ratio_in_profit(close_r: np.ndarray) -> float:
    """Among bars where close_r > 0, fraction where close_r >= previous in-profit bar."""
    in_profit = close_r[close_r > 0]
    if in_profit.size < 2:
        return 0.0
    return float(np.mean(in_profit[1:] >= in_profit[:-1]))


def _local_peaks_count(mfe_so_far_r: np.ndarray) -> int:
    """Count of bars where mfe_so_far_r > previous bar."""
    if mfe_so_far_r.size < 2:
        return 0
    return int(np.sum(mfe_so_far_r[1:] > mfe_so_far_r[:-1]))


def _pullback_magnitude_median(mfe_so_far_r: np.ndarray, close_r: np.ndarray) -> float:
    """Operational definition per §6 (per Open-08 closure):
       For consecutive bars where mfe_so_far_r increases (i.e. new peak observed),
       pullback = (earlier peak's mfe_so_far_r) - min(close_r between peaks).
       Median across consecutive peak pairs. 0 if <2 peaks.
    """
    if mfe_so_far_r.size < 2:
        return 0.0
    # Bars where mfe_so_far_r strictly increases vs previous bar.
    is_new_peak = np.zeros(mfe_so_far_r.size, dtype=bool)
    is_new_peak[1:] = mfe_so_far_r[1:] > mfe_so_far_r[:-1]
    is_new_peak[0] = True  # bar 0 anchors as first "peak" baseline
    peak_idx = np.where(is_new_peak)[0]
    if peak_idx.size < 2:
        return 0.0
    pullbacks: List[float] = []
    for i in range(peak_idx.size - 1):
        a = int(peak_idx[i])
        b = int(peak_idx[i + 1])
        earlier_peak_mfe = float(mfe_so_far_r[a])
        # min close_r between peaks (exclusive of endpoints? inclusive of earlier, exclusive of later).
        if b > a + 1:
            seg = close_r[a + 1:b]
            if seg.size == 0:
                continue
            dip = float(seg.min())
        else:
            # adjacent peaks; no dip between them.
            continue
        pullbacks.append(earlier_peak_mfe - dip)
    if not pullbacks:
        return 0.0
    return float(np.median(pullbacks))


def _time_to_peak_mfe_relative(
    mfe_so_far_r: np.ndarray, bars_held: int
) -> float:
    """time_to_peak_mfe / max(bars_held, 1), capped at 1.0.

    time_to_peak_mfe = bar_offset where mfe_so_far_r first reaches its max
    within the held window. If never in profit (max <= 0) -> 0.
    """
    if mfe_so_far_r.size == 0:
        return 0.0
    peak_value = float(mfe_so_far_r.max())
    if peak_value <= 0:
        return 0.0
    ttp = int(np.argmax(mfe_so_far_r))
    denom = max(bars_held, 1)
    return float(min(ttp / denom, 1.0))


def compute_path_features(trades_paths: pd.DataFrame, trades_all: pd.DataFrame) -> pd.DataFrame:
    """One row per trade with the four §6 features computed over held bars only."""
    held = trades_paths[trades_paths["is_held"] == 1].copy()
    held = held.sort_values(["trade_id", "bar_offset"], kind="mergesort").reset_index(drop=True)
    out_rows: List[Dict[str, Any]] = []
    for tid, sub in held.groupby("trade_id", sort=True):
        close_r = sub["close_r"].to_numpy(dtype=float)
        mfe_so_far_r = sub["mfe_so_far_r"].to_numpy(dtype=float)
        bars_held = int(len(sub) - 1)  # bar 0 to bar (n-1)
        mono = _monotonicity_ratio_in_profit(close_r)
        lp = _local_peaks_count(mfe_so_far_r)
        pull = _pullback_magnitude_median(mfe_so_far_r, close_r)
        ttp_rel = _time_to_peak_mfe_relative(mfe_so_far_r, bars_held)
        out_rows.append(
            {
                "trade_id": int(tid),
                "monotonicity_ratio_in_profit": mono,
                "local_peaks_count": int(lp),
                "pullback_magnitude_median": float(pull),
                "time_to_peak_mfe_relative": ttp_rel,
                "bars_held": bars_held,
            }
        )
    df = pd.DataFrame(out_rows)
    # Attach final_r for downstream reporting (not used for clustering).
    df = df.merge(
        trades_all[["trade_id", "final_r", "mfe_r", "mae_r", "exit_reason"]],
        on="trade_id", how="left",
    )
    return df.sort_values("trade_id").reset_index(drop=True)


def _degenerate_features(features: pd.DataFrame) -> Dict[str, Any]:
    flagged: List[Dict[str, Any]] = []
    for col in PATH_FEATURE_COLS:
        vals = features[col].to_numpy()
        if vals.size == 0:
            continue
        # Modal mass: fraction at the most-common rounded value (2 decimals).
        rounded = np.round(vals, 3)
        uniq, cnt = np.unique(rounded, return_counts=True)
        max_frac = float(cnt.max()) / float(vals.size)
        flagged.append({"feature": col, "max_modal_frac": max_frac, "degenerate": max_frac > DEGEN_FRAC})
    n_degen = sum(1 for f in flagged if f["degenerate"])
    return {"per_feature": flagged, "n_degenerate": n_degen, "halt": n_degen >= 2}


def _evaluate_k(
    X_scaled: np.ndarray, k: int, seed: int = 42
) -> Dict[str, Any]:
    km = KMeans(
        n_clusters=k, random_state=seed, n_init=10, max_iter=300,
        algorithm="lloyd",
    )
    labels = km.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels, random_state=seed)) if k > 1 else float("nan")
    sizes = pd.Series(labels).value_counts().sort_index()
    sizes_dict = {int(c): int(n) for c, n in sizes.items()}
    n_total = int(len(labels))
    max_frac = max(sizes_dict.values()) / n_total
    min_n = min(sizes_dict.values())
    gate_pass = (
        sil >= GATE_SILHOUETTE and
        max_frac <= GATE_MAX_CLUSTER_FRAC and
        min_n >= GATE_MIN_CLUSTER_N
    )
    return {
        "k": int(k),
        "silhouette": sil,
        "sizes": sizes_dict,
        "max_cluster_frac": float(max_frac),
        "min_cluster_n": int(min_n),
        "n_total": n_total,
        "labels": labels.tolist(),
        "centroids_scaled": km.cluster_centers_.tolist(),
        "gate_pass": bool(gate_pass),
        "gate_fail_reason": (
            None if gate_pass else
            ("silhouette" if sil < GATE_SILHOUETTE
             else "max_cluster_frac" if max_frac > GATE_MAX_CLUSTER_FRAC
             else "min_cluster_n")
        ),
    }


def _select_k(per_k: List[Dict[str, Any]]) -> Optional[int]:
    passing = [r for r in per_k if r["gate_pass"]]
    if not passing:
        return None
    best_sil = max(r["silhouette"] for r in passing)
    # Smaller K wins within tolerance.
    eligible = [r for r in passing if (best_sil - r["silhouette"]) <= K_TIE_TOLERANCE]
    eligible.sort(key=lambda r: r["k"])
    return int(eligible[0]["k"])


# §11 archetype patterns. Centroid pattern match against the four features.
# Order matters: first match wins. v2.1.2 Stepwise local_peaks ceiling = 5..50.
ARCHETYPE_PATTERNS: List[Dict[str, Any]] = [
    {
        "label": "Monotone ascent",
        "rule": lambda c: (c["mono"] >= 0.55 and c["lp"] <= 4 and c["ttp_rel"] >= 0.50),
    },
    {
        "label": "Stepwise climber",
        "rule": lambda c: (c["mono"] >= 0.50 and 5 <= c["lp"] <= 50 and c["pull"] <= 0.5 and c["ttp_rel"] >= 0.50),
    },
    {
        "label": "Early-peak hold",
        "rule": lambda c: (c["ttp_rel"] <= 0.30),
        # Differentiator (pct_peak_and_collapse) deferred to Step 3 evaluation.
    },
    {
        "label": "Random walk",
        "rule": lambda c: (c["lp"] >= 8 and c["mono"] <= 0.30 and c["pull"] >= 1.0),
    },
]


def _archetype_label(centroid: Dict[str, float]) -> str:
    for row in ARCHETYPE_PATTERNS:
        if row["rule"](centroid):
            return row["label"]
    return "Unclassified"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 Step 2 clustering.")
    parser.add_argument(
        "--in-dir", type=Path,
        default=_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_REPO_ROOT / "results" / "l_arc_9" / "step2_clustering",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trades_all = pd.read_csv(args.in_dir / "trades_all.csv")
    trades_paths = pd.read_csv(args.in_dir / "trades_paths.csv")

    features = compute_path_features(trades_paths, trades_all)
    features.to_csv(args.out_dir / "path_features.csv", index=False,
                    float_format="%.10g", lineterminator="\n")

    degen = _degenerate_features(features)
    if degen["halt"]:
        verdict = "KILL"
        summary = {
            "verdict": verdict,
            "reason": f"{degen['n_degenerate']} path features have >{int(DEGEN_FRAC * 100)}% modal mass",
            "degenerate_features": degen["per_feature"],
        }
        (args.out_dir / "step2_halt.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[step 2] HALT - degenerate features: {summary['reason']}")
        return 1

    # K-sweep.
    X = features[PATH_FEATURE_COLS].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    per_k: List[Dict[str, Any]] = []
    sil_rows: List[Dict[str, Any]] = []
    for k in K_RANGE:
        rec = _evaluate_k(X_scaled, k)
        per_k.append(rec)
        sil_rows.append({
            "k": rec["k"], "silhouette": rec["silhouette"],
            "n_clusters": k,
            "max_cluster_frac": rec["max_cluster_frac"],
            "min_cluster_n": rec["min_cluster_n"],
            "gate_pass": int(rec["gate_pass"]),
            "gate_fail_reason": rec["gate_fail_reason"] or "",
            "sizes": json.dumps(rec["sizes"]),
        })
        # Per-K text + clusters_K<k>.csv
        with (args.out_dir / f"silhouette_K{k}.txt").open("w", encoding="utf-8") as f:
            f.write(f"K={k}\n")
            f.write(f"silhouette: {rec['silhouette']:.4f}\n")
            f.write(f"sizes: {rec['sizes']}\n")
            f.write(f"max_cluster_frac: {rec['max_cluster_frac']:.4f}\n")
            f.write(f"min_cluster_n: {rec['min_cluster_n']}\n")
            f.write(f"gate_pass: {rec['gate_pass']}\n")
            if not rec["gate_pass"]:
                f.write(f"gate_fail_reason: {rec['gate_fail_reason']}\n")
        clusters_df = pd.DataFrame({"trade_id": features["trade_id"], "cluster_id": rec["labels"]})
        clusters_df.to_csv(
            args.out_dir / f"clusters_K{k}.csv", index=False,
            float_format="%.10g", lineterminator="\n",
        )
        # Centroids in raw (un-scaled) feature space — interpretability for §11.
        labels_arr = np.array(rec["labels"])
        cent_rows: List[Dict[str, Any]] = []
        for c in sorted(set(labels_arr.tolist())):
            mask = labels_arr == c
            n = int(mask.sum())
            mean_features = X[mask].mean(axis=0)
            cent_rows.append(
                {
                    "cluster_id": int(c),
                    "n": n,
                    "frac_of_pool": round(n / len(labels_arr), 4),
                    "monotonicity_ratio_in_profit": float(mean_features[0]),
                    "local_peaks_count": float(mean_features[1]),
                    "pullback_magnitude_median": float(mean_features[2]),
                    "time_to_peak_mfe_relative": float(mean_features[3]),
                }
            )
        cents_df = pd.DataFrame(cent_rows)
        cents_df.to_csv(
            args.out_dir / f"centroids_K{k}.csv", index=False,
            float_format="%.10g", lineterminator="\n",
        )

    pd.DataFrame(sil_rows).to_csv(
        args.out_dir / "silhouette_summary.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )

    k_sel = _select_k(per_k)
    if k_sel is None:
        # No K passes the gate -> arc dies at Step 2.
        verdict = "KILL"
        passing_k = [r["k"] for r in per_k if r["gate_pass"]]
        fail_reasons = {r["k"]: r["gate_fail_reason"] for r in per_k}
        summary = {
            "verdict": verdict,
            "reason": "no K passes silhouette/size gate at any K",
            "per_k_silhouette": {r["k"]: r["silhouette"] for r in per_k},
            "per_k_gate": {r["k"]: r["gate_pass"] for r in per_k},
            "per_k_fail_reason": fail_reasons,
            "passing_k": passing_k,
        }
        (args.out_dir / "step2_halt.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[step 2] KILL - {summary['reason']}")
        return 1

    # Archetype assignment at selected K.
    sel_rec = next(r for r in per_k if r["k"] == k_sel)
    labels_arr = np.array(sel_rec["labels"])
    arch_rows: List[Dict[str, Any]] = []
    for c in sorted(set(labels_arr.tolist())):
        mask = labels_arr == c
        n = int(mask.sum())
        mean_features = X[mask].mean(axis=0)
        centroid = {
            "mono": float(mean_features[0]),
            "lp": float(mean_features[1]),
            "pull": float(mean_features[2]),
            "ttp_rel": float(mean_features[3]),
        }
        label = _archetype_label(centroid)
        arch_rows.append(
            {
                "cluster_id": int(c),
                "n": n,
                "frac_of_pool": round(n / len(labels_arr), 4),
                "monotonicity_ratio_in_profit": centroid["mono"],
                "local_peaks_count": centroid["lp"],
                "pullback_magnitude_median": centroid["pull"],
                "time_to_peak_mfe_relative": centroid["ttp_rel"],
                "archetype_label": label,
            }
        )
    arch_df = pd.DataFrame(arch_rows)
    arch_df.to_csv(
        args.out_dir / "archetype_assignments.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )

    # Markdown summary.
    md: List[str] = []
    md.append("# Arc 9 Step 2 - Clustering")
    md.append("")
    md.append(f"Verdict: **PASS** (chosen K = {k_sel})")
    md.append("")
    md.append("## K-sweep (silhouette + gate)")
    md.append("")
    md.append("| K | silhouette | max_cluster_frac | min_cluster_n | gate | fail_reason |")
    md.append("|---|---|---|---|---|---|")
    for r in per_k:
        md.append(
            f"| {r['k']} | {r['silhouette']:.4f} | {r['max_cluster_frac']:.4f} | "
            f"{r['min_cluster_n']} | {'PASS' if r['gate_pass'] else 'FAIL'} | "
            f"{r['gate_fail_reason'] or '-'} |"
        )
    md.append("")
    md.append(f"K selection rule: highest silhouette satisfying gate, smaller K wins within {K_TIE_TOLERANCE} absolute. **Selected K = {k_sel}**.")
    md.append("")
    md.append(f"## Archetype assignment at K={k_sel}")
    md.append("")
    md.append("| cluster_id | n | frac | mono | local_peaks | pullback | ttp_rel | archetype |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in arch_rows:
        md.append(
            f"| {r['cluster_id']} | {r['n']} | {r['frac_of_pool']:.3f} | "
            f"{r['monotonicity_ratio_in_profit']:.3f} | {r['local_peaks_count']:.2f} | "
            f"{r['pullback_magnitude_median']:.3f} | {r['time_to_peak_mfe_relative']:.3f} | "
            f"**{r['archetype_label']}** |"
        )
    md.append("")
    md.append("## Path-feature degeneracy")
    md.append("")
    md.append("| feature | max_modal_frac | degenerate (>0.80) |")
    md.append("|---|---|---|")
    for d in degen["per_feature"]:
        md.append(f"| {d['feature']} | {d['max_modal_frac']:.3f} | {'yes' if d['degenerate'] else 'no'} |")
    md.append("")
    md.append(f"Degenerate count: {degen['n_degenerate']} (halt at >= 2)")
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append(f"- path_features.csv ({len(features)} rows)")
    md.append("- silhouette_summary.csv")
    md.append("- silhouette_K<k>.txt (K=3..7)")
    md.append("- clusters_K<k>.csv (K=3..7)")
    md.append("- centroids_K<k>.csv (K=3..7)")
    md.append("- archetype_assignments.csv (at K=" + str(k_sel) + ")")

    (args.out_dir / "STEP2_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[step 2] PASS - selected K={k_sel}")
    for r in arch_rows:
        print(f"  cluster {r['cluster_id']}: n={r['n']} ({r['frac_of_pool']:.1%}) -> {r['archetype_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
