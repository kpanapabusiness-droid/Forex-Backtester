"""v2.0 archetype diagnostic — driver.

Runs the full pipeline on KH-24 + Arc 1 + Arc 2, writes all CSVs +
DIAGNOSTIC_REPORT.md under results/v2_0_diagnostic/.

Pure computation — no protocol revision, no floor-setting, no verdicts.

Determinism: pins single-threaded BLAS / OpenMP execution. Multi-threaded
reductions introduce floating-point order non-determinism in KMeans
centroids; pinning to 1 thread gives byte-identical outputs across runs.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.v1_3_calibration.load_paths import load_paths  # noqa: E402, F401
from scripts.v2_0_diagnostic import characterize as CH  # noqa: E402
from scripts.v2_0_diagnostic import cluster as C  # noqa: E402
from scripts.v2_0_diagnostic import entry_features as EF  # noqa: E402
from scripts.v2_0_diagnostic import evidence as EV  # noqa: E402
from scripts.v2_0_diagnostic import labels as LB  # noqa: E402
from scripts.v2_0_diagnostic import overlap as OV  # noqa: E402
from scripts.v2_0_diagnostic import path_features as PF  # noqa: E402
from scripts.v2_0_diagnostic import predictability as PR  # noqa: E402
from scripts.v2_0_diagnostic import report as RP  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT  = REPO_ROOT / "results" / "v2_0_diagnostic"

DATASETS = ("kh24", "arc1", "arc2")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, lineterminator="\n")


def run() -> None:
    print(f"[v2.0-diagnostic] writing to {OUT_ROOT}")
    _ensure_dir(OUT_ROOT)

    per_dataset: dict[str, dict] = {}

    for name in DATASETS:
        print(f"\n--- {name} ---")
        out_dir = OUT_ROOT / name
        _ensure_dir(out_dir)

        # Step 1: path-shape features (compute directly to avoid re-loading).
        paths, meta = load_paths(name)
        print(f"  loaded {len(meta)} trades, {len(paths)} per-bar rows")
        features  = PF.compute_path_features(paths, meta)
        feat_dist = PF.summarise_distributions(features)
        degen     = PF.degenerate_flags(features)
        _write_csv(features,  out_dir / "path_features.csv")
        _write_csv(feat_dist, out_dir / "path_features_distributions.csv")
        _write_csv(degen,     out_dir / "path_features_degeneracy.csv")
        print("  path-shape features written")

        # Step 4 (a): entry features
        print("  computing entry features...")
        entry_feats = EF.compute_entry_features(name)
        _write_csv(entry_feats, out_dir / "entry_features_basic.csv")

        # Steps 2 + 3 + 4(b): cluster + characterise + predictability per K
        per_k_runs: dict[int, dict] = {}
        per_k_assignments: dict[int, pd.DataFrame] = {}
        for k in C.K_VALUES:
            assignments, centroids, sil, failed = C.cluster_one(features, k)
            per_k_assignments[k] = assignments
            _write_csv(assignments, out_dir / f"clusters_K{k}.csv")
            _write_csv(centroids,   out_dir / f"centroids_K{k}.csv")
            (out_dir / f"silhouette_K{k}.txt").write_text(f"{sil:.10f}\n", encoding="utf-8", newline="\n")

            summary, side_dists = CH.characterise(
                paths, meta, features, assignments, centroids, k,
            )
            _write_csv(summary, out_dir / f"archetype_summaries_K{k}.csv")
            for arch_id, dist in side_dists.items():
                _write_csv(dist, out_dir / f"archetype_{arch_id}_K{k}_distribution.csv")

            pred = PR.per_archetype_auc(entry_feats, assignments, k)
            _write_csv(pred, out_dir / f"predictability_K{k}.csv")

            lbls = LB.label_all(summary)

            per_k_runs[k] = {
                "assignments":   assignments,
                "centroids":     centroids,
                "silhouette":    sil,
                "failed":        failed,
                "summary":       summary,
                "side_dists":    side_dists,
                "predictability": pred,
                "labels":        lbls,
            }
            print(f"    K={k}: silhouette={sil:.4f}, failed={failed}")

        # Step 5: overlap (arc1, arc2 only)
        overlap_flags: dict[tuple[int, int], bool] = {}
        if name in OV.DUAL_GATE_PASSING:
            existing = OV.load_existing_clusters(name)
            for k, asgn in per_k_assignments.items():
                cm = OV.confusion_matrix(asgn, existing)
                _write_csv(cm.reset_index(), out_dir / f"overlap_matrix_K{k}.csv")
            os_df = OV.overlap_summary(per_k_assignments, existing, name)
            _write_csv(os_df, out_dir / "overlap_summary.csv")
            overlap_flags = OV.archetype_majority_overlap_flags(
                per_k_assignments, existing, name,
            )
        else:
            os_df = pd.DataFrame()

        # Inject overlap into per-K runs for evidence step.
        for k in C.K_VALUES:
            per_k_runs[k]["overlap_flags"] = {
                kv: v for kv, v in overlap_flags.items() if kv[0] == k
            }

        per_dataset[name] = {
            "n_trades":       int(len(meta)),
            "feature_dist":   feat_dist,
            "degen":          degen,
            "K":              per_k_runs,
            "overlap_summary": os_df,
        }

    # Step 6: v2.0 evidence flags
    evidence = EV.build_evidence(per_dataset)
    _write_csv(evidence, OUT_ROOT / "v2_0_evidence_flags.csv")
    n_qual = int(evidence["qualifies_as_v2_0_evidence"].sum())
    print(f"\n[v2.0-diagnostic] v2.0 evidence: {n_qual} qualifying tuple(s)")

    # Step 7: report
    RP.write_report(OUT_ROOT, per_dataset, evidence)
    print(f"[v2.0-diagnostic] report written: {OUT_ROOT / 'DIAGNOSTIC_REPORT.md'}")


if __name__ == "__main__":
    run()
