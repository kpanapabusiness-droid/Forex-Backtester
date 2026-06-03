"""KH-24 v2.0 self-test, Step 2 — path-shape clustering.

Reads Step 1's `trades_all.csv` + `trades_paths.csv`, computes the four §6
path-shape features per trade, runs the §6 K sweep with KMeans
(random_state=42, n_init=10, max_iter=300) on StandardScaler-normalised
features, evaluates the §6 gate, selects K, and matches each cluster's
centroid against §11 archetype patterns.

Outputs (under `results/arc_kh24_v2/step2/`):
    path_features.csv
    clusters_K3.csv ... clusters_K7.csv
    centroids_K3.csv ... centroids_K7.csv
    silhouette_K3.txt ... silhouette_K7.txt
    archetype_assignments.csv     (selected K only)
    step2_report.md

Usage:
    python -m scripts.arc_kh24_v2.step2.run_step2
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.arc_kh24_v2.step2._archetype import match_archetype  # noqa: E402
from scripts.arc_kh24_v2.step2._cluster import (  # noqa: E402
    GATE_MAX_CLUSTER_FRACTION,
    GATE_MIN_CLUSTER_SIZE,
    GATE_MIN_SILHOUETTE,
    K_SWEEP,
    evaluate_gate,
    run_kmeans_sweep,
    select_k,
)
from scripts.arc_kh24_v2.step2._degeneracy import (  # noqa: E402
    DEGENERACY_THRESHOLD,
    check_degeneracy,
)
from scripts.arc_kh24_v2.step2._features import (  # noqa: E402
    FEATURE_COLUMNS,
    compute_features_for_all_trades,
)

DEFAULT_STEP1_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step1"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step2"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n")


def _write_text(text: str, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def run(step1_dir: Path, out_dir: Path) -> dict:
    """Run Step 2 end-to-end. Returns a result dict for the report writer."""
    trades_df = pd.read_csv(step1_dir / "trades_all.csv")
    paths_df = pd.read_csv(step1_dir / "trades_paths.csv")

    # 1. Path-shape features.
    features_df = compute_features_for_all_trades(paths_df, trades_df)
    _write_csv(features_df, out_dir / "path_features.csv")

    # 2. Degeneracy check.
    degen = check_degeneracy(features_df)

    if degen.halt:
        # Halt before clustering — write features + report; no cluster artefacts.
        result = {
            "n_trades": len(features_df),
            "features_df": features_df,
            "degeneracy": degen,
            "cluster_results": None,
            "gates": None,
            "selected_k": None,
            "archetype_rows": None,
            "halt_reason": "STEP2_HALT_DEGENERATE",
        }
        return result

    # 3. KMeans sweep.
    cluster_results, scaler = run_kmeans_sweep(features_df, ks=K_SWEEP)

    # 4. Per-K artefacts.
    for k, res in cluster_results.items():
        clusters_df = pd.DataFrame(
            {"trade_id": features_df["trade_id"].values, "cluster_id": res.labels}
        )
        _write_csv(clusters_df, out_dir / f"clusters_K{k}.csv")

        centroid_rows = []
        for cid in range(k):
            row = {
                "cluster_id": cid,
                "cluster_size": int(res.cluster_sizes[cid]),
                "cluster_fraction": float(res.cluster_fractions[cid]),
            }
            for j, name in enumerate(FEATURE_COLUMNS):
                row[f"centroid_{_short_name(name)}"] = float(res.centroids_unstd[cid, j])
            for j, name in enumerate(FEATURE_COLUMNS):
                row[f"centroid_{_short_name(name)}_std"] = float(res.centroids_std[cid, j])
            centroid_rows.append(row)
        _write_csv(pd.DataFrame(centroid_rows), out_dir / f"centroids_K{k}.csv")

        _write_text(f"{res.silhouette:.10f}\n", out_dir / f"silhouette_K{k}.txt")

    # 5. §6 gate per K.
    gates = {k: evaluate_gate(res) for k, res in cluster_results.items()}
    selected_k = select_k(gates)

    archetype_rows: list[dict] | None = None
    if selected_k is not None:
        sel = cluster_results[selected_k]
        archetype_rows = []
        for cid in range(selected_k):
            cen = sel.centroids_unstd[cid]
            mono, peaks, pullback, ttp = (
                float(cen[0]),
                float(cen[1]),
                float(cen[2]),
                float(cen[3]),
            )
            label = match_archetype(mono, peaks, pullback, ttp)
            archetype_rows.append(
                {
                    "cluster_id": cid,
                    "cluster_size": int(sel.cluster_sizes[cid]),
                    "archetype_label": label.label,
                    "labelling_method": label.method,
                    "notes": label.notes,
                }
            )
        _write_csv(pd.DataFrame(archetype_rows), out_dir / "archetype_assignments.csv")

    return {
        "n_trades": len(features_df),
        "features_df": features_df,
        "degeneracy": degen,
        "cluster_results": cluster_results,
        "gates": gates,
        "selected_k": selected_k,
        "archetype_rows": archetype_rows,
        "halt_reason": None if (selected_k is not None) else "STEP2_FAIL_NO_VALID_K",
    }


def _short_name(feature_name: str) -> str:
    return {
        "monotonicity_ratio_in_profit": "monotonicity",
        "local_peaks_count": "local_peaks",
        "pullback_magnitude_median": "pullback",
        "time_to_peak_mfe_relative": "time_to_peak_rel",
    }[feature_name]


def write_report(result: dict, out_path: Path, *, determinism_status: str = "PENDING") -> None:
    """Render step2_report.md per §6 + the prompt's spec."""
    n = result["n_trades"]
    features_df = result["features_df"]
    degen = result["degeneracy"]
    cluster_results = result["cluster_results"]
    gates = result["gates"]
    selected_k = result["selected_k"]
    archetype_rows = result["archetype_rows"]
    halt = result["halt_reason"]

    lines: list[str] = []
    lines.append("# KH-24 v2.0 Step 2 — Path-shape clustering report")
    lines.append("")
    lines.append(
        "Generated by `scripts/arc_kh24_v2/step2/run_step2.py`; reads Step 1 outputs "
        "from `results/arc_kh24_v2/step1/`."
    )
    lines.append("")
    lines.append(f"- Pool size: **{n}** trades (Step 1 output)")
    lines.append(f"- K sweep: {{{', '.join(str(k) for k in K_SWEEP)}}}")
    lines.append(
        f"- §6 gate: silhouette ≥ {GATE_MIN_SILHOUETTE:.2f}, no cluster > "
        f"{int(GATE_MAX_CLUSTER_FRACTION * 100)}%, all clusters ≥ {GATE_MIN_CLUSTER_SIZE}"
    )
    lines.append("")

    # Gate summary
    lines.append("## §6 gate summary")
    lines.append("")
    if halt == "STEP2_HALT_DEGENERATE":
        lines.append(
            "**HALT — STEP2_HALT_DEGENERATE.** Two or more features degenerate (>80% at one value)."
        )
    elif halt == "STEP2_FAIL_NO_VALID_K":
        lines.append(
            "**FAIL — STEP2_FAIL_NO_VALID_K.** No K in the sweep passes all three §6 gate criteria."
        )
    else:
        lines.append(f"**PASS — selected K = {selected_k}.**")
    lines.append("")

    # Feature distributions
    lines.append("## Path-shape feature distributions (per §1 full-distribution discipline)")
    lines.append("")
    lines.append("| Feature | p5 | p25 | p50 | p75 | p95 | mean | std |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for col in FEATURE_COLUMNS:
        s = features_df[col]
        lines.append(
            f"| `{col}` | "
            f"{s.quantile(0.05):.3f} | {s.quantile(0.25):.3f} | "
            f"{s.quantile(0.50):.3f} | {s.quantile(0.75):.3f} | "
            f"{s.quantile(0.95):.3f} | {s.mean():.3f} | {s.std():.3f} |"
        )
    lines.append("")

    # Degeneracy
    lines.append("## Degenerate-feature check (§6 + §16 Open-08)")
    lines.append("")
    lines.append(
        f"Threshold: > {int(DEGENERACY_THRESHOLD * 100)}% of trades at the single most-common value "
        "(rounded to 4dp). Single-feature flag is a NOTE; two+ feature flags HALT the arc."
    )
    lines.append("")
    lines.append("| Feature | mode fraction | Flagged (> 80%) |")
    lines.append("|---|---:|:---:|")
    for col in FEATURE_COLUMNS:
        frac = degen.mode_fractions.get(col, 0.0)
        flag = "FLAG" if col in degen.flags else ""
        lines.append(f"| `{col}` | {frac:.4f} | {flag} |")
    lines.append("")
    if degen.halt:
        lines.append(
            "**Multi-feature degeneracy → STEP2_HALT_DEGENERATE. Cluster artefacts NOT written.**"
        )
        lines.append("")
        _write_text("\n".join(lines) + "\n", out_path)
        return
    if degen.flags:
        flag_names = ", ".join(f"`{n}` ({f:.1%})" for n, f in degen.flags.items())
        lines.append(
            f"Single-feature degeneracy noted ({flag_names}). Per §6 + §16 Open-08 this is "
            "a flag, not a halt — the arc advances. The flagged feature contributes weakly "
            "to cluster separation and is expected to refine post-Step-2 (Open-08 tracks "
            "the `pullback_magnitude_median` rethink)."
        )
        lines.append("")

    # Per-K gate table
    lines.append("## Per-K silhouette + gate evaluation")
    lines.append("")
    lines.append("| K | silhouette | max cluster fraction | min cluster size | gate |")
    lines.append("|---:|---:|---:|---:|:---:|")
    for k in K_SWEEP:
        g = gates[k]
        verdict = "PASS" if g.passes else "FAIL"
        lines.append(
            f"| {k} | {g.silhouette:.4f} | {g.max_cluster_fraction:.4f} | "
            f"{g.min_cluster_size} | {verdict} |"
        )
    lines.append("")

    if selected_k is None:
        lines.append("No K passes the §6 gate. STEP2_FAIL_NO_VALID_K. Arc halts.")
        lines.append("")
        _write_text("\n".join(lines) + "\n", out_path)
        return

    lines.append(
        f"K selection: highest silhouette among passing K with tie-break = smaller K (parsimony). "
        f"Selected **K = {selected_k}** (silhouette = {gates[selected_k].silhouette:.4f})."
    )
    lines.append("")

    # Selected-K cluster table
    sel = cluster_results[selected_k]
    lines.append(f"## Cluster centroids at K = {selected_k}")
    lines.append("")
    lines.append(
        "| cluster | size | fraction | monotonicity | local_peaks | pullback | time_to_peak_rel | archetype |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for cid in range(selected_k):
        cen = sel.centroids_unstd[cid]
        archetype = archetype_rows[cid]["archetype_label"]
        lines.append(
            f"| {cid} | {int(sel.cluster_sizes[cid])} | {sel.cluster_fractions[cid]:.4f} | "
            f"{cen[0]:.3f} | {cen[1]:.2f} | {cen[2]:.3f} | {cen[3]:.3f} | `{archetype}` |"
        )
    lines.append("")

    # Archetype labelling details
    lines.append(f"## §11 archetype assignments at K = {selected_k}")
    lines.append("")
    lines.append("| cluster | size | archetype | method | notes |")
    lines.append("|---:|---:|---|---|---|")
    for r in archetype_rows:
        lines.append(
            f"| {r['cluster_id']} | {r['cluster_size']} | `{r['archetype_label']}` | "
            f"{r['labelling_method']} | {r['notes']} |"
        )
    lines.append("")
    unresolved = [r for r in archetype_rows if r["labelling_method"] != "centroid_match"]
    if unresolved:
        lines.append(
            f"{len(unresolved)} of {selected_k} clusters carry a non-`centroid_match` label "
            "and are deferred to Step 3 per §11 boundary rule (empirical per-fold capture-ratio "
            "test) or because the archetype rule depends on Step-3-computed quantities "
            "(`pct_peak_and_collapse` for early-peak family; distribution shape for V-shape / "
            "Split-exit). Cluster-level granularity is retained in `clusters_K*.csv`; "
            "same-archetype aggregation is a Step 3+ concern (§6)."
        )
        lines.append("")

    # Determinism + reproducibility
    lines.append("## Determinism")
    lines.append("")
    lines.append(
        f"Two-run byte-identical CSV outputs: **{determinism_status}**. "
        "KMeans hyperparameters are exact per §6: `random_state=42`, `n_init=10`, "
        "`max_iter=300`. StandardScaler is fit once on the full 842-trade pool. "
        "See `tests/arc_kh24_v2/test_step2_determinism.py` for the CI-enforced test."
    )
    lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"§6 gate: **PASS** at K = {selected_k}.")
    lines.append("")
    lines.append("Ready to advance to Step 3 (capturability characterisation).")

    _write_text("\n".join(lines) + "\n", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step1-dir",
        default=str(DEFAULT_STEP1_DIR),
        help="Directory containing Step 1 outputs (default: results/arc_kh24_v2/step1).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory (default: results/arc_kh24_v2/step2).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing step2_report.md (used by the determinism check).",
    )
    parser.add_argument(
        "--determinism-status",
        default="PENDING",
        help="Status to record for determinism in the report (PASS/FAIL/PENDING).",
    )
    args = parser.parse_args(argv)

    step1_dir = Path(args.step1_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    result = run(step1_dir, out_dir)

    if not args.no_report:
        write_report(
            result, out_dir / "step2_report.md", determinism_status=args.determinism_status
        )

    # Print summary to stdout.
    if result["halt_reason"]:
        print(f"Step 2: {result['halt_reason']}")
        return 1
    sel = result["selected_k"]
    sil = result["gates"][sel].silhouette
    print(f"Step 2: PASS — selected K = {sel} (silhouette = {sil:.4f})")
    for k, g in result["gates"].items():
        print(
            f"  K={k}: silhouette={g.silhouette:.4f} maxFrac={g.max_cluster_fraction:.4f} "
            f"minSize={g.min_cluster_size} {'PASS' if g.passes else 'FAIL'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
