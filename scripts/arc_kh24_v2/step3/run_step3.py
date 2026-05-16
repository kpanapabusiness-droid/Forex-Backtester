"""KH-24 v2.0 self-test, Step 3 — capturability characterisation.

Reads Step 1 trades + paths and Step 2 cluster assignments + centroids;
computes per-cluster identity, forward geometry, distribution shape,
`pct_peak_and_collapse`; evaluates the §2 conjunctive capturability
gate; finalises archetype labels (early-peak family split, boundary
disposition); emits the §7 capturability pass list and the report.

Outputs (under `results/arc_kh24_v2/step3/`):
    archetype_summaries.csv
    archetype_<label>_distribution.csv      (one per cluster that needs detail)
    capturability_pass_list.csv
    step3_report.md

Usage:
    python -m scripts.arc_kh24_v2.step3.run_step3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.arc_kh24_v2.step3._archetype_finalize import finalise  # noqa: E402
from scripts.arc_kh24_v2.step3._capturability import evaluate  # noqa: E402
from scripts.arc_kh24_v2.step3._distribution import classify_shape  # noqa: E402
from scripts.arc_kh24_v2.step3._forward_geometry import (  # noqa: E402
    compute_cluster_forward_geometry,
    split_by_cluster,
)

DEFAULT_STEP1_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step1"
DEFAULT_STEP2_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step2"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "arc_kh24_v2" / "step3"


def _write_csv(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n")


def _write_text(text: str, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def run(step1_dir: Path, step2_dir: Path, out_dir: Path) -> dict:
    trades_df = pd.read_csv(step1_dir / "trades_all.csv")
    paths_df = pd.read_csv(step1_dir / "trades_paths.csv")
    clusters_df = pd.read_csv(step2_dir / "clusters_K5.csv")
    centroids_df = pd.read_csv(step2_dir / "centroids_K5.csv").set_index("cluster_id")
    arche_df = pd.read_csv(step2_dir / "archetype_assignments.csv").set_index("cluster_id")
    features_df = pd.read_csv(step2_dir / "path_features.csv")

    pool_size = int(len(trades_df))
    cluster_datas = split_by_cluster(trades_df, paths_df, clusters_df)

    summary_rows: list[dict] = []
    distribution_csvs: dict[int, pd.DataFrame] = {}
    pass_rows: list[dict] = []

    import numpy as np  # local — used for casting

    for cd in cluster_datas:
        fg = compute_cluster_forward_geometry(cd, features_df, pool_size)
        cid = cd.cluster_id

        # Centroid (from Step 2)
        centroid_row = centroids_df.loc[cid]
        mono_centroid = float(centroid_row["centroid_monotonicity"])
        peaks_centroid = float(centroid_row["centroid_local_peaks"])
        pullback_centroid = float(centroid_row["centroid_pullback"])
        ttp_centroid = float(centroid_row["centroid_time_to_peak_rel"])

        # Distribution shape from per-trade final_r
        final_r = cd.trades_df["final_r"].to_numpy(dtype=np.float64)
        shape_res = classify_shape(final_r)

        # Step 2 provisional label
        prov_label = str(arche_df.loc[cid, "archetype_label"])
        prov_method = str(arche_df.loc[cid, "labelling_method"])

        # §2 evaluation uses Definition B of frac_wrong_way (intent-aligned).
        cap = evaluate(
            cluster_id=cid,
            archetype_label=prov_label,
            centroid_monotonicity=mono_centroid,
            centroid_local_peaks=peaks_centroid,
            fwd_mfe_p50=fg["fwd_mfe_p50"],
            frac_reach_1R=fg["frac_reach_1R"],
            frac_wrong_way=fg["frac_wrong_way_def_b"],
            shape_tag=shape_res.tag,
            size_fraction_of_pool=fg["size_fraction_of_pool"],
        )

        final_lbl = finalise(
            provisional_label=prov_label,
            provisional_method=prov_method,
            pct_peak_and_collapse=fg["pct_peak_and_collapse"],
            survives_step2_floors=cap.overall_pass,
        )

        # Stats string for shape_tag — compact, single-line
        s = shape_res.stats
        shape_stats_str = (
            f"n={s.get('n', '?')}; mean={s.get('mean', float('nan')):.3f}; "
            f"std={s.get('std', float('nan')):.3f}; skew={s.get('skew', float('nan')):.3f}; "
            f"p95={s.get('p95', float('nan')):.3f}; "
            f"n_modes={s.get('n_modes_detected', 0)}; "
            f"sep={s.get('max_pair_separation', 0.0):.3f}; "
            f"sec/pri={s.get('secondary_to_primary_mass_ratio')}"
        )

        summary_rows.append(
            {
                "cluster_id": cid,
                "provisional_label_step2": prov_label,
                "provisional_method_step2": prov_method,
                "archetype_label_final": final_lbl.label,
                "labelling_method_final": final_lbl.method,
                "labelling_notes_final": final_lbl.notes,
                "size_count": fg["size_count"],
                "size_fraction_of_pool": fg["size_fraction_of_pool"],
                "centroid_monotonicity": mono_centroid,
                "centroid_local_peaks": peaks_centroid,
                "centroid_pullback": pullback_centroid,
                "centroid_time_to_peak_rel": ttp_centroid,
                "fwd_mfe_p5": fg["fwd_mfe_p5"],
                "fwd_mfe_p25": fg["fwd_mfe_p25"],
                "fwd_mfe_p50": fg["fwd_mfe_p50"],
                "fwd_mfe_p75": fg["fwd_mfe_p75"],
                "fwd_mfe_p95": fg["fwd_mfe_p95"],
                "final_r_mean": fg["final_r_mean"],
                "final_r_std": fg["final_r_std"],
                "final_r_t_stat": fg["final_r_t_stat"],
                "final_r_p5": fg["final_r_p5"],
                "final_r_p25": fg["final_r_p25"],
                "final_r_p50": fg["final_r_p50"],
                "final_r_p75": fg["final_r_p75"],
                "final_r_p95": fg["final_r_p95"],
                "mae_r_p5": fg["mae_r_p5"],
                "mae_r_p50": fg["mae_r_p50"],
                "mae_r_p95": fg["mae_r_p95"],
                "bars_held_p5": fg["bars_held_p5"],
                "bars_held_p50": fg["bars_held_p50"],
                "bars_held_p95": fg["bars_held_p95"],
                "frac_cap_240": fg["frac_cap_240"],
                "frac_reach_1R": fg["frac_reach_1R"],
                "frac_reach_2R": fg["frac_reach_2R"],
                "frac_wrong_way_def_a": fg["frac_wrong_way_def_a"],
                "frac_wrong_way_def_b": fg["frac_wrong_way_def_b"],
                "pct_peak_and_collapse": fg["pct_peak_and_collapse"],
                "shape_tag": shape_res.tag,
                "shape_tag_stats": shape_stats_str,
                "§2_clean_shape_pass": cap.clean_shape_pass,
                "§2_limited_oscillation_pass": cap.limited_oscillation_pass,
                "§2_magnitude_pass": cap.magnitude_pass,
                "§2_direction_pass": cap.direction_pass,
                "§2_shape_pass": cap.shape_pass,
                "§2_size_pass": cap.size_pass,
                "§2_overall_pass": cap.overall_pass,
                "§2_fail_reasons": " | ".join(cap.fail_reasons),
            }
        )

        # Per-cluster distribution CSV — we emit one for every cluster (size
        # is small and visibility into non-survivors is useful too).
        dist = cd.trades_df[
            ["trade_id", "pair", "entry_time", "bars_held", "final_r", "mfe_r", "mae_r"]
        ].copy()
        # Suffix the file name with both the final label and the cluster id so
        # two clusters that share an archetype don't collide.
        distribution_csvs[cid] = dist

        if cap.overall_pass:
            pass_rows.append(
                {
                    "cluster_id": cid,
                    "archetype_label_final": final_lbl.label,
                    "labelling_method_final": final_lbl.method,
                    "size_count": fg["size_count"],
                    "size_fraction_of_pool": fg["size_fraction_of_pool"],
                }
            )

    # Write archetype_summaries.csv
    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id").reset_index(drop=True)
    _write_csv(summary_df, out_dir / "archetype_summaries.csv")

    # Write per-cluster distribution CSVs (one file per cluster).
    for cid, dist_df in distribution_csvs.items():
        # Use the final label from the summary for the filename.
        lbl = summary_df.loc[summary_df["cluster_id"] == cid, "archetype_label_final"].iloc[0]
        # Sanitise — labels are already snake_case_or_hex; just lowercase.
        safe_lbl = str(lbl).lower()
        fname = f"archetype_{safe_lbl}_cluster{cid}_distribution.csv"
        _write_csv(dist_df.sort_values("trade_id").reset_index(drop=True), out_dir / fname)

    # Pass list
    pass_df = pd.DataFrame(
        pass_rows,
        columns=[
            "cluster_id",
            "archetype_label_final",
            "labelling_method_final",
            "size_count",
            "size_fraction_of_pool",
        ],
    )
    _write_csv(pass_df, out_dir / "capturability_pass_list.csv")

    return {
        "summary_df": summary_df,
        "pass_df": pass_df,
        "halt_reason": None if len(pass_df) > 0 else "STEP3_FAIL_NO_CAPTURABLE_ARCHETYPE",
    }


def write_report(result: dict, out_path: Path, *, determinism_status: str = "PENDING") -> None:
    summary_df = result["summary_df"]
    pass_df = result["pass_df"]
    halt = result["halt_reason"]

    lines: list[str] = []
    lines.append("# KH-24 v2.0 Step 3 — Capturability characterisation report")
    lines.append("")
    lines.append(
        "Generated by `scripts/arc_kh24_v2/step3/run_step3.py`. Reads Step 1 "
        "+ Step 2 outputs (cluster assignments at K = 5)."
    )
    lines.append("")
    lines.append(
        "- §2 gate: monotonicity ≥ 0.55, archetype-appropriate local_peaks ceiling, "
        "fwd_mfe_p50 ≥ 1.5R, frac_reach_1R ≥ 0.70, frac_wrong_way ≤ 0.30, "
        "shape_tag ∈ {tight_unimodal, heavy_right_tail}, size_fraction ≥ 0.10. "
        "Conjunctive — all six must pass."
    )
    lines.append("- §7 arc-level: ≥ 1 cluster passes → advance to Step 4. Zero → arc halts.")
    lines.append("")

    if halt:
        lines.append("## §7 arc-level gate")
        lines.append("")
        lines.append(
            "**FAIL — STEP3_FAIL_NO_CAPTURABLE_ARCHETYPE.** Zero clusters cleared the "
            "§2 floors conjunctively. The bare KH-24 signal does not pass v2.0 "
            "capturability at K = 5."
        )
    else:
        lines.append("## §7 arc-level gate")
        lines.append("")
        lines.append(f"**PASS — {len(pass_df)} cluster(s) cleared §2.**")
    lines.append("")

    lines.append("## frac_wrong_way definition disambiguation")
    lines.append("")
    lines.append("Both definitions reported per cluster in `archetype_summaries.csv`:")
    lines.append("")
    lines.append("- **Definition A** — `final_r <= -0.5` (trade ended materially adverse).")
    lines.append(
        "- **Definition B** — wrong-from-outset: `mae_so_far_r <= -1.0` occurred before any "
        "`mfe_so_far_r > 0.5` was reached; OR `mfe_so_far_r > 0.5` was never reached."
    )
    lines.append("")
    lines.append(
        "§2 evaluation uses **Definition B** because it matches the protocol's intent "
        '("wrong from the outset") and is consistent with the KH-24 anchor '
        "`frac_wrong_way = 0.04` from §14 — only Definition B can produce numbers that "
        "small on a hard-SL-only 1R-anchored design. Definition A is reported as a "
        "diagnostic. Chat may override the choice via a config switch in a follow-up."
    )
    lines.append("")

    lines.append("## Per-cluster §2 evaluation")
    lines.append("")
    lines.append(
        "| cid | label_final | size | mono | peaks | fwd_mfe_p50 | reach_1R | wrong_way_b | shape_tag | §2 |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|:---:|")
    for _, r in summary_df.iterrows():
        verdict = "PASS" if r["§2_overall_pass"] else "FAIL"
        lines.append(
            f"| {int(r['cluster_id'])} | `{r['archetype_label_final']}` | {int(r['size_count'])} | "
            f"{r['centroid_monotonicity']:.3f} | {r['centroid_local_peaks']:.2f} | "
            f"{r['fwd_mfe_p50']:.3f} | {r['frac_reach_1R']:.3f} | "
            f"{r['frac_wrong_way_def_b']:.3f} | `{r['shape_tag']}` | {verdict} |"
        )
    lines.append("")

    lines.append("## Per-cluster §2 criterion detail")
    lines.append("")
    lines.append(
        "| cid | clean_shape | limited_osc | magnitude | direction | shape | size | fail_reasons |"
    )
    lines.append("|---:|:---:|:---:|:---:|:---:|:---:|:---:|---|")
    for _, r in summary_df.iterrows():

        def _b(x):
            return "✓" if bool(x) else "✗"

        lines.append(
            f"| {int(r['cluster_id'])} | {_b(r['§2_clean_shape_pass'])} | "
            f"{_b(r['§2_limited_oscillation_pass'])} | {_b(r['§2_magnitude_pass'])} | "
            f"{_b(r['§2_direction_pass'])} | {_b(r['§2_shape_pass'])} | "
            f"{_b(r['§2_size_pass'])} | {r['§2_fail_reasons']} |"
        )
    lines.append("")

    lines.append("## Forward-geometry detail")
    lines.append("")
    lines.append("Distributions per cluster (per §1.9 full-distribution discipline).")
    lines.append("")
    lines.append(
        "| cid | mfe p5/p25/p50/p75/p95 | final_r mean / std / t-stat | final_r p5/p50/p95 | reach_1R | reach_2R | wrong_way A / B |"
    )
    lines.append("|---:|---|---|---|---:|---:|---|")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {int(r['cluster_id'])} | "
            f"{r['fwd_mfe_p5']:.2f} / {r['fwd_mfe_p25']:.2f} / {r['fwd_mfe_p50']:.2f} / "
            f"{r['fwd_mfe_p75']:.2f} / {r['fwd_mfe_p95']:.2f} | "
            f"{r['final_r_mean']:.3f} / {r['final_r_std']:.3f} / {r['final_r_t_stat']:.2f} | "
            f"{r['final_r_p5']:.2f} / {r['final_r_p50']:.2f} / {r['final_r_p95']:.2f} | "
            f"{r['frac_reach_1R']:.3f} | {r['frac_reach_2R']:.3f} | "
            f"{r['frac_wrong_way_def_a']:.3f} / {r['frac_wrong_way_def_b']:.3f} |"
        )
    lines.append("")

    lines.append("## Early-peak family split (pct_peak_and_collapse)")
    lines.append("")
    early_peak_clusters = summary_df[summary_df["provisional_label_step2"] == "early_peak_family"]
    if early_peak_clusters.empty:
        lines.append("No clusters carried the provisional `early_peak_family` label at Step 2.")
    else:
        lines.append(
            "Rule: `pct_peak_and_collapse < 0.30` → `early_peak_hold`; "
            "`>= 0.50` → `peak_and_collapse`; in `[0.30, 0.50)` → keep `early_peak_family` "
            "and defer to §11 empirical per-fold test at Step 5+ (no folds at Step 3)."
        )
        lines.append("")
        lines.append("| cid | size | pct_peak_and_collapse | final label |")
        lines.append("|---:|---:|---:|---|")
        for _, r in early_peak_clusters.iterrows():
            lines.append(
                f"| {int(r['cluster_id'])} | {int(r['size_count'])} | "
                f"{r['pct_peak_and_collapse']:.3f} | `{r['archetype_label_final']}` |"
            )
    lines.append("")

    lines.append("## bars_held distribution per cluster (cap-binding visibility)")
    lines.append("")
    lines.append("Step 1 noted 16.7% of pool trades hit the 240-bar cap. Per-cluster breakdown:")
    lines.append("")
    lines.append("| cid | p5 | p50 | p95 | frac at cap (>=240) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {int(r['cluster_id'])} | {int(r['bars_held_p5'])} | "
            f"{int(r['bars_held_p50'])} | {int(r['bars_held_p95'])} | "
            f"{r['frac_cap_240']:.3f} |"
        )
    lines.append("")

    lines.append("## Capturability pass list (→ Step 4)")
    lines.append("")
    if pass_df.empty:
        lines.append("Empty. No clusters cleared §2.")
    else:
        lines.append("| cid | archetype | method | size | size_fraction |")
        lines.append("|---:|---|---|---:|---:|")
        for _, r in pass_df.iterrows():
            lines.append(
                f"| {int(r['cluster_id'])} | `{r['archetype_label_final']}` | "
                f"{r['labelling_method_final']} | {int(r['size_count'])} | "
                f"{r['size_fraction_of_pool']:.3f} |"
            )
    lines.append("")

    lines.append("## Boundary-cluster resolution")
    lines.append("")
    survivors_with_boundary = summary_df[
        (summary_df["§2_overall_pass"]) & (summary_df["provisional_method_step2"] == "unresolved")
    ]
    if survivors_with_boundary.empty:
        lines.append(
            "No unresolved-boundary clusters survived §2; no §11 empirical "
            "capture-ratio test required at this step."
        )
    else:
        lines.append(
            f"{len(survivors_with_boundary)} unresolved-boundary cluster(s) survived §2 — "
            "label resolution deferred to §11 per-fold empirical capture-ratio test "
            "(Step 5+, when folds are introduced)."
        )
    lines.append("")

    lines.append("## Determinism")
    lines.append("")
    lines.append(
        f"Two-run byte-identical CSV outputs: **{determinism_status}**. KDE uses "
        "Scott's bandwidth and a fixed 200-point grid; `scipy.signal.find_peaks` "
        "with explicit prominence threshold. All deterministic for fixed inputs. "
        "See `tests/arc_kh24_v2/test_step3_determinism.py` for the CI test."
    )
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    if halt:
        lines.append("§7 arc-level gate: **FAIL**.")
        lines.append("")
        lines.append(
            "The bare KH-24 signal under v2.0 K = 5 produces zero capturable archetypes. "
            "The arc halts here. Possible next steps for chat consideration:"
        )
        lines.append(
            "- Inspect which §2 criterion killed the closest cluster (c2 with monotonicity "
            "0.878, ttp 0.27, size 13%) and whether the kill is by a small margin or "
            "structural."
        )
        lines.append(
            "- Compare against KH-24 K=4 archetype 3 reference (§14 anchor): monotonicity "
            "0.576, fwd_mfe_p50 5.40R, frac_reach_1R 1.00, frac_wrong_way 0.04. The "
            "difference is that KH-24's anchor used K = 4 clustering; here we use K = 5 "
            "(selected per §6 silhouette criterion)."
        )
        lines.append(
            "- Consider whether v2.0 protocol intent requires re-running clustering at "
            "K = 4 (the KH-24 anchor's K) for direct comparability."
        )
    else:
        lines.append(f"§7 arc-level gate: **PASS** ({len(pass_df)} cluster(s) → Step 4).")

    _write_text("\n".join(lines) + "\n", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1-dir", default=str(DEFAULT_STEP1_DIR))
    parser.add_argument("--step2-dir", default=str(DEFAULT_STEP2_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--determinism-status", default="PENDING")
    args = parser.parse_args(argv)

    result = run(
        Path(args.step1_dir).resolve(),
        Path(args.step2_dir).resolve(),
        Path(args.out_dir).resolve(),
    )

    if not args.no_report:
        write_report(
            result,
            Path(args.out_dir).resolve() / "step3_report.md",
            determinism_status=args.determinism_status,
        )

    # Console summary
    n_pass = len(result["pass_df"])
    if result["halt_reason"]:
        print(f"Step 3: {result['halt_reason']} — 0 capturable archetypes")
        return 1
    print(f"Step 3: PASS — {n_pass} archetype(s) cleared §2 floors")
    for _, r in result["pass_df"].iterrows():
        print(
            f"  cluster {int(r['cluster_id'])}: {r['archetype_label_final']} "
            f"(n={int(r['size_count'])}, {r['size_fraction_of_pool']:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
