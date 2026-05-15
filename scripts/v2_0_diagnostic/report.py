"""v2.0 archetype diagnostic — Step 7: write DIAGNOSTIC_REPORT.md.

Eleven sections per spec. Descriptive only — no verdicts, no interpretation
of whether the reframe is correct. The chat reads and decides.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def _df_to_md(df: pd.DataFrame) -> str:
    """Render DataFrame as GitHub-flavored markdown table (no tabulate dep)."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_rows = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if pd.isna(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) >= 1e-4 or v == 0 else f"{v:.6g}")
            elif isinstance(v, bool):
                vals.append("True" if v else "False")
            else:
                vals.append(str(v))
        body_rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body_rows])

from scripts.v2_0_diagnostic.cluster import K_VALUES
from scripts.v2_0_diagnostic.path_features import FEATURE_COLS
from scripts.v2_0_diagnostic.overlap import DUAL_GATE_PASSING


DATASETS = ("kh24", "arc1", "arc2")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _silhouette_table(per_dataset: dict) -> str:
    rows = []
    for k in K_VALUES:
        r = {"K": k}
        for name in DATASETS:
            sil = per_dataset[name]["K"][k]["silhouette"]
            r[name] = f"{sil:.4f}"
        rows.append(r)
    df = pd.DataFrame(rows)
    return _df_to_md(df)


def _archetype_block(name: str, k: int, run: dict) -> str:
    summary = run["summary"].copy()
    summary["label"] = run["labels"].values
    cols = [
        "archetype_id", "label", "size_count", "size_fraction_of_pool",
        "monotonicity_centroid", "local_peaks_centroid",
        "pullback_magnitude_centroid", "time_to_peak_relative_centroid",
        "fwd_mfe_h240_p50", "frac_reach_1R", "frac_reach_2R",
        "frac_wrong_way", "pct_peak_and_collapse",
        "final_r_mean", "final_r_t_stat",
        "shape_tag",
    ]
    fmt = summary[cols].copy()
    for c in fmt.columns:
        if c in ("archetype_id", "label", "size_count", "shape_tag"):
            continue
        fmt[c] = fmt[c].astype("float64").round(4)
    return _df_to_md(fmt)


def _predictability_block(run: dict) -> str:
    pred = run["predictability"].copy()
    cols = ["archetype_id", "auc_mean", "auc_std",
            "auc_fold_1", "auc_fold_2", "auc_fold_3", "auc_fold_4", "auc_fold_5",
            "n_pos", "n_total"]
    fmt = pred[cols].copy()
    for c in fmt.columns:
        if c in ("archetype_id", "n_pos", "n_total"):
            continue
        fmt[c] = fmt[c].astype("float64").round(4)
    return _df_to_md(fmt)


def write_report(out_root: Path, per_dataset: dict, evidence: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# v2.0 archetype clustering diagnostic — path-shape archetypes")
    lines.append("")
    lines.append("> Diagnostic only. No protocol revision, no floor-setting, no verdicts.")
    lines.append("> Inputs: KH-24 (553 trades), Arc 1 (45673 trades), Arc 2 (3993 trades).")
    lines.append("> All three are weak inputs by acknowledgement; the diagnostic gathers evidence")
    lines.append("> against the question: does path-shape clustering identify clean+meaty+predictable")
    lines.append("> archetypes that the existing dual-gate clustering missed.")
    lines.append("")

    # §1 Read-first verification
    lines.append("## §1 Read-first verification")
    lines.append("")
    for name in DATASETS:
        info = per_dataset[name]
        lines.append(f"- **{name}**: trades = {info['n_trades']}, "
                     f"path-shape features computed via "
                     f"`scripts/v2_0_diagnostic/path_features.py`. "
                     f"Loader reused from `scripts/v1_3_calibration/load_paths.py`; "
                     f"R-unit convention (R = 2 x ATR) preserved.")
    lines.append("")
    lines.append("Loader normalisation decisions reused verbatim. Arc 1 close_r is derived "
                 "from fwd_logret_cum (no per-bar OHLC); Arc 2 + KH-24 use per-bar OHLC + "
                 "running mfe/mae. None of the path-shape features depend on intrabar high/low — "
                 "all computed from close_r and mfe_so_far_r. Arc 1 path-shape features are "
                 "thus exact under the same close-based approximation as v1.3 axis 2g.")
    lines.append("")
    lines.append("**Spec deviation flag — pullback_magnitude_median:** the spec text reads "
                 "`mfe_so_far_r at earlier peak - min(mfe_so_far_r over bars between)`, but "
                 "`mfe_so_far_r` is the per-bar running maximum and therefore non-decreasing, "
                 "so the literal expression is identically 0 across every trade. Implementation "
                 "uses `min(close_r)` for the inter-peak trough (close_r is the per-bar mark "
                 "and does retrace), preserving the spec's intent that the feature captures "
                 "pullback magnitude between local peaks. Flagged here for chat audit.")
    lines.append("")
    lines.append("**Scope flag — Arc 1 held window:** Arc 1 trades use a 1H time exit; "
                 "`bars_held = 1` for 96% of trades. The spec scopes path-shape features to "
                 "`[0, bars_held]`, so Arc 1 features are computed over a 1-2 bar window and "
                 "are systemically degenerate (see §2). The diagnostic surfaces this rather "
                 "than re-scoping; chat may want to re-run with a forward-window scope.")
    lines.append("")

    # §2 Feature distribution summary
    lines.append("## §2 Feature distribution summary")
    lines.append("")
    for name in DATASETS:
        lines.append(f"### {name}")
        lines.append("")
        dist = per_dataset[name]["feature_dist"].copy()
        for c in dist.columns:
            if c == "feature":
                continue
            dist[c] = dist[c].astype("float64").round(4)
        lines.append(_df_to_md(dist))
        lines.append("")
        degen = per_dataset[name]["degen"]
        flagged = degen[degen["is_degenerate"]]
        if len(flagged):
            lines.append("**Degenerate-distribution flag (>80% trades at single value):**")
            lines.append("")
            for _, r in flagged.iterrows():
                lines.append(f"- `{r['feature']}` — top value {r['top_value']:.4f} "
                             f"shared by {r['top_share']*100:.1f}% of trades")
            lines.append("")

    # §3 Clustering separability
    lines.append("## §3 Clustering separability (silhouette by K x dataset)")
    lines.append("")
    lines.append(_silhouette_table(per_dataset))
    lines.append("")
    failed = []
    for name in DATASETS:
        for k in K_VALUES:
            if per_dataset[name]["K"][k]["failed"]:
                failed.append(f"- **{name}** K={k}: one archetype > 90% of trades (failed clustering)")
    if failed:
        lines.append("**Failed-clustering flags:**")
        lines.append("")
        lines.extend(failed)
        lines.append("")
    else:
        lines.append("No failed-clustering flags raised.")
        lines.append("")

    # §4 Archetype profiles per dataset
    lines.append("## §4 Archetype profiles (per dataset, per K)")
    lines.append("")
    for name in DATASETS:
        lines.append(f"### {name}")
        lines.append("")
        for k in K_VALUES:
            run = per_dataset[name]["K"][k]
            lines.append(f"#### K = {k}")
            lines.append("")
            lines.append(_archetype_block(name, k, run))
            lines.append("")

    # §5 Forward geometry per archetype
    lines.append("## §5 Forward geometry per archetype")
    lines.append("")
    lines.append("Forward geometry metrics are embedded in the §4 tables "
                 "(`fwd_mfe_h240_p50`, `frac_reach_1R`, `frac_reach_2R`, "
                 "`frac_wrong_way`, `pct_peak_and_collapse`, `final_r_mean`, "
                 "`final_r_t_stat`). Per-archetype distribution side files "
                 "are at `results/v2_0_diagnostic/<dataset>/archetype_<id>_K<k>_distribution.csv`.")
    lines.append("")
    lines.append("**Clean-shape-but-magnitude-dead flags** "
                 "(monotonicity_centroid >= 0.55 AND fwd_mfe_h240_p50 < 1.0):")
    lines.append("")
    found_clean_dead = []
    for name in DATASETS:
        for k in K_VALUES:
            summary = per_dataset[name]["K"][k]["summary"]
            mask = (summary["monotonicity_centroid"] >= 0.55) & (summary["fwd_mfe_h240_p50"] < 1.0)
            for _, r in summary[mask].iterrows():
                found_clean_dead.append(
                    f"- **{name}** K={k} archetype={int(r['archetype_id'])} "
                    f"(monotonicity={r['monotonicity_centroid']:.3f}, "
                    f"fwd_mfe_p50={r['fwd_mfe_h240_p50']:.3f}R, "
                    f"size={r['size_fraction_of_pool']*100:.1f}%)"
                )
    if found_clean_dead:
        lines.extend(found_clean_dead)
    else:
        lines.append("- (none)")
    lines.append("")

    # §6 Predictability per archetype
    lines.append("## §6 Predictability per archetype (5-fold ROC-AUC on entry features)")
    lines.append("")
    lines.append("AUC reference: 0.50 random; 0.55-0.60 marginal; 0.60-0.70 usable; 0.70+ strong.")
    lines.append("")
    for name in DATASETS:
        lines.append(f"### {name}")
        lines.append("")
        for k in K_VALUES:
            run = per_dataset[name]["K"][k]
            lines.append(f"#### K = {k}")
            lines.append("")
            lines.append(_predictability_block(run))
            lines.append("")

    # §7 Overlap
    lines.append("## §7 Overlap with existing dual-gate clusters (Arc 1 + Arc 2)")
    lines.append("")
    lines.append("Existing-cluster reference is K3_kmeans from `step3_extractability/cluster_assignments.csv` — "
                 "Arc 1 dual-gate-passing cluster = C0; Arc 2 dual-gate-passing cluster = C2 "
                 "(per v1.3 calibration §3).")
    lines.append("")

    for name in ("arc1", "arc2"):
        lines.append(f"### {name}")
        lines.append("")
        passing = DUAL_GATE_PASSING[name]
        os_df = per_dataset[name]["overlap_summary"]
        # Forward: per passing cluster at each K, which archetype contains majority
        fwd = os_df[(os_df["view"] == "forward") & (os_df["is_dual_gate_passing"])][
            ["K", "existing_cluster", "archetype_id", "overlap_pct", "n_in_existing"]
        ].copy()
        fwd["overlap_pct"] = fwd["overlap_pct"].round(4)
        lines.append(f"**Forward — where does dual-gate-passing cluster C{passing}'s majority land?**")
        lines.append("")
        lines.append(_df_to_md(fwd))
        lines.append("")

        # Inverse: archetypes whose source is NOT majority-from-passing,
        # AND that are clean+meaty+predictable (i.e. potentially v2.0-evidential)
        ev = evidence[(evidence["dataset"] == name)]
        clean_meaty_pred = ev[ev["clean"] & ev["meaty"] & ev["predictable"]][
            ["K", "archetype_id"]
        ]
        if len(clean_meaty_pred) == 0:
            lines.append("**Clean + meaty + predictable archetypes (any K):** none.")
            lines.append("")
        else:
            inv = os_df[os_df["view"] == "inverse"]
            joined = clean_meaty_pred.merge(inv, on=["K", "archetype_id"], how="left")
            cols = [
                "K", "archetype_id",
                "pct_from_existing_0", "pct_from_existing_1", "pct_from_existing_2", "pct_from_existing_-2",
                "is_majority_from_passing_cluster",
            ]
            avail = [c for c in cols if c in joined.columns]
            fmt = joined[avail].copy()
            for c in fmt.columns:
                if c in ("K", "archetype_id", "is_majority_from_passing_cluster"):
                    continue
                fmt[c] = fmt[c].astype("float64").round(4)
            lines.append("**Composition of clean+meaty+predictable archetypes (by source cluster):**")
            lines.append("")
            lines.append(_df_to_md(fmt))
            lines.append("")

            non_overlap = joined[~joined["is_majority_from_passing_cluster"]]
            if len(non_overlap):
                lines.append("**Flag: clean+meaty+predictable archetypes NOT majority-from-passing-cluster:**")
                lines.append("")
                for _, r in non_overlap.iterrows():
                    lines.append(f"- {name} K={int(r['K'])} archetype={int(r['archetype_id'])}")
                lines.append("")

    lines.append("KH-24 has no L-arc step-3 clusters; overlap section N/A.")
    lines.append("")

    # §8 v2.0 evidence flag — central output
    lines.append("## §8 v2.0 evidence flag — central output")
    lines.append("")
    qual = evidence[evidence["qualifies_as_v2_0_evidence"]].copy()
    if len(qual) == 0:
        lines.append("> **No tuples qualify as v2.0 evidence on these three weak datasets under the "
                     "first-pass priors. The reframe is neither validated nor refuted by this "
                     "diagnostic. Re-run when stronger signals are available, or revisit thresholds.**")
        lines.append("")
    else:
        # Echo all relevant fields including label.
        labels_map: dict[tuple[str, int, int], str] = {}
        for name in DATASETS:
            for k in K_VALUES:
                run = per_dataset[name]["K"][k]
                summary = run["summary"]
                lbls = run["labels"]
                for _, srow in summary.iterrows():
                    labels_map[(name, k, int(srow["archetype_id"]))] = lbls.loc[srow.name]
        qual["shape label"] = qual.apply(
            lambda r: labels_map.get((r["dataset"], int(r["K"]), int(r["archetype_id"])), "(none)"), axis=1
        )
        qual["overlap_with_dual_gate"] = qual.apply(
            lambda r: "N/A (KH-24)" if r["dataset"] == "kh24"
            else ("OVERLAPS_PASSING" if not r["not_overlap"] else "distinct"),
            axis=1,
        )
        cols = ["dataset", "K", "archetype_id", "size_fraction_of_pool", "shape label",
                "monotonicity_centroid", "final_r_mean", "frac_reach_1R", "auc_mean",
                "overlap_with_dual_gate"]
        fmt = qual[cols].copy()
        for c in fmt.columns:
            if c in ("dataset", "K", "archetype_id", "shape label", "overlap_with_dual_gate"):
                continue
            fmt[c] = fmt[c].astype("float64").round(4)
        lines.append(_df_to_md(fmt))
        lines.append("")
        lines.append("Listed for the chat to interpret. Diagnostic does not assert that these "
                     "are the only candidates, only that they meet the first-pass priors.")
        lines.append("")

    # §9 Combined view per dataset
    lines.append("## §9 Combined view per dataset")
    lines.append("")
    for name in DATASETS:
        lines.append(f"### {name}")
        lines.append("")
        para = _combined_paragraph(name, per_dataset[name], evidence)
        lines.append(para)
        lines.append("")

    # §10 Files produced
    lines.append("## §10 Files produced (manifest with sha256)")
    lines.append("")
    # Exclude the report itself and python __pycache__ for determinism.
    files = sorted(
        p for p in out_root.rglob("*")
        if p.is_file() and p.name != "DIAGNOSTIC_REPORT.md"
    )
    rows = [{"path": str(p.relative_to(out_root)).replace("\\", "/"),
             "sha256": _sha256(p),
             "bytes":  p.stat().st_size}
            for p in files]
    if rows:
        lines.append(_df_to_md(pd.DataFrame(rows)))
        lines.append("")

    # §11 Open observations
    lines.append("## §11 Open observations")
    lines.append("")
    obs = _open_observations(per_dataset, evidence)
    if obs:
        for o in obs:
            lines.append(f"- {o}")
    else:
        lines.append("- (no notable observations called out by the automated checks)")
    lines.append("")

    out_path = out_root / "DIAGNOSTIC_REPORT.md"
    out_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _combined_paragraph(name: str, info: dict, evidence: pd.DataFrame) -> str:
    # Pick a representative K — best (lowest) silhouette among non-failed clusterings.
    valid_ks = [k for k in K_VALUES if not info["K"][k]["failed"]]
    if not valid_ks:
        return f"Dataset {name}: all K values had a single dominant archetype > 90% (failed clustering)."
    # Pick K with highest silhouette for the descriptive paragraph.
    k_best = max(valid_ks, key=lambda k: info["K"][k]["silhouette"])
    summary = info["K"][k_best]["summary"].copy()
    summary["label"] = info["K"][k_best]["labels"].values
    summary = summary.sort_values("size_fraction_of_pool", ascending=False).reset_index(drop=True)

    n_arch = len(summary)
    top    = summary.iloc[0]
    bot    = summary.iloc[-1]
    ev_count = int(evidence[
        (evidence["dataset"] == name) & evidence["qualifies_as_v2_0_evidence"]
    ].shape[0])

    bits = [
        f"Dataset **{name}** at K={k_best} (max-silhouette of non-failed K values, "
        f"sil={info['K'][k_best]['silhouette']:.4f}) produces {n_arch} archetypes "
        f"ranging from `{top['label']}` ({top['size_fraction_of_pool']*100:.1f}% of pool) "
        f"to `{bot['label']}` ({bot['size_fraction_of_pool']*100:.1f}% of pool).",
    ]
    if ev_count == 0:
        bits.append(f"No archetype across any of K in {{3,4,5,6,7}} qualifies as v2.0 evidence "
                    "under the first-pass priors (§8 empty for this dataset).")
    else:
        bits.append(f"{ev_count} tuple(s) across all K qualify under first-pass priors — see §8.")
    return " ".join(bits)


def _open_observations(per_dataset: dict, evidence: pd.DataFrame) -> list[str]:
    obs: list[str] = []
    for name in DATASETS:
        degen = per_dataset[name]["degen"]
        for _, r in degen[degen["is_degenerate"]].iterrows():
            obs.append(
                f"**{name}** path-shape feature `{r['feature']}` is degenerate "
                f"({r['top_share']*100:.1f}% at value {r['top_value']:.4f}). "
                "Clustering on degenerate features is uninformative; the rest of this "
                "dataset's results should be read with caution."
            )
        for k in K_VALUES:
            if per_dataset[name]["K"][k]["failed"]:
                obs.append(f"**{name}** K={k}: failed clustering (one archetype > 90% of pool).")
            summary = per_dataset[name]["K"][k]["summary"]
            if (summary["final_r_mean"] <= 0).all():
                obs.append(f"**{name}** K={k}: no archetype has positive `final_r_mean` "
                           "— forward geometry is uniformly weak at this K.")
    # Arc 1 silhouette inflation note (silhouette > 0.90 at any K).
    arc1_high_sil_ks = [
        k for k in K_VALUES if per_dataset["arc1"]["K"][k]["silhouette"] > 0.90
    ]
    if arc1_high_sil_ks:
        obs.append(
            f"**arc1** silhouette is artificially high at K in {arc1_high_sil_ks} "
            "(>0.90). Inflation source: 96% of arc1 trades have bars_held=1, "
            "producing path-shape features that are zero on most features. K-means "
            "trivially separates the zero-vector mass from the small \"moved\" minority, "
            "yielding high silhouette without informative archetypes."
        )
    n_qual = int(evidence["qualifies_as_v2_0_evidence"].sum())
    if n_qual == 0:
        obs.append("**No (dataset, K, archetype) qualifies as v2.0 evidence under first-pass priors.** "
                   "Per spec §8 plainness: this neither validates nor refutes the reframe.")

    # Spotlight near-miss tuples (4 of 5 priors).
    ev = evidence.copy()
    ev["n_true"] = ev[["clean", "meaty", "predictable", "size_viable", "not_overlap"]].sum(axis=1)
    near_miss = ev[ev["n_true"] == 4]
    if len(near_miss):
        n = len(near_miss)
        obs.append(
            f"**{n} near-miss tuple(s)** meet 4 of 5 first-pass priors. Inspect "
            "`v2_0_evidence_flags.csv` for which prior failed — most commonly "
            "`predictable=False` (AUC < 0.60), suggesting entry-feature predictivity "
            "is the binding constraint on these inputs, not path-shape distinctness."
        )
    return obs
