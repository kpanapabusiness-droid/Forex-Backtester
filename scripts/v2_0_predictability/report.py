"""Step 5: synthesis report.

Seven sections per spec. Descriptive only — no verdicts beyond the
mechanical headline-finding categorisation (A/B/C) defined in §6.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

T_OBSERVE = (1, 3, 5, 10)
AUC_GATE = 0.60


def _df_to_md(df: pd.DataFrame, round_cols: bool = True) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if pd.isna(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.4f}".rstrip("0").rstrip(".") if round_cols else f"{v:.6g}")
            elif isinstance(v, bool):
                vals.append("True" if v else "False")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_report(
    out_root: Path,
    targets: pd.DataFrame,
    grouping: pd.DataFrame,
    grouped: pd.DataFrame,
    angle_a: pd.DataFrame,
    angle_b: pd.DataFrame,
    angle_c: pd.DataFrame,
    feature_lists: dict[str, list[str]],
) -> None:
    lines: list[str] = []

    lines.append("# v2.0 predictability investigation — is the wall real?")
    lines.append("")
    lines.append("> Tests whether the predictability wall from PR #129 (logistic AUC < 0.60 on")
    lines.append("> all v2.0 archetypes) is real or an artifact. Three angles per target:")
    lines.append("> A) RF on the same 8 features, B) expanded feature set, C) t>0 path-so-far.")
    lines.append("> Scope: target archetypes meeting RELAXED clean+meaty+size on KH-24 + Arc 2")
    lines.append("> (Arc 1 excluded — diagnostic's bars_held=1 issue makes its clusters invalid).")
    lines.append("> Computation only; §6 categorises the finding mechanically (A/B/C).")
    lines.append("")

    # §1 Targets + grouping
    lines.append("## §1 Target archetypes investigated")
    lines.append("")
    cols_t = [
        "dataset", "K", "archetype_id", "exit_family_tag",
        "size_fraction_of_pool", "monotonicity_centroid", "local_peaks_centroid",
        "pullback_magnitude_centroid", "time_to_peak_relative_centroid",
        "final_r_mean", "frac_reach_1R", "frac_wrong_way", "fwd_mfe_h240_p50",
        "auc_mean",
    ]
    g_with_tag = grouping[[
        "dataset", "K", "archetype_id", "exit_family_tag",
    ]].merge(targets, on=["dataset", "K", "archetype_id"], how="left")
    avail = [c for c in cols_t if c in g_with_tag.columns]
    fmt = g_with_tag[avail].copy()
    for c in fmt.columns:
        if c in ("dataset", "K", "archetype_id", "exit_family_tag"):
            continue
        fmt[c] = fmt[c].astype("float64").round(4)
    lines.append(_df_to_md(fmt))
    lines.append("")

    lines.append("**Exit-family groupings (size >= 2 archetypes sharing a tag):**")
    lines.append("")
    if len(grouped) and (grouped["n_archetypes_in_group"] >= 2).any():
        gsub = grouped[grouped["n_archetypes_in_group"] >= 2].copy()
        gsub["total_size_fraction"] = gsub["total_size_fraction"].round(4)
        lines.append(_df_to_md(gsub))
    else:
        lines.append("- (none — no exit-family tag had >= 2 archetypes within the same (dataset, K))")
    lines.append("")

    # §2 Angle A
    lines.append("## §2 Angle A — Random Forest on the 8 basic entry features")
    lines.append("")
    cols_a = [
        "dataset", "K", "target_id", "exit_family_tag", "target_size", "n_total",
        "auc_mean", "auc_std", "auc_logistic_baseline", "lift_vs_logistic",
    ]
    fmt = angle_a[cols_a].copy()
    for c in ("auc_mean", "auc_std", "auc_logistic_baseline", "lift_vs_logistic"):
        fmt[c] = fmt[c].astype("float64").round(4)
    lines.append(_df_to_md(fmt))
    lines.append("")
    a_clearing = angle_a[angle_a["auc_mean"] >= AUC_GATE]
    if len(a_clearing):
        lines.append(f"**Angle A clears AUC >= {AUC_GATE}** for:")
        for _, r in a_clearing.iterrows():
            lines.append(f"- {r['dataset']} K={int(r['K'])} {r['target_id']}: AUC = {r['auc_mean']:.4f}")
    else:
        lines.append(f"No Angle-A target clears AUC >= {AUC_GATE}.")
    lines.append("")

    # Lift-but-no-gate cases
    lift_only = angle_a[
        (angle_a["lift_vs_logistic"] >= 0.05) & (angle_a["auc_mean"] < AUC_GATE)
    ]
    if len(lift_only):
        lines.append(f"**Angle A non-linearity flag** — RF lifts >= 0.05 vs logistic but stays below {AUC_GATE}:")
        for _, r in lift_only.iterrows():
            lines.append(f"- {r['dataset']} K={int(r['K'])} {r['target_id']}: "
                         f"logistic {r['auc_logistic_baseline']:.4f} -> RF {r['auc_mean']:.4f} "
                         f"(+{r['lift_vs_logistic']:.4f})")
        lines.append("")

    # §3 Angle B
    lines.append("## §3 Angle B — expanded entry feature set")
    lines.append("")
    for ds, feats in feature_lists.items():
        lines.append(f"**{ds}** expanded feature list ({len(feats)} features):")
        lines.append("")
        lines.append("```")
        lines.append(", ".join(feats))
        lines.append("```")
        lines.append("")

    cols_b = [
        "dataset", "K", "target_id", "exit_family_tag", "target_size", "n_features_used",
        "auc_logistic_expanded_mean", "auc_rf_expanded_mean", "auc_logistic_baseline",
        "lift_logistic_expanded_vs_baseline", "lift_rf_expanded_vs_baseline",
    ]
    fmt = angle_b[cols_b].copy()
    for c in fmt.columns:
        if c in ("dataset", "K", "target_id", "exit_family_tag", "target_size", "n_features_used"):
            continue
        fmt[c] = fmt[c].astype("float64").round(4)
    lines.append(_df_to_md(fmt))
    lines.append("")
    b_clearing = angle_b[
        (angle_b["auc_logistic_expanded_mean"] >= AUC_GATE)
        | (angle_b["auc_rf_expanded_mean"] >= AUC_GATE)
    ]
    if len(b_clearing):
        lines.append(f"**Angle B clears AUC >= {AUC_GATE}** for:")
        for _, r in b_clearing.iterrows():
            best = max(r["auc_logistic_expanded_mean"], r["auc_rf_expanded_mean"])
            lines.append(f"- {r['dataset']} K={int(r['K'])} {r['target_id']}: best AUC = {best:.4f}")
    else:
        lines.append(f"No Angle-B target clears AUC >= {AUC_GATE}.")
    lines.append("")

    # §4 Angle C
    lines.append("## §4 Angle C — t > 0 path-so-far predictability")
    lines.append("")
    cols_c = ["dataset", "K", "target_id", "exit_family_tag", "auc_rf_basic_only"]
    for t in T_OBSERVE:
        cols_c.extend([f"target_size_at_t{t}", f"n_excluded_at_t{t}", f"auc_rf_mean_at_t{t}", f"lift_vs_entry_only_at_t{t}"])
    fmt = angle_c[cols_c].copy()
    for c in fmt.columns:
        if c in ("dataset", "K", "target_id", "exit_family_tag"):
            continue
        if "size" in c or "excluded" in c:
            continue
        fmt[c] = fmt[c].astype("float64").round(4)
    lines.append(_df_to_md(fmt))
    lines.append("")

    # Smallest t where AUC >= gate
    c_clearing_rows = []
    for _, r in angle_c.iterrows():
        for t in T_OBSERVE:
            auc = r[f"auc_rf_mean_at_t{t}"]
            if pd.notna(auc) and auc >= AUC_GATE:
                c_clearing_rows.append({
                    "dataset": r["dataset"], "K": int(r["K"]), "target_id": r["target_id"],
                    "smallest_t_clearing": t, "auc_at_t": auc,
                    "n_excluded_at_t": int(r[f"n_excluded_at_t{t}"]),
                    "target_size_at_t": int(r[f"target_size_at_t{t}"]),
                })
                break
    if c_clearing_rows:
        lines.append(f"**Angle C clears AUC >= {AUC_GATE}** at smallest t for:")
        lines.append("")
        df_clr = pd.DataFrame(c_clearing_rows)
        for c in ("auc_at_t",):
            df_clr[c] = df_clr[c].astype("float64").round(4)
        lines.append(_df_to_md(df_clr))
    else:
        lines.append(f"No Angle-C target clears AUC >= {AUC_GATE} at any of t in {{1,3,5,10}}.")
    lines.append("")

    # §5 Cross-angle synthesis
    lines.append("## §5 Cross-angle synthesis")
    lines.append("")
    synth_rows = []
    for _, r in angle_a.iterrows():
        ds, k, tid = r["dataset"], int(r["K"]), r["target_id"]
        b_row = angle_b[(angle_b["dataset"] == ds) & (angle_b["K"] == k) & (angle_b["target_id"] == tid)]
        c_row = angle_c[(angle_c["dataset"] == ds) & (angle_c["K"] == k) & (angle_c["target_id"] == tid)]
        b1 = b_row.iloc[0] if len(b_row) else None
        c1 = c_row.iloc[0] if len(c_row) else None
        any_clear = (
            (r["auc_mean"] >= AUC_GATE)
            or (b1 is not None and (b1["auc_logistic_expanded_mean"] >= AUC_GATE or b1["auc_rf_expanded_mean"] >= AUC_GATE))
            or (c1 is not None and any((not pd.isna(c1[f"auc_rf_mean_at_t{t}"])) and (c1[f"auc_rf_mean_at_t{t}"] >= AUC_GATE) for t in T_OBSERVE))
        )
        synth_rows.append({
            "dataset": ds, "K": k, "target_id": tid,
            "logistic_basic": r["auc_logistic_baseline"],
            "rf_basic": r["auc_mean"],
            "logistic_expanded": b1["auc_logistic_expanded_mean"] if b1 is not None else float("nan"),
            "rf_expanded":       b1["auc_rf_expanded_mean"]       if b1 is not None else float("nan"),
            "rf_at_t3": c1["auc_rf_mean_at_t3"] if c1 is not None else float("nan"),
            "rf_at_t5": c1["auc_rf_mean_at_t5"] if c1 is not None else float("nan"),
            "best_>=_0.60": any_clear,
        })
    synth = pd.DataFrame(synth_rows)
    for c in synth.columns:
        if c in ("dataset", "K", "target_id", "best_>=_0.60"):
            continue
        synth[c] = synth[c].astype("float64").round(4)
    lines.append(_df_to_md(synth))
    lines.append("")

    # §6 Headline finding
    lines.append("## §6 Headline finding")
    lines.append("")
    # Compute category
    cleared_entry = (
        any(angle_a["auc_mean"] >= AUC_GATE)
        or any(angle_b["auc_logistic_expanded_mean"] >= AUC_GATE)
        or any(angle_b["auc_rf_expanded_mean"] >= AUC_GATE)
    )
    cleared_t = any(
        any(pd.notna(r[f"auc_rf_mean_at_t{t}"]) and r[f"auc_rf_mean_at_t{t}"] >= AUC_GATE for t in T_OBSERVE)
        for _, r in angle_c.iterrows()
    )

    if cleared_entry:
        lines.append("### **(A) Wall broken at entry.**")
        lines.append("")
        lines.append(f"At least one target archetype has a non-t>0 angle AUC >= {AUC_GATE}.")
        lines.append("")
        targets_cleared_A = angle_a[angle_a["auc_mean"] >= AUC_GATE]
        if len(targets_cleared_A):
            lines.append("- Angle A (RF basic):")
            for _, r in targets_cleared_A.iterrows():
                lines.append(f"  - {r['dataset']} K={int(r['K'])} {r['target_id']}: AUC = {r['auc_mean']:.4f}")
        targets_cleared_B = angle_b[
            (angle_b["auc_logistic_expanded_mean"] >= AUC_GATE)
            | (angle_b["auc_rf_expanded_mean"] >= AUC_GATE)
        ]
        if len(targets_cleared_B):
            lines.append("- Angle B (expanded):")
            for _, r in targets_cleared_B.iterrows():
                best = max(r["auc_logistic_expanded_mean"], r["auc_rf_expanded_mean"])
                lines.append(f"  - {r['dataset']} K={int(r['K'])} {r['target_id']}: AUC = {best:.4f}")
        lines.append("")
        lines.append("**Reframe is validated; pipeline retains entry-time filter.**")
    elif cleared_t:
        lines.append("### **(B) Wall broken only with t > 0.**")
        lines.append("")
        lines.append(f"No entry-time angle clears AUC >= {AUC_GATE}, but Angle C does for at least one target.")
        lines.append("")
        if c_clearing_rows:
            lines.append("- Smallest-t-clearing targets:")
            for r in c_clearing_rows:
                lines.append(f"  - {r['dataset']} K={r['K']} {r['target_id']}: t={r['smallest_t_clearing']} -> AUC = {r['auc_at_t']:.4f}; excluded = {r['n_excluded_at_t']}")
        lines.append("")
        lines.append("**Reframe is validated under a different pipeline shape (deferred archetype identification).**")
    else:
        lines.append("### **(C) Wall holds.**")
        lines.append("")
        lines.append(f"No target archetype clears AUC >= {AUC_GATE} on any of the three angles.")
        lines.append("Predictability is the real binding constraint, not feature-set or modelling choices.")
        lines.append("")
        lines.append("**Reframe is structurally blocked on these inputs.**")
    lines.append("")

    # §7 Open observations
    lines.append("## §7 Open observations")
    lines.append("")
    obs = _open_observations(angle_a, angle_b, angle_c)
    if obs:
        for o in obs:
            lines.append(f"- {o}")
    else:
        lines.append("- (none flagged by automated checks)")
    lines.append("")

    # Manifest
    lines.append("## Files produced")
    lines.append("")
    files = sorted(
        p for p in out_root.rglob("*")
        if p.is_file() and p.name != "PREDICTABILITY_INVESTIGATION.md"
    )
    rows = [{
        "path": str(p.relative_to(out_root)).replace("\\", "/"),
        "sha256": _sha256(p),
        "bytes":  p.stat().st_size,
    } for p in files]
    if rows:
        lines.append(_df_to_md(pd.DataFrame(rows), round_cols=False))
    lines.append("")

    (out_root / "PREDICTABILITY_INVESTIGATION.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n",
    )


def _open_observations(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> list[str]:
    obs: list[str] = []

    # Targets where Angle B logistic dropped vs basic (overfitting flag).
    if "lift_logistic_expanded_vs_baseline" in b.columns:
        drops = b[b["lift_logistic_expanded_vs_baseline"] < -0.05]
        if len(drops):
            obs.append(
                f"{len(drops)} target(s) showed logistic-expanded AUC drop >= 0.05 vs basic logistic — "
                "possible overfitting or multicollinearity introduced by extra features."
            )

    # Large RF lifts within Angle A — non-linearity wins partially even if absolute stays low
    if "lift_vs_logistic" in a.columns:
        big_lift = a[a["lift_vs_logistic"] >= 0.05]
        if len(big_lift):
            obs.append(
                f"{len(big_lift)} target(s) lifted >= 0.05 from RF (basic) over logistic (basic) — "
                "non-linearity captures some signal that logistic misses; absolute level still matters for the gate."
            )

    # Sharp t>0 jumps
    rows = []
    for _, r in c.iterrows():
        base = r["auc_rf_basic_only"]
        for t in (1, 3, 5, 10):
            v = r[f"auc_rf_mean_at_t{t}"]
            if pd.notna(v) and pd.notna(base) and (v - base) >= 0.10:
                rows.append((r["dataset"], int(r["K"]), r["target_id"], t, v - base))
                break
    if rows:
        obs.append(
            f"{len(rows)} target(s) gained >= 0.10 AUC by observing path-so-far at small t. "
            "Suggests archetype identifiability is path-driven, not entry-driven."
        )

    # High-exclusion-rate flag at t=10 — compared against total dataset size
    # (not target_size which is positives only).
    totals = {"kh24": 553, "arc2": 3993}
    flagged = set()
    for _, r in c.iterrows():
        ds = r["dataset"]
        total = totals.get(ds)
        excl = r.get("n_excluded_at_t10")
        if total and pd.notna(excl) and (excl / total) >= 0.5 and ds not in flagged:
            obs.append(
                f"{ds}: {excl}/{total} ({excl/total*100:.1f}%) trades exited before t=10 — "
                "deferred-identification using t=10 cannot act on those, halving the addressable pool."
            )
            flagged.add(ds)

    # Entry-time clearing breakdown by dataset. Key by (K, target_id) since
    # archetype labels (e.g. "arch_3") repeat across K values.
    b_clear = b[(b["auc_logistic_expanded_mean"] >= AUC_GATE) | (b["auc_rf_expanded_mean"] >= AUC_GATE)]
    a_clear = a[a["auc_mean"] >= AUC_GATE]
    for ds in ("kh24", "arc2"):
        b_ds = b_clear[b_clear["dataset"] == ds]
        a_ds = a_clear[a_clear["dataset"] == ds]
        n_targets_ds = int((b["dataset"] == ds).sum())
        clear_keys = set(
            zip(a_ds["K"].tolist(), a_ds["target_id"].tolist())
        ).union(
            set(zip(b_ds["K"].tolist(), b_ds["target_id"].tolist()))
        )
        n_entry_clear = len(clear_keys)
        if n_entry_clear == 0:
            obs.append(
                f"**{ds}**: 0/{n_targets_ds} targets clear AUC >= {AUC_GATE} at entry. "
                "Predictability wall holds at entry on this dataset; only t>0 angle clears it."
            )
        else:
            keys_str = ", ".join(f"K={int(k)} {t}" for k, t in sorted(clear_keys))
            obs.append(
                f"**{ds}**: {n_entry_clear}/{n_targets_ds} target(s) clear AUC >= {AUC_GATE} at entry "
                f"(Angle A or B): {keys_str}. Entry-time filter feasible for those targets."
            )

    return obs
