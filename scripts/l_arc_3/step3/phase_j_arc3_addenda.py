"""Phase J — arc-3-specific addenda for L Arc 3 step 3.

Three arc-3-specific deliverables on top of the inherited arc 2 step 3 pipeline:

(1) cluster_direction_split.csv
    For each (algo, K) target+mirror cluster identified at Phase A.5 (v1.2 A2):
      - Fraction of trades by signal_bar_direction ∈ {up, down, doji}
      - Per-direction mean R, std R, full distribution
    This is the structural support for the step-4 candidate
    "filter to signal_bar_direction == up" (NOT a verdict — descriptive).

(2) signal_bar_direction BH-tier carry assessment
    Appends a row to cross_arc_portfolio_family.csv evaluating
    signal_bar_direction_is_up (binary 0/1) as a NEW PRE-REGISTRATION
    cross-arc carry per arc-open §4 step-2 finding. Encoded as a
    numeric (one-hot, "up" = 1; "down" + "doji" = 0). Reports BH tier
    per (algo, K) target.

(3) PHASE_L_ARC_3_STEP3.md addenda
    Appends arc-3-specific sections before the Verdict placeholder:
      §X variance-actually-wider summary (echoes step 2 finding;
          §4 prediction inverted)
      §Y up/down split structural note (echoes step 2 + cluster split)
      §Z Phase B fold-stability elevated narrative (F5/F6 callouts)
      §AA Phase G JPY-vs-non-JPY summary
      §AB Cross-arc carry status summary table
    Then writes ## Verdict placeholder table (six target/mirror columns
    × four dual-gate conditions) with PASS/FAIL cells populated from
    the actual phase outputs. ## Verdict text body remains
    "[planner to write]".

Descriptive only (op spec §11.5). The ## Verdict text is the ONE place
where the planner writes action-shaped reasoning; this script only sets up
the data structure the verdict reads.
"""
# ruff: noqa: E402, E701, E702, F841, I001, F401
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.l_arc_3.step3 import _common as C

STEP2_DIR = REPO_ROOT / "results" / "l_arc_3" / "step2_descriptive"
OUT = C.OUT_DIR
PHASE_DOC = OUT / "PHASE_L_ARC_3_STEP3.md"


def _load_signals_features() -> pd.DataFrame:
    df = pd.read_csv(STEP2_DIR / "signals_features.csv")
    return df


def _load_cluster_assignments() -> pd.DataFrame:
    return pd.read_csv(OUT / "cluster_assignments.csv")


def _load_target_selection() -> pd.DataFrame:
    return pd.read_csv(OUT / "cluster_target_selection.csv")


def _stats_for_arr(arr: np.ndarray) -> Dict[str, float]:
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    if n < 2:
        return {"n": n, "mean": float("nan"), "std": float("nan"),
                "p5": float("nan"), "p25": float("nan"), "p50": float("nan"),
                "p75": float("nan"), "p95": float("nan")}
    p5, p25, p50, p75, p95 = np.percentile(finite, [5, 25, 50, 75, 95])
    return {"n": n,
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)),
            "p5": float(p5), "p25": float(p25), "p50": float(p50),
            "p75": float(p75), "p95": float(p95)}


def cluster_direction_split() -> Path:
    feats = _load_signals_features()
    assigns = _load_cluster_assignments()
    targets = _load_target_selection()

    merged = feats.merge(assigns, on="trade_id", how="left")

    rows: List[Dict[str, Any]] = []
    for _, t in targets.iterrows():
        algo = str(t["algo"])
        K = int(t["K"])
        for role in ("target", "mirror"):
            cid_col = f"{role}_cluster_id"
            cid = int(t[cid_col])
            if cid in (-999, -2):
                continue
            if algo == "hdbscan":
                col_name = f"K{K}_hdbscan" if f"K{K}_hdbscan" in assigns.columns else "hdbscan"
            else:
                col_name = f"K{K}_{algo}"
            if col_name not in merged.columns:
                continue
            sub = merged[merged[col_name] == cid]
            n_cluster = len(sub)
            for direction in ("up", "down", "doji"):
                d_sub = sub[sub["signal_bar_direction"] == direction]
                n_dir = len(d_sub)
                net_r_arr = d_sub["net_r"].to_numpy(dtype=float)
                s = _stats_for_arr(net_r_arr)
                rows.append({
                    "algo": algo, "K": K, "role": role, "cluster_id": cid,
                    "direction": direction,
                    "n_in_cluster": n_cluster,
                    "n_in_direction_within_cluster": n_dir,
                    "frac_of_cluster": (n_dir / n_cluster) if n_cluster else float("nan"),
                    "mean_net_r": s["mean"], "std_net_r": s["std"],
                    "p5_net_r": s["p5"], "p25_net_r": s["p25"], "p50_net_r": s["p50"],
                    "p75_net_r": s["p75"], "p95_net_r": s["p95"],
                })
    out = OUT / "cluster_direction_split.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(f"  wrote {out}")
    return out


def signal_bar_direction_carry_tiers(perms: int = 200) -> Path:
    """Evaluate signal_bar_direction == 'up' (binary) as a new cross-arc carry
    against each (algo, K) target cluster. Compute AUC + hash-seeded permutation
    null + BH-corrected p-value (BH within this single-feature evaluation = the
    raw permutation p-value, since N=1 test).
    """
    feats = _load_signals_features()
    assigns = _load_cluster_assignments()
    targets = _load_target_selection()

    is_up = (feats["signal_bar_direction"] == "up").astype(int).to_numpy()
    merged = feats[["trade_id"]].merge(assigns, on="trade_id", how="left")

    rows: List[Dict[str, Any]] = []
    rng_base_seed = int.from_bytes(
        __import__("hashlib").sha256(b"l_arc_3_step3_signal_bar_direction").digest()[:4],
        "little",
    )
    rng = np.random.default_rng(rng_base_seed)

    for _, t in targets.iterrows():
        algo = str(t["algo"]); K = int(t["K"])
        target_cid = int(t["target_cluster_id"])
        if target_cid in (-999, -2):
            continue
        col_name = f"K{K}_{algo}" if algo != "hdbscan" else (f"K{K}_hdbscan" if f"K{K}_hdbscan" in merged.columns else "hdbscan")
        if col_name not in merged.columns:
            continue
        y_target = (merged[col_name].values == target_cid).astype(int)
        # Skip if degenerate (all same class).
        if len(set(y_target.tolist())) < 2:
            continue
        try:
            auc = float(roc_auc_score(y_target, is_up))
        except ValueError:
            continue
        # Permutation null on the predictor.
        perm_aucs = np.zeros(perms, dtype=float)
        for i in range(perms):
            perm = rng.permutation(is_up)
            try:
                perm_aucs[i] = float(roc_auc_score(y_target, perm))
            except ValueError:
                perm_aucs[i] = 0.5
        # Two-sided p-value relative to 0.5.
        eff = abs(auc - 0.5)
        p_val = float(np.mean(np.abs(perm_aucs - 0.5) >= eff))
        if p_val <= 0.05:
            tier = 1
        elif p_val <= 0.20:
            tier = 2
        else:
            tier = 3
        rows.append({
            "feature": "signal_bar_direction_is_up",
            "carry_family": "signal_bar_direction (new pre-registration; arc-3 step-2 finding)",
            "algo": algo, "K": K, "target_cluster_id": target_cid,
            "auc_vs_target": auc, "abs_lift_from_0.5": eff,
            "perm_p_value": p_val, "tier": tier,
            "n_perms": int(perms), "is_historical_arc1_carrier": False,
            "is_arc2_carry_promoted_to_mandatory": False,
            "is_new_pre_registration_arc3": True,
        })

    out = OUT / "signal_bar_direction_carry.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(f"  wrote {out}")
    return out


def _read_csv_or_none(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _compute_jpy_vs_nonjpy() -> Dict[str, Any]:
    feats = _load_signals_features()
    jpy_pairs = {"AUD_JPY", "CAD_JPY", "CHF_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "USD_JPY"}
    feats["is_jpy"] = feats["pair"].isin(jpy_pairs)
    out = {}
    for label, mask in (("jpy", feats["is_jpy"]), ("non_jpy", ~feats["is_jpy"])):
        sub = feats[mask]
        net_r = sub["net_r"].to_numpy(dtype=float)
        out[label] = {
            "n": int(len(sub)),
            "mean_net_r": float(np.nanmean(net_r)),
            "std_net_r": float(np.nanstd(net_r, ddof=1)) if len(net_r) >= 2 else float("nan"),
            "fold_pos_count": int((sub.groupby("fold_id")["net_r"].mean() > 0).sum()),
        }
    return out


def _build_verdict_table() -> str:
    """Populate the verdict placeholder table from the actual phase outputs.

    Six columns (K=2 kmeans target+mirror, K=2 hierarchical target+mirror,
    HDBSCAN target+mirror) × four dual-gate conditions.
    """
    target_sel = _load_target_selection()
    pred_auc = _read_csv_or_none(OUT / "predictor_AUC_signal_time.csv")
    pred_by_t = _read_csv_or_none(OUT / "predictor_AUC_by_cluster_by_t.csv")
    effect_sz = _read_csv_or_none(OUT / "cluster_effect_sizes.csv")
    size_elig = _read_csv_or_none(OUT / "cluster_size_eligibility.csv")
    stability = _read_csv_or_none(OUT / "cluster_stability.csv")

    # Build column keys: (algo, K, role) — only K=2 + HDBSCAN per task spec.
    cols: List[Tuple[str, int, str, int]] = []
    for _, t in target_sel.iterrows():
        algo = str(t["algo"]); K = int(t["K"])
        if algo == "hdbscan" or K == 2:
            for role in ("target", "mirror"):
                cid = int(t[f"{role}_cluster_id"])
                if cid in (-999, -2):
                    continue
                cols.append((algo, K, role, cid))

    def _auc_cell(algo: str, K: int, role: str, cid: int) -> str:
        if pred_auc.empty:
            return "n/a"
        # signal-time AUC for THIS cluster as target
        sub = pred_auc[
            (pred_auc["algo"] == algo) & (pred_auc["K"] == K) &
            (pred_auc["target_cluster_id"] == cid)
        ] if "target_cluster_id" in pred_auc.columns else pd.DataFrame()
        max_st = float("nan") if sub.empty else float(sub["pooled_auc"].max()) if "pooled_auc" in sub.columns else float(sub.select_dtypes(include="number").max().max())
        # held-bar AUC
        sub_t = pred_by_t[
            (pred_by_t["algo"] == algo) & (pred_by_t["K"] == K) &
            (pred_by_t["target_cluster_id"] == cid)
        ] if (not pred_by_t.empty and "target_cluster_id" in pred_by_t.columns) else pd.DataFrame()
        max_held = float("nan") if sub_t.empty else float(sub_t["pooled_auc"].max()) if "pooled_auc" in sub_t.columns else float("nan")
        cond_pass = (
            (np.isfinite(max_st) and max_st > 0.65)
            or (np.isfinite(max_held) and max_held > 0.70)
        )
        return f"{'PASS' if cond_pass else 'FAIL'} (sig-time max AUC {max_st:.3f}; held t≤20 max AUC {max_held:.3f})"

    def _effect_cell(algo: str, K: int, role: str, cid: int) -> str:
        if effect_sz.empty:
            return "n/a"
        sub = effect_sz[
            (effect_sz["algo"] == algo) & (effect_sz["K"] == K) &
            (effect_sz["cluster_id"] == cid)
        ] if "cluster_id" in effect_sz.columns else pd.DataFrame()
        if sub.empty:
            return "n/a (no row in cluster_effect_sizes.csv)"
        # Count number of effect-size criteria PASSed (op-spec §8)
        passes = 0; total = 0
        for col in sub.columns:
            if col.startswith("passes_") and col.endswith("_threshold"):
                total += 1
                if bool(sub.iloc[0][col]):
                    passes += 1
        if total == 0:
            return "n/a (no passes_* columns in cluster_effect_sizes.csv)"
        cond_pass = passes >= 2
        return f"{'PASS' if cond_pass else 'FAIL'} ({passes}/{total} thresholds)"

    def _size_cell(algo: str, K: int, role: str, cid: int) -> str:
        if size_elig.empty:
            return "n/a"
        sub = size_elig[
            (size_elig["algo"] == algo) & (size_elig["K"] == K) &
            (size_elig["cluster_id"] == cid)
        ] if "cluster_id" in size_elig.columns else pd.DataFrame()
        if sub.empty:
            return "n/a"
        frac = float(sub.iloc[0].get("frac_of_pool", float("nan")))
        size_pass_filter = frac >= 0.15
        size_pass_other = frac >= 0.05
        if size_pass_filter:
            return f"PASS filter-TO (size {frac:.3f})"
        if size_pass_other:
            return f"PASS exit/delayed only (size {frac:.3f})"
        return f"FAIL (size {frac:.3f})"

    def _stab_cell(algo: str, K: int, role: str, cid: int) -> str:
        if stability.empty:
            return "n/a"
        sub = stability[
            (stability["algo"] == algo) & (stability["K"] == K) &
            (stability["cluster_id"] == cid)
        ] if "cluster_id" in stability.columns else pd.DataFrame()
        if sub.empty:
            return "n/a"
        cv = sub.iloc[0].get("size_cv", float("nan"))
        ari = sub.iloc[0].get("ari_vs_pool", float("nan"))
        cv_pass = (np.isfinite(cv) and cv < 0.50)
        ari_pass = (np.isfinite(ari) and ari > 0.30)
        return f"{'PASS' if (cv_pass and ari_pass) else 'FAIL'} (size_CV {cv:.3f}; ARI {ari:.3f})"

    lines: List[str] = []
    lines.append("## Verdict")
    lines.append("")
    lines.append("Dual-gate condition matrix per (algorithm, K, target/mirror) — populated by Phase J from")
    lines.append("actual Phase A-H outputs. Verdict reasoning is **[planner to write]** below the table.")
    lines.append("")
    lines.append("| Condition | " + " | ".join(f"{a} K={k} {r}" for (a, k, r, _) in cols) + " |")
    lines.append("|---" + "|---" * len(cols) + "|")

    rows_data = [
        ("AUC > 0.65 (3c) OR > 0.70 (3d t≤20)", _auc_cell),
        ("Effect size (op-spec §8 thresholds)", _effect_cell),
        ("Cluster size ≥ 15% (filter-TO) OR ≥ 5% (exit/delayed)", _size_cell),
        ("Stability (per-fold size CV < 0.50 AND ARI > 0.30)", _stab_cell),
    ]
    for label, fn in rows_data:
        cells = [fn(a, k, r, cid) for (a, k, r, cid) in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Planner reasoning: [planner to write]**")
    lines.append("")
    lines.append("Verdict: **[PROCEED / AMBIGUOUS-PROBE / CLOSE — planner to fill]**")
    lines.append("")
    return "\n".join(lines)


def _build_arc3_sections() -> str:
    """Build the arc-3-specific §X-§AB sections to append to phase doc."""
    # Read step 2's variance compression report headline.
    var_path = STEP2_DIR / "variance_compression_report.txt"
    var_text = var_path.read_text(encoding="utf-8") if var_path.exists() else ""

    # Up/down split data — from step 2.
    ud_path = STEP2_DIR / "up_down_split.csv"
    ud = pd.read_csv(ud_path) if ud_path.exists() else pd.DataFrame()
    ud_pool = ud[ud["scope"] == "pool"] if not ud.empty else pd.DataFrame()

    # Cluster-direction-split (this phase J).
    cds_path = OUT / "cluster_direction_split.csv"
    cds = pd.read_csv(cds_path) if cds_path.exists() else pd.DataFrame()

    # Phase B per-fold breakdown (cluster_stability.csv).
    stab_path = OUT / "cluster_stability.csv"
    fold_break_path = OUT / "cluster_fold_breakdown.csv"
    stab = pd.read_csv(stab_path) if stab_path.exists() else pd.DataFrame()
    fold_break = pd.read_csv(fold_break_path) if fold_break_path.exists() else pd.DataFrame()

    # JPY-vs-non-JPY
    jpy = _compute_jpy_vs_nonjpy()

    # Cross-arc carry status — read cross_arc_portfolio_family.csv + signal_bar_direction_carry.csv.
    fam = _read_csv_or_none(OUT / "cross_arc_portfolio_family.csv")
    sbd = _read_csv_or_none(OUT / "signal_bar_direction_carry.csv")

    lines: List[str] = []

    # §X variance-actually-wider
    lines.append("## §X. Arc-open §4 variance-compression prediction INVERTED (step-2 carry-forward)")
    lines.append("")
    lines.append("Per arc-open §4 \"anticipated structural issue — regime-conditioning tautology\",")
    lines.append("vol-axis and concurrent-signal axis within-pool variance was expected to be COMPRESSED")
    lines.append("relative to arc 2. Step 2's [variance_compression_report.txt](../step2_descriptive/variance_compression_report.txt)")
    lines.append("shows the actual measurements:")
    lines.append("")
    lines.append("```")
    lines.append(var_text.strip())
    lines.append("```")
    lines.append("")
    lines.append("Descriptive observation: every measured feature (vol-axis and concurrent-signal axis)")
    lines.append("has arc-3 IQR ≥ arc-2 IQR. The §4 \"compression tautology\" framing does not hold;")
    lines.append("arc 3's within-pool dispersion is WIDER than arc 2's on the predicted-tautology axes.")
    lines.append("Step 3 phase C signal-time predictor scan reads the actual data, not the §4 expectation.")
    lines.append("")

    # §Y up/down split structural note
    lines.append("## §Y. Up/down bar split — structural note (step 2 + cluster-direction split)")
    lines.append("")
    lines.append("Step 2 surfaced a strong directional asymmetry on takes:")
    lines.append("")
    if not ud_pool.empty:
        lines.append("| direction | n_fires | n_takes | take_rate | frac_of_pool_takes | mean_net_r |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in ud_pool.iterrows():
            lines.append(
                f"| {r['direction']} | {int(r['n_fires']):,} | {int(r['n_takes']):,} | "
                f"{float(r['take_rate_within_direction']):.4f} | "
                f"{float(r['frac_of_pool_takes']):.4f} | {float(r['mean_net_r']):+.4f} |"
            )
    else:
        lines.append("(up_down_split.csv not available)")
    lines.append("")
    lines.append("Cluster-direction split (this phase J — `cluster_direction_split.csv`):")
    lines.append("")
    if not cds.empty:
        k2km = cds[(cds["algo"] == "kmeans") & (cds["K"] == 2)]
        if not k2km.empty:
            lines.append("K=2 kmeans:")
            lines.append("")
            lines.append("| role | cluster_id | direction | frac_of_cluster | mean_net_r |")
            lines.append("|---|---:|:---|---:|---:|")
            for _, r in k2km.iterrows():
                lines.append(
                    f"| {r['role']} | {int(r['cluster_id'])} | {r['direction']} | "
                    f"{float(r['frac_of_cluster']):.4f} | {float(r['mean_net_r']):+.4f} |"
                )
    lines.append("")
    lines.append("Structural support data; no verdict implication drawn here. Step 4 candidate")
    lines.append("derivation will read this table.")
    lines.append("")

    # §Z Phase B fold-stability F5/F6 narrative
    lines.append("## §Z. Phase B fold-stability — F5 / F6 cross-arc divergence (elevated narrative)")
    lines.append("")
    lines.append("Step 1 flagged fold-5 concurrence (arc 2 +93%, arc 3 +49%) and fold-6 divergence")
    lines.append("(arc 2 marginally positive, arc 3 −48.65%). Phase B reports per-fold cluster")
    lines.append("size + mean R for K=2 kmeans:")
    lines.append("")
    if not fold_break.empty:
        k2 = fold_break[(fold_break["algo"] == "kmeans") & (fold_break["K"] == 2)]
        if not k2.empty:
            lines.append("```")
            lines.append(k2.to_string(index=False))
            lines.append("```")
    else:
        lines.append("(cluster_fold_breakdown.csv not produced — see cluster_stability.csv for summary CV)")
    lines.append("")

    # §AA JPY-vs-non-JPY
    lines.append("## §AA. Phase G JPY vs non-JPY summary")
    lines.append("")
    lines.append("Per arc-open §4 cross-arc question: \"does any pair appear in the dragger-tail across")
    lines.append("multiple arcs? Arc 2 surfaced JPY-pairs as a positive-concentration set.\"")
    lines.append("")
    lines.append("| set | n | mean_net_r | std_net_r | folds with positive mean R |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| JPY pairs (7) | {jpy['jpy']['n']:,} | {jpy['jpy']['mean_net_r']:+.4f} | "
        f"{jpy['jpy']['std_net_r']:.4f} | {jpy['jpy']['fold_pos_count']}/7 |"
    )
    lines.append(
        f"| non-JPY pairs (21) | {jpy['non_jpy']['n']:,} | {jpy['non_jpy']['mean_net_r']:+.4f} | "
        f"{jpy['non_jpy']['std_net_r']:.4f} | {jpy['non_jpy']['fold_pos_count']}/7 |"
    )
    lines.append("")

    # §AB Cross-arc carry status table
    lines.append("## §AB. Cross-arc carry status summary")
    lines.append("")
    lines.append("Three carry families evaluated per arc-open §4 mandate:")
    lines.append("  1. Concurrent-signal sub-family (arc-2 Tier-1 carry, promoted to mandatory for arc 3)")
    lines.append("  2. Cross-pair / portfolio currency-basket family (arc-1 Tier-1 / arc-2 Tier-2/3 carry)")
    lines.append("  3. signal_bar_direction (NEW pre-registration per arc-3 step-2 finding)")
    lines.append("")
    lines.append("Carry-1 + Carry-2 (from `cross_arc_portfolio_family.csv`, Phase F):")
    lines.append("")
    if not fam.empty:
        keep_cols = [c for c in fam.columns if c in ("feature", "algo", "K", "target_cluster_id", "auc_vs_target", "perm_p_value", "tier")]
        lines.append("```")
        lines.append(fam[keep_cols].to_string(index=False) if keep_cols else fam.to_string(index=False))
        lines.append("```")
    else:
        lines.append("(cross_arc_portfolio_family.csv not produced)")
    lines.append("")
    lines.append("Carry-3 (from `signal_bar_direction_carry.csv`, Phase J — new pre-registration):")
    lines.append("")
    if not sbd.empty:
        lines.append("```")
        keep_cols = [c for c in sbd.columns if c in ("feature", "algo", "K", "target_cluster_id", "auc_vs_target", "perm_p_value", "tier")]
        lines.append(sbd[keep_cols].to_string(index=False) if keep_cols else sbd.to_string(index=False))
        lines.append("```")
    else:
        lines.append("(signal_bar_direction_carry.csv not produced)")
    lines.append("")

    return "\n".join(lines)


def append_to_phase_doc() -> None:
    """Append arc-3 sections + Verdict placeholder table to phase doc.

    If phase doc doesn't have a ## Verdict section yet, append at end.
    If it does, replace it with our populated version.
    """
    if not PHASE_DOC.exists():
        print(f"  WARN: {PHASE_DOC} does not exist; creating minimal scaffold")
        PHASE_DOC.write_text(
            "# PHASE_L_ARC_3_STEP3 — Extractability Assessment\n\n"
            "(main phase doc body not yet generated)\n\n",
            encoding="utf-8",
        )

    doc = PHASE_DOC.read_text(encoding="utf-8")
    arc3_block = _build_arc3_sections()
    verdict_block = _build_verdict_table()

    # Strip any pre-existing Verdict section.
    if "## Verdict" in doc:
        head = doc.split("## Verdict", 1)[0]
    else:
        head = doc

    # If our arc-3 block is already present, replace it.
    arc3_marker = "## §X. Arc-open §4 variance-compression prediction INVERTED"
    if arc3_marker in head:
        head = head.split(arc3_marker, 1)[0]

    new_doc = head.rstrip() + "\n\n" + arc3_block + "\n" + verdict_block + "\n"
    PHASE_DOC.write_text(new_doc, encoding="utf-8")
    print(f"  appended arc-3 sections + verdict placeholder to {PHASE_DOC}")


def main(rewrite_phase_doc: bool = False) -> None:
    print("[Phase J] computing cluster_direction_split.csv...")
    cluster_direction_split()
    print("[Phase J] computing signal_bar_direction carry tiers...")
    signal_bar_direction_carry_tiers()
    if rewrite_phase_doc:
        print("[Phase J] appending arc-3 sections + verdict placeholder to phase doc...")
        append_to_phase_doc()
    else:
        print("[Phase J] skipping phase-doc append (use --rewrite-phase-doc to enable).")
        print("[Phase J]   Rationale: append_to_phase_doc()'s split markers are fragile and can")
        print("[Phase J]   truncate an enriched phase doc. The doc is authored once at step-3 close;")
        print("[Phase J]   re-runs only refresh the two arc-3-specific CSVs.")
    print("[Phase J] done.")


if __name__ == "__main__":
    import sys
    main(rewrite_phase_doc="--rewrite-phase-doc" in sys.argv[1:])
