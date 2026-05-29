"""Arc 8 v3.0.2 Step 4 — extraction (canonical engine path).

Per L_PROTOCOL §2 Step 4 + Amendment 2 + intent doc §2 / §4:

Uses the canonical `core.steps.step_4_extraction.run_step_4` (PR #185)
with:
  - `train_end=2021-01-01T00:00:00Z` → restricts CV + classifier refit
    to IS-only (holdout-window training filter, PR #185 contract)
  - `persistence_dir=step_4/classifiers/` → joblib-pickles best-AUC
    classifier per candidate cluster with SHA256 manifest. A2/A6 at
    Step 5 instantiate via `build_a2_config_from_step4` /
    `build_a6_config_from_step4` (no Step 5 retrain per Amendment 2).
  - Lineage filter active → features tagged anything other than `clean`
    are excluded from training (PR #185 column-name reconcile).

Then evaluates Amendment 5 four-gate architecture admission per
candidate cluster, citing the admitting gate per architecture.

HALT trigger (soft canary per chat auto-run §3): if any candidate cluster's
best classifier mean OOS AUC exceeds 0.85, suspect a lookahead-class issue
and HALT for chat review. No drift HALT against any pre-existing baseline
(intent doc §4 — no applicable reference value under canonical 5ers_eet +
canonical W1 producer).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.determinism import seed_everything, write_text_deterministic
from core.manifest import write_manifest
from core.steps.step_4_extraction import run_step_4
from scripts.l_arc_8_v3_0_2.shared import ARC_NAME, RESULTS_ROOT, TRAIN_END_ISO

STEP_DIR: Path = RESULTS_ROOT / "step_4"
DIAGNOSTIC_PATH: Path = Path("docs/dispatches/arc_8_v3_0_2_diagnostic.md")

AMENDMENT_5_GATE_2_AUC: float = 0.65
IMPLAUSIBLY_HIGH_AUC_THRESHOLD: float = 0.85


def _amendment_5_admission(
    cap_df: pd.DataFrame,
    step4_per_cluster_aucs: dict[int, float],
    n_candidate_clusters: int,
) -> dict:
    """Resolve the architecture set per cluster under L_PROTOCOL Amendment 5.

    Returns dict keyed by cluster_id with:
      - archetype, mean_oos_auc, gates_fired, architectures_admitted,
        skip_reason (None if not skipped), amendment_1_archs (what would
        have been tested under prior rule, for `architectures_skipped_by_amendment_5`).
    """
    out: dict[int, dict] = {}
    for _, row in cap_df.iterrows():
        cid = int(row["cluster_id"])
        archetype = str(row["archetype"])
        is_candidate = bool(row["candidate"])

        # Choppy clusters skip ALL architectures regardless of other gates.
        if archetype == "Choppy":
            out[cid] = {
                "archetype": archetype,
                "mean_oos_auc": None,
                "gates_fired": [],
                "architectures_admitted": [],
                "amendment_1_archs": [],
                "skip_reason": "amendment_5_cluster_skip_choppy",
                "is_candidate": is_candidate,
            }
            continue

        # Non-candidate clusters (Step 3 dies_step3) are not evaluated at Step 5.
        if not is_candidate:
            out[cid] = {
                "archetype": archetype,
                "mean_oos_auc": None,
                "gates_fired": [],
                "architectures_admitted": [],
                "amendment_1_archs": [],
                "skip_reason": "step_3_not_candidate",
                "is_candidate": is_candidate,
            }
            continue

        auc = step4_per_cluster_aucs.get(cid)

        gates_fired: list[str] = []
        archs: set[str] = set()
        amendment_1_archs: set[str] = set()

        # Gate 1 — Shape-required (archetype-driven)
        archetype_g1_map = {
            "V-shape recovery": ["A3"],
            "Stepwise climber": ["A4"],
            "Bimodal": ["A4"],
        }
        g1_archs = archetype_g1_map.get(archetype, [])
        if g1_archs:
            gates_fired.append(f"Gate 1 ({archetype})")
            archs.update(g1_archs)
            amendment_1_archs.update(g1_archs)

        # Gate 2 — Classifier-driven (AUC >= 0.65 → add A2 + A6)
        if auc is not None and auc >= AMENDMENT_5_GATE_2_AUC:
            gates_fired.append(f"Gate 2 (AUC={auc:.4f} >= {AMENDMENT_5_GATE_2_AUC})")
            archs.update(["A2", "A6"])

        # Amendment 1 (prior rule) admitted A2/A6 for certain archetypes
        # regardless of AUC. Recording what Amendment 1 would have tested but
        # Amendment 5 skips → `architectures_skipped_by_amendment_5`.
        amendment_1_archs_by_archetype = {
            "V-shape recovery": ["A6"],
            "Stepwise climber": ["A2"],
            "Monotonic up": ["A2", "A6"],
        }
        amendment_1_archs.update(amendment_1_archs_by_archetype.get(archetype, []))

        # Gate 3 — Universal A1
        gates_fired.append("Gate 3 (universal A1)")
        archs.add("A1")
        amendment_1_archs.add("A1")

        # Gate 4 — Portfolio (A5) condition (a) only if >= 2 candidate clusters.
        # Amendment 5.1 also requires (b) at-least-one-PASS-tier-constituent
        # post-Step-5; that's resolved at closure time. At dispatch time we
        # only know whether (a) holds.
        if n_candidate_clusters >= 2:
            gates_fired.append(f"Gate 4(a) (n_candidate_clusters={n_candidate_clusters} >= 2)")
            archs.add("A5")
            amendment_1_archs.add("A5")

        out[cid] = {
            "archetype": archetype,
            "mean_oos_auc": auc,
            "gates_fired": gates_fired,
            "architectures_admitted": sorted(archs),
            "amendment_1_archs": sorted(amendment_1_archs),
            "skip_reason": None,
            "is_candidate": is_candidate,
        }
    return out


def _build_admission_md(admission: dict) -> str:
    lines = [
        f"# {ARC_NAME} — Amendment 5 Architecture Admission",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        "Per L_PROTOCOL Amendment 5 (`archive/L_PROTOCOL_v3_0_AMENDMENT_5.md`) — "
        "four-gate union evaluated at dispatch time from observed Step 4 AUC.",
        "",
        "## Per-cluster admission table",
        "",
        "| Cluster | Archetype | Is candidate? | Step 4 mean OOS AUC | Gates fired | Architectures admitted | Skip reason | Amendment 1 set (for `architectures_skipped_by_amendment_5`) |",
        "|---:|---|:---:|---:|---|---|---|---|",
    ]
    for cid in sorted(admission.keys()):
        info = admission[cid]
        auc_str = f"{info['mean_oos_auc']:.4f}" if info["mean_oos_auc"] is not None else "—"
        gates_str = "; ".join(info["gates_fired"]) if info["gates_fired"] else "—"
        archs_str = ", ".join(info["architectures_admitted"]) if info["architectures_admitted"] else "**(skip)**"
        skip_str = info["skip_reason"] or "—"
        a1_str = ", ".join(info["amendment_1_archs"]) if info["amendment_1_archs"] else "—"
        cand_str = "YES" if info["is_candidate"] else "no"
        lines.append(
            f"| {cid} | {info['archetype']} | {cand_str} | {auc_str} | "
            f"{gates_str} | {archs_str} | {skip_str} | {a1_str} |"
        )

    # architectures_skipped_by_amendment_5 (closure template v1.3.1 field)
    all_admitted: set[str] = set()
    all_amendment_1: set[str] = set()
    for cid, info in admission.items():
        all_admitted.update(info["architectures_admitted"])
        all_amendment_1.update(info["amendment_1_archs"])
    skipped = sorted(all_amendment_1 - all_admitted)

    lines += [
        "",
        "## Closure tracker field projection",
        "",
        f"- `architectures_skipped_by_amendment_5: {skipped}` (closure template v1.3.1)",
        f"- Total architectures admitted across clusters: {sorted(all_admitted)}",
        "",
        "## Step 5 config-count estimate (Bimodal 3-exit canonical slate)",
        "",
        "Per L_PROTOCOL §2 Step 5 Bimodal exit slate: "
        "`{sl_only, sl_partial_close_1r_runner_trail, sl_plus_tp_2r}` (3 exits).",
        "Per-arch counts assume 3 SL multipliers × 3 exits × 2 exposure caps.",
        "",
    ]
    for cid in sorted(admission.keys()):
        info = admission[cid]
        if not info["architectures_admitted"]:
            continue
        per_arch_estimate = {
            "A1": 18,   # 1 × 3 SLs × 3 exits × 2 exposures
            "A2": 18,   # 1 threshold × 3 SLs × 3 exits × 2 exposures
            "A3": 36,   # × 2 N-bar values (not Bimodal-admitted but included for completeness)
            "A4": 18,   # 1 classifier × 3 SLs × 3 exits × 2 exposures
            "A5": 1,    # portfolio composition only if >=2 clusters; deferred build
            "A6": 54,   # 3 threshold pairs × 3 SLs × 3 exits × 2 exposures
        }
        total = sum(per_arch_estimate.get(a, 0) for a in info["architectures_admitted"])
        archs = info["architectures_admitted"]
        lines.append(f"- Cluster {cid} ({info['archetype']}): {archs} → ~{total} configs")
    return "\n".join(lines)


def main() -> Path:
    seed_everything(42)
    t0 = time.perf_counter()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    (STEP_DIR / "classifiers").mkdir(exist_ok=True)
    step1 = RESULTS_ROOT / "step_1"
    step2 = RESULTS_ROOT / "step_2"
    step3 = RESULTS_ROOT / "step_3"

    pool = pd.read_parquet(step1 / "pool.parquet")
    feat = pd.read_parquet(step1 / "features.parquet")
    lineage = pd.read_csv(step1 / "feature_lineage.csv")
    assignments = pd.read_parquet(step2 / "cluster_assignments.parquet")
    cap = pd.read_csv(step3 / "capturability.csv")

    # Canonical engine path expects `cluster_id` (not `cluster_primary`).
    cluster_assignments = assignments[["trade_id", "cluster_primary"]].rename(
        columns={"cluster_primary": "cluster_id"}
    )

    # Identify candidate clusters per Step 3.
    candidate_cluster_ids = tuple(
        sorted(int(cid) for cid in cap.loc[cap["candidate"], "cluster_id"])
    )
    if not candidate_cluster_ids:
        # Per L_PROTOCOL §2 Step 3 failure-diag: highest-composite cluster.
        cid_fallback = int(cap.sort_values("capturability_composite", ascending=False).iloc[0]["cluster_id"])
        candidate_cluster_ids = (cid_fallback,)
        print(f"[step4] no candidates pass §3 floors; proceeding on cluster {cid_fallback}")
    print(f"[step4] candidate cluster ids: {candidate_cluster_ids}")
    print(f"[step4] train_end (PR #185 IS filter): {TRAIN_END_ISO}")

    s4 = run_step_4(
        trades=pool,
        feature_matrix=feat,
        cluster_assignments=cluster_assignments,
        feature_lineage=lineage,
        candidate_cluster_ids=candidate_cluster_ids,
        n_ts_folds=5,
        persistence_dir=STEP_DIR / "classifiers",
        arc_name=ARC_NAME,
        train_end=pd.Timestamp(TRAIN_END_ISO),
    )

    # Emit canonical artefacts.
    metrics_rows = []
    importance_rows = []
    for ce in s4.per_cluster:
        # Per-classifier mean AUC + fold AUCs
        by_clf: dict[str, list[float]] = {}
        thr_by_clf: dict[str, list[float]] = {}
        for fr in ce.classifier_fold_results:
            by_clf.setdefault(fr.classifier, []).append(fr.auc)
            thr_by_clf.setdefault(fr.classifier, []).append(fr.threshold_auc_best)
        for clf_name, aucs in by_clf.items():
            metrics_rows.append({
                "cluster_id": ce.cluster_id,
                "model": clf_name,
                "n_trades": ce.n_trades,
                "mean_auc": float(np.mean(aucs)),
                "std_auc": float(np.std(aucs)),
                "fold_aucs": ";".join(f"{a:.4f}" for a in aucs),
                "auc_best_threshold_mean": float(np.mean(thr_by_clf[clf_name])),
                "is_best": clf_name == ce.best_classifier,
                "best_threshold": ce.best_threshold if clf_name == ce.best_classifier else None,
                "fitted_classifier_path": (
                    str(ce.fitted_classifier_path) if clf_name == ce.best_classifier else None
                ),
            })
        for _, r in ce.feature_importance.iterrows():
            importance_rows.append({
                "cluster_id": ce.cluster_id,
                "feature": r["feature"],
                "mean_importance": r["mean_importance"],
                "std_importance": r["std_importance"],
            })

    metrics_df = pd.DataFrame(metrics_rows)
    importance_df = pd.DataFrame(importance_rows)
    metrics_path = STEP_DIR / "extraction_metrics.csv"
    importance_path = STEP_DIR / "feature_importance.csv"
    metrics_df.to_csv(metrics_path, index=False, lineterminator="\n")
    importance_df.to_csv(importance_path, index=False, lineterminator="\n")

    # Print per-cluster best-AUC summary
    per_cluster_best_aucs: dict[int, float] = {}
    for ce in s4.per_cluster:
        per_cluster_best_aucs[int(ce.cluster_id)] = float(ce.best_classifier_mean_auc)
        print(
            f"[step4] cluster {ce.cluster_id}: best={ce.best_classifier} "
            f"mean_oos_auc={ce.best_classifier_mean_auc:.4f} "
            f"threshold={ce.best_threshold:.3f} n={ce.n_trades}"
        )

    # Soft canary HALT per chat auto-run §3: implausibly high AUC (>0.85)
    # on any candidate cluster suggests a lookahead-class issue (e.g., feature
    # leakage). HALT for chat review.
    halt_reasons: list[str] = []
    for cid, auc in per_cluster_best_aucs.items():
        if auc > IMPLAUSIBLY_HIGH_AUC_THRESHOLD:
            halt_reasons.append(
                f"c{cid}: mean OOS AUC {auc:.4f} exceeds {IMPLAUSIBLY_HIGH_AUC_THRESHOLD} "
                f"canary threshold — suspect lookahead-class feature leak"
            )

    if halt_reasons:
        DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DIAGNOSTIC_PATH.write_text(
            f"# {ARC_NAME} — Step 4 HALT (implausibly-high AUC canary)\n\n"
            f"Per chat auto-run §3 (no specific baseline; AUC > {IMPLAUSIBLY_HIGH_AUC_THRESHOLD} canary):\n\n"
            + "\n".join(f"- {r}" for r in halt_reasons) + "\n\n"
            f"Surfaced from: `{STEP_DIR / 'extraction_metrics.csv'}`\n"
            f"Per-cluster best AUCs: {per_cluster_best_aucs}\n\n"
            f"Chat review required before Step 5.\n",
            encoding="utf-8", newline="\n",
        )
        print(f"[step4] HALT — see {DIAGNOSTIC_PATH}")
        raise SystemExit(f"HALT: implausibly-high AUC canary fired ({halt_reasons[0]})")

    # Amendment 5 four-gate evaluation.
    admission = _amendment_5_admission(
        cap_df=cap,
        step4_per_cluster_aucs=per_cluster_best_aucs,
        n_candidate_clusters=len(candidate_cluster_ids),
    )
    admission_md_path = STEP_DIR / "amendment_5_admission.md"
    write_text_deterministic(admission_md_path, _build_admission_md(admission))
    admission_json_path = STEP_DIR / "amendment_5_admission.json"
    admission_serializable = {
        str(cid): {
            "archetype": info["archetype"],
            "mean_oos_auc": float(info["mean_oos_auc"]) if info["mean_oos_auc"] is not None else None,
            "gates_fired": info["gates_fired"],
            "architectures_admitted": info["architectures_admitted"],
            "amendment_1_archs": info["amendment_1_archs"],
            "skip_reason": info["skip_reason"],
            "is_candidate": info["is_candidate"],
        }
        for cid, info in admission.items()
    }
    admission_json_path.write_text(
        json.dumps(admission_serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

    # Summary md.
    summary_path = STEP_DIR / "extraction_summary.md"
    summary = _build_summary_md(
        s4=s4,
        metrics_df=metrics_df,
        importance_df=importance_df,
        per_cluster_best_aucs=per_cluster_best_aucs,
        admission=admission,
    )
    write_text_deterministic(summary_path, summary)

    # Manifest.
    write_manifest(
        STEP_DIR / "manifest.json",
        artefacts=[metrics_path, importance_path, summary_path, admission_md_path, admission_json_path],
    )
    elapsed = time.perf_counter() - t0
    print(f"[step4] DONE in {elapsed:.1f}s — {len(s4.per_cluster)} classifier(s) persisted")
    return STEP_DIR


def _build_summary_md(
    s4,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    per_cluster_best_aucs: dict[int, float],
    admission: dict,
) -> str:
    lines = [
        f"# {ARC_NAME} — Step 4 Extraction Summary",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        "- Engine path: `core.steps.step_4_extraction.run_step_4` (canonical, PR #185)",
        f"- IS filter: `train_end={TRAIN_END_ISO}` — restricts CV + classifier refit to entry_time < cutoff",
        f"- Candidate clusters processed: {len(s4.per_cluster)}",
        "- Classifiers: RF + LGBM + Logistic at Appendix A defaults",
        "- CV: 5-fold TimeSeriesSplit",
        f"- HALT canary: AUC > {IMPLAUSIBLY_HIGH_AUC_THRESHOLD} on any candidate cluster (per chat auto-run §3)",
        "",
        "## Per-cluster, per-model OOS AUC",
        "",
        "| Cluster | Model | n trades | Mean OOS AUC | Std AUC | Fold AUCs | AUC-best threshold (mean) | Best? |",
        "|---:|---|---:|---:|---:|---|---:|:---:|",
    ]
    for _, r in metrics_df.iterrows():
        best_mark = "★" if r["is_best"] else ""
        lines.append(
            f"| {int(r['cluster_id'])} | {r['model']} | {int(r['n_trades']):,} | "
            f"{r['mean_auc']:.4f} | {r['std_auc']:.4f} | {r['fold_aucs']} | "
            f"{r['auc_best_threshold_mean']:.3f} | {best_mark} |"
        )

    lines += [
        "",
        "## Best classifier per cluster (drives Step 5 A2/A6 via Amendment 2)",
        "",
        "| Cluster | Best model | Mean AUC | Best threshold | Persisted classifier |",
        "|---:|---|---:|---:|---|",
    ]
    for ce in s4.per_cluster:
        lines.append(
            f"| {ce.cluster_id} | {ce.best_classifier} | "
            f"{ce.best_classifier_mean_auc:.4f} | {ce.best_threshold:.3f} | "
            f"`{ce.fitted_classifier_path}` |"
        )

    lines += [
        "",
        "## Amendment 5 Gate 2 decision (drives Step 5 A2/A6 admission)",
        "",
    ]
    for cid, info in admission.items():
        if info["mean_oos_auc"] is None:
            continue
        auc = info["mean_oos_auc"]
        if auc >= AMENDMENT_5_GATE_2_AUC:
            lines.append(
                f"- **c{cid} ({info['archetype']}):** mean OOS AUC {auc:.4f} >= {AMENDMENT_5_GATE_2_AUC} "
                f"→ Gate 2 FIRES → **A2 + A6 admitted**."
            )
        else:
            lines.append(
                f"- **c{cid} ({info['archetype']}):** mean OOS AUC {auc:.4f} < {AMENDMENT_5_GATE_2_AUC} "
                f"→ Gate 2 does NOT fire → **A2 + A6 skipped**."
            )

    lines += [
        "",
        "## Informational — prior arc references (NOT baselines)",
        "",
        "- Arc 8 v3.0 (UTC + multi_tf all-NaN) c2 V-shape RF mean OOS AUC: 0.5300. NOT a baseline — different methodology.",
        "- Prior `arc/l_arc_8_v3.0.2_halted` (UTC + W1 leaking) c2 V-shape mean AUC: 0.6938. NOT a baseline — driven by lookahead per verification doc §1.3.",
        "- v3.0.2 (5ers_eet + canonical W1) per-cluster AUCs above are the canonical observation under this methodology.",
        "",
        "## Excluded features (lineage filter, PR #185)",
        "",
    ]
    if s4.per_cluster and s4.per_cluster[0].excluded_features:
        lines.append("Per L_PROTOCOL §2 Step 4 + Amendment 2: features tagged anything other than `clean` excluded from training.")
        lines.append("")
        for f in s4.per_cluster[0].excluded_features:
            lines.append(f"- `{f}`")
    else:
        lines.append("(none — all features carried `causal_lineage = clean` tag)")

    lines += ["", "## Top-10 permutation-importance features per cluster", ""]
    for cid in sorted(importance_df["cluster_id"].unique()):
        sub = importance_df[importance_df["cluster_id"] == cid].sort_values(
            "mean_importance", ascending=False
        ).head(10)
        lines.append(f"### Cluster {int(cid)}")
        lines.append("")
        lines.append("| Feature | Mean importance | Std |")
        lines.append("|---|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['feature']} | {r['mean_importance']:+.5f} | {r['std_importance']:.5f} |")
        lines.append("")

    lines += [
        "",
        "## Amendment 5 architecture admission (drives Step 5)",
        "",
        "See `amendment_5_admission.md` for the per-cluster four-gate evaluation and "
        "`architectures_skipped_by_amendment_5` projection for the closure tracker.",
        "",
        "## Methodology notes",
        "",
        f"- Engine `core.steps.step_4_extraction.run_step_4` with `train_end={TRAIN_END_ISO}` "
        "and `persistence_dir=step_4/classifiers/`.",
        "- Target = binary cluster membership (cluster_id == cid).",
        "- TimeSeriesSplit preserves chronological order; PR #185 restricts CV + refit to IS-only.",
        "- Best-AUC classifier refit on full lineage-filtered IS pool; pickled with SHA256 + provenance manifest.",
        "- A2/A6 at Step 5 instantiate via `build_a2_config_from_step4` / `build_a6_config_from_step4` (no Step 5 retrain).",
        "- Amendment 5 four-gate selection applied at dispatch time per observed AUC (no chat override).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
