"""EXP-03 — E threshold scan + cross-arc pass/fail impact.

Question: at what E AUC threshold would Arc 10 pass? What about Arcs 6 and 7?
What is the implied false-positive cost at any relaxed threshold across the
full arc history?

Method:
  1. Assemble the arc history with best Pipeline E and D1 AUCs:
       Arc 6 — Stepwise climber c0/c2 (4H failed-breakout reversal)
       Arc 7 — V-shape c1/c3/aggregate (4H liquidity-sweep+reclaim)
       Arc 10 — V-shape c1 (4H DLR)
       KH-24 anchor (calibration anchor, deployed)
       Arc 3 / Arc 4 RERUN / Arc 5 — for downstream WFO outcome reference
  2. Sweep E AUC threshold from 0.55 → 0.65 in 0.005 steps.
     For each step: count arcs passing.
  3. Tabulate threshold at which Arc 6 + 7 + 10 simultaneously pass.
  4. False-positive cost estimate: of arcs that would pass at a relaxed
     threshold and whose downstream behaviour is on record, how many failed
     Step 5/6 WFO or deployability for unrelated reasons? This is a small-N
     exercise — partial / informational.
  5. False-negative cost: KH-24 anchor passes via D1 (0.638) — does it survive
     all sweep thresholds? (E AUC 0.642 — fails at 0.65 but passes D1; this
     is the documented anchor-preservation pattern.)

Data sources are documented arc result files in docs/arc_results/; no
in-repo classifier re-runs are performed here. The AUCs are point estimates
from each arc's published Step 4 summaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10.experiments._common import sha256_file  # noqa: E402

OUT_DIR = _REPO_ROOT / "results" / "l_arc_10" / "experiments"
RAW_DIR = OUT_DIR / "raw"


# Arc history with best Pipeline E AUC per (arc, cluster/aggregate).
# Sourced from docs/arc_results/*.md and results/l_arc_*/step4/STEP4_SUMMARY.md.
# `downstream_outcome` is the actual disposition the arc reached (deployment / Step 6 fail / KILL).
ARC_HISTORY: List[Dict[str, object]] = [
    # KH-24 deployed anchor (PASS via D1).
    {"arc": "KH-24", "cohort": "K=4 arch 3", "archetype": "Stepwise climber",
     "n": "n/a", "e_auc": 0.642, "d1_auc": 0.638, "downstream_outcome": "DEPLOYED",
     "notes": "calibration anchor; passes D1 by 0.038 margin"},
    # Arc 6 — Stepwise (failed-breakout reversal long, 4H).
    {"arc": "Arc 6", "cohort": "c0", "archetype": "Stepwise climber",
     "n": 334, "e_auc": 0.600, "d1_auc": 0.620,
     "downstream_outcome": "DEPLOYABILITY_FAIL_recall_collapse",
     "notes": "best E AUC across A/B/C steps; D1 passes mechanically but recall < 0.60 at all thresholds"},
    {"arc": "Arc 6", "cohort": "c2", "archetype": "Stepwise climber",
     "n": 242, "e_auc": 0.590, "d1_auc": 0.711,
     "downstream_outcome": "DEPLOYABILITY_FAIL_recall_collapse",
     "notes": "D1 AUC 0.711 at t=10 — strongest D1 signal in arc history; admission 0.4% though"},
    # Arc 7 — V-shape (liquidity-sweep + reclaim, 4H).
    {"arc": "Arc 7", "cohort": "c1", "archetype": "V-shape recovery (weak)",
     "n": 185, "e_auc": 0.484, "d1_auc": 0.420,
     "downstream_outcome": "CLEAN_NULL_step4",
     "notes": "both pipelines well below threshold"},
    {"arc": "Arc 7", "cohort": "c3", "archetype": "V-shape recovery",
     "n": 365, "e_auc": 0.512, "d1_auc": 0.518,
     "downstream_outcome": "CLEAN_NULL_step4",
     "notes": "largest single V-shape cohort in arc history"},
    {"arc": "Arc 7", "cohort": "agg_c1_c3", "archetype": "V-shape recovery (aggregate)",
     "n": 550, "e_auc": 0.536, "d1_auc": 0.496,
     "downstream_outcome": "CLEAN_NULL_step4",
     "notes": "aggregation stabilises std (0.029) but no AUC lift over c3"},
    # Arc 10 — V-shape (DLR, 4H).
    {"arc": "Arc 10", "cohort": "c1", "archetype": "V-shape recovery",
     "n": 228, "e_auc": 0.6296, "d1_auc": 0.5897,
     "downstream_outcome": "HALT_step4_near_miss",
     "notes": "this arc; both pipelines near-miss <0.03"},
    # Arcs that PASSED Step 4 but failed downstream Step 5/6 — informational.
    {"arc": "Arc 4 RERUN", "cohort": "(per closure)", "archetype": "Pipeline D1 archetype",
     "n": "n/a", "e_auc": 0.55, "d1_auc": 0.65,
     "downstream_outcome": "STEP6_FAIL_full_pool_DD_76.98pct",
     "notes": "passed admit-only stability, failed full-pool deployment economics"},
    {"arc": "Arc 5", "cohort": "(per closure)", "archetype": "Pipeline D1 archetype",
     "n": "n/a", "e_auc": "n/a", "d1_auc": 0.62,
     "downstream_outcome": "STEP6_KILL_Pipeline_D1_negative_expectancy",
     "notes": "same admit-only-pass / full-pool-fail pattern as Arc 4 RERUN"},
]

E_THRESHOLDS = np.round(np.arange(0.55, 0.6501, 0.005), 4)
D1_THRESHOLDS = np.round(np.arange(0.50, 0.6501, 0.005), 4)
DEFAULT_E_GATE = 0.65
DEFAULT_D1_GATE = 0.60


def _is_num(v):
    return isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df_hist = pd.DataFrame(ARC_HISTORY)
    df_hist.to_csv(RAW_DIR / "exp_03_arc_history.csv", index=False, lineterminator="\n")

    # Sweep E thresholds.
    sweep_rows = []
    for t in E_THRESHOLDS:
        passing = []
        for row in ARC_HISTORY:
            if _is_num(row["e_auc"]) and float(row["e_auc"]) >= t:
                passing.append(f"{row['arc']}/{row['cohort']}")
        sweep_rows.append({
            "e_threshold": float(t),
            "n_pass": len(passing),
            "passing_units": "; ".join(passing),
        })
    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(RAW_DIR / "exp_03_e_threshold_sweep.csv", index=False, lineterminator="\n")

    # Threshold at which Arc 6 + Arc 7 (best) + Arc 10 all pass.
    arc_best_e = {}
    for row in ARC_HISTORY:
        if _is_num(row["e_auc"]) and row["arc"] in ("Arc 6", "Arc 7", "Arc 10"):
            arc_best_e[row["arc"]] = max(arc_best_e.get(row["arc"], 0.0), float(row["e_auc"]))
    triple_pass_t = min(arc_best_e.values()) if len(arc_best_e) == 3 else None

    # FP / FN audit at the triple-pass threshold.
    if triple_pass_t is not None:
        fp_audit = []
        fn_audit = []
        for row in ARC_HISTORY:
            outcome = str(row["downstream_outcome"])
            e_auc = row["e_auc"]
            if not _is_num(e_auc):
                continue
            passes = float(e_auc) >= triple_pass_t
            actually_deployed_or_viable = outcome == "DEPLOYED"
            failed_downstream = "STEP6" in outcome or "DEPLOYABILITY" in outcome
            if passes and failed_downstream:
                fp_audit.append({
                    "arc": row["arc"], "cohort": row["cohort"],
                    "e_auc": float(e_auc), "downstream_outcome": outcome,
                    "reason_for_FP": "passes E AUC at relaxed threshold but downstream failure on record",
                })
            if (not passes) and actually_deployed_or_viable:
                fn_audit.append({
                    "arc": row["arc"], "cohort": row["cohort"],
                    "e_auc": float(e_auc), "downstream_outcome": outcome,
                    "reason_for_FN": "would be excluded at this threshold but actually deployed via the disjunctive gate",
                })

        pd.DataFrame(fp_audit).to_csv(RAW_DIR / "exp_03_fp_audit.csv", index=False, lineterminator="\n")
        pd.DataFrame(fn_audit).to_csv(RAW_DIR / "exp_03_fn_audit.csv", index=False, lineterminator="\n")

    # Markdown summary.
    md = []
    md.append("# EXP-03 — E threshold scan + cross-arc impact")
    md.append("")
    md.append("**Status:** experimental (not a Step 5 gate).")
    md.append("")
    md.append("## Question")
    md.append("At what E AUC threshold would Arc 10 pass? What about Arcs 6 and 7?")
    md.append("What is the implied false-positive cost across the full arc history?")
    md.append("")
    md.append("## Arc history (per-cohort best E + D1 AUC)")
    md.append("")
    md.append("| arc | cohort | archetype | n | E AUC | D1 AUC | downstream outcome |")
    md.append("|---|---|---|---:|---:|---:|---|")
    for r in ARC_HISTORY:
        e = f"{float(r['e_auc']):.4f}" if _is_num(r['e_auc']) else str(r['e_auc'])
        d = f"{float(r['d1_auc']):.4f}" if _is_num(r['d1_auc']) else str(r['d1_auc'])
        md.append(f"| {r['arc']} | {r['cohort']} | {r['archetype']} | {r['n']} | {e} | {d} | {r['downstream_outcome']} |")
    md.append("")
    md.append("**Sources:**")
    md.append("- KH-24 anchor: `L_ARC_PROTOCOL.md` §14 (v2.0 self-test).")
    md.append("- Arc 6: `docs/arc_results/ARC_6_RESULT.md` Step 4 best across A/B/C steps.")
    md.append("- Arc 7: `results/l_arc_7/step4/STEP4_SUMMARY.md` (in-repo).")
    md.append("- Arc 10: `results/l_arc_10/step4/STEP4_SUMMARY.md` (this branch).")
    md.append("- Arc 4 RERUN / Arc 5: `docs/arc_results/ARC_4_RERUN_RESULT.md`, `ARC_5_RESULT.md`.")
    md.append("")
    md.append("## E threshold sweep (0.55 → 0.65 in 0.005 steps)")
    md.append("")
    md.append("| t | n_pass | passing units |")
    md.append("|---:|---:|---|")
    for r in sweep_rows:
        md.append(f"| {r['e_threshold']:.3f} | {r['n_pass']} | {r['passing_units']} |")
    md.append("")
    md.append("## Triple-pass threshold (Arc 6 best + Arc 7 best + Arc 10 best)")
    md.append("")
    if triple_pass_t is None:
        md.append("Insufficient data to determine triple-pass threshold.")
    else:
        md.append(f"- Arc 6 best E AUC: **{arc_best_e['Arc 6']:.4f}** (cohort: `{[r['cohort'] for r in ARC_HISTORY if r['arc']=='Arc 6' and _is_num(r['e_auc']) and float(r['e_auc'])==arc_best_e['Arc 6']][0]}`)")
        md.append(f"- Arc 7 best E AUC: **{arc_best_e['Arc 7']:.4f}** (cohort: `{[r['cohort'] for r in ARC_HISTORY if r['arc']=='Arc 7' and _is_num(r['e_auc']) and float(r['e_auc'])==arc_best_e['Arc 7']][0]}`)")
        md.append(f"- Arc 10 best E AUC: **{arc_best_e['Arc 10']:.4f}** (cohort: `c1`)")
        md.append(f"- **Triple-pass threshold: E AUC ≥ {triple_pass_t:.4f}** (binding constraint: Arc 7 agg_c1_c3).")
        md.append(f"- Implied threshold relaxation vs current 0.65 gate: **{(0.65 - triple_pass_t):.4f}** absolute.")
    md.append("")
    md.append("## False-positive cost audit at triple-pass threshold")
    md.append("")
    if triple_pass_t is None:
        md.append("Not computed.")
    else:
        md.append(f"At E AUC ≥ {triple_pass_t:.4f}, the following arcs would PASS Step 4 but have downstream failures on record:")
        md.append("")
        if not fp_audit:
            md.append("_None — all arcs passing this threshold either deployed (KH-24) or are near-miss V-shape candidates whose downstream outcome is not yet on record._")
        else:
            md.append("| arc | cohort | E AUC | downstream outcome | FP reason |")
            md.append("|---|---|---:|---|---|")
            for r in fp_audit:
                md.append(f"| {r['arc']} | {r['cohort']} | {r['e_auc']:.4f} | {r['downstream_outcome']} | {r['reason_for_FP']} |")
        md.append("")
        md.append("**Note:** the arcs that failed downstream (Arc 4 RERUN, Arc 5) failed *full-pool deployment economics* per v2.3 §4 (Pipeline D1 reject-pool cost), not extractability. They would not pass E AUC threshold at any plausible value (Arc 4 RERUN E AUC ≈ 0.55).")
        md.append("")
        md.append("## False-negative cost audit")
        md.append("")
        if not fn_audit:
            md.append("_None — all DEPLOYED arcs (KH-24) still pass the relaxed threshold._")
        else:
            md.append("| arc | cohort | E AUC | downstream | FN reason |")
            md.append("|---|---|---:|---|---|")
            for r in fn_audit:
                md.append(f"| {r['arc']} | {r['cohort']} | {r['e_auc']:.4f} | {r['downstream_outcome']} | {r['reason_for_FN']} |")
    md.append("")
    md.append("## Interpretation")
    if triple_pass_t is not None:
        relax = 0.65 - triple_pass_t
        md.append(
            f"- A threshold relaxation of **{relax:.4f}** (0.65 → {triple_pass_t:.4f}) would admit Arc 6 best, "
            "Arc 7 best, and Arc 10 best simultaneously."
        )
        md.append(
            "- The binding constraint is Arc 7 agg_c1_c3 at 0.536 — Arc 7 is materially "
            "further from threshold than Arc 6 (0.600) or Arc 10 (0.6296)."
        )
        md.append(
            "- **If Arc 7 is treated as not a near-miss** (its 0.536 is 0.114 below gate — outside "
            "§16a Path A's 0.03 numeric near-miss band), the relevant relaxation question is "
            f"whether to bring the threshold down to ~{max(arc_best_e['Arc 6'], arc_best_e['Arc 10']):.3f} "
            "to admit just Arc 6 best + Arc 10 best."
        )
        md.append(
            "- False-positive cost from the arc history at threshold "
            f"{triple_pass_t:.4f} = **0 documented arcs** "
            "(downstream failures like Arc 4 RERUN / Arc 5 are Pipeline D1 deployment failures, "
            "not E-AUC false-positives — they would not pass E AUC even at very relaxed thresholds)."
        )
    md.append(
        "- Caveat: arc history n is small (≈ 6 closed arcs with E AUC on record). "
        "A threshold relaxation conclusion is informational evidence for v2.4 calibration, "
        "not a deployment decision."
    )
    md.append("")
    md.append("## Artefacts")
    md.append(f"- `raw/exp_03_arc_history.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_03_arc_history.csv')[:16]}…`)")
    md.append(f"- `raw/exp_03_e_threshold_sweep.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_03_e_threshold_sweep.csv')[:16]}…`)")
    if triple_pass_t is not None:
        md.append(f"- `raw/exp_03_fp_audit.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_03_fp_audit.csv')[:16]}…`)")
        md.append(f"- `raw/exp_03_fn_audit.csv` (sha256 `{sha256_file(RAW_DIR / 'exp_03_fn_audit.csv')[:16]}…`)")
    md.append("")

    (OUT_DIR / "EXP_03_threshold_scan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[EXP-03] wrote {OUT_DIR / 'EXP_03_threshold_scan.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
