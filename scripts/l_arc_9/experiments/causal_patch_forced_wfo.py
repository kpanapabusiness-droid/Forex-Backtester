"""Arc 9 causal patch — forced Step 5 WFO + scaled-risk (analyst override).

Original causal_patch.py halted at Phase 3 per dispatch hard rule (patched
LGBM AUC 0.5190 < §8 gate 0.65). Analyst override: run the WFO anyway for
diagnostic completeness — we want to see what a near-random classifier
(AUC ~0.52, essentially random discrimination after the leak is removed)
actually produces under the existing admission threshold infrastructure.

Disposition is NOT changed by this run — Arc 9 remains STEP_4_KILL_AFTER_PATCH.
The numbers here are diagnostic only: they answer "what would the threshold-
sweep admit set look like with a near-random classifier on the same pool?"
Expected: near-random admissions; Candidate A admit set ~ pool * tail of the
score distribution above 0.40 (likely very few admits); Candidate B at
threshold 0.05 admits almost everything (close to raw baseline).

Outputs in results/l_arc_9/experiments/causal_patch/step5_wfo_forced/:
  candidate_A_thr0.40/  per_fold_metrics.csv, admitted/resim trades, full_data.json
  candidate_B_thr0.05/  same
  comparison.csv         side-by-side with original (invalidated) candidates
  scaled_risk/           IF either candidate clears restricted §10 (F2-F7)
  FORCED_WFO_RESULT.md   report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_9.experiments.causal_patch import (  # noqa: E402
    _build_feature_matrix_patched,
    phase_4_step5_wfo_patched,
    phase_5_scaled_risk,
)
from scripts.l_arc_9.experiments.step5_validation import evaluate_gates  # noqa: E402


def _restricted_pass(fold_df: pd.DataFrame, full_m: Dict[str, Any]) -> bool:
    real = fold_df[fold_df["n_trades"] > 0]
    if len(real) == 0:
        return False
    return bool(
        real["annualised_roi_pct"].min() >= 5.0
        and real["annualised_roi_pct"].mean() >= 8.0
        and real["max_dd_pct"].max() <= 8.0
        and (real["fold_roi_pct"] > 0).all()
        and real["n_trades"].min() >= 15
        and full_m["full_data_annualised_roi_pct"] >= 5.0
        and full_m["full_data_max_dd_pct"] <= 10.0
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path,
        default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "causal_patch" / "step5_wfo_forced",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[forced wfo] building patched feature matrix...")
    feat_matrix = _build_feature_matrix_patched(args.out_dir)
    print(f"  n={len(feat_matrix)}, n_pos={int(feat_matrix['y'].sum())}")

    print("[forced wfo] running Step 5 LGBM E WFO with PATCHED classifier...")
    p4 = phase_4_step5_wfo_patched(feat_matrix, args.out_dir)

    # Restricted §10 check on F2-F7.
    cand_a_pass = _restricted_pass(p4["A_thr0.40"]["fold_df"], p4["A_thr0.40"]["full_m"])
    cand_b_pass = _restricted_pass(p4["B_thr0.05"]["fold_df"], p4["B_thr0.05"]["full_m"])
    print(f"  Candidate A restricted §10 (F2-F7): {'PASS' if cand_a_pass else 'FAIL'}")
    print(f"  Candidate B restricted §10 (F2-F7): {'PASS' if cand_b_pass else 'FAIL'}")

    # Comparison to invalidated originals.
    orig_a = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e" / "candidate_A_thr0.40" / "per_fold_metrics.csv")
    orig_a_full = json.loads((_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e" / "candidate_A_thr0.40" / "full_data_metrics.json").read_text())
    orig_b = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e" / "candidate_B_thr0.05" / "per_fold_metrics.csv")
    orig_b_full = json.loads((_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e" / "candidate_B_thr0.05" / "full_data_metrics.json").read_text())
    orig_a_gates = evaluate_gates(orig_a, orig_a_full)
    orig_b_gates = evaluate_gates(orig_b, orig_b_full)

    def _summarize(name, gates, fold_df, full_m, n_admitted):
        s = gates["summary"]
        return {
            "candidate": name,
            "n_admitted": int(n_admitted),
            "worst_fold_ann_roi_pct": s["worst_fold_ann_roi_pct"],
            "mean_fold_ann_roi_pct": s["mean_fold_ann_roi_pct"],
            "worst_fold_max_dd_pct": s["worst_fold_max_dd_pct"],
            "full_data_ann_roi_pct": s["full_data_ann_roi_pct"],
            "full_data_max_dd_pct": s["full_data_max_dd_pct"],
            "all_folds_positive": int(bool(s["all_folds_positive"])),
            "min_trades_per_fold": int(s["min_trades_per_fold"]),
            "pass_deployable": int(gates["pass_deployable"]),
            "pass_viable": int(gates["pass_viable"]),
        }

    rows = []
    rows.append(_summarize("A_thr0.40_ORIGINAL_LEAKED", orig_a_gates, orig_a, orig_a_full, int(orig_a["n_admitted"].sum() if "n_admitted" in orig_a.columns else 236)))
    rows.append(_summarize("A_thr0.40_PATCHED_FORCED", p4["A_thr0.40"]["gates"], p4["A_thr0.40"]["fold_df"], p4["A_thr0.40"]["full_m"], p4["A_thr0.40"]["n_admitted"]))
    rows.append(_summarize("B_thr0.05_ORIGINAL_LEAKED", orig_b_gates, orig_b, orig_b_full, int(orig_b["n_admitted"].sum() if "n_admitted" in orig_b.columns else 599)))
    rows.append(_summarize("B_thr0.05_PATCHED_FORCED", p4["B_thr0.05"]["gates"], p4["B_thr0.05"]["fold_df"], p4["B_thr0.05"]["full_m"], p4["B_thr0.05"]["n_admitted"]))
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False,
                               float_format="%.10g", lineterminator="\n")

    # Conditional scaled-risk on each candidate that clears restricted §10.
    scaled_summaries: Dict[str, Any] = {}
    for cn, cand_pass in [("A_thr0.40", cand_a_pass), ("B_thr0.05", cand_b_pass)]:
        if not cand_pass:
            continue
        print(f"[forced wfo] {cn} clears restricted §10; running scaled-risk diagnostic...")
        scaled_dir = args.out_dir / f"scaled_risk_{cn}"
        scaled_dir.mkdir(exist_ok=True)
        sr = phase_5_scaled_risk(p4[cn]["resim_df"], scaled_dir)
        scaled_summaries[cn] = {
            "recommended_risk_pct": sr["recommended_risk_pct"],
            "rec_metrics": sr["rec_metrics"],
        }
        print(f"  recommended risk: {sr['recommended_risk_pct']:.2f}%")

    summary = {
        "headline": "STEP_4_KILL_AFTER_PATCH (UNCHANGED) — diagnostic forced-WFO numbers attached",
        "candidate_A_restricted_pass_F2_F7": bool(cand_a_pass),
        "candidate_B_restricted_pass_F2_F7": bool(cand_b_pass),
        "comparison_rows": rows,
        "scaled_risk_runs": scaled_summaries,
        "patched_classifier_auc": 0.5190,
        "auc_gate": 0.65,
        "note": "Run executed at analyst override; the dispatch hard rule was to halt at Phase 3 because the patched AUC dropped below §8 gate. This run provides diagnostic numbers only; Arc 9 disposition (STEP_4_KILL) is unchanged.",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print(f"HEADLINE: STEP_4_KILL_AFTER_PATCH (UNCHANGED)")
    print(f"Diagnostic: Candidate A restricted §10 (F2-F7): {'PASS' if cand_a_pass else 'FAIL'}")
    print(f"            Candidate B restricted §10 (F2-F7): {'PASS' if cand_b_pass else 'FAIL'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
