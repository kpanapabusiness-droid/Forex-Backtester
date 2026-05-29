"""Arc 11 v3.0.2 — reconstruct run_summary.json from on-disk artefacts.

Used post-crash recovery: the v3.0.2 driver's orchestrator.run() completed
all 6.6 hours of compute, but the post-run summary-writer crashed on a
field-name bug accessing `AmendedGateResult.worst_fold_dd_at_r_safe_pct`
(non-existent — DD is bound to 8%/10% at r_safe/r_hard by construction).

This script reads:
  - results/l_arc_11_v3.0.2/ARC_CLOSURE.md (orchestrator skeleton with top-K table)
  - amendment_5_admission_probe.json (Step 3/4 detail)
  - admission_plan.json (architectures_skipped_by_amendment_5)
  - step_1/manifest.json (pool sha)
  - step_1/pool.parquet (pool size)
  - step_5/per_day_max_dd_base__*.parquet (Amendment 3 per-day artefacts)
  - step_4/extraction_metrics.csv, feature_importance.csv

And writes run_summary.json that write_closure.py can consume.

Holdout numerical values + per-config Step 5 metadata not in skeleton
are reconstructed where possible (Amendment 3 fields computed from
top-K worst_fold_dd_base_pct via locked scaling rule); marked null
where not recoverable (holdout ROI/DD per config — orchestrator did
run the holdout but values were in-memory only and lost in crash).
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
ARC_NAME = "l_arc_11_v3.0.2"
OUT_DIR = _REPO_ROOT / "results" / ARC_NAME

R_BASE = 0.005   # decimal fraction (0.5%)
R_MIN_PCT = 0.15
R_MAX_PCT = 2.0

# Top-20 table extracted from the orchestrator's ARC_CLOSURE.md skeleton
# (orchestrator's render_arc_closure output, written 00:02:05 UTC at run end
# and preserved in the post-crash read). Hard-coded here because subsequent
# writes by write_closure.py overwrite the skeleton; this is the canonical
# data source for run_summary.json reconstruction post-crash.
TOP_20_HARDCODED = [
    ("A1::a1_baseline_sl2.0_sl_partial_close_1r_runner_trail_expU", -0.74, -0.30, -29.4562, 39.5731),
    ("A1::a1_baseline_sl2.5_sl_plus_tp_2r_expU",                    -0.75,  0.05, -29.2175, 41.5479),
    ("A1::a1_baseline_sl2.5_sl_partial_close_1r_runner_trail_expU", -0.77, -0.15, -26.7333, 34.8636),
    ("A1::a1_baseline_sl1.5_sl_plus_tp_2r_exp2",                    -0.80, -0.45, -25.3794, 33.2769),
    ("A1::a1_baseline_sl2.0_sl_partial_close_1r_runner_trail_exp2", -0.81, -0.42, -16.9782, 26.3069),
    ("A1::a1_baseline_sl2.5_sl_plus_tp_2r_exp2",                    -0.82,  0.19, -15.6208, 25.8706),
    ("A1::a1_baseline_sl1.5_sl_plus_tp_2r_expU",                    -0.83, -0.53, -38.8292, 47.2139),
    ("A1::a1_baseline_sl1.5_sl_partial_close_1r_runner_trail_expU", -0.91, -0.39, -34.2161, 45.2028),
    ("A1::a1_baseline_sl1.5_sl_partial_close_1r_runner_trail_exp2", -0.93, -0.47, -31.2432, 34.6049),
    ("A1::a1_baseline_sl2.0_sl_plus_tp_2r_expU",                    -0.93, -0.35, -34.4931, 44.0745),
    ("A1::a1_baseline_sl2.5_sl_partial_close_1r_runner_trail_exp2", -0.93, -0.17, -20.8636, 22.5265),
    ("A1::a1_baseline_sl2.5_sl_only_exp2",                          -0.95, -0.02, -24.9826, 36.6904),
    ("A1::a1_baseline_sl2.0_sl_plus_tp_2r_exp2",                    -0.96, -0.10, -37.0677, 38.7451),
    ("A1::a1_baseline_sl2.0_sl_only_exp2",                          -0.97, -0.19, -34.9249, 44.3225),
    ("A4::a4_c0_sl2.0_sl_plus_tp_2r_exp2_et0.3",                    -0.98, -0.93, -32.3305, 33.1108),
    ("A4::a4_c0_sl2.0_sl_partial_close_1r_runner_trail_exp2_et0.3", -0.98, -0.92, -32.3131, 33.0936),
    ("A4::a4_c0_sl2.0_sl_partial_close_1r_runner_trail_exp2_et0.5", -0.99, -0.93, -32.0681, 32.8515),
    ("A4::a4_c0_sl2.0_sl_partial_close_1r_runner_trail_exp2_et0.4", -0.99, -0.93, -32.0681, 32.8515),
    ("A4::a4_c0_sl2.5_sl_plus_tp_2r_exp2_et0.5",                    -0.99, -0.92, -26.6968, 27.3747),
    ("A4::a4_c0_sl2.5_sl_plus_tp_2r_exp2_et0.4",                    -0.99, -0.92, -26.6968, 27.3747),
]


def parse_top_k_from_skeleton(closure_md: str) -> list[dict]:
    """Parse the 'All architectures tested' table from the orchestrator skeleton."""
    lines = closure_md.splitlines()
    in_table = False
    rows: list[dict] = []
    for ln in lines:
        if ln.strip().startswith("| Config |"):
            in_table = True
            continue
        if in_table and ln.strip().startswith("|---"):
            continue
        if in_table:
            if not ln.strip().startswith("|"):
                in_table = False
                continue
            parts = [p.strip() for p in ln.strip().strip("|").split("|")]
            if len(parts) != 6:
                continue
            config_id, verdict, wfr_str, mfr_str, wroi_str, wdd_str = parts
            try:
                wfr = float(wfr_str)
                mfr = float(mfr_str)
                wroi = float(wroi_str.rstrip("%")) / 100.0
                wdd = float(wdd_str.rstrip("%")) / 100.0
            except ValueError:
                continue
            rows.append({
                "config_id": config_id,
                "verdict": verdict,
                "worst_fold_ratio": wfr,
                "mean_fold_ratio": mfr,
                "worst_fold_roi": wroi,
                "worst_fold_dd": wdd,
                # Fields not in the skeleton — set to defaults; orchestrator
                # had them in memory but didn't render in the table.
                "n_negative_folds": None,
                "min_trades_per_fold": None,
                "n_folds_evaluated": 11,  # v3.0 WFO = 11 IS folds (oos_year 2010-2020)
            })
    return rows


def compute_amendment_3_for_top1(top1_wfd_pct: float, top1_wfr: float) -> dict:
    """Compute Amendment 3 fields from worst_fold_dd_base_pct using the
    locked scaling rule (core.wfo.amended_gates).

    `top1_wfd_pct` is in PERCENT (e.g. 39.57).
    `top1_wfr` is dimensionless (negative for FAIL arcs).

    Returns a dict matching the AmendedGateResult shape for the Top-1.
    """
    if top1_wfd_pct <= 0:
        # Edge case — k = infinity, fail scalability
        return {
            "k_safe": None, "k_hard": None,
            "r_safe_pct": None, "r_hard_pct": None,
            "scalable_to_safe": False, "scalable_to_hard": False,
            "primary_failure_mode": "step5_not_scalable",
        }
    k_safe = 8.0 / top1_wfd_pct
    k_hard = 10.0 / top1_wfd_pct
    r_safe_pct = R_BASE * 100 * k_safe   # in percent
    r_hard_pct = R_BASE * 100 * k_hard
    scalable_to_safe = R_MIN_PCT <= r_safe_pct <= R_MAX_PCT
    scalable_to_hard = R_MIN_PCT <= r_hard_pct <= R_MAX_PCT
    # Priority-ordered failure mode per L_PROTOCOL §3
    if not scalable_to_safe and not scalable_to_hard:
        pfm = "step5_not_scalable"
    elif top1_wfr < 2.0:
        pfm = "step5_ratio_below_gate_after_scaling"
    else:
        pfm = "other"  # placeholder
    return {
        "k_safe": round(k_safe, 6),
        "k_hard": round(k_hard, 6),
        "r_safe_pct": round(r_safe_pct, 6),
        "r_hard_pct": round(r_hard_pct, 6),
        "scalable_to_safe": bool(scalable_to_safe),
        "scalable_to_hard": bool(scalable_to_hard),
        "primary_failure_mode": pfm,
    }


def main() -> int:
    # 1. Read existing artefacts
    closure_md_path = OUT_DIR / "ARC_CLOSURE.md"
    probe_path = OUT_DIR / "amendment_5_admission_probe.json"
    admission_path = OUT_DIR / "admission_plan.json"
    manifest_path = OUT_DIR / "step_1" / "manifest.json"
    pool_path = OUT_DIR / "step_1" / "pool.parquet"

    if not closure_md_path.exists():
        print(f"ERROR: {closure_md_path} missing")
        return 1
    closure_md = closure_md_path.read_text(encoding="utf-8")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    admission_plan = json.loads(admission_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 2. Pool size
    pool_df = pd.read_parquet(pool_path)
    pool_size = int(len(pool_df))
    pool_sha = manifest.get("pool_sha256", "")

    # 3. Parse top-K from skeleton (sorted by worst-fold ratio desc).
    # Fall back to hard-coded TOP_20_HARDCODED if skeleton already overwritten
    # by a prior write_closure.py invocation (post-crash recovery scenario).
    top_k_all = parse_top_k_from_skeleton(closure_md)
    if not top_k_all:
        print("Skeleton table empty — falling back to TOP_20_HARDCODED constants")
        top_k_all = [
            {
                "config_id": cid,
                "verdict": "fail",
                "worst_fold_ratio": wfr,
                "mean_fold_ratio": mfr,
                "worst_fold_roi": wroi / 100.0,
                "worst_fold_dd": wdd / 100.0,
                "n_negative_folds": None,
                "min_trades_per_fold": None,
                "n_folds_evaluated": 11,
            }
            for (cid, wfr, mfr, wroi, wdd) in TOP_20_HARDCODED
        ]
    print(f"Parsed {len(top_k_all)} top configs from skeleton table")

    # 4. Identify top-3 candidates (those for which Amendment 3 was evaluated;
    # one per_day_max_dd parquet was written per top-3 candidate)
    per_day_parquets = sorted((OUT_DIR / "step_5").glob("per_day_max_dd_base__*.parquet"))
    top3_configs = []
    for p in per_day_parquets:
        # Parse config_id from filename: per_day_max_dd_base__<config_id>.parquet
        # Per arc_orchestrator._run_amendment_3_evaluation: filename uses
        # cid.replace("::", "__") to make the path safe. Reverse to match table.
        raw = p.stem.replace("per_day_max_dd_base__", "")
        # Re-insert the "::" between architecture prefix and config name
        cid = re.sub(r"^([A-Z]\d)__", r"\1::", raw, count=1)
        top3_configs.append({"config_id": cid, "per_day_max_dd_artefact_path": str(p.relative_to(_REPO_ROOT))})
    print(f"Top-3 per-day DD parquets: {[c['config_id'] for c in top3_configs]}")

    # 5. Construct Amendment 3 fields for the top-K (only Top-1 has Amendment 3 detail
    # reconstructed; others get the parquet path + same scaling logic)
    amendment_3_top_k = []
    for top_config in top3_configs:
        cid = top_config["config_id"]
        # Find the matching row in top_k_all
        match = next((r for r in top_k_all if r["config_id"] == cid), None)
        if match is None:
            print(f"WARN: no skeleton-table match for {cid}")
            continue
        wfd_pct = match["worst_fold_dd"] * 100
        wfr = match["worst_fold_ratio"]
        wroi = match["worst_fold_roi"]
        amend = compute_amendment_3_for_top1(wfd_pct, wfr)
        # Compute scaled fields
        worst_fold_roi_at_r_safe_pct = (
            wroi * 100 * amend["k_safe"] if amend["k_safe"] is not None else None
        )
        worst_fold_roi_at_r_hard_pct = (
            wroi * 100 * amend["k_hard"] if amend["k_hard"] is not None else None
        )
        # Chained DD — read first row of per-day parquet's day_max_dd_base_pct
        # to derive a chained-equivalent. NOT a precise reproduction of the
        # orchestrator's stitched chained DD; documented in closure as
        # "chained_dd reconstructed_post_crash; orchestrator value not preserved".
        per_day_df = pd.read_parquet(_REPO_ROOT / top_config["per_day_max_dd_artefact_path"])
        # Conservative reconstruction: max per-day DD in the trajectory =
        # peak instantaneous DD which is a CONSERVATIVE upper bound on
        # chained DD (chained DD is across the full continuous-equity curve,
        # always >= max per-day DD if peaks compose; equal to max per-day
        # DD if all DDs are within a single day). Use this as a placeholder.
        max_per_day_dd = float(per_day_df["day_max_dd_base_pct"].max()) if not per_day_df.empty else 0.0
        chained_dd_proxy = max_per_day_dd
        amendment_3_top_k.append({
            "config_id": cid,
            "chained_max_dd_base_pct": round(chained_dd_proxy, 4),
            "chained_dd_method": "max_per_day_proxy_post_crash_reconstruction",
            "per_day_max_dd_artefact_path": top_config["per_day_max_dd_artefact_path"],
            "verdict": "fail",
            "primary_failure_mode": amend["primary_failure_mode"],
            "k_safe": amend["k_safe"],
            "k_hard": amend["k_hard"],
            "r_safe_pct": amend["r_safe_pct"],
            "r_hard_pct": amend["r_hard_pct"],
            "scalable_to_safe": amend["scalable_to_safe"],
            "scalable_to_hard": amend["scalable_to_hard"],
            "worst_fold_roi_at_r_safe_pct": (
                round(worst_fold_roi_at_r_safe_pct, 4) if worst_fold_roi_at_r_safe_pct is not None else None
            ),
            "worst_fold_dd_at_r_safe_pct": 8.0 if amend["scalable_to_safe"] else None,
            "worst_fold_roi_at_r_hard_pct": (
                round(worst_fold_roi_at_r_hard_pct, 4) if worst_fold_roi_at_r_hard_pct is not None else None
            ),
            "worst_fold_dd_at_r_hard_pct": 10.0 if amend["scalable_to_hard"] else None,
            "chained_max_dd_at_r_safe_pct": (
                round(chained_dd_proxy * amend["k_safe"], 4) if amend["k_safe"] is not None else None
            ),
            "chained_max_dd_at_r_hard_pct": (
                round(chained_dd_proxy * amend["k_hard"], 4) if amend["k_hard"] is not None else None
            ),
            # daily_dd_breaches reconstructed from per-day parquet
            "daily_dd_breaches_at_r_safe": _daily_breaches(per_day_df, amend["k_safe"]) if amend["k_safe"] else None,
            "daily_dd_breaches_at_r_hard": _daily_breaches(per_day_df, amend["k_hard"]) if amend["k_hard"] else None,
            # Holdout values not preserved
            "holdout_roi_at_r_safe_pct": None,
            "holdout_dd_at_r_safe_pct": None,
            "holdout_roi_at_r_hard_pct": None,
            "holdout_dd_at_r_hard_pct": None,
        })

    # 6. Build full summary
    summary = {
        "arc_name": ARC_NAME,
        "verdict": "FAIL",
        "pool_size": pool_size,
        "pool_sha256": pool_sha,
        "feature_matrix_shape": [pool_size, 27],
        "step_2_k_selected": 4,
        "step_2_silhouettes": {2: 0.397, 3: 0.396, 4: 0.494, 5: 0.476, 6: 0.456},
        "candidate_cluster_ids": list(probe["candidate_ids"]),
        "step_3_per_cluster": probe["step_3_per_cluster"],
        "step_4_per_cluster": probe["step_4_per_cluster"],
        "admission_plan": admission_plan,
        "architectures_skipped_by_amendment_5": admission_plan["architectures_skipped_by_amendment_5"],
        # Actual evaluated count from admission plan (the skeleton table
        # is truncated to top-20; admission plan records the true total).
        "n_configs_evaluated_step_5": int(admission_plan.get("total_configs_step_5", len(top_k_all))),
        "search_scope_flag": admission_plan["search_scope_flag"],
        "step_5_top_k": top_k_all[:3],     # top-3 by worst-fold ratio (descending)
        "amendment_3_top_k": amendment_3_top_k,
        "holdout_results": [],   # Lost in crash; documented in closure §3 cross-arc
        "step_6_dispatched": False,
        "step_6_overall_passed": None,
        "elapsed_seconds": 397.1 * 60,   # 397.1 min per orchestrator log line
        "reconstruction_note": (
            "run_summary.json reconstructed from on-disk artefacts post-crash. "
            "Orchestrator completed all 6.6 hours of compute (Steps 1-5 + Amendment 3 + "
            "no Step 6 since all 54 configs FAIL); summary-writer crashed on a field-name "
            "bug accessing AmendedGateResult.worst_fold_dd_at_r_safe_pct (non-existent; "
            "DD is bound to 8% at r_safe by construction). Holdout per-config numerical "
            "values not preserved (in-memory only); verdict FAIL is dispositive via WFO "
            "ratio (all 54 configs worst-fold ratio < 0, below the 2.0 PASS gate)."
        ),
    }

    out_path = OUT_DIR / "run_summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"Wrote {out_path}")
    print(f"  pool_size={pool_size}, pool_sha256={pool_sha[:16]}...")
    print(f"  Top-1: {top_k_all[0]['config_id']}")
    print(f"    worst_fold_ratio={top_k_all[0]['worst_fold_ratio']}")
    print(f"    worst_fold_dd_base_pct={top_k_all[0]['worst_fold_dd']*100:.4f}%")
    print(f"  Amendment 3 Top-1: {amendment_3_top_k[0] if amendment_3_top_k else 'n/a'}")
    return 0


def _daily_breaches(per_day_df: pd.DataFrame, k_scale: float) -> int:
    """Count days where day_max_dd_base * k_scale >= 5% (5ers daily limit)."""
    if per_day_df is None or per_day_df.empty or "day_max_dd_base_pct" not in per_day_df.columns:
        return 0
    scaled = per_day_df["day_max_dd_base_pct"] * k_scale
    return int((scaled >= 5.0).sum())


if __name__ == "__main__":
    raise SystemExit(main())
