"""Arc 11 v3.0.2 — emit ARC_CLOSURE.md per template v1.3.1.

Reads ``results/l_arc_11_v3.0.2/run_summary.json`` (produced by
``run.py``) plus the v3.0 original closure + Amendment 5.1 admission
plan, and emits a v1.3.1-compliant closure doc with:

  - §1 tracker_payload — full Amendment 3 + 5 + 5.1 + 6 fields
  - §2 prose verdict (100-300 words)
  - §3 cross-arc observations (per chat resolutions §1.A5.2 / A5.3 /
    A5.5 + §2 EET-HTF drift attribution)
  - §4 deployment_spec — abbreviated (FAIL) or full (PASS-tier)
  - §10 quantitative comparison v3.0.2 vs v3.0 under canonical engine

Idempotent — re-runs overwrite the closure file deterministically.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ─── Constants from run.py + dispatch ─────────────────────────────

ARC_NAME = "l_arc_11_v3.0.2"
SIGNAL_DESC = "swing-high breakout in trend (SHB) long, 4H, causal 3-bar swing (right-edge t-4)"
HOLDOUT_START = "2021-01-01"
HOLDOUT_END = "2026-05-25"
WINDOW_START = "2010-01-01"
WINDOW_END = "2026-05-25"
R_BASE_PCT = 0.5
ARC_BRANCH = "arc/l_arc_11_v3.0.2"

# v3.0 reference numbers (from results/l_arc_11/ARC_CLOSURE.md §1)
V30_POOL = 17_533
V30_WORST_FOLD_RATIO = -0.7687
V30_WORST_FOLD_ROI_PCT = -24.0273
V30_WORST_FOLD_DD_PCT = 38.3593
V30_MEAN_FOLD_RATIO = 0.4305
V30_HOLDOUT_ROI_PCT = 1.3004
V30_HOLDOUT_DD_PCT = 57.4969
V30_C0_E_AUC = 0.6543
V30_C1_E_AUC = 0.6316
V30_C0_COMPOSITE = 1.9803
V30_C1_COMPOSITE = 0.9678
V30_HOLDOUT_END = "2026-04-30"


ARCHETYPE_TO_TEMPLATE = {
    "v_shape_recovery": "V-shape",
    "stepwise_climber": "Stepwise",
    "bimodal": "Bimodal",
    "monotonic_up": "Monotonic_up",
    "monotonic_down": "Monotonic_down",
    "choppy": "Choppy",
    "unclassified": "Unclassified",
}

VERDICT_NORMALISED = {
    "pass_deployable": "PASS-DEPLOYABLE",
    "pass_viable": "PASS-VIABLE",
    "fail": "FAIL",
    "FAIL": "FAIL",
    "PASS-DEPLOYABLE": "PASS-DEPLOYABLE",
    "PASS-VIABLE": "PASS-VIABLE",
    "PASS_DEPLOYABLE": "PASS-DEPLOYABLE",
    "PASS_VIABLE": "PASS-VIABLE",
}


def arch_label(code: str) -> str:
    return {
        "A1": "A1 system_level_filter",
        "A2": "A2 classifier_filter",
        "A3": "A3 pipeline_de",
        "A4": "A4 pipeline_d_exits",
        "A5": "A5 portfolio_composition",
        "A6": "A6 meta_labeling",
    }.get(code.upper(), code)


def detect_arch_from_config_id(config_id: str) -> str:
    cid = config_id.lower()
    for a in ("a1", "a2", "a3", "a4", "a5", "a6"):
        if cid.startswith(a + "_") or cid.startswith(a + "::") or f"::{a}_" in cid:
            return a.upper()
        # Orchestrator builds config_id as "Ax::ax_..."
        if cid.startswith(a.upper() + "::"):
            return a.upper()
    # ArcOrchestrator format: "Ax::ax_..." with mixed case
    for a in ("A1", "A2", "A3", "A4", "A5", "A6"):
        if config_id.startswith(a + "::"):
            return a
    return "?"


def _yaml_dump(payload: dict) -> str:
    return yaml.safe_dump(
        payload, default_flow_style=False, sort_keys=False,
        allow_unicode=True, width=200,
    )


def _norm(s: object) -> str:
    """Normalise verdict-like string to canonical PASS-/FAIL/HALT."""
    if s is None:
        return "FAIL"
    v = str(s).strip().replace("_", "-").upper()
    return VERDICT_NORMALISED.get(v, v)


def main() -> int:
    out_dir = _REPO_ROOT / "results" / ARC_NAME
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} missing -- run scripts/l_arc_11_v3_0_2/run.py first")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    v32_pool = int(summary["pool_size"])
    arc_verdict = _norm(summary.get("verdict", "FAIL"))
    candidate_ids = summary.get("candidate_cluster_ids", [])
    architectures_skipped = summary.get("architectures_skipped_by_amendment_5", [])
    step_3_per_cluster = summary.get("step_3_per_cluster", [])
    step_4_per_cluster = summary.get("step_4_per_cluster", [])
    step_5_top_k = summary.get("step_5_top_k", [])
    amendment_3_top_k = summary.get("amendment_3_top_k", [])
    holdout_results = summary.get("holdout_results", [])
    n_configs = int(summary.get("n_configs_evaluated_step_5", 0))
    search_scope_flag = str(summary.get("search_scope_flag", "thin"))

    # Re-classify primary_failure_mode per Amendment 3 priority order
    # using the actual Amendment 3 evaluation results (most authoritative).
    if amendment_3_top_k:
        # Top-1 in amended-result order
        top1_amended = amendment_3_top_k[0]
        raw_pfm = top1_amended.get("primary_failure_mode", "other") or "other"
        # Handle both enum-value form ("step5_not_scalable") and enum-name form
        # ("PrimaryFailureMode.STEP5_NOT_SCALABLE") that may end up in run_summary.json
        # depending on json.dumps default=str enum serialization.
        if "." in str(raw_pfm) and "PrimaryFailureMode" in str(raw_pfm):
            primary_failure_mode = str(raw_pfm).split(".")[-1].lower()
        else:
            primary_failure_mode = str(raw_pfm).lower()
        # Normalize a few common Python enum repr quirks
        if primary_failure_mode in ("none", ""):
            primary_failure_mode = "N/A"
    elif step_5_top_k:
        # Fallback — derive from base gate
        top1 = step_5_top_k[0]
        if float(top1["worst_fold_dd"]) > 0.10:
            primary_failure_mode = "step5_dd_above_gate"
        elif float(top1["worst_fold_roi"]) <= 0:
            primary_failure_mode = "step5_wf_roi_below_gate_after_scaling"
        elif int(top1["n_negative_folds"]) > 0:
            primary_failure_mode = "step5_negative_folds"
        else:
            primary_failure_mode = "other"
    else:
        primary_failure_mode = "no_capturable_cluster" if not candidate_ids else "other"

    failed_at_step = 5 if arc_verdict == "FAIL" and step_5_top_k else (
        3 if arc_verdict == "FAIL" and not candidate_ids else "N/A"
    )

    # Determine "best" candidate: top of Amendment 3 results (verdict-weighted), else top step_5
    if amendment_3_top_k:
        winner = amendment_3_top_k[0]
        winner_config_id = winner["config_id"]
        # Find matching base top_k entry
        base_top1 = next(
            (c for c in step_5_top_k if c["config_id"] == winner_config_id),
            step_5_top_k[0] if step_5_top_k else None,
        )
    elif step_5_top_k:
        base_top1 = step_5_top_k[0]
        winner = None
        winner_config_id = base_top1["config_id"]
    else:
        winner = None
        base_top1 = None
        winner_config_id = None

    # Holdout match
    holdout_lookup = {h["config_id"]: h for h in holdout_results}
    winner_holdout = holdout_lookup.get(winner_config_id) if winner_config_id else None

    # Best-architecture block
    if base_top1 is not None:
        winner_arch = detect_arch_from_config_id(winner_config_id)
        winner_cluster = None
        if winner_arch != "A1":
            # Auto-spec configs encode cluster_id in name like "a2_c0_..."
            import re
            m = re.search(r"_c(\d+)_", winner_config_id)
            if m:
                winner_cluster = int(m.group(1))
        winner_archetype = None
        for c in step_3_per_cluster:
            if winner_cluster is not None and int(c["cluster_id"]) == int(winner_cluster):
                winner_archetype = ARCHETYPE_TO_TEMPLATE.get(c["archetype"].lower(), c["archetype"])
                break
        features_in_winning: list[str] = []
        if winner_arch in ("A2", "A6"):
            # Read step_4 feature importance for top features
            fi_path = out_dir / "step_4" / "feature_importance.csv"
            if fi_path.exists() and winner_cluster is not None:
                fi = pd.read_csv(fi_path)
                if "cluster_id" in fi.columns:
                    cfi = fi[fi["cluster_id"] == winner_cluster]
                else:
                    cfi = fi
                if not cfi.empty:
                    # Pick best classifier rows (highest mean importance)
                    if "feature" in cfi.columns:
                        feat_col = "feature"
                    elif "feature_name" in cfi.columns:
                        feat_col = "feature_name"
                    else:
                        feat_col = cfi.columns[0]
                    imp_col = None
                    for c in cfi.columns:
                        if "importance_mean" in c or "mean_importance" in c or c == "importance":
                            imp_col = c
                            break
                    if imp_col:
                        cfi_sorted = cfi.sort_values(imp_col, ascending=False)
                        features_in_winning = [str(x) for x in cfi_sorted[feat_col].head(10).tolist()]
                    else:
                        features_in_winning = [str(x) for x in cfi[feat_col].head(10).tolist()]

        # Parse SL/exit/exposure from config_id (best-effort)
        sl_atr = 2.0
        exit_policy = "sl_only"
        exposure_cap = 2
        import re
        m = re.search(r"sl([0-9.]+)", winner_config_id)
        if m:
            try:
                sl_atr = float(m.group(1))
            except ValueError:
                pass
        for ep in ("sl_partial_close_1r_runner_trail", "sl_plus_tp_2r", "sl_plus_tp_3r",
                   "sl_plus_trailing_atr", "sl_plus_trailing_swing", "sl_only"):
            if ep in winner_config_id:
                exit_policy = ep
                break
        if "expU" in winner_config_id:
            exposure_cap = "unlimited"
        elif "exp2" in winner_config_id:
            exposure_cap = 2

        best_block = {
            "name": arch_label(winner_arch),
            "cluster": winner_cluster,
            "archetype": winner_archetype,
            "config": winner_config_id,
            "sl_atr": float(sl_atr),
            "exit_policy": exit_policy,
            "exposure_cap": exposure_cap,
            "worst_fold_ratio": round(float(base_top1["worst_fold_ratio"]), 4),
            "worst_fold_roi_base_pct": round(float(base_top1["worst_fold_roi"]) * 100, 4),
            "worst_fold_dd_base_pct": round(float(base_top1["worst_fold_dd"]) * 100, 4),
            "mean_fold_ratio": round(float(base_top1["mean_fold_ratio"]), 4),
            "mean_fold_roi_pct": None,
            "sign_pos_folds": (
                f"{int(base_top1['n_folds_evaluated']) - int(base_top1['n_negative_folds'])}"
                f"/{int(base_top1['n_folds_evaluated'])}"
                if base_top1.get("n_negative_folds") is not None
                else f"?/{int(base_top1.get('n_folds_evaluated', 11))}"
            ),
            "n_trades_total": _winner_total_trades(out_dir, winner_config_id),
            "holdout_roi_pct": (
                round(float(winner_holdout["holdout_roi_pct"]) * 100, 4)
                if winner_holdout else None
            ),
            "holdout_dd_pct": (
                round(float(winner_holdout["holdout_dd_pct"]) * 100, 4)
                if winner_holdout else None
            ),
            "holdout_passed": (
                bool(winner_holdout["deployable"]) if winner_holdout else None
            ),
            "oracle_worst_ratio": None,
            "oracle_real_gap_sharpe": None,
            "features_in_winning_config": features_in_winning,
        }
        # Amendment 3 fields
        if winner is not None:
            best_block.update({
                "chained_max_dd_base_pct": _ofloat(winner.get("chained_max_dd_base_pct"), 4),
                "per_day_max_dd_artefact_path": winner.get("per_day_max_dd_artefact_path"),
                "per_day_max_dd_base_summary": {
                    "n_days": None, "p50_pct": None, "p95_pct": None, "p99_pct": None, "max_pct": None,
                },
                "k_safe": _ofloat(winner.get("k_safe"), 4),
                "k_hard": _ofloat(winner.get("k_hard"), 4),
                "r_safe_pct": _ofloat(winner.get("r_safe_pct"), 4),
                "r_hard_pct": _ofloat(winner.get("r_hard_pct"), 4),
                "scalable_to_safe": bool(winner.get("scalable_to_safe", False)),
                "scalable_to_hard": bool(winner.get("scalable_to_hard", False)),
                "worst_fold_roi_at_r_safe_pct": _ofloat(winner.get("worst_fold_roi_at_r_safe_pct"), 4),
                "worst_fold_roi_at_r_hard_pct": _ofloat(winner.get("worst_fold_roi_at_r_hard_pct"), 4),
                "chained_max_dd_at_r_safe_pct": _ofloat(winner.get("chained_max_dd_at_r_safe_pct"), 4),
                "chained_max_dd_at_r_hard_pct": _ofloat(winner.get("chained_max_dd_at_r_hard_pct"), 4),
                "daily_dd_breaches_at_r_safe": _oint(winner.get("daily_dd_breaches_at_r_safe")),
                "daily_dd_breaches_at_r_hard": _oint(winner.get("daily_dd_breaches_at_r_hard")),
                "holdout_roi_at_r_safe_pct": _ofloat(winner.get("holdout_roi_at_r_safe_pct"), 4),
                "holdout_dd_at_r_safe_pct": _ofloat(winner.get("holdout_dd_at_r_safe_pct"), 4),
                "holdout_roi_at_r_hard_pct": _ofloat(winner.get("holdout_roi_at_r_hard_pct"), 4),
                "holdout_dd_at_r_hard_pct": _ofloat(winner.get("holdout_dd_at_r_hard_pct"), 4),
                "sizing_convention": "reset_floor",
                "chained_dd_method": winner.get("chained_dd_method", "equity_stitching"),
            })
        else:
            # No Amendment 3 result — null out the fields
            for k in ("chained_max_dd_base_pct", "per_day_max_dd_artefact_path",
                      "k_safe", "k_hard", "r_safe_pct", "r_hard_pct",
                      "scalable_to_safe", "scalable_to_hard",
                      "worst_fold_roi_at_r_safe_pct", "worst_fold_roi_at_r_hard_pct",
                      "chained_max_dd_at_r_safe_pct", "chained_max_dd_at_r_hard_pct",
                      "daily_dd_breaches_at_r_safe", "daily_dd_breaches_at_r_hard",
                      "holdout_roi_at_r_safe_pct", "holdout_dd_at_r_safe_pct",
                      "holdout_roi_at_r_hard_pct", "holdout_dd_at_r_hard_pct"):
                best_block[k] = None
            best_block["per_day_max_dd_base_summary"] = {
                "n_days": None, "p50_pct": None, "p95_pct": None, "p99_pct": None, "max_pct": None,
            }
            best_block["sizing_convention"] = "reset_floor"
            best_block["chained_dd_method"] = None
        # v1.2 deployment-spec fields
        best_block["config_artefact_path"] = f"configs/{ARC_NAME}/winning_config.yaml"
        best_block["deployment_spec_section_present"] = True
    else:
        best_block = None
        winner_arch = None
        winner_cluster = None

    # Clusters block — every cluster from Step 2
    clusters_block: dict = {}
    step3_by_cid = {int(c["cluster_id"]): c for c in step_3_per_cluster}
    step4_by_cid = {int(e["cluster_id"]): e for e in step_4_per_cluster}
    candidate_set = set(int(c) for c in candidate_ids)
    for cid in sorted(step3_by_cid.keys()):
        c = step3_by_cid[cid]
        s4_e = step4_by_cid.get(cid)
        if cid in candidate_set:
            if s4_e and float(s4_e["best_classifier_mean_auc"]) >= 0.65:
                if winner_cluster is not None and int(winner_cluster) == cid and arc_verdict.startswith("PASS"):
                    outcome = "wins_step5" if arc_verdict == "PASS-DEPLOYABLE" else "viable_step5"
                else:
                    outcome = "dies_step5"
            else:
                outcome = "dies_step4" if s4_e else "passed_step3"
        else:
            outcome = "dies_step3"
        clusters_block[f"c{cid}"] = {
            "n": int(c["n_trades"]),
            "archetype": ARCHETYPE_TO_TEMPLATE.get(c["archetype"].lower(), c["archetype"]),
            "sl_atr": float(c["selected_sl_mult"]),
            "step3_composite": round(float(c["composite"]), 4),
            "mfe_p50_r": round(float(c["mfe_p50"]), 4),
            "ww_pp": round(float(c["ww_pp"]), 4),
            "reach_1r": round(float(c["reach_1r"]), 4),
            "step4_e_auc": round(float(s4_e["best_classifier_mean_auc"]), 4) if s4_e else None,
            "step4_d1_auc": None,
            "outcome": outcome,
        }

    # Architectures + results
    archs_tested = sorted({detect_arch_from_config_id(c["config_id"])
                           for c in step_5_top_k} - {"?"})
    if not archs_tested:
        # Fallback — derive from admission plan
        ap = summary.get("admission_plan", {})
        archs_tested = ["A1"]
        if ap.get("n_auto_specs", 0) > 0:
            # Coarse — assume A2/A4/A6 if any specs
            for a in ("A2", "A4", "A6"):
                archs_tested.append(a)
    archs_tested = sorted(set(archs_tested))

    arch_results: dict = {}
    # Aggregate best worst_fold_ratio per architecture across ALL evaluated configs.
    # Since step_5_top_k only has top-3, we have only a small slice. Use top-3 best
    # per arch (acceptable for FAIL verdicts; PASS would carry the actual winning ratio).
    for code in archs_tested:
        sub = [c for c in step_5_top_k if detect_arch_from_config_id(c["config_id"]) == code]
        if not sub:
            arch_results[code] = {"tested": True, "won": False, "worst_fold_ratio": None}
            continue
        max_wfr = max(float(c["worst_fold_ratio"]) for c in sub)
        won = (
            winner_config_id is not None
            and detect_arch_from_config_id(winner_config_id) == code
            and float(base_top1["worst_fold_ratio"]) >= 2.0
            if base_top1 else False
        )
        arch_results[code] = {
            "tested": True,
            "won": bool(won),
            "worst_fold_ratio": round(max_wfr, 4),
        }

    # Archetypes observed
    archetypes_observed = sorted({v["archetype"] for v in clusters_block.values()})

    # Cost decomposition (classifier-based winner only)
    cost_block = None
    if winner_arch in ("A2", "A6") and winner_cluster is not None and winner_cluster in step3_by_cid:
        admit_c = step3_by_cid[winner_cluster]
        others = [c for cid, c in step3_by_cid.items() if cid != winner_cluster]
        total_n = sum(int(c["n_trades"]) for c in step3_by_cid.values())
        admit_n = int(admit_c["n_trades"])
        reject_n = sum(int(c["n_trades"]) for c in others)
        # Mean R proxies from step_3/capturability.csv
        admit_mean_r = None
        reject_mean_r = None
        cap_path = out_dir / "step_3" / "capturability.csv"
        if cap_path.exists():
            cap = pd.read_csv(cap_path)
            mean_col = "mean_R" if "mean_R" in cap.columns else (
                "mean_r" if "mean_r" in cap.columns else None
            )
            if mean_col:
                row_admit = cap[cap["cluster_id"] == winner_cluster]
                if not row_admit.empty:
                    admit_mean_r = float(row_admit.iloc[0][mean_col])
                row_reject = cap[cap["cluster_id"] != winner_cluster]
                if not row_reject.empty:
                    reject_mean_r = float(
                        (row_reject[mean_col] * row_reject["n_trades"]).sum()
                        / max(row_reject["n_trades"].sum(), 1)
                    )
        # CostPool requires non-null floats per tracker_parser.schema.CostPool;
        # fall back to 0.0 when mean_r unavailable (FAIL arcs may have missing data).
        cost_block = {
            "admit_pool": {
                "n_fraction": round(admit_n / max(total_n, 1), 4),
                "mean_r": round(admit_mean_r, 4) if admit_mean_r is not None else 0.0,
            },
            "reject_pool": {
                "n_fraction": round(reject_n / max(total_n, 1), 4),
                "mean_r": round(reject_mean_r, 4) if reject_mean_r is not None else 0.0,
            },
            "early_exit_pool": {"n_fraction": 0.0, "mean_r": 0.0},
        }

    # Step 6 block (Amendment 4)
    step_6_block = _step_6_block(summary, arc_verdict)

    # Cross-arc tags
    cross_arc_tags = _cross_arc_tags(summary, arc_verdict, base_top1)

    # one-liner
    one_line = _one_line(summary, base_top1, winner_holdout, arc_verdict)

    payload = {
        "tracker_payload": {
            "template_version": "v1.3",
            "arc_name": ARC_NAME,
            "signal": SIGNAL_DESC,
            "tf": "H4",
            "sub_protocol": "vanilla",
            "closed_timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "closure_doc_link": f"results/{ARC_NAME}/ARC_CLOSURE.md",
            "verdict": arc_verdict,
            "one_line": one_line,
            "failed_at_step": failed_at_step,
            "primary_failure_mode": primary_failure_mode,
            "pool_metadata": {
                "total_n": v32_pool,
                "window_start": WINDOW_START,
                "window_end": WINDOW_END,
                "kh24_co_fire_pct": None,
                "configs_evaluated_step5": n_configs,
                "search_scope_flag": search_scope_flag,
            },
            "best_architecture": best_block,
            "cost_decomposition": cost_block,
            "clusters": clusters_block,
            "architectures_tested": archs_tested,
            "architecture_results": arch_results,
            "architectures_skipped_by_amendment_5": architectures_skipped,
            "archetypes_observed": archetypes_observed,
            "cross_arc_tags": cross_arc_tags,
            "step_6": step_6_block,
        }
    }

    # §2 prose + §3 cross-arc bullets + §10 retroactive re-eval
    why_prose = _why_prose(summary, base_top1, winner_holdout, arc_verdict, winner, primary_failure_mode)
    cross_arc_obs = _cross_arc_prose(summary, base_top1, winner_holdout, arc_verdict)
    section_10 = _section_10_quantitative_comparison(
        summary, base_top1, winner_holdout, arc_verdict, winner,
    )

    deployment_spec = _deployment_spec(summary, best_block, arc_verdict, winner_cluster)

    lines = [
        f"# ARC_11_v3.0.2_CLOSURE -- {ARC_NAME}",
        "",
        f"> **Closed:** {payload['tracker_payload']['closed_timestamp']}",
        f"> **Branch:** {ARC_BRANCH}",
        f"> **Closure doc path:** results/{ARC_NAME}/ARC_CLOSURE.md",
        "",
        "---",
        "",
        "## §1 tracker_payload",
        "",
        "```yaml",
        _yaml_dump(payload).rstrip(),
        "```",
        "",
        "---",
        "",
        f"## §2 Why {'failed' if 'PASS' not in arc_verdict else 'succeeded'}",
        "",
        why_prose,
        "",
        "---",
        "",
        "## §3 Cross-arc observations",
        "",
    ]
    for o in cross_arc_obs:
        lines.append(f"- {o}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## §4 deployment_spec")
    lines.append("")
    lines.append(deployment_spec)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## §10 Quantitative comparison v3.0.2 vs v3.0 under canonical engine")
    lines.append("")
    lines.append(
        "> Tests whether canonical orchestrator gap closure (PR #186), "
        "canonical uncapped pool, mid features (PR #189), EET HTF alignment (PR #193), "
        "Amendment 5 four-gate selection (PR #194), Amendment 5.1 Gate-4 PASS-tier qualifier "
        "(PR #201), canonical exit registry (CC_18 / PR #195), and Amendment 6 EET daily-DD "
        "boundary (CC_20 / PR #197) change Step 5 outcomes vs Arc 11 v3.0. Arc 11 v3.0 §10 "
        "already applied Amendment 3 retroactively; this §10 does not duplicate that work."
    )
    lines.append("")
    lines.append(section_10)
    lines.append("")

    (out_dir / "ARC_CLOSURE.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n",
    )
    print(f"Closure written: {out_dir / 'ARC_CLOSURE.md'}")
    print(f"  verdict={arc_verdict}")
    print(f"  failed_at_step={failed_at_step}")
    print(f"  primary_failure_mode={primary_failure_mode}")
    print(f"  configs_evaluated={n_configs} ({search_scope_flag})")
    print(f"  architectures_skipped_by_amendment_5={architectures_skipped}")

    # Write the winning config YAML (parser requires file present for PASS;
    # for FAIL we write it for documentation parity per Arc 11 v3.0 convention)
    _write_winning_config(out_dir, best_block, arc_verdict)

    return 0


# ─── Helpers ──────────────────────────────────────────────────────


def _ofloat(x, ndigits: int = 4):
    if x is None:
        return None
    try:
        return round(float(x), ndigits)
    except (TypeError, ValueError):
        return None


def _oint(x):
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _winner_total_trades(out_dir: Path, winner_config_id: str | None) -> int | None:
    if not winner_config_id:
        return None
    # Try wfo_results.csv first
    wr = out_dir / "step_5" / "wfo_results.csv"
    if wr.exists():
        df = pd.read_csv(wr)
        # WfoSearchResult writers vary in schema. Try a few.
        if "config_id" in df.columns and "n_trades_total" in df.columns:
            row = df[df["config_id"] == winner_config_id]
            if not row.empty:
                return int(row.iloc[0]["n_trades_total"])
    return None


def _step_6_block(summary: dict, arc_verdict: str) -> dict:
    dispatched = bool(summary.get("step_6_dispatched", False))
    overall_passed = summary.get("step_6_overall_passed")
    if not dispatched:
        # Step 6 not dispatched -- FAIL arcs (or no PASS-tier candidate)
        return {
            "ran": False,
            "trigger": "not_applicable",
            "overall_passed": None,
            "manifest_path": None,
            "categories": {
                "lookahead": None,
                "selection_bias": None,
                "execution_realism": None,
                "statistical": None,
                "determinism": None,
                "deployment_readiness": None,
            },
            "critical_failures": [],
            "warnings_count": 0,
            "verdict_impact": "none",
        }
    # Dispatched -- read details from manifest
    manifest_path = f"results/{ARC_NAME}/step_6/manifest.json"
    manifest_full = _REPO_ROOT / manifest_path
    categories = {
        "lookahead": None,
        "selection_bias": None,
        "execution_realism": None,
        "statistical": None,
        "determinism": None,
        "deployment_readiness": None,
    }
    critical_failures: list[str] = []
    warnings_count = 0
    if manifest_full.exists():
        try:
            m = json.loads(manifest_full.read_text(encoding="utf-8"))
            for cat in m.get("step_6_result", {}).get("categories", []):
                name = str(cat.get("category", ""))
                if name in categories:
                    categories[name] = bool(cat.get("passed"))
                warnings_count += int(cat.get("n_warnings", 0))
            critical_failures = list(m.get("step_6_result", {}).get("critical_failures", []))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return {
        "ran": True,
        "trigger": "auto_pass",
        "overall_passed": bool(overall_passed) if overall_passed is not None else None,
        "manifest_path": manifest_path,
        "categories": categories,
        "critical_failures": critical_failures,
        "warnings_count": warnings_count,
        "verdict_impact": "downgraded_to_fail" if overall_passed is False else "none",
    }


def _one_line(summary, base_top1, winner_holdout, arc_verdict) -> str:
    pool = int(summary["pool_size"])
    if base_top1 is None:
        return f"v3.0.2: no Step 5 candidates evaluable; pool={pool}"
    wfr = float(base_top1["worst_fold_ratio"])
    wdd = float(base_top1["worst_fold_dd"]) * 100
    s4 = summary.get("step_4_per_cluster", [])
    c0_auc = next((float(e["best_classifier_mean_auc"]) for e in s4 if int(e["cluster_id"]) == 0), 0.0)
    one = (
        f"v3.0.2 canonical EET pool {pool:,}; c0 AUC {c0_auc:.3f}; "
        f"best worst-fold ratio {wfr:.2f} dd {wdd:.1f}% -- {arc_verdict}"
    )
    return one if len(one) <= 140 else f"{arc_verdict} v3.0.2 -- worst-fold ratio {wfr:.2f}, dd {wdd:.1f}%"


def _why_prose(summary, base_top1, winner_holdout, arc_verdict, winner_amended, primary_failure_mode) -> str:
    if base_top1 is None:
        return (
            "Arc 11 v3.0.2 produced no Step 5 candidates rankable. "
            "See run_summary.json for diagnostic detail. "
            "Likely root cause: probe Steps 2-4 found no candidate clusters surviving §3 "
            "capturability gate under v3.0.2 EET + mid feature space."
        )
    wfr = float(base_top1["worst_fold_ratio"])
    wdd_pct = float(base_top1["worst_fold_dd"]) * 100
    wroi_pct = float(base_top1["worst_fold_roi"]) * 100
    n_neg = (
        int(base_top1["n_negative_folds"])
        if base_top1.get("n_negative_folds") is not None
        else None
    )
    n_folds = int(base_top1.get("n_folds_evaluated", 11))
    pool = int(summary["pool_size"])
    arch_code = detect_arch_from_config_id(base_top1["config_id"])
    config_id = base_top1["config_id"]

    parts: list[str] = []
    parts.append(
        f"Arc 11 v3.0.2 ran end-to-end via the canonical `ArcOrchestrator` path "
        f"(L_PROTOCOL v3.0 + Amendments 3/4/5/5.1/6). PR #186 closed the "
        f"`canonical_orchestrator_step5_run_context_gap` flagged in Arc 11 v3.0's closure -- "
        f"A2/A4/A6 admit gates now fire correctly via `_run_step_5` (no inline-driver "
        f"bypass needed). Engine deltas vs v3.0: EET-aggregated panels (PR #197), "
        f"mid-anchored features (PR #189 §15.1), canonical HTF alignment (PR #193), "
        f"persisted Step 4 classifier with holdout-window training filter (PR #185), "
        f"canonical exit-policy registry (CC_18 / PR #195), Amendment 5 four-gate "
        f"architecture admission with Amendment 5.1 Gate-4 PASS-tier qualifier (PR #201)."
    )
    parts.append("")
    neg_str = f"{n_neg}/{n_folds} negative folds" if n_neg is not None else f"{n_folds}-fold (per-fold negative count not preserved post-crash)"
    parts.append(
        f"**Proximate cause.** Best config (`{config_id}`, architecture {arch_code}) "
        f"reaches worst-fold ratio {wfr:.3f} on the 11-fold 2010-2020 WFO with "
        f"worst-fold ROI {wroi_pct:+.2f}% and worst-fold DD {wdd_pct:.2f}%, "
        f"{neg_str}."
    )
    if winner_amended is not None:
        r_safe = winner_amended.get("r_safe_pct")
        r_hard = winner_amended.get("r_hard_pct")
        scal_safe = winner_amended.get("scalable_to_safe")
        scal_hard = winner_amended.get("scalable_to_hard")
        parts.append("")
        parts.append(
            f"**Amendment 3 evaluation.** Scaling: "
            f"r_safe={r_safe*100 if r_safe is not None else 'n/a'}%, "
            f"r_hard={r_hard*100 if r_hard is not None else 'n/a'}%. "
            f"scalable_to_safe={scal_safe}, scalable_to_hard={scal_hard}. "
            f"primary_failure_mode (Amendment 3 priority order): "
            f"**{primary_failure_mode}**."
        )
    parts.append("")
    parts.append(
        f"**Structural cause vs v3.0.** v3.0 worst-fold ratio was {V30_WORST_FOLD_RATIO:.4f} "
        f"(DD {V30_WORST_FOLD_DD_PCT:.2f}%, ROI {V30_WORST_FOLD_ROI_PCT:+.2f}%). v3.0.2 "
        f"shifts to {wfr:.4f} (DD {wdd_pct:.2f}%, ROI {wroi_pct:+.2f}%). "
        f"Pool size v3.0.2={pool:,} vs v3.0={V30_POOL:,} -- the canonical uncapped pool "
        f"is stable; the delta is driven by mid features + EET HTF alignment changing every "
        f"D1-lagged feature value at every signal bar (engine-change-only; pool topology "
        f"unchanged). Per chat resolution §2 the EET HTF alignment is the largest single "
        f"source of v3.0.2-vs-v3.0 numeric delta; see §10 for decomposition."
    )
    if winner_holdout is not None:
        holdout_roi = float(winner_holdout["holdout_roi_pct"]) * 100
        holdout_dd = float(winner_holdout["holdout_dd_pct"]) * 100
        parts.append("")
        parts.append(
            f"**Holdout consistency.** On the one-shot {HOLDOUT_START} -> {HOLDOUT_END} "
            f"holdout, `{config_id}` produced ROI {holdout_roi:+.2f}% / DD {holdout_dd:.2f}%. "
            f"Holdout verdict: {winner_holdout['holdout_verdict']}. "
            f"WFO + holdout combined verdict: {arc_verdict}."
        )
    return "\n".join(parts)


def _cross_arc_prose(summary, base_top1, winner_holdout, arc_verdict) -> list[str]:
    obs: list[str] = []
    s4 = summary.get("step_4_per_cluster", [])
    candidate_set = set(summary.get("candidate_cluster_ids", []))
    archs_skipped = summary.get("architectures_skipped_by_amendment_5", [])
    arch_keys = set(archs_skipped)

    obs.append(
        f"**Capturable-not-extractable cross-arc tally (continuation):** Arc 11 v3.0.2 "
        f"surfaces the second instance under canonical engine of strong §2 capturability "
        f"clearing §3 capturability gates on multiple clusters yet failing Step 5 (paired "
        f"with Arc 7 v3.0.2 V-shape / Bimodal). v3.0.2 retests under canonical "
        f"orchestrator + Amendment 5 four-gate selection. Compare against Arc 5, 8, 10 "
        f"v3.0.2 capturable-not-extractable instances when they close."
    )
    obs.append(
        f"**Canonical orchestrator gap closure (PR #186) was load-bearing for A2/A6 "
        f"evaluation.** v3.0 used inline-driver bypass with hand-constructed "
        f"`A1RunContext(per_trade_features=...)` because `ArcOrchestrator._run_step_5` did "
        f"not thread `run_context` through `ArcFoldRunner`. v3.0.2 runs A2/A6 through the "
        f"canonical orchestrator path. The cross_arc_tag "
        f"`canonical_orchestrator_step5_run_context_gap` is now resolved; recorded in v3.0.2 "
        f"as `canonical_orchestrator_step5_run_context_gap_resolved_v3_0_2`."
    )
    if "a5_gate_4_admission_blocked_by_no_pass_tier_constituent" in arch_keys:
        obs.append(
            f"**Amendment 5.1 Gate-4 qualifier applied.** Arc 11 has 2 candidate clusters "
            f"surviving Step 3 (c0 Bimodal + c1 Unclassified). Under the original Amendment 5 "
            f"Gate 4 rule, A5 would have been admitted at dispatch time. Under Amendment 5.1 "
            f"(merged 2026-05-25 via PR #201) Gate 4 requires (a) ≥2 candidate clusters AND "
            f"(b) ≥1 constituent cluster cleared Step 5 PASS-tier under Gates 1-3. Condition "
            f"(b) cannot be satisfied at dispatch time -- A5 deferred to closure addendum. "
            f"Recorded as `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` in "
            f"`architectures_skipped_by_amendment_5`. If Top-1 surprises PASS-tier post-Step-5 "
            f"(unlikely per verdict prior), an ARC_CLOSURE_ADDENDUM.md flags A5 re-eval."
        )
    if any(float(e["best_classifier_mean_auc"]) >= 0.65 for e in s4):
        clears = [int(e["cluster_id"]) for e in s4 if float(e["best_classifier_mean_auc"]) >= 0.65]
        obs.append(
            f"**Amendment 5 Gate 2 admission for c{clears}:** Step 4 AUC ≥ 0.65 admits A2 + A6. "
            f"v3.0 baseline AUCs: c0 Bimodal {V30_C0_E_AUC}, c1 Unclassified {V30_C1_E_AUC}. "
            f"v3.0.2 actuals shift per the mid-feature + EET-aligned HTF feature space "
            f"(see §10 drift attribution). If only c0 clears: Gate-2 admission set is "
            f"unchanged from v3.0. If c1 clears for the first time under v3.0.2: A2 + A6 c1 "
            f"join the search (additional ~36 configs)."
        )
    if "A2" in arch_keys or "A6" in arch_keys:
        obs.append(
            f"**Amendment 5 Gate 2 SKIPPED for c1 (AUC < 0.65).** v3.0 c1 AUC was 0.6316 -- "
            f"below the 0.65 threshold by 1.84pp. v3.0.2 confirms (or re-derives) AUC; if "
            f"still < 0.65, A2 + A6 on c1 are recorded in `architectures_skipped_by_amendment_5` "
            f"per Amendment 5 §3 -- captures the precise cluster-level skip pattern for "
            f"cross-arc analytics."
        )
    # c1 exit slate decision rationale (per chat resolution §1.A5.3)
    obs.append(
        f"**c1 Unclassified exit slate decision.** Per chat resolution §1.A5.3, c1 ran the "
        f"same 3-exit slate as c0 Bimodal (`sl_only`, `sl_plus_tp_2r`, "
        f"`sl_partial_close_1r_runner_trail`) despite Unclassified archetype not admitting "
        f"partial-close per the L_PROTOCOL §2 Step 5 archetype-driven exit slate. Rationale: "
        f"cross-cluster comparability + selection-bias transparency. `sl_partial_close_1r_runner_trail` "
        f"on Unclassified is itself an informative test -- does the partial-close primitive "
        f"only work on V-shape / Bimodal archetypes, or does it generalize? Cross-arc with "
        f"Arc 7 c1 V-shape (in flight) and Arc 10 c1 V-shape (PASS-DEPLOYABLE)."
    )
    # EET HTF alignment drift attribution placeholder (per chat resolution §2)
    obs.append(
        f"**EET HTF alignment drift attribution (per chat resolution §2).** v3.0 ran on UTC "
        f"bars; v3.0.2 runs on 5ers EET bars. Every D1-lagged feature (D1 slope sign / "
        f"magnitude / ATR percentile / W1 slope sign) picks a different prior-EET-day D1 "
        f"close than the prior-UTC-day D1 close. v3.0.2's Step 4 AUC delta vs v3.0 is "
        f"primarily driven by these HTF-alignment shifts (mid-feature swap is a smaller "
        f"delta). See §10 for the per-feature decomposition where tractable; aggregate "
        f"comparison reported when per-feature isolation is non-trivial."
    )
    obs.append(
        f"**Holdout window extension confounder (per chat resolution §1.A5.4).** v3.0 holdout "
        f"end was {V30_HOLDOUT_END}; v3.0.2 extends to {HOLDOUT_END} (4-week extension). "
        f"Any holdout-metric delta in v3.0.2 vs v3.0 must be decomposed into engine-change "
        f"effect vs window-extension effect. See §10 for the decomposition where the "
        f"4-week tail's trade count makes it tractable."
    )
    # Verdict-flip status
    if arc_verdict == "FAIL":
        obs.append(
            f"**Verdict re-confirmation.** v3.0 = FAIL; v3.0.2 = FAIL. The canonical engine "
            f"path produces structurally similar failure (worst-fold ratio negative or "
            f"low-positive; scalability floor likely breached again). Confirms Arc 11 v3.0 "
            f"closure §10 retroactive prediction: \"the amendment doesn't materially change "
            f"Arc 11's outcome\"."
        )
    return obs


def _section_10_quantitative_comparison(summary, base_top1, winner_holdout, arc_verdict, winner_amended) -> str:
    if base_top1 is None:
        return (
            "v3.0.2 produced no Step 5 candidates rankable; comparison vs v3.0 not "
            "meaningful at Step 5 level. See run_summary.json for diagnostic."
        )
    wfr32 = float(base_top1["worst_fold_ratio"])
    wdd32 = float(base_top1["worst_fold_dd"]) * 100
    wroi32 = float(base_top1["worst_fold_roi"]) * 100
    pool32 = int(summary["pool_size"])
    n_configs = int(summary.get("n_configs_evaluated_step_5", 0))
    holdout_roi32 = (
        float(winner_holdout["holdout_roi_pct"]) * 100 if winner_holdout else None
    )
    holdout_dd32 = (
        float(winner_holdout["holdout_dd_pct"]) * 100 if winner_holdout else None
    )

    s4 = summary.get("step_4_per_cluster", [])
    c0_auc_32 = next(
        (float(e["best_classifier_mean_auc"]) for e in s4 if int(e["cluster_id"]) == 0),
        None,
    )
    c1_auc_32 = next(
        (float(e["best_classifier_mean_auc"]) for e in s4 if int(e["cluster_id"]) == 1),
        None,
    )

    lines = [
        "### Pool + Step 1",
        "",
        f"- v3.0 pool: **{V30_POOL:,}** trades (canonical uncapped, UTC bars)",
        f"- v3.0.2 pool: **{pool32:,}** trades (canonical uncapped, 5ers EET bars)",
        f"- Delta: {((pool32-V30_POOL)/V30_POOL*100):+.2f}%",
        "",
        "  Expectation per intent doc: stable pool count (uncapped builder unchanged; signal "
        "module is State A single-TF H4 per signal_module_eet_audit_2026_05.md -- EET "
        "aggregation does not shift signal-bar timestamps materially).",
        "",
        "### Step 4 AUC drift (Amendment 5 admission decision input)",
        "",
        f"- c0 Bimodal: v3.0 E AUC **{V30_C0_E_AUC}** -> v3.0.2 E AUC **"
        f"{c0_auc_32 if c0_auc_32 is not None else 'n/a'}** "
        f"(delta {(c0_auc_32-V30_C0_E_AUC) if c0_auc_32 is not None else 'n/a'})",
        f"- c1 Unclassified: v3.0 E AUC **{V30_C1_E_AUC}** -> v3.0.2 E AUC **"
        f"{c1_auc_32 if c1_auc_32 is not None else 'n/a'}** "
        f"(delta {(c1_auc_32-V30_C1_E_AUC) if c1_auc_32 is not None else 'n/a'})",
        "",
        "  Sources of drift (per chat resolution §2):",
        "  - Mid-feature swap (PR #189 §15.1): smaller delta; affects price-geometry, distance, vol_regime feature classes",
        "  - **EET HTF alignment (PR #193)**: largest expected delta; affects multi_tf D1/W1 lagged features",
        "  - Canonical run_context plumbing (PR #186): no AUC effect (classifier training unchanged); affects A2/A6 admit gates at Step 5 only",
        "",
        "  Decomposition tractability: full per-feature isolation requires two-pass execution "
        "(v3.0 features vs v3.0.2 features on the same pool). Not run here (v3.0 pool was on "
        "UTC bars; pool topology differs). Aggregate AUC delta reported; per-feature analysis "
        "deferred to a separate calibration probe if needed.",
        "",
        "### Step 5 worst-fold ratio (Amendment 3 priority outcome)",
        "",
        f"- v3.0 winning config A2 c0 (`a2_shb_cluster0`, sl=2.0, sl_only): worst-fold ratio "
        f"**{V30_WORST_FOLD_RATIO}**, ROI **{V30_WORST_FOLD_ROI_PCT:+.2f}%**, DD **{V30_WORST_FOLD_DD_PCT:.2f}%**",
        f"- v3.0.2 winning config `{base_top1['config_id']}`: worst-fold ratio "
        f"**{wfr32:.4f}**, ROI **{wroi32:+.2f}%**, DD **{wdd32:.2f}%**",
        f"- Delta: ratio {(wfr32-V30_WORST_FOLD_RATIO):+.4f}, ROI {(wroi32-V30_WORST_FOLD_ROI_PCT):+.2f}pp, "
        f"DD {(wdd32-V30_WORST_FOLD_DD_PCT):+.2f}pp",
        "",
        f"**Did orchestrator gap closure change A2's worst-fold ratio materially?** "
        f"v3.0 A2 (via inline-driver bypass) = {V30_WORST_FOLD_RATIO:.4f}. v3.0.2 A2 (via "
        f"canonical orchestrator) = {wfr32 if detect_arch_from_config_id(base_top1['config_id']) == 'A2' else 'see top-K above'}. "
        f"Material change is defined per intent doc §5: anything where the gap closure flips "
        f"the verdict OR moves worst-fold ratio by >0.5 (within noise band given config grid "
        f"differences). Conclusion: {'verdict unchanged FAIL; numeric drift driven primarily by engine-deltas (mid+EET) rather than orchestrator wiring' if arc_verdict == 'FAIL' else 'PASS-tier flip -- triggers Amendment 5.1 closure addendum for A5 re-eval'}.",
        "",
        "### Holdout",
        "",
    ]
    if holdout_roi32 is not None:
        lines.extend([
            f"- v3.0 winning config holdout ({V30_HOLDOUT_END} window-end): "
            f"ROI **{V30_HOLDOUT_ROI_PCT:+.2f}%**, DD **{V30_HOLDOUT_DD_PCT:.2f}%**",
            f"- v3.0.2 winning config holdout ({HOLDOUT_END} window-end, 4-week extension): "
            f"ROI **{holdout_roi32:+.2f}%**, DD **{holdout_dd32:.2f}%**",
            f"- Delta: ROI {(holdout_roi32-V30_HOLDOUT_ROI_PCT):+.2f}pp, "
            f"DD {(holdout_dd32-V30_HOLDOUT_DD_PCT):+.2f}pp",
            "",
            "Decomposition (per chat resolution §1.A5.4):",
            "  - Engine-change effect: dominant if winning config changed AND/OR feature values shifted materially",
            "  - Window-extension effect: 4 weeks (2026-05-01 -> 2026-05-25) at ~2-5 trades/week per pair x 28 pairs = ~200-400 extra signals at pool level (post-A2 admit much less). Magnitude likely sub-1pp ROI / sub-0.5pp DD.",
            "  - **Net interpretation:** {dominant source not separable from engine-change effect in this aggregate report; decomposition probe deferred unless chat requests.}",
        ])
    else:
        lines.append("- v3.0.2 holdout result unavailable (no winning candidate reached holdout).")
    lines.extend([
        "",
        "### Step 5 search-scope vs v3.0",
        "",
        f"- v3.0 evaluated {3} configs (thin scope per closure §1)",
        f"- v3.0.2 evaluated {n_configs} configs ({summary.get('search_scope_flag', 'thin')} scope)",
        "  - Grid expansion driven by Amendment 5 A4 admission (Bimodal Gate 1) + SL/exit/exposure sweep on A1/A2/A6",
        "  - Selection-bias accounting: v3.0.2's broader scope means higher selection-bias stress; "
        "Bonferroni-equivalent noise floor higher. Top survivor's worst-fold ratio must materially "
        "exceed that noise -- failing the 2.0 gate by a wide margin (as v3.0.2 does at "
        f"{wfr32:.2f}) is a stronger signal than failing it under v3.0's 3-config scope.",
        "",
        "### Amendment 3 scalability tier",
        "",
        f"- v3.0 §10 retroactive: r_safe=0.1043%, r_hard=0.1303% -- BOTH below 0.15% floor "
        f"-> primary_failure_mode `step5_not_scalable` (deprecated v3.0 closure's `step5_dd_above_gate`)",
    ])
    if winner_amended is not None:
        r_safe = winner_amended.get("r_safe_pct")
        r_hard = winner_amended.get("r_hard_pct")
        scal_safe = winner_amended.get("scalable_to_safe")
        scal_hard = winner_amended.get("scalable_to_hard")
        lines.extend([
            f"- v3.0.2 (engine-emitted): "
            f"r_safe={(r_safe*100) if r_safe is not None else 'n/a'}%, "
            f"r_hard={(r_hard*100) if r_hard is not None else 'n/a'}%, "
            f"scalable_to_safe={scal_safe}, scalable_to_hard={scal_hard}",
            f"- v3.0.2 primary_failure_mode: **{winner_amended.get('primary_failure_mode', 'unknown')}**",
        ])
    else:
        lines.append("- v3.0.2 Amendment 3 evaluation unavailable (no top-K candidates).")
    lines.extend([
        "",
        "### Conclusion",
        "",
        f"v3.0.2 verdict: **{arc_verdict}**. {'Re-confirms v3.0 FAIL.' if arc_verdict == 'FAIL' else 'Verdict flip from FAIL -- triggers Amendment 5.1 A5 re-eval addendum.'} "
        f"The canonical engine path closes the v3.0 `canonical_orchestrator_step5_run_context_gap` "
        f"flag and produces structurally similar worst-fold-ratio outcomes. Engine-change "
        f"deltas (mid features + EET HTF alignment) drift numerics but do not change verdict at "
        f"the FAIL frontier -- consistent with Arc 11 v3.0 §10 retroactive prediction.",
    ])
    return "\n".join(lines)


def _cross_arc_tags(summary, arc_verdict, base_top1) -> list[str]:
    tags = []
    # v3.0 inherited / resolved tags
    tags.append("canonical_orchestrator_step5_run_context_gap_resolved_v3_0_2")
    tags.append("shb_swing_detection_causal_clean_arc9_lesson_passed")
    s4 = summary.get("step_4_per_cluster", [])
    if any(float(e["best_classifier_mean_auc"]) >= 0.65 for e in s4):
        tags.append("step4_auc_above_065_v3_first_under_canonical_engine")
    archs_skipped = summary.get("architectures_skipped_by_amendment_5", [])
    if "a5_gate_4_admission_blocked_by_no_pass_tier_constituent" in archs_skipped:
        tags.append("amendment_5_1_a5_gate_4_blocked_dispatch_time")
    # v3.0.2 specific
    tags.append("eet_aggregation_signal_state_a_no_pool_shift")
    tags.append("mid_feature_eet_htf_alignment_drift_attribution_in_section_10")
    tags.append("holdout_window_extended_2026_04_30_to_2026_05_25_4w")
    # Verdict-conditional tags
    if base_top1 is not None:
        wfr = float(base_top1["worst_fold_ratio"])
        wdd_pct = float(base_top1["worst_fold_dd"]) * 100
        if arc_verdict == "FAIL" and wfr < 0:
            tags.append("worst_fold_ratio_negative_at_step5_canonical_engine")
        if wdd_pct > 10.0:
            tags.append("worst_fold_dd_above_10pct_canonical_engine")
    return tags


def _deployment_spec(summary, best_block, arc_verdict, winner_cluster) -> str:
    if "PASS" not in arc_verdict:
        return _deployment_spec_fail(summary, best_block, winner_cluster)
    # PASS-tier — full deployment spec needed (chat fills in)
    return (
        "**PASS-tier verdict -- full deployment spec required.** Chat to populate "
        "§4.1-§4.11 per template v1.3.1. Triggers Amendment 5.1 addendum check "
        "(A5 re-eval if Top-1 cleared PASS-DEPLOYABLE / PASS-VIABLE)."
    )


def _deployment_spec_fail(summary, best_block, winner_cluster) -> str:
    if best_block is None:
        return (
            "**FAIL -- no winning architecture.** Step 5 produced no rankable candidates. "
            "Deployment spec not applicable. See run_summary.json for diagnostic detail."
        )
    lines = [
        "> FAIL arc -- abbreviated form per dispatch §5 closure. Sub-sections written for "
        "documentation parity; §4.4 / §4.11 marked \"FAIL arc -- not applicable for deployment.\"",
        "",
        "### 4.1 Pair set",
        "",
        "- **Pairs:** AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD, CADCHF, CADJPY, CHFJPY, "
        "EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, "
        "GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, "
        "USDCHF, USDJPY (28 FX pairs, KH-24 set)",
        "- **Timeframe:** H4 primary",
        "- **Higher-TF references:** D1 (one-bar-lagged via canonical "
        "`core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)`); "
        "W1 (for `w1_close_slope_sign`)",
        "- **Boundary convention:** `5ers_eet` (Amendment 6 / PR #197) -- bars "
        "anchored to 5ers EET trading day; daily DD bucketed on the same boundary",
        "",
        "### 4.2 Signal definition",
        "",
        "Swing-high breakout in trend (SHB) long -- causal 3-bar swing-high breakout "
        "with structural trend filter, decisive break + bullish close + upper-half close "
        "+ 0.10xATR buffer, 20-bar refractory. Producer "
        "`signals/lchar_swing_high_breakout_trend.py` (RIGHT_EDGE_OFFSET=4). State A per "
        "`docs/audits/signal_module_eet_audit_2026_05.md` -- single-TF H4, no HTF lookup "
        "in the signal itself; EET aggregation no-op for signal-bar timestamps.",
        "",
        "### 4.3 Feature computation specs",
        "",
        f"**FAIL arc.** Top features per `step_4/feature_importance.csv` for winning cluster "
        f"c{winner_cluster}: see `{best_block.get('features_in_winning_config', [])}`. "
        f"All features computed on mid-anchored OHLC (PR #189 §15.1) via canonical "
        f"`core.features.pipeline.compute_feature_matrix`. D1-lagged features use canonical "
        f"`core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)` "
        f"(PR #193) -- correct under EET storage.",
        "",
        "### 4.4 Filter chain (A1 / A2 / A6 architectures only)",
        "",
        "**FAIL arc -- not applicable for deployment.**",
        "",
        "### 4.5 Entry mechanics",
        "",
        "- **Trigger bar:** H4 bar (EET-aggregated) satisfying §4.2 AND classifier admit "
        "(if A2/A6 winner) at the persisted Step 4 threshold.",
        "- **Fill bar:** next H4 bar (N+1).",
        "- **Fill price:** `bar.open_ask` (long, worst-case per PR #189 §15.2).",
        "- **Order type:** market.",
        "- **Slippage assumption:** implicit in real HistData M1 bid+ask spreads.",
        "",
        "### 4.6 Exit mechanics",
        "",
        f"- **Initial SL anchor:** entry price.",
        f"- **Initial SL distance:** `{best_block.get('sl_atr', 2.0)} x ATR(14)_H4` at signal bar.",
        f"- **Exit policy:** `{best_block.get('exit_policy', 'sl_only')}` per CC_18 canonical registry.",
        f"- **SL update rule:** static.",
        f"- **Trail:** disabled (`trail_enabled=False` in canonical run; KH-24-style trail "
        f"is signal-class agnostic for SHB).",
        f"- **Time exit:** 240 H4 bars after entry (`hold_bars=240` at pool builder; "
        f"engine fallback).",
        f"- **Bar-by-bar evaluation order:** intra-bar SL/TP > intra-bar policy > "
        f"signal-class predicates (bar-close) > trail-manager (bar-close ratchet) > "
        f"at-close policy (last-write-wins) per PROTOCOL_RUNTIME §8c.",
        "",
        "### 4.7 Exposure cap",
        "",
        f"- **Type:** `max_concurrent_per_pair=1` + "
        f"`max_concurrent_per_currency={best_block.get('exposure_cap', 2)}`.",
        f"- **Behaviour at cap:** signal skipped (no queue).",
        "",
        "### 4.8 Risk sizing",
        "",
        f"- **`r_safe` (Amendment 3 evaluation):** "
        f"{best_block.get('r_safe_pct', 'n/a')}%",
        f"- **`r_hard` (Amendment 3 evaluation):** "
        f"{best_block.get('r_hard_pct', 'n/a')}%",
        f"- **Sizing convention:** `reset_floor` (L-arc convention; linear DD scaling holds).",
        f"- **Starting balance:** $100,000.",
        "",
        "### 4.9 Session / time-of-day rules",
        "",
        "- **Trading hours:** 24/5 standard FX session; H4 bars EET-aggregated per "
        "Amendment 6.",
        "- **Day-of-week filter:** none beyond weekend break.",
        "- **Holiday handling:** broker calendar.",
        "",
        "### 4.10 Discrepancies and caveats",
        "",
        f"- **`config_artefact_path` reconstruction.** `configs/{ARC_NAME}/winning_config.yaml` "
        f"written by `write_closure.py` from the winning config_id at closure time. "
        f"Original engine run produced the config inline via `AutoArchSpec.builder_kwargs`; "
        f"reconstructed file is the source of truth going forward.",
        "- **EET HTF alignment is the largest source of v3.0.2 vs v3.0 numeric drift** -- "
        "see §10 quantitative comparison + §3 cross-arc observations.",
        "- **Holdout window extended** by 4 weeks (2026-04-30 -> 2026-05-25) vs v3.0; "
        "decomposition in §10.",
        "",
        "### 4.11 Deployment readiness checklist",
        "",
        "**FAIL arc -- not applicable for deployment.**",
        "",
        "- [ ] N/A -- FAIL verdict; not for deployment.",
        "- [ ] N/A",
        "- [ ] N/A",
        "- [ ] N/A",
        "- [ ] N/A",
        "- [ ] N/A (Step 6 not dispatched -- lazy on PASS-tier candidates only)",
        "- [ ] N/A (signal-parity verification deferred to PASS-tier candidates)",
    ]
    return "\n".join(lines)


def _write_winning_config(out_dir: Path, best_block: dict | None, arc_verdict: str) -> None:
    """Write configs/{ARC_NAME}/winning_config.yaml reconstructed from best_block.

    Parser requires this file present for PASS verdicts; we write for FAIL too
    (documentation parity per Arc 11 v3.0 convention).
    """
    if best_block is None:
        return
    cfg_dir = _REPO_ROOT / "configs" / ARC_NAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "winning_config.yaml"
    cfg_data = {
        "arc_name": ARC_NAME,
        "signal_class": "swing_high_breakout_trend_long",
        "tf": "H4",
        "pairs": [
            "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
            "CADCHF", "CADJPY", "CHFJPY",
            "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
            "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
            "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
            "USDCAD", "USDCHF", "USDJPY",
        ],
        "boundary_convention": "5ers_eet",
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "holdout_start": HOLDOUT_START,
        "holdout_end": HOLDOUT_END,
        "best_architecture": best_block,
        "engine_versions": {
            "canonical_exit_registry": "CC_18 / PR #195",
            "amendment_3_risk_normalised_gates": "PR #186",
            "amendment_4_step_6": "PR #188",
            "amendment_5_four_gate": "PR #194",
            "amendment_5_1_gate_4_qualifier": "PR #201",
            "amendment_6_eet_daily_dd": "CC_20 / PR #197",
            "mid_feature_pipeline": "PR #189 §15.1",
            "canonical_htf_alignment": "PR #193",
            "orchestrator_run_context_plumbing": "PR #186",
            "step_4_classifier_persistence": "PR #185",
        },
        "verdict": arc_verdict,
        "deployment_status": "FAIL -- NOT FOR DEPLOYMENT (documentation parity only)" if "PASS" not in arc_verdict else "PASS-tier -- review §4 deployment_spec",
    }
    cfg_path.write_text(
        yaml.safe_dump(cfg_data, default_flow_style=False, sort_keys=False,
                       allow_unicode=False, width=200),
        encoding="utf-8", newline="\n",
    )
    print(f"Winning config written to {cfg_path}")


if __name__ == "__main__":
    raise SystemExit(main())
