"""Generate ARC_OPEN.md + ARC_CLOSURE.md per template v1.3.1 from run_summary.json.

Closure follows ``docs/templates/ARC_CLOSURE_TEMPLATE.md`` v1.3.1:
  - §1 tracker_payload YAML (Amendment 3 + Amendment 5 fields + step_6 block)
  - §2 Why succeeded/failed prose
  - §3 Cross-arc observations (W1 producer contamination + fix note)
  - §4 deployment_spec (REQUIRED on PASS verdicts)
  - §10 retroactive comparison vs PR #172 numbers if recoverable

Reads `results/l_arc_5_v3.0.2/run_summary.json`. Adapted from
`scripts/arc_7/write_closure.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SHAPE_TAG_TO_TEMPLATE = {
    "v_shape_recovery": "V-shape",
    "stepwise_climber": "Stepwise",
    "bimodal": "Bimodal",
    "monotonic_up": "Monotonic_up",
    "monotonic_down": "Monotonic_down",
    "choppy": "Choppy",
    "unclassified": "Unclassified",
}

ARCH_NAME_TO_FULL = {
    "A1": "A1 system_level_filter",
    "A2": "A2 classifier_filter",
    "A3": "A3 pipeline_de",
    "A4": "A4 pipeline_d_exits",
    "A5": "A5 portfolio_composition",
    "A6": "A6 meta_labeling",
}

ARC_NAME = "l_arc_5_v3.0.2"
SIGNAL_DESC = "mtf_alignment.2_down_mixed.kijun.h_120 (1H structural; H4+D1 aux)"


def _parse_cluster_from_config_id(config_id: str) -> int | None:
    for part in config_id.split("::"):
        if part.startswith("cl"):
            try:
                return int(part[2:])
            except ValueError:
                return None
    return None


def _parse_arch_from_config_id(config_id: str) -> str | None:
    parts = config_id.split("::")
    if parts and parts[0] in ARCH_NAME_TO_FULL:
        return parts[0]
    return None


def _parse_sl_from_config_id(config_id: str) -> float | None:
    for part in config_id.split("::"):
        if part.startswith("sl"):
            try:
                return float(part[2:])
            except ValueError:
                return None
    return None


def _parse_exit_policy_from_config_id(config_id: str) -> str | None:
    """Find canonical exit_policy registry name embedded in cid_str."""
    known = (
        "sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r",
        "sl_plus_trailing_atr", "sl_plus_trailing_swing",
        "sl_partial_close_1r_runner_trail",
    )
    for part in config_id.split("::"):
        if part in known:
            return part
    return None


def _parse_exp_from_config_id(config_id: str) -> str | int | None:
    for part in config_id.split("::"):
        if part.startswith("exp"):
            v = part[3:]
            return "unlimited" if v == "inf" else (int(v) if v.isdigit() else v)
    return None


def _build_clusters_block(summary: dict[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    clusters = summary.get("clusters", [])
    step4 = summary.get("step4_per_cluster", [])
    s4_by_id = {int(ce["cluster_id"]): ce for ce in step4}
    top3 = summary.get("step5", {}).get("top_3", [])
    top_cluster_ids: set[int] = set()
    for t in top3:
        cid = _parse_cluster_from_config_id(t["config_id"])
        if cid is not None:
            top_cluster_ids.add(cid)
    winning_verdict = summary.get("verdict", "FAIL")

    for c in clusters:
        cid = int(c["cluster_id"])
        s4 = s4_by_id.get(cid)
        if not c.get("is_candidate"):
            outcome = "dies_step3"
        elif s4 is None:
            outcome = "dies_step3"
        elif cid in top_cluster_ids:
            if winning_verdict.startswith("PASS-DEPLOYABLE"):
                outcome = "wins_step5"
            elif winning_verdict.startswith("PASS-VIABLE"):
                outcome = "viable_step5"
            else:
                outcome = "dies_step5"
        else:
            outcome = "dies_step5"
        out[f"c{cid}"] = {
            "n": int(c.get("n_trades", 0)),
            "archetype": SHAPE_TAG_TO_TEMPLATE.get(c.get("shape_tag", "unclassified"), "Unclassified"),
            "sl_atr": float(c.get("selected_sl", 2.0)),
            "step3_composite": float(c.get("capturability_composite", 0)),
            "mfe_p50_r": float(c.get("mfe_p50", 0)),
            "ww_pp": float(c.get("wrong_way_pp", 0)),
            "reach_1r": float(c.get("reach_1r", 0)),
            "step4_e_auc": float(s4.get("best_classifier_mean_auc", 0)) if s4 else None,
            "step4_d1_auc": None,
            "outcome": outcome,
        }
    return out


def _yaml_dump(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        if obj != obj:
            return "null"
        return f"{obj}"
    if isinstance(obj, str):
        special = any(ch in obj for ch in ":\n#&*!|>%@`")
        reserved = obj in ("yes", "no", "true", "false", "null", "")
        if special or reserved or obj.startswith(("-", "?", ",", "[", "]", "{", "}")):
            return json.dumps(obj)
        return obj
    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        if all(isinstance(x, (int, float, str, bool)) or x is None for x in obj):
            return "[" + ", ".join(_yaml_dump(x) for x in obj) + "]"
        lines = []
        for x in obj:
            lines.append(f"{sp}- {_yaml_dump(x, indent + 1).lstrip()}")
        return "\n".join(lines)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            if isinstance(v, dict) and v:
                lines.append(f"{sp}{k}:")
                lines.append(_yaml_dump(v, indent + 1))
            elif isinstance(v, (list, tuple)) and v and not all(
                isinstance(x, (int, float, str, bool)) or x is None for x in v
            ):
                lines.append(f"{sp}{k}:")
                lines.append(_yaml_dump(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {_yaml_dump(v, indent + 1)}")
        return "\n".join(lines)
    return repr(obj)


def _null_best_architecture() -> dict[str, Any]:
    return {
        "name": None, "cluster": None, "archetype": None, "config": None,
        "sl_atr": None, "exit_policy": None, "exposure_cap": None,
        "worst_fold_ratio": None, "worst_fold_roi_base_pct": None,
        "worst_fold_dd_base_pct": None, "mean_fold_ratio": None,
        "mean_fold_roi_pct": None, "sign_pos_folds": None, "n_trades_total": None,
        "holdout_roi_pct": None, "holdout_dd_pct": None, "holdout_passed": None,
        "oracle_worst_ratio": None, "oracle_real_gap_sharpe": None,
        "features_in_winning_config": [],
        "chained_max_dd_base_pct": None,
        "per_day_max_dd_artefact_path": None,
        "per_day_max_dd_base_summary": {
            "n_days": None, "p50_pct": None, "p95_pct": None,
            "p99_pct": None, "max_pct": None,
        },
        "k_safe": None, "k_hard": None, "r_safe_pct": None, "r_hard_pct": None,
        "scalable_to_safe": None, "scalable_to_hard": None,
        "worst_fold_roi_at_r_safe_pct": None, "worst_fold_roi_at_r_hard_pct": None,
        "chained_max_dd_at_r_safe_pct": None, "chained_max_dd_at_r_hard_pct": None,
        "daily_dd_breaches_at_r_safe": None, "daily_dd_breaches_at_r_hard": None,
        "holdout_roi_at_r_safe_pct": None, "holdout_dd_at_r_safe_pct": None,
        "holdout_roi_at_r_hard_pct": None, "holdout_dd_at_r_hard_pct": None,
        "sizing_convention": None, "chained_dd_method": None,
        "config_artefact_path": None, "deployment_spec_section_present": False,
    }


def _build_best_architecture(summary: dict[str, Any]) -> dict[str, Any]:
    top3 = summary.get("step5", {}).get("top_3", [])
    verdict = summary.get("verdict", "FAIL")
    if not top3:
        return _null_best_architecture()
    top1 = top3[0]
    cid = top1["config_id"]
    arch_key = _parse_arch_from_config_id(cid)
    cluster_id = _parse_cluster_from_config_id(cid)
    sl = _parse_sl_from_config_id(cid)
    exit_policy = _parse_exit_policy_from_config_id(cid) or "sl_only"
    exposure = _parse_exp_from_config_id(cid)

    archetype = None
    for c in summary.get("clusters", []):
        if int(c["cluster_id"]) == cluster_id:
            archetype = SHAPE_TAG_TO_TEMPLATE.get(c["shape_tag"], "Unclassified")
            break

    features = []
    if cluster_id is not None:
        for ce in summary.get("step4_per_cluster", []):
            if int(ce["cluster_id"]) == cluster_id:
                features = ce.get("top_features", [])[:10]
                break

    oracle_worst = None
    oracle = summary.get("oracle", {}).get(f"cluster_{cluster_id}", {})
    folds = oracle.get("fold_stats", [])
    if folds:
        oracle_worst = min(float(f["roi_dd_ratio"]) for f in folds)

    name = ARCH_NAME_TO_FULL.get(arch_key) if arch_key else None
    if verdict.startswith(("FAIL", "HALT")):
        # name retained — closure carries the best-explored arch even on FAIL
        pass

    config_artefact_path = (
        f"configs/{ARC_NAME.replace('.', '_')}/winning_config.yaml"
        if verdict.startswith("PASS") else None
    )

    return {
        "name": name,
        "cluster": cluster_id if cluster_id is not None else None,
        "archetype": archetype,
        "config": cid,
        "sl_atr": sl,
        "exit_policy": exit_policy,
        "exposure_cap": exposure,
        "worst_fold_ratio": float(top1.get("worst_fold_ratio", 0)),
        "worst_fold_roi_base_pct": float(top1.get("worst_fold_roi_base_pct", 0)),
        "worst_fold_dd_base_pct": float(top1.get("worst_fold_dd_base_pct", 0)),
        "mean_fold_ratio": float(top1.get("mean_fold_ratio", 0)),
        "mean_fold_roi_pct": None,
        "sign_pos_folds": None,
        "n_trades_total": None,
        "holdout_roi_pct": top1.get("holdout_roi_at_r_safe_pct"),
        "holdout_dd_pct": top1.get("holdout_dd_at_r_safe_pct"),
        "holdout_passed": (
            top1.get("holdout_roi_at_r_safe_pct") is not None
            and top1.get("holdout_roi_at_r_safe_pct") > 0
            and (top1.get("holdout_dd_at_r_safe_pct") or 0) <= 0.08
        ) if verdict.startswith("PASS-DEPLOYABLE") else None,
        "oracle_worst_ratio": oracle_worst,
        "oracle_real_gap_sharpe": None,
        "features_in_winning_config": features,
        "chained_max_dd_base_pct": float(top1.get("chained_max_dd_base_pct", 0)),
        "per_day_max_dd_artefact_path": top1.get("per_day_max_dd_artefact_path"),
        "per_day_max_dd_base_summary": {
            "n_days": None, "p50_pct": None, "p95_pct": None,
            "p99_pct": None, "max_pct": None,
        },
        "k_safe": float(top1.get("k_safe", 0)),
        "k_hard": float(top1.get("k_hard", 0)),
        "r_safe_pct": float(top1.get("r_safe_pct", 0)),
        "r_hard_pct": float(top1.get("r_hard_pct", 0)),
        "scalable_to_safe": bool(top1.get("scalable_to_safe", False)),
        "scalable_to_hard": bool(top1.get("scalable_to_hard", False)),
        "worst_fold_roi_at_r_safe_pct": float(top1.get("worst_fold_roi_at_r_safe_pct", 0)),
        "worst_fold_roi_at_r_hard_pct": float(top1.get("worst_fold_roi_at_r_hard_pct", 0)),
        "chained_max_dd_at_r_safe_pct": float(top1.get("chained_max_dd_at_r_safe_pct", 0)),
        "chained_max_dd_at_r_hard_pct": float(top1.get("chained_max_dd_at_r_hard_pct", 0)),
        "daily_dd_breaches_at_r_safe": int(top1.get("daily_dd_breaches_at_r_safe", 0)),
        "daily_dd_breaches_at_r_hard": int(top1.get("daily_dd_breaches_at_r_hard", 0)),
        "holdout_roi_at_r_safe_pct": top1.get("holdout_roi_at_r_safe_pct"),
        "holdout_dd_at_r_safe_pct": top1.get("holdout_dd_at_r_safe_pct"),
        "holdout_roi_at_r_hard_pct": top1.get("holdout_roi_at_r_hard_pct"),
        "holdout_dd_at_r_hard_pct": top1.get("holdout_dd_at_r_hard_pct"),
        "sizing_convention": top1.get("sizing_convention", "reset_floor"),
        "chained_dd_method": top1.get("chained_dd_method", "equity_stitching"),
        "config_artefact_path": config_artefact_path,
        "deployment_spec_section_present": verdict.startswith("PASS"),
    }


def _build_step_6_block(summary: dict[str, Any]) -> dict[str, Any]:
    s6 = summary.get("step6")
    if s6 is None:
        return {
            "ran": False,
            "trigger": "not_applicable",
            "overall_passed": None,
            "manifest_path": None,
            "categories": {
                "lookahead": None, "selection_bias": None,
                "execution_realism": None, "statistical": None,
                "determinism": None, "deployment_readiness": None,
            },
            "critical_failures": [],
            "warnings_count": 0,
            "verdict_impact": "none",
        }
    return {
        "ran": True,
        "trigger": s6.get("trigger", "auto_pass"),
        "overall_passed": s6.get("overall_passed"),
        "manifest_path": s6.get("manifest_path"),
        "categories": s6.get("categories", {}),
        "critical_failures": s6.get("critical_failures", []),
        "warnings_count": int(s6.get("warnings_count", 0)),
        "verdict_impact": s6.get("verdict_impact", "none"),
    }


def _build_one_line(verdict: str, summary: dict[str, Any]) -> str:
    pool = summary.get("pool_size", 0)
    if verdict == "HALT":
        return f"HALT — {summary.get('halt_reason', 'unspecified')}."
    top3 = summary.get("step5", {}).get("top_3", [])
    if verdict.startswith("PASS"):
        if top3:
            t = top3[0]
            return (
                f"{verdict}: pool {pool}, best `{t['config_id']}` worst-fold ratio "
                f"{float(t['worst_fold_ratio']):.2f} at r_safe={float(t['r_safe_pct']):.4%}."
            )
        return f"{verdict}: pool {pool}."
    n_candidates = sum(1 for c in summary.get("clusters", []) if c.get("is_candidate"))
    if top3:
        t = top3[0]
        return (
            f"FAIL: pool {pool}, {n_candidates} candidate clusters; top worst-fold ratio "
            f"{float(t['worst_fold_ratio']):.2f}, primary failure mode "
            f"`{t.get('primary_failure_mode', 'other')}`."
        )
    return f"FAIL: pool {pool}, {n_candidates} candidate clusters; no Step 5 candidates evaluated."


def _build_why_prose(verdict: str, summary: dict[str, Any], clusters: dict[str, dict]) -> str:
    pool = summary.get("pool_size", 0)
    k = summary.get("k_selected", "?")
    archetypes = sorted(set(c["archetype"] for c in clusters.values()))
    n_candidates = sum(1 for c in clusters.values() if c["outcome"] != "dies_step3")
    s4 = summary.get("step4_per_cluster", [])
    s4_aucs = {ce["cluster_id"]: float(ce["best_classifier_mean_auc"]) for ce in s4}
    top3 = summary.get("step5", {}).get("top_3", [])

    parts: list[str] = []
    parts.append(
        f"Pool size {pool:,} trades across 28 FX H1 bars 2010-01 → "
        f"{summary.get('window_end', '')} under 5ers EET bar boundary (PR #189 / Amendment 6). "
        f"Signal: `mtf_alignment.2_down_mixed.kijun.h_120` — 1H signal with H4 + D1 most-recent-completed Kijun-sign references, restored canonically per PR #193 (post-State-C zero-pool bug; see "
        f"`docs/audits/signal_module_eet_audit_2026_05.md`). "
        f"Step 2 selected K={k} (silhouettes {summary.get('silhouette_per_k', {})}). "
        f"Step 3 surfaced {len(clusters)} clusters; {n_candidates} flagged candidate "
        f"(archetypes: {', '.join(archetypes)})."
    )
    if s4:
        max_auc = max(s4_aucs.values()) if s4_aucs else 0.0
        parts.append(
            f" Step 4 RF/LGBM/LR per candidate cluster on the 27-feature catalogue — "
            f"max mean OOS AUC = {max_auc:.4f} (per cluster: {s4_aucs}). "
            f"Amendment 5 Gate 2 threshold = 0.65; persisted classifiers per PR #185."
        )
    if verdict.startswith("PASS"):
        if top3:
            t = top3[0]
            parts.append(
                f" **Best candidate** `{t['config_id']}` cleared Amendment 3 gates at "
                f"r_safe={float(t['r_safe_pct']):.4%} (k_safe={float(t['k_safe']):.2f}): "
                f"worst-fold ratio {float(t['worst_fold_ratio']):.2f}, worst-fold ROI at "
                f"r_safe {float(t['worst_fold_roi_at_r_safe_pct']):+.2%}, chained DD at "
                f"r_safe {float(t['chained_max_dd_at_r_safe_pct']):.2%}, "
                f"{int(t['daily_dd_breaches_at_r_safe'])} daily breaches @ r_safe. "
            )
            if t.get("holdout_roi_at_r_safe_pct") is not None:
                parts.append(
                    f"Holdout re-run at r_safe: "
                    f"ROI {float(t['holdout_roi_at_r_safe_pct']):+.2%}, DD "
                    f"{float(t['holdout_dd_at_r_safe_pct']):.2%}."
                )
    elif verdict == "HALT":
        parts.append(
            f" **Proximate cause:** {summary.get('halt_reason', 'unspecified')}. "
        )
    else:
        if top3:
            t = top3[0]
            parts.append(
                f" **Proximate cause:** best candidate `{t['config_id']}` reached worst-fold "
                f"ratio {float(t['worst_fold_ratio']):.2f} (need ≥ 2.0); primary failure "
                f"mode `{t.get('primary_failure_mode', 'other')}`. "
                f"Reason: {t.get('reason', 'n/a')}. "
                f"Worst-fold ROI base {float(t['worst_fold_roi_base_pct']):+.2%}, "
                f"worst-fold DD base {float(t['worst_fold_dd_base_pct']):.2%}, "
                f"chained DD base {float(t['chained_max_dd_base_pct']):.2%}."
            )
        if s4_aucs and max(s4_aucs.values()) < 0.65:
            parts.append(
                f" **Structural cause:** Step 4 entry-time classifiers could not separate "
                f"candidate-cluster membership above Amendment 5's Gate 2 threshold "
                f"(0.65); max mean OOS AUC {max(s4_aucs.values()):.4f}. "
                f"A2/A6 NOT admitted under Amendment 5."
            )
        s6 = summary.get("step6")
        if s6 and s6.get("verdict_impact") == "downgraded_to_fail":
            parts.append(
                f" Step 6 critical failures downgraded the Top-1 verdict: "
                f"{s6.get('critical_failures', [])}. "
            )

    return "".join(parts)


def _build_cross_arc_bullets(
    summary: dict[str, Any], clusters: dict[str, dict], verdict: str
) -> str:
    lines: list[str] = []
    s4 = summary.get("step4_per_cluster", [])
    s4_aucs = {ce["cluster_id"]: float(ce["best_classifier_mean_auc"]) for ce in s4}
    top3 = summary.get("step5", {}).get("top_3", [])
    oracle = summary.get("oracle", {})

    # Mandated W1 producer contamination + fix note (per resume §5, verbatim quote first)
    lines.append(
        "- **W1 producer contamination + canonical fix (verbatim, per cross-arc resume signal §5):** "
        "Prior `multi_tf.py::_w1_close_slope_sign` had within-period lookahead via `merge_asof`; "
        "fixed in `engine/w1_producer_canonical_alignment` (PR #208). "
        "Arc 5 ran with canonical producer. "
        "Cross-arc impact: Arc 8 originated detection; Arc 10 audit confirmed; "
        "Arc 5/8/10/11 all affected; Arc 7 v3.0.2 not affected (uses A2/A6 follow-up; ran post-fix)."
    )
    lines.append(
        "- **W1 producer fix — engineering detail.** Prior `core/features/multi_tf.py::_w1_close_slope_sign` "
        "used `pd.merge_asof` with within-period lookahead semantics (current W1 bar's close visible to H1 bars within the same W1 period). "
        "Engine fix `engine/w1_producer_canonical_alignment` shipped as PR #208 (merged "
        "2026-05-25; commit `ab03be9`) — `_w1_close_slope_sign` now uses canonical `core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)`. "
        "Arc 5 v3.0.2 is the FIRST Arc 5 run with the canonical W1 producer. "
        "Verified at smoke-test entry: "
        "`inspect.getsource(_w1_close_slope_sign)` contains `get_htf_value_at` and not `merge_asof`."
    )

    lines.append(
        "- **First Arc 5 run under canonical 5ers EET production substrate.** Prior Arc 5 v2.x ran under UTC + indicator-floor spreads; "
        "v3.0.1 was deleted from main after the signal module's State-C zero-pool failure was diagnosed (see `docs/audits/signal_module_eet_audit_2026_05.md`). "
        "This run uses the canonical `htf_alignment.get_htf_index_at(..., require_fully_closed=True)` signal module restored alongside the canonical utility; mid-price features (PR #189) + EET bar boundary (Amendment 6); "
        "27-feature catalogue + persisted classifiers (PR #185); Amendment 3 risk-normalised gates (PR #186); Amendment 5 four-gate AUC-driven architecture admission (PR #194); Step 6 framework auto-dispatch (PR #188 + #207 vacuous-pass patch). "
        "Not directly comparable to PR #172-era Arc 5 numbers; comparison in §10 is informational."
    )

    archetypes = sorted(set(c["archetype"] for c in clusters.values()))
    if archetypes:
        cluster_summary = "; ".join(f"{cid}={blk['archetype']}/{blk['outcome']}" for cid, blk in clusters.items())
        lines.append(
            f"- **Archetypes observed under v3.0.2 / EET / mid-price / canonical W1:** {', '.join(archetypes)}. "
            f"Cluster outcomes: {cluster_summary}."
        )

    # Amendment 5 admission summary
    a5_per_cluster = summary.get("amendment_5_per_cluster", {})
    a5_skipped = summary.get("architectures_skipped_under_amendment_5", [])
    if a5_per_cluster:
        admit_rows = []
        for cid, info in a5_per_cluster.items():
            auc_val = info.get('mean_oos_auc')
            auc_str = f"{auc_val:.4f}" if auc_val is not None else "n/a"
            admit_rows.append(
                f"c{cid}={info['archetype']}(AUC={auc_str}) → {info['admitted']}"
            )
        lines.append(
            f"- **Amendment 5 four-gate admission outcomes:** "
            f"{'; '.join(str(r) for r in admit_rows)}. "
            f"Architectures skipped vs Amendment 1's archetype-driven rule: "
            f"{sorted(a5_skipped) if a5_skipped else '(none)'}."
        )

    if oracle and top3:
        worst_realised = float(top3[0].get("worst_fold_ratio", 0))
        for cid_str, oracle_data in oracle.items():
            folds = oracle_data.get("fold_stats", [])
            if folds:
                oracle_worst = min(float(f["roi_dd_ratio"]) for f in folds)
                gap = oracle_worst - worst_realised
                if abs(gap) > 1.0:
                    lines.append(
                        f"- **Oracle-realised gap on {cid_str}:** {oracle_worst:.2f} (oracle) vs "
                        f"{worst_realised:.2f} (realised) = {gap:+.2f}. "
                        + ("Latent path-shape edge unrecovered by deployed-architecture set."
                           if gap > 1.0 else "")
                    )
                    break

    if s4_aucs:
        max_auc = max(s4_aucs.values())
        if max_auc >= 0.65:
            lines.append(
                f"- **Entry-feature AUC ≥ 0.65 deployability bar achieved.** Max mean OOS AUC "
                f"= {max_auc:.4f} on the v3 27-feature catalogue. Amendment 5 Gate 2 admits A2 + A6."
            )
        elif max_auc > 0:
            lines.append(
                f"- **Entry-feature AUC ceiling at {max_auc:.4f}** (below 0.65 Amendment 5 Gate 2 threshold). "
                f"A2/A6 NOT admitted; Amendment 5's tighter rule preserves Step 5 search budget."
            )

    s6 = summary.get("step6")
    if s6 and s6.get("ran"):
        lines.append(
            f"- **Step 6 framework auto-dispatched (Amendment 4 / PR #188).** "
            f"Overall passed: {s6.get('overall_passed')}, "
            f"trigger: `{s6.get('trigger', 'auto_pass')}`, "
            f"warnings: {s6.get('warnings_count', 0)}, "
            f"verdict impact: `{s6.get('verdict_impact', 'none')}`. "
            f"Critical failures: {s6.get('critical_failures', []) or 'none'}."
        )

    scope = summary.get("step5", {}).get("search_scope", "thin")
    n_configs = summary.get("step5", {}).get("n_configs", 0)
    lines.append(
        f"- **Selection-bias scope:** {scope} ({n_configs} configs evaluated). "
        f"Skipped exit-policy variants listed in `step_5/skipped_configs.md`."
    )

    return "\n".join(lines)


def _build_deployment_spec_section(summary: dict[str, Any], top1: dict[str, Any]) -> str:
    cid = top1["config_id"]
    cluster_id = _parse_cluster_from_config_id(cid)
    sl_atr = float(_parse_sl_from_config_id(cid) or 2.0)
    r_safe = float(top1.get("r_safe_pct", 0.005))
    arch = _parse_arch_from_config_id(cid)
    exit_policy = _parse_exit_policy_from_config_id(cid) or "sl_only"

    features = []
    if cluster_id is not None:
        for ce in summary.get("step4_per_cluster", []):
            if int(ce["cluster_id"]) == cluster_id:
                features = ce.get("top_features", [])[:10]
                break

    feature_descriptions = "\n".join(
        f"- **`{f}`** — see `docs/features_reference.md` for source bars / formula / lag rule."
        for f in features
    ) or "- (Step 4 produced no top features for the winning cluster.)"

    return f"""## §4 deployment_spec

### 4.1 Pair set

- **Pairs:** AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY NZDUSD USDCAD USDCHF USDJPY
- **Timeframe:** H1 (5ers EET boundary per PR #189 / Amendment 6)
- **Higher-TF references:** H4 + D1 (most-recent-completed via `get_htf_index_at(require_fully_closed=True)`)

### 4.2 Signal definition

```
At each H1 bar N close:
  kijun_sign_1H(N)    == -1   (1H mid-close below 1H Kijun-26)
  kijun_sign_4H_mr(N) == +1   (most-recent-completed 4H Kijun-26 above)
  kijun_sign_D1_mr(N) == -1   (most-recent-completed D1 Kijun-26 below)
fires entry at bar N+1 open_ask (long worst-case fill).
```

Kijun-26 computed on mid OHLC `(bid + ask) / 2`. Most-recent-completed enforced
via `core.signals.htf_alignment.get_htf_index_at(require_fully_closed=True)` —
matched H4/D1 bar's left-edge timestamp must be strictly before H1 close time.

### 4.3 Feature computation specs

Cluster-{cluster_id} Step-4 classifier top features (consumed by {arch}):

{feature_descriptions}

All features computed on mid OHLC `(close_bid + close_ask) / 2` per PR #189 §15.1
(except `spread_regime.*` which read raw `spread_close`). Multi-TF features
read canonical `core.signals.htf_alignment` helpers (no `merge_asof`).

### 4.4 Filter chain ({arch})

Architecture: `{arch}` instantiated via `core.steps.classifier_persistence.build_{arch.lower() if arch else 'a1'}_config_from_step4`.
Classifier: persisted at `results/{ARC_NAME}/step_4/classifiers/{cluster_id}.pkl` (SHA256-verified manifest).
Decision threshold: see §1 tracker_payload.best_architecture.

### 4.5 Entry mechanics

- Trigger bar: N (H1 close)
- Fill bar: N+1
- Fill price: `bar.open_ask` (long worst-case)
- Order type: market

### 4.6 Exit mechanics

- Initial SL: `entry_proxy − {sl_atr:.1f} × ATR(14)[N]` where entry_proxy = signal-bar `close_ask`
- Exit policy: `{exit_policy}` (registered in `core.sim.exit_policies`)
- Time exit: hold cap at signal-bar + 240 H1 bars

### 4.7 Exposure cap

- Type: per-currency
- Counter: `max_concurrent_per_currency` per config; `max_concurrent_per_pair = 1`
- Behaviour at cap: reject candidate at fill time

### 4.8 Risk sizing

- `r_safe` = **{r_safe:.4%}** (Amendment 3 per-arc derived)
- Risk basis: reset-floor (L arc convention)
- Floor reset: per L arc convention (account high-water mark)
- Position size: `floor × r_safe / |entry − sl|`
- Lot rounding: 0.01 (standard MT5)

### 4.9 Session / time-of-day rules

- Trading hours: 24/5 (5ers EET boundary)
- Day-of-week filter: none
- News-window filter: none in backtest; live EA may add
- Holiday handling: signal does not fire on bars with `bid_ask_data_quality != "ok"`

### 4.10 Discrepancies and caveats

- **First Arc-5 run under canonical 5ers EET + canonical W1 producer.** Not numerically comparable to Arc 5 v2.x results.
- **chained_dd_method = `equity_stitching`:** v3.0.1 default. v3.0.2 follow-up may replace with `full_window_sim` per chat directive Q6.

### 4.11 Deployment readiness checklist

- [ ] Config YAML at `configs/{ARC_NAME.replace('.', '_')}/winning_config.yaml` exists and is self-contained
- [ ] All features in §4.3 reproduce against KH-24 live MT5 data within tolerance
- [ ] Risk sizing at `r_safe = {r_safe:.4%}` confirmed feasible against 5ers broker minimums
- [ ] EA implementation matches §4.2-4.9 line-by-line
- [ ] Backtest-vs-EA byte-identical pre-deployment shadow run on 30 days of data
- [ ] Step 6 causal audit clean
- [ ] Signal parity verified against deployment venue (PR #189 hard requirement)
"""


def _build_section_10_retroactive(summary: dict[str, Any], verdict: str) -> str:
    """§10 — informational retroactive comparison vs prior Arc 5 v2.x.

    Per intent §I.3: framed as 'first test under canonical 5ers production
    conditions', NOT 'reproduces prior numbers'.
    """
    top3 = summary.get("step5", {}).get("top_3", [])
    top_str = "(no Top-1)" if not top3 else (
        f"Top-1 `{top3[0]['config_id']}` worst-fold ratio "
        f"{float(top3[0]['worst_fold_ratio']):.2f}, verdict `{top3[0].get('verdict', 'n/a')}`"
    )
    return f"""## §10 Retroactive comparison vs prior Arc 5 numbers (informational)

Per intent doc §I.3 and §J: **this arc is NOT directly comparable to PR #172-era
Arc 5 v3.0.0/v3.0.1 results.** The prior Arc 5 v3.0.1 was deleted from main
after the signal module's State-C zero-pool failure was diagnosed under EET
storage (see `docs/audits/signal_module_eet_audit_2026_05.md`); the original
Arc 5 v2.x ran under L_ARC_PROTOCOL v2.1.1 with UTC bar boundaries +
indicator-floor spreads + a different feature substrate entirely.

The Arc 5 v2.x closure (`docs/archive/arc_results/ARC_5_RESULT.md`) recorded
SHELVED Step 6 FAIL with no ship candidate (all three strategy candidates
closed Step 6 with negative worst-fold ROI at every risk-per-trade level under
PR2 + new spreads).

Arc 5 v3.0.2 (this run) is the FIRST Arc 5 test under canonical 5ers production
conditions: signal-parity engine (PR #189), canonical HTF alignment (PR #193),
canonical W1 producer (PR #208), mid-price features, EET bar boundary +
EET daily-DD bucketing (Amendment 6 / PR #197), Amendment 3 risk-normalised
gates (PR #186), Amendment 5 four-gate architecture admission (PR #194),
Step 6 framework auto-dispatch (PR #188 + #207).

This run's outcome: **{verdict}**. {top_str}.

Cross-substrate delta (v2.x SHELVED → v3.0.2 {verdict}) is not load-bearing —
the methodologies test different things. Recoverable v3.0.1-era pool / cluster
numbers from prior branches (now deleted) are not in scope.
"""


def render_arc_open(summary: dict[str, Any]) -> str:
    opened = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""# ARC_OPEN — {ARC_NAME}

```
arc_name: {ARC_NAME}
opened: {opened}
signal_class: mtf_alignment 2_down_mixed kijun (1H structural; H4+D1 aux)
signal_definition: {SIGNAL_DESC}
tf_mode: locked
tf: H1
sub_protocol: vanilla
pair_set: 28 FX (same as KH-24)
window: {summary.get('window_start', '2010-01-01')} → {summary.get('window_end', '')}
risk_per_trade: 0.50% (verdict at scaled r_safe / r_hard per Amendment 3)
boundary_convention: 5ers_eet
```

## Hypothesis

Run mtf_alignment.2_down_mixed.kijun.h_120 through L_PROTOCOL v3.0 + Amendments
1-6 under canonical 5ers EET substrate + canonical HTF alignment (PR #193) +
canonical W1 producer (PR #208). Prior v2.x finding (SHELVED Step 6 FAIL) is
historical record only; cluster archetypes, surviving cohorts, and verdicts
must come from v3.0.2 evidence alone.

## Expected failure modes

- AUC ceiling persists below 0.65 → Amendment 5 Gate 2 SKIPs A2/A6
- Scalability check fails → step5_not_scalable
- Holdout fails after IS pass → holdout_fail_after_is_pass
- Step 6 critical failure → step6_causal_audit_fail (verdict downgrade)
- Amendment 5.1 Gate 4 HALT (≥2 candidate clusters + ≥1 PASS-tier constituent; A5 engine not built)
"""


def render_closure(summary: dict[str, Any], closed_at: str) -> str:
    verdict = summary.get("verdict", "FAIL")
    top3 = summary.get("step5", {}).get("top_3", [])
    failed_at_step = "N/A" if verdict.startswith("PASS") else (
        "1" if verdict == "HALT" and "step_1" in summary.get("halt_reason", "")
        else ("5" if verdict == "HALT" and "gate_4" in summary.get("halt_reason", "") else "5")
    )
    primary_failure_mode = (
        "N/A" if verdict.startswith("PASS")
        else (top3[0].get("primary_failure_mode", "other") if top3
              else ("pool_too_small" if verdict == "HALT" else "other"))
    )

    clusters_block = _build_clusters_block(summary)
    archetypes_seen = sorted(set(c["archetype"] for c in clusters_block.values()))

    architectures_tested_set: set[str] = set()
    a5_per_cluster = summary.get("amendment_5_per_cluster", {})
    for info in a5_per_cluster.values():
        architectures_tested_set.update(info.get("admitted", []))
    architectures_tested = sorted(architectures_tested_set)

    architecture_results = {}
    winner_arch = _parse_arch_from_config_id(top3[0]["config_id"]) if top3 else None
    for ak in architectures_tested:
        won = (ak == winner_arch) and verdict.startswith("PASS")
        ratios = [
            float(t["worst_fold_ratio"]) for t in top3
            if _parse_arch_from_config_id(t["config_id"]) == ak
        ]
        architecture_results[ak] = {
            "tested": True, "won": won,
            "worst_fold_ratio": max(ratios) if ratios else None,
        }

    # Amendment 5 skipped-architectures field for the closure
    archs_skipped_a5 = summary.get("architectures_skipped_under_amendment_5", [])

    cross_arc_tags = [
        "arc_5_v3_0_2_first_run_canonical_5ers_eet",
        "w1_producer_canonical_fix_pr_208_applied",
        "amendment_5_four_gate_admission",
        "amendment_3_risk_normalised_gates_evaluated",
        "amendment_6_eet_daily_dd_boundary",
        "signal_module_eet_audit_state_a_restored_canonically",
    ]
    archetype_tag_map = {
        "V-shape": "v_shape_observed",
        "Stepwise": "stepwise_observed",
        "Bimodal": "bimodal_observed",
        "Monotonic_up": "monotonic_up_observed",
        "Monotonic_down": "monotonic_down_observed",
        "Choppy": "choppy_observed",
        "Unclassified": "unclassified_only",
    }
    for at in archetypes_seen:
        if at in archetype_tag_map:
            cross_arc_tags.append(archetype_tag_map[at])

    if top3 and bool(top3[0].get("scalable_to_safe", False)):
        cross_arc_tags.append("scalable_to_safe_within_bounds")
    s6 = summary.get("step6")
    if s6 and s6.get("ran"):
        cross_arc_tags.append("step_6_auto_dispatched_per_amendment_4")
    if verdict.startswith("PASS"):
        cross_arc_tags.append("pass_verdict_under_amendment_3")

    n_configs = int(summary.get("step5", {}).get("n_configs", 0))
    search_scope = "thin" if n_configs < 50 else ("normal" if n_configs < 100 else "broad")

    tracker_payload = {
        "template_version": "v1.3",
        "arc_name": ARC_NAME,
        "signal": SIGNAL_DESC,
        "tf": "H1",
        "sub_protocol": "vanilla",
        "closed_timestamp": closed_at,
        "closure_doc_link": f"results/{ARC_NAME}/ARC_CLOSURE.md",
        "verdict": verdict,
        "one_line": _build_one_line(verdict, summary),
        "failed_at_step": failed_at_step,
        "primary_failure_mode": primary_failure_mode,
        "pool_metadata": {
            "total_n": int(summary.get("pool_size", 0)),
            "window_start": summary.get("window_start", "2010-01-01"),
            "window_end": summary.get("window_end", ""),
            "kh24_co_fire_pct": None,
            "configs_evaluated_step5": n_configs,
            "search_scope_flag": search_scope,
        },
        "best_architecture": _build_best_architecture(summary),
        "cost_decomposition": None,
        "clusters": clusters_block,
        "architectures_tested": architectures_tested,
        "architecture_results": architecture_results,
        "architectures_skipped_by_amendment_5": archs_skipped_a5,
        "archetypes_observed": archetypes_seen,
        "cross_arc_tags": cross_arc_tags,
        "step_6": _build_step_6_block(summary),
    }

    yaml_body = _yaml_dump({"tracker_payload": tracker_payload}, indent=0)
    why = _build_why_prose(verdict, summary, clusters_block)
    bullets = _build_cross_arc_bullets(summary, clusters_block, verdict)
    section_10 = _build_section_10_retroactive(summary, verdict)

    closure = f"""# ARC_5_v3_0_2_CLOSURE — {ARC_NAME}

> **Closed:** {closed_at}
> **Branch:** arc/{ARC_NAME}
> **Closure doc path:** results/{ARC_NAME}/ARC_CLOSURE.md
> **Boundary convention:** 5ers_eet (canonical, end-to-end)

---

## §1 tracker_payload

```yaml
{yaml_body}
```

---

## §2 Why {"succeeded" if verdict.startswith("PASS") else "failed"}

{why}

---

## §3 Cross-arc observations

{bullets}

"""
    if verdict.startswith("PASS") and top3:
        closure += "\n---\n\n" + _build_deployment_spec_section(summary, top3[0])

    closure += "\n---\n\n" + section_10
    return closure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path,
        default=Path(f"results/{ARC_NAME}/run_summary.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path(f"results/{ARC_NAME}"))
    parser.add_argument("--closed-at", type=str, default=None)
    args = parser.parse_args(argv)

    if not args.summary.exists():
        print(f"ERROR: {args.summary} not found", file=sys.stderr)
        return 1
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    closed_at = args.closed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "ARC_OPEN.md").write_text(
        render_arc_open(summary), encoding="utf-8", newline="\n"
    )
    (out_dir / "ARC_CLOSURE.md").write_text(
        render_closure(summary, closed_at), encoding="utf-8", newline="\n"
    )

    print(f"Wrote {out_dir / 'ARC_OPEN.md'}")
    print(f"Wrote {out_dir / 'ARC_CLOSURE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
