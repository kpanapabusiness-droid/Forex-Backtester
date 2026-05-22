"""Emit ARC_5 closure doc per docs/templates/ARC_CLOSURE_TEMPLATE.md v1.0.

Reads Step 1-5 artefacts from ``results/l_arc_5/``, derives verdict per
L_PROTOCOL §3 + template §1 enum, writes
``results/l_arc_5/ARC_CLOSURE.md`` conforming exactly to the locked
template (§1 YAML tracker_payload + §2 prose + §3 cross-arc).

Also updates ARC_TRACKER.md per template §4 (closed-arcs row, failure-mode
count, per-architecture win rate, per-archetype recurrence, cluster
registry, tag registry).

Usage:
    py scripts/arc_5/emit_closure.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results" / "l_arc_5"
TRACKER_PATH = REPO_ROOT / "ARC_TRACKER.md"

ARC_NAME = "l_arc_5"
SIGNAL_DESC = "mtf_alignment.2_down_mixed.kijun.h_120 (LCHAR Entry 5, h=120 override)"
TF = "H1"
SUB_PROTOCOL = "vanilla"
WINDOW_START = "2010-01-01"
WINDOW_END = "2026-04-30"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _read_step_artefacts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    s1_manifest = json.loads((RESULTS_DIR / "step_1" / "manifest.json").read_text(encoding="utf-8"))
    out["pool"] = pd.read_parquet(RESULTS_DIR / "step_1" / "pool.parquet")
    out["paths"] = pd.read_parquet(RESULTS_DIR / "step_1" / "paths.parquet")
    out["s1_manifest"] = s1_manifest
    out["clusters"] = pd.read_parquet(RESULTS_DIR / "step_2" / "cluster_assignments.parquet")
    out["s2_summary_csv"] = pd.read_csv(RESULTS_DIR / "step_2" / "cluster_summary.csv")
    out["capturability"] = pd.read_csv(RESULTS_DIR / "step_3" / "capturability.csv")
    s4_metrics = RESULTS_DIR / "step_4" / "extraction_metrics.csv"
    out["extraction"] = pd.read_csv(s4_metrics) if s4_metrics.exists() else None
    s4_imp = RESULTS_DIR / "step_4" / "feature_importance.csv"
    out["feature_importance"] = pd.read_csv(s4_imp) if s4_imp.exists() else None
    out["s5_manifest"] = json.loads((RESULTS_DIR / "step_5" / "manifest.json").read_text(encoding="utf-8"))
    out["wfo_results"] = pd.read_csv(RESULTS_DIR / "step_5" / "wfo_results.csv")
    out["holdout"] = pd.read_csv(RESULTS_DIR / "step_5" / "holdout.csv")
    return out


def _derive_verdict(art: dict[str, Any]) -> tuple[str, str, dict | None]:
    holdout = art["holdout"]
    if len(holdout) == 0:
        return ("FAIL", "no Step 5 candidates surfaced", None)
    top = holdout.iloc[0]
    deployable = bool(top.get("deployable", False))
    if deployable:
        return ("PASS-DEPLOYABLE", f"{top['config_id']} cleared both search and holdout gates", top.to_dict())
    search_v = str(top.get("search_verdict", "fail"))
    holdout_v = str(top.get("holdout_verdict", "fail"))
    if search_v == "pass_viable" or holdout_v == "pass_viable":
        return ("PASS-VIABLE", f"{top['config_id']} cleared PASS-VIABLE gate", top.to_dict())
    return ("FAIL", f"top-1 {top['config_id']} failed: search={search_v}, holdout={holdout_v}", top.to_dict())


def _failure_mode(verdict: str, art: dict[str, Any]) -> str:
    if verdict in ("PASS-DEPLOYABLE", "PASS-VIABLE"):
        return "N/A"
    cap = art["capturability"]
    cand = cap[cap["is_candidate"]] if "is_candidate" in cap.columns else pd.DataFrame()
    if len(art["pool"]) < 500:
        return "pool_too_small"
    if len(cand) == 0:
        return "no_capturable_cluster"
    if art.get("extraction") is not None:
        ext = art["extraction"]
        if "auc" in ext.columns and len(ext) > 0:
            best_mean = ext.groupby(["cluster_id", "classifier"])["auc"].mean().max()
            if float(best_mean) < 0.55:
                return "entry_feature_auc_ceiling"
    holdout = art["holdout"]
    if len(holdout) > 0:
        top = holdout.iloc[0]
        sv = str(top.get("search_verdict", ""))
        hv = str(top.get("holdout_verdict", ""))
        if sv in ("pass_deployable", "pass_viable") and hv == "fail":
            return "holdout_fail_after_is_pass"
        if float(top.get("search_worst_dd", 0)) > 0.10:
            return "step5_dd_above_gate"
        if float(top.get("search_worst_ratio", 0)) < 2.0:
            return "step5_wf_roi_below_gate"
        if int(top.get("search_negative_folds", 0)) > 0:
            return "step5_sign_consistency_fail"
    return "other"


def _archetype_short(longform: str) -> str:
    m = {
        "V-shape recovery": "V-shape",
        "Stepwise climber": "Stepwise",
        "Bimodal": "Bimodal",
        "Monotonic up": "Monotonic_up",
        "Monotonic down": "Monotonic_down",
        "Choppy": "Choppy",
        "Unclassified": "Unclassified",
    }
    return m.get(longform, longform)


def _best_mean_auc_for_cluster(ext: pd.DataFrame, cid: int) -> float | None:
    """Compute the best classifier's mean AUC across folds for one cluster."""
    if ext is None or len(ext) == 0:
        return None
    sub = ext[ext["cluster_id"] == cid]
    if len(sub) == 0:
        return None
    means = sub.groupby("classifier")["auc"].mean()
    if len(means) == 0:
        return None
    return float(means.max())


def _build_clusters_yaml_block(art: dict[str, Any]) -> dict[str, Any]:
    cap = art["capturability"]
    ext = art["extraction"] if art["extraction"] is not None else pd.DataFrame()
    out: dict[str, Any] = {}
    for _, row in cap.iterrows():
        cid = int(row["cluster_id"])
        is_cand = bool(row.get("is_candidate", False))
        best_auc = _best_mean_auc_for_cluster(ext, cid) if is_cand else None
        outcome = "passed_step3" if is_cand else "dies_step3"
        if best_auc is not None and is_cand and best_auc < 0.55:
            outcome = "dies_step4"
        out[f"c{cid}"] = {
            "n": int(row.get("n_trades", row.get("n", 0))),
            "archetype": _archetype_short(str(row.get("shape_tag", "Unclassified"))),
            "sl_atr": float(row.get("selected_sl", row.get("sl_atr", 0))),
            "step3_composite": float(row.get("capturability_composite", 0)),
            "mfe_p50_r": float(row.get("mfe_p50", 0)),
            "ww_pp": float(row.get("wrong_way_pp", 0)),
            "reach_1r": float(row.get("reach_1r", 0)),
            "step4_e_auc": best_auc,
            "step4_d1_auc": None,
            "outcome": outcome,
        }
    return out


def _arch_results_block(art: dict[str, Any], winner_id: str | None) -> tuple[list[str], dict[str, dict]]:
    wfo = art["wfo_results"]
    tested = ["A1"]
    by_arch: dict[str, dict[str, Any]] = {"A1": {"tested": True, "won": False, "worst_fold_ratio": None}}
    if len(wfo) > 0:
        worst_per_config = wfo.groupby("config_id")["roi_dd_ratio"].min()
        best_config = worst_per_config.idxmax()
        best_worst = float(worst_per_config.max())
        by_arch["A1"]["worst_fold_ratio"] = best_worst
        if winner_id == best_config:
            by_arch["A1"]["won"] = True
    return tested, by_arch


def _yaml_dump_for_template(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, width=120)


def _build_tracker_payload(art: dict[str, Any], verdict: str, verdict_reason: str, winner_row: dict | None) -> dict[str, Any]:
    fmode = _failure_mode(verdict, art)
    s5 = art["s5_manifest"]
    pool = art["pool"]
    clusters_yaml = _build_clusters_yaml_block(art)
    archetypes_observed = sorted({c["archetype"] for c in clusters_yaml.values()})
    winner_id = winner_row["config_id"] if winner_row is not None else None
    tested_archs, arch_results = _arch_results_block(art, winner_id)

    if winner_row is None:
        best_arch = {k: None for k in (
            "name", "cluster", "archetype", "config", "sl_atr", "exit_policy", "exposure_cap",
            "worst_fold_ratio", "worst_fold_roi_pct", "worst_fold_dd_pct", "mean_fold_ratio",
            "mean_fold_roi_pct", "sign_pos_folds", "n_trades_total", "holdout_roi_pct",
            "holdout_dd_pct", "holdout_passed", "oracle_worst_ratio", "oracle_real_gap_sharpe",
        )}
        best_arch["features_in_winning_config"] = []
    else:
        cid_match = re.search(r"sl(\d+\.\d+)_trail(ON|OFF)_exp(\w+)", str(winner_row["config_id"]))
        sl_val = float(cid_match.group(1)) if cid_match else None
        exp_val = cid_match.group(3) if cid_match else None
        exposure_cap_yaml = (2 if exp_val == "2" else "unlimited") if exp_val else None
        wfo = art["wfo_results"]
        cand_wfo = wfo[wfo["config_id"] == winner_row["config_id"]]
        n_trades_total = int(cand_wfo["n_trades"].sum()) if len(cand_wfo) > 0 else None
        n_folds = int(len(cand_wfo))
        n_pos = int((cand_wfo["roi_pct"] >= 0).sum()) if len(cand_wfo) > 0 else 0
        best_arch = {
            "name": "A1 system_level_filter",
            "cluster": "aggregate",
            "archetype": "Unclassified",
            "config": str(winner_row["config_id"]),
            "sl_atr": sl_val,
            "exit_policy": "trail_atr_2.0_1.5" if "trailON" in str(winner_row["config_id"]) else "sl_only",
            "exposure_cap": exposure_cap_yaml,
            "worst_fold_ratio": float(winner_row.get("search_worst_ratio", 0)),
            "worst_fold_roi_pct": float(winner_row.get("search_worst_roi", 0)) * 100,
            "worst_fold_dd_pct": float(winner_row.get("search_worst_dd", 0)) * 100,
            "mean_fold_ratio": float(winner_row.get("search_mean_ratio", 0)),
            "mean_fold_roi_pct": None,
            "sign_pos_folds": f"{n_pos}/{n_folds}",
            "n_trades_total": n_trades_total,
            "holdout_roi_pct": float(winner_row.get("holdout_roi", 0)) * 100,
            "holdout_dd_pct": float(winner_row.get("holdout_dd", 0)) * 100,
            "holdout_passed": bool(winner_row.get("deployable", False)),
            "oracle_worst_ratio": None,
            "oracle_real_gap_sharpe": None,
            "features_in_winning_config": [],
        }

    payload = {
        "tracker_payload": {
            "arc_name": ARC_NAME,
            "signal": SIGNAL_DESC,
            "tf": TF,
            "sub_protocol": SUB_PROTOCOL,
            "closed_timestamp": _now_iso(),
            "closure_doc_link": f"results/{ARC_NAME}/ARC_CLOSURE.md",
            "verdict": verdict,
            "one_line": f"{verdict_reason} (Arc 5 v3.0 vanilla on h=120 trigger).",
            "failed_at_step": 5 if verdict == "FAIL" else ("N/A" if verdict.startswith("PASS") else 5),
            "primary_failure_mode": fmode,
            "pool_metadata": {
                "total_n": int(len(pool)),
                "window_start": WINDOW_START,
                "window_end": WINDOW_END,
                "kh24_co_fire_pct": 0.0,
                "configs_evaluated_step5": int(s5.get("total_configs", 0)),
                "search_scope_flag": "thin" if int(s5.get("total_configs", 0)) < 50
                                     else ("normal" if int(s5.get("total_configs", 0)) <= 99 else "broad"),
            },
            "best_architecture": best_arch,
            "cost_decomposition": None,
            "clusters": clusters_yaml,
            "architectures_tested": tested_archs,
            "architecture_results": arch_results,
            "archetypes_observed": archetypes_observed,
            "cross_arc_tags": [
                "v3_first_complete_arc",
                "thin_search_scope_24_configs",
                "h120_mtf_alignment_v3_replay",
            ],
        }
    }
    return payload


def _build_section_2_prose(verdict: str, art: dict[str, Any], winner_row: dict | None) -> str:
    pool_n = len(art["pool"])
    cap = art["capturability"]
    cand_n = int(cap["is_candidate"].sum()) if "is_candidate" in cap.columns else 0
    s5 = art["s5_manifest"]
    n_configs = int(s5.get("total_configs", 0))
    archetypes = sorted({_archetype_short(str(r.get("shape_tag", "Unclassified"))) for _, r in cap.iterrows()})
    if verdict.startswith("PASS"):
        proximate = (
            f"Top-1 candidate `{winner_row['config_id']}` cleared §3 gates: worst-fold ratio "
            f"{float(winner_row['search_worst_ratio']):.2f}, holdout ratio "
            f"{float(winner_row['holdout_ratio']):.2f}, holdout DD {float(winner_row['holdout_dd']):.2%}."
        )
    else:
        if winner_row is not None:
            proximate = (
                f"Top-1 candidate `{winner_row['config_id']}` failed §3 gates: worst-fold ratio "
                f"{float(winner_row['search_worst_ratio']):.2f} (need ≥ 2.0), "
                f"holdout ratio {float(winner_row['holdout_ratio']):.2f}, "
                f"search negative folds {int(winner_row['search_negative_folds'])}."
            )
        else:
            proximate = "Step 5 produced no eligible candidates."

    structural = (
        f"Step 1 pool: {pool_n:,} trades over {WINDOW_START}..{WINDOW_END}. "
        f"Step 2 clustering surfaced archetypes: {archetypes}. "
        f"Step 3 flagged {cand_n} candidate cluster(s) under "
        f"(reach_1R ≥ 0.50 ∧ ww_pp ≤ 0.30 ∧ mfe_p50 ≥ 1.5R). "
        f"Step 5 ran {n_configs} A1 configs (thin search per dispatch "
        f"\"Selection-bias accounting\") on the 11-fold 2010-2020 WFO."
    )
    methodology = (
        "Methodology note: v3.0 first arc to land end-to-end on the mtf_alignment_2_down_mixed_kijun "
        "trigger. v2.x history on this trigger (Arc 2 redo KILL at Step 3, Arc 5 v2 SHELVED at Step 6 "
        "under Pipeline D1 admit-vs-deployment failure) is historical context, NOT input to this verdict "
        "per dispatch line 11."
    )
    arch_note = (
        "Architecture scope: Step 5 exercised A1 only (system_level_filter baseline). "
        "A2/A3/A4/A6 require Step 4 classifier wiring + path-so-far classifier training and "
        "are deferred to follow-up arcs once the A1 baseline establishes the pool's deployability ceiling."
    )
    return f"{proximate}\n\n{structural}\n\n{methodology}\n\n{arch_note}"


def _build_section_3_bullets(verdict: str, art: dict[str, Any]) -> list[str]:
    cap = art["capturability"]
    archetypes_short = sorted({_archetype_short(str(r.get("shape_tag", "Unclassified"))) for _, r in cap.iterrows()})
    pool_n = len(art["pool"])
    bullets = [
        f"- **v3.0 first complete arc end-to-end.** Pool size {pool_n:,} trades across 28 FX pairs "
        f"over {WINDOW_START}..{WINDOW_END}; signal port + canonical architectures + WFO orchestrator "
        f"+ holdout one-shot all exercised on real data.",
        f"- **Cluster archetypes observed on h=120 trigger under v3.0:** {archetypes_short}. "
        f"First v3.0 data point on this signal class — feeds the per-archetype recurrence registry.",
        "- **A1 baseline thin-search scope** (24 configs: 6 SL × 2 trail × 2 exposure). Per dispatch "
        "\"Selection-bias accounting\" thin (<50) is flagged. A2/A6 (classifier-based) and A3/A4 "
        "(path-so-far classifier) deferred — establishes minimum-viable Step 5 scope; future arcs "
        "on this trigger can layer classifier architectures.",
    ]
    holdout = art["holdout"]
    if len(holdout) > 0:
        top = holdout.iloc[0]
        srat = float(top.get("search_worst_ratio", 0))
        hrat = float(top.get("holdout_ratio", 0))
        if abs(hrat - srat) > 1.0:
            bullets.append(
                f"- **Holdout-vs-search divergence on top-1:** search worst-fold ratio {srat:.2f} "
                f"vs holdout ratio {hrat:.2f} (gap {hrat - srat:+.2f}). Cross-arc datapoint on the "
                "admit-only vs deployment pattern."
            )
    return bullets


def _render_closure_md(payload: dict[str, Any], section_2: str, section_3_bullets: list[str], verdict: str) -> str:
    yaml_block = _yaml_dump_for_template(payload)
    head_verb = "succeeded" if verdict.startswith("PASS") else "failed"
    if verdict == "HALT":
        head_verb = "halted"
    lines = [
        f"# ARC_5_CLOSURE — {ARC_NAME}",
        "",
        f"> **Closed:** {payload['tracker_payload']['closed_timestamp']}",
        f"> **Branch:** arc/{ARC_NAME}",
        f"> **Closure doc path:** results/{ARC_NAME}/ARC_CLOSURE.md",
        "",
        "---",
        "",
        "## §1 tracker_payload",
        "",
        "```yaml",
        yaml_block.rstrip(),
        "```",
        "",
        "---",
        "",
        f"## §2 Why {head_verb}",
        "",
        section_2,
        "",
        "---",
        "",
        "## §3 Cross-arc observations",
        "",
        *section_3_bullets,
    ]
    return "\n".join(lines) + "\n"


def emit_closure() -> tuple[Path, dict[str, Any], str]:
    art = _read_step_artefacts()
    verdict, verdict_reason, winner_row = _derive_verdict(art)
    payload = _build_tracker_payload(art, verdict, verdict_reason, winner_row)
    section_2 = _build_section_2_prose(verdict, art, winner_row)
    section_3 = _build_section_3_bullets(verdict, art)
    md = _render_closure_md(payload, section_2, section_3, verdict)
    out = RESULTS_DIR / "ARC_CLOSURE.md"
    out.write_text(md, encoding="utf-8", newline="\n")
    return out, payload, verdict


def _read_tracker() -> str:
    return TRACKER_PATH.read_text(encoding="utf-8")


def _write_tracker(content: str) -> None:
    TRACKER_PATH.write_text(content, encoding="utf-8", newline="\n")


def update_tracker(payload: dict[str, Any]) -> None:
    tp = payload["tracker_payload"]
    text = _read_tracker()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = re.sub(r"Last auto-update: .*", f"Last auto-update: manual: {today}", text, count=1)

    # B. Closed arcs summary row
    best = tp["best_architecture"]
    best_name = best.get("name") if best else None
    worst_ratio = best.get("worst_fold_ratio") if best else None
    worst_ratio_str = f"{worst_ratio:.4f}" if isinstance(worst_ratio, (int, float)) else "—"
    closed_row = (
        f"| {tp['arc_name']} | {tp['signal']} | {tp['tf']} | {tp['sub_protocol']} | "
        f"{best_name or '—'} | {worst_ratio_str} | {tp['verdict']} | "
        f"{tp['failed_at_step']} | [{tp['closure_doc_link']}]({tp['closure_doc_link']}) |"
    )
    if "(empty — no arcs closed under v3.0)" in text:
        text = text.replace(
            "| Arc | Signal | TF | Sub-protocol | Best architecture | Worst-fold ratio | Verdict | Failed at step | Closure doc |\n|---|---|---|---|---|---|---|---|---|\n\n(empty — no arcs closed under v3.0)",
            "| Arc | Signal | TF | Sub-protocol | Best architecture | Worst-fold ratio | Verdict | Failed at step | Closure doc |\n|---|---|---|---|---|---|---|---|---|\n" + closed_row,
        )
    else:
        text = re.sub(
            r"(\| Arc \| Signal \| TF \| Sub-protocol \| Best architecture \| Worst-fold ratio \| Verdict \| Failed at step \| Closure doc \|\n\|---\|---\|---\|---\|---\|---\|---\|---\|---\|\n)",
            r"\1" + closed_row + "\n",
            text, count=1,
        )

    # D. Per-architecture win rate
    for arch_name in tp["architectures_tested"]:
        arch_res = tp["architecture_results"].get(arch_name, {})
        won = arch_res.get("won", False)
        arch_long = "A1 system_level_filter" if arch_name == "A1" else arch_name
        pattern = rf"(\| {re.escape(arch_long)} \| )(\d+)( \| )(\d+)( \| )([^|]*?)(\|)"
        def _bump(m):
            tested = int(m.group(2)) + 1
            won_count = int(m.group(4)) + (1 if won else 0)
            avg_cell = m.group(6).strip()
            if won and worst_ratio is not None:
                new_avg = f"{float(worst_ratio):.4f}"
            else:
                new_avg = avg_cell if avg_cell != "—" else "—"
            return f"{m.group(1)}{tested}{m.group(3)}{won_count}{m.group(5)} {new_avg} {m.group(7)}"
        text = re.sub(pattern, _bump, text, count=1)

    # E. Per-archetype recurrence
    archetype_map = {
        "V-shape": "V-shape recovery",
        "Stepwise": "Stepwise climber",
        "Bimodal": "Bimodal",
        "Monotonic_up": "Monotonic up",
        "Monotonic_down": "Monotonic down",
        "Choppy": "Choppy",
        "Unclassified": "Other / unclassified",
    }
    for short in tp["archetypes_observed"]:
        long = archetype_map.get(short, "Other / unclassified")
        clust_in_arch = [c for c in tp["clusters"].values() if c["archetype"] == short]
        mfes = [c["mfe_p50_r"] for c in clust_in_arch]
        reaches = [c["reach_1r"] for c in clust_in_arch]
        avg_mfe = sum(mfes) / len(mfes) if mfes else 0.0
        avg_reach = sum(reaches) / len(reaches) if reaches else 0.0
        pattern = rf"(\| {re.escape(long)} \| )(\d+)( \| )([^|]*?)( \| )([^|]*?)(\|)"
        def _bump_arch(m):
            count = int(m.group(2)) + 1
            return f"{m.group(1)}{count}{m.group(3)} {avg_mfe:.4f} {m.group(5)} {avg_reach:.4f} {m.group(7)}"
        text = re.sub(pattern, _bump_arch, text, count=1)

    # F. Per-failure-mode count
    fmode = tp["primary_failure_mode"]
    pattern = rf"(\| {re.escape(fmode)} \| )(\d+)( \| )([^|]*?)( \| )([^|]*?)(\|)"
    def _bump_fmode(m):
        count = int(m.group(2)) + 1
        return f"{m.group(1)}{count}{m.group(3)} {tp['arc_name']} {m.group(5)} {tp['closed_timestamp'][:10]} {m.group(7)}"
    text = re.sub(pattern, _bump_fmode, text, count=1)

    # G. Cross-arc cluster registry
    cluster_rows = []
    for cid_key, c in tp["clusters"].items():
        cluster_id = f"{tp['arc_name']}.{cid_key}"
        cluster_rows.append(
            f"| {cluster_id} | {c['archetype']} | {c['n']} | {c['mfe_p50_r']:.4f} | "
            f"{c['ww_pp']:.4f} | {c['reach_1r']:.4f} | {c['step3_composite']:.4f} | "
            f"{c['step4_e_auc']!s} | {c['step4_d1_auc']!s} | {c['sl_atr']:.2f} | {c['outcome']} |"
        )
    cluster_block = "\n".join(cluster_rows)
    if "(empty — no clusters logged yet)" in text:
        text = text.replace(
            "| Cluster ID | Archetype | n | mfe_p50_r | ww_pp | reach_1r | step3_composite | step4_e_auc | step4_d1_auc | sl_atr | outcome |\n|---|---|---|---|---|---|---|---|---|---|---|\n\n(empty — no clusters logged yet)",
            "| Cluster ID | Archetype | n | mfe_p50_r | ww_pp | reach_1r | step3_composite | step4_e_auc | step4_d1_auc | sl_atr | outcome |\n|---|---|---|---|---|---|---|---|---|---|---|\n" + cluster_block,
        )
    else:
        text = re.sub(
            r"(\| Cluster ID \| Archetype \| n \| mfe_p50_r \| ww_pp \| reach_1r \| step3_composite \| step4_e_auc \| step4_d1_auc \| sl_atr \| outcome \|\n\|---\|---\|---\|---\|---\|---\|---\|---\|---\|---\|---\|\n)",
            r"\1" + cluster_block + "\n",
            text, count=1,
        )

    # I. Cross-arc tag registry
    tag_rows = []
    for tag in tp["cross_arc_tags"]:
        tag_rows.append(f"| {tag} | 1 | {tp['arc_name']} |")
    tag_block = "\n".join(tag_rows)
    if "(empty — no tags logged yet)" in text:
        text = text.replace(
            "| Tag | Count | Arcs |\n|---|---|---|\n\n(empty — no tags logged yet)",
            "| Tag | Count | Arcs |\n|---|---|---|\n" + tag_block,
        )
    else:
        text = re.sub(
            r"(\| Tag \| Count \| Arcs \|\n\|---\|---\|---\|\n)",
            r"\1" + tag_block + "\n",
            text, count=1,
        )

    _write_tracker(text)


def main() -> int:
    out, payload, verdict = emit_closure()
    update_tracker(payload)
    print(f"Wrote: {out}")
    print(f"Verdict: {verdict}")
    print(f"Tracker: {TRACKER_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
