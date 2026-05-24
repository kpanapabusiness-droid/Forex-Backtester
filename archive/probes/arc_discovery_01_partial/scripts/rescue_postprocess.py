"""Post-process the rescue_dump.pkl into proper Step-1 artefacts.

Loads the pickle dumped by rescue_log_rows.py and runs the same Bonferroni
accounting + top-K ranking + IO functions that random_search.run_search would
have run had it completed. Writes the five Step-1 artefacts under
``results/arc_discovery_01_partial/step_1/discovery/`` so the data is
preserved alongside (not overwriting) any future full-run output.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from core.discovery.bonferroni import (
    bonferroni_survivors,
    build_bonferroni_report,
    rank_top_k,
)
from core.discovery.io import (
    render_bonferroni_survivors_md,
    render_causal_rejections_md,
    render_compute_budget_used_md,
    render_top_10_raw_md,
    write_full_search_log,
    write_manifest,
)


def main() -> None:
    pkl = Path("rescue_dump.pkl")
    with pkl.open("rb") as fh:
        payload = pickle.load(fh)

    log_rows = payload["log_rows"]
    causal_rejected = payload["causal_rejected"]
    pool_floor_rejected = payload["pool_floor_rejected"]
    specs_by_id = payload["specs_by_id"]
    cfg_summary = payload.get("cfg_summary", {})

    n_evaluated = sum(
        1 for r in log_rows if r.get("pool_floor_pass") and (r.get("pool_size") or 0) > 0
    )
    # We don't know n_generated from the rescue (we stopped at rule 511), so report
    # both: rules-completed (511) AND originally-planned (10000 from cfg_summary).
    rules_completed = len(log_rows)
    rules_planned = cfg_summary.get("n_rules") or 10000
    alpha = cfg_summary.get("alpha") or 0.05

    print(
        f"[postprocess] rescued: {rules_completed} log_rows, "
        f"{len(causal_rejected)} causal_rejected, "
        f"{len(pool_floor_rejected)} pool_floor_rejected, "
        f"{len(specs_by_id)} specs ; n_evaluated={n_evaluated}"
    )

    # Bonferroni report — primary uses N_evaluated (chat decision 3).
    report = build_bonferroni_report(
        n_generated=rules_completed,  # use ACTUAL completed, not planned 10k
        n_evaluated=n_evaluated,
        n_causal_rejected=len(causal_rejected),
        n_pool_floor_rejected=len(pool_floor_rejected),
        alpha=alpha,
    )
    print(
        f"[postprocess] bonferroni threshold (alpha/N_evaluated): "
        f"{report.threshold_primary:.3e}"
    )

    # Inject raw_rank into rows (mirrors random_search.run_search end-of-loop)
    eligible_rows = [r for r in log_rows if r.get("pool_floor_pass") and (r.get("pool_size") or 0) > 0]
    eligible_rows.sort(
        key=lambda r: (
            -float(r["mean_r"]) if r.get("mean_r") is not None else float("inf"),
            float(r.get("p_value") or 1.0),
            int(r["rule_id"]),
        )
    )
    rank_by_id = {int(r["rule_id"]): i + 1 for i, r in enumerate(eligible_rows)}
    for r in log_rows:
        r["raw_rank"] = rank_by_id.get(int(r["rule_id"]))
        # Re-stamp Bonferroni flags using the partial-run report
        p = r.get("p_value")
        if p is None:
            r["bonferroni_pass_primary"] = False
            r["bonferroni_pass_budget"] = False
            continue
        try:
            p_val = float(p)
        except (TypeError, ValueError):
            p_val = float("nan")
        if p_val != p_val:  # NaN
            r["bonferroni_pass_primary"] = False
            r["bonferroni_pass_budget"] = False
            continue
        r["bonferroni_pass_primary"] = bool(
            report.threshold_primary == report.threshold_primary
            and p_val < report.threshold_primary
        )
        r["bonferroni_pass_budget"] = bool(
            report.threshold_budget == report.threshold_budget
            and p_val < report.threshold_budget
        )

    ranked = rank_top_k(
        log_rows,
        k=cfg_summary.get("analysis_top_k") or 10,
        report=report,
        follow_up_top_k=cfg_summary.get("follow_up_top_k") or 3,
    )
    survivors = bonferroni_survivors(log_rows, report)

    out_root = Path("results/arc_discovery_01_partial")
    step1_dir = out_root / "step_1" / "discovery"
    step1_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "step_1" / "manifest.json"

    top_path = step1_dir / "top_10_raw.md"
    bonf_path = step1_dir / "bonferroni_survivors.md"
    log_path = step1_dir / "full_search_log.parquet"
    rej_path = step1_dir / "causal_audit_rejections.md"
    budget_path = step1_dir / "compute_budget_used.md"

    sha_log = write_full_search_log(log_path, log_rows)

    log_df = pd.DataFrame(log_rows)

    top_path.write_bytes(
        render_top_10_raw_md(
            ranked, specs_by_id, report,
            follow_up_top_k=cfg_summary.get("follow_up_top_k") or 3,
        ).encode("utf-8")
    )
    bonf_path.write_bytes(
        render_bonferroni_survivors_md(survivors, log_df, specs_by_id, report).encode("utf-8")
    )
    rej_path.write_bytes(
        render_causal_rejections_md(causal_rejected, rules_completed).encode("utf-8")
    )
    budget_path.write_bytes(
        render_compute_budget_used_md(
            report,
            wall_clock_seconds=132486.0,  # from the rescued log
            n_pairs=28,
            primary_tf="H1",
        ).encode("utf-8")
    )

    # Extra prepend to budget so reader knows this is PARTIAL.
    partial_note = (
        f"> **PARTIAL RUN — rescued via sys.remote_exec after wall-clock budget exceeded.**\n"
        f">\n"
        f"> Rules completed: **{rules_completed} of {rules_planned} planned** "
        f"({100.0 * rules_completed / rules_planned:.1f}%).\n"
        f"> Bonferroni denominator uses RULES-COMPLETED, not the originally-planned 10000.\n"
        f"> Rate observed: ~{rules_completed / (132486.0 / 3600):.1f} rules/hour at 28 pairs × H1 × 2010-2020 IS × no-time-exit.\n"
        f"> This partial output lives at `results/arc_discovery_01_partial/` so it does\n"
        f"> NOT collide with a future full run under amended exit policy at\n"
        f"> `results/arc_discovery_01/`.\n\n"
    )
    budget_path.write_bytes(
        (partial_note + budget_path.read_text(encoding="utf-8")).encode("utf-8")
    )

    write_manifest(
        manifest_path,
        artefact_paths={
            "top_10_raw": top_path,
            "bonferroni_survivors": bonf_path,
            "full_search_log": log_path,
            "causal_audit_rejections": rej_path,
            "compute_budget_used": budget_path,
        },
        extras={
            "run_status": "PARTIAL_RESCUED_VIA_REMOTE_EXEC",
            "rules_completed": rules_completed,
            "rules_planned": rules_planned,
            "n_evaluated": n_evaluated,
            "bonferroni_threshold_primary": report.threshold_primary,
            "wall_clock_seconds": 132486.0,
            "rate_rules_per_hour": rules_completed / (132486.0 / 3600),
            "kill_reason": "rate_too_slow_to_finish_locked_no_time_exit_policy",
        },
    )

    print(f"[postprocess] artefacts written under {step1_dir}")
    print(f"[postprocess]   top_10_raw.md            sha256={_sha(top_path)}")
    print(f"[postprocess]   bonferroni_survivors.md  sha256={_sha(bonf_path)}")
    print(f"[postprocess]   full_search_log.parquet  sha256={sha_log}")
    print(f"[postprocess]   causal_audit_rejections  sha256={_sha(rej_path)}")
    print(f"[postprocess]   compute_budget_used.md   sha256={_sha(budget_path)}")
    print(f"[postprocess]   manifest.json            (see file)")


def _sha(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
