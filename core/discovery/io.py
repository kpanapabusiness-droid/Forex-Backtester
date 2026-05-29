"""IO for arc_discovery_01 — write the five Step-1 discovery artefacts + manifest.

Artefacts per dispatch Override 4 / Task 4-7:

  * ``top_10_raw.md``                — top-10 rules by mean R + Bonferroni flags
  * ``bonferroni_survivors.md``      — rules clearing 0.05 / N_evaluated
  * ``full_search_log.parquet``      — every evaluated rule's row
  * ``causal_audit_rejections.md``   — rules rejected pre-evaluation
  * ``compute_budget_used.md``       — generated vs evaluated vs Bonferroni vs wall-clock
  * ``manifest.json``                — sha256 per artefact for two-run determinism

All file writes are byte-deterministic on identical inputs:
  * Parquet uses ``compression='snappy'`` + sorted columns + sorted rows.
  * Markdown uses LF line endings (``lineterminator='\n'``).
  * JSON uses ``sort_keys=True`` and trailing newline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from core.discovery.bonferroni import BonferroniReport, RankedRule
from core.discovery.grammar import RuleSpec, rule_to_pretty

# Locked parquet column order — DO NOT reorder without bumping the schema
# version reported in compute_budget_used.md.
SEARCH_LOG_COLUMNS: tuple[str, ...] = (
    "rule_id",
    "rule_spec_json",
    "n_atoms",
    "features_used",
    "causal_filter_pass",
    "pool_floor_pass",
    "causal_rejection_reason",
    "pool_size",
    "n_pairs_with_trades",
    "n_trail_activated",
    "mean_r",
    "std_r",
    "sharpe_lo",
    "r_p25",
    "r_p50",
    "r_p75",
    "win_rate",
    "mean_bars_held",
    "t_stat",
    "p_value",
    "raw_rank",
    "bonferroni_pass_primary",
    "bonferroni_pass_budget",
    # arc_discovery_02 Amendments A + C — added 2026-05-24
    "evaluation_timeout",
    "iterations_consumed",
    "time_exit_hit_pct",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_text(path: Path, text: str) -> None:
    """Write LF-terminated UTF-8 text deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always ensure exactly one trailing newline.
    if not text.endswith("\n"):
        text = text + "\n"
    path.write_bytes(text.encode("utf-8"))


def write_full_search_log(path: Path, rows: Sequence[dict]) -> str:
    """Write the parquet log + return its sha256.

    Rows missing schema columns get NaN-filled by pandas; extra columns
    are dropped. Row order: sort by ``rule_id`` ASC (deterministic).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=list(SEARCH_LOG_COLUMNS))
    for c in SEARCH_LOG_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[list(SEARCH_LOG_COLUMNS)]
    df = df.sort_values("rule_id", kind="mergesort").reset_index(drop=True)
    df.to_parquet(path, compression="snappy", index=False)
    return _sha256_file(path)


def render_top_10_raw_md(
    ranked: Sequence[RankedRule],
    specs_by_id: Mapping[int, RuleSpec],
    report: BonferroniReport,
    follow_up_top_k: int,
    arc_name: str = "arc_discovery_01",
    show_time_exit_hit_pct: bool = False,
) -> str:
    """Render the top-10 raw markdown table.

    Chat methodology constraint applied: top-K (default 3) are flagged
    ``follow_up_eligible=true``; ranks beyond that are analysis-only.

    ``show_time_exit_hit_pct``: when True (arc_discovery_02), adds the
    ``Time-exit hit %`` column showing the fraction of each rule's trades
    that exited via the 240-bar time cap.
    """
    lines: list[str] = []
    lines.append(f"# Top-10 raw performers — {arc_name}")
    lines.append("")
    lines.append("> Ranking metric: mean R per rule (dispatch Override 1).")
    lines.append(
        f"> Bonferroni threshold (primary, alpha/N_evaluated): "
        f"{report.threshold_primary:.3e} (N_evaluated={report.n_evaluated})"
    )
    lines.append(
        f"> Bonferroni threshold (budget, alpha/{report.n_generated}): "
        f"{report.threshold_budget:.3e}"
    )
    lines.append(
        f"> Follow-up arcs spawned: top-{follow_up_top_k} only (chat methodology constraint)."
    )
    lines.append(
        f"> Ranks {follow_up_top_k + 1}-{len(ranked)}: analysis-only; do NOT enter the deployment-track pipeline."
    )
    lines.append("")
    if show_time_exit_hit_pct:
        lines.append(
            "| Rank | Rule ID | Mean R | Pool size | p-value | Bonf. primary | Bonf. budget | Time-exit % | Follow-up | Rule spec |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
    else:
        lines.append(
            "| Rank | Rule ID | Mean R | Pool size | p-value | Bonf. primary | Bonf. budget | Follow-up | Rule spec |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in ranked:
        spec = specs_by_id.get(r.rule_id)
        spec_text = rule_to_pretty(spec) if spec is not None else "?"
        primary = "PASS" if r.bonferroni_pass_primary else "fail"
        budget = "PASS" if r.bonferroni_pass_budget else "fail"
        follow_up = "YES (deployment-track)" if r.follow_up_eligible else "no (analysis-only)"
        if show_time_exit_hit_pct:
            te_pct = (
                f"{100.0 * r.time_exit_hit_pct:.1f}%"
                if r.time_exit_hit_pct is not None
                else "—"
            )
            lines.append(
                f"| {r.rank} | {r.rule_id} | {r.mean_r:+.4f} | {r.pool_size} | "
                f"{r.p_value:.3e} | {primary} | {budget} | {te_pct} | {follow_up} | `{spec_text}` |"
            )
        else:
            lines.append(
                f"| {r.rank} | {r.rule_id} | {r.mean_r:+.4f} | {r.pool_size} | "
                f"{r.p_value:.3e} | {primary} | {budget} | {follow_up} | `{spec_text}` |"
            )
    lines.append("")
    return "\n".join(lines)


def render_bonferroni_survivors_md(
    survivor_ids: Sequence[int],
    log_df: pd.DataFrame,
    specs_by_id: Mapping[int, RuleSpec],
    report: BonferroniReport,
    arc_name: str = "arc_discovery_01",
) -> str:
    """Render bonferroni_survivors.md.

    Survivors are ranked by p-value ASC (most-significant first).
    """
    lines: list[str] = []
    lines.append(f"# Bonferroni survivors — {arc_name}")
    lines.append("")
    lines.append(
        f"> Primary threshold (alpha={report.alpha}, denominator=N_evaluated="
        f"{report.n_evaluated}): p < {report.threshold_primary:.3e}"
    )
    lines.append(
        f"> Budget threshold (alpha/{report.n_generated}, transparency only): p < {report.threshold_budget:.3e}"
    )
    lines.append("")
    if not survivor_ids:
        lines.append("**Zero rules cleared the primary Bonferroni threshold.**")
        lines.append("")
        lines.append(
            "This is a coherent result, not a failure. With 10k random rules over a "
            "clean-lineage feature space, the corrected significance bar is "
            "intentionally extremely strict. Top-10 raw performers (see "
            "`top_10_raw.md`) remain available for analysis."
        )
        return "\n".join(lines) + "\n"

    survivors = log_df[log_df["rule_id"].isin(survivor_ids)].copy()
    survivors = survivors.sort_values("p_value", kind="mergesort")
    lines.append(
        "| Rank | Rule ID | Mean R | Pool size | p-value | t-stat | Rule spec |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for i, (_, row) in enumerate(survivors.iterrows(), start=1):
        rule_id = int(row["rule_id"])
        spec = specs_by_id.get(rule_id)
        spec_text = rule_to_pretty(spec) if spec is not None else "?"
        lines.append(
            f"| {i} | {rule_id} | {float(row['mean_r']):+.4f} | "
            f"{int(row['pool_size'])} | {float(row['p_value']):.3e} | "
            f"{float(row['t_stat']):.3f} | `{spec_text}` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_causal_rejections_md(
    rejected: Sequence[dict],
    n_generated: int,
    arc_name: str = "arc_discovery_01",
) -> str:
    """Render causal_audit_rejections.md.

    ``rejected`` rows must carry ``rule_id`` and ``reason``. List
    truncates to first 200 rows (rest summarised by count) to keep the
    file readable.
    """
    lines: list[str] = []
    lines.append(f"# Causal-audit rejections — {arc_name}")
    lines.append("")
    lines.append(
        f"> Total rules generated: {n_generated}; rejected by causal filter: "
        f"{len(rejected)} ({100.0 * len(rejected) / max(n_generated, 1):.2f}%)."
    )
    lines.append("")
    lines.append("Rejection reasons:")
    from collections import Counter

    reasons = Counter(r.get("reason", "unknown") for r in rejected)
    for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{reason}` : {count}")
    lines.append("")
    head = list(rejected[:200])
    if head:
        lines.append("## First 200 rejected rules")
        lines.append("")
        lines.append("| Rule ID | Reason |")
        lines.append("|---|---|")
        for r in head:
            lines.append(f"| {int(r['rule_id'])} | {r.get('reason', '')} |")
        if len(rejected) > 200:
            lines.append("")
            lines.append(f"...and {len(rejected) - 200} more (see full_search_log.parquet).")
    lines.append("")
    return "\n".join(lines)


def render_compute_budget_used_md(
    report: BonferroniReport,
    wall_clock_seconds: float,
    n_pairs: int,
    primary_tf: str,
    schema_version: str = "1.0",
    arc_name: str = "arc_discovery_01",
    n_evaluation_timeouts: int = 0,
    time_exit_hit_pct_summary: dict | None = None,
    halted_at_aggregate_cap: bool = False,
    rules_run: int | None = None,
) -> str:
    """Render compute_budget_used.md per dispatch Task 7.

    arc_discovery_02 additions:
      * ``n_evaluation_timeouts`` — rules that exceeded the per-rule
        bar-iteration cap.
      * ``time_exit_hit_pct_summary`` — dict with mean / p50 / p90 of the
        time-exit-hit-rate distribution across evaluated rules.
      * ``halted_at_aggregate_cap`` — True if the aggregate wall-clock cap
        fired and the search stopped at a rule boundary before completing.
      * ``rules_run`` — count of rules whose evaluation finished. Less than
        ``report.n_generated`` only on aggregate-HALT.
    """
    secs = int(wall_clock_seconds)
    hms = f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
    actual_rules_run = rules_run if rules_run is not None else report.n_generated
    lines = [
        f"# Compute budget used — {arc_name}",
        "",
        f"- Total rules generated: **{report.n_generated}**",
        f"- Rules whose evaluation completed: **{actual_rules_run}**",
        f"- Rejected at causal lineage: **{report.n_causal_rejected}**",
        f"- Rejected at pool size floor: **{report.n_pool_floor_rejected}**",
        f"- Evaluation timeouts (per-rule bar-iter cap): **{n_evaluation_timeouts}**",
        f"- Other / degenerate rejections: **{report.n_other_rejected}**",
        f"- Successfully evaluated: **{report.n_evaluated}**",
        "",
        f"- Bonferroni alpha: {report.alpha}",
        f"- Primary threshold (alpha/N_evaluated): **{report.threshold_primary:.3e}**",
        f"- Budget threshold (alpha/{report.n_generated}): {report.threshold_budget:.3e}",
        "",
        f"- Pairs evaluated: {n_pairs}",
        f"- Primary TF: {primary_tf}",
        f"- Wall-clock time: **{hms}**",
        f"- Halted at aggregate-wallclock cap: **{halted_at_aggregate_cap}**",
        f"- Search-log schema version: {schema_version}",
        "",
    ]
    if time_exit_hit_pct_summary is not None:
        lines.append("## Time-exit hit rate distribution (across evaluated rules)")
        lines.append("")
        for k in ("mean", "p50", "p90", "max"):
            v = time_exit_hit_pct_summary.get(k)
            if v is not None:
                lines.append(f"- {k}: {100.0 * float(v):.1f}%")
        lines.append("")
    return "\n".join(lines)


def write_manifest(
    manifest_path: Path,
    artefact_paths: Mapping[str, Path],
    extras: Mapping[str, object] | None = None,
    arc_name: str = "arc_discovery_01",
) -> str:
    """Write the sha256 manifest sidecar and return its own sha256.

    Manifest schema:

        {
          "arc_name": "arc_discovery_01",
          "step": "step_1/discovery",
          "created_at": "<UTC ISO>",
          "artefacts": {
            "<logical_name>": {"path": "<relative>", "sha256": "<hex>"}
          },
          ...extras
        }
    """
    artefacts: dict[str, dict] = {}
    for logical_name, p in sorted(artefact_paths.items()):
        artefacts[logical_name] = {
            "path": str(p).replace("\\", "/"),
            "sha256": _sha256_file(p),
        }
    payload: dict[str, object] = {
        "arc_name": arc_name,
        "step": "step_1/discovery",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artefacts": artefacts,
    }
    if extras:
        payload.update(extras)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON-safe encoder for date/datetime/Path values that can sneak in via the
    # YAML config (window_start/end are parsed as datetime.date) or other extras.
    # arc_discovery_02 first hit this when the 15h full run died on the final
    # write because pyyaml gave dates not strings. Coerce here so the manifest
    # writer is robust to any future caller-supplied type.
    blob = json.dumps(payload, sort_keys=True, indent=2, default=_json_default)
    if not blob.endswith("\n"):
        blob = blob + "\n"
    manifest_path.write_bytes(blob.encode("utf-8"))
    return _sha256_file(manifest_path)


def _json_default(o):
    """JSON serializer for objects ``json`` doesn't handle natively."""
    from datetime import date, datetime
    if isinstance(o, datetime):
        return o.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o).replace("\\", "/")
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


@dataclass(frozen=True)
class WrittenArtefacts:
    """Sha256 fingerprints of every file written this run."""

    top_10_raw: str
    bonferroni_survivors: str
    full_search_log: str
    causal_audit_rejections: str
    compute_budget_used: str
    manifest: str


__all__ = (
    "SEARCH_LOG_COLUMNS",
    "write_full_search_log",
    "render_top_10_raw_md",
    "render_bonferroni_survivors_md",
    "render_causal_rejections_md",
    "render_compute_budget_used_md",
    "write_manifest",
    "WrittenArtefacts",
)
