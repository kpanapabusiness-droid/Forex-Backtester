"""Section 4 A-K mapping logic: ARC_CLOSURE §1 tracker_payload → ARC_TRACKER mutations.

Side effects:
- Mutates the TrackerState passed in.
- Reads/writes rolling state via rolling_state.json sidecar (see `rolling_state.load_state` / `save_state`).

Public entry point: `apply_payload(state, payload, rolling_state)` runs all sub-mappings A-J
in the order specified by the template. Section K (bad-payload handling) is handled upstream
in schema validation; this module assumes the payload is already valid.

Section 4A — Active arcs: REMOVE row matching arc_name (or warn if absent, per Q-6).
Section 4B — Closed arcs summary: APPEND row (10 cols incl. empty re_evaluated_verdict per Q-4).
Section 4C — Per-feature contribution: rolling-avg arithmetic per feature; verdict re-derived.
Section 4D — Per-architecture win rate: increment tested + won counters.
Section 4E — Per-archetype recurrence: per-cluster data points + arcs-appeared count.
Section 4F — Per-failure-mode count: increment matching row; update recent-example fields.
Section 4G — Cross-arc cluster registry: append one row per cluster, never deduplicate.
Section 4H — Cost-decomposition registry: append if non-null.
Section 4I — Cross-arc tag registry: find-or-create row per tag; append arc_name.
Section 4J — Last auto-update line: set to `parser: <closed_timestamp>` (Q-2).
"""

from __future__ import annotations

import logging
from typing import Any

from scripts.tracker_parser.tracker_io import TrackerState

logger = logging.getLogger(__name__)

EMDASH = "—"  # U+2014 em-dash, matches existing tracker convention


# ── Archetype normalisation ──────────────────────────────────────────────

ARCHETYPE_CANONICAL: dict[str, str] = {
    "v-shape": "V-shape recovery",
    "v_shape": "V-shape recovery",
    "v_shape_recovery": "V-shape recovery",
    "v-shape recovery": "V-shape recovery",
    "v-shape_recovery": "V-shape recovery",
    "v-shape-recovery": "V-shape recovery",
    "stepwise": "Stepwise climber",
    "stepwise_climber": "Stepwise climber",
    "stepwise climber": "Stepwise climber",
    "bimodal": "Bimodal",
    "monotonic_up": "Monotonic up",
    "monotonic up": "Monotonic up",
    "monotonic-up": "Monotonic up",
    "monotonic_down": "Monotonic down",
    "monotonic down": "Monotonic down",
    "monotonic-down": "Monotonic down",
    "choppy": "Choppy",
    "unclassified": "Unclassified",
    "other": "Other / unclassified",
    "other / unclassified": "Other / unclassified",
    "other/unclassified": "Other / unclassified",
}


def normalise_archetype(label: str) -> str:
    key = label.strip().lower()
    if key in ARCHETYPE_CANONICAL:
        return ARCHETYPE_CANONICAL[key]
    raise ValueError(
        f"archetype {label!r} not in normalisation map; canonical labels: "
        f"{sorted(set(ARCHETYPE_CANONICAL.values()))}"
    )


# ── Number formatting helpers ────────────────────────────────────────────


def fmt_or_dash(x: Any, places: int) -> str:
    if x is None:
        return EMDASH
    return f"{float(x):.{places}f}"


def fmt_yaml_repr(x: Any) -> str:
    """Match YAML float repr (Python shortest-repr) — used for Closed arcs summary worst-fold ratio."""
    if x is None:
        return EMDASH
    if isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


# ── Rolling state helpers ────────────────────────────────────────────────


def _fresh_feature_state() -> dict[str, float]:
    return {"n_with": 0, "sum_with": 0.0, "n_without": 0, "sum_without": 0.0}


def _fresh_architecture_state() -> dict[str, float]:
    return {"n_tested": 0, "n_won": 0, "sum_won_ratio": 0.0}


def _fresh_archetype_state() -> dict[str, float]:
    return {"n_arcs": 0, "n_clusters": 0, "sum_mfe": 0.0, "sum_reach": 0.0}


def _fresh_tag_state() -> dict[str, Any]:
    return {"count": 0, "arcs": []}


def _derive_feature_verdict(fstate: dict[str, float]) -> str:
    """LIFTS / NEUTRAL / HURTS / INSUFFICIENT per tracker schema."""
    n_with = int(fstate["n_with"])
    n_without = int(fstate["n_without"])
    if n_with < 3 or n_without == 0:
        return "INSUFFICIENT"
    avg_with = fstate["sum_with"] / n_with
    avg_without = fstate["sum_without"] / n_without
    delta = avg_with - avg_without
    if delta > 0.3:
        return "LIFTS"
    if delta < -0.3:
        return "HURTS"
    return "NEUTRAL"


def _render_feature_row(feature: str, fstate: dict[str, float]) -> list[str]:
    avg_with = (
        f"{fstate['sum_with'] / fstate['n_with']:.3f}" if fstate["n_with"] > 0 else EMDASH
    )
    avg_without = (
        f"{fstate['sum_without'] / fstate['n_without']:.3f}"
        if fstate["n_without"] > 0
        else EMDASH
    )
    verdict = _derive_feature_verdict(fstate)
    return [feature, str(int(fstate["n_with"])), avg_with, avg_without, verdict]


# ── Per-section mutators ─────────────────────────────────────────────────


def apply_active_arcs(state: TrackerState, payload: dict[str, Any]) -> None:
    """Section 4A — remove row matching arc_name; warn+continue if absent (Q-6)."""
    arc_name = payload["arc_name"]
    removed = state.remove_row_by_key("active_arcs", 0, arc_name)
    if not removed:
        logger.warning(
            "WARNING: %s not found in Active arcs — possibly already closed or never registered.",
            arc_name,
        )


def apply_closed_arcs_summary(state: TrackerState, payload: dict[str, Any]) -> None:
    """Section 4B — append 10-column row; column 8 (`Re-evaluated verdict`) is empty per Q-4."""
    ba = payload.get("best_architecture") or {}
    best_arch_name = ba.get("name")
    worst_fold_ratio = ba.get("worst_fold_ratio")
    failed_at = payload.get("failed_at_step")
    cells = [
        payload["arc_name"],
        payload["signal"],
        payload["tf"],
        payload["sub_protocol"],
        best_arch_name if best_arch_name else EMDASH,
        fmt_yaml_repr(worst_fold_ratio),
        payload["verdict"],
        "",  # Re-evaluated verdict — parser does not write (Q-4)
        str(failed_at) if failed_at is not None else "N/A",
        payload["closure_doc_link"],
    ]
    state.append_row("closed_arcs_summary", cells)


def apply_per_feature(
    state: TrackerState, payload: dict[str, Any], rolling: dict[str, Any]
) -> None:
    """Section 4C — rolling-avg per feature; re-derive verdict."""
    ba = payload.get("best_architecture") or {}
    ratio = ba.get("worst_fold_ratio")
    winning = list(ba.get("features_in_winning_config") or [])
    winning_set = set(winning)

    tr = state.tables["per_feature_contribution"]
    existing: dict[str, int] = {}
    for line_idx in tr.data_row_indices():
        cells = state.read_row(line_idx)
        existing[cells[0]] = line_idx

    features_dict = rolling.setdefault("features", {})
    affected: set[str] = set()

    # Step 1: update "with" series for features in winning config
    for feat in winning:
        fstate = features_dict.setdefault(feat, _fresh_feature_state())
        fstate["n_with"] = int(fstate["n_with"]) + 1
        if ratio is not None:
            fstate["sum_with"] = float(fstate["sum_with"]) + float(ratio)
        affected.add(feat)

    # Step 2: update "without" series for existing rows not in winning config
    for feat in existing:
        if feat not in winning_set and ratio is not None:
            fstate = features_dict.setdefault(feat, _fresh_feature_state())
            fstate["n_without"] = int(fstate["n_without"]) + 1
            fstate["sum_without"] = float(fstate["sum_without"]) + float(ratio)
            affected.add(feat)

    # Step 3: render — update existing rows; line indices stable under update_row
    for feat in affected:
        if feat in existing:
            new_cells = _render_feature_row(feat, features_dict[feat])
            state.update_row("per_feature_contribution", existing[feat], new_cells)

    # Step 4: append new feature rows in winning_features order
    for feat in winning:
        if feat not in existing:
            new_cells = _render_feature_row(feat, features_dict[feat])
            state.append_row("per_feature_contribution", new_cells)


def apply_per_architecture(
    state: TrackerState, payload: dict[str, Any], rolling: dict[str, Any]
) -> None:
    """Section 4D — increment n_tested for each architecture tested; n_won + Avg ratio when won if won."""
    archs_tested = payload.get("architectures_tested") or []
    arch_results = payload.get("architecture_results") or {}

    tr = state.tables["per_architecture_win_rate"]
    name_to_idx: dict[str, int] = {}
    short_to_full: dict[str, str] = {}
    for line_idx in tr.data_row_indices():
        cells = state.read_row(line_idx)
        full = cells[0]
        name_to_idx[full] = line_idx
        short = full.split(" ", 1)[0] if " " in full else full
        short_to_full[short] = full

    archs_dict = rolling.setdefault("architectures", {})

    for arch_short in archs_tested:
        if arch_short not in short_to_full:
            raise ValueError(
                f"architecture row for {arch_short!r} not found in tracker — "
                f"known: {sorted(short_to_full)}"
            )
        full = short_to_full[arch_short]
        astate = archs_dict.setdefault(arch_short, _fresh_architecture_state())
        astate["n_tested"] = int(astate["n_tested"]) + 1
        result = arch_results.get(arch_short, {}) or {}
        if result.get("won") is True:
            astate["n_won"] = int(astate["n_won"]) + 1
            r = result.get("worst_fold_ratio")
            if r is not None:
                astate["sum_won_ratio"] = float(astate["sum_won_ratio"]) + float(r)

        avg_when_won = (
            f"{astate['sum_won_ratio'] / astate['n_won']:.3f}"
            if astate["n_won"] > 0
            else EMDASH
        )
        new_cells = [
            full,
            str(int(astate["n_tested"])),
            str(int(astate["n_won"])),
            avg_when_won,
        ]
        state.update_row("per_architecture_win_rate", name_to_idx[full], new_cells)


def apply_per_archetype(
    state: TrackerState, payload: dict[str, Any], rolling: dict[str, Any]
) -> None:
    """Section 4E — per-archetype `Arcs where appeared` + rolling avgs from cluster data points."""
    observed_raw = payload.get("archetypes_observed") or []
    clusters = payload.get("clusters") or {}

    observed_canonical = {normalise_archetype(a) for a in observed_raw}

    cluster_data: dict[str, list[tuple[float, float]]] = {}
    for cid, cluster in clusters.items():
        canon = normalise_archetype(cluster["archetype"])
        cluster_data.setdefault(canon, []).append(
            (float(cluster["mfe_p50_r"]), float(cluster["reach_1r"]))
        )

    tr = state.tables["per_archetype_recurrence"]
    name_to_idx: dict[str, int] = {}
    for line_idx in tr.data_row_indices():
        cells = state.read_row(line_idx)
        name_to_idx[cells[0]] = line_idx

    archetypes_dict = rolling.setdefault("archetypes", {})
    touched = sorted(observed_canonical | set(cluster_data.keys()))

    for canon in touched:
        if canon not in name_to_idx:
            raise ValueError(
                f"per-archetype row {canon!r} not found in tracker — "
                f"existing rows: {sorted(name_to_idx)}; "
                f"add the row manually before re-running (parser does not create archetype rows)"
            )
        astate = archetypes_dict.setdefault(canon, _fresh_archetype_state())
        if canon in observed_canonical:
            astate["n_arcs"] = int(astate["n_arcs"]) + 1
        for mfe, reach in cluster_data.get(canon, []):
            astate["n_clusters"] = int(astate["n_clusters"]) + 1
            astate["sum_mfe"] = float(astate["sum_mfe"]) + mfe
            astate["sum_reach"] = float(astate["sum_reach"]) + reach

        avg_mfe = (
            f"{astate['sum_mfe'] / astate['n_clusters']:.2f}"
            if astate["n_clusters"] > 0
            else EMDASH
        )
        avg_reach = (
            f"{astate['sum_reach'] / astate['n_clusters']:.3f}"
            if astate["n_clusters"] > 0
            else EMDASH
        )
        new_cells = [canon, str(int(astate["n_arcs"])), avg_mfe, avg_reach]
        state.update_row("per_archetype_recurrence", name_to_idx[canon], new_cells)


def apply_per_failure_mode(state: TrackerState, payload: dict[str, Any]) -> None:
    """Section 4F — increment Count; update Recent example arc + date.

    Skip if primary_failure_mode == 'N/A' (PASS arcs).

    Skip if primary_failure_mode is the deferred-Amendment-3 sentinel:
    closure shipped PROVISIONAL with the real failure mode deferred to the
    addendum. The addendum closure will trigger the per-failure-mode update
    at that point (or write N/A on PASS confirmation).
    """
    mode = payload.get("primary_failure_mode")
    if mode == "N/A":
        return
    if mode == "PENDING_AMENDMENT_3_ADDENDUM":
        return

    closed_ts = str(payload["closed_timestamp"])
    closed_date = closed_ts[:10]  # YYYY-MM-DD prefix

    line_idx = state.find_row("per_failure_mode_count", 0, mode)
    if line_idx is None:
        raise ValueError(
            f"per-failure-mode row {mode!r} not found in tracker — "
            f"add the row manually (parser does not create failure-mode rows)"
        )
    cells = state.read_row(line_idx)
    new_count = int(cells[1]) + 1
    new_cells = [mode, str(new_count), payload["arc_name"], closed_date]
    state.update_row("per_failure_mode_count", line_idx, new_cells)


def apply_cluster_registry(state: TrackerState, payload: dict[str, Any]) -> None:
    """Section 4G — append one row per cluster. Never deduplicates.

    Idempotency relies on the closure-doc sha256 registry, not row-level dedup.
    """
    arc_name = payload["arc_name"]
    clusters = payload.get("clusters") or {}
    for cid in sorted(clusters.keys()):
        c = clusters[cid]
        cells = [
            f"{arc_name}.{cid}",
            c["archetype"],  # raw (not normalised) per tracker convention
            str(int(c["n"])),
            fmt_or_dash(c.get("mfe_p50_r"), 4),
            fmt_or_dash(c.get("ww_pp"), 4),
            fmt_or_dash(c.get("reach_1r"), 4),
            fmt_or_dash(c.get("step3_composite"), 4),
            fmt_or_dash(c.get("step4_e_auc"), 4),
            fmt_or_dash(c.get("step4_d1_auc"), 4),
            fmt_or_dash(c.get("sl_atr"), 1),
            c["outcome"],
        ]
        state.append_row("cross_arc_cluster_registry", cells)


def apply_cost_decomp(state: TrackerState, payload: dict[str, Any]) -> None:
    """Section 4H — append if cost_decomposition is non-null."""
    cost = payload.get("cost_decomposition")
    if cost is None:
        return
    admit = cost["admit_pool"]
    reject = cost["reject_pool"]
    early = cost["early_exit_pool"]
    cells = [
        payload["arc_name"],
        fmt_or_dash(admit["n_fraction"], 3),
        fmt_or_dash(admit["mean_r"], 4),
        fmt_or_dash(reject["n_fraction"], 3),
        fmt_or_dash(reject["mean_r"], 4),
        fmt_or_dash(early["n_fraction"], 1),
        fmt_or_dash(early["mean_r"], 1),
    ]
    state.append_row("cost_decomposition_registry", cells)


def apply_cross_arc_tags(
    state: TrackerState, payload: dict[str, Any], rolling: dict[str, Any]
) -> None:
    """Section 4I — find-or-create per tag; increment count; append arc_name to Arcs column."""
    arc_name = payload["arc_name"]
    tags = payload.get("cross_arc_tags") or []

    tags_dict = rolling.setdefault("tags", {})

    for tag in tags:
        line_idx = state.find_row("cross_arc_tag_registry", 0, tag)
        if line_idx is None:
            tags_dict[tag] = {"count": 1, "arcs": [arc_name]}
            state.append_row("cross_arc_tag_registry", [tag, "1", arc_name])
        else:
            cells = state.read_row(line_idx)
            count = int(cells[1]) + 1
            existing_arcs = cells[2] if len(cells) > 2 else ""
            if existing_arcs and existing_arcs != EMDASH:
                new_arcs = existing_arcs + ", " + arc_name
            else:
                new_arcs = arc_name
            tags_dict[tag] = {"count": count, "arcs": new_arcs.split(", ")}
            state.update_row(
                "cross_arc_tag_registry", line_idx, [tag, str(count), new_arcs]
            )


def apply_step_6_audit_registry(state: TrackerState, payload: dict[str, Any]) -> None:
    """Section 4-M (v1.3 / Amendment 4) — append one row per auto-dispatched Step 6 run.

    Append-only. Manual CLI invocations DO NOT add rows (per chat Q6).
    Skips silently when ``payload`` carries no ``step_6`` block (pre-v1.3 closures).
    Skips when ``step_6.ran == false`` (Step 6 didn't dispatch — typical for FAIL closures).
    Skips when ``step_6.trigger == "manual"``.
    """
    if "step_6_audit_registry" not in state.tables:
        # Older trackers without the section — silently no-op.
        return
    step6 = payload.get("step_6")
    if not isinstance(step6, dict):
        return
    if not step6.get("ran"):
        return
    if step6.get("trigger") == "manual":
        return

    arc_name = payload["arc_name"]
    verdict = payload["verdict"]
    overall = step6.get("overall_passed")
    cats = step6.get("categories") or {}
    n_crit = sum(
        1 for v in cats.values() if v is False  # explicit False; null/None means not run
    )
    n_warn = int(step6.get("warnings_count") or 0)
    manifest_path = step6.get("manifest_path") or EMDASH
    cells = [
        arc_name,
        verdict,
        "true" if step6.get("ran") else "false",
        "true" if overall else ("false" if overall is False else EMDASH),
        str(n_crit),
        str(n_warn),
        str(manifest_path),
    ]
    state.append_row("step_6_audit_registry", cells)


def apply_last_auto_update(state: TrackerState, payload: dict[str, Any]) -> None:
    """Section 4J — format `closed_timestamp` as YYYY-MM-DD HH:MM:SS in the Last auto-update line.

    Determinism: timestamp comes from closure's closed_timestamp field (Q-2).
    """
    ts = str(payload["closed_timestamp"])
    formatted = _format_parser_timestamp(ts)
    state.set_last_auto_update(formatted)


def _format_parser_timestamp(closed_ts: str) -> str:
    """Convert ISO 8601 `closed_timestamp` to `YYYY-MM-DD HH:MM:SS` (UTC, no TZ suffix).

    Handles trailing `Z`, fractional seconds, +00:00 offsets.
    """
    s = closed_ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Replace 'T' separator with space for a YYYY-MM-DD HH:MM:SS rendering
    # but we want UTC. Parse strictly.
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Fall back: try parsing as plain date if no time
        dt = datetime.fromisoformat(s + "T00:00:00+00:00")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")


# ── Top-level orchestration ──────────────────────────────────────────────


def apply_payload(
    state: TrackerState, payload: dict[str, Any], rolling: dict[str, Any]
) -> None:
    """Apply all Section 4 mappings in spec order.

    Mutates state and rolling in place.
    """
    apply_active_arcs(state, payload)
    apply_closed_arcs_summary(state, payload)
    apply_per_feature(state, payload, rolling)
    apply_per_architecture(state, payload, rolling)
    apply_per_archetype(state, payload, rolling)
    apply_per_failure_mode(state, payload)
    apply_cluster_registry(state, payload)
    apply_cost_decomp(state, payload)
    apply_cross_arc_tags(state, payload, rolling)
    apply_step_6_audit_registry(state, payload)
    apply_last_auto_update(state, payload)
