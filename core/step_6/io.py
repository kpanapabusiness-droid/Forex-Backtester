"""Build :class:`Step6Inputs` from two sources: live orchestrator
result (auto-dispatch) and on-disk closure directory (manual CLI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from core.step_6.inputs import Step6Inputs


def from_arc_orchestrator_result(
    result: Any,
    arc_root: Path,
    *,
    pool_trades: pd.DataFrame | None = None,
    pool_paths: pd.DataFrame | None = None,
    feature_matrix: pd.DataFrame | None = None,
    feature_lineage: pd.DataFrame | None = None,
    panels: Mapping[str, Any] | None = None,
    signal_module_name: str | None = None,
    primary_tf: str | None = None,
    pair_set: tuple[str, ...] = (),
    holdout_start: pd.Timestamp | None = None,
) -> Step6Inputs:
    """Auto-dispatch path — builds from a live :class:`ArcOrchestratorResult`.

    The orchestrator passes the live ``ArcPool`` + its config + Step 4
    output directly so categories don't have to re-read from disk.
    """
    best_arch_name = None
    best_config_id = None
    best_features: tuple[str, ...] = ()
    r_safe_pct = None
    r_hard_pct = None
    sizing_convention = None
    configs_evaluated = None

    amended_wfo = getattr(result, "amended_wfo", None)
    if amended_wfo is not None and amended_wfo.amended_results:
        # Best amended candidate is the one with the highest verdict rank
        # — per arc_orchestrator's existing ranking logic.
        best_amended = amended_wfo.amended_results[0]
        best_config_id = best_amended.config_id
        gate = best_amended.amended_gate
        r_safe_pct = gate.r_safe_pct if gate.scalable_to_safe else None
        r_hard_pct = gate.r_hard_pct if gate.scalable_to_hard else None
        sizing_convention = gate.sizing_convention

    wfo_search = getattr(result, "wfo_search", None)
    if wfo_search is not None:
        configs_evaluated = wfo_search.n_candidates_evaluated
        if wfo_search.top_k:
            top1 = wfo_search.top_k[0]
            best_arch_name = (
                top1.config.architecture_name
                if hasattr(top1.config, "architecture_name")
                else top1.config_id.split("::")[0] if "::" in top1.config_id else None
            )

    # Feature names: prefer Step 4 fitted classifier feature_order if present;
    # otherwise use the full feature_matrix columns minus trade_id.
    step_4 = getattr(result, "step_4", None)
    if step_4 is not None and step_4.per_cluster:
        first_cluster = step_4.per_cluster[0]
        if first_cluster.fitted_classifier_feature_order is not None:
            best_features = first_cluster.fitted_classifier_feature_order
        else:
            best_features = first_cluster.used_features

    if not best_features and feature_matrix is not None:
        best_features = tuple(
            c for c in feature_matrix.columns if c != "trade_id"
        )

    pool = getattr(result, "pool", None)
    if pool_trades is None and pool is not None:
        pool_trades = getattr(pool, "trades", None)
    if pool_paths is None and pool is not None:
        pool_paths = getattr(pool, "paths", None)

    return Step6Inputs(
        arc_name=result.arc_name,
        arc_root=arc_root,
        best_candidate_config_id=best_config_id,
        best_candidate_architecture=best_arch_name,
        best_candidate_features=best_features,
        arc_orchestrator_result=result,
        pool_trades=pool_trades,
        pool_paths=pool_paths,
        feature_matrix=feature_matrix,
        feature_lineage=feature_lineage,
        panels=panels,
        signal_module_name=signal_module_name,
        primary_tf=primary_tf,
        pair_set=pair_set,
        r_safe_pct=r_safe_pct,
        r_hard_pct=r_hard_pct,
        sizing_convention=sizing_convention,
        configs_evaluated_step5=configs_evaluated,
        holdout_start=holdout_start,
    )


def from_closure_dir(closure_dir: Path, *, arc_name: str | None = None) -> Step6Inputs:
    """Manual CLI path — read closure doc + on-disk artefacts.

    Reads ARC_CLOSURE.md's §1 tracker_payload to recover best_architecture
    fields. Loads parquet pool + feature matrix if present. Anything
    missing stays None — category modules check and record info-level
    skips.
    """
    closure_dir = Path(closure_dir)
    if closure_dir.is_file():
        # Caller passed the closure doc path itself; the directory is its parent.
        closure_dir = closure_dir.parent
    if not closure_dir.is_dir():
        raise FileNotFoundError(f"closure dir not found: {closure_dir}")

    payload: Mapping[str, Any] | None = None
    closure_md = closure_dir / "ARC_CLOSURE.md"
    if closure_md.exists():
        payload = _extract_tracker_payload(closure_md.read_text(encoding="utf-8"))

    arc_name = arc_name or (payload or {}).get("arc_name") or closure_dir.name

    best_arch_name = None
    best_config_id = None
    best_features: tuple[str, ...] = ()
    r_safe_pct = None
    r_hard_pct = None
    sizing_convention = None
    configs_evaluated = None
    primary_tf = None
    pair_set: tuple[str, ...] = ()
    window_start = None
    window_end = None
    holdout_start = None

    if payload:
        primary_tf = payload.get("tf")
        pool_meta = payload.get("pool_metadata") or {}
        configs_evaluated = pool_meta.get("configs_evaluated_step5")
        ws = pool_meta.get("window_start")
        we = pool_meta.get("window_end")
        if ws:
            window_start = pd.Timestamp(ws)
        if we:
            window_end = pd.Timestamp(we)
        ba = payload.get("best_architecture") or {}
        if ba:
            best_arch_name = ba.get("name")
            best_config_id = ba.get("config")
            best_features = tuple(ba.get("features_in_winning_config") or ())
            r_safe_pct = ba.get("r_safe_pct")
            r_hard_pct = ba.get("r_hard_pct")
            sizing_convention = ba.get("sizing_convention")

    # On-disk data
    pool_trades = _maybe_read_parquet(closure_dir / "step_1" / "pool.parquet")
    pool_paths = _maybe_read_parquet(closure_dir / "step_1" / "paths.parquet")
    feature_matrix = _maybe_read_parquet(closure_dir / "step_1" / "feature_matrix.parquet")
    feature_lineage = _maybe_read_csv(closure_dir / "step_1" / "feature_lineage.csv")

    # If best_features wasn't in the closure payload (older arcs), fall back
    # to the persisted classifier feature_order.
    if not best_features:
        clf_manifest = closure_dir / "step_4" / "classifiers" / "manifest.json"
        if clf_manifest.exists():
            import json as _json
            data = _json.loads(clf_manifest.read_text(encoding="utf-8"))
            for entry in (data.get("classifiers") or {}).values():
                order = entry.get("feature_order")
                if order:
                    best_features = tuple(order)
                    break

    return Step6Inputs(
        arc_name=arc_name,
        arc_root=closure_dir,
        best_candidate_config_id=best_config_id,
        best_candidate_architecture=best_arch_name,
        best_candidate_features=best_features,
        closure_payload=payload,
        pool_trades=pool_trades,
        pool_paths=pool_paths,
        feature_matrix=feature_matrix,
        feature_lineage=feature_lineage,
        primary_tf=primary_tf,
        pair_set=pair_set,
        window_start=window_start,
        window_end=window_end,
        r_safe_pct=r_safe_pct,
        r_hard_pct=r_hard_pct,
        sizing_convention=sizing_convention,
        configs_evaluated_step5=configs_evaluated,
        holdout_start=holdout_start,
    )


# ── helpers ─────────────────────────────────────────────────────────


def _extract_tracker_payload(text: str) -> Mapping[str, Any] | None:
    """Extract the §1 tracker_payload YAML block from ARC_CLOSURE.md.

    Mirrors :func:`scripts.tracker_parser.extract.extract_payload_from_text`
    but tolerant: returns ``None`` rather than raising on a missing block,
    so manual CLI invocation on older docs degrades gracefully.
    """
    import re

    heading = re.search(r"^## §1 tracker_payload\s*$", text, re.MULTILINE)
    if heading is None:
        return None
    body = text[heading.end():]
    fence = re.search(r"^```yaml\s*$", body, re.MULTILINE)
    if fence is None:
        return None
    yaml_start = fence.end()
    close_idx = body.find("\n```", yaml_start)
    if close_idx == -1:
        return None
    try:
        parsed = yaml.safe_load(body[yaml_start:close_idx])
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    inner = parsed.get("tracker_payload")
    if not isinstance(inner, dict):
        return None
    return inner


def _maybe_read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _maybe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


__all__ = ("from_arc_orchestrator_result", "from_closure_dir")
