"""Step 6 input bundle.

Wraps the data + paths that the six audit categories need. Built two ways:

  - :func:`from_arc_orchestrator_result` — auto-dispatch path, live in-memory
    objects from :class:`core.arc.arc_orchestrator.ArcOrchestratorResult`
  - :func:`from_closure_dir` — manual CLI path, reads everything off disk
    from a closure directory (``results/<arc>/``)

Categories check for the presence of optional fields and record an
``info`` severity ``CheckResult`` when a field is missing rather than
raising — manual CLI on an old arc may have a thinner bundle than an
auto-dispatched run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class Step6Inputs:
    """Everything Step 6 categories may consult.

    Required:
      - ``arc_name`` — for manifest + log lines
      - ``arc_root`` — the on-disk results/<arc> dir (manifests, parquet etc.)

    Optional (None when not available for the invocation mode):
      - ``best_candidate_config_id`` — Top-1 verdict-carrier (e.g. ``"A1::cfg_42"``)
      - ``best_candidate_architecture`` — short name ("A1"/"A2"/...)
      - ``best_candidate_features`` — feature names in the winning filter/classifier
      - ``arc_orchestrator_result`` — auto-dispatch path; live object
      - ``closure_payload`` — parsed §1 tracker_payload dict (manual CLI path)
      - ``pool_trades`` — Step 1 trades DataFrame, when loadable
      - ``pool_paths`` — Step 1 forward-path DataFrame, when loadable
      - ``feature_matrix`` — entry-time feature matrix
      - ``feature_lineage`` — feature lineage DataFrame (name + lineage tag)
      - ``signal_module_name`` — fully qualified module path of the signal producer
      - ``primary_tf`` — primary timeframe (e.g. "1H", "4H")
      - ``pair_set`` — pairs in scope
      - ``window_start`` / ``window_end`` — arc window bounds
      - ``r_safe_pct`` / ``r_hard_pct`` — Amendment 3 scaled risk values
      - ``sizing_convention`` — "reset_floor" | "equity_pct"
      - ``configs_evaluated_step5`` — total config count (for §6.2 Bonferroni)
      - ``holdout_start`` — UTC date marking start of holdout window
    """

    arc_name: str
    arc_root: Path

    best_candidate_config_id: str | None = None
    best_candidate_architecture: str | None = None
    best_candidate_features: tuple[str, ...] = ()

    arc_orchestrator_result: Any | None = None
    closure_payload: Mapping[str, Any] | None = None

    pool_trades: pd.DataFrame | None = None
    pool_paths: pd.DataFrame | None = None
    feature_matrix: pd.DataFrame | None = None
    feature_lineage: pd.DataFrame | None = None
    panels: Mapping[str, Any] | None = None

    signal_module_name: str | None = None
    primary_tf: str | None = None
    pair_set: tuple[str, ...] = ()
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None

    r_safe_pct: float | None = None
    r_hard_pct: float | None = None
    sizing_convention: str | None = None

    configs_evaluated_step5: int | None = None
    holdout_start: pd.Timestamp | None = None

    # Per-arc extras (e.g. classifier path, broker spread floor file). Free-form.
    extras: Mapping[str, Any] = field(default_factory=dict)


__all__ = ("Step6Inputs",)
