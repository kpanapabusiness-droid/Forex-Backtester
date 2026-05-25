"""ArcOrchestrator — runs L_PROTOCOL v3.0 Steps 1→5 sequentially.

Given an arc config + signal module + panels, the orchestrator:

  1. Step 1 via :func:`core.arc.arc_pool_builder.build_arc_pool`
  2. Step 2 via :func:`core.steps.step_2_clustering.run_step_2`
  3. Step 3 via :func:`core.steps.step_3_capturability.run_step_3`
  4. Step 4 via :func:`core.steps.step_4_extraction.run_step_4`
     (only for clusters flagged candidate at Step 3)
  5. Step 5 via :class:`core.runners.arc_fold_runner.ArcFoldRunner`
     across selected architectures × configs
  6. Step 6 invoked conditionally if any Step 5 candidate clears
     PASS-DEPLOYABLE or PASS-VIABLE — at v3.0 Step 6 is a stub that
     reports "audit deferred to chat".

Sub-protocols (per :mod:`core.arc.sub_protocol`) can override any step
by registering a callable. At v3.0 the registry is empty.

The orchestrator drafts an ARC_CLOSURE.md skeleton via
:mod:`core.arc._closure_template`. Final closure prose is chat's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.arc._closure_template import render_arc_closure, render_arc_open
from core.arc.arc_pool_builder import (
    ArcPool,
    ArcPoolConfig,
    build_arc_pool,
    write_arc_pool,
)
from core.arc.signal_protocol import SignalEvaluation, SignalModule
from core.arc.sub_protocol import resolve_step_override
from core.architectures._protocol import Architecture, StrategyResult
from core.architectures.a1_system_level_filter import A1RunContext
from core.runners._fold_stats_helpers import compute_per_day_max_dd
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.step_6.dispatch import (
    DispatchOutcome,
    maybe_dispatch_step_6,
    replace_top_1_with_step6_fail,
)
from core.step_6.manifest import AuditConfig as Step6AuditConfig
from core.steps.classifier_persistence import (
    build_a2_config_from_step4,
    build_a3_config_from_step4,
    build_a4_config_from_step4,
    build_a6_config_from_step4,
)
from core.steps.path_classifier_per_fold import (
    PerFoldTrainingInputs,
    build_path_classifier_fits_per_fold,
    build_per_trade_entry_features,
)
from core.steps.step_2_clustering import Step2Result, run_step_2
from core.steps.step_3_capturability import Step3Result, run_step_3
from core.steps.step_4_extraction import Step4Result, run_step_4
from core.wfo.amended_gates import (
    AmendedGateResult,
    AmendedVerdict,
    classify_amended_fold_stats,
)
from core.wfo.chained_dd import (
    compute_chained_max_dd_from_continuous_equity,
    stitch_per_fold_oos_equity,
)
from core.wfo.folds import Fold, WfoStructure, build_v3_folds
from core.wfo.gates import FoldStats
from core.wfo.holdout_rerun import rescale_arch_config_risk
from core.wfo.orchestrator import (
    CandidateSearchResult,
    WfoSearchResult,
    run_holdout,
    run_search,
)

# Architecture auto-builders keyed by ``Architecture.architecture_name``.
# A2 / A6 load Step 4's persisted classifier from disk and bind it to
# the returned config (no Step 5 retraining). A3 / A4 retrain per fold
# via :mod:`core.steps.path_classifier_per_fold`; their builders here
# return a config with `classifier_fit=None`, and the orchestrator
# threads per-fold fits via ``A1RunContext.path_classifier_fits`` at
# Step 5 dispatch time. See L_PROTOCOL §2 Step 5 "Architecture-specific
# retraining policy".
_AUTO_BUILDERS = {
    "A2": build_a2_config_from_step4,
    "A3": build_a3_config_from_step4,
    "A4": build_a4_config_from_step4,
    "A6": build_a6_config_from_step4,
}

# Architectures that need per-fold path-classifier retraining (their
# fits live in run_context, not in arch_config).
_PER_FOLD_RETRAIN_ARCHS = frozenset({"A3", "A4"})


@dataclass(frozen=True)
class AutoArchSpec:
    """Tells :class:`ArcOrchestrator` to build an architecture config
    from the Step 4 result at Step 5 dispatch time.

    Used for A2 / A6 — the orchestrator looks up ``cluster_id`` in the
    Step 4 result and invokes
    :func:`core.steps.classifier_persistence.build_a2_config_from_step4`
    (or the A6 equivalent) to instantiate the config, loading the
    persisted classifier from disk. ``builder_kwargs`` pass through to
    the builder (e.g. ``threshold_override`` for A2, or
    ``lower_threshold`` / ``upper_threshold`` for A6).
    """

    architecture: Architecture
    cluster_id: int
    builder_kwargs: Mapping[str, Any] = field(default_factory=dict)


def _build_per_trade_features(
    pool_trades: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """Build the (pair, signal_time) -> feature_dict lookup that A1
    filter rules and A2 / A6 classifier admit gates read via
    :class:`A1RunContext`.

    NaN / inf feature values are coerced to 0.0 — the architecture
    layer's admit logic still rejects rows where the coercion would
    mislead a classifier (it checks ``np.all(np.isfinite(row))`` before
    calling ``predict_proba``, see
    :mod:`core.architectures.a2_classifier_filter`). Mirrors the
    convention from ``scripts/l_arc_11/run.py:build_per_trade_features``
    so the orchestrator path produces identical lookups to the canonical
    driver path.
    """
    pool = pool_trades.copy()
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True)
    pool_by_tid = pool.set_index("trade_id")
    out: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    fm = feature_matrix
    if "trade_id" in fm.columns:
        fm = fm.set_index("trade_id")
    for tid, row in fm.iterrows():
        if tid not in pool_by_tid.index:
            continue
        prow = pool_by_tid.loc[tid]
        key = (str(prow["pair"]), pd.Timestamp(prow["signal_time"]))
        out[key] = {
            col: float(row[col]) if pd.notna(row[col]) else 0.0
            for col in row.index
        }
    return out


@dataclass(frozen=True)
class ArcConfig:
    """Top-level arc configuration."""

    arc_name: str
    signal_class: str
    pair_set: tuple[str, ...]
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None
    risk_pct: float = 0.005
    sub_protocol: str | None = None
    sl_atr_mult: float = 2.0
    hold_bars: int = 240
    output_dir: Path | None = None
    feature_matrix: pd.DataFrame | None = None  # Step 1 -> Step 4 feature input
    feature_lineage: pd.DataFrame | None = None
    architectures: tuple[Architecture, ...] = ()
    architecture_configs: tuple[Any, ...] = ()  # 1:1 with architectures
    # A2 / A6 specs that the orchestrator builds from Step 4 at Step 5
    # dispatch time. See :class:`AutoArchSpec`. Empty by default.
    auto_arch_specs: tuple[AutoArchSpec, ...] = ()
    wfo_structure: WfoStructure | None = None  # None -> build_v3_folds()
    # Step 6 auto-dispatch (Amendment 4). Default behavior is to run Step 6
    # automatically on any PASS-tier candidate after the amended gate clears
    # §3 constraints #1-9. ``skip_step_6=True`` is an escape hatch — chat
    # may set it for diagnostic / dev runs that don't need the audit.
    skip_step_6: bool = False
    step_6_audit_config: Step6AuditConfig | None = None
    # Amendment 3 §"Sizing convention" — when any candidate uses
    # ``sizing_convention="equity_pct"``, the scalability gate FAILs
    # by default. Set this to True only after chat approval of a
    # separate scaling treatment.
    accept_equity_pct: bool = False
    hypothesis: str = ""
    expected_failure_modes: str = ""


@dataclass(frozen=True)
class CandidateAmendedResult:
    """Per-candidate Amendment 3 evaluation bundle.

    Extension dataclass per chat directive Q4 — does NOT amend
    :class:`core.wfo.orchestrator.WfoSearchResult` in place.
    Backwards compatibility with pre-Amendment-3 readers preserved.

    ``chained_dd_method`` records HOW the chained equity was
    reconstructed for this candidate:

      - ``"equity_stitching"`` (v3.0.1 default): per-fold OOS equity
        series multiplicatively chained with continuity adjustment.
        Cheap; assumes fold-independence approximately.
      - ``"full_window_sim"`` (v3.0.2 follow-up): a single sim spans
        IS + holdout per top-K candidate, producing true continuous
        equity. Per chat directive Q6 the gold standard; deferred to
        a separate PR.

    Recorded per-candidate so analysts know which reconstruction the
    chained DD came from. Tracker payload includes the same field
    per ``ARC_CLOSURE_TEMPLATE.md`` v1.2 §1 schema (chat decision
    PR-186 review item 1).
    """

    config_id: str
    chained_max_dd_base_pct: float
    chained_dd_method: str   # "equity_stitching" | "full_window_sim"
    per_day_max_dd_artefact_path: Path | None
    amended_gate: AmendedGateResult
    # Holdout trade counts at the three risk tiers (Amendment 4 §6.5
    # risk-leak detection). All three are observed under IDENTICAL
    # signal-evaluation logic — only the sizing scales — so they MUST
    # match. Divergence is a critical engine bug (the 2026-05-25 Arc 7
    # rerun at r=2% vs r=0.5% surfaced exactly this — 36-38% fewer
    # trades under the higher risk). ``None`` when the corresponding
    # holdout sim did not run (scaling not feasible, no holdout fold).
    holdout_n_trades_at_r_base: int | None = None
    holdout_n_trades_at_r_safe: int | None = None
    holdout_n_trades_at_r_hard: int | None = None


@dataclass(frozen=True)
class AmendedWfoSearchResult:
    """Amendment 3 extension of :class:`WfoSearchResult`.

    Holds per-top-K Amendment 3 evaluations alongside the original
    WfoSearchResult so closure writers / verdict logic / tracker
    payloads can read every Amendment 3 field without round-tripping
    through the legacy result type.
    """

    base: WfoSearchResult
    amended_results: tuple[CandidateAmendedResult, ...]


@dataclass
class ArcOrchestratorResult:
    """Complete output of one arc run."""

    arc_name: str
    pool: ArcPool
    step_2: Step2Result
    step_3: Step3Result
    step_4: Step4Result | None
    wfo_search: WfoSearchResult | None
    holdout_results: tuple | None
    arc_open_md: str
    arc_closure_md: str
    verdict: str  # "PASS_DEPLOYABLE" | "PASS_VIABLE" | "FAIL" | "INCOMPLETE"
    raw_strategy_results: tuple[StrategyResult, ...] = field(default_factory=tuple)
    # Amendment 3 extension — None when no top-K candidates exist
    amended_wfo: AmendedWfoSearchResult | None = None
    # Amendment 4 extension — None when Step 6 did not dispatch (no PASS-tier
    # candidate cleared §3 constraints #1-9, or skip_step_6=True).
    step_6_dispatch: DispatchOutcome | None = None


class ArcOrchestrator:
    """The umbrella runner for an arc."""

    def __init__(
        self,
        arc_config: ArcConfig,
        signal_module: SignalModule,
        panels: Mapping[str, Panel],
    ) -> None:
        self.cfg = arc_config
        self.signal_module = signal_module
        self.panels = panels

    # ── per-step helpers (delegate to sub-protocol override if registered) ─

    def _run_step_1(self) -> ArcPool:
        override = resolve_step_override(self.cfg.sub_protocol, "step_1")
        if override is not None:
            return override(self.signal_module, self.panels, self.cfg)
        pool_cfg = ArcPoolConfig(
            arc_name=self.cfg.arc_name,
            sl_atr_mult=self.cfg.sl_atr_mult,
            hold_bars=self.cfg.hold_bars,
            risk_pct=self.cfg.risk_pct,
            window_start=self.cfg.window_start.date() if self.cfg.window_start else None,
            window_end=self.cfg.window_end.date() if self.cfg.window_end else None,
        )
        return build_arc_pool(self.signal_module, self.panels, pool_cfg)

    def _run_step_2(self, pool: ArcPool) -> Step2Result:
        override = resolve_step_override(self.cfg.sub_protocol, "step_2")
        if override is not None:
            return override(pool)
        return run_step_2(pool.trades, pool.paths)

    def _run_step_3(self, pool: ArcPool, s2: Step2Result) -> Step3Result:
        override = resolve_step_override(self.cfg.sub_protocol, "step_3")
        if override is not None:
            return override(pool, s2)
        return run_step_3(
            pool.trades, pool.paths, s2.cluster_assignments,
            declared_sl_mult=self.cfg.sl_atr_mult,
            cluster_centroids=s2.centroids,
        )

    def _resolve_output_dir(self) -> Path:
        """Resolve the on-disk output root used by Step 4 persistence
        and final ``write()``. Falls back to ``results/<arc_name>``."""
        return Path(self.cfg.output_dir or f"results/{self.cfg.arc_name}")

    def _resolve_train_end(self) -> pd.Timestamp | None:
        """Read the holdout-window start from ``cfg.wfo_structure`` and
        return it as the Step 4 ``train_end``. Returns ``None`` when no
        holdout window is configured — Step 4 then sees every trade in
        the pool (the v3.0 pre-fix behavior). Holdout exclusion is a
        chat-locked contract: when a holdout exists, Step 4's CV and
        the persisted classifier MUST NOT see it.
        """
        wfo_struct = self.cfg.wfo_structure or build_v3_folds()
        if wfo_struct.holdout is None:
            return None
        ts = pd.Timestamp(wfo_struct.holdout.oos_start)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts

    def _run_step_4(
        self, pool: ArcPool, s2: Step2Result, s3: Step3Result
    ) -> Step4Result | None:
        override = resolve_step_override(self.cfg.sub_protocol, "step_4")
        if override is not None:
            return override(pool, s2, s3)
        candidate_ids = tuple(c.cluster_id for c in s3.per_cluster if c.is_candidate)
        if not candidate_ids:
            return None
        if self.cfg.feature_matrix is None:
            return None
        # Persistence at Step 4 time so Step 5 auto-builders can load
        # the classifier downstream within the same run() call.
        persistence_dir = self._resolve_output_dir() / "step_4" / "classifiers"
        return run_step_4(
            pool.trades,
            self.cfg.feature_matrix,
            s2.cluster_assignments,
            feature_lineage=self.cfg.feature_lineage,
            candidate_cluster_ids=candidate_ids,
            persistence_dir=persistence_dir,
            arc_name=self.cfg.arc_name,
            train_end=self._resolve_train_end(),
        )

    def _build_candidates_and_contexts(
        self,
        *,
        signal_eval: SignalEvaluation,
        pool: ArcPool,
        s2: Step2Result,
        s4: Step4Result | None,
        wfo_struct: WfoStructure,
    ) -> tuple[
        list[tuple[str, tuple[Architecture, Any]]],
        dict[str, tuple[Architecture, A1RunContext]],
        A1RunContext,
    ]:
        """Build the Step 5 candidate list + per-candidate run_contexts.

        Centralised so that ``_run_step_5`` (search loop) and the
        ``run()`` holdout block share the same A3 / A4 per-fold fits
        (built once for the full ``wfo_struct.folds + (holdout,)`` set).

        Returns ``(candidates, per_candidate_arch, base_ctx)`` where:

          - ``candidates`` is the ``(config_id, (architecture, conf))``
            list passed to :func:`run_search` / :func:`run_holdout`.
          - ``per_candidate_arch`` maps ``config_id ->
            (Architecture, A1RunContext)`` — A3 / A4 specs get specific
            ``path_classifier_fits`` keyed by ``fold_id``.
          - ``base_ctx`` is the A1RunContext used by A1/A2/A5/A6 (and
            any candidate not pre-registered in the per-candidate map).
        """
        candidates: list[tuple[str, tuple[Architecture, Any]]] = []
        per_candidate_arch: dict[str, tuple[Architecture, A1RunContext]] = {}

        per_trade_features = None
        if self.cfg.feature_matrix is not None:
            per_trade_features = _build_per_trade_features(
                pool.trades, self.cfg.feature_matrix
            )
        base_ctx = A1RunContext(per_trade_features=per_trade_features)

        # Lazily build shared entry-features lookup (A3/A4 only).
        shared_entry_features: Mapping[tuple[str, pd.Timestamp], Mapping[str, float]] | None = None

        def _ensure_entry_features() -> Mapping[tuple[str, pd.Timestamp], Mapping[str, float]]:
            nonlocal shared_entry_features
            if shared_entry_features is None:
                inputs = PerFoldTrainingInputs(
                    pool_trades=pool.trades,
                    pool_paths=pool.paths,
                    cluster_assignments=s2.cluster_assignments,
                    panels=self.panels,
                    primary_tf=signal_eval.primary_tf,
                    candidate_cluster_id=None,
                    n_defer=5,
                )
                shared_entry_features = build_per_trade_entry_features(inputs)
            return shared_entry_features

        # Explicit (architecture, config) pairs supplied by the caller.
        for arch, conf in zip(self.cfg.architectures, self.cfg.architecture_configs):
            cid = f"{arch.architecture_name}::{getattr(conf, 'config_id', repr(conf))}"
            candidates.append((cid, (arch, conf)))
            per_candidate_arch[cid] = (arch, base_ctx)

        # Auto-built configs (A2 / A3 / A4 / A6)
        for spec in self.cfg.auto_arch_specs:
            if s4 is None:
                raise RuntimeError(
                    "auto_arch_specs supplied but Step 4 did not run "
                    "(no candidate clusters from Step 3, or no "
                    "feature_matrix on ArcConfig)"
                )
            arch_name = spec.architecture.architecture_name
            builder = _AUTO_BUILDERS.get(arch_name)
            if builder is None:
                raise RuntimeError(
                    f"auto_arch_specs does not support architecture "
                    f"{arch_name}; supported: {sorted(_AUTO_BUILDERS)}"
                )
            conf = builder(s4, cluster_id=spec.cluster_id, **dict(spec.builder_kwargs))
            cid = f"{arch_name}::{getattr(conf, 'config_id', repr(conf))}"
            candidates.append((cid, (spec.architecture, conf)))

            if arch_name in _PER_FOLD_RETRAIN_ARCHS:
                # A3 / A4 — build per-fold fits including the holdout
                # fold so the same map covers both search and holdout
                # paths. Fits keyed by Fold.fold_id.
                folds_for_fits: tuple[Fold, ...] = wfo_struct.folds
                if wfo_struct.holdout is not None:
                    folds_for_fits = folds_for_fits + (wfo_struct.holdout,)
                n_defer = int(getattr(conf, "n_defer", 5))
                inputs = PerFoldTrainingInputs(
                    pool_trades=pool.trades,
                    pool_paths=pool.paths,
                    cluster_assignments=s2.cluster_assignments,
                    panels=self.panels,
                    primary_tf=signal_eval.primary_tf,
                    candidate_cluster_id=int(spec.cluster_id) if arch_name == "A3" else None,
                    n_defer=n_defer,
                )
                fits = build_path_classifier_fits_per_fold(
                    inputs=inputs,
                    folds=folds_for_fits,
                    arch=arch_name,  # type: ignore[arg-type]
                )
                entry_feats = _ensure_entry_features()
                ctx = A1RunContext(
                    per_trade_features=per_trade_features,
                    per_trade_entry_features=entry_feats,
                    path_classifier_fits=fits,
                )
                per_candidate_arch[cid] = (spec.architecture, ctx)
            else:
                per_candidate_arch[cid] = (spec.architecture, base_ctx)

        return candidates, per_candidate_arch, base_ctx

    def _run_step_5(
        self,
        signal_eval: SignalEvaluation,
        pool: ArcPool,
        s2: Step2Result,
        s4: Step4Result | None,
    ) -> WfoSearchResult | None:
        override = resolve_step_override(self.cfg.sub_protocol, "step_5")
        if override is not None:
            return override(signal_eval, self.panels)
        # No work if neither explicit configs nor auto specs were supplied
        if not self.cfg.architectures and not self.cfg.auto_arch_specs:
            return None

        wfo_struct = self.cfg.wfo_structure or build_v3_folds()
        candidates, per_candidate_arch, base_ctx = self._build_candidates_and_contexts(
            signal_eval=signal_eval, pool=pool, s2=s2, s4=s4, wfo_struct=wfo_struct,
        )
        # Stash for the run() holdout block — see ``run()`` below.
        self._last_per_candidate_arch = per_candidate_arch
        self._last_base_ctx = base_ctx
        # Side-channel: collect per-(config_id, fold_id) StrategyResults
        # so the Amendment 3 evaluation can read the per-fold OOS equity
        # series without re-running sims. Keyed by config_id with a
        # nested dict keyed by fold_id.
        self._last_strategy_results: dict[str, dict[int, StrategyResult]] = {}

        if not candidates:
            return None

        def _runner(fold: Fold, paired: tuple[Architecture, Any]) -> Any:
            arch, conf = paired
            cid = f"{arch.architecture_name}::{getattr(conf, 'config_id', repr(conf))}"
            _arch, ctx = per_candidate_arch.get(cid, (arch, base_ctx))
            r = ArcFoldRunner(
                architecture=arch,
                signal_evaluation=signal_eval,
                panels=self.panels,
                run_context=ctx,
            )
            stats = r(fold, conf)
            if r.last_result is not None:
                self._last_strategy_results.setdefault(cid, {})[fold.fold_id] = r.last_result
            return stats

        return run_search(
            wfo_struct,
            candidates,
            fold_runner=_runner,
            min_is_days=365,
            top_k=3,
        )

    # ── Amendment 3 evaluation pass ───────────────────────────────────

    def _run_amendment_3_evaluation(
        self,
        *,
        s5: WfoSearchResult,
        holdout_results: tuple,
        signal_eval: SignalEvaluation,
        wfo_struct: WfoStructure,
    ) -> AmendedWfoSearchResult:
        """Per-top-K candidate Amendment 3 evaluation.

        For each top-K candidate:
          1. Stitch per-fold OOS equity + holdout OOS equity into a
             continuous-equity proxy (per chat directive Q6 the gold
             standard is a full-window sim; this implementation uses
             equity-stitching per the v3.0.1 scope, documented in
             :mod:`core.wfo.chained_dd`).
          2. Compute ``chained_max_dd_base_pct`` from that.
          3. Emit per-day max-DD parquet (UTC broker-day boundary).
          4. Re-run holdout at ``r_safe`` / ``r_hard`` per the candidate's
             scaling factors, if scalable.
          5. Apply :func:`classify_amended_fold_stats`.

        Returns :class:`AmendedWfoSearchResult` containing per-top-K
        :class:`CandidateAmendedResult`. Extension dataclass per chat
        directive Q4 — does NOT amend ``WfoSearchResult`` in place.
        """
        out_dir = self._resolve_output_dir()
        step5_dir = out_dir / "step_5"
        step5_dir.mkdir(parents=True, exist_ok=True)

        per_candidate_arch = getattr(self, "_last_per_candidate_arch", {})
        base_ctx = getattr(self, "_last_base_ctx", A1RunContext())
        strategy_results = getattr(self, "_last_strategy_results", {})

        holdout_by_cid: dict[str, Any] = {}
        for h in holdout_results or ():
            holdout_by_cid[h.config_id] = h

        amended_results: list[CandidateAmendedResult] = []

        for cand in s5.top_k:
            cid = cand.config_id
            # 1. Stitch per-fold OOS equity + holdout OOS equity
            per_fold_equity: list[pd.Series] = []
            for f_stats in cand.fold_stats:
                sr = strategy_results.get(cid, {}).get(f_stats.fold_id)
                if sr is not None and len(sr.equity_curve) > 0:
                    per_fold_equity.append(sr.equity_curve)
            # Holdout OOS equity (one-shot)
            h_match = holdout_by_cid.get(cid)
            if h_match is not None and wfo_struct.holdout is not None:
                # Need to re-execute holdout to capture the StrategyResult's
                # equity — run_holdout currently only returns FoldStats.
                # Optimisation TODO: extend run_holdout to keep equity.
                # For this PR we approximate by running the holdout sim
                # explicitly here per top-K candidate (cheap; only top-K).
                holdout_sr = self._rerun_holdout_capture_equity(
                    cand=cand, fold=wfo_struct.holdout,
                    signal_eval=signal_eval,
                    per_candidate_arch=per_candidate_arch,
                    base_ctx=base_ctx,
                )
                if holdout_sr is not None and len(holdout_sr.equity_curve) > 0:
                    per_fold_equity.append(holdout_sr.equity_curve)

            _arch_unwrapped, arch_config = self._unwrap_cand_config(cand)
            if arch_config is None:
                arch_config = cand.config  # fall back to raw if not a tuple
            starting_balance = float(getattr(arch_config, "starting_balance", 100_000.0))
            chained_equity = stitch_per_fold_oos_equity(
                per_fold_equity, starting_balance=starting_balance,
            )
            chained_dd = compute_chained_max_dd_from_continuous_equity(chained_equity)

            # 3. Per-day max-DD parquet (full IS+holdout trajectory).
            # Amendment 6: boundary follows the engine's Panel convention.
            primary_tf = self.signal_module.primary_tf
            panel_convention = getattr(
                self.panels.get(primary_tf), "boundary_convention", "utc"
            )
            per_day_df = compute_per_day_max_dd(
                chained_equity,
                pair_set=",".join(self.cfg.pair_set),
                boundary_convention=panel_convention,
            )
            parquet_path: Path | None = None
            if not per_day_df.empty:
                # Use cid-safe filename
                safe_cid = cid.replace("::", "__").replace("/", "_")
                parquet_path = step5_dir / f"per_day_max_dd_base__{safe_cid}.parquet"
                per_day_df.to_parquet(
                    parquet_path, engine="pyarrow", compression="snappy", index=False,
                )

            # 4. Re-run holdout at scaled risk(s)
            from core.wfo.amended_gates import compute_scaling_factors
            worst_fold_dd_base = max(
                (f.max_dd_pct for f in cand.fold_stats), default=0.0
            )
            scaling = compute_scaling_factors(worst_fold_dd_base, r_base=self.cfg.risk_pct)
            holdout_safe = None
            holdout_hard = None
            if scaling.scalable_to_safe:
                holdout_safe = self._rerun_holdout_at_scaled_risk(
                    cand=cand, fold=wfo_struct.holdout, k_scale=scaling.k_safe,
                    signal_eval=signal_eval, per_candidate_arch=per_candidate_arch,
                    base_ctx=base_ctx,
                )
            if scaling.scalable_to_hard:
                holdout_hard = self._rerun_holdout_at_scaled_risk(
                    cand=cand, fold=wfo_struct.holdout, k_scale=scaling.k_hard,
                    signal_eval=signal_eval, per_candidate_arch=per_candidate_arch,
                    base_ctx=base_ctx,
                )

            # 5. Classify with amended gate logic
            sizing_convention = str(getattr(arch_config, "sizing_convention", "reset_floor"))
            amended_gate = classify_amended_fold_stats(
                folds=cand.fold_stats,
                chained_max_dd_base_pct=chained_dd,
                per_day_max_dd_df=per_day_df if not per_day_df.empty else None,
                holdout_stats_at_r_safe=holdout_safe,
                holdout_stats_at_r_hard=holdout_hard,
                sizing_convention=sizing_convention,
                accept_equity_pct=self.cfg.accept_equity_pct,
                r_base=self.cfg.risk_pct,
            )

            # §6.5 risk-leak detection inputs: capture trade counts at
            # each scaled-risk holdout tier (Amendment 4 §6.5). Equal
            # counts → admit logic is risk-independent (correct);
            # divergent counts surface engine bugs like the 2026-05-25
            # Arc 7 r=2% vs r=0.5% incident.
            base_holdout_match = holdout_by_cid.get(cid)
            holdout_n_base = (
                int(base_holdout_match.holdout_stats.n_trades)
                if base_holdout_match is not None
                else None
            )
            amended_results.append(CandidateAmendedResult(
                config_id=cid,
                chained_max_dd_base_pct=chained_dd,
                # v3.0.1 reconstructs chained equity via stitching;
                # v3.0.2 follow-up replaces with full_window_sim per
                # chat directive Q6. Recorded per-candidate so
                # downstream analysts know which method produced the
                # chained DD value.
                chained_dd_method="equity_stitching",
                per_day_max_dd_artefact_path=parquet_path,
                amended_gate=amended_gate,
                holdout_n_trades_at_r_base=holdout_n_base,
                holdout_n_trades_at_r_safe=(
                    int(holdout_safe.n_trades) if holdout_safe is not None else None
                ),
                holdout_n_trades_at_r_hard=(
                    int(holdout_hard.n_trades) if holdout_hard is not None else None
                ),
            ))

        return AmendedWfoSearchResult(base=s5, amended_results=tuple(amended_results))

    def _unwrap_cand_config(
        self, cand: CandidateSearchResult,
    ) -> tuple[Architecture, Any] | tuple[None, None]:
        """Unwrap a candidate's ``config`` field.

        The search loop stores candidates as ``(config_id, (arch, conf))``;
        ``CandidateSearchResult.config`` therefore holds the tuple
        ``(architecture, arch_config)``. Returns ``(arch, conf)`` if so;
        ``(None, None)`` if the shape is unexpected.
        """
        cfg = cand.config
        if isinstance(cfg, tuple) and len(cfg) == 2:
            return cfg[0], cfg[1]
        return None, None

    def _rerun_holdout_capture_equity(
        self,
        *,
        cand: CandidateSearchResult,
        fold: Fold,
        signal_eval: SignalEvaluation,
        per_candidate_arch: dict[str, tuple[Architecture, A1RunContext]],
        base_ctx: A1RunContext,
    ) -> StrategyResult | None:
        """Re-run the holdout sim for ``cand`` and return its StrategyResult.

        The existing :func:`run_holdout` only surfaces FoldStats; we
        need the equity series for chained DD + per-day DD emission.
        Cheap because only top-K candidates run through here.
        """
        cid = cand.config_id
        arch, _ctx = per_candidate_arch.get(cid, (None, base_ctx))  # type: ignore[assignment]
        _arch_from_cand, conf = self._unwrap_cand_config(cand)
        if arch is None:
            arch = _arch_from_cand
        if arch is None or conf is None:
            return None
        ctx = per_candidate_arch.get(cid, (arch, base_ctx))[1]
        r = ArcFoldRunner(
            architecture=arch,
            signal_evaluation=signal_eval,
            panels=self.panels,
            run_context=ctx,
        )
        r(fold, conf)
        return r.last_result

    def _rerun_holdout_at_scaled_risk(
        self,
        *,
        cand: CandidateSearchResult,
        fold: Fold,
        k_scale: float,
        signal_eval: SignalEvaluation,
        per_candidate_arch: dict[str, tuple[Architecture, A1RunContext]],
        base_ctx: A1RunContext,
    ) -> "FoldStats | None":
        """Re-run holdout at ``risk_pct * k_scale`` per Amendment 3 §5.3."""
        cid = cand.config_id
        arch_in_map, _ctx = per_candidate_arch.get(cid, (None, base_ctx))  # type: ignore[assignment]
        arch_from_cand, conf = self._unwrap_cand_config(cand)
        arch = arch_in_map or arch_from_cand
        if arch is None or conf is None:
            return None
        try:
            scaled_conf = rescale_arch_config_risk(conf, k_scale=k_scale)
        except (TypeError, ValueError):
            return None
        ctx = per_candidate_arch.get(cid, (arch, base_ctx))[1]
        r = ArcFoldRunner(
            architecture=arch,
            signal_evaluation=signal_eval,
            panels=self.panels,
            run_context=ctx,
        )
        stats = r(fold, scaled_conf)
        return stats

    # ── full run ──────────────────────────────────────────────────────

    def run(self) -> ArcOrchestratorResult:
        opened = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        arc_open_md = render_arc_open({
            "arc_name": self.cfg.arc_name,
            "opened": opened,
            "signal_class": self.cfg.signal_class,
            "signal_definition": self.signal_module.signal_name,
            "tf_mode": "locked",
            "tf": self.signal_module.primary_tf,
            "sub_protocol": self.cfg.sub_protocol or "vanilla",
            "pair_set": ", ".join(self.cfg.pair_set),
            "window_start": str(self.cfg.window_start) if self.cfg.window_start else "open",
            "window_end": str(self.cfg.window_end) if self.cfg.window_end else "open",
            "risk_per_trade": f"{self.cfg.risk_pct:.2%}",
            "hypothesis": self.cfg.hypothesis or "(not specified)",
            "expected_failure_modes": self.cfg.expected_failure_modes or "(not specified)",
        })

        pool = self._run_step_1()
        signal_eval = pool.signal_evaluation
        s2 = self._run_step_2(pool)
        s3 = self._run_step_3(pool, s2)
        s4 = self._run_step_4(pool, s2, s3)
        s5 = self._run_step_5(signal_eval, pool, s2, s4)

        # Holdout: top-K candidates from search re-evaluated on holdout window.
        # Uses the per-candidate context map stashed by _run_step_5 so A3/A4
        # candidates see the same per-fold path-classifier fits at holdout
        # (the holdout's fold_id is already keyed in the fits map per
        # _build_candidates_and_contexts).
        holdout = None
        if s5 is not None and s5.top_k:
            wfo_struct = self.cfg.wfo_structure or build_v3_folds()
            if wfo_struct.holdout is not None:
                per_candidate_arch = getattr(self, "_last_per_candidate_arch", {})
                base_ctx = getattr(self, "_last_base_ctx", A1RunContext())

                def _runner(fold: Fold, paired: tuple[Architecture, Any]):
                    arch, conf = paired
                    cid = f"{arch.architecture_name}::{getattr(conf, 'config_id', repr(conf))}"
                    _arch, ctx = per_candidate_arch.get(cid, (arch, base_ctx))
                    r = ArcFoldRunner(
                        architecture=arch,
                        signal_evaluation=signal_eval,
                        panels=self.panels,
                        run_context=ctx,
                    )
                    return r(fold, conf)
                holdout = run_holdout(wfo_struct, s5.top_k, fold_runner=_runner)

        # Amendment 3 evaluation pass (top-K candidates only)
        amended_wfo: AmendedWfoSearchResult | None = None
        if s5 is not None and s5.top_k:
            wfo_struct_amend = self.cfg.wfo_structure or build_v3_folds()
            amended_wfo = self._run_amendment_3_evaluation(
                s5=s5,
                holdout_results=holdout or (),
                signal_eval=signal_eval,
                wfo_struct=wfo_struct_amend,
            )

        # Amendment 4: Step 6 causal-audit auto-dispatch.
        # Post-gate per chat Q1 — runs only after the amended gate clears
        # §3 constraints #1-9 for at least one candidate.
        step_6_dispatch: DispatchOutcome | None = None
        if (
            amended_wfo is not None
            and amended_wfo.amended_results
            and not self.cfg.skip_step_6
        ):
            arc_root = self._resolve_output_dir()
            arc_root.mkdir(parents=True, exist_ok=True)
            wfo_struct_amend = self.cfg.wfo_structure or build_v3_folds()
            holdout_start = None
            holdout_fold_id = None
            if wfo_struct_amend.holdout is not None:
                holdout_start = pd.Timestamp(wfo_struct_amend.holdout.oos_start)
                if holdout_start.tzinfo is None:
                    holdout_start = holdout_start.tz_localize("UTC")
                holdout_fold_id = int(wfo_struct_amend.holdout.fold_id)
            # Panel boundary convention threaded through for §6.3
            # propagation check (Amendment 6).
            primary_tf_name = signal_eval.primary_tf
            panel_boundary_convention = getattr(
                self.panels.get(primary_tf_name), "boundary_convention", None,
            )
            step_6_dispatch = maybe_dispatch_step_6(
                arc_orchestrator_result=_LightOrchestratorView(
                    arc_name=self.cfg.arc_name,
                    pool=pool, step_4=s4, wfo_search=s5,
                    amended_wfo=amended_wfo,
                ),
                amended_wfo=amended_wfo,
                arc_root=arc_root,
                audit_config=self.cfg.step_6_audit_config,
                holdout_start=holdout_start,
                panels=self.panels,
                feature_matrix=self.cfg.feature_matrix,
                feature_lineage=self.cfg.feature_lineage,
                signal_module_name=type(self.signal_module).__module__,
                primary_tf=primary_tf_name,
                pair_set=tuple(self.cfg.pair_set),
                strategy_results=getattr(self, "_last_strategy_results", None),
                holdout_results=tuple(holdout or ()),
                holdout_fold_id=holdout_fold_id,
                r_base_pct=float(self.cfg.risk_pct),
                panel_boundary_convention=panel_boundary_convention,
            )
            # Per Amendment 4 + chat Q1: if Step 6 critical-failed, downgrade
            # the Top-1 candidate by re-classifying its gate with
            # causal_audit_clean=False.
            if step_6_dispatch.downgrade_top_1:
                amended_wfo = replace_top_1_with_step6_fail(amended_wfo)

        # Verdict — source from Amendment 3 result if present, else legacy.
        verdict = "INCOMPLETE"
        if amended_wfo is not None and amended_wfo.amended_results:
            # Pick the best amended candidate by verdict ranking
            # (PASS-DEPLOYABLE > PASS-VIABLE > FAIL).
            ranked_amended = sorted(
                amended_wfo.amended_results,
                key=lambda r: _amended_verdict_rank(r.amended_gate.verdict),
                reverse=True,
            )
            best_amended = ranked_amended[0]
            verdict = best_amended.amended_gate.verdict.value.upper()
        elif s5 is not None and s5.top_k:
            best = s5.top_k[0]
            verdict = best.gate.verdict.value.upper()

        # Closure doc skeleton
        arch_table = _build_architectures_table(s5)
        arc_closure_md = render_arc_closure({
            "arc_name": self.cfg.arc_name,
            "verdict_headline": f"Verdict: **{verdict}**",
            "best_architecture": _best_architecture_summary(s5),
            "architectures_ranked_table": arch_table,
            "step_1_summary": _step_1_summary(pool),
            "step_2_summary": s2.summary_md,
            "step_3_summary": s3.summary_md,
            "step_4_summary": s4.summary_md if s4 else "(no candidate clusters; Step 4 skipped)",
            "step_5_summary": _step_5_summary(s5),
            "step_6_summary": _step_6_summary(step_6_dispatch),
            "pass_or_failed_phrase": "passed" if "PASS" in verdict else "failed",
            "why_explanation": "(populated by chat in closure prose)",
            "improvements_list": "(populated by chat)",
            "adjacent_ideas": "(populated by chat)",
        })

        return ArcOrchestratorResult(
            arc_name=self.cfg.arc_name,
            pool=pool,
            step_2=s2,
            step_3=s3,
            step_4=s4,
            wfo_search=s5,
            holdout_results=holdout,
            arc_open_md=arc_open_md,
            arc_closure_md=arc_closure_md,
            verdict=verdict,
            amended_wfo=amended_wfo,
            step_6_dispatch=step_6_dispatch,
        )

    def write(self, result: ArcOrchestratorResult, out_dir: Path | None = None) -> Path:
        """Serialise the orchestrator result to disk.

        Layout:
          out_dir/
            ARC_OPEN.md
            ARC_CLOSURE.md
            step_1/  (from write_arc_pool)
            step_2/cluster_assignments.parquet + cluster_summary.md
            step_3/capturability.csv + capturability_summary.md
            step_4/extraction_metrics.csv + feature_importance.csv + extraction_summary.md
        """
        out = Path(out_dir or self.cfg.output_dir or f"results/{self.cfg.arc_name}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "ARC_OPEN.md").write_text(result.arc_open_md, encoding="utf-8", newline="\n")
        (out / "ARC_CLOSURE.md").write_text(result.arc_closure_md, encoding="utf-8", newline="\n")
        write_arc_pool(result.pool, out)
        # Step 2 outputs
        s2_dir = out / "step_2"
        s2_dir.mkdir(exist_ok=True)
        result.step_2.cluster_assignments.to_parquet(
            s2_dir / "cluster_assignments.parquet", engine="pyarrow", compression="snappy", index=False
        )
        result.step_2.cluster_summary.to_csv(
            s2_dir / "cluster_summary.csv", index=False, lineterminator="\n"
        )
        (s2_dir / "cluster_summary.md").write_text(result.step_2.summary_md, encoding="utf-8", newline="\n")
        # Step 3
        s3_dir = out / "step_3"
        s3_dir.mkdir(exist_ok=True)
        result.step_3.capturability_csv.to_csv(
            s3_dir / "capturability.csv", index=False, lineterminator="\n"
        )
        (s3_dir / "capturability_summary.md").write_text(result.step_3.summary_md, encoding="utf-8", newline="\n")
        # Step 4 (optional)
        if result.step_4 is not None:
            s4_dir = out / "step_4"
            s4_dir.mkdir(exist_ok=True)
            result.step_4.extraction_metrics.to_csv(
                s4_dir / "extraction_metrics.csv", index=False, lineterminator="\n"
            )
            result.step_4.feature_importance.to_csv(
                s4_dir / "feature_importance.csv", index=False, lineterminator="\n"
            )
            (s4_dir / "extraction_summary.md").write_text(result.step_4.summary_md, encoding="utf-8", newline="\n")
        return out


# ── markdown helpers ───────────────────────────────────────────────────


def _amended_verdict_rank(v: AmendedVerdict) -> int:
    """Higher value = better verdict, for sorting amended candidates."""
    if v == AmendedVerdict.PASS_DEPLOYABLE:
        return 2
    if v == AmendedVerdict.PASS_VIABLE:
        return 1
    return 0


def _step_1_summary(pool: ArcPool) -> str:
    lines = [f"Pool size: **{len(pool.trades):,}** trades."]
    lines.append(f"Pool sha256: `{pool.pool_sha256[:16]}...`")
    lines.append("")
    lines.append("Integrity:")
    lines.append("")
    lines.append("| Check | Status |")
    lines.append("|---|---|")
    for row in pool.integrity:
        lines.append(f"| {row.check} | {row.status} |")
    return "\n".join(lines)


def _build_architectures_table(s5: WfoSearchResult | None) -> str:
    if s5 is None or not s5.candidates:
        return "(no Step 5 results)"
    lines = ["| Config | Verdict | Worst ratio | Mean ratio | Worst ROI | Worst DD |"]
    lines.append("|---|---|---:|---:|---:|---:|")
    ranked = sorted(
        s5.candidates,
        key=lambda r: (r.gate.worst_fold_ratio, r.gate.mean_fold_ratio),
        reverse=True,
    )
    for r in ranked[:20]:
        g = r.gate
        lines.append(
            f"| {r.config_id} | {g.verdict.value} | {g.worst_fold_ratio:.2f} | "
            f"{g.mean_fold_ratio:.2f} | {g.worst_fold_roi:+.4%} | {g.worst_fold_dd:.4%} |"
        )
    return "\n".join(lines)


def _best_architecture_summary(s5: WfoSearchResult | None) -> str:
    if s5 is None or not s5.top_k:
        return "(no Step 5 results)"
    best = s5.top_k[0]
    return (
        f"**{best.config_id}** — verdict {best.gate.verdict.value}, "
        f"worst ratio {best.gate.worst_fold_ratio:.2f}, "
        f"worst ROI {best.gate.worst_fold_roi:+.4%}, "
        f"worst DD {best.gate.worst_fold_dd:.4%}."
    )


@dataclass(frozen=True)
class _LightOrchestratorView:
    """Minimal duck-type passed to ``core.step_6.io.from_arc_orchestrator_result``.

    The orchestrator hasn't constructed its full ArcOrchestratorResult by the
    time Step 6 dispatches (verdict + closure rendering happen after). This
    shim exposes just the fields the Step 6 io builder reads.
    """

    arc_name: str
    pool: Any
    step_4: Any
    wfo_search: Any
    amended_wfo: Any


def _step_6_summary(dispatch: DispatchOutcome | None) -> str:
    if dispatch is None:
        return "(not dispatched — no PASS-tier candidate cleared §3 #1-9 OR skip_step_6=True)"
    if not dispatch.dispatched:
        return "(not dispatched — no PASS-tier candidate cleared §3 #1-9)"
    res = dispatch.step_6_result
    if res is None:
        return "(dispatched; no result captured)"
    lines = [
        f"Step 6 result: **overall_passed={bool(res.overall_passed)}**",
        f"Trigger: `{res.trigger.value}` · Verdict impact: `{res.verdict_impact.value}`",
        "",
        "| Category | Passed | Critical fails | Warnings |",
        "|---|---|---:|---:|",
    ]
    for c in res.categories:
        lines.append(
            f"| {c.category} | {bool(c.passed)} | "
            f"{c.n_critical_fails}/{c.n_critical} | {c.n_warnings} |"
        )
    if dispatch.divergence_warning:
        lines.append("")
        lines.append("> ⚠ Top-1 feature set differs from other PASS-tier candidates — chat may want to audit them too (per Amendment 4 §Q2).")
    crit = res.critical_failures()
    if crit:
        lines.append("")
        lines.append("**Critical failures:**")
        for name in crit:
            lines.append(f"- `{name}`")
    return "\n".join(lines)


def _step_5_summary(s5: WfoSearchResult | None) -> str:
    if s5 is None:
        return "(Step 5 not run — missing architectures or configs)"
    lines = [
        f"Candidates evaluated: **{s5.n_candidates_evaluated}** "
        f"(structure: {s5.structure_name})",
        "",
        f"Top-{len(s5.top_k)} by worst-fold ratio:",
        "",
    ]
    for k, c in enumerate(s5.top_k, start=1):
        lines.append(f"{k}. {c.config_id} — {c.gate.verdict.value} (ratio {c.gate.worst_fold_ratio:.2f})")
    return "\n".join(lines)


__all__ = (
    "ArcConfig",
    "ArcOrchestrator",
    "ArcOrchestratorResult",
    "AutoArchSpec",
)
