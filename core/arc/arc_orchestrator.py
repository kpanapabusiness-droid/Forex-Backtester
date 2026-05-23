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
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.steps.classifier_persistence import (
    build_a2_config_from_step4,
    build_a6_config_from_step4,
)
from core.steps.step_2_clustering import Step2Result, run_step_2
from core.steps.step_3_capturability import Step3Result, run_step_3
from core.steps.step_4_extraction import Step4Result, run_step_4
from core.wfo.folds import Fold, WfoStructure, build_v3_folds
from core.wfo.orchestrator import (
    WfoSearchResult,
    run_holdout,
    run_search,
)

# Architectures that consume Step 4's persisted classifier through one
# of the named builders. Keyed by ``Architecture.architecture_name``.
# A3 / A4 retrain their own classifier per fold (see L_PROTOCOL §2 Step
# 5 "Architecture-specific retraining policy") so they do not appear
# here.
_AUTO_BUILDERS = {
    "A2": build_a2_config_from_step4,
    "A6": build_a6_config_from_step4,
}


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
    invoke_step_6: bool = False  # set True after PASS-tier candidate detected
    hypothesis: str = ""
    expected_failure_modes: str = ""


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

    def _run_step_5(
        self,
        signal_eval: SignalEvaluation,
        pool: ArcPool,
        s4: Step4Result | None,
    ) -> WfoSearchResult | None:
        override = resolve_step_override(self.cfg.sub_protocol, "step_5")
        if override is not None:
            return override(signal_eval, self.panels)
        # No work if neither explicit configs nor auto specs were supplied
        if not self.cfg.architectures and not self.cfg.auto_arch_specs:
            return None

        wfo_struct = self.cfg.wfo_structure or build_v3_folds()
        candidates: list[tuple[str, Any]] = []

        # Explicit (architecture, config) pairs supplied by the caller
        for arch, conf in zip(self.cfg.architectures, self.cfg.architecture_configs):
            cid = f"{arch.architecture_name}::{getattr(conf, 'config_id', repr(conf))}"
            candidates.append((cid, (arch, conf)))

        # Auto-built configs from Step 4 (A2 / A6)
        for spec in self.cfg.auto_arch_specs:
            if s4 is None:
                raise RuntimeError(
                    "auto_arch_specs supplied but Step 4 did not run "
                    "(no candidate clusters from Step 3, or no "
                    "feature_matrix on ArcConfig)"
                )
            builder = _AUTO_BUILDERS.get(spec.architecture.architecture_name)
            if builder is None:
                raise RuntimeError(
                    f"auto_arch_specs does not support architecture "
                    f"{spec.architecture.architecture_name}; supported: "
                    f"{sorted(_AUTO_BUILDERS)}"
                )
            conf = builder(s4, cluster_id=spec.cluster_id, **dict(spec.builder_kwargs))
            cid = f"{spec.architecture.architecture_name}::{getattr(conf, 'config_id', repr(conf))}"
            candidates.append((cid, (spec.architecture, conf)))

        if not candidates:
            return None

        # Per-trade features for A1 filter rules + A2 / A6 admit gates.
        # A1 / A5 ignore the context; building it once is cheap.
        per_trade_features = None
        if self.cfg.feature_matrix is not None:
            per_trade_features = _build_per_trade_features(
                pool.trades, self.cfg.feature_matrix
            )
        run_context = A1RunContext(per_trade_features=per_trade_features)

        def _runner(fold: Fold, paired: tuple[Architecture, Any]) -> Any:
            arch, conf = paired
            r = ArcFoldRunner(
                architecture=arch,
                signal_evaluation=signal_eval,
                panels=self.panels,
                run_context=run_context,
            )
            return r(fold, conf)

        return run_search(
            wfo_struct,
            candidates,
            fold_runner=_runner,
            min_is_days=365,
            top_k=3,
        )

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
        s5 = self._run_step_5(signal_eval, pool, s4)

        # Holdout: top-K candidates from search re-evaluated on holdout window
        holdout = None
        if s5 is not None and s5.top_k:
            wfo_struct = self.cfg.wfo_structure or build_v3_folds()
            if wfo_struct.holdout is not None:
                per_trade_features = None
                if self.cfg.feature_matrix is not None:
                    per_trade_features = _build_per_trade_features(
                        pool.trades, self.cfg.feature_matrix
                    )
                run_context = A1RunContext(per_trade_features=per_trade_features)

                def _runner(fold: Fold, paired: tuple[Architecture, Any]):
                    arch, conf = paired
                    r = ArcFoldRunner(
                        architecture=arch,
                        signal_evaluation=signal_eval,
                        panels=self.panels,
                        run_context=run_context,
                    )
                    return r(fold, conf)
                holdout = run_holdout(wfo_struct, s5.top_k, fold_runner=_runner)

        # Verdict
        verdict = "INCOMPLETE"
        if s5 is not None and s5.top_k:
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
            "step_6_summary": "(lazy — deferred to chat at PASS verdict)",
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
