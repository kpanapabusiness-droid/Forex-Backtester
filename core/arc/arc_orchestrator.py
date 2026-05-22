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
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.steps.step_2_clustering import Step2Result, run_step_2
from core.steps.step_3_capturability import Step3Result, run_step_3
from core.steps.step_4_extraction import Step4Result, run_step_4
from core.wfo.folds import Fold, WfoStructure, build_v3_folds
from core.wfo.orchestrator import (
    WfoSearchResult,
    run_holdout,
    run_search,
)


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
        return run_step_4(
            pool.trades,
            self.cfg.feature_matrix,
            s2.cluster_assignments,
            feature_lineage=self.cfg.feature_lineage,
            candidate_cluster_ids=candidate_ids,
        )

    def _run_step_5(self, signal_eval: SignalEvaluation) -> WfoSearchResult | None:
        override = resolve_step_override(self.cfg.sub_protocol, "step_5")
        if override is not None:
            return override(signal_eval, self.panels)
        if not self.cfg.architectures or not self.cfg.architecture_configs:
            return None
        wfo_struct = self.cfg.wfo_structure or build_v3_folds()
        candidates: list[tuple[str, Any]] = []
        # Pair (architecture, config) tuples; config_id makes them addressable
        for arch, conf in zip(self.cfg.architectures, self.cfg.architecture_configs):
            cid = f"{arch.architecture_name}::{getattr(conf, 'config_id', repr(conf))}"
            candidates.append((cid, (arch, conf)))

        # Wrap each (arch, conf) into a fold-runner adapter
        def _runner(fold: Fold, paired: tuple[Architecture, Any]) -> Any:
            arch, conf = paired
            r = ArcFoldRunner(
                architecture=arch,
                signal_evaluation=signal_eval,
                panels=self.panels,
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
        s5 = self._run_step_5(signal_eval)

        # Holdout: top-K candidates from search re-evaluated on holdout window
        holdout = None
        if s5 is not None and s5.top_k:
            wfo_struct = self.cfg.wfo_structure or build_v3_folds()
            if wfo_struct.holdout is not None:
                def _runner(fold: Fold, paired: tuple[Architecture, Any]):
                    arch, conf = paired
                    r = ArcFoldRunner(
                        architecture=arch,
                        signal_evaluation=signal_eval,
                        panels=self.panels,
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
)
