"""Random-search orchestration loop — generate 10k rules, evaluate, log.

Sequence per rule:

  1. Generate ``RuleSpec`` from the seeded RNG (in ``grammar``).
  2. Causal-filter check (defensive — the feature_pool already restricted
     to clean lineage, but we re-check to log per-rule rejection
     reasons).
  3. For each pair: compile rule against the pair's feature matrix +
     quantile grid -> boolean trigger mask.
  4. Build trade pool across pairs by simulating each trigger under the
     locked exit policy.
  5. Pool floor: drop the rule if ``pool_size < pool_floor``.
  6. Compute per-rule metrics (mean R, std, Sharpe-Lo, quantiles, t/p).
  7. Append row to the search log.

The loop is single-process / sequential for determinism. Per-rule
evaluation is O(N) over the boolean mask + O(triggers) for sim; with
the feature matrix pre-shared across rules (Option beta, chat
decision 2), the dominant per-rule cost is the trigger-mask compile.

Per-pair feature matrices are computed ONCE before the loop. They are
the largest memory consumers (each ~28 features * 70k bars * 8 bytes
== ~16MB per pair, ~450MB across 28 pairs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from core.discovery.bonferroni import (
    BonferroniReport,
    RankedRule,
    bonferroni_survivors,
    build_bonferroni_report,
    rank_top_k,
)
from core.discovery.causal_filter import check_rule_causal
from core.discovery.grammar import GrammarConfig, RuleSpec, generate_rule_population
from core.discovery.metrics import RuleMetrics, compute_rule_metrics
from core.discovery.pool_simulator import DiscoveryExitConfig, simulate_pair_pool
from core.discovery.quantile_grid import QuantileGrid
from core.discovery.rule_engine import compile_rule


@dataclass(frozen=True)
class PairFixture:
    """One pair's pre-loaded fixtures.

    Memory expectation:
      pair_df  : ~70k rows * ~11 cols * 8 bytes = ~6MB
      features : ~70k rows * ~28 cols * 8 bytes = ~16MB
      atr      : ~70k rows * 8 bytes = ~0.5MB
    Across 28 pairs: ~640MB. Comfortable on a 16GB machine; for tighter
    machines, the search loop can be sharded by pair.
    """

    pair: str
    pair_df: pd.DataFrame
    feature_matrix: pd.DataFrame
    atr_series: pd.Series


@dataclass(frozen=True)
class SearchConfig:
    """Locked search-loop parameters.

    arc_discovery_01 used this without ``iteration_budget_per_rule`` or
    ``total_wallclock_cap_hours`` (Amendment C from arc_discovery_02). Both
    new fields default to ``None`` to preserve _01 backward compatibility.

    arc_discovery_02 Amendment F adds checkpoint params for interruptible
    runs — ``checkpoint_every`` rules a fresh CHECKPOINT.md + incremental
    full_search_log.parquet is written under ``checkpoint_dir`` so a Ctrl-C
    kill leaves the latest checkpoint as the partial deliverable.
    """

    n_rules: int
    random_seed: int
    pool_floor: int
    grammar_cfg: GrammarConfig
    exit_cfg: DiscoveryExitConfig
    follow_up_top_k: int = 3
    analysis_top_k: int = 10
    alpha: float = 0.05
    iteration_budget_per_rule: int | None = None  # Amendment C; None == no cap
    total_wallclock_cap_hours: float | None = None  # rule-boundary HALT (Amendment F: None == disabled)
    checkpoint_every: int = 0                       # Amendment F; 0 == no checkpoints
    checkpoint_dir: Path | None = None
    checkpoint_arc_name: str = ""


@dataclass
class SearchResult:
    """Aggregated result of one full search run.

    ``log_rows`` is the list of search-log row dicts (one per generated
    rule, including rejections). ``specs_by_id`` maps rule_id ->
    RuleSpec for downstream pretty-printing.

    ``halted_at_aggregate_cap`` flags whether the search aborted at the
    rule boundary because the aggregate wall-clock cap fired. ``rules_run``
    is the count of rules whose evaluation completed (passed or failed) —
    less than ``cfg.n_rules`` only when aggregate-HALT fired.
    """

    log_rows: list[dict]
    causal_rejected: list[dict]
    pool_floor_rejected: list[dict]
    bonferroni_report: BonferroniReport
    ranked_top: tuple[RankedRule, ...]
    survivors: tuple[int, ...]
    specs_by_id: dict[int, RuleSpec]
    wall_clock_seconds: float
    halted_at_aggregate_cap: bool = False
    rules_run: int = 0


def run_search(
    fixtures: Sequence[PairFixture],
    grid: QuantileGrid,
    lineage_df: pd.DataFrame,
    cfg: SearchConfig,
    accepted_lineage: Sequence[str] = ("clean",),
    exclude_classes: Sequence[str] = (),
    feature_pool: Sequence[str] | None = None,
    progress_every: int = 500,
    pre_filtered_rules: Sequence[RuleSpec] | None = None,
    extra_pre_eval_rejected: Sequence[dict] | None = None,
) -> SearchResult:
    """Run the full random-search loop.

    Parameters
    ----------
    fixtures, grid, lineage_df, cfg, accepted_lineage, exclude_classes,
    feature_pool, progress_every
        See module docstring. Unchanged from arc_discovery_01.
    pre_filtered_rules
        arc_discovery_02 Amendment D — when provided, skip in-loop generation
        and use this list as the rule set. The caller is responsible for any
        pre-evaluation filtering (e.g., density-band rejection); this loop
        treats them as the canonical rule set.
    extra_pre_eval_rejected
        arc_discovery_02 Amendment D — rows for rules rejected by upstream
        filters (e.g., density filter). Merged into the search log alongside
        the causal-filter rejections so the artefacts stay complete.
    """
    from core.discovery.causal_filter import clean_feature_pool

    if feature_pool is None:
        feature_pool = clean_feature_pool(
            lineage_df, accepted=accepted_lineage, exclude_classes=exclude_classes
        )
    if not feature_pool:
        raise ValueError(
            "Clean-feature pool is empty — causal filter rejected every registered feature."
        )

    if pre_filtered_rules is not None:
        rules = tuple(pre_filtered_rules)
    else:
        rules = generate_rule_population(
            n=cfg.n_rules,
            seed=cfg.random_seed,
            feature_pool=feature_pool,
            cfg=cfg.grammar_cfg,
        )
    specs_by_id: dict[int, RuleSpec] = {r.rule_id: r for r in rules}

    log_rows: list[dict] = []
    causal_rejected: list[dict] = []
    pool_floor_rejected: list[dict] = []
    evaluation_timeouts: list[dict] = []  # Amendment C bookkeeping
    next_trade_id = 0

    aggregate_cap_seconds: float | None = (
        cfg.total_wallclock_cap_hours * 3600.0
        if cfg.total_wallclock_cap_hours is not None
        else None
    )

    t0 = time.perf_counter()
    n_evaluated = 0
    halted_at_aggregate_cap = False
    rules_run = 0

    for spec in rules:
        # Rule-boundary aggregate-wall-clock HALT check (Amendment C / chat decision 4).
        if aggregate_cap_seconds is not None:
            if (time.perf_counter() - t0) >= aggregate_cap_seconds:
                halted_at_aggregate_cap = True
                print(
                    f"[discovery] HALT — aggregate wall-clock cap "
                    f"({cfg.total_wallclock_cap_hours}h) reached at rule "
                    f"{spec.rule_id}. Stopping search at rule boundary.",
                    flush=True,
                )
                break

        if progress_every and (spec.rule_id + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (spec.rule_id + 1) / max(elapsed, 1e-9)
            print(
                f"[discovery] rule {spec.rule_id + 1}/{cfg.n_rules} | "
                f"evaluated={n_evaluated} | "
                f"causal_rej={len(causal_rejected)} | "
                f"pool_floor_rej={len(pool_floor_rejected)} | "
                f"eval_timeout={len(evaluation_timeouts)} | "
                f"elapsed={elapsed:.1f}s ({rate:.1f} rules/s)",
                flush=True,
            )

        # 1. Causal filter (defensive — feature_pool already restricted).
        check = check_rule_causal(
            spec, lineage_df, accepted=accepted_lineage, exclude_classes=exclude_classes
        )
        if not check.passed:
            causal_rejected.append({"rule_id": spec.rule_id, "reason": check.reason})
            log_rows.append(
                _log_row_for_rejection(
                    spec,
                    causal_pass=False,
                    pool_floor_pass=False,
                    causal_reason=check.reason,
                )
            )
            rules_run += 1
            continue

        # 2-4. Compile rule against each pair; collect trades.
        # Track per-rule iteration budget; share across pairs.
        rule_iteration_budget_remaining: int | None = (
            cfg.iteration_budget_per_rule if cfg.iteration_budget_per_rule is not None else None
        )
        rule_iterations_consumed = 0
        pool_trades = []
        rule_aborted = False
        for fx in fixtures:
            try:
                trigger_mask = compile_rule(spec, fx.feature_matrix, grid)
            except KeyError as e:
                # Should not happen post-causal-filter, but trap for diagnostic.
                causal_rejected.append(
                    {"rule_id": spec.rule_id, "reason": f"compile_keyerror: {e}"}
                )
                log_rows.append(
                    _log_row_for_rejection(
                        spec,
                        causal_pass=True,
                        pool_floor_pass=False,
                        causal_reason=f"compile_keyerror: {e}",
                    )
                )
                pool_trades = None
                break
            pair_trades, next_trade_id, pair_iters, pair_aborted = simulate_pair_pool(
                pair=fx.pair,
                pair_df=fx.pair_df,
                trigger_mask=trigger_mask,
                atr_series=fx.atr_series,
                cfg=cfg.exit_cfg,
                next_trade_id=next_trade_id,
                iteration_budget=rule_iteration_budget_remaining,
            )
            rule_iterations_consumed += pair_iters
            if rule_iteration_budget_remaining is not None:
                rule_iteration_budget_remaining = max(
                    0, rule_iteration_budget_remaining - pair_iters
                )
            pool_trades.extend(pair_trades)
            if pair_aborted:
                rule_aborted = True
                break
        if pool_trades is None:
            rules_run += 1
            continue

        if rule_aborted:
            # Amendment C — bar-iteration cap fired; record timeout, discard partial trades.
            evaluation_timeouts.append(
                {
                    "rule_id": spec.rule_id,
                    "iterations_consumed": rule_iterations_consumed,
                    "cap": cfg.iteration_budget_per_rule,
                }
            )
            log_rows.append(
                _log_row_for_timeout(
                    spec, iterations_consumed=rule_iterations_consumed
                )
            )
            rules_run += 1
            continue

        pool_size = len(pool_trades)
        if pool_size < cfg.pool_floor:
            pool_floor_rejected.append(
                {"rule_id": spec.rule_id, "pool_size": pool_size, "floor": cfg.pool_floor}
            )
            log_rows.append(
                _log_row_for_rejection(
                    spec,
                    causal_pass=True,
                    pool_floor_pass=False,
                    causal_reason="",
                    pool_size=pool_size,
                )
            )
            rules_run += 1
            continue

        metrics = compute_rule_metrics(pool_trades)
        log_rows.append(
            _log_row_for_evaluated(
                spec, metrics, iterations_consumed=rule_iterations_consumed
            )
        )
        n_evaluated += 1
        rules_run += 1

        # Amendment F — checkpoint every N rules.
        if (
            cfg.checkpoint_every
            and cfg.checkpoint_dir is not None
            and rules_run % cfg.checkpoint_every == 0
        ):
            _write_checkpoint(
                checkpoint_dir=cfg.checkpoint_dir,
                arc_name=cfg.checkpoint_arc_name or "arc_discovery",
                log_rows=log_rows,
                causal_rejected=causal_rejected,
                pool_floor_rejected=pool_floor_rejected,
                evaluation_timeouts=evaluation_timeouts,
                n_generated=cfg.n_rules,
                n_evaluated=n_evaluated,
                rules_run=rules_run,
                alpha=cfg.alpha,
                follow_up_top_k=cfg.follow_up_top_k,
                analysis_top_k=cfg.analysis_top_k,
                wall_clock_elapsed=time.perf_counter() - t0,
            )

    wall = time.perf_counter() - t0

    # Bonferroni accounting + ranking
    report = build_bonferroni_report(
        n_generated=cfg.n_rules,
        n_evaluated=n_evaluated,
        n_causal_rejected=len(causal_rejected),
        n_pool_floor_rejected=len(pool_floor_rejected),
        alpha=cfg.alpha,
    )

    # Inject raw rank into rows for the parquet log.
    eligible_rows = [r for r in log_rows if r["pool_floor_pass"] and r["pool_size"] > 0]
    eligible_rows.sort(
        key=lambda r: (
            -float(r["mean_r"]) if r["mean_r"] is not None else float("inf"),
            float(r.get("p_value") or 1.0),
            int(r["rule_id"]),
        )
    )
    rank_by_id = {int(r["rule_id"]): i + 1 for i, r in enumerate(eligible_rows)}
    for r in log_rows:
        r["raw_rank"] = rank_by_id.get(int(r["rule_id"]))

    ranked = rank_top_k(
        log_rows, k=cfg.analysis_top_k, report=report, follow_up_top_k=cfg.follow_up_top_k
    )
    survivors = bonferroni_survivors(log_rows, report)

    # Stamp bonferroni flags into log rows (so parquet carries them).
    for r in log_rows:
        p = r.get("p_value")
        if p is None or not _is_finite_float(p):
            r["bonferroni_pass_primary"] = False
            r["bonferroni_pass_budget"] = False
            continue
        r["bonferroni_pass_primary"] = bool(
            _is_finite_float(report.threshold_primary) and float(p) < report.threshold_primary
        )
        r["bonferroni_pass_budget"] = bool(
            _is_finite_float(report.threshold_budget) and float(p) < report.threshold_budget
        )

    return SearchResult(
        log_rows=log_rows,
        causal_rejected=causal_rejected,
        pool_floor_rejected=pool_floor_rejected,
        bonferroni_report=report,
        ranked_top=ranked,
        survivors=survivors,
        specs_by_id=specs_by_id,
        wall_clock_seconds=wall,
        halted_at_aggregate_cap=halted_at_aggregate_cap,
        rules_run=rules_run,
    )


# ── helpers ────────────────────────────────────────────────────────────


def _log_row_for_rejection(
    spec: RuleSpec,
    *,
    causal_pass: bool,
    pool_floor_pass: bool,
    causal_reason: str,
    pool_size: int = 0,
) -> dict:
    return {
        "rule_id": int(spec.rule_id),
        "rule_spec_json": spec.to_json(),
        "n_atoms": int(spec.n_atoms),
        "features_used": ",".join(spec.features_used()),
        "causal_filter_pass": bool(causal_pass),
        "pool_floor_pass": bool(pool_floor_pass),
        "causal_rejection_reason": causal_reason,
        "pool_size": int(pool_size),
        "n_pairs_with_trades": 0,
        "n_trail_activated": 0,
        "mean_r": None,
        "std_r": None,
        "sharpe_lo": None,
        "r_p25": None,
        "r_p50": None,
        "r_p75": None,
        "win_rate": None,
        "mean_bars_held": None,
        "t_stat": None,
        "p_value": None,
        "raw_rank": None,
        "bonferroni_pass_primary": False,
        "bonferroni_pass_budget": False,
        # Amendment C — these flags are False for non-timeout rejections.
        "evaluation_timeout": False,
        "iterations_consumed": 0,
        "time_exit_hit_pct": None,
    }


def _log_row_for_timeout(spec: RuleSpec, *, iterations_consumed: int) -> dict:
    """Row for a rule that exceeded the per-rule iteration budget (Amendment C)."""
    return {
        "rule_id": int(spec.rule_id),
        "rule_spec_json": spec.to_json(),
        "n_atoms": int(spec.n_atoms),
        "features_used": ",".join(spec.features_used()),
        "causal_filter_pass": True,
        "pool_floor_pass": False,
        "causal_rejection_reason": "",
        "pool_size": 0,
        "n_pairs_with_trades": 0,
        "n_trail_activated": 0,
        "mean_r": None,
        "std_r": None,
        "sharpe_lo": None,
        "r_p25": None,
        "r_p50": None,
        "r_p75": None,
        "win_rate": None,
        "mean_bars_held": None,
        "t_stat": None,
        "p_value": None,
        "raw_rank": None,
        "bonferroni_pass_primary": False,
        "bonferroni_pass_budget": False,
        "evaluation_timeout": True,
        "iterations_consumed": int(iterations_consumed),
        "time_exit_hit_pct": None,
    }


def _log_row_for_evaluated(
    spec: RuleSpec, m: RuleMetrics, *, iterations_consumed: int = 0
) -> dict:
    def _safe(v: float) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        return f

    return {
        "rule_id": int(spec.rule_id),
        "rule_spec_json": spec.to_json(),
        "n_atoms": int(spec.n_atoms),
        "features_used": ",".join(spec.features_used()),
        "causal_filter_pass": True,
        "pool_floor_pass": True,
        "causal_rejection_reason": "",
        "pool_size": int(m.pool_size),
        "n_pairs_with_trades": int(m.n_pairs_with_trades),
        "n_trail_activated": int(m.n_trail_activated),
        "mean_r": _safe(m.mean_r),
        "std_r": _safe(m.std_r),
        "sharpe_lo": _safe(m.sharpe_lo),
        "r_p25": _safe(m.r_p25),
        "r_p50": _safe(m.r_p50),
        "r_p75": _safe(m.r_p75),
        "win_rate": _safe(m.win_rate),
        "mean_bars_held": _safe(m.mean_bars_held),
        "t_stat": _safe(m.t_stat),
        "p_value": _safe(m.p_value),
        "raw_rank": None,        # filled in post-loop
        "bonferroni_pass_primary": False,  # filled in post-loop
        "bonferroni_pass_budget": False,
        # Amendment C — successful evaluation; timeout flag stays False.
        "evaluation_timeout": False,
        "iterations_consumed": int(iterations_consumed),
        "time_exit_hit_pct": _safe(m.time_exit_hit_pct),
    }


def _is_finite_float(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _write_checkpoint(
    *,
    checkpoint_dir: Path,
    arc_name: str,
    log_rows: Sequence[dict],
    causal_rejected: Sequence[dict],
    pool_floor_rejected: Sequence[dict],
    evaluation_timeouts: Sequence[dict],
    n_generated: int,
    n_evaluated: int,
    rules_run: int,
    alpha: float,
    follow_up_top_k: int,
    analysis_top_k: int,
    wall_clock_elapsed: float,
) -> None:
    """Amendment F — periodic checkpoint flush.

    Writes:
      * ``full_search_log.parquet`` — incremental snapshot (atomic via tmp + rename)
      * ``CHECKPOINT.md`` — current top-K + Bonferroni survivors + counts

    Best-effort: failures are caught and printed; the main search loop is
    never blocked by checkpoint IO errors.
    """
    from core.discovery.io import (
        render_bonferroni_survivors_md,
        render_top_10_raw_md,
        write_full_search_log,
    )

    try:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path = checkpoint_dir / "full_search_log.parquet"

        # Atomic-ish write: pandas to_parquet to a tmp path then replace.
        tmp_path = log_path.with_suffix(".parquet.tmp")
        sha_log = write_full_search_log(tmp_path, list(log_rows))
        tmp_path.replace(log_path)

        # Build a report + rank for the CHECKPOINT.md
        report = build_bonferroni_report(
            n_generated=n_generated,
            n_evaluated=n_evaluated,
            n_causal_rejected=len(causal_rejected),
            n_pool_floor_rejected=len(pool_floor_rejected),
            alpha=alpha,
        )

        # Inject raw_rank into rows for ranking (same logic as end-of-search).
        eligible = [
            r for r in log_rows
            if r.get("pool_floor_pass") and (r.get("pool_size") or 0) > 0
        ]
        eligible_sorted = sorted(
            eligible,
            key=lambda r: (
                -float(r["mean_r"]) if r.get("mean_r") is not None else float("inf"),
                float(r.get("p_value") or 1.0),
                int(r["rule_id"]),
            ),
        )
        rank_by_id = {int(r["rule_id"]): i + 1 for i, r in enumerate(eligible_sorted)}

        # Build a synthetic rows list with bonferroni flags + rank stamped in,
        # WITHOUT mutating the loop's log_rows (the post-loop final pass does that).
        rows_for_rank: list[dict] = []
        for r in log_rows:
            rcopy = dict(r)
            rcopy["raw_rank"] = rank_by_id.get(int(r["rule_id"]))
            p_val = r.get("p_value")
            if p_val is None or not _is_finite_float(p_val):
                rcopy["bonferroni_pass_primary"] = False
                rcopy["bonferroni_pass_budget"] = False
            else:
                rcopy["bonferroni_pass_primary"] = bool(
                    _is_finite_float(report.threshold_primary)
                    and float(p_val) < report.threshold_primary
                )
                rcopy["bonferroni_pass_budget"] = bool(
                    _is_finite_float(report.threshold_budget)
                    and float(p_val) < report.threshold_budget
                )
            rows_for_rank.append(rcopy)

        ranked = rank_top_k(
            rows_for_rank, k=analysis_top_k,
            report=report, follow_up_top_k=follow_up_top_k,
        )
        survivors = bonferroni_survivors(rows_for_rank, report)

        # Build specs_by_id from the JSON column (best effort — preserves rule lookup).
        specs_by_id: dict[int, RuleSpec] = {}
        for r in rows_for_rank:
            try:
                rid = int(r["rule_id"])
            except (KeyError, TypeError, ValueError):
                continue
            # Skip — we don't have the RuleSpec in the row, only the JSON spec.
            # The top-K renderer falls back to "?" if missing; this is fine
            # for a checkpoint preview (chat reads it knowing it's interim).
            _ = rid

        # Render preview markdown.
        top_md = render_top_10_raw_md(
            ranked, specs_by_id, report,
            follow_up_top_k=follow_up_top_k,
            arc_name=arc_name,
            show_time_exit_hit_pct=True,
        )
        log_df = pd.DataFrame(rows_for_rank)
        surv_md = render_bonferroni_survivors_md(
            survivors, log_df, specs_by_id, report, arc_name=arc_name,
        )

        secs = int(wall_clock_elapsed)
        hms = f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"

        checkpoint_md = (
            f"# {arc_name} — CHECKPOINT\n"
            f"\n"
            f"> **Status:** in-progress checkpoint flush. Overwritten every "
            f"{rules_run}-th rule. If the run is interrupted, this is the latest "
            f"partial deliverable.\n"
            f"\n"
            f"## State\n"
            f"\n"
            f"- Checkpoint timestamp (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f"- Wall-clock elapsed: {hms}\n"
            f"- Rules planned: {n_generated}\n"
            f"- Rules completed (rules_run): {rules_run}\n"
            f"- Rules evaluated (cleared all gates): {n_evaluated}\n"
            f"- Causal-filter rejected: {len(causal_rejected)}\n"
            f"- Pool-floor rejected: {len(pool_floor_rejected)}\n"
            f"- Iter-cap timeouts: {len(evaluation_timeouts)}\n"
            f"\n"
            f"## Bonferroni (at checkpoint)\n"
            f"\n"
            f"- alpha = {alpha}\n"
            f"- N_evaluated = {n_evaluated}\n"
            f"- threshold_primary = alpha / N_evaluated = "
            f"{report.threshold_primary:.3e}\n"
            f"- threshold for FULL run (alpha / 10000) = 5.000e-06 (for reference)\n"
            f"- survivors at this checkpoint: {len(survivors)}\n"
            f"\n"
            f"---\n"
            f"\n"
            f"{top_md}\n"
            f"\n"
            f"---\n"
            f"\n"
            f"{surv_md}\n"
        )
        cp_tmp = checkpoint_dir / "CHECKPOINT.md.tmp"
        cp_path = checkpoint_dir / "CHECKPOINT.md"
        cp_tmp.write_bytes(checkpoint_md.encode("utf-8"))
        cp_tmp.replace(cp_path)

        print(
            f"[checkpoint] rules_run={rules_run} n_evaluated={n_evaluated} "
            f"survivors={len(survivors)} log_sha={sha_log[:12]}",
            flush=True,
        )
    except Exception as exc:
        print(f"[checkpoint] WARNING — checkpoint flush failed: {exc!r}", flush=True)


__all__ = (
    "PairFixture",
    "SearchConfig",
    "SearchResult",
    "run_search",
)
