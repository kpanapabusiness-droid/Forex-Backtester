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
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from core.discovery.bonferroni import (
    BonferroniReport,
    RankedRule,
    bonferroni_survivors,
    build_bonferroni_report,
    rank_top_k,
)
from core.discovery.causal_filter import CausalCheckResult, check_rule_causal
from core.discovery.grammar import GrammarConfig, RuleSpec, generate_rule_population
from core.discovery.metrics import RuleMetrics, compute_rule_metrics, empty_metrics
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
    """Locked search-loop parameters from configs/arc_discovery_01.yaml."""

    n_rules: int
    random_seed: int
    pool_floor: int
    grammar_cfg: GrammarConfig
    exit_cfg: DiscoveryExitConfig
    follow_up_top_k: int = 3
    analysis_top_k: int = 10
    alpha: float = 0.05


@dataclass
class SearchResult:
    """Aggregated result of one full search run.

    ``log_rows`` is the list of search-log row dicts (one per generated
    rule, including rejections). ``specs_by_id`` maps rule_id ->
    RuleSpec for downstream pretty-printing.
    """

    log_rows: list[dict]
    causal_rejected: list[dict]
    pool_floor_rejected: list[dict]
    bonferroni_report: BonferroniReport
    ranked_top: tuple[RankedRule, ...]
    survivors: tuple[int, ...]
    specs_by_id: dict[int, RuleSpec]
    wall_clock_seconds: float


def run_search(
    fixtures: Sequence[PairFixture],
    grid: QuantileGrid,
    lineage_df: pd.DataFrame,
    cfg: SearchConfig,
    accepted_lineage: Sequence[str] = ("clean",),
    exclude_classes: Sequence[str] = (),
    feature_pool: Sequence[str] | None = None,
    progress_every: int = 500,
) -> SearchResult:
    """Run the full random-search loop.

    Parameters
    ----------
    fixtures
        One ``PairFixture`` per pair, with ``pair_df``, ``feature_matrix``,
        and ``atr_series`` pre-aligned to the same DatetimeIndex.
    grid
        Per-feature quantile threshold table.
    lineage_df
        Output of ``core.features.pipeline.feature_lineage_dataframe()``.
    cfg
        :class:`SearchConfig` with all locked parameters.
    accepted_lineage
        Lineage tags allowed through the causal filter. Default: ("clean",).
    exclude_classes
        Feature classes to additionally exclude (e.g., experimental).
    feature_pool
        If provided, the explicit clean-feature pool to draw atoms from.
        If ``None``, the pool is computed via
        :func:`core.discovery.causal_filter.clean_feature_pool`.
    progress_every
        Print a progress line every N rules. Set to 0 to silence.

    Returns
    -------
    :class:`SearchResult` with the search log, rejection lists, Bonferroni
    report, top-K rankings, survivors, and wall-clock time.
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
    next_trade_id = 0

    t0 = time.perf_counter()
    n_evaluated = 0

    for spec in rules:
        if progress_every and (spec.rule_id + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (spec.rule_id + 1) / max(elapsed, 1e-9)
            print(
                f"[discovery] rule {spec.rule_id + 1}/{cfg.n_rules} | "
                f"evaluated={n_evaluated} | "
                f"causal_rej={len(causal_rejected)} | pool_floor_rej={len(pool_floor_rejected)} | "
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
            continue

        # 2-4. Compile rule against each pair; collect trades.
        pool_trades = []
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
            pair_trades, next_trade_id = simulate_pair_pool(
                pair=fx.pair,
                pair_df=fx.pair_df,
                trigger_mask=trigger_mask,
                atr_series=fx.atr_series,
                cfg=cfg.exit_cfg,
                next_trade_id=next_trade_id,
            )
            pool_trades.extend(pair_trades)
        if pool_trades is None:
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
            continue

        metrics = compute_rule_metrics(pool_trades)
        log_rows.append(_log_row_for_evaluated(spec, metrics))
        n_evaluated += 1

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
    }


def _log_row_for_evaluated(spec: RuleSpec, m: RuleMetrics) -> dict:
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
    }


def _is_finite_float(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


__all__ = (
    "PairFixture",
    "SearchConfig",
    "SearchResult",
    "run_search",
)
