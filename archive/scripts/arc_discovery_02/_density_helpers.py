"""Shared density-filter wiring for run_discovery + preflight_smoke.

Loads the calibration fixture (6 months EURUSD H4 + feature matrix) and
generates rules deterministically until N pass the density band, capped by
max_generation_attempts. Density-rejected rules are returned alongside so
the search log records every attempt for audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from core.data.aggregator import aggregate
from core.discovery.density_filter import (
    CalibrationFixture,
    DensityBand,
    check_density,
    restrict_to_window,
)
from core.discovery.grammar import GrammarConfig, RuleSpec, generate_rule_population
from core.discovery.quantile_grid import QuantileGrid
from core.features.pipeline import compute_feature_matrix
from core.sim.panel import Panel


@dataclass(frozen=True)
class DensityFilterOutcome:
    """Result of the density-filter pre-pass.

    ``accepted`` — first N rules that passed the band (in seed=42 generation
    order, rule_ids preserved). Length = target_n_passes or less if the
    generation-attempt cap fired.
    ``rejected_rows`` — full rejection records, one per density-rejected
    candidate, ready to merge into the search log + causal_audit_rejections.
    ``n_attempts`` — total rules generated (accepted + rejected).
    ``cap_hit`` — True if max_generation_attempts was reached before
    target_n_passes accepted rules accumulated.
    """

    accepted: tuple[RuleSpec, ...]
    rejected_rows: tuple[dict, ...]
    n_attempts: int
    cap_hit: bool


def build_calibration_fixture(
    pair: str,
    tf: str,
    calibration_window_start: str,
    calibration_window_end: str,
    histdata_root: Path,
    cache_root: Path,
    clean_features: Sequence[str],
    grid: QuantileGrid,
) -> CalibrationFixture:
    """Load + restrict + compute feature matrix for the calibration sample.

    ``grid`` is the FULL-WINDOW (2010-2020) quantile grid — the calibration
    fixture only narrows the bar range, not the threshold reference.
    """
    df = aggregate(
        pair, tf,
        histdata_root=str(histdata_root),
        cache_root=str(cache_root),
        use_cache=True,
    )
    df_calib = restrict_to_window(df, calibration_window_start, calibration_window_end)
    panel = Panel(pair_dfs={pair: df_calib}, tf=tf)
    fm = compute_feature_matrix(pair, df_calib, panel=panel, names=list(clean_features))
    return CalibrationFixture(
        pair=pair,
        feature_matrix=fm.matrix,
        grid=grid,
        window_start=calibration_window_start,
        window_end=calibration_window_end,
    )


def density_filtered_rules(
    target_n_passes: int,
    seed: int,
    feature_pool: Sequence[str],
    grammar_cfg: GrammarConfig,
    fixture: CalibrationFixture,
    band: DensityBand,
    max_generation_attempts: int,
    progress_every: int = 5000,
) -> DensityFilterOutcome:
    """Generate rules deterministically; keep first ``target_n_passes`` that pass.

    A single ``generate_rule_population(n=max_generation_attempts, seed=seed)``
    call materialises the full candidate pool up front. We iterate through it
    in order, applying the density check, until either:
      - ``target_n_passes`` candidates accept (early stop), or
      - all ``max_generation_attempts`` candidates have been tried.
    """
    candidates = generate_rule_population(
        n=max_generation_attempts,
        seed=seed,
        feature_pool=feature_pool,
        cfg=grammar_cfg,
    )

    accepted: list[RuleSpec] = []
    rejected_rows: list[dict] = []
    n_attempts = 0
    for spec in candidates:
        n_attempts += 1
        res = check_density(spec, fixture, band)
        if res.passed:
            accepted.append(spec)
            if progress_every and (len(accepted) % progress_every == 0):
                print(
                    f"[density] accepted {len(accepted)}/{target_n_passes} | "
                    f"attempts={n_attempts} | "
                    f"reject_rate={len(rejected_rows) / max(n_attempts, 1):.3f}",
                    flush=True,
                )
            if len(accepted) >= target_n_passes:
                break
        else:
            rejected_rows.append(
                {
                    "rule_id": int(spec.rule_id),
                    "rule_spec_json": spec.to_json(),
                    "reason": f"trigger_density_out_of_band rate={res.observed_rate:.4f}",
                    "observed_rate": float(res.observed_rate),
                    "n_atoms": int(spec.n_atoms),
                    "features_used": ",".join(spec.features_used()),
                }
            )

    cap_hit = len(accepted) < target_n_passes
    return DensityFilterOutcome(
        accepted=tuple(accepted),
        rejected_rows=tuple(rejected_rows),
        n_attempts=n_attempts,
        cap_hit=cap_hit,
    )


def density_rejected_log_row(rejected: dict) -> dict:
    """Convert a density-rejected dict into a search-log row (parquet schema)."""
    return {
        "rule_id": int(rejected["rule_id"]),
        "rule_spec_json": rejected["rule_spec_json"],
        "n_atoms": int(rejected["n_atoms"]),
        "features_used": rejected["features_used"],
        "causal_filter_pass": True,        # density is upstream of causal
        "pool_floor_pass": False,
        "causal_rejection_reason": rejected["reason"],
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
        "evaluation_timeout": False,
        "iterations_consumed": 0,
        "time_exit_hit_pct": None,
    }


__all__ = (
    "DensityFilterOutcome",
    "build_calibration_fixture",
    "density_filtered_rules",
    "density_rejected_log_row",
)
