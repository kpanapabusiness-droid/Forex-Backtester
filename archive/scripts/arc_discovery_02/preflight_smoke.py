"""Pre-flight 50-rule smoke test per arc_discovery_02 dispatch §5.

Generates the FIRST 50 rules via the same deterministic generator the full
run uses (seed=42), evaluates them under the amended pipeline (60-bar time
exit + pool floor 500 + 5M bar-iteration cap + density band [0.005, 0.04]),
and writes a preflight_smoke_test.md with timing distribution + 10k
extrapolation.

Chat go/no-go criteria (revised 2026-05-25 after Amendment F — checkpointed
interruptible runs):
  * Pool-floor pass ≥ 70% AND iter-cap fires ≤ 5%   → GO (launch full 10k)
  * Pool-floor pass < 70% OR iter-cap fires > 5%    → HALT, surface to chat
  * Wall-clock projection: REPORTED but not blocking.

The script exits 0 on GO, 1 on HALT.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from core.discovery.causal_filter import clean_feature_pool
from core.discovery.density_filter import DensityBand
from core.discovery.grammar import generate_rule_population
from core.discovery.pool_simulator import simulate_pair_pool
from core.discovery.quantile_grid import build_quantile_grid
from core.discovery.rule_engine import compile_rule
from core.features.pipeline import feature_lineage_dataframe
from scripts.arc_discovery_02._density_helpers import (
    build_calibration_fixture,
    density_filtered_rules,
)
from scripts.arc_discovery_02.run_discovery import (
    ARC_NAME,
    DEFAULT_CONFIG_PATH,
    _build_fixtures,
    _exit_cfg_from_yaml,
    _grammar_cfg_from_yaml,
    _load_config,
    _write_text,
)

SMOKE_N_RULES = 50


def _evaluate_one_rule(spec, fixtures, grid, exit_cfg, pool_floor, iteration_budget) -> dict:
    """Evaluate a single rule end-to-end and return its timing+outcome record."""
    rule_start = time.perf_counter()
    next_trade_id = 0
    pool_trades = []
    rule_iterations_consumed = 0
    rule_budget = iteration_budget
    aborted = False
    for fx in fixtures:
        trigger_mask = compile_rule(spec, fx.feature_matrix, grid)
        pair_trades, next_trade_id, pair_iters, pair_aborted = simulate_pair_pool(
            pair=fx.pair, pair_df=fx.pair_df, trigger_mask=trigger_mask,
            atr_series=fx.atr_series, cfg=exit_cfg, next_trade_id=next_trade_id,
            iteration_budget=rule_budget,
        )
        rule_iterations_consumed += pair_iters
        if rule_budget is not None:
            rule_budget = max(0, rule_budget - pair_iters)
        pool_trades.extend(pair_trades)
        if pair_aborted:
            aborted = True
            break
    elapsed = time.perf_counter() - rule_start
    pool_size = len(pool_trades)
    pool_floor_pass = (not aborted) and (pool_size >= pool_floor)
    n_time_exit = sum(1 for t in pool_trades if t.exit_reason == "time_exit")
    time_exit_pct = (n_time_exit / pool_size) if pool_size > 0 else 0.0
    return {
        "rule_id": int(spec.rule_id),
        "n_atoms": int(spec.n_atoms),
        "pool_size": pool_size,
        "iterations_consumed": rule_iterations_consumed,
        "evaluation_timeout": aborted,
        "pool_floor_pass": pool_floor_pass,
        "time_exit_hit_pct": time_exit_pct,
        "wall_clock_seconds": elapsed,
    }


def run_smoke(config_path: Path) -> tuple[Path, int]:
    """Run the 50-rule smoke. Returns (output_md_path, exit_code).

    exit_code: 0 = GO, 1 = HALT, 2 = BORDERLINE.
    """
    cfg = _load_config(config_path)
    pairs = list(cfg["pair_set"])
    tf = str(cfg["arc"]["tf"])
    window_start = str(cfg["window"]["start"])
    window_end = str(cfg["window"]["end"])
    histdata_root = Path(cfg["data"]["histdata_root"])
    cache_root = Path(cfg["data"]["cache_root"])

    lineage_df = feature_lineage_dataframe()
    accepted = tuple(cfg["causal_filter"]["accepted_lineage"])
    exclude_classes = tuple(cfg["causal_filter"].get("exclude_classes", []) or [])
    clean = clean_feature_pool(
        lineage_df, accepted=accepted, exclude_classes=exclude_classes
    )
    if not clean:
        raise RuntimeError("Causal filter rejected every registered feature.")
    print(f"[smoke] clean-feature pool: {len(clean)} features", flush=True)

    setup_start = time.perf_counter()
    print(f"[smoke] loading {len(pairs)} pairs at TF={tf}", flush=True)
    fixtures = _build_fixtures(
        pairs=pairs, tf=tf, histdata_root=histdata_root, cache_root=cache_root,
        window_start=window_start, window_end=window_end, clean_features=clean,
    )
    print("[smoke] building quantile grid", flush=True)
    grid = build_quantile_grid(
        feature_matrices={fx.pair: fx.feature_matrix for fx in fixtures},
        quantiles=cfg["grammar"]["threshold_quantiles"],
        feature_names=clean,
    )
    usable_features = grid.features_with_data(min_non_nan=100)
    setup_elapsed = time.perf_counter() - setup_start
    print(f"[smoke] setup complete in {setup_elapsed:.1f}s", flush=True)
    density_filter_start = time.perf_counter()

    grammar_cfg = _grammar_cfg_from_yaml(cfg["grammar"])
    exit_cfg = _exit_cfg_from_yaml(cfg["exit_policy"])
    pool_floor = int(cfg["search"]["pool_size_floor"])
    iteration_budget = int(cfg["caps"]["iteration_budget_per_rule"])
    random_seed = int(cfg["determinism"]["random_state"])

    # Amendment D — pre-filter by density on the calibration fixture so the
    # smoke evaluates the first 50 rules that ALSO pass the density band
    # (the same first-50 the full run will see).
    density_cfg = cfg.get("density_filter", {})
    density_outcome = None
    if density_cfg.get("enabled", False):
        band = DensityBand(
            lo=float(density_cfg["band_lo"]),
            hi=float(density_cfg["band_hi"]),
        )
        print(
            f"[smoke] building density-filter calibration fixture "
            f"({density_cfg['calibration_pair']}, "
            f"{density_cfg['calibration_window_start']} to "
            f"{density_cfg['calibration_window_end']})",
            flush=True,
        )
        calib_fixture = build_calibration_fixture(
            pair=str(density_cfg["calibration_pair"]),
            tf=tf,
            calibration_window_start=str(density_cfg["calibration_window_start"]),
            calibration_window_end=str(density_cfg["calibration_window_end"]),
            histdata_root=histdata_root,
            cache_root=cache_root,
            clean_features=usable_features,
            grid=grid,
        )
        print(f"[smoke] applying density filter [{band.lo}, {band.hi}]", flush=True)
        density_outcome = density_filtered_rules(
            target_n_passes=SMOKE_N_RULES,
            seed=random_seed,
            feature_pool=usable_features,
            grammar_cfg=grammar_cfg,
            fixture=calib_fixture,
            band=band,
            max_generation_attempts=int(density_cfg["max_generation_attempts"]),
            progress_every=0,
        )
        rules = density_outcome.accepted
        print(
            f"[smoke] density filter: {len(rules)} accepted / "
            f"{density_outcome.n_attempts} attempted "
            f"({len(density_outcome.rejected_rows)} rejected); "
            f"cap_hit={density_outcome.cap_hit}",
            flush=True,
        )
    else:
        rules = generate_rule_population(
            n=SMOKE_N_RULES, seed=random_seed,
            feature_pool=usable_features, cfg=grammar_cfg,
        )
    density_filter_elapsed = time.perf_counter() - density_filter_start
    print(f"[smoke] density filter pass complete in {density_filter_elapsed:.1f}s", flush=True)
    print(f"[smoke] evaluating {len(rules)} rules", flush=True)

    rows: list[dict] = []
    loop_start = time.perf_counter()
    for i, spec in enumerate(rules, start=1):
        rec = _evaluate_one_rule(
            spec, fixtures, grid, exit_cfg, pool_floor, iteration_budget
        )
        rows.append(rec)
        if i % 10 == 0:
            elapsed = time.perf_counter() - loop_start
            rate = i / max(elapsed, 1e-9)
            print(
                f"[smoke] {i}/{SMOKE_N_RULES} | "
                f"floor_pass={sum(1 for r in rows if r['pool_floor_pass'])} | "
                f"timeouts={sum(1 for r in rows if r['evaluation_timeout'])} | "
                f"elapsed={elapsed:.1f}s ({rate:.2f} rules/s)",
                flush=True,
            )
    total_loop_elapsed = time.perf_counter() - loop_start

    # Aggregate stats
    n = len(rows)
    n_floor_pass = sum(1 for r in rows if r["pool_floor_pass"])
    n_timeouts = sum(1 for r in rows if r["evaluation_timeout"])
    n_clean_fail_floor = sum(
        1 for r in rows if (not r["evaluation_timeout"]) and (not r["pool_floor_pass"])
    )
    times = [r["wall_clock_seconds"] for r in rows]
    time_p50 = statistics.median(times) if times else 0.0
    time_p90 = (
        statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times) if times else 0.0
    )
    time_max = max(times) if times else 0.0
    pool_floor_pass_rate = n_floor_pass / n if n > 0 else 0.0
    iter_cap_fire_rate = n_timeouts / n if n > 0 else 0.0

    # Extrapolation includes density-filter pass cost (which scales with attempts,
    # but attempts scale linearly with accepted rules → 200× extrap factor holds).
    per_smoke_loop_sec = total_loop_elapsed + density_filter_elapsed
    extrapolated_10k_hours = (per_smoke_loop_sec * (10000 / max(n, 1))) / 3600.0

    # Chat go/no-go (Amendment F — wall-clock is informational only).
    halt = (
        pool_floor_pass_rate < 0.70
        or iter_cap_fire_rate > 0.05
    )
    if halt:
        verdict = "HALT"
        exit_code = 1
    else:
        verdict = "GO"
        exit_code = 0

    output_path = (
        Path(cfg["output"]["arc_root"])
        / cfg["output"]["step1_discovery_subdir"]
        / cfg["output"]["artefacts"]["preflight_smoke"]
    )

    density_lines: list[str] = []
    if density_outcome is not None:
        accept_rate = (
            len(density_outcome.accepted) / max(density_outcome.n_attempts, 1)
        )
        density_lines = [
            f"| Density-filter attempts (to find first {SMOKE_N_RULES}) | {density_outcome.n_attempts} |",
            f"| Density-filter accept rate | {100.0 * accept_rate:.1f}% |",
            f"| Density-filter rejected | {len(density_outcome.rejected_rows)} |",
            f"| Density-filter cap hit | {density_outcome.cap_hit} |",
            f"| Density-filter wall-clock | {density_filter_elapsed:.1f}s |",
        ]

    lines: list[str] = [
        f"# Pre-flight smoke test — {ARC_NAME}",
        "",
        f"> 50-rule smoke per dispatch §5 + Amendment D (density filter). "
        f"First {SMOKE_N_RULES} density-passed rules from the same seed={random_seed} "
        f"generator the full 10k run uses.",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total rules evaluated | {n} |",
        f"| Pool-floor pass count | {n_floor_pass} ({100.0 * pool_floor_pass_rate:.1f}%) |",
        f"| Evaluation timeouts (iter-cap) | {n_timeouts} ({100.0 * iter_cap_fire_rate:.1f}%) |",
        f"| Clean fail-floor (below 500 trades) | {n_clean_fail_floor} |",
        f"| Wall-clock per rule: p50 / p90 / max | {time_p50:.2f}s / {time_p90:.2f}s / {time_max:.2f}s |",
        f"| Setup wall-clock (panels + features + grid) | {setup_elapsed:.1f}s |",
        *density_lines,
        f"| 50-rule loop wall-clock | {total_loop_elapsed:.1f}s |",
        f"| **Extrapolated 10k wall-clock (incl. density-filter pass)** | **{extrapolated_10k_hours:.1f}h** |",
        "",
        "## Chat go/no-go criteria (Amendment F — wall-clock informational only)",
        "",
        "| Criterion | Threshold | Observed | Blocking? |",
        "|---|---|---|---|",
        f"| Pool-floor pass rate | ≥ 70% | {100.0 * pool_floor_pass_rate:.1f}% | YES |",
        f"| Iter-cap fire rate | ≤ 5% | {100.0 * iter_cap_fire_rate:.1f}% | YES |",
        f"| 10k wall-clock projection | (informational) | {extrapolated_10k_hours:.1f}h | NO — checkpointed |",
        "",
        "## Per-rule timing (first 50)",
        "",
        "| Rule ID | n_atoms | Pool size | Iters consumed | Eval timeout | Floor pass | Time-exit % | Wall-clock (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['rule_id']} | {r['n_atoms']} | {r['pool_size']} | "
            f"{r['iterations_consumed']} | {r['evaluation_timeout']} | "
            f"{r['pool_floor_pass']} | {100.0 * r['time_exit_hit_pct']:.1f}% | "
            f"{r['wall_clock_seconds']:.2f} |"
        )
    lines.append("")
    md = "\n".join(lines)
    _write_text(output_path, md)
    print(f"[smoke] wrote {output_path}")
    print(f"[smoke] verdict: {verdict} (exit code {exit_code})")
    return output_path, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="arc_discovery_02 50-rule pre-flight smoke.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    _, exit_code = run_smoke(args.config)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
