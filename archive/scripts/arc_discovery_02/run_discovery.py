"""Top-level CLI for arc_discovery_02.

Same shape as scripts/arc_discovery_01/run_discovery.py with three amendments:

  A. ``DiscoveryExitConfig.time_exit_bars`` set from config (240 at 4H).
  B. ``pool_size_floor: 500`` per amended config.
  C. ``iteration_budget_per_rule`` + ``total_wallclock_cap_hours`` threaded
     into the SearchConfig.

Writes outputs under ``results/arc_discovery_02/step_1/`` with the same
artefact layout as _01 — plus ``preflight_smoke_test.md`` produced by
``preflight_smoke.py`` (separate entry point).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from core.data.aggregator import aggregate
from core.discovery.causal_filter import clean_feature_pool
from core.discovery.density_filter import DensityBand
from core.discovery.grammar import Combinator, GrammarConfig, Op
from core.discovery.io import (
    WrittenArtefacts,
    render_bonferroni_survivors_md,
    render_causal_rejections_md,
    render_compute_budget_used_md,
    render_top_10_raw_md,
    write_full_search_log,
    write_manifest,
)
from core.discovery.pool_simulator import DiscoveryExitConfig
from core.discovery.quantile_grid import build_quantile_grid
from core.discovery.random_search import (
    PairFixture,
    SearchConfig,
    SearchResult,
    run_search,
)
from core.features.pipeline import compute_feature_matrix, feature_lineage_dataframe
from core.sim.panel import Panel
from scripts.arc_discovery_02._density_helpers import (
    build_calibration_fixture,
    density_filtered_rules,
    density_rejected_log_row,
)

DEFAULT_CONFIG_PATH = Path("configs/arc_discovery_02.yaml")
ARC_NAME = "arc_discovery_02"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _restrict_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    tz = df.index.tz
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def _build_fixtures(
    pairs: Iterable[str],
    tf: str,
    histdata_root: Path,
    cache_root: Path,
    window_start: str,
    window_end: str,
    clean_features: tuple[str, ...],
) -> list[PairFixture]:
    fixtures: list[PairFixture] = []
    panel_pair_dfs: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df = aggregate(
            pair, tf,
            histdata_root=str(histdata_root),
            cache_root=str(cache_root),
            use_cache=True,
        )
        df = _restrict_window(df, window_start, window_end)
        panel_pair_dfs[pair] = df
    panel = Panel(pair_dfs=panel_pair_dfs, tf=tf)

    for pair, df in panel_pair_dfs.items():
        fm = compute_feature_matrix(pair, df, panel=panel, names=list(clean_features))
        if "atr_14" in fm.matrix.columns:
            atr = fm.matrix["atr_14"]
        else:
            from core.features._helpers import mid_close, mid_high, mid_low, wilder_atr
            atr = wilder_atr(
                mid_high(df), mid_low(df), mid_close(df), period=14
            ).shift(1).rename("atr_14")
        fixtures.append(
            PairFixture(
                pair=pair,
                pair_df=df,
                feature_matrix=fm.matrix,
                atr_series=atr,
            )
        )
    return fixtures


def _grammar_cfg_from_yaml(g: dict) -> GrammarConfig:
    op_map = {
        "gt": Op.GT, "lt": Op.LT, "ge": Op.GE,
        "le": Op.LE, "eq": Op.EQ, "ne": Op.NE,
    }
    return GrammarConfig(
        max_atoms_per_rule=int(g["max_atoms_per_rule"]),
        threshold_quantiles=tuple(float(q) for q in g["threshold_quantiles"]),
        allow_not=bool(g.get("allow_not", True)),
        operators=tuple(op_map[o] for o in g["operators"]),
        combinators=tuple(Combinator(c) for c in g["combinators"]),
    )


def _exit_cfg_from_yaml(e: dict) -> DiscoveryExitConfig:
    return DiscoveryExitConfig(
        initial_sl_atr_mult=float(e["initial_sl_atr_mult"]),
        trail_activation_atr_mult=float(e["trail_activation_atr_mult"]),
        trail_distance_atr_mult=float(e["trail_distance_atr_mult"]),
        time_exit_bars=int(e["time_exit_bars"]) if e.get("time_exit_bars") is not None else None,
    )


def build_search_config(
    cfg: dict,
    n_rules_override: int | None = None,
    checkpoint_dir_override: Path | None = None,
) -> SearchConfig:
    """Construct SearchConfig from the YAML dict, applying any overrides."""
    wallclock_cap = cfg["caps"].get("total_wallclock_cap_hours")
    cp_cfg = cfg.get("checkpoint", {})
    cp_every = int(cp_cfg.get("every_n_rules", 0)) if cp_cfg.get("enabled", False) else 0
    cp_dir = checkpoint_dir_override
    if cp_dir is None and cp_every > 0:
        cp_dir = (
            Path(cfg["output"]["arc_root"])
            / cfg["output"]["step1_discovery_subdir"]
        )
    return SearchConfig(
        n_rules=int(n_rules_override or cfg["search"]["budget"]),
        random_seed=int(cfg["determinism"]["random_state"]),
        pool_floor=int(cfg["search"]["pool_size_floor"]),
        grammar_cfg=_grammar_cfg_from_yaml(cfg["grammar"]),
        exit_cfg=_exit_cfg_from_yaml(cfg["exit_policy"]),
        follow_up_top_k=int(cfg["follow_up"]["spawn_follow_up_top_k"]),
        analysis_top_k=int(cfg["follow_up"]["analysis_report_top_k"]),
        alpha=float(cfg["bonferroni"]["alpha"]),
        iteration_budget_per_rule=int(cfg["caps"]["iteration_budget_per_rule"]),
        total_wallclock_cap_hours=(
            float(wallclock_cap) if wallclock_cap is not None else None
        ),
        checkpoint_every=cp_every,
        checkpoint_dir=cp_dir,
        checkpoint_arc_name=ARC_NAME,
    )


def write_outputs(
    result: SearchResult,
    cfg: dict,
    search_cfg: SearchConfig,
    n_pairs: int,
    output_root: Path,
) -> WrittenArtefacts:
    step1_dir = output_root / cfg["output"]["step1_discovery_subdir"]
    step1_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "step_1" / "manifest.json"

    top_10_path = step1_dir / cfg["output"]["artefacts"]["top_10_raw"]
    bonf_path = step1_dir / cfg["output"]["artefacts"]["bonferroni_survivors"]
    log_path = step1_dir / cfg["output"]["artefacts"]["full_search_log"]
    rejections_path = step1_dir / cfg["output"]["artefacts"]["causal_audit_rejections"]
    budget_path = step1_dir / cfg["output"]["artefacts"]["compute_budget_used"]

    sha_log = write_full_search_log(log_path, result.log_rows)
    log_df = pd.DataFrame(result.log_rows)

    # Compute time-exit-hit-pct distribution across evaluated rules.
    evaluated = log_df[log_df["pool_floor_pass"] == True]  # noqa: E712
    n_evaluation_timeouts = int(
        (log_df.get("evaluation_timeout", pd.Series(dtype=bool)) == True).sum()  # noqa: E712
    )
    time_exit_summary: dict | None = None
    if "time_exit_hit_pct" in evaluated.columns and len(evaluated) > 0:
        pcts = evaluated["time_exit_hit_pct"].dropna()
        if len(pcts) > 0:
            time_exit_summary = {
                "mean": float(pcts.mean()),
                "p50": float(pcts.quantile(0.50)),
                "p90": float(pcts.quantile(0.90)),
                "max": float(pcts.max()),
            }

    top_10_md = render_top_10_raw_md(
        result.ranked_top,
        result.specs_by_id,
        result.bonferroni_report,
        follow_up_top_k=search_cfg.follow_up_top_k,
        arc_name=ARC_NAME,
        show_time_exit_hit_pct=True,
    )
    _write_text(top_10_path, top_10_md)

    bonf_md = render_bonferroni_survivors_md(
        result.survivors, log_df, result.specs_by_id, result.bonferroni_report,
        arc_name=ARC_NAME,
    )
    _write_text(bonf_path, bonf_md)

    rej_md = render_causal_rejections_md(
        result.causal_rejected, search_cfg.n_rules, arc_name=ARC_NAME,
    )
    _write_text(rejections_path, rej_md)

    budget_md = render_compute_budget_used_md(
        result.bonferroni_report,
        wall_clock_seconds=result.wall_clock_seconds,
        n_pairs=n_pairs,
        primary_tf=cfg["arc"]["tf"],
        arc_name=ARC_NAME,
        n_evaluation_timeouts=n_evaluation_timeouts,
        time_exit_hit_pct_summary=time_exit_summary,
        halted_at_aggregate_cap=result.halted_at_aggregate_cap,
        rules_run=result.rules_run,
    )
    _write_text(budget_path, budget_md)

    density_outcome = getattr(result, "_density_outcome", None)
    density_extras = {}
    if density_outcome is not None:
        density_extras = {
            "density_filter_enabled": True,
            "density_n_attempts": int(density_outcome.n_attempts),
            "density_n_accepted": int(len(density_outcome.accepted)),
            "density_n_rejected": int(len(density_outcome.rejected_rows)),
            "density_cap_hit": bool(density_outcome.cap_hit),
        }
    else:
        density_extras = {"density_filter_enabled": False}

    manifest_extras = {
        "n_pairs": n_pairs,
        "primary_tf": cfg["arc"]["tf"],
        "window_start": cfg["window"]["start"],
        "window_end": cfg["window"]["end"],
        "wall_clock_seconds": float(result.wall_clock_seconds),
        "halted_at_aggregate_cap": bool(result.halted_at_aggregate_cap),
        "rules_run": int(result.rules_run),
        **density_extras,
        "follow_up_top_k": search_cfg.follow_up_top_k,
        "analysis_top_k": search_cfg.analysis_top_k,
        "bonferroni_alpha": search_cfg.alpha,
        "bonferroni_threshold_primary": (
            None
            if result.bonferroni_report.threshold_primary
            != result.bonferroni_report.threshold_primary
            else result.bonferroni_report.threshold_primary
        ),
        "bonferroni_threshold_budget": result.bonferroni_report.threshold_budget,
        "n_generated": result.bonferroni_report.n_generated,
        "n_evaluated": result.bonferroni_report.n_evaluated,
        "n_causal_rejected": result.bonferroni_report.n_causal_rejected,
        "n_pool_floor_rejected": result.bonferroni_report.n_pool_floor_rejected,
        "n_evaluation_timeouts": n_evaluation_timeouts,
        "n_survivors": len(result.survivors),
        "iteration_budget_per_rule": search_cfg.iteration_budget_per_rule,
        "total_wallclock_cap_hours": search_cfg.total_wallclock_cap_hours,
    }
    sha_manifest = write_manifest(
        manifest_path,
        artefact_paths={
            "top_10_raw": top_10_path,
            "bonferroni_survivors": bonf_path,
            "full_search_log": log_path,
            "causal_audit_rejections": rejections_path,
            "compute_budget_used": budget_path,
        },
        extras=manifest_extras,
        arc_name=ARC_NAME,
    )
    return WrittenArtefacts(
        top_10_raw=_sha256(top_10_path),
        bonferroni_survivors=_sha256(bonf_path),
        full_search_log=sha_log,
        causal_audit_rejections=_sha256(rejections_path),
        compute_budget_used=_sha256(budget_path),
        manifest=sha_manifest,
    )


def run(
    config_path: Path,
    n_rules_override: int | None = None,
    pairs_override: list[str] | None = None,
    output_root_override: Path | None = None,
) -> WrittenArtefacts:
    cfg = _load_config(config_path)

    pairs = pairs_override or list(cfg["pair_set"])
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
    print(f"[{ARC_NAME}] clean-feature pool: {len(clean)} features", flush=True)

    print(f"[{ARC_NAME}] loading {len(pairs)} pairs at TF={tf} from {cache_root}", flush=True)
    fixtures = _build_fixtures(
        pairs=pairs,
        tf=tf,
        histdata_root=histdata_root,
        cache_root=cache_root,
        window_start=window_start,
        window_end=window_end,
        clean_features=clean,
    )

    print(f"[{ARC_NAME}] building quantile grid", flush=True)
    grid = build_quantile_grid(
        feature_matrices={fx.pair: fx.feature_matrix for fx in fixtures},
        quantiles=cfg["grammar"]["threshold_quantiles"],
        feature_names=clean,
    )

    usable_features = grid.features_with_data(min_non_nan=100)
    if len(usable_features) < len(clean):
        dropped = set(clean) - set(usable_features)
        print(
            f"[{ARC_NAME}] dropping {len(dropped)} features with < 100 non-NaN obs: "
            f"{sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}",
            flush=True,
        )

    search_cfg = build_search_config(cfg, n_rules_override=n_rules_override)

    # Amendment D — generation-time density filter (chat 2026-05-25).
    # Builds calibration fixture, generates up to max_generation_attempts rules,
    # accepts the first n_rules that fall in the density band.
    density_cfg = cfg.get("density_filter", {})
    density_outcome = None
    pre_filtered_rules = None
    if density_cfg.get("enabled", False):
        band = DensityBand(
            lo=float(density_cfg["band_lo"]),
            hi=float(density_cfg["band_hi"]),
        )
        print(
            f"[{ARC_NAME}] building density-filter calibration fixture "
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
        print(
            f"[{ARC_NAME}] density filter: target={search_cfg.n_rules} rules in "
            f"[{band.lo}, {band.hi}]; max_attempts={density_cfg['max_generation_attempts']}",
            flush=True,
        )
        density_outcome = density_filtered_rules(
            target_n_passes=search_cfg.n_rules,
            seed=search_cfg.random_seed,
            feature_pool=usable_features,
            grammar_cfg=search_cfg.grammar_cfg,
            fixture=calib_fixture,
            band=band,
            max_generation_attempts=int(density_cfg["max_generation_attempts"]),
        )
        pre_filtered_rules = density_outcome.accepted
        print(
            f"[{ARC_NAME}] density filter: {len(density_outcome.accepted)} accepted / "
            f"{density_outcome.n_attempts} attempted "
            f"({len(density_outcome.rejected_rows)} rejected); "
            f"cap_hit={density_outcome.cap_hit}",
            flush=True,
        )

    result = run_search(
        fixtures=fixtures,
        grid=grid,
        lineage_df=lineage_df,
        cfg=search_cfg,
        accepted_lineage=accepted,
        exclude_classes=exclude_classes,
        feature_pool=usable_features,
        pre_filtered_rules=pre_filtered_rules,
    )

    # Append density-rejected rows to the search log + causal_rejected list so
    # the full search log + causal_audit_rejections.md include them.
    if density_outcome is not None and density_outcome.rejected_rows:
        for r in density_outcome.rejected_rows:
            result.log_rows.append(density_rejected_log_row(r))
            result.causal_rejected.append(
                {"rule_id": int(r["rule_id"]), "reason": r["reason"]}
            )

    # Bundle density outcome onto the result for downstream reporting.
    setattr(result, "_density_outcome", density_outcome)

    print(
        f"[{ARC_NAME}] search done: evaluated={result.bonferroni_report.n_evaluated} "
        f"/ generated={search_cfg.n_rules} ; rules_run={result.rules_run} ; "
        f"timeouts={sum(1 for r in result.log_rows if r.get('evaluation_timeout'))} ; "
        f"survivors={len(result.survivors)} ; "
        f"halted_at_cap={result.halted_at_aggregate_cap}",
        flush=True,
    )

    output_root = output_root_override or Path(cfg["output"]["arc_root"])

    # Aggregate-cap HALT case: dump under archive/probes/arc_discovery_02_partial/
    # (matching the _01 archive convention) instead of the main results path.
    if result.halted_at_aggregate_cap:
        partial_root = Path("archive/probes/arc_discovery_02_partial")
        print(
            f"[{ARC_NAME}] HALT triggered — writing partial dump to {partial_root}",
            flush=True,
        )
        return write_outputs(result, cfg, search_cfg, len(fixtures), partial_root)

    return write_outputs(result, cfg, search_cfg, len(fixtures), output_root)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text = text + "\n"
    path.write_bytes(text.encode("utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run arc_discovery_02 signal-discovery probe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-rules", type=int, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    pairs_override = None
    if args.pairs:
        pairs_override = [p.strip() for p in args.pairs.split(",") if p.strip()]

    artefacts = run(
        config_path=args.config,
        n_rules_override=args.n_rules,
        pairs_override=pairs_override,
        output_root_override=args.output_root,
    )
    print(f"[{ARC_NAME}] artefacts written:")
    for k, v in artefacts.__dict__.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
