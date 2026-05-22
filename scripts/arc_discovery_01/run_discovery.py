"""Top-level CLI for arc_discovery_01.

Loads:
  - 28 FX pairs at the primary TF (H1) from data/cache (built via PR-A loader)
  - v3 feature matrix per pair via core.features.pipeline.compute_feature_matrix
  - Quantile grid on the pooled training-window distribution
  - Causal-filter accepted feature pool

Runs:
  - core.discovery.random_search.run_search (10k rules by default)

Writes:
  - results/arc_discovery_01/step_1/discovery/top_10_raw.md
  - results/arc_discovery_01/step_1/discovery/bonferroni_survivors.md
  - results/arc_discovery_01/step_1/discovery/full_search_log.parquet
  - results/arc_discovery_01/step_1/discovery/causal_audit_rejections.md
  - results/arc_discovery_01/step_1/discovery/compute_budget_used.md
  - results/arc_discovery_01/step_1/manifest.json

Determinism: random_state=42 throughout; n_jobs=1 single-process.
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
    run_search,
)
from core.features.pipeline import compute_feature_matrix, feature_lineage_dataframe
from core.sim.panel import Panel

DEFAULT_CONFIG_PATH = Path("configs/arc_discovery_01.yaml")


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
    """Load + restrict + compute features for each pair.

    Restricts to the configured training window BEFORE computing features
    so quantile thresholds reflect the IS distribution only. The
    feature pipeline shifts by 1 internally so lookahead is preserved.
    """
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
        # ATR(14) — sourced from the feature matrix's atr_14 column if available;
        # otherwise computed inline. (price_geometry registers atr_14 with shift(1).)
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

    # Causal-filter accepted feature pool (computed before grammar generation).
    lineage_df = feature_lineage_dataframe()
    accepted = tuple(cfg["causal_filter"]["accepted_lineage"])
    exclude_classes = tuple(cfg["causal_filter"].get("exclude_classes", []) or [])
    clean = clean_feature_pool(
        lineage_df, accepted=accepted, exclude_classes=exclude_classes
    )
    if not clean:
        raise RuntimeError("Causal filter rejected every registered feature.")
    print(f"[discovery] clean-feature pool: {len(clean)} features", flush=True)

    print(f"[discovery] loading {len(pairs)} pairs at TF={tf} from {cache_root}", flush=True)
    fixtures = _build_fixtures(
        pairs=pairs,
        tf=tf,
        histdata_root=histdata_root,
        cache_root=cache_root,
        window_start=window_start,
        window_end=window_end,
        clean_features=clean,
    )

    print("[discovery] building quantile grid", flush=True)
    grid = build_quantile_grid(
        feature_matrices={fx.pair: fx.feature_matrix for fx in fixtures},
        quantiles=cfg["grammar"]["threshold_quantiles"],
        feature_names=clean,
    )

    # Drop features the grid couldn't compute (insufficient non-NaN observations).
    usable_features = grid.features_with_data(min_non_nan=100)
    if len(usable_features) < len(clean):
        dropped = set(clean) - set(usable_features)
        print(
            f"[discovery] dropping {len(dropped)} features with < 100 non-NaN obs: "
            f"{sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}",
            flush=True,
        )

    search_cfg = SearchConfig(
        n_rules=int(n_rules_override or cfg["search"]["budget"]),
        random_seed=int(cfg["determinism"]["random_state"]),
        pool_floor=int(cfg["search"]["pool_size_floor"]),
        grammar_cfg=_grammar_cfg_from_yaml(cfg["grammar"]),
        exit_cfg=_exit_cfg_from_yaml(cfg["exit_policy"]),
        follow_up_top_k=int(cfg["follow_up"]["spawn_follow_up_top_k"]),
        analysis_top_k=int(cfg["follow_up"]["analysis_report_top_k"]),
        alpha=float(cfg["bonferroni"]["alpha"]),
    )

    result = run_search(
        fixtures=fixtures,
        grid=grid,
        lineage_df=lineage_df,
        cfg=search_cfg,
        accepted_lineage=accepted,
        exclude_classes=exclude_classes,
        feature_pool=usable_features,
    )

    print(
        f"[discovery] search done: evaluated={result.bonferroni_report.n_evaluated} "
        f"/ generated={search_cfg.n_rules} ; survivors={len(result.survivors)} ; "
        f"top-{search_cfg.analysis_top_k} mean R range: "
        f"{result.ranked_top[-1].mean_r:+.4f}..{result.ranked_top[0].mean_r:+.4f}"
        if result.ranked_top
        else "[discovery] search done; no eligible top-K rules",
        flush=True,
    )

    # Write outputs
    output_root = output_root_override or Path(cfg["output"]["arc_root"])
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

    top10_md = render_top_10_raw_md(
        result.ranked_top,
        result.specs_by_id,
        result.bonferroni_report,
        follow_up_top_k=search_cfg.follow_up_top_k,
    )
    _write_text(top_10_path, top10_md)
    bonf_md = render_bonferroni_survivors_md(
        result.survivors, log_df, result.specs_by_id, result.bonferroni_report
    )
    _write_text(bonf_path, bonf_md)
    rej_md = render_causal_rejections_md(result.causal_rejected, search_cfg.n_rules)
    _write_text(rejections_path, rej_md)
    budget_md = render_compute_budget_used_md(
        result.bonferroni_report,
        wall_clock_seconds=result.wall_clock_seconds,
        n_pairs=len(fixtures),
        primary_tf=tf,
    )
    _write_text(budget_path, budget_md)

    manifest_extras = {
        "config_path": str(config_path).replace("\\", "/"),
        "n_pairs": len(fixtures),
        "primary_tf": tf,
        "window_start": window_start,
        "window_end": window_end,
        "wall_clock_seconds": float(result.wall_clock_seconds),
        "follow_up_top_k": search_cfg.follow_up_top_k,
        "analysis_top_k": search_cfg.analysis_top_k,
        "bonferroni_alpha": search_cfg.alpha,
        "bonferroni_threshold_primary": (
            None
            if result.bonferroni_report.threshold_primary != result.bonferroni_report.threshold_primary
            else result.bonferroni_report.threshold_primary
        ),
        "bonferroni_threshold_budget": result.bonferroni_report.threshold_budget,
        "n_generated": result.bonferroni_report.n_generated,
        "n_evaluated": result.bonferroni_report.n_evaluated,
        "n_causal_rejected": result.bonferroni_report.n_causal_rejected,
        "n_pool_floor_rejected": result.bonferroni_report.n_pool_floor_rejected,
        "n_survivors": len(result.survivors),
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
    )

    # Sha256 of the markdown files (read post-write to ensure deterministic encoding)
    artefacts = WrittenArtefacts(
        top_10_raw=_sha256(top_10_path),
        bonferroni_survivors=_sha256(bonf_path),
        full_search_log=sha_log,
        causal_audit_rejections=_sha256(rejections_path),
        compute_budget_used=_sha256(budget_path),
        manifest=sha_manifest,
    )
    return artefacts


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
    parser = argparse.ArgumentParser(description="Run arc_discovery_01 signal-discovery probe.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Locked-parameters YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--n-rules",
        type=int,
        default=None,
        help="Override search.budget (e.g. for smoke tests). Default: from YAML.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pair-subset override (e.g. EURUSD,GBPUSD). Default: from YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the output arc root (for ad-hoc runs).",
    )
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
    print("[discovery] artefacts written:")
    print(f"  top_10_raw.md            : {artefacts.top_10_raw}")
    print(f"  bonferroni_survivors.md  : {artefacts.bonferroni_survivors}")
    print(f"  full_search_log.parquet  : {artefacts.full_search_log}")
    print(f"  causal_audit_rejections  : {artefacts.causal_audit_rejections}")
    print(f"  compute_budget_used.md   : {artefacts.compute_budget_used}")
    print(f"  manifest.json            : {artefacts.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
