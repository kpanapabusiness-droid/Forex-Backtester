"""Arc 7 v3.0 — EXIT-EXTRACTION re-run (L_PROTOCOL v3.0 + Amendment 3).

Direct replication test of the Arc 10 recovery on Arc 7's liquidity-sweep-reclaim
V-shape pool: does the same exit-extraction mechanism that recovered Arc 10 (full
pool A1 + asymmetric `sl_partial_close_1r_runner_trail` exit, NOT entry selection)
clear Arc 7's capturable-but-not-extractable pool?

PER USER DIRECTION (2026-06-02): use the reset-floor Step-5 search engine
(`scripts/arc_7/run_arc_7.py` path, Amendment 3) for the full sweep AND gate — NOT
the governed fixed-initial harness. Gate at r_base on TRAILING worst-fold DD ->
Amendment-3 r_safe. Governed fixed-initial is a deferred confirmation run on
finalists only, IF anything clears PASS-tier (not run here).

This driver is a composition driver — it reuses run_arc_7's canonical primitives
(panel build, Step 1-4, ArcFoldRunner, Amendment-3 gate) and adds:
  1. The dispatch exit set wired via A1Config.exit_policy / A3Config.exit_policy
     (run_arc_7's built-in builders only expose trail_enabled∈{T,F}).
  2. The fixed dispatch grid: A1 (full pool) + A3 (deferred entry, best V-shape
     cluster) × exits {sl_partial_close_1r_runner_trail, sl_plus_tp_2r,
     sl_plus_tp_3r, sl_only} × SL {2.5,3.0,3.5} × exposure {2, unlimited}.
  3. Two-stage WFO: Stage A triage (all configs, F1/F6/F8) -> Stage B gate
     (top-3, full 11-fold + holdout); verdict ONLY from Stage B.

DATA: this worktree has no market data; the H4/D1/W1 5ers_eet cache lives only in
sibling worktree nice-mirzakhani-baa9e5. --cache-root points there (read-only).
Step 5 re-simulates each config against the live panels, so the SL grid and exit
policies are all simulable from the cache (no forward-price fabrication).

Determinism: random_state=42, n_jobs=1, lineterminator='\n' via core.determinism.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool, write_arc_pool  # noqa: E402
from core.architectures.a1_system_level_filter import (  # noqa: E402
    A1Architecture,
    A1Config,
    A1RunContext,
)
from core.architectures.a3_pipeline_de import A3Architecture, A3Config  # noqa: E402
from core.determinism import seed_everything  # noqa: E402
from core.runners._fold_stats_helpers import compute_per_day_max_dd  # noqa: E402
from core.runners.arc_fold_runner import ArcFoldRunner  # noqa: E402
from core.steps.path_classifier_per_fold import (  # noqa: E402
    PerFoldTrainingInputs,
    build_path_classifier_fits_per_fold,
    build_per_trade_entry_features,
)
from core.steps.step_2_clustering import run_step_2  # noqa: E402
from core.steps.step_3_capturability import run_step_3  # noqa: E402
from core.steps.step_4_extraction import run_step_4  # noqa: E402
from core.strategies.liquidity_sweep_reclaim_long.signal_module import (  # noqa: E402
    LiquiditySweepReclaimLongSignal,
)
from core.wfo.amended_gates import (  # noqa: E402
    classify_amended_fold_stats,
    compute_scaling_factors,
)
from core.wfo.chained_dd import (  # noqa: E402
    compute_chained_max_dd_from_continuous_equity,
    stitch_per_fold_oos_equity,
)
from core.wfo.folds import Fold, build_v3_folds  # noqa: E402
from core.wfo.gates import FoldStats  # noqa: E402

# Reuse run_arc_7's validated helpers (panel build, Step-1 features, manifest).
from scripts.arc_7.run_arc_7 import (  # noqa: E402
    PAIRS_28,
    _build_panel_5ers_eet,
    _build_per_trade_features,
    _compute_step1_feature_matrix,
    _write_manifest,
)

log = logging.getLogger("arc7_v3_exit_extraction")

# ── Dispatch grid constants ──────────────────────────────────────────────
EXITS: tuple[str, ...] = (
    "sl_partial_close_1r_runner_trail",  # the proven Arc 10 extractor (MUST include)
    "sl_plus_tp_2r",
    "sl_plus_tp_3r",
    "sl_only",  # baseline
)
SLS: tuple[float, ...] = (2.5, 3.0, 3.5)
EXPS: tuple[int | None, ...] = (2, None)  # per-currency cap {2, unlimited}
N_DEFER = 5  # A3 deferred-entry bars (fixed; gives 4×3×2=24 A3 configs)
R_BASE = 0.005  # engine emits at r_base 0.5% (Amendment 3 scales to r_safe)
OPERATING_R = 0.004  # 0.40% — the DD-screen reference tier
DD_SCREEN_PCT = 0.08  # Stage-A screen: trailing DD@0.40% > 8% -> drop
TRIAGE_FOLD_IDS = (1, 6, 8)  # F1 2010 (flash-crash tail), F6 2015, F8 2017
STARTING_BALANCE = 100_000.0


@dataclass
class ConfigResult:
    config_id: str
    arch: str
    exit_policy: str
    sl: float
    exp: int | None
    cluster_id: int  # -1 = full pool (A1)
    per_fold: dict[int, FoldStats]  # fold_id -> stats (at r_base)
    equity_by_fold: dict[int, pd.Series]


def _exp_str(cap: int | None) -> str:
    return str(cap) if cap is not None else "inf"


def _dd_at_operating(dd_base: float) -> float:
    """Reset-floor linear scaling: DD@0.40% = DD@r_base × (0.40/0.50)."""
    return dd_base * (OPERATING_R / R_BASE)


def _run_config_over_folds(
    arch: Any,
    cfg: Any,
    ctx: A1RunContext,
    signal_eval: Any,
    panels: Mapping[str, Any],
    folds: list[Fold],
) -> dict[int, tuple[FoldStats, pd.Series]]:
    runner = ArcFoldRunner(
        architecture=arch, signal_evaluation=signal_eval, panels=panels, run_context=ctx
    )
    out: dict[int, tuple[FoldStats, pd.Series]] = {}
    for f in folds:
        try:
            stats = runner(f, cfg)
            eq = runner.last_result.equity_curve if runner.last_result is not None else pd.Series(dtype=float)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s fold %d errored: %s", getattr(cfg, "config_id", "?"), f.fold_id, exc)
            stats = FoldStats(
                fold_id=f.fold_id, n_trades=0, roi_pct=0.0, max_dd_pct=0.0,
                days_breaching_daily_5pct=0, roi_dd_ratio=0.0,
            )
            eq = pd.Series(dtype=float)
        out[f.fold_id] = (stats, eq)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sibling = REPO_ROOT.parent / "nice-mirzakhani-baa9e5"  # .claude/worktrees/nice-mirzakhani-baa9e5
    p.add_argument("--cache-root", type=Path, default=sibling / "data" / "cache")
    p.add_argument("--histdata-root", type=Path, default=sibling / "data" / "histdata")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "arc_7_v3.0_exit_extraction")
    p.add_argument("--window-start", type=str, default="2010-01-01")
    p.add_argument("--window-end", type=str, default="2025-12-31")
    p.add_argument("--smoke", action="store_true", help="stop after Step 1 (panels+pool validation)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--verbose", action="count", default=1)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", force=True,
    )
    seed_everything(42)
    t_start = time.perf_counter()
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log.info("Arc 7 v3.0 exit-extraction — cache=%s out=%s", args.cache_root, out)

    # ── Step 0: panels (H4 + D1 + W1, 5ers_eet) ──────────────────────────
    log.info("Step 0 — building H4 + D1 + W1 panels (5ers_eet, 28 pairs) from cache")
    t0 = time.perf_counter()
    panel_h4 = _build_panel_5ers_eet(list(PAIRS_28), "H4", histdata_root=args.histdata_root, cache_root=args.cache_root)
    panel_d1 = _build_panel_5ers_eet(list(PAIRS_28), "D1", histdata_root=args.histdata_root, cache_root=args.cache_root)
    panel_w1 = _build_panel_5ers_eet(list(PAIRS_28), "W1", histdata_root=args.histdata_root, cache_root=args.cache_root)
    object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})
    panels = {"H4": panel_h4, "D1": panel_d1, "W1": panel_w1}
    log.info("Step 0 — panels built in %.1fs", time.perf_counter() - t0)

    # ── Step 1: pool ─────────────────────────────────────────────────────
    log.info("Step 1 — liquidity_sweep_reclaim_long pool")
    t0 = time.perf_counter()
    signal_module = LiquiditySweepReclaimLongSignal()
    pool_cfg = ArcPoolConfig(
        arc_name="arc_7_v3.0_exit_extraction",
        sl_atr_mult=2.0, hold_bars=240, risk_pct=R_BASE,
        window_start=pd.Timestamp(args.window_start).date(),
        window_end=pd.Timestamp(args.window_end).date(),
    )
    pool = build_arc_pool(signal_module, panels, pool_cfg)
    write_arc_pool(pool, out)
    log.info("Step 1 — pool %d trades in %.1fs (sha=%s)", len(pool.trades), time.perf_counter() - t0, pool.pool_sha256[:16])

    if args.smoke:
        log.info("SMOKE — stopping after Step 1. pool=%d sha=%s", len(pool.trades), pool.pool_sha256)
        (out / "smoke_summary.json").write_text(
            json.dumps({"pool_size": int(len(pool.trades)), "pool_sha256": pool.pool_sha256,
                        "window_start": str(pool_cfg.window_start), "window_end": str(pool_cfg.window_end)},
                       indent=2) + "\n", encoding="utf-8", newline="\n")
        return 0

    feat_matrix, feat_lineage = _compute_step1_feature_matrix(pool, panel_h4, panel_h4.pair_dfs)
    fm_for_step4 = feat_matrix.reset_index() if len(feat_matrix) > 0 else feat_matrix
    per_trade_features = _build_per_trade_features(pool.trades, feat_matrix) if len(feat_matrix) > 0 else {}

    # ── Step 2-3: clustering + capturability (candidate V-shape clusters) ──
    log.info("Step 2 — clustering")
    step2 = run_step_2(pool.trades, pool.paths)
    s2_dir = out / "step_2"
    s2_dir.mkdir(exist_ok=True)
    step2.cluster_assignments.to_parquet(s2_dir / "cluster_assignments.parquet", engine="pyarrow", compression="snappy", index=False)
    step2.cluster_summary.to_csv(s2_dir / "cluster_summary.csv", index=False, lineterminator="\n")
    (s2_dir / "cluster_summary.md").write_text(step2.summary_md, encoding="utf-8", newline="\n")

    log.info("Step 3 — capturability")
    step3 = run_step_3(pool.trades, pool.paths, step2.cluster_assignments,
                       declared_sl_mult=pool_cfg.sl_atr_mult, cluster_centroids=step2.centroids)
    s3_dir = out / "step_3"
    s3_dir.mkdir(exist_ok=True)
    step3.capturability_csv.to_csv(s3_dir / "capturability.csv", index=False, lineterminator="\n")
    (s3_dir / "capturability_summary.md").write_text(step3.summary_md, encoding="utf-8", newline="\n")
    candidate_clusters = [c for c in step3.per_cluster if c.is_candidate]
    log.info("Step 3 — %d candidate clusters: %s", len(candidate_clusters),
             [(c.cluster_id, c.shape_tag, round(c.capturability_composite, 3)) for c in candidate_clusters])

    # ── Step 4: entry-AUC for the RECORD only (NON-BLOCKING) ──────────────
    s4 = None
    train_end_ts = pd.Timestamp("2021-01-01", tz="UTC")
    if candidate_clusters and len(feat_matrix) > 0:
        log.info("Step 4 — extraction (entry-AUC for record; non-blocking)")
        try:
            s4 = run_step_4(
                pool.trades, fm_for_step4, step2.cluster_assignments,
                feature_lineage=feat_lineage,
                candidate_cluster_ids=tuple(c.cluster_id for c in candidate_clusters),
                persistence_dir=out / "step_4" / "classifiers",
                arc_name="arc_7_v3.0_exit_extraction", train_end=train_end_ts,
            )
            s4_dir = out / "step_4"
            s4_dir.mkdir(exist_ok=True)
            s4.extraction_metrics.to_csv(s4_dir / "extraction_metrics.csv", index=False, lineterminator="\n")
            (s4_dir / "extraction_summary.md").write_text(s4.summary_md, encoding="utf-8", newline="\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("Step 4 errored (non-blocking): %s", exc)

    # ── Step 5: build the dispatch grid ───────────────────────────────────
    wfo = build_v3_folds(holdout_end=pd.Timestamp(args.window_end).date())
    all_search_folds = list(wfo.folds)  # F1..F11
    fold_by_id = {f.fold_id: f for f in all_search_folds}
    base_ctx = A1RunContext(per_trade_features=per_trade_features) if per_trade_features else A1RunContext()

    # A1 — full pool (primary). No filter rules; exit policy governs exits.
    configs: list[tuple[str, Any, Any, A1RunContext]] = []  # (cid, arch, cfg, ctx)
    for ep in EXITS:
        for sl in SLS:
            for cap in EXPS:
                cid = f"A1::full::{ep}::sl{sl:.1f}::exp{_exp_str(cap)}"
                cfg = A1Config(
                    config_id=cid, sl_atr_mult=sl, trail_enabled=False,
                    risk_pct=R_BASE, starting_balance=STARTING_BALANCE,
                    max_concurrent_per_currency=cap, max_concurrent_per_pair=1,
                    max_concurrent_total=None, exit_policy=ep,
                )
                configs.append((cid, A1Architecture(), cfg, base_ctx))

    # A3 — deferred entry on the strongest candidate V-shape cluster (secondary).
    a3_cluster = None
    if candidate_clusters and s4 is not None:
        a3_cluster = max(candidate_clusters, key=lambda c: c.capturability_composite)
        log.info("A3 secondary on cluster %d (%s, composite %.3f)",
                 a3_cluster.cluster_id, a3_cluster.shape_tag, a3_cluster.capturability_composite)
        try:
            a3_inputs = PerFoldTrainingInputs(
                pool_trades=pool.trades, pool_paths=pool.paths,
                cluster_assignments=step2.cluster_assignments, panels=panels,
                primary_tf=signal_module.primary_tf, candidate_cluster_id=a3_cluster.cluster_id,
                n_defer=N_DEFER,
            )
            folds_for_fits = all_search_folds + ([wfo.holdout] if wfo.holdout else [])
            a3_fits = build_path_classifier_fits_per_fold(inputs=a3_inputs, folds=folds_for_fits, arch="A3")
            entry_feats = build_per_trade_entry_features(PerFoldTrainingInputs(
                pool_trades=pool.trades, pool_paths=pool.paths,
                cluster_assignments=step2.cluster_assignments, panels=panels,
                primary_tf=signal_module.primary_tf, candidate_cluster_id=None, n_defer=N_DEFER,
            ))
            a3_ctx = A1RunContext(
                per_trade_features=per_trade_features,
                per_trade_entry_features=entry_feats,
                path_classifier_fits=a3_fits,
            )
            for ep in EXITS:
                for sl in SLS:
                    for cap in EXPS:
                        cid = f"A3::cl{a3_cluster.cluster_id}::{ep}::sl{sl:.1f}::n{N_DEFER}::exp{_exp_str(cap)}"
                        cfg = A3Config(
                            config_id=cid, n_defer=N_DEFER, sl_atr_mult=sl, trail_enabled=False,
                            risk_pct=R_BASE, starting_balance=STARTING_BALANCE,
                            max_concurrent_per_currency=cap, max_concurrent_per_pair=1,
                            max_concurrent_total=None, exit_policy=ep,
                        )
                        configs.append((cid, A3Architecture(), cfg, a3_ctx))
        except Exception as exc:  # noqa: BLE001
            log.warning("A3 grid build failed (non-blocking): %s\n%s", exc, traceback.format_exc())
    else:
        log.warning("A3 skipped — no candidate clusters or no Step 4 result")

    log.info("Step 5 — %d configs in grid", len(configs))

    # ── Stage A: triage on F1/F6/F8 (all configs) ─────────────────────────
    triage_folds = [fold_by_id[i] for i in TRIAGE_FOLD_IDS if i in fold_by_id]
    log.info("Stage A — triage %d configs × folds %s", len(configs), [f.fold_id for f in triage_folds])
    stage_a_rows: list[dict] = []
    t_a = time.perf_counter()
    for i, (cid, arch, cfg, ctx) in enumerate(configs, 1):
        res = _run_config_over_folds(arch, cfg, ctx, pool.signal_evaluation, panels, triage_folds)
        # worst-of-N across folds where the config actually traded (n>0)
        traded = {fid: s for fid, (s, _) in res.items() if s.n_trades > 0}
        if traded:
            worst_roi = min(s.roi_pct for s in traded.values())
            worst_dd_base = max(s.max_dd_pct for s in traded.values())
            ratios = [(s.roi_pct / s.max_dd_pct) if s.max_dd_pct > 1e-9 else (np.inf if s.roi_pct > 0 else 0.0)
                      for s in traded.values()]
            worst_ratio = min(ratios)
            min_trades = min(s.n_trades for s in traded.values())
            dd_screen = _dd_at_operating(max(s.max_dd_pct for s in traded.values()))
            screened_out = dd_screen > DD_SCREEN_PCT
        else:
            worst_roi = worst_dd_base = worst_ratio = 0.0
            min_trades = 0
            dd_screen = 0.0
            screened_out = True  # no trades on any triage fold
        stage_a_rows.append(dict(
            config_id=cid, arch=cfg.__class__.__name__.replace("Config", ""),
            exit_policy=getattr(cfg, "exit_policy", None), sl=cfg.sl_atr_mult,
            exp=_exp_str(getattr(cfg, "max_concurrent_per_currency", None)),
            n_triage_folds_traded=len(traded),
            worst_roi_pct=worst_roi, worst_dd_base_pct=worst_dd_base,
            worst_dd_at_0p40_pct=dd_screen, worst_ratio=worst_ratio,
            min_trades=min_trades, screened_out=bool(screened_out),
            **{f"f{fid}_roi": (res[fid][0].roi_pct if fid in res else np.nan) for fid in TRIAGE_FOLD_IDS},
            **{f"f{fid}_dd": (res[fid][0].max_dd_pct if fid in res else np.nan) for fid in TRIAGE_FOLD_IDS},
            **{f"f{fid}_n": (res[fid][0].n_trades if fid in res else 0) for fid in TRIAGE_FOLD_IDS},
        ))
        if i % 6 == 0 or i == len(configs):
            el = time.perf_counter() - t_a
            log.info("  Stage A %d/%d (%.0fs, avg %.1fs/cfg)", i, len(configs), el, el / i)
    stage_a = pd.DataFrame(stage_a_rows)
    s5_dir = out / "step_5"
    s5_dir.mkdir(exist_ok=True)
    stage_a.sort_values(["screened_out", "worst_ratio"], ascending=[True, False]).to_csv(
        s5_dir / "stage_a_triage.csv", index=False, lineterminator="\n")

    # Rank surviving configs by worst-of-3 ratio; widen top-3 -> top-5 if within ~10%.
    survivors = stage_a[~stage_a["screened_out"]].copy().sort_values("worst_ratio", ascending=False)
    if survivors.empty:
        log.warning("Stage A — ALL configs screened out; taking top-3 by ratio anyway for Stage B record")
        survivors = stage_a.sort_values("worst_ratio", ascending=False)
    top_n = args.top_k
    if len(survivors) > args.top_k:
        r = survivors["worst_ratio"].to_numpy()
        if r[0] > 0 and (r[args.top_k - 1] - r[args.top_k]) / max(abs(r[0]), 1e-9) < 0.10:
            top_n = min(5, len(survivors))
            log.info("Stage A — top-3 within ~10%% noise; widening Stage B to top-%d", top_n)
    stage_b_cids = list(survivors["config_id"].head(top_n))
    log.info("Stage A -> Stage B finalists: %s", stage_b_cids)

    # ── Stage B: full 11-fold + holdout on finalists; Amendment-3 gate ─────
    cfg_by_cid = {cid: (arch, cfg, ctx) for cid, arch, cfg, ctx in configs}
    stage_b_results: list[dict] = []
    per_fold_rows: list[dict] = []
    pdmdd_written: dict[str, str] = {}
    for cid in stage_b_cids:
        arch, cfg, ctx = cfg_by_cid[cid]
        log.info("Stage B — %s (full 11-fold + holdout)", cid)
        folds_b = all_search_folds + ([wfo.holdout] if wfo.holdout else [])
        res = _run_config_over_folds(arch, cfg, ctx, pool.signal_evaluation, panels, folds_b)
        search_stats = [res[f.fold_id][0] for f in all_search_folds]
        holdout_stats = res[wfo.holdout.fold_id][0] if wfo.holdout else None
        # chained DD from stitched OOS equity (search folds + holdout)
        equities = [res[f.fold_id][1] for f in folds_b if len(res[f.fold_id][1]) > 0]
        chained_eq = stitch_per_fold_oos_equity(equities, starting_balance=STARTING_BALANCE) if equities else pd.Series(dtype=float)
        chained_dd = compute_chained_max_dd_from_continuous_equity(chained_eq) if len(chained_eq) else 0.0
        per_day = compute_per_day_max_dd(chained_eq, pair_set=",".join(PAIRS_28)) if len(chained_eq) else pd.DataFrame()
        pdmdd_path = None
        if not per_day.empty:
            safe = cid.replace("::", "__").replace("/", "_")
            pdmdd_path = s5_dir / f"per_day_max_dd__{safe}.parquet"
            per_day.to_parquet(pdmdd_path, engine="pyarrow", compression="snappy", index=False)
            pdmdd_written[cid] = str(pdmdd_path)
        # Amendment-3 scaling from worst-fold DD@r_base
        worst_dd_base = max((s.max_dd_pct for s in search_stats), default=0.0)
        _ = compute_scaling_factors(worst_dd_base, r_base=R_BASE)  # scaling recomputed inside the gate
        gate = classify_amended_fold_stats(
            folds=search_stats, chained_max_dd_base_pct=chained_dd,
            per_day_max_dd_df=per_day if not per_day.empty else None,
            holdout_stats_at_r_safe=None, holdout_stats_at_r_hard=None,
            sizing_convention="reset_floor", accept_equity_pct=False, r_base=R_BASE,
        )
        for f in folds_b:
            s = res[f.fold_id][0]
            per_fold_rows.append(dict(
                config_id=cid, fold_id=f.fold_id,
                segment=("holdout" if (wfo.holdout and f.fold_id == wfo.holdout.fold_id) else "search"),
                calendar_year=(2009 + f.fold_id if f.fold_id <= 11 else 2021),
                n_trades=s.n_trades, roi_pct=s.roi_pct, max_dd_base_pct=s.max_dd_pct,
                dd_at_0p40_pct=_dd_at_operating(s.max_dd_pct),
                days_breaching_daily_5pct=s.days_breaching_daily_5pct, roi_dd_ratio=s.roi_dd_ratio,
            ))
        stage_b_results.append(dict(
            config_id=cid, verdict=gate.verdict.value,
            primary_failure_mode=getattr(gate.primary_failure_mode, "value", str(gate.primary_failure_mode)),
            worst_fold_ratio=gate.worst_fold_ratio, mean_fold_ratio=gate.mean_fold_ratio,
            worst_fold_roi_base_pct=gate.worst_fold_roi_base_pct,
            worst_fold_dd_base_pct=gate.worst_fold_dd_base_pct,
            worst_fold_dd_at_0p40_pct=_dd_at_operating(gate.worst_fold_dd_base_pct),
            n_negative_folds=gate.n_negative_folds, min_trades_per_fold=gate.min_trades_per_fold,
            chained_max_dd_base_pct=chained_dd,
            k_safe=gate.k_safe, r_safe_pct=gate.r_safe_pct,
            scalable_to_safe=gate.scalable_to_safe, scalable_to_hard=gate.scalable_to_hard,
            worst_fold_roi_at_r_safe_pct=gate.worst_fold_roi_at_r_safe_pct,
            chained_max_dd_at_r_safe_pct=gate.chained_max_dd_at_r_safe_pct,
            daily_dd_breaches_at_r_safe=gate.daily_dd_breaches_at_r_safe,
            holdout_roi_pct=(holdout_stats.roi_pct if holdout_stats else None),
            holdout_dd_base_pct=(holdout_stats.max_dd_pct if holdout_stats else None),
            holdout_n_trades=(holdout_stats.n_trades if holdout_stats else None),
            per_day_max_dd_path=pdmdd_written.get(cid),
        ))

    pd.DataFrame(per_fold_rows).to_csv(s5_dir / "per_fold.csv", index=False, lineterminator="\n")
    stage_b = pd.DataFrame(stage_b_results)
    # rank Stage B by verdict tier then worst-fold ratio
    tier = {"pass_deployable": 2, "pass_viable": 1, "fail": 0}
    stage_b["_tier"] = stage_b["verdict"].map(lambda v: tier.get(v, 0))
    stage_b = stage_b.sort_values(["_tier", "worst_fold_ratio"], ascending=[False, False]).drop(columns="_tier")
    stage_b.to_csv(s5_dir / "wfo_results.csv", index=False, lineterminator="\n")

    best = stage_b.iloc[0].to_dict() if len(stage_b) else {}
    final_verdict = str(best.get("verdict", "incomplete")).upper().replace("_", "-")
    log.info("Stage B — final verdict: %s (best %s)", final_verdict, best.get("config_id"))

    # ── run summary ───────────────────────────────────────────────────────
    summary = dict(
        arc_name="arc_7_v3.0_exit_extraction",
        ran_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        wall_time_seconds=time.perf_counter() - t_start,
        pool_size=int(len(pool.trades)), pool_sha256=pool.pool_sha256,
        window_start=str(pool_cfg.window_start), window_end=str(pool_cfg.window_end),
        boundary_convention="5ers_eet",
        k_selected=int(step2.k_selected),
        silhouette_per_k={int(k): float(v) for k, v in step2.silhouette_per_k.items()},
        candidate_clusters=[dict(cluster_id=int(c.cluster_id), shape_tag=c.shape_tag,
                                 composite=float(c.capturability_composite),
                                 selected_sl=float(c.selected_sl), mfe_p50=float(c.mfe_p50),
                                 reach_1r=float(c.reach_1r), wrong_way_pp=float(c.wrong_way_pp),
                                 n_trades=int(c.n_trades)) for c in candidate_clusters],
        step4_entry_auc=[dict(cluster_id=int(ce.cluster_id), best_classifier=ce.best_classifier,
                              best_auc=float(ce.best_classifier_mean_auc), n_trades=int(ce.n_trades))
                         for ce in (s4.per_cluster if s4 else [])],
        a3_cluster=(int(a3_cluster.cluster_id) if a3_cluster else None),
        n_configs=len(configs), n_a1=sum(1 for c in configs if c[0].startswith("A1")),
        n_a3=sum(1 for c in configs if c[0].startswith("A3")),
        triage_folds=list(TRIAGE_FOLD_IDS),
        n_screened_out=int(stage_a["screened_out"].sum()),
        stage_b_finalists=stage_b_cids,
        final_verdict=final_verdict, best_config_id=best.get("config_id"),
        stage_b=stage_b_results,
    )
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")

    _write_manifest(s5_dir / "manifest.json",
                    [s5_dir / "stage_a_triage.csv", s5_dir / "wfo_results.csv", s5_dir / "per_fold.csv",
                     *s5_dir.glob("per_day_max_dd__*.parquet")])
    log.info("DONE in %.1fs — verdict %s", time.perf_counter() - t_start, final_verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
