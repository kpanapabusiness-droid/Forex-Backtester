"""Arc 7 v3.0.2 driver — A2/A6 follow-up retry under architecture-map override.

Reuses Arc 7 v3.0.1's Step 1-4 artefacts at ``results/l_arc_7/``. Runs only
Step 5 onwards with the augmented architecture set:

  - c0 (Bimodal, LR AUC 0.6192 < 0.65): {A1, A4}  (unchanged from v3.0.1)
  - c1 (Unclassified, RF AUC 0.6642 ≥ 0.65): {A1, A2, A6}  (override augments)
  - A5 portfolio across both candidate clusters

Step 6 auto-dispatches on PASS-tier candidate per Amendment 4.

Per L_PROTOCOL §1: random_state=42, n_jobs=1, lineterminator='\\n'.

Invocation::

    python -m scripts.l_arc_7_v3_0_2.run \\
        --histdata-root C:/Users/panap/Documents/Forex-Backtester/data/histdata \\
        --cache-root    C:/Users/panap/Documents/Forex-Backtester/data/cache \\
        --out-root      results/l_arc_7_v3.0.2 \\
        --v301-root     results/l_arc_7 \\
        --wall-time-budget-h 5.5

Outputs ``results/l_arc_7_v3.0.2/`` per L_PROTOCOL §6 + Amendment 3 + Step 6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.arc.signal_protocol import (  # noqa: E402
    SignalEvaluation,
)
from core.architectures._protocol import Architecture, StrategyResult  # noqa: E402
from core.architectures.a1_system_level_filter import (  # noqa: E402
    A1Architecture,
    A1Config,
    A1RunContext,
)
from core.architectures.a2_classifier_filter import A2Architecture  # noqa: E402
from core.architectures.a4_pipeline_d_exits import A4Architecture  # noqa: E402
from core.architectures.a5_portfolio_composition import (  # noqa: E402
    A5Architecture,
    A5Config,
)
from core.architectures.a6_meta_labeling import A6Architecture  # noqa: E402
from core.data.aggregator import aggregate  # noqa: E402
from core.determinism import seed_everything  # noqa: E402
from core.runners._fold_stats_helpers import compute_per_day_max_dd  # noqa: E402
from core.runners.arc_fold_runner import ArcFoldRunner  # noqa: E402
from core.runners.oracle_fold_runner import OracleFoldRunner  # noqa: E402
from core.sim.panel import Panel  # noqa: E402
from core.step_6.dispatch import (  # noqa: E402
    maybe_dispatch_step_6,
    replace_top_1_with_step6_fail,
)
from core.steps.classifier_persistence import (  # noqa: E402
    build_a2_config_from_step4,
    build_a4_config_from_step4,
    build_a6_config_from_step4,
)
from core.steps.path_classifier_per_fold import (  # noqa: E402
    PerFoldTrainingInputs,
    build_path_classifier_fits_per_fold,
    build_per_trade_entry_features,
)
from core.steps.step_3_capturability import (  # noqa: E402
    SL_MULT_SWEEP,
    ClusterCapturability,
)
from core.steps.step_4_extraction import (  # noqa: E402
    ClusterExtraction,
    Step4Result,
)
from core.strategies.liquidity_sweep_reclaim_long.signal_module import (  # noqa: E402
    LiquiditySweepReclaimLongSignal,
)
from core.wfo.amended_gates import (  # noqa: E402
    AmendedGateResult,
    AmendedVerdict,
    classify_amended_fold_stats,
    compute_scaling_factors,
)
from core.wfo.chained_dd import (  # noqa: E402
    compute_chained_max_dd_from_continuous_equity,
    stitch_per_fold_oos_equity,
)
from core.wfo.folds import Fold, build_v3_folds  # noqa: E402
from core.wfo.gates import FoldStats  # noqa: E402
from core.wfo.holdout_rerun import rescale_arch_config_risk  # noqa: E402
from core.wfo.orchestrator import (  # noqa: E402
    CandidateSearchResult,
    WfoSearchResult,
    run_search,
)

PAIRS_28: tuple[str, ...] = (
    "AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY "
    "EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD "
    "GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY "
    "NZDUSD USDCAD USDCHF USDJPY"
).split()
assert len(PAIRS_28) == 28

AUC_DEPLOYABILITY_BAR = 0.65   # dispatch override threshold


def _setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity >= 1 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


log = logging.getLogger("arc7_v3_0_2")


# ──────────────────────────────────────────────────────────────────────
# Panel build (reuse 5ers_eet cache from v3.0.1)
# ──────────────────────────────────────────────────────────────────────


def _build_panel_5ers_eet(
    pairs: list[str], tf: str, *, histdata_root: Path, cache_root: Path,
) -> Panel:
    pair_dfs: dict[str, pd.DataFrame] = {}
    for p in sorted(pairs):
        pair_dfs[p] = aggregate(
            p, tf,
            histdata_root=histdata_root, cache_root=cache_root,
            boundary_convention="5ers_eet",
        )
    return Panel.from_frames(pair_dfs, tf=tf)


# ──────────────────────────────────────────────────────────────────────
# Reconstruct SignalEvaluation from the original signal module + panels
# (cheap — signal evaluation on the warm cache is ~30s)
# ──────────────────────────────────────────────────────────────────────


def _rebuild_signal_evaluation(panels: Mapping[str, Panel]) -> SignalEvaluation:
    """Re-evaluate the signal module on the same panels to produce a fresh
    SignalEvaluation. Step 1 pool is loaded from disk; SignalEvaluation
    drives Step 5 fold runners (signal_mask + atr per pair)."""
    signal_module = LiquiditySweepReclaimLongSignal()
    return signal_module.evaluate(panels)


# ──────────────────────────────────────────────────────────────────────
# Load v3.0.1 Step 1-4 artefacts
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class V301Artefacts:
    pool_trades: pd.DataFrame
    pool_paths: pd.DataFrame
    cluster_assignments: pd.DataFrame
    cluster_summary: pd.DataFrame
    capturability: pd.DataFrame    # Step 3 csv
    extraction_metrics: pd.DataFrame   # Step 4 csv
    feature_importance: pd.DataFrame   # Step 4 csv
    classifiers_manifest: dict
    candidate_clusters: list[ClusterCapturability]  # rebuilt from CSV
    step4_per_cluster: list[ClusterExtraction]   # rebuilt from CSV + manifest


def _load_v301(v301_root: Path) -> V301Artefacts:
    """Load v3.0.1 Step 1-4 artefacts from disk."""
    s1 = v301_root / "step_1"
    s2 = v301_root / "step_2"
    s3 = v301_root / "step_3"
    s4 = v301_root / "step_4"

    pool_trades = pd.read_parquet(s1 / "pool.parquet")
    pool_paths = pd.read_parquet(s1 / "paths.parquet")
    cluster_assignments = pd.read_parquet(s2 / "cluster_assignments.parquet")
    cluster_summary = pd.read_csv(s2 / "cluster_summary.csv")
    capturability = pd.read_csv(s3 / "capturability.csv")
    extraction_metrics = pd.read_csv(s4 / "extraction_metrics.csv")
    feature_importance = pd.read_csv(s4 / "feature_importance.csv")
    classifiers_manifest = json.loads((s4 / "classifiers" / "manifest.json").read_text(encoding="utf-8"))

    # Reconstruct candidate_clusters from capturability.csv
    candidate_clusters: list[ClusterCapturability] = []
    for _, r in capturability.iterrows():
        if not bool(r["is_candidate"]):
            continue
        cap = ClusterCapturability(
            cluster_id=int(r["cluster_id"]),
            n_trades=int(r["n_trades"]),
            shape_tag=str(r["shape_tag"]),
            reach_1r=float(r["reach_1r"]),
            reach_2r=float(r.get("reach_2r", 0)),
            reach_3r=float(r.get("reach_3r", 0)),
            mfe_p25=float(r.get("mfe_p25", 0)),
            mfe_p50=float(r["mfe_p50"]),
            mfe_p75=float(r.get("mfe_p75", 0)),
            mfe_p90=float(r.get("mfe_p90", 0)),
            wrong_way_pp=float(r["wrong_way_pp"]),
            ttp_p25=int(r.get("ttp_p25", 0)),
            ttp_p50=int(r.get("ttp_p50", 0)),
            ttp_p75=int(r.get("ttp_p75", 0)),
            mean_r=float(r.get("mean_r", 0)),
            final_r_p25=float(r.get("final_r_p25", 0)),
            final_r_p50=float(r.get("final_r_p50", 0)),
            sl_sweep={},  # not used downstream
            selected_sl=float(r["selected_sl"]),
            capturability_composite=float(r["capturability_composite"]),
            is_candidate=True,
        )
        candidate_clusters.append(cap)

    # Reconstruct Step4Result.per_cluster from extraction_metrics + classifiers manifest
    step4_per_cluster: list[ClusterExtraction] = []
    for cid_str, entry in classifiers_manifest.get("classifiers", {}).items():
        cid = int(cid_str)
        # Per-fold AUC for this cluster's best classifier (introspection only)
        best_name = entry.get("classifier_name", "rf")
        # Top features for this cluster — from feature_importance.csv
        fi = feature_importance[feature_importance["cluster_id"] == cid]
        top_feats_agg = (
            fi.groupby("feature")["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "mean_importance", "std": "std_importance"})
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )
        ce = ClusterExtraction(
            cluster_id=cid,
            n_trades=int(entry.get("trained_on_pool_size", 0)),
            excluded_features=(),
            used_features=tuple(entry.get("feature_order", [])),
            classifier_fold_results=(),
            best_classifier=best_name,
            best_classifier_mean_auc=float(entry.get("auc_oos_cv5", 0)),
            best_threshold=float(entry.get("best_threshold", 0.5)),
            feature_importance=top_feats_agg,
            fitted_classifier_path=Path(entry["path"]) if Path(entry["path"]).is_absolute()
                else (s4 / "classifiers" / entry["path"]),
            fitted_classifier_type=entry.get("classifier_type"),
            fitted_classifier_feature_order=tuple(entry.get("feature_order", [])),
        )
        step4_per_cluster.append(ce)

    return V301Artefacts(
        pool_trades=pool_trades,
        pool_paths=pool_paths,
        cluster_assignments=cluster_assignments,
        cluster_summary=cluster_summary,
        capturability=capturability,
        extraction_metrics=extraction_metrics,
        feature_importance=feature_importance,
        classifiers_manifest=classifiers_manifest,
        candidate_clusters=candidate_clusters,
        step4_per_cluster=step4_per_cluster,
    )


# ──────────────────────────────────────────────────────────────────────
# Step 5 candidate-grid builders (same shape as v3.0.1 driver)
# ──────────────────────────────────────────────────────────────────────


def _sl_window(selected_sl: float) -> tuple[float, ...]:
    sweep = list(SL_MULT_SWEEP)
    try:
        idx = sweep.index(selected_sl)
    except ValueError:
        idx = int(np.argmin([abs(s - selected_sl) for s in sweep]))
    lo = max(0, idx - 1)
    hi = min(len(sweep), idx + 2)
    return tuple(sweep[lo:hi])


def _exposure_choices() -> tuple[int | None, ...]:
    return (2, None)


def _build_a1_configs(cid: int, selected_sl: float) -> list[tuple[str, tuple[A1Architecture, A1Config]]]:
    out: list[tuple[str, tuple[A1Architecture, A1Config]]] = []
    for sl in _sl_window(selected_sl):
        for trail in (True, False):
            for cap in _exposure_choices():
                cid_str = f"A1::cl{cid}::sl{sl:.1f}::trail{int(trail)}::exp{cap if cap else 'inf'}"
                cfg = A1Config(
                    config_id=cid_str,
                    sl_atr_mult=sl, trail_enabled=trail, risk_pct=0.005,
                    max_concurrent_per_currency=cap, max_concurrent_per_pair=1,
                    max_concurrent_total=None,
                )
                out.append((f"A1::{cid_str}", (A1Architecture(), cfg)))
    return out


def _build_a2_configs(
    cid: int, selected_sl: float, s4: Step4Result
) -> list[tuple[str, tuple[A2Architecture, Any]]]:
    out: list[tuple[str, tuple[A2Architecture, Any]]] = []
    for sl in _sl_window(selected_sl):
        for cap in _exposure_choices():
            try:
                cid_str = f"A2::cl{cid}::sl{sl:.1f}::exp{cap if cap else 'inf'}"
                cfg = build_a2_config_from_step4(
                    s4, cluster_id=cid, config_id=cid_str,
                    sl_atr_mult=sl, trail_enabled=True, risk_pct=0.005,
                    max_concurrent_per_currency=cap, max_concurrent_per_pair=1,
                    max_concurrent_total=None,
                )
            except ValueError as exc:
                log.warning("A2 cluster %d skipped: %s", cid, exc)
                return []
            out.append((f"A2::{cid_str}", (A2Architecture(), cfg)))
    return out


def _build_a4_configs(
    cid: int, selected_sl: float, s4: Step4Result
) -> list[tuple[str, tuple[A4Architecture, Any]]]:
    out: list[tuple[str, tuple[A4Architecture, Any]]] = []
    for sl in _sl_window(selected_sl):
        for thr in (0.3, 0.4, 0.5):
            for cap in _exposure_choices():
                try:
                    cid_str = f"A4::cl{cid}::sl{sl:.1f}::thr{thr:.1f}::exp{cap if cap else 'inf'}"
                    cfg = build_a4_config_from_step4(
                        s4, cluster_id=cid, exit_threshold=thr, config_id=cid_str,
                        sl_atr_mult=sl, trail_enabled=True, risk_pct=0.005,
                        max_concurrent_per_currency=cap, max_concurrent_per_pair=1,
                        max_concurrent_total=None,
                    )
                except ValueError as exc:
                    log.warning("A4 cluster %d skipped: %s", cid, exc)
                    return []
                out.append((f"A4::{cid_str}", (A4Architecture(), cfg)))
    return out


def _build_a6_configs(
    cid: int, selected_sl: float, s4: Step4Result
) -> list[tuple[str, tuple[A6Architecture, Any]]]:
    out: list[tuple[str, tuple[A6Architecture, Any]]] = []
    for sl in _sl_window(selected_sl):
        for lo, hi in ((0.3, 0.5), (0.4, 0.6), (0.5, 0.7)):
            for cap in _exposure_choices():
                try:
                    cid_str = f"A6::cl{cid}::sl{sl:.1f}::thr{lo:.1f}-{hi:.1f}::exp{cap if cap else 'inf'}"
                    cfg = build_a6_config_from_step4(
                        s4, cluster_id=cid, lower_threshold=lo, upper_threshold=hi,
                        config_id=cid_str, sl_atr_mult=sl, trail_enabled=True,
                        risk_pct=0.005, max_concurrent_per_currency=cap,
                        max_concurrent_per_pair=1, max_concurrent_total=None,
                    )
                except ValueError as exc:
                    log.warning("A6 cluster %d skipped: %s", cid, exc)
                    return []
                out.append((f"A6::{cid_str}", (A6Architecture(), cfg)))
    return out


# ──────────────────────────────────────────────────────────────────────
# Amendment 3 evaluation (composed canonical primitives)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateAmendedResult:
    config_id: str
    chained_max_dd_base_pct: float
    chained_dd_method: str
    per_day_max_dd_artefact_path: Path | None
    amended_gate: AmendedGateResult


@dataclass(frozen=True)
class AmendedWfoSearchResult:
    base: WfoSearchResult
    amended_results: tuple[CandidateAmendedResult, ...]


def _run_amendment_3_for_candidate(
    *, cand: CandidateSearchResult,
    strategy_results_by_cid_fold: Mapping[str, Mapping[int, StrategyResult]],
    holdout_result_by_cid: Mapping[str, StrategyResult],
    panels: Mapping[str, Panel],
    signal_eval: SignalEvaluation,
    per_candidate_ctx: Mapping[str, A1RunContext],
    base_ctx: A1RunContext,
    pair_set: tuple[str, ...], out_dir: Path,
    r_base: float, holdout_fold: Fold | None,
) -> CandidateAmendedResult:
    cid = cand.config_id
    per_fold_equity: list[pd.Series] = []
    for f_stats in cand.fold_stats:
        sr = strategy_results_by_cid_fold.get(cid, {}).get(f_stats.fold_id)
        if sr is not None and len(sr.equity_curve) > 0:
            per_fold_equity.append(sr.equity_curve)
    h_sr = holdout_result_by_cid.get(cid)
    if h_sr is not None and len(h_sr.equity_curve) > 0:
        per_fold_equity.append(h_sr.equity_curve)

    arch, arch_config = cand.config
    starting_balance = float(getattr(arch_config, "starting_balance", 100_000.0))
    chained_equity = stitch_per_fold_oos_equity(per_fold_equity, starting_balance=starting_balance)
    chained_dd = compute_chained_max_dd_from_continuous_equity(chained_equity)

    per_day_df = compute_per_day_max_dd(chained_equity, pair_set=",".join(pair_set))
    parquet_path: Path | None = None
    if not per_day_df.empty:
        safe_cid = cid.replace("::", "__").replace("/", "_")
        parquet_path = out_dir / "step_5" / f"per_day_max_dd_base__{safe_cid}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        per_day_df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

    worst_fold_dd_base = max((f.max_dd_pct for f in cand.fold_stats), default=0.0)
    scaling = compute_scaling_factors(worst_fold_dd_base, r_base=r_base)

    def _rerun_at_scale(k: float) -> FoldStats | None:
        if holdout_fold is None or k == float("inf") or k <= 0:
            return None
        try:
            scaled_conf = rescale_arch_config_risk(arch_config, k_scale=k)
        except (TypeError, ValueError):
            return None
        ctx = per_candidate_ctx.get(cid, base_ctx)
        runner = ArcFoldRunner(
            architecture=arch, signal_evaluation=signal_eval,
            panels=panels, run_context=ctx,
        )
        return runner(holdout_fold, scaled_conf)

    holdout_safe = _rerun_at_scale(scaling.k_safe) if scaling.scalable_to_safe else None
    holdout_hard = _rerun_at_scale(scaling.k_hard) if scaling.scalable_to_hard else None

    sizing_convention = str(getattr(arch_config, "sizing_convention", "reset_floor"))
    amended_gate = classify_amended_fold_stats(
        folds=cand.fold_stats,
        chained_max_dd_base_pct=chained_dd,
        per_day_max_dd_df=per_day_df if not per_day_df.empty else None,
        holdout_stats_at_r_safe=holdout_safe,
        holdout_stats_at_r_hard=holdout_hard,
        sizing_convention=sizing_convention,
        accept_equity_pct=False,
        r_base=r_base,
    )

    return CandidateAmendedResult(
        config_id=cid, chained_max_dd_base_pct=chained_dd,
        chained_dd_method="equity_stitching",
        per_day_max_dd_artefact_path=parquet_path,
        amended_gate=amended_gate,
    )


def _amended_verdict_rank(v: AmendedVerdict) -> int:
    if v == AmendedVerdict.PASS_DEPLOYABLE:
        return 2
    if v == AmendedVerdict.PASS_VIABLE:
        return 1
    return 0


# Light view for Step 6 dispatch
@dataclass(frozen=True)
class _LightOrchestratorView:
    arc_name: str
    pool: Any
    step_4: Any
    wfo_search: Any
    amended_wfo: Any


@dataclass(frozen=True)
class _FauxStep4:
    """Bundle Step 4 reconstruction for the Step 6 dispatch's IO builder."""

    per_cluster: tuple[ClusterExtraction, ...]


@dataclass(frozen=True)
class _FauxPool:
    """Bundle Step 1 reconstruction for the Step 6 dispatch's IO builder."""

    trades: pd.DataFrame
    paths: pd.DataFrame
    pool_sha256: str = ""
    signal_evaluation: Any = None


# ──────────────────────────────────────────────────────────────────────
# Manifest helpers
# ──────────────────────────────────────────────────────────────────────


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(manifest_path: Path, artefacts: list[Path]) -> None:
    root = manifest_path.parent
    entries: dict[str, dict] = {}
    for a in sorted(artefacts):
        if not a.exists():
            continue
        rel = str(a.relative_to(root)).replace("\\", "/")
        entries[rel] = {"sha256": _file_sha256(a), "size_bytes": a.stat().st_size}
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artefacts": entries,
    }
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


# ──────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histdata-root", type=Path, default=Path("data/histdata"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--out-root", type=Path, default=Path("results/l_arc_7_v3.0.2"))
    parser.add_argument("--v301-root", type=Path, default=Path("results/l_arc_7"))
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--wall-time-budget-h", type=float, default=5.5)
    parser.add_argument("--window-end", type=str, default="2026-04-30")
    parser.add_argument("--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    seed_everything(42)
    t_start = time.perf_counter()

    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    v301_root = args.v301_root.resolve()

    log.info("Arc 7 v3.0.2 driver — out=%s, v301_root=%s", out, v301_root)

    # ── Load v3.0.1 artefacts ────────────────────────────────────────
    log.info("Loading v3.0.1 Step 1-4 artefacts ...")
    t0 = time.perf_counter()
    v301 = _load_v301(v301_root)
    log.info(
        "Loaded %d trades, %d candidate clusters, %d Step-4 extractions (%.1fs)",
        len(v301.pool_trades), len(v301.candidate_clusters),
        len(v301.step4_per_cluster), time.perf_counter() - t0,
    )

    # Step4Result reconstruction (for build_a{2,4,6}_config_from_step4)
    s4_result = Step4Result(
        per_cluster=tuple(v301.step4_per_cluster),
        extraction_metrics=v301.extraction_metrics,
        feature_importance=v301.feature_importance,
        summary_md="(reused from v3.0.1)",
    )

    # ── Build panels (warm cache for 5ers_eet from v3.0.1) ───────────
    log.info("Building H4/D1/W1 panels (5ers_eet, warm cache from v3.0.1)")
    t0 = time.perf_counter()
    panel_h4 = _build_panel_5ers_eet(list(PAIRS_28), "H4", histdata_root=args.histdata_root, cache_root=args.cache_root)
    panel_d1 = _build_panel_5ers_eet(list(PAIRS_28), "D1", histdata_root=args.histdata_root, cache_root=args.cache_root)
    panel_w1 = _build_panel_5ers_eet(list(PAIRS_28), "W1", histdata_root=args.histdata_root, cache_root=args.cache_root)
    object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})
    panels: dict[str, Panel] = {"H4": panel_h4, "D1": panel_d1, "W1": panel_w1}
    log.info("Panels built in %.1fs", time.perf_counter() - t0)

    # ── Re-evaluate signal (cheap; needed for Step 5 fold runners) ───
    log.info("Re-evaluating signal on panels ...")
    t0 = time.perf_counter()
    signal_module = LiquiditySweepReclaimLongSignal()
    signal_eval = signal_module.evaluate(panels)
    log.info("Signal evaluation in %.1fs", time.perf_counter() - t0)

    # Build a FauxPool that wraps trades/paths for Step 6 dispatch consumption
    faux_pool = _FauxPool(
        trades=v301.pool_trades, paths=v301.pool_paths,
        pool_sha256=v301.classifiers_manifest.get("pool_sha256", ""),
        signal_evaluation=signal_eval,
    )

    # ── Architecture-map override resolution ─────────────────────────
    log.info("Applying architecture-map override (any cluster AUC >= %s adds A2+A6)", AUC_DEPLOYABILITY_BAR)
    cluster_arch_sets: dict[int, tuple[str, ...]] = {}
    cluster_step4: dict[int, ClusterExtraction] = {ce.cluster_id: ce for ce in v301.step4_per_cluster}

    archetype_default = {
        "bimodal": ("A1", "A4"),
        "unclassified": ("A1",),
        "v_shape_recovery": ("A1", "A3", "A6"),
        "stepwise_climber": ("A1", "A2", "A4"),
        "monotonic_up": ("A1", "A2", "A6"),
    }
    for cc in v301.candidate_clusters:
        base_archs = list(archetype_default.get(cc.shape_tag, ("A1",)))
        ce = cluster_step4.get(cc.cluster_id)
        if ce is not None and ce.best_classifier_mean_auc >= AUC_DEPLOYABILITY_BAR:
            for aug in ("A2", "A6"):
                if aug not in base_archs:
                    base_archs.append(aug)
        cluster_arch_sets[cc.cluster_id] = tuple(base_archs)
        log.info(
            "  cluster %d (%s, AUC %.4f) → %s",
            cc.cluster_id, cc.shape_tag, ce.best_classifier_mean_auc if ce else float("nan"),
            base_archs,
        )

    # ── Per-fold path-classifier fits (A4 only — c0 needs it; A3 unused) ──
    wfo_struct = build_v3_folds(holdout_end=pd.Timestamp(args.window_end).date())
    folds_for_fits = wfo_struct.folds
    if wfo_struct.holdout is not None:
        folds_for_fits = folds_for_fits + (wfo_struct.holdout,)

    a4_fits: Mapping[int, Any] | None = None
    if any("A4" in archs for archs in cluster_arch_sets.values()):
        log.info("Building A4 per-fold path-classifier fits (cluster-independent)")
        t0 = time.perf_counter()
        a4_inputs = PerFoldTrainingInputs(
            pool_trades=v301.pool_trades, pool_paths=v301.pool_paths,
            cluster_assignments=None, panels=panels,
            primary_tf="H4", candidate_cluster_id=None, n_defer=5,
        )
        a4_fits = build_path_classifier_fits_per_fold(
            inputs=a4_inputs, folds=folds_for_fits, arch="A4",
        )
        log.info("A4 fits built in %.1fs", time.perf_counter() - t0)

    # ── Build per_trade entry features for A4 ────────────────────────
    entry_features = None
    if a4_fits is not None:
        log.info("Building per-trade entry features for A4 runtime")
        ef_inputs = PerFoldTrainingInputs(
            pool_trades=v301.pool_trades, pool_paths=v301.pool_paths,
            cluster_assignments=None, panels=panels,
            primary_tf="H4", candidate_cluster_id=None, n_defer=5,
        )
        entry_features = build_per_trade_entry_features(ef_inputs)

    # ── Build per_trade features for A2/A6 admit gates ───────────────
    # A2/A6 read per_trade_features[(pair, signal_time)] -> dict of features.
    # The 15-feature path-classifier schema (for A4) is different from the
    # 27-feature Step 4 catalogue (for A2/A6). A2/A6 consume the classifier's
    # feature_order which is the Step 4 lineage-clean catalogue.
    # We reconstruct the Step 1 per-trade feature lookup from the persisted
    # classifier's feature_order applied to each trade's signal_time.
    log.info("Reconstructing per-trade feature lookup for A2/A6 admit gates ...")
    t0 = time.perf_counter()
    per_trade_features = _build_per_trade_features_for_a2_a6(
        pool_trades=v301.pool_trades,
        signal_eval=signal_eval,
        panel_h4=panel_h4,
        feature_order=v301.step4_per_cluster[0].fitted_classifier_feature_order
        if v301.step4_per_cluster else (),
    )
    log.info("Per-trade features built (%d trades) in %.1fs",
             len(per_trade_features), time.perf_counter() - t0)

    base_ctx = A1RunContext(per_trade_features=per_trade_features)

    # ── Build Step 5 candidate grid ──────────────────────────────────
    log.info("Building Step 5 candidate grid (augmented per override)")
    candidates_for_search: list[tuple[str, tuple[Architecture, Any]]] = []
    per_candidate_ctx: dict[str, A1RunContext] = {}
    skipped_configs_doc: list[str] = []

    for cc in v301.candidate_clusters:
        archs = cluster_arch_sets[cc.cluster_id]
        sl_window = list(_sl_window(cc.selected_sl))
        log.info(
            "  cluster %d (%s) → %s, sl_window %s",
            cc.cluster_id, cc.shape_tag, archs, sl_window,
        )

        if "A1" in archs:
            a1s = _build_a1_configs(cc.cluster_id, cc.selected_sl)
            for cid_full, paired in a1s:
                candidates_for_search.append((cid_full, paired))
                per_candidate_ctx[cid_full] = base_ctx

        if "A2" in archs:
            a2s = _build_a2_configs(cc.cluster_id, cc.selected_sl, s4_result)
            if not a2s:
                skipped_configs_doc.append(f"A2/cluster{cc.cluster_id}: no fitted classifier")
            for cid_full, paired in a2s:
                candidates_for_search.append((cid_full, paired))
                per_candidate_ctx[cid_full] = base_ctx

        if "A4" in archs:
            a4s = _build_a4_configs(cc.cluster_id, cc.selected_sl, s4_result)
            if a4s and a4_fits is not None and entry_features is not None:
                a4_ctx = A1RunContext(
                    per_trade_features=per_trade_features,
                    per_trade_entry_features=entry_features,
                    path_classifier_fits=a4_fits,
                )
                for cid_full, paired in a4s:
                    candidates_for_search.append((cid_full, paired))
                    per_candidate_ctx[cid_full] = a4_ctx

        if "A6" in archs:
            a6s = _build_a6_configs(cc.cluster_id, cc.selected_sl, s4_result)
            if not a6s:
                skipped_configs_doc.append(f"A6/cluster{cc.cluster_id}: no fitted classifier")
            for cid_full, paired in a6s:
                candidates_for_search.append((cid_full, paired))
                per_candidate_ctx[cid_full] = base_ctx

    skipped_configs_doc.append(
        "Exit policies sl_plus_tp_2r, sl_plus_tp_3r, sl_partial_close_1r_runner_trail, "
        "sl_plus_trailing_swing, time_exit_n_bars: NOT TESTED — v3 A1/A2/A6 Configs "
        "don't expose tp_atr_mult / partial-close mechanics. Tested trail axis "
        "trail_enabled ∈ {True (sl+trailing_atr), False (sl_only)}."
    )

    log.info("Step 5 — %d configs for search (pre-A5)", len(candidates_for_search))
    n_configs = len(candidates_for_search)

    # ── Run search ──────────────────────────────────────────────────
    eligible_folds = [f for f in wfo_struct.folds if f.is_days >= 365]
    n_total = len(candidates_for_search) * len(eligible_folds)
    log.info("Step 5 — running %d (candidate × fold) pairs", n_total)
    t_search = time.perf_counter()
    n_done = 0
    wall_budget_s = args.wall_time_budget_h * 3600

    strategy_results_by_cid_fold: dict[str, dict[int, StrategyResult]] = {}

    def _runner(fold: Fold, paired: tuple[Architecture, Any]) -> FoldStats:
        nonlocal n_done
        arch, conf = paired
        cid = f"{arch.architecture_name}::{getattr(conf, 'config_id', repr(conf))}"
        ctx = per_candidate_ctx.get(cid, base_ctx)
        try:
            r = ArcFoldRunner(
                architecture=arch, signal_evaluation=signal_eval,
                panels=panels, run_context=ctx,
            )
            stats = r(fold, conf)
            if r.last_result is not None:
                strategy_results_by_cid_fold.setdefault(cid, {})[fold.fold_id] = r.last_result
        except Exception as exc:  # noqa: BLE001
            log.warning("candidate %s fold %d errored: %s", cid, fold.fold_id, exc)
            stats = FoldStats(
                fold_id=fold.fold_id, n_trades=0, roi_pct=0.0,
                max_dd_pct=0.0, days_breaching_daily_5pct=0, roi_dd_ratio=0.0,
            )
        n_done += 1
        if n_done % 10 == 0 or n_done == n_total:
            el = time.perf_counter() - t_search
            avg = el / max(1, n_done)
            eta = avg * (n_total - n_done)
            log.info("  WFO progress %d/%d (avg %.1fs/run, ETA %.0fs)", n_done, n_total, avg, eta)
        if (time.perf_counter() - t_start) > wall_budget_s:
            raise RuntimeError(f"Wall-time budget ({args.wall_time_budget_h}h) exceeded mid-search")
        return stats

    try:
        s5: WfoSearchResult = run_search(
            wfo_struct, candidates_for_search,
            fold_runner=_runner, min_is_days=365, top_k=3,
        )
    except RuntimeError as exc:
        log.error("Step 5 search aborted: %s", exc)
        return _emergency_halt(out, faux_pool, "wall_budget_exceeded_step_5")

    # ── A5 portfolio composition (≥ 2 candidate clusters) ────────────
    a5_candidates: list[tuple[str, tuple[A5Architecture, A5Config]]] = []
    a5_strategy_per_fold: dict[str, dict[int, StrategyResult]] = {}
    if len(v301.candidate_clusters) >= 2:
        log.info("A5 portfolio composition — combining top-1 per cluster per fold")
        # Pick best per-cluster A1/A2/A4/A6 architecture per cluster
        per_cluster_best: dict[int, str] = {}
        for cc in v301.candidate_clusters:
            cands_for_cluster = [
                cand for cand in s5.candidates
                if f"::cl{cc.cluster_id}::" in cand.config_id
            ]
            if cands_for_cluster:
                best = max(cands_for_cluster, key=lambda c: c.gate.worst_fold_ratio)
                per_cluster_best[cc.cluster_id] = best.config_id
        if len(per_cluster_best) >= 2:
            log.info("  A5 constituents per cluster: %s", per_cluster_best)
            # Build per-fold A5 StrategyResult by composing constituents
            a5_id = "A5::portfolio::" + "+".join(
                per_cluster_best[k].split("::", 2)[-1]
                for k in sorted(per_cluster_best.keys())
            )
            a5_arch = A5Architecture()
            a5_fold_stats: list[FoldStats] = []
            for fold in eligible_folds:
                constituents = []
                for cid_target in per_cluster_best.values():
                    sr = strategy_results_by_cid_fold.get(cid_target, {}).get(fold.fold_id)
                    if sr is not None:
                        constituents.append(sr)
                if len(constituents) < 2:
                    continue
                a5_cfg = A5Config(
                    config_id=a5_id, constituents=tuple(constituents),
                    starting_balance=100_000.0,
                )
                a5_result = a5_arch.run(
                    arch_config=a5_cfg, config_id=a5_id, fold=fold,
                )
                a5_strategy_per_fold.setdefault(a5_id, {})[fold.fold_id] = a5_result
                a5_fold_stats.append(a5_result.fold_stats)
            # Add A5 as a synthetic CandidateSearchResult into s5.top_k
            if a5_fold_stats:
                from core.wfo.gates import classify_fold_stats
                a5_gate = classify_fold_stats(tuple(a5_fold_stats))
                a5_cand_sr = CandidateSearchResult(
                    config=(a5_arch, a5_cfg), config_id=a5_id,
                    fold_stats=tuple(a5_fold_stats), gate=a5_gate,
                )
                # Inject A5 into candidates + re-rank top_k
                all_cands = list(s5.candidates) + [a5_cand_sr]
                ranked = sorted(
                    all_cands,
                    key=lambda r: (r.gate.worst_fold_ratio, r.gate.mean_fold_ratio),
                    reverse=True,
                )
                s5 = WfoSearchResult(
                    structure_name=s5.structure_name,
                    n_candidates_evaluated=s5.n_candidates_evaluated + 1,
                    candidates=tuple(all_cands),
                    top_k=tuple(ranked[:3]),
                )
                a5_candidates.append((a5_id, (a5_arch, a5_cfg)))
                n_configs += 1
                # Add A5 strategy results to main map so Amendment 3 chains it
                strategy_results_by_cid_fold[a5_id] = a5_strategy_per_fold[a5_id]
                # No per_candidate_ctx for A5 — its arch ignores ctx
                per_candidate_ctx[a5_id] = base_ctx
                log.info("A5 added: %d fold stats, gate verdict %s, worst-fold ratio %.4f",
                         len(a5_fold_stats), a5_gate.verdict.value, a5_gate.worst_fold_ratio)
        else:
            log.info("A5 skipped: <2 clusters with viable candidates")
    else:
        log.info("A5 skipped: <2 candidate clusters in v3.0.1 Step 3")

    # ── Holdout (top-K) ─────────────────────────────────────────────
    holdout_results_by_cid: dict[str, StrategyResult] = {}
    holdout_fold_stats: dict[str, FoldStats] = {}
    if s5.top_k and wfo_struct.holdout is not None:
        log.info("Holdout — top-%d candidates × 1 holdout fold", len(s5.top_k))
        for cand in s5.top_k:
            arch, conf = cand.config
            cid = cand.config_id
            if arch.architecture_name == "A5":
                # A5 holdout: re-compose from constituents' holdout results.
                # Constituents would need re-fetch; simplify: A5 holdout uses
                # constituent fold approach below.
                constituent_results = []
                for sr_cluster_id in per_cluster_best.values():
                    arch_inner_pair = next(
                        (paired for cid_check, paired in candidates_for_search if cid_check == sr_cluster_id),
                        None,
                    )
                    if arch_inner_pair is None:
                        continue
                    inner_arch, inner_conf = arch_inner_pair
                    inner_ctx = per_candidate_ctx.get(sr_cluster_id, base_ctx)
                    try:
                        rr = ArcFoldRunner(
                            architecture=inner_arch, signal_evaluation=signal_eval,
                            panels=panels, run_context=inner_ctx,
                        )
                        rr(wfo_struct.holdout, inner_conf)
                        if rr.last_result is not None:
                            constituent_results.append(rr.last_result)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("A5 holdout constituent %s errored: %s", sr_cluster_id, exc)
                if len(constituent_results) >= 2:
                    a5_cfg = A5Config(
                        config_id=cid, constituents=tuple(constituent_results),
                        starting_balance=100_000.0,
                    )
                    a5_h = arch.run(arch_config=a5_cfg, config_id=cid, fold=wfo_struct.holdout)
                    holdout_fold_stats[cid] = a5_h.fold_stats
                    holdout_results_by_cid[cid] = a5_h
                continue
            ctx = per_candidate_ctx.get(cid, base_ctx)
            try:
                r = ArcFoldRunner(
                    architecture=arch, signal_evaluation=signal_eval,
                    panels=panels, run_context=ctx,
                )
                stats = r(wfo_struct.holdout, conf)
                holdout_fold_stats[cid] = stats
                if r.last_result is not None:
                    holdout_results_by_cid[cid] = r.last_result
            except Exception as exc:  # noqa: BLE001
                log.warning("holdout %s errored: %s\n%s", cid, exc, traceback.format_exc())
                holdout_fold_stats[cid] = FoldStats(
                    fold_id=wfo_struct.holdout.fold_id, n_trades=0, roi_pct=0.0,
                    max_dd_pct=0.0, days_breaching_daily_5pct=0, roi_dd_ratio=0.0,
                )

    # ── Amendment 3 evaluation per top-K ─────────────────────────────
    log.info("Amendment 3 — per-top-K evaluation")
    amended_results: list[CandidateAmendedResult] = []
    for cand in s5.top_k:
        try:
            res = _run_amendment_3_for_candidate(
                cand=cand,
                strategy_results_by_cid_fold=strategy_results_by_cid_fold,
                holdout_result_by_cid=holdout_results_by_cid,
                panels=panels, signal_eval=signal_eval,
                per_candidate_ctx=per_candidate_ctx, base_ctx=base_ctx,
                pair_set=PAIRS_28, out_dir=out, r_base=0.005,
                holdout_fold=wfo_struct.holdout,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Amendment 3 eval for %s errored: %s\n%s", cand.config_id, exc, traceback.format_exc())
            continue
        amended_results.append(res)
    amended_wfo = AmendedWfoSearchResult(base=s5, amended_results=tuple(amended_results))

    # ── Step 6 auto-dispatch ────────────────────────────────────────
    step_6_dispatch = None
    if amended_wfo.amended_results:
        log.info("Step 6 — maybe_dispatch_step_6 (auto)")
        train_end_ts = pd.Timestamp("2021-01-01", tz="UTC")
        try:
            step_6_dispatch = maybe_dispatch_step_6(
                arc_orchestrator_result=_LightOrchestratorView(
                    arc_name="l_arc_7_v3_0_2",
                    pool=faux_pool,
                    step_4=_FauxStep4(per_cluster=tuple(v301.step4_per_cluster)),
                    wfo_search=s5, amended_wfo=amended_wfo,
                ),
                amended_wfo=amended_wfo,
                arc_root=out,
                holdout_start=train_end_ts,
                panels=panels,
                feature_lineage=None,
                signal_module_name=type(signal_module).__module__,
                primary_tf="H4",
                pair_set=PAIRS_28,
            )
            if step_6_dispatch.downgrade_top_1:
                log.warning("Step 6 critical-failed; downgrading Top-1 verdict")
                amended_wfo = replace_top_1_with_step6_fail(amended_wfo)
        except Exception as exc:  # noqa: BLE001
            log.error("Step 6 dispatch errored: %s\n%s", exc, traceback.format_exc())

    # ── Determine final verdict ──────────────────────────────────────
    final_verdict = "INCOMPLETE"
    top_amended: CandidateAmendedResult | None = None
    if amended_wfo.amended_results:
        ranked = sorted(
            amended_wfo.amended_results,
            key=lambda r: _amended_verdict_rank(r.amended_gate.verdict), reverse=True,
        )
        top_amended = ranked[0]
        final_verdict = top_amended.amended_gate.verdict.value.upper().replace("_", "-")
    elif s5.top_k:
        final_verdict = s5.top_k[0].gate.verdict.value.upper()

    log.info("Final verdict: %s", final_verdict)

    # ── Oracle WFO per candidate cluster (locked to winning exit policy) ──
    log.info("Oracle WFO per cluster (locked to winning exit policy)")
    win_trail_enabled = True
    win_sl = 2.0
    if top_amended is not None:
        for cand in s5.top_k:
            if cand.config_id == top_amended.config_id:
                _arch, win_conf = cand.config
                win_trail_enabled = bool(getattr(win_conf, "trail_enabled", True))
                win_sl = float(getattr(win_conf, "sl_atr_mult", 2.0))
                break

    oracle_rows: list[dict] = []
    for cc in v301.candidate_clusters:
        base_a1 = A1Config(
            config_id=f"oracle_cl{cc.cluster_id}",
            sl_atr_mult=win_sl, trail_enabled=win_trail_enabled, risk_pct=0.005,
            max_concurrent_per_currency=2, max_concurrent_per_pair=1,
            max_concurrent_total=None,
        )
        oracle_runner = OracleFoldRunner(
            signal_evaluation=signal_eval, panels=panels,
            cluster_assignments=v301.cluster_assignments,
            candidate_cluster_id=cc.cluster_id, trades=v301.pool_trades,
        )
        for fold in wfo_struct.folds:
            if fold.is_days < 365:
                continue
            try:
                stats = oracle_runner(fold, base_a1)
            except Exception as exc:  # noqa: BLE001
                log.warning("oracle cl%d fold%d errored: %s", cc.cluster_id, fold.fold_id, exc)
                stats = FoldStats(
                    fold_id=fold.fold_id, n_trades=0, roi_pct=0.0,
                    max_dd_pct=0.0, days_breaching_daily_5pct=0, roi_dd_ratio=0.0,
                )
            oracle_rows.append({
                "cluster_id": cc.cluster_id, "fold_id": stats.fold_id,
                "n_trades": stats.n_trades, "roi_pct": stats.roi_pct,
                "max_dd_pct": stats.max_dd_pct, "roi_dd_ratio": stats.roi_dd_ratio,
            })

    # ── Write Step 5 outputs ─────────────────────────────────────────
    s5_dir = out / "step_5"
    s5_dir.mkdir(exist_ok=True)
    wfo_rows: list[dict] = []
    per_fold_rows: list[dict] = []
    for cid_str, fold_dict in strategy_results_by_cid_fold.items():
        for fid, sr in fold_dict.items():
            wfo_rows.append({
                "config_id": cid_str, "fold_id": fid,
                "n_trades": sr.fold_stats.n_trades, "roi_pct": sr.fold_stats.roi_pct,
                "max_dd_pct": sr.fold_stats.max_dd_pct,
                "days_breaching_daily_5pct": sr.fold_stats.days_breaching_daily_5pct,
                "roi_dd_ratio": sr.fold_stats.roi_dd_ratio,
            })
            per_fold_rows.append({
                "config_id": cid_str, "fold_id": fid,
                "n_trades": sr.fold_stats.n_trades, "roi_pct": sr.fold_stats.roi_pct,
                "max_dd_pct": sr.fold_stats.max_dd_pct,
                "roi_dd_ratio": sr.fold_stats.roi_dd_ratio,
            })
    pd.DataFrame(wfo_rows).to_csv(s5_dir / "wfo_results.csv", index=False, lineterminator="\n")
    pd.DataFrame(per_fold_rows).to_csv(s5_dir / "per_fold_metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(oracle_rows).to_csv(s5_dir / "wfo_oracle.csv", index=False, lineterminator="\n")
    holdout_rows = [
        {"config_id": cid, "fold_id": s.fold_id, "n_trades": s.n_trades,
         "roi_pct": s.roi_pct, "max_dd_pct": s.max_dd_pct,
         "roi_dd_ratio": s.roi_dd_ratio}
        for cid, s in holdout_fold_stats.items()
    ]
    pd.DataFrame(holdout_rows).to_csv(s5_dir / "holdout_results.csv", index=False, lineterminator="\n")

    if n_configs >= 100:
        search_scope = "broad"
    elif n_configs >= 50:
        search_scope = "normal"
    else:
        search_scope = "thin"

    summary_md = ["# Step 5 — WFO Summary (v3.0.2)", ""]
    summary_md.append(f"**Total configs evaluated:** {n_configs} ({search_scope})")
    summary_md.append("")
    summary_md.append("## Architecture-map override applied")
    summary_md.append("")
    for cc in v301.candidate_clusters:
        ce = cluster_step4.get(cc.cluster_id)
        augmented = ce and ce.best_classifier_mean_auc >= AUC_DEPLOYABILITY_BAR
        summary_md.append(
            f"- c{cc.cluster_id} ({cc.shape_tag}, AUC {(ce.best_classifier_mean_auc if ce else 0.0):.4f}): "
            f"{cluster_arch_sets[cc.cluster_id]} "
            f"{'(augmented per override)' if augmented else ''}"
        )
    summary_md.append("")
    summary_md.append("## Top-K Amendment 3 results")
    summary_md.append("")
    summary_md.append("| Rank | Config | Verdict (Amended) | Worst ratio | r_safe | r_hard | Chained DD base | Daily breaches @r_safe |")
    summary_md.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for k, ar in enumerate(sorted(amended_wfo.amended_results, key=lambda r: _amended_verdict_rank(r.amended_gate.verdict), reverse=True)[:3], start=1):
        g = ar.amended_gate
        summary_md.append(
            f"| {k} | `{ar.config_id}` | {g.verdict.value} | {g.worst_fold_ratio:.2f} | "
            f"{g.r_safe_pct:.4%} | {g.r_hard_pct:.4%} | {ar.chained_max_dd_base_pct:.4%} | "
            f"{g.daily_dd_breaches_at_r_safe} |"
        )
    (s5_dir / "wfo_summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8", newline="\n")

    (s5_dir / "skipped_configs.md").write_text(
        "# Step 5 — Skipped Configs (explicit list per dispatch)\n\n"
        + "\n".join(f"- {x}" for x in skipped_configs_doc) + "\n",
        encoding="utf-8", newline="\n",
    )

    s5_manifest_paths = [
        s5_dir / "wfo_results.csv", s5_dir / "per_fold_metrics.csv",
        s5_dir / "wfo_oracle.csv", s5_dir / "holdout_results.csv",
        s5_dir / "wfo_summary.md", s5_dir / "skipped_configs.md",
    ]
    s5_manifest_paths.extend(s5_dir.glob("per_day_max_dd_base__*.parquet"))
    _write_manifest(s5_dir / "manifest.json", s5_manifest_paths)

    # ── Run summary JSON ─────────────────────────────────────────────
    summary_payload = {
        "arc_name": "l_arc_7_v3_0_2",
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_time_seconds": time.perf_counter() - t_start,
        "boundary_convention": "5ers_eet",
        "pool_size": int(len(v301.pool_trades)),
        "pool_sha256": v301.classifiers_manifest.get("pool_sha256", ""),
        "window_start": "2010-01-01",
        "window_end": str(args.window_end),
        "reuses_v301_artefacts": True,
        "k_selected": int(v301.cluster_summary["cluster_id"].max() + 1) if len(v301.cluster_summary) > 0 else 0,
        "architecture_map_override_applied": True,
        "auc_deployability_bar": AUC_DEPLOYABILITY_BAR,
        "clusters": [
            {
                "cluster_id": int(cc.cluster_id),
                "n_trades": int(cc.n_trades),
                "shape_tag": cc.shape_tag,
                "reach_1r": float(cc.reach_1r),
                "mfe_p50": float(cc.mfe_p50),
                "wrong_way_pp": float(cc.wrong_way_pp),
                "selected_sl": float(cc.selected_sl),
                "capturability_composite": float(cc.capturability_composite),
                "is_candidate": True,
                "architectures_under_override": list(cluster_arch_sets[cc.cluster_id]),
            }
            for cc in v301.candidate_clusters
        ],
        "step4_per_cluster": [
            {
                "cluster_id": int(ce.cluster_id),
                "n_trades": int(ce.n_trades),
                "best_classifier": ce.best_classifier,
                "best_classifier_mean_auc": float(ce.best_classifier_mean_auc),
                "best_threshold": float(ce.best_threshold),
                "auc_above_065": ce.best_classifier_mean_auc >= AUC_DEPLOYABILITY_BAR,
                "top_features": ce.feature_importance.head(10)["feature"].astype(str).tolist() if len(ce.feature_importance) > 0 else [],
                "fitted_classifier_path": str(ce.fitted_classifier_path) if ce.fitted_classifier_path else None,
            }
            for ce in v301.step4_per_cluster
        ],
        "step5": {
            "n_configs": n_configs,
            "search_scope": search_scope,
            "skipped_configs": skipped_configs_doc,
            "top_3": [
                _amended_to_dict(ar) for ar in sorted(
                    amended_wfo.amended_results,
                    key=lambda r: _amended_verdict_rank(r.amended_gate.verdict),
                    reverse=True,
                )[:3]
            ],
        },
        "step6": (
            None if step_6_dispatch is None or not step_6_dispatch.dispatched
            else {
                "dispatched": True,
                "trigger": step_6_dispatch.step_6_result.trigger.value if step_6_dispatch.step_6_result else "auto_pass",
                "overall_passed": bool(step_6_dispatch.step_6_result.overall_passed) if step_6_dispatch.step_6_result else None,
                "manifest_path": str(step_6_dispatch.out_dir / "manifest.json") if step_6_dispatch.out_dir else None,
                "categories": {
                    c.category: bool(c.passed)
                    for c in (step_6_dispatch.step_6_result.categories if step_6_dispatch.step_6_result else [])
                },
                "critical_failures": (
                    step_6_dispatch.step_6_result.critical_failures() if step_6_dispatch.step_6_result else []
                ),
                "warnings_count": sum(
                    c.n_warnings for c in (step_6_dispatch.step_6_result.categories if step_6_dispatch.step_6_result else [])
                ),
                "verdict_impact": (
                    "downgraded_to_fail" if step_6_dispatch.downgrade_top_1 else "none"
                ),
                "divergence_warning": bool(step_6_dispatch.divergence_warning),
            }
        ),
        "verdict": final_verdict,
        "best_config_id": top_amended.config_id if top_amended else None,
        "winning_exit_policy": "sl_plus_trailing_atr" if win_trail_enabled else "sl_only",
        "winning_sl_atr_mult": win_sl,
        "oracle": {
            f"cluster_{cl_id}": {
                "fold_stats": [r for r in oracle_rows if r["cluster_id"] == cl_id],
            }
            for cl_id in sorted({r["cluster_id"] for r in oracle_rows})
        },
    }
    (out / "run_summary.json").write_text(
        json.dumps(summary_payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )

    log.info("Arc 7 v3.0.2 driver — done in %.1fs (wall)", time.perf_counter() - t_start)
    return 0


def _amended_to_dict(ar: CandidateAmendedResult) -> dict:
    g = ar.amended_gate
    return {
        "config_id": ar.config_id,
        "verdict": g.verdict.value,
        "primary_failure_mode": g.primary_failure_mode.value,
        "reason": g.reason,
        "worst_fold_ratio": float(g.worst_fold_ratio),
        "worst_fold_roi_base_pct": float(g.worst_fold_roi_base_pct),
        "worst_fold_dd_base_pct": float(g.worst_fold_dd_base_pct),
        "mean_fold_ratio": float(g.mean_fold_ratio),
        "n_negative_folds": int(g.n_negative_folds),
        "min_trades_per_fold": int(g.min_trades_per_fold),
        "chained_max_dd_base_pct": float(ar.chained_max_dd_base_pct),
        "chained_dd_method": ar.chained_dd_method,
        "per_day_max_dd_artefact_path": str(ar.per_day_max_dd_artefact_path) if ar.per_day_max_dd_artefact_path else None,
        "k_safe": float(g.k_safe),
        "k_hard": float(g.k_hard),
        "r_safe_pct": float(g.r_safe_pct),
        "r_hard_pct": float(g.r_hard_pct),
        "scalable_to_safe": bool(g.scalable_to_safe),
        "scalable_to_hard": bool(g.scalable_to_hard),
        "worst_fold_roi_at_r_safe_pct": float(g.worst_fold_roi_at_r_safe_pct),
        "worst_fold_roi_at_r_hard_pct": float(g.worst_fold_roi_at_r_hard_pct),
        "chained_max_dd_at_r_safe_pct": float(g.chained_max_dd_at_r_safe_pct),
        "chained_max_dd_at_r_hard_pct": float(g.chained_max_dd_at_r_hard_pct),
        "daily_dd_breaches_at_r_safe": int(g.daily_dd_breaches_at_r_safe),
        "daily_dd_breaches_at_r_hard": int(g.daily_dd_breaches_at_r_hard),
        "holdout_roi_at_r_safe_pct": g.holdout_roi_at_r_safe_pct,
        "holdout_dd_at_r_safe_pct": g.holdout_dd_at_r_safe_pct,
        "holdout_roi_at_r_hard_pct": g.holdout_roi_at_r_hard_pct,
        "holdout_dd_at_r_hard_pct": g.holdout_dd_at_r_hard_pct,
        "sizing_convention": g.sizing_convention,
    }


def _build_per_trade_features_for_a2_a6(
    pool_trades: pd.DataFrame, signal_eval: SignalEvaluation,
    panel_h4: Panel, feature_order: tuple[str, ...],
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """Reconstruct per-trade Step-1 feature lookup for A2/A6 admit gates.

    Computes the Step-1 27-feature catalogue per pair via
    core.features.pipeline.compute_feature_matrix (with panel.aux for
    multi_tf), then indexes at each trade's signal_time.

    Returns (pair, signal_time) → {feature_name: value} restricted to
    feature_order columns.
    """
    from core.features.pipeline import compute_feature_matrix

    matrices: dict[str, pd.DataFrame] = {}
    for pair in sorted(panel_h4.pairs):
        df = panel_h4.pair_dfs[pair]
        res = compute_feature_matrix(pair, df, panel=panel_h4)
        matrices[pair] = res.matrix

    out: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    feat_keys = list(feature_order) if feature_order else None
    for _, trade in pool_trades.iterrows():
        pair = trade["pair"]
        signal_time = pd.Timestamp(trade["signal_time"])
        m = matrices.get(pair)
        if m is None or signal_time not in m.index:
            continue
        row = m.loc[signal_time]
        if feat_keys is None:
            out[(pair, signal_time)] = {
                col: float(row[col]) if pd.notna(row[col]) else 0.0
                for col in row.index
            }
        else:
            out[(pair, signal_time)] = {
                col: (float(row[col]) if (col in row.index and pd.notna(row[col])) else 0.0)
                for col in feat_keys
            }
    return out


def _emergency_halt(out: Path, pool: Any, reason: str) -> int:
    (out / "run_summary.json").write_text(
        json.dumps(
            {"arc_name": "l_arc_7_v3_0_2", "verdict": "HALT",
             "halt_reason": reason, "pool_size": int(len(pool.trades))},
            indent=2,
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
