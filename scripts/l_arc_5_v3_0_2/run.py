"""Arc 5 v3.0.2 driver — mtf_alignment.2_down_mixed.kijun.h_120, vanilla
L_PROTOCOL v3.0 + Amendments 1-6, A5 deferred per Amendment 5.1.

Composition driver adapted from `scripts/arc_7/run_arc_7.py`. Differences:

  - Signal: MtfAlignment2DownMixedKijunSignal (H1 primary; H4 + D1 auxiliary)
  - Panels: H1 (primary) + H4 + D1 + W1 (W1 for features only)
  - Architecture admission: Amendment 5 four-gate dispatch-time rule
    (Gate 1 archetype→A3/A4, Gate 2 AUC≥0.65→A2/A6, Gate 3 universal A1,
    Gate 4 A5 deferred per Amendment 5.1).
  - Output root: results/l_arc_5_v3.0.2/

Per L_PROTOCOL §1: random_state=42, n_jobs=1, lineterminator='\n' throughout
via core.determinism. Boundary convention: 5ers_eet end-to-end (Amendment 6).

Invocation::

    python -m scripts.l_arc_5_v3_0_2.run \\
        --histdata-root C:/Users/panap/Documents/Forex-Backtester/data/histdata \\
        --cache-root    C:/Users/panap/Documents/Forex-Backtester/data/cache \\
        --out-root      results/l_arc_5_v3.0.2

Outputs `results/l_arc_5_v3.0.2/` per L_PROTOCOL §6 + Amendment 3/4 artefacts.
Closure doc written by `scripts/l_arc_5_v3_0_2/write_closure.py`.
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

from core.arc.arc_pool_builder import (  # noqa: E402
    ArcPool,
    ArcPoolConfig,
    build_arc_pool,
    write_arc_pool,
)
from core.architectures._protocol import Architecture, StrategyResult  # noqa: E402
from core.architectures.a1_system_level_filter import (  # noqa: E402
    A1Architecture,
    A1Config,
    A1RunContext,
)
from core.architectures.a2_classifier_filter import A2Architecture  # noqa: E402
from core.architectures.a3_pipeline_de import A3Architecture  # noqa: E402
from core.architectures.a4_pipeline_d_exits import A4Architecture  # noqa: E402
from core.architectures.a6_meta_labeling import A6Architecture  # noqa: E402
from core.data.aggregator import aggregate  # noqa: E402
from core.determinism import seed_everything  # noqa: E402
from core.features.pipeline import compute_feature_matrix  # noqa: E402
from core.parallel import default_pool_size  # noqa: E402
from core.runners._fold_stats_helpers import compute_per_day_max_dd  # noqa: E402
from core.runners.arc_fold_runner import ArcFoldRunner  # noqa: E402
from core.runners.oracle_fold_runner import OracleFoldRunner  # noqa: E402
from core.sim.panel import Panel  # noqa: E402
from core.step_6.dispatch import (  # noqa: E402
    maybe_dispatch_step_6,
    replace_top_1_with_step6_fail,
)
from core.steps._shape_tags import (  # noqa: E402
    BIMODAL,
    CHOPPY,
    MONOTONIC_DOWN,
    MONOTONIC_UP,
    STEPWISE,
    UNCLASSIFIED,
    V_SHAPE,
)
from core.steps.classifier_persistence import (  # noqa: E402
    build_a2_config_from_step4,
    build_a3_config_from_step4,
    build_a4_config_from_step4,
    build_a6_config_from_step4,
)
from core.steps.path_classifier_per_fold import (  # noqa: E402
    PerFoldTrainingInputs,
    build_path_classifier_fits_per_fold,
    build_per_trade_entry_features,
)
from core.steps.step_2_clustering import Step2Result, run_step_2  # noqa: E402
from core.steps.step_3_capturability import (  # noqa: E402
    SL_MULT_SWEEP,
    Step3Result,
    run_step_3,
)
from core.steps.step_4_extraction import Step4Result, run_step_4  # noqa: E402
from core.signals.mtf_alignment_2_down_mixed_kijun import (  # noqa: E402
    MtfAlignment2DownMixedKijunSignal,
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

# Amendment 5 four-gate admission rule (per intent doc §F).
# Gate 1 (archetype): V-shape→A3; Stepwise→A4; Bimodal→A4;
#         Monotonic_up/Monotonic_down/Unclassified→(none from Gate 1)
#         Choppy→skip all
# Gate 2 (AUC): A2 + A6 if cluster mean OOS AUC ≥ 0.65
# Gate 3: A1 universally (every non-Choppy cluster)
# Gate 4: A5 NOT admitted at dispatch (Amendment 5.1 deferral)
GATE_2_AUC_THRESHOLD = 0.65

ARCHETYPE_TO_GATE_1: dict[str, tuple[str, ...]] = {
    V_SHAPE: ("A3",),
    STEPWISE: ("A4",),
    BIMODAL: ("A4",),
    MONOTONIC_UP: (),
    MONOTONIC_DOWN: (),
    UNCLASSIFIED: (),
    CHOPPY: (),
}

# Arc 7-era (Amendment 1) reference, used to populate
# `architectures_skipped_by_amendment_5` when Amendment 5's set differs.
AMENDMENT_1_ARCHETYPE_MAP: dict[str, tuple[str, ...]] = {
    V_SHAPE: ("A1", "A3", "A6"),
    STEPWISE: ("A1", "A2", "A4"),
    BIMODAL: ("A1", "A4"),
    MONOTONIC_UP: ("A1", "A2", "A6"),
    MONOTONIC_DOWN: (),
    CHOPPY: (),
    UNCLASSIFIED: ("A1",),
}


def resolve_admitted_architectures(
    archetype: str, mean_oos_auc: float | None
) -> tuple[str, ...]:
    """Amendment 5 four-gate resolver — returns admitted arch codes for a
    surviving cluster.

    Gate 1 from archetype (A3/A4); Gate 2 from AUC ≥ 0.65 (A2/A6); Gate 3
    universal A1 (every non-Choppy cluster); Gate 4 (A5) NOT admitted at
    dispatch time per Amendment 5.1.

    Choppy → empty tuple (cluster skipped entirely).
    """
    if archetype == CHOPPY:
        return ()
    admitted: set[str] = set()
    # Gate 1
    admitted.update(ARCHETYPE_TO_GATE_1.get(archetype, ()))
    # Gate 2 — admit A2 + A6 when AUC threshold cleared
    if mean_oos_auc is not None and mean_oos_auc >= GATE_2_AUC_THRESHOLD:
        admitted.update(("A2", "A6"))
    # Gate 3 — universal A1 (non-Choppy)
    admitted.add("A1")
    # Gate 4 — A5 deferred (handled post-Step-5)
    return tuple(sorted(admitted))


_ARCHITECTURE_CLASSES = {
    "A1": A1Architecture,
    "A2": A2Architecture,
    "A3": A3Architecture,
    "A4": A4Architecture,
    "A6": A6Architecture,
}


def _setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity >= 1 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


log = logging.getLogger("arc5_v3_0_2")


def _build_panel_5ers_eet(
    pairs: list[str],
    tf: str,
    *,
    histdata_root: Path,
    cache_root: Path,
) -> Panel:
    """Aggregate `pairs` × `tf` under 5ers_eet boundary; return Panel.

    Serial aggregation (the parallel helper doesn't accept boundary_convention).
    Cache-warm → ~seconds; cache-cold → ~30s/pair.
    """
    pair_dfs: dict[str, pd.DataFrame] = {}
    for p in sorted(pairs):
        pair_dfs[p] = aggregate(
            p, tf,
            histdata_root=histdata_root, cache_root=cache_root,
            boundary_convention="5ers_eet",
        )
    return Panel.from_frames(pair_dfs, tf=tf, boundary_convention="5ers_eet")


# ──────────────────────────────────────────────────────────────────────
# Step 1 helpers
# ──────────────────────────────────────────────────────────────────────


def _compute_step1_feature_matrix(
    pool: ArcPool,
    panel: Panel,
    pair_dfs: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the 27-feature matrix at each trade's signal_time.

    Per-pair: compute_feature_matrix once over the full pair df (H1 panel
    with aux d1/w1), then extract row at each Step 1 trade's signal_time.

    Returns (feature_matrix_df, feature_lineage_df). Renames lineage →
    causal_lineage for Step 4 compatibility (engine bug workaround).
    """
    log.info("computing Step-1 feature matrix per pair …")
    t0 = time.perf_counter()
    matrices: dict[str, pd.DataFrame] = {}
    lineage_ref: pd.DataFrame | None = None
    for pair in sorted(pair_dfs):
        df = pair_dfs[pair]
        result = compute_feature_matrix(pair, df, panel=panel)
        matrices[pair] = result.matrix
        if lineage_ref is None:
            lineage_ref = result.lineage
    assert lineage_ref is not None

    rows: list[dict] = []
    for _, trade in pool.trades.iterrows():
        pair = trade["pair"]
        signal_time = pd.Timestamp(trade["signal_time"])
        m = matrices.get(pair)
        if m is None or signal_time not in m.index:
            continue
        row = m.loc[signal_time].to_dict()
        row["trade_id"] = int(trade["trade_id"])
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    if "trade_id" in feat_df.columns:
        feat_df = feat_df.set_index("trade_id").sort_index()

    all_nan_cols = [c for c in feat_df.columns if feat_df[c].isna().all()]
    if all_nan_cols:
        log.warning("Dropping all-NaN feature columns: %s", all_nan_cols)
        feat_df = feat_df.drop(columns=all_nan_cols)
        lineage_ref = lineage_ref[~lineage_ref["name"].isin(all_nan_cols)]

    lineage_renamed = lineage_ref.rename(columns={"lineage": "causal_lineage"})

    log.info(
        "step-1 feature matrix: %d trades × %d features in %.1fs",
        len(feat_df), max(0, len(feat_df.columns)),
        time.perf_counter() - t0,
    )
    return feat_df, lineage_renamed


def _build_per_trade_features(
    pool_trades: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """Build (pair, signal_time) → feature_dict for A1 filter_rules + A2/A6
    classifier admit gates.
    """
    pool = pool_trades.copy()
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True)
    pool_by_tid = pool.set_index("trade_id")
    fm = feature_matrix
    if "trade_id" in fm.columns:
        fm = fm.set_index("trade_id")
    out: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
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


# ──────────────────────────────────────────────────────────────────────
# Step 5 candidate-grid builders
# ──────────────────────────────────────────────────────────────────────


def _sl_window_around(selected_sl: float) -> tuple[float, ...]:
    """±1 step around selected_sl in SL_MULT_SWEEP."""
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


def _exit_policy_slate(archetype: str) -> tuple[str, ...]:
    """L_PROTOCOL §2 Step 5 / intent §E exit-policy slate per archetype.

    All registered in core.sim.exit_policies and supported as A1/A2/A3/A4/A6
    Config kwarg `exit_policy` (PR #195).
    """
    if archetype == V_SHAPE:
        return ("sl_only", "sl_plus_tp_2r", "sl_partial_close_1r_runner_trail")
    if archetype == STEPWISE:
        return ("sl_only", "sl_plus_trailing_atr", "sl_plus_trailing_swing")
    if archetype == BIMODAL:
        return ("sl_only", "sl_partial_close_1r_runner_trail", "sl_plus_tp_2r")
    # Monotonic_up / Monotonic_down / Unclassified
    return ("sl_only", "sl_plus_tp_2r")


def _build_a1_configs(
    cluster_id: int, selected_sl: float, archetype: str
) -> list[tuple[str, tuple[A1Architecture, A1Config]]]:
    out: list[tuple[str, tuple[A1Architecture, A1Config]]] = []
    for sl in _sl_window_around(selected_sl):
        for policy in _exit_policy_slate(archetype):
            for cap in _exposure_choices():
                cid_str = (
                    f"A1::cl{cluster_id}::sl{sl:.1f}"
                    f"::{policy}::exp{cap if cap else 'inf'}"
                )
                cfg = A1Config(
                    config_id=cid_str,
                    sl_atr_mult=sl,
                    trail_enabled=False,  # exit policy handles trailing semantics
                    risk_pct=0.005,
                    max_concurrent_per_currency=cap,
                    max_concurrent_per_pair=1,
                    max_concurrent_total=None,
                    exit_policy=policy if policy != "sl_only" else None,
                )
                out.append((f"A1::{cid_str}", (A1Architecture(), cfg)))
    return out


def _build_a2_configs(
    cluster_id: int, selected_sl: float, archetype: str, s4: Step4Result
) -> list[tuple[str, tuple[A2Architecture, Any]]]:
    out: list[tuple[str, tuple[A2Architecture, Any]]] = []
    for sl in _sl_window_around(selected_sl):
        for policy in _exit_policy_slate(archetype):
            for cap in _exposure_choices():
                try:
                    cid_str = (
                        f"A2::cl{cluster_id}::sl{sl:.1f}"
                        f"::{policy}::exp{cap if cap else 'inf'}"
                    )
                    cfg = build_a2_config_from_step4(
                        s4,
                        cluster_id=cluster_id,
                        config_id=cid_str,
                        sl_atr_mult=sl,
                        trail_enabled=False,
                        risk_pct=0.005,
                        max_concurrent_per_currency=cap,
                        max_concurrent_per_pair=1,
                        max_concurrent_total=None,
                        exit_policy=policy if policy != "sl_only" else None,
                    )
                except ValueError as exc:
                    log.warning("A2 cluster %d skipped: %s", cluster_id, exc)
                    return []
                out.append((f"A2::{cid_str}", (A2Architecture(), cfg)))
    return out


def _build_a3_configs(
    cluster_id: int, selected_sl: float, archetype: str, s4: Step4Result
) -> list[tuple[str, tuple[A3Architecture, Any]]]:
    out: list[tuple[str, tuple[A3Architecture, Any]]] = []
    for sl in _sl_window_around(selected_sl):
        for n in (3, 5):
            for policy in _exit_policy_slate(archetype):
                for cap in _exposure_choices():
                    try:
                        cid_str = (
                            f"A3::cl{cluster_id}::sl{sl:.1f}::n{n}"
                            f"::{policy}::exp{cap if cap else 'inf'}"
                        )
                        cfg = build_a3_config_from_step4(
                            s4,
                            cluster_id=cluster_id,
                            n_defer=n,
                            config_id=cid_str,
                            sl_atr_mult=sl,
                            trail_enabled=False,
                            risk_pct=0.005,
                            max_concurrent_per_currency=cap,
                            max_concurrent_per_pair=1,
                            max_concurrent_total=None,
                            exit_policy=policy if policy != "sl_only" else None,
                        )
                    except ValueError as exc:
                        log.warning("A3 cluster %d skipped: %s", cluster_id, exc)
                        return []
                    out.append((f"A3::{cid_str}", (A3Architecture(), cfg)))
    return out


def _build_a4_configs(
    cluster_id: int, selected_sl: float, archetype: str, s4: Step4Result
) -> list[tuple[str, tuple[A4Architecture, Any]]]:
    out: list[tuple[str, tuple[A4Architecture, Any]]] = []
    for sl in _sl_window_around(selected_sl):
        for thr in (0.3, 0.4, 0.5):
            for policy in _exit_policy_slate(archetype):
                for cap in _exposure_choices():
                    try:
                        cid_str = (
                            f"A4::cl{cluster_id}::sl{sl:.1f}::thr{thr:.1f}"
                            f"::{policy}::exp{cap if cap else 'inf'}"
                        )
                        cfg = build_a4_config_from_step4(
                            s4,
                            cluster_id=cluster_id,
                            exit_threshold=thr,
                            config_id=cid_str,
                            sl_atr_mult=sl,
                            trail_enabled=False,
                            risk_pct=0.005,
                            max_concurrent_per_currency=cap,
                            max_concurrent_per_pair=1,
                            max_concurrent_total=None,
                            exit_policy=policy if policy != "sl_only" else None,
                        )
                    except ValueError as exc:
                        log.warning("A4 cluster %d skipped: %s", cluster_id, exc)
                        return []
                    out.append((f"A4::{cid_str}", (A4Architecture(), cfg)))
    return out


def _build_a6_configs(
    cluster_id: int, selected_sl: float, archetype: str, s4: Step4Result
) -> list[tuple[str, tuple[A6Architecture, Any]]]:
    out: list[tuple[str, tuple[A6Architecture, Any]]] = []
    for sl in _sl_window_around(selected_sl):
        for lo, hi in ((0.3, 0.5), (0.4, 0.6), (0.5, 0.7)):
            for policy in _exit_policy_slate(archetype):
                for cap in _exposure_choices():
                    try:
                        cid_str = (
                            f"A6::cl{cluster_id}::sl{sl:.1f}::thr{lo:.1f}-{hi:.1f}"
                            f"::{policy}::exp{cap if cap else 'inf'}"
                        )
                        cfg = build_a6_config_from_step4(
                            s4,
                            cluster_id=cluster_id,
                            lower_threshold=lo,
                            upper_threshold=hi,
                            config_id=cid_str,
                            sl_atr_mult=sl,
                            trail_enabled=False,
                            risk_pct=0.005,
                            max_concurrent_per_currency=cap,
                            max_concurrent_per_pair=1,
                            max_concurrent_total=None,
                            exit_policy=policy if policy != "sl_only" else None,
                        )
                    except ValueError as exc:
                        log.warning("A6 cluster %d skipped: %s", cluster_id, exc)
                        return []
                    out.append((f"A6::{cid_str}", (A6Architecture(), cfg)))
    return out


# ──────────────────────────────────────────────────────────────────────
# Amendment 3 evaluation
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
    *,
    cand: CandidateSearchResult,
    strategy_results_by_cid_fold: Mapping[str, Mapping[int, StrategyResult]],
    holdout_result_by_cid: Mapping[str, StrategyResult],
    panels: Mapping[str, Panel],
    signal_eval: Any,
    per_candidate_ctx: Mapping[str, A1RunContext],
    base_ctx: A1RunContext,
    pair_set: tuple[str, ...],
    out_dir: Path,
    r_base: float,
    holdout_fold: Fold | None,
) -> CandidateAmendedResult:
    """Per-top-K Amendment 3 evaluation per orchestrator pattern."""
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
    chained_equity = stitch_per_fold_oos_equity(
        per_fold_equity, starting_balance=starting_balance
    )
    chained_dd = compute_chained_max_dd_from_continuous_equity(chained_equity)

    per_day_df = compute_per_day_max_dd(
        chained_equity, pair_set=",".join(pair_set),
        boundary_convention="5ers_eet",
    )
    parquet_path: Path | None = None
    if not per_day_df.empty:
        safe_cid = cid.replace("::", "__").replace("/", "_")
        parquet_path = out_dir / "step_5" / f"per_day_max_dd_base__{safe_cid}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        per_day_df.to_parquet(
            parquet_path, engine="pyarrow", compression="snappy", index=False,
        )

    worst_fold_dd_base = max(
        (f.max_dd_pct for f in cand.fold_stats), default=0.0
    )
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
            architecture=arch,
            signal_evaluation=signal_eval,
            panels=panels,
            run_context=ctx,
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
        config_id=cid,
        chained_max_dd_base_pct=chained_dd,
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


@dataclass(frozen=True)
class _LightOrchestratorView:
    arc_name: str
    pool: Any
    step_4: Any
    wfo_search: Any
    amended_wfo: Any


# ──────────────────────────────────────────────────────────────────────
# Manifest helpers
# ──────────────────────────────────────────────────────────────────────


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(manifest_path: Path, artefacts: list[Path], extras: dict | None = None) -> None:
    root = manifest_path.parent
    entries: dict[str, dict] = {}
    for a in sorted(artefacts):
        if not a.exists():
            continue
        rel = str(a.relative_to(root)).replace("\\", "/")
        entries[rel] = {"sha256": _file_sha256(a), "size_bytes": a.stat().st_size}
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artefacts": entries,
    }
    if extras:
        payload.update(extras)
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


# ──────────────────────────────────────────────────────────────────────
# Halt diagnostics
# ──────────────────────────────────────────────────────────────────────


def _emergency_halt(out: Path, pool: ArcPool | None, reason: str) -> int:
    """Emit minimal summary; return 1 (HALT)."""
    payload: dict[str, Any] = {
        "arc_name": "l_arc_5_v3.0.2",
        "verdict": "HALT",
        "halt_reason": reason,
    }
    if pool is not None:
        payload["pool_size"] = int(len(pool.trades))
    (out / "run_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return 1


def _write_a5_gate_4_halt_diagnostic(
    repo_root: Path,
    n_candidate_clusters: int,
    pass_tier_constituents: list[str],
) -> None:
    """Write the Amendment 5.1 Gate 4 HALT diagnostic doc."""
    diag_path = repo_root / "docs" / "dispatches" / "arc_5_v3_0_2_a5_gate_4_halt_diagnostic.md"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = f"""# Arc 5 v3.0.2 — Amendment 5.1 Gate 4 HALT diagnostic

> **Generated:** {ts}
> **Arc:** `l_arc_5_v3.0.2`
> **Branch:** `arc/l_arc_5_v3.0.2`

## Trigger

Amendment 5.1 Gate 4 admission conditions BOTH hold post-Step-5:

- (a) ≥ 2 candidate clusters survived Step 3 — observed: {n_candidate_clusters} candidates.
- (b) ≥ 1 constituent cleared PASS-DEPLOYABLE or PASS-VIABLE under Gate 1/2/3 architecture — observed PASS-tier candidates:

```
{json.dumps(pass_tier_constituents, indent=2)}
```

Per Amendment 5.1 §"Gate 4 PASS-tier-constituent qualifier" and intent doc §I.4,
A5 admission is now permitted but the A5 engine is not yet built. HALT.

## Required next step

Open a fresh chat dispatch to:
1. Build the A5 portfolio-composition engine
2. Compose the cluster constituents into A5 admission
3. Re-evaluate this arc under A5 (closure addendum, not full rerun)

## Artefacts preserved

- `results/l_arc_5_v3.0.2/step_1/` through `step_5/` — IS WFO + holdout + Amendment 3 per top-K.
- `results/l_arc_5_v3.0.2/run_summary.json` — full search state including Top-K + per-cluster oracle WFO.
- Arc branch `arc/l_arc_5_v3.0.2` retained for A5 addendum work.

End of diagnostic.
"""
    diag_path.write_text(body, encoding="utf-8", newline="\n")
    log.error("Wrote A5 Gate 4 HALT diagnostic to %s", diag_path)


# ──────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--histdata-root", type=Path,
        default=Path("C:/Users/panap/Documents/Forex-Backtester/data/histdata"),
    )
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("C:/Users/panap/Documents/Forex-Backtester/data/cache"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/l_arc_5_v3.0.2"))
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--wall-time-budget-h", type=float, default=8.0)
    parser.add_argument("--window-start", type=str, default="2010-01-01")
    parser.add_argument("--window-end", type=str, default=None)
    parser.add_argument("--stop-after-step", type=int, default=None,
                        help="If set, stop after running this step (1..5).")
    parser.add_argument(
        "--a1-only", action="store_true", default=False,
        help=(
            "Override Amendment 5 architecture admission to A1 only for "
            "every surviving non-Choppy cluster. Amendment-5 four-gate "
            "reasoning still recorded in the closure for audit trail; "
            "architectures otherwise admitted under Gates 1/2 are reported "
            "as compute-budget skipped (NOT Amendment-5 skipped). Used for "
            "compute-budget-constrained reruns after a prior full-grid run "
            "exhausted wall time."
        ),
    )
    parser.add_argument("--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    seed_everything(42)
    t_start = time.perf_counter()

    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    pool_size = args.pool_size or default_pool_size(28)
    histdata_root = args.histdata_root.resolve()
    cache_root = args.cache_root.resolve()
    log.info(
        "Arc 5 v3.0.2 driver — pool_size=%d, out_root=%s, histdata_root=%s, cache_root=%s",
        pool_size, out, histdata_root, cache_root,
    )

    # ── Step 0: build panels (H1 primary + H4 + D1 + W1; all 5ers_eet) ──
    log.info("Step 0 — building H1 + H4 + D1 + W1 panels (5ers_eet boundary, 28 pairs)")
    t0 = time.perf_counter()
    panel_h1 = _build_panel_5ers_eet(
        list(PAIRS_28), "H1",
        histdata_root=histdata_root, cache_root=cache_root,
    )
    panel_h4 = _build_panel_5ers_eet(
        list(PAIRS_28), "H4",
        histdata_root=histdata_root, cache_root=cache_root,
    )
    panel_d1 = _build_panel_5ers_eet(
        list(PAIRS_28), "D1",
        histdata_root=histdata_root, cache_root=cache_root,
    )
    panel_w1 = _build_panel_5ers_eet(
        list(PAIRS_28), "W1",
        histdata_root=histdata_root, cache_root=cache_root,
    )
    log.info("Step 0 — panels built in %.1fs", time.perf_counter() - t0)

    # multi_tf features expect panel.aux["d1"] / panel.aux["w1"]; attach to H1 panel.
    # Panel is frozen dataclass — use object.__setattr__ (same pattern as Arc 7).
    object.__setattr__(panel_h1, "aux", {"d1": panel_d1, "w1": panel_w1})
    panels: dict[str, Panel] = {
        "H1": panel_h1, "H4": panel_h4, "D1": panel_d1, "W1": panel_w1,
    }

    # ── Step 1: pool + integrity ─────────────────────────────────────
    log.info("Step 1 — applying mtf_alignment_2_down_mixed_kijun signal")
    t0 = time.perf_counter()
    signal_module = MtfAlignment2DownMixedKijunSignal()
    window_start = pd.Timestamp(args.window_start).date()
    if args.window_end:
        window_end = pd.Timestamp(args.window_end).date()
    else:
        today = datetime.now(timezone.utc).date()
        first_of_month = today.replace(day=1)
        window_end = first_of_month - pd.Timedelta(days=1).to_pytimedelta()
    pool_cfg = ArcPoolConfig(
        arc_name="l_arc_5_v3.0.2",
        sl_atr_mult=2.0,
        hold_bars=240,
        risk_pct=0.005,
        window_start=window_start,
        window_end=window_end,
    )
    pool = build_arc_pool(signal_module, panels, pool_cfg)
    log.info(
        "Step 1 — pool size %d trades in %.1fs",
        len(pool.trades), time.perf_counter() - t0,
    )
    write_arc_pool(pool, out)

    # Augment Step 1 manifest with the resolved absolute cache path (intent §I.1)
    step_1_manifest_path = out / "step_1" / "manifest.json"
    if step_1_manifest_path.exists():
        existing = json.loads(step_1_manifest_path.read_text(encoding="utf-8"))
        existing["cache_root_absolute"] = str(cache_root)
        existing["histdata_root_absolute"] = str(histdata_root)
        existing["boundary_convention"] = "5ers_eet"
        step_1_manifest_path.write_text(
            json.dumps(existing, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )

    if len(pool.trades) < 100:
        log.error(
            "Step 1 HALT — pool size %d looks pathological (<100); surfacing for chat",
            len(pool.trades),
        )
        return _emergency_halt(out, pool, f"step_1_pool_pathological_n_{len(pool.trades)}")

    if args.stop_after_step == 1:
        log.info("Stopping after Step 1 per --stop-after-step")
        (out / "run_summary.json").write_text(
            json.dumps({
                "arc_name": "l_arc_5_v3.0.2",
                "verdict": "STOPPED_AFTER_STEP_1",
                "pool_size": int(len(pool.trades)),
                "boundary_convention": "5ers_eet",
            }, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        return 0

    # ── Step 1 features ──────────────────────────────────────────────
    feat_matrix, feat_lineage_renamed = _compute_step1_feature_matrix(
        pool, panel_h1, panel_h1.pair_dfs
    )
    fm_for_step4 = feat_matrix.reset_index() if len(feat_matrix) > 0 else feat_matrix

    per_trade_features = (
        _build_per_trade_features(pool.trades, feat_matrix)
        if len(feat_matrix) > 0 else {}
    )

    # ── Step 2: clustering ───────────────────────────────────────────
    log.info("Step 2 — path-shape clustering K∈{2..6}")
    t0 = time.perf_counter()
    step2: Step2Result = run_step_2(pool.trades, pool.paths)
    s2_dir = out / "step_2"
    s2_dir.mkdir(exist_ok=True)
    step2.cluster_assignments.to_parquet(
        s2_dir / "cluster_assignments.parquet",
        engine="pyarrow", compression="snappy", index=False,
    )
    step2.cluster_summary.to_csv(
        s2_dir / "cluster_summary.csv", index=False, lineterminator="\n"
    )
    (s2_dir / "cluster_summary.md").write_text(
        step2.summary_md, encoding="utf-8", newline="\n"
    )
    _write_manifest(s2_dir / "manifest.json", [
        s2_dir / "cluster_assignments.parquet",
        s2_dir / "cluster_summary.csv",
        s2_dir / "cluster_summary.md",
    ])
    log.info(
        "Step 2 — K=%d, %d clusters, silhouettes=%s, %.1fs",
        step2.k_selected, len(step2.cluster_summary),
        {k: round(v, 4) for k, v in step2.silhouette_per_k.items()},
        time.perf_counter() - t0,
    )

    # ── Step 3: capturability + archetype ────────────────────────────
    log.info("Step 3 — capturability + SL sweep + archetype labels")
    t0 = time.perf_counter()
    step3: Step3Result = run_step_3(
        pool.trades, pool.paths, step2.cluster_assignments,
        declared_sl_mult=pool_cfg.sl_atr_mult,
        cluster_centroids=step2.centroids,
    )
    s3_dir = out / "step_3"
    s3_dir.mkdir(exist_ok=True)
    step3.capturability_csv.to_csv(
        s3_dir / "capturability.csv", index=False, lineterminator="\n"
    )
    (s3_dir / "capturability_summary.md").write_text(
        step3.summary_md, encoding="utf-8", newline="\n"
    )
    _write_manifest(s3_dir / "manifest.json", [
        s3_dir / "capturability.csv", s3_dir / "capturability_summary.md",
    ])
    candidate_clusters = [c for c in step3.per_cluster if c.is_candidate]
    log.info(
        "Step 3 — %d clusters, %d candidates: %s, %.1fs",
        len(step3.per_cluster), len(candidate_clusters),
        [(c.cluster_id, c.shape_tag) for c in candidate_clusters],
        time.perf_counter() - t0,
    )

    # ── Step 4: extraction with persistence + train_end ──────────────
    s4: Step4Result | None = None
    train_end_ts = pd.Timestamp("2021-01-01", tz="UTC")
    if candidate_clusters and len(feat_matrix) > 0:
        log.info("Step 4 — extraction per candidate cluster (train_end=%s)", train_end_ts)
        t0 = time.perf_counter()
        s4_dir = out / "step_4"
        persistence_dir = s4_dir / "classifiers"
        s4 = run_step_4(
            pool.trades,
            fm_for_step4,
            step2.cluster_assignments,
            feature_lineage=feat_lineage_renamed,
            candidate_cluster_ids=tuple(c.cluster_id for c in candidate_clusters),
            persistence_dir=persistence_dir,
            arc_name="l_arc_5_v3.0.2",
            train_end=train_end_ts,
        )
        s4_dir.mkdir(exist_ok=True)
        s4.extraction_metrics.to_csv(
            s4_dir / "extraction_metrics.csv", index=False, lineterminator="\n"
        )
        s4.feature_importance.to_csv(
            s4_dir / "feature_importance.csv", index=False, lineterminator="\n"
        )
        (s4_dir / "extraction_summary.md").write_text(
            s4.summary_md, encoding="utf-8", newline="\n"
        )
        _write_manifest(s4_dir / "manifest.json", [
            s4_dir / "extraction_metrics.csv",
            s4_dir / "feature_importance.csv",
            s4_dir / "extraction_summary.md",
        ])
        log.info(
            "Step 4 — %d cluster extractions in %.1fs",
            len(s4.per_cluster), time.perf_counter() - t0,
        )
    else:
        log.info("Step 4 — skipped (no candidate clusters or empty feature matrix)")

    if args.stop_after_step == 4:
        log.info("Stopping after Step 4 per --stop-after-step")
        return 0

    # ── Step 5: Amendment 5 four-gate resolution + WFO ──────────────
    log.info("Step 5 — architecture grid + WFO + Amendment 3 + Step 6")
    wfo_struct = build_v3_folds(holdout_end=window_end)
    folds_for_fits = wfo_struct.folds
    if wfo_struct.holdout is not None:
        folds_for_fits = folds_for_fits + (wfo_struct.holdout,)

    # Map Step 4 AUCs per cluster (for Gate 2)
    s4_auc_by_cluster: dict[int, float] = {}
    if s4 is not None:
        for ce in s4.per_cluster:
            s4_auc_by_cluster[int(ce.cluster_id)] = float(ce.best_classifier_mean_auc)

    a3_fits_per_cluster: dict[int, Mapping[int, Any]] = {}
    a4_fits: Mapping[int, Any] | None = None

    shared_entry_features: Mapping | None = None

    def _ensure_entry_features() -> Mapping:
        nonlocal shared_entry_features
        if shared_entry_features is None:
            inputs = PerFoldTrainingInputs(
                pool_trades=pool.trades,
                pool_paths=pool.paths,
                cluster_assignments=step2.cluster_assignments,
                panels=panels,
                primary_tf=signal_module.primary_tf,
                candidate_cluster_id=None,
                n_defer=5,
            )
            shared_entry_features = build_per_trade_entry_features(inputs)
        return shared_entry_features

    candidates_for_search: list[tuple[str, tuple[Architecture, Any]]] = []
    per_candidate_ctx: dict[str, A1RunContext] = {}
    skipped_configs_doc: list[str] = []
    base_ctx = (
        A1RunContext(per_trade_features=per_trade_features)
        if per_trade_features else A1RunContext()
    )

    # Track Amendment 5 admission decisions per cluster
    amendment_5_per_cluster: dict[int, dict[str, Any]] = {}
    # Architectures that Amendment 1 would have admitted but Amendment 5 skipped
    architectures_skipped_under_amendment_5: set[str] = set()
    # Architectures Amendment 5 admitted but --a1-only stripped (compute-budget skip;
    # distinct from Amendment-5 skip — recorded in cross-arc §3 of closure, not in
    # the v1.3.1 architectures_skipped_by_amendment_5 field).
    architectures_skipped_by_a1_only: set[str] = set()

    for cc in candidate_clusters:
        cluster_id = int(cc.cluster_id)
        archetype = cc.shape_tag
        selected_sl = cc.selected_sl
        mean_auc = s4_auc_by_cluster.get(cluster_id)
        admitted_amendment_5 = resolve_admitted_architectures(archetype, mean_auc)
        if args.a1_only and archetype != CHOPPY:
            admitted = tuple(sorted(set(admitted_amendment_5) & {"A1"}))
            a1_only_stripped_here = set(admitted_amendment_5) - {"A1"}
            architectures_skipped_by_a1_only.update(a1_only_stripped_here)
        else:
            admitted = admitted_amendment_5
            a1_only_stripped_here = set()
        amendment_1_set = set(AMENDMENT_1_ARCHETYPE_MAP.get(archetype, ()))
        skipped_here = amendment_1_set - set(admitted_amendment_5)
        architectures_skipped_under_amendment_5.update(skipped_here)
        amendment_5_per_cluster[cluster_id] = {
            "archetype": archetype,
            "mean_oos_auc": mean_auc,
            "gate_1": list(ARCHETYPE_TO_GATE_1.get(archetype, ())),
            "gate_2_auc_threshold": GATE_2_AUC_THRESHOLD,
            "gate_2_admitted": (
                ("A2", "A6") if (mean_auc is not None and mean_auc >= GATE_2_AUC_THRESHOLD)
                else ()
            ),
            "gate_3_universal_a1": archetype != CHOPPY,
            "amendment_5_admitted": list(admitted_amendment_5),
            "admitted": list(admitted),
            "a1_only_stripped": sorted(a1_only_stripped_here),
            "amendment_1_set": sorted(amendment_1_set),
            "skipped_under_amendment_5": sorted(skipped_here),
        }
        log.info(
            "  cluster %d (%s, AUC=%s) → Amendment 5 admitted %s%s",
            cluster_id, archetype,
            f"{mean_auc:.4f}" if mean_auc is not None else "n/a",
            admitted_amendment_5,
            (f" -> tested {admitted} after --a1-only override (compute-budget skip: {sorted(a1_only_stripped_here)})"
             if args.a1_only else " [Amendment 5 four-gate]"),
        )
        if not admitted:
            skipped_configs_doc.append(
                f"cluster{cluster_id}: archetype={archetype} skipped all architectures under Amendment 5"
            )
            continue

        # A1
        if "A1" in admitted:
            a1s = _build_a1_configs(cluster_id, selected_sl, archetype)
            for cid, paired in a1s:
                candidates_for_search.append((cid, paired))
                per_candidate_ctx[cid] = base_ctx

        # A2
        if "A2" in admitted:
            if s4 is None:
                skipped_configs_doc.append(
                    f"A2/cluster{cluster_id}: Step 4 result absent"
                )
            else:
                a2s = _build_a2_configs(cluster_id, selected_sl, archetype, s4)
                if not a2s:
                    skipped_configs_doc.append(
                        f"A2/cluster{cluster_id}: no fitted classifier (insufficient Step 4 class balance)"
                    )
                for cid, paired in a2s:
                    candidates_for_search.append((cid, paired))
                    per_candidate_ctx[cid] = base_ctx

        # A3
        if "A3" in admitted:
            if s4 is None:
                skipped_configs_doc.append(
                    f"A3/cluster{cluster_id}: Step 4 result absent (sanity check)"
                )
            else:
                a3s = _build_a3_configs(cluster_id, selected_sl, archetype, s4)
                if a3s:
                    if cluster_id not in a3_fits_per_cluster:
                        log.info("  building A3 per-fold fits for cluster %d ...", cluster_id)
                        a3_inputs = PerFoldTrainingInputs(
                            pool_trades=pool.trades,
                            pool_paths=pool.paths,
                            cluster_assignments=step2.cluster_assignments,
                            panels=panels,
                            primary_tf=signal_module.primary_tf,
                            candidate_cluster_id=cluster_id,
                            n_defer=5,
                        )
                        a3_fits_per_cluster[cluster_id] = build_path_classifier_fits_per_fold(
                            inputs=a3_inputs, folds=folds_for_fits, arch="A3",
                        )
                    entry_feats = _ensure_entry_features()
                    a3_ctx = A1RunContext(
                        per_trade_features=per_trade_features,
                        per_trade_entry_features=entry_feats,
                        path_classifier_fits=a3_fits_per_cluster[cluster_id],
                    )
                    for cid, paired in a3s:
                        candidates_for_search.append((cid, paired))
                        per_candidate_ctx[cid] = a3_ctx

        # A4
        if "A4" in admitted:
            if s4 is None:
                skipped_configs_doc.append(
                    f"A4/cluster{cluster_id}: Step 4 result absent (sanity check)"
                )
            else:
                a4s = _build_a4_configs(cluster_id, selected_sl, archetype, s4)
                if a4s:
                    if a4_fits is None:
                        log.info("  building A4 per-fold fits (cluster-independent)...")
                        a4_inputs = PerFoldTrainingInputs(
                            pool_trades=pool.trades,
                            pool_paths=pool.paths,
                            cluster_assignments=None,
                            panels=panels,
                            primary_tf=signal_module.primary_tf,
                            candidate_cluster_id=None,
                            n_defer=5,
                        )
                        a4_fits = build_path_classifier_fits_per_fold(
                            inputs=a4_inputs, folds=folds_for_fits, arch="A4",
                        )
                    entry_feats = _ensure_entry_features()
                    a4_ctx = A1RunContext(
                        per_trade_features=per_trade_features,
                        per_trade_entry_features=entry_feats,
                        path_classifier_fits=a4_fits,
                    )
                    for cid, paired in a4s:
                        candidates_for_search.append((cid, paired))
                        per_candidate_ctx[cid] = a4_ctx

        # A6
        if "A6" in admitted:
            if s4 is None:
                skipped_configs_doc.append(
                    f"A6/cluster{cluster_id}: Step 4 result absent"
                )
            else:
                a6s = _build_a6_configs(cluster_id, selected_sl, archetype, s4)
                if not a6s:
                    skipped_configs_doc.append(
                        f"A6/cluster{cluster_id}: no fitted classifier"
                    )
                for cid, paired in a6s:
                    candidates_for_search.append((cid, paired))
                    per_candidate_ctx[cid] = base_ctx

    if not candidates_for_search:
        log.warning("Step 5 — no candidates; running A1 no-filter baseline")
        a1s = _build_a1_configs(-1, 2.0, UNCLASSIFIED)
        for cid, paired in a1s:
            candidates_for_search.append((cid, paired))
            per_candidate_ctx[cid] = base_ctx

    log.info("Step 5 — total config count: %d", len(candidates_for_search))
    if len(candidates_for_search) >= 100:
        search_scope = "broad"
    elif len(candidates_for_search) >= 50:
        search_scope = "normal"
    else:
        search_scope = "thin"

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
                architecture=arch,
                signal_evaluation=pool.signal_evaluation,
                panels=panels,
                run_context=ctx,
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
            raise RuntimeError(
                f"Wall-time budget ({args.wall_time_budget_h}h) exceeded mid-search"
            )
        return stats

    try:
        s5: WfoSearchResult = run_search(
            wfo_struct, candidates_for_search,
            fold_runner=_runner, min_is_days=365, top_k=3,
        )
    except RuntimeError as exc:
        log.error("Step 5 search aborted: %s", exc)
        return _emergency_halt(out, pool, "wall_budget_exceeded_step_5")

    # ── Holdout ─────────────────────────────────────────────────────
    holdout_results_by_cid: dict[str, StrategyResult] = {}
    holdout_fold_stats: dict[str, FoldStats] = {}
    if s5.top_k and wfo_struct.holdout is not None:
        log.info("Holdout — top-%d candidates × 1 holdout fold", len(s5.top_k))
        for cand in s5.top_k:
            arch, conf = cand.config
            cid = cand.config_id
            ctx = per_candidate_ctx.get(cid, base_ctx)
            try:
                r = ArcFoldRunner(
                    architecture=arch,
                    signal_evaluation=pool.signal_evaluation,
                    panels=panels,
                    run_context=ctx,
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

    # ── Amendment 3 evaluation ───────────────────────────────────────
    log.info("Amendment 3 — per-top-K evaluation")
    amended_results: list[CandidateAmendedResult] = []
    for cand in s5.top_k:
        try:
            res = _run_amendment_3_for_candidate(
                cand=cand,
                strategy_results_by_cid_fold=strategy_results_by_cid_fold,
                holdout_result_by_cid=holdout_results_by_cid,
                panels=panels,
                signal_eval=pool.signal_evaluation,
                per_candidate_ctx=per_candidate_ctx,
                base_ctx=base_ctx,
                pair_set=PAIRS_28,
                out_dir=out,
                r_base=0.005,
                holdout_fold=wfo_struct.holdout,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Amendment 3 eval for %s errored: %s\n%s",
                cand.config_id, exc, traceback.format_exc(),
            )
            continue
        amended_results.append(res)
    amended_wfo = AmendedWfoSearchResult(base=s5, amended_results=tuple(amended_results))

    # ── Amendment 5.1 Gate 4 check (post-Step-5) ───────────────────
    n_candidates_for_a5 = len(candidate_clusters)
    pass_tier_results = [
        r for r in amended_results
        if r.amended_gate.verdict in (AmendedVerdict.PASS_DEPLOYABLE, AmendedVerdict.PASS_VIABLE)
    ]
    a5_gate_4_condition_a = n_candidates_for_a5 >= 2
    a5_gate_4_condition_b = len(pass_tier_results) >= 1

    if a5_gate_4_condition_a and a5_gate_4_condition_b:
        log.error(
            "Amendment 5.1 Gate 4 HALT — both conditions hold: %d candidates, %d PASS-tier constituents. "
            "A5 engine not built — HALT for fresh chat dispatch.",
            n_candidates_for_a5, len(pass_tier_results),
        )
        pass_tier_constituents = [r.config_id for r in pass_tier_results]
        _write_a5_gate_4_halt_diagnostic(
            REPO_ROOT, n_candidates_for_a5, pass_tier_constituents
        )
        # Persist partial run_summary first so the closure has something to read
        _persist_run_summary(
            out=out,
            t_start=t_start,
            pool=pool,
            window_start=window_start,
            window_end=window_end,
            step2=step2,
            step3=step3,
            s4=s4,
            candidates_for_search=candidates_for_search,
            search_scope=search_scope,
            skipped_configs_doc=skipped_configs_doc,
            amended_wfo=amended_wfo,
            step_6_dispatch=None,
            verdict="HALT",
            halt_reason="amendment_5_1_gate_4_a5_not_built",
            best_amended=None,
            oracle_rows=[],
            win_trail_enabled=False,
            win_sl=2.0,
            win_exit_policy=None,
            amendment_5_per_cluster=amendment_5_per_cluster,
            architectures_skipped_under_amendment_5=architectures_skipped_under_amendment_5,
            amendment_5_1_gate_4_status={
                "condition_a_n_candidates_ge_2": a5_gate_4_condition_a,
                "condition_b_pass_tier_present": a5_gate_4_condition_b,
                "pass_tier_constituents": pass_tier_constituents,
            },
            architectures_skipped_by_a1_only=architectures_skipped_by_a1_only,
            a1_only_mode=bool(args.a1_only),
        )
        return 2

    if a5_gate_4_condition_a and not a5_gate_4_condition_b:
        # Record Gate 4 admission blocked
        architectures_skipped_under_amendment_5.add(
            "a5_gate_4_admission_blocked_by_no_pass_tier_constituent"
        )

    # ── Step 6 auto-dispatch (Amendment 4) ───────────────────────────
    step_6_dispatch = None
    if amended_wfo.amended_results:
        log.info("Step 6 — maybe_dispatch_step_6 (auto)")
        try:
            step_6_dispatch = maybe_dispatch_step_6(
                arc_orchestrator_result=_LightOrchestratorView(
                    arc_name="l_arc_5_v3.0.2",
                    pool=pool, step_4=s4, wfo_search=s5,
                    amended_wfo=amended_wfo,
                ),
                amended_wfo=amended_wfo,
                arc_root=out,
                holdout_start=train_end_ts,
                panels=panels,
                feature_matrix=fm_for_step4 if len(feat_matrix) > 0 else None,
                feature_lineage=feat_lineage_renamed,
                signal_module_name=type(signal_module).__module__,
                primary_tf=signal_module.primary_tf,
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
        ranked_amended = sorted(
            amended_wfo.amended_results,
            key=lambda r: _amended_verdict_rank(r.amended_gate.verdict),
            reverse=True,
        )
        top_amended = ranked_amended[0]
        final_verdict = top_amended.amended_gate.verdict.value.upper().replace("_", "-")
    elif s5.top_k:
        final_verdict = s5.top_k[0].gate.verdict.value.upper()

    log.info("Final verdict: %s", final_verdict)

    # ── Oracle WFO per candidate cluster ────────────────────────────
    log.info("Oracle WFO per candidate cluster (locked to winning exit policy)")
    win_trail_enabled = False
    win_sl = 2.0
    win_exit_policy: str | None = None
    if top_amended is not None:
        for cand in s5.top_k:
            if cand.config_id == top_amended.config_id:
                _arch, win_conf = cand.config
                win_trail_enabled = bool(getattr(win_conf, "trail_enabled", False))
                win_sl = float(getattr(win_conf, "sl_atr_mult", 2.0))
                win_exit_policy = getattr(win_conf, "exit_policy", None)
                break

    oracle_rows: list[dict] = []
    for cc in candidate_clusters:
        log.info(
            "  Oracle cluster %d (sl=%.1f, policy=%s, trail=%s)",
            cc.cluster_id, win_sl, win_exit_policy, win_trail_enabled,
        )
        base_a1 = A1Config(
            config_id=f"oracle_cl{cc.cluster_id}",
            sl_atr_mult=win_sl,
            trail_enabled=win_trail_enabled,
            risk_pct=0.005,
            max_concurrent_per_currency=2,
            max_concurrent_per_pair=1,
            max_concurrent_total=None,
            exit_policy=win_exit_policy,
        )
        oracle_runner = OracleFoldRunner(
            signal_evaluation=pool.signal_evaluation,
            panels=panels,
            cluster_assignments=step2.cluster_assignments,
            candidate_cluster_id=cc.cluster_id,
            trades=pool.trades,
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
                "n_trades": sr.fold_stats.n_trades,
                "roi_pct": sr.fold_stats.roi_pct,
                "max_dd_pct": sr.fold_stats.max_dd_pct,
                "days_breaching_daily_5pct": sr.fold_stats.days_breaching_daily_5pct,
                "roi_dd_ratio": sr.fold_stats.roi_dd_ratio,
            })
            per_fold_rows.append({
                "config_id": cid_str, "fold_id": fid,
                "n_trades": sr.fold_stats.n_trades,
                "roi_pct": sr.fold_stats.roi_pct,
                "max_dd_pct": sr.fold_stats.max_dd_pct,
                "roi_dd_ratio": sr.fold_stats.roi_dd_ratio,
            })
    pd.DataFrame(wfo_rows).to_csv(
        s5_dir / "wfo_results.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(per_fold_rows).to_csv(
        s5_dir / "per_fold_metrics.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(oracle_rows).to_csv(
        s5_dir / "wfo_oracle.csv", index=False, lineterminator="\n"
    )
    holdout_rows = [
        {
            "config_id": cid, "fold_id": s.fold_id, "n_trades": s.n_trades,
            "roi_pct": s.roi_pct, "max_dd_pct": s.max_dd_pct,
            "roi_dd_ratio": s.roi_dd_ratio,
        }
        for cid, s in holdout_fold_stats.items()
    ]
    pd.DataFrame(holdout_rows).to_csv(
        s5_dir / "holdout_results.csv", index=False, lineterminator="\n"
    )

    summary_md = ["# Step 5 — WFO Summary", ""]
    summary_md.append(f"**Total configs evaluated:** {s5.n_candidates_evaluated} ({search_scope})")
    summary_md.append("")
    summary_md.append("## Amendment 5 architecture admission per cluster")
    summary_md.append("")
    summary_md.append("| Cluster | Archetype | Mean OOS AUC | Admitted | Skipped (vs Amendment 1) |")
    summary_md.append("|---:|---|---:|---|---|")
    for cid_int, info in sorted(amendment_5_per_cluster.items()):
        auc_str = f"{info['mean_oos_auc']:.4f}" if info['mean_oos_auc'] is not None else "n/a"
        summary_md.append(
            f"| {cid_int} | {info['archetype']} | {auc_str} | "
            f"{', '.join(info['admitted']) or '(none)'} | "
            f"{', '.join(info['skipped_under_amendment_5']) or '(none)'} |"
        )
    summary_md.append("")
    summary_md.append("## Top-K Amendment 3 results")
    summary_md.append("")
    summary_md.append("| Rank | Config | Verdict (Amended) | Worst ratio | r_safe | r_hard | Chained DD base | Daily breaches @r_safe |")
    summary_md.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for k, ar in enumerate(
        sorted(
            amended_wfo.amended_results,
            key=lambda r: _amended_verdict_rank(r.amended_gate.verdict),
            reverse=True,
        )[:3],
        start=1,
    ):
        g = ar.amended_gate
        summary_md.append(
            f"| {k} | `{ar.config_id}` | {g.verdict.value} | {g.worst_fold_ratio:.2f} | "
            f"{g.r_safe_pct:.4%} | {g.r_hard_pct:.4%} | {ar.chained_max_dd_base_pct:.4%} | "
            f"{g.daily_dd_breaches_at_r_safe} |"
        )
    (s5_dir / "wfo_summary.md").write_text(
        "\n".join(summary_md) + "\n", encoding="utf-8", newline="\n"
    )
    (s5_dir / "skipped_configs.md").write_text(
        "# Step 5 — Skipped Configs\n\n"
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
    _persist_run_summary(
        out=out,
        t_start=t_start,
        pool=pool,
        window_start=window_start,
        window_end=window_end,
        step2=step2,
        step3=step3,
        s4=s4,
        candidates_for_search=candidates_for_search,
        search_scope=search_scope,
        skipped_configs_doc=skipped_configs_doc,
        amended_wfo=amended_wfo,
        step_6_dispatch=step_6_dispatch,
        verdict=final_verdict,
        halt_reason=None,
        best_amended=top_amended,
        oracle_rows=oracle_rows,
        win_trail_enabled=win_trail_enabled,
        win_sl=win_sl,
        win_exit_policy=win_exit_policy,
        amendment_5_per_cluster=amendment_5_per_cluster,
        architectures_skipped_under_amendment_5=architectures_skipped_under_amendment_5,
        amendment_5_1_gate_4_status={
            "condition_a_n_candidates_ge_2": a5_gate_4_condition_a,
            "condition_b_pass_tier_present": a5_gate_4_condition_b,
            "pass_tier_constituents": [r.config_id for r in pass_tier_results],
        },
        architectures_skipped_by_a1_only=architectures_skipped_by_a1_only,
        a1_only_mode=bool(args.a1_only),
    )

    log.info("Arc 5 v3.0.2 driver — done in %.1fs (wall)", time.perf_counter() - t_start)
    return 0


def _persist_run_summary(
    *,
    out: Path,
    t_start: float,
    pool: ArcPool,
    window_start: Any,
    window_end: Any,
    step2: Step2Result,
    step3: Step3Result,
    s4: Step4Result | None,
    candidates_for_search: list[Any],
    search_scope: str,
    skipped_configs_doc: list[str],
    amended_wfo: AmendedWfoSearchResult,
    step_6_dispatch: Any,
    verdict: str,
    halt_reason: str | None,
    best_amended: CandidateAmendedResult | None,
    oracle_rows: list[dict],
    win_trail_enabled: bool,
    win_sl: float,
    win_exit_policy: str | None,
    amendment_5_per_cluster: dict[int, dict[str, Any]],
    architectures_skipped_under_amendment_5: set[str],
    amendment_5_1_gate_4_status: dict[str, Any],
    architectures_skipped_by_a1_only: set[str] | None = None,
    a1_only_mode: bool = False,
) -> None:
    summary_payload: dict[str, Any] = {
        "arc_name": "l_arc_5_v3.0.2",
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_time_seconds": time.perf_counter() - t_start,
        "pool_size": int(len(pool.trades)),
        "pool_sha256": pool.pool_sha256,
        "window_start": str(window_start),
        "window_end": str(window_end),
        "boundary_convention": "5ers_eet",
        "primary_tf": "H1",
        "k_selected": int(step2.k_selected),
        "silhouette_per_k": {int(k): float(v) for k, v in step2.silhouette_per_k.items()},
        "clusters": [
            {
                "cluster_id": int(c.cluster_id),
                "n_trades": int(c.n_trades),
                "shape_tag": c.shape_tag,
                "reach_1r": float(c.reach_1r),
                "mfe_p50": float(c.mfe_p50),
                "wrong_way_pp": float(c.wrong_way_pp),
                "selected_sl": float(c.selected_sl),
                "capturability_composite": float(c.capturability_composite),
                "is_candidate": bool(c.is_candidate),
            }
            for c in step3.per_cluster
        ],
        "step4_per_cluster": [
            {
                "cluster_id": int(ce.cluster_id),
                "n_trades": int(ce.n_trades),
                "best_classifier": ce.best_classifier,
                "best_classifier_mean_auc": float(ce.best_classifier_mean_auc),
                "best_threshold": float(ce.best_threshold),
                "excluded_features": list(ce.excluded_features),
                "top_features": (
                    ce.feature_importance.head(10)["feature"].astype(str).tolist()
                    if len(ce.feature_importance) > 0 else []
                ),
                "fitted_classifier_path": (
                    str(ce.fitted_classifier_path) if ce.fitted_classifier_path else None
                ),
            }
            for ce in (s4.per_cluster if s4 else [])
        ],
        "amendment_5_per_cluster": amendment_5_per_cluster,
        "architectures_skipped_under_amendment_5": sorted(architectures_skipped_under_amendment_5),
        "amendment_5_1_gate_4_status": amendment_5_1_gate_4_status,
        "a1_only_mode": bool(a1_only_mode),
        "architectures_skipped_by_a1_only_compute_budget": (
            sorted(architectures_skipped_by_a1_only) if architectures_skipped_by_a1_only else []
        ),
        "step5": {
            "n_configs": len(candidates_for_search),
            "search_scope": search_scope,
            "skipped_configs": skipped_configs_doc,
            "top_3": [
                {
                    "config_id": ar.config_id,
                    "verdict": ar.amended_gate.verdict.value,
                    "primary_failure_mode": ar.amended_gate.primary_failure_mode.value,
                    "reason": ar.amended_gate.reason,
                    "worst_fold_ratio": float(ar.amended_gate.worst_fold_ratio),
                    "worst_fold_roi_base_pct": float(ar.amended_gate.worst_fold_roi_base_pct),
                    "worst_fold_dd_base_pct": float(ar.amended_gate.worst_fold_dd_base_pct),
                    "mean_fold_ratio": float(ar.amended_gate.mean_fold_ratio),
                    "n_negative_folds": int(ar.amended_gate.n_negative_folds),
                    "min_trades_per_fold": int(ar.amended_gate.min_trades_per_fold),
                    "chained_max_dd_base_pct": float(ar.chained_max_dd_base_pct),
                    "chained_dd_method": ar.chained_dd_method,
                    "per_day_max_dd_artefact_path": (
                        str(ar.per_day_max_dd_artefact_path)
                        if ar.per_day_max_dd_artefact_path else None
                    ),
                    "k_safe": float(ar.amended_gate.k_safe),
                    "k_hard": float(ar.amended_gate.k_hard),
                    "r_safe_pct": float(ar.amended_gate.r_safe_pct),
                    "r_hard_pct": float(ar.amended_gate.r_hard_pct),
                    "scalable_to_safe": bool(ar.amended_gate.scalable_to_safe),
                    "scalable_to_hard": bool(ar.amended_gate.scalable_to_hard),
                    "worst_fold_roi_at_r_safe_pct": float(ar.amended_gate.worst_fold_roi_at_r_safe_pct),
                    "worst_fold_roi_at_r_hard_pct": float(ar.amended_gate.worst_fold_roi_at_r_hard_pct),
                    "chained_max_dd_at_r_safe_pct": float(ar.amended_gate.chained_max_dd_at_r_safe_pct),
                    "chained_max_dd_at_r_hard_pct": float(ar.amended_gate.chained_max_dd_at_r_hard_pct),
                    "daily_dd_breaches_at_r_safe": int(ar.amended_gate.daily_dd_breaches_at_r_safe),
                    "daily_dd_breaches_at_r_hard": int(ar.amended_gate.daily_dd_breaches_at_r_hard),
                    "holdout_roi_at_r_safe_pct": ar.amended_gate.holdout_roi_at_r_safe_pct,
                    "holdout_dd_at_r_safe_pct": ar.amended_gate.holdout_dd_at_r_safe_pct,
                    "holdout_roi_at_r_hard_pct": ar.amended_gate.holdout_roi_at_r_hard_pct,
                    "holdout_dd_at_r_hard_pct": ar.amended_gate.holdout_dd_at_r_hard_pct,
                    "sizing_convention": ar.amended_gate.sizing_convention,
                }
                for ar in sorted(
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
                "trigger": (
                    step_6_dispatch.step_6_result.trigger.value
                    if step_6_dispatch.step_6_result else "auto_pass"
                ),
                "overall_passed": (
                    bool(step_6_dispatch.step_6_result.overall_passed)
                    if step_6_dispatch.step_6_result else None
                ),
                "manifest_path": (
                    str(step_6_dispatch.out_dir / "manifest.json")
                    if step_6_dispatch.out_dir else None
                ),
                "categories": {
                    c.category: bool(c.passed)
                    for c in (
                        step_6_dispatch.step_6_result.categories
                        if step_6_dispatch.step_6_result else []
                    )
                },
                "critical_failures": (
                    step_6_dispatch.step_6_result.critical_failures()
                    if step_6_dispatch.step_6_result else []
                ),
                "warnings_count": sum(
                    c.n_warnings for c in (
                        step_6_dispatch.step_6_result.categories
                        if step_6_dispatch.step_6_result else []
                    )
                ),
                "verdict_impact": (
                    "downgraded_to_fail" if step_6_dispatch.downgrade_top_1 else "none"
                ),
                "divergence_warning": bool(step_6_dispatch.divergence_warning),
            }
        ),
        "verdict": verdict,
        "halt_reason": halt_reason,
        "best_config_id": best_amended.config_id if best_amended else None,
        "oracle": {
            f"cluster_{cl_id}": {
                "fold_stats": [r for r in oracle_rows if r["cluster_id"] == cl_id],
            }
            for cl_id in sorted({r["cluster_id"] for r in oracle_rows})
        },
        "winning_exit_policy": win_exit_policy if win_exit_policy else ("sl_plus_trailing_atr" if win_trail_enabled else "sl_only"),
        "winning_sl_atr_mult": win_sl,
    }
    (out / "run_summary.json").write_text(
        json.dumps(summary_payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
