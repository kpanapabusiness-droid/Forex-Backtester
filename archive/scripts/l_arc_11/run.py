"""Arc 11 — canonical-infra driver.

Uses ONLY canonical v3 infrastructure (PR #168). Bypasses
``ArcOrchestrator._run_step_5`` (which has a wiring gap: doesn't plumb
``run_context`` through to ``ArcFoldRunner``, which means A2/A6/A4
architectures requiring ``per_trade_features`` cannot be evaluated via
the orchestrator's Step 5). Instead invokes the canonical
``run_search`` / ``run_holdout`` via ``ArcFoldRunner`` directly with
proper ``A1RunContext`` — this is legitimate canonical usage; the
orchestrator's helper just doesn't wire it.

The driver:

1. Builds H4 / D1 / W1 panels from the v3 cache (warm).
2. Constructs an :class:`SHBSignalModule` for the SHB signal.
3. Builds the trade pool via ``build_arc_pool``.
4. Builds the v3 27-feature matrix (per-trade entry-time features).
5. Runs Steps 2 / 3 / 4 via the canonical step runners.
6. Pre-fits A2/A6 classifier on the 2010-2020 training window
   (Amendment 2: "no retrain at Step 5").
7. Builds per_trade_features lookup for A2/A6 run_context.
8. Runs Step 5 via ``run_search`` / ``run_holdout`` directly on
   ``ArcFoldRunner`` instances (one per architecture).
9. Emits ARC_CLOSURE.md in v1.0 template format
   (``docs/templates/ARC_CLOSURE_TEMPLATE.md``).
10. Writes all artefacts to ``results/l_arc_11/``.

Surfaces an infrastructure finding: ``ArcOrchestrator._run_step_5``
silently runs A2/A6/A4 as no-admit baselines because it does not
construct ``A1RunContext`` from ``cfg.feature_matrix``. Documented in
the closure §3 cross_arc_tags.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool, write_arc_pool  # noqa: E402
from core.architectures._protocol import StrategyResult  # noqa: E402
from core.architectures.a1_system_level_filter import (  # noqa: E402
    A1Architecture,
    A1Config,
    A1RunContext,
)
from core.architectures.a2_classifier_filter import A2Architecture, A2Config  # noqa: E402
from core.architectures.a6_meta_labeling import A6Architecture, A6Config  # noqa: E402
from core.determinism import RANDOM_STATE, seed_everything  # noqa: E402
from core.features.pipeline import compute_feature_matrix  # noqa: E402
from core.runners.arc_fold_runner import ArcFoldRunner  # noqa: E402
from core.sim.panel import Panel  # noqa: E402
from core.steps._classifier_defaults import build_lgbm, build_lr, build_rf  # noqa: E402
from core.steps.step_2_clustering import run_step_2  # noqa: E402
from core.steps.step_3_capturability import run_step_3  # noqa: E402
from core.steps.step_4_extraction import run_step_4  # noqa: E402
from core.strategies.shb.signal_module import SHBSignalModule  # noqa: E402
from core.wfo.folds import Fold, build_v3_folds  # noqa: E402
from core.wfo.gates import FoldStats  # noqa: E402
from core.wfo.orchestrator import (  # noqa: E402
    CandidateHoldoutResult,
    CandidateSearchResult,
    run_holdout,
    run_search,
)

ARC_NAME = "l_arc_11"
PAIRS: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)
HISTDATA_ROOT = r"C:\Users\panap\Documents\Forex-Backtester\data\histdata"
CACHE_ROOT = r"C:\Users\panap\Documents\Forex-Backtester\data\cache"
WINDOW_START = date(2010, 1, 1)
WINDOW_END = date(2026, 4, 30)
TRAIN_END = date(2020, 12, 31)
HOLDOUT_START = date(2021, 1, 1)
SL_ATR_MULT = 2.0
HOLD_BARS = 240
RISK_PCT = 0.005
STARTING_BALANCE = 100_000.0
MIN_IS_DAYS = 365
TOP_K_HOLDOUT = 3


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[arc_11 canonical {ts}] {msg}", flush=True)


# ─── AuxPanel wrapper for multi_tf features ─────────────────────────


class AuxPanel:
    def __init__(self, h4_panel: Panel, aux: dict | None = None) -> None:
        self._h4 = h4_panel
        self.pair_dfs = h4_panel.pair_dfs
        self.tf = h4_panel.tf
        self.aux = aux or {}

    @property
    def pairs(self):
        return self._h4.pairs

    def snapshot_at(self, t):
        return self._h4.snapshot_at(t)


# ─── Helpers ────────────────────────────────────────────────────────


def build_panel(tf: str) -> Panel:
    t0 = time.time()
    panel = Panel.from_pairs(
        list(PAIRS), tf,
        histdata_root=HISTDATA_ROOT,
        cache_root=CACHE_ROOT,
    )
    _log(f"{tf} panel built in {time.time() - t0:.1f}s ({len(panel.pairs)} pairs)")
    return panel


def build_feature_matrix(
    pool_trades: pd.DataFrame,
    h4_panel: Panel,
    d1_panel: Panel,
    w1_panel: Panel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = AuxPanel(h4_panel, aux={"d1": d1_panel, "w1": w1_panel})
    rows: list[pd.DataFrame] = []
    lineage_df = None
    _log("Computing v3 feature matrix per pair at signal bars")
    for pair in sorted(h4_panel.pairs):
        t0 = time.time()
        pair_df = h4_panel.pair_dfs[pair]
        if pair_df.empty:
            continue
        fm = compute_feature_matrix(pair, pair_df, panel=panel)
        if lineage_df is None:
            lineage_df = fm.lineage
        pair_trades = pool_trades[pool_trades["pair"] == pair]
        if pair_trades.empty:
            continue
        signal_times = pd.DatetimeIndex(pair_trades["signal_time"].values, tz="UTC")
        valid_idx = fm.matrix.index.intersection(signal_times)
        if len(valid_idx) == 0:
            continue
        sub = fm.matrix.loc[valid_idx].copy()
        sub["signal_time"] = sub.index
        sub["pair"] = pair
        rows.append(sub.reset_index(drop=True))
        _log(f"  features {pair}: {len(sub)} signal rows in {time.time() - t0:.1f}s")

    if not rows:
        return pd.DataFrame(), lineage_df if lineage_df is not None else pd.DataFrame()
    features_all = pd.concat(rows, ignore_index=True)
    pool_keys = pool_trades[["trade_id", "pair", "signal_time"]].copy()
    pool_keys["signal_time"] = pd.to_datetime(pool_keys["signal_time"], utc=True)
    features_all["signal_time"] = pd.to_datetime(features_all["signal_time"], utc=True)
    merged = features_all.merge(pool_keys, on=["pair", "signal_time"], how="inner")
    feat_cols = [c for c in merged.columns if c not in ("signal_time", "pair", "trade_id")]
    fm_indexed = merged[["trade_id"] + feat_cols].set_index("trade_id")
    return fm_indexed, lineage_df


def build_classifier(name: str):
    if name == "rf":
        return build_rf()
    if name == "lgbm":
        return build_lgbm()
    if name == "lr":
        return build_lr()
    raise ValueError(f"unknown classifier {name!r}")


def prefit_classifier(
    pool_trades: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    candidate_cluster_id: int,
    train_end: date,
    classifier_name: str,
) -> tuple[Any, tuple[str, ...]]:
    pool = pool_trades.copy()
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True)
    train_cutoff = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
    train_trades = pool[pool["entry_time"] < train_cutoff]
    train_ids = set(train_trades["trade_id"])
    fm_train = feature_matrix.loc[feature_matrix.index.isin(train_ids)]
    ca_train = cluster_assignments[cluster_assignments["trade_id"].isin(train_ids)]
    if fm_train.empty or ca_train.empty:
        raise RuntimeError("No training-window data for prefit")
    y_map = ca_train.set_index("trade_id")["cluster_id"].to_dict()
    y = np.array([
        1 if y_map.get(tid, -1) == candidate_cluster_id else 0
        for tid in fm_train.index
    ], dtype=int)
    X = fm_train.copy().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    model = build_classifier(classifier_name)
    model.fit(X, y)
    _log(f"  classifier ({classifier_name}) fit on {len(X)} train trades; "
         f"pos={int(y.sum())}/{len(y)}")
    return model, tuple(fm_train.columns)


def build_per_trade_features(
    pool_trades: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """Build (pair, signal_time) -> feature_dict lookup for A2/A6 run_context."""
    pool = pool_trades.copy()
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True)
    pool_by_tid = pool.set_index("trade_id")
    out: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for tid, row in feature_matrix.iterrows():
        if tid not in pool_by_tid.index:
            continue
        prow = pool_by_tid.loc[tid]
        key = (str(prow["pair"]), pd.Timestamp(prow["signal_time"]))
        out[key] = {col: float(row[col]) if pd.notna(row[col]) else 0.0
                    for col in row.index}
    return out


# ─── Main ──────────────────────────────────────────────────────────


def main() -> int:
    seed_everything(RANDOM_STATE)
    out_dir = _REPO_ROOT / "results" / ARC_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. Panels
    h4_panel = build_panel("H4")
    d1_panel = build_panel("D1")
    w1_panel = build_panel("W1")

    # 2. Signal + pool
    signal_module = SHBSignalModule()
    panels = {"H4": h4_panel}
    _log("Building Step 1 trade pool (canonical build_arc_pool)")
    pool_cfg = ArcPoolConfig(
        arc_name=ARC_NAME,
        sl_atr_mult=SL_ATR_MULT,
        hold_bars=HOLD_BARS,
        risk_pct=RISK_PCT,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )
    pool = build_arc_pool(signal_module, panels, pool_cfg)
    _log(f"  pool n={len(pool.trades)}, pool_sha256={pool.pool_sha256[:16]}...")
    write_arc_pool(pool, out_dir)

    # 3. Feature matrix
    feature_matrix, feature_lineage = build_feature_matrix(
        pool.trades, h4_panel, d1_panel, w1_panel,
    )
    _log(f"  feature_matrix shape: {feature_matrix.shape}; lineage rows: {len(feature_lineage)}")

    # 4. Steps 2, 3, 4
    _log("Step 2 (clustering)")
    s2 = run_step_2(pool.trades, pool.paths)
    _log(f"  best K = {s2.k_selected}; silhouettes = {s2.silhouette_per_k}")
    s2_dir = out_dir / "step_2"
    s2_dir.mkdir(exist_ok=True)
    s2.cluster_assignments.to_parquet(s2_dir / "cluster_assignments.parquet",
                                      engine="pyarrow", compression="snappy", index=False)
    s2.cluster_summary.to_csv(s2_dir / "cluster_summary.csv", index=False, lineterminator="\n")
    (s2_dir / "cluster_summary.md").write_text(s2.summary_md, encoding="utf-8", newline="\n")

    _log("Step 3 (capturability)")
    s3 = run_step_3(
        pool.trades, pool.paths, s2.cluster_assignments,
        declared_sl_mult=SL_ATR_MULT, cluster_centroids=s2.centroids,
    )
    s3_dir = out_dir / "step_3"
    s3_dir.mkdir(exist_ok=True)
    s3.capturability_csv.to_csv(s3_dir / "capturability.csv", index=False, lineterminator="\n")
    (s3_dir / "capturability_summary.md").write_text(s3.summary_md, encoding="utf-8", newline="\n")
    candidate_ids = tuple(c.cluster_id for c in s3.per_cluster if c.is_candidate)
    _log(f"  candidate clusters: {candidate_ids}")

    _log("Step 4 (extraction)")
    s4 = None
    primary_cluster_id = None
    primary_classifier_name = None
    primary_threshold = 0.5
    if candidate_ids:
        s4 = run_step_4(
            pool.trades,
            feature_matrix.reset_index(),
            s2.cluster_assignments,
            feature_lineage=feature_lineage,
            candidate_cluster_ids=candidate_ids,
        )
        s4_dir = out_dir / "step_4"
        s4_dir.mkdir(exist_ok=True)
        s4.extraction_metrics.to_csv(s4_dir / "extraction_metrics.csv", index=False, lineterminator="\n")
        s4.feature_importance.to_csv(s4_dir / "feature_importance.csv", index=False, lineterminator="\n")
        (s4_dir / "extraction_summary.md").write_text(s4.summary_md, encoding="utf-8", newline="\n")
        if s4.per_cluster:
            # Pick the highest-AUC cluster
            primary = max(s4.per_cluster, key=lambda e: e.best_classifier_mean_auc)
            primary_cluster_id = int(primary.cluster_id)
            primary_classifier_name = primary.best_classifier
            primary_threshold = float(primary.best_threshold)
            _log(f"  primary cluster {primary_cluster_id}: "
                 f"{primary_classifier_name} AUC {primary.best_classifier_mean_auc:.4f} "
                 f"threshold {primary_threshold:.4f}")

    # 5. Pre-fit classifier + build per_trade_features
    classifier = None
    feat_order: tuple[str, ...] = ()
    if primary_cluster_id is not None and primary_classifier_name is not None:
        classifier, feat_order = prefit_classifier(
            pool_trades=pool.trades,
            feature_matrix=feature_matrix,
            cluster_assignments=s2.cluster_assignments,
            candidate_cluster_id=primary_cluster_id,
            train_end=TRAIN_END,
            classifier_name=primary_classifier_name,
        )

    per_trade_features = build_per_trade_features(pool.trades, feature_matrix)
    _log(f"  per_trade_features lookup: {len(per_trade_features)} entries")

    # 6. Architecture configs
    archs_configs: list[tuple[str, Any, Any]] = []  # (config_id, architecture, arch_config)

    a1_cfg = A1Config(
        config_id="a1_shb_baseline",
        sl_atr_mult=SL_ATR_MULT,
        trail_enabled=False,
        risk_pct=RISK_PCT,
        starting_balance=STARTING_BALANCE,
        max_concurrent_per_pair=1,
        max_concurrent_per_currency=2,
        time_exit_bars=HOLD_BARS,
    )
    archs_configs.append((a1_cfg.config_id, A1Architecture(), a1_cfg))

    # A2 + A6 only if a classifier was successfully prefitted
    if classifier is not None:
        a2_cfg = A2Config(
            config_id=f"a2_shb_cluster{primary_cluster_id}",
            classifier=classifier,
            threshold=primary_threshold,
            classifier_feature_order=feat_order,
            sl_atr_mult=SL_ATR_MULT,
            trail_enabled=False,
            risk_pct=RISK_PCT,
            starting_balance=STARTING_BALANCE,
            max_concurrent_per_pair=1,
            max_concurrent_per_currency=2,
        )
        a6_cfg = A6Config(
            config_id=f"a6_shb_metalabel_0.4_0.6_c{primary_cluster_id}",
            classifier=classifier,
            lower_threshold=0.4,
            upper_threshold=0.6,
            classifier_feature_order=feat_order,
            sl_atr_mult=SL_ATR_MULT,
            trail_enabled=False,
            risk_pct=RISK_PCT,
            starting_balance=STARTING_BALANCE,
            max_concurrent_per_pair=1,
            max_concurrent_per_currency=2,
        )
        archs_configs.extend([
            (a2_cfg.config_id, A2Architecture(), a2_cfg),
            (a6_cfg.config_id, A6Architecture(), a6_cfg),
        ])
    _log(f"Architectures + configs to evaluate: {[c[0] for c in archs_configs]}")

    # 7. Step 5 — run_search per architecture, then aggregate top-K
    wfo_struct = build_v3_folds()
    _log(f"WFO structure: {wfo_struct.n_folds} folds; "
         f"holdout {wfo_struct.holdout.oos_start} to {wfo_struct.holdout.oos_end}")

    all_search_results: list[CandidateSearchResult] = []
    last_results_by_config: dict[str, list[StrategyResult]] = {}
    for cfg_id, arch, arch_cfg in archs_configs:
        _log(f"Running search for {cfg_id}")
        # A2/A6 need run_context with per_trade_features; A1 ignores it
        ctx = A1RunContext(per_trade_features=per_trade_features)
        runner = ArcFoldRunner(
            architecture=arch,
            signal_evaluation=pool.signal_evaluation,
            panels=panels,
            run_context=ctx,
        )
        # Wrap into a fold-runner callable that the orchestrator expects
        per_config_results: list[StrategyResult] = []
        def fold_runner(fold: Fold, config: Any, _runner=runner, _store=per_config_results) -> FoldStats:
            stats = _runner(fold, config)
            if _runner.last_result is not None:
                _store.append(_runner.last_result)
            return stats
        search_result = run_search(
            wfo_struct,
            [(cfg_id, arch_cfg)],
            fold_runner=fold_runner,
            min_is_days=MIN_IS_DAYS,
            top_k=1,
        )
        all_search_results.extend(search_result.candidates)
        last_results_by_config[cfg_id] = per_config_results
        if search_result.top_k:
            g = search_result.top_k[0].gate
            _log(f"  {cfg_id}: verdict={g.verdict.value} ratio={g.worst_fold_ratio:.3f} "
                 f"roi={g.worst_fold_roi:+.4f} dd={g.worst_fold_dd:.4f} "
                 f"min_n={g.min_trades_per_fold}")

    # 8. Rank across architectures, take top-K for holdout
    ranked = sorted(
        all_search_results,
        key=lambda r: (r.gate.worst_fold_ratio, r.gate.mean_fold_ratio),
        reverse=True,
    )
    top_k = tuple(ranked[:TOP_K_HOLDOUT])
    _log(f"Top-{len(top_k)} for holdout: {[c.config_id for c in top_k]}")

    # 9. Holdout one-shot per top-K
    holdout_results: list[CandidateHoldoutResult] = []
    for tk in top_k:
        # Match each top-K back to its architecture
        match = next((a for a in archs_configs if a[0] == tk.config_id), None)
        if match is None:
            continue
        cfg_id, arch, arch_cfg = match
        ctx = A1RunContext(per_trade_features=per_trade_features)
        runner = ArcFoldRunner(
            architecture=arch,
            signal_evaluation=pool.signal_evaluation,
            panels=panels,
            run_context=ctx,
        )
        def holdout_runner(fold: Fold, config: Any, _runner=runner) -> FoldStats:
            return _runner(fold, config)
        h_result = run_holdout(
            wfo_struct,
            (tk,),
            fold_runner=holdout_runner,
        )
        holdout_results.extend(h_result)

    for hr in holdout_results:
        sg = hr.search_gate
        hg = hr.holdout_gate
        _log(f"  HOLDOUT {hr.config_id}: search={sg.verdict.value} "
             f"holdout={hg.verdict.value} deployable={hr.deployable}")

    # 10. Build artefacts
    wfo_search_csv = pd.DataFrame([
        {
            "config_id": c.config_id,
            "verdict": c.gate.verdict.value,
            "worst_fold_ratio": c.gate.worst_fold_ratio,
            "worst_fold_roi": c.gate.worst_fold_roi,
            "worst_fold_dd": c.gate.worst_fold_dd,
            "mean_fold_ratio": c.gate.mean_fold_ratio,
            "n_negative_folds": c.gate.n_negative_folds,
            "min_trades_per_fold": c.gate.min_trades_per_fold,
            "max_dd_across_folds": c.gate.max_dd_across_folds,
            "n_folds": len(c.fold_stats),
        }
        for c in all_search_results
    ])
    step5_dir = out_dir / "step_5"
    step5_dir.mkdir(exist_ok=True)
    wfo_search_csv.to_csv(step5_dir / "wfo_results.csv", index=False, lineterminator="\n")

    # Per-fold per-config CSV
    per_fold_rows = []
    for c in all_search_results:
        for f in c.fold_stats:
            per_fold_rows.append({
                "config_id": c.config_id,
                "fold_id": f.fold_id,
                "n_trades": f.n_trades,
                "roi_pct": f.roi_pct,
                "max_dd_pct": f.max_dd_pct,
                "days_breaching_daily_5pct": f.days_breaching_daily_5pct,
                "roi_dd_ratio": f.roi_dd_ratio,
            })
    if per_fold_rows:
        pd.DataFrame(per_fold_rows).to_csv(
            step5_dir / "per_fold_metrics.csv", index=False, lineterminator="\n"
        )

    holdout_csv = pd.DataFrame([
        {
            "config_id": h.config_id,
            "search_verdict": h.search_gate.verdict.value,
            "search_worst_fold_ratio": h.search_gate.worst_fold_ratio,
            "holdout_verdict": h.holdout_gate.verdict.value,
            "holdout_roi_pct": h.holdout_stats.roi_pct,
            "holdout_dd_pct": h.holdout_stats.max_dd_pct,
            "holdout_ratio": h.holdout_stats.roi_dd_ratio,
            "holdout_n_trades": h.holdout_stats.n_trades,
            "deployable": h.deployable,
        }
        for h in holdout_results
    ])
    if not holdout_csv.empty:
        holdout_csv.to_csv(step5_dir / "holdout_results.csv", index=False, lineterminator="\n")

    # Verdict
    if not ranked:
        arc_verdict = "FAIL"
    else:
        best = ranked[0]
        h_match = next((h for h in holdout_results if h.config_id == best.config_id), None)
        if h_match is not None:
            if h_match.deployable:
                arc_verdict = "PASS-DEPLOYABLE"
            elif (best.gate.verdict.value == "pass_viable"
                  and h_match.holdout_gate.verdict.value in ("pass_deployable", "pass_viable")):
                arc_verdict = "PASS-VIABLE"
            else:
                arc_verdict = "FAIL"
        else:
            arc_verdict = best.gate.verdict.value.replace("_", "-").upper()

    summary = {
        "arc_name": ARC_NAME,
        "verdict": arc_verdict,
        "pool_size": int(len(pool.trades)),
        "pool_sha256": pool.pool_sha256,
        "k_selected": int(s2.k_selected),
        "silhouettes": {int(k): float(v) for k, v in s2.silhouette_per_k.items()},
        "candidate_cluster_ids": [int(c.cluster_id) for c in s3.per_cluster if c.is_candidate],
        "primary_cluster": primary_cluster_id,
        "primary_classifier": primary_classifier_name,
        "primary_threshold": primary_threshold,
        "n_configs_evaluated": len(all_search_results),
        "n_folds": wfo_struct.n_folds,
        "search_results": [
            {
                "config_id": c.config_id,
                "verdict": c.gate.verdict.value,
                "worst_fold_ratio": float(c.gate.worst_fold_ratio),
                "worst_fold_roi": float(c.gate.worst_fold_roi),
                "worst_fold_dd": float(c.gate.worst_fold_dd),
                "mean_fold_ratio": float(c.gate.mean_fold_ratio),
                "n_negative_folds": int(c.gate.n_negative_folds),
                "min_trades_per_fold": int(c.gate.min_trades_per_fold),
                "n_folds_evaluated": len(c.fold_stats),
            }
            for c in ranked
        ],
        "holdout_results": [
            {
                "config_id": h.config_id,
                "search_verdict": h.search_gate.verdict.value,
                "holdout_verdict": h.holdout_gate.verdict.value,
                "holdout_roi_pct": float(h.holdout_stats.roi_pct),
                "holdout_dd_pct": float(h.holdout_stats.max_dd_pct),
                "holdout_ratio": float(h.holdout_stats.roi_dd_ratio),
                "deployable": bool(h.deployable),
            }
            for h in holdout_results
        ],
        "step_4_per_cluster": [
            {
                "cluster_id": int(e.cluster_id),
                "best_classifier": e.best_classifier,
                "best_classifier_mean_auc": float(e.best_classifier_mean_auc),
                "best_threshold": float(e.best_threshold),
                "n_trades": int(e.n_trades),
                "top_10_features": e.feature_importance.head(10)["feature"].tolist()
                    if not e.feature_importance.empty else [],
            }
            for e in (s4.per_cluster if s4 else ())
        ],
        "step_3_per_cluster": [
            {
                "cluster_id": int(c.cluster_id),
                "n_trades": int(c.n_trades),
                "shape_tag": c.shape_tag,
                "selected_sl_mult": float(c.selected_sl_mult),
                "composite": float(c.composite),
                "reach_1r": float(c.reach_1r),
                "mfe_p50": float(c.mfe_p50),
                "ww_pp": float(c.ww_pp),
                "is_candidate": bool(c.is_candidate),
            }
            for c in s3.per_cluster
        ],
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    _log(f"Verdict: {arc_verdict}; total elapsed: {summary['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
