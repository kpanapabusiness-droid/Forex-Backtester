"""Arc 8 v3.0.2 Step 5 — direct `run_search` + `run_holdout` driver (Arc 11 pattern).

Per chat decision (Step 5 Pipeline Decision §1 — Option 1): bypasses
``ArcOrchestrator.run()`` because my custom Step 1-4 outputs use an
arc-local schema that doesn't match the canonical engine's path-feature
expectations. Uses ``core.wfo.orchestrator.run_search`` /
``core.wfo.orchestrator.run_holdout`` directly with my pre-computed
Step 1-4 outputs as inputs. Amendment 3 evaluation invoked via a stub
``ArcOrchestrator`` instance per chat directive. Step 6 auto-dispatches
via ``maybe_dispatch_step_6`` if Top-1 lands PASS-tier.

Inputs (read from disk):
  - results/l_arc_8_v3.0.2/step_1/pool.parquet
  - results/l_arc_8_v3.0.2/step_1/paths.parquet
  - results/l_arc_8_v3.0.2/step_1/features.parquet
  - results/l_arc_8_v3.0.2/step_2/cluster_assignments.parquet
  - results/l_arc_8_v3.0.2/step_3/capturability.csv
  - results/l_arc_8_v3.0.2/step_4/amendment_5_admission.json

Architecture set per Amendment 5 admission (c2 Bimodal, AUC 0.48 < 0.65):
  {A1, A4}, 18 configs each = 36 total.

Exit slate (Bimodal canonical per L_PROTOCOL §2 Step 5):
  {sl_only, sl_partial_close_1r_runner_trail, sl_plus_tp_2r}.

SL multipliers (Step 3 c2 optimum 1.5 +/- 1 step): {1.0, 1.5, 2.0}.
Exposure caps: {max_per_currency=2, unlimited}.

WFO: 11-fold 2010-2020 + holdout 2021-2025 (build_v3_folds with
holdout_end=2025-12-31).
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from core.arc.arc_orchestrator import ArcConfig, ArcOrchestrator
from core.arc.arc_pool_builder import ArcPool
from core.arc.signal_protocol import SignalEvaluation
from core.architectures._protocol import Architecture
from core.architectures.a1_system_level_filter import (
    A1Architecture,
    A1Config,
    A1RunContext,
)
from core.architectures.a4_pipeline_d_exits import A4Architecture, A4Config
from core.determinism import seed_everything, write_text_deterministic
from core.manifest import write_manifest
from core.parallel import build_panel_parallel, default_pool_size
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.step_6.dispatch import maybe_dispatch_step_6
from core.steps.path_classifier_per_fold import (
    PerFoldTrainingInputs,
    build_path_classifier_fits_per_fold,
    build_per_trade_entry_features,
)
from core.strategies.pullback_resume_hhhl_long import PullbackResumeHHHLLongSignal
from core.wfo.folds import Fold, build_v3_folds
from core.wfo.orchestrator import run_holdout, run_search
from scripts.l_arc_8_v3_0_2.shared import (
    ARC_NAME,
    BOUNDARY_CONVENTION,
    CACHE_ROOT,
    HISTDATA_ROOT,
    PAIRS_28,
    RESULTS_ROOT,
    RISK_PCT,
    SIGNAL_PARAMS,
    WINDOW_END_TARGET,
    WINDOW_START,
)

STEP_DIR: Path = RESULTS_ROOT / "step_5"

SL_MULTIPLIERS: tuple[float, ...] = (1.0, 1.5, 2.0)
EXIT_POLICIES_BIMODAL: tuple[str, ...] = (
    "sl_only",
    "sl_partial_close_1r_runner_trail",
    "sl_plus_tp_2r",
)
EXPOSURE_CAPS: tuple[tuple[str, int | None], ...] = (
    ("max_per_currency_2", 2),
    ("unlimited", None),
)

HOLDOUT_END_DATE: date = date(2025, 12, 31)


def _build_panels() -> dict[str, Panel]:
    """Build H4 / D1 / W1 panels at 5ers_eet, sliced to the arc window."""
    win_start = pd.Timestamp(WINDOW_START, tz="UTC")
    win_end = pd.Timestamp(WINDOW_END_TARGET, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    pool_size = default_pool_size(len(PAIRS_28))
    print(f"[step5] Building panels (pool_size={pool_size}, boundary={BOUNDARY_CONVENTION})...")
    panel_h4 = build_panel_parallel(
        list(PAIRS_28), "H4",
        histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
        pool_size=pool_size, boundary_convention=BOUNDARY_CONVENTION,
    )
    panel_h4 = Panel.from_frames(
        {p: df.loc[win_start:win_end] for p, df in panel_h4.pair_dfs.items()},
        tf="H4", boundary_convention=BOUNDARY_CONVENTION,
    )
    panel_d1 = build_panel_parallel(
        list(PAIRS_28), "D1",
        histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
        pool_size=pool_size, boundary_convention=BOUNDARY_CONVENTION,
    )
    panel_d1 = Panel.from_frames(
        {p: df.loc[win_start:win_end] for p, df in panel_d1.pair_dfs.items()},
        tf="D1", boundary_convention=BOUNDARY_CONVENTION,
    )
    panel_w1 = build_panel_parallel(
        list(PAIRS_28), "W1",
        histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
        pool_size=pool_size, boundary_convention=BOUNDARY_CONVENTION,
    )
    panel_w1 = Panel.from_frames(
        {p: df.loc[win_start:win_end] for p, df in panel_w1.pair_dfs.items()},
        tf="W1", boundary_convention=BOUNDARY_CONVENTION,
    )
    object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})
    return {"H4": panel_h4, "D1": panel_d1, "W1": panel_w1}


def _canonicalise_paths(paths_my: pd.DataFrame) -> pd.DataFrame:
    """Convert my arc-local paths schema to the canonical engine schema.

    My schema: trade_id, pair, bar_idx, bar_time, low_r, high_r, close_r.
    Canonical: trade_id, bar_offset, close_r, mfe_so_far_r, mae_so_far_r.

    mfe_so_far_r = cummax(high_r) per trade.
    mae_so_far_r = cummin(low_r) per trade.
    """
    df = paths_my.sort_values(["trade_id", "bar_idx"]).copy()
    df["bar_offset"] = df["bar_idx"].astype(int)
    g = df.groupby("trade_id", sort=False)
    df["mfe_so_far_r"] = g["high_r"].cummax()
    df["mae_so_far_r"] = g["low_r"].cummin()
    return df[["trade_id", "bar_offset", "close_r", "mfe_so_far_r", "mae_so_far_r"]].copy()


def _canonicalise_pool(pool_my: pd.DataFrame) -> pd.DataFrame:
    """Rename my pool fields to match canonical expectations.

    My pool fields: ... atr_at_entry, sl_price ...
    Canonical (for path_classifier_per_fold): atr_at_signal, signal_time, entry_time, ...

    Returns a copy with both old + canonical names so consumers find what they need.
    """
    df = pool_my.copy()
    if "atr_at_signal" not in df.columns and "atr_at_entry" in df.columns:
        df["atr_at_signal"] = df["atr_at_entry"]
    if "sl_at_entry_price" not in df.columns and "sl_price" in df.columns:
        df["sl_at_entry_price"] = df["sl_price"]
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    return df


def _build_per_trade_features(
    pool_trades: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """Build (pair, signal_time) -> feature dict lookup from my features.parquet.

    Mirrors `core.arc.arc_orchestrator._build_per_trade_features` semantics.
    Drops metadata cols before iterating.
    """
    fm = feature_matrix.copy()
    meta_cols = [c for c in ("pair", "signal_time") if c in fm.columns]
    if "trade_id" in fm.columns:
        fm = fm.set_index("trade_id")
    fm = fm.drop(columns=meta_cols, errors="ignore")
    pool_by_tid = pool_trades.set_index("trade_id")
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


def _build_a1_configs() -> list[A1Config]:
    """18 A1 configs: 3 SLs × 3 exits × 2 exposures."""
    configs: list[A1Config] = []
    for sl in SL_MULTIPLIERS:
        for exit_policy in EXIT_POLICIES_BIMODAL:
            for exp_name, exp_cap in EXPOSURE_CAPS:
                configs.append(A1Config(
                    config_id=f"a1_sl{sl:.1f}_exit-{exit_policy}_exp-{exp_name}",
                    sl_atr_mult=sl,
                    trail_enabled=False,
                    risk_pct=RISK_PCT,
                    starting_balance=100_000.0,
                    max_concurrent_total=None,
                    max_concurrent_per_pair=1,
                    max_concurrent_per_currency=exp_cap,
                    time_exit_bars=240,
                    sizing_convention="reset_floor",
                    exit_policy=exit_policy,
                ))
    return configs


def _build_a4_configs() -> list[A4Config]:
    """18 A4 configs: 3 SLs × 3 exits × 2 exposures. classifier_fit attached at fold-time via A1RunContext."""
    configs: list[A4Config] = []
    for sl in SL_MULTIPLIERS:
        for exit_policy in EXIT_POLICIES_BIMODAL:
            for exp_name, exp_cap in EXPOSURE_CAPS:
                configs.append(A4Config(
                    config_id=f"a4_sl{sl:.1f}_exit-{exit_policy}_exp-{exp_name}",
                    classifier_fit=None,  # threaded via A1RunContext.path_classifier_fits
                    exit_threshold=0.4,
                    sl_atr_mult=sl,
                    trail_enabled=False,
                    risk_pct=RISK_PCT,
                    starting_balance=100_000.0,
                    max_concurrent_per_pair=1,
                    max_concurrent_per_currency=exp_cap,
                    per_trade_entry_features=None,  # threaded via A1RunContext
                    sizing_convention="reset_floor",
                    exit_policy=exit_policy,
                ))
    return configs


def main() -> Path:
    seed_everything(42)
    t0 = time.perf_counter()
    STEP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Step 1-4 artefacts ────────────────────────────────────
    print("[step5] Loading Step 1-4 artefacts...")
    pool_my = pd.read_parquet(RESULTS_ROOT / "step_1" / "pool.parquet")
    paths_my = pd.read_parquet(RESULTS_ROOT / "step_1" / "paths.parquet")
    feat_df = pd.read_parquet(RESULTS_ROOT / "step_1" / "features.parquet")
    lineage_df = pd.read_csv(RESULTS_ROOT / "step_1" / "feature_lineage.csv")
    assignments = pd.read_parquet(RESULTS_ROOT / "step_2" / "cluster_assignments.parquet")
    cap_df = pd.read_csv(RESULTS_ROOT / "step_3" / "capturability.csv")
    admission = json.loads((RESULTS_ROOT / "step_4" / "amendment_5_admission.json").read_text(encoding="utf-8"))
    print(f"[step5] pool: {len(pool_my)} trades, paths: {len(paths_my)} rows, features: {feat_df.shape}")

    # Identify candidate cluster from Step 4 admission.
    candidate_clusters = [
        int(cid) for cid, info in admission.items()
        if info.get("architectures_admitted") and info["is_candidate"]
    ]
    if len(candidate_clusters) != 1:
        raise SystemExit(f"Expected 1 candidate cluster; got {candidate_clusters}.")
    candidate_cid = candidate_clusters[0]
    admitted_archs = set(admission[str(candidate_cid)]["architectures_admitted"])
    print(f"[step5] Candidate cluster c{candidate_cid}; admitted archs: {sorted(admitted_archs)}")
    if admitted_archs != {"A1", "A4"}:
        print(f"[step5] WARN: expected {{A1, A4}}; got {admitted_archs}. Proceeding intersected.")

    # ── Canonicalise pool + paths for canonical engine consumption ─
    pool = _canonicalise_pool(pool_my)
    paths_canon = _canonicalise_paths(paths_my)

    # Cluster assignments — canonical cluster_id from my cluster_primary.
    cluster_assignments_canon = assignments[["trade_id", "cluster_primary"]].rename(
        columns={"cluster_primary": "cluster_id"}
    )

    # ── Build panels ───────────────────────────────────────────────
    panels = _build_panels()
    print(f"[step5] Panels built ({time.perf_counter() - t0:.1f}s).")

    # ── Signal evaluation ──────────────────────────────────────────
    signal_module = PullbackResumeHHHLLongSignal(params=SIGNAL_PARAMS)
    print("[step5] Running signal evaluation...")
    signal_eval = signal_module.evaluate(panels)

    # ── Per-trade feature lookup ──────────────────────────────────
    per_trade_features = _build_per_trade_features(pool, feat_df)
    print(f"[step5] per_trade_features: {len(per_trade_features)} entries")

    # ── WFO structure ─────────────────────────────────────────────
    wfo_struct = build_v3_folds(holdout_end=HOLDOUT_END_DATE)
    holdout_fold = wfo_struct.holdout
    if holdout_fold is None:
        raise SystemExit("WFO holdout fold required but missing")
    print(f"[step5] WFO: {len(wfo_struct.folds)} search folds, 1 holdout (oos {holdout_fold.oos_start}..{holdout_fold.oos_end}).")

    # ── A4 per-fold path-classifier fits ──────────────────────────
    print("[step5] Building A4 per-fold path-classifier fits...")
    folds_for_fits: tuple[Fold, ...] = wfo_struct.folds + (holdout_fold,)
    a4_inputs = PerFoldTrainingInputs(
        pool_trades=pool,
        pool_paths=paths_canon,
        cluster_assignments=cluster_assignments_canon,
        panels=panels,
        primary_tf="H4",
        candidate_cluster_id=None,  # A4 trains on final_r > 0, not cluster
        n_defer=5,
    )
    a4_fits = build_path_classifier_fits_per_fold(
        inputs=a4_inputs, folds=folds_for_fits, arch="A4"
    )
    entry_feats = build_per_trade_entry_features(a4_inputs)
    print(f"[step5] A4 fits: {len(a4_fits)} folds; entry_feats: {len(entry_feats)} trades")

    # ── Build candidates list ─────────────────────────────────────
    a1_arch = A1Architecture()
    a4_arch = A4Architecture()
    a1_configs = _build_a1_configs()
    a4_configs = _build_a4_configs()

    candidates: list[tuple[str, tuple[Architecture, object]]] = []
    per_candidate_arch: dict[str, tuple[Architecture, A1RunContext]] = {}
    base_ctx = A1RunContext(per_trade_features=per_trade_features)
    a4_ctx = A1RunContext(
        per_trade_features=per_trade_features,
        per_trade_entry_features=entry_feats,
        path_classifier_fits=a4_fits,
    )

    for conf in a1_configs:
        cid = f"{a1_arch.architecture_name}::{conf.config_id}"
        candidates.append((cid, (a1_arch, conf)))
        per_candidate_arch[cid] = (a1_arch, base_ctx)
    for conf in a4_configs:
        cid = f"{a4_arch.architecture_name}::{conf.config_id}"
        candidates.append((cid, (a4_arch, conf)))
        per_candidate_arch[cid] = (a4_arch, a4_ctx)

    print(f"[step5] Candidates: {len(candidates)} (A1={len(a1_configs)} + A4={len(a4_configs)})")

    # Side-channel for Amendment 3 evaluation: capture StrategyResults per (cid, fold_id).
    strategy_results: dict[str, dict[int, object]] = {}

    def _fold_runner(fold: Fold, paired: tuple[Architecture, object]):
        arch, conf = paired
        cid = f"{arch.architecture_name}::{getattr(conf, 'config_id', repr(conf))}"
        _arch, ctx = per_candidate_arch[cid]
        r = ArcFoldRunner(
            architecture=arch,
            signal_evaluation=signal_eval,
            panels=panels,
            run_context=ctx,
        )
        stats = r(fold, conf)
        if r.last_result is not None:
            strategy_results.setdefault(cid, {})[fold.fold_id] = r.last_result
        return stats

    # ── Run search ────────────────────────────────────────────────
    print("[step5] Running WFO search...")
    t_search = time.perf_counter()
    s5 = run_search(
        wfo_struct,
        candidates,
        fold_runner=_fold_runner,
        min_is_days=365,
        top_k=3,
    )
    print(f"[step5] Search complete ({time.perf_counter() - t_search:.1f}s); top_k: {[c.config_id for c in s5.top_k]}")

    # ── Run holdout on top-K ──────────────────────────────────────
    print("[step5] Running holdout on top-K...")
    t_holdout = time.perf_counter()
    holdout = run_holdout(wfo_struct, s5.top_k, fold_runner=_fold_runner)
    print(f"[step5] Holdout complete ({time.perf_counter() - t_holdout:.1f}s)")

    # ── Stub ArcOrchestrator for Amendment 3 evaluation ──────────
    print("[step5] Running Amendment 3 evaluation...")
    arc_cfg = ArcConfig(
        arc_name=ARC_NAME,
        signal_class="pullback_resume_hhhl_long",
        pair_set=PAIRS_28,
        window_start=pd.Timestamp(WINDOW_START, tz="UTC"),
        window_end=pd.Timestamp(WINDOW_END_TARGET, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        risk_pct=RISK_PCT,
        sub_protocol="vanilla",
        output_dir=RESULTS_ROOT,
        feature_matrix=feat_df,
        feature_lineage=lineage_df,
        wfo_structure=wfo_struct,
        skip_step_6=False,
        accept_equity_pct=False,
    )
    stub_orch = ArcOrchestrator(arc_cfg, signal_module, panels)
    stub_orch._last_per_candidate_arch = per_candidate_arch
    stub_orch._last_base_ctx = base_ctx
    stub_orch._last_strategy_results = strategy_results
    t_amend = time.perf_counter()
    amended_wfo = stub_orch._run_amendment_3_evaluation(
        s5=s5,
        holdout_results=holdout,
        signal_eval=signal_eval,
        wfo_struct=wfo_struct,
    )
    print(f"[step5] Amendment 3 complete ({time.perf_counter() - t_amend:.1f}s)")
    for ar in amended_wfo.amended_results:
        print(
            f"[step5]   {ar.config_id}: verdict={ar.amended_gate.verdict.value} "
            f"chained_dd={ar.chained_max_dd_base_pct:.4%}"
        )

    # ── Step 6 auto-dispatch ──────────────────────────────────────
    step_6_dispatch = None
    pass_tier = [ar for ar in amended_wfo.amended_results if ar.amended_gate.verdict.value in ("pass_deployable", "pass_viable")]
    if pass_tier:
        print(f"[step5] Step 6 auto-dispatch — {len(pass_tier)} PASS-tier candidate(s)...")
        # Build _LightOrchestratorView shim. Step 6's io builder pulls from pool + step_4 + wfo_search.
        # For direct-driver path, we shim pool and step_4 as the data the io builder needs.
        # Since step_4 has no candidate cluster classifier admission active (AUC < 0.65 → A2/A6 skipped),
        # Step 6's lookahead category will use feature_matrix + feature_lineage we pass in.
        from core.arc.arc_orchestrator import _LightOrchestratorView
        # Pool shim — Step 6 reads pool_trades + pool_paths for some checks.
        # The full ArcPool dataclass is heavier; for Step 6 dispatch it suffices if the shim
        # exposes .trades and .paths attributes.

        class _PoolShim:
            def __init__(self, trades, paths):
                self.trades = trades
                self.paths = paths

        class _Step4Shim:
            per_cluster = ()
            summary_md = ""

        pool_shim = _PoolShim(pool, paths_canon)
        step_4_shim = _Step4Shim()
        light_view = _LightOrchestratorView(
            arc_name=ARC_NAME,
            pool=pool_shim,
            step_4=step_4_shim,
            wfo_search=s5,
            amended_wfo=amended_wfo,
        )
        holdout_start = pd.Timestamp(holdout_fold.oos_start, tz="UTC")
        try:
            step_6_dispatch = maybe_dispatch_step_6(
                arc_orchestrator_result=light_view,
                amended_wfo=amended_wfo,
                arc_root=RESULTS_ROOT,
                audit_config=None,
                holdout_start=holdout_start,
                panels=panels,
                feature_matrix=feat_df,
                feature_lineage=lineage_df,
                signal_module_name=type(signal_module).__module__,
                primary_tf="H4",
                pair_set=tuple(PAIRS_28),
            )
            if step_6_dispatch:
                print(f"[step5] Step 6 dispatched: {step_6_dispatch.dispatched}; result present: {step_6_dispatch.step_6_result is not None}")
                if step_6_dispatch.downgrade_top_1:
                    from core.step_6.dispatch import replace_top_1_with_step6_fail
                    print("[step5] Step 6 critical-failed; downgrading Top-1")
                    amended_wfo = replace_top_1_with_step6_fail(amended_wfo)
        except Exception as exc:
            print(f"[step5] WARN: Step 6 dispatch raised {type(exc).__name__}: {exc}")
            print("[step5] Continuing without Step 6; closure will note as deferred.")
    else:
        print("[step5] No PASS-tier candidates; Step 6 not dispatched (Amendment 4).")

    # ── Resolve verdict ───────────────────────────────────────────
    from core.arc.arc_orchestrator import _amended_verdict_rank
    if amended_wfo.amended_results:
        ranked = sorted(amended_wfo.amended_results, key=lambda r: _amended_verdict_rank(r.amended_gate.verdict), reverse=True)
        verdict = ranked[0].amended_gate.verdict.value.upper()
    elif s5.top_k:
        verdict = s5.top_k[0].gate.verdict.value.upper()
    else:
        verdict = "INCOMPLETE"
    print(f"[step5] Verdict: {verdict}")

    # ── Persist Step 5 artefacts ──────────────────────────────────
    _emit_step_5_artefacts(
        s5=s5, amended_wfo=amended_wfo, step_6_dispatch=step_6_dispatch,
        verdict=verdict, candidate_cid=candidate_cid,
    )

    elapsed = time.perf_counter() - t0
    print(f"[step5] DONE in {elapsed:.1f}s — verdict {verdict}")
    return STEP_DIR


def _emit_step_5_artefacts(s5, amended_wfo, step_6_dispatch, verdict, candidate_cid) -> None:
    """Write step_5/ artefacts based on the in-memory result objects."""
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    artefacts: list[Path] = []

    # Safety: pickle raw result objects FIRST so a downstream emission bug
    # doesn't lose the 90+ minutes of search compute.
    import pickle
    pickle_path = STEP_DIR / "_raw_results.pkl"
    try:
        with open(pickle_path, "wb") as fh:
            pickle.dump({
                "verdict": verdict,
                "candidate_cid": candidate_cid,
                "s5": s5,
                "amended_wfo": amended_wfo,
                "step_6_dispatched": bool(step_6_dispatch and step_6_dispatch.dispatched),
            }, fh)
        print(f"[step5] Raw results pickled → {pickle_path}")
    except Exception as exc:
        print(f"[step5] WARN: failed to pickle raw results: {type(exc).__name__}: {exc}")

    # wfo_results.csv — per-config aggregate (top-K and below).
    rows = []
    # WfoSearchResult.candidates = full list (incl. top_k); top_k is the ranked top-K subset.
    top_k_ids = {c.config_id for c in s5.top_k}
    all_results = list(s5.candidates)
    for cand in all_results:
        fold_stats = cand.fold_stats
        worst_ratio = min(
            (f.roi_pct / f.max_dd_pct if f.max_dd_pct > 0 else float("inf"))
            for f in fold_stats
        ) if fold_stats else None
        worst_dd = max((f.max_dd_pct for f in fold_stats), default=0.0)
        mean_roi = (sum(f.roi_pct for f in fold_stats) / len(fold_stats)) if fold_stats else 0.0
        rows.append({
            "config_id": cand.config_id,
            "n_folds": len(fold_stats),
            "n_positive_folds": sum(1 for f in fold_stats if f.roi_pct > 0),
            "worst_fold_ratio": worst_ratio if worst_ratio != float("inf") else None,
            "worst_fold_dd_pct": worst_dd,
            "mean_fold_roi_pct": mean_roi,
            "verdict": cand.gate.verdict.value if cand.gate else "unknown",
            "in_top_k": cand.config_id in top_k_ids,
        })
    wfo_df = pd.DataFrame(rows).sort_values("worst_fold_ratio", ascending=False, na_position="last")
    wfo_path = STEP_DIR / "wfo_results.csv"
    wfo_df.to_csv(wfo_path, index=False, lineterminator="\n")
    artefacts.append(wfo_path)

    # per_fold_metrics.csv — all candidates × folds long-format.
    pf_rows = []
    for cand in all_results:
        for f in cand.fold_stats:
            pf_rows.append({
                "config_id": cand.config_id,
                "fold_id": f.fold_id,
                "n_trades": f.n_trades,
                "roi_pct": f.roi_pct,
                "max_dd_pct": f.max_dd_pct,
                "ratio": (f.roi_pct / f.max_dd_pct) if f.max_dd_pct > 0 else None,
            })
    per_fold_df = pd.DataFrame(pf_rows)
    pf_path = STEP_DIR / "per_fold_metrics.csv"
    per_fold_df.to_csv(pf_path, index=False, lineterminator="\n")
    artefacts.append(pf_path)

    # architectures_ranked.md.
    arch_lines = [
        f"# {ARC_NAME} — Step 5 architectures ranked",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        f"Candidate cluster: c{candidate_cid} (Bimodal). Arch set per Amendment 5: {{A1, A4}}.",
        "Exit slate (Bimodal canonical): {sl_only, sl_partial_close_1r_runner_trail, sl_plus_tp_2r}.",
        "",
        "## Top-K candidates by worst-fold ratio",
        "",
        "| Rank | Config | Worst-fold ratio | Worst-fold DD% | Mean fold ROI% | Base verdict | Amended verdict |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    amended_by_cid = {ar.config_id: ar for ar in (amended_wfo.amended_results if amended_wfo else ())}
    for k, cand in enumerate(s5.top_k, start=1):
        fs = cand.fold_stats
        wr = min((f.roi_pct / f.max_dd_pct if f.max_dd_pct > 0 else float("inf")) for f in fs) if fs else None
        wdd = max((f.max_dd_pct for f in fs), default=0.0)
        mr = (sum(f.roi_pct for f in fs) / len(fs)) if fs else 0.0
        bv = cand.gate.verdict.value if cand.gate else "unknown"
        ar = amended_by_cid.get(cand.config_id)
        av = ar.amended_gate.verdict.value if ar else "—"
        arch_lines.append(
            f"| {k} | `{cand.config_id}` | "
            f"{wr if wr is not None and wr != float('inf') else '—'} | "
            f"{wdd:.2%} | {mr:.2%} | {bv} | {av} |"
        )

    # Amendment 3 per top-K.
    arch_lines += ["", "## Amendment 3 evaluation (top-K)", ""]
    if amended_wfo and amended_wfo.amended_results:
        arch_lines += [
            "| Config | Amended verdict | Chained DD% | k_safe | k_hard | r_safe% (deploy) | r_hard% (deploy) | r_safe intrinsic% | r_hard intrinsic% | safe capped | hard capped | scalable_safe | scalable_hard | primary_failure_mode |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---|",
        ]
        for ar in amended_wfo.amended_results:
            ag = ar.amended_gate
            sf = getattr(ag, "scaling_factors", None)
            def _fmt(v, pct=False):
                if v is None:
                    return "—"
                return f"{v * 100:.3f}" if pct else f"{v:.3f}"
            k_safe = _fmt(getattr(sf, "k_safe", None)) if sf else "—"
            k_hard = _fmt(getattr(sf, "k_hard", None)) if sf else "—"
            r_safe = _fmt(getattr(sf, "r_safe_pct", None), pct=True) if sf else "—"
            r_hard = _fmt(getattr(sf, "r_hard_pct", None), pct=True) if sf else "—"
            r_safe_intr = _fmt(getattr(sf, "r_safe_intrinsic_pct", None), pct=True) if sf else "—"
            r_hard_intr = _fmt(getattr(sf, "r_hard_intrinsic_pct", None), pct=True) if sf else "—"
            safe_capped = "Y" if sf and getattr(sf, "r_safe_capped_at_rmax", False) else "N"
            hard_capped = "Y" if sf and getattr(sf, "r_hard_capped_at_rmax", False) else "N"
            scalable_safe = "Y" if sf and getattr(sf, "scalable_to_safe", False) else "N"
            scalable_hard = "Y" if sf and getattr(sf, "scalable_to_hard", False) else "N"
            pfm = getattr(ag, "primary_failure_mode", None) or "—"
            arch_lines.append(
                f"| `{ar.config_id}` | {ag.verdict.value} | {ar.chained_max_dd_base_pct:.4%} | "
                f"{k_safe} | {k_hard} | {r_safe} | {r_hard} | {r_safe_intr} | {r_hard_intr} | "
                f"{safe_capped} | {hard_capped} | {scalable_safe} | {scalable_hard} | {pfm} |"
            )
    else:
        arch_lines.append("(no top-K candidates; Amendment 3 skipped)")

    # Step 6.
    arch_lines += ["", "## Step 6 (auto-dispatch per Amendment 4)", ""]
    if step_6_dispatch and step_6_dispatch.dispatched:
        res = step_6_dispatch.step_6_result
        if res:
            arch_lines.append(f"- Dispatched. Overall passed: **{bool(res.overall_passed)}**")
            arch_lines.append(f"- Trigger: `{res.trigger.value}`; Verdict impact: `{res.verdict_impact.value}`")
        else:
            arch_lines.append("- Dispatched; no result captured.")
    else:
        arch_lines.append("- Not dispatched (no PASS-tier candidate cleared §3 constraints #1-9).")

    arch_md_path = STEP_DIR / "architectures_ranked.md"
    write_text_deterministic(arch_md_path, "\n".join(arch_lines))
    artefacts.append(arch_md_path)

    # best_candidate.md.
    best_lines = [
        f"# {ARC_NAME} — Best candidate (Step 5)",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
    ]
    if s5.top_k:
        best = s5.top_k[0]
        best_amended = amended_by_cid.get(best.config_id)
        best_lines += [
            f"## Top-1 config: `{best.config_id}`",
            "",
            f"Base gate verdict: `{best.gate.verdict.value if best.gate else 'unknown'}`",
            f"Amended gate verdict: `{best_amended.amended_gate.verdict.value if best_amended else 'unknown'}`",
            "",
            "### Per-fold stats (search)",
            "",
            "| Fold | n | ROI% | DD% | Ratio |",
            "|---:|---:|---:|---:|---:|",
        ]
        for f in best.fold_stats:
            ratio = f.roi_pct / f.max_dd_pct if f.max_dd_pct > 0 else float("inf")
            best_lines.append(
                f"| {f.fold_id} | {f.n_trades} | {f.roi_pct:.2%} | {f.max_dd_pct:.2%} | "
                f"{('inf' if ratio == float('inf') else f'{ratio:.3f}')} |"
            )
        if best_amended:
            ag = best_amended.amended_gate
            sf = getattr(ag, "scaling_factors", None)
            best_lines += [
                "",
                "### Amendment 3 evaluation (Top-1)",
                "",
                f"- chained DD (base): {best_amended.chained_max_dd_base_pct:.4%} (method: {best_amended.chained_dd_method})",
                f"- primary_failure_mode: `{getattr(ag, 'primary_failure_mode', None) or '—'}`",
            ]
            if sf is not None:
                best_lines += [
                    f"- r_safe intrinsic: {getattr(sf, 'r_safe_intrinsic_pct', 0) * 100:.3f}% (capped at r_max: {getattr(sf, 'r_safe_capped_at_rmax', False)})",
                    f"- r_hard intrinsic: {getattr(sf, 'r_hard_intrinsic_pct', 0) * 100:.3f}% (capped at r_max: {getattr(sf, 'r_hard_capped_at_rmax', False)})",
                    f"- r_safe deploy: {getattr(sf, 'r_safe_pct', 0) * 100:.3f}% (k_safe: {getattr(sf, 'k_safe', 0):.3f})",
                    f"- r_hard deploy: {getattr(sf, 'r_hard_pct', 0) * 100:.3f}% (k_hard: {getattr(sf, 'k_hard', 0):.3f})",
                    f"- scalable_to_safe: {getattr(sf, 'scalable_to_safe', False)}",
                    f"- scalable_to_hard: {getattr(sf, 'scalable_to_hard', False)}",
                ]
    else:
        best_lines.append("(no Step 5 candidates surfaced)")
    best_md_path = STEP_DIR / "best_candidate.md"
    write_text_deterministic(best_md_path, "\n".join(best_lines))
    artefacts.append(best_md_path)

    # per_day_max_dd parquets emitted in step_5/ by Amendment 3 evaluation.
    pdmd_files = sorted(STEP_DIR.glob("per_day_max_dd_base__*.parquet"))

    # Manifest.
    write_manifest(STEP_DIR / "manifest.json", artefacts=artefacts + pdmd_files)


if __name__ == "__main__":
    main()
