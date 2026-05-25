"""Arc 11 v3.0.2 — canonical ArcOrchestrator driver under EET semantics.

Drives Arc 11 v3.0.2 end-to-end via the canonical
:class:`core.arc.arc_orchestrator.ArcOrchestrator` path (which threads
``run_context`` through ``_run_step_5`` post-PR-#186 — closes the
``canonical_orchestrator_step5_run_context_gap`` flag from Arc 11 v3.0).

Engine deltas vs Arc 11 v3.0:

- Panels built with ``boundary_convention="5ers_eet"`` (PR #197 /
  Amendment 6); cache namespace ``data/cache/<TF>_5ers_eet/<PAIR>.parquet``
- Feature matrix on mid-anchored OHLC (PR #189 §15.1) via the canonical
  ``core.features.pipeline``
- Multi-TF features use canonical ``core.signals.htf_alignment``
  (PR #193) — byte-identical to legacy under UTC; correct under EET
- A2/A6 use ``classifier_persistence`` (PR #185) — no Step 5 retrain
- A4 uses per-fold path classifier via ``path_classifier_per_fold``
- Step 5 admission set per L_PROTOCOL Amendment 5 four-gate rule:
  A1 (Gate 3) + A2 c0 (Gate 2 if AUC>=0.65) + A4 c0 (Gate 1 Bimodal) +
  A6 c0 (Gate 2 if AUC>=0.65)
- A5 deferred to closure addendum per Amendment 5.1 (PR #201) —
  Gate 4(b) cannot be satisfied at dispatch time
- Amendment 3 risk-normalised gates emitted automatically per top-K
- Step 6 auto-dispatched if Top-1 PASS-tier (unlikely per dispatch §6)

Holdout window: 2021-01-01 -> 2026-05-25 (4-week extension vs v3.0).

Operational design: single Python invocation; logs to
``results/l_arc_11_v3.0.2/run.log`` for progress monitoring; writes all
v3.0.2 artefacts to ``results/l_arc_11_v3.0.2/``. The orchestrator's
default closure skeleton is augmented post-run via
``write_closure.py`` (separate script) to populate §1 tracker_payload
+ §2 prose + §3 cross-arc + §4 deployment_spec + §10 retroactive
re-eval per chat resolutions in the v3.0.2 intent doc.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

# Force UTF-8 stdout/stderr on Windows so log lines with non-ASCII chars
# from downstream libraries don't crash with UnicodeEncodeError under cp1252.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.architectures.a1_system_level_filter import (  # noqa: E402
    A1Architecture, A1Config,
)
from core.architectures.a2_classifier_filter import A2Architecture  # noqa: E402
from core.architectures.a4_pipeline_d_exits import A4Architecture  # noqa: E402
from core.architectures.a6_meta_labeling import A6Architecture  # noqa: E402
from core.arc.arc_orchestrator import (  # noqa: E402
    ArcConfig, ArcOrchestrator, AutoArchSpec,
)
from core.determinism import RANDOM_STATE, seed_everything  # noqa: E402
from core.features.pipeline import compute_feature_matrix  # noqa: E402
from core.sim.panel import Panel  # noqa: E402
from core.strategies.shb.signal_module import SHBSignalModule  # noqa: E402
from core.wfo.folds import build_v3_folds  # noqa: E402

ARC_NAME = "l_arc_11_v3.0.2"
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
WINDOW_START = dt.date(2010, 1, 1)
WINDOW_END = dt.date(2026, 5, 25)        # Holdout end per chat resolution §1.A5.4
HOLDOUT_START = dt.date(2021, 1, 1)
HOLDOUT_END = dt.date(2026, 5, 25)
SL_ATR_MULT_DEFAULT = 2.0                 # Pool builder anchor; Step 5 sweeps separately
HOLD_BARS = 240
RISK_PCT = 0.005
STARTING_BALANCE = 100_000.0
MIN_IS_DAYS = 365
TOP_K_HOLDOUT = 3

# Step 5 grid (per L_PROTOCOL §2 Step 5 + chat resolution §1.A5.3)
SL_GRID = (1.5, 2.0, 2.5)                 # centred on v3.0 Step 3 optimum 1.5 ±1 step
EXITS_DEFAULT = ("sl_only", "sl_plus_tp_2r", "sl_partial_close_1r_runner_trail")
EXITS_BIMODAL = ("sl_partial_close_1r_runner_trail", "sl_plus_tp_2r")  # A4 c0 Bimodal
EXPOSURE_GRID = (2, None)                 # max_concurrent_per_currency: 2 OR unlimited
A6_THRESHOLDS_DEFAULT = (0.4, 0.6)        # (lower, upper) — single pair per chat A5.3
A4_EXIT_THRESHOLDS = (0.3, 0.4, 0.5)      # L_PROTOCOL Amendment 2 A4 spec

OUT_DIR = _REPO_ROOT / "results" / ARC_NAME
LOG_FILE: Path | None = None


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    line = f"[arc_11_v3.0.2 {ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


# ─── Panel construction under EET ──────────────────────────────────


class AuxPanel:
    """Wraps the primary H4 panel with auxiliary D1 / W1 panels so the
    canonical feature pipeline can compute multi-TF features. Mirrors
    the pattern from scripts/l_arc_11/run.py."""

    def __init__(self, h4_panel: Panel, aux: dict | None = None) -> None:
        self._h4 = h4_panel
        self.pair_dfs = h4_panel.pair_dfs
        self.tf = h4_panel.tf
        self.aux = aux or {}
        self.boundary_convention = h4_panel.boundary_convention

    @property
    def pairs(self):
        return self._h4.pairs

    def snapshot_at(self, t):
        return self._h4.snapshot_at(t)


def build_panel_eet(tf: str) -> Panel:
    t0 = time.time()
    panel = Panel.from_pairs(
        list(PAIRS), tf,
        histdata_root=HISTDATA_ROOT,
        cache_root=CACHE_ROOT,
        boundary_convention="5ers_eet",
    )
    _log(f"{tf} panel built in {time.time() - t0:.1f}s "
         f"({len(panel.pairs)} pairs; convention={panel.boundary_convention})")
    return panel


def build_feature_matrix(
    pool_trades: pd.DataFrame,
    h4_panel: Panel,
    d1_panel: Panel,
    w1_panel: Panel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-trade entry-time feature matrix indexed by trade_id."""
    panel = AuxPanel(h4_panel, aux={"d1": d1_panel, "w1": w1_panel})
    rows: list[pd.DataFrame] = []
    lineage_df = None
    _log("Computing v3 feature matrix per pair at signal bars")
    for pair in sorted(h4_panel.pairs):
        t0 = time.time()
        pair_df = h4_panel.pair_dfs.get(pair)
        if pair_df is None or pair_df.empty:
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


# ─── Step 5 grid construction ──────────────────────────────────────


def build_a1_grid() -> tuple[
    tuple[A1Architecture, ...], tuple[A1Config, ...]
]:
    """Build explicit A1 grid: 3 SLs x 3 exits x 2 exposure = 18 configs."""
    archs: list[A1Architecture] = []
    cfgs: list[A1Config] = []
    for sl in SL_GRID:
        for exit_policy in EXITS_DEFAULT:
            for exposure in EXPOSURE_GRID:
                exposure_tag = "expU" if exposure is None else f"exp{exposure}"
                cfg = A1Config(
                    config_id=f"a1_baseline_sl{sl}_{exit_policy}_{exposure_tag}",
                    sl_atr_mult=sl,
                    trail_enabled=False,
                    risk_pct=RISK_PCT,
                    starting_balance=STARTING_BALANCE,
                    max_concurrent_per_pair=1,
                    max_concurrent_per_currency=exposure,
                    time_exit_bars=HOLD_BARS,
                    exit_policy=exit_policy,
                )
                archs.append(A1Architecture())
                cfgs.append(cfg)
    return tuple(archs), tuple(cfgs)


def build_auto_arch_specs_for_cluster(
    cluster_id: int,
    *,
    admit_a2: bool,
    admit_a4: bool,
    admit_a6: bool,
) -> tuple[AutoArchSpec, ...]:
    """Build AutoArchSpecs for A2/A4/A6 grids per cluster admission flags.

    Per L_PROTOCOL Amendment 2 retraining policy:
      - A2 / A6: use Step 4 persisted classifier, no retrain
      - A4: per-fold path classifier (orchestrator threads via
        ``A1RunContext.path_classifier_fits``)

    Builder kwargs sweep SL x exit x exposure (x threshold pair for A6,
    x exit_threshold for A4).
    """
    specs: list[AutoArchSpec] = []

    # A2: 3 SLs x 3 exits x 2 exposure = 18 configs at default Step-4-best threshold
    if admit_a2:
        for sl in SL_GRID:
            for exit_policy in EXITS_DEFAULT:
                for exposure in EXPOSURE_GRID:
                    exposure_tag = "expU" if exposure is None else f"exp{exposure}"
                    specs.append(AutoArchSpec(
                        architecture=A2Architecture(),
                        cluster_id=cluster_id,
                        builder_kwargs={
                            "config_id": (
                                f"a2_c{cluster_id}_sl{sl}_{exit_policy}_{exposure_tag}"
                            ),
                            "sl_atr_mult": sl,
                            "trail_enabled": False,
                            "exit_policy": exit_policy,
                            "risk_pct": RISK_PCT,
                            "starting_balance": STARTING_BALANCE,
                            "max_concurrent_per_pair": 1,
                            "max_concurrent_per_currency": exposure,
                        },
                    ))

    # A4: 3 SLs x 2 Bimodal exits x 2 exposure x 3 exit_thresholds = 36
    if admit_a4:
        for sl in SL_GRID:
            for exit_policy in EXITS_BIMODAL:
                for exposure in EXPOSURE_GRID:
                    for et in A4_EXIT_THRESHOLDS:
                        exposure_tag = "expU" if exposure is None else f"exp{exposure}"
                        specs.append(AutoArchSpec(
                            architecture=A4Architecture(),
                            cluster_id=cluster_id,
                            builder_kwargs={
                                "config_id": (
                                    f"a4_c{cluster_id}_sl{sl}_{exit_policy}_"
                                    f"{exposure_tag}_et{et}"
                                ),
                                "exit_threshold": et,
                                "sl_atr_mult": sl,
                                "trail_enabled": False,
                                "exit_policy": exit_policy,
                                "risk_pct": RISK_PCT,
                                "starting_balance": STARTING_BALANCE,
                                "max_concurrent_per_pair": 1,
                                "max_concurrent_per_currency": exposure,
                            },
                        ))

    # A6: 3 SLs x 3 exits x 2 exposure x 1 threshold pair (0.4/0.6) = 18
    # Single threshold pair (default) per chat resolution §1.A5.3 — broader
    # threshold-pair grids inflate config count without adding diagnostic value
    # at the verdict-prior-FAIL frontier.
    if admit_a6:
        lo, hi = A6_THRESHOLDS_DEFAULT
        for sl in SL_GRID:
            for exit_policy in EXITS_DEFAULT:
                for exposure in EXPOSURE_GRID:
                    exposure_tag = "expU" if exposure is None else f"exp{exposure}"
                    specs.append(AutoArchSpec(
                        architecture=A6Architecture(),
                        cluster_id=cluster_id,
                        builder_kwargs={
                            "config_id": (
                                f"a6_c{cluster_id}_sl{sl}_{exit_policy}_"
                                f"{exposure_tag}_lo{lo}_hi{hi}"
                            ),
                            "lower_threshold": lo,
                            "upper_threshold": hi,
                            "sl_atr_mult": sl,
                            "trail_enabled": False,
                            "exit_policy": exit_policy,
                            "risk_pct": RISK_PCT,
                            "starting_balance": STARTING_BALANCE,
                            "max_concurrent_per_pair": 1,
                            "max_concurrent_per_currency": exposure,
                        },
                    ))

    return tuple(specs)


# ─── Probe Steps 2-4 to decide cluster-level admission ────────────


def _probe_for_admission(
    pool, feature_matrix: pd.DataFrame, feature_lineage: pd.DataFrame,
    train_end: pd.Timestamp,
) -> dict:
    """Pre-probe Step 4 to get per-cluster AUC for Amendment 5 admission.

    Runs Steps 2/3/4 in lightweight form (no persistence) to determine
    which clusters survive Step 3 and which clusters' Step 4 AUC >= 0.65.
    Returns a dict with admission flags + Step 3/4 metrics for closure
    diagnostic prose.

    This is a pre-orchestrator probe so we can pass the correct
    auto_arch_specs to ArcConfig (Amendment 5 enforcement is
    dispatch-time per PROTOCOL_RUNTIME §8a). The full orchestrator
    run re-executes Steps 1-4 deterministically — the probe's compute
    is small (no per-fold WFO) and amortised by avoiding wrong
    architecture admissions.
    """
    from core.steps.step_2_clustering import run_step_2
    from core.steps.step_3_capturability import run_step_3
    from core.steps.step_4_extraction import run_step_4
    s2 = run_step_2(pool.trades, pool.paths)
    _log(f"  probe Step 2: K_selected={s2.k_selected}; "
         f"silhouettes={ {int(k): round(float(v),3) for k,v in s2.silhouette_per_k.items()} }")
    s3 = run_step_3(
        pool.trades, pool.paths, s2.cluster_assignments,
        declared_sl_mult=SL_ATR_MULT_DEFAULT, cluster_centroids=s2.centroids,
    )
    candidate_ids = tuple(int(c.cluster_id) for c in s3.per_cluster if c.is_candidate)
    cluster_archetypes = {
        int(c.cluster_id): str(c.shape_tag) for c in s3.per_cluster
    }
    _log(f"  probe Step 3: candidate_ids={candidate_ids}")
    for c in s3.per_cluster:
        flag = "* candidate" if c.is_candidate else "  "
        _log(f"    c{int(c.cluster_id)} ({c.shape_tag}): "
             f"n={c.n_trades} composite={c.capturability_composite:.3f} "
             f"reach_1r={c.reach_1r:.3f} mfe_p50={c.mfe_p50:.3f} "
             f"ww_pp={c.wrong_way_pp:.3f} sl_opt={c.selected_sl:.1f} {flag}")
    admission: dict = {
        "candidate_ids": candidate_ids,
        "cluster_archetypes": cluster_archetypes,
        "n_candidates": len(candidate_ids),
        "step_3_per_cluster": [
            {
                "cluster_id": int(c.cluster_id),
                "archetype": str(c.shape_tag),
                "n_trades": int(c.n_trades),
                "composite": float(c.capturability_composite),
                "reach_1r": float(c.reach_1r),
                "mfe_p50": float(c.mfe_p50),
                "ww_pp": float(c.wrong_way_pp),
                "selected_sl_mult": float(c.selected_sl),
                "is_candidate": bool(c.is_candidate),
            }
            for c in s3.per_cluster
        ],
    }
    if not candidate_ids:
        admission["step_4_per_cluster"] = []
        admission["admit_per_cluster"] = {}
        return admission
    # Step 4 probe — NO persistence (orchestrator re-runs with persistence)
    s4 = run_step_4(
        pool.trades,
        feature_matrix.reset_index(),
        s2.cluster_assignments,
        feature_lineage=feature_lineage,
        candidate_cluster_ids=candidate_ids,
        train_end=train_end,
    )
    step4_metrics: list = []
    admit_per_cluster: dict[int, dict[str, bool]] = {}
    for ce in s4.per_cluster:
        cid = int(ce.cluster_id)
        auc = float(ce.best_classifier_mean_auc)
        archetype = cluster_archetypes.get(cid, "Unclassified")
        admit_a2_a6 = auc >= 0.65                     # Gate 2
        # Case-insensitive archetype check — step_2_clustering emits lowercase
        # tags ("bimodal", "stepwise_climber") while L_PROTOCOL Gate 1 spec uses
        # title-case labels ("Bimodal", "Stepwise"). Normalize both sides.
        arch_norm = archetype.lower().replace(" ", "_") if archetype else ""
        admit_a4 = arch_norm in ("bimodal", "stepwise_climber", "stepwise climber")  # Gate 1 subset
        admit_per_cluster[cid] = {
            "A2": admit_a2_a6,
            "A4": admit_a4,
            "A6": admit_a2_a6,
        }
        step4_metrics.append({
            "cluster_id": cid,
            "archetype": archetype,
            "best_classifier": ce.best_classifier,
            "best_classifier_mean_auc": auc,
            "best_threshold": float(ce.best_threshold),
            "admit_a2_a6_gate2": admit_a2_a6,
            "admit_a4_gate1": admit_a4,
        })
        _log(f"    Step 4 c{cid} ({archetype}): "
             f"best={ce.best_classifier} AUC={auc:.4f} "
             f"thr={ce.best_threshold:.4f} | "
             f"Gate2 admits A2/A6={admit_a2_a6} | Gate1 admits A4={admit_a4}")
    admission["step_4_per_cluster"] = step4_metrics
    admission["admit_per_cluster"] = admit_per_cluster
    return admission


# ─── Main ──────────────────────────────────────────────────────────


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global LOG_FILE
    LOG_FILE = OUT_DIR / "run.log"
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    LOG_FILE.touch()

    seed_everything(RANDOM_STATE)
    t_start = time.time()
    _log(f"=== Arc 11 v3.0.2 START ===")
    _log(f"Window: {WINDOW_START} -> {WINDOW_END} (holdout {HOLDOUT_START} -> {HOLDOUT_END})")
    _log(f"r_base={RISK_PCT:.4f}, starting_balance={STARTING_BALANCE:.0f}")

    # 1. Build panels (EET cache warm)
    h4_panel = build_panel_eet("H4")
    d1_panel = build_panel_eet("D1")
    w1_panel = build_panel_eet("W1")

    # 2. Build pool (canonical, uncapped) for admission probe
    from core.arc.arc_pool_builder import (
        ArcPoolConfig, build_arc_pool, write_arc_pool,
    )
    signal_module = SHBSignalModule()
    panels = {"H4": h4_panel, "D1": d1_panel, "W1": w1_panel}
    _log("Building Step 1 trade pool (canonical build_arc_pool, EET-aware)")
    pool_cfg = ArcPoolConfig(
        arc_name=ARC_NAME,
        sl_atr_mult=SL_ATR_MULT_DEFAULT,
        hold_bars=HOLD_BARS,
        risk_pct=RISK_PCT,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )
    pool = build_arc_pool(signal_module, panels, pool_cfg)
    _log(f"  pool n={len(pool.trades)}, pool_sha256={pool.pool_sha256[:16]}...")
    write_arc_pool(pool, OUT_DIR)

    # 3. Build feature matrix
    feature_matrix, feature_lineage = build_feature_matrix(
        pool.trades, h4_panel, d1_panel, w1_panel,
    )
    _log(f"  feature_matrix shape: {feature_matrix.shape}; "
         f"lineage rows: {len(feature_lineage) if feature_lineage is not None else 0}")

    # 4. Probe Steps 2-4 to determine Amendment 5 admission
    _log("=== Pre-orchestrator probe: Steps 2-4 for Amendment 5 admission ===")
    train_end = pd.Timestamp(HOLDOUT_START, tz="UTC")
    admission = _probe_for_admission(pool, feature_matrix, feature_lineage, train_end)

    # Save probe summary for closure synthesis
    probe_json = OUT_DIR / "amendment_5_admission_probe.json"
    probe_json.write_text(
        json.dumps(admission, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    _log(f"  probe summary written to {probe_json}")

    # 5. Compute Amendment 5 (post-5.1) architecture admission set
    n_candidates = admission["n_candidates"]
    a5_gate_4_a_satisfied = n_candidates >= 2
    a5_skipped_by_5_1 = a5_gate_4_a_satisfied  # Gate 4(b) cannot hold at dispatch

    # A1: always admitted (Gate 3 universal). Build explicit grid.
    a1_archs, a1_cfgs = build_a1_grid()
    _log(f"A1 grid: {len(a1_cfgs)} explicit configs")

    # Auto-arch specs per candidate cluster
    auto_specs: list[AutoArchSpec] = []
    architectures_skipped: list[str] = []
    a2_a6_skipped_clusters: list[int] = []
    for cid in admission["candidate_ids"]:
        flags = admission["admit_per_cluster"][cid]
        cluster_specs = build_auto_arch_specs_for_cluster(
            cid,
            admit_a2=flags["A2"],
            admit_a4=flags["A4"],
            admit_a6=flags["A6"],
        )
        auto_specs.extend(cluster_specs)
        if not flags["A2"]:
            a2_a6_skipped_clusters.append(cid)
    if a5_skipped_by_5_1:
        architectures_skipped.append("a5_gate_4_admission_blocked_by_no_pass_tier_constituent")
    if a2_a6_skipped_clusters:
        # Per Amendment 5 Gate 2: when AUC<0.65 the cluster skips A2+A6.
        # The closure field is flat — record A2 and A6 once if ANY cluster
        # was excluded (cross-arc analytics; per-cluster context in §3 prose).
        if "A2" not in architectures_skipped:
            architectures_skipped.append("A2")
        if "A6" not in architectures_skipped:
            architectures_skipped.append("A6")
    _log(f"Auto-arch specs: {len(auto_specs)} (across "
         f"{n_candidates} candidate clusters)")
    _log(f"architectures_skipped_by_amendment_5: {architectures_skipped}")

    # Persist the planned admission set for closure synthesis
    admission_plan = {
        "amendment_5_1_cutoff_applied": True,
        "n_candidate_clusters": n_candidates,
        "a5_gate_4_a_satisfied": a5_gate_4_a_satisfied,
        "a5_skipped_by_5_1": a5_skipped_by_5_1,
        "a2_a6_skipped_clusters": a2_a6_skipped_clusters,
        "architectures_skipped_by_amendment_5": architectures_skipped,
        "n_a1_configs": len(a1_cfgs),
        "n_auto_specs": len(auto_specs),
        "total_configs_step_5": len(a1_cfgs) + len(auto_specs),
        "search_scope_flag": _scope_flag(len(a1_cfgs) + len(auto_specs)),
    }
    (OUT_DIR / "admission_plan.json").write_text(
        json.dumps(admission_plan, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    _log(f"  admission plan: {admission_plan['total_configs_step_5']} configs "
         f"({admission_plan['search_scope_flag']})")

    # 6. Build ArcConfig and run orchestrator end-to-end
    wfo_struct = build_v3_folds(
        holdout_start=HOLDOUT_START,
        holdout_end=HOLDOUT_END,
    )
    _log(f"WFO structure: {len(wfo_struct.folds)} IS folds; "
         f"holdout {wfo_struct.holdout.oos_start} -> {wfo_struct.holdout.oos_end}")
    arc_cfg = ArcConfig(
        arc_name=ARC_NAME,
        signal_class="swing_high_breakout_trend_long",
        pair_set=PAIRS,
        window_start=pd.Timestamp(WINDOW_START),
        window_end=pd.Timestamp(WINDOW_END),
        risk_pct=RISK_PCT,
        sub_protocol="vanilla",
        sl_atr_mult=SL_ATR_MULT_DEFAULT,
        hold_bars=HOLD_BARS,
        output_dir=OUT_DIR,
        feature_matrix=feature_matrix.reset_index(),
        feature_lineage=feature_lineage,
        architectures=a1_archs,
        architecture_configs=a1_cfgs,
        auto_arch_specs=tuple(auto_specs),
        wfo_structure=wfo_struct,
        skip_step_6=False,                # Amendment 4 auto-dispatch if PASS-tier
        accept_equity_pct=False,
        hypothesis=(
            "v3.0.2 re-run under canonical engine "
            "(EET aggregation + mid features + EET HTF alignment + "
            "Amendment 5/5.1 admission + canonical exits + Amendment 6 "
            "EET daily-DD boundary). Verdict prior FAIL; diagnostic value in "
            "whether orchestrator gap closure + canonical pool change Step 5 outcomes."
        ),
        expected_failure_modes=(
            "step5_not_scalable (Arc 11 v3.0 §10 re-class); "
            "negative worst-fold ratio; "
            "scalability floor breach (r_safe < 0.15%); "
            "non-zero daily DD breaches at r_base; "
            "trade count per fold below 25 floor"
        ),
    )

    _log("=== Invoking ArcOrchestrator.run() ===")
    t_orch = time.time()
    orch = ArcOrchestrator(arc_cfg, signal_module, panels)
    result = orch.run()
    _log(f"=== ArcOrchestrator.run() complete in {(time.time()-t_orch)/60:.1f} min ===")
    _log(f"Verdict (orchestrator): {result.verdict}")

    # 7. Write artefacts (orchestrator's write writes step_1/2/3/4 + ARC_OPEN + ARC_CLOSURE)
    orch.write(result, OUT_DIR)
    _log(f"Artefacts written to {OUT_DIR}")

    # 8. Persist run summary for closure synthesis
    summary = {
        "arc_name": ARC_NAME,
        "verdict": result.verdict,
        "pool_size": int(len(pool.trades)),
        "pool_sha256": pool.pool_sha256,
        "feature_matrix_shape": list(feature_matrix.shape),
        "step_2_k_selected": int(result.step_2.k_selected) if result.step_2 else None,
        "step_2_silhouettes": {
            int(k): float(v) for k, v in result.step_2.silhouette_per_k.items()
        } if result.step_2 else None,
        "candidate_cluster_ids": list(admission["candidate_ids"]),
        "step_3_per_cluster": admission["step_3_per_cluster"],
        "step_4_per_cluster": admission["step_4_per_cluster"],
        "admission_plan": admission_plan,
        "architectures_skipped_by_amendment_5": architectures_skipped,
        "n_configs_evaluated_step_5": (
            len(result.wfo_search.candidates) if result.wfo_search else 0
        ),
        "search_scope_flag": admission_plan["search_scope_flag"],
        "step_5_top_k": [
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
            for c in (result.wfo_search.top_k if result.wfo_search else ())
        ],
        "amendment_3_top_k": [
            {
                "config_id": ar.config_id,
                "chained_max_dd_base_pct": float(ar.chained_max_dd_base_pct),
                "chained_dd_method": ar.chained_dd_method,
                "per_day_max_dd_artefact_path": (
                    str(ar.per_day_max_dd_artefact_path)
                    if ar.per_day_max_dd_artefact_path else None
                ),
                "verdict": ar.amended_gate.verdict.value,
                "primary_failure_mode": (
                    ar.amended_gate.primary_failure_mode.value
                    if hasattr(ar.amended_gate.primary_failure_mode, "value")
                    else str(ar.amended_gate.primary_failure_mode)
                ),
                "k_safe": float(ar.amended_gate.k_safe) if ar.amended_gate.k_safe is not None else None,
                "k_hard": float(ar.amended_gate.k_hard) if ar.amended_gate.k_hard is not None else None,
                "r_safe_pct": float(ar.amended_gate.r_safe_pct) if ar.amended_gate.r_safe_pct is not None else None,
                "r_hard_pct": float(ar.amended_gate.r_hard_pct) if ar.amended_gate.r_hard_pct is not None else None,
                "scalable_to_safe": bool(ar.amended_gate.scalable_to_safe),
                "scalable_to_hard": bool(ar.amended_gate.scalable_to_hard),
                "worst_fold_roi_at_r_safe_pct": _ofloat(ar.amended_gate.worst_fold_roi_at_r_safe_pct),
                # worst_fold_dd_at_r_safe_pct: bound to 8% at r_safe by construction (k_safe = 8 / worst_fold_dd_base)
                "worst_fold_dd_at_r_safe_pct": 8.0,
                "worst_fold_roi_at_r_hard_pct": _ofloat(ar.amended_gate.worst_fold_roi_at_r_hard_pct),
                # worst_fold_dd_at_r_hard_pct: bound to 10% at r_hard by construction
                "worst_fold_dd_at_r_hard_pct": 10.0,
                "chained_max_dd_at_r_safe_pct": _ofloat(ar.amended_gate.chained_max_dd_at_r_safe_pct),
                "chained_max_dd_at_r_hard_pct": _ofloat(ar.amended_gate.chained_max_dd_at_r_hard_pct),
                "daily_dd_breaches_at_r_safe": _oint(ar.amended_gate.daily_dd_breaches_at_r_safe),
                "daily_dd_breaches_at_r_hard": _oint(ar.amended_gate.daily_dd_breaches_at_r_hard),
                "holdout_roi_at_r_safe_pct": _ofloat(getattr(ar.amended_gate, "holdout_roi_at_r_safe_pct", None)),
                "holdout_dd_at_r_safe_pct": _ofloat(getattr(ar.amended_gate, "holdout_dd_at_r_safe_pct", None)),
                "holdout_roi_at_r_hard_pct": _ofloat(getattr(ar.amended_gate, "holdout_roi_at_r_hard_pct", None)),
                "holdout_dd_at_r_hard_pct": _ofloat(getattr(ar.amended_gate, "holdout_dd_at_r_hard_pct", None)),
            }
            for ar in (result.amended_wfo.amended_results if result.amended_wfo else ())
        ],
        "holdout_results": [
            {
                "config_id": h.config_id,
                "search_verdict": h.search_gate.verdict.value,
                "holdout_verdict": h.holdout_gate.verdict.value,
                "holdout_roi_pct": float(h.holdout_stats.roi_pct),
                "holdout_dd_pct": float(h.holdout_stats.max_dd_pct),
                "holdout_ratio": float(h.holdout_stats.roi_dd_ratio),
                "holdout_n_trades": int(h.holdout_stats.n_trades),
                "deployable": bool(h.deployable),
            }
            for h in (result.holdout_results or ())
        ],
        "step_6_dispatched": result.step_6_dispatch is not None and result.step_6_dispatch.dispatched,
        "step_6_overall_passed": (
            bool(result.step_6_dispatch.step_6_result.overall_passed)
            if result.step_6_dispatch is not None
            and result.step_6_dispatch.step_6_result is not None
            else None
        ),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    (OUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    _log(f"Verdict: {result.verdict}; "
         f"top-K candidates: {[t.config_id for t in (result.wfo_search.top_k if result.wfo_search else ())]}")
    _log(f"Total elapsed: {(time.time() - t_start)/60:.1f} min")
    _log("=== Arc 11 v3.0.2 END ===")
    return 0


def _scope_flag(n: int) -> str:
    if n < 50:
        return "thin"
    if n < 100:
        return "normal"
    return "broad"


def _ofloat(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _oint(x):
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
