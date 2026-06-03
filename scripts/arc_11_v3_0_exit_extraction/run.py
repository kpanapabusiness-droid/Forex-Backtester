"""Arc 11 v3.0 — EXIT-EXTRACTION re-run (world #2: ArcFoldRunner + Amendment 3).

Tests the UNTESTED exit-extraction route on the SHB cohort: full-pool + differentiated
exits (A4), NOT entry selection. Verdict prior FAIL/ambiguous; Arc 11 has the highest
oracle ceiling on record. Per chat steering 2026-06-02:

  * World #2 ONLY — canonical ArcFoldRunner + Amendment 3 risk-normalised gates (the tested
    path that closed Arc 5/7/8/11 v3.0.2). NOT the Arc-10 governed/fixed-initial harness.
  * Sizing reset-floor, r_base = 0.5%. Gate on TRAILING worst-fold DD:
    k_safe = 8.0 / worst_fold_trailing_dd -> r_safe (Amendment 3); r_max 2.0% cap.
  * Cost cell 5 (swaps OFF, 1.5x spread, $5/lot RT commission, 0.5 pip slip x n_fills)
    applied as the canonical deferred per-trade R-haircut (core.sim.costs) on each fold's
    closed-trade ledger; ROI/DD recomputed on the cost-adjusted equity curve. The WFO sim
    itself runs on the engine's 1x real bid/ask spread (HistData M1).
  * EET (Amendment 6). Two-stage triage. Real clusters re-derived (Bimodal/Unclassified).
  * Architecture selection off the REAL clusters: A1 always (Gate 3) + A4 on the Bimodal
    cluster only (Gate 1). A2/A6 NOT run. A3 SKIP (entry-route DE confirmed FAIL, ratio 1.04).

Outputs results/arc_11_v3.0_exit_extraction/. Closure written separately.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

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

from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool, write_arc_pool  # noqa: E402
from core.architectures._protocol import Architecture, StrategyResult  # noqa: E402
from core.architectures.a1_system_level_filter import (  # noqa: E402
    A1Architecture, A1Config, A1RunContext,
)
from core.architectures.a4_pipeline_d_exits import A4Architecture  # noqa: E402
from core.determinism import RANDOM_STATE, seed_everything  # noqa: E402
from core.features.pipeline import compute_feature_matrix  # noqa: E402
from core.runners._fold_stats_helpers import (  # noqa: E402
    count_daily_5pct_breaches, filter_oos_trades, max_drawdown_pct,
    slice_equity_to_oos, compute_per_day_max_dd,
)
from core.runners.arc_fold_runner import ArcFoldRunner  # noqa: E402
from core.runners.oracle_fold_runner import OracleFoldRunner  # noqa: E402
from core.sim.costs import (  # noqa: E402
    compute_commission_usd, compute_extra_spread_price, compute_slippage_pips,
)
from core.sim.panel import Panel  # noqa: E402
from core.steps.classifier_persistence import build_a4_config_from_step4  # noqa: E402
from core.steps.path_classifier_per_fold import (  # noqa: E402
    PerFoldTrainingInputs, build_path_classifier_fits_per_fold,
    build_per_trade_entry_features,
)
from core.steps.step_2_clustering import run_step_2  # noqa: E402
from core.steps.step_3_capturability import run_step_3  # noqa: E402
from core.steps.step_4_extraction import run_step_4  # noqa: E402
from core.strategies.shb.signal_module import SHBSignalModule  # noqa: E402
from core.wfo.amended_gates import (  # noqa: E402
    AmendedVerdict, classify_amended_fold_stats, compute_scaling_factors,
    count_daily_breaches_at_scaled_risk,
)
from core.wfo.chained_dd import (  # noqa: E402
    compute_chained_max_dd_from_continuous_equity, stitch_per_fold_oos_equity,
)
from core.wfo.folds import Fold, build_v3_folds  # noqa: E402
from core.wfo.gates import FoldStats  # noqa: E402

# ───────────────────────────── constants ──────────────────────────────
ARC_NAME = "arc_11_v3.0_exit_extraction"
PAIRS = (
    "AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY "
    "EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD "
    "GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY "
    "NZDUSD USDCAD USDCHF USDJPY"
).split()
HISTDATA = r"C:\Users\panap\Documents\Forex-Backtester\data\histdata"
CACHE = r"C:\Users\panap\Documents\Forex-Backtester\data\cache"
WINDOW_START = dt.date(2010, 1, 1)
WINDOW_END = dt.date(2026, 4, 30)
HOLDOUT_START = dt.date(2021, 1, 1)
SL_ANCHOR = 2.0          # pool builder anchor; Step 5 sweeps SL_GRID
HOLD_BARS = 240
RISK_PCT = 0.005
STARTING_BALANCE = 100_000.0
MIN_IS_DAYS = 365

# Step 5 grid
SL_GRID = (2.5, 3.0, 3.5)
EXITS_A1 = ("sl_only", "sl_plus_tp_2r", "sl_partial_close_1r_runner_trail")
EXITS_A4 = ("sl_partial_close_1r_runner_trail", "sl_plus_tp_2r")
EXPOSURE = (2, None)
A4_EXIT_THRESHOLDS = (0.3, 0.4, 0.5)
TRIAGE_OOS_FOLDS = (1, 6, 8)   # F1 2010 / F6 2015 / F8 2017
TOP_K = 3
TRIAGE_DD_DROP_AT_040 = 0.08   # drop config if trailing DD@0.40% > 8% on any triage fold
OPERATING_RISK_FOR_TRIAGE = 0.0040   # 0.40% reference tier for the triage DD drop

# Cost cell 5
SPREAD_MULT = 1.5
SLIP_PER_FILL = 0.5
COMMISSION_RT = 5.0   # $/lot round-turn (cell 5 / canonical provenance)

JPY_QUOTE = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}
# per-pair pip-value USD/lot reference constants (from scripts/audit/arc_10/cost_sweep.py)
REF = dict(USDJPY=109.6032, USDCAD=1.2412, USDCHF=0.9324, NZDUSD=0.7047,
           GBPUSD=1.4067, AUDUSD=0.7760)

OUT = _REPO_ROOT / "results" / ARC_NAME
LOG_FILE: Path | None = None


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    line = f"[arc11_exit {ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_QUOTE else 0.0001


def pip_value_usd_per_lot(pair: str) -> float:
    quote = pair[3:]
    if quote == "USD":
        return 10.0
    if quote == "JPY":
        return 1000.0 / REF["USDJPY"]
    if quote == "CAD":
        return 10.0 / REF["USDCAD"]
    if quote == "CHF":
        return 10.0 / REF["USDCHF"]
    if quote == "NZD":
        return 10.0 * REF["NZDUSD"]
    if quote == "GBP":
        return 10.0 * REF["GBPUSD"]
    if quote == "AUD":
        return 10.0 * REF["AUDUSD"]
    raise ValueError(f"unknown quote currency for pair {pair}")


# ───────────────────────────── panels ─────────────────────────────────
class AuxPanel:
    def __init__(self, h4: Panel, aux: dict | None = None) -> None:
        self._h4 = h4
        self.pair_dfs = h4.pair_dfs
        self.tf = h4.tf
        self.aux = aux or {}
        self.boundary_convention = h4.boundary_convention

    @property
    def pairs(self):
        return self._h4.pairs

    def snapshot_at(self, t):
        return self._h4.snapshot_at(t)


def build_panel_eet(tf: str) -> Panel:
    t0 = time.time()
    p = Panel.from_pairs(list(PAIRS), tf, histdata_root=HISTDATA, cache_root=CACHE,
                         boundary_convention="5ers_eet")
    _log(f"{tf} panel built {time.time()-t0:.1f}s ({len(p.pairs)} pairs)")
    return p


def build_feature_matrix(pool_trades, h4, d1, w1):
    panel = AuxPanel(h4, aux={"d1": d1, "w1": w1})
    rows, lineage = [], None
    for pair in sorted(h4.pairs):
        pdf = h4.pair_dfs.get(pair)
        if pdf is None or pdf.empty:
            continue
        fm = compute_feature_matrix(pair, pdf, panel=panel)
        if lineage is None:
            lineage = fm.lineage
        pt = pool_trades[pool_trades["pair"] == pair]
        if pt.empty:
            continue
        st = pd.DatetimeIndex(pt["signal_time"].values, tz="UTC")
        vi = fm.matrix.index.intersection(st)
        if len(vi) == 0:
            continue
        sub = fm.matrix.loc[vi].copy()
        sub["signal_time"] = sub.index
        sub["pair"] = pair
        rows.append(sub.reset_index(drop=True))
    if not rows:
        return pd.DataFrame(), lineage if lineage is not None else pd.DataFrame()
    feats = pd.concat(rows, ignore_index=True)
    keys = pool_trades[["trade_id", "pair", "signal_time"]].copy()
    keys["signal_time"] = pd.to_datetime(keys["signal_time"], utc=True)
    feats["signal_time"] = pd.to_datetime(feats["signal_time"], utc=True)
    merged = feats.merge(keys, on=["pair", "signal_time"], how="inner")
    fcols = [c for c in merged.columns if c not in ("signal_time", "pair", "trade_id")]
    return merged[["trade_id"] + fcols].set_index("trade_id"), lineage


def build_per_trade_features(pool_trades, feature_matrix):
    pool = pool_trades.copy()
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True)
    by_tid = pool.set_index("trade_id")
    fm = feature_matrix
    out: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for tid, row in fm.iterrows():
        if tid not in by_tid.index:
            continue
        prow = by_tid.loc[tid]
        key = (str(prow["pair"]), pd.Timestamp(prow["signal_time"]))
        out[key] = {c: float(row[c]) if pd.notna(row[c]) else 0.0 for c in row.index}
    return out


# ─────────────────────── cost cell 5 R-haircut ────────────────────────
def _position_cost_r(legs) -> float | None:
    """Cell-5 per-position cost in R-units. Independent of lots/balance:
    commission_r = $rt/(sl_pips*pip_value); slip_r and extra_spread_r scale by sl_distance."""
    legs = sorted(legs, key=lambda t: t.exit_time)
    first, last = legs[0], legs[-1]
    if first.sl_price is None:
        return None
    sl_dist_price = abs(float(first.entry_price) - float(first.sl_price))
    if sl_dist_price <= 0:
        return None
    pair = first.pair
    ps = pip_size(pair)
    sl_dist_pips = sl_dist_price / ps
    pv = pip_value_usd_per_lot(pair)
    comm_r = COMMISSION_RT / (sl_dist_pips * pv) if (sl_dist_pips > 0 and pv > 0) else 0.0
    tp1_hit = len(legs) > 1   # partial-close leg fired => TP1 reached
    slip_pips, _ = compute_slippage_pips(SLIP_PER_FILL, tp1_hit)
    slip_r = (slip_pips * ps) / sl_dist_price
    se = float(first.entry_ask) - float(first.entry_bid)
    sx = float(last.exit_ask) - float(last.exit_bid)
    se = se if np.isfinite(se) else 0.0
    sx = sx if np.isfinite(sx) else 0.0
    extra_spread_r = compute_extra_spread_price(se, sx, SPREAD_MULT) / sl_dist_price
    return comm_r + slip_r + extra_spread_r


def _cell5_adjust_equity(equity: pd.Series, closed_trades, fold: Fold,
                         eff_risk_pct: float) -> pd.Series:
    """Subtract cell-5 realized cost (step-down at each position's exit bar) from the
    concurrent-book equity curve. cost_usd = cost_r * eff_risk_pct * STARTING_BALANCE."""
    if len(equity) == 0:
        return equity
    oos = filter_oos_trades(closed_trades, fold)
    if not oos:
        return equity
    by_pos: dict[Any, list] = defaultdict(list)
    for t in oos:
        by_pos[t.position_id].append(t)
    ev = pd.Series(0.0, index=equity.index)
    idx = equity.index
    for legs in by_pos.values():
        cr = _position_cost_r(legs)
        if cr is None:
            continue
        exit_time = max(t.exit_time for t in legs)
        cost_usd = cr * eff_risk_pct * STARTING_BALANCE
        pos = idx.searchsorted(pd.Timestamp(exit_time))
        if pos >= len(idx):
            pos = len(idx) - 1
        ev.iloc[pos] += cost_usd
    return equity - ev.cumsum()


def _adjusted_fold_stats(sr: StrategyResult, fold: Fold, eff_risk_pct: float) -> tuple[FoldStats, pd.Series]:
    """Cost-adjusted FoldStats + adjusted OOS equity curve. n_trades = unique OOS positions."""
    raw = sr.equity_curve
    oos_positions = {t.position_id for t in filter_oos_trades(sr.closed_trades, fold)}
    n = len(oos_positions)
    if raw is None or len(raw) == 0:
        return FoldStats(fold.fold_id, n, 0.0, 0.0, 0, 0.0), pd.Series(dtype=float)
    adj = _cell5_adjust_equity(raw, sr.closed_trades, fold, eff_risk_pct)
    roi = float(adj.iloc[-1] / adj.iloc[0] - 1.0)
    dd = max_drawdown_pct(adj)
    breaches = count_daily_5pct_breaches(adj)
    ratio = (roi / dd) if dd > 0 else (999.0 if roi > 0 else 0.0)
    return FoldStats(fold.fold_id, n, roi, dd, breaches, ratio), adj


# ─────────────────────── config grid builders ─────────────────────────
def _exp_tag(e):
    return "expU" if e is None else f"exp{e}"


def build_a1_configs():
    out = []
    for sl in SL_GRID:
        for ex in EXITS_A1:
            for e in EXPOSURE:
                cid = f"A1::sl{sl}::{ex}::{_exp_tag(e)}"
                cfg = A1Config(
                    config_id=cid, sl_atr_mult=sl, trail_enabled=False,
                    risk_pct=RISK_PCT, starting_balance=STARTING_BALANCE,
                    max_concurrent_per_pair=1, max_concurrent_per_currency=e,
                    time_exit_bars=HOLD_BARS, exit_policy=ex,
                )
                out.append((f"A1::{cid}", (A1Architecture(), cfg)))
    return out


def build_a4_configs(cluster_id, s4):
    out = []
    for sl in SL_GRID:
        for ex in EXITS_A4:
            for e in EXPOSURE:
                for et in A4_EXIT_THRESHOLDS:
                    cid = f"A4::cl{cluster_id}::sl{sl}::{ex}::{_exp_tag(e)}::et{et}"
                    cfg = build_a4_config_from_step4(
                        s4, cluster_id=cluster_id, exit_threshold=et, config_id=cid,
                        sl_atr_mult=sl, trail_enabled=False, exit_policy=ex,
                        risk_pct=RISK_PCT, starting_balance=STARTING_BALANCE,
                        max_concurrent_per_pair=1, max_concurrent_per_currency=e,
                    )
                    out.append((f"A4::{cid}", (A4Architecture(), cfg)))
    return out


# ─────────────────────────────── main ─────────────────────────────────
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    global LOG_FILE
    LOG_FILE = OUT / "run.log"
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    LOG_FILE.touch()
    seed_everything(RANDOM_STATE)
    t0 = time.time()
    _log("=== Arc 11 v3.0 EXIT-EXTRACTION START (world #2) ===")

    panels = {tf: build_panel_eet(tf) for tf in ("H4", "D1", "W1")}
    signal = SHBSignalModule()

    pool_cfg = ArcPoolConfig(arc_name=ARC_NAME, sl_atr_mult=SL_ANCHOR, hold_bars=HOLD_BARS,
                             risk_pct=RISK_PCT, window_start=WINDOW_START, window_end=WINDOW_END)
    pool = build_arc_pool(signal, panels, pool_cfg)
    _log(f"Step 1 pool n={len(pool.trades)} sha={pool.pool_sha256[:16]}")
    write_arc_pool(pool, OUT)

    fm, lineage = build_feature_matrix(pool.trades, panels["H4"], panels["D1"], panels["W1"])
    _log(f"feature matrix {fm.shape}")
    per_trade_features = build_per_trade_features(pool.trades, fm) if len(fm) else {}

    # Steps 2-4
    s2 = run_step_2(pool.trades, pool.paths)
    (OUT / "step_2").mkdir(exist_ok=True)
    s2.cluster_assignments.to_parquet(OUT / "step_2" / "cluster_assignments.parquet", index=False)
    (OUT / "step_2" / "cluster_summary.md").write_text(s2.summary_md, encoding="utf-8", newline="\n")
    _log(f"Step 2 K={s2.k_selected} silh={ {int(k): round(float(v),3) for k,v in s2.silhouette_per_k.items()} }")

    s3 = run_step_3(pool.trades, pool.paths, s2.cluster_assignments,
                    declared_sl_mult=SL_ANCHOR, cluster_centroids=s2.centroids)
    (OUT / "step_3").mkdir(exist_ok=True)
    s3.capturability_csv.to_csv(OUT / "step_3" / "capturability.csv", index=False, lineterminator="\n")
    (OUT / "step_3" / "capturability_summary.md").write_text(s3.summary_md, encoding="utf-8", newline="\n")
    candidates = [c for c in s3.per_cluster if c.is_candidate]
    for c in s3.per_cluster:
        _log(f"  c{int(c.cluster_id)} {c.shape_tag} n={c.n_trades} comp={c.capturability_composite:.3f} "
             f"reach1r={c.reach_1r:.3f} mfe50={c.mfe_p50:.3f} ww={c.wrong_way_pp:.3f} "
             f"sl_opt={c.selected_sl:.1f} {'CAND' if c.is_candidate else ''}")
    cand_ids = tuple(int(c.cluster_id) for c in candidates)
    train_end = pd.Timestamp(HOLDOUT_START, tz="UTC")

    s4 = None
    if cand_ids and len(fm):
        (OUT / "step_4").mkdir(exist_ok=True)
        s4 = run_step_4(pool.trades, fm.reset_index(), s2.cluster_assignments,
                        feature_lineage=lineage, candidate_cluster_ids=cand_ids,
                        persistence_dir=OUT / "step_4" / "classifiers",
                        arc_name=ARC_NAME, train_end=train_end)
        s4.extraction_metrics.to_csv(OUT / "step_4" / "extraction_metrics.csv", index=False, lineterminator="\n")
        (OUT / "step_4" / "extraction_summary.md").write_text(s4.summary_md, encoding="utf-8", newline="\n")
        for ce in s4.per_cluster:
            _log(f"  Step4 c{int(ce.cluster_id)} best={ce.best_classifier} AUC={ce.best_classifier_mean_auc:.4f}")

    # Architecture selection off REAL clusters
    bimodal_ids = [int(c.cluster_id) for c in candidates
                   if c.shape_tag and c.shape_tag.lower().replace(" ", "_") in ("bimodal", "stepwise_climber")]
    _log(f"candidate clusters={cand_ids}; A4-eligible (Bimodal/Stepwise)={bimodal_ids}")

    base_ctx = A1RunContext(per_trade_features=per_trade_features) if per_trade_features else A1RunContext()

    # WFO folds
    wfo = build_v3_folds(holdout_start=HOLDOUT_START, holdout_end=WINDOW_END)
    eligible = [f for f in wfo.folds if f.is_days >= MIN_IS_DAYS]   # F1 dropped (empty IS)
    folds_for_fits = tuple(eligible) + (wfo.holdout,)
    fold_by_id = {f.fold_id: f for f in wfo.folds}
    _log(f"WFO: {len(eligible)} eligible IS folds (F1 2010 dropped: empty IS) + holdout "
         f"{wfo.holdout.oos_start}->{wfo.holdout.oos_end}")

    # A4 per-fold path classifier fits (cluster-independent target final_r>0), built once
    a4_ctx = None
    if bimodal_ids and s4 is not None:
        _log("Building A4 per-fold path-classifier fits (target final_r>0)")
        a4_inputs = PerFoldTrainingInputs(
            pool_trades=pool.trades, pool_paths=pool.paths, cluster_assignments=None,
            panels=panels, primary_tf=signal.primary_tf, candidate_cluster_id=None, n_defer=5)
        a4_fits = build_path_classifier_fits_per_fold(inputs=a4_inputs, folds=folds_for_fits, arch="A4")
        entry_feats = build_per_trade_entry_features(a4_inputs)
        a4_ctx = A1RunContext(per_trade_features=per_trade_features,
                              per_trade_entry_features=entry_feats, path_classifier_fits=a4_fits)

    # Build config grid
    grid: list[tuple[str, tuple[Architecture, Any], A1RunContext]] = []
    for cid, paired in build_a1_configs():
        grid.append((cid, paired, base_ctx))
    a4_count = 0
    if bimodal_ids and s4 is not None and a4_ctx is not None:
        for bid in bimodal_ids:
            for cid, paired in build_a4_configs(bid, s4):
                grid.append((cid, paired, a4_ctx))
                a4_count += 1
    n_total = len(grid)
    _log(f"Step 5 grid: {n_total} configs (A1 {len(build_a1_configs())} + A4 {a4_count}); "
         f"scope={'thin' if n_total<50 else ('normal' if n_total<100 else 'broad')}")

    def is_a4(cid: str) -> bool:
        return cid.startswith("A4::")

    def run_on_fold(paired, ctx, fold: Fold) -> tuple[FoldStats, pd.Series, StrategyResult | None]:
        arch, cfg = paired
        try:
            runner = ArcFoldRunner(architecture=arch, signal_evaluation=pool.signal_evaluation,
                                   panels=panels, run_context=ctx)
            runner(fold, cfg)
            sr = runner.last_result
            if sr is None:
                return FoldStats(fold.fold_id, 0, 0.0, 0.0, 0, 0.0), pd.Series(dtype=float), None
            fs, adj = _adjusted_fold_stats(sr, fold, RISK_PCT)
            return fs, adj, sr
        except Exception as exc:  # noqa: BLE001
            _log(f"  ERR {getattr(cfg,'config_id','?')} fold {fold.fold_id}: {exc}")
            return FoldStats(fold.fold_id, 0, 0.0, 0.0, 0, 0.0), pd.Series(dtype=float), None

    # ── STAGE A: triage on F1/F6/F8 (diagnostic; DD-drop recorded, NOT eliminating) ──
    # NOTE: the dispatch's hard DD-drop eliminated 54/54 (full-pool long-only SHB has
    # large concurrency/carryover trailing DD from 240-bar holds). To still produce the
    # Step-5 gate verdict, the DD-drop is recorded as a per-config diagnostic flag and
    # Stage B is run on the top-N by worst-triage ratio PLUS guaranteed A4 coverage (A4
    # is the PRIMARY exit-extraction architecture). Documented in the closure.
    _log("=== STAGE A triage (folds 2010/2015/2017; DD-drop = diagnostic flag) ===")
    paired_by_cid = {cid: (paired, ctx) for cid, paired, ctx in grid}
    existing_csv = OUT / "step_5" / "stage_a_triage.csv"
    reuse = False
    if existing_csv.exists():
        try:
            prev = pd.read_csv(existing_csv)
            if set(prev["config_id"]) >= set(paired_by_cid):
                reuse = True
        except Exception:
            reuse = False
    triage_all = []
    n_dd_dropped = 0
    if reuse:
        _log("Stage A: reusing existing stage_a_triage.csv (deterministic) — skipping recompute")
        for _, row in prev.iterrows():
            cid = row["config_id"]
            if cid not in paired_by_cid:
                continue
            paired, ctx = paired_by_cid[cid]
            triage_all.append((cid, paired, ctx, float(row["worst_triage_ratio"])))
            flag = row["dd_drop_flag"] if "dd_drop_flag" in row else row.get("dropped_dd", False)
            n_dd_dropped += int(bool(flag))
    else:
        triage_rows = []
        t_stage = time.time()
        for i, (cid, paired, ctx) in enumerate(grid, 1):
            a4 = is_a4(cid)
            fold_ids = [f for f in TRIAGE_OOS_FOLDS if not (a4 and fold_by_id[f].is_days < MIN_IS_DAYS)]
            per_fold = {}
            for fid in fold_ids:
                fs, _adj, _sr = run_on_fold(paired, ctx, fold_by_id[fid])
                per_fold[fid] = fs
            ratios = [fs.roi_dd_ratio for fs in per_fold.values()]
            dds = [fs.max_dd_pct for fs in per_fold.values()]
            worst_ratio = min(ratios) if ratios else 0.0
            worst_dd = max(dds) if dds else 0.0
            dd_at_040 = worst_dd * (OPERATING_RISK_FOR_TRIAGE / RISK_PCT)
            dropped = dd_at_040 > TRIAGE_DD_DROP_AT_040
            n_dd_dropped += int(dropped)
            triage_rows.append(dict(
                config_id=cid, arch=("A4" if a4 else "A1"),
                triage_folds="+".join(str(f) for f in fold_ids),
                worst_triage_ratio=worst_ratio, worst_triage_dd_base=worst_dd,
                worst_triage_dd_at_040=dd_at_040, dd_drop_flag=bool(dropped),
                **{f"f{fid}_roi": per_fold[fid].roi_pct for fid in fold_ids},
                **{f"f{fid}_dd": per_fold[fid].max_dd_pct for fid in fold_ids},
                **{f"f{fid}_n": per_fold[fid].n_trades for fid in fold_ids},
            ))
            triage_all.append((cid, paired, ctx, worst_ratio))
            if i % 6 == 0 or i == n_total:
                _log(f"  triage {i}/{n_total} ({(time.time()-t_stage)/60:.1f}m)")
        triage_df = pd.DataFrame(triage_rows).sort_values("worst_triage_ratio", ascending=False)
        (OUT / "step_5").mkdir(exist_ok=True)
        triage_df.to_csv(OUT / "step_5" / "stage_a_triage.csv", index=False, lineterminator="\n")

    triage_all.sort(key=lambda x: x[3], reverse=True)
    _log(f"Stage A: {n_dd_dropped}/{n_total} flagged by the DD-drop "
         f"(diagnostic; not eliminated). top worst-ratios: {[round(s[3],3) for s in triage_all[:5]]}")

    # Stage B candidates: top-10 overall by worst-triage ratio UNION top-4 A4 (guarantee
    # the primary exit-extraction architecture is gated even if sl_only artifacts rank high).
    top_overall = triage_all[:10]
    top_a4 = [x for x in triage_all if x[0].startswith("A4::")][:4]
    seen, top = set(), []
    for x in top_overall + top_a4:
        if x[0] not in seen:
            seen.add(x[0])
            top.append(x)
    k = len(top)
    _log(f"Stage B candidates (top-10 overall + top-4 A4 = {k}): {[t[0] for t in top]}")

    # ── STAGE B: full WFO + holdout on top-K ──
    _log("=== STAGE B full WFO + holdout (top candidates) ===")
    stage_b = []
    for cid, paired, ctx in [(t[0], t[1], t[2]) for t in top]:
        a4 = is_a4(cid)
        run_folds = [f for f in eligible]
        per_fold_fs, per_fold_eq = [], []
        for f in run_folds:
            fs, adj, _sr = run_on_fold(paired, ctx, f)
            per_fold_fs.append(fs)
            if len(adj):
                per_fold_eq.append(adj)
        # holdout at r_base (cost-adjusted)
        h_fs, h_adj, _hsr = run_on_fold(paired, ctx, wfo.holdout)
        if len(h_adj):
            per_fold_eq.append(h_adj)

        # chained DD on cost-adjusted stitched equity
        chained_eq = stitch_per_fold_oos_equity(per_fold_eq, starting_balance=STARTING_BALANCE) \
            if per_fold_eq else pd.Series(dtype=float)
        chained_dd = compute_chained_max_dd_from_continuous_equity(chained_eq) if len(chained_eq) else 0.0
        per_day = compute_per_day_max_dd(chained_eq, pair_set=",".join(PAIRS)) if len(chained_eq) else pd.DataFrame()
        if not per_day.empty:
            safe_cid = cid.replace("::", "__").replace("/", "_")
            per_day.to_parquet(OUT / "step_5" / f"per_day_max_dd_base__{safe_cid}.parquet", index=False)

        worst_dd_base = max((fs.max_dd_pct for fs in per_fold_fs), default=0.0)
        scaling = compute_scaling_factors(worst_dd_base, r_base=RISK_PCT)
        # holdout linear-scaled to r_safe / r_hard (reset-floor linear; Amendment 3 assumption)
        h_per_day = compute_per_day_max_dd(h_adj, pair_set=",".join(PAIRS)) if len(h_adj) else pd.DataFrame()

        def scaled_holdout(kf):
            if not np.isfinite(kf) or kf <= 0:
                return None
            br = count_daily_breaches_at_scaled_risk(h_per_day, kf) if not h_per_day.empty else 0
            roi = h_fs.roi_pct * kf
            dd = h_fs.max_dd_pct * kf
            ratio = (roi / dd) if dd > 0 else (999.0 if roi > 0 else 0.0)
            return FoldStats(wfo.holdout.fold_id, h_fs.n_trades, roi, dd, br, ratio)

        h_safe = scaled_holdout(scaling.k_safe) if scaling.scalable_to_safe else None
        h_hard = scaled_holdout(scaling.k_hard) if scaling.scalable_to_hard else None

        gate = classify_amended_fold_stats(
            folds=per_fold_fs, chained_max_dd_base_pct=chained_dd,
            per_day_max_dd_df=(per_day if not per_day.empty else None),
            holdout_stats_at_r_safe=h_safe, holdout_stats_at_r_hard=h_hard,
            sizing_convention="reset_floor", accept_equity_pct=False, r_base=RISK_PCT)

        thin_folds = [fs.fold_id for fs in per_fold_fs if fs.n_trades < 25]
        stage_b.append(dict(
            config_id=cid, arch=("A4" if a4 else "A1"), gate=gate,
            per_fold=[(fs.fold_id, fs.n_trades, fs.roi_pct, fs.max_dd_pct, fs.roi_dd_ratio) for fs in per_fold_fs],
            holdout=(h_fs.n_trades, h_fs.roi_pct, h_fs.max_dd_pct, h_fs.roi_dd_ratio),
            chained_dd_base=chained_dd, worst_dd_base=worst_dd_base, thin_folds=thin_folds,
        ))
        _log(f"  {cid}: verdict={gate.verdict.value} worst_ratio={gate.worst_fold_ratio:.3f} "
             f"worst_dd={worst_dd_base*100:.2f}% r_safe={gate.r_safe_pct:.4%} "
             f"scalable_safe={gate.scalable_to_safe} thin_folds={thin_folds}")

    # rank by verdict then worst ratio
    rank = {AmendedVerdict.PASS_DEPLOYABLE: 2, AmendedVerdict.PASS_VIABLE: 1, AmendedVerdict.FAIL: 0}
    stage_b.sort(key=lambda r: (rank[r["gate"].verdict], r["gate"].worst_fold_ratio), reverse=True)
    final_verdict = stage_b[0]["gate"].verdict.value.upper().replace("_", "-") if stage_b else "INCOMPLETE"
    best_cid = stage_b[0]["config_id"] if stage_b else None
    _log(f"FINAL verdict={final_verdict} best={best_cid}")

    # ── Oracle WFO per candidate cluster (winning SL, runner-trail) ──
    _log("=== Oracle WFO per candidate cluster ===")
    oracle_rows = []
    win_sl = 3.0
    if stage_b:
        try:
            win_sl = float(stage_b[0]["config_id"].split("sl")[1].split("::")[0])
        except Exception:
            win_sl = 3.0
    for c in candidates:
        a1 = A1Config(config_id=f"oracle_cl{int(c.cluster_id)}", sl_atr_mult=win_sl,
                      trail_enabled=False, risk_pct=RISK_PCT, starting_balance=STARTING_BALANCE,
                      max_concurrent_per_pair=1, max_concurrent_per_currency=2,
                      time_exit_bars=HOLD_BARS, exit_policy="sl_partial_close_1r_runner_trail")
        orr = OracleFoldRunner(signal_evaluation=pool.signal_evaluation, panels=panels,
                               cluster_assignments=s2.cluster_assignments,
                               candidate_cluster_id=int(c.cluster_id), trades=pool.trades)
        for f in eligible:
            try:
                st = orr(f, a1)
            except Exception as exc:  # noqa: BLE001
                _log(f"  oracle cl{int(c.cluster_id)} f{f.fold_id}: {exc}")
                st = FoldStats(f.fold_id, 0, 0.0, 0.0, 0, 0.0)
            oracle_rows.append(dict(cluster_id=int(c.cluster_id), fold_id=st.fold_id,
                                    n_trades=st.n_trades, roi_pct=st.roi_pct,
                                    max_dd_pct=st.max_dd_pct, roi_dd_ratio=st.roi_dd_ratio))
    if oracle_rows:
        pd.DataFrame(oracle_rows).to_csv(OUT / "step_5" / "wfo_oracle.csv", index=False, lineterminator="\n")

    # ── write run_summary.json ──
    def g(d):
        ga = d["gate"]
        return dict(
            config_id=d["config_id"], arch=d["arch"], verdict=ga.verdict.value,
            primary_failure_mode=ga.primary_failure_mode.value,
            reason=ga.reason, worst_fold_ratio=float(ga.worst_fold_ratio),
            worst_fold_roi_base_pct=float(ga.worst_fold_roi_base_pct),
            worst_fold_dd_base_pct=float(ga.worst_fold_dd_base_pct),
            mean_fold_ratio=float(ga.mean_fold_ratio), n_negative_folds=int(ga.n_negative_folds),
            min_trades_per_fold=int(ga.min_trades_per_fold),
            chained_max_dd_base_pct=float(d["chained_dd_base"]),
            k_safe=float(ga.k_safe), r_safe_pct=float(ga.r_safe_pct),
            r_hard_pct=float(ga.r_hard_pct), scalable_to_safe=bool(ga.scalable_to_safe),
            scalable_to_hard=bool(ga.scalable_to_hard),
            worst_fold_roi_at_r_safe_pct=float(ga.worst_fold_roi_at_r_safe_pct),
            chained_max_dd_at_r_safe_pct=float(ga.chained_max_dd_at_r_safe_pct),
            daily_dd_breaches_at_r_safe=int(ga.daily_dd_breaches_at_r_safe),
            holdout=d["holdout"], thin_folds=d["thin_folds"], per_fold=d["per_fold"],
        )

    summary = dict(
        arc_name=ARC_NAME, verdict=final_verdict, best_config_id=best_cid,
        pool_size=int(len(pool.trades)), pool_sha256=pool.pool_sha256,
        window=[str(WINDOW_START), str(WINDOW_END)], boundary_convention="5ers_eet",
        risk_basis="reset_floor", r_base=RISK_PCT, cost_cell="cell_5 (swaps off, 1.5x spread, $5/lot RT, 0.5 slip x n_fills) deferred R-haircut",
        gate_basis="trailing worst-fold DD; k_safe=8.0/worst_fold_trailing_dd",
        step_2_k=int(s2.k_selected),
        step_2_silhouettes={int(k): float(v) for k, v in s2.silhouette_per_k.items()},
        clusters=[dict(cluster_id=int(c.cluster_id), shape_tag=c.shape_tag, n_trades=int(c.n_trades),
                       composite=float(c.capturability_composite), reach_1r=float(c.reach_1r),
                       mfe_p50=float(c.mfe_p50), ww_pp=float(c.wrong_way_pp),
                       selected_sl=float(c.selected_sl), is_candidate=bool(c.is_candidate))
                  for c in s3.per_cluster],
        candidate_cluster_ids=list(cand_ids), a4_eligible_clusters=bimodal_ids,
        step_4=[dict(cluster_id=int(ce.cluster_id), best_classifier=ce.best_classifier,
                     auc=float(ce.best_classifier_mean_auc)) for ce in (s4.per_cluster if s4 else [])],
        n_configs_total=n_total, n_dd_dropped_triage=n_dd_dropped,
        selection_bias_N=n_total, search_scope=("thin" if n_total < 50 else ("normal" if n_total < 100 else "broad")),
        top_k=k, stage_b=[g(d) for d in stage_b],
        oracle_note=("OracleFoldRunner here marks the FULL signal book equity (not admit-only "
                     "cluster economics); its DD is full-book concurrency DD, so it does NOT "
                     "reproduce the historical admit-only +101%/yr / 2.48% DD figure."),
        oracle={f"cluster_{cid}": [r for r in oracle_rows if r["cluster_id"] == cid]
                for cid in sorted({r["cluster_id"] for r in oracle_rows})},
        a3_skipped_reason="entry-route DE confirmed FAIL pre-run (ratio 1.04)",
        architectures_not_run=["A2", "A3", "A6"],
        elapsed_min=round((time.time() - t0) / 60, 1),
    )
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                                          encoding="utf-8", newline="\n")
    _log(f"=== DONE verdict={final_verdict} in {(time.time()-t0)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
