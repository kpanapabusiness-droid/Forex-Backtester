"""Arc 7 v3.0.2 best-config re-run at r_base = 2.0%.

> **Non-canonical status.** This script is a DIAGNOSTIC analysis tool,
> NOT the canonical Arc 7 v3.0.2 simulator. The canonical script is
> [scripts/l_arc_7_v3_0_2/run.py](../l_arc_7_v3_0_2/run.py), which
> generated the Arc 7 v3.0.2 PASS-tier closure recorded at
> [results/l_arc_7_v3.0.2/](../../results/l_arc_7_v3.0.2/). Any
> deployment-decision rerun of the canonical config at a different
> risk should invoke that script directly, not this one. The two
> scripts produce identical results when given matched inputs (see
> [docs/dispatches/driver_script_divergence_investigation.md](../../docs/dispatches/driver_script_divergence_investigation.md)),
> but this script's defaults differ in one respect documented at
> §"Driver-script defaults vs the canonical orchestrator" below.

Single driver for the two dispatched deliverables. Reuses Arc 7
v3.0.1 Step 1-4 artefacts at ``results/l_arc_7/`` (pool, clusters,
classifiers) and the canonical v3.0.2 best config:

  config_id: A6::A6::cl1::sl2.0::thr0.5-0.7::exp2
  architecture: A6 (meta-labeling)
  cluster: c1 (Unclassified, RF AUC 0.6642)
  SL: 2.0 x ATR + trailing ATR
  exposure: 2 per currency
  thresholds: lower=0.5, upper=0.7

The ONLY config change from v3.0.2 canonical is ``risk_pct = 0.005 -> 0.02``.

Driver-script defaults vs the canonical orchestrator
----------------------------------------------------
The original version of this script (2026-05-25 ``analysis/arc_7_wfo_rerun_r2pct``
branch) defaulted ``--window-end`` to ``"2025-12-31"``, four months earlier
than the canonical orchestrator's ``"2026-04-30"``. That mismatch is the
load-bearing source of the "87 vs 136" holdout trade-count discrepancy
flagged in [docs/analysis/arc_7_r2pct_rerun.md](../../docs/analysis/arc_7_r2pct_rerun.md)
§4.1. Per the post-investigation reconciliation, this script now defaults
to ``"2026-04-30"`` to match the orchestrator. See
[docs/dispatches/driver_script_divergence_investigation.md](../../docs/dispatches/driver_script_divergence_investigation.md)
for the full investigation trail.

If you need the analytical numbers under a different holdout window,
pass ``--window-end YYYY-MM-DD`` explicitly. The script's *logic* is
canonical-equivalent; only the default argument was previously off.

Deliverables:

  A) Canonical WFO at r=2% across the v3 fold structure
     (11-fold IS 2010..2020 + 2021..2025 holdout). Outputs to
     ``results/analysis/arc_7_r2pct/wfo/``.

  B) Single continuous backtest 2010-01-01 .. 2025-12-31 at r=2%.
     Mechanism (dispatch §5.1 Option 1): one Fold with
     ``oos_start=2010-01-01``, ``oos_end=2025-12-31``; the architecture's
     ``_slice_panels_to_fold(...)`` warmup (60d) seeds indicators.
     Outputs to ``results/analysis/arc_7_r2pct/full_sim/``.

NOT an arc, NOT an L_PROTOCOL closure. No tracker updates. No Step 6
dispatch. Output is a read-only analysis artefact.

Invocation::

    python -m scripts.analysis.arc_7_r2pct_rerun \\
        --histdata-root C:/Users/panap/Documents/Forex-Backtester/data/histdata \\
        --cache-root    C:/Users/panap/Documents/Forex-Backtester/data/cache \\
        --out-root      results/analysis/arc_7_r2pct \\
        --v301-root     results/l_arc_7 \\
        --deliverable   both
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.arc.signal_protocol import SignalEvaluation  # noqa: E402
from core.architectures.a1_system_level_filter import A1RunContext  # noqa: E402
from core.architectures.a6_meta_labeling import A6Architecture  # noqa: E402
from core.data.aggregator import aggregate  # noqa: E402
from core.determinism import seed_everything  # noqa: E402
from core.runners._fold_stats_helpers import compute_per_day_max_dd  # noqa: E402
from core.runners.arc_fold_runner import ArcFoldRunner  # noqa: E402
from core.sim.panel import Panel  # noqa: E402
from core.steps.classifier_persistence import build_a6_config_from_step4  # noqa: E402
from core.steps.step_3_capturability import ClusterCapturability  # noqa: E402
from core.steps.step_4_extraction import (  # noqa: E402
    ClusterExtraction,
    Step4Result,
)
from core.strategies.liquidity_sweep_reclaim_long.signal_module import (  # noqa: E402
    LiquiditySweepReclaimLongSignal,
)
from core.wfo.chained_dd import (  # noqa: E402
    compute_chained_max_dd_from_continuous_equity,
    stitch_per_fold_oos_equity,
)
from core.wfo.folds import Fold, build_v3_folds  # noqa: E402

PAIRS_28: tuple[str, ...] = (
    "AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY "
    "EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD "
    "GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY "
    "NZDUSD USDCAD USDCHF USDJPY"
).split()
assert len(PAIRS_28) == 28

WINNING_CONFIG_ID = "A6::A6::cl1::sl2.0::thr0.5-0.7::exp2"
CLUSTER_ID = 1
SL_MULT = 2.0
LOWER_THR = 0.5
UPPER_THR = 0.7
MAX_PER_CURRENCY = 2
R_BASE = 0.02
R_CANONICAL = 0.005
STARTING_BALANCE = 100_000.0

log = logging.getLogger("arc7_r2pct")


def _setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity >= 1 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(manifest_path: Path, artefacts: list[Path], extra: dict | None = None) -> None:
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
    if extra:
        payload.update(extra)
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


# ──────────────────────────────────────────────────────────────────────
# V3.0.1 artefact loading (subset of scripts/l_arc_7_v3_0_2/run.py)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class V301Subset:
    pool_trades: pd.DataFrame
    cluster_assignments: pd.DataFrame
    capturability: pd.DataFrame
    extraction_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    classifiers_manifest: dict
    candidate_clusters: list[ClusterCapturability]
    step4_per_cluster: list[ClusterExtraction]


def _load_v301(v301_root: Path) -> V301Subset:
    s1 = v301_root / "step_1"
    s2 = v301_root / "step_2"
    s3 = v301_root / "step_3"
    s4 = v301_root / "step_4"
    pool_trades = pd.read_parquet(s1 / "pool.parquet")
    cluster_assignments = pd.read_parquet(s2 / "cluster_assignments.parquet")
    capturability = pd.read_csv(s3 / "capturability.csv")
    extraction_metrics = pd.read_csv(s4 / "extraction_metrics.csv")
    feature_importance = pd.read_csv(s4 / "feature_importance.csv")
    classifiers_manifest = json.loads((s4 / "classifiers" / "manifest.json").read_text(encoding="utf-8"))

    candidate_clusters: list[ClusterCapturability] = []
    for _, r in capturability.iterrows():
        if not bool(r["is_candidate"]):
            continue
        candidate_clusters.append(ClusterCapturability(
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
            sl_sweep={},
            selected_sl=float(r["selected_sl"]),
            capturability_composite=float(r["capturability_composite"]),
            is_candidate=True,
        ))

    step4_per_cluster: list[ClusterExtraction] = []
    for cid_str, entry in classifiers_manifest.get("classifiers", {}).items():
        cid = int(cid_str)
        fi = feature_importance[feature_importance["cluster_id"] == cid]
        top_feats_agg = (
            fi.groupby("feature")["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "mean_importance", "std": "std_importance"})
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )
        step4_per_cluster.append(ClusterExtraction(
            cluster_id=cid,
            n_trades=int(entry.get("trained_on_pool_size", 0)),
            excluded_features=(),
            used_features=tuple(entry.get("feature_order", [])),
            classifier_fold_results=(),
            best_classifier=entry.get("classifier_name", "rf"),
            best_classifier_mean_auc=float(entry.get("auc_oos_cv5", 0)),
            best_threshold=float(entry.get("best_threshold", 0.5)),
            feature_importance=top_feats_agg,
            fitted_classifier_path=Path(entry["path"]) if Path(entry["path"]).is_absolute()
                else (s4 / "classifiers" / entry["path"]),
            fitted_classifier_type=entry.get("classifier_type"),
            fitted_classifier_feature_order=tuple(entry.get("feature_order", [])),
        ))

    return V301Subset(
        pool_trades=pool_trades,
        cluster_assignments=cluster_assignments,
        capturability=capturability,
        extraction_metrics=extraction_metrics,
        feature_importance=feature_importance,
        classifiers_manifest=classifiers_manifest,
        candidate_clusters=candidate_clusters,
        step4_per_cluster=step4_per_cluster,
    )


# ──────────────────────────────────────────────────────────────────────
# Panel build + per-trade feature lookup (same as v3.0.2 driver)
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


def _build_per_trade_features(
    pool_trades: pd.DataFrame, panel_h4: Panel, feature_order: tuple[str, ...],
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """Reconstruct (pair, signal_time) -> {feature_name: value} for A6 admit."""
    from core.features.pipeline import compute_feature_matrix

    matrices: dict[str, pd.DataFrame] = {}
    for pair in sorted(panel_h4.pairs):
        df = panel_h4.pair_dfs[pair]
        res = compute_feature_matrix(pair, df, panel=panel_h4)
        matrices[pair] = res.matrix

    out: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    feat_keys = list(feature_order)
    for _, trade in pool_trades.iterrows():
        pair = trade["pair"]
        signal_time = pd.Timestamp(trade["signal_time"])
        m = matrices.get(pair)
        if m is None or signal_time not in m.index:
            continue
        row = m.loc[signal_time]
        out[(pair, signal_time)] = {
            col: (float(row[col]) if (col in row.index and pd.notna(row[col])) else 0.0)
            for col in feat_keys
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# Build the (sole) winning A6 config at r=2%
# ──────────────────────────────────────────────────────────────────────


def _build_winning_a6_config(s4: Step4Result, *, risk_pct: float, exit_policy: str | None = None) -> Any:
    return build_a6_config_from_step4(
        s4, cluster_id=CLUSTER_ID,
        lower_threshold=LOWER_THR, upper_threshold=UPPER_THR,
        config_id=WINNING_CONFIG_ID,
        sl_atr_mult=SL_MULT, trail_enabled=True,
        risk_pct=risk_pct, max_concurrent_per_currency=MAX_PER_CURRENCY,
        max_concurrent_per_pair=1, max_concurrent_total=None,
        starting_balance=STARTING_BALANCE,
        exit_policy=exit_policy,
    )


# ──────────────────────────────────────────────────────────────────────
# Summary stats — shared helpers
# ──────────────────────────────────────────────────────────────────────


def _max_dd_duration_days(equity: pd.Series) -> int:
    """Calendar days from prior peak to deepest trough in equity series."""
    if equity is None or len(equity) == 0:
        return 0
    s = equity.dropna()
    if len(s) == 0:
        return 0
    cmax = s.cummax()
    dd = (cmax - s) / cmax
    if not len(dd):
        return 0
    trough_idx = dd.idxmax()
    # peak before trough = first cmax-equal point at or before trough_idx
    peak_value = cmax.loc[trough_idx]
    peak_candidates = s.loc[:trough_idx]
    peak_candidates = peak_candidates[peak_candidates >= peak_value]
    if len(peak_candidates) == 0:
        return 0
    peak_idx = peak_candidates.index[0]
    return int((trough_idx - peak_idx).days)


def _annualized_sharpe(equity: pd.Series) -> float:
    """Sharpe on H4-bar log-returns annualized to 252*6 H4 bars/yr."""
    if equity is None or len(equity) < 2:
        return 0.0
    s = equity.dropna()
    if len(s) < 2:
        return 0.0
    rets = np.log(s / s.shift(1)).dropna()
    if rets.std() == 0 or len(rets) == 0:
        return 0.0
    # H4 bars per year (forex 5x24 ~ 6 H4 bars/day x 252 sessions)
    bars_per_year = 252 * 6
    return float(rets.mean() / rets.std() * np.sqrt(bars_per_year))


def _worst_rolling_window_dd(equity: pd.Series, *, window_days: int = 30) -> float:
    """Worst rolling N-day DD: max of (max(equity[t-N..t]) - min(equity[t-N..t])) / max."""
    if equity is None or len(equity) == 0:
        return 0.0
    s = equity.dropna()
    if len(s) == 0:
        return 0.0
    rolling = s.rolling(window=f"{window_days}D", min_periods=1)
    rolling_max = rolling.max()
    rolling_min = rolling.min()
    rolling_dd = (rolling_max - rolling_min) / rolling_max
    return float(rolling_dd.max()) if len(rolling_dd) else 0.0


def _top_n_drawdown_periods(equity: pd.Series, *, n: int = 5) -> list[dict]:
    """Identify the N deepest distinct drawdown periods (peak -> trough -> recovery)."""
    if equity is None or len(equity) == 0:
        return []
    s = equity.dropna()
    if len(s) == 0:
        return []
    periods: list[dict] = []
    cmax = s.cummax()
    is_underwater = s < cmax
    if not is_underwater.any():
        return []
    # Find runs of underwater
    start_idx = None
    runs: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    last_peak = s.iloc[0]
    last_peak_ts = s.index[0]
    for ts, val in s.items():
        if val >= last_peak:
            if start_idx is not None:
                # Close out an underwater run that recovered
                run_s = s.loc[start_idx:ts]
                trough_ts = run_s.idxmin()
                runs.append((start_idx, trough_ts, ts))
                start_idx = None
            last_peak = val
            last_peak_ts = ts
        else:
            if start_idx is None:
                start_idx = last_peak_ts
    # Open-ended underwater at the end
    if start_idx is not None:
        run_s = s.loc[start_idx:]
        trough_ts = run_s.idxmin()
        runs.append((start_idx, trough_ts, run_s.index[-1]))
    for peak_ts, trough_ts, end_ts in runs:
        peak_val = s.loc[peak_ts]
        trough_val = s.loc[trough_ts]
        depth_pct = (peak_val - trough_val) / peak_val
        recovery_days = (end_ts - trough_ts).days if end_ts >= trough_ts else None
        periods.append({
            "peak_ts": str(peak_ts.date()) if hasattr(peak_ts, "date") else str(peak_ts),
            "trough_ts": str(trough_ts.date()) if hasattr(trough_ts, "date") else str(trough_ts),
            "end_ts": str(end_ts.date()) if hasattr(end_ts, "date") else str(end_ts),
            "depth_pct": float(depth_pct),
            "trough_to_recovery_days": int(recovery_days) if recovery_days is not None else None,
            "duration_days": int((end_ts - peak_ts).days),
        })
    periods.sort(key=lambda p: p["depth_pct"], reverse=True)
    return periods[:n]


def _years_with_negative_return(equity: pd.Series) -> list[int]:
    if equity is None or len(equity) == 0:
        return []
    s = equity.dropna()
    if len(s) == 0:
        return []
    yearly = s.resample("YE").last()
    if len(yearly) < 2:
        return []
    yearly_ret = yearly.pct_change().dropna()
    neg_years = [int(ts.year) for ts, v in yearly_ret.items() if v < 0]
    return neg_years


def _build_trade_ledger(run_result: Any, sl_lookup_atr: float) -> pd.DataFrame:
    """Convert closed_trades into a flat ledger DataFrame.

    Computes R per trade via sl_distance = abs(entry_price - sl_price)
    (preferred when present) or pnl / size as fallback for legacy rows.
    """
    rows: list[dict] = []
    for t in run_result.closed_trades:
        sl_price = getattr(t, "sl_price", None)
        if sl_price is not None and sl_price > 0:
            sl_distance = abs(t.entry_price - sl_price)
            r_at_close = (
                ((t.exit_price - t.entry_price) * (1 if t.direction.sign > 0 else -1))
                / sl_distance if sl_distance > 0 else 0.0
            )
        else:
            r_at_close = 0.0
        rows.append({
            "position_id": t.position_id,
            "parent_position_id": t.parent_position_id,
            "pair": t.pair,
            "direction": "LONG" if t.direction.sign > 0 else "SHORT",
            "entry_time": pd.Timestamp(t.entry_time),
            "exit_time": pd.Timestamp(t.exit_time),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "size": float(t.size),
            "pnl": float(t.pnl),
            "exit_reason": str(t.exit_reason),
            "sl_price": float(sl_price) if sl_price is not None else float("nan"),
            "r_at_close": float(r_at_close),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "position_id", "parent_position_id", "pair", "direction",
            "entry_time", "exit_time", "entry_price", "exit_price",
            "size", "pnl", "exit_reason", "sl_price", "r_at_close",
        ])
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Deliverable A — canonical WFO at r=2%
# ──────────────────────────────────────────────────────────────────────


def run_deliverable_a(
    *,
    out_dir: Path,
    panels: Mapping[str, Panel],
    signal_eval: SignalEvaluation,
    base_ctx: A1RunContext,
    s4_result: Step4Result,
    window_end_str: str,
) -> dict:
    """Run WFO across 11 v3 folds + holdout at r_base = 2.0%."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build winning config at r=2%
    cfg = _build_winning_a6_config(s4_result, risk_pct=R_BASE)
    arch = A6Architecture()

    wfo_struct = build_v3_folds(holdout_end=date.fromisoformat(window_end_str))
    eligible_folds = [f for f in wfo_struct.folds if f.is_days >= 365]
    log.info("Deliverable A: %d eligible IS folds + 1 holdout fold", len(eligible_folds))

    runner = ArcFoldRunner(
        architecture=arch, signal_evaluation=signal_eval,
        panels=panels, run_context=base_ctx,
    )

    # Run IS folds
    per_fold_rows: list[dict] = []
    per_fold_equity: list[pd.Series] = []
    t0 = time.perf_counter()
    for f in eligible_folds:
        log.info("Fold %d: oos=%s..%s", f.fold_id, f.oos_start, f.oos_end)
        stats = runner(f, cfg)
        sr = runner.last_result
        per_fold_rows.append({
            "fold_id": f.fold_id,
            "oos_start": f.oos_start.isoformat(),
            "oos_end": f.oos_end.isoformat(),
            "n_trades": stats.n_trades,
            "roi_pct": stats.roi_pct,
            "max_dd_pct": stats.max_dd_pct,
            "days_breaching_daily_5pct": stats.days_breaching_daily_5pct,
            "roi_dd_ratio": stats.roi_dd_ratio,
        })
        if sr is not None and len(sr.equity_curve) > 0:
            per_fold_equity.append(sr.equity_curve)
    is_elapsed = time.perf_counter() - t0
    log.info("Deliverable A IS folds done in %.1fs", is_elapsed)

    # Holdout one-shot
    holdout_row: dict | None = None
    holdout_equity: pd.Series | None = None
    if wfo_struct.holdout is not None:
        log.info("Holdout: oos=%s..%s", wfo_struct.holdout.oos_start, wfo_struct.holdout.oos_end)
        t0 = time.perf_counter()
        stats = runner(wfo_struct.holdout, cfg)
        sr = runner.last_result
        holdout_row = {
            "fold_id": wfo_struct.holdout.fold_id,
            "oos_start": wfo_struct.holdout.oos_start.isoformat(),
            "oos_end": wfo_struct.holdout.oos_end.isoformat(),
            "n_trades": stats.n_trades,
            "roi_pct": stats.roi_pct,
            "max_dd_pct": stats.max_dd_pct,
            "days_breaching_daily_5pct": stats.days_breaching_daily_5pct,
            "roi_dd_ratio": stats.roi_dd_ratio,
        }
        if sr is not None and len(sr.equity_curve) > 0:
            holdout_equity = sr.equity_curve
        log.info("Holdout done in %.1fs", time.perf_counter() - t0)

    # Write per-fold results
    wfo_df = pd.DataFrame(per_fold_rows)
    wfo_df.to_csv(out_dir / "wfo_results.csv", index=False, lineterminator="\n")

    holdout_df = pd.DataFrame([holdout_row]) if holdout_row is not None else pd.DataFrame()
    holdout_df.to_csv(out_dir / "holdout_results.csv", index=False, lineterminator="\n")

    # Aggregate summary stats
    rois = [r["roi_pct"] for r in per_fold_rows]
    dds = [r["max_dd_pct"] for r in per_fold_rows]
    ratios = [r["roi_dd_ratio"] for r in per_fold_rows]
    ntrades = [r["n_trades"] for r in per_fold_rows]
    worst_idx = int(np.argmin(ratios)) if ratios else -1
    worst_dd_idx = int(np.argmax(dds)) if dds else -1

    # Chained DD across IS folds + holdout (Amendment 3 method)
    chained_equity = stitch_per_fold_oos_equity(
        per_fold_equity + ([holdout_equity] if holdout_equity is not None else []),
        starting_balance=STARTING_BALANCE,
    )
    chained_dd = compute_chained_max_dd_from_continuous_equity(chained_equity)

    # Daily DD across stitched chain (5ers EET boundary)
    per_day_df = compute_per_day_max_dd(
        chained_equity, pair_set=",".join(PAIRS_28),
        boundary_convention="5ers_eet",
    )
    if not per_day_df.empty:
        per_day_df.to_parquet(
            out_dir / "per_day_max_dd_chained.parquet",
            engine="pyarrow", compression="snappy", index=False,
        )

    summary_md: list[str] = []
    summary_md.append("# Step 5 — WFO Summary (Arc 7 r=2%)\n")
    summary_md.append(f"**Config:** `{WINNING_CONFIG_ID}`")
    summary_md.append(f"**r_base:** {R_BASE:.4%}  (canonical: {R_CANONICAL:.4%})")
    summary_md.append(f"**Boundary convention:** 5ers_eet")
    summary_md.append(f"**Window end:** {window_end_str}")
    summary_md.append("")
    summary_md.append("## Per-fold OOS metrics")
    summary_md.append("")
    summary_md.append("| Fold | OOS start | n_trades | ROI % | DD % | Daily 5% breaches | Ratio |")
    summary_md.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in per_fold_rows:
        summary_md.append(
            f"| {r['fold_id']} | {r['oos_start']} | {r['n_trades']} | "
            f"{r['roi_pct']*100:.2f}% | {r['max_dd_pct']*100:.2f}% | "
            f"{r['days_breaching_daily_5pct']} | {r['roi_dd_ratio']:.2f} |"
        )
    if holdout_row is not None:
        summary_md.append(
            f"| H ({holdout_row['fold_id']}) | {holdout_row['oos_start']} | {holdout_row['n_trades']} | "
            f"{holdout_row['roi_pct']*100:.2f}% | {holdout_row['max_dd_pct']*100:.2f}% | "
            f"{holdout_row['days_breaching_daily_5pct']} | {holdout_row['roi_dd_ratio']:.2f} |"
        )
    summary_md.append("")
    summary_md.append("## Aggregate")
    summary_md.append("")
    if ratios:
        summary_md.append(f"- Worst-fold ratio: {min(ratios):.2f}")
        summary_md.append(f"- Mean-fold ratio: {float(np.mean(ratios)):.2f}")
        summary_md.append(f"- Worst-fold ROI %: {min(rois)*100:.2f}% (fold {per_fold_rows[worst_idx]['fold_id']})")
        summary_md.append(f"- Worst-fold DD %: {max(dds)*100:.2f}% (fold {per_fold_rows[worst_dd_idx]['fold_id']})")
        summary_md.append(f"- All folds positive ROI: {all(r > 0 for r in rois)}")
        summary_md.append(f"- Min trades / fold: {min(ntrades)}")
    summary_md.append(f"- Chained max DD (IS + holdout): {chained_dd*100:.2f}%")
    (out_dir / "wfo_summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8", newline="\n")

    summary_payload = {
        "deliverable": "A_wfo_r2pct",
        "config_id": WINNING_CONFIG_ID,
        "r_base": R_BASE,
        "n_is_folds": len(eligible_folds),
        "per_fold": per_fold_rows,
        "holdout": holdout_row,
        "worst_fold_ratio": float(min(ratios)) if ratios else None,
        "mean_fold_ratio": float(np.mean(ratios)) if ratios else None,
        "worst_fold_roi_pct": float(min(rois)) if rois else None,
        "worst_fold_dd_pct": float(max(dds)) if dds else None,
        "min_trades_per_fold": int(min(ntrades)) if ntrades else None,
        "all_folds_positive": all(r > 0 for r in rois) if rois else None,
        "chained_max_dd_pct": float(chained_dd),
        "is_elapsed_seconds": is_elapsed,
    }
    (out_dir / "summary_stats.json").write_text(
        json.dumps(summary_payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )

    artefacts = [
        out_dir / "wfo_results.csv",
        out_dir / "holdout_results.csv",
        out_dir / "wfo_summary.md",
        out_dir / "summary_stats.json",
    ]
    if (out_dir / "per_day_max_dd_chained.parquet").exists():
        artefacts.append(out_dir / "per_day_max_dd_chained.parquet")
    _write_manifest(
        out_dir / "manifest.json", artefacts,
        extra={
            "config_id": WINNING_CONFIG_ID,
            "r_base": R_BASE,
            "boundary_convention": "5ers_eet",
            "window_end": window_end_str,
        },
    )
    return summary_payload


# ──────────────────────────────────────────────────────────────────────
# Deliverable B — single continuous sim 2010-2025 at r=2%
# ──────────────────────────────────────────────────────────────────────


def run_deliverable_b(
    *,
    out_dir: Path,
    panels: Mapping[str, Panel],
    signal_eval: SignalEvaluation,
    base_ctx: A1RunContext,
    s4_result: Step4Result,
    sim_start_str: str,
    sim_end_str: str,
) -> dict:
    """Single contiguous backtest 2010-01-01 .. 2025-12-31 at r=2%.

    Mechanism (dispatch §5.1 Option 1): construct a Fold with
    oos_start = sim_start, oos_end = sim_end (and an empty IS — the
    architecture doesn't refit per-fold for A6). The architecture's
    _slice_panels_to_fold(...) gives the standard 60-day warmup.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_start = date.fromisoformat(sim_start_str)
    sim_end = date.fromisoformat(sim_end_str)
    cont_fold = Fold(
        fold_id=999,
        is_start=sim_start,
        is_end=date(sim_start.year - 1, 12, 31),  # empty IS
        oos_start=sim_start,
        oos_end=sim_end,
    )

    cfg = _build_winning_a6_config(s4_result, risk_pct=R_BASE)
    arch = A6Architecture()

    log.info("Deliverable B: continuous sim %s .. %s at r=%.2f%%",
             sim_start, sim_end, R_BASE * 100)
    t0 = time.perf_counter()
    runner = ArcFoldRunner(
        architecture=arch, signal_evaluation=signal_eval,
        panels=panels, run_context=base_ctx,
    )
    stats = runner(cont_fold, cfg)
    sr = runner.last_result
    elapsed = time.perf_counter() - t0
    log.info("Deliverable B sim done in %.1fs (%.1fmin)", elapsed, elapsed / 60.0)

    if sr is None or len(sr.equity_curve) == 0:
        log.error("Deliverable B produced empty equity curve")
        return {"deliverable": "B_full_sim_r2pct", "status": "empty"}

    # Equity curve — full mark-to-market series from MultiPairBacktester
    equity_full = sr.run_result.equity_curve  # includes pre-OOS warmup
    # Restrict to sim window
    start_ts = pd.Timestamp(sim_start, tz="UTC")
    end_ts = pd.Timestamp(sim_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    equity = equity_full.loc[start_ts:end_ts]
    if len(equity) == 0:
        log.error("Deliverable B sim equity is empty after restricting to window")
        return {"deliverable": "B_full_sim_r2pct", "status": "empty"}

    # Persist equity curve
    eq_df = pd.DataFrame({
        "timestamp_utc": equity.index,
        "equity": equity.values,
    })
    eq_df.to_parquet(out_dir / "equity_curve.parquet", engine="pyarrow",
                     compression="snappy", index=False)

    # Trade ledger
    ledger = _build_trade_ledger(sr.run_result, sl_lookup_atr=SL_MULT)
    if not ledger.empty:
        # Restrict to OOS window via entry_time
        ledger = ledger[(ledger["entry_time"] >= start_ts) & (ledger["entry_time"] <= end_ts)]
    ledger.to_parquet(out_dir / "trade_ledger.parquet", engine="pyarrow",
                      compression="snappy", index=False)

    # Per-day max-DD across full window
    per_day_df = compute_per_day_max_dd(
        equity, pair_set=",".join(PAIRS_28),
        boundary_convention="5ers_eet",
    )
    if not per_day_df.empty:
        per_day_df.to_parquet(out_dir / "per_day_max_dd.parquet",
                              engine="pyarrow", compression="snappy", index=False)

    # ── Stats ─────────────────────────────────────────────────────
    starting_eq = float(equity.iloc[0])
    ending_eq = float(equity.iloc[-1])
    total_ret_pct = (ending_eq / starting_eq) - 1.0
    max_dd = compute_chained_max_dd_from_continuous_equity(equity)
    max_dd_dur_days = _max_dd_duration_days(equity)
    sharpe = _annualized_sharpe(equity)
    n_trades = int(len(ledger))
    # Whole-position closes only (exclude partial-leg duplicates)
    whole_closes = ledger[ledger["parent_position_id"].isna()] if not ledger.empty else ledger
    wins = whole_closes[whole_closes["pnl"] > 0]
    win_rate = float(len(wins) / len(whole_closes)) if len(whole_closes) > 0 else 0.0
    avg_r = float(whole_closes["r_at_close"].mean()) if len(whole_closes) > 0 else 0.0

    worst_day_dd = float(per_day_df["day_max_dd_base_pct"].max()) if not per_day_df.empty else 0.0
    daily_breaches = int((per_day_df["day_max_dd_base_pct"] > 0.05).sum()) if not per_day_df.empty else 0
    worst_30d_dd = _worst_rolling_window_dd(equity, window_days=30)
    top_dds = _top_n_drawdown_periods(equity, n=5)
    neg_years = _years_with_negative_return(equity)

    # Survival check
    max_dd_breached = bool(max_dd >= 0.10)
    daily_dd_breached = bool(daily_breaches > 0)
    survived = not (max_dd_breached or daily_dd_breached)
    margin_to_max_dd = float(0.10 - max_dd)
    margin_to_daily_dd = float(0.05 - worst_day_dd)

    summary_payload = {
        "deliverable": "B_full_sim_r2pct",
        "config_id": WINNING_CONFIG_ID,
        "r_base": R_BASE,
        "sim_mechanism": "single_fold_extended_oos",
        "sim_start": sim_start_str,
        "sim_end": sim_end_str,
        "boundary_convention": "5ers_eet",
        "starting_equity": starting_eq,
        "ending_equity": ending_eq,
        "total_return_pct": float(total_ret_pct),
        "max_drawdown_pct": float(max_dd),
        "max_dd_duration_days": int(max_dd_dur_days),
        "sharpe_annualized": float(sharpe),
        "n_trades": n_trades,
        "n_whole_closes": int(len(whole_closes)),
        "win_rate": win_rate,
        "average_r_per_trade": avg_r,
        "worst_single_day_dd_pct": worst_day_dd,
        "daily_breaches_5pct": daily_breaches,
        "worst_rolling_30day_dd_pct": float(worst_30d_dd),
        "top_5_drawdowns": top_dds,
        "years_with_negative_return": neg_years,
        "fivers_survival": {
            "limits": {"max_dd_pct": 0.10, "daily_dd_pct": 0.05},
            "max_dd_breached": max_dd_breached,
            "daily_dd_breached": daily_dd_breached,
            "survived": survived,
            "margin_to_max_dd_pct": margin_to_max_dd,
            "margin_to_daily_dd_pct": margin_to_daily_dd,
        },
        "wall_time_seconds": elapsed,
    }
    (out_dir / "summary_stats.json").write_text(
        json.dumps(summary_payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )

    artefacts = [
        out_dir / "equity_curve.parquet",
        out_dir / "trade_ledger.parquet",
        out_dir / "summary_stats.json",
    ]
    if (out_dir / "per_day_max_dd.parquet").exists():
        artefacts.append(out_dir / "per_day_max_dd.parquet")
    _write_manifest(
        out_dir / "manifest.json", artefacts,
        extra={
            "config_id": WINNING_CONFIG_ID,
            "r_base": R_BASE,
            "boundary_convention": "5ers_eet",
            "sim_start": sim_start_str,
            "sim_end": sim_end_str,
            "sim_mechanism": "single_fold_extended_oos",
        },
    )
    return summary_payload


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histdata-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path,
                        default=Path("results/analysis/arc_7_r2pct"))
    parser.add_argument("--v301-root", type=Path, default=Path("results/l_arc_7"))
    parser.add_argument("--deliverable", choices=("A", "B", "both"), default="both")
    parser.add_argument("--window-end", type=str, default="2026-04-30",
                        help="WFO holdout end (Deliverable A) and continuous sim end (Deliverable B). "
                             "Default matches the canonical orchestrator (scripts/l_arc_7_v3_0_2/run.py); see "
                             "docs/dispatches/driver_script_divergence_investigation.md.")
    parser.add_argument("--sim-start", type=str, default="2010-01-01",
                        help="Continuous sim start (Deliverable B)")
    parser.add_argument("--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    seed_everything(42)
    t_start = time.perf_counter()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("arc_7_r2pct rerun — out=%s, v301_root=%s, deliverable=%s",
             out_root, args.v301_root, args.deliverable)

    # ── Load v3.0.1 Step 1-4 artefacts ──────────────────────────────
    log.info("Loading v3.0.1 Step 1-4 artefacts ...")
    t0 = time.perf_counter()
    v301 = _load_v301(args.v301_root.resolve())
    log.info("Loaded %d trades, %d candidate clusters, %d classifiers (%.1fs)",
             len(v301.pool_trades), len(v301.candidate_clusters),
             len(v301.step4_per_cluster), time.perf_counter() - t0)

    s4_result = Step4Result(
        per_cluster=tuple(v301.step4_per_cluster),
        extraction_metrics=v301.extraction_metrics,
        feature_importance=v301.feature_importance,
        summary_md="(reused from v3.0.1)",
    )

    # ── Build panels ────────────────────────────────────────────────
    log.info("Building H4/D1/W1 panels (5ers_eet, cache=%s)", args.cache_root)
    t0 = time.perf_counter()
    panel_h4 = _build_panel_5ers_eet(list(PAIRS_28), "H4",
                                     histdata_root=args.histdata_root,
                                     cache_root=args.cache_root)
    panel_d1 = _build_panel_5ers_eet(list(PAIRS_28), "D1",
                                     histdata_root=args.histdata_root,
                                     cache_root=args.cache_root)
    panel_w1 = _build_panel_5ers_eet(list(PAIRS_28), "W1",
                                     histdata_root=args.histdata_root,
                                     cache_root=args.cache_root)
    object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})
    panels: dict[str, Panel] = {"H4": panel_h4, "D1": panel_d1, "W1": panel_w1}
    log.info("Panels built in %.1fs", time.perf_counter() - t0)

    # ── Re-evaluate signal ──────────────────────────────────────────
    log.info("Re-evaluating signal on panels ...")
    t0 = time.perf_counter()
    signal_module = LiquiditySweepReclaimLongSignal()
    signal_eval = signal_module.evaluate(panels)
    log.info("Signal evaluation in %.1fs", time.perf_counter() - t0)

    # ── Per-trade features for A6 admit gate ────────────────────────
    log.info("Building per-trade feature lookup for A6 admit gate ...")
    t0 = time.perf_counter()
    feature_order = v301.step4_per_cluster[0].fitted_classifier_feature_order if v301.step4_per_cluster else ()
    per_trade_features = _build_per_trade_features(
        v301.pool_trades, panel_h4, feature_order,
    )
    log.info("Per-trade features built (%d trades) in %.1fs",
             len(per_trade_features), time.perf_counter() - t0)
    base_ctx = A1RunContext(per_trade_features=per_trade_features)

    # ── Run deliverables ────────────────────────────────────────────
    deliverable_a_summary: dict | None = None
    deliverable_b_summary: dict | None = None

    if args.deliverable in ("A", "both"):
        log.info("=" * 60)
        log.info("DELIVERABLE A — Canonical WFO at r=%.2f%%", R_BASE * 100)
        log.info("=" * 60)
        deliverable_a_summary = run_deliverable_a(
            out_dir=out_root / "wfo",
            panels=panels, signal_eval=signal_eval, base_ctx=base_ctx,
            s4_result=s4_result, window_end_str=args.window_end,
        )

    if args.deliverable in ("B", "both"):
        log.info("=" * 60)
        log.info("DELIVERABLE B — Continuous sim %s..%s at r=%.2f%%",
                 args.sim_start, args.window_end, R_BASE * 100)
        log.info("=" * 60)
        deliverable_b_summary = run_deliverable_b(
            out_dir=out_root / "full_sim",
            panels=panels, signal_eval=signal_eval, base_ctx=base_ctx,
            s4_result=s4_result,
            sim_start_str=args.sim_start, sim_end_str=args.window_end,
        )

    # ── Combined run summary ────────────────────────────────────────
    run_summary = {
        "run_name": "arc_7_r2pct_rerun",
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_time_seconds": time.perf_counter() - t_start,
        "config_id": WINNING_CONFIG_ID,
        "cluster_id": CLUSTER_ID,
        "r_base": R_BASE,
        "r_canonical": R_CANONICAL,
        "boundary_convention": "5ers_eet",
        "v301_pool_size": int(len(v301.pool_trades)),
        "v301_classifier_path": str(v301.step4_per_cluster[CLUSTER_ID].fitted_classifier_path)
            if v301.step4_per_cluster else None,
        "deliverable_a": deliverable_a_summary,
        "deliverable_b": deliverable_b_summary,
    }
    (out_root / "run_summary.json").write_text(
        json.dumps(run_summary, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )

    log.info("arc_7_r2pct rerun done in %.1fs (%.1fmin)",
             time.perf_counter() - t_start,
             (time.perf_counter() - t_start) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
