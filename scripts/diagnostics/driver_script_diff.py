"""Driver-script divergence diagnostic.

Runs the Arc 7 holdout fold via TWO paths side-by-side:

  * **V302**: uses helpers from ``scripts/l_arc_7_v3_0_2/run.py``
    (the canonical Arc 7 v3.0.2 driver). Imports the same
    ``_build_panel_5ers_eet``, ``_build_per_trade_features_for_a2_a6``,
    ``_load_v301``, ``_build_a6_configs`` byte-for-byte.

  * **ANALYSIS**: uses helpers from
    ``scripts/analysis/arc_7_r2pct_rerun.py`` (the standalone analysis
    re-run driver). Imports the same ``_build_panel_5ers_eet``,
    ``_build_per_trade_features``, ``_load_v301``,
    ``_build_winning_a6_config``.

Both run the SAME (engine, data, winning config, risk=0.005, window).
If the trade counts differ, the difference is purely in the driver's
own construction of inputs to ``ArcFoldRunner``.

Outputs:
  * Per-driver closed-trade ledger as parquet.
  * Per-driver admit-attempts via the same instrumentation harness
    used by ``risk_leak_diagnosis.py``.
  * Side-by-side comparison summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.architectures.a1_system_level_filter import A1RunContext  # noqa: E402
from core.architectures.a6_meta_labeling import A6Architecture  # noqa: E402
from core.determinism import seed_everything  # noqa: E402
from core.runners.arc_fold_runner import ArcFoldRunner  # noqa: E402
from core.steps.classifier_persistence import build_a6_config_from_step4  # noqa: E402
from core.steps.step_4_extraction import Step4Result  # noqa: E402
from core.strategies.liquidity_sweep_reclaim_long.signal_module import (  # noqa: E402
    LiquiditySweepReclaimLongSignal,
)
from core.wfo.folds import build_v3_folds  # noqa: E402

# ANALYSIS driver helpers
from scripts.analysis.arc_7_r2pct_rerun import (  # noqa: E402
    CLUSTER_ID,
    LOWER_THR,
    MAX_PER_CURRENCY,
    SL_MULT,
    STARTING_BALANCE,
    UPPER_THR,
    WINNING_CONFIG_ID,
)
from scripts.analysis.arc_7_r2pct_rerun import (  # noqa: E402
    PAIRS_28 as ANALYSIS_PAIRS,
)
from scripts.analysis.arc_7_r2pct_rerun import (  # noqa: E402
    _build_panel_5ers_eet as analysis_build_panel,
)
from scripts.analysis.arc_7_r2pct_rerun import (  # noqa: E402
    _build_per_trade_features as analysis_build_features,
)
from scripts.analysis.arc_7_r2pct_rerun import (  # noqa: E402
    _load_v301 as analysis_load_v301,
)

# V302 driver helpers
from scripts.l_arc_7_v3_0_2.run import (  # noqa: E402
    PAIRS_28 as V302_PAIRS,
)
from scripts.l_arc_7_v3_0_2.run import (  # noqa: E402
    _build_a6_configs as v302_build_a6_configs,
)
from scripts.l_arc_7_v3_0_2.run import (  # noqa: E402
    _build_panel_5ers_eet as v302_build_panel,
)
from scripts.l_arc_7_v3_0_2.run import (  # noqa: E402
    _build_per_trade_features_for_a2_a6 as v302_build_features,
)
from scripts.l_arc_7_v3_0_2.run import (  # noqa: E402
    _load_v301 as v302_load_v301,
)

log = logging.getLogger("driver_script_diff")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _trade_ledger(closed_trades) -> pd.DataFrame:
    rows = []
    for t in closed_trades:
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
            "sl_price": float(t.sl_price) if t.sl_price is not None else float("nan"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_v302_path(*, histdata_root: Path, cache_root: Path, v301_root: Path,
                  window_end: str, out_dir: Path) -> dict:
    """Reproduce the v3.0.2 driver's setup, run holdout for the winning A6 config."""
    log.info("[V302] Loading v3.0.1 artefacts")
    v301 = v302_load_v301(v301_root)
    s4_result = Step4Result(
        per_cluster=tuple(v301.step4_per_cluster),
        extraction_metrics=v301.extraction_metrics,
        feature_importance=v301.feature_importance,
        summary_md="(reused)",
    )

    log.info("[V302] Building panels")
    panel_h4 = v302_build_panel(list(V302_PAIRS), "H4",
                                histdata_root=histdata_root, cache_root=cache_root)
    panel_d1 = v302_build_panel(list(V302_PAIRS), "D1",
                                histdata_root=histdata_root, cache_root=cache_root)
    panel_w1 = v302_build_panel(list(V302_PAIRS), "W1",
                                histdata_root=histdata_root, cache_root=cache_root)
    object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})
    panels = {"H4": panel_h4, "D1": panel_d1, "W1": panel_w1}

    log.info("[V302] Evaluating signal")
    signal_module = LiquiditySweepReclaimLongSignal()
    signal_eval = signal_module.evaluate(panels)

    log.info("[V302] Building per-trade features (v3.0.2-style)")
    per_trade_features = v302_build_features(
        pool_trades=v301.pool_trades,
        signal_eval=signal_eval,
        panel_h4=panel_h4,
        feature_order=v301.step4_per_cluster[0].fitted_classifier_feature_order
        if v301.step4_per_cluster else (),
    )
    base_ctx = A1RunContext(per_trade_features=per_trade_features)

    # Build the winning A6 config via v3.0.2 driver's _build_a6_configs
    log.info("[V302] Building A6 configs via _build_a6_configs")
    # Find the candidate cluster + its selected_sl
    target_cluster = None
    for cc in v301.candidate_clusters:
        if cc.cluster_id == CLUSTER_ID:
            target_cluster = cc
            break
    if target_cluster is None:
        raise RuntimeError(f"Candidate cluster {CLUSTER_ID} not found")

    a6_configs = v302_build_a6_configs(
        cid=target_cluster.cluster_id,
        selected_sl=target_cluster.selected_sl,
        s4=s4_result,
    )
    # Pick the winning config: A6::cl1::sl2.0::thr0.5-0.7::exp2
    target_cid_suffix = "A6::cl1::sl2.0::thr0.5-0.7::exp2"
    target_pair = None
    for cid_full, paired in a6_configs:
        if cid_full == f"A6::{target_cid_suffix}":
            target_pair = paired
            break
    if target_pair is None:
        available = [cid_full for cid_full, _ in a6_configs]
        raise RuntimeError(
            f"Winning A6 config not found. Available: {available}"
        )
    arch, cfg = target_pair
    log.info("[V302] Selected config: config_id=%s, exit_policy=%s, "
             "risk_pct=%s, exp_per_cur=%s",
             cfg.config_id, cfg.exit_policy, cfg.risk_pct,
             cfg.max_concurrent_per_currency)

    # Run holdout
    wfo_struct = build_v3_folds(holdout_end=date.fromisoformat(window_end))
    log.info("[V302] Running holdout %s..%s",
             wfo_struct.holdout.oos_start, wfo_struct.holdout.oos_end)
    t0 = time.perf_counter()
    seed_everything(42)
    runner = ArcFoldRunner(
        architecture=arch, signal_evaluation=signal_eval,
        panels=panels, run_context=base_ctx,
    )
    stats = runner(wfo_struct.holdout, cfg)
    elapsed = time.perf_counter() - t0
    log.info("[V302] Done in %.1fs: n_trades=%d roi=%.4f dd=%.4f",
             elapsed, stats.n_trades, stats.roi_pct, stats.max_dd_pct)

    # Persist ledger
    sr = runner.last_result
    ledger = _trade_ledger(sr.closed_trades) if sr is not None else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ledger.empty:
        ledger.to_parquet(out_dir / "v302_holdout_ledger.parquet",
                          engine="pyarrow", index=False)

    return {
        "n_trades": int(stats.n_trades),
        "roi_pct": float(stats.roi_pct),
        "max_dd_pct": float(stats.max_dd_pct),
        "n_admit_targets_in_pool": len(per_trade_features),
        "config_id": cfg.config_id,
        "config_starting_balance": float(cfg.starting_balance),
        "config_exit_policy": cfg.exit_policy,
        "config_max_concurrent_per_currency": cfg.max_concurrent_per_currency,
        "elapsed_seconds": elapsed,
    }


def run_analysis_path(*, histdata_root: Path, cache_root: Path, v301_root: Path,
                      window_end: str, out_dir: Path) -> dict:
    """Reproduce the analysis-script driver's setup, run holdout."""
    log.info("[ANALYSIS] Loading v3.0.1 artefacts")
    v301 = analysis_load_v301(v301_root)
    s4_result = Step4Result(
        per_cluster=tuple(v301.step4_per_cluster),
        extraction_metrics=v301.extraction_metrics,
        feature_importance=v301.feature_importance,
        summary_md="(reused)",
    )

    log.info("[ANALYSIS] Building panels")
    panel_h4 = analysis_build_panel(list(ANALYSIS_PAIRS), "H4",
                                    histdata_root=histdata_root, cache_root=cache_root)
    panel_d1 = analysis_build_panel(list(ANALYSIS_PAIRS), "D1",
                                    histdata_root=histdata_root, cache_root=cache_root)
    panel_w1 = analysis_build_panel(list(ANALYSIS_PAIRS), "W1",
                                    histdata_root=histdata_root, cache_root=cache_root)
    object.__setattr__(panel_h4, "aux", {"d1": panel_d1, "w1": panel_w1})
    panels = {"H4": panel_h4, "D1": panel_d1, "W1": panel_w1}

    log.info("[ANALYSIS] Evaluating signal")
    signal_module = LiquiditySweepReclaimLongSignal()
    signal_eval = signal_module.evaluate(panels)

    log.info("[ANALYSIS] Building per-trade features (analysis-style)")
    feature_order = v301.step4_per_cluster[0].fitted_classifier_feature_order \
        if v301.step4_per_cluster else ()
    per_trade_features = analysis_build_features(
        v301.pool_trades, panel_h4, feature_order,
    )
    base_ctx = A1RunContext(per_trade_features=per_trade_features)

    # Build winning config the analysis-script way
    cfg = build_a6_config_from_step4(
        s4_result, cluster_id=CLUSTER_ID,
        lower_threshold=LOWER_THR, upper_threshold=UPPER_THR,
        config_id=WINNING_CONFIG_ID,
        sl_atr_mult=SL_MULT, trail_enabled=True,
        risk_pct=0.005, max_concurrent_per_currency=MAX_PER_CURRENCY,
        max_concurrent_per_pair=1, max_concurrent_total=None,
        starting_balance=STARTING_BALANCE,
        exit_policy=None,
    )
    log.info("[ANALYSIS] config_id=%s exit_policy=%s risk_pct=%s exp_per_cur=%s",
             cfg.config_id, cfg.exit_policy, cfg.risk_pct,
             cfg.max_concurrent_per_currency)

    wfo_struct = build_v3_folds(holdout_end=date.fromisoformat(window_end))
    log.info("[ANALYSIS] Running holdout %s..%s",
             wfo_struct.holdout.oos_start, wfo_struct.holdout.oos_end)
    t0 = time.perf_counter()
    seed_everything(42)
    arch = A6Architecture()
    runner = ArcFoldRunner(
        architecture=arch, signal_evaluation=signal_eval,
        panels=panels, run_context=base_ctx,
    )
    stats = runner(wfo_struct.holdout, cfg)
    elapsed = time.perf_counter() - t0
    log.info("[ANALYSIS] Done in %.1fs: n_trades=%d roi=%.4f dd=%.4f",
             elapsed, stats.n_trades, stats.roi_pct, stats.max_dd_pct)

    sr = runner.last_result
    ledger = _trade_ledger(sr.closed_trades) if sr is not None else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ledger.empty:
        ledger.to_parquet(out_dir / "analysis_holdout_ledger.parquet",
                          engine="pyarrow", index=False)

    return {
        "n_trades": int(stats.n_trades),
        "roi_pct": float(stats.roi_pct),
        "max_dd_pct": float(stats.max_dd_pct),
        "n_admit_targets_in_pool": len(per_trade_features),
        "config_id": cfg.config_id,
        "config_starting_balance": float(cfg.starting_balance),
        "config_exit_policy": cfg.exit_policy,
        "config_max_concurrent_per_currency": cfg.max_concurrent_per_currency,
        "elapsed_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histdata-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--v301-root", type=Path, default=Path("results/l_arc_7"))
    parser.add_argument("--window-end", type=str, default="2026-04-30")
    parser.add_argument("--out-root", type=Path,
                        default=Path("results/analysis/driver_script_diff"))
    args = parser.parse_args(argv)
    _setup_logging()
    seed_everything(42)

    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    v302_result = run_v302_path(
        histdata_root=args.histdata_root,
        cache_root=args.cache_root,
        v301_root=args.v301_root.resolve(),
        window_end=args.window_end,
        out_dir=out / "v302",
    )
    analysis_result = run_analysis_path(
        histdata_root=args.histdata_root,
        cache_root=args.cache_root,
        v301_root=args.v301_root.resolve(),
        window_end=args.window_end,
        out_dir=out / "analysis",
    )

    log.info("=" * 60)
    log.info("SIDE-BY-SIDE")
    log.info("=" * 60)
    log.info(" V302:     %s", v302_result)
    log.info(" ANALYSIS: %s", analysis_result)
    log.info(" delta_n_trades = V302 - ANALYSIS = %d",
             v302_result["n_trades"] - analysis_result["n_trades"])

    import json
    (out / "summary.json").write_text(json.dumps({
        "v302": v302_result, "analysis": analysis_result,
        "delta_n_trades": v302_result["n_trades"] - analysis_result["n_trades"],
        "window_end": args.window_end,
    }, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
