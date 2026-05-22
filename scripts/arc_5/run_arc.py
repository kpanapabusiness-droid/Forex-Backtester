"""Arc 5 (v3.0) — master runner.

Executes L_PROTOCOL v3.0 Steps 1 → 5 end-to-end on the
mtf_alignment.2_down_mixed.kijun.h_120 signal. Vanilla sub-protocol.

Outputs land under ``results/l_arc_5/`` per L_PROTOCOL §6.

Usage:
    py scripts/arc_5/run_arc.py
"""

from __future__ import annotations

import io
import json
import sys
import time
import traceback

# Force UTF-8 on stdout/stderr for Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from datetime import date, datetime, timezone
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
from core.architectures.a1_system_level_filter import A1Architecture, A1Config  # noqa: E402
from core.determinism import seed_everything, write_text_deterministic  # noqa: E402
from core.features.pipeline import compute_feature_matrix  # noqa: E402
from core.runners.arc_fold_runner import ArcFoldRunner  # noqa: E402
from core.signals.mtf_alignment_2_down_mixed_kijun import (  # noqa: E402
    MtfAlignment2DownMixedKijunSignal,
)
from core.sim.panel import Panel  # noqa: E402
from core.steps.step_2_clustering import run_step_2  # noqa: E402
from core.steps.step_3_capturability import run_step_3  # noqa: E402
from core.steps.step_4_extraction import run_step_4  # noqa: E402
from core.wfo.folds import Fold, build_v3_folds  # noqa: E402
from core.wfo.orchestrator import run_holdout, run_search  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
ARC_NAME = "l_arc_5"
RESULTS_DIR = REPO_ROOT / "results" / "l_arc_5"
HISTDATA_ROOT = Path("C:/Users/panap/Documents/Forex-Backtester/data/histdata")
CACHE_ROOT = Path("C:/Users/panap/Documents/Forex-Backtester/data/cache")
WINDOW_START = date(2010, 1, 1)
WINDOW_END = date(2026, 4, 30)
PAIRS: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)
SL_AT_STEP1 = 2.0
HOLD_BARS = 120
RISK_PCT = 0.005
STARTING_BALANCE = 100_000.0
RANDOM_STATE = 42


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


# ──────────────────────────────────────────────────────────────────────
# Panel + aux wrapper for multi_tf features
# ──────────────────────────────────────────────────────────────────────


class PanelWithAux:
    """Duck-typed Panel wrapper exposing ``aux`` for multi_tf features.

    The v3 Panel is a frozen dataclass without ``aux`` but
    ``core.features.multi_tf`` looks up ``panel.aux['d1']`` / ['w1'].
    """

    def __init__(self, base: Panel, aux: dict[str, Panel]) -> None:
        self._base = base
        self.aux = aux

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _build_panels() -> dict[str, Panel]:
    panels: dict[str, Panel] = {}
    for tf in ("H1", "H4", "D1", "W1"):
        _log(f"[SETUP] building {tf} panel (28 pairs)")
        panels[tf] = Panel.from_pairs(
            list(PAIRS), tf,
            histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
        )
    return panels


# ──────────────────────────────────────────────────────────────────────
# Step 1
# ──────────────────────────────────────────────────────────────────────


def step_1(panels: dict[str, Panel]) -> ArcPool:
    pool_path = RESULTS_DIR / "step_1" / "pool.parquet"
    paths_path = RESULTS_DIR / "step_1" / "paths.parquet"
    manifest_path = RESULTS_DIR / "step_1" / "manifest.json"
    signal_module = MtfAlignment2DownMixedKijunSignal()
    if pool_path.exists() and paths_path.exists() and manifest_path.exists():
        _log("[STEP 1] reusing existing pool.parquet (idempotent skip)")
        trades = pd.read_parquet(pool_path)
        paths = pd.read_parquet(paths_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Re-evaluate signal for downstream Step 5 consumers (only carries Series state)
        evaluation = signal_module.evaluate(panels)
        return ArcPool(
            trades=trades, paths=paths, integrity=(),
            signal_evaluation=evaluation,
            pool_sha256=manifest.get("pool_sha256", ""),
            manifest={"signal_name": evaluation.signal_name, "primary_tf": evaluation.primary_tf},
        )
    _log("[STEP 1] building signal + pool")
    cfg = ArcPoolConfig(
        arc_name=ARC_NAME,
        sl_atr_mult=SL_AT_STEP1,
        hold_bars=HOLD_BARS,
        risk_pct=RISK_PCT,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )
    pool = build_arc_pool(signal_module, panels, cfg)
    _log(f"[STEP 1] pool: {len(pool.trades):,} trades, sha256={pool.pool_sha256[:12]}")
    write_arc_pool(pool, RESULTS_DIR)
    return pool


# ──────────────────────────────────────────────────────────────────────
# Feature matrix at signal-bar timestamps (for Step 4)
# ──────────────────────────────────────────────────────────────────────


def compute_signal_features(panels: dict[str, Panel], pool: ArcPool) -> tuple[pd.DataFrame, pd.DataFrame]:
    _log("[FEATURES] computing 27-feature matrix per signal bar")
    panel_h1 = panels["H1"]
    panel_with_aux = PanelWithAux(panel_h1, aux={"d1": panels["D1"], "w1": panels["W1"]})

    trades = pool.trades.sort_values(["pair", "signal_time"]).reset_index(drop=True)
    per_pair_feats: list[pd.DataFrame] = []
    lineage_df = None
    for pair in sorted(trades["pair"].unique()):
        h1_df = panel_h1.pair_dfs[pair]
        fm = compute_feature_matrix(pair, h1_df, panel=panel_with_aux)
        if lineage_df is None:
            lineage_df = fm.lineage
        sig_times = trades.loc[trades["pair"] == pair, "signal_time"].values
        sub = fm.matrix.reindex(pd.DatetimeIndex(sig_times, tz="UTC")).copy()
        sub.insert(0, "pair", pair)
        sub.insert(1, "signal_time", sub.index)
        per_pair_feats.append(sub.reset_index(drop=True))
    feats_long = pd.concat(per_pair_feats, ignore_index=True)
    feats_long["signal_time"] = pd.to_datetime(feats_long["signal_time"], utc=True)
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
    # Align to trade order (one row per trade_id)
    merged = trades[["trade_id", "pair", "signal_time"]].merge(
        feats_long, on=["pair", "signal_time"], how="left"
    )
    feature_cols = [c for c in merged.columns if c not in ("trade_id", "pair", "signal_time")]
    feature_matrix = merged[["trade_id", *feature_cols]].set_index("trade_id")
    # Pipeline emits 'lineage' column; Step 4 expects 'causal_lineage'. Rename.
    if lineage_df is not None and "lineage" in lineage_df.columns and "causal_lineage" not in lineage_df.columns:
        lineage_df = lineage_df.rename(columns={"lineage": "causal_lineage"})
    counts = lineage_df["causal_lineage"].value_counts().to_dict() if lineage_df is not None else {}
    _log(f"[FEATURES] matrix shape: {feature_matrix.shape}; lineage counts: {counts}")
    return feature_matrix, lineage_df


# ──────────────────────────────────────────────────────────────────────
# Step 5 — A1 config sweep
# ──────────────────────────────────────────────────────────────────────


def build_a1_configs() -> list[tuple[A1Architecture, A1Config]]:
    """A1 search grid: SL × trail × exposure-cap.

    SL: {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} — 6
    Trail: {enabled, disabled} — 2
    Exposure per-currency: {2, None (unlimited)} — 2
    → 24 configs (thin search; dispatch §"Selection-bias accounting" thin < 50).
    """
    arch = A1Architecture()
    configs: list[tuple[A1Architecture, A1Config]] = []
    for sl in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        for trail in (True, False):
            for exp_cap in (2, None):
                cid = f"A1_sl{sl:.1f}_trail{'ON' if trail else 'OFF'}_exp{'2' if exp_cap == 2 else 'unl'}"
                cfg = A1Config(
                    config_id=cid,
                    sl_atr_mult=sl,
                    trail_enabled=trail,
                    risk_pct=RISK_PCT,
                    starting_balance=STARTING_BALANCE,
                    max_concurrent_per_currency=exp_cap,
                    max_concurrent_per_pair=1,
                )
                configs.append((arch, cfg))
    return configs


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    seed_everything(RANDOM_STATE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPO_ROOT / "logs" / "arc_5_runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Arc 5 v3.0 — runner start. results_dir={RESULTS_DIR}")
    t0 = time.time()

    try:
        panels = _build_panels()
        pool = step_1(panels)
        if len(pool.trades) < 200:
            _log(f"HALT: pool size {len(pool.trades)} < 200 threshold")
            return 2

        # Feature matrix for Step 4
        feature_matrix, lineage_df = compute_signal_features(panels, pool)

        # Step 2 — clustering
        _log("[STEP 2] clustering K∈{2..6}")
        s2 = run_step_2(pool.trades, pool.paths)
        s2_dir = RESULTS_DIR / "step_2"
        s2_dir.mkdir(exist_ok=True)
        s2.cluster_assignments.to_parquet(
            s2_dir / "cluster_assignments.parquet", engine="pyarrow", compression="snappy", index=False,
        )
        s2.cluster_summary.to_csv(s2_dir / "cluster_summary.csv", index=False, lineterminator="\n")
        write_text_deterministic(s2_dir / "cluster_summary.md", s2.summary_md)
        _log(f"[STEP 2] K={s2.k_selected}, silhouette={s2.silhouette_per_k[s2.k_selected]:.4f}")

        # Step 3 — capturability
        _log("[STEP 3] capturability + SL sweep")
        s3 = run_step_3(
            pool.trades, pool.paths, s2.cluster_assignments,
            declared_sl_mult=SL_AT_STEP1, cluster_centroids=s2.centroids,
        )
        s3_dir = RESULTS_DIR / "step_3"
        s3_dir.mkdir(exist_ok=True)
        s3.capturability_csv.to_csv(s3_dir / "capturability.csv", index=False, lineterminator="\n")
        write_text_deterministic(s3_dir / "capturability_summary.md", s3.summary_md)
        cand_ids = tuple(c.cluster_id for c in s3.per_cluster if c.is_candidate)
        _log(f"[STEP 3] candidate clusters: {list(cand_ids)}")

        # Step 4 — extraction (only if candidates)
        s4 = None
        if cand_ids:
            _log(f"[STEP 4] training classifiers per candidate cluster ({len(cand_ids)})")
            s4 = run_step_4(
                pool.trades, feature_matrix, s2.cluster_assignments,
                feature_lineage=lineage_df, candidate_cluster_ids=cand_ids,
            )
            s4_dir = RESULTS_DIR / "step_4"
            s4_dir.mkdir(exist_ok=True)
            s4.extraction_metrics.to_csv(s4_dir / "extraction_metrics.csv", index=False, lineterminator="\n")
            s4.feature_importance.to_csv(s4_dir / "feature_importance.csv", index=False, lineterminator="\n")
            write_text_deterministic(s4_dir / "extraction_summary.md", s4.summary_md)
            _log(f"[STEP 4] complete: {len(s4.per_cluster)} candidates evaluated")
        else:
            _log("[STEP 4] skipped — no candidate clusters from Step 3")

        # Step 5 — A1 architecture search via WFO orchestrator
        _log("[STEP 5] building A1 config grid")
        a1_configs = build_a1_configs()
        candidates: list[tuple[str, tuple[A1Architecture, A1Config]]] = []
        for arch, conf in a1_configs:
            candidates.append((conf.config_id, (arch, conf)))
        total_configs = len(candidates)
        _log(f"[STEP 5] {total_configs} configs (thin search)")

        signal_eval = pool.signal_evaluation
        # ArcFoldRunner needs to be created with the architecture + signal_eval + panels.
        # We can't reuse one ArcFoldRunner across architectures, but since all configs
        # use the same A1Architecture instance, one runner per (arch, conf) call is fine.
        def fold_runner(fold: Fold, paired: tuple[A1Architecture, A1Config]):
            arch, conf = paired
            r = ArcFoldRunner(
                architecture=arch, signal_evaluation=signal_eval, panels=panels,
            )
            return r(fold, conf)

        wfo_struct = build_v3_folds(holdout_end=WINDOW_END)
        _log(f"[STEP 5] WFO structure: {wfo_struct.n_folds} folds, holdout {wfo_struct.holdout.oos_start}..{wfo_struct.holdout.oos_end}")
        _log(f"[STEP 5] running search ({total_configs} configs × {wfo_struct.n_folds} folds; min_is_days=365)")
        search_result = run_search(
            wfo_struct, candidates, fold_runner=fold_runner,
            min_is_days=365, top_k=3,
        )
        _log(f"[STEP 5] search complete: {search_result.n_candidates_evaluated} candidates evaluated; top-3 to holdout")

        # Holdout
        _log("[STEP 5] running holdout one-shot for top-3")
        holdout_results = run_holdout(wfo_struct, search_result.top_k, fold_runner=fold_runner)
        _log(f"[STEP 5] holdout complete")

        # Persist Step 5 artefacts
        s5_dir = RESULTS_DIR / "step_5"
        s5_dir.mkdir(exist_ok=True)

        # wfo_results.csv — per-(config, fold) rows
        wfo_rows: list[dict[str, Any]] = []
        for cand in search_result.candidates:
            for fs in cand.fold_stats:
                wfo_rows.append({
                    "config_id": cand.config_id, "fold_id": fs.fold_id,
                    "n_trades": fs.n_trades, "roi_pct": fs.roi_pct,
                    "max_dd_pct": fs.max_dd_pct, "roi_dd_ratio": fs.roi_dd_ratio,
                    "days_breaching_daily_5pct": fs.days_breaching_daily_5pct,
                })
        pd.DataFrame(wfo_rows).to_csv(s5_dir / "wfo_results.csv", index=False, lineterminator="\n")

        # architectures_ranked.md
        ranked = sorted(
            search_result.candidates,
            key=lambda r: (r.gate.worst_fold_ratio, r.gate.mean_fold_ratio), reverse=True,
        )
        rank_lines = [
            "# Arc 5 v3.0 — Step 5 Architectures Ranked",
            f"Generated: {_now_iso()}",
            f"Total configs: {total_configs} (selection-bias flag: thin search, <50)",
            "",
            "| Rank | Config | Verdict | Worst ratio | Mean ratio | Worst ROI | Worst DD | Min trades |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        for i, c in enumerate(ranked, start=1):
            g = c.gate
            rank_lines.append(
                f"| {i} | {c.config_id} | {g.verdict.value} | {g.worst_fold_ratio:.4f} | "
                f"{g.mean_fold_ratio:.4f} | {g.worst_fold_roi:+.4%} | {g.worst_fold_dd:.4%} | {g.min_trades_per_fold} |"
            )
        write_text_deterministic(s5_dir / "architectures_ranked.md", "\n".join(rank_lines))

        # holdout summary
        hold_rows: list[dict[str, Any]] = []
        for h in holdout_results:
            hold_rows.append({
                "config_id": h.config_id, "search_verdict": h.search_gate.verdict.value,
                "search_worst_ratio": h.search_gate.worst_fold_ratio,
                "search_mean_ratio": h.search_gate.mean_fold_ratio,
                "search_worst_roi": h.search_gate.worst_fold_roi,
                "search_worst_dd": h.search_gate.worst_fold_dd,
                "search_negative_folds": h.search_gate.n_negative_folds,
                "holdout_n_trades": h.holdout_stats.n_trades,
                "holdout_roi": h.holdout_stats.roi_pct,
                "holdout_dd": h.holdout_stats.max_dd_pct,
                "holdout_ratio": h.holdout_stats.roi_dd_ratio,
                "holdout_breaches": h.holdout_stats.days_breaching_daily_5pct,
                "holdout_verdict": h.holdout_gate.verdict.value,
                "deployable": h.deployable,
            })
        pd.DataFrame(hold_rows).to_csv(s5_dir / "holdout.csv", index=False, lineterminator="\n")

        # best_candidate.md
        best = search_result.top_k[0] if search_result.top_k else None
        best_holdout = holdout_results[0] if holdout_results else None
        best_lines = [
            "# Arc 5 v3.0 — Best Step 5 Candidate",
            f"Generated: {_now_iso()}",
            "",
        ]
        if best is not None:
            best_lines += [
                f"**Config:** {best.config_id}",
                f"**Architecture:** A1 system_level_filter",
                f"**Search verdict:** {best.gate.verdict.value}",
                "",
                "## Search WFO (eligible folds)",
                f"- Worst-fold ratio: {best.gate.worst_fold_ratio:.4f}",
                f"- Mean-fold ratio: {best.gate.mean_fold_ratio:.4f}",
                f"- Worst-fold ROI: {best.gate.worst_fold_roi:+.4%}",
                f"- Worst-fold DD: {best.gate.worst_fold_dd:.4%}",
                f"- Negative folds: {best.gate.n_negative_folds}",
                f"- Min trades/fold: {best.gate.min_trades_per_fold}",
                "",
            ]
        if best_holdout is not None:
            best_lines += [
                "## Holdout one-shot 2021-2026-04",
                f"- Trades: {best_holdout.holdout_stats.n_trades}",
                f"- ROI: {best_holdout.holdout_stats.roi_pct:+.4%}",
                f"- DD: {best_holdout.holdout_stats.max_dd_pct:.4%}",
                f"- Ratio: {best_holdout.holdout_stats.roi_dd_ratio:.4f}",
                f"- Verdict: {best_holdout.holdout_gate.verdict.value}",
                f"- Deployable: {best_holdout.deployable}",
            ]
        write_text_deterministic(s5_dir / "best_candidate.md", "\n".join(best_lines))

        # Step 5 manifest
        s5_manifest = {
            "step": 5, "generated_at": _now_iso(),
            "total_configs": total_configs,
            "n_folds_search": wfo_struct.n_folds,
            "top_3_ids": [c.config_id for c in search_result.top_k],
            "winner_config_id": search_result.top_k[0].config_id if search_result.top_k else None,
            "winner_verdict": search_result.top_k[0].gate.verdict.value if search_result.top_k else None,
            "holdout_top_verdict": holdout_results[0].holdout_gate.verdict.value if holdout_results else None,
            "deployable_after_holdout": any(h.deployable for h in holdout_results) if holdout_results else False,
        }
        write_text_deterministic(s5_dir / "manifest.json", json.dumps(s5_manifest, indent=2, sort_keys=True, default=str))

        elapsed = time.time() - t0
        _log(f"[DONE] elapsed {elapsed/60.0:.1f} min")
        return 0
    except Exception as exc:
        _log(f"FATAL: {exc}")
        traceback.print_exc()
        diag = [
            "# Arc 5 v3.0 — Runner Diagnostic",
            f"Generated: {_now_iso()}",
            "",
            "## Exception",
            f"```\n{exc}\n```",
            "",
            "## Traceback",
            f"```\n{traceback.format_exc()}\n```",
        ]
        diag_path = REPO_ROOT / "docs" / "dispatches" / "arc_5_diagnostic.md"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text("\n".join(diag), encoding="utf-8", newline="\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
