"""Drive a KH-24 anchor reproduction run end-to-end.

Loads H4 + D1 + H1 panels for the 28-pair universe across the run's
full window, then iterates the supplied WFO structure folds and runs
``MultiPairBacktester`` per fold. Per-fold artefacts (equity curve,
closed trades, fold stats) land under ``out_root/per_fold/``; a
roll-up ``fold_by_fold.parquet`` + human-readable ``summary.md`` +
sha256 ``manifest.json`` land at ``out_root/``.

Usage::

    py -m scripts.anchor.run_anchor \\
        --mode A \\
        --histdata-root C:/Users/panap/Documents/Forex-Backtester/data/histdata \\
        --cache-root   C:/Users/panap/AppData/Local/Temp/pr_e2_cache \\
        --out-root     results/anchor_kh24_7fold \\
        --pool-size    11

Mode A: 7-fold rolling Oct 2020 -> Jan 2026 (matches KH-24 published lineage).
Mode B: 11-fold expanding-IS 2010-2020 + one-shot holdout 2021-present.

Both modes use the locked KH24Config from PR-E.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.parallel import build_panel_parallel, default_pool_size
from core.sim.multipair_backtester import MultiPairBacktester
from core.sim.panel import Panel
from core.strategies.kh24.kh24 import KH24Config, build_kh24_runtime
from core.wfo.folds import Fold, WfoStructure, build_kh24_anchor_folds, build_v3_folds

PAIRS_28 = (
    "AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY "
    "EURAUD EURCAD EURCHF EURGBP EURJPY EURNZD EURUSD GBPAUD "
    "GBPCAD GBPCHF GBPJPY GBPNZD GBPUSD NZDCAD NZDCHF NZDJPY "
    "NZDUSD USDCAD USDCHF USDJPY"
).split()


# ── per-fold mechanics ──────────────────────────────────────────────


def _slice_panel(panel: Panel, start: pd.Timestamp, end: pd.Timestamp) -> Panel:
    """Inclusive slice on every pair_df."""
    sliced = {p: df.loc[start:end] for p, df in panel.pair_dfs.items()}
    return Panel.from_frames(sliced, tf=panel.tf)


def _max_drawdown_pct(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    cmax = equity.cummax()
    dd = (cmax - equity) / cmax
    return float(dd.max()) if len(dd) else 0.0


def _count_daily_5pct_breaches(equity: pd.Series) -> int:
    if len(equity) == 0:
        return 0
    daily = equity.resample("1D").agg(["first", "min"]).dropna()
    if len(daily) == 0:
        return 0
    return int(((daily["first"] - daily["min"]) / daily["first"] > 0.05).sum())


def _trade_metrics(closed_trades: tuple) -> dict[str, float]:
    """Win%, mean R, profit factor from the closed-trade ledger."""
    if not closed_trades:
        return {"win_pct": 0.0, "mean_r": 0.0, "profit_factor": 0.0, "n_trades": 0}
    pnls = np.array([t.pnl for t in closed_trades])
    # R = pnl / risk_at_entry. We approximate R by signed PnL relative to the
    # 1R hard SL distance: R = pnl / (entry_price - sl_price) × size = pnl / risk_amt
    # Without per-trade risk_amount stored, we use a proxy: each trade was sized
    # at 1% floor / sl_distance, so risk_amount = sl_distance × size. For the
    # anchor comparison we only need win% + mean PnL (currency); R values can
    # be derived externally.
    wins = (pnls > 0).sum()
    n = len(pnls)
    win_pct = float(wins / n) if n else 0.0
    gross_win = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()
    pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    if math.isinf(pf):
        pf = 999.0
    return {"win_pct": win_pct, "mean_pnl": float(pnls.mean()), "profit_factor": pf, "n_trades": int(n)}


def _run_one_fold(
    fold: Fold,
    panel_h4: Panel,
    panel_d1: Panel,
    panel_h1: Panel,
    config: KH24Config,
    warmup_days: int = 60,
) -> dict:
    """Execute KH-24 over one fold's OOS window; return per-fold artefacts."""
    slice_start = pd.Timestamp(fold.oos_start, tz="UTC") - pd.Timedelta(days=warmup_days)
    slice_end = pd.Timestamp(fold.oos_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    h4 = _slice_panel(panel_h4, slice_start, slice_end)
    d1 = _slice_panel(panel_d1, slice_start, slice_end)
    h1 = _slice_panel(panel_h1, slice_start, slice_end)

    runtime = build_kh24_runtime(h4, d1, h1, config=config)
    bt = MultiPairBacktester(
        panel=h4,
        account=runtime.account,
        strategy=runtime.strategy,
        trail_manager=runtime.trail_manager,
        exit_predicates=runtime.exit_predicates,
    )
    result = bt.run()
    equity = result.equity_curve
    # Restrict equity + trades to the OOS window (drop warmup)
    oos_start_ts = pd.Timestamp(fold.oos_start, tz="UTC")
    equity_oos = equity.loc[oos_start_ts:slice_end]
    trades_oos = tuple(t for t in result.closed_trades if t.entry_time >= oos_start_ts)

    if len(equity_oos) > 1:
        roi_pct = float(equity_oos.iloc[-1] / equity_oos.iloc[0] - 1.0)
    else:
        roi_pct = 0.0
    max_dd = _max_drawdown_pct(equity_oos)
    daily_breaches = _count_daily_5pct_breaches(equity_oos)
    trade_stats = _trade_metrics(trades_oos)
    ratio = roi_pct / max_dd if max_dd > 0 else (999.0 if roi_pct > 0 else 0.0)

    return {
        "fold_id": fold.fold_id,
        "oos_start": fold.oos_start.isoformat(),
        "oos_end": fold.oos_end.isoformat(),
        "n_trades": trade_stats["n_trades"],
        "roi_pct": roi_pct,
        "max_dd_pct": max_dd,
        "win_pct": trade_stats["win_pct"],
        "mean_pnl": trade_stats["mean_pnl"],
        "profit_factor": trade_stats["profit_factor"],
        "daily_5pct_breaches": daily_breaches,
        "roi_dd_ratio": ratio,
        "equity_curve": equity_oos,
        "closed_trades": trades_oos,
    }


# ── manifest + IO ──────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(manifest_path: Path, artefacts: list[Path]) -> None:
    root = manifest_path.parent
    entries: dict[str, dict] = {}
    for a in sorted(artefacts):
        rel = str(a.relative_to(root)).replace("\\", "/")
        entries[rel] = {"sha256": _sha256_file(a), "size_bytes": a.stat().st_size}
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artefacts": entries,
    }
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def _trades_to_dataframe(trades: tuple) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "position_id", "pair", "direction", "entry_time", "entry_price",
                "exit_time", "exit_price", "size", "pnl", "exit_reason",
            ]
        )
    return pd.DataFrame(
        [
            {
                "position_id": t.position_id,
                "pair": t.pair,
                "direction": t.direction.value,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ]
    )


def _write_outputs(out_root: Path, mode: str, fold_results: list[dict]) -> list[Path]:
    """Write per-fold equity / trades + roll-up; return list of artefact paths."""
    out_root.mkdir(parents=True, exist_ok=True)
    per_fold_dir = out_root / "per_fold"
    per_fold_dir.mkdir(exist_ok=True)

    written: list[Path] = []
    rows = []
    for r in fold_results:
        fid = r["fold_id"]
        # Per-fold equity
        eq_path = per_fold_dir / f"fold_{fid:02d}_equity.parquet"
        r["equity_curve"].to_frame(name="equity").to_parquet(
            eq_path, engine="pyarrow", compression="snappy"
        )
        written.append(eq_path)
        # Per-fold trades
        trades_path = per_fold_dir / f"fold_{fid:02d}_trades.parquet"
        _trades_to_dataframe(r["closed_trades"]).to_parquet(
            trades_path, engine="pyarrow", compression="snappy"
        )
        written.append(trades_path)
        rows.append(
            {k: r[k] for k in (
                "fold_id", "oos_start", "oos_end", "n_trades", "roi_pct",
                "max_dd_pct", "win_pct", "mean_pnl", "profit_factor",
                "daily_5pct_breaches", "roi_dd_ratio",
            )}
        )

    fold_table = pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)
    fold_path = out_root / "fold_by_fold.parquet"
    fold_table.to_parquet(fold_path, engine="pyarrow", compression="snappy")
    written.append(fold_path)

    summary = _build_summary_md(mode, fold_table)
    summary_path = out_root / "summary.md"
    summary_path.write_text(summary, encoding="utf-8", newline="\n")
    written.append(summary_path)

    manifest_path = out_root / "manifest.json"
    _write_manifest(manifest_path, written)
    written.append(manifest_path)
    return written


def _build_summary_md(mode: str, fold_table: pd.DataFrame) -> str:
    lines = [
        f"# KH-24 Anchor Run — Mode {mode}",
        "",
        "Run via `scripts.anchor.run_anchor` (PR-E.2). KH-24 strategy locked to deployed config; runs on HistData M1-derived H4/D1/H1 panels via the v3 backtester.",
        "",
        "## Fold-by-fold metrics",
        "",
        "| Fold | OOS window | Trades | ROI | DD | Win% | Mean PnL | PF | Daily breaches | ROI/DD |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in fold_table.iterrows():
        lines.append(
            f"| {int(r['fold_id'])} "
            f"| {r['oos_start']} → {r['oos_end']} "
            f"| {int(r['n_trades']):,} "
            f"| {r['roi_pct']:+.4%} "
            f"| {r['max_dd_pct']:.4%} "
            f"| {r['win_pct']:.2%} "
            f"| {r['mean_pnl']:+.2f} "
            f"| {r['profit_factor']:.2f} "
            f"| {int(r['daily_5pct_breaches'])} "
            f"| {r['roi_dd_ratio']:.2f} |"
        )
    # Aggregate row
    total_trades = int(fold_table["n_trades"].sum())
    worst_roi = float(fold_table["roi_pct"].min())
    worst_dd = float(fold_table["max_dd_pct"].max())
    n_positive = int((fold_table["roi_pct"] > 0).sum())
    n_total = len(fold_table)
    lines += [
        "",
        f"**Total trades:** {total_trades:,}",
        f"**Worst-fold ROI:** {worst_roi:+.4%}",
        f"**Worst-fold DD:** {worst_dd:.4%}",
        f"**Positive folds:** {n_positive}/{n_total}",
    ]
    return "\n".join(lines) + "\n"


# ── orchestrator ──────────────────────────────────────────────────


def run(
    mode: str,
    structure: WfoStructure,
    histdata_root: Path,
    cache_root: Path,
    out_root: Path,
    pool_size: int | None = None,
    config: KH24Config | None = None,
) -> Path:
    """Run KH-24 across the structure; write outputs; return out_root."""
    config = config or KH24Config()
    print(f"[anchor] Mode {mode} — {structure.name} structure, {structure.n_folds} folds")
    print(f"[anchor] Loading H4 + D1 + H1 panels (pool_size={pool_size})...")
    t0 = time.perf_counter()
    panel_h4 = build_panel_parallel(
        list(PAIRS_28), "H4",
        histdata_root=histdata_root, cache_root=cache_root, pool_size=pool_size,
    )
    panel_d1 = build_panel_parallel(
        list(PAIRS_28), "D1",
        histdata_root=histdata_root, cache_root=cache_root, pool_size=pool_size,
    )
    panel_h1 = build_panel_parallel(
        list(PAIRS_28), "H1",
        histdata_root=histdata_root, cache_root=cache_root, pool_size=pool_size,
    )
    print(f"[anchor] Panels built in {time.perf_counter() - t0:.1f}s")

    folds = list(structure.folds)
    if structure.holdout is not None:
        folds.append(structure.holdout)
    fold_results: list[dict] = []
    for fold in folds:
        # Skip folds with empty IS (v3 fold 1)
        if fold.is_empty_is and fold.fold_id == 1 and mode == "B":
            print(f"[anchor] Skipping fold {fold.fold_id} (empty IS)")
            continue
        t_fold = time.perf_counter()
        result = _run_one_fold(fold, panel_h4, panel_d1, panel_h1, config)
        el = time.perf_counter() - t_fold
        print(
            f"[anchor] fold {result['fold_id']:>2d} "
            f"[{result['oos_start'][:10]} -> {result['oos_end'][:10]}] "
            f"trades={result['n_trades']:>3d} "
            f"ROI={result['roi_pct']:+.4%} "
            f"DD={result['max_dd_pct']:.4%} "
            f"({el:.1f}s)"
        )
        fold_results.append(result)

    written = _write_outputs(out_root, mode, fold_results)
    print(f"[anchor] Wrote {len(written)} artefacts to {out_root}")
    print(f"[anchor] Manifest: {out_root / 'manifest.json'}")
    return out_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A", "B"], required=True,
                        help="A=7-fold KH-24 anchor; B=11-fold v3 + holdout")
    parser.add_argument("--histdata-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "A":
        structure = build_kh24_anchor_folds()
    else:
        structure = build_v3_folds()

    pool_size = args.pool_size if args.pool_size is not None else default_pool_size(28)
    run(
        mode=args.mode,
        structure=structure,
        histdata_root=args.histdata_root,
        cache_root=args.cache_root,
        out_root=args.out_root,
        pool_size=pool_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
