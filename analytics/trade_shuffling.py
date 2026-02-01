# analytics/trade_shuffling.py
# Phase 5.1 — Trade shuffling / path risk: shuffle trade order, measure DD and pain metrics.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from analytics.metrics import compute_max_drawdown_pct


def max_consecutive_losses(pnl_series: pd.Series) -> int:
    """
    Count the longest streak of trades with negative pnl.
    Empty or all-non-negative returns 0.
    """
    if pnl_series is None or (hasattr(pnl_series, "empty") and pnl_series.empty):
        return 0
    pnl = pd.to_numeric(pnl_series, errors="coerce").fillna(0.0)
    if pnl.empty:
        return 0
    is_loss = (pnl < 0).astype(int)
    if is_loss.sum() == 0:
        return 0
    groups = (is_loss.diff() != 0).cumsum()
    streak_lengths = is_loss.groupby(groups).sum()
    return int(streak_lengths.max()) if not streak_lengths.empty else 0


def _worst_peak_to_trough_dollars(equity: pd.Series) -> float:
    """Largest drop from a peak to a trough in dollars (negative or zero)."""
    if equity is None or (hasattr(equity, "empty") and equity.empty):
        return 0.0
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty or len(eq) < 2:
        return 0.0
    peak = eq.cummax()
    drawdown_dollars = eq - peak
    return float(drawdown_dollars.min()) if np.isfinite(drawdown_dollars.min()) else 0.0


def run_trade_shuffling(
    trades_df: pd.DataFrame,
    starting_balance: float,
    n_sims: int,
    seed: int,
    out_dir: str | Path,
) -> Dict[str, Any]:
    """
    Run n_sims shuffles of trade order; for each sim compute max_dd_pct, max_consec_losses,
    ending_equity, ending_roi_pct. Write runs CSV, summary JSON, and summary TXT.
    Returns summary dict. Handles empty trades gracefully.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pnl_col = "pnl"
    if trades_df is None or trades_df.empty:
        runs_df = pd.DataFrame(
            columns=[
                "sim_id",
                "max_dd_pct",
                "max_consec_losses",
                "ending_equity",
                "ending_roi_pct",
                "worst_peak_to_trough_dollars",
            ]
        )
        runs_df.to_csv(out_path / "trade_shuffling_runs.csv", index=False)
        summary = {
            "n_sims": n_sims,
            "seed": seed,
            "starting_balance": starting_balance,
            "n_trades": 0,
            "max_dd_pct_p05": 0.0,
            "max_dd_pct_p50": 0.0,
            "max_dd_pct_p95": 0.0,
            "max_dd_pct_worst": 0.0,
            "max_consec_losses_p05": 0,
            "max_consec_losses_p50": 0,
            "max_consec_losses_p95": 0,
            "ruin_probs": {"P_dd_le_10": 0.0, "P_dd_le_15": 0.0, "P_dd_le_20": 0.0},
        }
        (out_path / "trade_shuffling_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (out_path / "trade_shuffling_summary.txt").write_text(
            "Trade Shuffling Summary (Phase 5.1)\nNo trades — no simulations run.\n",
            encoding="utf-8",
        )
        return summary

    if pnl_col not in trades_df.columns:
        pnl_series = pd.Series(dtype=float)
    else:
        pnl_series = pd.to_numeric(trades_df[pnl_col], errors="coerce").fillna(0.0)
    if pnl_series.empty:
        pnl_series = pd.Series([0.0])

    rng = np.random.default_rng(seed)
    n_trades = len(pnl_series)
    rows = []

    for sim_id in range(n_sims):
        perm = rng.permutation(n_trades)
        shuffled_pnl = pnl_series.iloc[perm].reset_index(drop=True)
        equity = starting_balance + shuffled_pnl.cumsum()
        equity = pd.Series(equity.values)
        max_dd_pct = compute_max_drawdown_pct(equity)
        max_consec = max_consecutive_losses(shuffled_pnl)
        ending_equity = float(equity.iloc[-1]) if len(equity) > 0 else starting_balance
        ending_roi_pct = (
            (ending_equity / starting_balance - 1.0) * 100.0
            if starting_balance > 0
            else 0.0
        )
        worst_ptt = _worst_peak_to_trough_dollars(equity)
        rows.append(
            {
                "sim_id": sim_id,
                "max_dd_pct": round(max_dd_pct, 4),
                "max_consec_losses": max_consec,
                "ending_equity": round(ending_equity, 2),
                "ending_roi_pct": round(ending_roi_pct, 4),
                "worst_peak_to_trough_dollars": round(worst_ptt, 2),
            }
        )

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out_path / "trade_shuffling_runs.csv", index=False)

    max_dd = runs_df["max_dd_pct"]
    max_consec_s = runs_df["max_consec_losses"]
    summary = {
        "n_sims": n_sims,
        "seed": seed,
        "starting_balance": starting_balance,
        "n_trades": int(n_trades),
        "max_dd_pct_p05": float(max_dd.quantile(0.05)),
        "max_dd_pct_p50": float(max_dd.quantile(0.50)),
        "max_dd_pct_p95": float(max_dd.quantile(0.95)),
        "max_dd_pct_worst": float(max_dd.min()),
        "max_consec_losses_p05": int(max_consec_s.quantile(0.05)),
        "max_consec_losses_p50": int(max_consec_s.quantile(0.50)),
        "max_consec_losses_p95": int(max_consec_s.quantile(0.95)),
        "ruin_probs": {
            "P_dd_le_10": float((max_dd <= -10).mean()),
            "P_dd_le_15": float((max_dd <= -15).mean()),
            "P_dd_le_20": float((max_dd <= -20).mean()),
        },
    }
    (out_path / "trade_shuffling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "Trade Shuffling Summary (Phase 5.1)",
        "-----------------------------------",
        f"n_sims             : {summary['n_sims']}",
        f"seed               : {summary['seed']}",
        f"starting_balance   : {summary['starting_balance']}",
        f"n_trades           : {summary['n_trades']}",
        "",
        "Max DD (%)",
        f"  p05               : {summary['max_dd_pct_p05']:.2f}",
        f"  p50               : {summary['max_dd_pct_p50']:.2f}",
        f"  p95               : {summary['max_dd_pct_p95']:.2f}",
        f"  worst             : {summary['max_dd_pct_worst']:.2f}",
        "",
        "Max consecutive losses",
        f"  p05               : {summary['max_consec_losses_p05']}",
        f"  p50               : {summary['max_consec_losses_p50']}",
        f"  p95               : {summary['max_consec_losses_p95']}",
        "",
        "Ruin probabilities",
        f"  P(dd <= -10%)     : {summary['ruin_probs']['P_dd_le_10']:.4f}",
        f"  P(dd <= -15%)     : {summary['ruin_probs']['P_dd_le_15']:.4f}",
        f"  P(dd <= -20%)     : {summary['ruin_probs']['P_dd_le_20']:.4f}",
    ]
    (out_path / "trade_shuffling_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    return summary
