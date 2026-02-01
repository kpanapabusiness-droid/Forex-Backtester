#!/usr/bin/env python3
"""Print FULL backtest metrics and paths from results dir. Usage: python scripts/print_full_metrics.py [results_dir]"""
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results"
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    results_dir = results_dir.resolve()

    summary_path = results_dir / "summary.txt"
    trades_path = results_dir / "trades.csv"
    equity_path = results_dir / "equity_curve.csv"

    print("=" * 60)
    print("FULL BACKTEST (spreads ON) — Output directory:", results_dir)
    print("=" * 60)

    if summary_path.exists():
        text = summary_path.read_text(encoding="utf-8")
        print("\n--- Summary metrics ---")
        for line in text.splitlines():
            line = line.strip()
            if any(x in line for x in ["Total Trades", "ROI ($)", "ROI (%)", "Max Drawdown", "Wins", "Losses", "Scratches"]):
                print(" ", line)
    else:
        print(" (summary.txt not found)")

    print("\n--- Absolute paths ---")
    for name, p in [("summary.txt", summary_path), ("trades.csv", trades_path), ("equity_curve.csv", equity_path)]:
        print(f"  {name}: {p}")

    if trades_path.exists():
        import pandas as pd
        df = pd.read_csv(trades_path)
        pnl = pd.to_numeric(df.get("pnl", 0), errors="coerce").fillna(0)
        total_pnl = float(pnl.sum())
        if "win" in df.columns and "loss" in df.columns and "scratch" in df.columns:
            win_mask = df["win"].fillna(0).astype(bool)
            loss_mask = df["loss"].fillna(0).astype(bool)
            scratch_mask = df["scratch"].fillna(0).astype(bool)
            print("\n--- Trades breakdown (win/loss/scratch from CSV columns) ---")
        else:
            scratch_threshold = 0.01
            scratch_mask = pnl.abs() < scratch_threshold
            win_mask = pnl > 0
            loss_mask = pnl < 0
            print(f"\n--- Trades breakdown (scratch = |pnl| < {scratch_threshold}) ---")
        win_pnl = float(pnl[win_mask].sum())
        lose_pnl = float(pnl[loss_mask].sum())
        scratch_pnl = float(pnl[scratch_mask].sum())
        n_win = int(win_mask.sum())
        n_lose = int(loss_mask.sum())
        n_scratch = int(scratch_mask.sum())
        print(f"  Total PnL sum:        {total_pnl:.2f}")
        print(f"  Winners  PnL sum:    {win_pnl:.2f}  (count: {n_win})")
        print(f"  Losers   PnL sum:    {lose_pnl:.2f}  (count: {n_lose})")
        print(f"  Scratches PnL sum:   {scratch_pnl:.2f}  (count: {n_scratch})")
    print()

if __name__ == "__main__":
    main()
