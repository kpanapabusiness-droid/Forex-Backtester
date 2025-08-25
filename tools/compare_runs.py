#!/usr/bin/env python3
import argparse, json, os, pandas as pd
from pathlib import Path
import backtester

def load_metrics(run_path: Path):
    mp = run_path / "metrics.json"
    if mp.exists():
        return json.loads(mp.read_text())
    # fallback: derive from trades.csv
    tp = run_path / "trades.csv"
    if not tp.exists():
        return None
    df = pd.read_csv(tp)
    wins = int((df.get("win")==True).sum()) if "win" in df else int((df["pnl"]>0).sum())
    losses = int((df.get("loss")==True).sum()) if "loss" in df else int((df["pnl"]<0).sum())
    scratches = int((df.get("scratch")==True).sum()) if "scratch" in df else int((df["pnl"]==0).sum())
    total_pnl = float(df["pnl"].fillna(0).sum()) if "pnl" in df else 0.0
    return {"total_trades": len(df), "wins":wins, "losses":losses, "scratches":scratches, "roi_abs": total_pnl}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default=None, help="results_history folder")
    parser.add_argument("--a", help="run_id A (older)")
    parser.add_argument("--b", help="run_id B (newer)")
    args = parser.parse_args()

    cfg = backtester.load_config("config.yaml")
    hist_cfg = args.history or cfg.get("results_history_dir","results/results_history")
    hist_dir = (backtester.ROOT / hist_cfg).resolve()

    runs = sorted([p for p in hist_dir.iterdir() if p.is_dir()])
    if not runs:
        print("No runs found in", hist_dir)
        return

    # pick latest two if not provided
    if not args.a or not args.b:
        runs_sorted = sorted(runs, key=lambda p: p.stat().st_mtime)
        a, b = runs_sorted[-2], runs_sorted[-1] if len(runs_sorted) >= 2 else (runs_sorted[0], runs_sorted[0])
    else:
        a = hist_dir / args.a
        b = hist_dir / args.b

    ma = load_metrics(a); mb = load_metrics(b)
    if not ma or not mb:
        print("Missing metrics for one of the runs.")
        return

    def fmt(m): return f"{m:.2f}" if isinstance(m, float) else str(m)
    print(f"\nCompare runs\nA: {a.name}\nB: {b.name}\n")
    keys = ["total_trades","wins","losses","scratches","roi_abs","win_rate_pct","roi_pct"]
    for k in keys:
        va = ma.get(k, 0); vb = mb.get(k, 0)
        diff = (vb - va) if isinstance(va, (int,float)) and isinstance(vb,(int,float)) else "-"
        print(f"{k:16}: {fmt(va):>10}  ->  {fmt(vb):>10}   Δ {fmt(diff)}")

if __name__ == "__main__":
    main()
