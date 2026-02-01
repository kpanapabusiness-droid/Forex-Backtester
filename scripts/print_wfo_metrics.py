#!/usr/bin/env python3
"""Print WFO fold table and worst-fold metrics. Usage: python scripts/print_wfo_metrics.py [wfo_root]"""
import sys
from pathlib import Path


def _parse_summary(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if "Total Trades" in line:
            try:
                out["trades"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "ROI (%)" in line and "equity" not in line.lower():
            try:
                out["roi_pct"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Max Drawdown (%)" in line:
            try:
                out["max_dd_pct"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
    return out


def main():
    root = Path(__file__).resolve().parent.parent
    wfo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results" / "wfo"
    if not wfo_root.is_absolute():
        wfo_root = root / wfo_root
    wfo_root = wfo_root.resolve()

    run_dirs = sorted([d for d in wfo_root.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        print("No WFO run directories found under", wfo_root)
        return
    run_dir = run_dirs[0]
    print("=" * 60)
    print("WFO BACKTEST (spreads ON) — Newest run:", run_dir)
    print("=" * 60)

    folds_data = []
    for i in range(1, 5):
        fold_name = f"fold_{i:02d}"
        oos_summary = run_dir / fold_name / "out_of_sample" / "summary.txt"
        m = _parse_summary(oos_summary)
        m["fold"] = fold_name
        folds_data.append(m)

    print("\n--- Fold table (OOS) ---")
    print(f"  {'Fold':<10} {'ROI (%)':>10} {'Max DD (%)':>12} {'Trades':>8}")
    print("  " + "-" * 42)
    worst_roi_fold = None
    worst_dd_fold = None
    worst_roi_val = 1e9
    worst_dd_val = 0.0
    for m in folds_data:
        roi = m.get("roi_pct", 0.0)
        dd = m.get("max_dd_pct", 0.0)
        tr = m.get("trades", 0)
        print(f"  {m['fold']:<10} {roi:>10.2f} {dd:>12.2f} {tr:>8}")
        if roi < worst_roi_val:
            worst_roi_val = roi
            worst_roi_fold = m
        if dd < worst_dd_val:
            worst_dd_val = dd
            worst_dd_fold = m
    print()
    print("--- Worst-fold (by ROI %) ---")
    if worst_roi_fold:
        print(f"  Fold: {worst_roi_fold['fold']}")
        print(f"  ROI (%): {worst_roi_fold.get('roi_pct', 0):.2f}")
        print(f"  Max DD (%): {worst_roi_fold.get('max_dd_pct', 0):.2f}")
        print(f"  Trade count: {worst_roi_fold.get('trades', 0)}")
        worst_path = run_dir / worst_roi_fold["fold"] / "out_of_sample" / "summary.txt"
        print(f"  Absolute path: {worst_path.resolve()}")
    print()
    print("--- Worst-fold (by Max DD) ---")
    if worst_dd_fold:
        print(f"  Fold: {worst_dd_fold['fold']}")
        print(f"  ROI (%): {worst_dd_fold.get('roi_pct', 0):.2f}")
        print(f"  Max DD (%): {worst_dd_fold.get('max_dd_pct', 0):.2f}")
        print(f"  Trade count: {worst_dd_fold.get('trades', 0)}")
    print()


if __name__ == "__main__":
    main()
