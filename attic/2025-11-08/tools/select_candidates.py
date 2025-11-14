from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def select_candidates(lb_csv: Path, stab_csv: Path, min_trades: int, min_profitable_years: float) -> pd.DataFrame:
    if not lb_csv.exists() or not stab_csv.exists():
        return pd.DataFrame()
    lb = pd.read_csv(lb_csv)
    stab = pd.read_csv(stab_csv)

    # Overall leaderboard is per-c1; stability is per pair×c1. Aggregate stability by c1.
    stab_agg = (
        stab.copy()
        .groupby('c1', as_index=False)
        .agg(
            trades=('trades', 'sum'),
            profitable_years_pct=('profitable_years_pct', 'mean'),
            monthly_cv=('monthly_cv', 'mean'),
        )
    )

    merged = lb.merge(stab_agg, on='c1', how='left', suffixes=('', '_stab'))
    merged['trades'] = merged['trades'].fillna(0).astype(int)
    merged['profitable_years_pct'] = merged['profitable_years_pct'].fillna(0.0)

    filt = (
        (merged['trades'] >= int(min_trades))
        & (merged['profitable_years_pct'] >= float(min_profitable_years))
        & (~merged['monthly_cv'].isna())
    )
    sel = merged.loc[filt].copy()
    if sel.empty:
        return sel
    # Sort by ROI desc, then MaxDD asc
    sel = sel.sort_values(['roi_pct', 'max_dd_pct'], ascending=[False, True])
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lb', default='results/leaderboard_c1_only_exits_overall.csv', type=str)
    ap.add_argument('--stab', default='results/stability/c1_only_exits_stability.csv', type=str)
    ap.add_argument('--out', default='results/candidates/c1_only_exits_candidates.csv', type=str)
    ap.add_argument('--min-trades', default=300, type=int)
    ap.add_argument('--min-profitable-years', default=60.0, type=float)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = select_candidates(Path(args.lb), Path(args.stab), args.min_trades, args.min_profitable_years)
    df.to_csv(out_path, index=False)
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()


