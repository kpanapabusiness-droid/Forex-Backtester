from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _walk_trades(root: Path) -> List[Tuple[str, str, Path]]:
    out: List[Tuple[str, str, Path]] = []
    if not root.exists():
        return out
    for pair_dir in sorted(root.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair = pair_dir.name
        for c1_dir in sorted(pair_dir.iterdir()):
            if not c1_dir.is_dir():
                continue
            c1 = c1_dir.name
            trades = c1_dir / 'trades.csv'
            if trades.exists():
                out.append((pair, c1, trades))
    return out


def _read_trades(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, encoding='utf-8', engine='python')
    # Normalize dates
    for col in ['entry_date', 'exit_date', 'date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def _extract_r_or_pnl(df: pd.DataFrame) -> pd.Series:
    if 'r_multiple' in df.columns:
        return pd.to_numeric(df['r_multiple'], errors='coerce').fillna(0.0)
    if 'pnl' in df.columns:
        return pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
    if 'return_pct' in df.columns:
        # interpret return_pct as %; convert to R proxy by sum of pct/100
        return pd.to_numeric(df['return_pct'], errors='coerce').fillna(0.0) / 100.0
    return pd.Series(dtype=float)


def _year_from_row(df: pd.DataFrame) -> pd.Series:
    for col in ['exit_date', 'entry_date', 'date']:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors='coerce')
            if s.notna().any():
                return s.dt.year
    return pd.Series(index=df.index, dtype='Int64')


def _month_key(df: pd.DataFrame) -> pd.Series:
    for col in ['exit_date', 'entry_date', 'date']:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors='coerce')
            if s.notna().any():
                return s.dt.to_period('M').astype(str)
    return pd.Series(index=df.index, dtype=object)


def compute_stability(root: Path, min_trades_flag: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pair, c1, csv_path in _walk_trades(root):
        df = _read_trades(csv_path)
        trades = int(len(df.index))
        r_or_pnl = _extract_r_or_pnl(df)

        # Yearly totals (R or PnL proxy)
        years = _year_from_row(df)
        yearly = pd.DataFrame({'year': years, 'value': r_or_pnl}).dropna(subset=['year'])
        yearly_tot = yearly.groupby('year', as_index=False)['value'].sum()
        n_years = int(len(yearly_tot.index))
        profitable_years = int((yearly_tot['value'] > 0.0).sum()) if n_years > 0 else 0
        profitable_years_pct = float(profitable_years / n_years * 100.0) if n_years > 0 else 0.0

        # Monthly coefficient of variation (std/mean) of monthly totals
        mkey = _month_key(df)
        monthly = pd.DataFrame({'ym': mkey, 'value': r_or_pnl}).dropna(subset=['ym'])
        monthly_tot = monthly.groupby('ym', as_index=False)['value'].sum()
        if len(monthly_tot.index) >= 2:
            m_mean = float(monthly_tot['value'].mean())
            m_std = float(monthly_tot['value'].std(ddof=1))
            monthly_cv = float(m_std / m_mean) if m_mean != 0.0 else float('nan')
        else:
            monthly_cv = float('nan')

        low_sample_flag = bool(trades < min_trades_flag)

        rows.append(
            {
                'pair': pair,
                'c1': c1,
                'trades': trades,
                'years': n_years,
                'profitable_years_pct': profitable_years_pct,
                'monthly_cv': monthly_cv,
                'low_sample_flag': low_sample_flag,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/c1_only_exits', type=str)
    ap.add_argument('--out', default='results/stability/c1_only_exits_stability.csv', type=str)
    ap.add_argument('--min-trades', default=100, type=int)
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = compute_stability(root, args.min_trades)
    df.to_csv(out_csv, index=False)
    print(f'Wrote: {out_csv}')


if __name__ == '__main__':
    main()


