from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_top_rows(lb_csv: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(lb_csv)
    if df.empty:
        return df
    # overall top across pairs; if leaderboard is sorted per pair, we still take global top
    df_sorted = df.sort_values(['roi_pct', 'win_rate_pct', 'max_dd_pct'], ascending=[False, False, True])
    # dedupe by c1 if present to diversify
    if 'c1' in df_sorted.columns:
        df_sorted = df_sorted.drop_duplicates(subset=['c1'], keep='first')
    return df_sorted.head(top_n)


def _equity_path(root: Path, pair: str, c1: str) -> Path:
    return root / pair / c1 / 'equity_curve.csv'


def _load_equity(eq_path: Path) -> pd.DataFrame:
    df = pd.read_csv(eq_path)
    # Expect columns: date,equity
    if 'date' in df.columns and 'equity' in df.columns:
        df = df[['date', 'equity']].copy()
    else:
        # Attempt to infer
        date_col = next((c for c in df.columns if c.lower() == 'date'), df.columns[0])
        eq_col = next((c for c in df.columns if 'equity' in c.lower()), df.columns[-1])
        df = df[[date_col, eq_col]].copy()
        df.columns = ['date', 'equity']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'equity']).sort_values('date')
    return df


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd


def plot_equity(df_sel: pd.DataFrame, root: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted = 0
    for _, row in df_sel.iterrows():
        pair = str(row['pair']) if 'pair' in row else None
        c1 = str(row['c1']) if 'c1' in row else None
        if not pair or not c1:
            continue
        eq_path = _equity_path(root, pair, c1)
        if not eq_path.exists():
            continue
        eq_df = _load_equity(eq_path)
        plt.plot(eq_df['date'], eq_df['equity'], label=f'{pair}-{c1}')
        plotted += 1
    if plotted == 0:
        print('No equity_curve.csv found for selection; nothing plotted.')
        return
    plt.title('C1-only exits: Top-N Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend(loc='best')
    out_path = outdir / 'c1_only_exits_topN_equity.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Wrote: {out_path}')


def plot_drawdowns(df_sel: pd.DataFrame, root: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted = 0
    for _, row in df_sel.iterrows():
        pair = str(row['pair']) if 'pair' in row else None
        c1 = str(row['c1']) if 'c1' in row else None
        if not pair or not c1:
            continue
        eq_path = _equity_path(root, pair, c1)
        if not eq_path.exists():
            continue
        eq_df = _load_equity(eq_path)
        dd = _compute_drawdown(eq_df['equity'])
        plt.plot(eq_df['date'], dd, label=f'{pair}-{c1}')
        plotted += 1
    if plotted == 0:
        print('No equity_curve.csv found for selection; nothing plotted.')
        return
    plt.title('C1-only exits: Top-N Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (fraction)')
    plt.legend(loc='best')
    out_path = outdir / 'c1_only_exits_topN_drawdowns.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Wrote: {out_path}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-leaderboard', default='results/leaderboard_c1_only_exits_by_pair.csv', type=str)
    ap.add_argument('--root', default='results/c1_only_exits', type=str)
    ap.add_argument('--top', default=12, type=int)
    ap.add_argument('--outdir', default='results/plots', type=str)
    args = ap.parse_args()

    lb = Path(args.from_leaderboard)
    root = Path(args.root)
    outdir = Path(args.outdir)

    df_sel = _read_top_rows(lb, args.top)
    if df_sel.empty:
        print('Leaderboard is empty or missing; nothing to plot.')
        return
    plot_equity(df_sel, root, outdir)
    plot_drawdowns(df_sel, root, outdir)


if __name__ == '__main__':
    main()


