from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Non-C1 names to exclude
NON_C1 = {
    'chandelier_exit',
    'baseline',
    'volume',
    'exit_chandelier',
    'exit_trailing',
    'exit_baseline',
}

# Helper: check if c1 name is a confirmation indicator (not excluded)
def _is_c1_indicator(c1: str) -> bool:
    c1_low = str(c1).lower()
    if c1_low in NON_C1:
        return False
    if c1_low.startswith('exit_'):
        return False
    return True


def _parse_float(text: str) -> float:
    if text is None:
        return 0.0
    # strip common labels and symbols, keep digits, minus, dot
    s = str(text)
    s = s.replace('%', '').strip()
    m = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)
    try:
        return float(m[0]) if m else 0.0
    except Exception:
        return 0.0


def _parse_summary_txt(path: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        'roi_pct': None,
        'win_rate_pct': None,
        'avg_r': None,
        'max_dd_pct': None,
        'trades': None,
        'sharpe': None,
        'sortino': None,
    }
    if not path.exists():
        return out
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return out

    for ln in lines:
        low = ln.lower()
        if ':' not in low:
            continue
        key, val = ln.split(':', 1)
        k = key.strip().lower()
        v = val.strip()
        if 'roi' in k and '%' in v or 'roi%' in k or k.startswith('roi'):
            out['roi_pct'] = _parse_float(v)
        elif ('win' in k and 'rate' in k) or k.strip() == 'winrate':
            out['win_rate_pct'] = _parse_float(v)
        elif 'avg' in k and ('r' == k.strip()[-1] or 'avg r' in k or 'average r' in k):
            out['avg_r'] = _parse_float(v)
        elif 'maxdd' in k or ('max' in k and 'dd' in k) or 'max dd' in k:
            out['max_dd_pct'] = _parse_float(v)
        elif 'trades' in k:
            out['trades'] = _parse_float(v)
        elif 'sharpe' in k:
            out['sharpe'] = _parse_float(v)
        elif 'sortino' in k:
            out['sortino'] = _parse_float(v)
    return out


def _parse_trades_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding='utf-8', errors='ignore')
    except Exception:
        return None
    if df.empty:
        return None
    # Normalize date column
    for col in ['exit_date', 'entry_date', 'date', 'close_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            break
    # Ensure leg column (if missing, try to infer from thread_id or other logic)
    if 'leg' not in df.columns:
        # Try to infer: if thread_id exists, use modulo or other logic
        if 'thread_id' in df.columns:
            df['leg'] = df['thread_id'].apply(lambda x: 'A' if (x or 0) % 2 == 0 else 'B')
        else:
            # Default: assume all are leg A for win-rate purposes
            df['leg'] = 'A'
    # Ensure r column (R multiple)
    if 'r' not in df.columns:
        if 'r_multiple' in df.columns:
            df['r'] = pd.to_numeric(df['r_multiple'], errors='coerce').fillna(0.0)
        elif 'pnl' in df.columns:
            # Proxy: assume fixed R=1.0 per trade, so r = pnl / (some base)
            # For now, use pnl directly as proxy (will need starting balance for true R)
            df['r'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0) / 100.0  # rough proxy
        else:
            df['r'] = 0.0
    # Normalize exit_reason
    if 'exit_reason' not in df.columns:
        df['exit_reason'] = ''
    df['exit_reason'] = df['exit_reason'].astype(str).str.upper()
    return df


def _compute_wr_from_leg_a(df: pd.DataFrame) -> Optional[float]:
    leg_a = df[df['leg'].isin(['A', 'a'])].copy()
    if leg_a.empty:
        return None
    wins = len(leg_a[leg_a['exit_reason'].str.contains('TP1', na=False)])
    losses = len(leg_a[leg_a['exit_reason'].str.contains('SL', na=False)])
    # Exclude scratches (flips before TP1)
    non_scratch = wins + losses
    if non_scratch == 0:
        return None
    return float(wins / non_scratch * 100.0)


def _compute_avg_r_all(df: pd.DataFrame) -> Optional[float]:
    if 'r' not in df.columns or df['r'].isna().all():
        return None
    r_vals = pd.to_numeric(df['r'], errors='coerce').dropna()
    if r_vals.empty:
        return None
    return float(r_vals.mean())


def _build_monthly_series(df: pd.DataFrame) -> Optional[pd.Series]:
    date_col = None
    for col in ['exit_date', 'entry_date', 'date', 'close_time']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        return None
    df_with_date = df[[date_col, 'r']].copy()
    df_with_date = df_with_date.dropna(subset=[date_col])
    if df_with_date.empty:
        return None
    df_with_date['ym'] = pd.to_datetime(df_with_date[date_col]).dt.to_period('M').astype(str)
    monthly = df_with_date.groupby('ym', as_index=False)['r'].sum()
    return monthly.set_index('ym')['r']


def _compute_sharpe_sortino(monthly: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if monthly is None or monthly.empty:
        return None, None
    mean_r = float(monthly.mean())
    std_r = float(monthly.std(ddof=1))
    if std_r <= 0:
        sharpe = 0.0 if mean_r == 0 else None
    else:
        sharpe = float(mean_r / std_r)
    # Sortino: use downside std
    negative = monthly[monthly < 0]
    if negative.empty:
        sortino = 0.0 if mean_r == 0 else None
    else:
        downside_std = float(negative.std(ddof=1))
        if downside_std <= 0:
            sortino = 0.0 if mean_r == 0 else None
        else:
            sortino = float(mean_r / downside_std)
    return sharpe, sortino


def _compute_max_dd_pct_from_equity(ec_path: Path, trades_df: Optional[pd.DataFrame]) -> Optional[float]:
    # Try equity_curve.csv first
    if ec_path.exists():
        try:
            ec_df = pd.read_csv(ec_path, encoding='utf-8', errors='ignore')
            if 'equity' in ec_df.columns:
                equity = pd.to_numeric(ec_df['equity'], errors='coerce').dropna()
            elif 'cumulative_r' in ec_df.columns:
                equity = 1.0 + pd.to_numeric(ec_df['cumulative_r'], errors='coerce').fillna(0.0).cumsum()
            else:
                equity = None
            if equity is not None and not equity.empty:
                peak = equity.cummax()
                dd = (equity / peak - 1.0) * 100.0
                return float(dd.min())
        except Exception:
            pass
    # Fallback: reconstruct from trades
    if trades_df is not None and 'r' in trades_df.columns:
        date_col = None
        for col in ['exit_date', 'entry_date', 'date', 'close_time']:
            if col in trades_df.columns:
                date_col = col
                break
        if date_col:
            df_sorted = trades_df[[date_col, 'r']].copy()
            df_sorted = df_sorted.dropna(subset=[date_col]).sort_values(date_col)
            if not df_sorted.empty:
                r_vals = pd.to_numeric(df_sorted['r'], errors='coerce').fillna(0.0)
                equity = 1.0 + r_vals.cumsum()
                peak = equity.cummax()
                dd = (equity / peak - 1.0) * 100.0
                return float(dd.min())
    return None


def _walk_runs(root: Path) -> List[Tuple[str, str, Path]]:
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
            out.append((pair, c1, c1_dir))
    return out


def aggregate(root: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    stats = {
        'total_scanned': 0,
        'kept': 0,
        'dropped_non_c1': 0,
        'dropped_zero_trades': 0,
        'computed_from_csv': 0,
    }
    for pair, c1, run_dir in _walk_runs(root):
        stats['total_scanned'] += 1
        # Exclude non-C1
        if not _is_c1_indicator(c1):
            stats['dropped_non_c1'] += 1
            continue
        summary = _parse_summary_txt(run_dir / 'summary.txt')
        trades_csv_path = run_dir / 'trades.csv'
        equity_csv_path = run_dir / 'equity_curve.csv'
        trades_df = _parse_trades_csv(trades_csv_path)
        # Get trades count
        trades_count = summary.get('trades')
        if trades_count is None or trades_count == 0.0:
            if trades_df is not None:
                trades_count = float(len(trades_df))
            else:
                trades_count = 0.0
            summary['trades'] = trades_count
        # Exclude zero trades
        if trades_count is None or trades_count == 0:
            stats['dropped_zero_trades'] += 1
            continue
        # Fill missing metrics from CSV
        computed_any = False
        if trades_df is not None:
            # Win rate from Leg A
            if summary.get('win_rate_pct') is None:
                wr = _compute_wr_from_leg_a(trades_df)
                if wr is not None:
                    summary['win_rate_pct'] = wr
                    computed_any = True
            # Avg R
            if summary.get('avg_r') is None:
                avg_r = _compute_avg_r_all(trades_df)
                if avg_r is not None:
                    summary['avg_r'] = avg_r
                    computed_any = True
            # Sharpe/Sortino from monthly series
            if summary.get('sharpe') is None or summary.get('sortino') is None:
                monthly = _build_monthly_series(trades_df)
                sharpe, sortino = _compute_sharpe_sortino(monthly)
                if sharpe is not None and summary.get('sharpe') is None:
                    summary['sharpe'] = sharpe
                    computed_any = True
                if sortino is not None and summary.get('sortino') is None:
                    summary['sortino'] = sortino
                    computed_any = True
            # Max DD
            if summary.get('max_dd_pct') is None:
                max_dd = _compute_max_dd_pct_from_equity(equity_csv_path, trades_df)
                if max_dd is not None:
                    summary['max_dd_pct'] = max_dd
                    computed_any = True
        if computed_any:
            stats['computed_from_csv'] += 1
        # Fill None with 0.0 for output
        rows.append(
            {
                'pair': pair,
                'c1': c1,
                'trades': int(summary.get('trades', 0.0) or 0.0),
                'roi_pct': float(summary.get('roi_pct', 0.0) or 0.0),
                'win_rate_pct': float(summary.get('win_rate_pct', 0.0) or 0.0),
                'avg_r': float(summary.get('avg_r', 0.0) or 0.0),
                'max_dd_pct': float(summary.get('max_dd_pct', 0.0) or 0.0),
                'sharpe': float(summary.get('sharpe', 0.0) or 0.0),
                'sortino': float(summary.get('sortino', 0.0) or 0.0),
            }
        )
        stats['kept'] += 1
    return pd.DataFrame(rows), stats


def build_leaderboards(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    # By pair: sort within each pair
    by_pair = (
        df.copy()
        .assign(
            roi_pct=df['roi_pct'].fillna(0.0),
            win_rate_pct=df['win_rate_pct'].fillna(0.0),
            max_dd_pct=df['max_dd_pct'].fillna(0.0),
        )
        .sort_values(['pair', 'roi_pct', 'win_rate_pct', 'max_dd_pct'], ascending=[True, False, False, True])
    )

    # Overall by C1: average metrics across pairs, sum trades
    overall = (
        df.copy()
        .groupby('c1', as_index=False)
        .agg(
            trades=('trades', 'sum'),
            roi_pct=('roi_pct', 'mean'),
            win_rate_pct=('win_rate_pct', 'mean'),
            avg_r=('avg_r', 'mean'),
            max_dd_pct=('max_dd_pct', 'mean'),
            sharpe=('sharpe', 'mean'),
            sortino=('sortino', 'mean'),
        )
        .fillna(0.0)
        .sort_values(['roi_pct', 'max_dd_pct'], ascending=[False, True])
    )
    return by_pair, overall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/c1_only_exits', type=str)
    ap.add_argument('--outdir', default='results', type=str)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_df, stats = aggregate(root)
    # Write all runs table
    all_csv = outdir / 'c1_only_exits_all_runs.csv'
    all_df.to_csv(all_csv, index=False)

    by_pair_df, overall_df = build_leaderboards(all_df)

    lb_by_pair = outdir / 'leaderboard_c1_only_exits_by_pair.csv'
    lb_overall = outdir / 'leaderboard_c1_only_exits_overall.csv'

    by_pair_df.to_csv(lb_by_pair, index=False)
    overall_df.to_csv(lb_overall, index=False)

    print(f'Wrote: {all_csv}')
    print(f'Wrote: {lb_by_pair}')
    print(f'Wrote: {lb_overall}')
    print()
    print('Summary:')
    print(f'  Total scanned: {stats["total_scanned"]}')
    print(f'  Kept: {stats["kept"]}')
    print(f'  Dropped (non-C1): {stats["dropped_non_c1"]}')
    print(f'  Dropped (trades=0): {stats["dropped_zero_trades"]}')
    print(f'  Rows with CSV-computed metrics: {stats["computed_from_csv"]}')


if __name__ == '__main__':
    main()


