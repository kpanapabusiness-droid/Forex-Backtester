from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def _is_c1_indicator(c1: str) -> bool:
    c1_low = str(c1).lower()
    if c1_low in NON_C1:
        return False
    if c1_low.startswith('exit_'):
        return False
    return True


def _parse_float(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    s = s.replace('%', '').strip()
    m = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)
    try:
        return float(m[0]) if m else None
    except Exception:
        return None


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
    for col in ['exit_date', 'close_date', 'date', 'close_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            break
    # Ensure leg column
    if 'leg' not in df.columns:
        if 'thread_id' in df.columns:
            df['leg'] = df['thread_id'].apply(lambda x: 'A' if (x or 0) % 2 == 0 else 'B')
        else:
            df['leg'] = 'A'
    # Ensure r column
    if 'r' not in df.columns:
        if 'r_multiple' in df.columns:
            df['r'] = pd.to_numeric(df['r_multiple'], errors='coerce')
        elif 'pnl' in df.columns:
            df['r'] = pd.to_numeric(df['pnl'], errors='coerce') / 100.0
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
    non_scratch = wins + losses
    if non_scratch == 0:
        return None
    return float(wins / non_scratch * 100.0)


def _compute_avg_r_all(df: pd.DataFrame) -> Optional[float]:
    if 'r' not in df.columns:
        return None
    r_vals = pd.to_numeric(df['r'], errors='coerce').dropna()
    if r_vals.empty:
        return None
    return float(r_vals.mean())


def _build_monthly_returns(df: pd.DataFrame) -> Optional[pd.Series]:
    date_col = None
    for col in ['exit_date', 'close_date', 'date', 'close_time']:
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


def _compute_max_dd_pct(ec_path: Path, trades_df: Optional[pd.DataFrame]) -> Optional[float]:
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
    if trades_df is not None and 'r' in trades_df.columns:
        date_col = None
        for col in ['exit_date', 'close_date', 'date', 'close_time']:
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


def _walk_runs(root: Path, pair_filter: Optional[Set[str]] = None) -> List[tuple[str, str, Path]]:
    out: List[tuple[str, str, Path]] = []
    if not root.exists():
        return out
    for pair_dir in sorted(root.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair = pair_dir.name
        if pair_filter and pair not in pair_filter:
            continue
        for c1_dir in sorted(pair_dir.iterdir()):
            if not c1_dir.is_dir():
                continue
            c1 = c1_dir.name
            out.append((pair, c1, c1_dir))
    return out


def build_table(root: Path, pair_filter: Optional[Set[str]] = None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    runs = _walk_runs(root, pair_filter)
    
    for pair, c1, run_dir in runs:
        if not _is_c1_indicator(c1):
            continue
        
        summary = _parse_summary_txt(run_dir / 'summary.txt')
        trades_csv = run_dir / 'trades.csv'
        equity_csv = run_dir / 'equity_curve.csv'
        trades_df = _parse_trades_csv(trades_csv)
        
        # Trades count
        trades_count = summary.get('trades')
        if trades_count is None or trades_count == 0.0:
            if trades_df is not None:
                trades_count = float(len(trades_df))
            else:
                trades_count = 0.0
        if trades_count is None or trades_count == 0:
            continue
        
        # Fill missing metrics from CSV
        if trades_df is not None:
            if summary.get('win_rate_pct') is None:
                wr = _compute_wr_from_leg_a(trades_df)
                if wr is not None:
                    summary['win_rate_pct'] = wr
            
            if summary.get('avg_r') is None:
                avg_r = _compute_avg_r_all(trades_df)
                if avg_r is not None:
                    summary['avg_r'] = avg_r
            
            if summary.get('sharpe') is None or summary.get('sortino') is None:
                monthly = _build_monthly_returns(trades_df)
                sharpe, sortino = _compute_sharpe_sortino(monthly)
                if sharpe is not None and summary.get('sharpe') is None:
                    summary['sharpe'] = sharpe
                if sortino is not None and summary.get('sortino') is None:
                    summary['sortino'] = sortino
            
            if summary.get('max_dd_pct') is None:
                max_dd = _compute_max_dd_pct(equity_csv, trades_df)
                if max_dd is not None:
                    summary['max_dd_pct'] = max_dd
        
        rows.append({
            'pair': pair,
            'c1': c1,
            'trades': int(trades_count),
            'roi_pct': float(summary.get('roi_pct', 0.0) or 0.0),
            'win_rate_pct': float(summary.get('win_rate_pct', 0.0) or 0.0),
            'avg_r': float(summary.get('avg_r', 0.0) or 0.0),
            'max_dd_pct': float(summary.get('max_dd_pct', 0.0) or 0.0),
            'sharpe': float(summary.get('sharpe', 0.0) or 0.0),
            'sortino': float(summary.get('sortino', 0.0) or 0.0),
        })
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    # Sort: pair ASC, roi_pct DESC, win_rate_pct DESC, max_dd_pct ASC
    df = df.sort_values(
        ['pair', 'roi_pct', 'win_rate_pct', 'max_dd_pct'],
        ascending=[True, False, False, True]
    )
    
    return df


def print_preview(df: pd.DataFrame, top_n: int) -> None:
    if df.empty:
        print('No data to preview.')
        return
    
    # Take top N overall (not per pair)
    preview = df.head(top_n).copy()
    
    # Format columns for display
    print('\n' + '=' * 100)
    print(f'Top {len(preview)} rows (overall):')
    print('=' * 100)
    
    # Header
    header = f"{'pair':<12} {'c1':<25} {'trades':>8} {'roi_pct':>10} {'win_rate_pct':>12} {'avg_r':>8} {'max_dd_pct':>12} {'sharpe':>8} {'sortino':>8}"
    print(header)
    print('-' * 100)
    
    # Rows
    for _, row in preview.iterrows():
        line = (
            f"{str(row['pair']):<12} "
            f"{str(row['c1']):<25} "
            f"{int(row['trades']):>8} "
            f"{float(row['roi_pct']):>10.2f} "
            f"{float(row['win_rate_pct']):>12.2f} "
            f"{float(row['avg_r']):>8.3f} "
            f"{float(row['max_dd_pct']):>12.2f} "
            f"{float(row['sharpe']):>8.3f} "
            f"{float(row['sortino']):>8.3f}"
        )
        print(line)
    print('=' * 100 + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description='Build consolidated indicator table from C1-only exits results')
    ap.add_argument('--root', default='results/c1_only_exits', type=str, help='Root directory to scan')
    ap.add_argument('--out', default='results', type=str, help='Output directory')
    ap.add_argument('--top', default=20, type=int, help='Number of top rows to preview')
    ap.add_argument('--pair', action='append', help='Filter by pair (can be passed multiple times)')
    args = ap.parse_args()
    
    root = Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    
    pair_filter = set(args.pair) if args.pair else None
    
    df = build_table(root, pair_filter)
    
    if df.empty:
        print('No valid runs found.')
        return
    
    out_csv = outdir / 'indicator_table_full.csv'
    df.to_csv(out_csv, index=False)
    print(f'✅ Wrote: {out_csv} ({len(df)} rows)')
    
    print_preview(df, args.top)


if __name__ == '__main__':
    main()

