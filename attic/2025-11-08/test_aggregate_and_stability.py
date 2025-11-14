from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


def _results_root() -> Path:
    return Path('results') / 'c1_only_exits'


def test_aggregate_outputs_shape(tmp_path: Path):
    root = _results_root()
    outdir = Path('results')

    # If no real results, synthesize a tiny structure in tmp and run there
    if not root.exists():
        synth = tmp_path / 'c1_only_exits'
        pair = synth / 'EUR_USD' / 'C1_TEST'
        pair.mkdir(parents=True)
        (pair / 'summary.txt').write_text('ROI%: 12.5\nWin Rate %: 55\nAvg R: 0.4\nMax DD %: -8.0\nTrades: 150\n', encoding='utf-8')
        (pair / 'trades.csv').write_text('id,entry_date,exit_date,r_multiple\n1,2020-01-01,2020-01-05,1.0\n', encoding='utf-8')
        root = synth
        outdir = tmp_path

    # Run aggregator
    import subprocess, sys
    subprocess.check_call([sys.executable, 'tools/aggregate_c1_only.py', '--root', str(root), '--outdir', str(outdir)])

    all_csv = Path(outdir) / 'c1_only_exits_all_runs.csv'
    overall_csv = Path(outdir) / 'leaderboard_c1_only_exits_overall.csv'
    assert all_csv.exists()
    assert overall_csv.exists()

    all_df = pd.read_csv(all_csv)
    for col in ['pair', 'c1', 'trades', 'roi_pct', 'win_rate_pct', 'avg_r', 'max_dd_pct']:
        assert col in all_df.columns

    lb_df = pd.read_csv(overall_csv)
    assert 'c1' in lb_df.columns


def test_stability_outputs_shape(tmp_path: Path):
    root = _results_root()
    out_csv = Path('results') / 'stability' / 'c1_only_exits_stability.csv'

    if not root.exists():
        synth = tmp_path / 'c1_only_exits'
        pair = synth / 'EUR_USD' / 'C1_TEST'
        pair.mkdir(parents=True)
        (pair / 'trades.csv').write_text(
            'entry_date,exit_date,r_multiple\n'
            '2020-01-01,2020-01-10,1.0\n'
            '2020-02-01,2020-02-05,-0.5\n'
            '2021-03-01,2021-03-12,0.7\n',
            encoding='utf-8',
        )
        root = synth
        out_csv = tmp_path / 'stability' / 'c1_only_exits_stability.csv'

    import subprocess, sys
    subprocess.check_call([sys.executable, 'analytics/stability_scan.py', '--root', str(root), '--out', str(out_csv), '--min-trades', '1'])

    if out_csv.exists():
        df = pd.read_csv(out_csv)
        for col in ['pair', 'c1', 'trades', 'years', 'profitable_years_pct', 'monthly_cv', 'low_sample_flag']:
            assert col in df.columns


