# tests/test_summary_drawdown.py
# Integration: summary writer includes Max Drawdown (%) when equity curve exists

from pathlib import Path

import pandas as pd

from core.utils import summarize_results


def test_summary_includes_max_drawdown_when_equity_exists(tmp_path):
    results_dir = Path(tmp_path)
    trades = pd.DataFrame(
        {
            "pair": ["AUD_JPY"],
            "entry_date": ["2020-01-01"],
            "exit_date": ["2020-01-02"],
            "pnl": [100.0],
            "pnl_dollars": [100.0],
            "win": [True],
            "loss": [False],
            "scratch": [False],
        }
    )
    trades.to_csv(results_dir / "trades.csv", index=False)
    equity = pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02", "2020-01-03"], "equity": [10000.0, 11000.0, 9900.0]}
    )
    equity.to_csv(results_dir / "equity_curve.csv", index=False)

    text, metrics = summarize_results(results_dir, starting_balance=10000.0)

    assert "Max Drawdown (%)" in text
    assert "max_dd_pct" in metrics
    assert metrics["max_dd_pct"] < 0
