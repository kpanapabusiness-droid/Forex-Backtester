import os
import shutil
import tempfile

from core import backtester


def test_write_results_creates_files():
    tmpdir = tempfile.mkdtemp()
    try:
        # minimal trades row
        trades = [
            {
                "pair": "AUD_JPY",
                "entry_date": "2020-01-03",
                "entry_price": 100.0,
                "direction": "long",
                "direction_int": 1,
                "atr_at_entry_price": 1.0,
                "atr_at_entry_pips": 10.0,
                "lots_total": 0.1,
                "lots_half": 0.05,
                "lots_runner": 0.05,
                "pip_value_per_lot": 10.0,
                "tp1_level": 101.0,
                "sl_level": 98.5,
                "be_level": 100.0,
                "ts_active": False,
                "ts_level": None,
                "entry_idx": 3,
                "exit_date": "2020-01-05",
                "exit_price": 102.0,
                "exit_reason": "indicator",
                "pnl": 123.45,
                "win": True,
                "loss": False,
                "scratch": False,
            }
        ]
        backtester.write_results(trades, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "trades.csv"))
        assert os.path.exists(os.path.join(tmpdir, "summary.txt"))
    finally:
        shutil.rmtree(tmpdir)
