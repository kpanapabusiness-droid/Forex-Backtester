# tests/test_trade_shuffling.py
# Phase 5.1 — Trade shuffling: unit and integration tests (non-research)

import pandas as pd
import pytest

from analytics.trade_shuffling import max_consecutive_losses, run_trade_shuffling
from scripts.phase5_trade_shuffling import load_oos_trades_from_wfo


class TestMaxConsecutiveLosses:
    """Unit tests for max_consecutive_losses on known pnl sequences."""

    def test_empty_returns_zero(self):
        assert max_consecutive_losses(pd.Series(dtype=float)) == 0

    def test_all_wins_returns_zero(self):
        pnl = pd.Series([10.0, 20.0, 5.0])
        assert max_consecutive_losses(pnl) == 0

    def test_single_loss_returns_one(self):
        pnl = pd.Series([10.0, -5.0, 20.0])
        assert max_consecutive_losses(pnl) == 1

    def test_two_consecutive_losses(self):
        pnl = pd.Series([10.0, -5.0, -3.0, 20.0])
        assert max_consecutive_losses(pnl) == 2

    def test_two_streaks_takes_max(self):
        pnl = pd.Series([-1.0, -2.0, 10.0, -3.0, -4.0, -5.0, 1.0])
        assert max_consecutive_losses(pnl) == 3

    def test_all_losses(self):
        pnl = pd.Series([-1.0, -2.0, -3.0])
        assert max_consecutive_losses(pnl) == 3

    def test_zero_pnl_not_loss(self):
        pnl = pd.Series([-1.0, 0.0, -1.0])
        assert max_consecutive_losses(pnl) == 1


class TestTradeShufflingDeterministic:
    """Same seed produces same summary stats."""

    def test_deterministic_with_seed(self, tmp_path):
        trades_df = pd.DataFrame({"pnl": [100.0, -50.0, 80.0, -30.0, 20.0]})
        s1 = run_trade_shuffling(
            trades_df, starting_balance=10_000.0, n_sims=100, seed=42, out_dir=tmp_path / "a"
        )
        s2 = run_trade_shuffling(
            trades_df, starting_balance=10_000.0, n_sims=100, seed=42, out_dir=tmp_path / "b"
        )
        assert s1["max_dd_pct_p50"] == s2["max_dd_pct_p50"]
        assert s1["max_dd_pct_worst"] == s2["max_dd_pct_worst"]
        assert s1["max_consec_losses_p50"] == s2["max_consec_losses_p50"]
        runs1 = pd.read_csv(tmp_path / "a" / "trade_shuffling_runs.csv")
        runs2 = pd.read_csv(tmp_path / "b" / "trade_shuffling_runs.csv")
        assert (runs1["max_dd_pct"] == runs2["max_dd_pct"]).all()

    def test_different_seed_different_stats(self, tmp_path):
        trades_df = pd.DataFrame({"pnl": [100.0, -50.0, 80.0, -30.0, 20.0] * 4})
        run_trade_shuffling(
            trades_df, starting_balance=10_000.0, n_sims=50, seed=1, out_dir=tmp_path / "a"
        )
        run_trade_shuffling(
            trades_df, starting_balance=10_000.0, n_sims=50, seed=999, out_dir=tmp_path / "b"
        )
        runs1 = pd.read_csv(tmp_path / "a" / "trade_shuffling_runs.csv")
        runs2 = pd.read_csv(tmp_path / "b" / "trade_shuffling_runs.csv")
        assert not (runs1["max_dd_pct"].values == runs2["max_dd_pct"].values).all()


class TestTradeShufflingIntegration:
    """Integration: output files, max_dd_pct <= 0, ruin in [0,1], deterministic."""

    def test_output_files_created(self, tmp_path):
        trades_df = pd.DataFrame({"pnl": [50.0, -20.0, 30.0, -10.0, 40.0]})
        run_trade_shuffling(
            trades_df, starting_balance=10_000.0, n_sims=100, seed=123, out_dir=tmp_path
        )
        assert (tmp_path / "trade_shuffling_runs.csv").exists()
        assert (tmp_path / "trade_shuffling_summary.json").exists()
        assert (tmp_path / "trade_shuffling_summary.txt").exists()

    def test_max_dd_pct_all_non_positive(self, tmp_path):
        trades_df = pd.DataFrame({"pnl": [50.0, -20.0, 30.0, -10.0, 40.0]})
        run_trade_shuffling(
            trades_df, starting_balance=10_000.0, n_sims=100, seed=123, out_dir=tmp_path
        )
        runs = pd.read_csv(tmp_path / "trade_shuffling_runs.csv")
        assert (runs["max_dd_pct"] <= 0).all()

    def test_ruin_probs_in_zero_one(self, tmp_path):
        trades_df = pd.DataFrame({"pnl": [50.0, -20.0, 30.0, -10.0, 40.0]})
        summary = run_trade_shuffling(
            trades_df, starting_balance=10_000.0, n_sims=100, seed=123, out_dir=tmp_path
        )
        rp = summary["ruin_probs"]
        assert 0 <= rp["P_dd_le_10"] <= 1
        assert 0 <= rp["P_dd_le_15"] <= 1
        assert 0 <= rp["P_dd_le_20"] <= 1

    def test_empty_trades_graceful(self, tmp_path):
        summary = run_trade_shuffling(
            pd.DataFrame(), starting_balance=10_000.0, n_sims=100, seed=123, out_dir=tmp_path
        )
        assert summary["n_trades"] == 0
        assert (tmp_path / "trade_shuffling_runs.csv").exists()
        assert (tmp_path / "trade_shuffling_summary.json").exists()
        runs = pd.read_csv(tmp_path / "trade_shuffling_runs.csv")
        assert len(runs) == 0


class TestLoadOosTradesFromWfo:
    """WFO OOS trade discovery: direct run dir, parent with runs, no trades raises."""

    def test_case_a_direct_run_dir_with_folds_loads_trades(self, tmp_path):
        """wfo_results_dir points at a run with fold_*/out_of_sample/trades.csv → loads trades."""
        run_dir = tmp_path / "run_01"
        run_dir.mkdir()
        (run_dir / "fold_01" / "out_of_sample").mkdir(parents=True)
        trades_path = run_dir / "fold_01" / "out_of_sample" / "trades.csv"
        pd.DataFrame({"pnl": [10.0, -5.0], "pair": ["EUR_USD", "EUR_USD"]}).to_csv(
            trades_path, index=False
        )
        df, resolved_dir, n_folds = load_oos_trades_from_wfo(run_dir)
        assert len(df) == 2
        assert resolved_dir.resolve() == run_dir.resolve()
        assert n_folds == 1

    def test_case_b_parent_with_multiple_runs_chooses_latest(self, tmp_path):
        """wfo_results_dir points at parent with run subdirs → latest by mtime is chosen."""
        parent = tmp_path / "wfo"
        parent.mkdir()
        run_old = parent / "20260101_120000"
        run_new = parent / "20260102_120000"
        run_old.mkdir()
        run_new.mkdir()
        for run_dir in (run_old, run_new):
            (run_dir / "fold_01" / "out_of_sample").mkdir(parents=True)
            pd.DataFrame({"pnl": [1.0], "pair": ["X"]}).to_csv(
                run_dir / "fold_01" / "out_of_sample" / "trades.csv", index=False
            )
        import time
        time.sleep(0.02)
        (run_new / "touch").write_text("")
        df, resolved_dir, n_folds = load_oos_trades_from_wfo(parent)
        assert len(df) >= 1
        assert resolved_dir.resolve() == run_new.resolve()
        assert n_folds == 1

    def test_case_c_no_folds_raises(self, tmp_path):
        """No fold_*/out_of_sample/trades.csv → ValueError (not silent 0 trades)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError) as exc_info:
            load_oos_trades_from_wfo(empty_dir)
        msg = exc_info.value.args[0]
        assert "No OOS trades" in msg or "no OOS" in msg.lower()
        assert "--wfo-results-dir" in msg

    def test_case_c_no_run_subdirs_raises(self, tmp_path):
        """Parent dir exists but has no run subdirs with folds → ValueError."""
        parent = tmp_path / "wfo"
        parent.mkdir()
        (parent / "other_file.txt").write_text("x")
        with pytest.raises(ValueError) as exc_info:
            load_oos_trades_from_wfo(parent)
        assert "No OOS trades" in exc_info.value.args[0] or "no OOS" in exc_info.value.args[0].lower()

    def test_nonexistent_dir_raises(self, tmp_path):
        """Nonexistent wfo_results_dir → ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_oos_trades_from_wfo(tmp_path / "does_not_exist")
        assert "does not exist" in exc_info.value.args[0]
        assert "--wfo-results-dir" in exc_info.value.args[0]

    def test_allow_empty_trades_returns_empty_df(self, tmp_path):
        """When allow_empty=True, empty fold CSVs return (empty df, run_dir, n_folds) without raising."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "fold_01" / "out_of_sample").mkdir(parents=True)
        pd.DataFrame(columns=["pnl", "pair"]).to_csv(
            run_dir / "fold_01" / "out_of_sample" / "trades.csv", index=False
        )
        df, resolved, n_folds = load_oos_trades_from_wfo(run_dir, allow_empty=True)
        assert df.empty
        assert resolved.resolve() == run_dir.resolve()
        assert n_folds == 1
