"""Schema + invariant checks for v1.3 results/kh24/trades_paths.csv.

This is a first-emission sanity test, not a byte-level regression. Verifies:
  - the expected columns are present
  - per-trade bar_offset starts at 0
  - mfe_so_far_r is monotone non-decreasing within trade
  - mae_so_far_r is monotone non-increasing within trade
  - every trade_id in trades_paths.csv has a matching row in trades_all.csv
  - most trades have forward-window rows beyond bars_held
  - most trades reach bar_offset=240 (the configured forward window cap)
"""
import os

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS_FILE = os.path.join(REPO_ROOT, "results", "kh24", "trades_paths.csv")
TRADES_FILE = os.path.join(REPO_ROOT, "results", "kh24", "trades_all.csv")

EXPECTED_COLUMNS = {
    "trade_id", "pair", "bar_offset",
    "high_r", "low_r", "close_r",
    "mfe_so_far_r", "mae_so_far_r", "is_held",
}

pytestmark = pytest.mark.skipif(
    not (os.path.exists(PATHS_FILE) and os.path.exists(TRADES_FILE)),
    reason=(
        "results/kh24/trades_paths.csv or trades_all.csv not present — "
        "run the KH-24 backtester first: "
        "python scripts/phase_kgl_v2_4h_wfo.py -c configs/wfo_kh24.yaml"
    ),
)


def test_schema():
    df = pd.read_csv(PATHS_FILE)
    assert EXPECTED_COLUMNS.issubset(set(df.columns)), (
        f"Missing required columns: {EXPECTED_COLUMNS - set(df.columns)}"
    )


def test_bar_offset_starts_at_zero_per_trade():
    df = pd.read_csv(PATHS_FILE)
    per_trade_min = df.groupby("trade_id")["bar_offset"].min()
    assert (per_trade_min == 0).all(), (
        f"Some trades do not start at bar_offset=0: "
        f"{per_trade_min[per_trade_min != 0].head().to_dict()}"
    )


def test_mfe_so_far_monotone_per_trade():
    df = pd.read_csv(PATHS_FILE).sort_values(["trade_id", "bar_offset"])
    diff = df.groupby("trade_id")["mfe_so_far_r"].diff().dropna()
    # Floating-point tolerance: allow tiny negative drift from re-serialisation.
    assert (diff >= -1e-12).all(), (
        "mfe_so_far_r is not monotone non-decreasing within trade. "
        f"Worst negative diff: {diff.min()}"
    )


def test_mae_so_far_monotone_per_trade():
    df = pd.read_csv(PATHS_FILE).sort_values(["trade_id", "bar_offset"])
    diff = df.groupby("trade_id")["mae_so_far_r"].diff().dropna()
    assert (diff <= 1e-12).all(), (
        "mae_so_far_r is not monotone non-increasing within trade. "
        f"Worst positive diff: {diff.max()}"
    )


def test_trade_ids_match_trades_all():
    """Every trade_id in trades_paths.csv must exist in trades_all.csv."""
    paths = pd.read_csv(PATHS_FILE)
    trades = pd.read_csv(TRADES_FILE)
    assert set(paths["trade_id"]) == set(trades["trade_id"]), (
        f"trade_id mismatch: "
        f"paths-only={set(paths['trade_id']) - set(trades['trade_id'])} "
        f"trades-only={set(trades['trade_id']) - set(paths['trade_id'])}"
    )


def test_forward_window_present():
    """Most trades should have bar_offset rows beyond bars_held."""
    paths = pd.read_csv(PATHS_FILE)
    trades = pd.read_csv(TRADES_FILE)[["trade_id", "bars_held"]]
    merged = paths.merge(trades, on="trade_id")
    forward_rows = merged[merged["bar_offset"] > merged["bars_held"]]
    n_with_forward = forward_rows["trade_id"].nunique()
    n_total = trades.shape[0]
    assert n_with_forward / n_total > 0.5, (
        f"Forward window appears missing or empty: only "
        f"{n_with_forward}/{n_total} trades have forward-window rows."
    )


def test_max_bar_offset_capped_at_240_or_end_of_data():
    """Most trades should hit exactly bar_offset=240 (configured cap)."""
    df = pd.read_csv(PATHS_FILE)
    per_trade_max = df.groupby("trade_id")["bar_offset"].max()
    pct_at_240 = (per_trade_max == 240).mean()
    assert pct_at_240 > 0.8, (
        f"Expected most trades to have full 240-bar forward window; "
        f"only {pct_at_240:.1%} did. Sample non-240 maxes: "
        f"{per_trade_max[per_trade_max != 240].head().to_dict()}"
    )


def test_is_held_consistent_with_bars_held():
    """is_held=1 iff bar_offset <= bars_held (spec definition)."""
    paths = pd.read_csv(PATHS_FILE)
    trades = pd.read_csv(TRADES_FILE)[["trade_id", "bars_held"]]
    merged = paths.merge(trades, on="trade_id")
    expected_is_held = (merged["bar_offset"] <= merged["bars_held"]).astype(int)
    mismatch = (merged["is_held"] != expected_is_held).sum()
    assert mismatch == 0, (
        f"{mismatch} per-bar rows have is_held inconsistent with bar_offset <= bars_held."
    )
