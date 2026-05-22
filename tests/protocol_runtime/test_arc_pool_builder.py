"""Tests for arc_pool_builder.

  - Pool size > 0 for a synthetic periodic signal
  - Determinism: two-run sha256 byte-identical
  - Integrity rows include the expected checks
  - Path table covers every trade
"""

from __future__ import annotations

import pytest

from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from tests.protocol_runtime._fixtures import build_synthetic_arc_pool_inputs


def _run_builder():
    signal, panels = build_synthetic_arc_pool_inputs(n_bars=600)
    cfg = ArcPoolConfig(arc_name="synth_test", sl_atr_mult=2.0, hold_bars=24)
    return build_arc_pool(signal, panels, cfg)


def test_pool_nonempty_on_synthetic_data() -> None:
    pool = _run_builder()
    assert len(pool.trades) > 0
    assert len(pool.paths) > 0


def test_two_run_determinism() -> None:
    pool_1 = _run_builder()
    pool_2 = _run_builder()
    assert pool_1.pool_sha256 == pool_2.pool_sha256


def test_integrity_rows_present() -> None:
    pool = _run_builder()
    checks = {row.check for row in pool.integrity}
    assert "pool_size_min" in checks
    assert "coverage_window" in checks
    assert "lookahead_declared_lineage" in checks


def test_path_table_covers_every_trade() -> None:
    pool = _run_builder()
    trade_ids = set(pool.trades["trade_id"].astype(int))
    path_ids = set(pool.paths["trade_id"].astype(int))
    assert trade_ids == path_ids


def test_pool_schema_columns() -> None:
    pool = _run_builder()
    expected = {
        "pair", "trade_id", "signal_time", "entry_time", "entry_price",
        "atr_at_signal", "sl_at_entry_price", "exit_time", "exit_price",
        "exit_reason", "bars_held", "final_r", "mfe_r", "mae_r",
    }
    assert set(pool.trades.columns) >= expected
