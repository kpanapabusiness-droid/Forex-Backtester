"""Tests for arc_pool_builder.

  - Pool size > 0 for a synthetic periodic signal
  - Determinism: two-run sha256 byte-identical
  - Integrity rows include the expected checks
  - Path table covers every trade
"""

from __future__ import annotations

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
        # SL-honest meta-label provenance (HONEST_ENGINE_SWEEP.md Part D).
        "bars_to_1r_mfe",
    }
    assert set(pool.trades.columns) >= expected


def test_bars_to_1r_mfe_is_sl_honest() -> None:
    """The emitted bars_to_1r_mfe is the take-the-loss producer's output:
    where set it is (a) <= bars_held and (b) backed by mfe_r >= 1.0 (the
    high genuinely reached +1R); a same-bar +1R/SL trade leaves it NaN even
    though raw mfe_r touched +1R. See core.sim.honest_label."""
    import numpy as np

    pool = _run_builder()
    df = pool.trades
    assert "bars_to_1r_mfe" in df.columns
    reached = df["bars_to_1r_mfe"].notna()
    # Every honest +1R reach must be at/within the holding window.
    assert bool((df.loc[reached, "bars_to_1r_mfe"] <= df.loc[reached, "bars_held"]).all())
    # And must be corroborated by the raw excursion having touched +1R.
    assert bool((df.loc[reached, "mfe_r"] >= 1.0 - 1e-9).all())
    # A hard_sl trade may have raw mfe_r >= 1.0 yet NaN bars_to_1r_mfe (the
    # +1R high landed on the stop bar → take-the-loss says "not reached").
    # The reverse — a set bars_to_1r_mfe whose mfe_r < 1R — must never occur.
    assert not bool(((df["bars_to_1r_mfe"].notna()) & (df["mfe_r"] < 1.0 - 1e-9)).any())
