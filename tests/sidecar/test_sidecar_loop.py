"""Integration test for the main loop: one iteration end-to-end with a fake MT5."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from deployment.sidecar.config import load_sidecar_config
from deployment.sidecar.sidecar import _loop_iteration, main_loop, verify_mt5_h4_alignment
from deployment.sidecar.state_manager import load_state


def test_loop_iteration_writes_heartbeat_and_state(
    fake_mt5, winning_config_path, sidecar_root
):
    cfg = load_sidecar_config(winning_config_path, None, sidecar_root)
    state = load_state(cfg.state_path)
    ok, pairs_ok = _loop_iteration(cfg, fake_mt5, state)
    assert ok is True
    assert set(pairs_ok) == set(cfg.pairs)
    # Heartbeat file written.
    hb = json.loads(cfg.heartbeat_path.read_text(encoding="utf-8"))
    assert hb["last_loop_complete_utc"]
    # State has last_processed_bar_utc populated for each pair.
    state_after = load_state(cfg.state_path)
    for pair in cfg.pairs:
        assert pair in state_after.last_processed_bar_utc


def test_main_loop_runs_fixed_iterations(
    fake_mt5, winning_config_path, sidecar_root
):
    cfg = load_sidecar_config(winning_config_path, None, sidecar_root)
    # Force the loop to wake immediately by stubbing the H4-close
    # computation to "now"-ish.
    fake_now = datetime(2026, 5, 27, 15, 59, 59, tzinfo=timezone.utc)
    sleeps: list[float] = []
    main_loop(
        cfg,
        fake_mt5,
        iterations=2,
        sleep_func=sleeps.append,
        now_func=lambda: fake_now,
    )
    state = load_state(cfg.state_path)
    assert state.restart_count >= 1
    # Two iterations × at least one sleep each (the "wait until close") = 2 sleeps.
    assert len(sleeps) == 2


def test_verify_mt5_h4_alignment_accepts_anchored_bars(fake_mt5):
    # The conftest's FakeMt5 generates UTC-anchored bars.
    verify_mt5_h4_alignment(fake_mt5, probe_symbol="EURUSD", probe_count=6)


def test_verify_mt5_h4_alignment_rejects_drifted_bars(fake_mt5):
    """If the broker emits bars at e.g. 21:00 / 01:00 UTC (EET-anchored),
    the probe must refuse."""
    from tests.sidecar.conftest import _synth_h4_panel
    import pandas as pd

    drifted = _synth_h4_panel(n_bars=24, start_iso="2026-01-01T01:00:00")
    fake_mt5.h4_panel = drifted
    with pytest.raises(Exception, match="non-UTC-anchored bars"):
        verify_mt5_h4_alignment(fake_mt5, probe_symbol="EURUSD", probe_count=6)
