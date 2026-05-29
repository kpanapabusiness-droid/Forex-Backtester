"""Tests for the ``--quick-test`` diagnostic flag.

Quick-test bypasses the wait-for-next-H4-close sleep and runs exactly one
cycle immediately against the most-recently-closed H4 bar, so the staleness
investigation can iterate without waiting up to 4h for the next live
boundary. It must still exercise the real production path: anchor probe,
heartbeat, identical output filenames.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import deployment.sidecar.mt5_data_fetcher as mt5_data_fetcher
import deployment.sidecar.sidecar as sidecar_mod
from deployment.sidecar.__main__ import main as cli_main
from deployment.sidecar.config import load_sidecar_config
from deployment.sidecar.sidecar import initialize_and_run, main_loop
from deployment.sidecar.state_manager import load_state


def test_quick_test_runs_one_iteration_without_sleeping(
    fake_mt5, winning_config_path, sidecar_root
):
    cfg = load_sidecar_config(winning_config_path, None, sidecar_root)
    sleeps: list[float] = []
    main_loop(
        cfg,
        fake_mt5,
        quick_test=True,
        sleep_func=sleeps.append,
        now_func=lambda: datetime(2026, 5, 27, 15, 17, tzinfo=timezone.utc),
    )
    # The defining property: no wait-for-next-H4 sleep at all.
    assert sleeps == []
    # One cycle actually ran: heartbeat + state were written.
    hb = json.loads(cfg.heartbeat_path.read_text(encoding="utf-8"))
    assert hb["last_loop_complete_utc"]
    state = load_state(cfg.state_path)
    for pair in cfg.pairs:
        assert pair in state.last_processed_bar_utc


def test_quick_test_exits_after_one_iteration(
    fake_mt5, winning_config_path, sidecar_root, monkeypatch
):
    cfg = load_sidecar_config(winning_config_path, None, sidecar_root)
    calls: list[int] = []

    real_iter = sidecar_mod._loop_iteration

    def _counting_iter(*args, **kwargs):
        calls.append(1)
        return real_iter(*args, **kwargs)

    monkeypatch.setattr(sidecar_mod, "_loop_iteration", _counting_iter)
    main_loop(
        cfg,
        fake_mt5,
        quick_test=True,
        sleep_func=lambda _s: None,
        now_func=lambda: datetime(2026, 5, 27, 15, 17, tzinfo=timezone.utc),
    )
    assert sum(calls) == 1


def test_quick_test_runs_anchor_probe_and_heartbeat(
    fake_mt5, winning_config_path, sidecar_root, monkeypatch
):
    """Through the real entry point: anchor probe still fires, heartbeat
    still written. (conftest's fake panel is UTC-anchored so the probe
    passes.)"""
    cfg = load_sidecar_config(winning_config_path, None, sidecar_root)

    # Inject the fake MT5 instead of importing the Windows-only library.
    monkeypatch.setattr(mt5_data_fetcher, "import_mt5", lambda: fake_mt5)

    probe_calls: list[str] = []
    real_probe = sidecar_mod.verify_mt5_h4_alignment

    def _spy_probe(*args, **kwargs):
        probe_calls.append(kwargs.get("convention", "utc"))
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(sidecar_mod, "verify_mt5_h4_alignment", _spy_probe)

    initialize_and_run(cfg, quick_test=True)

    # Anchor probe ran (broker-grid correctness gate is never skipped).
    assert len(probe_calls) == 1
    # Heartbeat written by the single cycle.
    hb = json.loads(cfg.heartbeat_path.read_text(encoding="utf-8"))
    assert hb["last_loop_complete_utc"]
    # MT5 was cleanly shut down.
    assert fake_mt5.shutdown_calls == 1


def test_quick_test_cli_implies_single_iteration(
    fake_mt5, winning_config_path, sidecar_root, monkeypatch
):
    """``--quick-test`` with no ``--iterations`` must still run exactly one
    cycle. Verified at the CLI wiring: iterations=1 + quick_test=True reach
    initialize_and_run."""
    captured: dict[str, object] = {}

    def _fake_run(cfg, *, iterations=None, quick_test=False):
        captured["iterations"] = iterations
        captured["quick_test"] = quick_test

    monkeypatch.setattr("deployment.sidecar.__main__.initialize_and_run", _fake_run)

    rc = cli_main(
        [
            "--winning-config",
            str(winning_config_path),
            "--sidecar-root",
            str(sidecar_root),
            "--quick-test",
        ]
    )
    assert rc == 0
    assert captured["quick_test"] is True
    assert captured["iterations"] == 1


def test_quick_test_works_under_eet_convention(
    fake_mt5, tmp_path, sidecar_root, monkeypatch
):
    """Both boundary conventions must work with --quick-test. EET path: the
    prev-close log helper and loop run without error."""
    content = (winning_eet_config := tmp_path / "winning_config.yaml")
    content.write_text(
        """
arc_name: l_arc_10_v3.0.2_eet
verdict: PASS-DEPLOYABLE
boundary_convention: 5ers_eet

signal:
  name: dlr_d1_swing_low_rejection_long
  module: signals.lchar_dlr_long
  version: v0.1

direction: long

architecture:
  name: A1
  variant: system_level_filter

stop_loss:
  type: atr_multiple
  atr_period: 14
  multiplier: 3.5
  anchor: mid
  reference: entry_price

exit_policy:
  name: sl_partial_close_1r_runner_trail
  partial_close_at: 1.0
  partial_close_fraction: 0.5
  runner_trail_atr_below_peak: 1.0
  update_frequency: bar_close

time_exit:
  max_bars: 240

timeframes:
  primary: H4
  anchor: D1

pairs:
  - EURUSD
  - GBPUSD

risk:
  r_safe_pct: 0.004336

fills:
  entry_long: open_ask
  spread_source: histdata_m1_bid_ask
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_sidecar_config(winning_eet_config, None, sidecar_root)
    sleeps: list[float] = []
    main_loop(
        cfg,
        fake_mt5,
        quick_test=True,
        sleep_func=sleeps.append,
        now_func=lambda: datetime(2026, 5, 27, 15, 17, tzinfo=timezone.utc),
    )
    assert sleeps == []
    hb = json.loads(cfg.heartbeat_path.read_text(encoding="utf-8"))
    assert hb["last_loop_complete_utc"]
