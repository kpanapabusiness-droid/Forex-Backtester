"""Tests for the engine-generalisation PR (signal adapter, timeframe
pluggability, time-exit cap, spread-floor wiring).

Byte-identical KH-24 is verified manually (full WFO is too slow for CI);
this file covers the new code paths via unit / focused integration tests.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.signal_adapter import (  # noqa: E402
    ALLOWED_AUX_KEYS,
    import_class,
    validate_aux_declaration,
)
from core.spread_floor import (  # noqa: E402
    STATE_CFG_KEY,
    SpreadFloorState,
    apply_spread_floor_to_pips,
    load_spread_floor,
)
from signals.kb_exhaustion_bar_adapter import KbExhaustionBarAdapter  # noqa: E402

# ─── Part 1: SignalAdapter protocol + helpers ────────────────────────────────


class _StubAdapter:
    """Minimal SignalAdapter for tests — supplies a constant mask."""

    def __init__(self, aux_keys: list[str] | None = None) -> None:
        self._aux = list(aux_keys) if aux_keys is not None else []

    def required_aux_data(self) -> list[str]:
        return list(self._aux)

    def compute_signal_mask(self, df, aux):
        return np.zeros(len(df), dtype=int), {}

    def compute_signal_atr(self, df, idx, aux):
        return 1.0

    def compute_entry_features(self, df, idx, aux):
        return {}

    def per_trade_init(self, trade, df, idx, aux):
        trade.setdefault("test_field", True)


def test_import_class_valid():
    cls = import_class("signals.kb_exhaustion_bar_adapter:KbExhaustionBarAdapter")
    assert cls is KbExhaustionBarAdapter


def test_import_class_missing_colon():
    with pytest.raises(ImportError, match="module.path:ClassName"):
        import_class("not_a_valid_spec")


def test_import_class_missing_module():
    with pytest.raises(ImportError, match="cannot import module"):
        import_class("does.not.exist:Whatever")


def test_import_class_missing_class():
    with pytest.raises(ImportError, match="class .* not found"):
        import_class("signals.kb_exhaustion_bar_adapter:NoSuchClass")


def test_validate_aux_accepts_known_keys():
    validate_aux_declaration(["h1", "d1"], "test")  # must not raise
    validate_aux_declaration([], "test")
    validate_aux_declaration(["d1"], "test")


def test_validate_aux_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown aux keys"):
        validate_aux_declaration(["h1", "moon_phase"], "test_spec")


def test_allowed_aux_keys_contract():
    # Doc invariant — engine code paths depend on these two keys exactly.
    assert ALLOWED_AUX_KEYS == frozenset({"h1", "d1"})


def test_kb_adapter_required_aux():
    a = KbExhaustionBarAdapter()
    assert a.required_aux_data() == ["h1", "d1"]


def test_kb_adapter_accepts_empty_kwargs():
    # Engine passes signal_adapter_kwargs as **{} when block is empty.
    KbExhaustionBarAdapter(**{})  # must not raise


def test_kb_adapter_per_trade_init_stamps_defaults():
    a = KbExhaustionBarAdapter()
    n = 20
    df = pd.DataFrame(
        {
            "open":  [1.0] * n,
            "high":  [1.1] * n,
            "low":   [0.9] * n,
            "close": [1.05] * n,
        }
    )
    trade: dict = {}
    a.per_trade_init(trade, df, idx=5, aux={})
    assert trade["kh13_triggered"] is False
    assert trade["kh14_triggered"] is False
    # Entry bar idx+1 has close > open → first_bar_dir = 1
    assert trade["first_bar_dir"] == 1


def test_kb_adapter_per_trade_init_preserves_existing():
    a = KbExhaustionBarAdapter()
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
    trade = {"kh13_triggered": True}  # pre-set
    a.per_trade_init(trade, df, idx=0, aux={})
    # setdefault must NOT overwrite existing values
    assert trade["kh13_triggered"] is True


# ─── Part 4: spread_floor (the engine wiring uses it as a no-op for KH-24) ──


def test_spread_floor_disabled_when_block_absent():
    state = load_spread_floor({})
    assert isinstance(state, SpreadFloorState)
    assert state.enabled is False


def test_spread_floor_disabled_explicit():
    state = load_spread_floor({"spread_floor": {"enabled": False}})
    assert state.enabled is False


def test_spread_floor_enabled_requires_source_and_sha():
    with pytest.raises(ValueError, match="'source' and 'expected_body_sha256'"):
        load_spread_floor({"spread_floor": {"enabled": True}})


def test_spread_floor_enabled_missing_source_raises():
    cfg = {
        "spread_floor": {
            "enabled": True,
            "source": "nonexistent/path/spreads.yaml",
            "expected_body_sha256": "0" * 64,
        }
    }
    with pytest.raises(FileNotFoundError):
        load_spread_floor(cfg)


def test_apply_spread_floor_noop_when_state_missing():
    # cfg lacks STATE_CFG_KEY → no floor, returns input unchanged.
    assert apply_spread_floor_to_pips({}, "AUD_USD", 1.5) == 1.5


def test_apply_spread_floor_noop_when_disabled():
    cfg = {STATE_CFG_KEY: SpreadFloorState(enabled=False)}
    assert apply_spread_floor_to_pips(cfg, "AUD_USD", 1.5) == 1.5


def test_apply_spread_floor_raises_when_below():
    state = SpreadFloorState(enabled=True, floors_pips={"AUD_USD": 2.0})
    cfg = {STATE_CFG_KEY: state}
    # raw 1.0 < floor 2.0 → return floor
    assert apply_spread_floor_to_pips(cfg, "AUD_USD", 1.0) == 2.0
    assert state.n_applications == 1
    # raw 3.0 > floor → return raw
    assert apply_spread_floor_to_pips(cfg, "AUD_USD", 3.0) == 3.0
    assert state.n_applications == 1  # unchanged
    # pair not in floors_pips → return raw
    assert apply_spread_floor_to_pips(cfg, "EUR_USD", 1.0) == 1.0


# ─── Part 2 + Part 3: engine integration via subprocess ──────────────────────


def _engine_cmd(cfg_path: Path, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "phase_kgl_v2_4h_wfo.py"),
        "--config",
        str(cfg_path),
        "--out-dir",
        str(out_dir),
    ]


def _make_tiny_synthetic_data(pair_dir: Path, pair: str, freq: str, n_bars: int) -> None:
    """Write a tiny CSV the engine's load_pair_csv can read.

    Synthetic deterministic OHLCV with no signal-triggering structure — used
    to exercise engine plumbing without depending on real market data files.
    """
    pair_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq=freq)
    closes = 1.0 + 0.0001 * np.arange(n_bars)
    df = pd.DataFrame(
        {
            "date":   dates,
            "open":   closes,
            "high":   closes + 0.001,
            "low":    closes - 0.001,
            "close":  closes,
            "volume": 1000.0,
            "spread": 1.0,
        }
    )
    df.to_csv(pair_dir / f"{pair}.csv", index=False)


def test_engine_main_aux_missing_raises(tmp_path):
    """Adapter declares ['h1', 'd1'] but cfg.data.aux omits h1 → fail loud."""
    cfg = {
        "phase": "test",
        "signal_tf": "H4",
        "data": {
            "signal_tf_dir": "data/4hr",
            "aux": {"d1": "data/daily"},  # h1 missing
        },
        "signal_adapter": "signals.kb_exhaustion_bar_adapter:KbExhaustionBarAdapter",
        "signal_adapter_kwargs": {},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_dir = tmp_path / "out"

    result = subprocess.run(_engine_cmd(cfg_path, out_dir), capture_output=True, text=True)
    assert result.returncode != 0, "Engine should have errored"
    combined = (result.stdout + result.stderr)
    assert "missing" in combined.lower() or "aux" in combined.lower(), combined[-500:]


def test_engine_main_bad_adapter_spec_raises(tmp_path):
    cfg = {
        "phase": "test",
        "signal_adapter": "nonexistent.module:NoSuchAdapter",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_dir = tmp_path / "out"
    result = subprocess.run(_engine_cmd(cfg_path, out_dir), capture_output=True, text=True)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "cannot import module" in combined or "not found" in combined


def test_engine_main_rejects_invalid_time_exit_bars(tmp_path):
    cfg = {
        "phase": "test",
        "signal_adapter": "signals.kb_exhaustion_bar_adapter:KbExhaustionBarAdapter",
        "signal_adapter_kwargs": {},
        "time_exit_bars": 0,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_dir = tmp_path / "out"
    result = subprocess.run(_engine_cmd(cfg_path, out_dir), capture_output=True, text=True)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "time_exit_bars" in combined


def test_engine_main_rejects_spread_floor_missing_source(tmp_path):
    cfg = {
        "phase": "test",
        "signal_adapter": "signals.kb_exhaustion_bar_adapter:KbExhaustionBarAdapter",
        "signal_adapter_kwargs": {},
        "spread_floor": {
            "enabled": True,
            "source": "configs/does_not_exist.yaml",
            "expected_body_sha256": "0" * 64,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_dir = tmp_path / "out"
    result = subprocess.run(_engine_cmd(cfg_path, out_dir), capture_output=True, text=True)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "spread_floor" in combined or "not found" in combined


# Two-run KH-24 determinism verification was performed manually during PR
# preparation (sha256 08118567a6ef…58e80ab0 reproduced exactly across the
# pre-PR main and post-PR branch heads). Not encoded as a pytest test
# because the engine's run_kgl_v2 calls OUT_ROOT.relative_to(PROJECT_ROOT)
# which requires --out-dir to live inside the worktree — incompatible with
# pytest's tmp_path. The byte-identical proof lives in the PR description.
