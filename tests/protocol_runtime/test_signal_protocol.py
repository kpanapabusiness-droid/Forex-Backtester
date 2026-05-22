"""Tests for SignalModule Protocol + validation."""

from __future__ import annotations

import pytest

from core.arc.signal_protocol import SignalModule, validate_panels
from tests.protocol_runtime._fixtures import SyntheticSignal, build_synthetic_panel


def test_synthetic_signal_conforms_to_protocol() -> None:
    s = SyntheticSignal()
    assert isinstance(s, SignalModule)
    assert s.signal_name == "synthetic_periodic"
    assert s.primary_tf == "H4"
    assert s.causal_lineage == "clean"


def test_validate_panels_missing_tf_raises() -> None:
    s = SyntheticSignal(auxiliary_tfs=("D1",))
    panel_h4 = build_synthetic_panel(pairs=("EURUSD",), n_bars=100)
    with pytest.raises(ValueError, match="requires panels"):
        validate_panels(s, {"H4": panel_h4})


def test_validate_panels_pair_set_mismatch_raises() -> None:
    from core.sim.panel import Panel
    s = SyntheticSignal(auxiliary_tfs=("D1",))
    panel_h4 = build_synthetic_panel(pairs=("EURUSD", "GBPUSD"), n_bars=100)
    panel_d1_raw = build_synthetic_panel(pairs=("EURUSD",), n_bars=20)
    d1 = Panel.from_frames(panel_d1_raw.pair_dfs, tf="D1")
    with pytest.raises(ValueError, match="pair sets"):
        validate_panels(s, {"H4": panel_h4, "D1": d1})


def test_evaluate_returns_per_pair_for_every_pair() -> None:
    s = SyntheticSignal()
    panel = build_synthetic_panel(pairs=("EURUSD", "GBPUSD"), n_bars=200)
    ev = s.evaluate({"H4": panel})
    assert set(ev.per_pair) == {"EURUSD", "GBPUSD"}
    for state in ev.per_pair.values():
        assert state.signal_mask.dtype == bool
        assert (state.signal_mask.sum() > 0)
