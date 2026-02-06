from __future__ import annotations

from scripts.phase7_run_volume_veto_wfo import (
    discover_volume_indicators,
    probe_volume_indicator,
)


def test_phase7_discovery_finds_runnable_indicator() -> None:
    """Discovery should find volume indicators and at least one runnable candidate."""
    names = discover_volume_indicators()
    assert names, "Expected at least one discovered volume indicator"

    runnable = [name for name in names if probe_volume_indicator(name)[0]]
    assert runnable, "Expected at least one runnable volume indicator"


def test_phase7_stub_indicator_detected() -> None:
    """
    Known stub/non-functional indicator must be detected as not runnable.

    volume_william_vix_fix is implemented as an always-zero stub in volume_funcs.
    """
    ok, _reason = probe_volume_indicator("william_vix_fix")
    assert ok is False

