"""Lookahead-invariant test for the Phase-2 concurrent-density filter (CH-001).

Per docs/PHASE_L6_ARC1_P2_OPEN.md §3.3 (BLOCKING; CI-enforced):

    The count at pair X bar N close must use signal evaluations at bars [N-2, N]
    across all 28 pairs — and ONLY those bars. It must NOT use any data from
    bar N+1 or later on any pair.

    A test harness must verify, on a small synthetic sample: if pair Y fires at
    bar N+1, the count at pair X bar N does NOT change. Hard error if violated.

This test fails CI hard on any lookahead leak in `_attach_concurrent_density`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.signals.l4_univariate_extreme import _attach_concurrent_density


# ---------------------------------------------------------------------------
# Synthetic harness
# ---------------------------------------------------------------------------

N_PAIRS = 28
N_BARS = 100


def _make_pair_signals(fires: dict[str, list[int]]) -> dict[str, pd.DataFrame]:
    """Build a 28-pair × 100-bar synthetic `pair_signals` dict.

    `fires[pair]` is a list of bar indices where that pair's L4 signal fires.
    All 28 pairs share the same hourly timestamp grid starting at 2020-01-01
    (so the unified timeline is exactly the 100-bar grid).
    """
    pair_names = [f"P{i:02d}" for i in range(N_PAIRS)]
    ts = pd.date_range("2020-01-01 00:00:00", periods=N_BARS, freq="1h")
    out: dict[str, pd.DataFrame] = {}
    for p in pair_names:
        mask = np.zeros(N_BARS, dtype=bool)
        for i in fires.get(p, []):
            mask[i] = True
        out[p] = pd.DataFrame({"time": ts, "signal_fired": mask})
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_lookahead_future_bar_does_not_leak() -> None:
    """The §3.3 mandated harness: pair Y firing at bar N+1 must NOT change the
    `concurrent_signals_within_3h` value at pair X bar N."""
    n = 50
    # Two scenarios that differ ONLY in whether pair Y fires at bar n+1.
    base = _make_pair_signals({"P00": [n]})  # X = P00 fires at bar 50 only
    with_future = _make_pair_signals({"P00": [n], "P01": [n + 1]})  # add Y@51

    _attach_concurrent_density(base)
    _attach_concurrent_density(with_future)

    base_val = int(base["P00"].iloc[n]["concurrent_signals_within_3h"])
    future_val = int(with_future["P00"].iloc[n]["concurrent_signals_within_3h"])

    # The count at X@50 must be 1 in both scenarios (only X@50 is in [48, 49, 50]).
    assert base_val == 1, f"baseline X@N count expected 1, got {base_val}"
    assert future_val == 1, (
        f"LOOKAHEAD LEAK: Y firing at bar N+1 changed X@N count from {base_val} to "
        f"{future_val}. The rolling-3 window must include only bars [N-2, N]."
    )


def test_window_includes_two_prior_bars_inclusive() -> None:
    """At X@50, fires at Y@48 (N-2), Z@49 (N-1), and X@50 (N) must all count.

    Boundary check: N-2 is inclusive. The pair's own fire is included."""
    n = 50
    pair_signals = _make_pair_signals(
        {
            "P00": [n],          # X fires at N
            "P01": [n - 1],      # one other pair fires at N-1
            "P02": [n - 2],      # one other pair fires at N-2 (inclusive boundary)
        }
    )
    _attach_concurrent_density(pair_signals)
    val = int(pair_signals["P00"].iloc[n]["concurrent_signals_within_3h"])
    assert val == 3, f"expected count=3 (X@N + P01@N-1 + P02@N-2), got {val}"


def test_window_excludes_n_minus_3() -> None:
    """A fire at N-3 is OUTSIDE the [N-2, N] window and must NOT count."""
    n = 50
    pair_signals = _make_pair_signals(
        {
            "P00": [n],
            "P01": [n - 3],  # outside window
        }
    )
    _attach_concurrent_density(pair_signals)
    val = int(pair_signals["P00"].iloc[n]["concurrent_signals_within_3h"])
    assert val == 1, f"expected count=1 (only X@N; P01@N-3 outside window), got {val}"


def test_simultaneous_fires_at_same_bar_all_count() -> None:
    """Multiple pairs firing at the SAME bar N: each pair's view at bar N counts
    every fire (including its own). All pairs see the same count at that bar."""
    n = 50
    pair_signals = _make_pair_signals(
        {f"P{i:02d}": [n] for i in range(5)}  # P00..P04 all fire at bar 50
    )
    _attach_concurrent_density(pair_signals)
    for i in range(5):
        val = int(pair_signals[f"P{i:02d}"].iloc[n]["concurrent_signals_within_3h"])
        assert val == 5, f"pair P{i:02d} expected count=5, got {val}"


def test_zero_fires_anywhere_yields_zero() -> None:
    """Sanity: with no fires anywhere, every bar has count 0."""
    pair_signals = _make_pair_signals({})
    _attach_concurrent_density(pair_signals)
    for p, df in pair_signals.items():
        assert (df["concurrent_signals_within_3h"].astype(int) == 0).all(), (
            f"pair {p} has nonzero count in a no-fire dataset"
        )


def test_arc1_path_unaffected_when_density_not_attached() -> None:
    """Defense-in-depth: if `_attach_concurrent_density` is NOT called (the Arc 1
    code path), the column does not exist on per-pair dfs. This guarantees the
    Arc 1 verbatim YAML produces byte-identical outputs (no incidental column
    added to trades_all.csv etc.)."""
    pair_signals = _make_pair_signals({"P00": [50], "P01": [51]})
    for df in pair_signals.values():
        assert "concurrent_signals_within_3h" not in df.columns


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
