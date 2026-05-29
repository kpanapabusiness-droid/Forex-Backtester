"""Trigger-density filter — arc_discovery_02 Amendment D.

After the smoke run of arc_discovery_02 demonstrated that the random-grammar
rule distribution under 4H × 28 pairs is sharply bimodal (76% of rules either
fire on 20-40% of bars or fire on < 0.5%), a generation-time density filter
removes both tails before any expensive simulation runs.

The check is cheap:
  1. Cache a calibration fixture once — 6 months of EURUSD H4 + the full
     feature matrix on that window.
  2. For each candidate rule, compile its trigger mask against the
     calibration fixture (no simulation, just boolean ops over ~1100 bars).
  3. Compute observed trigger rate = mask.sum() / len(mask).
  4. Accept if ``band.lo <= rate <= band.hi`` (default [0.005, 0.08]).
  5. Reject otherwise; log the rate so chat can audit the band.

The quantile-threshold grid used to compile the rule MUST be the full-window
grid (built on 2010-2020 pooled), NOT a 6-month grid — otherwise we'd be
measuring "fires on its own quantile" which is trivially 50% by construction.

Determinism: pure function of (rule, fixture, grid, band). No RNG. The same
50000-rule generation + filter pass on the same fixture is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.discovery.grammar import RuleSpec
from core.discovery.quantile_grid import QuantileGrid
from core.discovery.rule_engine import compile_rule


@dataclass(frozen=True)
class DensityBand:
    """Inclusive bounds on accepted trigger rate."""

    lo: float = 0.005   # 0.5% — rules below this are too rare for stat power
    hi: float = 0.08    # 8%  — rules above this dominate compute (the smoke pathology)

    def accepts(self, rate: float) -> bool:
        return self.lo <= rate <= self.hi


@dataclass(frozen=True)
class CalibrationFixture:
    """Single-pair feature-matrix slice used to estimate per-rule trigger density.

    ``feature_matrix`` is a small DataFrame (typically EURUSD H4 over 6 months)
    aligned to the same column schema as the search-time per-pair feature
    matrices. ``grid`` is the FULL-WINDOW QuantileGrid used at search time.
    """

    pair: str
    feature_matrix: pd.DataFrame
    grid: QuantileGrid
    window_start: str
    window_end: str

    @property
    def n_bars(self) -> int:
        return len(self.feature_matrix.index)


@dataclass(frozen=True)
class DensityCheckResult:
    """Outcome of running the density filter on one rule."""

    passed: bool
    observed_rate: float
    band: DensityBand

    @property
    def reason(self) -> str:
        if self.passed:
            return "density_in_band"
        if self.observed_rate < self.band.lo:
            return f"trigger_density_too_low ({self.observed_rate:.4f} < {self.band.lo})"
        return f"trigger_density_too_high ({self.observed_rate:.4f} > {self.band.hi})"


def check_density(
    spec: RuleSpec,
    fixture: CalibrationFixture,
    band: DensityBand,
) -> DensityCheckResult:
    """Compile rule against the calibration fixture and assess its density."""
    mask = compile_rule(spec, fixture.feature_matrix, fixture.grid)
    arr = mask.to_numpy(dtype=bool, copy=False)
    if arr.size == 0:
        # Degenerate fixture — treat as out-of-band low.
        return DensityCheckResult(passed=False, observed_rate=0.0, band=band)
    rate = float(arr.sum()) / float(arr.size)
    return DensityCheckResult(passed=band.accepts(rate), observed_rate=rate, band=band)


def restrict_to_window(
    df: pd.DataFrame, window_start: str, window_end: str
) -> pd.DataFrame:
    """Slice ``df`` to the inclusive [start, end] date range (tz-aware)."""
    tz = df.index.tz
    start_ts = pd.Timestamp(window_start, tz=tz)
    end_ts = pd.Timestamp(window_end, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


__all__ = (
    "DensityBand",
    "CalibrationFixture",
    "DensityCheckResult",
    "check_density",
    "restrict_to_window",
)
