"""Per-pair non-zero spread floor (L6.0 §7).

Implements the L6.0 §7 spread-cost rule:

    effective_spread[bar, pair] = max(observed_spread[bar, pair], min_nonzero_spread[pair])

The floor file (default `configs/spread_floors_5ers.yaml`) is sha256-locked at
methodology lock. This module:

- Loads the floor table via PyYAML
- Verifies the body sha256 against the cfg-supplied `expected_body_sha256` using
  `compute_body_sha256` from `scripts.lchar.compute_spread_floors` (single
  source of truth — no duplicated hash logic)
- Holds runtime state (per-pair pips floor + counters) that is queried by
  `core.backtester.resolve_spread_pips` via `apply_spread_floor_to_pips`
- Emits two log lines (`SPREAD_FLOOR:` at startup, `SPREAD_FLOOR_APPLICATIONS:`
  at end-of-run) for the integrating backtest entrypoint

The floor file stores native MT5 points; we convert to pips at load time using
`spreads.points_per_pip` (default 10), matching the existing `resolve_spread_pips`
convention. Application is in pips space (the unit `resolve_spread_pips` returns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from scripts.lchar.compute_spread_floors import compute_body_sha256

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Sentinel cfg key for the runtime state attached during run_backtest.
STATE_CFG_KEY: str = "_spread_floor_state"


@dataclass
class SpreadFloorState:
    enabled: bool
    floors_pips: dict[str, float] = field(default_factory=dict)
    source_path: Optional[str] = None
    hash_check: str = "N/A"  # "PASS" or "N/A"
    points_per_pip: float = 10.0
    n_applications: int = 0
    n_total_entry_bars: int = 0


def _resolve_source_path(source: str) -> Path:
    p = Path(source)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_spread_floor(cfg: dict[str, Any]) -> SpreadFloorState:
    """Build a SpreadFloorState from cfg.

    cfg shape:
        spread_floor:
          enabled: bool                     # default false
          source: <path to YAML>            # required when enabled
          expected_body_sha256: <hex64>     # required when enabled

    On `enabled: true`, this function:
      - Verifies the file body sha256 matches `expected_body_sha256`
      - Parses `floors:` and pre-computes per-pair pips floor (native / points_per_pip)
      - Returns a SpreadFloorState with `enabled=True, hash_check="PASS"`

    On absent block or `enabled: false`, returns a disabled state (no-op semantics).
    """
    block = (cfg or {}).get("spread_floor") or {}
    points_per_pip = float(((cfg or {}).get("spreads") or {}).get("points_per_pip", 10.0))

    if not block.get("enabled", False):
        return SpreadFloorState(
            enabled=False, points_per_pip=points_per_pip, source_path=None, hash_check="N/A"
        )

    source = block.get("source")
    expected = block.get("expected_body_sha256")
    if not source or not expected:
        raise ValueError(
            "spread_floor.enabled=true requires both 'source' and 'expected_body_sha256'"
        )

    path = _resolve_source_path(str(source))
    if not path.exists():
        raise FileNotFoundError(
            f"spread_floor.source not found: {path}\n"
            "  see: docs/L6_0_METHODOLOGY_LOCK.md §7"
        )

    actual = compute_body_sha256(path)
    if actual != expected:
        raise ValueError(
            "spread_floor body sha256 mismatch — refusing to load.\n"
            f"  source:   {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "  L6.0 §7 locks this file at methodology lock; any post-lock\n"
            "  modification requires explicit re-planning per L6.0 §17.\n"
            "  see: docs/L6_0_METHODOLOGY_LOCK.md §7"
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    floors_section = data.get("floors") or {}
    floors_pips: dict[str, float] = {}
    for pair, stats in floors_section.items():
        native = float(stats["min_nonzero_spread_native"])
        floors_pips[pair] = native / points_per_pip

    return SpreadFloorState(
        enabled=True,
        floors_pips=floors_pips,
        source_path=str(path),
        hash_check="PASS",
        points_per_pip=points_per_pip,
    )


def apply_spread_floor_to_pips(
    cfg: dict[str, Any], pair: str, pips: float
) -> float:
    """Cap-floor a resolved spread (in pips) using cfg's runtime spread-floor state.

    No-op semantics when state is absent (legacy/unrelated callers) or disabled.
    Always increments `n_total_entry_bars` so the summary line reflects total
    queries even when the floor is off — this gives a stable denominator.
    """
    state = (cfg or {}).get(STATE_CFG_KEY)
    if not isinstance(state, SpreadFloorState):
        return pips
    state.n_total_entry_bars += 1
    if not state.enabled:
        return pips
    floor_pips = state.floors_pips.get(pair)
    if floor_pips is None:
        return pips
    if pips < floor_pips:
        state.n_applications += 1
        return floor_pips
    return pips


def format_startup_log(state: SpreadFloorState) -> str:
    src = state.source_path if state.enabled else None
    return (
        f"SPREAD_FLOOR: enabled={state.enabled}, "
        f"hash_check={state.hash_check}, "
        f"floor_source={src}"
    )


def format_summary_log(state: SpreadFloorState) -> str:
    n = state.n_applications
    m = state.n_total_entry_bars
    pct = (n / m) if m > 0 else 0.0
    return (
        f"SPREAD_FLOOR_APPLICATIONS: count={n}, "
        f"total_entry_bars={m}, "
        f"pct_floored={pct:.6f}"
    )
