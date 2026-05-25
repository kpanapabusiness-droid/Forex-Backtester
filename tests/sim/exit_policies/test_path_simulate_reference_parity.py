"""Reference-implementation parity for canonical path-simulate.

The canonical ``core.sim.exit_policies.path_simulate`` module was extracted
from ``scripts/l_arc_10_v3/step_5.py:_apply_exit_policy:99-253`` (the
historical hand-rolled simulator that produced Arc 10's PASS-DEPLOYABLE
verdict). This test confirms byte-identical behaviour between the two
across a synthetic range of recorded path inputs.

Tolerances (per chat answer Q5 — mid-only parity is byte-identical because
both implementations work in R-units off a single mid-anchored price
stream; no bid/ask wing is added):
  * Per-trade final_r: BYTE-IDENTICAL (not just within ±0.01R).
  * Per-trade bars_held: BYTE-IDENTICAL.

If this test ever fails: the canonical implementation has drifted from
the reference. Either (a) the reference changed in scripts/l_arc_10_v3/
step_5.py (likely a separate intentional change — update the parity
fixture), or (b) we accidentally introduced a semantic divergence in
the canonical path_simulate module (this is the bug the test is here
to catch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# sys.path mutation above must precede these imports → E402 explicitly waived.
from core.sim.exit_policies import simulate_path as canonical_simulate_path  # noqa: E402
from scripts.l_arc_10_v3.step_5 import _apply_exit_policy as reference_apply  # noqa: E402

POLICIES = (
    "sl_only",
    "sl_plus_tp_2r",
    "sl_plus_tp_3r",
    "sl_plus_trailing_atr",
    "sl_plus_trailing_swing",
    "sl_partial_close_1r_runner_trail",
)

SL_MULTS = (1.5, 2.0, 2.5, 3.0)


def _make_path_df(
    *,
    mae_seq: list[float],
    mfe_seq: list[float],
    close_seq: list[float],
    is_held: list[int] | None = None,
) -> pd.DataFrame:
    n = len(mae_seq)
    assert len(mfe_seq) == n and len(close_seq) == n
    return pd.DataFrame({
        "bar_offset": list(range(n)),
        "mae_so_far_r": mae_seq,
        "mfe_so_far_r": mfe_seq,
        "close_r": close_seq,
        "is_held": is_held if is_held is not None else [1] * n,
    })


# Reproducible synthetic scenarios that exercise different code paths
SCENARIOS = {
    # Winner that hits +2R cleanly
    "winner_tp2r": _make_path_df(
        mae_seq=[0.0, -0.1, -0.2, -0.1, -0.05],
        mfe_seq=[0.0, 0.5, 1.2, 1.8, 2.2],
        close_seq=[0.0, 0.4, 1.0, 1.6, 2.1],
    ),
    # Winner that just barely reaches +1R and trails back
    "winner_runner_trail": _make_path_df(
        mae_seq=[0.0, -0.2, -0.1, -0.05, -0.1, -0.2, -0.3],
        mfe_seq=[0.0, 0.5, 1.1, 1.3, 1.3, 1.3, 1.3],
        close_seq=[0.0, 0.4, 1.0, 1.2, 0.8, 0.2, -0.1],
    ),
    # Loser hit SL early
    "loser_sl_early": _make_path_df(
        mae_seq=[0.0, -0.5, -1.1, -1.5, -2.0],
        mfe_seq=[0.0, 0.1, 0.2, 0.1, 0.0],
        close_seq=[0.0, -0.3, -1.0, -1.3, -1.6],
    ),
    # Time exit at end of held window — no SL, no profitable threshold
    "time_exit_breakeven": _make_path_df(
        mae_seq=[0.0, -0.3, -0.3, -0.2, -0.1, 0.0],
        mfe_seq=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7],
        close_seq=[0.0, 0.1, 0.3, 0.4, 0.5, 0.6],
    ),
    # SL hit after some profit — tests SL-preempts-trail / SL-after-tp1
    "winner_then_sl": _make_path_df(
        mae_seq=[0.0, -0.2, -0.1, -0.05, -0.8, -1.5],
        mfe_seq=[0.0, 0.5, 1.2, 1.5, 1.5, 1.5],
        close_seq=[0.0, 0.4, 1.0, 1.3, 0.0, -1.0],
    ),
    # Empty path (edge case)
    "empty_path": pd.DataFrame(
        columns=["bar_offset", "mae_so_far_r", "mfe_so_far_r", "close_r", "is_held"]
    ),
    # Path with NaN (degenerate but possible)
    "with_nan": _make_path_df(
        mae_seq=[0.0, -0.1, float("nan"), -0.2, -0.3],
        mfe_seq=[0.0, 0.5, 1.1, 1.3, 1.0],
        close_seq=[0.0, 0.4, 1.0, 1.2, 0.8],
    ),
    # Big winner that trails out after peak
    "big_winner_trail": _make_path_df(
        mae_seq=[0.0, -0.1, -0.05, 0.0, -0.5, -0.8],
        mfe_seq=[0.0, 1.5, 2.5, 3.5, 3.5, 3.5],
        close_seq=[0.0, 1.0, 2.0, 3.0, 1.5, 0.5],
    ),
    # is_held shortened (time exit earlier than path length)
    "early_time_exit": _make_path_df(
        mae_seq=[0.0, -0.2, -0.3, -0.2, -0.1, -0.05],
        mfe_seq=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9],
        close_seq=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7],
        is_held=[1, 1, 1, 0, 0, 0],  # closed at bar 2
    ),
}


def _make_trade_row(final_r: float = 0.0, bars_held: int = 0) -> pd.Series:
    """Minimal trade row — only used by the empty-path early-return branch."""
    return pd.Series({"final_r": final_r, "bars_held": bars_held})


# ────────────────────────────────────────────────────────────────────────
# Tests: canonical vs reference byte-identity
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("sl_mult", SL_MULTS)
@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_canonical_matches_reference_byte_identical(
    policy: str, sl_mult: float, scenario_name: str
) -> None:
    """Canonical path simulator emits identical (final_r, bars_held) to
    reference hand-rolled across all policies × SL multipliers × scenarios.
    """
    path_df = SCENARIOS[scenario_name]
    trade_row = _make_trade_row(final_r=0.0, bars_held=len(path_df))

    canonical_r, canonical_bars = canonical_simulate_path(
        policy, trade_row, path_df, sl_mult
    )
    reference_r, reference_bars = reference_apply(
        trade_row, path_df, sl_mult, policy
    )

    # Byte-identical contract on R values (same float operations in the same
    # order; no bid/ask spread perturbation in path replay mode).
    assert canonical_r == reference_r, (
        f"final_r mismatch | policy={policy} sl_mult={sl_mult} "
        f"scenario={scenario_name} | canonical={canonical_r} "
        f"reference={reference_r}"
    )
    assert canonical_bars == reference_bars, (
        f"bars_held mismatch | policy={policy} sl_mult={sl_mult} "
        f"scenario={scenario_name} | canonical={canonical_bars} "
        f"reference={reference_bars}"
    )


def test_unknown_policy_falls_back_to_sl_only() -> None:
    """Reference behaviour: unknown policy name → sl_only fallthrough.
    Canonical preserves this contract.
    """
    path_df = SCENARIOS["winner_tp2r"]
    trade_row = _make_trade_row()
    canonical = canonical_simulate_path("nonsense", trade_row, path_df, 2.0)
    reference = reference_apply(trade_row, path_df, 2.0, "nonsense")
    assert canonical == reference


def test_available_path_simulators_matches_canonical_registry() -> None:
    """The path-simulator dispatcher's name list must be a SUPERSET of the
    live-engine registry (every live policy needs a path-replay too).
    """
    from core.sim.exit_policies import available_path_simulators, available_policies

    live = set(available_policies())
    replay = set(available_path_simulators())
    missing_in_replay = live - replay
    assert not missing_in_replay, (
        f"Policies in live registry but missing from path-simulate: "
        f"{sorted(missing_in_replay)}"
    )
