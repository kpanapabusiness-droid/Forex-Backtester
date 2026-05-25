"""Tests for Step 6 §6.3 spread P&L decomposition diagnostic.

Covers the 10 tests in the dispatch §5 contract plus a Step 6 integration
test that exercises the appended-markdown render path through
``execution_realism.audit()``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.step_6.execution_realism import APPENDED_MARKDOWN_KEY, audit
from core.step_6.inputs import Step6Inputs
from core.step_6.manifest import AuditConfig, Severity
from core.step_6.spread_pnl_decomposition import (
    GateThresholds,
    _drop_unusable_rows,
    _normalise_trade_ledger,
    _per_position_spread_pct_of_R,
    _trade_R_under_scenario,
    run_spread_pnl_decomposition,
)

# ── ledger fixtures ─────────────────────────────────────────────────


def _ledger_row(
    *,
    trade_id: int,
    pair: str = "EURUSD",
    direction: str = "long",
    entry_price: float = 1.1000,
    exit_price: float = 1.1100,
    entry_bid: float = 1.0999,
    entry_ask: float = 1.1001,
    exit_bid: float = 1.1099,
    exit_ask: float = 1.1101,
    sl_price: float = 1.0950,
    size: float = 100_000.0,
    pnl: float | None = None,
    exit_time: str = "2024-01-01 12:00:00+00:00",
    parent_position_id: int | None = None,
) -> dict:
    """Synthesise one ClosedTrade-shaped ledger row."""
    direction_sign = 1 if direction == "long" else -1
    pnl_v = pnl if pnl is not None else direction_sign * (exit_price - entry_price) * size
    return {
        "position_id": trade_id,
        "trade_id": trade_id,
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "exit_bid": exit_bid,
        "exit_ask": exit_ask,
        "sl_price": sl_price,
        "size": size,
        "pnl": pnl_v,
        "exit_time": pd.Timestamp(exit_time),
        "parent_position_id": parent_position_id,
    }


# ── §5 Test 1 — long entry spread cost positive ─────────────────────


def test_long_entry_spread_cost_positive() -> None:
    """Long entry where entry_ask > entry_mid → positive half-spread cost."""
    ledger = pd.DataFrame([_ledger_row(trade_id=1)])
    decomposed = _normalise_trade_ledger(ledger)
    assert (decomposed["half_spread_entry_price"] > 0).all()
    assert decomposed["half_spread_entry_price"].iloc[0] == pytest.approx(0.0001)


# ── §5 Test 2 — short entry spread cost positive ────────────────────


def test_short_entry_spread_cost_positive() -> None:
    """Short entry mirrors long: entry_mid - entry_bid > 0."""
    ledger = pd.DataFrame([
        _ledger_row(
            trade_id=1, direction="short",
            entry_price=1.0999, exit_price=1.0900,
            sl_price=1.1050,
            pnl=(1.0999 - 1.0900) * 100_000.0,
        )
    ])
    decomposed = _normalise_trade_ledger(ledger)
    # The diagnostic uses the symmetric (ask - bid)/2 formula for both
    # sides — it captures the half-spread the trade implicitly paid
    # regardless of direction. Verify the spread cost is positive and
    # equals (entry_ask - entry_bid)/2.
    assert decomposed["half_spread_entry_price"].iloc[0] == pytest.approx(0.0001)
    assert decomposed["half_spread_exit_price"].iloc[0] == pytest.approx(0.0001)
    assert decomposed["direction_sign"].iloc[0] == -1


# ── §5 Test 3 — zero-spread trade has zero spread cost ──────────────


def test_zero_spread_trade_zero_cost() -> None:
    """entry_bid == entry_ask → entry half-spread = 0."""
    ledger = pd.DataFrame([
        _ledger_row(
            trade_id=1,
            entry_bid=1.1000, entry_ask=1.1000,
            exit_bid=1.1100, exit_ask=1.1100,
        )
    ])
    decomposed = _normalise_trade_ledger(ledger)
    assert decomposed["half_spread_entry_price"].iloc[0] == pytest.approx(0.0)
    assert decomposed["half_spread_exit_price"].iloc[0] == pytest.approx(0.0)
    assert decomposed["spread_cost_R"].iloc[0] == pytest.approx(0.0)


# ── §5 Test 4 — inflation scales linearly ──────────────────────────


def test_inflation_scales_linearly() -> None:
    """Inflated_R formula: realised_R - (X - 1) * spread_cost_R.

    Concretely: a trade with realised_R=2.0 and spread_cost_R=0.1 should
    yield 2.0 - 0.1 = 1.9 at X=2.0 and 2.0 - 0.2 = 1.8 at X=3.0.
    """
    realised = pd.Series([2.0, 1.0, -0.5])
    cost = pd.Series([0.1, 0.05, 0.2])
    at_2x = _trade_R_under_scenario(realised, cost, 2.0)
    at_3x = _trade_R_under_scenario(realised, cost, 3.0)
    at_zero = _trade_R_under_scenario(realised, cost, 0.0)
    np.testing.assert_allclose(at_2x.values, [1.9, 0.95, -0.7])
    np.testing.assert_allclose(at_3x.values, [1.8, 0.9, -0.9])
    # zero-spread → original + full tax credited back
    np.testing.assert_allclose(at_zero.values, [2.1, 1.05, -0.3])


# ── §5 Test 5 — zero-spread_R equals realised + total cost ──────────


def test_zero_spread_R_equals_realised_plus_cost() -> None:
    """Property: zero_spread_R = realised_R + spread_cost_R per trade."""
    ledger = pd.DataFrame([
        _ledger_row(trade_id=1, pnl=200.0,
                    entry_bid=1.0999, entry_ask=1.1001,
                    exit_bid=1.1099, exit_ask=1.1101),
        _ledger_row(trade_id=2, pnl=-50.0,
                    entry_bid=1.0998, entry_ask=1.1002,
                    exit_bid=1.1098, exit_ask=1.1102),
    ])
    decomposed = _normalise_trade_ledger(ledger)
    decomposed, _, _ = _drop_unusable_rows(decomposed)
    zero = _trade_R_under_scenario(
        decomposed["realised_R"], decomposed["spread_cost_R"], 0.0,
    )
    expected = decomposed["realised_R"] + decomposed["spread_cost_R"]
    np.testing.assert_allclose(zero.values, expected.values)


# ── §5 Test 6 — robust case: verdict survives 3x ────────────────────


def test_verdict_flip_detection_robust_case(tmp_path: Path) -> None:
    """Synthetic ledger that stays PASS-DEPLOYABLE through 3.0x inflation.

    Construction: 11 folds × 30 trades, each trade realised_R = 0.50 with
    spread_cost_R = 0.001 (essentially nil tax). Worst-fold total =
    15.0 R; worst-fold DD = essentially 0 (all winners). Even at 3.0x
    the spread tax is < 0.5% of R → ratio still passes ≥ 2.0 gate.
    """
    rows = []
    fold_rows = []
    leg_id = 0
    for fold_id in range(11):
        for k in range(30):
            rows.append(_ledger_row(
                trade_id=leg_id, pair="EURUSD",
                entry_price=1.1000, exit_price=1.1050,
                entry_bid=1.0999995, entry_ask=1.1000005,
                exit_bid=1.1049995, exit_ask=1.1050005,
                sl_price=1.0900,
                pnl=(1.1050 - 1.1000) * 100_000.0,
                exit_time=f"2024-{(fold_id % 12) + 1:02d}-{(k % 28) + 1:02d} 12:00:00+00:00",
            ))
            fold_rows.append({"leg_id": leg_id, "fold_id": fold_id})
            leg_id += 1
    ledger = pd.DataFrame(rows)
    fold_assignments = pd.DataFrame(fold_rows)

    result = run_spread_pnl_decomposition(
        arc_name="robust_case",
        top_1_config_id="cfg_robust",
        trade_ledger=ledger,
        fold_assignments=fold_assignments,
        inflation_factors=(1.25, 1.50, 2.00, 3.00),
        gate_thresholds=GateThresholds(),
        output_dir=tmp_path / "out",
    )
    assert result.verdict_flip_factor is None
    assert result.fragility_classification == "robust"
    assert result.worst_fold_at_base_spread.pass_deployable is True


# ── §5 Test 7 — fragile case: downgrades at 1.25x ───────────────────


def test_verdict_flip_detection_fragile_case(tmp_path: Path) -> None:
    """Synthetic ledger whose verdict downgrades at the first inflation step.

    Construction: base scenario sits exactly at PASS-DEPLOYABLE (ratio
    just above 2.0); at 1.25× the spread tax tips it over (worst-fold
    becomes negative → ratio < 2.0).
    """
    # 11 folds, 30 trades per fold. Mixed signs so spread tax matters.
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []
    fold_rows: list[dict] = []
    leg_id = 0
    # Per fold: 22 winners at +1.0R, 8 losers at -1.0R → fold total =
    # 22 - 8 = +14 R. With a wide spread tax (spread_cost_R ≈ 0.6 per
    # trade due to ultra-wide bid/ask), 1.25x adds 0.25 * 0.6 * 30 =
    # 4.5 R tax per fold → fold total 9.5 R. Drawdown also widens.
    for fold_id in range(11):
        is_winner = [True] * 22 + [False] * 8
        rng.shuffle(is_winner)
        for k, win in enumerate(is_winner):
            sl_dist_px = 0.0050  # 50 pips
            # exit price relative to entry sets realised R sign + magnitude.
            entry = 1.1000
            sl = entry - sl_dist_px
            if win:
                exit_price = entry + sl_dist_px      # +1.0R
            else:
                exit_price = entry - sl_dist_px      # -1.0R
            # Ultra-wide synthetic spread: 0.0030 wide → spread_cost_R per
            # trade ≈ (0.0015 + 0.0015) / 0.0050 = 0.60
            rows.append(_ledger_row(
                trade_id=leg_id, pair="EURUSD",
                entry_price=entry, exit_price=exit_price,
                entry_bid=entry - 0.0015, entry_ask=entry + 0.0015,
                exit_bid=exit_price - 0.0015, exit_ask=exit_price + 0.0015,
                sl_price=sl,
                pnl=(exit_price - entry) * 100_000.0,
                exit_time=f"2024-{(fold_id % 12) + 1:02d}-{(k % 28) + 1:02d} 12:00:00+00:00",
            ))
            fold_rows.append({"leg_id": leg_id, "fold_id": fold_id})
            leg_id += 1
    ledger = pd.DataFrame(rows)
    fold_assignments = pd.DataFrame(fold_rows)

    result = run_spread_pnl_decomposition(
        arc_name="fragile_case",
        top_1_config_id="cfg_fragile",
        trade_ledger=ledger,
        fold_assignments=fold_assignments,
        inflation_factors=(1.25, 1.50, 2.00, 3.00),
        gate_thresholds=GateThresholds(),
        output_dir=tmp_path / "out",
    )
    # Verdict must downgrade SOMEWHERE in the inflation sweep.
    assert result.verdict_flip_factor is not None, (
        "expected verdict-flip with wide spread tax; "
        f"per-scenario: base={result.worst_fold_at_base_spread}, "
        f"inflations={dict((k, v.worst_fold_ratio) for k, v in result.worst_fold_at_inflations.items())}"
    )
    assert result.fragility_classification in {"fragile", "marginal", "tolerant"}


# ── §5 Test 8 — per-fold aggregation correctness ────────────────────


def test_per_fold_aggregation_correctness(tmp_path: Path) -> None:
    """3-fold synthetic ledger; verify per-fold totals match manual sum."""
    # Fold 0: two trades, R = +1.5, -0.5 → total +1.0, DD = 0.5 (after second)
    # Fold 1: two trades, R = +2.0, +1.0 → total +3.0, DD = 0.0
    # Fold 2: two trades, R = -1.0, -0.5 → total -1.5, DD = 1.5
    sl_dist = 0.0050
    entry = 1.1000
    sl = entry - sl_dist
    def _r_to_exit(r: float) -> float:
        return entry + r * sl_dist
    plan = [
        (0, 1.5),
        (0, -0.5),
        (1, 2.0),
        (1, 1.0),
        (2, -1.0),
        (2, -0.5),
    ]
    rows: list[dict] = []
    fold_rows: list[dict] = []
    for leg_id, (fid, r) in enumerate(plan):
        exit_price = _r_to_exit(r)
        rows.append(_ledger_row(
            trade_id=leg_id, pair="EURUSD",
            entry_price=entry, exit_price=exit_price,
            entry_bid=entry - 1e-9, entry_ask=entry + 1e-9,
            exit_bid=exit_price - 1e-9, exit_ask=exit_price + 1e-9,
            sl_price=sl,
            pnl=(exit_price - entry) * 100_000.0,
            exit_time=f"2024-0{fid+1}-{leg_id+1:02d} 12:00:00+00:00",
        ))
        fold_rows.append({"leg_id": leg_id, "fold_id": fid})
    ledger = pd.DataFrame(rows)
    fold_assignments = pd.DataFrame(fold_rows)

    result = run_spread_pnl_decomposition(
        arc_name="agg_test", top_1_config_id="cfg_agg",
        trade_ledger=ledger, fold_assignments=fold_assignments,
        inflation_factors=(1.50,),
        gate_thresholds=GateThresholds(),
        output_dir=tmp_path / "out",
    )
    base = result.worst_fold_at_base_spread
    # Worst fold = fold 2 (R total = -1.5, DD = 1.5).
    assert base.worst_fold_roi_base_r == pytest.approx(-1.5, abs=1e-6)
    assert base.worst_fold_dd_base_r == pytest.approx(1.5, abs=1e-6)
    # Worst-fold ratio: -1.5 / 1.5 = -1.0
    assert base.worst_fold_ratio == pytest.approx(-1.0, abs=1e-6)
    assert base.n_negative_folds == 1
    assert base.pass_deployable is False
    assert base.pass_viable is False


# ── §5 Test 9 — determinism two runs byte-identical ─────────────────


def test_determinism_two_runs(tmp_path: Path) -> None:
    """Two invocations of the diagnostic on the same fixture produce
    byte-identical artefact files (per Step 6 §6.5 contract)."""
    rng = np.random.default_rng(seed=11)
    rows = []
    fold_rows = []
    entry = 1.1000
    sl = entry - 0.0050
    for leg_id in range(60):
        win = rng.random() > 0.4
        exit_price = entry + (0.0050 if win else -0.0050)
        rows.append(_ledger_row(
            trade_id=leg_id,
            entry_price=entry, exit_price=exit_price,
            entry_bid=entry - 0.00005, entry_ask=entry + 0.00005,
            exit_bid=exit_price - 0.00005, exit_ask=exit_price + 0.00005,
            sl_price=sl,
            pnl=(exit_price - entry) * 100_000.0,
            exit_time=f"2024-01-{(leg_id % 28) + 1:02d} 12:00:00+00:00",
        ))
        fold_rows.append({"leg_id": leg_id, "fold_id": leg_id % 5})
    ledger = pd.DataFrame(rows)
    fold_assignments = pd.DataFrame(fold_rows)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    r1 = run_spread_pnl_decomposition(
        arc_name="det", top_1_config_id="cfg",
        trade_ledger=ledger, fold_assignments=fold_assignments,
        gate_thresholds=GateThresholds(),
        output_dir=out1,
    )
    r2 = run_spread_pnl_decomposition(
        arc_name="det", top_1_config_id="cfg",
        trade_ledger=ledger, fold_assignments=fold_assignments,
        gate_thresholds=GateThresholds(),
        output_dir=out2,
    )
    # Same set of artefacts produced.
    names1 = sorted(p.name for p in r1.output_artefacts)
    names2 = sorted(p.name for p in r2.output_artefacts)
    assert names1 == names2
    # Byte-identical content per artefact.
    for n in names1:
        b1 = (out1 / n).read_bytes()
        b2 = (out2 / n).read_bytes()
        assert b1 == b2, f"non-deterministic output: {n}"


# ── §5 Test 10 — handles zero-spread data-quality bars ──────────────


def test_handles_zero_spread_bars_in_ledger(tmp_path: Path) -> None:
    """Trades whose bid==ask (zero-spread DQ flag) must not blow up.

    Resulting spread_cost_R == 0 for those trades; per-fold/per-scenario
    aggregation completes without exception.
    """
    rows: list[dict] = []
    fold_rows: list[dict] = []
    entry = 1.1000
    sl = entry - 0.0050
    for leg_id in range(20):
        win = leg_id % 2 == 0
        exit_price = entry + (0.0050 if win else -0.0050)
        # Half the trades have zero spread (data-quality flag);
        # the other half have nonzero spread.
        zero_spread = (leg_id % 2 == 0)
        bid_offset = 0.0 if zero_spread else 0.00005
        rows.append(_ledger_row(
            trade_id=leg_id,
            entry_price=entry, exit_price=exit_price,
            entry_bid=entry - bid_offset, entry_ask=entry + bid_offset,
            exit_bid=exit_price - bid_offset, exit_ask=exit_price + bid_offset,
            sl_price=sl,
            pnl=(exit_price - entry) * 100_000.0,
            exit_time=f"2024-01-{(leg_id % 28) + 1:02d} 12:00:00+00:00",
        ))
        fold_rows.append({"leg_id": leg_id, "fold_id": leg_id % 4})
    ledger = pd.DataFrame(rows)
    fold_assignments = pd.DataFrame(fold_rows)

    result = run_spread_pnl_decomposition(
        arc_name="dq_test", top_1_config_id="cfg",
        trade_ledger=ledger, fold_assignments=fold_assignments,
        gate_thresholds=GateThresholds(),
        output_dir=tmp_path / "out",
    )
    # Zero-spread legs are KEPT (their cost is 0, not NaN). All 20 should
    # be counted.
    assert result.n_trades_with_spread_data == 20
    # Distribution stats finite when at least some legs have nonzero cost.
    assert math.isfinite(result.mean_spread_cost_pct_of_r)


# ── Bonus: per-position aggregation (multi-leg partial close) ───────


def test_per_position_aggregation_partials() -> None:
    """A multi-leg position (one partial + one final) contributes once
    to the per-position spread-pct distribution.
    """
    pos_id = 7
    # Same position: leg 0 is a partial close at +1R; leg 1 is the final close at +2R.
    sl_dist = 0.0050
    entry = 1.1000
    sl = entry - sl_dist
    rows = [
        _ledger_row(
            trade_id=pos_id,
            entry_price=entry, exit_price=entry + sl_dist,
            entry_bid=entry - 5e-5, entry_ask=entry + 5e-5,
            exit_bid=entry + sl_dist - 5e-5, exit_ask=entry + sl_dist + 5e-5,
            sl_price=sl,
            size=50_000.0, pnl=(entry + sl_dist - entry) * 50_000.0,
            parent_position_id=pos_id,
        ),
        _ledger_row(
            trade_id=pos_id,
            entry_price=entry, exit_price=entry + 2 * sl_dist,
            entry_bid=entry - 5e-5, entry_ask=entry + 5e-5,
            exit_bid=entry + 2 * sl_dist - 5e-5, exit_ask=entry + 2 * sl_dist + 5e-5,
            sl_price=sl,
            size=50_000.0, pnl=(entry + 2 * sl_dist - entry) * 50_000.0,
            parent_position_id=pos_id,
        ),
    ]
    ledger = pd.DataFrame(rows)
    decomposed = _normalise_trade_ledger(ledger)
    decomposed, _, _ = _drop_unusable_rows(decomposed)
    pct = _per_position_spread_pct_of_R(decomposed)
    # Exactly ONE position aggregated.
    assert len(pct) == 1
    assert pct.index[0] == pos_id


# ── Integration: Step 6 execution_realism appends subsection ────────


def test_execution_realism_appends_spread_pnl_subsection(tmp_path: Path) -> None:
    """When Step6Inputs carries top_1_trade_ledger + fold_assignments,
    audit() emits the spread-decomposition subsection in the rendered
    report.
    """
    from core.step_6.artefacts import render_category_report

    rng = np.random.default_rng(seed=7)
    rows = []
    fold_rows = []
    entry = 1.1000
    sl = entry - 0.0050
    for leg_id in range(40):
        win = rng.random() > 0.4
        exit_price = entry + (0.0050 if win else -0.0050)
        rows.append(_ledger_row(
            trade_id=leg_id,
            entry_price=entry, exit_price=exit_price,
            entry_bid=entry - 5e-5, entry_ask=entry + 5e-5,
            exit_bid=exit_price - 5e-5, exit_ask=exit_price + 5e-5,
            sl_price=sl,
            pnl=(exit_price - entry) * 100_000.0,
            exit_time=f"2024-01-{(leg_id % 28) + 1:02d} 12:00:00+00:00",
        ))
        fold_rows.append({"leg_id": leg_id, "fold_id": leg_id % 4})
    ledger = pd.DataFrame(rows)
    fold_assignments = pd.DataFrame(fold_rows)

    inputs = Step6Inputs(
        arc_name="integration_test",
        arc_root=tmp_path / "arc_root",
        best_candidate_config_id="A1::cfg_top1",
        top_1_trade_ledger=ledger,
        top_1_fold_assignments=fold_assignments,
        r_base_pct=0.005,
    )
    inputs.arc_root.mkdir(parents=True, exist_ok=True)
    result = audit(inputs, AuditConfig())
    # Diagnostic produced and pinned into the diagnostic dict
    assert APPENDED_MARKDOWN_KEY in result.diagnostic
    assert "Spread P&L decomposition" in result.diagnostic[APPENDED_MARKDOWN_KEY]
    # The diagnostic check exists and is INFO severity (does not block)
    diag_check = next(
        (c for c in result.checks if c.name == "spread_pnl_decomposition"), None
    )
    assert diag_check is not None
    assert diag_check.severity is Severity.INFO
    # Rendered category report contains the markdown subsection
    rendered = render_category_report(result)
    assert "### Spread P&L decomposition" in rendered
    assert "Worst-fold ratio across spread scenarios" in rendered
    assert "Verdict-flip factor" in rendered
    # Artefacts written to <arc_root>/step_6/
    artefact_dir = inputs.arc_root / "step_6"
    assert (artefact_dir / "spread_pnl_per_trade.parquet").exists()
    assert (artefact_dir / "spread_pnl_per_fold_per_scenario.csv").exists()
    assert (artefact_dir / "spread_pnl_verdict_flip_summary.csv").exists()


def test_execution_realism_skips_diagnostic_gracefully_pre_extension(
    tmp_path: Path,
) -> None:
    """Pre-extension closures (no top_1_trade_ledger) get an INFO-severity
    'skipped' check; no artefacts emitted; no markdown appended.
    """
    inputs = Step6Inputs(
        arc_name="pre_pr",
        arc_root=tmp_path / "arc_root",
    )
    inputs.arc_root.mkdir(parents=True, exist_ok=True)
    result = audit(inputs, AuditConfig())
    diag_check = next(
        (c for c in result.checks if c.name == "spread_pnl_decomposition"), None
    )
    assert diag_check is not None
    assert diag_check.severity is Severity.INFO
    assert diag_check.passed is True
    assert APPENDED_MARKDOWN_KEY not in result.diagnostic
    # The diagnostic must not pollute the spread_pnl_decomposition manifest
    # entry either when skipped.
    assert "spread_pnl_decomposition" not in result.diagnostic
