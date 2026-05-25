"""End-to-end smoke test: spread P&L diagnostic runs when ledger is wired.

Per dispatch §6 DoD #2: "Spread P&L decomposition wired into orchestrator
and runs end-to-end on a synthetic PASS scenario."

This test bypasses the orchestrator's heavy machinery and exercises just
the §6.3 diagnostic path through the public Step 6 surface: build
Step6Inputs with a populated ``top_1_trade_ledger`` + ``top_1_fold_assignments``,
run ``audit_exec``, and confirm the spread P&L artefacts appear on disk
and the check is not a silent skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.step_6 import AuditConfig
from core.step_6.execution_realism import audit as audit_exec
from core.step_6.inputs import Step6Inputs
from tests.step_6._fixtures import build_clean_inputs


def _build_ledger(n_trades: int = 60, *, seed: int = 42) -> pd.DataFrame:
    """Synthetic closed-trade ledger with the full extended schema."""
    rng = np.random.default_rng(seed)
    entry_price = 1.10 + rng.normal(0, 0.001, n_trades)
    sl_distance = 0.005
    sl_price = entry_price - sl_distance
    # Half the trades win at +1R, half lose at -1R-ish
    direction_pnl = rng.choice([-sl_distance, 2 * sl_distance], size=n_trades)
    exit_price = entry_price + direction_pnl
    half_spread = 0.00005  # 0.5 pip
    entry_bid = entry_price - half_spread
    entry_ask = entry_price + half_spread
    exit_bid = exit_price - half_spread
    exit_ask = exit_price + half_spread
    return pd.DataFrame({
        "position_id": range(1, n_trades + 1),
        "pair": "EURUSD",
        "direction": "long",
        "entry_time": pd.date_range("2018-01-01", periods=n_trades, freq="D", tz="UTC"),
        "entry_price": entry_price,
        "exit_time": pd.date_range("2018-01-02", periods=n_trades, freq="D", tz="UTC"),
        "exit_price": exit_price,
        "size": 10_000.0,
        "pnl": direction_pnl * 10_000.0,
        "exit_reason": "stop_loss",
        "parent_position_id": None,
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "exit_bid": exit_bid,
        "exit_ask": exit_ask,
        "sl_price": sl_price,
    })


def test_spread_pnl_diagnostic_runs_end_to_end(tmp_path: Path):
    inp = build_clean_inputs(tmp_path)
    ledger = _build_ledger(60)
    # Two IS folds + one "holdout" fold
    fold_assignments = pd.DataFrame({
        "leg_id": list(range(60)),
        "fold_id": [1] * 25 + [2] * 25 + [99] * 10,
    })
    inputs = Step6Inputs(
        arc_name=inp.arc_name, arc_root=inp.arc_root,
        best_candidate_config_id="A1::cfg_smoke",
        best_candidate_architecture="A1",
        best_candidate_features=inp.best_candidate_features,
        pool_trades=inp.pool_trades,
        feature_matrix=inp.feature_matrix,
        feature_lineage=inp.feature_lineage,
        primary_tf=inp.primary_tf, pair_set=inp.pair_set,
        r_safe_pct=inp.r_safe_pct,
        sizing_convention=inp.sizing_convention,
        configs_evaluated_step5=inp.configs_evaluated_step5,
        top_1_trade_ledger=ledger,
        top_1_fold_assignments=fold_assignments,
        holdout_fold_id=99,
        r_base_pct=0.005,
    )
    result = audit_exec(inputs, AuditConfig())
    diag = next(c for c in result.checks if c.name == "spread_pnl_decomposition")
    # Diagnostic ran (not "skipped — no top-1 trade ledger ...")
    assert "skipped" not in diag.message.lower()
    assert diag.evidence["fragility_classification"] in (
        "robust", "tolerant", "marginal", "fragile",
    )
    # Three artefacts should now live under arc_root/step_6/
    out_dir = tmp_path / "step_6"
    assert (out_dir / "spread_pnl_per_trade.parquet").exists()
    assert (out_dir / "spread_pnl_per_fold_per_scenario.csv").exists()
    assert (out_dir / "spread_pnl_verdict_flip_summary.csv").exists()


def test_spread_pnl_skipped_when_ledger_absent(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = audit_exec(inputs, AuditConfig())
    diag = next(c for c in result.checks if c.name == "spread_pnl_decomposition")
    assert "skipped" in diag.message.lower()
