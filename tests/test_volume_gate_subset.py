from __future__ import annotations

from pathlib import Path
from typing import Set, Tuple

import pandas as pd

from core.backtester import run_backtest


def _run_config_to_dir(tmp_path: Path, config_rel: str, slug: str) -> Path:
    """
    Run backtester with a given config path into an isolated results dir and
    return the path to trades.csv.
    """
    results_dir = tmp_path / slug
    run_backtest(config_path=Path("configs") / config_rel, results_dir=str(results_dir))

    trades_path = results_dir / "trades.csv"
    assert trades_path.exists(), f"Expected trades.csv at {trades_path}"
    return trades_path


def _identity_set(trades: pd.DataFrame) -> Set[Tuple]:
    """
    Build a stable identity set for trades.

    Uses columns that are part of the canonical schema and uniquely
    characterize a trade under the current engine:
      pair, entry_date, direction_int, entry_price, tp1_price, sl_price
    """
    if trades.empty:
        return set()

    cols = ["pair", "entry_date", "direction_int", "entry_price", "tp1_price", "sl_price"]
    for col in cols:
        assert col in trades.columns, f"Missing expected column '{col}' in trades.csv"

    return set(
        tuple(trades[c].iloc[i] for c in cols)
        for i in range(len(trades))
    )


def test_volume_on_trades_are_subset_of_off(tmp_path: Path) -> None:
    """
    Stronger invariant: trades with volume enabled must be a subset of trades
    with volume disabled (no trade substitution).
    """
    off_trades_path = _run_config_to_dir(tmp_path, "phase2_volume_demo_off.yaml", "demo_off")
    on_trades_path = _run_config_to_dir(tmp_path, "phase2_volume_demo_on.yaml", "demo_on")

    off_df = pd.read_csv(off_trades_path)
    on_df = pd.read_csv(on_trades_path)

    off_ids = _identity_set(off_df)
    on_ids = _identity_set(on_df)

    # Basic sanity: count invariant should still hold
    assert len(on_df) <= len(off_df), (
        f"Volume ON produced more trades than OFF: on={len(on_df)}, off={len(off_df)}"
    )

    missing = on_ids - off_ids
    assert not missing, (
        "Found trades present with volume ON that are not present with volume OFF. "
        f"Example identities (up to 5): {list(missing)[:5]}"
    )

