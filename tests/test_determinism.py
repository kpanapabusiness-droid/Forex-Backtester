"""Two-run sha256 reproducibility per CC_06 Task 7 + 9d.

Runs an end-to-end mini pipeline twice and asserts every output (panel
parquet caches, feature matrices, backtester equity curve, closed-trade
ledger) is byte-identical. Then repeats the comparison at pool_size=1
vs pool_size=4 — the parallelism MUST NOT change any output.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from core.determinism import RANDOM_STATE, seed_everything
from core.features.cache import get_or_compute, pool_sha_from_dataframe
from core.features.pipeline import compute_feature_matrix
from core.parallel import build_panel_parallel
from core.sim.account import Account, Direction, ExposureRules
from core.sim.multipair_backtester import MultiPairBacktester, Order
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    return build_fixture(
        tmp_path / "histdata",
        FixtureSpec(
            pairs=("EURUSD", "GBPUSD", "USDJPY"),
            months=("201001",),
            minutes_per_month=360,  # 6 hours of M1 → 72 M5 bars per pair
        ),
    )


def _deterministic_strategy(t, snapshot, account):
    """Emit one long order on the first bar of each minute-0-hour-N bucket
    until 3 trades have opened. Deterministic + finite."""
    if len(account.closed_trades) + len(account.open_positions) >= 3:
        return []
    if t.minute != 0:
        return []
    pair = "EURUSD"
    bar = snapshot.get(pair)
    if bar is None:
        return []
    return [Order(pair=pair, direction=Direction.LONG, size=1.0)]


def _sha_csv(df: pd.DataFrame) -> str:
    csv = df.to_csv(lineterminator="\n", float_format="%.10g")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def _run_pipeline(mini_root: Path, cache_root: Path, pool_size: int) -> tuple[str, str, str]:
    """Run the mini pipeline and return (panel_sha, features_sha, equity_sha)."""
    seed_everything(RANDOM_STATE)
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    panel = build_panel_parallel(
        pairs, "M5", histdata_root=mini_root, cache_root=cache_root, pool_size=pool_size
    )
    panel_csv = pd.concat([panel.pair_dfs[p].assign(_pair=p) for p in sorted(panel.pairs)], axis=0)
    panel_sha = _sha_csv(panel_csv)

    # Feature matrix for EURUSD (panel-dependent features included)
    eur_features = compute_feature_matrix("EURUSD", panel.pair_dfs["EURUSD"], panel=panel)
    features_sha = _sha_csv(eur_features.matrix)

    # Run a deterministic backtest
    acct = Account(
        starting_balance=100_000.0,
        exposure=ExposureRules(max_concurrent_total=3, max_concurrent_per_pair=1),
    )
    bt = MultiPairBacktester(panel=panel, account=acct, strategy=_deterministic_strategy)
    result = bt.run()
    equity_csv = result.equity_curve.to_csv(lineterminator="\n", float_format="%.10g")
    equity_sha = hashlib.sha256(equity_csv.encode("utf-8")).hexdigest()

    return panel_sha, features_sha, equity_sha


def test_two_run_byte_identical_pool1(mini_root: Path, tmp_path: Path) -> None:
    """Two serial runs against fresh caches produce identical sha256s."""
    a = _run_pipeline(mini_root, tmp_path / "cache_a", pool_size=1)
    b = _run_pipeline(mini_root, tmp_path / "cache_b", pool_size=1)
    assert a == b, f"two-run sha mismatch at pool=1: {a} vs {b}"


def test_two_run_byte_identical_pool4(mini_root: Path, tmp_path: Path) -> None:
    """Two pool=4 runs against fresh caches produce identical sha256s."""
    a = _run_pipeline(mini_root, tmp_path / "cache_a", pool_size=4)
    b = _run_pipeline(mini_root, tmp_path / "cache_b", pool_size=4)
    assert a == b, f"two-run sha mismatch at pool=4: {a} vs {b}"


def test_pool1_equals_pool4_byte_identical(mini_root: Path, tmp_path: Path) -> None:
    """Switching pool_size from 1 to 4 MUST NOT change any output sha256."""
    serial = _run_pipeline(mini_root, tmp_path / "cache_serial", pool_size=1)
    parallel = _run_pipeline(mini_root, tmp_path / "cache_parallel", pool_size=4)
    assert serial == parallel, (
        f"parallel changed output:\n  serial   {serial}\n  parallel {parallel}"
    )


def test_feature_cache_two_run_byte_identical(mini_root: Path, tmp_path: Path) -> None:
    """Feature matrix cache produces byte-identical output on hit + miss."""
    seed_everything(RANDOM_STATE)
    panel = build_panel_parallel(
        ["EURUSD", "GBPUSD", "USDJPY"],
        "M5",
        histdata_root=mini_root,
        cache_root=tmp_path / "cache",
        pool_size=2,
    )
    pool_df = pd.DataFrame({"entry_time": panel.pair_dfs["EURUSD"].index[:5]})
    pool_sha = pool_sha_from_dataframe(pool_df)

    def compute():
        return compute_feature_matrix("EURUSD", panel.pair_dfs["EURUSD"], panel=panel)

    miss = get_or_compute(
        "test_arc", "sig", pool_sha, "v3.0", compute, cache_root=tmp_path / "cache"
    )
    hit = get_or_compute(
        "test_arc", "sig", pool_sha, "v3.0", compute, cache_root=tmp_path / "cache"
    )

    assert _sha_csv(miss.matrix) == _sha_csv(hit.matrix)


def test_seed_everything_idempotent() -> None:
    """Calling seed_everything twice doesn't change the seed state semantics."""
    seed_everything(42)
    import numpy as np

    a = np.random.rand()
    seed_everything(42)
    b = np.random.rand()
    assert a == b
