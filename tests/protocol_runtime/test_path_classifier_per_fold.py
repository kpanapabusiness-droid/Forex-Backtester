"""Per-fold path-classifier orchestration tests for A3 / A4.

Covers core.steps.path_classifier_per_fold:

  - derive_per_fold_seed determinism + cross-process stability
  - build_per_trade_entry_features shape
  - build_path_classifier_fits_per_fold for A3 (cluster membership target)
  - build_path_classifier_fits_per_fold for A4 (final_r > 0 target)
  - CostDecomposition aggregation

Uses synthetic pool + paths fixtures so determinism is asserted on
exact numerical outputs.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.architectures._path_classifier import PathClassifierFit
from core.sim.panel import Panel
from core.steps.path_classifier_per_fold import (
    A4_TRAIN_DECIDE_OFFSET,
    PerFoldTrainingInputs,
    PoolFraction,
    build_path_classifier_fits_per_fold,
    build_per_trade_entry_features,
    compute_cost_decomposition_from_decisions,
    derive_per_fold_seed,
)
from core.wfo.folds import Fold

# ── derive_per_fold_seed ───────────────────────────────────────────


def test_derive_per_fold_seed_deterministic() -> None:
    a = derive_per_fold_seed(arc_seed=42, fold_id=3, arch_name="A3", cluster_id=1)
    b = derive_per_fold_seed(arc_seed=42, fold_id=3, arch_name="A3", cluster_id=1)
    assert a == b
    assert 0 <= a < 2**32


def test_derive_per_fold_seed_varies_with_inputs() -> None:
    base = derive_per_fold_seed(arc_seed=42, fold_id=3, arch_name="A3", cluster_id=1)
    # Same arc_seed but different fold_id, arch, or cluster_id -> different seed
    assert base != derive_per_fold_seed(arc_seed=42, fold_id=4, arch_name="A3", cluster_id=1)
    assert base != derive_per_fold_seed(arc_seed=42, fold_id=3, arch_name="A4", cluster_id=1)
    assert base != derive_per_fold_seed(arc_seed=42, fold_id=3, arch_name="A3", cluster_id=2)
    assert base != derive_per_fold_seed(arc_seed=43, fold_id=3, arch_name="A3", cluster_id=1)


def test_derive_per_fold_seed_handles_none_cluster_id() -> None:
    # A4 doesn't have cluster_id; passing None should still produce a
    # stable, in-range seed.
    s = derive_per_fold_seed(arc_seed=42, fold_id=1, arch_name="A4", cluster_id=None)
    assert 0 <= s < 2**32


# ── Fixture builders ────────────────────────────────────────────────


def _make_pair_df(n: int = 100, start: str = "2018-01-01", seed: int = 42) -> pd.DataFrame:
    """Synthetic H4 bars with bid+ask spread + ATR-like volatility."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq="4h", tz="UTC")
    base = 1.1000 + rng.normal(0, 0.002, n).cumsum() * 0.01
    high = base + rng.uniform(0.0005, 0.0020, n)
    low = base - rng.uniform(0.0005, 0.0020, n)
    open_ = base + rng.normal(0, 0.0005, n)
    close = base + rng.normal(0, 0.0005, n)
    return pd.DataFrame(
        {
            "open_bid": open_, "open_ask": open_ + 0.0001,
            "high_bid": high, "high_ask": high + 0.0001,
            "low_bid": low, "low_ask": low + 0.0001,
            "close_bid": close, "close_ask": close + 0.0001,
            "volume": rng.uniform(100, 1000, n),
            "spread_close": 0.0001,
            "bid_ask_data_quality": ["ok"] * n,
        },
        index=idx,
    )


def _make_synthetic_pool(n_trades: int = 30, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Panel]:
    """Build a synthetic Step 1 pool + paths + cluster_assignments + Panel.

    Returns (trades, paths, cluster_assignments, panel).
    """
    rng = np.random.default_rng(seed)
    df = _make_pair_df(n=100, seed=seed)
    panel = Panel(tf="H4", pair_dfs={"EURUSD": df})

    # Pick signal bars spread across the pair df (leaving room for path bars)
    signal_indices = rng.choice(range(25, 80), size=n_trades, replace=False)
    signal_indices.sort()

    trades_rows: list[dict] = []
    paths_rows: list[dict] = []
    for tid, sig_idx in enumerate(signal_indices, start=1):
        sig_t = df.index[sig_idx]
        entry_idx = sig_idx + 1
        entry_t = df.index[entry_idx]
        atr = 0.0010
        entry_px = float(df["open_ask"].iat[entry_idx])
        sl_px = entry_px - 2.0 * atr
        sl_dist = entry_px - sl_px
        # 10-bar trade
        hold = 10
        bars_held = 0
        final_r = 0.0
        for off in range(0, hold + 1):
            bidx = entry_idx + off
            if bidx >= len(df):
                break
            close_r = float(((df["close_bid"].iat[bidx]) - entry_px) / sl_dist)
            mfe_r = float(((df["high_bid"].iat[bidx]) - entry_px) / sl_dist)
            mae_r = float(((df["low_bid"].iat[bidx]) - entry_px) / sl_dist)
            paths_rows.append({
                "trade_id": tid,
                "bar_offset": off,
                "timestamp": df.index[bidx],
                "close_r": close_r,
                "mfe_so_far_r": max(0.0, mfe_r),
                "mae_so_far_r": min(0.0, mae_r),
            })
            if off == hold:
                bars_held = off
                final_r = close_r
        trades_rows.append({
            "pair": "EURUSD",
            "trade_id": tid,
            "signal_time": sig_t,
            "entry_time": entry_t,
            "entry_price": entry_px,
            "atr_at_signal": atr,
            "sl_at_entry_price": sl_px,
            "exit_time": df.index[entry_idx + bars_held],
            "exit_price": entry_px + final_r * sl_dist,
            "exit_reason": "time_exit",
            "bars_held": bars_held,
            "final_r": final_r,
            "mfe_r": max(0.0, final_r),
            "mae_r": min(0.0, final_r),
        })

    trades = pd.DataFrame(trades_rows)
    paths = pd.DataFrame(paths_rows)
    # 50/50 cluster split for A3 target
    cluster_assignments = pd.DataFrame({
        "trade_id": trades["trade_id"],
        "cluster_id": [tid % 2 for tid in trades["trade_id"]],
        "k_selected": 2,
    })
    return trades, paths, cluster_assignments, panel


def _make_folds() -> tuple[Fold, ...]:
    """Two folds covering the synthetic 2018 fixture window."""
    return (
        Fold(
            fold_id=1,
            is_start=date(2018, 1, 1),
            is_end=date(2018, 1, 10),
            oos_start=date(2018, 1, 11),
            oos_end=date(2018, 1, 15),
        ),
        Fold(
            fold_id=2,
            is_start=date(2018, 1, 1),
            is_end=date(2018, 1, 15),
            oos_start=date(2018, 1, 16),
            oos_end=date(2018, 1, 18),
        ),
    )


# ── build_per_trade_entry_features ──────────────────────────────────


def test_build_per_trade_entry_features_shape() -> None:
    trades, _paths, _ca, panel = _make_synthetic_pool()
    inputs = PerFoldTrainingInputs(
        pool_trades=trades,
        pool_paths=_paths,
        cluster_assignments=_ca,
        panels={"H4": panel},
        primary_tf="H4",
        candidate_cluster_id=1,
        n_defer=5,
    )
    out = build_per_trade_entry_features(inputs)
    # Should have one entry per pool trade
    assert len(out) == len(trades)
    # Keys are (pair, signal_time) tuples
    pair, sig_t = next(iter(out))
    assert pair == "EURUSD"
    assert isinstance(sig_t, pd.Timestamp)
    # Values contain the 8 ENTRY_FEATURE_KEYS
    from core.features_path_so_far import ENTRY_FEATURE_KEYS
    feats = next(iter(out.values()))
    assert set(feats.keys()) == set(ENTRY_FEATURE_KEYS)


# ── build_path_classifier_fits_per_fold — A3 ─────────────────────────


def test_a3_per_fold_fits_keyed_by_fold_id() -> None:
    trades, paths, ca, panel = _make_synthetic_pool()
    inputs = PerFoldTrainingInputs(
        pool_trades=trades,
        pool_paths=paths,
        cluster_assignments=ca,
        panels={"H4": panel},
        primary_tf="H4",
        candidate_cluster_id=1,
        n_defer=5,
    )
    folds = _make_folds()
    fits = build_path_classifier_fits_per_fold(inputs=inputs, folds=folds, arch="A3")
    assert set(fits.keys()) == {1, 2}
    for fid, fit in fits.items():
        assert isinstance(fit, PathClassifierFit)
        # feature_order should be the 15-feature locked schema
        from core.features_path_so_far import ALL_FEATURE_KEYS
        assert set(fit.feature_order) == set(ALL_FEATURE_KEYS)


def test_a3_per_fold_fits_determinism() -> None:
    trades, paths, ca, panel = _make_synthetic_pool()
    inputs = PerFoldTrainingInputs(
        pool_trades=trades, pool_paths=paths, cluster_assignments=ca,
        panels={"H4": panel}, primary_tf="H4",
        candidate_cluster_id=1, n_defer=5,
    )
    folds = _make_folds()
    a = build_path_classifier_fits_per_fold(inputs=inputs, folds=folds, arch="A3")
    b = build_path_classifier_fits_per_fold(inputs=inputs, folds=folds, arch="A3")
    # Threshold + fit_auc should be byte-identical across runs
    for fid in a:
        assert a[fid].threshold == b[fid].threshold
        # NaN-tolerant comparison for fit_auc (degenerate fits return NaN)
        if not (np.isnan(a[fid].fit_auc) and np.isnan(b[fid].fit_auc)):
            assert a[fid].fit_auc == b[fid].fit_auc


# ── build_path_classifier_fits_per_fold — A4 ─────────────────────────


def test_a4_per_fold_fits_use_final_r_target() -> None:
    trades, paths, ca, panel = _make_synthetic_pool()
    inputs = PerFoldTrainingInputs(
        pool_trades=trades, pool_paths=paths, cluster_assignments=None,
        panels={"H4": panel}, primary_tf="H4",
        candidate_cluster_id=None, n_defer=A4_TRAIN_DECIDE_OFFSET,
    )
    folds = _make_folds()
    fits = build_path_classifier_fits_per_fold(inputs=inputs, folds=folds, arch="A4")
    assert set(fits.keys()) == {1, 2}
    for fit in fits.values():
        assert isinstance(fit, PathClassifierFit)


def test_a4_does_not_require_cluster_assignments() -> None:
    """Sanity: A4 target = `final_r > 0`; no cluster info needed."""
    trades, paths, _ca, panel = _make_synthetic_pool()
    inputs = PerFoldTrainingInputs(
        pool_trades=trades, pool_paths=paths, cluster_assignments=None,
        panels={"H4": panel}, primary_tf="H4",
        candidate_cluster_id=None, n_defer=5,
    )
    folds = _make_folds()
    # Should not raise
    fits = build_path_classifier_fits_per_fold(inputs=inputs, folds=folds, arch="A4")
    assert len(fits) == 2


# ── Cost decomposition ──────────────────────────────────────────────


def test_cost_decomposition_three_buckets() -> None:
    decisions = [
        {"bucket": "admit", "r": 1.0},
        {"bucket": "admit", "r": 0.5},
        {"bucket": "admit", "r": -0.5},
        {"bucket": "reject", "r": -0.2},
        {"bucket": "reject", "r": -0.3},
        {"bucket": "early_exit", "r": -1.0},
    ]
    decomp = compute_cost_decomposition_from_decisions(decisions)
    assert decomp.admit_pool.n == 3
    assert decomp.admit_pool.n_fraction == pytest.approx(3 / 6)
    assert decomp.admit_pool.mean_r == pytest.approx((1.0 + 0.5 - 0.5) / 3)
    assert decomp.reject_pool.n == 2
    assert decomp.reject_pool.mean_r == pytest.approx(-0.25)
    assert decomp.early_exit_pool.n == 1
    assert decomp.early_exit_pool.mean_r == pytest.approx(-1.0)


def test_cost_decomposition_empty_buckets() -> None:
    decomp = compute_cost_decomposition_from_decisions([])
    assert decomp.admit_pool == PoolFraction(n_fraction=0.0, mean_r=0.0, n=0)
    assert decomp.reject_pool == PoolFraction(n_fraction=0.0, mean_r=0.0, n=0)
    assert decomp.early_exit_pool == PoolFraction(n_fraction=0.0, mean_r=0.0, n=0)
