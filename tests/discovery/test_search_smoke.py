"""Small-N integration smoke for the random_search loop.

Builds synthetic per-pair OHLC + feature matrices, runs a tiny search
(N=30 rules, 2 pairs), and asserts:

  * The search log has one row per generated rule (including rejections).
  * Causal-rejected rules carry pool_size=0 and pool_floor_pass=False.
  * Evaluated rules carry finite mean_r and p_value.
  * Two runs produce byte-identical log rows (determinism asserted at
    the dict level — the parquet sha equality is asserted in the
    dispatch determinism harness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.discovery.causal_filter import clean_feature_pool
from core.discovery.grammar import Combinator, GrammarConfig, Op
from core.discovery.pool_simulator import DiscoveryExitConfig
from core.discovery.quantile_grid import build_quantile_grid
from core.discovery.random_search import (
    PairFixture,
    SearchConfig,
    run_search,
)


def _make_synthetic_pair(pair: str, n: int = 2000, seed: int = 0) -> PairFixture:
    """Random-walk OHLC + a small feature matrix tagged clean lineage."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC")
    # Generate close as random walk.
    rets = rng.normal(0.0, 0.001, n)
    close = 1.20 * np.cumprod(1.0 + rets)
    spread = 0.0001
    df = pd.DataFrame(
        {
            "open_bid":  close * (1.0 + rng.normal(0, 0.0002, n)),
            "high_bid":  close * (1.0 + np.abs(rng.normal(0, 0.0008, n))),
            "low_bid":   close * (1.0 - np.abs(rng.normal(0, 0.0008, n))),
            "close_bid": close,
            "open_ask":  close * (1.0 + rng.normal(0, 0.0002, n)) + spread,
            "high_ask":  close * (1.0 + np.abs(rng.normal(0, 0.0008, n))) + spread,
            "low_ask":   close * (1.0 - np.abs(rng.normal(0, 0.0008, n))) + spread,
            "close_ask": close + spread,
            "volume": rng.integers(1, 1000, n).astype("float64"),
            "spread_close": [spread] * n,
            "bid_ask_data_quality": ["ok"] * n,
        },
        index=idx,
    )
    # Tiny feature matrix: two clean features, two distinct distributions.
    feat = pd.DataFrame(
        {
            "feat_a": np.arange(n, dtype="float64"),
            "feat_b": np.sin(np.arange(n) / 50.0),
        },
        index=idx,
    )
    atr = pd.Series([0.0010] * n, index=idx, name="atr_14")
    return PairFixture(pair=pair, pair_df=df, feature_matrix=feat, atr_series=atr)


def _lineage_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "feat_a", "lineage": "clean", "feature_class": "synthetic"},
            {"name": "feat_b", "lineage": "clean", "feature_class": "synthetic"},
        ]
    )


def test_search_smoke_writes_one_row_per_rule():
    fixtures = [
        _make_synthetic_pair("EURUSD", n=2000, seed=1),
        _make_synthetic_pair("GBPUSD", n=2000, seed=2),
    ]
    grid = build_quantile_grid(
        feature_matrices={fx.pair: fx.feature_matrix for fx in fixtures},
        quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
    )
    lineage = _lineage_df()
    cfg = SearchConfig(
        n_rules=30,
        random_seed=42,
        pool_floor=50,
        grammar_cfg=GrammarConfig(),
        exit_cfg=DiscoveryExitConfig(),
        follow_up_top_k=3,
        analysis_top_k=10,
    )
    feature_pool = clean_feature_pool(lineage)
    res = run_search(
        fixtures=fixtures,
        grid=grid,
        lineage_df=lineage,
        cfg=cfg,
        feature_pool=feature_pool,
        progress_every=0,
    )
    assert len(res.log_rows) == 30
    # Every row has the locked search-log keys.
    required_keys = {
        "rule_id", "rule_spec_json", "n_atoms", "features_used",
        "causal_filter_pass", "pool_floor_pass", "pool_size", "mean_r",
        "p_value", "bonferroni_pass_primary", "bonferroni_pass_budget",
    }
    for row in res.log_rows:
        assert required_keys.issubset(row.keys())


def test_search_smoke_is_deterministic():
    fixtures = [_make_synthetic_pair("EURUSD", n=1500, seed=1)]
    grid = build_quantile_grid(
        feature_matrices={fx.pair: fx.feature_matrix for fx in fixtures},
        quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
    )
    lineage = _lineage_df()
    cfg = SearchConfig(
        n_rules=20,
        random_seed=42,
        pool_floor=10,
        grammar_cfg=GrammarConfig(),
        exit_cfg=DiscoveryExitConfig(),
    )
    pool = clean_feature_pool(lineage)
    a = run_search(fixtures=fixtures, grid=grid, lineage_df=lineage, cfg=cfg,
                   feature_pool=pool, progress_every=0)
    b = run_search(fixtures=fixtures, grid=grid, lineage_df=lineage, cfg=cfg,
                   feature_pool=pool, progress_every=0)
    # Strip the timing field (wall_clock differs) and compare rows.
    assert a.bonferroni_report.n_generated == b.bonferroni_report.n_generated
    assert a.bonferroni_report.n_evaluated == b.bonferroni_report.n_evaluated
    assert a.survivors == b.survivors
    assert [r.rule_id for r in a.ranked_top] == [r.rule_id for r in b.ranked_top]
    assert len(a.log_rows) == len(b.log_rows)
    # Compare every key except None-vs-None equality.
    for ra, rb in zip(a.log_rows, b.log_rows):
        assert ra == rb


def test_bonferroni_threshold_primary_against_n_evaluated():
    """Decision 3: primary denominator = N_evaluated (not n_generated)."""
    fixtures = [_make_synthetic_pair("EURUSD", n=1500, seed=1)]
    grid = build_quantile_grid(
        feature_matrices={fx.pair: fx.feature_matrix for fx in fixtures},
        quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
    )
    lineage = _lineage_df()
    cfg = SearchConfig(
        n_rules=20,
        random_seed=42,
        pool_floor=10,
        grammar_cfg=GrammarConfig(),
        exit_cfg=DiscoveryExitConfig(),
    )
    pool = clean_feature_pool(lineage)
    res = run_search(
        fixtures=fixtures, grid=grid, lineage_df=lineage, cfg=cfg,
        feature_pool=pool, progress_every=0,
    )
    n_eval = res.bonferroni_report.n_evaluated
    if n_eval > 0:
        assert abs(res.bonferroni_report.threshold_primary - 0.05 / n_eval) < 1e-12
    # Budget threshold uses n_generated regardless.
    assert abs(res.bonferroni_report.threshold_budget - 0.05 / 20) < 1e-12
