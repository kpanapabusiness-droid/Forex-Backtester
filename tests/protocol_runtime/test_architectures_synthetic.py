"""Smoke tests for architectures A1-A6 on synthetic data.

These cover the wiring (architecture.run() returns a StrategyResult with
sensible shape), not the underlying methodology. KH-24 anchor test
covers correctness for A1.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.architectures._path_classifier import fit_path_classifier
from core.architectures._protocol import StrategyResult
from core.architectures.a1_system_level_filter import A1Architecture, A1Config, A1RunContext
from core.architectures.a2_classifier_filter import A2Architecture, A2Config
from core.architectures.a5_portfolio_composition import A5Architecture, A5Config
from core.architectures.a6_meta_labeling import A6Architecture, A6Config
from core.features_path_so_far import ALL_FEATURE_KEYS
from core.wfo.folds import Fold
from tests.protocol_runtime._fixtures import (
    SyntheticSignal,
    build_synthetic_panel,
)


def _fold() -> Fold:
    return Fold(
        fold_id=1,
        is_start=date(2018, 1, 1),
        is_end=date(2018, 6, 30),
        oos_start=date(2018, 7, 1),
        oos_end=date(2018, 12, 31),
    )


def _panel_eval():
    panel = build_synthetic_panel(
        pairs=("EURUSD", "GBPUSD"), n_bars=800, start="2018-01-01"
    )
    signal = SyntheticSignal()
    eval_ = signal.evaluate({"H4": panel})
    return panel, eval_


def test_a1_runs_on_synthetic_signal() -> None:
    panel, eval_ = _panel_eval()
    cfg = A1Config(
        config_id="a1_smoke",
        sl_atr_mult=2.0,
        trail_enabled=False,
        risk_pct=0.005,
        max_concurrent_per_currency=2,
        max_concurrent_per_pair=1,
    )
    res = A1Architecture().run(
        signal_evaluation=eval_,
        panels={"H4": panel},
        fold=_fold(),
        arch_config=cfg,
        config_id=cfg.config_id,
    )
    assert isinstance(res, StrategyResult)
    assert res.architecture == "A1"
    assert res.fold_stats.fold_id == 1


def test_a2_admit_gate_with_dummy_classifier() -> None:
    panel, eval_ = _panel_eval()

    class _AdmitAll:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(X.shape[0]), np.ones(X.shape[0])])

    cfg = A2Config(
        config_id="a2_smoke",
        classifier=_AdmitAll(),
        threshold=0.5,
        classifier_feature_order=("f0",),
        trail_enabled=False,
        max_concurrent_per_currency=2,
    )
    # Build per-trade feature lookup: every signal-time gets f0=1.0
    pair = "EURUSD"
    df = panel.pair_dfs[pair]
    feats = {(pair, ts): {"f0": 1.0} for ts in df.index}
    ctx = A1RunContext(per_trade_features=feats)
    res = A2Architecture().run(
        signal_evaluation=eval_,
        panels={"H4": panel},
        fold=_fold(),
        arch_config=cfg,
        config_id=cfg.config_id,
        run_context=ctx,
    )
    assert isinstance(res, StrategyResult)
    assert res.architecture == "A2"


def test_a6_sizing_with_dummy_classifier() -> None:
    panel, eval_ = _panel_eval()

    class _Always06:
        def predict_proba(self, X):
            return np.column_stack([np.full(X.shape[0], 0.4), np.full(X.shape[0], 0.6)])

    cfg = A6Config(
        config_id="a6_smoke",
        classifier=_Always06(),
        lower_threshold=0.4,
        upper_threshold=0.6,
        classifier_feature_order=("f0",),
        trail_enabled=False,
    )
    pair = "EURUSD"
    df = panel.pair_dfs[pair]
    feats = {(pair, ts): {"f0": 1.0} for ts in df.index}
    ctx = A1RunContext(per_trade_features=feats)
    res = A6Architecture().run(
        signal_evaluation=eval_,
        panels={"H4": panel},
        fold=_fold(),
        arch_config=cfg,
        config_id=cfg.config_id,
        run_context=ctx,
    )
    assert isinstance(res, StrategyResult)
    assert res.architecture == "A6"


def test_path_classifier_fits_on_synthetic_features() -> None:
    rng = np.random.default_rng(42)
    n = 200
    y = (np.arange(n) % 2)
    # Feature matrix shaped per ALL_FEATURE_KEYS
    cols = list(ALL_FEATURE_KEYS)
    X = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    # Inject a strong signal in one column
    X["close_r_at_t"] = X["close_r_at_t"] + y * 2.0
    fit = fit_path_classifier(X, y)
    assert fit.fit_auc > 0.7


def test_a5_combines_two_constituents() -> None:
    panel, eval_ = _panel_eval()
    fold = _fold()
    base_cfg = A1Config(
        config_id="a5_constituent",
        sl_atr_mult=2.0,
        trail_enabled=False,
        max_concurrent_per_currency=2,
    )
    res_a = A1Architecture().run(
        signal_evaluation=eval_,
        panels={"H4": panel},
        fold=fold,
        arch_config=base_cfg,
        config_id="a_v1",
    )
    res_b = A1Architecture().run(
        signal_evaluation=eval_,
        panels={"H4": panel},
        fold=fold,
        arch_config=base_cfg,
        config_id="a_v2",
    )
    combined = A5Architecture().run(
        arch_config=A5Config(
            config_id="a5_smoke",
            constituents=(res_a, res_b),
            starting_balance=100_000.0,
        ),
        config_id="a5_smoke",
        fold=fold,
    )
    assert combined.architecture == "A5"
    assert combined.metadata["n_constituents"] == 2
