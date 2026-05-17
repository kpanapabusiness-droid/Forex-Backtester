"""Tests for Pipeline D1 plumbing (PR 1 of 2).

Covers:

* :mod:`core.features_path_so_far` — feature builder validation and the
  no-lookahead invariant.
* :mod:`core.d1_pipeline` — ``D1Hook`` initialization checks, the
  ``Close`` / ``ApplyPolicy`` decision boundary, multi-archetype max-P
  tie-breaking, and the engine helper ``apply_d1_hook_per_bar`` (correct
  bar-offset gating + next-open vs last-bar fill).
* Engine integration — the patched lines in
  ``scripts/phase_kgl_v2_4h_wfo.py`` exist, the global ``D1_HOOK`` defaults
  to ``None`` (baseline byte-identical), and the SL formula stays
  ``entry - 2.0 * ATR(14)`` for all original trades.

Tests use a stub classifier — no joblib / sklearn import required.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from core.d1_pipeline import (
    ApplyPolicy,
    Close,
    D1ArchetypeConfig,
    D1Hook,
    Hold,
    apply_d1_hook_per_bar,
)
from core.exit_policies import StepwiseClimberPolicy
from core.features_path_so_far import (
    ALL_FEATURE_KEYS,
    ENTRY_FEATURE_KEYS,
    PATH_FEATURE_KEYS,
    build_entry_features_at_signal_bar,
    build_features_at_t,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stub classifier — deterministic P from a single path feature.
# ---------------------------------------------------------------------------


class StubClassifier:
    """Returns P(class=1) per a single-feature rule.

    Default: P=0.9 if features[``mfe_idx``] > ``threshold``, else P=0.1.
    Tests can subclass or pass ``fixed_p`` for a constant probability.
    """

    def __init__(
        self,
        feature_order: list[str],
        *,
        gate_feature: str = "mfe_so_far_r_at_t",
        threshold: float = 1.0,
        p_high: float = 0.9,
        p_low: float = 0.1,
        fixed_p: float | None = None,
    ):
        self.n_features_in_ = len(feature_order)
        self._gate_idx = feature_order.index(gate_feature)
        self._threshold = float(threshold)
        self._p_high = float(p_high)
        self._p_low = float(p_low)
        self._fixed_p = fixed_p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.zeros((n, 2), dtype=np.float64)
        for k in range(n):
            if self._fixed_p is not None:
                p = float(self._fixed_p)
            else:
                p = self._p_high if X[k, self._gate_idx] > self._threshold else self._p_low
            out[k, 1] = p
            out[k, 0] = 1.0 - p
        return out


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def entry_features_finite() -> dict[str, float]:
    """Locked finite entry-feature dict for use across tests."""
    return {
        "body_to_range_ratio": 0.6,
        "upper_wick_ratio": 0.2,
        "lower_wick_ratio": 0.2,
        "range_to_atr_14": 1.1,
        "ret_5bar_atr": 0.3,
        "ret_20bar_atr": 0.5,
        "pos_in_20bar_range": 0.7,
        "rsi_14": 60.0,
    }


def _bar_path(close_rs: list[float]) -> list[dict[str, float]]:
    """Build a v1.3-shape bar_path with running mfe/mae bookkeeping.

    high_r / low_r are pinned at close_r ± 0.1; running mfe is the max of
    high_r so far, running mae the min of low_r so far.
    """
    rows: list[dict[str, float]] = []
    mfe = -np.inf
    mae = np.inf
    for k, cr in enumerate(close_rs):
        hr = float(cr) + 0.1
        lr = float(cr) - 0.1
        mfe = max(mfe, hr)
        mae = min(mae, lr)
        rows.append(
            {
                "bar_offset": k,
                "high_r": hr,
                "low_r": lr,
                "close_r": float(cr),
                "mfe_so_far_r": float(mfe),
                "mae_so_far_r": float(mae),
                "is_held": 1,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 1. Feature builder validation.
# ---------------------------------------------------------------------------


class TestFeatureBuilderValidation:
    def test_known_keys_all_present(self):
        assert set(ALL_FEATURE_KEYS) == set(ENTRY_FEATURE_KEYS) | set(PATH_FEATURE_KEYS)
        assert len(set(ENTRY_FEATURE_KEYS) & set(PATH_FEATURE_KEYS)) == 0

    def test_unknown_feature_raises(self, entry_features_finite):
        order = list(ALL_FEATURE_KEYS) + ["bogus_feature"]
        with pytest.raises(ValueError, match="unexpected: bogus_feature"):
            build_features_at_t(
                bar_path_so_far=_bar_path([0.0, 0.5, 1.0]),
                entry_features=entry_features_finite,
                t=2,
                feature_order=order,
            )

    def test_missing_feature_raises(self, entry_features_finite):
        order = [k for k in ALL_FEATURE_KEYS if k != "rsi_14"]
        with pytest.raises(ValueError, match="missing: rsi_14"):
            build_features_at_t(
                bar_path_so_far=_bar_path([0.0, 0.5, 1.0]),
                entry_features=entry_features_finite,
                t=2,
                feature_order=order,
            )

    def test_duplicate_feature_raises(self, entry_features_finite):
        order = list(ALL_FEATURE_KEYS) + ["rsi_14"]
        with pytest.raises(ValueError, match="duplicate"):
            build_features_at_t(
                bar_path_so_far=_bar_path([0.0]),
                entry_features=entry_features_finite,
                t=0,
                feature_order=order,
            )

    def test_entry_features_missing_raises(self):
        bad_entry = {k: 0.0 for k in ENTRY_FEATURE_KEYS if k != "rsi_14"}
        with pytest.raises(ValueError, match="rsi_14"):
            build_features_at_t(
                bar_path_so_far=_bar_path([0.0, 1.0]),
                entry_features=bad_entry,
                t=1,
                feature_order=list(ALL_FEATURE_KEYS),
            )

    def test_negative_t_raises(self, entry_features_finite):
        with pytest.raises(ValueError, match="t must be >= 0"):
            build_features_at_t(
                bar_path_so_far=_bar_path([0.0, 0.5]),
                entry_features=entry_features_finite,
                t=-1,
                feature_order=list(ALL_FEATURE_KEYS),
            )

    def test_returns_numpy_float64(self, entry_features_finite):
        feats = build_features_at_t(
            bar_path_so_far=_bar_path([0.0, 0.5, 1.5]),
            entry_features=entry_features_finite,
            t=2,
            feature_order=list(ALL_FEATURE_KEYS),
        )
        assert isinstance(feats, np.ndarray)
        assert feats.dtype == np.float64
        assert feats.shape == (len(ALL_FEATURE_KEYS),)


# ---------------------------------------------------------------------------
# 2. Feature builder: no-lookahead invariant.
# ---------------------------------------------------------------------------


class TestFeatureBuilderNoLookahead:
    def test_features_at_t_ignore_post_t_rows(self, entry_features_finite):
        """Mutating bar_offset > t rows must not change features at t."""
        order = list(ALL_FEATURE_KEYS)
        path_short = _bar_path([0.0, 0.5, 1.0])           # bars 0..2
        path_long = _bar_path([0.0, 0.5, 1.0, 99.9, -99.9])  # extends with wild bars

        feats_short = build_features_at_t(
            bar_path_so_far=path_short,
            entry_features=entry_features_finite,
            t=2,
            feature_order=order,
        )
        feats_long = build_features_at_t(
            bar_path_so_far=path_long,
            entry_features=entry_features_finite,
            t=2,
            feature_order=order,
        )
        np.testing.assert_array_equal(feats_short, feats_long)

    def test_t_must_exist_in_path(self, entry_features_finite):
        with pytest.raises(ValueError, match="no row with bar_offset == 5"):
            build_features_at_t(
                bar_path_so_far=_bar_path([0.0, 0.5, 1.0]),  # bars 0..2
                entry_features=entry_features_finite,
                t=5,
                feature_order=list(ALL_FEATURE_KEYS),
            )


# ---------------------------------------------------------------------------
# 3. Path feature semantics.
# ---------------------------------------------------------------------------


class TestPathFeatureSemantics:
    """Spot-checks that the derived path features compute the spec values."""

    def test_close_r_and_mfe_mae_at_t(self, entry_features_finite):
        # Path: close_r = [0.0, 0.5, -0.2, 1.2]; we evaluate at t=3.
        path = _bar_path([0.0, 0.5, -0.2, 1.2])
        order = ["close_r_at_t", "mfe_so_far_r_at_t", "mae_so_far_r_at_t"] + [
            k for k in ALL_FEATURE_KEYS
            if k not in ("close_r_at_t", "mfe_so_far_r_at_t", "mae_so_far_r_at_t")
        ]
        feats = build_features_at_t(path, entry_features_finite, 3, order)
        assert feats[0] == pytest.approx(1.2)             # close_r_at_t
        assert feats[1] == pytest.approx(1.3, abs=1e-9)   # mfe_so_far_r_at_t = max high_r = 1.2+0.1
        assert feats[2] == pytest.approx(-0.3, abs=1e-9)  # mae_so_far_r_at_t = min low_r = -0.2-0.1

    def test_bars_in_profit_count(self, entry_features_finite):
        # close_r values: 1 positive, 1 zero, 2 negative -> bars_in_profit = 1.
        path = _bar_path([0.5, 0.0, -0.1, -0.2])
        order = ["bars_in_profit_at_t"] + [
            k for k in ALL_FEATURE_KEYS if k != "bars_in_profit_at_t"
        ]
        feats = build_features_at_t(path, entry_features_finite, 3, order)
        assert feats[0] == pytest.approx(1.0)

    def test_velocity_first_t(self, entry_features_finite):
        # mfe_so_far_r_at_t = 1.1 (= 1.0+0.1), t = 4 → velocity = 1.1 / 4.
        path = _bar_path([0.0, 0.5, 1.0, 0.7, 0.3])
        order = ["velocity_first_t"] + [
            k for k in ALL_FEATURE_KEYS if k != "velocity_first_t"
        ]
        feats = build_features_at_t(path, entry_features_finite, 4, order)
        assert feats[0] == pytest.approx(1.1 / 4)


# ---------------------------------------------------------------------------
# 4. D1Hook initialization.
# ---------------------------------------------------------------------------


class TestD1HookInit:
    def test_empty_archetypes_raises(self):
        with pytest.raises(ValueError, match="at least one archetype"):
            D1Hook([])

    def test_mixed_bar_offset_t_raises(self):
        order = list(ALL_FEATURE_KEYS)
        policy = StepwiseClimberPolicy(
            archetype_sl_r=1.0, r_in_atr=3.0, mfe_lock_r=1.0, trail_from_high_r=0.75
        )
        a = D1ArchetypeConfig(
            label="x",
            feature_order=tuple(order),
            decision_threshold=0.5,
            bar_offset_t=3,
            per_fold_classifiers={"F1": StubClassifier(order)},
            exit_policy=policy,
        )
        b = D1ArchetypeConfig(
            label="y",
            feature_order=tuple(order),
            decision_threshold=0.5,
            bar_offset_t=5,
            per_fold_classifiers={"F1": StubClassifier(order)},
            exit_policy=policy,
        )
        with pytest.raises(ValueError, match="must share bar_offset_t"):
            D1Hook([a, b])

    def test_bad_feature_order_raises(self):
        policy = StepwiseClimberPolicy(
            archetype_sl_r=1.0, r_in_atr=3.0, mfe_lock_r=1.0, trail_from_high_r=0.75
        )
        a = D1ArchetypeConfig(
            label="x",
            feature_order=tuple(list(ALL_FEATURE_KEYS) + ["bogus"]),
            decision_threshold=0.5,
            bar_offset_t=3,
            per_fold_classifiers={"F1": StubClassifier(list(ALL_FEATURE_KEYS))},
            exit_policy=policy,
        )
        with pytest.raises(ValueError, match="bogus"):
            D1Hook([a])

    def test_classifier_feature_count_mismatch_raises(self):
        # Mutate the stub's n_features_in_ to a value that mismatches the
        # length of feature_order — provokes the explicit validation message.
        order = list(ALL_FEATURE_KEYS)
        stub = StubClassifier(order)
        stub.n_features_in_ = 99
        a = D1ArchetypeConfig(
            label="x",
            feature_order=tuple(order),
            decision_threshold=0.5,
            bar_offset_t=3,
            per_fold_classifiers={"F1": stub},
            exit_policy=StepwiseClimberPolicy(
                archetype_sl_r=1.0, r_in_atr=3.0,
                mfe_lock_r=1.0, trail_from_high_r=0.75,
            ),
        )
        with pytest.raises(ValueError, match="expects 99 features"):
            D1Hook([a])

    def test_threshold_out_of_range_raises(self):
        order = list(ALL_FEATURE_KEYS)
        a = D1ArchetypeConfig(
            label="x",
            feature_order=tuple(order),
            decision_threshold=1.5,
            bar_offset_t=3,
            per_fold_classifiers={"F1": StubClassifier(order)},
            exit_policy=StepwiseClimberPolicy(
                archetype_sl_r=1.0, r_in_atr=3.0,
                mfe_lock_r=1.0, trail_from_high_r=0.75,
            ),
        )
        with pytest.raises(ValueError, match="decision_threshold"):
            D1Hook([a])

    def test_bar_offset_t_must_be_positive(self):
        order = list(ALL_FEATURE_KEYS)
        a = D1ArchetypeConfig(
            label="x",
            feature_order=tuple(order),
            decision_threshold=0.5,
            bar_offset_t=0,
            per_fold_classifiers={"F1": StubClassifier(order)},
            exit_policy=StepwiseClimberPolicy(
                archetype_sl_r=1.0, r_in_atr=3.0,
                mfe_lock_r=1.0, trail_from_high_r=0.75,
            ),
        )
        with pytest.raises(ValueError, match="bar_offset_t"):
            D1Hook([a])

    def test_empty_per_fold_classifiers_raises(self):
        order = list(ALL_FEATURE_KEYS)
        a = D1ArchetypeConfig(
            label="x",
            feature_order=tuple(order),
            decision_threshold=0.5,
            bar_offset_t=3,
            per_fold_classifiers={},
            exit_policy=StepwiseClimberPolicy(
                archetype_sl_r=1.0, r_in_atr=3.0,
                mfe_lock_r=1.0, trail_from_high_r=0.75,
            ),
        )
        with pytest.raises(ValueError, match="per_fold_classifiers is empty"):
            D1Hook([a])


# ---------------------------------------------------------------------------
# 5. D1Hook.evaluate — Close / ApplyPolicy decision.
# ---------------------------------------------------------------------------


def _default_policy() -> StepwiseClimberPolicy:
    return StepwiseClimberPolicy(
        archetype_sl_r=1.0, r_in_atr=3.0, mfe_lock_r=1.0, trail_from_high_r=0.75
    )


def _make_single_hook(
    threshold: float = 0.5, t: int = 3, fold_id: str = "F1"
) -> D1Hook:
    order = list(ALL_FEATURE_KEYS)
    arch = D1ArchetypeConfig(
        label="stepwise_climber",
        feature_order=tuple(order),
        decision_threshold=threshold,
        bar_offset_t=t,
        per_fold_classifiers={
            fold_id: StubClassifier(
                order, gate_feature="mfe_so_far_r_at_t", threshold=1.0,
            ),
        },
        exit_policy=_default_policy(),
    )
    return D1Hook([arch])


class TestD1HookEvaluate:
    def test_returns_apply_policy_when_mfe_high(self, entry_features_finite):
        hook = _make_single_hook(threshold=0.5, t=3)
        # mfe_so_far_r at t=3 = 1.5 + 0.1 = 1.6 > stub threshold 1.0 → P=0.9 ≥ 0.5.
        path = _bar_path([0.0, 0.5, 1.0, 1.5])
        decision = hook.evaluate(
            trade={}, bar_path_so_far=path, entry_features=entry_features_finite,
            t=3, cached={}, fold_id="F1",
        )
        assert isinstance(decision, ApplyPolicy)
        assert decision.label == "stepwise_climber"
        assert decision.probability == pytest.approx(0.9)
        assert isinstance(decision.policy, StepwiseClimberPolicy)

    def test_returns_close_when_mfe_low(self, entry_features_finite):
        hook = _make_single_hook(threshold=0.5, t=3)
        path = _bar_path([0.0, 0.1, 0.2, 0.3])  # max high_r = 0.4 < 1.0 → P=0.1
        decision = hook.evaluate(
            trade={}, bar_path_so_far=path, entry_features=entry_features_finite,
            t=3, cached={}, fold_id="F1",
        )
        assert isinstance(decision, Close)
        assert decision.reason == "d1_untradeable"

    def test_raises_when_t_does_not_match(self, entry_features_finite):
        hook = _make_single_hook(threshold=0.5, t=3)
        path = _bar_path([0.0, 0.5, 1.0, 1.5])
        with pytest.raises(ValueError, match="configured t=3"):
            hook.evaluate(
                trade={}, bar_path_so_far=path,
                entry_features=entry_features_finite, t=2, cached={}, fold_id="F1",
            )

    def test_nan_entry_features_abstain(self):
        hook = _make_single_hook(threshold=0.5, t=3)
        # All entry features NaN → archetype abstains. With only one
        # archetype, no archetype has valid features → Close (not Hold,
        # because the classifier IS available for this fold).
        nan_entry = {k: float("nan") for k in ENTRY_FEATURE_KEYS}
        path = _bar_path([0.0, 0.5, 1.0, 1.5])
        decision = hook.evaluate(
            trade={}, bar_path_so_far=path, entry_features=nan_entry,
            t=3, cached={}, fold_id="F1",
        )
        assert isinstance(decision, Close)

    def test_returns_hold_when_fold_has_no_classifier(self, entry_features_finite):
        """Trades in a fold absent from per_fold_classifiers fall through.

        Use-case: Arc 4 F1 has no training data; the YAML registers
        F2..F7 only. F1 trades return Hold and run the legacy cascade.
        """
        hook = _make_single_hook(threshold=0.5, t=3, fold_id="F2")
        path = _bar_path([0.0, 0.5, 1.0, 1.5])
        decision = hook.evaluate(
            trade={}, bar_path_so_far=path, entry_features=entry_features_finite,
            t=3, cached={}, fold_id="F1",
        )
        assert isinstance(decision, Hold)


# ---------------------------------------------------------------------------
# 6. Multi-archetype max-P + tie-break.
# ---------------------------------------------------------------------------


class TestMultiArchetype:
    def _two_arch_hook(self, p_a: float, p_b: float) -> D1Hook:
        order = list(ALL_FEATURE_KEYS)
        policy = _default_policy()
        arch_a = D1ArchetypeConfig(
            label="a",
            feature_order=tuple(order),
            decision_threshold=0.5,
            bar_offset_t=3,
            per_fold_classifiers={"F1": StubClassifier(order, fixed_p=p_a)},
            exit_policy=policy,
        )
        arch_b = D1ArchetypeConfig(
            label="b",
            feature_order=tuple(order),
            decision_threshold=0.5,
            bar_offset_t=3,
            per_fold_classifiers={"F1": StubClassifier(order, fixed_p=p_b)},
            exit_policy=policy,
        )
        return D1Hook([arch_a, arch_b])

    def test_max_p_wins(self, entry_features_finite):
        hook = self._two_arch_hook(p_a=0.6, p_b=0.9)
        decision = hook.evaluate(
            trade={}, bar_path_so_far=_bar_path([0.0, 0.5, 1.0, 1.5]),
            entry_features=entry_features_finite, t=3, cached={}, fold_id="F1",
        )
        assert isinstance(decision, ApplyPolicy)
        assert decision.label == "b"
        assert decision.probability == pytest.approx(0.9)

    def test_tie_resolves_to_first_in_config_order(self, entry_features_finite):
        hook = self._two_arch_hook(p_a=0.8, p_b=0.8)
        decision = hook.evaluate(
            trade={}, bar_path_so_far=_bar_path([0.0, 0.5, 1.0, 1.5]),
            entry_features=entry_features_finite, t=3, cached={}, fold_id="F1",
        )
        assert isinstance(decision, ApplyPolicy)
        assert decision.label == "a"

    def test_all_below_threshold_returns_close(self, entry_features_finite):
        hook = self._two_arch_hook(p_a=0.4, p_b=0.45)
        decision = hook.evaluate(
            trade={}, bar_path_so_far=_bar_path([0.0, 0.5, 1.0, 1.5]),
            entry_features=entry_features_finite, t=3, cached={}, fold_id="F1",
        )
        assert isinstance(decision, Close)


# ---------------------------------------------------------------------------
# 7. apply_d1_hook_per_bar — engine helper (fill resolution, bar gating).
# ---------------------------------------------------------------------------


def _trade_with(t: int, close_rs: list[float], entry_features: dict[str, float]) -> dict:
    return {
        "entry_idx": 0,
        "bar_path": _bar_path(close_rs),
        "entry_features": entry_features,
        "entry_px": 1.0,
        "atr": 0.01,
        "sl_px": 1.0 - 2.0 * 0.01,  # pre-t SL (entry − 2 × ATR)
        "signal_direction": "long",
    }


class TestApplyD1HookPerBar:
    def test_none_hook_is_noop(self, entry_features_finite):
        cached = {"o": np.array([1.0, 1.01, 1.02, 1.03, 1.04])}
        trade = _trade_with(3, [0.0, 0.5, 1.0, 1.5], entry_features_finite)
        out = apply_d1_hook_per_bar(
            hook=None, trade=trade, j=3, entry_idx=0, cached=cached, c_j=1.03,
            fold_id="F1",
        )
        assert out == (None, None, None)

    def test_wrong_bar_is_noop(self, entry_features_finite):
        hook = _make_single_hook(threshold=0.5, t=3)
        cached = {"o": np.array([1.0, 1.01, 1.02, 1.03, 1.04])}
        trade = _trade_with(3, [0.0, 0.5], entry_features_finite)
        # j - entry_idx = 1 != bar_offset_t = 3.
        out = apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=1, entry_idx=0, cached=cached, c_j=1.01,
            fold_id="F1",
        )
        assert out == (None, None, None)

    def test_close_fills_next_bar_open(self, entry_features_finite):
        hook = _make_single_hook(threshold=0.5, t=3)
        opens = np.array([1.0, 1.01, 1.02, 1.03, 1.0399, 1.05])
        cached = {"o": opens}
        trade = _trade_with(3, [0.0, 0.1, 0.2, 0.3], entry_features_finite)  # low mfe → Close
        out = apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=3, entry_idx=0, cached=cached, c_j=1.03,
            fold_id="F1",
        )
        exit_px, exit_bar, exit_reason = out
        assert exit_reason == "d1_untradeable"
        assert exit_bar == 4
        assert exit_px == pytest.approx(1.0399)
        # Audit fields stamped on trade dict (used by trades_all.csv).
        assert trade["d1_decision"] == "close"
        assert trade["classifier_fold_id"] == "F1"

    def test_close_last_bar_fallback_uses_c_j(self, entry_features_finite):
        hook = _make_single_hook(threshold=0.5, t=3)
        # opens has length 4 → indices 0..3. At j=3, j+1=4 is past end.
        opens = np.array([1.0, 1.01, 1.02, 1.03])
        cached = {"o": opens}
        trade = _trade_with(3, [0.0, 0.1, 0.2, 0.3], entry_features_finite)
        out = apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=3, entry_idx=0, cached=cached, c_j=1.0301,
            fold_id="F1",
        )
        exit_px, exit_bar, exit_reason = out
        assert exit_reason == "d1_untradeable"
        assert exit_bar == 3
        assert exit_px == pytest.approx(1.0301)

    def test_apply_policy_installs_policy_on_trade(self, entry_features_finite):
        """ApplyPolicy: helper installs the policy on the trade dict,
        mutates SL via apply_at_accept, and returns no-exit tuple so the
        engine cascade continues.
        """
        hook = _make_single_hook(threshold=0.5, t=3)
        cached = {"o": np.array([1.0, 1.01, 1.02, 1.03, 1.04])}
        # High mfe path → P=0.9 ≥ 0.5 → ApplyPolicy.
        trade = _trade_with(3, [0.0, 0.5, 1.0, 1.5], entry_features_finite)
        pre_sl = trade["sl_px"]
        out = apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=3, entry_idx=0, cached=cached, c_j=1.03,
            fold_id="F1",
        )
        assert out == (None, None, None)
        # apply_at_accept replaced the pre-t SL with entry − 3 × ATR.
        assert trade["sl_px"] == pytest.approx(1.0 - 3.0 * 0.01)
        assert trade["sl_px"] != pre_sl
        assert isinstance(trade["exit_policy"], StepwiseClimberPolicy)
        assert trade["d1_decision"] == "apply_policy"
        assert trade["classifier_fold_id"] == "F1"
        assert trade["d1_archetype_label"] == "stepwise_climber"

    def test_missing_fold_classifier_returns_hold(self, entry_features_finite):
        """Trade with fold_id absent from per_fold_classifiers → Hold →
        helper returns no-exit; trade falls through to legacy cascade.
        """
        hook = _make_single_hook(threshold=0.5, t=3, fold_id="F2")
        cached = {"o": np.array([1.0, 1.01, 1.02, 1.03, 1.04])}
        trade = _trade_with(3, [0.0, 0.5, 1.0, 1.5], entry_features_finite)
        out = apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=3, entry_idx=0, cached=cached, c_j=1.03,
            fold_id="F1",
        )
        assert out == (None, None, None)
        assert trade["d1_decision"] == "no_d1"
        # No policy installed.
        assert "exit_policy" not in trade or trade.get("exit_policy") is None

    def test_missing_entry_features_fall_through(self):
        """Re-entry trades (no entry_features) must fall through without erroring."""
        hook = _make_single_hook(threshold=0.5, t=3)
        cached = {"o": np.array([1.0, 1.01, 1.02, 1.03, 1.04])}
        trade = {
            "entry_idx": 0,
            "bar_path": _bar_path([0.0, 0.5, 1.0, 1.5]),
            # no "entry_features" key
        }
        out = apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=3, entry_idx=0, cached=cached, c_j=1.03,
            fold_id="F1",
        )
        assert out == (None, None, None)


# ---------------------------------------------------------------------------
# 8. Entry-feature builder — signal-bar definitions.
# ---------------------------------------------------------------------------


def _synth_ohlc(n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.001, size=n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.001, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.001, size=n))
    return open_, high, low, close


class TestEntryFeatureBuilder:
    def test_all_eight_keys_present(self):
        o, h, l_, c = _synth_ohlc(60)
        feats = build_entry_features_at_signal_bar(o, h, l_, c, 0.001, 40)
        assert set(feats.keys()) == set(ENTRY_FEATURE_KEYS)

    def test_warmup_returns_nan(self):
        # signal_bar_idx=10 → ret_20bar_atr, pos_in_20bar_range, rsi_14 may be NaN.
        o, h, l_, c = _synth_ohlc(60)
        feats = build_entry_features_at_signal_bar(o, h, l_, c, 0.001, 10)
        assert not np.isfinite(feats["ret_20bar_atr"])
        assert not np.isfinite(feats["pos_in_20bar_range"])

    def test_no_lookahead(self):
        # Mutate future bars; signal-bar features at idx=40 must stay invariant.
        o, h, l_, c = _synth_ohlc(80)
        feats1 = build_entry_features_at_signal_bar(o, h, l_, c, 0.001, 40)
        c2 = c.copy()
        c2[50:] *= 5.0
        h2 = h.copy()
        h2[50:] *= 5.0
        l2 = l_.copy()
        l2[50:] *= 5.0
        feats2 = build_entry_features_at_signal_bar(o, h2, l2, c2, 0.001, 40)
        for k in ENTRY_FEATURE_KEYS:
            assert feats1[k] == feats2[k], f"feature {k} changed under future-bar mutation"


# ---------------------------------------------------------------------------
# 9. from_yaml_dict factory — stub-injected classifier.
# ---------------------------------------------------------------------------


class TestFromYamlDict:
    def test_inline_per_fold_classifiers_are_used(self):
        """PR 2 schema: per_fold_classifiers dict + exit_policy block."""
        order = list(ALL_FEATURE_KEYS)
        stubs = {f: StubClassifier(order, fixed_p=0.99) for f in ("F2", "F3")}
        yaml_block = [
            {
                "label": "stepwise_climber_arc4_cluster1",
                "feature_order": order,
                "decision_threshold": 0.1647,
                "bar_offset_t": 1,
                "per_fold_classifiers": stubs,
                "exit_policy": {
                    "type": "stepwise_climber",
                    "archetype_sl_r": 1.0,
                    "r_in_atr": 3.0,
                    "mfe_lock_r": 1.0,
                    "trail_from_high_r": 0.75,
                },
            }
        ]
        hook = D1Hook.from_yaml_dict(yaml_block)
        assert hook.bar_offset_t == 1
        assert hook.archetypes[0].label == "stepwise_climber_arc4_cluster1"
        assert hook.archetypes[0].per_fold_classifiers["F2"] is stubs["F2"]
        assert isinstance(hook.archetypes[0].exit_policy, StepwiseClimberPolicy)
        assert hook.archetypes[0].exit_policy.archetype_sl_r == 1.0

    def test_empty_block_raises(self):
        with pytest.raises(ValueError, match="empty"):
            D1Hook.from_yaml_dict([])

    def test_pr1_schema_fails_loud(self):
        """PR 1 ``classifier_path`` schema is no longer accepted."""
        order = list(ALL_FEATURE_KEYS)
        yaml_block = [
            {
                "label": "old_schema",
                "feature_order": order,
                "decision_threshold": 0.5,
                "bar_offset_t": 3,
                "classifier_path": "some/path.joblib",
            }
        ]
        with pytest.raises(ValueError, match="PR 1 schema"):
            D1Hook.from_yaml_dict(yaml_block)

    def test_missing_exit_policy_raises(self):
        order = list(ALL_FEATURE_KEYS)
        stub = StubClassifier(order)
        yaml_block = [
            {
                "label": "no_policy",
                "feature_order": order,
                "decision_threshold": 0.5,
                "bar_offset_t": 3,
                "per_fold_classifiers": {"F1": stub},
                # no exit_policy
            }
        ]
        with pytest.raises(ValueError, match="exit_policy.*required"):
            D1Hook.from_yaml_dict(yaml_block)

    def test_missing_per_fold_classifiers_raises(self):
        order = list(ALL_FEATURE_KEYS)
        yaml_block = [
            {
                "label": "no_classifiers",
                "feature_order": order,
                "decision_threshold": 0.5,
                "bar_offset_t": 3,
                "exit_policy": {
                    "type": "stepwise_climber",
                    "archetype_sl_r": 1.0,
                    "r_in_atr": 3.0,
                    "mfe_lock_r": 1.0,
                    "trail_from_high_r": 0.75,
                },
                # no per_fold_classifiers
            }
        ]
        with pytest.raises(ValueError, match="per_fold_classifiers.*required"):
            D1Hook.from_yaml_dict(yaml_block)

    def test_unknown_exit_policy_type_raises(self):
        order = list(ALL_FEATURE_KEYS)
        yaml_block = [
            {
                "label": "bogus_policy",
                "feature_order": order,
                "decision_threshold": 0.5,
                "bar_offset_t": 3,
                "per_fold_classifiers": {"F1": StubClassifier(order)},
                "exit_policy": {"type": "not_a_real_policy"},
            }
        ]
        with pytest.raises(ValueError, match="unknown exit_policy type"):
            D1Hook.from_yaml_dict(yaml_block)


# ---------------------------------------------------------------------------
# 10. Engine integration — patched lines exist; baseline is byte-identical.
# ---------------------------------------------------------------------------


ENGINE_PATH = PROJECT_ROOT / "scripts" / "phase_kgl_v2_4h_wfo.py"


class TestEnginePatch:
    """The engine patch is exercised end-to-end by the WFO pytest suite.
    Here we verify the static contract: globals default to off, the hook
    call site is exactly where the spec says, and SL still uses 2.0×ATR.
    """

    def test_engine_file_has_d1_hook_default_none(self):
        src = ENGINE_PATH.read_text(encoding="utf-8")
        assert re.search(r"^D1_HOOK:\s*\"?D1Hook\s*\|\s*None\"?\s*=\s*None", src, re.M), (
            "D1_HOOK must default to None at module scope"
        )

    def test_engine_calls_helper_with_close_branch(self):
        src = ENGINE_PATH.read_text(encoding="utf-8")
        assert "apply_d1_hook_per_bar(" in src
        # The hook call sits in the per-trade exit cascade — before SL.
        d1_call_idx = src.index("apply_d1_hook_per_bar(")
        sl_priority_idx = src.index("# Priority 1: Intrabar hard SL")
        assert d1_call_idx < sl_priority_idx, (
            "D1 hook must fire before the Priority-1 SL check"
        )

    def test_entry_features_stored_only_when_hook_active(self):
        src = ENGINE_PATH.read_text(encoding="utf-8")
        # The build_entry_features call must be guarded by an `if D1_HOOK is not None:`.
        m = re.search(
            r"if\s+D1_HOOK\s+is\s+not\s+None:\s*\n\s+entry_features_dict\s*=\s*"
            r"build_entry_features_at_signal_bar",
            src,
        )
        assert m, (
            "entry_features computation must be guarded by `if D1_HOOK is not None:` "
            "to preserve byte-identity when D1 is off"
        )

    def test_sl_formula_is_2_0_times_atr_long(self):
        """Pre-t SL must remain entry - 2.0 × ATR (KH-24 baseline).

        Asserted at the source level rather than the runtime level — the
        SL formula on the long path is `entry_px - SL_MULT * a` with
        SL_MULT = 2.0 at module scope.
        """
        src = ENGINE_PATH.read_text(encoding="utf-8")
        assert re.search(r"^SL_MULT\s*=\s*2\.0", src, re.M), "SL_MULT must be 2.0"
        assert "sl_px        = entry_px - SL_MULT * a" in src, (
            "long-side SL formula must be `entry_px - SL_MULT * a`"
        )

    def test_d1_hook_none_means_no_engine_state_change(self):
        """Static contract: with D1_HOOK = None, the hook block is a no-op."""
        src = ENGINE_PATH.read_text(encoding="utf-8")
        # Locate the hook block and assert the early-return guard exists.
        m = re.search(
            r"if\s+D1_HOOK\s+is\s+not\s+None:\s*\n\s+_d1_px,\s*_d1_bar,\s*_d1_reason"
            r"\s*=\s*apply_d1_hook_per_bar\(",
            src,
        )
        assert m, "hook block must be guarded by `if D1_HOOK is not None:`"

    def test_engine_passes_fold_id_to_hook(self):
        """PR 2: helper receives fold_id from trade dict so the per-fold
        classifier dispatch works.
        """
        src = ENGINE_PATH.read_text(encoding="utf-8")
        m = re.search(
            r"apply_d1_hook_per_bar\([^)]*fold_id\s*=\s*trade\.get\(\"fold_id\"\)",
            src,
            re.S,
        )
        assert m, "engine call to apply_d1_hook_per_bar must thread fold_id"

    def test_engine_invokes_per_bar_policy_update(self):
        """PR 2: per-bar update fires every bar from accept onward,
        before Priority-1 SL.
        """
        src = ENGINE_PATH.read_text(encoding="utf-8")
        update_idx = src.find('trade["exit_policy"].update_per_bar')
        sl_priority_idx = src.index("# Priority 1: Intrabar hard SL")
        assert update_idx != -1, (
            "engine must call trade['exit_policy'].update_per_bar(...)"
        )
        assert update_idx < sl_priority_idx, (
            "per-bar policy update must run before the Priority-1 SL check"
        )

    def test_engine_stamps_fold_id_at_trade_open(self):
        """PR 2: trade dict gains fold_id at entry — used to select the
        per-fold classifier.
        """
        src = ENGINE_PATH.read_text(encoding="utf-8")
        assert "_fold_id_for_entry(bar_date)" in src
        m = re.search(
            r"\"fold_id\":\s+_fold_id_for_entry\(bar_date\)",
            src,
        )
        assert m, "trade dict must carry fold_id at open"

    def test_engine_audit_columns_gated_on_hook(self):
        """PR 2: trades_all.csv gains new D1 audit columns only when
        D1_HOOK is configured — KH-24 baseline runs stay byte-identical.
        """
        src = ENGINE_PATH.read_text(encoding="utf-8")
        m = re.search(
            r"_d1_audit_extras:\s*dict\s*=\s*\{\}\s*\n\s+if\s+D1_HOOK\s+is\s+not\s+None:",
            src,
        )
        assert m, (
            "audit columns must be gated on `if D1_HOOK is not None:` so "
            "the baseline trades_all.csv stays byte-identical"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
