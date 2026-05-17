"""Tests for Pipeline D1 archetype exit policies (PR 2).

Two layers:

* Unit tests on :class:`~core.exit_policies.StepwiseClimberPolicy` —
  ``apply_at_accept`` swaps the pre-t SL for ``entry − archetype_sl_r ×
  r_in_atr × ATR``; ``update_per_bar`` ratchets SL via MFE-lock + trail.
* Integration tests on :func:`~core.d1_pipeline.apply_d1_hook_per_bar`
  + stubs — per-fold classifier dispatch, ApplyPolicy installs the
  policy on the trade dict, audit fields surface in the completed trade.

No joblib / sklearn — stubs implement ``predict_proba``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.d1_pipeline import (
    ApplyPolicy,
    Close,
    D1ArchetypeConfig,
    D1Hook,
    apply_d1_hook_per_bar,
)
from core.exit_policies import (
    StepwiseClimberPolicy,
    build_exit_policy,
)
from core.features_path_so_far import ALL_FEATURE_KEYS

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


# Arc 4 cluster 1 parameters — verbatim from results/l_arc_4/step4/
# cluster_1_D1_policy.yaml (archetype_R_frame_atr_mult: 3.0, pre/post-t SL
# multipliers 2.0/3.0, MFE-lock at 1.0, trail 0.75 from new high).
ARC4_C1_KW = dict(
    archetype_sl_r=1.0,
    r_in_atr=3.0,
    mfe_lock_r=1.0,
    trail_from_high_r=0.75,
)


def _entry_features_finite() -> dict[str, float]:
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
    """v1.3 bar_path rows with running mfe/mae over close_r ± 0.1.

    bar_offset starts at 0 and increments.
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


def _new_trade(
    entry_px: float = 1.0,
    atr: float = 0.01,
    direction: str = "long",
    bar_path: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    pre_t_sl = entry_px - 2.0 * atr if direction == "long" else entry_px + 2.0 * atr
    return {
        "entry_idx": 0,
        "entry_px": entry_px,
        "atr": atr,
        "sl_px": pre_t_sl,
        "signal_direction": direction,
        "bar_path": bar_path if bar_path is not None else _bar_path([0.0]),
        "entry_features": _entry_features_finite(),
    }


# ---------------------------------------------------------------------------
# 1. StepwiseClimberPolicy.apply_at_accept.
# ---------------------------------------------------------------------------


class TestApplyAtAccept:
    def test_long_replaces_pre_t_sl_with_archetype_sl(self):
        """SL replaces pre-t 2×ATR with entry − archetype_sl_r × r_in_atr × ATR."""
        policy = StepwiseClimberPolicy(**ARC4_C1_KW)
        trade = _new_trade(entry_px=1.0, atr=0.01, direction="long")
        # Pre-t SL is at entry − 2 × ATR = 1.0 − 0.02 = 0.98.
        assert trade["sl_px"] == pytest.approx(0.98)
        policy.apply_at_accept(trade, entry_px=1.0, atr_at_entry=0.01)
        # Cluster 1: SL = entry − 1.0 × 3.0 × ATR = entry − 3 × ATR = 0.97.
        assert trade["sl_px"] == pytest.approx(0.97)
        assert trade["mfe_lock_fired_bar"] is None

    def test_short_replaces_pre_t_sl_with_archetype_sl(self):
        policy = StepwiseClimberPolicy(**ARC4_C1_KW)
        trade = _new_trade(entry_px=1.0, atr=0.01, direction="short")
        # Pre-t SL short = entry + 2 × ATR = 1.02.
        assert trade["sl_px"] == pytest.approx(1.02)
        policy.apply_at_accept(trade, entry_px=1.0, atr_at_entry=0.01)
        # Short: SL = entry + 3 × ATR = 1.03.
        assert trade["sl_px"] == pytest.approx(1.03)


# ---------------------------------------------------------------------------
# 2. StepwiseClimberPolicy.update_per_bar — MFE-lock + trail dynamics.
# ---------------------------------------------------------------------------


class TestUpdatePerBar:
    def _bootstrap(
        self, mfe_so_far_r_list: list[float], direction: str = "long"
    ) -> tuple[StepwiseClimberPolicy, dict[str, Any]]:
        policy = StepwiseClimberPolicy(**ARC4_C1_KW)
        trade = _new_trade(direction=direction)
        policy.apply_at_accept(trade, entry_px=1.0, atr_at_entry=0.01)
        # Replace the trade's bar_path with a synthetic path whose
        # mfe_so_far_r matches the test's expected MFE trajectory at
        # each bar.
        rows: list[dict[str, float]] = []
        for k, mfe_r in enumerate(mfe_so_far_r_list):
            rows.append(
                {
                    "bar_offset": k + 1,
                    "high_r": float(mfe_r),
                    "low_r": -0.5,
                    "close_r": float(mfe_r) - 0.1,
                    "mfe_so_far_r": float(mfe_r),
                    "mae_so_far_r": -0.5,
                    "is_held": 1,
                }
            )
        trade["bar_path"] = rows
        return policy, trade

    def test_before_lock_low_mfe_keeps_post_t_sl(self):
        """MFE < 1R and on the same bar as accept (no prior bar after
        accept) → trail doesn't fire yet, SL stays at the post-t value.
        """
        policy, trade = self._bootstrap([0.3])
        policy.update_per_bar(
            trade, trade["bar_path"][0], entry_px=1.0, atr_at_entry=0.01
        )
        # mfe = 0.3 R < mfe_lock_r → lock did not fire; trail doesn't
        # operate before lock either (per spec). SL stays at the
        # archetype's post-t value (entry − 3 × ATR).
        assert trade["sl_px"] == pytest.approx(0.97)
        assert trade["mfe_lock_fired_bar"] is None

    def test_lock_fires_at_one_r(self):
        """First bar with MFE ≥ 1R: lock fires AND trail arms same-bar.

        Effective SL = max(post-t SL, BE, trail_stop). With MFE = 1R:
        trail = entry + (1.0 − 0.75) × R = entry + 0.25R. max(entry-3R,
        entry, entry+0.25R) = entry + 0.25R. Matches Step 5 simulator
        (`scripts/l_arc_4/step5_stability.py:294-298`).
        """
        policy, trade = self._bootstrap([1.0])
        policy.update_per_bar(
            trade, trade["bar_path"][0], entry_px=1.0, atr_at_entry=0.01
        )
        assert trade["mfe_lock_fired_bar"] == 1
        # Trail at MFE=1R = entry + 0.25R = 1.0 + 0.0075.
        assert trade["sl_px"] == pytest.approx(1.0 + 0.25 * 3.0 * 0.01)

    def test_lock_doesnt_reverse_on_subsequent_drawdown(self):
        """Once lock fires, dropping back below 1R doesn't reset SL.

        mfe_so_far_r is monotonic (running max), but close_r can dip
        below entry post-peak. The lock persists.
        """
        policy, trade = self._bootstrap([1.0, 1.0])
        # First bar: lock fires.
        policy.update_per_bar(
            trade, trade["bar_path"][0], entry_px=1.0, atr_at_entry=0.01
        )
        # Second bar: same MFE peak, but close_r could be very low.
        # Engineer a row with close_r = -1.5 R, but mfe_so_far_r stays 1.0.
        trade["bar_path"][1]["close_r"] = -1.5
        policy.update_per_bar(
            trade, trade["bar_path"][1], entry_px=1.0, atr_at_entry=0.01
        )
        # On the bar AFTER the lock fire, trail kicks in:
        # trail_stop = entry + (1.0 - 0.75) × R = entry + 0.25 R = 1.0075.
        # SL ratchets max(1.0, 1.0075) = 1.0075. SL never falls.
        assert trade["sl_px"] == pytest.approx(1.0 + 0.25 * 3.0 * 0.01)
        assert trade["mfe_lock_fired_bar"] == 1  # unchanged

    def test_trail_overtakes_be_at_high_mfe(self):
        """At MFE = 1.75 R (post-lock bar), trail_stop = entry + 1.0 R.
        At MFE = 2.0 R, trail = entry + 1.25 R. SL ratchets up.
        """
        policy, trade = self._bootstrap([1.0, 1.75, 2.0])
        policy.update_per_bar(
            trade, trade["bar_path"][0], entry_px=1.0, atr_at_entry=0.01
        )
        policy.update_per_bar(
            trade, trade["bar_path"][1], entry_px=1.0, atr_at_entry=0.01
        )
        # MFE = 1.75 → trail = entry + (1.75 − 0.75) × 3 × ATR = entry + 1.0 × 0.03 = 1.03.
        assert trade["sl_px"] == pytest.approx(1.03)
        policy.update_per_bar(
            trade, trade["bar_path"][2], entry_px=1.0, atr_at_entry=0.01
        )
        # MFE = 2.0 → trail = entry + (2.0 − 0.75) × 3 × ATR = entry + 1.25 × 0.03 = 1.0375.
        assert trade["sl_px"] == pytest.approx(1.0375)

    def test_worked_example_arc4_c1_trajectory(self):
        """Spec worked example: pre-t → accept → lock → 2R → 3R.

        Arc 4 cluster 1 long with entry = 1.0, ATR = 0.01 (R = 0.03).
        Matches Step 5 simulator's same-bar trail-arm semantics:

        * pre-t:  entry − 2 × ATR = 0.98
        * accept: entry − 3 × ATR = 0.97
        * lock @ MFE=1R (same-bar trail): entry + 0.25R = 1.0075
        * MFE=2R: entry + (2 − 0.75) × R = entry + 1.25R = 1.0375
        * MFE=3R: entry + (3 − 0.75) × R = entry + 2.25R = 1.0675
        """
        policy = StepwiseClimberPolicy(**ARC4_C1_KW)
        trade = _new_trade(entry_px=1.0, atr=0.01, direction="long")
        # 1) pre-t SL (engine-set, before policy installs)
        assert trade["sl_px"] == pytest.approx(0.98)
        # 2) accept
        policy.apply_at_accept(trade, entry_px=1.0, atr_at_entry=0.01)
        assert trade["sl_px"] == pytest.approx(0.97)
        # Build a 3-bar post-accept path: MFE = 1.0, 2.0, 3.0.
        rows = [
            {
                "bar_offset": k + 1, "high_r": v, "low_r": -0.5, "close_r": v - 0.1,
                "mfe_so_far_r": v, "mae_so_far_r": -0.5, "is_held": 1,
            }
            for k, v in enumerate([1.0, 2.0, 3.0])
        ]
        trade["bar_path"] = rows
        # 3) lock fires + trail arms same-bar
        policy.update_per_bar(trade, rows[0], entry_px=1.0, atr_at_entry=0.01)
        assert trade["sl_px"] == pytest.approx(1.0075)
        # 4) MFE = 2R
        policy.update_per_bar(trade, rows[1], entry_px=1.0, atr_at_entry=0.01)
        assert trade["sl_px"] == pytest.approx(1.0375)
        # 5) MFE = 3R
        policy.update_per_bar(trade, rows[2], entry_px=1.0, atr_at_entry=0.01)
        assert trade["sl_px"] == pytest.approx(1.0675)

    def test_short_direction_mirrors_long(self):
        """Short trades: BE ceiling at entry, trail pulls SL DOWN as MFE
        grows. Same-bar trail-arm semantics mirrored from long.
        """
        policy = StepwiseClimberPolicy(**ARC4_C1_KW)
        trade = _new_trade(entry_px=1.0, atr=0.01, direction="short")
        policy.apply_at_accept(trade, entry_px=1.0, atr_at_entry=0.01)
        # Short post-t SL = entry + 3 × ATR = 1.03.
        assert trade["sl_px"] == pytest.approx(1.03)
        rows = [
            {
                "bar_offset": k + 1, "high_r": -0.5, "low_r": -v, "close_r": -(v - 0.1),
                "mfe_so_far_r": v, "mae_so_far_r": -0.5, "is_held": 1,
            }
            for k, v in enumerate([1.0, 2.0])
        ]
        trade["bar_path"] = rows
        policy.update_per_bar(trade, rows[0], entry_px=1.0, atr_at_entry=0.01)
        # Short lock fires + trail same-bar at MFE=1R:
        # trail = entry − (1 − 0.75) × R = 1 − 0.0075 = 0.9925.
        # SL = min(1.03, 1.0, 0.9925) = 0.9925.
        assert trade["sl_px"] == pytest.approx(0.9925)
        policy.update_per_bar(trade, rows[1], entry_px=1.0, atr_at_entry=0.01)
        # Short trail at MFE=2R: SL = entry − (2 − 0.75) × R = 1 − 0.0375 = 0.9625.
        assert trade["sl_px"] == pytest.approx(0.9625)

    def test_sl_only_ratchets_never_loosens(self):
        """SL never moves AWAY from price — trail dipping wouldn't lower
        the SL once it has been pulled tighter.
        """
        policy, trade = self._bootstrap([1.0, 2.0])
        policy.update_per_bar(
            trade, trade["bar_path"][0], entry_px=1.0, atr_at_entry=0.01
        )
        policy.update_per_bar(
            trade, trade["bar_path"][1], entry_px=1.0, atr_at_entry=0.01
        )
        # SL is at entry + 1.25 R = 1.0375 after MFE=2 bar.
        peak_sl = trade["sl_px"]
        # Now feed a bar where mfe_so_far_r is still 2.0 (it's a running
        # max) but the engine called update_per_bar again — SL must not
        # loosen.
        trade["bar_path"].append(
            {
                "bar_offset": 3, "high_r": 0.5, "low_r": -0.5, "close_r": 0.0,
                "mfe_so_far_r": 2.0, "mae_so_far_r": -0.5, "is_held": 1,
            }
        )
        policy.update_per_bar(
            trade, trade["bar_path"][2], entry_px=1.0, atr_at_entry=0.01
        )
        assert trade["sl_px"] == pytest.approx(peak_sl)


# ---------------------------------------------------------------------------
# 3. build_exit_policy factory.
# ---------------------------------------------------------------------------


class TestBuildExitPolicy:
    def test_stepwise_climber_factory(self):
        spec = {
            "type": "stepwise_climber",
            "archetype_sl_r": 1.0,
            "r_in_atr": 3.0,
            "mfe_lock_r": 1.0,
            "trail_from_high_r": 0.75,
        }
        policy = build_exit_policy(spec)
        assert isinstance(policy, StepwiseClimberPolicy)
        assert policy.archetype_sl_r == 1.0
        assert policy.r_in_atr == 3.0
        assert policy.mfe_lock_r == 1.0
        assert policy.trail_from_high_r == 0.75

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown exit_policy type"):
            build_exit_policy({"type": "made_up_policy"})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            build_exit_policy({"archetype_sl_r": 1.0})


# ---------------------------------------------------------------------------
# 4. Per-fold classifier dispatch (integration).
# ---------------------------------------------------------------------------


class _ConstantClassifier:
    """``predict_proba`` returns a fixed P regardless of features."""

    def __init__(self, n_features: int, p: float):
        self.n_features_in_ = n_features
        self._p = float(p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        out = np.zeros((X.shape[0], 2), dtype=np.float64)
        out[:, 1] = self._p
        out[:, 0] = 1.0 - self._p
        return out


def _make_per_fold_hook(
    p_by_fold: dict[str, float], t: int = 1, threshold: float = 0.1647
) -> D1Hook:
    order = list(ALL_FEATURE_KEYS)
    per_fold = {
        fold: _ConstantClassifier(len(order), p)
        for fold, p in p_by_fold.items()
    }
    arch = D1ArchetypeConfig(
        label="stepwise_climber_arc4_cluster1",
        feature_order=tuple(order),
        decision_threshold=threshold,
        bar_offset_t=t,
        per_fold_classifiers=per_fold,
        exit_policy=StepwiseClimberPolicy(**ARC4_C1_KW),
    )
    return D1Hook([arch])


class TestPerFoldDispatch:
    def test_dispatch_picks_fold_specific_classifier(self):
        """F3 classifier P=0.99 accepts; F2 classifier P=0.05 rejects.

        A trade in F3 should be accepted (ApplyPolicy); a trade in F2
        should be rejected (Close).
        """
        hook = _make_per_fold_hook({"F2": 0.05, "F3": 0.99})
        path = _bar_path([0.0, 0.5])  # bars 0..1
        # Trade in F3:
        d_f3 = hook.evaluate(
            trade={}, bar_path_so_far=path, entry_features=_entry_features_finite(),
            t=1, cached={}, fold_id="F3",
        )
        assert isinstance(d_f3, ApplyPolicy)
        # Trade in F2:
        d_f2 = hook.evaluate(
            trade={}, bar_path_so_far=path, entry_features=_entry_features_finite(),
            t=1, cached={}, fold_id="F2",
        )
        assert isinstance(d_f2, Close)

    def test_helper_passes_fold_id_through(self):
        hook = _make_per_fold_hook({"F2": 0.05, "F3": 0.99})
        cached = {"o": np.array([1.0, 1.01, 1.02])}
        trade_f3 = _new_trade(bar_path=_bar_path([0.0, 0.5]))
        out_f3 = apply_d1_hook_per_bar(
            hook=hook, trade=trade_f3, j=1, entry_idx=0,
            cached=cached, c_j=1.01, fold_id="F3",
        )
        assert out_f3 == (None, None, None)  # ApplyPolicy → no exit
        assert trade_f3.get("d1_decision") == "apply_policy"
        assert trade_f3.get("classifier_fold_id") == "F3"
        assert isinstance(trade_f3.get("exit_policy"), StepwiseClimberPolicy)

        trade_f2 = _new_trade(bar_path=_bar_path([0.0, 0.5]))
        out_f2 = apply_d1_hook_per_bar(
            hook=hook, trade=trade_f2, j=1, entry_idx=0,
            cached=cached, c_j=1.01, fold_id="F2",
        )
        assert out_f2[2] == "d1_untradeable"  # Close
        assert trade_f2.get("d1_decision") == "close"
        assert trade_f2.get("classifier_fold_id") == "F2"

    def test_unknown_fold_returns_hold(self):
        """Trade in fold not present in per_fold_classifiers (e.g. Arc 4 F1)
        → Hold → engine falls through to legacy cascade.
        """
        hook = _make_per_fold_hook({"F2": 0.99, "F3": 0.99})
        cached = {"o": np.array([1.0, 1.01, 1.02])}
        trade = _new_trade(bar_path=_bar_path([0.0, 0.5]))
        out = apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=1, entry_idx=0,
            cached=cached, c_j=1.01, fold_id="F1",
        )
        assert out == (None, None, None)
        assert trade.get("d1_decision") == "no_d1"

    def test_apply_policy_install_sets_audit_fields(self):
        """Audit fields populated by helper survive in trade dict for
        completed-trade record consumption.
        """
        hook = _make_per_fold_hook({"F3": 0.99})
        cached = {"o": np.array([1.0, 1.01, 1.02])}
        trade = _new_trade(bar_path=_bar_path([0.0, 0.5]))
        apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=1, entry_idx=0,
            cached=cached, c_j=1.01, fold_id="F3",
        )
        assert trade["mfe_lock_fired_bar"] is None  # not yet fired
        assert trade["trail_active_from_bar"] is not None
        assert trade["d1_archetype_label"] == "stepwise_climber_arc4_cluster1"
        assert trade["d1_probability"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 5. Reject branch: Close decision installs no policy.
# ---------------------------------------------------------------------------


class TestRejectBranch:
    def test_close_does_not_install_policy(self):
        """Verify Close (rejection) leaves the trade's exit_policy unset
        — preserving PR 1 behaviour for trades that fail the gate.
        """
        hook = _make_per_fold_hook({"F2": 0.05})
        cached = {"o": np.array([1.0, 1.01, 1.02])}
        trade = _new_trade(bar_path=_bar_path([0.0, 0.5]))
        out = apply_d1_hook_per_bar(
            hook=hook, trade=trade, j=1, entry_idx=0,
            cached=cached, c_j=1.01, fold_id="F2",
        )
        assert out[2] == "d1_untradeable"
        # No exit_policy key was added.
        assert "exit_policy" not in trade or trade["exit_policy"] is None


# ---------------------------------------------------------------------------
# 6. Determinism — two-run byte-identity of a stub end-to-end run.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_run_byte_identical(self):
        """Two identical hook invocations on the same trade dict yield
        identical SL trajectories. Catches accidental float drift or
        nondeterministic policy ordering.
        """
        hook = _make_per_fold_hook({"F3": 0.99})
        cached = {"o": np.array([1.0, 1.01, 1.02, 1.03, 1.04])}

        def simulate() -> list[float]:
            trade = _new_trade(bar_path=_bar_path([0.0, 0.5]))
            apply_d1_hook_per_bar(
                hook=hook, trade=trade, j=1, entry_idx=0,
                cached=cached, c_j=1.01, fold_id="F3",
            )
            # Run two post-accept bars with MFE = 1.5, 2.5.
            sls: list[float] = [trade["sl_px"]]
            for bar_idx, mfe in enumerate([1.5, 2.5], start=2):
                row = {
                    "bar_offset": bar_idx, "high_r": mfe, "low_r": -0.5,
                    "close_r": mfe - 0.1, "mfe_so_far_r": mfe, "mae_so_far_r": -0.5,
                    "is_held": 1,
                }
                trade["bar_path"].append(row)
                trade["exit_policy"].update_per_bar(
                    trade, row, entry_px=trade["entry_px"], atr_at_entry=trade["atr"],
                )
                sls.append(trade["sl_px"])
            return sls

        run1 = simulate()
        run2 = simulate()
        assert run1 == run2


# ---------------------------------------------------------------------------
# 7. Engine-patch static contract (PR 2 patch points exist).
# ---------------------------------------------------------------------------


class TestEnginePatchPR2:
    def test_fold_id_helper_present(self):
        from pathlib import Path

        engine_src = (Path(__file__).resolve().parents[1]
                      / "scripts" / "phase_kgl_v2_4h_wfo.py").read_text(
            encoding="utf-8"
        )
        assert "def _fold_id_for_entry" in engine_src
        assert "f\"F{int(f['fold'])}\"" in engine_src

    def test_engine_imports_exit_policies(self):
        from pathlib import Path

        engine_src = (Path(__file__).resolve().parents[1]
                      / "scripts" / "phase_kgl_v2_4h_wfo.py").read_text(
            encoding="utf-8"
        )
        # Indirect — core.d1_pipeline re-exports nothing from
        # exit_policies, so the engine doesn't need to import the policy
        # module directly. It accesses the policy via trade["exit_policy"].
        # Just check that the per-bar update line is wired.
        assert 'trade["exit_policy"].update_per_bar' in engine_src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
