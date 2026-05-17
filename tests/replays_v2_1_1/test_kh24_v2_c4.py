"""Unit tests for Open-18 Replay #2 KH-24 v2.0 c4 under v2.1.1.

Covers SL truncation, peak_mfe_bar identification, pre-peak monotonicity,
frac_wrong_way_pre_peak (Def C), MFE-based shape_tag classifier, and cap-binding rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.replays_v2_1_1.kh24_v2_c4.step3 import (  # noqa: E402
    classify_shape_tag,
    evaluate_sl_for_cluster,
    monotonicity_pre_peak,
    peak_mfe_bar_in_truncated,
    truncate_path,
    wrong_way_pre_peak_hit,
)


FLOORS = {
    "monotonicity_pre_peak_min": 0.55,
    "fwd_mfe_h240_p50_min": 1.5,
    "frac_reach_1R_min": 0.70,
    "frac_wrong_way_pre_peak_max": 0.30,
    "size_fraction_min": 0.10,
    "shape_tag_admitted": ["tight_unimodal", "heavy_right_tail", "bimodal_separated"],
}
BIMODAL_CFG = {
    "hartigan_dip_p_max": 0.05,
    "min_mode_mass": 0.20,
    "min_mode_separation_r": 1.0,
}


# ---------------------------------------------------------------------------------------

def test_sl_truncation_first_hit_at_known_bar():
    """SL=1.5 ATR ⇒ close_r threshold -0.75 (atr_units_per_R=2.0). MAE crosses at bar 3."""
    mae = np.array([-0.10, -0.30, -0.60, -0.80, -1.20, -0.50])
    trunc_idx, sl_hit = truncate_path(mae, close_r_threshold=-0.75)
    assert sl_hit is True
    assert trunc_idx == 3, f"expected bar 3 (first crossing of -0.75), got {trunc_idx}"


def test_sl_never_hit_returns_last_bar():
    """Path that never hits the SL threshold — truncation idx is last bar, sl_hit=False."""
    mae = np.array([-0.10, -0.20, -0.30, -0.40])
    trunc_idx, sl_hit = truncate_path(mae, close_r_threshold=-0.75)
    assert sl_hit is False
    assert trunc_idx == 3, "should return last bar index when never hit"


def test_peak_mfe_bar_identified_within_truncated_path():
    """Peak MFE is at bar 4 in raw path but truncation puts it at bar 2."""
    mfe = np.array([0.0, 0.2, 0.8, 0.5, 1.5, 1.2])
    # If truncated to first 3 bars (bars 0..2), peak should be at bar 2
    peak_bar = peak_mfe_bar_in_truncated(mfe[:3])
    assert peak_bar == 2

    # On full path peak is at bar 4
    peak_bar_full = peak_mfe_bar_in_truncated(mfe)
    assert peak_bar_full == 4


def test_monotonicity_pre_peak_manual_calculation():
    """In-profit sequence: [0.1, 0.05, 0.2, 0.15, 0.3]. Diffs: -0.05, +0.15, -0.05,
    +0.15. Non-decreasing fraction: 2/4 = 0.50.
    """
    close_r = np.array([-0.1, 0.1, 0.05, -0.05, 0.2, 0.15, 0.3])
    result = monotonicity_pre_peak(close_r)
    assert result == pytest.approx(2 / 4)


def test_monotonicity_zero_in_profit_returns_zero():
    """Path with no in-profit bars (all close_r <= 0)."""
    close_r = np.array([-0.1, -0.05, -0.2, 0.0])
    assert monotonicity_pre_peak(close_r) == 0.0


def test_frac_wrong_way_pre_peak_def_c_only_counts_pre_peak():
    """Two synthetic trades:
       Trade A: MAE crosses threshold at bar 1 (BEFORE peak at bar 4) — should count.
       Trade B: MAE crosses at bar 5 (AFTER peak at bar 2) — should NOT count for
                wrong_way_pre_peak (post-peak collapse).
    """
    # Trade A — pre-peak hit
    mae_a = np.array([-0.10, -0.80, -0.30, -0.20, -0.15])  # threshold -0.75 hit at bar 1
    peak_bar_a = 4  # peak of MFE assumed at bar 4
    pre_peak_a = mae_a[: peak_bar_a + 1]
    assert wrong_way_pre_peak_hit(pre_peak_a, close_r_threshold=-0.75) is True

    # Trade B — post-peak hit only
    mae_b = np.array([-0.10, -0.20, -0.30, -0.40, -0.50, -0.80])  # hits at bar 5
    peak_bar_b = 2
    pre_peak_b = mae_b[: peak_bar_b + 1]
    assert wrong_way_pre_peak_hit(pre_peak_b, close_r_threshold=-0.75) is False


def test_shape_tag_heavy_right_tail():
    """Right-skewed unimodal distribution (long tail): p95/p50 > 3.0."""
    rng = np.random.default_rng(42)
    # Exponential gives strong right tail
    vals = rng.exponential(scale=1.0, size=300)
    res = classify_shape_tag(vals, BIMODAL_CFG)
    assert res.tag == "heavy_right_tail", (
        f"expected heavy_right_tail, got {res.tag} "
        f"(p95/p50={res.p95_over_p50}, dip_p={res.dip_p})"
    )


def test_shape_tag_tight_unimodal():
    """Tight unimodal Gaussian centred at 2.0 with small std: p95/p50 ≈ 1.x."""
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=2.0, scale=0.2, size=300)
    res = classify_shape_tag(vals, BIMODAL_CFG)
    assert res.tag == "tight_unimodal", (
        f"expected tight_unimodal, got {res.tag} "
        f"(p95/p50={res.p95_over_p50}, dip_p={res.dip_p})"
    )


def test_shape_tag_bimodal_separated():
    """Two well-separated modes (centres 0.5 and 3.0, each n=200, std 0.15)."""
    rng = np.random.default_rng(7)
    mode1 = rng.normal(loc=0.5, scale=0.15, size=200)
    mode2 = rng.normal(loc=3.0, scale=0.15, size=200)
    vals = np.concatenate([mode1, mode2])
    res = classify_shape_tag(vals, BIMODAL_CFG)
    assert res.tag == "bimodal_separated", (
        f"expected bimodal_separated, got {res.tag} "
        f"(dip_p={res.dip_p}, modes={res.mode_locations}, "
        f"masses={res.mode_masses}, sep={res.mode_separation})"
    )


def test_cap_binding_rate_half_bound():
    """Synthetic pool: 2 trades cap-bound (SL never hit), 2 trades SL-hit early.
    cap_binding_rate should be 0.5.
    """
    rows = []
    # Two cap-bound: 4 bars, never crosses threshold
    for tid in ("T1_cap", "T2_cap"):
        for b in range(4):
            rows.append({
                "trade_id": tid,
                "bar_offset": b,
                "close_r": 0.5 + 0.1 * b,
                "mfe_so_far_r": 0.5 + 0.1 * b,
                "mae_so_far_r": -0.05,
            })
    # Two SL-hit: at bar 1, MAE goes to -2.0 (well below any tested threshold)
    for tid in ("T3_hit", "T4_hit"):
        for b in range(4):
            rows.append({
                "trade_id": tid,
                "bar_offset": b,
                "close_r": -0.5 if b >= 1 else 0.1,
                "mfe_so_far_r": 0.1,
                "mae_so_far_r": -2.0 if b >= 1 else 0.0,
            })
    cluster_paths = pd.DataFrame(rows)
    # SL = 1.5 ATR ⇒ threshold -0.75; mae_so_far_r=-2.0 trips, -0.05 does not.
    res = evaluate_sl_for_cluster(
        cluster_paths=cluster_paths,
        sl_atr=1.5,
        atr_units_per_R_at_simulation=2.0,
        pool_size=4,
        floors=FLOORS,
        bimodal_cfg=BIMODAL_CFG,
        local_peaks_relaxed_max=35.0,
    )
    assert res.cap_binding_rate == pytest.approx(0.5)
    assert res.n_trades == 4
