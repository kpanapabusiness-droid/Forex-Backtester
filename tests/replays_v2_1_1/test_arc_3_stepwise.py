"""Unit tests for Open-18 Replay #1 Arc 3 Stepwise climber v2.1.1 Step 3 logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.replays_v2_1_1.arc_3_stepwise.shape_classifier import classify_shape_tag  # noqa: E402
from scripts.replays_v2_1_1.arc_3_stepwise.step3 import (  # noqa: E402
    bimodal_separated_test,
    capturability_composite,
    compute_trade_path_result,
    truncate_path_at_sl,
)


def _path(close_r, mfe, mae):
    """Make synthetic trade path arrays with matched bar_offsets starting at 0."""
    bar_offsets = np.arange(len(close_r), dtype=int)
    return bar_offsets, np.array(close_r, float), np.array(mfe, float), np.array(mae, float)


def test_sl_truncation_logic():
    # SL=1.5 ATR at 2 ATR/R simulation -> threshold = -0.75.
    # Path mae: [0.0, -0.2, -0.5, -0.8, -0.4]. First bar where mae <= -0.75 is index 3.
    close_r = [0.0, 0.1, 0.05, -0.3, 0.2]
    mfe = [0.0, 0.1, 0.1, 0.1, 0.2]
    mae = [0.0, -0.2, -0.5, -0.8, -0.4]
    bo, cr, mf, ma = _path(close_r, mfe, mae)
    idx = truncate_path_at_sl(bo, cr, mf, ma, sl_threshold=-0.75)
    assert idx == 3, f"expected truncate at bar 3, got {idx}"

    # No SL hit case: all mae > threshold -> last index.
    mae_no_hit = [0.0, -0.1, -0.2, -0.3, -0.4]
    bo, cr, mf, ma = _path(close_r, mfe, mae_no_hit)
    idx = truncate_path_at_sl(bo, cr, mf, ma, sl_threshold=-0.75)
    assert idx == 4, f"expected truncate at last bar (4) on no-hit, got {idx}"


def test_peak_mfe_bar_identification():
    # Peak should be at bar 3 (max mfe = 1.2). SL=2.0 ATR -> threshold=-1.0; mae never reaches it.
    close_r = [0.0, 0.3, 0.8, 1.2, 1.0, 0.6]
    mfe = [0.0, 0.3, 0.8, 1.2, 1.2, 1.2]
    mae = [0.0, -0.1, -0.05, -0.1, -0.3, -0.5]
    bo, cr, mf, ma = _path(close_r, mfe, mae)
    tr = compute_trade_path_result(
        trade_id=1,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=2.0,
        atr_per_R_simulation=2.0,
    )
    assert tr.sl_hit is False, "should not hit SL"
    assert tr.peak_mfe_bar == 3, f"expected peak at bar 3, got {tr.peak_mfe_bar}"
    # peak_mfe_r_candidate at SL=2.0 ATR -> redenom factor = 2/2 = 1.0; peak stays 1.2
    assert abs(tr.peak_mfe_r_candidate - 1.2) < 1e-9


def test_pre_peak_monotonicity_calculation():
    # Pre-peak in-profit bars: [0.3, 0.8, 1.2]. All three at-or-above prior; mono = 3/3 = 1.0.
    close_r = [0.0, 0.3, 0.8, 1.2, 0.5, -0.2]
    mfe = [0.0, 0.3, 0.8, 1.2, 1.2, 1.2]
    mae = [0.0, -0.05, -0.05, -0.05, -0.2, -0.5]
    bo, cr, mf, ma = _path(close_r, mfe, mae)
    tr = compute_trade_path_result(
        trade_id=1,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=2.0,
        atr_per_R_simulation=2.0,
    )
    assert tr.peak_mfe_bar == 3
    # in-profit pre-peak vals: [0.3, 0.8, 1.2]; all >= prior. Mono = 3/3 = 1.0.
    assert abs(tr.pre_peak_monotonicity - 1.0) < 1e-9

    # Now with a regression: in-profit pre-peak [0.5, 0.3, 0.8]. 2 of 3 monotone (first counted + 0.8 >= 0.3).
    close_r2 = [0.0, 0.5, 0.3, 0.8, 0.2]
    mfe2 = [0.0, 0.5, 0.5, 0.8, 0.8]
    mae2 = [0.0, -0.05, -0.05, -0.05, -0.3]
    bo, cr, mf, ma = _path(close_r2, mfe2, mae2)
    tr2 = compute_trade_path_result(
        trade_id=2,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=2.0,
        atr_per_R_simulation=2.0,
    )
    assert tr2.peak_mfe_bar == 3
    # in-profit pre-peak vals: [0.5, 0.3, 0.8]
    # mono per spec: among bars where close_r > 0, fraction where close_r >= prev in-profit bar (first counts).
    # [0.5 (first, +1), 0.3 (0.3<0.5, no), 0.8 (0.8>=0.3, +1)] -> 2/3
    assert abs(tr2.pre_peak_monotonicity - (2.0 / 3.0)) < 1e-9


def test_frac_wrong_way_pre_peak():
    """Def C: wrong_way fires iff MAE <= threshold occurs on or before peak_mfe_bar.

    Two trades: A has MAE-before-peak (Def C should fire). B has MAE-after-peak
    (Def C should NOT fire — post-peak adverse moves are excluded by v2.1.1).
    Both at X=2.0 ATR / threshold=-1.0; neither hits SL within the path.
    """
    # Trade A: peak_mfe at bar 5 (mfe=1.0). MAE crosses -1.0 at bar 2 (BEFORE peak). Def C fires.
    close_r_A = [0.0, -0.4, -0.8, 0.2, 0.6, 1.0, 0.5]
    mfe_A = [0.0, 0.0, 0.0, 0.2, 0.6, 1.0, 1.0]
    mae_A = [0.0, -0.4, -1.05, -1.05, -1.05, -1.05, -1.05]
    bo, cr, mf, ma = _path(close_r_A, mfe_A, mae_A)
    tr_A = compute_trade_path_result(
        trade_id=1,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=2.0,
        atr_per_R_simulation=2.0,  # threshold -1.0
    )
    assert tr_A.sl_hit is True  # mae <= -1.0 at bar 2
    assert tr_A.truncate_bar == 2
    # After truncation: peak_mfe_bar within bars 0..2 — max mfe = 0.0, argmax at bar 0.
    # Pre-peak window = {bar 0}. mae[0]=0 > -1.0. wrong_way=False. This is *correct* Def C —
    # a trade with peak at entry bar can't have pre-peak adverse moves (no bars before bar 0).
    assert tr_A.peak_mfe_bar == 0
    assert tr_A.pre_peak_wrong_way is False

    # Trade A': peak_mfe at bar 5 NEVER stops out — wider SL so it survives, MAE before peak fires.
    # threshold -2.0 at SL=4 ATR — mae -1.05 > -2.0, no SL hit. Pre-peak 0..5 includes bar 2.
    # But we need a Def-C wrong_way at *that* threshold. Restructure: at SL=4 ATR threshold -2.0,
    # we need mae <= -2.0 somewhere in 0..5. Make mae deeper.
    mae_A2 = [0.0, -0.4, -2.05, -2.05, -2.05, -2.05, -2.05]
    bo, cr, mf, ma = _path(close_r_A, mfe_A, mae_A2)
    tr_A2 = compute_trade_path_result(
        trade_id=2,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=4.0,
        atr_per_R_simulation=2.0,  # threshold -2.0
    )
    assert tr_A2.sl_hit is True
    assert tr_A2.truncate_bar == 2
    assert tr_A2.peak_mfe_bar == 0
    assert tr_A2.pre_peak_wrong_way is False  # peak at bar 0, no pre-peak bars before it

    # Trade A'': peak at bar 5 BEFORE the SL hits — SL never hit within path; MAE crosses before peak.
    mae_A3 = [0.0, -0.4, -0.9, -0.9, -0.9, -0.9, -0.4]  # crosses -0.75 at bar 2, peak at 5
    bo, cr, mf, ma = _path(close_r_A, mfe_A, mae_A3)
    # mae[2]=-0.9 <= -0.75 -> sl_hit True; but peak is at bar 5 which is past SL...
    # Truncation at bar 2 -> peak in 0..2 = bar 0. pre-peak False. To test "MAE before peak in
    # a surviving trade", we need NO SL hit at the sweep level. Use threshold deeper than mae.
    tr_A3b = compute_trade_path_result(
        trade_id=4,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=2.0,
        atr_per_R_simulation=2.0,  # threshold -1.0; mae -0.9 > -1.0, no hit
    )
    assert tr_A3b.sl_hit is False
    # Now peak_mfe_bar=5 (mfe=1.0 at bar 5). Pre-peak 0..5 has bars where mae=-0.9... but is -0.9 <= -1.0? No.
    # Need an actually crossing threshold. Use threshold -0.5 -> SL=1.0 ATR.
    tr_A3c = compute_trade_path_result(
        trade_id=5,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=1.0,
        atr_per_R_simulation=2.0,  # threshold -0.5; mae -0.9 hits at bar 2
    )
    assert tr_A3c.sl_hit is True
    assert tr_A3c.truncate_bar == 2
    assert tr_A3c.peak_mfe_bar == 0  # in truncated 0..2, max mfe is 0 at bar 0

    # The cleanest Def-C "MAE before peak" demonstration: trade survives in truncated path,
    # peak is mid-path, and an adverse MAE crosses threshold before the peak bar.
    close_r_C = [0.0, 0.3, 0.5, 0.8, 1.0, 0.4]
    mfe_C = [0.0, 0.3, 0.5, 0.8, 1.0, 1.0]
    mae_C = [0.0, -0.05, -0.6, -0.6, -0.6, -0.6]  # crosses -0.5 at bar 2, peak at bar 4
    bo, cr, mf, ma = _path(close_r_C, mfe_C, mae_C)
    tr_C = compute_trade_path_result(
        trade_id=6,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=1.0,
        atr_per_R_simulation=2.0,  # threshold -0.5
    )
    # MAE -0.6 <= -0.5 at bar 2. SL hit -> truncate at bar 2.
    # Within 0..2 max mfe = 0.5 at bar 2. Peak=2. Pre-peak = bars 0..2. mae[2]=-0.6 <= -0.5 -> wrong=True.
    assert tr_C.sl_hit is True
    assert tr_C.truncate_bar == 2
    assert tr_C.peak_mfe_bar == 2
    assert tr_C.pre_peak_wrong_way is True, "Def C: MAE at peak bar counts as pre-peak"

    # Trade B: MAE strictly AFTER peak. peak_mfe at bar 2, MAE crosses at bar 4.
    close_r_B = [0.0, 0.4, 1.0, 0.6, -0.4, -0.6]
    mfe_B = [0.0, 0.4, 1.0, 1.0, 1.0, 1.0]
    mae_B = [0.0, -0.05, -0.05, -0.3, -0.6, -0.8]
    bo, cr, mf, ma = _path(close_r_B, mfe_B, mae_B)
    tr_B = compute_trade_path_result(
        trade_id=7,
        bar_offsets=bo,
        close_r=cr,
        mfe_so_far_r=mf,
        mae_so_far_r=ma,
        sl_atr=1.0,
        atr_per_R_simulation=2.0,  # threshold -0.5
    )
    # mae crosses -0.5 at bar 4 -> SL hit, truncate at 4.
    assert tr_B.sl_hit is True
    assert tr_B.truncate_bar == 4
    # Within 0..4 max mfe = 1.0 at bar 2 (first occurrence). Peak=2. Pre-peak 0..2.
    # mae[0..2] = [0, -0.05, -0.05] -> none <= -0.5. wrong=False.
    assert tr_B.peak_mfe_bar == 2
    assert tr_B.pre_peak_wrong_way is False, "Def C: post-peak MAE does NOT count"


def test_hartigan_dip_wrapper_bimodal_and_unimodal():
    rng = np.random.default_rng(42)
    # Clearly bimodal: two gaussians at 0 and 3, sigma 0.3, n=200.
    bm = np.concatenate(
        [
            rng.normal(0.0, 0.3, 100),
            rng.normal(3.0, 0.3, 100),
        ]
    )
    out = bimodal_separated_test(
        bm, hartigan_dip_p_max=0.05, min_mode_mass=0.20, min_mode_separation_r=1.0
    )
    assert out["pass_dip"] is True, f"expected dip p<0.05 for clear bimodal, got p={out['dip_p']}"
    assert out["pass_separation"] is True, "modes ~3R apart"
    assert out["pass_mode_mass"] is True, "both modes have ~50% mass"
    assert out["bimodal_separated"] is True

    # Clearly unimodal: gaussian sigma=1.0, n=300.
    um = rng.normal(2.0, 1.0, 300)
    out_um = bimodal_separated_test(
        um, hartigan_dip_p_max=0.05, min_mode_mass=0.20, min_mode_separation_r=1.0
    )
    assert out_um["pass_dip"] is False, (
        f"expected dip p>=0.05 for unimodal, got p={out_um['dip_p']}"
    )
    assert out_um["bimodal_separated"] is False


def test_capturability_composite():
    # composite = (mono - 0.55) + (reach_1R - 0.70) + (0.30 - wrong_way)
    m = {
        "monotonicity_pre_peak_centroid": 0.65,  # +0.10
        "frac_reach_1R": 0.80,  # +0.10
        "frac_wrong_way_pre_peak": 0.15,  # +0.15
    }
    assert abs(capturability_composite(m) - 0.35) < 1e-9


def test_shape_tag_classifier_reuse_pin():
    """Pin the v2.0 classifier on a fixture matching Arc 3's Stepwise aggregate v2.0 output (bimodal).

    Arc 3 closure ARC_3_RESULT.md row reports Stepwise climber aggregate at simulation SL=2 ATR
    had shape_tag = bimodal. Fixture: two-mode distribution with valley.
    """
    rng = np.random.default_rng(7)
    # Two distinct modes with a valley between — matches Arc 3 Stepwise aggregate final_r profile.
    sample = np.concatenate(
        [
            rng.normal(-1.0, 0.05, 250),  # tight stop-out cluster
            rng.normal(2.0, 0.6, 250),  # survivor cluster
        ]
    )
    tag, log = classify_shape_tag(sample)
    assert tag == "bimodal", (
        f"expected v2.0 classifier to tag this as bimodal, got {tag}; log={log}"
    )

    # Pin: a clearly tight_unimodal distribution.
    tight = rng.normal(1.0, 0.3, 200)
    tag2, _ = classify_shape_tag(tight)
    assert tag2 == "tight_unimodal", f"expected tight_unimodal, got {tag2}"
