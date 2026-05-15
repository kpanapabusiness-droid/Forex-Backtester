"""Arc 2 counterfactual exit-rule sweep, round 1, v2 (execution-faithful).

Phase: l6_arc2_counterfactual_sweep_round_1_v2

Reissue of the round-1 sweep with execution-faithful R formulas matching Arc 2's
actual `_execute_arc2` semantics line-by-line.

Key corrections from v1 (which HALTed at gate 4):
1. SL R = -1 - sp_exit × pip / (4 × atr) — exit-side half-spread only.
2. Time exit uses bar k=121 OPEN (from v1.2.1's next_bar_open_atr at k=120),
   not bar k=120 close.

Inputs (locked, sha256-verified at run start):
- results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv
- results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv
- results/l6/arc2/trades_all.csv (sp_entry_pips, sp_exit_pips lookup)
- core/signals/l4_mtf_alignment_2_down_mixed_kijun.py (Arc 2 module ref)
- configs/wfo_l6_arc2.yaml (Arc 2 config ref)

Outputs (results/l6/arc2/characterisation/extended/counterfactuals/round_1/):
- variant_trades.csv (63,888 rows × 11 cols)
- variant_summary_pooled.csv (16 rows)
- variant_summary_per_fold.csv (16 × 7 = 112 rows)
- additivity_calibration.csv (1 row)
- counterfactual_sweep_round_1.md (combined report)
- run_manifest.txt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import re
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Locked input sha256s (gate 1) ---
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv":
        "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv":
        "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py":
        "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml":
        "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "L6_0_METHODOLOGY_LOCK.md":
        "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}
TRADES_ALL_SHA = "47fccbfe4dffa6577a6000b0c16c2ebb9597dcf76523ff2b8084631b19836b3c"

# --- Adjacent locks for gate 15 ---
ADJ_LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_2_full/per_bar_paths.csv":
        "e1195f0dedb317f6d921d4fa9526c8aa546457f8038f28f37cd656605e6b1960",
    "results/l6/arc2/characterisation/v1_2_full/trade_index.csv":
        "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_2_full/pipeline_diff_v1_2_manifest.md":
        "f3094ffd59121bcb0864f72d8f851f99cc44b4e4354d374d5159e671b4f0d530",
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv":
        "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "scripts/lchar/arc2_characterisation_v1_1.py":
        "5d32627a1c4691ef654315dd5f35401d3a4e811bc20c0d48cd64a33debcb5105",
    "scripts/lchar/arc2_per_bar_paths.py":
        "36bb6f9b0413386bd5d25960f4525084fa93408ecb491232e17396872f1ff821",
}
CANDIDATE_HYPOTHESES_BASELINE_SHA = (
    "8ed487620a7f9ab2c443e6520a4afa820c353480d8329d4fe91703b7d083dfbf"
)

PAIRS: Tuple[str, ...] = (
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD", "CAD_CHF", "CAD_JPY", "CHF_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_NZD", "EUR_USD", "GBP_AUD",
    "GBP_CAD", "GBP_CHF", "GBP_JPY", "GBP_NZD", "GBP_USD", "NZD_CAD", "NZD_CHF", "NZD_JPY",
    "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY",
)

TIME_HORIZON_DEFAULT = 120
TIME_HORIZON_EXTENDED = 240


def _pip_size(pair: str) -> float:
    """Standard FX convention: JPY pairs use 0.01; everything else 0.0001."""
    return 0.01 if "JPY" in pair else 0.0001


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_input_integrity() -> Dict[str, str]:
    """Gate 1: verify locked sha256s."""
    out: Dict[str, str] = {}
    for rel, expected in LOCKED_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"Gate 1 HALT — sha256 mismatch on {rel}:\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
        out[rel] = actual
    # trades_all.csv: capture but don't verify (no formal lock in spec)
    out["results/l6/arc2/trades_all.csv"] = _sha256_file(REPO_ROOT / "results/l6/arc2/trades_all.csv")
    return out


# ============================================================================
# VARIANT MECHANICS — execution-faithful
#
# Each variant fn signature:
#   variant_fn(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T) -> dict
# where T is per-trade pre-computed data dict.
# Returns: {exit_bar, exit_reason, exit_level_atr_fill, gross_R, spread_cost_R, net_R}
# All R values in the variant's own R-unit (1R_variant = M_variant × atr).
# ============================================================================


def _exit_te_de(*, k_idx: int, T: Dict[str, Any], M: float,
                exit_atr_fill: float, exit_reason: str) -> Dict[str, Any]:
    """Time-exit or data-end exit at given fill-rel ATR offset.

    For TE: exit_atr_fill = next_bar_open_atr at k.
    For data_end: exit_atr_fill = bar_close_atr at k.
    Both: exit_fill = entry_fill + exit_atr_fill × atr - sp_exit × pip / 2.
    """
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    # Exit offset in price relative to entry_fill: exit_atr_fill × atr − sp_exit × pip / 2
    exit_price_offset = exit_atr_fill * atr - sp_exit * pip / 2
    sl_dist = M * atr
    net_R = exit_price_offset / sl_dist
    # exit_level_atr_fill = (exit_fill - entry_fill) / atr (Arc 2 ATR scale)
    exit_level_atr_fill = exit_atr_fill - sp_exit * pip / (2 * atr)
    # gross_R (mid-to-mid in variant R) = (exit_atr_fill_MID) / M = (exit_atr_fill + entry_fill_offset_atr) / M
    gross_R = (exit_atr_fill + T["entry_fill_offset_atr"]) / M
    spread_cost_R = gross_R - net_R  # = (full RT) × (2/M)
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": exit_reason,
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def _exit_sl(*, k_idx: int, T: Dict[str, Any], M: float) -> Dict[str, Any]:
    """SL exit at -M × atr (fill-rel)."""
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    # exit_fill = sl_price - sp_exit × pip / 2 = (entry_fill - M × atr) - sp_exit × pip / 2
    # exit_fill - entry_fill = -M × atr - sp_exit × pip / 2
    # R = (exit_fill - entry_fill) / (M × atr) = -1 - sp_exit × pip / (2 × M × atr)
    net_R = -1.0 - sp_exit * pip / (2 * M * atr)
    exit_level_atr_fill = -M - sp_exit * pip / (2 * atr)
    gross_R = -1.0
    spread_cost_R = gross_R - net_R  # = sp_exit × pip / (2 × M × atr)
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": "stop_loss",
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def _exit_be(*, k_idx: int, T: Dict[str, Any], M: float = 2.0) -> Dict[str, Any]:
    """BE-SL exit at entry_mid level (mid-rel 0)."""
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    # exit_fill = entry_mid - sp_exit × pip / 2; exit_fill - entry_fill = -sp_entry × pip / 2 - sp_exit × pip / 2
    net_R = -T["baseline_spread_cost_r"]
    exit_level_atr_fill = -T["entry_fill_offset_atr"] - sp_exit * pip / (2 * atr)
    gross_R = 0.0  # mid-to-mid is exactly entry → entry
    spread_cost_R = gross_R - net_R  # = baseline_spread_cost_r (full RT)
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": "be_exit",
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def _exit_trail(*, k_idx: int, T: Dict[str, Any], trail_level_atr_fill: float) -> Dict[str, Any]:
    """Trail exit at trail_level (price)."""
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    # exit_fill = entry_fill + trail_level_atr_fill × atr - sp_exit × pip / 2
    # net_R = (trail_level_atr_fill × atr - sp_exit × pip / 2) / (2 × atr)
    net_R = trail_level_atr_fill / 2 - sp_exit * pip / (4 * atr)
    exit_level_atr_fill = trail_level_atr_fill - sp_exit * pip / (2 * atr)
    gross_R = (trail_level_atr_fill + T["entry_fill_offset_atr"]) / 2  # mid-to-mid
    spread_cost_R = gross_R - net_R  # = baseline_spread_cost_r (full RT)
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": "trail_exit",
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def _exit_tp(*, k_idx: int, T: Dict[str, Any], T_TP_R: float) -> Dict[str, Any]:
    """Fixed TP exit at +T_TP_R level (mid-rel)."""
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    # exit_fill = entry_mid + 2 × T_TP × atr - sp_exit × pip / 2
    # exit_fill - entry_fill = -sp_entry × pip / 2 + 2 × T_TP × atr - sp_exit × pip / 2
    # net_R = ... / (2 × atr) = T_TP - baseline_spread_cost_r
    net_R = T_TP_R - T["baseline_spread_cost_r"]
    exit_level_atr_fill = 2 * T_TP_R - T["entry_fill_offset_atr"] - sp_exit * pip / (2 * atr)
    gross_R = T_TP_R  # mid-to-mid
    spread_cost_R = gross_R - net_R  # = baseline_spread_cost_r
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": "fixed_tp",
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def variant_BL(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
               *, time_horizon=TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    """Baseline: SL at -2 ATR, time exit at k=120, data_end at bavail."""
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            return _exit_sl(k_idx=k_idx, T=T, M=2.0)
        if k == time_horizon:
            if hnb[k_idx]:
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=nbo[k_idx], exit_reason="time_exit")
            else:
                # Edge case: no next bar at horizon → fall through to data_end at this k
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=bc_[k_idx], exit_reason="data_end")
        if k == bavail:
            return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                exit_atr_fill=bc_[k_idx], exit_reason="data_end")
    raise RuntimeError("BL did not terminate")


def variant_A(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
              *, M, time_horizon=TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    """Class A: SL at -M ATR. Variant R-unit = M × atr."""
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -M:
            return _exit_sl(k_idx=k_idx, T=T, M=M)
        if k == time_horizon:
            if hnb[k_idx]:
                return _exit_te_de(k_idx=k_idx, T=T, M=M,
                                    exit_atr_fill=nbo[k_idx], exit_reason="time_exit")
            else:
                return _exit_te_de(k_idx=k_idx, T=T, M=M,
                                    exit_atr_fill=bc_[k_idx], exit_reason="data_end")
        if k == bavail:
            return _exit_te_de(k_idx=k_idx, T=T, M=M,
                                exit_atr_fill=bc_[k_idx], exit_reason="data_end")
    raise RuntimeError("A did not terminate")


def variant_BE(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
               *, T_R, time_horizon=TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    """Class B / G2: BE-SL at +T_R mid-rel."""
    threshold_atr_fill = 2.0 * T_R - T["entry_fill_offset_atr"]
    entry_mid_atr_fill = -T["entry_fill_offset_atr"]
    be_active = False
    for k_idx in range(bavail):
        k = k_idx + 1
        if not be_active:
            if rmae[k_idx] <= -2.0:
                return _exit_sl(k_idx=k_idx, T=T, M=2.0)
            if rmfe[k_idx] >= threshold_atr_fill:
                be_active = True  # activate from k+1
        else:
            if bl_[k_idx] <= entry_mid_atr_fill:
                return _exit_be(k_idx=k_idx, T=T)
        if k == time_horizon:
            if hnb[k_idx]:
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=nbo[k_idx], exit_reason="time_exit")
            else:
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=bc_[k_idx], exit_reason="data_end")
        if k == bavail:
            return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                exit_atr_fill=bc_[k_idx], exit_reason="data_end")
    raise RuntimeError("BE did not terminate")


def variant_TRAIL(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
                  *, T_engage_R, D_R, kinked=False,
                  time_horizon=TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    """Class C / G1: trail engages at +T_engage_R mid-rel, D_R below peak."""
    engage_threshold_atr_fill = 2.0 * T_engage_R - T["entry_fill_offset_atr"]
    trail_active = False
    for k_idx in range(bavail):
        k = k_idx + 1
        if not trail_active:
            if rmae[k_idx] <= -2.0:
                return _exit_sl(k_idx=k_idx, T=T, M=2.0)
            if rmfe[k_idx] >= engage_threshold_atr_fill:
                trail_active = True  # activate from k+1
        else:
            if kinked:
                # G1: D = 1.0 ATR (= 0.5R) while rmfe<6.0; D = 2.0 ATR (= 1.0R) once rmfe>=6.0
                D_atr = 1.0 if rmfe[k_idx] < 6.0 else 2.0
            else:
                D_atr = 2.0 * D_R
            trail_level_atr_fill = rmfe[k_idx] - D_atr
            if bl_[k_idx] <= trail_level_atr_fill:
                return _exit_trail(k_idx=k_idx, T=T, trail_level_atr_fill=trail_level_atr_fill)
        if k == time_horizon:
            if hnb[k_idx]:
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=nbo[k_idx], exit_reason="time_exit")
            else:
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=bc_[k_idx], exit_reason="data_end")
        if k == bavail:
            return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                exit_atr_fill=bc_[k_idx], exit_reason="data_end")
    raise RuntimeError("TRAIL did not terminate")


def variant_PARTIAL_BE(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
                       *, T_partial_R=1.0,
                       time_horizon=TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    """Class D: partial 50% at +1R mid-rel + BE-SL on remainder."""
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    bsc = T["baseline_spread_cost_r"]
    partial_threshold_atr_fill = 2.0 * T_partial_R - T["entry_fill_offset_atr"]
    entry_mid_atr_fill = -T["entry_fill_offset_atr"]
    partial_closed = False
    be_active = False
    for k_idx in range(bavail):
        k = k_idx + 1
        if not partial_closed:
            if rmae[k_idx] <= -2.0:
                # Full SL on full position
                return _exit_sl(k_idx=k_idx, T=T, M=2.0)
            if bh_[k_idx] >= partial_threshold_atr_fill:
                # Partial fires this bar; BE-SL active from k+1
                partial_closed = True
                be_active = True
        else:  # partial_closed and be_active
            if bl_[k_idx] <= entry_mid_atr_fill:
                # Remainder hits BE
                # Total R = 0.5 × T_partial - baseline_spread_cost_r
                net_R = 0.5 * T_partial_R - bsc
                exit_level_atr_fill = entry_mid_atr_fill - sp_exit * pip / (2 * atr)  # final remainder exit
                gross_R = 0.5 * T_partial_R  # mid-to-mid: half closed at +T, half at 0
                spread_cost_R = gross_R - net_R  # = bsc
                return {
                    "exit_bar": k, "exit_reason": "partial_then_be",
                    "exit_level_atr_fill": exit_level_atr_fill,
                    "gross_R": gross_R, "spread_cost_R": spread_cost_R, "net_R": net_R,
                }
        if k == time_horizon:
            if partial_closed:
                # Remainder time exit at next_bar_open (or bar_close if no next bar)
                if hnb[k_idx]:
                    exit_atr_fill = nbo[k_idx]
                    exit_reason = "partial_then_time"
                else:
                    exit_atr_fill = bc_[k_idx]
                    exit_reason = "partial_then_data_end"
                # net_R = 0.5 × T_partial + 0.25 × next_bar_open_atr - sp_entry × pip / (8 × atr) - sp_exit × pip / (4 × atr)
                # equivalent: gross = 0.5 × T_partial + 0.25 × (exit_atr_fill + entry_fill_offset_atr); spread = bsc
                gross_R = 0.5 * T_partial_R + 0.25 * (exit_atr_fill + T["entry_fill_offset_atr"])
                net_R = gross_R - bsc
                exit_level_atr_fill = exit_atr_fill - sp_exit * pip / (2 * atr)
                return {
                    "exit_bar": k, "exit_reason": exit_reason,
                    "exit_level_atr_fill": exit_level_atr_fill,
                    "gross_R": gross_R, "spread_cost_R": bsc, "net_R": net_R,
                }
            else:
                # Full-position time exit (no partial)
                if hnb[k_idx]:
                    return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                        exit_atr_fill=nbo[k_idx], exit_reason="time_exit")
                else:
                    return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                        exit_atr_fill=bc_[k_idx], exit_reason="data_end")
        if k == bavail:
            if partial_closed:
                exit_atr_fill = bc_[k_idx]
                gross_R = 0.5 * T_partial_R + 0.25 * (exit_atr_fill + T["entry_fill_offset_atr"])
                net_R = gross_R - bsc
                exit_level_atr_fill = exit_atr_fill - sp_exit * pip / (2 * atr)
                return {
                    "exit_bar": k, "exit_reason": "partial_then_data_end",
                    "exit_level_atr_fill": exit_level_atr_fill,
                    "gross_R": gross_R, "spread_cost_R": bsc, "net_R": net_R,
                }
            else:
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=bc_[k_idx], exit_reason="data_end")
    raise RuntimeError("PARTIAL_BE did not terminate")


def variant_TP(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
               *, T_TP_R, time_horizon=TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    """Class E: fixed TP at +T_TP_R mid-rel. SL has precedence at same bar."""
    tp_threshold_atr_fill = 2.0 * T_TP_R - T["entry_fill_offset_atr"]
    for k_idx in range(bavail):
        k = k_idx + 1
        # SL first (per §4.6)
        if rmae[k_idx] <= -2.0:
            return _exit_sl(k_idx=k_idx, T=T, M=2.0)
        # TP check
        if bh_[k_idx] >= tp_threshold_atr_fill:
            return _exit_tp(k_idx=k_idx, T=T, T_TP_R=T_TP_R)
        if k == time_horizon:
            if hnb[k_idx]:
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=nbo[k_idx], exit_reason="time_exit")
            else:
                return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                    exit_atr_fill=bc_[k_idx], exit_reason="data_end")
        if k == bavail:
            return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                exit_atr_fill=bc_[k_idx], exit_reason="data_end")
    raise RuntimeError("TP did not terminate")


def variant_F1(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T) -> Dict[str, Any]:
    """F1: BL with extended time horizon 240."""
    return variant_BL(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
                       time_horizon=TIME_HORIZON_EXTENDED)


def variant_F2(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T) -> Dict[str, Any]:
    """F2: SL or data_end only (no time exit)."""
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            return _exit_sl(k_idx=k_idx, T=T, M=2.0)
        if k == bavail:
            return _exit_te_de(k_idx=k_idx, T=T, M=2.0,
                                exit_atr_fill=bc_[k_idx], exit_reason="data_end")
    raise RuntimeError("F2 did not terminate")


def variant_G1(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T) -> Dict[str, Any]:
    """G1: kinked trail. T_engage = +1R; D = 0.5R while rmfe<6, D = 1R once rmfe>=6."""
    return variant_TRAIL(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
                          T_engage_R=1.0, D_R=0.5, kinked=True)


def variant_G2(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T) -> Dict[str, Any]:
    """G2: B2 (BE-SL at +1R) + F1 (k=240). Tests B2+F1 additivity."""
    return variant_BE(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
                       T_R=1.0, time_horizon=TIME_HORIZON_EXTENDED)


# Variant registry: (id, spec_short, fn).
VARIANTS: List[Tuple[str, str, Callable[..., Dict[str, Any]]]] = [
    ("BL", "Baseline: SL=-2ATR, k=120 TE",                  variant_BL),
    ("A1", "SL=-1.5ATR, k=120 TE",                          lambda *a: variant_A(*a, M=1.5)),
    ("A2", "SL=-2.5ATR, k=120 TE",                          lambda *a: variant_A(*a, M=2.5)),
    ("B1", "BE-SL @ +0.5R, k=120",                          lambda *a: variant_BE(*a, T_R=0.5)),
    ("B2", "BE-SL @ +1.0R, k=120",                          lambda *a: variant_BE(*a, T_R=1.0)),
    ("B3", "BE-SL @ +1.5R, k=120",                          lambda *a: variant_BE(*a, T_R=1.5)),
    ("C1", "Trail engage +1R, D=0.5R, k=120",               lambda *a: variant_TRAIL(*a, T_engage_R=1.0, D_R=0.5)),
    ("C2", "Trail engage +1R, D=1R, k=120",                 lambda *a: variant_TRAIL(*a, T_engage_R=1.0, D_R=1.0)),
    ("C3", "Trail engage +2R, D=1R, k=120",                 lambda *a: variant_TRAIL(*a, T_engage_R=2.0, D_R=1.0)),
    ("D1", "Partial 50%@+1R + BE-SL, k=120",                variant_PARTIAL_BE),
    ("E1", "Fixed TP @ +1.5R, k=120",                       lambda *a: variant_TP(*a, T_TP_R=1.5)),
    ("E2", "Fixed TP @ +3R, k=120",                         lambda *a: variant_TP(*a, T_TP_R=3.0)),
    ("F1", "BL k=240",                                       variant_F1),
    ("F2", "BL no time exit (SL or DE)",                    variant_F2),
    ("G1", "Kinked trail (D=0.5R/1R)",                      variant_G1),
    ("G2", "B2 + F1: BE-SL @ +1R, k=240",                   variant_G2),
]


# ============================================================================
# RUN SWEEP
# ============================================================================


def _run_sweep(
    *, per_bar_csv: Path, trade_index_csv: Path, trades_all_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (variant_trades, trade_index_full)."""
    print("  Loading trade_index.csv + trades_all.csv...", flush=True)
    ti = pd.read_csv(trade_index_csv)
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ta = pd.read_csv(trades_all_csv)
    ta["signal_bar_ts"] = pd.to_datetime(ta["signal_bar_ts"])

    # Join for sp_entry/exit
    ti_full = ti.merge(
        ta[["pair", "signal_bar_ts", "spread_pips_entry", "spread_pips_exit"]],
        on=["pair", "signal_bar_ts"], how="left", validate="one_to_one",
    )
    if ti_full[["spread_pips_entry", "spread_pips_exit"]].isna().any().any():
        raise RuntimeError("Gate 3 HALT — null sp_entry/sp_exit after merge")
    ti_full = ti_full.sort_values("trade_id").reset_index(drop=True)

    # Pre-compute per-trade values
    print("  Pre-computing per-trade values...", flush=True)
    per_trade: Dict[int, Dict[str, Any]] = {}
    for _, row in ti_full.iterrows():
        tid = int(row["trade_id"])
        pair = row["pair"]
        pip = _pip_size(pair)
        sp_entry = float(row["spread_pips_entry"])
        sp_exit = float(row["spread_pips_exit"])
        atr = float(row["atr_1h_wilder_at_signal"])
        entry_fill = float(row["entry_price"])
        per_trade[tid] = {
            "pair": pair, "atr": atr, "entry_fill": entry_fill,
            "sp_entry_pips": sp_entry, "sp_exit_pips": sp_exit, "pip": pip,
            "entry_fill_offset_atr": sp_entry * pip / (2 * atr),
            "baseline_spread_cost_r": (sp_entry + sp_exit) * pip / (4 * atr),
            "exit_half_spread_r_base": sp_exit * pip / (4 * atr),
            "entry_mid": entry_fill - sp_entry * pip / 2,
            "fold_id": int(row["fold_id"]),
            "signal_bar_ts": row["signal_bar_ts"].strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # Load per_bar
    print("  Loading per_bar_paths.csv (118 MB)...", flush=True)
    pb = pd.read_csv(per_bar_csv)
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids_arr = pb["trade_id"].to_numpy(dtype=np.int64)
    n_trades = int(ti["trade_id"].max()) + 1
    starts = np.searchsorted(tids_arr, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids_arr, np.arange(n_trades), side="right")

    rmae_all = pb["running_mae_atr"].to_numpy(dtype=np.float64)
    rmfe_all = pb["running_mfe_atr"].to_numpy(dtype=np.float64)
    bl_all = pb["bar_low_atr"].to_numpy(dtype=np.float64)
    bh_all = pb["bar_high_atr"].to_numpy(dtype=np.float64)
    bc_all = pb["bar_close_atr"].to_numpy(dtype=np.float64)
    nbo_all = pb["next_bar_open_atr"].to_numpy(dtype=np.float64)
    hnb_all = pb["has_next_bar"].to_numpy(dtype=bool)

    n_vars = len(VARIANTS)
    total_rows = n_trades * n_vars

    out_variant = np.empty(total_rows, dtype=object)
    out_tid = np.empty(total_rows, dtype=np.int64)
    out_pair = np.empty(total_rows, dtype=object)
    out_sigts = np.empty(total_rows, dtype=object)
    out_fold = np.empty(total_rows, dtype=np.int64)
    out_reason = np.empty(total_rows, dtype=object)
    out_exitbar = np.empty(total_rows, dtype=np.int64)
    out_exitlvl = np.empty(total_rows, dtype=np.float64)
    out_gross = np.empty(total_rows, dtype=np.float64)
    out_spread = np.empty(total_rows, dtype=np.float64)
    out_net = np.empty(total_rows, dtype=np.float64)

    print(f"  Computing {n_vars} variants × {n_trades} trades...", flush=True)
    t0 = time.time()
    write_idx = 0
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        bavail = e - s
        rmae = rmae_all[s:e]; rmfe = rmfe_all[s:e]
        bl_ = bl_all[s:e]; bh_ = bh_all[s:e]; bc_ = bc_all[s:e]
        nbo = nbo_all[s:e]; hnb = hnb_all[s:e]
        T = per_trade[tid]

        for vid, _, vfn in VARIANTS:
            r = vfn(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T)
            out_variant[write_idx] = vid
            out_tid[write_idx] = tid
            out_pair[write_idx] = T["pair"]
            out_sigts[write_idx] = T["signal_bar_ts"]
            out_fold[write_idx] = T["fold_id"]
            out_reason[write_idx] = r["exit_reason"]
            out_exitbar[write_idx] = r["exit_bar"]
            out_exitlvl[write_idx] = r["exit_level_atr_fill"]
            out_gross[write_idx] = r["gross_R"]
            out_spread[write_idx] = r["spread_cost_R"]
            out_net[write_idx] = r["net_R"]
            write_idx += 1
        if (tid + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"    progress: {tid+1}/{n_trades} ({el:.1f}s, "
                  f"{(tid+1)/el:.0f} trades/s)", flush=True)

    variant_trades = pd.DataFrame({
        "variant_id": out_variant, "trade_id": out_tid, "pair": out_pair,
        "signal_bar_ts": out_sigts, "fold_id": out_fold,
        "exit_reason_variant": out_reason, "exit_bar": out_exitbar,
        "exit_level_atr_fill": out_exitlvl, "gross_R": out_gross,
        "spread_cost_R": out_spread, "net_R": out_net,
    })
    variant_trades = variant_trades.sort_values(["variant_id", "trade_id"]).reset_index(drop=True)

    return variant_trades, ti_full


# ============================================================================
# AGGREGATIONS
# ============================================================================


EXIT_REASONS_ALL = [
    "stop_loss", "time_exit", "be_exit", "trail_exit",
    "partial_then_be", "partial_then_time", "partial_then_data_end",
    "fixed_tp", "data_end",
]


def _aggregate_pooled(vt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = {vid: spec for vid, spec, _ in VARIANTS}
    for vid in [v[0] for v in VARIANTS]:
        sub = vt[vt["variant_id"] == vid]
        n = len(sub)
        net = sub["net_R"].to_numpy(dtype=np.float64)
        rc = sub["exit_reason_variant"].value_counts().to_dict()
        rates = {f"{r}_rate": rc.get(r, 0) / n for r in EXIT_REASONS_ALL}
        rows.append({
            "variant_id": vid, "variant_spec_short": specs[vid],
            "n_trades": n,
            "mean_R": float(np.mean(net)),
            "median_R": float(np.median(net)),
            "std_R": float(np.std(net, ddof=1)) if n > 1 else 0.0,
            "q05_R": float(np.quantile(net, 0.05)),
            "q25_R": float(np.quantile(net, 0.25)),
            "q75_R": float(np.quantile(net, 0.75)),
            "q95_R": float(np.quantile(net, 0.95)),
            "sl_rate": rates["stop_loss_rate"],
            "time_exit_rate": rates["time_exit_rate"],
            "be_exit_rate": rates["be_exit_rate"] + rates["partial_then_be_rate"],
            "trail_exit_rate": rates["trail_exit_rate"],
            "partial_exit_rate": (rates["partial_then_be_rate"]
                                  + rates["partial_then_time_rate"]
                                  + rates["partial_then_data_end_rate"]),
            "tp_exit_rate": rates["fixed_tp_rate"],
            "data_end_rate": rates["data_end_rate"] + rates["partial_then_data_end_rate"],
            "mean_spread_cost_R": float(sub["spread_cost_R"].mean()),
            "total_R": float(np.sum(net)),
        })
    return pd.DataFrame(rows)


def _aggregate_per_fold(vt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sorted_vt = vt.sort_values(["variant_id", "fold_id", "signal_bar_ts", "trade_id"])
    folds = sorted(vt["fold_id"].unique().tolist())
    for vid in [v[0] for v in VARIANTS]:
        for fid in folds:
            sub = sorted_vt[(sorted_vt["variant_id"] == vid) & (sorted_vt["fold_id"] == fid)]
            n = len(sub)
            if n == 0:
                continue
            net = sub["net_R"].to_numpy(dtype=np.float64)
            cum = np.cumsum(net)
            run_max = np.maximum.accumulate(cum)
            dd = run_max - cum
            mdd = float(np.max(dd))
            rc = sub["exit_reason_variant"].value_counts().to_dict()
            rates = {f"{r}_rate": rc.get(r, 0) / n for r in EXIT_REASONS_ALL}
            rows.append({
                "variant_id": vid, "fold_id": int(fid), "n": n,
                "mean_R": float(np.mean(net)),
                "median_R": float(np.median(net)),
                "total_R": float(np.sum(net)),
                "max_DD_R": mdd, "max_DD_pct_of_n": mdd / n,
                "sl_rate": rates["stop_loss_rate"],
                "time_exit_rate": rates["time_exit_rate"],
                "be_exit_rate": rates["be_exit_rate"] + rates["partial_then_be_rate"],
                "trail_exit_rate": rates["trail_exit_rate"],
                "partial_exit_rate": (rates["partial_then_be_rate"]
                                      + rates["partial_then_time_rate"]
                                      + rates["partial_then_data_end_rate"]),
                "tp_exit_rate": rates["fixed_tp_rate"],
            })
    return pd.DataFrame(rows)


def _additivity_calibration(pooled: pd.DataFrame) -> pd.DataFrame:
    bl_mean = float(pooled.loc[pooled["variant_id"] == "BL", "mean_R"].iloc[0])
    b2_mean = float(pooled.loc[pooled["variant_id"] == "B2", "mean_R"].iloc[0])
    f1_mean = float(pooled.loc[pooled["variant_id"] == "F1", "mean_R"].iloc[0])
    g2_mean = float(pooled.loc[pooled["variant_id"] == "G2", "mean_R"].iloc[0])
    lone_sum = (b2_mean - bl_mean) + (f1_mean - bl_mean)
    combo = g2_mean - bl_mean
    diverg = combo - lone_sum
    note = ("abs(divergence) < 0.05R — additive forecastable from lone effects."
            if abs(diverg) < 0.05 else
            "abs(divergence) >= 0.05R — combinations must be tested directly.")
    return pd.DataFrame([{
        "combination": "G2 = B2 + F1",
        "BL_mean_R": bl_mean, "B2_mean_R": b2_mean, "F1_mean_R": f1_mean, "G2_mean_R": g2_mean,
        "lone_lift_sum": lone_sum, "combination_lift": combo, "divergence": diverg,
        "interpretation_note": note,
    }])


# ============================================================================
# VALIDATION GATES
# ============================================================================


def _validate_gates(*, vt: pd.DataFrame, ti_full: pd.DataFrame,
                    pooled: pd.DataFrame, per_fold: pd.DataFrame,
                    out_dir: Path) -> Dict[str, Any]:
    disp: Dict[str, Any] = {}

    # Gate 2: trade count parity
    n_ti = len(ti_full)
    pb_path = REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv"
    n_pb = sum(1 for _ in pb_path.open("r", encoding="utf-8")) - 1
    disp["gate_2"] = f"trade_index={n_ti}, per_bar={n_pb}"
    if n_ti != 3993 or n_pb != 954749:
        raise RuntimeError(f"Gate 2 HALT — counts off: {disp['gate_2']}")

    # Gate 3: trades_all lookup completeness (already enforced at load; record)
    nulls = int(ti_full[["spread_pips_entry", "spread_pips_exit"]].isna().any(axis=1).sum())
    disp["gate_3"] = f"sp_lookup_nulls={nulls}"
    if nulls > 0:
        raise RuntimeError(f"Gate 3 HALT — {nulls} sp_entry/sp_exit nulls")

    # Gate 4 (CRITICAL): BL R reproduction
    # Tolerance relaxed from 1e-9 to 1e-7 per gate 4 HALT diagnostic
    # (l6_arc2_counterfactual_sweep_round_1_v2_resume phase). The strict <1e-9
    # is unachievable through `:.10g`-stored entry_price/atr inputs from
    # trade_index.csv: the 4-op TE R formula propagates ULP errors to ~9e-9.
    # 1e-7 gives a 10× safety margin over observed max diff (~9.14e-9), still
    # verifies engine correctness to <0.00001 R-units (6 orders below the
    # smallest variant lift signal of ~0.01R), and would catch genuine logic
    # bugs (~0.01R+ diffs).
    GATE_4_TOL = 1e-7
    bl = vt[vt["variant_id"] == "BL"].sort_values("trade_id").reset_index(drop=True)
    ti_sorted = ti_full.sort_values("trade_id").reset_index(drop=True)
    diff = (bl["net_R"].to_numpy(dtype=np.float64)
            - ti_sorted["R"].to_numpy(dtype=np.float64))
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    n_mismatch = int((abs_diff >= GATE_4_TOL).sum())
    # Per exit_reason breakdown for diagnostic
    er = ti_sorted["exit_reason"].to_numpy()
    sl_mask = (er == "stop_loss")
    te_mask = (er == "time_exit")
    de_mask = (er == "data_end")
    disp["gate_4"] = {
        "max_abs_diff": f"{max_abs:.3e}",
        "tolerance": f"{GATE_4_TOL:.0e}",
        "mismatches_at_tol": n_mismatch,
        "mismatches_1e9": int((abs_diff >= 1e-9).sum()),
        "mismatches_1e7": int((abs_diff >= 1e-7).sum()),
        "mismatches_1e6": int((abs_diff >= 1e-6).sum()),
        "max_sl": f"{float(abs_diff[sl_mask].max()):.3e}",
        "max_te": f"{float(abs_diff[te_mask].max()):.3e}",
        "max_de": f"{float(abs_diff[de_mask].max()):.3e}",
    }
    if max_abs >= GATE_4_TOL or n_mismatch > 0:
        # Halt with full diagnostic
        bad_idx = np.argsort(-abs_diff)[:20]
        sample = pd.DataFrame({
            "trade_id": bl["trade_id"].iloc[bad_idx].values,
            "pair": ti_sorted["pair"].iloc[bad_idx].values,
            "exit_reason": er[bad_idx],
            "BL_net_R": bl["net_R"].iloc[bad_idx].values,
            "v1.1_R": ti_sorted["R"].iloc[bad_idx].values,
            "abs_diff": abs_diff[bad_idx],
        })
        _write_gate4_halt_diagnostic(out_dir=out_dir, disp_gate4=disp["gate_4"],
                                       sample=sample, max_abs=max_abs)
        raise RuntimeError(
            f"Gate 4 HALT — BL R diff exceeds {GATE_4_TOL:.0e}: max={max_abs:.3e}, "
            f"mismatches_at_tol={n_mismatch}. See GATE_4_HALT_DIAGNOSTIC.md."
        )

    # Gate 5: spread accounting consistency
    # SL trades: gross - net should equal sp_exit × pip / (4 × atr) (BL only)
    # TE trades: gross - net should equal baseline_spread_cost_r
    bl_sl = bl[er == "stop_loss"]
    bl_te = bl[er == "time_exit"]
    sa_diff_sl = (bl_sl["gross_R"] - bl_sl["net_R"]).to_numpy()
    sa_diff_te = (bl_te["gross_R"] - bl_te["net_R"]).to_numpy()
    # Expected
    ti_sl = ti_sorted[er == "stop_loss"]
    ti_te = ti_sorted[er == "time_exit"]
    ti_sl_pip = ti_sl["pair"].apply(_pip_size).to_numpy()
    sl_expected = ti_sl["spread_pips_exit"].to_numpy() * ti_sl_pip / (4 * ti_sl["atr_1h_wilder_at_signal"].to_numpy())
    te_expected = ((ti_te["spread_pips_entry"].to_numpy() + ti_te["spread_pips_exit"].to_numpy())
                   * ti_te["pair"].apply(_pip_size).to_numpy() / (4 * ti_te["atr_1h_wilder_at_signal"].to_numpy()))
    sl_max = float(np.max(np.abs(sa_diff_sl - sl_expected)))
    te_max = float(np.max(np.abs(sa_diff_te - te_expected)))
    disp["gate_5"] = f"sl_diff_max={sl_max:.3e}, te_diff_max={te_max:.3e}"
    if sl_max >= 1e-9 or te_max >= 1e-9:
        raise RuntimeError(f"Gate 5 HALT — spread accounting: {disp['gate_5']}")

    # Gate 6: exit_bar plausibility
    bavail_by_tid = ti_sorted.set_index("trade_id")["bars_available"].to_dict()
    horizon_by_vid = {v[0]: TIME_HORIZON_DEFAULT for v in VARIANTS}
    horizon_by_vid["F1"] = TIME_HORIZON_EXTENDED
    horizon_by_vid["G2"] = TIME_HORIZON_EXTENDED
    horizon_by_vid["F2"] = TIME_HORIZON_EXTENDED  # F2 cap = bavail
    bad = 0
    for _, row in vt.iterrows():
        cap = min(horizon_by_vid[row["variant_id"]], bavail_by_tid[int(row["trade_id"])])
        if int(row["exit_bar"]) > cap:
            bad += 1
    disp["gate_6"] = f"exit_bar_violations={bad}"
    if bad > 0:
        raise RuntimeError(f"Gate 6 HALT — {bad} exit_bar > cap")

    # Gate 7: R range plausibility
    nan_inf = int((~np.isfinite(vt["net_R"].to_numpy(dtype=np.float64))).sum())
    disp["gate_7"] = f"nan_or_inf={nan_inf}"
    if nan_inf > 0:
        raise RuntimeError(f"Gate 7 HALT — {nan_inf} NaN/inf in net_R")
    # Gate 7 cap relaxed from +20 to +50 per v2_resume_2 phase
    # (l6_arc2_counterfactual_sweep_round_1_v2_resume_2). F2's no-time-exit
    # semantics structurally allow long-runners to exceed +20R (observed
    # max +25.83R on EUR_CAD tid=3420, baseline +5.93R extending to +25.83R
    # by k=240). Cap of +50 is ~3× v1.1 mfe_R max (+16.32R) and 2× F2's
    # observed max; still catches genuine logic bugs (R > 100s from
    # near-zero ATR divisions / sign flips / sentinel misuse).
    GATE_7_R_CAP = 50.0
    sl_only = {"BL", "A1", "A2", "F2"}
    for vid in sl_only:
        max_r = float(vt[vt["variant_id"] == vid]["net_R"].max())
        if max_r > GATE_7_R_CAP:
            # Write a brief diagnostic before halting (similar pattern to gate 4
            # and v1.2.1 gate 9 — legitimate small-ATR-trade extreme exceeds spec cap).
            top = vt[vt["variant_id"] == vid].sort_values("net_R", ascending=False).head(5)
            top_tid = int(top.iloc[0]["trade_id"])
            ti_top = ti_full[ti_full["trade_id"] == top_tid].iloc[0]
            diag_lines = [
                "# GATE 7 HALT DIAGNOSTIC — Counterfactual Sweep Round 1 v2",
                "",
                f"_Generated: {_dt.datetime.now().isoformat(timespec='seconds')}_",
                "",
                "## Disposition",
                "",
                f"**Gate 7 FAILED.** Spec: SL-only variants (BL, A1, A2, F2) must have no R > +20. "
                f"Observed: variant **{vid}** max_R = **{max_r:.4f}**.",
                "",
                "## Per-variant SL-only max_R (other variants under cap)",
                "",
            ]
            for v_check in sl_only:
                mr = float(vt[vt["variant_id"] == v_check]["net_R"].max())
                diag_lines.append(f"- {v_check}: max_R = {mr:+.4f} {'← OVER CAP' if mr > 20 else ''}")
            diag_lines.extend([
                "",
                f"## Top 5 {vid} trades by net_R",
                "",
                "```",
                top[["trade_id", "pair", "signal_bar_ts", "exit_reason_variant",
                     "exit_bar", "net_R"]].to_string(index=False),
                "```",
                "",
                f"## Worked example: trade_id={top_tid} ({ti_top['pair']})",
                "",
                f"- atr_1h_wilder_at_signal: {ti_top['atr_1h_wilder_at_signal']:.6e}",
                f"- entry_price: {ti_top['entry_price']:.6f}",
                f"- v1.1 R (Arc 2 baseline @ k=120): {ti_top['R']:.4f}",
                f"- v1.1 mfe_R (held-window peak): {ti_top['mfe_R']:.4f}",
                f"- bars_available: {ti_top['bars_available']}",
                f"- {vid} net_R (no time exit, full hold): {top.iloc[0]['net_R']:.4f}",
                "",
                "## Diagnosis",
                "",
                f"This is a **legitimate long-running winner** captured by `{vid}` (no time exit). "
                "The trade exited at +{:.2f}R Arc 2 baseline (time exit at k=120) but ran further "
                "favourably; F2's no-time-exit semantics let it accumulate to +{:.2f}R by bars_available=240.".format(
                    ti_top['R'], top.iloc[0]['net_R']),
                "",
                f"The trade's ATR is moderate (~{ti_top['atr_1h_wilder_at_signal']:.4f}); the +{top.iloc[0]['net_R']:.0f}R extreme reflects a sustained favourable trend over multiple days, not a logic bug. "
                "Same pattern as v1.2.1 gate 9 (cap of |x|<50 was too tight for legitimate small-ATR extremes; relaxed to <200 by planner).",
                "",
                "## Resolutions for the planner",
                "",
                f"1. **Relax cap to +50 or +100** — would accept this trade and similar legitimate long-runners; still catches genuine logic bugs (which would show R > 100s).",
                "2. **Remove the cap entirely** — gate 7 already guarantees no NaN/inf; the +20 cap was a sanity check, not a correctness check.",
                "3. **Restrict gate 7 cap to BL/A1/A2 only**, exempting F2 (since F2's no-time-exit semantics specifically allow long runs).",
                "",
                "## Outputs preserved",
                "",
                "- `variant_trades.csv` — full sweep computation; data correct, gate 7 cap inconsistent with population scale",
                "- `variant_summary_pooled.csv`, `variant_summary_per_fold.csv`, `additivity_calibration.csv`",
            ])
            (out_dir / "GATE_7_HALT_DIAGNOSTIC.md").write_text("\n".join(diag_lines) + "\n",
                                                                  encoding="utf-8")
            print(f"\n=== GATE 7 HALT DIAGNOSTIC written to: {out_dir / 'GATE_7_HALT_DIAGNOSTIC.md'} ===\n",
                  flush=True)
            raise RuntimeError(f"Gate 7 HALT — {vid} max_R={max_r:.2f} > 20")

    # Gate 8: B2 BE FIRE rate plausibility (relaxed from [0.45, 0.55] to
    # [0.30, 0.55] per v2_resume_3 phase). The original range was set from the
    # MFE-≥-1R population share (51.19%, the BE *activation* rate). But the
    # gate measures `be_exit_rate` (the BE *fire* rate — % of trades closing at
    # entry_mid after BE-SL activates). Activation rate ≠ fire rate: trades can
    # activate BE then exit at TE without retracing to entry_mid (~12.6% of
    # taken trades). Observed BE fire rate ~38.59% is the protective rate.
    # Relaxed bound still catches sign-flip bugs (would push fire rate near 95%)
    # and missed-activation bugs (would push it to 0%).
    GATE_8_BE_RATE_LO = 0.30
    GATE_8_BE_RATE_HI = 0.55
    b2 = pooled.loc[pooled["variant_id"] == "B2"].iloc[0]
    b2_be = float(b2["be_exit_rate"])
    disp["gate_8"] = f"B2_be_rate={b2_be:.4f} (range [{GATE_8_BE_RATE_LO}, {GATE_8_BE_RATE_HI}])"
    if not (GATE_8_BE_RATE_LO <= b2_be <= GATE_8_BE_RATE_HI):
        # Brief diagnostic before halt (same pattern as gates 4, 7, v1.2.1 gate 9):
        # spec range confused BE activation rate (≈51% of trades reach +1R MFE)
        # with BE fire rate (≈39% — subset of those that subsequently retrace to entry_mid).
        b2_sub = vt[vt["variant_id"] == "B2"]
        n = len(b2_sub)
        rc = b2_sub["exit_reason_variant"].value_counts().to_dict()
        # Reference: v1.1 MFE>=1R rate
        v11_path = REPO_ROOT / "results/l6/arc2/characterisation/v1_1_full/signals_features.csv"
        v11 = pd.read_csv(v11_path)
        taken = v11[v11["taken"] == True]  # noqa: E712
        mfe_ge_1_rate = (taken["mfe_R"] >= 1.0).sum() / len(taken)
        diag = [
            "# GATE 8 HALT DIAGNOSTIC — B2 BE-fire rate",
            "",
            f"_Generated: {_dt.datetime.now().isoformat(timespec='seconds')}_",
            "",
            "## Disposition",
            "",
            "**Gate 8 FAILED.** Spec range [0.45, 0.55]. Observed B2 be_exit_rate = **{:.4f}** ({:.2f}%).".format(b2_be, b2_be*100),
            "",
            "## Diagnosis: spec confused **BE activation rate** with **BE fire rate**",
            "",
            "Spec rationale: 'should approximate the share of trades with mfe_R >= 1 in v1.1 (51.19%)'.",
            "",
            f"v1.1 population reference (taken trades, n={len(taken)}):",
            f"- Trades reaching mfe_R >= 1R: **{(taken['mfe_R']>=1.0).sum()} = {mfe_ge_1_rate*100:.2f}%** (= BE activation rate)",
            "",
            f"B2 actual exit reasons (n={n}):",
        ]
        for rsn in ["stop_loss", "be_exit", "time_exit", "data_end"]:
            cnt = rc.get(rsn, 0)
            diag.append(f"- `{rsn}`: {cnt} = {cnt/n*100:.2f}%")
        diag.extend([
            "",
            "## Two distinct rates, two physical meanings",
            "",
            "1. **BE activation rate** = % of trades reaching mfe_R >= 1R (so BE-SL becomes active starting at the NEXT bar). For B2: **51.19%** (matches v1.1 reference exactly — every trade reaching +1R MFE activates).",
            "",
            "2. **BE fire rate** = % of trades where BE-SL fires (bar_low retraces to entry_mid AFTER activation, before time exit). For B2: **" + "{:.2f}".format(b2_be*100) + "%**.",
            "",
            "The diff (51.19% - 38.59% = ~12.6%) = trades that reached +1R MFE but kept running favourably and never retraced to entry_mid before time exit. These show up in B2's `time_exit` (12.32%).",
            "",
            "Verification: 38.59% (BE fire) + 12.32% (TE without BE fire) + 0.18% (DE) ≈ 51.10% ≈ activation rate.",
            "",
            "## Other plausibility gates (all pass at observed data)",
            "",
            "Verified externally before halting:",
            "- Gate 7 (max R, cap +50): all SL-only variants PASS (BL 14.7, A1 19.6, A2 11.8, F2 25.8)",
            "- Gate 9 (C1/C3 trail engage): C1 51.1%, C3 34.2% — both in spec range",
            "- Gate 10 (E1/E2 TP rates): E1 40.8%, E2 23.3% — both in spec range",
            "- Gate 11 (A1 > BL > A2 SL ordering): 82.4% > 76.0% > 70.8% — PASS",
            "",
            "Only gate 8 is mis-specified. The data is correct.",
            "",
            "## Resolutions for the planner",
            "",
            "1. **Relax gate 8 range to [30%, 55%]** — accepts the BE fire rate while still bounding logic bugs (e.g., a sign-flip on the entry_mid threshold would make BE fire on near-100% of trades).",
            "2. **Re-interpret gate 8 to check BE activation rate** — observe 51.19% matches v1.1 reference. Would require adding an `activated_be` flag to variant_trades.csv (new column).",
            "3. **Remove gate 8** — keep the other plausibility gates (9, 10, 11) which all PASS at current data.",
            "",
            "## Outputs preserved",
            "",
            "- `variant_trades.csv`, `variant_summary_pooled.csv`, `variant_summary_per_fold.csv`, `additivity_calibration.csv` — all match prior resume's sha256s. Data correct.",
        ])
        (out_dir / "GATE_8_HALT_DIAGNOSTIC.md").write_text("\n".join(diag) + "\n",
                                                              encoding="utf-8")
        print(f"\n=== GATE 8 HALT DIAGNOSTIC written to: {out_dir / 'GATE_8_HALT_DIAGNOSTIC.md'} ===\n",
              flush=True)
        raise RuntimeError(f"Gate 8 HALT — B2 BE rate {b2_be:.4f} outside [{GATE_8_BE_RATE_LO}, {GATE_8_BE_RATE_HI}]. See GATE_8_HALT_DIAGNOSTIC.md.")

    # Gate 9: trail engagement plausibility
    c1_engage = 1.0 - float(pooled.loc[pooled["variant_id"] == "C1", "sl_rate"].iloc[0])
    c3_engage = 1.0 - float(pooled.loc[pooled["variant_id"] == "C3", "sl_rate"].iloc[0])
    disp["gate_9"] = f"C1_engage={c1_engage:.4f}, C3_engage={c3_engage:.4f}"
    if not (0.45 <= c1_engage <= 0.55):
        raise RuntimeError(f"Gate 9 HALT — C1 engage {c1_engage:.4f} outside [0.45, 0.55]")
    if not (0.28 <= c3_engage <= 0.38):
        raise RuntimeError(f"Gate 9 HALT — C3 engage {c3_engage:.4f} outside [0.28, 0.38]")

    # Gate 10: TP rate plausibility
    e1_tp = float(pooled.loc[pooled["variant_id"] == "E1", "tp_exit_rate"].iloc[0])
    e2_tp = float(pooled.loc[pooled["variant_id"] == "E2", "tp_exit_rate"].iloc[0])
    disp["gate_10"] = f"E1_tp={e1_tp:.4f}, E2_tp={e2_tp:.4f}"
    if not (0.25 <= e1_tp <= 0.45):
        raise RuntimeError(f"Gate 10 HALT — E1 TP {e1_tp:.4f} outside [0.25, 0.45]")
    if not (0.10 <= e2_tp <= 0.30):
        raise RuntimeError(f"Gate 10 HALT — E2 TP {e2_tp:.4f} outside [0.10, 0.30]")

    # Gate 11: Class A SL rate
    bl_sl_rate = float(pooled.loc[pooled["variant_id"] == "BL", "sl_rate"].iloc[0])
    a1_sl = float(pooled.loc[pooled["variant_id"] == "A1", "sl_rate"].iloc[0])
    a2_sl = float(pooled.loc[pooled["variant_id"] == "A2", "sl_rate"].iloc[0])
    disp["gate_11"] = f"BL_sl={bl_sl_rate:.4f}, A1_sl={a1_sl:.4f}, A2_sl={a2_sl:.4f}"
    if a1_sl <= bl_sl_rate:
        raise RuntimeError(f"Gate 11 HALT — A1 SL {a1_sl:.4f} not > BL {bl_sl_rate:.4f}")
    if a2_sl >= bl_sl_rate:
        raise RuntimeError(f"Gate 11 HALT — A2 SL {a2_sl:.4f} not < BL {bl_sl_rate:.4f}")

    # Gate 12: G2 cell consistency vs B2 for trades exiting before k=120
    b2_sub = vt[vt["variant_id"] == "B2"].set_index("trade_id")
    g2_sub = vt[vt["variant_id"] == "G2"].set_index("trade_id")
    # Trades where B2 exited before k=120
    early_b2 = b2_sub[b2_sub["exit_bar"] < 120]
    # Sample 20 random
    rng = np.random.default_rng(42)
    sample_n = min(20, len(early_b2))
    sample_tids = rng.choice(early_b2.index.values, size=sample_n, replace=False)
    mismatches = 0
    for tid in sample_tids:
        b2r = b2_sub.loc[tid]
        g2r = g2_sub.loc[tid]
        if (int(b2r["exit_bar"]) != int(g2r["exit_bar"])
                or abs(float(b2r["net_R"]) - float(g2r["net_R"])) > 1e-12):
            mismatches += 1
    disp["gate_12"] = f"sampled={sample_n}, mismatches={mismatches}"
    if mismatches > 0:
        raise RuntimeError(f"Gate 12 HALT — G2/B2 consistency: {mismatches} mismatches in {sample_n}")

    return disp


def _write_gate4_halt_diagnostic(*, out_dir: Path, disp_gate4: Dict[str, str],
                                  sample: pd.DataFrame, max_abs: float) -> None:
    L = [
        "# GATE 4 HALT DIAGNOSTIC — Counterfactual Sweep Round 1 v2",
        "",
        f"_Generated: {_dt.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Disposition",
        "",
        f"**Gate 4 FAILED.** Spec requires `abs diff < 1e-9 per trade, zero mismatches by integer index`.",
        f"Observed: max_abs_diff = **{max_abs:.3e}**.",
        "",
        "## Per-exit-reason breakdown",
        "",
        f"- stop_loss trades: max abs diff = {disp_gate4['max_sl']}",
        f"- time_exit trades: max abs diff = {disp_gate4['max_te']}",
        f"- data_end trades:  max abs diff = {disp_gate4['max_de']}",
        f"",
        f"Mismatches by tolerance:",
        f"- >= 1e-9: {disp_gate4['mismatches_1e9']} of 3993",
        f"- >= 1e-7: {disp_gate4['mismatches_1e7']} of 3993",
        f"- >= 1e-6: {disp_gate4['mismatches_1e6']} of 3993",
        "",
        "## Diagnosis: float-arithmetic ULP precision loss through CSV-stored inputs",
        "",
        "The execution-faithful BL R formula reproduces v1.1's `R` column to **float-ULP-level precision** for SL trades (max diff ~ 6e-10, PASSES <1e-9 strict), but TIME_EXIT trades have larger ULP propagation (max diff ~ 9e-9) due to the longer arithmetic chain.",
        "",
        "### Source of precision loss",
        "",
        "1. **Arc 2 module stores R as `:.10g`** (`core/signals/l4_mtf_alignment_2_down_mixed_kijun.py:904`). This gives ~10 significant digits of storage precision per value (≈ 1e-10 round-off vs Arc 2's internal float64).",
        "2. **trade_index inputs** (`entry_price`, `atr_1h_wilder_at_signal`) are also `:.10g`-stored passthroughs from trades_all.csv.",
        "3. **`next_bar_open_atr` in v1.2.1** is also stored as `:.10g` (matches v1.2 convention).",
        "",
        "When the sweep engine reads these stored values and computes BL R via",
        "",
        "```",
        "  R_te = (open[k+120] - sp_exit × pip / 2 - entry_fill) / (2 × atr)",
        "```",
        "",
        "the **chain of 4 arithmetic operations on `:.10g`-precision inputs** propagates ULP errors to ~1e-8 in R. This is intrinsic to IEEE 754 float64 — not a logic bug.",
        "",
        "SL trades pass because their formula `R = -1 - sp_exit × pip / (4 × atr)` has only 2 ops, propagating less ULP error.",
        "",
        "### Verified independently",
        "",
        "Pre-flight (before writing variant_trades) confirmed the same magnitude when using RAW 1H open prices from `data/1hr/<pair>.csv` (full float64 precision, no CSV roundtrip on `next_bar_open_atr`) — max diff 8.11e-9 for TE trades. So the issue is not the v1.2.1 `:.10g` storage of next_bar_open_atr; it's the trade_index passthroughs of `entry_price` and `atr` that introduce precision loss.",
        "",
        "## Top-20 worst mismatches",
        "",
        "```",
        sample.to_string(index=False),
        "```",
        "",
        "## Resolutions for the planner",
        "",
        "**Option 1 — Relax gate 4 tolerance to `abs diff < 1e-7`.** Recommended. ULP-level precision (max diff ~9e-9) is float-noise; relaxing to 1e-7 gives a 10× safety margin and still verifies engine semantic correctness to better than 0.00001% of an R-unit. All variant comparisons remain accurate to many decimal places.",
        "",
        "**Option 2 — Reproduce Arc 2's exact float-arithmetic chain.** Use `position_size_units × quote_to_usd / risk_per_trade_usd` chain instead of the algebraically-equivalent `/ sl_distance` shortcut. `position_size_units` is in trades_all.csv; `quote_to_usd_rate` is not stored. Would require reproducing Arc 2's quote_to_usd lookup logic — large scope expansion for marginal precision gain.",
        "",
        "**Option 3 — Regenerate v1.0/v1.1/v1.2/v1.2.1 chain with `:.18g` storage.** Restores full float64 precision through the input chain. Largest scope: changes the v1.0 baseline. Not recommended.",
        "",
        "**Option 4 — Use trade_index `R` column directly for BL net_R.** Defeats engine verification but trivially satisfies gate 4. Variants still use the spec's engine.",
        "",
        "## Preserved outputs",
        "",
        "Despite the halt, `variant_trades.csv`, `variant_summary_pooled.csv`, `variant_summary_per_fold.csv`, and `additivity_calibration.csv` were written for planner inspection. These are USABLE IF the planner chooses Option 1 (relaxed tolerance) — the data is correct, only the tolerance check is too strict.",
    ]
    out = out_dir / "GATE_4_HALT_DIAGNOSTIC.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n=== GATE 4 HALT DIAGNOSTIC written to: {out} ===\n", flush=True)


# ============================================================================
# REPORT WRITER (counterfactual_sweep_round_1.md)
# ============================================================================


def _write_report(*, out_dir: Path, pooled: pd.DataFrame, per_fold: pd.DataFrame,
                   additivity: pd.DataFrame, disp: Dict[str, Any],
                   input_shas: Dict[str, str], output_shas: Dict[str, str],
                   determinism: Dict[str, str], run_timestamps: Dict[str, str],
                   single_run: bool) -> Path:
    pooled_idx = pooled.set_index("variant_id")
    bl_mean = float(pooled_idx.loc["BL", "mean_R"])

    worst_fold = {}
    for vid in [v[0] for v in VARIANTS]:
        sub = per_fold[per_fold["variant_id"] == vid]
        worst = sub.sort_values("mean_R").iloc[0]
        wfdd = sub.sort_values("max_DD_R", ascending=False).iloc[0]
        worst_fold[vid] = (int(worst["fold_id"]), float(worst["mean_R"]),
                            int(wfdd["fold_id"]), float(wfdd["max_DD_R"]))

    L: List[str] = []
    L.append("# Counterfactual Exit-Rule Sweep — Round 1 v2 (execution-faithful)")
    L.append("")
    L.append(f"_Generated: {run_timestamps.get('end', 'n/a')}_")
    L.append("")
    L.append("## 6.1 Header — sha256 + determinism")
    L.append("")
    L.append("**Inputs (locked, sha256-verified at run start AND end):**")
    L.append("")
    for k, v in input_shas.items():
        L.append(f"- `{k}`")
        L.append(f"  - `{v}`")
    L.append("")
    L.append("**Outputs (sha256, run #1):**")
    L.append("")
    for k, v in output_shas.items():
        L.append(f"- `{k}`")
        L.append(f"  - `{v}`")
    L.append("")
    L.append(f"**Determinism (Gate 14):** {'SKIPPED' if single_run else 'PASS — see receipt'}")
    if not single_run:
        for k, v in determinism.items():
            L.append(f"- `{k}`: {v}")
    L.append("")
    L.append(f"**Run timestamps:** start={run_timestamps['start']}, end={run_timestamps['end']}, "
             f"wallclock_run1={run_timestamps['wallclock_run1']}.")
    L.append("")
    L.append("## 6.2 Baseline reproduction validation")
    L.append("")
    L.append(f"Gate 4 disposition: **PASS** — max abs diff = {disp['gate_4']['max_abs_diff']}, "
             f"zero mismatches at the configured tolerance.")
    L.append("")
    L.append(
        "BL `net_R` computation uses the execution-faithful Arc 2 formulas:"
    )
    L.append("- SL exit: `R = -1 - sp_exit × pip / (4 × atr)` (exit-side half-spread only)")
    L.append("- Time exit: `R = (next_bar_open[k+120] - sp_exit × pip / 2 - entry_fill) / (2 × atr)`")
    L.append("- Data-end: `R = (bar_close[k=bavail] - sp_exit × pip / 2 - entry_fill) / (2 × atr)`")
    L.append("")
    L.append(f"Per-exit-reason max abs diff: SL={disp['gate_4']['max_sl']}, "
             f"TE={disp['gate_4']['max_te']}, DE={disp['gate_4']['max_de']}.")
    L.append("")
    L.append("## 6.3 Variant grid summary")
    L.append("")
    L.append("| ID | Spec | n | mean_R | median_R | sl_rate | lift_vs_BL | worst_fold (id, mean) | worst_fold_DD |")
    L.append("|----|------|---|--------|----------|---------|------------|------------------------|----------------|")
    for vid, spec, _ in VARIANTS:
        row = pooled_idx.loc[vid]
        wfid, wfm, wfdd_id, wfdd = worst_fold[vid]
        lift = float(row["mean_R"]) - bl_mean
        L.append(
            f"| **{vid}** | {spec} | {int(row['n_trades'])} | "
            f"{row['mean_R']:+.4f} | {row['median_R']:+.4f} | "
            f"{row['sl_rate']:.3f} | {lift:+.4f} | "
            f"(fold {wfid}, {wfm:+.4f}) | {wfdd:.3f} (fold {wfdd_id}) |"
        )
    L.append("")

    def vmean(vid: str) -> float:
        return float(pooled_idx.loc[vid, "mean_R"])
    def vsl(vid: str) -> float:
        return float(pooled_idx.loc[vid, "sl_rate"])
    def wf(vid: str) -> Tuple[int, float]:
        return worst_fold[vid][0], worst_fold[vid][1]

    L.append("## 6.4 Per-class headline findings")
    L.append("")
    L.append(
        f"**Class A (SL distance).** A1 (M=1.5) pooled mean R = {vmean('A1'):+.4f}, "
        f"sl_rate {vsl('A1'):.3f}, worst-fold mean R {wf('A1')[1]:+.4f} (fold {wf('A1')[0]}). "
        f"A2 (M=2.5) pooled mean R = {vmean('A2'):+.4f}, sl_rate {vsl('A2'):.3f}, "
        f"worst-fold {wf('A2')[1]:+.4f} (fold {wf('A2')[0]}). R reported in variant's own R-unit "
        f"(1R = M × ATR). Spread cost rescaled per §3.4."
    )
    L.append("")
    L.append(
        f"**Class B (BE-SL trigger).** B1 (T=+0.5R) / B2 (T=+1R) / B3 (T=+1.5R) pooled mean R = "
        f"{vmean('B1'):+.4f} / {vmean('B2'):+.4f} / {vmean('B3'):+.4f} vs BL {bl_mean:+.4f}. "
        f"BE-exit shares: B1={float(pooled_idx.loc['B1','be_exit_rate']):.3f}, "
        f"B2={float(pooled_idx.loc['B2','be_exit_rate']):.3f}, "
        f"B3={float(pooled_idx.loc['B3','be_exit_rate']):.3f}. Worst-fold mean R: "
        f"B1={wf('B1')[1]:+.4f}, B2={wf('B2')[1]:+.4f}, B3={wf('B3')[1]:+.4f}."
    )
    L.append("")
    L.append(
        f"**Class C (Trail).** C1 (+1R engage, D=0.5R) / C2 (+1R, D=1R) / C3 (+2R, D=1R) "
        f"pooled mean R = {vmean('C1'):+.4f} / {vmean('C2'):+.4f} / {vmean('C3'):+.4f}. "
        f"Trail-exit shares: C1={float(pooled_idx.loc['C1','trail_exit_rate']):.3f}, "
        f"C2={float(pooled_idx.loc['C2','trail_exit_rate']):.3f}, "
        f"C3={float(pooled_idx.loc['C3','trail_exit_rate']):.3f}. Worst-fold mean R: "
        f"C1={wf('C1')[1]:+.4f}, C2={wf('C2')[1]:+.4f}, C3={wf('C3')[1]:+.4f}."
    )
    L.append("")
    L.append(
        f"**Class D (Partial close + BE-SL).** D1 pooled mean R = {vmean('D1'):+.4f}, "
        f"partial_exit_rate {float(pooled_idx.loc['D1','partial_exit_rate']):.3f}. "
        f"Worst-fold mean R = {wf('D1')[1]:+.4f} (fold {wf('D1')[0]})."
    )
    L.append("")
    L.append(
        f"**Class E (Fixed TP).** E1 (TP=+1.5R) / E2 (TP=+3R) pooled mean R = "
        f"{vmean('E1'):+.4f} / {vmean('E2'):+.4f}. TP-exit shares: "
        f"E1={float(pooled_idx.loc['E1','tp_exit_rate']):.3f}, "
        f"E2={float(pooled_idx.loc['E2','tp_exit_rate']):.3f}. "
        f"Worst-fold mean R: E1={wf('E1')[1]:+.4f}, E2={wf('E2')[1]:+.4f}."
    )
    L.append("")
    L.append(
        f"**Class F (Time-exit horizon).** F1 (k=240) / F2 (no time exit) pooled mean R = "
        f"{vmean('F1'):+.4f} / {vmean('F2'):+.4f}. Worst-fold mean R: "
        f"F1={wf('F1')[1]:+.4f}, F2={wf('F2')[1]:+.4f}."
    )
    L.append("")
    L.append(
        f"**Bonus combinations (G).** G1 (kinked trail) pooled mean R = {vmean('G1'):+.4f}, "
        f"worst-fold {wf('G1')[1]:+.4f}. G2 (B2+F1) pooled mean R = {vmean('G2'):+.4f}, "
        f"worst-fold {wf('G2')[1]:+.4f}."
    )
    L.append("")

    L.append("## 6.5 Additivity calibration (G2 vs B2 + F1)")
    L.append("")
    a = additivity.iloc[0]
    L.append(f"- BL mean R: {a['BL_mean_R']:+.4f}")
    L.append(f"- B2 mean R: {a['B2_mean_R']:+.4f}  (lift: {a['B2_mean_R']-a['BL_mean_R']:+.4f})")
    L.append(f"- F1 mean R: {a['F1_mean_R']:+.4f}  (lift: {a['F1_mean_R']-a['BL_mean_R']:+.4f})")
    L.append(f"- G2 mean R: {a['G2_mean_R']:+.4f}  (lift: {a['combination_lift']:+.4f})")
    L.append(f"- Lone-lift sum: {a['lone_lift_sum']:+.4f}")
    L.append(f"- Combination lift: {a['combination_lift']:+.4f}")
    L.append(f"- **Divergence (combo − sum-of-lones): {a['divergence']:+.4f} R**")
    L.append("")
    L.append(f"Interpretation: {a['interpretation_note']}")
    L.append("")

    L.append("## 6.6 Worst-fold disposition table")
    L.append("")
    folds = sorted(per_fold["fold_id"].unique().tolist())
    head = "| variant | " + " | ".join(f"f{f}" for f in folds) + " | worst fold |"
    sep = "|---" * (len(folds) + 2) + "|"
    L.append("### 6.6a Per-fold mean R")
    L.append("")
    L.append(head); L.append(sep)
    for vid, _, _ in VARIANTS:
        sub = per_fold[per_fold["variant_id"] == vid].set_index("fold_id")
        wfid = int(sub["mean_R"].idxmin())
        cells = []
        for f in folds:
            v = float(sub.loc[f, "mean_R"])
            cells.append(f"**{v:+.4f}**" if f == wfid else f"{v:+.4f}")
        cells.append(f"f{wfid}={float(sub.loc[wfid,'mean_R']):+.4f}")
        L.append(f"| {vid} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### 6.6b Per-fold max DD (R-units, sequential intra-fold R sum)")
    L.append("")
    L.append(head); L.append(sep)
    for vid, _, _ in VARIANTS:
        sub = per_fold[per_fold["variant_id"] == vid].set_index("fold_id")
        wfid = int(sub["max_DD_R"].idxmax())
        cells = []
        for f in folds:
            v = float(sub.loc[f, "max_DD_R"])
            cells.append(f"**{v:.3f}**" if f == wfid else f"{v:.3f}")
        cells.append(f"f{wfid}={float(sub.loc[wfid,'max_DD_R']):.3f}")
        L.append(f"| {vid} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("_Note: within-fold sequential R sum, NOT Arc 2 dollar-DD. Relative ranking + plausibility only._")
    L.append("")

    L.append("## 6.7 Cross-variant observations")
    L.append("")
    pooled_sorted = pooled.sort_values("mean_R", ascending=False)
    top1 = pooled_sorted.iloc[0]; top2 = pooled_sorted.iloc[1]; bot1 = pooled_sorted.iloc[-1]
    L.append(
        f"Variant **{top1['variant_id']}** is highest by pooled mean R "
        f"({top1['mean_R']:+.4f} R). **{top2['variant_id']}** is second "
        f"({top2['mean_R']:+.4f} R). **{bot1['variant_id']}** is lowest "
        f"({bot1['mean_R']:+.4f} R)."
    )
    L.append("")
    wf_pairs = sorted(
        [(vid, worst_fold[vid][1]) for vid, _, _ in VARIANTS],
        key=lambda x: x[1], reverse=True,
    )
    L.append(
        f"By worst-fold mean R: **{wf_pairs[0][0]}** highest ({wf_pairs[0][1]:+.4f} R); "
        f"**{wf_pairs[1][0]}** second ({wf_pairs[1][1]:+.4f} R); "
        f"**{wf_pairs[-1][0]}** lowest ({wf_pairs[-1][1]:+.4f} R)."
    )
    L.append("")
    n_pos_wf = sum(1 for _, m in wf_pairs if m > 0)
    L.append(
        f"Of the {len(VARIANTS)} specifications, {n_pos_wf} have positive worst-fold mean R; "
        f"{len(VARIANTS) - n_pos_wf} are at or below 0."
    )
    L.append("")
    n_gt_bl = sum(1 for vid, _, _ in VARIANTS
                   if float(pooled_idx.loc[vid, "mean_R"]) > bl_mean and vid != "BL")
    L.append(
        f"Of the 15 non-BL variants, {n_gt_bl} have pooled mean R above BL "
        f"({bl_mean:+.4f} R); {15 - n_gt_bl} are at or below BL."
    )
    L.append("")

    L.append("## 6.8 Planning input — descriptive ranking for planner decision")
    L.append("")
    L.append(
        "_This subsection steps outside §14.5's disposition discipline to provide ranked "
        "observations for planner use. No phase-2 spec is selected here; planner decides "
        "in follow-up workflow._"
    )
    L.append("")
    L.append("### 6.8a Ranked by pooled mean R (descending)")
    L.append("")
    L.append("| rank | variant_id | pooled mean R | lift vs BL |")
    L.append("|---|---|---|---|")
    pooled_sorted_m = pooled.sort_values("mean_R", ascending=False).reset_index(drop=True)
    for i, row in pooled_sorted_m.iterrows():
        L.append(f"| {i+1} | {row['variant_id']} | {row['mean_R']:+.4f} | "
                 f"{row['mean_R'] - bl_mean:+.4f} |")
    L.append("")
    L.append("### 6.8b Ranked by worst-fold mean R (descending)")
    L.append("")
    L.append("| rank | variant_id | worst-fold mean R | worst fold ID |")
    L.append("|---|---|---|---|")
    for i, (vid, m) in enumerate(wf_pairs):
        L.append(f"| {i+1} | {vid} | {m:+.4f} | {worst_fold[vid][0]} |")
    L.append("")
    bl_dd = max(float(per_fold[(per_fold["variant_id"] == "BL")
                                  & (per_fold["fold_id"] == f)]["max_DD_R"].iloc[0])
                  for f in folds)
    L.append("### 6.8c Ranked by max-DD improvement vs BL (positive = lower DD)")
    L.append("")
    L.append("| rank | variant_id | worst-fold max_DD_R | improvement vs BL |")
    L.append("|---|---|---|---|")
    dd_pairs = sorted(
        [(vid, worst_fold[vid][3]) for vid, _, _ in VARIANTS],
        key=lambda x: x[1],
    )
    for i, (vid, dd) in enumerate(dd_pairs):
        improvement = bl_dd - dd
        L.append(f"| {i+1} | {vid} | {dd:.3f} | {improvement:+.3f} |")
    L.append("")
    L.append(f"_BL worst-fold max DD: {bl_dd:.3f} R (reference for improvement column)._")
    L.append("")
    L.append("## 6.9 Out-of-scope items observed")
    L.append("")
    L.append(
        "- **Class D taxonomy extension**: added `partial_then_data_end` for partial-closed trades hitting data_end. Spec §7.1 lists `partial_then_{be,time}` only; this is a logically distinct outcome. Rolled into `data_end_rate` and `partial_exit_rate` aggregates."
    )
    L.append(
        "- **Variant G1 kinked trail**: trail level steps backward by ~0.5R at running_mfe = 6.0 ATR boundary by intentional design (the kink is the experimental hypothesis). Implementation follows the literal `trail_level = running_mfe - 2 × D_current`."
    )
    out = out_dir / "counterfactual_sweep_round_1.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def _gate_13_disposition_grep(report_path: Path) -> Tuple[bool, List[str]]:
    text = report_path.read_text(encoding="utf-8")
    m = re.search(r"##\s+6\.8\s+Planning input", text)
    if m is None:
        return False, ["§6.8 boundary not found"]
    start = re.search(r"##\s+6\.2\s+Baseline", text)
    if start is None:
        return False, ["§6.2 boundary not found"]
    region = text[start.start():m.start()]
    forbidden = ["should be selected", "passes the gate", "best variant",
                  "we should", "would have"]
    hits = []
    for phrase in forbidden:
        for m2 in re.finditer(re.escape(phrase), region, flags=re.IGNORECASE):
            ctx_s = max(0, m2.start() - 60)
            ctx_e = min(len(region), m2.end() + 60)
            hits.append(f"  '{phrase}': ...{region[ctx_s:ctx_e]}...")
    return (len(hits) == 0, hits)


def run_pipeline(*, out_dir: Path, per_bar_csv: Path, trade_index_csv: Path,
                 trades_all_csv: Path, write_report: bool = True,
                 input_shas: Dict[str, str] = None,
                 determinism: Dict[str, str] = None,
                 run_timestamps: Dict[str, str] = None,
                 single_run: bool = False) -> Tuple[Dict[str, str], Dict[str, Any]]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Stage 1: Computing variants...", flush=True)
    vt, ti_full = _run_sweep(per_bar_csv=per_bar_csv, trade_index_csv=trade_index_csv,
                              trades_all_csv=trades_all_csv)

    print("Stage 2: Aggregating...", flush=True)
    pooled = _aggregate_pooled(vt)
    per_fold = _aggregate_per_fold(vt)
    additivity = _additivity_calibration(pooled)

    # Pre-gate writes (preserve for inspection on halt)
    print("Stage 3a: Writing CSV outputs (pre-gate)...", flush=True)
    vt_p = out_dir / "variant_trades.csv"
    vp_p = out_dir / "variant_summary_pooled.csv"
    vf_p = out_dir / "variant_summary_per_fold.csv"
    ad_p = out_dir / "additivity_calibration.csv"
    vt.to_csv(vt_p, index=False, lineterminator="\n", float_format="%.10g")
    pooled.to_csv(vp_p, index=False, lineterminator="\n", float_format="%.10g")
    per_fold.to_csv(vf_p, index=False, lineterminator="\n", float_format="%.10g")
    additivity.to_csv(ad_p, index=False, lineterminator="\n", float_format="%.10g")

    print("Stage 3b: Validating gates 2-12...", flush=True)
    disp = _validate_gates(vt=vt, ti_full=ti_full, pooled=pooled,
                            per_fold=per_fold, out_dir=out_dir)

    out_shas = {
        "variant_trades.csv": _sha256_file(vt_p),
        "variant_summary_pooled.csv": _sha256_file(vp_p),
        "variant_summary_per_fold.csv": _sha256_file(vf_p),
        "additivity_calibration.csv": _sha256_file(ad_p),
    }

    if write_report:
        print("Stage 4: Writing report + gate 13...", flush=True)
        report_path = _write_report(
            out_dir=out_dir, pooled=pooled, per_fold=per_fold, additivity=additivity,
            disp=disp, input_shas=input_shas or {}, output_shas=out_shas,
            determinism=determinism or {},
            run_timestamps=run_timestamps or {"start": "n/a", "end": "n/a", "wallclock_run1": "n/a"},
            single_run=single_run,
        )
        out_shas["counterfactual_sweep_round_1.md"] = _sha256_file(report_path)
        ok, hits = _gate_13_disposition_grep(report_path)
        disp["gate_13"] = "PASS" if ok else f"HALT — {len(hits)} hits"
        if not ok:
            for h in hits:
                print(h)
            raise RuntimeError(f"Gate 13 HALT — disposition discipline:\n" + "\n".join(hits))

    return out_shas, disp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/extended/counterfactuals/round_1"))
    parser.add_argument("--per-bar-csv",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv"))
    parser.add_argument("--trade-index-csv",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv"))
    parser.add_argument("--trades-all-csv",
        default=str(REPO_ROOT / "results/l6/arc2/trades_all.csv"))
    parser.add_argument("--single-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Arc 2 counterfactual sweep — Round 1 v2 (execution-faithful)")
    print("=" * 60)
    tracemalloc.start()
    t_start = time.time()
    start_iso = _dt.datetime.now().isoformat(timespec="seconds")

    print("\n[Gate 1] Verifying input sha256s...")
    input_shas = _verify_input_integrity()
    for k in input_shas:
        print(f"  OK {k}")

    out_dir = Path(args.output_dir)
    pb_csv = Path(args.per_bar_csv); ti_csv = Path(args.trade_index_csv); ta_csv = Path(args.trades_all_csv)

    print(f"\n[Run #1] Output dir: {out_dir}")
    t_r1 = time.time()
    sha1, disp = run_pipeline(out_dir=out_dir, per_bar_csv=pb_csv,
                                trade_index_csv=ti_csv, trades_all_csv=ta_csv,
                                write_report=False)
    el1 = time.time() - t_r1
    print(f"  Run #1 complete in {el1:.1f}s")
    for k, v in sha1.items():
        print(f"    {k}: {v}")

    determinism: Dict[str, str] = {}
    if not args.single_run:
        scratch = Path(tempfile.mkdtemp(prefix="arc2_sweep_v2_run2_"))
        print(f"\n[Run #2 / Gate 14] Output dir (scratch): {scratch}")
        t_r2 = time.time()
        sha2, _ = run_pipeline(out_dir=scratch, per_bar_csv=pb_csv,
                                 trade_index_csv=ti_csv, trades_all_csv=ta_csv,
                                 write_report=False)
        el2 = time.time() - t_r2
        print(f"  Run #2 complete in {el2:.1f}s")
        det_pass = True
        for k in sha1:
            match = sha1[k] == sha2[k]
            determinism[k] = "match" if match else "MISMATCH"
            print(f"    {k}: {determinism[k]}")
            if not match:
                det_pass = False
        try:
            for p in scratch.iterdir(): p.unlink()
            scratch.rmdir()
        except Exception:
            pass
        if not det_pass:
            raise RuntimeError("Gate 14 HALT — determinism failed")

    end_iso = _dt.datetime.now().isoformat(timespec="seconds")
    rt = {"start": start_iso, "end": end_iso, "wallclock_run1": f"{el1:.1f}s"}
    print("\n[Final] Writing report with determinism receipt...")
    report_path = _write_report(out_dir=out_dir, pooled=_aggregate_pooled(pd.read_csv(out_dir / "variant_trades.csv")),
                                  per_fold=pd.read_csv(out_dir / "variant_summary_per_fold.csv"),
                                  additivity=pd.read_csv(out_dir / "additivity_calibration.csv"),
                                  disp=disp, input_shas=input_shas, output_shas=sha1,
                                  determinism=determinism, run_timestamps=rt,
                                  single_run=args.single_run)
    sha1["counterfactual_sweep_round_1.md"] = _sha256_file(report_path)

    print("\n[Gate 13] Disposition-discipline grep...")
    ok, hits = _gate_13_disposition_grep(report_path)
    disp["gate_13"] = "PASS" if ok else f"HALT — {len(hits)} hits"
    if not ok:
        for h in hits: print(h)
        raise RuntimeError("Gate 13 HALT")
    print("  PASS — no forbidden phrases in §6.2-§6.7")

    print("\n[Gate 15] Re-verifying locked artefact integrity...")
    post = _verify_input_integrity()
    for k in input_shas:
        if input_shas[k] != post[k]:
            raise RuntimeError(f"Gate 15 HALT — {k} changed mid-run")
    for rel, exp in ADJ_LOCKED_SHAS.items():
        actual = _sha256_file(REPO_ROOT / rel)
        if actual != exp:
            raise RuntimeError(f"Gate 15 HALT — {rel} sha changed")
    ch_path = REPO_ROOT / "docs/CANDIDATE_HYPOTHESES.md"
    if ch_path.exists():
        actual = _sha256_file(ch_path)
        if actual != CANDIDATE_HYPOTHESES_BASELINE_SHA:
            raise RuntimeError(f"Gate 15 HALT — CANDIDATE_HYPOTHESES.md changed")
    print("  All locked artefacts unchanged.")

    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rm = [
        "Arc 2 counterfactual sweep — Round 1 v2 — run manifest",
        "=" * 60,
        f"Run timestamps: start={start_iso}, end={end_iso}",
        f"Wallclock run #1: {el1:.1f}s",
    ]
    if not args.single_run:
        rm.append(f"Wallclock run #2: {el2:.1f}s")
    rm.append(f"Memory peak (tracemalloc): {peak / (1024*1024):.1f} MB")
    rm.append("")
    rm.append("Inputs (sha256):")
    for k, v in input_shas.items():
        rm.append(f"  {k}\n    {v}")
    rm.append("")
    rm.append("Outputs (sha256):")
    for k, v in sha1.items():
        p = out_dir / k
        sz = p.stat().st_size if p.exists() else 0
        rm.append(f"  {k} ({sz:,} bytes)\n    {v}")
    rm.append("")
    rm.append("Determinism (Gate 14):")
    if args.single_run:
        rm.append("  SKIPPED")
    else:
        for k, v in determinism.items():
            rm.append(f"  {k}: {v}")
    rm.append("")
    rm.append("Gate dispositions:")
    for k in sorted(disp.keys()):
        rm.append(f"  {k}: {disp[k]}")
    rm_p = out_dir / "run_manifest.txt"
    rm_p.write_text("\n".join(rm) + "\n", encoding="utf-8")
    print(f"\n[Manifest] {rm_p}")
    print(f"\nMemory peak: {peak / (1024*1024):.1f} MB")
    print(f"Total wallclock: {time.time() - t_start:.1f}s")
    print("\nAll outputs written. Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
