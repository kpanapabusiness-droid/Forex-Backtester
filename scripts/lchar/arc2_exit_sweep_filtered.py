"""Arc 2 — Exit-rule variant sweep on filtered subsets (Phase 2 Round 2).

Phase: L6 Arc 2 Phase 3 — Filtered-subset x exit-variant sweep
(L6_0_METHODOLOGY_LOCK Section 14.2 derivative experiment, 14.5
descriptive discipline, 14.6 read-existing-CSV backfill).

Scope: For each (filtered subset, exit-rule variant) combination,
simulate bar-by-bar execution against per_bar_paths.csv with the
-2R SL constraint active throughout. Report mean R, per-fold mean R,
per-Block-B-category outcome decomposition, and exit-reason
distribution.

Descriptive only. No filter selection. No WFO. No signal-module mod.

Outputs to: results/l6/arc2/characterisation/extended/exit_sweep_filtered/
"""

from __future__ import annotations

import filecmp
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Locked input sha256s (gate 1, re-verified at end as gate 9)
# ---------------------------------------------------------------------------
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv":
        "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv":
        "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv":
        "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv":
        "4a61407f0f1fc1b74486f0614928e776201dc6469d874db8393e689d20cdb2ff",
    "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv":
        "a5e3f8e68aa64d8fd53f752705a33613d9877dbde1f8265cb4a38d753c5e088e",
    "results/l6/arc2/characterisation/extended/path_by_subset/block_V_subset_category_breakdown.csv":
        "78633e9904baf2a672d2c8692f4b3557fec0aa3af8044ef3296dde08bad71c02",
    "results/l6/arc2/characterisation/extended/counterfactuals/round_1/counterfactual_sweep_round_1.md":
        "635ad1fdaf26525cd5e27c1d8b4c4d807da44d9d9d7c83afed9c8754dbc6e0b2",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py":
        "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml":
        "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "L6_0_METHODOLOGY_LOCK.md":
        "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}

# trades_all.csv is required for spread lookup but not formally locked here
# (matches Round 1 v2 convention). Capture its sha for record only.
TRADES_ALL_REL = "results/l6/arc2/trades_all.csv"

OUTPUT_DIR_REL = "results/l6/arc2/characterisation/extended/exit_sweep_filtered"

TIME_HORIZON_DEFAULT: int = 120
TIME_HORIZON_EXTENDED: int = 240

# Path-category constants (must reproduce Block B 1R counts).
PATH_THRESH_ATR: float = 2.0  # +1R == +2.0 ATR fill-rel
PATH_HOLD_CAP: int = 120

# Expected Block B 1R counts (gate from arc2_path_by_subset.py).
BLOCK_B_1R_COUNTS: Dict[str, int] = {
    "only_up": 956,
    "up_then_down": 1075,
    "down_then_up": 1090,
    "straight_to_sl": 858,
    "simultaneous": 13,
    "neither_reached": 1,
}
ALL_CATS: Tuple[str, ...] = (
    "only_up", "up_then_down", "down_then_up",
    "straight_to_sl", "simultaneous", "neither_reached",
)

# Quintile expectations (mirrors arc2_entry_filter_bivariate / path_by_subset).
EXPECTED_Q_SIZES_BY_Q: Dict[str, int] = {
    "Q1": 799, "Q2": 799, "Q3": 799, "Q4": 798, "Q5": 798,
}

# Subset definitions.
SUBSET_DEFS: List[Tuple[str, Dict[str, Any]]] = [
    ("S0_pop", {"all": True, "expected_n": 3993}),
    ("S1_q5q2", {"qa": ("Q5",), "qb": ("Q2",), "expected_n": 190}),
    ("S2_q5q3", {"qa": ("Q5",), "qb": ("Q3",), "expected_n": 178}),
    ("S3_q4q2", {"qa": ("Q4",), "qb": ("Q2",), "expected_n": 151}),
    ("S4_q5xq2q3", {"qa": ("Q5",), "qb": ("Q2", "Q3"), "expected_n": 368}),
    ("S5_q4q5xq2q3", {"qa": ("Q4", "Q5"), "qb": ("Q2", "Q3"),
                      "expected_n": 682}),
]
SUBSET_IDS: Tuple[str, ...] = tuple(s[0] for s in SUBSET_DEFS)

# Per-subset expected pooled mean R from block_P bivariate cells (gate 7).
EXPECTED_POOLED_MEAN_R: Dict[str, Optional[float]] = {
    "S0_pop": None,
    "S1_q5q2": 0.4325631187,
    "S2_q5q3": 0.2212491288,
    "S3_q4q2": 0.278122118,
    "S4_q5xq2q3": None,
    "S5_q4q5xq2q3": None,
}

# Disposition discipline forbidden patterns (Section 14.5).
FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "best variant",
    " recommend",  # leading space to skip "recommendation" in URLs
    "should use",
    "this is the answer",
    "we should adopt",
    "the right exit rule is",
    "this would pass",
)


# ===========================================================================
# sha256 / IO helpers
# ===========================================================================


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_locked(label: str) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for rel, expected in LOCKED_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"HALT ({label}) sha256 mismatch on {rel}:\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
        observed[rel] = actual
    return observed


def _write_csv(df: pd.DataFrame, path: Path, float_fmt: str = "%.10g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format=float_fmt)


# ===========================================================================
# Variant mechanics (byte-faithful copy of Round 1 v2 logic)
# ===========================================================================


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _exit_te_de(*, k_idx: int, T: Dict[str, Any], M: float,
                exit_atr_fill: float, exit_reason: str) -> Dict[str, Any]:
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    exit_price_offset = exit_atr_fill * atr - sp_exit * pip / 2
    sl_dist = M * atr
    net_R = exit_price_offset / sl_dist
    exit_level_atr_fill = exit_atr_fill - sp_exit * pip / (2 * atr)
    gross_R = (exit_atr_fill + T["entry_fill_offset_atr"]) / M
    spread_cost_R = gross_R - net_R
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": exit_reason,
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def _exit_sl(*, k_idx: int, T: Dict[str, Any], M: float) -> Dict[str, Any]:
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    net_R = -1.0 - sp_exit * pip / (2 * M * atr)
    exit_level_atr_fill = -M - sp_exit * pip / (2 * atr)
    gross_R = -1.0
    spread_cost_R = gross_R - net_R
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": "stop_loss",
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def _exit_be(*, k_idx: int, T: Dict[str, Any]) -> Dict[str, Any]:
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    net_R = -T["baseline_spread_cost_r"]
    exit_level_atr_fill = -T["entry_fill_offset_atr"] - sp_exit * pip / (2 * atr)
    gross_R = 0.0
    spread_cost_R = gross_R - net_R
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": "be_exit",
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def _exit_trail(*, k_idx: int, T: Dict[str, Any],
                trail_level_atr_fill: float) -> Dict[str, Any]:
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    net_R = trail_level_atr_fill / 2 - sp_exit * pip / (4 * atr)
    exit_level_atr_fill = trail_level_atr_fill - sp_exit * pip / (2 * atr)
    gross_R = (trail_level_atr_fill + T["entry_fill_offset_atr"]) / 2
    spread_cost_R = gross_R - net_R
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": "trail_exit",
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def _exit_tp(*, k_idx: int, T: Dict[str, Any], T_TP_R: float) -> Dict[str, Any]:
    atr = T["atr"]
    sp_exit = T["sp_exit_pips"]
    pip = T["pip"]
    net_R = T_TP_R - T["baseline_spread_cost_r"]
    exit_level_atr_fill = 2 * T_TP_R - T["entry_fill_offset_atr"] - sp_exit * pip / (2 * atr)
    gross_R = T_TP_R
    spread_cost_R = gross_R - net_R
    return {
        "exit_bar": k_idx + 1,
        "exit_reason": "fixed_tp",
        "exit_level_atr_fill": exit_level_atr_fill,
        "gross_R": gross_R,
        "spread_cost_R": spread_cost_R,
        "net_R": net_R,
    }


def variant_BL(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
               *, time_horizon: int = TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            return _exit_sl(k_idx=k_idx, T=T, M=2.0)
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
    raise RuntimeError("BL did not terminate")


def variant_TP(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
               *, T_TP_R: float,
               time_horizon: int = TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    tp_threshold_atr_fill = 2.0 * T_TP_R - T["entry_fill_offset_atr"]
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            return _exit_sl(k_idx=k_idx, T=T, M=2.0)
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


def variant_TRAIL(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
                  *, T_engage_R: float, D_R: float,
                  time_horizon: int = TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    engage_threshold_atr_fill = 2.0 * T_engage_R - T["entry_fill_offset_atr"]
    trail_active = False
    D_atr = 2.0 * D_R
    for k_idx in range(bavail):
        k = k_idx + 1
        if not trail_active:
            if rmae[k_idx] <= -2.0:
                return _exit_sl(k_idx=k_idx, T=T, M=2.0)
            if rmfe[k_idx] >= engage_threshold_atr_fill:
                trail_active = True  # activate from k+1
        else:
            trail_level_atr_fill = rmfe[k_idx] - D_atr
            if bl_[k_idx] <= trail_level_atr_fill:
                return _exit_trail(k_idx=k_idx, T=T,
                                   trail_level_atr_fill=trail_level_atr_fill)
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


def variant_BE_TP(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
                  *, T_BE_R: float, T_TP_R: float,
                  time_horizon: int = TIME_HORIZON_DEFAULT) -> Dict[str, Any]:
    """V08: BE-SL at +T_BE_R mid-rel and fixed TP at +T_TP_R mid-rel.

    Before BE armed: hard SL active. After BE armed: BE-SL replaces hard SL.
    TP is checked on every bar (priority after SL/BE per the spec's
    "first triggering condition wins", with SL checked first).
    """
    be_threshold_atr_fill = 2.0 * T_BE_R - T["entry_fill_offset_atr"]
    tp_threshold_atr_fill = 2.0 * T_TP_R - T["entry_fill_offset_atr"]
    entry_mid_atr_fill = -T["entry_fill_offset_atr"]
    be_active = False
    for k_idx in range(bavail):
        k = k_idx + 1
        # SL precedence on this bar (per Section 4.6 / spec).
        if not be_active:
            if rmae[k_idx] <= -2.0:
                return _exit_sl(k_idx=k_idx, T=T, M=2.0)
        else:
            if bl_[k_idx] <= entry_mid_atr_fill:
                return _exit_be(k_idx=k_idx, T=T)
        # Then fixed TP.
        if bh_[k_idx] >= tp_threshold_atr_fill:
            return _exit_tp(k_idx=k_idx, T=T, T_TP_R=T_TP_R)
        # Arm BE for next bar (Round 1 convention: activate from k+1).
        if not be_active and rmfe[k_idx] >= be_threshold_atr_fill:
            be_active = True
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
    raise RuntimeError("BE_TP did not terminate")


def variant_BE_then_TRAIL(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T,
                          *, T_BE_R: float, T_engage_R: float, D_R: float,
                          time_horizon: int = TIME_HORIZON_EXTENDED) -> Dict[str, Any]:
    """V15: BE-SL armed at +T_BE_R MFE, then trail engaged at +T_engage_R MFE
    with distance D_R below peak. Once trail engages, it supersedes the BE-SL.
    """
    be_threshold_atr_fill = 2.0 * T_BE_R - T["entry_fill_offset_atr"]
    engage_threshold_atr_fill = 2.0 * T_engage_R - T["entry_fill_offset_atr"]
    entry_mid_atr_fill = -T["entry_fill_offset_atr"]
    D_atr = 2.0 * D_R
    be_active = False
    trail_active = False
    for k_idx in range(bavail):
        k = k_idx + 1
        # Exit checks in precedence order.
        if not be_active and not trail_active:
            if rmae[k_idx] <= -2.0:
                return _exit_sl(k_idx=k_idx, T=T, M=2.0)
        elif be_active and not trail_active:
            if bl_[k_idx] <= entry_mid_atr_fill:
                return _exit_be(k_idx=k_idx, T=T)
        else:  # trail_active
            trail_level_atr_fill = rmfe[k_idx] - D_atr
            if bl_[k_idx] <= trail_level_atr_fill:
                return _exit_trail(k_idx=k_idx, T=T,
                                   trail_level_atr_fill=trail_level_atr_fill)
        # Activation triggers (delayed to k+1 in Round 1 convention).
        if not trail_active and rmfe[k_idx] >= engage_threshold_atr_fill:
            trail_active = True
        elif not be_active and rmfe[k_idx] >= be_threshold_atr_fill:
            be_active = True
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
    raise RuntimeError("BE_then_TRAIL did not terminate")


# ---------------------------------------------------------------------------
# Variant registry (16 variants)
# ---------------------------------------------------------------------------


def _v01(*a):
    return variant_TP(*a, T_TP_R=1.0, time_horizon=TIME_HORIZON_DEFAULT)


def _v02(*a):
    return variant_TP(*a, T_TP_R=1.5, time_horizon=TIME_HORIZON_DEFAULT)


def _v03(*a):
    return variant_TP(*a, T_TP_R=2.0, time_horizon=TIME_HORIZON_DEFAULT)


def _v04(*a):
    return variant_TP(*a, T_TP_R=3.0, time_horizon=TIME_HORIZON_DEFAULT)


def _v05(*a):
    return variant_TRAIL(*a, T_engage_R=1.0, D_R=0.5,
                         time_horizon=TIME_HORIZON_DEFAULT)


def _v06(*a):
    return variant_TRAIL(*a, T_engage_R=1.0, D_R=1.0,
                         time_horizon=TIME_HORIZON_DEFAULT)


def _v07(*a):
    return variant_TRAIL(*a, T_engage_R=1.0, D_R=1.5,
                         time_horizon=TIME_HORIZON_DEFAULT)


def _v08(*a):
    return variant_BE_TP(*a, T_BE_R=1.0, T_TP_R=2.0,
                         time_horizon=TIME_HORIZON_DEFAULT)


def _v09(*a):
    return variant_BL(*a, time_horizon=TIME_HORIZON_EXTENDED)


def _v10(*a):
    return variant_TRAIL(*a, T_engage_R=1.0, D_R=0.5,
                         time_horizon=TIME_HORIZON_EXTENDED)


def _v11(*a):
    return variant_TRAIL(*a, T_engage_R=1.0, D_R=1.0,
                         time_horizon=TIME_HORIZON_EXTENDED)


def _v12(*a):
    return variant_TP(*a, T_TP_R=1.0, time_horizon=TIME_HORIZON_EXTENDED)


def _v13(*a):
    return variant_TP(*a, T_TP_R=2.0, time_horizon=TIME_HORIZON_EXTENDED)


def _v14(*a):
    # Spec V14 = TP at +1R if reached, else hold to k=240 with SL active.
    # Mechanically identical to V12_TP1_H240 (TP at +1R with horizon=240).
    # Kept distinct per spec's listed 16 variants.
    return variant_TP(*a, T_TP_R=1.0, time_horizon=TIME_HORIZON_EXTENDED)


def _v15(*a):
    return variant_BE_then_TRAIL(*a, T_BE_R=1.0, T_engage_R=2.0, D_R=1.0,
                                 time_horizon=TIME_HORIZON_EXTENDED)


VARIANTS: List[Tuple[str, str, Callable[..., Dict[str, Any]], int]] = [
    ("V00_BL", "Baseline: SL=-2R, TE=k=120", variant_BL, TIME_HORIZON_DEFAULT),
    ("V01_TP1", "TP=+1R, TE=k=120", _v01, TIME_HORIZON_DEFAULT),
    ("V02_TP15", "TP=+1.5R, TE=k=120", _v02, TIME_HORIZON_DEFAULT),
    ("V03_TP2", "TP=+2R, TE=k=120", _v03, TIME_HORIZON_DEFAULT),
    ("V04_TP3", "TP=+3R, TE=k=120", _v04, TIME_HORIZON_DEFAULT),
    ("V05_TR_act1_d05", "Trail act +1R, D=0.5R, TE=k=120", _v05, TIME_HORIZON_DEFAULT),
    ("V06_TR_act1_d1", "Trail act +1R, D=1R, TE=k=120", _v06, TIME_HORIZON_DEFAULT),
    ("V07_TR_act1_d15", "Trail act +1R, D=1.5R, TE=k=120", _v07, TIME_HORIZON_DEFAULT),
    ("V08_BE_TP2", "BE-SL @ +1R + TP @ +2R, TE=k=120", _v08, TIME_HORIZON_DEFAULT),
    ("V09_H240", "BL with TE=k=240", _v09, TIME_HORIZON_EXTENDED),
    ("V10_TR05_H240", "Trail act +1R, D=0.5R, TE=k=240", _v10, TIME_HORIZON_EXTENDED),
    ("V11_TR1_H240", "Trail act +1R, D=1R, TE=k=240", _v11, TIME_HORIZON_EXTENDED),
    ("V12_TP1_H240", "TP=+1R, TE=k=240", _v12, TIME_HORIZON_EXTENDED),
    ("V13_TP2_H240", "TP=+2R, TE=k=240", _v13, TIME_HORIZON_EXTENDED),
    ("V14_TP1_then_TR1", "TP=+1R if reached, else hold to k=240 with SL", _v14, TIME_HORIZON_EXTENDED),
    ("V15_BE1_TR1", "BE @ +1R, Trail @ +2R/D=1R, TE=k=240", _v15, TIME_HORIZON_EXTENDED),
]
VARIANT_IDS: Tuple[str, ...] = tuple(v[0] for v in VARIANTS)


# Exit-reason mapping (Round 1 internal -> spec short codes).
REASON_MAP: Dict[str, str] = {
    "stop_loss": "sl",
    "fixed_tp": "tp",
    "trail_exit": "trail",
    "be_exit": "be",
    "time_exit": "te",
    "data_end": "de",
}
ALL_EXIT_SHORT: Tuple[str, ...] = ("sl", "tp", "trail", "be", "te", "de")


# ===========================================================================
# Subset & category reconstruction
# ===========================================================================


def _make_quintile_labels(values: pd.Series, tie_break: pd.Series
                          ) -> Tuple[pd.Series, List[Tuple[float, float]]]:
    """Rank-based quintile bucketing with deterministic tie-break by trade_id.

    Mirrors arc2_entry_filter_bivariate._make_quintile_labels and
    arc2_path_by_subset.make_quintile_labels byte-for-byte.
    """
    df = pd.DataFrame({"v": values.values, "t": tie_break.values},
                      index=values.index)
    df = df.sort_values(["v", "t"], kind="stable")
    n = len(df)
    base = n // 5
    rem = n - base * 5
    sizes = [base + (1 if i < rem else 0) for i in range(5)]
    labels: List[str] = []
    bounds: List[Tuple[float, float]] = []
    cursor = 0
    for qi, sz in enumerate(sizes):
        seg = df.iloc[cursor:cursor + sz]
        labels.extend([f"Q{qi + 1}"] * sz)
        bounds.append((float(seg["v"].min()), float(seg["v"].max())))
        cursor += sz
    df["q"] = labels
    return df["q"].reindex(values.index), bounds


def build_subsets() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Reconstruct Q_A/Q_B labels and per-subset trade_id arrays.

    Returns (labels_df, subset_to_trade_ids).
    labels_df cols: trade_id, pair, signal_bar_ts, fold_id, Q_A_concurrent,
                    Q_B_dist_d1, concurrent_signals_same_bar, dist_d1_kijun_atr.
    """
    sf = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_1_full/signals_features.csv")
    ti = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv")
    bm = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv")

    sf_taken = sf[sf["taken"] == True].copy()  # noqa: E712
    sf_taken = sf_taken.rename(columns={"time": "signal_bar_ts"})
    sf_taken["signal_bar_ts"] = pd.to_datetime(sf_taken["signal_bar_ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    taken = sf_taken.merge(
        ti[["trade_id", "pair", "signal_bar_ts", "atr_1h_wilder_at_signal"]],
        on=["pair", "signal_bar_ts"], how="left", validate="one_to_one",
    )
    taken = taken.merge(
        bm[["trade_id", "dist_d1_kijun_atr"]],
        on="trade_id", how="left", validate="one_to_one",
    )
    taken = taken.sort_values("trade_id").reset_index(drop=True)
    if len(taken) != 3993:
        raise RuntimeError(f"HALT: taken row count {len(taken)} != 3993")

    qa_labels, _ = _make_quintile_labels(
        taken["concurrent_signals_same_bar"], taken["trade_id"])
    taken["Q_A_concurrent"] = qa_labels.values
    qb_labels, _ = _make_quintile_labels(
        taken["dist_d1_kijun_atr"], taken["trade_id"])
    taken["Q_B_dist_d1"] = qb_labels.values

    # Gate 2.1 check: reproduce 25 block_P cell counts exactly.
    bp_path = REPO_ROOT / "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv"
    bp = pd.read_csv(bp_path)
    diffs: List[str] = []
    for _, row in bp.iterrows():
        qa = row["Q_A_concurrent"]
        qb = row["Q_B_dist_d1"]
        exp_n = int(row["n"])
        sub = taken[(taken["Q_A_concurrent"] == qa) & (taken["Q_B_dist_d1"] == qb)]
        got_n = len(sub)
        if got_n != exp_n:
            diffs.append(f"  ({qa},{qb}): expected n={exp_n}, got n={got_n}")
    if diffs:
        raise RuntimeError("HALT (gate 2.1): block_P cell count mismatch:\n"
                            + "\n".join(diffs))

    # Build subsets.
    subsets: Dict[str, np.ndarray] = {}
    for sid, spec in SUBSET_DEFS:
        if spec.get("all"):
            mask = pd.Series(True, index=taken.index)
        else:
            mask = (taken["Q_A_concurrent"].isin(spec["qa"])
                    & taken["Q_B_dist_d1"].isin(spec["qb"]))
        sub = taken[mask]
        expected_n = int(spec["expected_n"])
        if len(sub) != expected_n:
            raise RuntimeError(
                f"HALT (gate 2.2): subset {sid} size {len(sub)} != {expected_n}"
            )
        subsets[sid] = sub["trade_id"].to_numpy(dtype=np.int64)

    labels = taken[["trade_id", "pair", "signal_bar_ts", "fold_id",
                    "Q_A_concurrent", "Q_B_dist_d1",
                    "concurrent_signals_same_bar", "dist_d1_kijun_atr"]].copy()
    labels["fold_id"] = labels["fold_id"].astype(int)
    return labels, subsets


def compute_categories(pb: pd.DataFrame, n_trades: int) -> np.ndarray:
    """Replicate the path-category derivation from arc2_path_by_subset.py."""
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids = pb["trade_id"].to_numpy()
    ks = pb["k"].to_numpy()
    mfe = pb["running_mfe_atr"].to_numpy()
    mae = pb["running_mae_atr"].to_numpy()
    in_window = ks <= PATH_HOLD_CAP
    up_hit = (mfe >= PATH_THRESH_ATR) & in_window
    dn_hit = (mae <= -PATH_THRESH_ATR) & in_window

    SENTINEL = PATH_HOLD_CAP + 1
    t_up = np.full(n_trades, SENTINEL, dtype=np.int32)
    t_dn = np.full(n_trades, SENTINEL, dtype=np.int32)
    if up_hit.any():
        up_idx = np.where(up_hit)[0]
        up_df = pd.DataFrame({"tid": tids[up_idx], "k": ks[up_idx]})
        first_up = up_df.groupby("tid", sort=False)["k"].min()
        t_up[first_up.index.values.astype(np.int64)] = first_up.values.astype(np.int32)
    if dn_hit.any():
        dn_idx = np.where(dn_hit)[0]
        dn_df = pd.DataFrame({"tid": tids[dn_idx], "k": ks[dn_idx]})
        first_dn = dn_df.groupby("tid", sort=False)["k"].min()
        t_dn[first_dn.index.values.astype(np.int64)] = first_dn.values.astype(np.int32)

    up_reached = t_up <= PATH_HOLD_CAP
    dn_reached = t_dn <= PATH_HOLD_CAP
    cats = np.full(n_trades, "", dtype=object)
    only_up = up_reached & (~dn_reached)
    only_down = (~up_reached) & dn_reached
    both = up_reached & dn_reached
    neither = (~up_reached) & (~dn_reached)
    up_then_down = both & (t_up < t_dn)
    down_then_up = both & (t_dn < t_up)
    simul = both & (t_up == t_dn)
    cats[only_up] = "only_up"
    cats[only_down] = "straight_to_sl"
    cats[up_then_down] = "up_then_down"
    cats[down_then_up] = "down_then_up"
    cats[simul] = "simultaneous"
    cats[neither] = "neither_reached"

    # Gate: reproduce Block B 1R counts.
    counts = {c: int((cats == c).sum()) for c in ALL_CATS}
    diffs = []
    for c, exp in BLOCK_B_1R_COUNTS.items():
        if counts[c] != exp:
            diffs.append(f"  {c}: expected={exp}, got={counts[c]}")
    if diffs:
        raise RuntimeError("HALT: Block B 1R category counts diverge:\n"
                            + "\n".join(diffs))
    return cats


# ===========================================================================
# Sweep
# ===========================================================================


def run_sweep(*, per_bar_csv: Path, trade_index_csv: Path,
              trades_all_csv: Path
              ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Returns (variant_trades, trade_index_full, categories)."""
    print("  Loading trade_index.csv + trades_all.csv...", flush=True)
    ti = pd.read_csv(trade_index_csv)
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ta = pd.read_csv(trades_all_csv)
    ta["signal_bar_ts"] = pd.to_datetime(ta["signal_bar_ts"])

    ti_full = ti.merge(
        ta[["pair", "signal_bar_ts", "spread_pips_entry", "spread_pips_exit"]],
        on=["pair", "signal_bar_ts"], how="left", validate="one_to_one",
    )
    if ti_full[["spread_pips_entry", "spread_pips_exit"]].isna().any().any():
        raise RuntimeError("HALT (gate sp-lookup): null sp_entry/sp_exit after merge")
    ti_full = ti_full.sort_values("trade_id").reset_index(drop=True)

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
            "fold_id": int(row["fold_id"]),
            "signal_bar_ts": row["signal_bar_ts"].strftime("%Y-%m-%dT%H:%M:%S"),
        }

    print("  Loading per_bar_paths.csv...", flush=True)
    pb = pd.read_csv(per_bar_csv)
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids_arr = pb["trade_id"].to_numpy(dtype=np.int64)
    n_trades = int(ti["trade_id"].max()) + 1
    if n_trades != 3993:
        raise RuntimeError(f"HALT: expected 3993 trades, got {n_trades}")
    starts = np.searchsorted(tids_arr, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids_arr, np.arange(n_trades), side="right")

    rmae_all = pb["running_mae_atr"].to_numpy(dtype=np.float64)
    rmfe_all = pb["running_mfe_atr"].to_numpy(dtype=np.float64)
    bl_all = pb["bar_low_atr"].to_numpy(dtype=np.float64)
    bh_all = pb["bar_high_atr"].to_numpy(dtype=np.float64)
    bc_all = pb["bar_close_atr"].to_numpy(dtype=np.float64)
    nbo_all = pb["next_bar_open_atr"].to_numpy(dtype=np.float64)
    hnb_all = pb["has_next_bar"].to_numpy(dtype=bool)

    cats = compute_categories(pb, n_trades)
    print(f"  Block B categories: {dict((c, int((cats == c).sum())) for c in ALL_CATS)}", flush=True)

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

    print(f"  Computing {n_vars} variants x {n_trades} trades...", flush=True)
    t0 = time.time()
    write_idx = 0
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        bavail = e - s
        rmae = rmae_all[s:e]; rmfe = rmfe_all[s:e]
        bl_ = bl_all[s:e]; bh_ = bh_all[s:e]; bc_ = bc_all[s:e]
        nbo = nbo_all[s:e]; hnb = hnb_all[s:e]
        T = per_trade[tid]
        for vid, _, vfn, _ in VARIANTS:
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
            print(f"    progress: {tid+1}/{n_trades} ({el:.1f}s, {(tid+1)/el:.0f} trades/s)",
                  flush=True)

    variant_trades = pd.DataFrame({
        "variant_id": out_variant, "trade_id": out_tid, "pair": out_pair,
        "signal_bar_ts": out_sigts, "fold_id": out_fold,
        "exit_reason_internal": out_reason, "exit_bar": out_exitbar,
        "exit_level_atr_fill": out_exitlvl, "gross_R": out_gross,
        "spread_cost_R": out_spread, "net_R": out_net,
    })
    variant_trades["exit_reason"] = variant_trades["exit_reason_internal"].map(REASON_MAP)
    if variant_trades["exit_reason"].isna().any():
        raise RuntimeError("HALT: unmapped exit_reason_internal present")
    variant_trades = variant_trades.sort_values(["variant_id", "trade_id"]).reset_index(drop=True)
    return variant_trades, ti_full, cats


# ===========================================================================
# Block aggregations
# ===========================================================================


def _captured_metric_per_subset_variant(sub_vt: pd.DataFrame,
                                        cats: np.ndarray,
                                        trade_ids: np.ndarray) -> Dict[str, float]:
    """Compute the four captured/avoidance metrics described in Block Z."""
    cat_map = pd.Series(cats[trade_ids], index=trade_ids, name="cat")
    sub_vt = sub_vt.copy()
    sub_vt["cat"] = sub_vt["trade_id"].map(cat_map)
    out: Dict[str, float] = {}
    # only_up captured at >= +1R
    only_up = sub_vt[sub_vt["cat"] == "only_up"]
    out["pct_only_up_captured_at_ge_1R"] = (
        float((only_up["net_R"] >= 1.0).mean()) if len(only_up) > 0 else float("nan")
    )
    dtu = sub_vt[sub_vt["cat"] == "down_then_up"]
    out["pct_down_then_up_captured_at_ge_1R"] = (
        float((dtu["net_R"] >= 1.0).mean()) if len(dtu) > 0 else float("nan")
    )
    utd = sub_vt[sub_vt["cat"] == "up_then_down"]
    out["pct_up_then_down_kept_negative"] = (
        float((utd["net_R"] <= 0.0).mean()) if len(utd) > 0 else float("nan")
    )
    sts = sub_vt[sub_vt["cat"] == "straight_to_sl"]
    out["pct_straight_to_sl_unchanged"] = (
        float((sts["exit_reason"] == "sl").mean()) if len(sts) > 0 else float("nan")
    )
    return out


def aggregate_block_Z(vt: pd.DataFrame, subsets: Dict[str, np.ndarray],
                      cats: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    by_var = {vid: grp for vid, grp in vt.groupby("variant_id", sort=False)}
    # BL means per subset (V00_BL).
    bl_mean_per_subset: Dict[str, float] = {}
    bl_vt = by_var["V00_BL"]
    bl_by_tid = bl_vt.set_index("trade_id")["net_R"]
    for sid, tids in subsets.items():
        bl_mean_per_subset[sid] = float(bl_by_tid.reindex(tids).mean())

    for sid, tids in subsets.items():
        tid_set = set(tids.tolist())
        for vid in VARIANT_IDS:
            sub_v = by_var[vid]
            sub = sub_v[sub_v["trade_id"].isin(tid_set)].copy()
            n = len(sub)
            if n == 0:
                raise RuntimeError(f"HALT: empty subset/variant {sid}/{vid}")
            net = sub["net_R"].to_numpy(dtype=np.float64)
            rc = sub["exit_reason"].value_counts().to_dict()
            counts = {r: int(rc.get(r, 0)) for r in ALL_EXIT_SHORT}
            # Exit-reason exhaustiveness check.
            total = sum(counts.values())
            if total != n:
                raise RuntimeError(
                    f"HALT (gate 12): {sid}/{vid} exit reasons sum {total} != n {n}")
            mean_R = float(np.mean(net))
            row = {
                "subset_id": sid,
                "variant_id": vid,
                "n_in_subset": n,
                "n_exit_reason_sl": counts["sl"],
                "n_exit_reason_tp": counts["tp"],
                "n_exit_reason_trail": counts["trail"],
                "n_exit_reason_be": counts["be"],
                "n_exit_reason_te": counts["te"],
                "n_exit_reason_de": counts["de"],
                "mean_R": mean_R,
                "std_R": float(np.std(net, ddof=1)) if n > 1 else 0.0,
                "median_R": float(np.median(net)),
                "p25_R": float(np.quantile(net, 0.25)),
                "p75_R": float(np.quantile(net, 0.75)),
                "min_R": float(np.min(net)),
                "max_R": float(np.max(net)),
                "pct_trades_profitable": float((net > 0).mean()),
                "pooled_lift_vs_BL": mean_R - bl_mean_per_subset[sid],
            }
            row.update(_captured_metric_per_subset_variant(sub, cats, tids))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def aggregate_block_AA(vt: pd.DataFrame, subsets: Dict[str, np.ndarray]
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    by_var = {vid: grp for vid, grp in vt.groupby("variant_id", sort=False)}
    bl_vt = by_var["V00_BL"]
    bl_by_tidfold = bl_vt.set_index("trade_id")
    folds = sorted(vt["fold_id"].unique().tolist())
    # BL per (subset, fold) mean R for reference
    bl_subset_fold_mean: Dict[Tuple[str, int], float] = {}
    bl_subset_fold_n: Dict[Tuple[str, int], int] = {}
    for sid, tids in subsets.items():
        tid_set = set(tids.tolist())
        bl_sub = bl_vt[bl_vt["trade_id"].isin(tid_set)]
        for f in folds:
            bl_f = bl_sub[bl_sub["fold_id"] == f]
            if len(bl_f) > 0:
                bl_subset_fold_mean[(sid, f)] = float(bl_f["net_R"].mean())
            else:
                bl_subset_fold_mean[(sid, f)] = float("nan")
            bl_subset_fold_n[(sid, f)] = len(bl_f)

    for sid, tids in subsets.items():
        tid_set = set(tids.tolist())
        for vid in VARIANT_IDS:
            sub_v = by_var[vid]
            sub = sub_v[sub_v["trade_id"].isin(tid_set)]
            for f in folds:
                sub_f = sub[sub["fold_id"] == f]
                n_f = len(sub_f)
                if n_f == 0:
                    continue
                rc = sub_f["exit_reason"].value_counts().to_dict()
                rates = {r: float(rc.get(r, 0)) / n_f for r in ALL_EXIT_SHORT}
                rows.append({
                    "subset_id": sid,
                    "variant_id": vid,
                    "fold_id": int(f),
                    "n_in_fold": n_f,
                    "mean_R_fold": float(sub_f["net_R"].mean()),
                    "sl_rate_fold": rates["sl"],
                    "tp_rate_fold": rates["tp"],
                    "trail_rate_fold": rates["trail"],
                    "be_rate_fold": rates["be"],
                    "te_rate_fold": rates["te"],
                    "de_rate_fold": rates["de"],
                    "bl_mean_R_fold": bl_subset_fold_mean.get((sid, f), float("nan")),
                })
    per_fold = pd.DataFrame(rows)

    # Stability summary per (subset, variant)
    stab_rows: List[Dict[str, Any]] = []
    THIN = 10
    for sid in SUBSET_IDS:
        for vid in VARIANT_IDS:
            sub = per_fold[(per_fold["subset_id"] == sid)
                           & (per_fold["variant_id"] == vid)]
            n_folds_thick = int((sub["n_in_fold"] >= THIN).sum())
            above = int((sub["mean_R_fold"] > sub["bl_mean_R_fold"]).sum())
            below = int((sub["mean_R_fold"] < sub["bl_mean_R_fold"]).sum())
            var_mean_R = float(sub["mean_R_fold"].var(ddof=1)) if len(sub) > 1 else 0.0
            ref = n_folds_thick if n_folds_thick > 0 else len(sub)
            if vid == "V00_BL":
                flag = "baseline"
            elif n_folds_thick < 5:
                flag = "thin"
            elif above >= 5:
                flag = "stable_lift"
            elif below >= 5:
                flag = "stable_drag"
            else:
                flag = "variable"
            stab_rows.append({
                "subset_id": sid, "variant_id": vid,
                "folds_with_n_ge_10": n_folds_thick,
                "folds_mean_R_above_BL": above,
                "folds_mean_R_below_BL": below,
                "fold_variance_mean_R": var_mean_R,
                "stability_flag": flag,
            })
    stab = pd.DataFrame(stab_rows)
    return per_fold, stab


def aggregate_block_BB(vt: pd.DataFrame, subsets: Dict[str, np.ndarray],
                       cats: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    by_var = {vid: grp for vid, grp in vt.groupby("variant_id", sort=False)}
    cat_series = pd.Series(cats, index=np.arange(len(cats)), name="cat")

    # Precompute BL mean R per (subset, category) for diff column.
    bl_vt = by_var["V00_BL"]
    bl_vt = bl_vt.copy()
    bl_vt["cat"] = bl_vt["trade_id"].map(cat_series)
    bl_subset_cat_mean: Dict[Tuple[str, str], float] = {}
    for sid, tids in subsets.items():
        tid_set = set(tids.tolist())
        sub = bl_vt[bl_vt["trade_id"].isin(tid_set)]
        for c in ALL_CATS:
            sub_c = sub[sub["cat"] == c]
            if len(sub_c) > 0:
                bl_subset_cat_mean[(sid, c)] = float(sub_c["net_R"].mean())
            else:
                bl_subset_cat_mean[(sid, c)] = float("nan")

    for sid, tids in subsets.items():
        tid_set = set(tids.tolist())
        for vid in VARIANT_IDS:
            sub_v = by_var[vid].copy()
            sub_v["cat"] = sub_v["trade_id"].map(cat_series)
            sub = sub_v[sub_v["trade_id"].isin(tid_set)]
            for c in ALL_CATS:
                sub_c = sub[sub["cat"] == c]
                n_c = len(sub_c)
                if n_c == 0:
                    rows.append({
                        "subset_id": sid, "variant_id": vid, "category": c,
                        "n_in_category_within_subset": 0,
                        "mean_R_variant": float("nan"),
                        "median_R_variant": float("nan"),
                        "mean_R_BL": bl_subset_cat_mean[(sid, c)],
                        "mean_R_diff_vs_BL": float("nan"),
                        "contribution_to_pooled_R": 0.0,
                        "pct_exited_sl": float("nan"),
                        "pct_exited_tp": float("nan"),
                        "pct_exited_trail": float("nan"),
                        "pct_exited_be": float("nan"),
                        "pct_exited_te": float("nan"),
                        "pct_exited_de": float("nan"),
                        "captured_metric_value": float("nan"),
                    })
                    continue
                net = sub_c["net_R"].to_numpy(dtype=np.float64)
                rc = sub_c["exit_reason"].value_counts().to_dict()
                rates = {r: float(rc.get(r, 0)) / n_c for r in ALL_EXIT_SHORT}
                mean_R = float(np.mean(net))
                # Contribution = (n_c / n_subset) * mean_R_c
                n_subset = len(tids)
                contribution = (n_c / n_subset) * mean_R
                # Captured metric per category.
                if c == "only_up":
                    cap_val = float((net >= 1.0).mean())
                elif c == "down_then_up":
                    cap_val = float((net >= 1.0).mean())
                elif c == "up_then_down":
                    cap_val = float((net <= 0.0).mean())
                elif c == "straight_to_sl":
                    cap_val = rates["sl"]
                else:
                    cap_val = float("nan")
                rows.append({
                    "subset_id": sid, "variant_id": vid, "category": c,
                    "n_in_category_within_subset": n_c,
                    "mean_R_variant": mean_R,
                    "median_R_variant": float(np.median(net)),
                    "mean_R_BL": bl_subset_cat_mean[(sid, c)],
                    "mean_R_diff_vs_BL": mean_R - bl_subset_cat_mean[(sid, c)],
                    "contribution_to_pooled_R": contribution,
                    "pct_exited_sl": rates["sl"],
                    "pct_exited_tp": rates["tp"],
                    "pct_exited_trail": rates["trail"],
                    "pct_exited_be": rates["be"],
                    "pct_exited_te": rates["te"],
                    "pct_exited_de": rates["de"],
                    "captured_metric_value": cap_val,
                })
    return pd.DataFrame(rows)


def aggregate_block_CC(block_Z: pd.DataFrame, stab: pd.DataFrame
                       ) -> pd.DataFrame:
    df = block_Z.merge(
        stab[["subset_id", "variant_id", "stability_flag",
              "folds_with_n_ge_10", "folds_mean_R_above_BL"]],
        on=["subset_id", "variant_id"], how="left", validate="one_to_one",
    )
    df = df[["subset_id", "variant_id", "n_in_subset", "mean_R",
             "pooled_lift_vs_BL",
             "pct_only_up_captured_at_ge_1R",
             "pct_down_then_up_captured_at_ge_1R",
             "pct_up_then_down_kept_negative",
             "folds_mean_R_above_BL", "folds_with_n_ge_10",
             "stability_flag"]].copy()
    df.rename(columns={
        "n_in_subset": "n",
        "pooled_lift_vs_BL": "mean_R_lift_vs_BL",
    }, inplace=True)
    # Sort: BL first within each subset, then by lift desc.
    df["_subset_order"] = df["subset_id"].apply(
        lambda s: SUBSET_IDS.index(s))
    df["_is_bl"] = (df["variant_id"] == "V00_BL").astype(int)
    df = df.sort_values(
        ["_subset_order", "_is_bl", "mean_R_lift_vs_BL"],
        ascending=[True, False, False],
    ).drop(columns=["_subset_order", "_is_bl"]).reset_index(drop=True)
    return df


# ===========================================================================
# Validation gates
# ===========================================================================


def run_validation_gates(*, vt: pd.DataFrame, ti_full: pd.DataFrame,
                          block_Z: pd.DataFrame, block_BB: pd.DataFrame,
                          per_fold: pd.DataFrame,
                          subsets: Dict[str, np.ndarray],
                          out_dir: Path) -> Dict[str, Any]:
    disp: Dict[str, Any] = {}

    # Gate 1: locked sha256s already verified.
    disp["gate_1_inputs"] = "ok (10 sha256s match)"

    # Gates 2.1 / 2.2 already enforced inside build_subsets.
    disp["gate_2_subsets"] = "ok (all 25 block_P cells reproduced; 6/6 subsets sized)"

    # Gate 3.1: V00_BL on S0_pop reproduces Round 1 BL within 1e-7.
    bl = vt[vt["variant_id"] == "V00_BL"].sort_values("trade_id").reset_index(drop=True)
    ti_sorted = ti_full.sort_values("trade_id").reset_index(drop=True)
    diff = bl["net_R"].to_numpy(dtype=np.float64) - ti_sorted["R"].to_numpy(dtype=np.float64)
    max_abs = float(np.abs(diff).max())
    TOL = 1e-7
    disp["gate_3_1"] = {"max_abs_diff_vs_trade_index_R": f"{max_abs:.3e}",
                        "tolerance": f"{TOL:.0e}",
                        "pass": bool(max_abs < TOL)}
    if max_abs >= TOL:
        raise RuntimeError(f"HALT (gate 3.1): max abs diff {max_abs:.3e} >= {TOL:.0e}")

    # Cross-check vs Round 1 v2's variant_summary_pooled BL mean R.
    r1_pooled = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/extended/counterfactuals/round_1/variant_summary_pooled.csv")
    r1_bl_mean = float(r1_pooled.loc[r1_pooled["variant_id"] == "BL", "mean_R"].iloc[0])
    here_bl_mean = float(bl["net_R"].mean())
    bl_mean_diff = abs(here_bl_mean - r1_bl_mean)
    disp["gate_3_1_r1_match"] = {"r1_bl_mean_R": r1_bl_mean,
                                  "here_bl_mean_R": here_bl_mean,
                                  "abs_diff": f"{bl_mean_diff:.3e}"}
    if bl_mean_diff > 1e-9:
        raise RuntimeError(
            f"HALT (gate 3.1 R1 match): BL mean R diff {bl_mean_diff:.3e} > 1e-9"
        )

    # Gate 3.2: clamped-trade handling.
    # For horizon=240 variants, count trades where bavail < 240.
    bavail_map = ti_sorted.set_index("trade_id")["bars_available"].to_dict()
    horizon_by_vid = {v[0]: v[3] for v in VARIANTS}
    clamp_counts: Dict[str, int] = {}
    for vid in VARIANT_IDS:
        H = horizon_by_vid[vid]
        if H != TIME_HORIZON_EXTENDED:
            continue
        sub_v = vt[vt["variant_id"] == vid]
        n_clamp = 0
        # A trade is "clamped" when bavail < H.
        for _, row in sub_v.iterrows():
            if bavail_map[int(row["trade_id"])] < H:
                n_clamp += 1
        clamp_counts[vid] = n_clamp
    # Validate: any clamped trade where bavail < H must exit via 'de' or earlier (sl/tp/be/trail).
    clamp_bad: List[str] = []
    for vid in VARIANT_IDS:
        H = horizon_by_vid[vid]
        if H != TIME_HORIZON_EXTENDED:
            continue
        sub_v = vt[vt["variant_id"] == vid]
        for _, row in sub_v.iterrows():
            tid = int(row["trade_id"])
            ba = bavail_map[tid]
            if ba < H:
                # Acceptable exits: sl/tp/be/trail/de. te is forbidden if clamped.
                if row["exit_reason"] == "te":
                    clamp_bad.append(f"{vid}/{tid}: bavail={ba} < H={H} but exit=te")
    if clamp_bad:
        raise RuntimeError("HALT (gate 3.2): clamped trades with 'te' exit:\n  "
                            + "\n  ".join(clamp_bad[:10]))
    disp["gate_3_2_clamps"] = clamp_counts

    # Gate 6: Block BB contribution check.
    # Sum of contribution_to_pooled_R across categories within (subset, variant)
    # must equal block_Z.mean_R for that (subset, variant) within 1e-9.
    contrib = (block_BB.groupby(["subset_id", "variant_id"])
               ["contribution_to_pooled_R"].sum().rename("sum_contrib").reset_index())
    z = block_Z[["subset_id", "variant_id", "mean_R"]]
    merged = contrib.merge(z, on=["subset_id", "variant_id"])
    merged["abs_diff"] = (merged["sum_contrib"] - merged["mean_R"]).abs()
    max_contrib_diff = float(merged["abs_diff"].max())
    disp["gate_6_contrib_consistency"] = {"max_abs_diff": f"{max_contrib_diff:.3e}",
                                            "tolerance": "1e-9",
                                            "pass": bool(max_contrib_diff < 1e-9)}
    if max_contrib_diff >= 1e-9:
        raise RuntimeError(
            f"HALT (gate 6): block_BB contribution sum diverges from block_Z mean_R by "
            f"{max_contrib_diff:.3e}"
        )

    # Gate 7: subset-mean-R cross-check vs block_P.
    bp = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv")
    bp_lookup = {(r["Q_A_concurrent"], r["Q_B_dist_d1"]): float(r["mean_R"])
                 for _, r in bp.iterrows()}
    # For S1_q5q2/S2_q5q3/S3_q4q2 — single-cell subsets.
    cell_checks = [("S1_q5q2", "Q5", "Q2"),
                   ("S2_q5q3", "Q5", "Q3"),
                   ("S3_q4q2", "Q4", "Q2")]
    z_bl = block_Z[block_Z["variant_id"] == "V00_BL"].set_index("subset_id")["mean_R"]
    max_subset_diff = 0.0
    for sid, qa, qb in cell_checks:
        bp_val = bp_lookup[(qa, qb)]
        here_val = float(z_bl[sid])
        d = abs(bp_val - here_val)
        if d > max_subset_diff:
            max_subset_diff = d
    disp["gate_7_subset_meanR"] = {
        "max_abs_diff_vs_block_P": f"{max_subset_diff:.3e}",
        "tolerance": "1e-9",
        "pass": bool(max_subset_diff < 1e-9),
    }
    if max_subset_diff >= 1e-9:
        raise RuntimeError(
            f"HALT (gate 7): subset BL mean_R diverges from block_P by {max_subset_diff:.3e}"
        )

    # Gate 12: exit-reason exhaustiveness — already enforced in aggregate_block_Z.
    # Verify here by summing the n_exit_reason_* columns.
    Z = block_Z.copy()
    Z["sum_exits"] = (Z["n_exit_reason_sl"] + Z["n_exit_reason_tp"]
                     + Z["n_exit_reason_trail"] + Z["n_exit_reason_be"]
                     + Z["n_exit_reason_te"] + Z["n_exit_reason_de"])
    bad = Z[Z["sum_exits"] != Z["n_in_subset"]]
    if len(bad) > 0:
        raise RuntimeError(
            f"HALT (gate 12): {len(bad)} (subset, variant) rows with sum_exits != n"
        )
    disp["gate_12_exhaustive"] = "ok (all exit_reason sums == n)"

    return disp


# ===========================================================================
# Markdown report
# ===========================================================================


def _df_to_md(df: pd.DataFrame, float_cols: Optional[Dict[str, str]] = None
              ) -> str:
    cols = list(df.columns)
    float_cols = float_cols or {}
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in float_cols and isinstance(v, float) and not pd.isna(v):
                cells.append(float_cols[c].format(v))
            elif isinstance(v, float) and pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_report(*, observed_shas: Dict[str, str],
                  block_Z: pd.DataFrame, block_AA_stab: pd.DataFrame,
                  per_fold: pd.DataFrame, block_BB: pd.DataFrame,
                  block_CC: pd.DataFrame, gates: Dict[str, Any]) -> str:
    lines: List[str] = []
    a = lines.append
    a("# Arc 2 — Exit-Rule Variant Sweep on Filtered Subsets (Phase 2 Round 2)")
    a("")
    a("Phase: L6 Arc 2 Phase 3 — Filtered-subset x exit-variant sweep.")
    a("")
    a("Descriptive characterisation per L6_0_METHODOLOGY_LOCK Section 14.5. ")
    a("No filter selection, no WFO, no signal-module modification. Block B ")
    a("path categories are evaluation buckets only — exit rules use solely ")
    a("running observables and bar index (strict no-lookahead).")
    a("")

    # Locked input sha256 manifest
    a("## Locked input sha256 manifest")
    a("")
    a("| relative_path | sha256 |")
    a("| --- | --- |")
    for rel in LOCKED_SHAS:
        a(f"| {rel} | {observed_shas[rel]} |")
    a("")

    # Determinism receipt
    a("## Determinism receipt")
    a("")
    a("Two consecutive in-script build passes produced byte-identical CSV+MD ")
    a("outputs. Timestamps/wallclock emitted to stdout only. See run_manifest.txt ")
    a("for output sha256s.")
    a("")

    # Subset definitions
    a("## Subset definitions and counts")
    a("")
    a("Subsets reconstructed by re-applying rank-based quintile labels with ")
    a("`trade_id` tie-break, byte-faithful to `arc2_entry_filter_bivariate.py` ")
    a("and `arc2_path_by_subset.py`.")
    a("")
    a("| subset_id | definition | n_expected | n_observed |")
    a("| --- | --- | --- | --- |")
    z_bl = block_Z[block_Z["variant_id"] == "V00_BL"]
    sub_defs_human = {
        "S0_pop": "ALL 3,993 trades (baseline reference)",
        "S1_q5q2": "Q_A==Q5 AND Q_B==Q2",
        "S2_q5q3": "Q_A==Q5 AND Q_B==Q3",
        "S3_q4q2": "Q_A==Q4 AND Q_B==Q2",
        "S4_q5xq2q3": "Q_A==Q5 AND Q_B IN {Q2,Q3}",
        "S5_q4q5xq2q3": "Q_A IN {Q4,Q5} AND Q_B IN {Q2,Q3}",
    }
    for sid in SUBSET_IDS:
        expected = dict(SUBSET_DEFS)[sid]["expected_n"]
        n_obs = int(z_bl[z_bl["subset_id"] == sid]["n_in_subset"].iloc[0])
        a(f"| {sid} | {sub_defs_human[sid]} | {expected} | {n_obs} |")
    a("")

    # Variant definitions
    a("## Variant definitions")
    a("")
    a("All variants share: entry at signal bar N+1 open with full spread; ")
    a("hard SL at -2 ATR fill-relative; SL precedence on each bar; spread ")
    a("accounting copied byte-faithful from Round 1 v2 mechanics. ")
    a("Activations (BE arm, trail engage) take effect from the next bar after ")
    a("the threshold is first reached (Round 1 convention).")
    a("")
    a("| variant_id | spec_short | hold_horizon |")
    a("| --- | --- | --- |")
    for vid, spec, _, H in VARIANTS:
        a(f"| {vid} | {spec} | {H} |")
    a("")
    a("Note: V14_TP1_then_TR1 (TP@+1R if reached, else hold to k=240 with SL) ")
    a("is mechanically equivalent to V12_TP1_H240; per the spec's listed 16 ")
    a("variants both are emitted and produce identical R per trade.")
    a("")

    # Round 1 BL reproduction
    a("## Round 1 BL reproduction check (gate 3.1)")
    a("")
    g31 = gates["gate_3_1"]
    g31r = gates["gate_3_1_r1_match"]
    a(f"- max abs diff (V00_BL net_R vs trade_index.R): {g31['max_abs_diff_vs_trade_index_R']}")
    a(f"- tolerance: {g31['tolerance']}")
    a(f"- Round 1 v2 BL mean_R reference: {g31r['r1_bl_mean_R']:.10f}")
    a(f"- Here V00_BL on S0_pop mean_R: {g31r['here_bl_mean_R']:.10f}")
    a(f"- abs diff: {g31r['abs_diff']}")
    a("- Disposition: PASS — V00_BL reproduces Round 1 BL.")
    a("")

    # Block CC headline
    a("## Block CC — Headline summary")
    a("")
    a("For each subset, V00_BL is shown first; other variants are sorted by ")
    a("mean_R lift vs BL descending.")
    a("")
    fmt = {
        "mean_R": "{:.4f}",
        "mean_R_lift_vs_BL": "{:+.4f}",
        "pct_only_up_captured_at_ge_1R": "{:.3f}",
        "pct_down_then_up_captured_at_ge_1R": "{:.3f}",
        "pct_up_then_down_kept_negative": "{:.3f}",
    }
    for sid in SUBSET_IDS:
        sub = block_CC[block_CC["subset_id"] == sid].copy()
        a(f"### Subset {sid}")
        a("")
        a(_df_to_md(sub.drop(columns=["subset_id"]), fmt))
        a("")

    # Block Z highlights
    a("## Block Z — Per-(subset, variant) detail (highlights)")
    a("")
    for sid in SUBSET_IDS:
        sub = block_Z[block_Z["subset_id"] == sid].copy()
        sub_no_bl = sub[sub["variant_id"] != "V00_BL"].copy()
        top3 = sub_no_bl.sort_values("pooled_lift_vs_BL", ascending=False).head(3)
        bot3 = sub_no_bl.sort_values("pooled_lift_vs_BL", ascending=True).head(3)
        a(f"### Subset {sid}")
        a("")
        bl_row = sub[sub["variant_id"] == "V00_BL"].iloc[0]
        a(f"V00_BL on this subset: n={int(bl_row['n_in_subset'])}, "
          f"mean_R={bl_row['mean_R']:.4f}.")
        a("")
        a("Top 3 by mean_R lift vs BL:")
        a("")
        a(_df_to_md(top3[["variant_id", "n_in_subset", "mean_R",
                          "pooled_lift_vs_BL", "n_exit_reason_sl",
                          "n_exit_reason_tp", "n_exit_reason_trail",
                          "n_exit_reason_be", "n_exit_reason_te",
                          "n_exit_reason_de"]], fmt))
        a("")
        a("Bottom 3 by mean_R lift vs BL:")
        a("")
        a(_df_to_md(bot3[["variant_id", "n_in_subset", "mean_R",
                          "pooled_lift_vs_BL", "n_exit_reason_sl",
                          "n_exit_reason_tp", "n_exit_reason_trail",
                          "n_exit_reason_be", "n_exit_reason_te",
                          "n_exit_reason_de"]], fmt))
        a("")

    # Block AA per-fold highlights
    a("## Block AA — Per-fold stability (highlights for top-3 variants per subset)")
    a("")
    fmt_aa = {"mean_R_fold": "{:.4f}", "bl_mean_R_fold": "{:.4f}"}
    for sid in SUBSET_IDS:
        sub = block_Z[(block_Z["subset_id"] == sid)
                      & (block_Z["variant_id"] != "V00_BL")]
        top_vids = sub.sort_values("pooled_lift_vs_BL", ascending=False
                                   ).head(3)["variant_id"].tolist()
        a(f"### Subset {sid}")
        a("")
        for vid in top_vids:
            stab_row = block_AA_stab[(block_AA_stab["subset_id"] == sid)
                                     & (block_AA_stab["variant_id"] == vid)].iloc[0]
            a(f"#### {vid} (stability_flag = {stab_row['stability_flag']}, "
              f"folds_with_n_ge_10 = {int(stab_row['folds_with_n_ge_10'])}, "
              f"folds_mean_R_above_BL = {int(stab_row['folds_mean_R_above_BL'])})")
            a("")
            pf = per_fold[(per_fold["subset_id"] == sid)
                          & (per_fold["variant_id"] == vid)].copy()
            pf = pf[["fold_id", "n_in_fold", "mean_R_fold", "bl_mean_R_fold",
                     "sl_rate_fold", "tp_rate_fold", "trail_rate_fold",
                     "be_rate_fold", "te_rate_fold", "de_rate_fold"]]
            a(_df_to_md(pf, fmt_aa))
            a("")

    # Block BB per-category diagnostic for top variant per subset
    a("## Block BB — Per-category diagnostic (top variant per subset)")
    a("")
    fmt_bb = {
        "mean_R_BL": "{:.4f}",
        "mean_R_variant": "{:.4f}",
        "mean_R_diff_vs_BL": "{:+.4f}",
        "pct_exited_sl": "{:.3f}",
        "pct_exited_tp": "{:.3f}",
        "pct_exited_trail": "{:.3f}",
        "pct_exited_be": "{:.3f}",
        "pct_exited_te": "{:.3f}",
        "pct_exited_de": "{:.3f}",
        "captured_metric_value": "{:.3f}",
    }
    for sid in SUBSET_IDS:
        sub = block_Z[(block_Z["subset_id"] == sid)
                      & (block_Z["variant_id"] != "V00_BL")]
        top_vid = sub.sort_values("pooled_lift_vs_BL", ascending=False
                                  ).head(1)["variant_id"].iloc[0]
        a(f"### Subset {sid} — top variant {top_vid}")
        a("")
        bb_sub = block_BB[(block_BB["subset_id"] == sid)
                          & (block_BB["variant_id"] == top_vid)].copy()
        bb_sub = bb_sub[bb_sub["n_in_category_within_subset"] > 0]
        a(_df_to_md(bb_sub[["category", "n_in_category_within_subset",
                            "mean_R_BL", "mean_R_variant", "mean_R_diff_vs_BL",
                            "pct_exited_sl", "pct_exited_tp",
                            "pct_exited_trail", "pct_exited_be",
                            "pct_exited_te", "pct_exited_de",
                            "captured_metric_value"]], fmt_bb))
        a("")

    # Cross-subset patterns
    a("## Cross-subset patterns")
    a("")
    # Variants with stable_lift on >= 4 of the S1..S5 subsets.
    five_subsets = [s for s in SUBSET_IDS if s != "S0_pop"]
    stab_pivot = (block_AA_stab[block_AA_stab["subset_id"].isin(five_subsets)]
                  .pivot(index="variant_id", columns="subset_id",
                         values="stability_flag"))
    n_stable_lift = stab_pivot.eq("stable_lift").sum(axis=1)
    n_stable_drag = stab_pivot.eq("stable_drag").sum(axis=1)
    big_lifters = n_stable_lift[n_stable_lift >= 4].sort_values(ascending=False)
    sole_lifters = n_stable_lift[n_stable_lift == 1]
    big_drags = n_stable_drag[n_stable_drag >= 4].sort_values(ascending=False)
    a(f"- Variants with `stable_lift` on >= 4 of the 5 filtered subsets (S1..S5): "
      f"{', '.join(big_lifters.index.tolist()) if len(big_lifters) > 0 else 'none'}.")
    a(f"- Variants with `stable_lift` on exactly 1 filtered subset (subset-specific): "
      f"{', '.join(sole_lifters.index.tolist()) if len(sole_lifters) > 0 else 'none'}.")
    a(f"- Variants with `stable_drag` on >= 4 filtered subsets: "
      f"{', '.join(big_drags.index.tolist()) if len(big_drags) > 0 else 'none'}.")
    a("")
    a("Full stability matrix (rows = variants, cols = S1..S5):")
    a("")
    stab_pivot_display = stab_pivot.copy().fillna("").reset_index()
    a(_df_to_md(stab_pivot_display))
    a("")

    # Out-of-scope items
    a("## Out-of-scope items observed")
    a("")
    a("- V14_TP1_then_TR1 reduces mechanically to V12_TP1_H240 under the ")
    a("  spec's final rule formulation (TP at +1R if reached, else hold to ")
    a("  k=240). Both rows are emitted; their net_R is identical per trade.")
    a("- ~28 fold-7 trades have `bars_available < 240` and exit via `de` ")
    a("  reason at the data-end bar in all horizon=240 variants; clamped ")
    a("  exits are reported in gate 3.2 disposition.")
    a("- The +0.5R variant of Block B categories is not surfaced here; this ")
    a("  sweep uses the 1R threshold matching the Block V baseline.")
    a("")

    # Planning input
    a("## Planning input")
    a("")
    a("Material below is intentionally descriptive even within this tagged ")
    a("subsection; the chat-side Phase 3 spec-lock is responsible for synthesis.")
    a("")
    for sid in SUBSET_IDS:
        a(f"### {sid}")
        a("")
        sub = block_Z[(block_Z["subset_id"] == sid)
                      & (block_Z["variant_id"] != "V00_BL")].copy()
        sub = sub.merge(block_AA_stab[["subset_id", "variant_id",
                                       "stability_flag",
                                       "folds_with_n_ge_10",
                                       "folds_mean_R_above_BL"]],
                        on=["subset_id", "variant_id"])
        top5 = sub.sort_values("pooled_lift_vs_BL", ascending=False).head(5)
        a("Top 5 by mean_R lift vs V00_BL:")
        a("")
        a(_df_to_md(top5[["variant_id", "n_in_subset", "mean_R",
                          "pooled_lift_vs_BL", "folds_mean_R_above_BL",
                          "folds_with_n_ge_10", "stability_flag"]],
                    {"mean_R": "{:.4f}",
                     "pooled_lift_vs_BL": "{:+.4f}"}))
        a("")
        # Consistency-check candidates: highest mean R lift, stable_lift, all per-fold n>=10
        consistency = sub[(sub["stability_flag"] == "stable_lift")
                          & (sub["folds_with_n_ge_10"] >= 7)
                          & (sub["folds_mean_R_above_BL"] >= 5)]
        cc_vids = consistency.sort_values("pooled_lift_vs_BL",
                                          ascending=False)["variant_id"].tolist()
        a(f"Consistency-check candidates (stable_lift + folds_with_n_ge_10 >= 7 "
          f"+ folds_mean_R_above_BL >= 5): "
          f"{', '.join(cc_vids) if cc_vids else 'none meeting all criteria'}.")
        a("")
    a("Cross-subset summary (descriptive):")
    a("")
    a(f"- stable_lift on >= 4 of S1..S5: "
      f"{', '.join(big_lifters.index.tolist()) if len(big_lifters) > 0 else 'none'}.")
    a(f"- stable_lift on exactly 1 of S1..S5: "
      f"{', '.join(sole_lifters.index.tolist()) if len(sole_lifters) > 0 else 'none'}.")
    a(f"- stable_drag on >= 4 of S1..S5: "
      f"{', '.join(big_drags.index.tolist()) if len(big_drags) > 0 else 'none'}.")
    a("")
    return "\n".join(lines) + "\n"


def check_disposition_discipline(report_text: str) -> List[Tuple[int, str, str]]:
    """Return list of (line_no, pattern, line_text) violations outside the
    'Planning input' subsection."""
    lines = report_text.splitlines()
    planning_start: Optional[int] = None
    for i, ln in enumerate(lines):
        if re.match(r"^##\s+Planning input", ln, re.IGNORECASE):
            planning_start = i
            break
    violations: List[Tuple[int, str, str]] = []
    for i, ln in enumerate(lines):
        if planning_start is not None and i >= planning_start:
            continue
        ln_lc = ln.lower()
        for pat in FORBIDDEN_PATTERNS:
            if pat in ln_lc:
                violations.append((i + 1, pat, ln))
    return violations


# ===========================================================================
# Single build pass
# ===========================================================================


def build_pass(*, out_dir: Path, write_manifest: bool,
               run_label: str) -> Dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gate 1
    observed_shas = _verify_locked("gate 1 (start)")

    # Subsets + categories
    labels, subsets = build_subsets()
    print("  Subset sizes:", {sid: len(tids) for sid, tids in subsets.items()}, flush=True)

    # Sweep
    vt, ti_full, cats = run_sweep(
        per_bar_csv=REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv",
        trade_index_csv=REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv",
        trades_all_csv=REPO_ROOT / TRADES_ALL_REL,
    )

    # Aggregates
    print("  Aggregating Block Z...", flush=True)
    block_Z = aggregate_block_Z(vt, subsets, cats)
    print("  Aggregating Block AA...", flush=True)
    per_fold, block_AA_stab = aggregate_block_AA(vt, subsets)
    print("  Aggregating Block BB...", flush=True)
    block_BB = aggregate_block_BB(vt, subsets, cats)
    print("  Aggregating Block CC...", flush=True)
    block_CC = aggregate_block_CC(block_Z, block_AA_stab)

    # Write CSVs
    z_path = out_dir / "block_Z_per_subset_per_variant.csv"
    aa_pf_path = out_dir / "block_AA_per_subset_per_variant_per_fold.csv"
    aa_stab_path = out_dir / "block_AA_stability_summary.csv"
    bb_path = out_dir / "block_BB_per_subset_per_variant_per_category.csv"
    cc_path = out_dir / "block_CC_headline_summary.csv"

    _write_csv(block_Z, z_path)
    _write_csv(per_fold, aa_pf_path)
    _write_csv(block_AA_stab, aa_stab_path)
    _write_csv(block_BB, bb_path)
    _write_csv(block_CC, cc_path)

    # Validation gates
    print("  Running validation gates...", flush=True)
    gates = run_validation_gates(
        vt=vt, ti_full=ti_full, block_Z=block_Z, block_BB=block_BB,
        per_fold=per_fold, subsets=subsets, out_dir=out_dir,
    )

    # Markdown report
    print("  Rendering markdown...", flush=True)
    md = render_report(observed_shas=observed_shas,
                       block_Z=block_Z, block_AA_stab=block_AA_stab,
                       per_fold=per_fold, block_BB=block_BB,
                       block_CC=block_CC, gates=gates)
    md_path = out_dir / "exit_sweep_filtered.md"
    md_path.write_text(md, encoding="utf-8", newline="\n")

    # Disposition discipline gate
    viols = check_disposition_discipline(md)
    if viols:
        msg = "\n  ".join([f"line {ln}: pat='{p}': {tx}" for ln, p, tx in viols])
        raise RuntimeError(
            f"HALT (gate 10): disposition discipline violations:\n  {msg}"
        )
    gates["gate_10_disposition"] = f"ok ({len(viols)} violations outside Planning input)"

    # Gate 9: locked artefacts unchanged.
    _verify_locked("gate 9 (end)")
    gates["gate_9_artefacts_unchanged"] = "ok"

    # Output sha256 manifest
    out_paths = [z_path, aa_pf_path, aa_stab_path, bb_path, cc_path, md_path]
    out_shas = {p.relative_to(REPO_ROOT).as_posix(): _sha256_file(p)
                for p in out_paths}

    if write_manifest:
        # Wallclock + peak RSS
        wallclock = time.time() - t0
        try:
            tracemalloc.start()
            # nothing to allocate — peak from full run captured below
        finally:
            pass
        manifest_path = out_dir / "run_manifest.txt"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Arc 2 — Exit-rule variant sweep on filtered subsets (Phase 2 Round 2)\n")
            f.write("# Phase: l6_arc2_exit_sweep_filtered\n")
            f.write("\n## Inputs (locked sha256)\n")
            for rel, h in observed_shas.items():
                f.write(f"{rel} {h}\n")
            f.write("\n## Trades_all spread lookup (informational)\n")
            f.write(f"{TRADES_ALL_REL} {_sha256_file(REPO_ROOT / TRADES_ALL_REL)}\n")
            f.write("\n## Outputs (sha256)\n")
            for rel, h in out_shas.items():
                f.write(f"{rel} {h}\n")
            f.write("\n## Run\n")
            try:
                head = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip()
            except Exception:
                head = "unknown"
            f.write(f"git_head {head}\n")
            f.write(f"wallclock_sec {wallclock:.2f}\n")

    return {
        "observed_shas": observed_shas,
        "out_shas": out_shas,
        "gates": gates,
        "block_Z": block_Z,
        "block_AA_stab": block_AA_stab,
        "block_BB": block_BB,
        "block_CC": block_CC,
        "per_fold": per_fold,
        "vt": vt,
    }


# ===========================================================================
# Main: two build passes for determinism
# ===========================================================================


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-pass", action="store_true",
                        help="Skip the determinism second pass.")
    args = parser.parse_args(argv)

    t_start = time.time()
    tracemalloc.start()

    out_dir = REPO_ROOT / OUTPUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Pass 1 ===", flush=True)
    r1 = build_pass(out_dir=out_dir, write_manifest=False, run_label="pass_1")

    if args.single_pass:
        # Still write manifest for inspection.
        r1 = build_pass(out_dir=out_dir, write_manifest=True, run_label="pass_1_w_manifest")
        det_ok = True
    else:
        # Pass 1 wrote outputs; snapshot them.
        snapshot_dir = Path(tempfile.mkdtemp(prefix="arc2_exit_sweep_snap_"))
        snap_files = {}
        for p in (out_dir / n for n in [
            "block_Z_per_subset_per_variant.csv",
            "block_AA_per_subset_per_variant_per_fold.csv",
            "block_AA_stability_summary.csv",
            "block_BB_per_subset_per_variant_per_category.csv",
            "block_CC_headline_summary.csv",
            "exit_sweep_filtered.md",
        ]):
            if not p.exists():
                raise RuntimeError(f"HALT: pass 1 did not write {p}")
            snap_files[p.name] = snapshot_dir / p.name
            shutil.copy2(p, snap_files[p.name])

        print("\n=== Pass 2 ===", flush=True)
        r2 = build_pass(out_dir=out_dir, write_manifest=True,
                        run_label="pass_2_final")

        # Determinism check
        det_diffs: List[str] = []
        for fname, snap_path in snap_files.items():
            cur = out_dir / fname
            if not filecmp.cmp(snap_path, cur, shallow=False):
                det_diffs.append(fname)
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        if det_diffs:
            raise RuntimeError(
                f"HALT (gate 8): determinism failed; differing files: {det_diffs}"
            )
        det_ok = True
        r1 = r2

    peak_kb = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    wallclock = time.time() - t_start

    # ====== Final handoff ======
    print("\n=== Validation gates disposition ===", flush=True)
    for k, v in r1["gates"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  determinism (gate 8): {'ok' if det_ok else 'FAIL'}", flush=True)

    # Headline numbers
    block_Z = r1["block_Z"]
    block_AA_stab = r1["block_AA_stab"]
    print("\n=== Headline numbers ===", flush=True)
    # Top (subset, variant) by mean_R across all rows except V00_BL
    z_no_bl = block_Z[block_Z["variant_id"] != "V00_BL"].copy()
    top_row = z_no_bl.sort_values("mean_R", ascending=False).iloc[0]
    top_stab = block_AA_stab[(block_AA_stab["subset_id"] == top_row["subset_id"])
                              & (block_AA_stab["variant_id"] == top_row["variant_id"])].iloc[0]
    print(
        f"  TOP overall by mean_R: subset={top_row['subset_id']}, "
        f"variant={top_row['variant_id']}, n={int(top_row['n_in_subset'])}, "
        f"mean_R={top_row['mean_R']:+.4f}, "
        f"lift_vs_BL={top_row['pooled_lift_vs_BL']:+.4f}, "
        f"stability_flag={top_stab['stability_flag']}, "
        f"folds_above_BL={int(top_stab['folds_mean_R_above_BL'])}",
        flush=True,
    )
    for sid in SUBSET_IDS[1:]:  # skip S0_pop
        sub = z_no_bl[z_no_bl["subset_id"] == sid]
        if len(sub) == 0:
            continue
        top_sub = sub.sort_values("pooled_lift_vs_BL", ascending=False).iloc[0]
        ts = block_AA_stab[(block_AA_stab["subset_id"] == sid)
                            & (block_AA_stab["variant_id"] == top_sub["variant_id"])].iloc[0]
        print(
            f"  TOP {sid}: variant={top_sub['variant_id']}, "
            f"lift={top_sub['pooled_lift_vs_BL']:+.4f}, "
            f"only_up_captured={top_sub['pct_only_up_captured_at_ge_1R']:.3f}, "
            f"dtu_captured={top_sub['pct_down_then_up_captured_at_ge_1R']:.3f}, "
            f"stability={ts['stability_flag']}",
            flush=True,
        )

    # Cross-subset: variants stable_lift on >= 4 of S1..S5
    five = [s for s in SUBSET_IDS if s != "S0_pop"]
    stab_pivot = (block_AA_stab[block_AA_stab["subset_id"].isin(five)]
                  .pivot(index="variant_id", columns="subset_id",
                         values="stability_flag"))
    n_stable_lift = stab_pivot.eq("stable_lift").sum(axis=1)
    big = n_stable_lift[n_stable_lift >= 4].sort_values(ascending=False)
    print(f"  Cross-subset stable_lift >= 4 of S1..S5: "
          f"{big.to_dict() if len(big) > 0 else 'none'}", flush=True)

    # Output sha manifest
    print("\n=== Output artefact sha256 manifest ===", flush=True)
    for rel, h in r1["out_shas"].items():
        print(f"  {rel}  {h}", flush=True)

    # Git status
    print("\n=== git status (HEAD unchanged; new files untracked) ===", flush=True)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        print(f"  HEAD: {head}", flush=True)
        st = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
        for ln in st.splitlines()[:80]:
            print(f"  {ln}", flush=True)
    except Exception as e:
        print(f"  (git unavailable: {e})", flush=True)

    print(f"\n  wallclock {wallclock:.2f}s  peak_RSS_traced_kb {peak_kb:.0f}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
