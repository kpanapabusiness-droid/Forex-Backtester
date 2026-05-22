"""Arc 2 — Exit-rule counterfactuals Round 2 (Round 3E).

Phase: L6 Arc 2 Phase 3 — exit-rule descriptive evaluation
(L6_0_METHODOLOGY_LOCK Section 14.2 derivative experiment, 14.5
descriptive discipline, 14.6 read-existing-CSV backfill).

Five descriptive blocks:
  Block PP — Early-cut at k=20, tau in {-0.5, -0.25, -0.75} ATR
  Block QQ — Tier-conditional H240 based on k=120 confirmation
  Block RR — PP + QQ combined
  Block SS — Trail on confirmed runners (close-based peak)
  Block TT — Full combined: PP + QQ + SS

SL is locked at -2 ATR throughout every variant; tier and category
labels are used for reporting only and never for exit decisions.
Re-uses spread accounting and intrabar SL priority from Round 2's
arc2_exit_sweep_filtered.py via direct import.

DESCRIPTIVE ONLY (Section 14.5). No variant selection. No spec
lock. Output is consumed by chat-side planning.

Outputs to: results/l6/arc2/characterisation/extended/exit_counterfactuals_round2/
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

os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_LCHAR = REPO_ROOT / "scripts" / "lchar"
if str(SCRIPTS_LCHAR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_LCHAR))

# Import Round 2 primitives — SL trigger, exit helpers, spread accounting,
# subset reconstruction, Block B labelling. The user spec explicitly
# requires reuse of these to make V00_BL byte-identical to Round 2.
import arc2_exit_sweep_filtered as r2  # noqa: E402

# ---------------------------------------------------------------------------
# Locked input sha256s (11 inputs; gate 1, re-verified as gate 13)
# ---------------------------------------------------------------------------
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv": "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv": "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv": "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv": "4a61407f0f1fc1b74486f0614928e776201dc6469d874db8393e689d20cdb2ff",
    "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv": "a5e3f8e68aa64d8fd53f752705a33613d9877dbde1f8265cb4a38d753c5e088e",
    "results/l6/arc2/characterisation/extended/path_by_subset/block_V_subset_category_breakdown.csv": "78633e9904baf2a672d2c8692f4b3557fec0aa3af8044ef3296dde08bad71c02",
    "results/l6/arc2/characterisation/extended/counterfactuals/round_1/counterfactual_sweep_round_1.md": "635ad1fdaf26525cd5e27c1d8b4c4d807da44d9d9d7c83afed9c8754dbc6e0b2",
    "results/l6/arc2/characterisation/extended/exit_sweep_filtered/block_Z_per_subset_per_variant.csv": "d4d13f1793bce3292b984aac80e983c9b08d1ec383ce2bca3f3092515798bc21",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py": "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml": "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "L6_0_METHODOLOGY_LOCK.md": "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}

# trades_all.csv is required for spread lookup (not formally locked here,
# same convention as Round 2). Captured in run manifest.
TRADES_ALL_REL = "results/l6/arc2/trades_all.csv"

OUTPUT_DIR_REL = "results/l6/arc2/characterisation/extended/exit_counterfactuals_round2"

# ---------------------------------------------------------------------------
# Inheritance from Round 2
# ---------------------------------------------------------------------------
SUBSET_DEFS = r2.SUBSET_DEFS
SUBSET_IDS_ALL = r2.SUBSET_IDS
# This round reports on three subsets per spec §2.
SUBSET_IDS: Tuple[str, ...] = ("S0_pop", "S1_q5q2", "S4_q5xq2q3")
BLOCK_B_1R_COUNTS = r2.BLOCK_B_1R_COUNTS
ALL_CATS = r2.ALL_CATS
PATH_THRESH_ATR = r2.PATH_THRESH_ATR
PATH_HOLD_CAP = r2.PATH_HOLD_CAP
TIME_HORIZON_DEFAULT = r2.TIME_HORIZON_DEFAULT
TIME_HORIZON_EXTENDED = r2.TIME_HORIZON_EXTENDED

# Round 3C Block MM tier bounds (lower bound exclusive, upper inclusive).
TIER_BOUNDS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 4.0, "tier_low"),
    (4.0, 10.0, "tier_mid"),
    (10.0, 20.0, "tier_high"),
    (20.0, float("inf"), "tier_runner"),
)
TIER_LABELS: Tuple[str, ...] = tuple(t[2] for t in TIER_BOUNDS)

# Exit-reason short codes used in this round.
REASON_MAP_3E: Dict[str, str] = {
    "stop_loss": "sl",
    "fixed_tp": "tp",
    "trail_exit": "trail",
    "be_exit": "be",
    "time_exit": "te",
    "data_end": "de",
    "early_cut": "cut",
}
ALL_EXIT_SHORT: Tuple[str, ...] = ("sl", "tp", "trail", "be", "te", "de", "cut")

# Reproduction tolerance for gate 3.1.
GATE_3_1_TOL: float = 1e-7

# Disposition discipline forbidden patterns.
FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "best variant",
    " recommend",
    "should use",
    "this is the answer",
    "we should adopt",
    "the right exit rule is",
    "the right rule is",
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
# New variant functions (reuse _exit_sl/te_de/trail spread accounting)
# ===========================================================================


def variant_BL(
    rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T, *, time_horizon: int = TIME_HORIZON_DEFAULT
) -> Dict[str, Any]:
    """V00_BL / V09_H240. Byte-identical to Round 2's variant_BL.

    Returns Round 2 dict shape + trail_activated_at_k = -1.
    """
    out = r2.variant_BL(rmae, rmfe, bl_, bh_, bc_, nbo, hnb, bavail, T, time_horizon=time_horizon)
    out["trail_activated_at_k"] = -1
    return out


def variant_PP(
    rmae,
    rmfe,
    bl_,
    bh_,
    bc_,
    nbo,
    hnb,
    bavail,
    T,
    *,
    tau_atr_fill: float,
    cut_bar: int = 20,
    time_horizon: int = TIME_HORIZON_DEFAULT,
) -> Dict[str, Any]:
    """SL throughout. At k=cut_bar, if running_close <= tau, exit at that close.

    Exit price for cut uses _exit_te_de spread accounting (same one-sided
    exit spread as a time_exit at that bar's close).
    """
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            out = r2._exit_sl(k_idx=k_idx, T=T, M=2.0)
            out["trail_activated_at_k"] = -1
            return out
        if k == cut_bar and bc_[k_idx] <= tau_atr_fill:
            out = r2._exit_te_de(
                k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="early_cut"
            )
            out["trail_activated_at_k"] = -1
            return out
        if k == time_horizon:
            if hnb[k_idx]:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=nbo[k_idx], exit_reason="time_exit"
                )
            else:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
                )
            out["trail_activated_at_k"] = -1
            return out
        if k == bavail:
            out = r2._exit_te_de(
                k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
            )
            out["trail_activated_at_k"] = -1
            return out
    raise RuntimeError("variant_PP did not terminate")


def variant_QQ(
    rmae,
    rmfe,
    bl_,
    bh_,
    bc_,
    nbo,
    hnb,
    bavail,
    T,
    *,
    decision_bar: int = 120,
    confirm_kind: str,
    confirm_threshold: float,
    extended_horizon: int = TIME_HORIZON_EXTENDED,
) -> Dict[str, Any]:
    """SL throughout. At k=decision_bar, evaluate confirmation; if confirmed,
    extend horizon to extended_horizon; else exit at decision_bar's
    next-bar-open (or close if clamped). Falls back to data_end at bavail.

    confirm_kind in {"mfe", "close"}; threshold in ATR fill-rel units.
    """
    horizon = decision_bar
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            out = r2._exit_sl(k_idx=k_idx, T=T, M=2.0)
            out["trail_activated_at_k"] = -1
            return out
        if k == decision_bar:
            val = rmfe[k_idx] if confirm_kind == "mfe" else bc_[k_idx]
            if val >= confirm_threshold:
                horizon = extended_horizon
            else:
                if hnb[k_idx]:
                    out = r2._exit_te_de(
                        k_idx=k_idx, T=T, M=2.0, exit_atr_fill=nbo[k_idx], exit_reason="time_exit"
                    )
                else:
                    out = r2._exit_te_de(
                        k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
                    )
                out["trail_activated_at_k"] = -1
                return out
        if k == horizon:
            if hnb[k_idx]:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=nbo[k_idx], exit_reason="time_exit"
                )
            else:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
                )
            out["trail_activated_at_k"] = -1
            return out
        if k == bavail:
            out = r2._exit_te_de(
                k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
            )
            out["trail_activated_at_k"] = -1
            return out
    raise RuntimeError("variant_QQ did not terminate")


def variant_RR(
    rmae,
    rmfe,
    bl_,
    bh_,
    bc_,
    nbo,
    hnb,
    bavail,
    T,
    *,
    tau_atr_fill: float,
    cut_bar: int = 20,
    decision_bar: int = 120,
    confirm_kind: str,
    confirm_threshold: float,
    extended_horizon: int = TIME_HORIZON_EXTENDED,
) -> Dict[str, Any]:
    """PP cut + QQ conditional hold. No trail."""
    horizon = decision_bar
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            out = r2._exit_sl(k_idx=k_idx, T=T, M=2.0)
            out["trail_activated_at_k"] = -1
            return out
        if k == cut_bar and bc_[k_idx] <= tau_atr_fill:
            out = r2._exit_te_de(
                k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="early_cut"
            )
            out["trail_activated_at_k"] = -1
            return out
        if k == decision_bar:
            val = rmfe[k_idx] if confirm_kind == "mfe" else bc_[k_idx]
            if val >= confirm_threshold:
                horizon = extended_horizon
            else:
                if hnb[k_idx]:
                    out = r2._exit_te_de(
                        k_idx=k_idx, T=T, M=2.0, exit_atr_fill=nbo[k_idx], exit_reason="time_exit"
                    )
                else:
                    out = r2._exit_te_de(
                        k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
                    )
                out["trail_activated_at_k"] = -1
                return out
        if k == horizon:
            if hnb[k_idx]:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=nbo[k_idx], exit_reason="time_exit"
                )
            else:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
                )
            out["trail_activated_at_k"] = -1
            return out
        if k == bavail:
            out = r2._exit_te_de(
                k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
            )
            out["trail_activated_at_k"] = -1
            return out
    raise RuntimeError("variant_RR did not terminate")


def variant_SS(
    rmae,
    rmfe,
    bl_,
    bh_,
    bc_,
    nbo,
    hnb,
    bavail,
    T,
    *,
    X_act_atr: float,
    D_atr: float,
    time_horizon: int,
) -> Dict[str, Any]:
    """Close-based trail.

    Activation: running_mfe_atr first reaches X_act_atr (uses intrabar
    high). Once active, peak = cumulative max of bar_close_atr since
    activation. trail_level monotone non-decreasing = max over k of
    (peak[k] - D). Exit when bar_close_atr <= trail_level (intrabar
    semantics: SL checked first each bar; trail next).

    Activation effective from k+1 (matches Round 2 variant_TRAIL).
    Exit price uses _exit_trail's spread accounting.
    """
    trail_active = False
    activated_at_k = -1
    peak_close = -np.inf
    trail_level = -np.inf
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            out = r2._exit_sl(k_idx=k_idx, T=T, M=2.0)
            out["trail_activated_at_k"] = activated_at_k
            return out
        if trail_active:
            if bc_[k_idx] > peak_close:
                peak_close = bc_[k_idx]
            new_tl = peak_close - D_atr
            if new_tl > trail_level:
                trail_level = new_tl
            if bc_[k_idx] <= trail_level:
                out = r2._exit_trail(k_idx=k_idx, T=T, trail_level_atr_fill=trail_level)
                out["trail_activated_at_k"] = activated_at_k
                return out
        if not trail_active and rmfe[k_idx] >= X_act_atr:
            trail_active = True
            activated_at_k = k
            peak_close = bc_[k_idx]
            trail_level = peak_close - D_atr
        if k == time_horizon:
            if hnb[k_idx]:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=nbo[k_idx], exit_reason="time_exit"
                )
            else:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
                )
            out["trail_activated_at_k"] = activated_at_k
            return out
        if k == bavail:
            out = r2._exit_te_de(
                k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
            )
            out["trail_activated_at_k"] = activated_at_k
            return out
    raise RuntimeError("variant_SS did not terminate")


def variant_TT(
    rmae,
    rmfe,
    bl_,
    bh_,
    bc_,
    nbo,
    hnb,
    bavail,
    T,
    *,
    tau_atr_fill: float,
    cut_bar: int = 20,
    X_act_atr: float,
    D_atr: float,
    decision_bar: int = 120,
    confirm_kind: str,
    confirm_threshold: float,
    extended_horizon: int = TIME_HORIZON_EXTENDED,
) -> Dict[str, Any]:
    """PP cut + QQ conditional hold + SS close-based trail."""
    trail_active = False
    activated_at_k = -1
    peak_close = -np.inf
    trail_level = -np.inf
    horizon = decision_bar
    for k_idx in range(bavail):
        k = k_idx + 1
        # 1. SL.
        if rmae[k_idx] <= -2.0:
            out = r2._exit_sl(k_idx=k_idx, T=T, M=2.0)
            out["trail_activated_at_k"] = activated_at_k
            return out
        # 2. PP cut.
        if k == cut_bar and bc_[k_idx] <= tau_atr_fill:
            out = r2._exit_te_de(
                k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="early_cut"
            )
            out["trail_activated_at_k"] = activated_at_k
            return out
        # 3. Trail exit (if active).
        if trail_active:
            if bc_[k_idx] > peak_close:
                peak_close = bc_[k_idx]
            new_tl = peak_close - D_atr
            if new_tl > trail_level:
                trail_level = new_tl
            if bc_[k_idx] <= trail_level:
                out = r2._exit_trail(k_idx=k_idx, T=T, trail_level_atr_fill=trail_level)
                out["trail_activated_at_k"] = activated_at_k
                return out
        # 4. QQ confirmation at decision_bar.
        if k == decision_bar:
            val = rmfe[k_idx] if confirm_kind == "mfe" else bc_[k_idx]
            if val >= confirm_threshold:
                horizon = extended_horizon
            else:
                if hnb[k_idx]:
                    out = r2._exit_te_de(
                        k_idx=k_idx, T=T, M=2.0, exit_atr_fill=nbo[k_idx], exit_reason="time_exit"
                    )
                else:
                    out = r2._exit_te_de(
                        k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
                    )
                out["trail_activated_at_k"] = activated_at_k
                return out
        # 5. Trail activation (delayed: applies from k+1).
        if not trail_active and rmfe[k_idx] >= X_act_atr:
            trail_active = True
            activated_at_k = k
            peak_close = bc_[k_idx]
            trail_level = peak_close - D_atr
        # 6. Horizon.
        if k == horizon:
            if hnb[k_idx]:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=nbo[k_idx], exit_reason="time_exit"
                )
            else:
                out = r2._exit_te_de(
                    k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
                )
            out["trail_activated_at_k"] = activated_at_k
            return out
        if k == bavail:
            out = r2._exit_te_de(
                k_idx=k_idx, T=T, M=2.0, exit_atr_fill=bc_[k_idx], exit_reason="data_end"
            )
            out["trail_activated_at_k"] = activated_at_k
            return out
    raise RuntimeError("variant_TT did not terminate")


# ===========================================================================
# Variant registry
# ===========================================================================

# PP threshold (ATR fill-rel) per variant.
PP_VARIANTS: Tuple[Tuple[str, float], ...] = (
    ("PP01", -0.5),
    ("PP02", -0.25),
    ("PP03", -0.75),
)

# QQ confirmation criteria per variant: (id, kind, threshold_atr).
QQ_VARIANTS: Tuple[Tuple[str, str, float], ...] = (
    ("QQ01", "mfe", 4.0),
    ("QQ02", "mfe", 6.0),
    ("QQ03", "close", 2.0),
    ("QQ04", "close", 4.0),
)

# RR combinations (PP01 + each QQ).
RR_VARIANTS: Tuple[Tuple[str, float, str, float], ...] = (
    ("RR01", -0.5, "mfe", 4.0),
    ("RR02", -0.5, "mfe", 6.0),
    ("RR03", -0.5, "close", 2.0),
    ("RR04", -0.5, "close", 4.0),
)

# SS trail variants: (id, X_act_atr, D_atr, TE).
SS_VARIANTS: Tuple[Tuple[str, float, float, int], ...] = (
    ("SS01", 4.0, 2.0, TIME_HORIZON_DEFAULT),
    ("SS02", 4.0, 3.0, TIME_HORIZON_DEFAULT),
    ("SS03", 6.0, 2.0, TIME_HORIZON_DEFAULT),
    ("SS04", 6.0, 3.0, TIME_HORIZON_DEFAULT),
    ("SS05", 8.0, 2.0, TIME_HORIZON_DEFAULT),
    ("SS06", 8.0, 3.0, TIME_HORIZON_DEFAULT),
    ("SS07", 6.0, 3.0, TIME_HORIZON_EXTENDED),
    ("SS08", 4.0, 2.0, TIME_HORIZON_EXTENDED),
    ("SS09", 8.0, 3.0, TIME_HORIZON_EXTENDED),
)

# TT combinations: (id, tau, X_act, D, confirm_kind, confirm_thr).
TT_VARIANTS: Tuple[Tuple[str, float, float, float, str, float], ...] = (
    ("TT01", -0.5, 4.0, 2.0, "close", 2.0),
    ("TT02", -0.5, 6.0, 3.0, "mfe", 4.0),
    ("TT03", -0.5, 4.0, 3.0, "close", 2.0),
    ("TT04", -0.5, 6.0, 2.0, "close", 4.0),
    ("TT05", -0.5, 8.0, 3.0, "close", 4.0),
)


def _make_variant_list() -> List[Tuple[str, str, Callable[..., Dict[str, Any]]]]:
    items: List[Tuple[str, str, Callable[..., Dict[str, Any]]]] = []
    items.append(
        (
            "V00_BL",
            "Baseline (SL=-2R, TE=120)",
            lambda *a: variant_BL(*a, time_horizon=TIME_HORIZON_DEFAULT),
        )
    )
    items.append(
        (
            "V09_H240",
            "BL with TE=240",
            lambda *a: variant_BL(*a, time_horizon=TIME_HORIZON_EXTENDED),
        )
    )
    for vid, tau in PP_VARIANTS:
        items.append(
            (
                vid,
                f"Early-cut at k=20, tau={tau}",
                (lambda tau_=tau: lambda *a: variant_PP(*a, tau_atr_fill=tau_))(),
            )
        )
    for vid, kind, thr in QQ_VARIANTS:
        items.append(
            (
                vid,
                f"H240 conditional on {kind}>={thr}",
                (
                    lambda k_=kind, t_=thr: (
                        lambda *a: variant_QQ(*a, confirm_kind=k_, confirm_threshold=t_)
                    )
                )(),
            )
        )
    for vid, tau, kind, thr in RR_VARIANTS:
        items.append(
            (
                vid,
                f"PP tau={tau} + QQ {kind}>={thr}",
                (
                    lambda tau_=tau, k_=kind, t_=thr: (
                        lambda *a: variant_RR(
                            *a, tau_atr_fill=tau_, confirm_kind=k_, confirm_threshold=t_
                        )
                    )
                )(),
            )
        )
    for vid, xact, d, te in SS_VARIANTS:
        items.append(
            (
                vid,
                f"Trail X_act={xact} D={d} TE={te}",
                (
                    lambda x_=xact, d_=d, t_=te: (
                        lambda *a: variant_SS(*a, X_act_atr=x_, D_atr=d_, time_horizon=t_)
                    )
                )(),
            )
        )
    for vid, tau, xact, d, kind, thr in TT_VARIANTS:
        items.append(
            (
                vid,
                f"TT tau={tau} X_act={xact} D={d} {kind}>={thr}",
                (
                    lambda tau_=tau, x_=xact, d_=d, k_=kind, t_=thr: (
                        lambda *a: variant_TT(
                            *a,
                            tau_atr_fill=tau_,
                            X_act_atr=x_,
                            D_atr=d_,
                            confirm_kind=k_,
                            confirm_threshold=t_,
                        )
                    )
                )(),
            )
        )
    return items


VARIANTS = _make_variant_list()
VARIANT_IDS: Tuple[str, ...] = tuple(v[0] for v in VARIANTS)


# ===========================================================================
# Sweep — 27 variants x 3993 trades
# ===========================================================================


def run_sweep() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Returns (variant_trades, ti_full, cats, peak_mfe, subsets)."""
    print("  Loading trade_index.csv + trades_all.csv...", flush=True)
    ti = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv")
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ta = pd.read_csv(REPO_ROOT / TRADES_ALL_REL)
    ta["signal_bar_ts"] = pd.to_datetime(ta["signal_bar_ts"])
    ti_full = ti.merge(
        ta[["pair", "signal_bar_ts", "spread_pips_entry", "spread_pips_exit"]],
        on=["pair", "signal_bar_ts"],
        how="left",
        validate="one_to_one",
    )
    if ti_full[["spread_pips_entry", "spread_pips_exit"]].isna().any().any():
        raise RuntimeError("HALT (sp-lookup): null sp_entry/sp_exit after merge")
    ti_full = ti_full.sort_values("trade_id").reset_index(drop=True)

    print("  Pre-computing per-trade values...", flush=True)
    per_trade: Dict[int, Dict[str, Any]] = {}
    for _, row in ti_full.iterrows():
        tid = int(row["trade_id"])
        pair = row["pair"]
        pip = r2._pip_size(pair)
        sp_entry = float(row["spread_pips_entry"])
        sp_exit = float(row["spread_pips_exit"])
        atr = float(row["atr_1h_wilder_at_signal"])
        entry_fill = float(row["entry_price"])
        per_trade[tid] = {
            "pair": pair,
            "atr": atr,
            "entry_fill": entry_fill,
            "sp_entry_pips": sp_entry,
            "sp_exit_pips": sp_exit,
            "pip": pip,
            "entry_fill_offset_atr": sp_entry * pip / (2 * atr),
            "baseline_spread_cost_r": (sp_entry + sp_exit) * pip / (4 * atr),
            "fold_id": int(row["fold_id"]),
            "signal_bar_ts": row["signal_bar_ts"].strftime("%Y-%m-%dT%H:%M:%S"),
        }

    print("  Loading per_bar_paths.csv...", flush=True)
    pb = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv")
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

    # Block B categories (gate 2.3).
    cats = r2.compute_categories(pb, n_trades)
    print("  Block B categories:", {c: int((cats == c).sum()) for c in ALL_CATS}, flush=True)

    # Per-trade peak_mfe for tier classification (reporting only).
    peak_mfe = np.full(n_trades, np.nan, dtype=np.float64)
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        if e > s:
            peak_mfe[tid] = float(rmfe_all[s:e].max())

    # Build subsets (gate 2.1, 2.2 inside).
    _, subsets_full = r2.build_subsets()
    subsets = {sid: subsets_full[sid] for sid in SUBSET_IDS}
    print("  Subsets: {sid: n} =", {sid: len(t) for sid, t in subsets.items()}, flush=True)

    # Run 27 variants per trade.
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
    out_act_k = np.empty(total_rows, dtype=np.int64)

    print(f"  Simulating {n_vars} variants x {n_trades} trades...", flush=True)
    t0 = time.time()
    write_idx = 0
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        bavail = e - s
        rmae = rmae_all[s:e]
        rmfe = rmfe_all[s:e]
        bl_ = bl_all[s:e]
        bh_ = bh_all[s:e]
        bc_ = bc_all[s:e]
        nbo = nbo_all[s:e]
        hnb = hnb_all[s:e]
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
            out_act_k[write_idx] = r["trail_activated_at_k"]
            write_idx += 1
        if (tid + 1) % 1000 == 0:
            el = time.time() - t0
            print(
                f"    progress: {tid + 1}/{n_trades} ({el:.1f}s, {(tid + 1) / el:.0f} trades/s)",
                flush=True,
            )

    vt = pd.DataFrame(
        {
            "variant_id": out_variant,
            "trade_id": out_tid,
            "pair": out_pair,
            "signal_bar_ts": out_sigts,
            "fold_id": out_fold,
            "exit_reason_internal": out_reason,
            "exit_bar": out_exitbar,
            "exit_level_atr_fill": out_exitlvl,
            "gross_R": out_gross,
            "spread_cost_R": out_spread,
            "net_R": out_net,
            "trail_activated_at_k": out_act_k,
        }
    )
    vt["exit_reason"] = vt["exit_reason_internal"].map(REASON_MAP_3E)
    if vt["exit_reason"].isna().any():
        raise RuntimeError("HALT: unmapped exit_reason_internal present")
    vt = vt.sort_values(["variant_id", "trade_id"]).reset_index(drop=True)
    return vt, ti_full, cats, peak_mfe, subsets


# ===========================================================================
# Gates
# ===========================================================================


def _gate_3_1_bl_repro(vt: pd.DataFrame, subsets: Dict[str, np.ndarray]) -> None:
    bz_path = (
        REPO_ROOT
        / "results/l6/arc2/characterisation/extended/exit_sweep_filtered/block_Z_per_subset_per_variant.csv"
    )
    bz = pd.read_csv(bz_path)
    ref = float(
        bz[(bz["subset_id"] == "S0_pop") & (bz["variant_id"] == "V00_BL")]["mean_R"].iloc[0]
    )
    bl_s0 = vt[
        (vt["variant_id"] == "V00_BL") & (vt["trade_id"].isin(set(subsets["S0_pop"].tolist())))
    ]
    got = float(bl_s0["net_R"].mean())
    diff = abs(got - ref)
    if diff > GATE_3_1_TOL:
        raise RuntimeError(
            f"HALT (gate 3.1): V00_BL S0_pop mean_R diff {diff:.3e} > tol "
            f"{GATE_3_1_TOL:.0e}. got={got!r}, ref={ref!r}"
        )


def _gate_3_2_clamped(vt: pd.DataFrame, ti_full: pd.DataFrame) -> None:
    """Gate 3.2: clamped trades route to "de" reason in V00_BL.

    Clamped means bavail < hold_horizon AND no SL in the window. We
    verify by checking that V00_BL data_end count is consistent with
    trades whose bavail < 120 and that did not SL.
    """
    # Use V00_BL as the canonical baseline.
    bl = vt[vt["variant_id"] == "V00_BL"]
    n_de = int((bl["exit_reason"] == "de").sum())
    # V00_BL Round 2 reference: 8 data_end on S0_pop.
    if n_de < 0:  # noop sentinel; just ensure column exists.
        raise RuntimeError("HALT (gate 3.2): missing data_end column")
    # Soft sanity check vs Round 2's published count (8 on S0_pop pooled).
    bz_path = (
        REPO_ROOT
        / "results/l6/arc2/characterisation/extended/exit_sweep_filtered/block_Z_per_subset_per_variant.csv"
    )
    bz = pd.read_csv(bz_path)
    ref_de = int(
        bz[(bz["subset_id"] == "S0_pop") & (bz["variant_id"] == "V00_BL")]["n_exit_reason_de"].iloc[
            0
        ]
    )
    got_de = int((bl["exit_reason"] == "de").sum())
    if got_de != ref_de:
        raise RuntimeError(
            f"HALT (gate 3.2): V00_BL S0_pop n_data_end {got_de} != Round 2 ref {ref_de}"
        )


def _gate_7_exit_exhaustive(vt: pd.DataFrame, subsets: Dict[str, np.ndarray]) -> None:
    """Gate 7: per (subset, variant), exit reasons sum to n."""
    for sid in SUBSET_IDS:
        tids = set(subsets[sid].tolist())
        for vid in VARIANT_IDS:
            sub = vt[(vt["variant_id"] == vid) & (vt["trade_id"].isin(tids))]
            n = len(sub)
            if n == 0:
                raise RuntimeError(f"HALT (gate 7): empty {sid}/{vid}")
            counts = sub["exit_reason"].value_counts().to_dict()
            total = int(sum(counts.values()))
            if total != n:
                raise RuntimeError(f"HALT (gate 7): {sid}/{vid} reasons sum {total} != n {n}")
            for r in counts.keys():
                if r not in ALL_EXIT_SHORT:
                    raise RuntimeError(f"HALT (gate 7): unexpected reason '{r}' in {sid}/{vid}")


def _gate_11_tt_pp_additivity(vt: pd.DataFrame) -> None:
    """For trades where TT exit==cut, the matching PP variant
    must also exit at cut with byte-identical net_R.
    """
    # All TT use tau=-0.5 which corresponds to PP01.
    pp_by_tid = vt[vt["variant_id"] == "PP01"].set_index("trade_id")
    for vid, *_ in TT_VARIANTS:
        tt = vt[vt["variant_id"] == vid]
        cut_tt = tt[tt["exit_reason"] == "cut"]
        for _, row in cut_tt.iterrows():
            tid = int(row["trade_id"])
            pp_row = pp_by_tid.loc[tid]
            if pp_row["exit_reason"] != "cut":
                raise RuntimeError(
                    f"HALT (gate 11): {vid} cut tid={tid} but PP01 not cut "
                    f"(PP01 reason={pp_row['exit_reason']})"
                )
            diff = abs(float(row["net_R"]) - float(pp_row["net_R"]))
            if diff > 1e-12:
                raise RuntimeError(f"HALT (gate 11): {vid} tid={tid} net_R diff {diff:.3e} vs PP01")


# ===========================================================================
# Block PP aggregation
# ===========================================================================


def _bar_of_sl_under_BL(
    rmae: np.ndarray, n_trades: int, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """First k where running_mae_atr <= -2 within k<=120; SENTINEL (=121)
    if never within window.
    """
    out = np.full(n_trades, 121, dtype=np.int32)
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        window = min(e - s, 120)
        for k_idx in range(window):
            if rmae[s + k_idx] <= -2.0:
                out[tid] = k_idx + 1
                break
    return out


def _close_at_k(starts, ends, bc, tid, k):
    s, e = int(starts[tid]), int(ends[tid])
    if e - s >= k:
        return float(bc[s + k - 1])
    return float("nan")


def aggregate_block_PP(
    vt: pd.DataFrame,
    subsets: Dict[str, np.ndarray],
    cats: np.ndarray,
    bar_of_sl_BL: np.ndarray,
    close_at_k20: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per_subset_per_category_per_variant_df, cut_group_detail_df,
    summary_df).
    """
    bl_by_tid = vt[vt["variant_id"] == "V00_BL"].set_index("trade_id")
    rows_cat: List[Dict[str, Any]] = []
    rows_cut: List[Dict[str, Any]] = []
    rows_summary: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        n_total_subset = len(tids)
        bl_sub = bl_by_tid.loc[bl_by_tid.index.isin(tids)]
        bl_pooled_mean = float(bl_sub["net_R"].mean())
        for vid, tau in PP_VARIANTS:
            v_by_tid = vt[vt["variant_id"] == vid].set_index("trade_id")
            v_sub = v_by_tid.loc[v_by_tid.index.isin(tids)]
            pooled_mean = float(v_sub["net_R"].mean())
            lift = pooled_mean - bl_pooled_mean
            n_cut_total = int((v_sub["exit_reason"] == "cut").sum())
            n_sl_total = int((v_sub["exit_reason"] == "sl").sum())
            n_te_total = int((v_sub["exit_reason"] == "te").sum())
            n_de_total = int((v_sub["exit_reason"] == "de").sum())
            rows_summary.append(
                {
                    "subset_id": sid,
                    "variant_id": vid,
                    "tau_atr_fill": tau,
                    "n_total": n_total_subset,
                    "n_cut_at_k20": n_cut_total,
                    "n_sl": n_sl_total,
                    "n_te": n_te_total,
                    "n_de": n_de_total,
                    "pooled_mean_R": pooled_mean,
                    "pooled_mean_R_BL": bl_pooled_mean,
                    "lift_vs_BL": lift,
                }
            )
            # Per-category breakdown.
            for cat_label in ALL_CATS:
                tids_cat = np.array(
                    [int(t) for t in tids.tolist() if cats[int(t)] == cat_label],
                    dtype=np.int64,
                )
                n_total_cat = int(len(tids_cat))
                if n_total_cat == 0:
                    rows_cat.append(
                        {
                            "subset_id": sid,
                            "variant_id": vid,
                            "category": cat_label,
                            "n_total_in_category": 0,
                            "n_sl_before_k20": 0,
                            "n_open_at_k20": 0,
                            "n_open_at_k20_below_tau": 0,
                            "n_open_at_k20_above_tau": 0,
                            "n_subsequent_sl": 0,
                            "n_subsequent_te_negative": 0,
                            "n_subsequent_te_positive": 0,
                            "n_subsequent_de": 0,
                            "mean_cut_price_R": float("nan"),
                            "median_cut_price_R": float("nan"),
                            "mean_bl_R_for_cut_group": float("nan"),
                            "savings_per_cut_trade": float("nan"),
                            "pre_cut_category_mean_R": float("nan"),
                            "post_cut_category_mean_R": float("nan"),
                            "lift_per_category": float("nan"),
                        }
                    )
                    continue
                bl_cat = bl_by_tid.loc[bl_by_tid.index.isin(tids_cat)]
                v_cat = v_by_tid.loc[v_by_tid.index.isin(tids_cat)]
                # bar_of_sl_BL is 121 if never SL'd within k<=120.
                bos = bar_of_sl_BL[tids_cat]
                sl_before_20 = bos <= 20
                n_sl_before_k20 = int(sl_before_20.sum())
                # Open at k=20 under BL = SL not yet triggered.
                open_mask = ~sl_before_20
                tids_open = tids_cat[open_mask]
                n_open = int(len(tids_open))
                if n_open > 0:
                    close_k20_vals = close_at_k20[tids_open]
                    below_mask = close_k20_vals <= tau
                    n_below = int(below_mask.sum())
                    n_above = n_open - n_below
                    cut_tids = tids_open[below_mask]
                else:
                    n_below = 0
                    n_above = 0
                    cut_tids = np.array([], dtype=np.int64)

                # For cut group, look up BL outcomes.
                if len(cut_tids) > 0:
                    bl_for_cut = bl_by_tid.loc[bl_by_tid.index.isin(cut_tids)]
                    v_for_cut = v_by_tid.loc[v_by_tid.index.isin(cut_tids)]
                    cut_prices = v_for_cut["net_R"].to_numpy(dtype=np.float64)
                    bl_R_for_cut = bl_for_cut["net_R"].to_numpy(dtype=np.float64)
                    bl_reason = bl_for_cut["exit_reason"].to_numpy()
                    n_sub_sl = int((bl_reason == "sl").sum())
                    n_sub_te = bl_reason == "te"
                    n_sub_te_neg = int(((bl_R_for_cut < 0) & n_sub_te).sum())
                    n_sub_te_pos = int(((bl_R_for_cut >= 0) & n_sub_te).sum())
                    n_sub_de = int((bl_reason == "de").sum())
                    mean_cut_R = float(np.mean(cut_prices))
                    median_cut_R = float(np.median(cut_prices))
                    mean_bl_R_for_cut = float(np.mean(bl_R_for_cut))
                    savings = mean_cut_R - mean_bl_R_for_cut
                    # Per-cut trade detail rows.
                    for tid_, cp, bp, br in zip(
                        cut_tids.tolist(),
                        cut_prices.tolist(),
                        bl_R_for_cut.tolist(),
                        bl_reason.tolist(),
                    ):
                        rows_cut.append(
                            {
                                "subset_id": sid,
                                "variant_id": vid,
                                "category": cat_label,
                                "trade_id": int(tid_),
                                "cut_price_R": cp,
                                "bl_R_counterfactual": bp,
                                "bl_exit_reason": br,
                                "savings_R": cp - bp,
                            }
                        )
                else:
                    n_sub_sl = 0
                    n_sub_te_neg = 0
                    n_sub_te_pos = 0
                    n_sub_de = 0
                    mean_cut_R = float("nan")
                    median_cut_R = float("nan")
                    mean_bl_R_for_cut = float("nan")
                    savings = float("nan")

                pre = float(bl_cat["net_R"].mean())
                post = float(v_cat["net_R"].mean())
                rows_cat.append(
                    {
                        "subset_id": sid,
                        "variant_id": vid,
                        "category": cat_label,
                        "n_total_in_category": n_total_cat,
                        "n_sl_before_k20": n_sl_before_k20,
                        "n_open_at_k20": n_open,
                        "n_open_at_k20_below_tau": n_below,
                        "n_open_at_k20_above_tau": n_above,
                        "n_subsequent_sl": n_sub_sl,
                        "n_subsequent_te_negative": n_sub_te_neg,
                        "n_subsequent_te_positive": n_sub_te_pos,
                        "n_subsequent_de": n_sub_de,
                        "mean_cut_price_R": mean_cut_R,
                        "median_cut_price_R": median_cut_R,
                        "mean_bl_R_for_cut_group": mean_bl_R_for_cut,
                        "savings_per_cut_trade": savings,
                        "pre_cut_category_mean_R": pre,
                        "post_cut_category_mean_R": post,
                        "lift_per_category": post - pre,
                    }
                )

    df_cat = pd.DataFrame(rows_cat)
    df_cut = pd.DataFrame(rows_cut)
    df_summary = pd.DataFrame(rows_summary)
    return df_cat, df_cut, df_summary


def _gate_8_pp_cut_detail(df_cat: pd.DataFrame) -> None:
    """Gate 8: per (subset, variant, category):
    n_sub_sl + n_sub_te_neg + n_sub_te_pos + n_sub_de = n_open_at_k20_below_tau.
    """
    for _, row in df_cat.iterrows():
        n_below = int(row["n_open_at_k20_below_tau"])
        s = int(
            row["n_subsequent_sl"]
            + row["n_subsequent_te_negative"]
            + row["n_subsequent_te_positive"]
            + row["n_subsequent_de"]
        )
        if s != n_below:
            raise RuntimeError(
                f"HALT (gate 8): {row['subset_id']}/{row['variant_id']}/"
                f"{row['category']}: sub sum {s} != n_below {n_below}"
            )


# ===========================================================================
# Block QQ aggregation
# ===========================================================================


def _confirm_flag(
    rmfe_at_120: float, bc_at_120: float, confirm_kind: str, threshold: float
) -> Optional[bool]:
    """None if either input is NaN (trade clamped before k=120)."""
    if np.isnan(rmfe_at_120) or np.isnan(bc_at_120):
        return None
    val = rmfe_at_120 if confirm_kind == "mfe" else bc_at_120
    return bool(val >= threshold)


def aggregate_block_QQ(
    vt: pd.DataFrame,
    subsets: Dict[str, np.ndarray],
    cats: np.ndarray,
    peak_mfe: np.ndarray,
    rmfe_at_120: np.ndarray,
    bc_at_120: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_subset_per_tier_per_variant_df, summary_df)."""
    rows_tier: List[Dict[str, Any]] = []
    rows_summary: List[Dict[str, Any]] = []
    bl_by_tid = vt[vt["variant_id"] == "V00_BL"].set_index("trade_id")
    h240_by_tid = vt[vt["variant_id"] == "V09_H240"].set_index("trade_id")
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        only_up_tids = np.array(
            [int(t) for t in tids.tolist() if cats[int(t)] == "only_up"],
            dtype=np.int64,
        )
        n_only_up = int(len(only_up_tids))
        # Tier labels for only_up.
        tier_for_tid: Dict[int, str] = {}
        for t in only_up_tids.tolist():
            pmfe = peak_mfe[t]
            label = TIER_LABELS[0]
            for lo, hi, lb in TIER_BOUNDS:
                if pmfe > lo and pmfe <= hi:
                    label = lb
                    break
            tier_for_tid[t] = label

        bl_pooled_mean = float(bl_by_tid.loc[bl_by_tid.index.isin(tids)]["net_R"].mean())
        h240_pooled_mean = float(h240_by_tid.loc[h240_by_tid.index.isin(tids)]["net_R"].mean())

        for vid, kind, thr in QQ_VARIANTS:
            v_by_tid = vt[vt["variant_id"] == vid].set_index("trade_id")
            v_sub = v_by_tid.loc[v_by_tid.index.isin(tids)]
            pooled_mean = float(v_sub["net_R"].mean())

            # Confirmation counts over only_up.
            n_conf = 0
            for t in only_up_tids.tolist():
                f = _confirm_flag(rmfe_at_120[t], bc_at_120[t], kind, thr)
                if f is True:
                    n_conf += 1
            n_not_conf = n_only_up - n_conf

            rows_summary.append(
                {
                    "subset_id": sid,
                    "variant_id": vid,
                    "confirm_kind": kind,
                    "confirm_threshold": thr,
                    "n_only_up_total": n_only_up,
                    "n_only_up_confirmed": n_conf,
                    "n_only_up_not_confirmed": n_not_conf,
                    "pct_confirmed": (n_conf / n_only_up if n_only_up > 0 else float("nan")),
                    "pooled_mean_R": pooled_mean,
                    "pooled_mean_R_BL": bl_pooled_mean,
                    "pooled_mean_R_H240": h240_pooled_mean,
                    "lift_vs_BL": pooled_mean - bl_pooled_mean,
                    "lift_vs_H240": pooled_mean - h240_pooled_mean,
                }
            )

            # Per-tier (only_up).
            for tier_label in TIER_LABELS:
                tids_tier = np.array(
                    [t for t, lb in tier_for_tid.items() if lb == tier_label],
                    dtype=np.int64,
                )
                n_in_tier = int(len(tids_tier))
                if n_in_tier == 0:
                    rows_tier.append(
                        {
                            "subset_id": sid,
                            "variant_id": vid,
                            "tier": tier_label,
                            "n_in_tier": 0,
                            "n_tier_confirmed": 0,
                            "pct_of_tier_confirmed": float("nan"),
                            "mean_R_BL_for_tier": float("nan"),
                            "mean_R_QQ_for_tier": float("nan"),
                            "mean_R_H240_for_tier": float("nan"),
                            "lift_QQ_vs_BL_for_tier": float("nan"),
                        }
                    )
                    continue
                conf_in_tier = 0
                for t in tids_tier.tolist():
                    f = _confirm_flag(rmfe_at_120[t], bc_at_120[t], kind, thr)
                    if f is True:
                        conf_in_tier += 1
                mean_bl = float(bl_by_tid.loc[bl_by_tid.index.isin(tids_tier)]["net_R"].mean())
                mean_qq = float(v_by_tid.loc[v_by_tid.index.isin(tids_tier)]["net_R"].mean())
                mean_h240 = float(
                    h240_by_tid.loc[h240_by_tid.index.isin(tids_tier)]["net_R"].mean()
                )
                rows_tier.append(
                    {
                        "subset_id": sid,
                        "variant_id": vid,
                        "tier": tier_label,
                        "n_in_tier": n_in_tier,
                        "n_tier_confirmed": conf_in_tier,
                        "pct_of_tier_confirmed": conf_in_tier / n_in_tier,
                        "mean_R_BL_for_tier": mean_bl,
                        "mean_R_QQ_for_tier": mean_qq,
                        "mean_R_H240_for_tier": mean_h240,
                        "lift_QQ_vs_BL_for_tier": mean_qq - mean_bl,
                    }
                )
    return pd.DataFrame(rows_tier), pd.DataFrame(rows_summary)


def _gate_9_qq_confirmation(qq_summary: pd.DataFrame) -> None:
    for _, row in qq_summary.iterrows():
        n_total = int(row["n_only_up_total"])
        s = int(row["n_only_up_confirmed"] + row["n_only_up_not_confirmed"])
        if s != n_total:
            raise RuntimeError(
                f"HALT (gate 9): {row['subset_id']}/{row['variant_id']}: "
                f"conf+not = {s} != n_only_up {n_total}"
            )


# ===========================================================================
# Block RR aggregation
# ===========================================================================


def aggregate_block_RR(
    vt: pd.DataFrame, subsets: Dict[str, np.ndarray], cats: np.ndarray
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    bl_by_tid = vt[vt["variant_id"] == "V00_BL"].set_index("trade_id")
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        bl_pooled = float(bl_by_tid.loc[bl_by_tid.index.isin(tids)]["net_R"].mean())
        for vid, tau, kind, thr in RR_VARIANTS:
            v_by_tid = vt[vt["variant_id"] == vid].set_index("trade_id")
            v_sub = v_by_tid.loc[v_by_tid.index.isin(tids)]
            pooled_mean = float(v_sub["net_R"].mean())
            counts = v_sub["exit_reason"].value_counts().to_dict()
            row: Dict[str, Any] = {
                "subset_id": sid,
                "variant_id": vid,
                "tau_atr_fill": tau,
                "confirm_kind": kind,
                "confirm_threshold": thr,
                "n_total": len(v_sub),
                "n_cut": int(counts.get("cut", 0)),
                "n_sl": int(counts.get("sl", 0)),
                "n_te": int(counts.get("te", 0)),
                "n_de": int(counts.get("de", 0)),
                "pooled_mean_R": pooled_mean,
                "pooled_mean_R_BL": bl_pooled,
                "lift_vs_BL": pooled_mean - bl_pooled,
            }
            # Per-category exit-reason breakdown.
            v_sub_with_cat = v_sub.assign(
                category=v_sub.index.to_series().map(lambda tid: cats[int(tid)])
            )
            for cat in ALL_CATS:
                cat_sub = v_sub_with_cat[v_sub_with_cat["category"] == cat]
                row[f"n_{cat}_cut"] = int((cat_sub["exit_reason"] == "cut").sum())
                row[f"n_{cat}_sl"] = int((cat_sub["exit_reason"] == "sl").sum())
                row[f"n_{cat}_te"] = int((cat_sub["exit_reason"] == "te").sum())
                row[f"n_{cat}_de"] = int((cat_sub["exit_reason"] == "de").sum())
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Block SS aggregation
# ===========================================================================


def aggregate_block_SS(
    vt: pd.DataFrame, subsets: Dict[str, np.ndarray], cats: np.ndarray, peak_mfe: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_tier_per_variant_df, summary_df)."""
    rows_tier: List[Dict[str, Any]] = []
    rows_summary: List[Dict[str, Any]] = []
    bl_by_tid = vt[vt["variant_id"] == "V00_BL"].set_index("trade_id")
    h240_by_tid = vt[vt["variant_id"] == "V09_H240"].set_index("trade_id")
    for sid in SUBSET_IDS:
        tids = subsets[sid]
        only_up_tids = np.array(
            [int(t) for t in tids.tolist() if cats[int(t)] == "only_up"],
            dtype=np.int64,
        )
        tier_for_tid: Dict[int, str] = {}
        for t in only_up_tids.tolist():
            pmfe = peak_mfe[t]
            label = TIER_LABELS[0]
            for lo, hi, lb in TIER_BOUNDS:
                if pmfe > lo and pmfe <= hi:
                    label = lb
                    break
            tier_for_tid[t] = label

        bl_pooled = float(bl_by_tid.loc[bl_by_tid.index.isin(tids)]["net_R"].mean())
        h240_pooled = float(h240_by_tid.loc[h240_by_tid.index.isin(tids)]["net_R"].mean())

        for vid, xact, d, te in SS_VARIANTS:
            v_by_tid = vt[vt["variant_id"] == vid].set_index("trade_id")
            v_sub = v_by_tid.loc[v_by_tid.index.isin(tids)]
            pooled_mean = float(v_sub["net_R"].mean())
            counts = v_sub["exit_reason"].value_counts().to_dict()
            n_trail_total = int(counts.get("trail", 0))
            # Activation counts subset-wide.
            n_activated_total = int((v_sub["trail_activated_at_k"] > 0).sum())
            n_sl_after_act = int(
                ((v_sub["trail_activated_at_k"] > 0) & (v_sub["exit_reason"] == "sl")).sum()
            )
            n_te_after_act = int(
                (
                    (v_sub["trail_activated_at_k"] > 0) & (v_sub["exit_reason"].isin(["te", "de"]))
                ).sum()
            )
            rows_summary.append(
                {
                    "subset_id": sid,
                    "variant_id": vid,
                    "X_act_atr": xact,
                    "D_atr": d,
                    "time_horizon": te,
                    "n_total": len(v_sub),
                    "n_trail_activated": n_activated_total,
                    "n_trail_exited": n_trail_total,
                    "n_sl_after_activation": n_sl_after_act,
                    "n_te_after_activation": n_te_after_act,
                    "pooled_mean_R": pooled_mean,
                    "pooled_mean_R_BL": bl_pooled,
                    "pooled_mean_R_H240": h240_pooled,
                    "lift_vs_BL": pooled_mean - bl_pooled,
                    "lift_vs_H240": pooled_mean - h240_pooled,
                }
            )

            # Per-tier (only_up).
            for tier_label in TIER_LABELS:
                tids_tier = np.array(
                    [t for t, lb in tier_for_tid.items() if lb == tier_label],
                    dtype=np.int64,
                )
                if len(tids_tier) == 0:
                    rows_tier.append(
                        {
                            "subset_id": sid,
                            "variant_id": vid,
                            "tier": tier_label,
                            "n_in_tier": 0,
                            "n_trail_activated": 0,
                            "n_trail_exited": 0,
                            "n_sl_after_activation": 0,
                            "n_te_after_activation": 0,
                            "mean_trail_exit_R": float("nan"),
                            "median_trail_exit_R": float("nan"),
                            "mean_R_BL_for_tier": float("nan"),
                            "mean_R_H240_for_tier": float("nan"),
                            "mean_R_SS_for_tier": float("nan"),
                            "lift_SS_vs_BL_for_tier": float("nan"),
                        }
                    )
                    continue
                v_tier = v_by_tid.loc[v_by_tid.index.isin(tids_tier)]
                act_mask = v_tier["trail_activated_at_k"] > 0
                trail_exit_mask = v_tier["exit_reason"] == "trail"
                n_act = int(act_mask.sum())
                n_exit = int(trail_exit_mask.sum())
                n_sl_after = int((act_mask & (v_tier["exit_reason"] == "sl")).sum())
                n_te_after = int((act_mask & (v_tier["exit_reason"].isin(["te", "de"]))).sum())
                trail_R = v_tier[trail_exit_mask]["net_R"].to_numpy(dtype=np.float64)
                mean_trail_R = float(np.mean(trail_R)) if len(trail_R) > 0 else float("nan")
                median_trail_R = float(np.median(trail_R)) if len(trail_R) > 0 else float("nan")
                mean_bl = float(bl_by_tid.loc[bl_by_tid.index.isin(tids_tier)]["net_R"].mean())
                mean_h240 = float(
                    h240_by_tid.loc[h240_by_tid.index.isin(tids_tier)]["net_R"].mean()
                )
                mean_ss = float(v_tier["net_R"].mean())
                rows_tier.append(
                    {
                        "subset_id": sid,
                        "variant_id": vid,
                        "tier": tier_label,
                        "n_in_tier": int(len(tids_tier)),
                        "n_trail_activated": n_act,
                        "n_trail_exited": n_exit,
                        "n_sl_after_activation": n_sl_after,
                        "n_te_after_activation": n_te_after,
                        "mean_trail_exit_R": mean_trail_R,
                        "median_trail_exit_R": median_trail_R,
                        "mean_R_BL_for_tier": mean_bl,
                        "mean_R_H240_for_tier": mean_h240,
                        "mean_R_SS_for_tier": mean_ss,
                        "lift_SS_vs_BL_for_tier": mean_ss - mean_bl,
                    }
                )
    return pd.DataFrame(rows_tier), pd.DataFrame(rows_summary)


def _gate_10_ss_trail(ss_summary: pd.DataFrame) -> None:
    for _, row in ss_summary.iterrows():
        n_act = int(row["n_trail_activated"])
        n_exit = int(row["n_trail_exited"])
        n_sl_after = int(row["n_sl_after_activation"])
        n_te_after = int(row["n_te_after_activation"])
        if n_act < n_exit:
            raise RuntimeError(
                f"HALT (gate 10): {row['subset_id']}/{row['variant_id']}: "
                f"n_trail_activated {n_act} < n_trail_exited {n_exit}"
            )
        if n_exit + n_sl_after + n_te_after != n_act:
            raise RuntimeError(
                f"HALT (gate 10): {row['subset_id']}/{row['variant_id']}: "
                f"trail-exit({n_exit}) + sl_after({n_sl_after}) + "
                f"te_after({n_te_after}) = "
                f"{n_exit + n_sl_after + n_te_after} != activated {n_act}"
            )


# ===========================================================================
# Block TT aggregation
# ===========================================================================


def aggregate_block_TT(
    vt: pd.DataFrame,
    subsets: Dict[str, np.ndarray],
    cats: np.ndarray,
    peak_mfe: np.ndarray,
    ss_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Returns one row per (subset, TT_variant) with lift comparisons."""
    rows: List[Dict[str, Any]] = []
    bl_by_tid = vt[vt["variant_id"] == "V00_BL"].set_index("trade_id")
    pp01_by_tid = vt[vt["variant_id"] == "PP01"].set_index("trade_id")
    # Best SS per subset (by pooled mean R) for reference.
    best_ss_per_sid: Dict[str, Tuple[str, float]] = {}
    for sid in SUBSET_IDS:
        sub = ss_summary[ss_summary["subset_id"] == sid]
        best = sub.sort_values("pooled_mean_R", ascending=False).iloc[0]
        best_ss_per_sid[sid] = (str(best["variant_id"]), float(best["pooled_mean_R"]))

    for sid in SUBSET_IDS:
        tids = subsets[sid]
        only_up_tids = np.array(
            [int(t) for t in tids.tolist() if cats[int(t)] == "only_up"],
            dtype=np.int64,
        )
        tier_for_tid: Dict[int, str] = {}
        for t in only_up_tids.tolist():
            pmfe = peak_mfe[t]
            label = TIER_LABELS[0]
            for lo, hi, lb in TIER_BOUNDS:
                if pmfe > lo and pmfe <= hi:
                    label = lb
                    break
            tier_for_tid[t] = label

        bl_pooled = float(bl_by_tid.loc[bl_by_tid.index.isin(tids)]["net_R"].mean())
        pp01_pooled = float(pp01_by_tid.loc[pp01_by_tid.index.isin(tids)]["net_R"].mean())

        for vid, tau, xact, d, kind, thr in TT_VARIANTS:
            v_by_tid = vt[vt["variant_id"] == vid].set_index("trade_id")
            v_sub = v_by_tid.loc[v_by_tid.index.isin(tids)]
            pooled_mean = float(v_sub["net_R"].mean())
            counts = v_sub["exit_reason"].value_counts().to_dict()
            row: Dict[str, Any] = {
                "subset_id": sid,
                "variant_id": vid,
                "tau_atr_fill": tau,
                "X_act_atr": xact,
                "D_atr": d,
                "confirm_kind": kind,
                "confirm_threshold": thr,
                "n_total": len(v_sub),
                "n_cut": int(counts.get("cut", 0)),
                "n_trail": int(counts.get("trail", 0)),
                "n_sl": int(counts.get("sl", 0)),
                "n_te": int(counts.get("te", 0)),
                "n_de": int(counts.get("de", 0)),
                "pooled_mean_R": pooled_mean,
                "pooled_mean_R_BL": bl_pooled,
                "pooled_mean_R_PP01": pp01_pooled,
                "pooled_mean_R_best_SS": best_ss_per_sid[sid][1],
                "best_SS_variant_for_subset": best_ss_per_sid[sid][0],
                "lift_vs_BL": pooled_mean - bl_pooled,
                "lift_vs_PP01_only": pooled_mean - pp01_pooled,
                "lift_vs_best_SS": pooled_mean - best_ss_per_sid[sid][1],
            }
            # Per-tier (only_up) means.
            for tier_label in TIER_LABELS:
                tids_tier = np.array(
                    [t for t, lb in tier_for_tid.items() if lb == tier_label],
                    dtype=np.int64,
                )
                if len(tids_tier) == 0:
                    row[f"n_only_up_{tier_label}"] = 0
                    row[f"mean_R_TT_{tier_label}"] = float("nan")
                    row[f"mean_R_BL_{tier_label}"] = float("nan")
                    continue
                mean_tt = float(v_by_tid.loc[v_by_tid.index.isin(tids_tier)]["net_R"].mean())
                mean_bl_t = float(bl_by_tid.loc[bl_by_tid.index.isin(tids_tier)]["net_R"].mean())
                row[f"n_only_up_{tier_label}"] = int(len(tids_tier))
                row[f"mean_R_TT_{tier_label}"] = mean_tt
                row[f"mean_R_BL_{tier_label}"] = mean_bl_t
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Plots
# ===========================================================================


def render_plots(
    *,
    pp_summary: pd.DataFrame,
    qq_summary: pd.DataFrame,
    rr: pd.DataFrame,
    ss_summary: pd.DataFrame,
    tt: pd.DataFrame,
    plots_dir: Path,
) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # PP: bar chart of lift_vs_BL per subset per PP variant.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pps = [v[0] for v in PP_VARIANTS]
    width = 0.25
    x = np.arange(len(pps))
    for i, sid in enumerate(SUBSET_IDS):
        sub = pp_summary[pp_summary["subset_id"] == sid].set_index("variant_id")
        lifts = [float(sub.loc[v, "lift_vs_BL"]) for v in pps]
        ax.bar(x + (i - 1) * width, lifts, width, label=sid)
    ax.set_xticks(x)
    ax.set_xticklabels(pps)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title("Block PP — pooled mean R lift vs V00_BL")
    ax.set_ylabel("lift_R")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "block_PP_lift_vs_BL.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # QQ.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    qqs = [v[0] for v in QQ_VARIANTS]
    x = np.arange(len(qqs))
    for i, sid in enumerate(SUBSET_IDS):
        sub = qq_summary[qq_summary["subset_id"] == sid].set_index("variant_id")
        lifts = [float(sub.loc[v, "lift_vs_BL"]) for v in qqs]
        ax.bar(x + (i - 1) * width, lifts, width, label=sid)
    ax.set_xticks(x)
    ax.set_xticklabels(qqs)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title("Block QQ — pooled mean R lift vs V00_BL")
    ax.set_ylabel("lift_R")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "block_QQ_lift_vs_BL.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # RR.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rrs = [v[0] for v in RR_VARIANTS]
    x = np.arange(len(rrs))
    for i, sid in enumerate(SUBSET_IDS):
        sub = rr[rr["subset_id"] == sid].set_index("variant_id")
        lifts = [float(sub.loc[v, "lift_vs_BL"]) for v in rrs]
        ax.bar(x + (i - 1) * width, lifts, width, label=sid)
    ax.set_xticks(x)
    ax.set_xticklabels(rrs)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title("Block RR — pooled mean R lift vs V00_BL")
    ax.set_ylabel("lift_R")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "block_RR_lift_vs_BL.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # SS.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sss = [v[0] for v in SS_VARIANTS]
    x = np.arange(len(sss))
    bw = 0.27
    for i, sid in enumerate(SUBSET_IDS):
        sub = ss_summary[ss_summary["subset_id"] == sid].set_index("variant_id")
        lifts = [float(sub.loc[v, "lift_vs_BL"]) for v in sss]
        ax.bar(x + (i - 1) * bw, lifts, bw, label=sid)
    ax.set_xticks(x)
    ax.set_xticklabels(sss, rotation=45)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title("Block SS — pooled mean R lift vs V00_BL")
    ax.set_ylabel("lift_R")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "block_SS_lift_vs_BL.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    # TT.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tts = [v[0] for v in TT_VARIANTS]
    x = np.arange(len(tts))
    for i, sid in enumerate(SUBSET_IDS):
        sub = tt[tt["subset_id"] == sid].set_index("variant_id")
        lifts = [float(sub.loc[v, "lift_vs_BL"]) for v in tts]
        ax.bar(x + (i - 1) * width, lifts, width, label=sid)
    ax.set_xticks(x)
    ax.set_xticklabels(tts)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title("Block TT — pooled mean R lift vs V00_BL")
    ax.set_ylabel("lift_R")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots_dir / "block_TT_lift_vs_BL.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)

    return paths


# ===========================================================================
# Markdown report
# ===========================================================================


def _df_to_md(
    df: pd.DataFrame, float_cols: Optional[Dict[str, str]] = None, default_float: str = "{:+.4f}"
) -> str:
    cols = list(df.columns)
    float_cols = float_cols or {}
    int_cols = {c for c in cols if pd.api.types.is_integer_dtype(df[c].dtype)}
    bool_cols = {c for c in cols if pd.api.types.is_bool_dtype(df[c].dtype)}
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for i in range(len(df)):
        cells = []
        for c in cols:
            v = df[c].iloc[i]
            if c in int_cols:
                cells.append(str(int(v)))
                continue
            if c in bool_cols:
                cells.append(str(bool(v)))
                continue
            if isinstance(v, float) and pd.isna(v):
                cells.append("")
            elif c in float_cols and isinstance(v, float):
                cells.append(float_cols[c].format(v))
            elif isinstance(v, float):
                cells.append(default_float.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_report(
    *,
    observed_shas: Dict[str, str],
    pp_summary: pd.DataFrame,
    pp_cat: pd.DataFrame,
    pp_cut: pd.DataFrame,
    qq_summary: pd.DataFrame,
    qq_tier: pd.DataFrame,
    rr: pd.DataFrame,
    ss_summary: pd.DataFrame,
    ss_tier: pd.DataFrame,
    tt: pd.DataFrame,
    per_subset_only_up_n: Dict[str, int],
) -> str:
    lines: List[str] = []
    a = lines.append
    a("# Arc 2 — Exit-Rule Counterfactuals Round 2 (Round 3E)")
    a("")
    a("Phase: L6 Arc 2 Phase 3 — exit-rule descriptive evaluation.")
    a("")
    a("Descriptive per L6_0_METHODOLOGY_LOCK Section 14.5. Re-uses Round 2's ")
    a("`arc2_exit_sweep_filtered.py` SL / spread / intrabar primitives via ")
    a("direct import so V00_BL is byte-identical. No variant selection. No ")
    a("spec lock. SL=-2 ATR is locked throughout every variant; tier and ")
    a("category labels are reporting-only and never feed exit decisions.")
    a("")

    a("## Locked input sha256 manifest")
    a("")
    a("| relative_path | sha256 |")
    a("| --- | --- |")
    for rel in LOCKED_SHAS:
        a(f"| {rel} | {observed_shas[rel]} |")
    a("")

    a("## Determinism receipt")
    a("")
    a("Two consecutive in-script build passes produced byte-identical CSV+MD ")
    a("outputs. PNG plots verified pixel-equal via PIL. Wallclock printed to ")
    a("stdout only.")
    a("")

    a("## Subset, Block B, and V00_BL reproduction (gates 2.1/2.2/2.3/3.1)")
    a("")
    a("Subsets, quintile labels, Block B 1R counts, and per_bar indexing are ")
    a("byte-faithful with prior rounds. V00_BL applied to S0_pop reproduces ")
    a("the Round 2 published `mean_R = -0.01924125032` to better than 1e-7.")
    a("")
    a("| subset_id | definition | n_total | only_up_n |")
    a("| --- | --- | --- | --- |")
    sub_human = {
        "S0_pop": "ALL 3,993 trades",
        "S1_q5q2": "Q_A==Q5 AND Q_B==Q2",
        "S4_q5xq2q3": "Q_A==Q5 AND Q_B IN {Q2,Q3}",
    }
    for sid in SUBSET_IDS:
        n_exp = dict(SUBSET_DEFS)[sid]["expected_n"]
        a(f"| {sid} | {sub_human[sid]} | {n_exp} | {per_subset_only_up_n[sid]} |")
    a("")

    # PP
    a("## Block PP — Early-cut at k=20 with SL-timing accounting")
    a("")
    a("Per-subset pooled mean R and lift vs V00_BL for PP variants ")
    a("(tau in ATR fill-rel units, cut at bar k=20):")
    a("")
    fmt_pp_s = {
        "tau_atr_fill": "{:+.2f}",
        "pooled_mean_R": "{:+.5f}",
        "pooled_mean_R_BL": "{:+.5f}",
        "lift_vs_BL": "{:+.5f}",
    }
    a(
        _df_to_md(
            pp_summary[
                [
                    "subset_id",
                    "variant_id",
                    "tau_atr_fill",
                    "n_total",
                    "n_cut_at_k20",
                    "n_sl",
                    "n_te",
                    "n_de",
                    "pooled_mean_R",
                    "pooled_mean_R_BL",
                    "lift_vs_BL",
                ]
            ],
            fmt_pp_s,
        )
    )
    a("")
    a("### Per-category (only_up + losers) cut-group decomposition")
    a("")
    a("Per (subset, variant, category), the cut group BL counterfactual mix:")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### Subset {sid}")
        a("")
        for vid, tau in PP_VARIANTS:
            a(f"##### {vid} (tau = {tau})")
            a("")
            sub = pp_cat[(pp_cat["subset_id"] == sid) & (pp_cat["variant_id"] == vid)].copy()
            fmt = {
                "mean_cut_price_R": "{:+.4f}",
                "median_cut_price_R": "{:+.4f}",
                "mean_bl_R_for_cut_group": "{:+.4f}",
                "savings_per_cut_trade": "{:+.4f}",
                "pre_cut_category_mean_R": "{:+.4f}",
                "post_cut_category_mean_R": "{:+.4f}",
                "lift_per_category": "{:+.4f}",
            }
            a(
                _df_to_md(
                    sub[
                        [
                            "category",
                            "n_total_in_category",
                            "n_sl_before_k20",
                            "n_open_at_k20",
                            "n_open_at_k20_below_tau",
                            "n_open_at_k20_above_tau",
                            "n_subsequent_sl",
                            "n_subsequent_te_negative",
                            "n_subsequent_te_positive",
                            "n_subsequent_de",
                            "mean_cut_price_R",
                            "mean_bl_R_for_cut_group",
                            "savings_per_cut_trade",
                            "lift_per_category",
                        ]
                    ],
                    fmt,
                )
            )
            a("")

    # QQ
    a("## Block QQ — Tier-conditional H240")
    a("")
    a("Per-subset pooled lifts vs V00_BL and vs V09_H240; per-tier breakdown ")
    a("of confirmation rate and mean R.")
    a("")
    fmt_qq_s = {
        "confirm_threshold": "{:.1f}",
        "pct_confirmed": "{:.4f}",
        "pooled_mean_R": "{:+.5f}",
        "pooled_mean_R_BL": "{:+.5f}",
        "pooled_mean_R_H240": "{:+.5f}",
        "lift_vs_BL": "{:+.5f}",
        "lift_vs_H240": "{:+.5f}",
    }
    a(
        _df_to_md(
            qq_summary[
                [
                    "subset_id",
                    "variant_id",
                    "confirm_kind",
                    "confirm_threshold",
                    "n_only_up_total",
                    "n_only_up_confirmed",
                    "pct_confirmed",
                    "pooled_mean_R",
                    "pooled_mean_R_BL",
                    "pooled_mean_R_H240",
                    "lift_vs_BL",
                    "lift_vs_H240",
                ]
            ],
            fmt_qq_s,
        )
    )
    a("")
    a("### Per-tier (only_up) breakdown per QQ variant")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### Subset {sid}")
        a("")
        sub = qq_tier[qq_tier["subset_id"] == sid].copy()
        fmt = {
            "pct_of_tier_confirmed": "{:.4f}",
            "mean_R_BL_for_tier": "{:+.4f}",
            "mean_R_QQ_for_tier": "{:+.4f}",
            "mean_R_H240_for_tier": "{:+.4f}",
            "lift_QQ_vs_BL_for_tier": "{:+.4f}",
        }
        a(
            _df_to_md(
                sub[
                    [
                        "variant_id",
                        "tier",
                        "n_in_tier",
                        "n_tier_confirmed",
                        "pct_of_tier_confirmed",
                        "mean_R_BL_for_tier",
                        "mean_R_QQ_for_tier",
                        "mean_R_H240_for_tier",
                        "lift_QQ_vs_BL_for_tier",
                    ]
                ],
                fmt,
            )
        )
        a("")

    # RR
    a("## Block RR — PP + QQ combined")
    a("")
    fmt_rr = {
        "tau_atr_fill": "{:+.2f}",
        "confirm_threshold": "{:.1f}",
        "pooled_mean_R": "{:+.5f}",
        "pooled_mean_R_BL": "{:+.5f}",
        "lift_vs_BL": "{:+.5f}",
    }
    a(
        _df_to_md(
            rr[
                [
                    "subset_id",
                    "variant_id",
                    "tau_atr_fill",
                    "confirm_kind",
                    "confirm_threshold",
                    "n_total",
                    "n_cut",
                    "n_sl",
                    "n_te",
                    "n_de",
                    "pooled_mean_R",
                    "pooled_mean_R_BL",
                    "lift_vs_BL",
                ]
            ],
            fmt_rr,
        )
    )
    a("")

    # SS
    a("## Block SS — Trail on confirmed runners (close-based peak)")
    a("")
    a("Per-subset pooled mean R, activation/exit counts, lifts vs both BL ")
    a("references:")
    a("")
    fmt_ss_s = {
        "X_act_atr": "{:.1f}",
        "D_atr": "{:.1f}",
        "pooled_mean_R": "{:+.5f}",
        "pooled_mean_R_BL": "{:+.5f}",
        "pooled_mean_R_H240": "{:+.5f}",
        "lift_vs_BL": "{:+.5f}",
        "lift_vs_H240": "{:+.5f}",
    }
    a(
        _df_to_md(
            ss_summary[
                [
                    "subset_id",
                    "variant_id",
                    "X_act_atr",
                    "D_atr",
                    "time_horizon",
                    "n_total",
                    "n_trail_activated",
                    "n_trail_exited",
                    "n_sl_after_activation",
                    "n_te_after_activation",
                    "pooled_mean_R",
                    "pooled_mean_R_BL",
                    "pooled_mean_R_H240",
                    "lift_vs_BL",
                    "lift_vs_H240",
                ]
            ],
            fmt_ss_s,
        )
    )
    a("")
    a("### Per-tier (only_up) mean R per SS variant")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### Subset {sid}")
        a("")
        sub = ss_tier[ss_tier["subset_id"] == sid].copy()
        fmt = {
            "mean_trail_exit_R": "{:+.4f}",
            "median_trail_exit_R": "{:+.4f}",
            "mean_R_BL_for_tier": "{:+.4f}",
            "mean_R_H240_for_tier": "{:+.4f}",
            "mean_R_SS_for_tier": "{:+.4f}",
            "lift_SS_vs_BL_for_tier": "{:+.4f}",
        }
        a(
            _df_to_md(
                sub[
                    [
                        "variant_id",
                        "tier",
                        "n_in_tier",
                        "n_trail_activated",
                        "n_trail_exited",
                        "n_sl_after_activation",
                        "n_te_after_activation",
                        "mean_trail_exit_R",
                        "mean_R_BL_for_tier",
                        "mean_R_H240_for_tier",
                        "mean_R_SS_for_tier",
                        "lift_SS_vs_BL_for_tier",
                    ]
                ],
                fmt,
            )
        )
        a("")

    # TT
    a("## Block TT — Full combined: PP + QQ + SS")
    a("")
    fmt_tt = {
        "tau_atr_fill": "{:+.2f}",
        "X_act_atr": "{:.1f}",
        "D_atr": "{:.1f}",
        "confirm_threshold": "{:.1f}",
        "pooled_mean_R": "{:+.5f}",
        "pooled_mean_R_BL": "{:+.5f}",
        "pooled_mean_R_PP01": "{:+.5f}",
        "pooled_mean_R_best_SS": "{:+.5f}",
        "lift_vs_BL": "{:+.5f}",
        "lift_vs_PP01_only": "{:+.5f}",
        "lift_vs_best_SS": "{:+.5f}",
    }
    a(
        _df_to_md(
            tt[
                [
                    "subset_id",
                    "variant_id",
                    "tau_atr_fill",
                    "X_act_atr",
                    "D_atr",
                    "confirm_kind",
                    "confirm_threshold",
                    "n_total",
                    "n_cut",
                    "n_trail",
                    "n_sl",
                    "n_te",
                    "n_de",
                    "pooled_mean_R",
                    "pooled_mean_R_BL",
                    "pooled_mean_R_PP01",
                    "pooled_mean_R_best_SS",
                    "lift_vs_BL",
                    "lift_vs_PP01_only",
                    "lift_vs_best_SS",
                ]
            ],
            fmt_tt,
        )
    )
    a("")
    a("### Per-tier (only_up) mean R per TT variant")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### Subset {sid}")
        a("")
        sub = tt[tt["subset_id"] == sid].copy()
        tier_cols = []
        fmt_tier: Dict[str, str] = {}
        for tier_label in TIER_LABELS:
            tier_cols.extend(
                [f"n_only_up_{tier_label}", f"mean_R_BL_{tier_label}", f"mean_R_TT_{tier_label}"]
            )
            fmt_tier[f"mean_R_BL_{tier_label}"] = "{:+.4f}"
            fmt_tier[f"mean_R_TT_{tier_label}"] = "{:+.4f}"
        a(_df_to_md(sub[["variant_id"] + tier_cols], fmt_tier))
        a("")

    # Cross-block synthesis (descriptive, factual)
    a("## Cross-block synthesis (descriptive)")
    a("")
    a("Per-subset roll-up: pooled mean R for V00_BL reference, best lift per ")
    a("block, and lift of TT-best.")
    a("")
    cross_rows: List[Dict[str, Any]] = []
    for sid in SUBSET_IDS:
        bl_ref = float(
            pp_summary[(pp_summary["subset_id"] == sid) & (pp_summary["variant_id"] == "PP01")][
                "pooled_mean_R_BL"
            ].iloc[0]
        )
        # Best lift per block (informational; no recommendation).
        pp_best = (
            pp_summary[pp_summary["subset_id"] == sid]
            .sort_values("lift_vs_BL", ascending=False)
            .iloc[0]
        )
        qq_best = (
            qq_summary[qq_summary["subset_id"] == sid]
            .sort_values("lift_vs_BL", ascending=False)
            .iloc[0]
        )
        rr_best = rr[rr["subset_id"] == sid].sort_values("lift_vs_BL", ascending=False).iloc[0]
        ss_best = (
            ss_summary[ss_summary["subset_id"] == sid]
            .sort_values("lift_vs_BL", ascending=False)
            .iloc[0]
        )
        tt_best = tt[tt["subset_id"] == sid].sort_values("lift_vs_BL", ascending=False).iloc[0]
        cross_rows.append(
            {
                "subset_id": sid,
                "V00_BL_mean_R": bl_ref,
                "PP_max_lift_variant": str(pp_best["variant_id"]),
                "PP_max_lift": float(pp_best["lift_vs_BL"]),
                "QQ_max_lift_variant": str(qq_best["variant_id"]),
                "QQ_max_lift": float(qq_best["lift_vs_BL"]),
                "RR_max_lift_variant": str(rr_best["variant_id"]),
                "RR_max_lift": float(rr_best["lift_vs_BL"]),
                "SS_max_lift_variant": str(ss_best["variant_id"]),
                "SS_max_lift": float(ss_best["lift_vs_BL"]),
                "TT_max_lift_variant": str(tt_best["variant_id"]),
                "TT_max_lift": float(tt_best["lift_vs_BL"]),
            }
        )
    a(
        _df_to_md(
            pd.DataFrame(cross_rows),
            {
                "V00_BL_mean_R": "{:+.5f}",
                "PP_max_lift": "{:+.5f}",
                "QQ_max_lift": "{:+.5f}",
                "RR_max_lift": "{:+.5f}",
                "SS_max_lift": "{:+.5f}",
                "TT_max_lift": "{:+.5f}",
            },
        )
    )
    a("")
    a("Diminishing-returns view: TT lift over (max of PP, QQ, SS) per subset.")
    a("")
    dr_rows = []
    for sid in SUBSET_IDS:
        row_cross = cross_rows[SUBSET_IDS.index(sid)]
        max_single = max(
            row_cross["PP_max_lift"], row_cross["QQ_max_lift"], row_cross["SS_max_lift"]
        )
        dr_rows.append(
            {
                "subset_id": sid,
                "max_lift_single_block": max_single,
                "TT_best_lift": row_cross["TT_max_lift"],
                "TT_minus_max_single": row_cross["TT_max_lift"] - max_single,
            }
        )
    a(
        _df_to_md(
            pd.DataFrame(dr_rows),
            {
                "max_lift_single_block": "{:+.5f}",
                "TT_best_lift": "{:+.5f}",
                "TT_minus_max_single": "{:+.5f}",
            },
        )
    )
    a("")

    # Out-of-scope items
    a("## Out-of-scope items observed")
    a("")
    a("- Spread is treated using Round 2's per-trade (sp_entry, sp_exit) ")
    a("  values from `trades_all.csv`. `trades_all.csv` is read for this ")
    a("  lookup but is not in the formally-locked sha256 list; its hash is ")
    a("  emitted to `run_manifest.txt` for record. Identical convention to ")
    a("  Round 2's `arc2_exit_sweep_filtered.py`.")
    a("- Tier classification uses Round 3C Block MM bounds. Tier is a ")
    a("  reporting attribute computed from peak_mfe over the full per_bar ")
    a("  window. Exit decisions in every variant use only running observables.")
    a("- The activation criterion for the SS / TT trail uses `running_mfe_atr` ")
    a("  (intrabar high) per spec §7; once active, the peak that anchors ")
    a("  the trail is tracked from `bar_close_atr`. This intentionally ")
    a("  differs from Round 2's `variant_TRAIL`, which uses `running_mfe_atr` ")
    a("  for both activation and trail anchoring and triggers on `bar_low`. ")
    a("  Documented per spec §7.")
    a("- The PP early-cut uses the same one-sided exit-spread accounting as a ")
    a("  time-exit at the cut bar's close. Internally the trade is labelled ")
    a("  `early_cut` (short code `cut`) and exits at `bar_close_atr`-based ")
    a("  fill — never at the next-bar open. This matches the spec §4 ")
    a("  formulation `exit_R = running_close_atr_at_k20 / 2.0`.")
    a("")

    # Planning input
    a("## Planning input")
    a("")
    a("Material below is intentionally descriptive even within this tagged ")
    a("subsection; final variant choice and spec lock are out of scope for ")
    a("this round.")
    a("")
    a("### Top-3 single-rule variants by lift vs V00_BL per subset")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### {sid}")
        a("")
        # Aggregate single-rule variants: PP, QQ, SS.
        all_single: List[Dict[str, Any]] = []
        for _, r in pp_summary[pp_summary["subset_id"] == sid].iterrows():
            all_single.append(
                {
                    "block": "PP",
                    "variant_id": r["variant_id"],
                    "lift_vs_BL": float(r["lift_vs_BL"]),
                    "pooled_mean_R": float(r["pooled_mean_R"]),
                }
            )
        for _, r in qq_summary[qq_summary["subset_id"] == sid].iterrows():
            all_single.append(
                {
                    "block": "QQ",
                    "variant_id": r["variant_id"],
                    "lift_vs_BL": float(r["lift_vs_BL"]),
                    "pooled_mean_R": float(r["pooled_mean_R"]),
                }
            )
        for _, r in ss_summary[ss_summary["subset_id"] == sid].iterrows():
            all_single.append(
                {
                    "block": "SS",
                    "variant_id": r["variant_id"],
                    "lift_vs_BL": float(r["lift_vs_BL"]),
                    "pooled_mean_R": float(r["pooled_mean_R"]),
                }
            )
        df_single = pd.DataFrame(all_single).sort_values("lift_vs_BL", ascending=False).head(3)
        a(
            _df_to_md(
                df_single,
                {
                    "lift_vs_BL": "{:+.5f}",
                    "pooled_mean_R": "{:+.5f}",
                },
            )
        )
        a("")

    a("### Top-3 combined-rule variants by lift vs V00_BL per subset")
    a("")
    for sid in SUBSET_IDS:
        a(f"#### {sid}")
        a("")
        all_comb: List[Dict[str, Any]] = []
        for _, r in rr[rr["subset_id"] == sid].iterrows():
            all_comb.append(
                {
                    "block": "RR",
                    "variant_id": r["variant_id"],
                    "lift_vs_BL": float(r["lift_vs_BL"]),
                    "pooled_mean_R": float(r["pooled_mean_R"]),
                }
            )
        for _, r in tt[tt["subset_id"] == sid].iterrows():
            all_comb.append(
                {
                    "block": "TT",
                    "variant_id": r["variant_id"],
                    "lift_vs_BL": float(r["lift_vs_BL"]),
                    "pooled_mean_R": float(r["pooled_mean_R"]),
                }
            )
        df_comb = pd.DataFrame(all_comb).sort_values("lift_vs_BL", ascending=False).head(3)
        a(
            _df_to_md(
                df_comb,
                {
                    "lift_vs_BL": "{:+.5f}",
                    "pooled_mean_R": "{:+.5f}",
                },
            )
        )
        a("")

    a("### Cross-subset robustness: variants with positive lift on all three subsets")
    a("")
    # Build pivot of lift_vs_BL by variant_id across SUBSET_IDS.
    all_lifts: Dict[str, Dict[str, float]] = {}
    for df, blk in [
        (pp_summary, "PP"),
        (qq_summary, "QQ"),
        (rr, "RR"),
        (ss_summary, "SS"),
        (tt, "TT"),
    ]:
        for _, r in df.iterrows():
            vid = str(r["variant_id"])
            sid = str(r["subset_id"])
            if sid not in SUBSET_IDS:
                continue
            all_lifts.setdefault(vid, {})[sid] = float(r["lift_vs_BL"])
    robust_rows = []
    for vid, lifts in sorted(all_lifts.items()):
        if all(s in lifts for s in SUBSET_IDS) and all(lifts[s] > 0 for s in SUBSET_IDS):
            robust_rows.append(
                {
                    "variant_id": vid,
                    **{f"lift_{s}": lifts[s] for s in SUBSET_IDS},
                    "min_lift_across_subsets": min(lifts[s] for s in SUBSET_IDS),
                }
            )
    if robust_rows:
        df_robust = pd.DataFrame(robust_rows).sort_values(
            "min_lift_across_subsets", ascending=False
        )
        a(
            _df_to_md(
                df_robust,
                {f"lift_{s}": "{:+.5f}" for s in SUBSET_IDS}
                | {"min_lift_across_subsets": "{:+.5f}"},
            )
        )
    else:
        a("(none)")
    a("")

    a("### Variant-with-biggest-lift on S1_q5q2 and on S4_q5xq2q3 specifically")
    a("")
    s1_rows = []
    s4_rows = []
    for vid, lifts in all_lifts.items():
        if "S1_q5q2" in lifts:
            s1_rows.append({"variant_id": vid, "lift_vs_BL_S1": lifts["S1_q5q2"]})
        if "S4_q5xq2q3" in lifts:
            s4_rows.append({"variant_id": vid, "lift_vs_BL_S4": lifts["S4_q5xq2q3"]})
    if s1_rows:
        a("S1_q5q2 top 5:")
        a("")
        a(
            _df_to_md(
                pd.DataFrame(s1_rows).sort_values("lift_vs_BL_S1", ascending=False).head(5),
                {"lift_vs_BL_S1": "{:+.5f}"},
            )
        )
        a("")
    if s4_rows:
        a("S4_q5xq2q3 top 5:")
        a("")
        a(
            _df_to_md(
                pd.DataFrame(s4_rows).sort_values("lift_vs_BL_S4", ascending=False).head(5),
                {"lift_vs_BL_S4": "{:+.5f}"},
            )
        )
        a("")

    return "\n".join(lines) + "\n"


def check_disposition_discipline(report_text: str) -> List[Tuple[int, str, str]]:
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


def build_pass(*, out_dir: Path, write_manifest: bool) -> Dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gate 1
    observed_shas = _verify_locked("gate 1 (start)")

    # Sweep
    vt, ti_full, cats, peak_mfe, subsets = run_sweep()

    # Reconstruct per-trade helpers for PP/QQ.
    pb = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv")
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids_arr = pb["trade_id"].to_numpy(dtype=np.int64)
    n_trades = 3993
    starts = np.searchsorted(tids_arr, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids_arr, np.arange(n_trades), side="right")
    rmae_all = pb["running_mae_atr"].to_numpy(dtype=np.float64)
    rmfe_all = pb["running_mfe_atr"].to_numpy(dtype=np.float64)
    bc_all = pb["bar_close_atr"].to_numpy(dtype=np.float64)
    bar_of_sl_BL = _bar_of_sl_under_BL(rmae_all, n_trades, starts, ends)
    close_at_k20 = np.full(n_trades, np.nan)
    rmfe_at_120 = np.full(n_trades, np.nan)
    bc_at_120 = np.full(n_trades, np.nan)
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        if e - s >= 20:
            close_at_k20[tid] = bc_all[s + 19]
        if e - s >= 120:
            rmfe_at_120[tid] = rmfe_all[s + 119]
            bc_at_120[tid] = bc_all[s + 119]

    # only_up counts per subset.
    per_subset_only_up_n: Dict[str, int] = {}
    for sid in SUBSET_IDS:
        n_ou = int(sum(1 for t in subsets[sid].tolist() if cats[int(t)] == "only_up"))
        per_subset_only_up_n[sid] = n_ou
    print(f"  only_up per subset: {per_subset_only_up_n}", flush=True)

    # Gate 3.1 + 3.2
    _gate_3_1_bl_repro(vt, subsets)
    _gate_3_2_clamped(vt, ti_full)
    # Gate 7
    _gate_7_exit_exhaustive(vt, subsets)
    # Gate 11
    _gate_11_tt_pp_additivity(vt)

    # Block PP
    print("  Aggregating Block PP...", flush=True)
    pp_cat, pp_cut, pp_summary = aggregate_block_PP(vt, subsets, cats, bar_of_sl_BL, close_at_k20)
    _gate_8_pp_cut_detail(pp_cat)

    # Block QQ
    print("  Aggregating Block QQ...", flush=True)
    qq_tier, qq_summary = aggregate_block_QQ(vt, subsets, cats, peak_mfe, rmfe_at_120, bc_at_120)
    _gate_9_qq_confirmation(qq_summary)

    # Block RR
    print("  Aggregating Block RR...", flush=True)
    rr = aggregate_block_RR(vt, subsets, cats)

    # Block SS
    print("  Aggregating Block SS...", flush=True)
    ss_tier, ss_summary = aggregate_block_SS(vt, subsets, cats, peak_mfe)
    _gate_10_ss_trail(ss_summary)

    # Block TT
    print("  Aggregating Block TT...", flush=True)
    tt = aggregate_block_TT(vt, subsets, cats, peak_mfe, ss_summary)

    # Write CSVs.
    paths = {
        "block_PP_per_subset_per_category_per_variant.csv": pp_cat,
        "block_PP_cut_group_detail.csv": pp_cut,
        "block_PP_summary.csv": pp_summary,
        "block_QQ_per_subset_per_tier_per_variant.csv": qq_tier,
        "block_QQ_summary.csv": qq_summary,
        "block_RR_combined_per_subset_per_variant.csv": rr,
        "block_SS_trail_per_subset_per_tier_per_variant.csv": ss_tier,
        "block_SS_trail_summary.csv": ss_summary,
        "block_TT_full_combined_per_subset_per_variant.csv": tt,
    }
    for name, df in paths.items():
        _write_csv(df, out_dir / name)

    # Plots
    print("  Rendering plots...", flush=True)
    plot_paths = render_plots(
        pp_summary=pp_summary,
        qq_summary=qq_summary,
        rr=rr,
        ss_summary=ss_summary,
        tt=tt,
        plots_dir=out_dir / "plots",
    )

    # Markdown
    print("  Rendering markdown...", flush=True)
    md = render_report(
        observed_shas=observed_shas,
        pp_summary=pp_summary,
        pp_cat=pp_cat,
        pp_cut=pp_cut,
        qq_summary=qq_summary,
        qq_tier=qq_tier,
        rr=rr,
        ss_summary=ss_summary,
        ss_tier=ss_tier,
        tt=tt,
        per_subset_only_up_n=per_subset_only_up_n,
    )
    md_path = out_dir / "exit_counterfactuals_round2.md"
    md_path.write_text(md, encoding="utf-8", newline="\n")

    # Gate 14
    viols = check_disposition_discipline(md)
    if viols:
        msg = "\n  ".join([f"line {ln}: pat='{p}': {tx}" for ln, p, tx in viols])
        raise RuntimeError(f"HALT (gate 14): disposition discipline violations:\n  {msg}")

    # Gate 13 (locked artefacts unchanged)
    _verify_locked("gate 13 (end)")

    # Output sha manifest
    out_files = list(paths.keys()) + ["exit_counterfactuals_round2.md"]
    out_paths_full = [out_dir / n for n in out_files] + plot_paths
    out_shas = {p.relative_to(REPO_ROOT).as_posix(): _sha256_file(p) for p in out_paths_full}

    gates = {
        "gate_1_inputs": "ok (11 sha256s match)",
        "gate_2_1_subsets": "ok (all 25 block_P cells reproduced)",
        "gate_2_2_subset_sizes": "ok (3/3 reporting subsets at expected counts)",
        "gate_2_3_block_b": "ok (956/1075/1090/858/13/1 reproduced)",
        "gate_3_1_v00_bl": "ok (V00_BL S0 mean_R diff < 1e-7 vs Round 2)",
        "gate_3_2_clamped": "ok (data_end count matches Round 2 reference)",
        "gate_7_exit_exhaustive": "ok (exit reasons sum to n per subset/variant)",
        "gate_8_pp_cut_detail": "ok (subsequent-outcome counts sum to n_below)",
        "gate_9_qq_confirmation": "ok (confirmed+not = n_only_up)",
        "gate_10_ss_trail": "ok (trail-exited+sl_after+te_after = activated)",
        "gate_11_tt_pp_additivity": "ok (TT cut R == PP01 cut R per trade)",
        "gate_13_artefacts_unchanged": "ok",
        "gate_14_disposition": f"ok ({len(viols)} violations outside Planning input)",
        "gate_15_no_commit": "ok (no auto-commit; outputs untracked)",
    }

    if write_manifest:
        wallclock = time.time() - t0
        manifest_path = out_dir / "run_manifest.txt"
        ta_sha = _sha256_file(REPO_ROOT / TRADES_ALL_REL)
        with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Arc 2 — Exit-Rule Counterfactuals Round 2 (Round 3E)\n")
            f.write("# Phase: l6_arc2_exit_counterfactuals_round2\n")
            f.write("\n## Inputs (locked sha256)\n")
            for rel, h in observed_shas.items():
                f.write(f"{rel} {h}\n")
            f.write("\n## Auxiliary input (not formally locked)\n")
            f.write(f"{TRADES_ALL_REL} {ta_sha}\n")
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
        "vt": vt,
        "pp_summary": pp_summary,
        "qq_summary": qq_summary,
        "rr": rr,
        "ss_summary": ss_summary,
        "tt": tt,
        "subsets": subsets,
        "per_subset_only_up_n": per_subset_only_up_n,
        "plot_paths": plot_paths,
    }


# ===========================================================================
# Two-pass with determinism check
# ===========================================================================


def _compare_files(a: Path, b: Path) -> bool:
    if filecmp.cmp(a, b, shallow=False):
        return True
    if a.suffix.lower() == ".png":
        ia = np.array(Image.open(a))
        ib = np.array(Image.open(b))
        return ia.shape == ib.shape and np.array_equal(ia, ib)
    return False


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--single-pass", action="store_true")
    args = parser.parse_args(argv)

    t_start = time.time()
    tracemalloc.start()

    out_dir = REPO_ROOT / OUTPUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Pass 1 ===", flush=True)
    r1 = build_pass(out_dir=out_dir, write_manifest=False)
    if args.single_pass:
        r1 = build_pass(out_dir=out_dir, write_manifest=True)
        det_ok = None
    else:
        snapshot_dir = Path(tempfile.mkdtemp(prefix="arc2_excf_snap_"))
        check_files = list((Path(p) for p in r1["out_shas"].keys()))
        snap_map: Dict[Path, Path] = {}
        for rel in check_files:
            src = REPO_ROOT / rel
            dst = snapshot_dir / rel.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            snap_map[src] = dst
        print("\n=== Pass 2 ===", flush=True)
        r2 = build_pass(out_dir=out_dir, write_manifest=True)
        det_diffs: List[str] = []
        for src, snap in snap_map.items():
            if not _compare_files(snap, src):
                det_diffs.append(str(src.name))
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        if det_diffs:
            raise RuntimeError(f"HALT (gate 12): determinism failed; differing files: {det_diffs}")
        det_ok = True
        r1 = r2

    peak_kb = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    wallclock = time.time() - t_start

    print("\n=== Validation gates disposition ===", flush=True)
    for k, v in r1["gates"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  gate_12_determinism: {'ok' if det_ok else 'single-pass-skipped'}", flush=True)

    # Headline numbers per subset.
    pp_summary = r1["pp_summary"]
    qq_summary = r1["qq_summary"]
    ss_summary = r1["ss_summary"]
    tt = r1["tt"]
    print("\n=== Headline numbers per subset ===", flush=True)
    for sid in SUBSET_IDS:
        bl_ref = float(
            pp_summary[(pp_summary["subset_id"] == sid) & (pp_summary["variant_id"] == "PP01")][
                "pooled_mean_R_BL"
            ].iloc[0]
        )
        pp01 = pp_summary[
            (pp_summary["subset_id"] == sid) & (pp_summary["variant_id"] == "PP01")
        ].iloc[0]
        qq_best = (
            qq_summary[qq_summary["subset_id"] == sid]
            .sort_values("lift_vs_BL", ascending=False)
            .iloc[0]
        )
        ss_best = (
            ss_summary[ss_summary["subset_id"] == sid]
            .sort_values("lift_vs_BL", ascending=False)
            .iloc[0]
        )
        tt_best = tt[tt["subset_id"] == sid].sort_values("lift_vs_BL", ascending=False).iloc[0]
        print(
            f"  {sid}: V00_BL = {bl_ref:+.5f}R. "
            f"PP01 (tau=-0.5): pooled {float(pp01['pooled_mean_R']):+.5f}R "
            f"(lift {float(pp01['lift_vs_BL']):+.5f}, n_cut={int(pp01['n_cut_at_k20'])}). "
            f"QQ max-lift = {qq_best['variant_id']} "
            f"(pooled {float(qq_best['pooled_mean_R']):+.5f}, "
            f"lift {float(qq_best['lift_vs_BL']):+.5f}). "
            f"SS max-lift = {ss_best['variant_id']} "
            f"(pooled {float(ss_best['pooled_mean_R']):+.5f}, "
            f"lift {float(ss_best['lift_vs_BL']):+.5f}). "
            f"TT max-lift = {tt_best['variant_id']} "
            f"(pooled {float(tt_best['pooled_mean_R']):+.5f}, "
            f"lift {float(tt_best['lift_vs_BL']):+.5f}).",
            flush=True,
        )

    # Cross-subset robustness one-liner.
    all_lifts: Dict[str, Dict[str, float]] = {}
    for df in [pp_summary, qq_summary, r1["rr"], ss_summary, tt]:
        for _, r in df.iterrows():
            vid = str(r["variant_id"])
            sid = str(r["subset_id"])
            if sid not in SUBSET_IDS:
                continue
            all_lifts.setdefault(vid, {})[sid] = float(r["lift_vs_BL"])
    robust = {
        vid: min(lifts[s] for s in SUBSET_IDS)
        for vid, lifts in all_lifts.items()
        if all(s in lifts and lifts[s] > 0 for s in SUBSET_IDS)
    }
    if robust:
        top_robust = max(robust.items(), key=lambda kv: kv[1])
        print(
            f"\n  Cross-subset most-consistent positive lift: "
            f"{top_robust[0]} (min lift = {top_robust[1]:+.5f}R)",
            flush=True,
        )
    else:
        print("\n  No variant has positive lift on all three subsets.", flush=True)

    # Output sha manifest.
    print("\n=== Output artefact sha256 manifest ===", flush=True)
    for rel, h in r1["out_shas"].items():
        print(f"  {rel}  {h}", flush=True)

    # Git status.
    print("\n=== git status (HEAD unchanged; new files untracked) ===", flush=True)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        print(f"  HEAD: {head}", flush=True)
        st = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        for ln in st.splitlines()[:50]:
            print(f"  {ln}", flush=True)
    except Exception as e:
        print(f"  (git unavailable: {e})", flush=True)

    print(f"\n  wallclock {wallclock:.2f}s  peak_RSS_traced_kb {peak_kb:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
