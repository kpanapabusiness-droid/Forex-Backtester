"""Arc 2 counterfactual exit-rule sweep, round 1.

Phase: l6_arc2_counterfactual_sweep_round_1
Purpose: descriptive characterisation of 15 pre-specified exit-rule variants
(plus 1 baseline reproduction control) on the Arc 2 taken-trade set, using
the v1.2 per-bar path data.

Strictly characterisation-only per L6.0 v1.1 §14.6. No phase-2 spec is
locked here; results inform planner decisions in a separate workflow.

Inputs (all read-only, sha256-verified at run start):
- results/l6/arc2/characterisation/v1_2_full/per_bar_paths.csv
- results/l6/arc2/characterisation/v1_2_full/trade_index.csv
- results/l6/arc2/characterisation/v1_2_full/pipeline_diff_v1_2_manifest.md

Outputs (results/l6/arc2/characterisation/extended/counterfactuals/round_1/):
- variant_trades.csv         — long-format per-(variant, trade) result
- variant_summary_pooled.csv — one row per variant
- variant_summary_per_fold.csv — one row per (variant, fold)
- additivity_calibration.csv — G2 lone-vs-combination divergence
- counterfactual_sweep_round_1.md — combined synthesis report
- run_manifest.txt           — sha256 manifest + determinism receipt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
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

# --- Locked input sha256s (gate 1) ----------------------------------------
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_2_full/per_bar_paths.csv": "e1195f0dedb317f6d921d4fa9526c8aa546457f8038f28f37cd656605e6b1960",
    "results/l6/arc2/characterisation/v1_2_full/trade_index.csv": "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_2_full/pipeline_diff_v1_2_manifest.md": "f3094ffd59121bcb0864f72d8f851f99cc44b4e4354d374d5159e671b4f0d530",
}

# --- Methodology + adjacent locked artefacts (gate 13) --------------------
ADJ_LOCKED_SHAS: Dict[str, str] = {
    "L6_0_METHODOLOGY_LOCK.md": "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv": "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py": "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml": "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "scripts/lchar/arc2_characterisation_v1_1.py": "5d32627a1c4691ef654315dd5f35401d3a4e811bc20c0d48cd64a33debcb5105",
    "scripts/lchar/arc2_per_bar_paths.py": "36bb6f9b0413386bd5d25960f4525084fa93408ecb491232e17396872f1ff821",
}

# --- CANDIDATE_HYPOTHESES.md baseline (gate 15) ---------------------------
CANDIDATE_HYPOTHESES_BASELINE_SHA = (
    "8ed487620a7f9ab2c443e6520a4afa820c353480d8329d4fe91703b7d083dfbf"
)

# --- Variant grid -----------------------------------------------------------
# Each variant is (id, spec_short, fn, params).
TIME_HORIZON_DEFAULT = 120
TIME_HORIZON_EXTENDED = 240


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_input_integrity() -> Dict[str, str]:
    """Gate 1: verify the 3 v1.2 input sha256s. HALT on any mismatch."""
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
    return out


# ============================================================================
# VARIANT MECHANICS
#
# Every variant fn signature:
#   variant_fn(rmae, rmfe, bl_, bc_, bavail, spread_baseline) -> dict
# returning {exit_bar, exit_reason, gross_R, spread_cost_R, net_R}.
#
# rmae, rmfe, bl_, bc_ are 1-D float64 numpy arrays of length bavail
# (running_mae_atr, running_mfe_atr, bar_low_atr, bar_close_atr).
# All values are in ATR-distance units.
# k = k_idx + 1 throughout (bars are 1-indexed externally, 0-indexed in arrays).
# ============================================================================


def variant_BL(
    rmae, rmfe, bl_, bc_, bavail, spread_baseline, *, time_horizon=TIME_HORIZON_DEFAULT
) -> Dict[str, Any]:
    """Baseline reproduction: SL at -2.0 ATR, time exit at k=120."""
    spread = spread_baseline
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            return {
                "exit_bar": k,
                "exit_reason": "stop_loss",
                "gross_R": -1.0,
                "spread_cost_R": spread,
                "net_R": -1.0 - spread,
            }
        if k == time_horizon:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "time_exit",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
        if k == bavail:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "data_end",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
    raise RuntimeError("BL did not terminate")


def variant_A(
    rmae, rmfe, bl_, bc_, bavail, spread_baseline, *, M, time_horizon=TIME_HORIZON_DEFAULT
) -> Dict[str, Any]:
    """Class A: SL at -M*ATR. Variant R-unit = M*ATR.

    spread_cost_R_variant = spread_cost_R_baseline * (2.0 / M).
    """
    spread = spread_baseline * (2.0 / M)
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -M:
            return {
                "exit_bar": k,
                "exit_reason": "stop_loss",
                "gross_R": -1.0,
                "spread_cost_R": spread,
                "net_R": -1.0 - spread,
            }
        if k == time_horizon:
            gross = bc_[k_idx] / M
            return {
                "exit_bar": k,
                "exit_reason": "time_exit",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
        if k == bavail:
            gross = bc_[k_idx] / M
            return {
                "exit_bar": k,
                "exit_reason": "data_end",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
    raise RuntimeError("A did not terminate")


def variant_BE(
    rmae, rmfe, bl_, bc_, bavail, spread_baseline, *, T_trigger, time_horizon=TIME_HORIZON_DEFAULT
) -> Dict[str, Any]:
    """Class B: BE-SL at trigger T_trigger (ATR units).

    SL stays at -2.0 ATR until trigger. Trigger fires at first bar where
    running_mfe_atr >= T_trigger; BE-SL active starting NEXT bar. Once
    BE-SL active, exit at BE if bar_low_atr <= 0.
    """
    spread = spread_baseline
    be_active = False
    for k_idx in range(bavail):
        k = k_idx + 1
        if not be_active:
            if rmae[k_idx] <= -2.0:
                return {
                    "exit_bar": k,
                    "exit_reason": "stop_loss",
                    "gross_R": -1.0,
                    "spread_cost_R": spread,
                    "net_R": -1.0 - spread,
                }
            if rmfe[k_idx] >= T_trigger:
                be_active = True
                # Don't check BE on this bar; fall through to time/data-end.
        else:
            if bl_[k_idx] <= 0.0:
                return {
                    "exit_bar": k,
                    "exit_reason": "be_exit",
                    "gross_R": 0.0,
                    "spread_cost_R": spread,
                    "net_R": 0.0 - spread,
                }
        if k == time_horizon:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "time_exit",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
        if k == bavail:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "data_end",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
    raise RuntimeError("BE did not terminate")


def variant_TRAIL(
    rmae,
    rmfe,
    bl_,
    bc_,
    bavail,
    spread_baseline,
    *,
    T_engage,
    D_atr,
    kinked=False,
    time_horizon=TIME_HORIZON_DEFAULT,
) -> Dict[str, Any]:
    """Class C / G1: trail engages at T_engage (ATR), distance D_atr (ATR).

    For kinked=True (G1): D switches from 1.0 ATR to 2.0 ATR once running_mfe
    crosses 6.0 ATR. (Spec confirms this is intentional 'kink' — trail can step
    backward at the boundary by ~1 ATR. Per literal §3.2 formula.)
    """
    spread = spread_baseline
    trail_active = False
    for k_idx in range(bavail):
        k = k_idx + 1
        if not trail_active:
            if rmae[k_idx] <= -2.0:
                return {
                    "exit_bar": k,
                    "exit_reason": "stop_loss",
                    "gross_R": -1.0,
                    "spread_cost_R": spread,
                    "net_R": -1.0 - spread,
                }
            if rmfe[k_idx] >= T_engage:
                trail_active = True
                # Don't check trail on this bar.
        else:
            if kinked:
                D_cur = 1.0 if rmfe[k_idx] < 6.0 else 2.0
            else:
                D_cur = D_atr
            trail_level = rmfe[k_idx] - D_cur
            if bl_[k_idx] <= trail_level:
                gross = trail_level / 2.0
                return {
                    "exit_bar": k,
                    "exit_reason": "trail_exit",
                    "gross_R": gross,
                    "spread_cost_R": spread,
                    "net_R": gross - spread,
                }
        if k == time_horizon:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "time_exit",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
        if k == bavail:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "data_end",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
    raise RuntimeError("TRAIL did not terminate")


def variant_PARTIAL_BE(
    rmae,
    rmfe,
    bl_,
    bc_,
    bavail,
    spread_baseline,
    *,
    T_partial=2.0,
    time_horizon=TIME_HORIZON_DEFAULT,
) -> Dict[str, Any]:
    """Class D: partial close 50% at +1R, BE-SL on remainder.

    Convention: spread is split half/half across the two exits. So:
    - Pre-partial SL: gross = -1.0, net = -1.0 - spread (full spread, single exit).
    - Partial then BE: gross = 0.5*1 + 0.5*0 = 0.5, net = 0.5 - spread.
    - Partial then time: gross = 0.5*1 + 0.5*bar_close_R, net = gross - spread.
    - Partial then data_end: same as partial-then-time but exit_reason flag changes.
    """
    spread = spread_baseline
    partial_closed = False
    partial_gross = 0.0
    for k_idx in range(bavail):
        k = k_idx + 1
        if not partial_closed:
            if rmae[k_idx] <= -2.0:
                return {
                    "exit_bar": k,
                    "exit_reason": "stop_loss",
                    "gross_R": -1.0,
                    "spread_cost_R": spread,
                    "net_R": -1.0 - spread,
                }
            if rmfe[k_idx] >= T_partial:
                partial_closed = True
                partial_gross = 0.5 * 1.0  # half position at +1R, gross
                # Don't check BE this bar.
        else:
            if bl_[k_idx] <= 0.0:
                gross = partial_gross + 0.5 * 0.0  # remainder at BE
                return {
                    "exit_bar": k,
                    "exit_reason": "partial_then_be",
                    "gross_R": gross,
                    "spread_cost_R": spread,
                    "net_R": gross - spread,
                }
        if k == time_horizon:
            if partial_closed:
                rem_gross = 0.5 * (bc_[k_idx] / 2.0)
                gross = partial_gross + rem_gross
                return {
                    "exit_bar": k,
                    "exit_reason": "partial_then_time",
                    "gross_R": gross,
                    "spread_cost_R": spread,
                    "net_R": gross - spread,
                }
            else:
                gross = bc_[k_idx] / 2.0
                return {
                    "exit_bar": k,
                    "exit_reason": "time_exit",
                    "gross_R": gross,
                    "spread_cost_R": spread,
                    "net_R": gross - spread,
                }
        if k == bavail:
            if partial_closed:
                rem_gross = 0.5 * (bc_[k_idx] / 2.0)
                gross = partial_gross + rem_gross
                # Use partial_then_data_end as taxonomy extension; document
                # in manifest. Spec §5.2 lists partial_then_{be,time,sl} but
                # data_end on a partial-closed trade is logically distinct.
                return {
                    "exit_bar": k,
                    "exit_reason": "partial_then_data_end",
                    "gross_R": gross,
                    "spread_cost_R": spread,
                    "net_R": gross - spread,
                }
            else:
                gross = bc_[k_idx] / 2.0
                return {
                    "exit_bar": k,
                    "exit_reason": "data_end",
                    "gross_R": gross,
                    "spread_cost_R": spread,
                    "net_R": gross - spread,
                }
    raise RuntimeError("PARTIAL_BE did not terminate")


def variant_TP(
    rmae, rmfe, bl_, bc_, bavail, spread_baseline, *, T_TP, time_horizon=TIME_HORIZON_DEFAULT
) -> Dict[str, Any]:
    """Class E: fixed TP at T_TP (ATR units). SL at -2.0 ATR.

    Walk bar-by-bar. SL takes precedence at each bar (spec §3.4 step 1).
    TP fires at first bar where running_mfe_atr >= T_TP and SL not already fired.
    Exit at exactly T_TP / 2 in R-units (deterministic fill, not bar high).
    """
    spread = spread_baseline
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            return {
                "exit_bar": k,
                "exit_reason": "stop_loss",
                "gross_R": -1.0,
                "spread_cost_R": spread,
                "net_R": -1.0 - spread,
            }
        if rmfe[k_idx] >= T_TP:
            gross = T_TP / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "fixed_tp",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
        if k == time_horizon:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "time_exit",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
        if k == bavail:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "data_end",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
    raise RuntimeError("TP did not terminate")


def variant_F1(rmae, rmfe, bl_, bc_, bavail, spread_baseline) -> Dict[str, Any]:
    """F1: BL with extended time horizon = 240."""
    return variant_BL(
        rmae, rmfe, bl_, bc_, bavail, spread_baseline, time_horizon=TIME_HORIZON_EXTENDED
    )


def variant_F2(rmae, rmfe, bl_, bc_, bavail, spread_baseline) -> Dict[str, Any]:
    """F2: no time exit. SL at -2.0 ATR. Otherwise exit at bars_available."""
    spread = spread_baseline
    for k_idx in range(bavail):
        k = k_idx + 1
        if rmae[k_idx] <= -2.0:
            return {
                "exit_bar": k,
                "exit_reason": "stop_loss",
                "gross_R": -1.0,
                "spread_cost_R": spread,
                "net_R": -1.0 - spread,
            }
        if k == bavail:
            gross = bc_[k_idx] / 2.0
            return {
                "exit_bar": k,
                "exit_reason": "data_end",
                "gross_R": gross,
                "spread_cost_R": spread,
                "net_R": gross - spread,
            }
    raise RuntimeError("F2 did not terminate")


def variant_G1(rmae, rmfe, bl_, bc_, bavail, spread_baseline) -> Dict[str, Any]:
    """G1: kinked trail. T_engage=2.0 (ATR), D=1.0 ATR while rmfe<6.0,
    D=2.0 ATR once rmfe>=6.0. SL stays at -2.0 ATR until engage.
    """
    return variant_TRAIL(
        rmae, rmfe, bl_, bc_, bavail, spread_baseline, T_engage=2.0, D_atr=1.0, kinked=True
    )


def variant_G2(rmae, rmfe, bl_, bc_, bavail, spread_baseline) -> Dict[str, Any]:
    """G2: BE-SL at +1R + extended hold to k=240. Combination of B2 + F1."""
    return variant_BE(
        rmae,
        rmfe,
        bl_,
        bc_,
        bavail,
        spread_baseline,
        T_trigger=2.0,
        time_horizon=TIME_HORIZON_EXTENDED,
    )


# Variant registry: (id, spec_short, fn).
# fn must be (rmae, rmfe, bl_, bc_, bavail, spread_baseline) -> dict
VARIANTS: List[Tuple[str, str, Callable[..., Dict[str, Any]]]] = [
    ("BL", "Baseline: SL=-2.0ATR, k=120 time exit", variant_BL),
    ("A1", "SL=-1.5ATR, k=120 time exit", lambda *a: variant_A(*a, M=1.5)),
    ("A2", "SL=-2.5ATR, k=120 time exit", lambda *a: variant_A(*a, M=2.5)),
    ("B1", "BE-SL @ +0.5R (rmfe>=1.0), k=120", lambda *a: variant_BE(*a, T_trigger=1.0)),
    ("B2", "BE-SL @ +1.0R (rmfe>=2.0), k=120", lambda *a: variant_BE(*a, T_trigger=2.0)),
    ("B3", "BE-SL @ +1.5R (rmfe>=3.0), k=120", lambda *a: variant_BE(*a, T_trigger=3.0)),
    (
        "C1",
        "Trail engage +1.0R, D=0.5R (1.0 ATR), k=120",
        lambda *a: variant_TRAIL(*a, T_engage=2.0, D_atr=1.0),
    ),
    (
        "C2",
        "Trail engage +1.0R, D=1.0R (2.0 ATR), k=120",
        lambda *a: variant_TRAIL(*a, T_engage=2.0, D_atr=2.0),
    ),
    (
        "C3",
        "Trail engage +2.0R, D=1.0R (2.0 ATR), k=120",
        lambda *a: variant_TRAIL(*a, T_engage=4.0, D_atr=2.0),
    ),
    ("D1", "Partial 50% @ +1R + BE-SL, k=120", variant_PARTIAL_BE),
    ("E1", "Fixed TP @ +1.5R (rmfe>=3.0), k=120", lambda *a: variant_TP(*a, T_TP=3.0)),
    ("E2", "Fixed TP @ +3.0R (rmfe>=6.0), k=120", lambda *a: variant_TP(*a, T_TP=6.0)),
    ("F1", "BL with k=240 (extended hold)", variant_F1),
    ("F2", "BL with no time exit (SL or data_end)", variant_F2),
    ("G1", "Kinked trail (T_engage=+1R, D=0.5R<+3R, D=1R>=+3R)", variant_G1),
    ("G2", "B2 + F1: BE-SL @ +1R + k=240", variant_G2),
]


# ============================================================================
# COMPUTATION ORCHESTRATION
# ============================================================================


def _run_sweep(per_bar_csv: Path, trade_index_csv: Path) -> pd.DataFrame:
    """Compute all 16 variants × 3993 trades and return the long-format DataFrame.

    Memory plan: load both CSVs into memory (~150 MB peak with pandas overhead).
    Group per_bar by trade_id once; per-trade compute is a tight numpy loop.
    """
    print("  Loading trade_index.csv...", flush=True)
    ti = pd.read_csv(trade_index_csv)
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ti = ti.sort_values("trade_id").reset_index(drop=True)

    print("  Loading per_bar_paths.csv (~100 MB)...", flush=True)
    pb = pd.read_csv(per_bar_csv)
    # Sort once for deterministic group iteration order.
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)

    # Build a per-trade lookup of slice indices (start, stop) into pb.
    # Since pb is sorted by trade_id and trade_ids are 0..3992 contiguous,
    # we can compute boundaries fast via np.searchsorted.
    tids = pb["trade_id"].to_numpy(dtype=np.int64)
    n_trades = int(ti["trade_id"].max()) + 1
    starts = np.searchsorted(tids, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids, np.arange(n_trades), side="right")

    rmae_all = pb["running_mae_atr"].to_numpy(dtype=np.float64)
    rmfe_all = pb["running_mfe_atr"].to_numpy(dtype=np.float64)
    bl_all = pb["bar_low_atr"].to_numpy(dtype=np.float64)
    bc_all = pb["bar_close_atr"].to_numpy(dtype=np.float64)

    # Trade-level fields.
    ti_pair = ti["pair"].to_numpy()
    ti_sigts = ti["signal_bar_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
    ti_fold = ti["fold_id"].to_numpy(dtype=np.int64)
    ti_spread = ti["spread_cost_r"].to_numpy(dtype=np.float64)

    n_variants = len(VARIANTS)
    total_rows = n_trades * n_variants

    # Pre-allocate output buffers.
    out_variant = np.empty(total_rows, dtype=object)
    out_tid = np.empty(total_rows, dtype=np.int64)
    out_pair = np.empty(total_rows, dtype=object)
    out_sigts = np.empty(total_rows, dtype=object)
    out_fold = np.empty(total_rows, dtype=np.int64)
    out_reason = np.empty(total_rows, dtype=object)
    out_exitbar = np.empty(total_rows, dtype=np.int64)
    out_gross = np.empty(total_rows, dtype=np.float64)
    out_spread = np.empty(total_rows, dtype=np.float64)
    out_net = np.empty(total_rows, dtype=np.float64)

    print(f"  Computing {n_variants} variants × {n_trades} trades...", flush=True)
    write_idx = 0
    t_start = time.time()
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        bavail = e - s
        rmae = rmae_all[s:e]
        rmfe = rmfe_all[s:e]
        bl_ = bl_all[s:e]
        bc_ = bc_all[s:e]
        spread = float(ti_spread[tid])
        pair = ti_pair[tid]
        sigts = ti_sigts[tid]
        fold = ti_fold[tid]

        for vidx, (vid, _vspec, vfn) in enumerate(VARIANTS):
            r = vfn(rmae, rmfe, bl_, bc_, bavail, spread)
            out_variant[write_idx] = vid
            out_tid[write_idx] = tid
            out_pair[write_idx] = pair
            out_sigts[write_idx] = sigts
            out_fold[write_idx] = fold
            out_reason[write_idx] = r["exit_reason"]
            out_exitbar[write_idx] = r["exit_bar"]
            out_gross[write_idx] = r["gross_R"]
            out_spread[write_idx] = r["spread_cost_R"]
            out_net[write_idx] = r["net_R"]
            write_idx += 1

        if (tid + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(
                f"    progress: {tid + 1}/{n_trades} trades  "
                f"({elapsed:.1f}s, {(tid + 1) / elapsed:.0f} trades/s)",
                flush=True,
            )

    out_df = pd.DataFrame(
        {
            "variant_id": out_variant,
            "trade_id": out_tid,
            "pair": out_pair,
            "signal_bar_ts": out_sigts,
            "fold_id": out_fold,
            "exit_reason_variant": out_reason,
            "exit_bar": out_exitbar,
            "gross_R": out_gross,
            "spread_cost_R": out_spread,
            "net_R": out_net,
        }
    )
    return out_df


# ============================================================================
# AGGREGATION
# ============================================================================


EXIT_REASONS_ALL = [
    "stop_loss",
    "time_exit",
    "be_exit",
    "trail_exit",
    "partial_then_be",
    "partial_then_time",
    "partial_then_sl",
    "partial_then_data_end",
    "fixed_tp",
    "data_end",
]


def _aggregate_pooled(variant_trades: pd.DataFrame) -> pd.DataFrame:
    """Per-variant pooled summary."""
    rows = []
    spec_lookup = {vid: spec for vid, spec, _ in VARIANTS}
    for vid in [v[0] for v in VARIANTS]:
        sub = variant_trades[variant_trades["variant_id"] == vid]
        n = len(sub)
        net = sub["net_R"].to_numpy(dtype=np.float64)
        reason_counts = sub["exit_reason_variant"].value_counts().to_dict()
        rates = {f"{r}_rate": reason_counts.get(r, 0) / n for r in EXIT_REASONS_ALL}
        # Map the spec-named rate aggregates expected by §5 / report.
        sl_rate = rates["stop_loss_rate"]
        time_exit_rate = rates["time_exit_rate"]
        be_exit_rate = rates["be_exit_rate"] + rates["partial_then_be_rate"]
        trail_exit_rate = rates["trail_exit_rate"]
        partial_exit_rate = (
            rates["partial_then_be_rate"]
            + rates["partial_then_time_rate"]
            + rates["partial_then_sl_rate"]
            + rates["partial_then_data_end_rate"]
        )
        tp_exit_rate = rates["fixed_tp_rate"]
        data_end_rate = rates["data_end_rate"] + rates["partial_then_data_end_rate"]

        rows.append(
            {
                "variant_id": vid,
                "variant_spec_short": spec_lookup[vid],
                "n_trades": n,
                "mean_R": float(np.mean(net)),
                "median_R": float(np.median(net)),
                "std_R": float(np.std(net, ddof=1)) if n > 1 else 0.0,
                "q05_R": float(np.quantile(net, 0.05)),
                "q25_R": float(np.quantile(net, 0.25)),
                "q75_R": float(np.quantile(net, 0.75)),
                "q95_R": float(np.quantile(net, 0.95)),
                "sl_rate": sl_rate,
                "time_exit_rate": time_exit_rate,
                "be_exit_rate": be_exit_rate,
                "trail_exit_rate": trail_exit_rate,
                "partial_exit_rate": partial_exit_rate,
                "tp_exit_rate": tp_exit_rate,
                "data_end_rate": data_end_rate,
                "mean_spread_cost_R": float(sub["spread_cost_R"].mean()),
                "total_R": float(np.sum(net)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_per_fold(variant_trades: pd.DataFrame) -> pd.DataFrame:
    """Per-(variant, fold) summary including max-DD per fold."""
    rows = []
    # Sort once by (variant_id, fold_id, signal_bar_ts) for deterministic DD calc.
    sorted_vt = variant_trades.sort_values(["variant_id", "fold_id", "signal_bar_ts", "trade_id"])
    folds = sorted(variant_trades["fold_id"].unique().tolist())
    for vid in [v[0] for v in VARIANTS]:
        for fid in folds:
            sub = sorted_vt[(sorted_vt["variant_id"] == vid) & (sorted_vt["fold_id"] == fid)]
            n = len(sub)
            if n == 0:
                continue
            net = sub["net_R"].to_numpy(dtype=np.float64)
            cum = np.cumsum(net)
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            max_dd = float(np.max(dd))
            reason_counts = sub["exit_reason_variant"].value_counts().to_dict()
            rates = {f"{r}_rate": reason_counts.get(r, 0) / n for r in EXIT_REASONS_ALL}
            be_rate = rates["be_exit_rate"] + rates["partial_then_be_rate"]
            partial_rate = (
                rates["partial_then_be_rate"]
                + rates["partial_then_time_rate"]
                + rates["partial_then_sl_rate"]
                + rates["partial_then_data_end_rate"]
            )
            rows.append(
                {
                    "variant_id": vid,
                    "fold_id": int(fid),
                    "n": n,
                    "mean_R": float(np.mean(net)),
                    "median_R": float(np.median(net)),
                    "total_R": float(np.sum(net)),
                    "max_DD_R": max_dd,
                    "max_DD_pct_of_n": max_dd / n,
                    "sl_rate": rates["stop_loss_rate"],
                    "time_exit_rate": rates["time_exit_rate"],
                    "be_exit_rate": be_rate,
                    "trail_exit_rate": rates["trail_exit_rate"],
                    "partial_exit_rate": partial_rate,
                    "tp_exit_rate": rates["fixed_tp_rate"],
                }
            )
    return pd.DataFrame(rows)


def _additivity_calibration(pooled: pd.DataFrame) -> pd.DataFrame:
    """Compute G2 lone-vs-combination divergence per spec §6.5."""
    bl_mean = float(pooled.loc[pooled["variant_id"] == "BL", "mean_R"].iloc[0])
    b2_mean = float(pooled.loc[pooled["variant_id"] == "B2", "mean_R"].iloc[0])
    f1_mean = float(pooled.loc[pooled["variant_id"] == "F1", "mean_R"].iloc[0])
    g2_mean = float(pooled.loc[pooled["variant_id"] == "G2", "mean_R"].iloc[0])
    lone_lift_sum = (b2_mean - bl_mean) + (f1_mean - bl_mean)
    combination_lift = g2_mean - bl_mean
    divergence = combination_lift - lone_lift_sum
    if abs(divergence) < 0.05:
        note = (
            "abs(divergence) < 0.05R — additive-class combinations "
            "are forecastable from lone effects in this round."
        )
    else:
        note = (
            "abs(divergence) >= 0.05R — additive-class combinations "
            "cannot be forecast from lone effects; round 2 must test "
            "combinations directly."
        )
    return pd.DataFrame(
        [
            {
                "combination": "G2 = B2 + F1",
                "BL_mean_R": bl_mean,
                "B2_mean_R": b2_mean,
                "F1_mean_R": f1_mean,
                "G2_mean_R": g2_mean,
                "lone_lift_sum": lone_lift_sum,
                "combination_lift": combination_lift,
                "divergence": divergence,
                "interpretation_note": note,
            }
        ]
    )


# ============================================================================
# VALIDATION GATES (gates 2-10; gates 11/12/13/14/15 in main)
# ============================================================================


def _validate_gates_2_to_10(
    *,
    variant_trades: pd.DataFrame,
    trade_index: pd.DataFrame,
    pooled: pd.DataFrame,
) -> Dict[str, Any]:
    disp: Dict[str, Any] = {}

    # ----- Gate 2: trade count parity -----
    n_ti = len(trade_index)
    pb_path = REPO_ROOT / "results/l6/arc2/characterisation/v1_2_full/per_bar_paths.csv"
    n_pb = sum(1 for _ in pb_path.open("r", encoding="utf-8")) - 1
    disp["gate_2"] = f"trade_index={n_ti}, per_bar_rows={n_pb}"
    if n_ti != 3993:
        raise RuntimeError(f"Gate 2 HALT — trade_index rows={n_ti}, expected 3993")
    if n_pb != 954749:
        raise RuntimeError(f"Gate 2 HALT — per_bar rows={n_pb}, expected 954749")

    # ----- Gate 3: variant grid completeness -----
    n_total = len(variant_trades)
    expected = 3993 * len(VARIANTS)
    disp["gate_3"] = f"variant_trades_rows={n_total}, expected={expected}"
    if n_total != expected:
        raise RuntimeError(f"Gate 3 HALT — variant_trades rows={n_total}, expected={expected}")
    # Per-variant count check
    counts = variant_trades.groupby("variant_id").size().to_dict()
    bad = {v: c for v, c in counts.items() if c != 3993}
    if bad:
        raise RuntimeError(f"Gate 3 HALT — per-variant trade counts off: {bad}")

    # ----- Gate 4: baseline reproduction -----
    # BL net_R must match v1.1 R column (== trade_index R column passthrough) per trade.
    bl_sub = variant_trades[variant_trades["variant_id"] == "BL"].copy()
    bl_sub = bl_sub.sort_values("trade_id").reset_index(drop=True)
    ti_sorted = trade_index.sort_values("trade_id").reset_index(drop=True)
    if not (bl_sub["trade_id"].values == ti_sorted["trade_id"].values).all():
        raise RuntimeError("Gate 4 HALT — trade_id ordering mismatch BL vs trade_index")
    diff = bl_sub["net_R"].to_numpy(dtype=np.float64) - ti_sorted["R"].to_numpy(dtype=np.float64)
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    n_mismatch = int((abs_diff >= 1e-9).sum())
    disp["gate_4"] = f"max_abs_diff={max_abs:.3e}, mismatches(>=1e-9)={n_mismatch}"
    if n_mismatch > 0 or max_abs >= 1e-9:
        # Diagnostic
        bad_idx = np.argsort(-abs_diff)[:10]
        sample = pd.DataFrame(
            {
                "trade_id": bl_sub["trade_id"].iloc[bad_idx].values,
                "BL_net_R": bl_sub["net_R"].iloc[bad_idx].values,
                "v1.1_R": ti_sorted["R"].iloc[bad_idx].values,
                "abs_diff": abs_diff[bad_idx],
                "BL_exit_reason": bl_sub["exit_reason_variant"].iloc[bad_idx].values,
                "BL_exit_bar": bl_sub["exit_bar"].iloc[bad_idx].values,
                "v1.2_held_bars": ti_sorted["held_bars"].iloc[bad_idx].values,
                "v1.2_exit_reason": ti_sorted["exit_reason"].iloc[bad_idx].values,
            }
        )
        raise RuntimeError(
            f"Gate 4 HALT — BL doesn't reproduce v1.1 R column. "
            f"max_abs_diff={max_abs:.3e}, mismatches(>=1e-9)={n_mismatch}.\n"
            f"Top-10 worst mismatches:\n{sample.to_string()}"
        )

    # ----- Gate 5: variant exit-bar plausibility -----
    # exit_bar <= min(time_horizon_for_variant, bars_available)
    bavail_by_tid = trade_index.set_index("trade_id")["bars_available"].to_dict()
    horizon_by_vid = {v[0]: TIME_HORIZON_DEFAULT for v in VARIANTS}
    horizon_by_vid["F1"] = TIME_HORIZON_EXTENDED
    horizon_by_vid["G2"] = TIME_HORIZON_EXTENDED
    horizon_by_vid["F2"] = TIME_HORIZON_EXTENDED  # F2 has no time exit; bavail is the cap.
    bad_count = 0
    bad_samples = []
    for _, row in variant_trades.iterrows():
        tid = int(row["trade_id"])
        vid = row["variant_id"]
        h = horizon_by_vid[vid]
        bavail = bavail_by_tid[tid]
        cap = min(h, bavail)
        if int(row["exit_bar"]) > cap:
            bad_count += 1
            if len(bad_samples) < 5:
                bad_samples.append((vid, tid, int(row["exit_bar"]), h, bavail))
    disp["gate_5"] = f"exit_bar_violations={bad_count}"
    if bad_count > 0:
        raise RuntimeError(
            f"Gate 5 HALT — {bad_count} exit_bar > cap. "
            f"Sample (vid,tid,exit_bar,horizon,bavail): {bad_samples}"
        )

    # ----- Gate 6: variant R range plausibility -----
    sl_only_variants = {"BL", "A1", "A2", "F2"}
    nan_or_inf = ~np.isfinite(variant_trades["net_R"].to_numpy(dtype=np.float64))
    n_bad = int(nan_or_inf.sum())
    disp["gate_6"] = f"nan_or_inf_count={n_bad}"
    if n_bad > 0:
        raise RuntimeError(f"Gate 6 HALT — {n_bad} NaN/inf values in net_R")
    for vid in sl_only_variants:
        sub = variant_trades[variant_trades["variant_id"] == vid]
        max_r = float(sub["net_R"].max())
        if max_r > 20.0:
            raise RuntimeError(f"Gate 6 HALT — {vid} has max net_R={max_r:.2f} > +20")
    disp["gate_6"] += "; SL-only variants max R within +20"

    # ----- Gate 7: BE-SL rate plausibility (B2: 45-55%) -----
    b2_pooled = pooled.loc[pooled["variant_id"] == "B2"].iloc[0]
    b2_be = float(b2_pooled["be_exit_rate"])
    disp["gate_7"] = f"B2 be_exit_rate={b2_be:.4f}"
    if not (0.45 <= b2_be <= 0.55):
        raise RuntimeError(
            f"Gate 7 HALT — B2 be_exit_rate={b2_be:.4f} outside [0.45, 0.55]. "
            f"Block C reference: 51.19% of trades reached mfe_R >= 1."
        )

    # ----- Gate 8: trail engagement rate plausibility -----
    # Engagement rate = share of trades that did NOT hit SL before T_engage was reached.
    # Equivalent to share of trades whose exit_reason is trail_exit OR time_exit OR data_end.
    # Compute as 1 - sl_rate.
    c1_pooled = pooled.loc[pooled["variant_id"] == "C1"].iloc[0]
    c2_pooled = pooled.loc[pooled["variant_id"] == "C2"].iloc[0]
    c3_pooled = pooled.loc[pooled["variant_id"] == "C3"].iloc[0]
    c1_engage = 1.0 - float(c1_pooled["sl_rate"])
    c2_engage = 1.0 - float(c2_pooled["sl_rate"])
    c3_engage = 1.0 - float(c3_pooled["sl_rate"])
    disp["gate_8"] = (
        f"C1 engage={c1_engage:.4f}, C2 engage={c2_engage:.4f}, C3 engage={c3_engage:.4f}"
    )
    # C1/C2 engage should be near 51.19% (mfe_R >= 1.0 in v1.1); C3 near 33.43% (mfe_R >= 2.0).
    # Tolerance: ±5pp around the headline figure.
    if not (0.46 <= c1_engage <= 0.57) or not (0.46 <= c2_engage <= 0.57):
        raise RuntimeError(
            f"Gate 8 HALT — C1/C2 engage rates outside [0.46, 0.57]. "
            f"C1={c1_engage:.4f}, C2={c2_engage:.4f}. Reference 51.19%."
        )
    if not (0.28 <= c3_engage <= 0.39):
        raise RuntimeError(
            f"Gate 8 HALT — C3 engage rate {c3_engage:.4f} outside [0.28, 0.39]. Reference 33.43%."
        )

    # ----- Gate 9: Fixed TP rate plausibility -----
    e1_pooled = pooled.loc[pooled["variant_id"] == "E1"].iloc[0]
    e2_pooled = pooled.loc[pooled["variant_id"] == "E2"].iloc[0]
    e1_tp = float(e1_pooled["tp_exit_rate"])
    e2_tp = float(e2_pooled["tp_exit_rate"])
    disp["gate_9"] = f"E1 tp_rate={e1_tp:.4f}, E2 tp_rate={e2_tp:.4f}"
    if not (0.25 <= e1_tp <= 0.45):
        raise RuntimeError(
            f"Gate 9 HALT — E1 tp_rate={e1_tp:.4f} outside [0.25, 0.45]. "
            f"Reference: mfe_R >= 1.5 → 40.75% in v1.1, minus those hitting SL first."
        )
    if not (0.05 <= e2_tp <= 0.25):
        raise RuntimeError(f"Gate 9 HALT — E2 tp_rate={e2_tp:.4f} outside [0.05, 0.25]")

    # ----- Gate 10: spread cost convention (Class A spot-check) -----
    # spread_cost_R_variant = spread_cost_R_baseline * (2.0 / M)
    # For 3 sample trades, verify A1 (M=1.5) and A2 (M=2.5) spread_cost_R values.
    sample_tids = [0, 1500, 3000]
    spot_lines = []
    spot_pass = True
    bl_sub_idx = variant_trades[variant_trades["variant_id"] == "BL"].set_index("trade_id")
    a1_sub_idx = variant_trades[variant_trades["variant_id"] == "A1"].set_index("trade_id")
    a2_sub_idx = variant_trades[variant_trades["variant_id"] == "A2"].set_index("trade_id")
    for tid in sample_tids:
        bl_spread = float(bl_sub_idx.loc[tid, "spread_cost_R"])
        a1_spread = float(a1_sub_idx.loc[tid, "spread_cost_R"])
        a2_spread = float(a2_sub_idx.loc[tid, "spread_cost_R"])
        a1_expected = bl_spread * (2.0 / 1.5)
        a2_expected = bl_spread * (2.0 / 2.5)
        d1 = abs(a1_spread - a1_expected)
        d2 = abs(a2_spread - a2_expected)
        spot_lines.append(
            f"  tid={tid}: BL_spread={bl_spread:.6e}, "
            f"A1={a1_spread:.6e} (exp {a1_expected:.6e}, diff {d1:.2e}), "
            f"A2={a2_spread:.6e} (exp {a2_expected:.6e}, diff {d2:.2e})"
        )
        if d1 > 1e-12 or d2 > 1e-12:
            spot_pass = False
    disp["gate_10"] = "; ".join(spot_lines) + (" — PASS" if spot_pass else " — HALT")
    if not spot_pass:
        raise RuntimeError(
            "Gate 10 HALT — Class A spread_cost_R formula violated.\n" + "\n".join(spot_lines)
        )

    return disp


# ============================================================================
# GATE 4 HALT DIAGNOSTIC WRITER
# ============================================================================


def _write_gate4_halt_diagnostic(
    *,
    out_dir: Path,
    variant_trades: pd.DataFrame,
    trade_index: pd.DataFrame,
    err_msg: str,
) -> Path:
    """Write a clear halt diagnostic explaining the root cause of gate 4 failure.

    The spec's BL formula (gross_R = -1 on SL, gross_R = bar_close_atr/2 on TE)
    cannot reproduce v1.1's R column exactly because:

    1. Arc 2's actual SL R formula uses only the EXIT-SIDE spread:
         R_sl = -1 - sp_exit_pips * pip_size / (4 * atr)
       The spec's formula uses the FULL round-trip spread_cost_R from
       trade_index.csv, which equals (sp_entry + sp_exit) * pip / (4 * atr).
       For trades with sp_entry != sp_exit (most large-spread trades), this
       diverges by sp_entry / (4 * atr).

    2. Arc 2's actual time-exit R uses the OPEN of bar at index entry_idx + 120
       (i.e., bar k=121 in the 1-indexed scheme), not the CLOSE of bar k=120.
       The per_bar_paths.csv only contains high/low/close per bar — no bar_open.

    Both gaps require additional inputs (trades_all.csv for sp_entry/exit;
    1H pair data for bar_open) that are not in the prompt's input list. The
    BL spec is therefore structurally unable to satisfy gate 4 as written.

    Per the prompt's pre-commit rule: "If during execution a variant's
    specification is found ambiguous, halt and report rather than infer".
    """
    bl_sub = (
        variant_trades[variant_trades["variant_id"] == "BL"]
        .sort_values("trade_id")
        .reset_index(drop=True)
    )
    ti_sorted = trade_index.sort_values("trade_id").reset_index(drop=True)
    diff = bl_sub["net_R"].to_numpy(dtype=np.float64) - ti_sorted["R"].to_numpy(dtype=np.float64)
    abs_diff = np.abs(diff)
    n_mismatch_1e9 = int((abs_diff >= 1e-9).sum())
    n_mismatch_1e3 = int((abs_diff >= 1e-3).sum())
    n_mismatch_1e1 = int((abs_diff >= 1e-1).sum())
    max_abs = float(abs_diff.max())

    # Compute SL-only and TE-only diffs.
    sl_mask = (ti_sorted["exit_reason"] == "stop_loss").to_numpy()
    te_mask = (ti_sorted["exit_reason"] == "time_exit").to_numpy()
    de_mask = (ti_sorted["exit_reason"] == "data_end").to_numpy()
    sl_max = float(abs_diff[sl_mask].max()) if sl_mask.any() else 0.0
    te_max = float(abs_diff[te_mask].max()) if te_mask.any() else 0.0
    de_max = float(abs_diff[de_mask].max()) if de_mask.any() else 0.0

    # Top-20 worst mismatches.
    bad_idx = np.argsort(-abs_diff)[:20]
    samples = pd.DataFrame(
        {
            "trade_id": bl_sub["trade_id"].iloc[bad_idx].values,
            "pair": ti_sorted["pair"].iloc[bad_idx].values,
            "exit_reason": ti_sorted["exit_reason"].iloc[bad_idx].values,
            "held_bars": ti_sorted["held_bars"].iloc[bad_idx].values,
            "BL_net_R_(spec)": bl_sub["net_R"].iloc[bad_idx].values,
            "v1.1_R_(actual)": ti_sorted["R"].iloc[bad_idx].values,
            "abs_diff": abs_diff[bad_idx],
            "spread_cost_R": ti_sorted["spread_cost_r"].iloc[bad_idx].values,
            "atr_at_signal": ti_sorted["atr_1h_wilder_at_signal"].iloc[bad_idx].values,
        }
    )

    L: List[str] = []
    L.append("# GATE 4 HALT DIAGNOSTIC — Counterfactual Sweep Round 1")
    L.append("")
    L.append(f"_Generated: {_dt.datetime.now().isoformat(timespec='seconds')}_")
    L.append("")
    L.append("## Disposition")
    L.append("")
    L.append(
        "**Gate 4 FAILED.** Per prompt §7 gate 4: 'If BL doesn\\'t reproduce v1.1\\'s R column "
        "exactly, the sweep engine has a bug; no variant result is trustworthy.' — and the "
        "prompt's pre-commit rule: 'If during execution a variant\\'s specification is found "
        "ambiguous, halt and report rather than infer.'"
    )
    L.append("")
    L.append(
        "This is **not** an engine bug. It is a **structural gap between the spec's BL formula "
        "and the actual Arc 2 execution module's R formula**, with **insufficient inputs** in "
        "the prompt's allowed set to bridge the gap exactly. The script halts so the planner "
        "can decide how to proceed (see §5 below)."
    )
    L.append("")
    L.append("## Numerical disposition")
    L.append("")
    L.append(f"- Max abs diff (any trade): **{max_abs:.6e}** R")
    L.append(f"- Mismatches at >= 1e-9 tol: **{n_mismatch_1e9}** of 3993 trades")
    L.append(f"- Mismatches at >= 1e-3 tol: {n_mismatch_1e3} of 3993")
    L.append(f"- Mismatches at >= 1e-1 tol: {n_mismatch_1e1} of 3993")
    L.append("")
    L.append("Per-exit-reason max abs diff:")
    L.append(f"- stop_loss trades: max diff = {sl_max:.6e} R")
    L.append(f"- time_exit trades: max diff = {te_max:.6e} R")
    L.append(f"- data_end trades:  max diff = {de_max:.6e} R")
    L.append("")
    L.append("## Root cause (two structural gaps)")
    L.append("")
    L.append("### Gap A: Arc 2 SL R formula uses EXIT-side spread only")
    L.append("")
    L.append(
        "Arc 2 module (`core/signals/l4_mtf_alignment_2_down_mixed_kijun.py:584-635`) computes:"
    )
    L.append("")
    L.append("```")
    L.append("entry_fill = entry_mid + sp_entry_pips * pip / 2.0    # entry pays half-spread")
    L.append(
        "sl_price   = entry_fill - 2.0 * atr                   # SL referenced from entry_fill"
    )
    L.append("# On SL hit:")
    L.append("exit_fill  = sl_price - sp_exit_pips * pip / 2.0      # exit pays half-spread")
    L.append(
        "R          = (exit_fill - entry_fill) / sl_distance   # = -1 - sp_exit_pips*pip/(4*atr)"
    )
    L.append("```")
    L.append("")
    L.append("So the Arc 2 SL R formula is:")
    L.append("")
    L.append("```")
    L.append("R_sl_actual = -1 - sp_exit_pips * pip / (4 * atr)")
    L.append("```")
    L.append("")
    L.append("The v1.1 manifest §3.4 defines `spread_cost_r` as the FULL round-trip cost:")
    L.append("```")
    L.append("spread_cost_r = (sp_entry + sp_exit) * pip / (4 * atr)")
    L.append("```")
    L.append("")
    L.append(
        "The spec §3.5 BL formula `gross_R = -1.0 on SL` plus net = gross - spread_cost_r yields:"
    )
    L.append("```")
    L.append("R_sl_spec = -1 - spread_cost_r = -1 - (sp_entry + sp_exit) * pip / (4 * atr)")
    L.append("```")
    L.append("")
    L.append(
        "These differ by `sp_entry * pip / (4 * atr)`. For trades where `sp_entry == sp_exit` "
        "(common: same-session entry+exit), the divergence is `spread_cost_r / 2` ≈ small. For "
        "trades where `sp_entry != sp_exit` (different sessions, spread floor changes, etc.), "
        "the divergence can be huge."
    )
    L.append("")
    L.append("**Worked example — trade_id=3282 (CAD_CHF, held=2, SL):**")
    L.append("```")
    L.append(
        "sp_entry = 24.9 pips, sp_exit = 7.3 pips (asymmetric: held only 2 bars across session change)"
    )
    L.append("pip = 0.0001, atr = 7.238e-04")
    L.append("R_sl_actual  = -1 - 7.3 * 0.0001 / (4 * 7.238e-04)  = -1.252138")
    L.append("R_sl_spec    = -1 - (24.9+7.3)*0.0001 / (4*7.238e-04) = -2.112172")
    L.append("absolute diff = 0.860")
    L.append("```")
    L.append("")
    L.append("### Gap B: Arc 2 TE uses bar-OPEN at k=121, not bar-CLOSE at k=120")
    L.append("")
    L.append("Arc 2 module (lines 604, 639-650) computes time exit at:")
    L.append("```")
    L.append("time_exit_idx = entry_idx + HOLD_BARS  # = entry_idx + 120")
    L.append("te_row = df.iloc[time_exit_idx]        # bar at k=121 (1-indexed)")
    L.append("exit_mid = float(te_row['open'])       # OPEN of bar k=121")
    L.append("exit_fill = exit_mid - sp_exit*pip/2.0 # exit pays half-spread")
    L.append("R = (exit_fill - entry_fill) / sl_distance")
    L.append("```")
    L.append("")
    L.append(
        "The spec §3.5 BL formula uses `bar_close_atr / 2.0` from per_bar_paths at k=120, "
        "which is `(close[entry_idx+119] - entry_price) / atr / 2`. This:"
    )
    L.append("- Uses the **wrong bar** (k=120 close vs k=121 open)")
    L.append(
        "- Uses **entry_price = entry_fill** (already includes sp_entry/2) as the reference, "
        "but `te_open` is the bar-open price (mid). The `bar_close_atr / 2` term doesn't decompose "
        "cleanly into a fill-aware expression."
    )
    L.append("")
    L.append("**Worked example — trade_id=2110 (NZD_CHF, held=120, TE):**")
    L.append("```")
    L.append("Actual v1.1 R = +0.8676")
    L.append("Spec BL R = bar_close_atr_at_k120 / 2 - spread_cost_r = 4.266/2 - 0.254 = 1.879")
    L.append("Diff = -1.011")
    L.append("```")
    L.append("")
    L.append("## Inputs needed to bridge the gap exactly")
    L.append("")
    L.append("To make BL exactly reproduce v1.1 R per gate 4, the script needs:")
    L.append(
        "- **`spread_pips_entry`, `spread_pips_exit` per trade** — available in `results/l6/arc2/trades_all.csv` "
        "(NOT in trade_index.csv passthrough). Allows exact SL R computation."
    )
    L.append(
        "- **bar_open[entry_idx + 120] per trade** — available in `data/1hr/<pair>.csv` "
        "(NOT in per_bar_paths.csv). Allows exact TE R computation."
    )
    L.append("")
    L.append(
        "Both are tracked locked artefacts (gate 13 protects them), so reading is safe — but they "
        "are **not in the prompt's listed inputs** (§1)."
    )
    L.append("")
    L.append("## Top-20 worst BL_R vs v1.1_R mismatches")
    L.append("")
    L.append("```")
    L.append(samples.to_string(index=False))
    L.append("```")
    L.append("")
    L.append("## Resolutions for the planner")
    L.append("")
    L.append(
        "**Option 1 — Expand inputs to allow exact reproduction.** Permit reading "
        "`results/l6/arc2/trades_all.csv` (for sp_entry/exit) and `data/1hr/<pair>.csv` "
        "(for bar_open[k=121]). Reformulate the BL net_R formula:"
    )
    L.append("```")
    L.append("R_sl  = -1 - sp_exit_pips * pip_size / (4 * atr_1h_wilder_at_signal)")
    L.append(
        "R_te  = (te_open - entry_fill) / (2 * atr_1h_wilder_at_signal) - sp_exit_pips * pip / (4 * atr)"
    )
    L.append("       where entry_fill = entry_price (== entry_fill from trades_all.csv)")
    L.append("```")
    L.append(
        "Variants would also need to use the corrected SL R / TE R formulas for internal consistency."
    )
    L.append("")
    L.append(
        "**Option 2 — Relax gate 4 tolerance.** Accept that BL approximates v1.1 R within some "
        "looser bound (e.g., 1e-3 pooled mean R, with most trades exact and a few outliers from "
        "spread-asymmetric trades and TE bar-open vs bar-close). Document the gap explicitly. "
        "Internal consistency (BL vs variants) remains; cross-comparison with v1.1 R is approximate."
    )
    L.append("")
    L.append(
        "**Option 3 — Use v1.1 R column directly for BL.** Source BL's net_R from trade_index.csv `R` "
        "column verbatim; engine still computes exit_bar/exit_reason from per_bar_paths but does not "
        "compute net_R. Gate 4 passes by construction; engine is verified for exit detection only. "
        "Variants use the spec's simplified per_bar engine; comparison vs BL is approximate."
    )
    L.append("")
    L.append(
        "**Option 4 — Rewrite BL spec.** Update §3.5 to define BL net_R directly via a formula that "
        "the engine can produce from per_bar_paths + trade_index. E.g., `R_sl = -1 - spread_cost_r/2` "
        "(half-RT proxy assuming sp_entry == sp_exit) and accept the systematic bias from sp_entry "
        "≠ sp_exit and bar_close vs bar_open as known approximation error."
    )
    L.append("")
    L.append("## Preserved outputs (for planner inspection)")
    L.append("")
    L.append(
        "Despite the halt, the script wrote the variant computations under the spec's literal BL "
        "formula. These are SUSPECT vs gate 4 expectations but useful for planner inspection:"
    )
    L.append("")
    L.append("- `variant_trades.csv` — long-format per-(variant, trade) result (63,888 rows)")
    L.append("- `variant_summary_pooled.csv` — per-variant pooled stats")
    L.append("- `variant_summary_per_fold.csv` — per-(variant, fold) stats")
    L.append("- `additivity_calibration.csv` — G2 lone-vs-combination divergence")
    L.append("")
    L.append(
        "If the planner chooses **Option 2** (relaxed gate 4), these outputs are usable as-is, "
        "with the documented bias. If **Option 1, 3, or 4** is chosen, the variants need recomputation."
    )
    L.append("")
    L.append("## Original error message")
    L.append("")
    L.append("```")
    L.append(err_msg)
    L.append("```")
    out = out_dir / "GATE_4_HALT_DIAGNOSTIC.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# ============================================================================
# REPORT WRITER (counterfactual_sweep_round_1.md)
# ============================================================================


def _write_combined_report(
    *,
    out_dir: Path,
    pooled: pd.DataFrame,
    per_fold: pd.DataFrame,
    additivity: pd.DataFrame,
    disp: Dict[str, Any],
    input_shas: Dict[str, str],
    output_shas: Dict[str, str],
    determinism: Dict[str, str],
    run_timestamps: Dict[str, str],
    single_run: bool,
) -> Path:
    """Write the combined synthesis report. Disposition-discipline compliant
    in §6.2-§6.7 (no forbidden phrases). §6.8 is the tagged planning input.
    """
    pooled_idx = pooled.set_index("variant_id")
    bl_mean = float(pooled_idx.loc["BL", "mean_R"])

    # Worst-fold per variant.
    worst_fold_info: Dict[str, Tuple[int, float, float]] = {}
    for vid in [v[0] for v in VARIANTS]:
        sub = per_fold[per_fold["variant_id"] == vid]
        worst = sub.sort_values("mean_R").iloc[0]
        wf_dd = sub.sort_values("max_DD_R", ascending=False).iloc[0]
        worst_fold_info[vid] = (
            int(worst["fold_id"]),
            float(worst["mean_R"]),
            int(wf_dd["fold_id"]),
            float(wf_dd["max_DD_R"]),
        )

    # Per-fold rank of worst-fold mean_R among 7 folds.
    rank_info: Dict[str, int] = {}
    for vid in [v[0] for v in VARIANTS]:
        sub = per_fold[per_fold["variant_id"] == vid].sort_values("mean_R")
        rank_info[vid] = int(sub.iloc[0]["fold_id"])  # the fold ID of worst

    L: List[str] = []
    L.append("# Counterfactual Exit-Rule Sweep — Round 1")
    L.append("")
    L.append(f"_Generated: {run_timestamps.get('end', 'n/a')} (UTC-naive local timestamp)._")
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
    L.append("**Determinism (Gate 12) — two consecutive runs:**")
    L.append("")
    if single_run:
        L.append("- SKIPPED (--single-run)")
    else:
        for k, v in determinism.items():
            L.append(f"- `{k}`: {v}")
    L.append("")
    L.append(
        f"**Run timestamps:** start={run_timestamps['start']}, end={run_timestamps['end']}, "
        f"wallclock_run1={run_timestamps['wallclock_run1']}."
    )
    L.append("")

    # ---- 6.2 Baseline reproduction ----
    L.append("## 6.2 Baseline reproduction validation")
    L.append("")
    L.append(f"Gate 4 disposition: **PASS** — {disp['gate_4']}.")
    L.append("")
    L.append(
        "BL's per-trade `net_R` was compared against the v1.1 `R` column "
        "(passthrough into `trade_index.csv`) for all 3,993 trades. "
        "Maximum absolute difference is reported above; zero mismatches at the "
        "1e-9 tolerance threshold confirms the sweep engine reproduces the "
        "Arc 2 baseline R formula exactly. Variant results downstream are "
        "computed by the same engine."
    )
    L.append("")

    # ---- 6.3 Variant grid summary table ----
    L.append("## 6.3 Variant grid summary")
    L.append("")
    L.append(
        "| ID | Spec | n | mean_R | median_R | sl_rate | lift_vs_BL | worst_fold (id, mean_R) | worst_fold_DD_R |"
    )
    L.append(
        "|----|------|---|--------|----------|---------|------------|-------------------------|------------------|"
    )
    for vid, spec, _ in VARIANTS:
        row = pooled_idx.loc[vid]
        wf_id, wf_mean, wf_dd_id, wf_dd = worst_fold_info[vid]
        lift = float(row["mean_R"]) - bl_mean
        L.append(
            f"| **{vid}** | {spec} | {int(row['n_trades'])} | "
            f"{row['mean_R']:+.4f} | {row['median_R']:+.4f} | "
            f"{row['sl_rate']:.3f} | {lift:+.4f} | "
            f"(fold {wf_id}, {wf_mean:+.4f}) | "
            f"{wf_dd:.3f} (fold {wf_dd_id}) |"
        )
    L.append("")

    # ---- 6.4 Per-class headline findings ----
    L.append("## 6.4 Per-class headline findings")
    L.append("")

    def vmean(vid: str) -> float:
        return float(pooled_idx.loc[vid, "mean_R"])

    def vsl(vid: str) -> float:
        return float(pooled_idx.loc[vid, "sl_rate"])

    def vmedian(vid: str) -> float:
        return float(pooled_idx.loc[vid, "median_R"])

    def wf(vid: str) -> Tuple[int, float]:
        return worst_fold_info[vid][0], worst_fold_info[vid][1]

    L.append(
        f"**Class A (SL distance).** A1 (SL=−1.5×ATR) produced pooled mean R of "
        f"{vmean('A1'):+.4f} with sl_rate {vsl('A1'):.3f} and median R "
        f"{vmedian('A1'):+.4f}; worst-fold mean R was {wf('A1')[1]:+.4f} (fold {wf('A1')[0]}). "
        f"A2 (SL=−2.5×ATR) produced pooled mean R of {vmean('A2'):+.4f}, "
        f"sl_rate {vsl('A2'):.3f}, median R {vmedian('A2'):+.4f}; worst-fold mean R "
        f"{wf('A2')[1]:+.4f} (fold {wf('A2')[0]}). Both reported in the variant's own R-unit "
        f"(1R = M×ATR); spread_cost_R rescaled per §3.6 (PASS gate 10)."
    )
    L.append("")
    L.append(
        f"**Class B (BE-SL trigger).** B1 (T=+0.5R), B2 (T=+1R), B3 (T=+1.5R) produced "
        f"pooled mean R of {vmean('B1'):+.4f} / {vmean('B2'):+.4f} / {vmean('B3'):+.4f} "
        f"versus BL {bl_mean:+.4f}. BE-exit shares: "
        f"B1={float(pooled_idx.loc['B1', 'be_exit_rate']):.3f}, "
        f"B2={float(pooled_idx.loc['B2', 'be_exit_rate']):.3f}, "
        f"B3={float(pooled_idx.loc['B3', 'be_exit_rate']):.3f}. "
        f"Worst-fold mean R: B1={wf('B1')[1]:+.4f} (fold {wf('B1')[0]}), "
        f"B2={wf('B2')[1]:+.4f} (fold {wf('B2')[0]}), "
        f"B3={wf('B3')[1]:+.4f} (fold {wf('B3')[0]})."
    )
    L.append("")
    L.append(
        f"**Class C (Trail).** C1 (engage +1R, D=0.5R), C2 (engage +1R, D=1R), "
        f"C3 (engage +2R, D=1R) produced pooled mean R of "
        f"{vmean('C1'):+.4f} / {vmean('C2'):+.4f} / {vmean('C3'):+.4f}. "
        f"Trail-exit shares: C1={float(pooled_idx.loc['C1', 'trail_exit_rate']):.3f}, "
        f"C2={float(pooled_idx.loc['C2', 'trail_exit_rate']):.3f}, "
        f"C3={float(pooled_idx.loc['C3', 'trail_exit_rate']):.3f}. "
        f"Worst-fold mean R: C1={wf('C1')[1]:+.4f} (fold {wf('C1')[0]}), "
        f"C2={wf('C2')[1]:+.4f} (fold {wf('C2')[0]}), "
        f"C3={wf('C3')[1]:+.4f} (fold {wf('C3')[0]})."
    )
    L.append("")
    L.append(
        f"**Class D (Partial close + BE-SL).** D1 (50% partial @+1R + BE-SL on remainder) "
        f"produced pooled mean R of {vmean('D1'):+.4f}, partial_exit_rate "
        f"{float(pooled_idx.loc['D1', 'partial_exit_rate']):.3f}. Worst-fold mean R "
        f"{wf('D1')[1]:+.4f} (fold {wf('D1')[0]})."
    )
    L.append("")
    L.append(
        f"**Class E (Fixed TP).** E1 (TP=+1.5R) and E2 (TP=+3R) produced pooled mean R of "
        f"{vmean('E1'):+.4f} / {vmean('E2'):+.4f}. TP-exit shares: "
        f"E1={float(pooled_idx.loc['E1', 'tp_exit_rate']):.3f}, "
        f"E2={float(pooled_idx.loc['E2', 'tp_exit_rate']):.3f}. "
        f"Worst-fold mean R: E1={wf('E1')[1]:+.4f} (fold {wf('E1')[0]}), "
        f"E2={wf('E2')[1]:+.4f} (fold {wf('E2')[0]})."
    )
    L.append("")
    L.append(
        f"**Class F (Time-exit horizon).** F1 (k=240) and F2 (no time exit) produced pooled "
        f"mean R of {vmean('F1'):+.4f} / {vmean('F2'):+.4f}. Worst-fold mean R: "
        f"F1={wf('F1')[1]:+.4f} (fold {wf('F1')[0]}), "
        f"F2={wf('F2')[1]:+.4f} (fold {wf('F2')[0]})."
    )
    L.append("")
    L.append(
        f"**Bonus combinations (G).** G1 (kinked trail) produced pooled mean R "
        f"{vmean('G1'):+.4f}, worst-fold {wf('G1')[1]:+.4f} (fold {wf('G1')[0]}). "
        f"G2 (B2 + F1) produced pooled mean R {vmean('G2'):+.4f}, worst-fold "
        f"{wf('G2')[1]:+.4f} (fold {wf('G2')[0]})."
    )
    L.append("")

    # ---- 6.5 Additivity calibration ----
    L.append("## 6.5 Additivity calibration (G2 vs B2 + F1)")
    L.append("")
    add = additivity.iloc[0]
    L.append(f"- BL mean R: {add['BL_mean_R']:+.4f}")
    L.append(
        f"- B2 mean R: {add['B2_mean_R']:+.4f}  (lift vs BL: {add['B2_mean_R'] - add['BL_mean_R']:+.4f})"
    )
    L.append(
        f"- F1 mean R: {add['F1_mean_R']:+.4f}  (lift vs BL: {add['F1_mean_R'] - add['BL_mean_R']:+.4f})"
    )
    L.append(f"- G2 mean R: {add['G2_mean_R']:+.4f}  (lift vs BL: {add['combination_lift']:+.4f})")
    L.append(f"- Lone-lift sum (B2 lift + F1 lift): {add['lone_lift_sum']:+.4f}")
    L.append(f"- Combination lift (G2 lift): {add['combination_lift']:+.4f}")
    L.append(f"- Divergence (combination − sum-of-lones): **{add['divergence']:+.4f}** R")
    L.append("")
    L.append(f"Interpretation: {add['interpretation_note']}")
    L.append("")

    # ---- 6.6 Worst-fold disposition table ----
    L.append("## 6.6 Worst-fold disposition table")
    L.append("")
    L.append("Per-fold mean R and max DD across all 7 folds, worst fold value bolded.")
    L.append("")
    L.append("### 6.6a Per-fold mean R")
    L.append("")
    folds = sorted(per_fold["fold_id"].unique().tolist())
    head = "| variant | " + " | ".join(f"f{f}" for f in folds) + " | worst fold |"
    sep = "|---" * (len(folds) + 2) + "|"
    L.append(head)
    L.append(sep)
    for vid, _spec, _ in VARIANTS:
        sub = per_fold[per_fold["variant_id"] == vid].set_index("fold_id")
        worst_fold_id = int(sub["mean_R"].idxmin())
        cells = []
        for f in folds:
            v = float(sub.loc[f, "mean_R"])
            if f == worst_fold_id:
                cells.append(f"**{v:+.4f}**")
            else:
                cells.append(f"{v:+.4f}")
        cells.append(f"f{worst_fold_id}={float(sub.loc[worst_fold_id, 'mean_R']):+.4f}")
        L.append(f"| {vid} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### 6.6b Per-fold max DD (R-units, fold-internal sequential R sum)")
    L.append("")
    L.append(head)
    L.append(sep)
    for vid, _spec, _ in VARIANTS:
        sub = per_fold[per_fold["variant_id"] == vid].set_index("fold_id")
        worst_fold_id = int(sub["max_DD_R"].idxmax())
        cells = []
        for f in folds:
            v = float(sub.loc[f, "max_DD_R"])
            if f == worst_fold_id:
                cells.append(f"**{v:.3f}**")
            else:
                cells.append(f"{v:.3f}")
        cells.append(f"f{worst_fold_id}={float(sub.loc[worst_fold_id, 'max_DD_R']):.3f}")
        L.append(f"| {vid} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(
        "_Note (per §5.1): max DD here is a within-fold sequential R sum, NOT "
        "the Arc 2 dollar-DD. Sufficient for relative ranking and rough "
        "plausibility, not for gate-pass prediction._"
    )
    L.append("")

    # ---- 6.7 Cross-variant observations ----
    L.append("## 6.7 Cross-variant observations")
    L.append("")
    # Sort by pooled mean R; report ranking statements (permitted: "highest by pooled mean R").
    pooled_sorted = pooled.sort_values("mean_R", ascending=False)
    top1 = pooled_sorted.iloc[0]
    top2 = pooled_sorted.iloc[1]
    bot1 = pooled_sorted.iloc[-1]
    L.append(
        f"Variant **{top1['variant_id']}** is highest by pooled mean R "
        f"({top1['mean_R']:+.4f} R). Variant **{top2['variant_id']}** is second "
        f"({top2['mean_R']:+.4f} R). Variant **{bot1['variant_id']}** is lowest "
        f"by pooled mean R ({bot1['mean_R']:+.4f} R)."
    )
    L.append("")
    # Worst-fold mean R ranking.
    wf_pairs = sorted(
        [(vid, worst_fold_info[vid][1]) for vid, _, _ in VARIANTS],
        key=lambda x: x[1],
        reverse=True,
    )
    L.append(
        f"By worst-fold mean R: **{wf_pairs[0][0]}** is highest "
        f"({wf_pairs[0][1]:+.4f} R), **{wf_pairs[1][0]}** is second "
        f"({wf_pairs[1][1]:+.4f} R); **{wf_pairs[-1][0]}** is lowest "
        f"({wf_pairs[-1][1]:+.4f} R)."
    )
    L.append("")
    # Number of variants with positive worst-fold mean R.
    n_pos_wf = sum(1 for _, m in wf_pairs if m > 0)
    L.append(
        f"Of the {len(VARIANTS)} specifications, {n_pos_wf} have positive worst-fold "
        f"mean R, {len(VARIANTS) - n_pos_wf} have non-positive worst-fold mean R."
    )
    L.append("")
    # Number of variants with mean R > BL mean R.
    n_gt_bl = sum(
        1
        for vid, _, _ in VARIANTS
        if float(pooled_idx.loc[vid, "mean_R"]) > bl_mean and vid != "BL"
    )
    L.append(
        f"Of the 15 non-BL variants, {n_gt_bl} have pooled mean R higher than BL "
        f"({bl_mean:+.4f} R), and {15 - n_gt_bl} have pooled mean R at or below BL."
    )
    L.append("")

    # ---- 6.8 Planning input (clearly tagged) ----
    L.append("## 6.8 Planning input — descriptive ranking for planner decision")
    L.append("")
    L.append(
        "_This subsection steps outside §14.5's disposition discipline to provide "
        "ranked observations for planner use. No phase-2 spec is selected here; the "
        "planner uses these rankings in a follow-up workflow._"
    )
    L.append("")
    # By pooled mean R.
    L.append("### 6.8a Ranked by pooled mean R (descending)")
    L.append("")
    L.append("| rank | variant_id | pooled mean R | lift vs BL |")
    L.append("|---|---|---|---|")
    pooled_sorted_mean = pooled.sort_values("mean_R", ascending=False).reset_index(drop=True)
    for i, row in pooled_sorted_mean.iterrows():
        L.append(
            f"| {i + 1} | {row['variant_id']} | {row['mean_R']:+.4f} | "
            f"{row['mean_R'] - bl_mean:+.4f} |"
        )
    L.append("")
    # By worst-fold mean R.
    L.append("### 6.8b Ranked by worst-fold mean R (descending)")
    L.append("")
    L.append("| rank | variant_id | worst-fold mean R | worst fold ID |")
    L.append("|---|---|---|---|")
    for i, (vid, m) in enumerate(
        sorted(
            [(vid, worst_fold_info[vid][1]) for vid, _, _ in VARIANTS],
            key=lambda x: x[1],
            reverse=True,
        )
    ):
        L.append(f"| {i + 1} | {vid} | {m:+.4f} | {worst_fold_info[vid][0]} |")
    L.append("")
    # By max DD improvement vs BL (i.e., variant max DD lower than BL → improvement).
    bl_worst_dd = max(
        float(
            per_fold[(per_fold["variant_id"] == "BL") & (per_fold["fold_id"] == f)][
                "max_DD_R"
            ].iloc[0]
        )
        for f in folds
    )
    L.append(
        "### 6.8c Ranked by max-DD-improvement vs BL (descending; positive = lower worst-fold DD than BL)"
    )
    L.append("")
    L.append("| rank | variant_id | worst-fold max_DD_R | improvement vs BL worst-fold DD |")
    L.append("|---|---|---|---|")
    dd_pairs = sorted(
        [(vid, worst_fold_info[vid][3]) for vid, _, _ in VARIANTS],
        key=lambda x: x[1],
    )
    for i, (vid, dd) in enumerate(dd_pairs):
        improvement = bl_worst_dd - dd
        L.append(f"| {i + 1} | {vid} | {dd:.3f} | {improvement:+.3f} |")
    L.append("")
    L.append(
        f"_BL worst-fold max DD: {bl_worst_dd:.3f} R (used as reference for the "
        "improvement column)._"
    )
    L.append("")

    # ---- 6.9 Out-of-scope items ----
    L.append("## 6.9 Out-of-scope items observed")
    L.append("")
    L.append(
        "- Class D (Partial close) introduced an additional exit-reason taxonomy value, "
        "`partial_then_data_end`, for partial-closed trades whose remainder ran into the "
        "data-end clamp before BE-SL or time horizon. The spec §5.2 lists "
        "`partial_then_{be,time,sl}` but not `partial_then_data_end`. The new value is "
        "rolled into the `data_end_rate` and `partial_exit_rate` aggregates per spec "
        "intent. Logged for completeness; no action taken."
    )
    L.append(
        "- Variant G1's trail level steps backward by ~1 ATR at the running_mfe ≥ 6.0 "
        "boundary by intentional design (the kink is the experimental hypothesis). The "
        "§3.2 parenthetical 'monotonic non-decreasing' is true for non-kinked variants "
        "(C1/C2/C3) but does not apply to G1 by construction. Implementation follows the "
        "literal `trail_level = running_mfe - D_current` formula. Logged."
    )
    L.append(
        "- For Class D `partial_then_time` and `partial_then_data_end`, spec §3.3 "
        "step 3 stated 'exit at bar close × 0.5 + partial_R' without explicit second-half "
        "spread mention. Implementation applies the symmetric convention from §3.3 "
        "step 2 (half-spread on each leg → full spread on the trade). Net R formula: "
        "`gross = 0.5 + 0.5 × bar_close_R; net = gross − spread_cost_R`. Logged."
    )
    L.append(
        "- Counterfactual variant grid is closed (16 specifications). No additional "
        "variants observed during execution that warrant inclusion in this round."
    )
    L.append("")

    out = out_dir / "counterfactual_sweep_round_1.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# ============================================================================
# DISPOSITION-DISCIPLINE CHECK (Gate 11)
# ============================================================================


def _gate_11_disposition_grep(report_path: Path) -> Tuple[bool, List[str]]:
    """Grep §6.2-§6.7 for forbidden phrases. §6.8 is exempt."""
    text = report_path.read_text(encoding="utf-8")
    # Identify §6.8 boundary.
    m = re.search(r"##\s+6\.8\s+Planning input", text)
    if m is None:
        return False, ["§6.8 boundary not found in report"]
    start_m = re.search(r"##\s+6\.2\s+Baseline", text)
    if start_m is None:
        return False, ["§6.2 boundary not found in report"]
    region = text[start_m.start() : m.start()]
    forbidden = [
        "should be selected",
        "passes the gate",
        "best variant",
        "we should",
        "would have",
    ]
    hits: List[str] = []
    for phrase in forbidden:
        # Case-insensitive search per gate spec.
        if re.search(re.escape(phrase), region, flags=re.IGNORECASE):
            # Get context.
            for m2 in re.finditer(re.escape(phrase), region, flags=re.IGNORECASE):
                ctx_start = max(0, m2.start() - 60)
                ctx_end = min(len(region), m2.end() + 60)
                hits.append(f"  '{phrase}': ...{region[ctx_start:ctx_end]}...")
    return (len(hits) == 0, hits)


# ============================================================================
# MAIN
# ============================================================================


def run_pipeline(
    *,
    out_dir: Path,
    per_bar_csv: Path,
    trade_index_csv: Path,
    write_report: bool = True,
    input_shas: Dict[str, str] = None,
    determinism: Dict[str, str] = None,
    single_run: bool = False,
    run_timestamps: Dict[str, str] = None,
) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trade_index = pd.read_csv(trade_index_csv)
    trade_index["signal_bar_ts"] = pd.to_datetime(trade_index["signal_bar_ts"])
    trade_index = trade_index.sort_values("trade_id").reset_index(drop=True)

    print("Stage 1: Computing variants...", flush=True)
    variant_trades = _run_sweep(per_bar_csv, trade_index_csv)

    # Sort for deterministic output.
    variant_trades = variant_trades.sort_values(["variant_id", "trade_id"]).reset_index(drop=True)

    # ----- Aggregations -----
    print("Stage 2: Aggregating...", flush=True)
    pooled = _aggregate_pooled(variant_trades)
    per_fold = _aggregate_per_fold(variant_trades)
    additivity = _additivity_calibration(pooled)

    # ----- Pre-gate: write variant outputs (preserved for inspection even if gates HALT) -----
    print("Stage 4a: Writing CSV outputs (pre-gate, for inspection)...", flush=True)
    vt_path = out_dir / "variant_trades.csv"
    vp_path = out_dir / "variant_summary_pooled.csv"
    vf_path = out_dir / "variant_summary_per_fold.csv"
    ad_path = out_dir / "additivity_calibration.csv"

    variant_trades.to_csv(vt_path, index=False, lineterminator="\n", float_format="%.10g")
    pooled.to_csv(vp_path, index=False, lineterminator="\n", float_format="%.10g")
    per_fold.to_csv(vf_path, index=False, lineterminator="\n", float_format="%.10g")
    additivity.to_csv(ad_path, index=False, lineterminator="\n", float_format="%.10g")

    # ----- Validation gates 2-10 -----
    print("Stage 4b: Validating gates 2-10...", flush=True)
    try:
        disp = _validate_gates_2_to_10(
            variant_trades=variant_trades,
            trade_index=trade_index,
            pooled=pooled,
        )
    except RuntimeError as e:
        # Write a halt diagnostic if gate 4 fails.
        err_msg = str(e)
        if "Gate 4 HALT" in err_msg:
            halt_path = _write_gate4_halt_diagnostic(
                out_dir=out_dir,
                variant_trades=variant_trades,
                trade_index=trade_index,
                err_msg=err_msg,
            )
            print(f"\n=== GATE 4 HALT DIAGNOSTIC written to: {halt_path} ===\n")
        raise

    # ----- Write combined report -----
    if write_report:
        print("Stage 5: Writing report...", flush=True)
        # Compute output sha256s for the report.
        out_shas = {
            "variant_trades.csv": _sha256_file(vt_path),
            "variant_summary_pooled.csv": _sha256_file(vp_path),
            "variant_summary_per_fold.csv": _sha256_file(vf_path),
            "additivity_calibration.csv": _sha256_file(ad_path),
        }
        report_path = _write_combined_report(
            out_dir=out_dir,
            pooled=pooled,
            per_fold=per_fold,
            additivity=additivity,
            disp=disp,
            input_shas=input_shas or {},
            output_shas=out_shas,
            determinism=determinism or {},
            run_timestamps=run_timestamps
            or {"start": "n/a", "end": "n/a", "wallclock_run1": "n/a"},
            single_run=single_run,
        )
        out_shas["counterfactual_sweep_round_1.md"] = _sha256_file(report_path)

        # ----- Gate 11: disposition discipline grep -----
        print("Stage 6: Gate 11 disposition-discipline check...", flush=True)
        ok, hits = _gate_11_disposition_grep(report_path)
        disp["gate_11"] = "PASS" if ok else f"HALT — {len(hits)} forbidden-phrase hits"
        if not ok:
            for h in hits:
                print(h, flush=True)
            raise RuntimeError(
                f"Gate 11 HALT — {len(hits)} forbidden phrase hits in §6.2-§6.7:\n"
                + "\n".join(hits)
            )
    else:
        out_shas = {
            "variant_trades.csv": _sha256_file(vt_path),
            "variant_summary_pooled.csv": _sha256_file(vp_path),
            "variant_summary_per_fold.csv": _sha256_file(vf_path),
            "additivity_calibration.csv": _sha256_file(ad_path),
        }

    return out_shas, variant_trades, pooled, per_fold, additivity, disp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT / "results/l6/arc2/characterisation/extended/counterfactuals/round_1"
        ),
    )
    parser.add_argument(
        "--per-bar-csv",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_full/per_bar_paths.csv"),
    )
    parser.add_argument(
        "--trade-index-csv",
        default=str(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_full/trade_index.csv"),
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Skip the determinism re-run (development only).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Arc 2 counterfactual exit-rule sweep — Round 1")
    print("=" * 60)

    # Track memory peak.
    tracemalloc.start()
    t_overall_start = time.time()
    start_iso = _dt.datetime.now().isoformat(timespec="seconds")

    # ----- Gate 1: input integrity -----
    print("\n[Gate 1] Verifying 3 v1.2 input sha256s...")
    input_shas = _verify_input_integrity()
    for k in input_shas:
        print(f"  OK {k}")

    out_dir = Path(args.output_dir)
    per_bar_csv = Path(args.per_bar_csv)
    trade_index_csv = Path(args.trade_index_csv)

    # ----- Run #1 -----
    print(f"\n[Run #1] Output dir: {out_dir}")
    t_run1_start = time.time()
    sha1, variant_trades, pooled, per_fold, additivity, disp = run_pipeline(
        out_dir=out_dir,
        per_bar_csv=per_bar_csv,
        trade_index_csv=trade_index_csv,
        write_report=False,  # Defer report until after determinism run.
    )
    elapsed1 = time.time() - t_run1_start
    print(f"  Run #1 complete in {elapsed1:.1f}s")
    for k, v in sha1.items():
        print(f"    {k}: {v}")

    # ----- Run #2: determinism (Gate 12) -----
    determinism: Dict[str, str] = {}
    if not args.single_run:
        scratch = Path(tempfile.mkdtemp(prefix="arc2_sweep_run2_"))
        print(f"\n[Run #2 / Gate 12] Output dir (scratch): {scratch}")
        t_run2_start = time.time()
        sha2, _, _, _, _, _ = run_pipeline(
            out_dir=scratch,
            per_bar_csv=per_bar_csv,
            trade_index_csv=trade_index_csv,
            write_report=False,
        )
        elapsed2 = time.time() - t_run2_start
        print(f"  Run #2 complete in {elapsed2:.1f}s")
        det_pass = True
        for k in sha1:
            match = sha1[k] == sha2[k]
            determinism[k] = "match" if match else "MISMATCH"
            print(f"    {k}: {determinism[k]}")
            if not match:
                det_pass = False
        # Cleanup scratch.
        try:
            for p in scratch.iterdir():
                p.unlink()
            scratch.rmdir()
        except Exception:
            pass
        if not det_pass:
            raise RuntimeError("Gate 12 HALT — determinism failed; outputs differ.")

    # ----- Now write the report (with determinism receipt baked in) -----
    end_iso = _dt.datetime.now().isoformat(timespec="seconds")
    run_timestamps = {
        "start": start_iso,
        "end": end_iso,
        "wallclock_run1": f"{elapsed1:.1f}s",
    }
    report_path = _write_combined_report(
        out_dir=out_dir,
        pooled=pooled,
        per_fold=per_fold,
        additivity=additivity,
        disp=disp,
        input_shas=input_shas,
        output_shas=sha1,
        determinism=determinism,
        run_timestamps=run_timestamps,
        single_run=args.single_run,
    )
    sha1["counterfactual_sweep_round_1.md"] = _sha256_file(report_path)

    # ----- Gate 11: disposition discipline -----
    print("\n[Gate 11] Disposition-discipline grep on §6.2-§6.7...")
    ok, hits = _gate_11_disposition_grep(report_path)
    disp["gate_11"] = "PASS" if ok else f"HALT — {len(hits)} hits"
    if not ok:
        print(f"  HALT — {len(hits)} forbidden-phrase hits:")
        for h in hits:
            print(h)
        raise RuntimeError("Gate 11 HALT — see hits above")
    print("  PASS — no forbidden phrases in §6.2-§6.7")

    # ----- Gate 13: locked artefact integrity (post-run) -----
    print("\n[Gate 13] Re-verifying locked artefact integrity post-run...")
    post_input_shas = _verify_input_integrity()
    for k in input_shas:
        if input_shas[k] != post_input_shas[k]:
            raise RuntimeError(f"Gate 13 HALT — input {k} sha256 changed mid-run")
    # Adjacent locked files.
    for rel, expected in ADJ_LOCKED_SHAS.items():
        actual = _sha256_file(REPO_ROOT / rel)
        if actual != expected:
            raise RuntimeError(
                f"Gate 13 HALT — adjacent locked file {rel} sha256 changed.\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
    print("  All locked artefacts unchanged.")

    # ----- Gate 15: CANDIDATE_HYPOTHESES.md unchanged -----
    print("\n[Gate 15] Verifying docs/CANDIDATE_HYPOTHESES.md unchanged...")
    ch_path = REPO_ROOT / "docs/CANDIDATE_HYPOTHESES.md"
    if ch_path.exists():
        actual = _sha256_file(ch_path)
        if actual != CANDIDATE_HYPOTHESES_BASELINE_SHA:
            raise RuntimeError(
                f"Gate 15 HALT — CANDIDATE_HYPOTHESES.md sha256 changed.\n"
                f"  baseline: {CANDIDATE_HYPOTHESES_BASELINE_SHA}\n"
                f"  observed: {actual}"
            )
        print(f"  PASS — sha256 unchanged: {actual}")
    else:
        print("  CANDIDATE_HYPOTHESES.md not present; treating as no-change.")

    # Memory snapshot.
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ----- run_manifest.txt -----
    rm_path = out_dir / "run_manifest.txt"
    rm: List[str] = []
    rm.append("Arc 2 counterfactual exit-rule sweep — Round 1 — run manifest")
    rm.append("=" * 60)
    rm.append(f"Run timestamps: start={start_iso}, end={end_iso}")
    rm.append(f"Wallclock run #1: {elapsed1:.1f}s")
    if not args.single_run:
        rm.append(f"Wallclock run #2 (determinism): {elapsed2:.1f}s")
    rm.append(f"Memory peak (tracemalloc): {peak / (1024 * 1024):.1f} MB")
    rm.append("")
    rm.append("Inputs (sha256, locked at start AND end):")
    for k, v in input_shas.items():
        rm.append(f"  {k}\n    {v}")
    rm.append("")
    rm.append("Outputs (sha256, run #1):")
    for k, v in sha1.items():
        p = out_dir / k
        size = p.stat().st_size if p.exists() else 0
        rm.append(f"  {k} ({size:,} bytes)\n    {v}")
    rm.append("")
    rm.append("Adjacent locked artefacts (sha256, verified unchanged):")
    for k, v in ADJ_LOCKED_SHAS.items():
        rm.append(f"  {k}\n    {v}")
    rm.append("")
    rm.append("CANDIDATE_HYPOTHESES.md (Gate 15):")
    rm.append(f"  baseline / observed: {CANDIDATE_HYPOTHESES_BASELINE_SHA}")
    rm.append("")
    rm.append("Determinism (Gate 12):")
    if args.single_run:
        rm.append("  SKIPPED (--single-run)")
    else:
        for k, v in determinism.items():
            rm.append(f"  {k}: {v}")
    rm.append("")
    rm.append("Gate dispositions:")
    for k in sorted(disp.keys()):
        rm.append(f"  {k}: {disp[k]}")
    rm.append("")
    rm.append("No auto-commit (Gate 14): script never invokes git. Caller verifies.")
    rm_path.write_text("\n".join(rm) + "\n", encoding="utf-8")

    print(f"\n[Manifest] {rm_path}")
    print(f"\nMemory peak: {peak / (1024 * 1024):.1f} MB (target < 1024 MB)")
    overall = time.time() - t_overall_start
    print(f"Total wallclock: {overall:.1f}s")
    print("\nAll outputs written. Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
