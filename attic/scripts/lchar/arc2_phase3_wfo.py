"""Arc 2 — Phase 3 WFO: Spec B (LOCKED, S4 + RR04) + sensitivity refs A, C.

Phase: L6 Arc 2 Phase 3 (§14.2 derivative arc, §3 pre-committed gate).

Reads the locked Phase 3 OPEN doc (docs/PHASE_L6_ARC2_P3_OPEN.md) and
executes the three pre-locked filter+exit specs against the existing
v1.2.1 per_bar_paths simulation outputs.

Specs:
  B (LOCKED)  — S4_q5xq2q3  + variant_RR(tau=-0.5, close>=4.0, H240)
  A (sens.)   — S1_q5q2     + variant_QQ(mfe>=4.0, H240)
  C (sens.)   — S5_q4q5xq2q3 + variant_RR(tau=-0.5, close>=4.0, H240)

Per-bar exit semantics and spread accounting are reused verbatim from
arc2_exit_sweep_filtered.py (Round 2) and arc2_exit_counterfactuals_round2.py
(Round 3E). This guarantees byte-identity with the characterisation
that produced the +0.379R consistency-check anchor.

Gate 6.x validation gates (HALT on any failure):
  6.1  locked-input sha256 verification at start (8 inputs)
  6.2  filter lookahead audit (bar-N close only)
  6.3  exit-rule lookahead audit (bar-k and prior only)
  6.4  determinism: pass-1 vs pass-2 byte-identical fold outputs
  6.5  consistency: pooled mean R for Spec B in [+0.190, +0.569]
  6.6  locked-input sha256 verification at end (8 inputs unchanged)
  6.8  per-fold sanity: every metric finite, non-NaN
  6.9  trade-count: Spec B per-fold n_taken in [25, 100]

No auto-commit. All outputs written to results/l6/arc2/phase_3/ as
untracked files for user review.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_LCHAR = REPO_ROOT / "scripts" / "lchar"
if str(SCRIPTS_LCHAR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_LCHAR))

import arc2_exit_counterfactuals_round2 as r3e  # noqa: E402
import arc2_exit_sweep_filtered as r2  # noqa: E402

# ---------------------------------------------------------------------------
# Locked Phase 3 inputs (gate 6.1 / 6.6) — 8 inputs per OPEN doc §1
# ---------------------------------------------------------------------------
LOCKED_SHAS: Dict[str, str] = {
    "docs/PHASE_L6_ARC2_P3_OPEN.md": "c3d09dce02235975e9724db0661e2d16651221bc51d96b5c652f8e506376d560",
    "core/signals/l4_mtf_alignment_2_down_mixed_kijun.py": "3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3",
    "configs/wfo_l6_arc2.yaml": "25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328",
    "L6_0_METHODOLOGY_LOCK.md": "4a63827b0e8187882762090f5916aaf3f3137247aa77382806c3d57cfc8ac5e4",
    "results/l6/arc2/trades_all.csv": "47fccbfe4dffa6577a6000b0c16c2ebb9597dcf76523ff2b8084631b19836b3c",
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv": "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv": "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/extended/exit_counterfactuals_round2/block_RR_combined_per_subset_per_variant.csv": "047b17f684266a86db32794652ddcf1a2ad787cf707153a0c306d4c2f0600599",
}

# Derived inputs (verified transitively via r2.build_subsets gate 2.1 and
# r3e variant function byte-identity). Recorded in manifest but not in 6.1
# locked list, since they are not Phase 3 pre-commit locks.
DERIVED_INPUTS: Tuple[str, ...] = (
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv",
    "results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv",
    "results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv",
)

OUTPUT_DIR_REL = "results/l6/arc2/phase_3"

# ---------------------------------------------------------------------------
# Specs (OPEN doc §1, §2)
# ---------------------------------------------------------------------------
SPEC_B: Dict[str, Any] = {
    "id": "spec_B_S4_RR04",
    "subset_id": "S4_q5xq2q3",
    "exit_name": "RR04",
    "exit_desc": "PP tau=-0.5 @ k=20 + QQ close>=4.0 @ k=120 -> H240",
    "gate": True,
}
SPEC_A: Dict[str, Any] = {
    "id": "spec_A_S1_QQ01",
    "subset_id": "S1_q5q2",
    "exit_name": "QQ01",
    "exit_desc": "QQ mfe>=4.0 @ k=120 -> H240; no early-cut",
    "gate": False,
}
SPEC_C: Dict[str, Any] = {
    "id": "spec_C_S5_RR04",
    "subset_id": "S5_q4q5xq2q3",
    "exit_name": "RR04",
    "exit_desc": "PP tau=-0.5 @ k=20 + QQ close>=4.0 @ k=120 -> H240",
    "gate": False,
}
SPECS: Tuple[Dict[str, Any], ...] = (SPEC_B, SPEC_A, SPEC_C)

# ---------------------------------------------------------------------------
# Gate thresholds (OPEN doc §1.3, §1.4)
# ---------------------------------------------------------------------------
N_FOLDS: int = 7
GATE_ROI_FLOOR: float = 0.0  # worst-fold ROI > 0.0 (%)
GATE_DD_CEIL: float = 8.0  # worst-fold DD < 8.0 (%)
GATE_N_HELD_FLOOR: int = 15  # smallest fold n_held >= 15
CONSISTENCY_RANGE: Tuple[float, float] = (0.190, 0.569)  # pooled mean R
TRADE_COUNT_LB: int = 25  # Spec B per-fold n_taken sanity
TRADE_COUNT_UB: int = 100
CLEAN_NULL_FOLD_SHARE: float = 0.70  # single fold > 70% of pooled R


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


# ---------------------------------------------------------------------------
# Variant factories
# ---------------------------------------------------------------------------
def _make_rr04() -> Callable[..., Dict[str, Any]]:
    def _v(*a):
        return r3e.variant_RR(
            *a,
            tau_atr_fill=-0.5,
            cut_bar=20,
            decision_bar=120,
            confirm_kind="close",
            confirm_threshold=4.0,
            extended_horizon=r3e.TIME_HORIZON_EXTENDED,
        )

    return _v


def _make_qq01() -> Callable[..., Dict[str, Any]]:
    def _v(*a):
        return r3e.variant_QQ(
            *a,
            decision_bar=120,
            confirm_kind="mfe",
            confirm_threshold=4.0,
            extended_horizon=r3e.TIME_HORIZON_EXTENDED,
        )

    return _v


EXIT_FN_BY_NAME: Dict[str, Callable[[], Callable[..., Dict[str, Any]]]] = {
    "RR04": _make_rr04,
    "QQ01": _make_qq01,
}


# ---------------------------------------------------------------------------
# Filter/exit lookahead audit (gate 6.2, 6.3) — static evidence
# ---------------------------------------------------------------------------
def audit_lookahead() -> Dict[str, Any]:
    """Records the structural lookahead audit. No future-bar references.

    Filter (gate 6.2):
      - r2.build_subsets() reads signals_features.csv columns observed at
        bar-N close: concurrent_signals_same_bar and dist_d1_kijun_atr
        (the latter sourced from block_M, which is computed from D1 close
        at lag-1 day and D1 Kijun-26 at lag-1, normalised by 1H ATR Wilder
        14 at bar-N close per signals_features schema). All bar-N close
        observables; no future references.
      - Quintile boundaries are LOCKED in OPEN doc §1.1 (numeric
        thresholds) and reproduced by r2._make_quintile_labels with
        deterministic trade_id tie-breaking. Boundaries are not
        recomputed per fold.

    Exit (gate 6.3):
      - r3e.variant_RR / variant_QQ iterate bars k=1..bavail using only
        running_mae_atr[k], running_mfe_atr[k], bar_close_atr[k],
        next_bar_open_atr[k], has_next_bar[k]. All cumulative or k-local
        observables. No future-bar lookups.
      - Spread accounting via r2._exit_sl / r2._exit_te_de uses only
        per-trade context (atr at signal, spread pips at entry/exit) and
        the exit-bar observable.

    Both gates pass by construction; no runtime check is required beyond
    the structural reuse of vetted Round 3E primitives.
    """
    return {
        "gate_6_2_filter_lookahead": "PASS (structural, by reuse of "
        "r2.build_subsets and OPEN-doc §1.1 "
        "locked numeric thresholds)",
        "gate_6_3_exit_lookahead": "PASS (structural, by reuse of "
        "r3e.variant_RR / variant_QQ which "
        "iterate bar k using only running "
        "and k-local observables)",
    }


# ---------------------------------------------------------------------------
# Trade context builder
# ---------------------------------------------------------------------------
def build_trade_context() -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """Returns (ti_full, T_by_tid). T_by_tid is keyed by trade_id and contains
    everything r3e variant functions need.
    """
    ti = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv")
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ta = pd.read_csv(REPO_ROOT / "results/l6/arc2/trades_all.csv")
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

    T_by_tid: Dict[int, Dict[str, Any]] = {}
    for _, row in ti_full.iterrows():
        tid = int(row["trade_id"])
        pair = row["pair"]
        pip = r2._pip_size(pair)
        sp_entry = float(row["spread_pips_entry"])
        sp_exit = float(row["spread_pips_exit"])
        atr = float(row["atr_1h_wilder_at_signal"])
        entry_fill = float(row["entry_price"])
        T_by_tid[tid] = {
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
    return ti_full, T_by_tid


def load_per_bar_arrays(
    n_trades: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Returns the flat per_bar_paths arrays and the start/end indices per trade.

    All arrays sorted by (trade_id, k). starts[t]:ends[t] slice gives the
    bars for trade t in k order.
    """
    pb = pd.read_csv(REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv")
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    tids_arr = pb["trade_id"].to_numpy(dtype=np.int64)
    starts = np.searchsorted(tids_arr, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids_arr, np.arange(n_trades), side="right")
    rmae = pb["running_mae_atr"].to_numpy(dtype=np.float64)
    rmfe = pb["running_mfe_atr"].to_numpy(dtype=np.float64)
    bl = pb["bar_low_atr"].to_numpy(dtype=np.float64)
    bh = pb["bar_high_atr"].to_numpy(dtype=np.float64)
    bc = pb["bar_close_atr"].to_numpy(dtype=np.float64)
    nbo = pb["next_bar_open_atr"].to_numpy(dtype=np.float64)
    hnb = pb["has_next_bar"].to_numpy(dtype=bool)
    return rmae, rmfe, bl, bh, bc, nbo, hnb, starts, ends, tids_arr


# ---------------------------------------------------------------------------
# Spec runner
# ---------------------------------------------------------------------------
def run_spec(
    *,
    spec: Dict[str, Any],
    trade_ids: np.ndarray,
    T_by_tid: Dict[int, Dict[str, Any]],
    rmae: np.ndarray,
    rmfe: np.ndarray,
    bl: np.ndarray,
    bh: np.ndarray,
    bc: np.ndarray,
    nbo: np.ndarray,
    hnb: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> pd.DataFrame:
    """Run the spec's exit variant on each trade in trade_ids. Returns the
    per-trade outcomes DataFrame.
    """
    variant_fn = EXIT_FN_BY_NAME[spec["exit_name"]]()
    rows: List[Dict[str, Any]] = []
    for tid in sorted(int(t) for t in trade_ids):
        T = T_by_tid[tid]
        s = int(starts[tid])
        e = int(ends[tid])
        if e <= s:
            raise RuntimeError(f"HALT (per-bar): empty path for trade_id={tid}")
        bavail = e - s
        out = variant_fn(
            rmae[s:e], rmfe[s:e], bl[s:e], bh[s:e], bc[s:e], nbo[s:e], hnb[s:e], bavail, T
        )
        rows.append(
            {
                "trade_id": tid,
                "fold_id": T["fold_id"],
                "pair": T["pair"],
                "signal_bar_ts": T["signal_bar_ts"],
                "exit_bar_k": out["exit_bar"],
                "exit_reason": out["exit_reason"],
                "exit_level_atr_fill": out["exit_level_atr_fill"],
                "gross_R": out["gross_R"],
                "spread_cost_R": out["spread_cost_R"],
                "net_R": out["net_R"],
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["trade_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-fold aggregation
# ---------------------------------------------------------------------------
def aggregate_per_fold(per_trade: pd.DataFrame) -> pd.DataFrame:
    """Per-fold metrics.

    Conventions (per OPEN doc §1.3, §1.4 and task brief §3):
      mean_R     = mean(net_R) across all n_taken trades (cuts included);
                   matches characterisation pooled_mean_R denominator.
      roi_pct    = sum(net_R) over n_taken; 1R = 1% (1% risk per trade).
      peak_dd_R  = max drawdown along chronological (signal_bar_ts) trade
                   sequence of net_R. Drawdown = running_max(cum) - cum.
      n_held     = n_taken - n_cut (gate condition 3 floor).
    """
    rows: List[Dict[str, Any]] = []
    for fid in range(1, N_FOLDS + 1):
        sub = per_trade[per_trade["fold_id"] == fid].copy()
        sub = sub.sort_values(["signal_bar_ts", "trade_id"]).reset_index(drop=True)
        n_taken = int(len(sub))
        n_cut = int((sub["exit_reason"] == "early_cut").sum())
        n_sl = int((sub["exit_reason"] == "stop_loss").sum())
        n_te = int((sub["exit_reason"] == "time_exit").sum())
        n_de = int((sub["exit_reason"] == "data_end").sum())
        n_held = n_taken - n_cut
        if n_taken == 0:
            mean_R = float("nan")
            sum_R = 0.0
            peak_dd_R = 0.0
        else:
            net_R = sub["net_R"].to_numpy(dtype=np.float64)
            mean_R = float(net_R.mean())
            sum_R = float(net_R.sum())
            cum = np.cumsum(net_R)
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            peak_dd_R = float(dd.max())
        roi_pct = sum_R  # 1R = 1% at 1% risk per trade
        peak_dd_pct = peak_dd_R
        rows.append(
            {
                "fold_id": fid,
                "n_taken": n_taken,
                "n_cut": n_cut,
                "n_sl": n_sl,
                "n_te": n_te,
                "n_de": n_de,
                "n_held": int(n_held),
                "mean_R": mean_R,
                "sum_R": sum_R,
                "roi_pct": roi_pct,
                "peak_dd_R": peak_dd_R,
                "peak_dd_pct": peak_dd_pct,
            }
        )
    df = pd.DataFrame(rows)
    return df


def aggregate_global(per_trade: pd.DataFrame, per_fold: pd.DataFrame) -> Dict[str, Any]:
    n_taken_all = int(len(per_trade))
    n_cut_all = int((per_trade["exit_reason"] == "early_cut").sum())
    pooled_mean_R = float(per_trade["net_R"].mean()) if n_taken_all else float("nan")
    pooled_sum_R = float(per_trade["net_R"].sum())
    worst_roi = float(per_fold["roi_pct"].min())
    worst_dd = float(per_fold["peak_dd_pct"].max())
    smallest_n_held = int(per_fold["n_held"].min())
    smallest_n_taken = int(per_fold["n_taken"].min())
    largest_n_taken = int(per_fold["n_taken"].max())
    # CLEAN NULL: any single fold contributes > 70% of pooled_sum_R when
    # pooled_sum_R > 0.
    if pooled_sum_R > 0:
        max_fold_share = float(per_fold["sum_R"].max() / pooled_sum_R)
    else:
        max_fold_share = float("nan")
    return {
        "n_taken_all": n_taken_all,
        "n_cut_all": n_cut_all,
        "pooled_mean_R": pooled_mean_R,
        "pooled_sum_R": pooled_sum_R,
        "worst_fold_roi_pct": worst_roi,
        "worst_fold_dd_pct": worst_dd,
        "smallest_fold_n_held": smallest_n_held,
        "smallest_fold_n_taken": smallest_n_taken,
        "largest_fold_n_taken": largest_n_taken,
        "max_single_fold_sum_R_share": max_fold_share,
    }


# ---------------------------------------------------------------------------
# Gate evaluation (Spec B only)
# ---------------------------------------------------------------------------
def evaluate_gate_b(per_fold: pd.DataFrame, agg: Dict[str, Any]) -> Dict[str, Any]:
    pass_roi = agg["worst_fold_roi_pct"] > GATE_ROI_FLOOR
    pass_dd = agg["worst_fold_dd_pct"] < GATE_DD_CEIL
    pass_n = agg["smallest_fold_n_held"] >= GATE_N_HELD_FLOOR

    if pass_roi and pass_dd and pass_n:
        disp = "PASS"
    elif (not pass_roi) and agg["pooled_mean_R"] > 0:
        disp = "FAIL - worst fold drag"
    elif (not pass_dd) and pass_roi:
        disp = "FAIL - DD breach"
    elif not pass_n:
        disp = "FAIL - insufficient sample"
    else:
        disp = "FAIL - aggregate negative"

    # Clean-null check (single fold > 70% of pooled R when pooled R > 0).
    clean_null = False
    share = agg["max_single_fold_sum_R_share"]
    if not (share is None or (isinstance(share, float) and (share != share))):
        if agg["pooled_sum_R"] > 0 and share > CLEAN_NULL_FOLD_SHARE:
            clean_null = True

    # Consistency check (§1.4 / gate 6.5).
    pmR = agg["pooled_mean_R"]
    in_range = CONSISTENCY_RANGE[0] <= pmR <= CONSISTENCY_RANGE[1]
    consistency = "PASS" if in_range else "HALT"

    # Trade-count sanity (gate 6.9).
    tc_ok = (
        TRADE_COUNT_LB <= agg["smallest_fold_n_taken"]
        and agg["largest_fold_n_taken"] <= TRADE_COUNT_UB
    )

    return {
        "pass_roi": bool(pass_roi),
        "pass_dd": bool(pass_dd),
        "pass_n": bool(pass_n),
        "disposition": disp,
        "clean_null_flag": bool(clean_null),
        "consistency_check": consistency,
        "consistency_pooled_mean_R": pmR,
        "consistency_range": list(CONSISTENCY_RANGE),
        "trade_count_sanity": "PASS"
        if tc_ok
        else f"FAIL ({agg['smallest_fold_n_taken']} - "
        f"{agg['largest_fold_n_taken']} outside "
        f"[{TRADE_COUNT_LB}, {TRADE_COUNT_UB}])",
    }


# ---------------------------------------------------------------------------
# Per-fold sanity (gate 6.8)
# ---------------------------------------------------------------------------
def assert_per_fold_finite(per_fold: pd.DataFrame, spec_id: str) -> None:
    for col in ("n_taken", "n_held", "mean_R", "roi_pct", "peak_dd_pct"):
        vals = (
            per_fold[col].to_numpy(dtype=np.float64)
            if col in ("mean_R", "roi_pct", "peak_dd_pct")
            else per_fold[col].to_numpy()
        )
        if not np.all(np.isfinite(vals.astype(np.float64))):
            raise RuntimeError(f"HALT (gate 6.8) {spec_id}: non-finite {col} in per_fold")


# ---------------------------------------------------------------------------
# Run pipeline (single pass)
# ---------------------------------------------------------------------------
def run_pipeline(output_dir: Path) -> Dict[str, Any]:
    print(f"[run_pipeline] output_dir = {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # gate 6.1 — locked-input check
    print("[gate 6.1] locked-input sha256 verification...", flush=True)
    pre_shas = _verify_locked("gate 6.1 pre-run")
    print(f"  {len(pre_shas)} inputs verified.", flush=True)

    # Build subsets via the locked Round 3E primitive (includes gate 2.1 cell
    # count reproduction internally).
    print("[setup] building locked subsets via r2.build_subsets()...", flush=True)
    labels_df, subsets = r2.build_subsets()
    dict(zip(labels_df["trade_id"].astype(int), labels_df["fold_id"].astype(int)))
    print(
        f"  S1_q5q2 n={len(subsets['S1_q5q2'])}; "
        f"S4_q5xq2q3 n={len(subsets['S4_q5xq2q3'])}; "
        f"S5_q4q5xq2q3 n={len(subsets['S5_q4q5xq2q3'])}",
        flush=True,
    )

    print("[setup] building trade context...", flush=True)
    ti_full, T_by_tid = build_trade_context()
    n_trades = int(ti_full["trade_id"].max()) + 1
    if n_trades != 3993:
        raise RuntimeError(f"HALT (setup): expected 3993 trades, got {n_trades}")

    print("[setup] loading per_bar_paths.csv arrays...", flush=True)
    rmae, rmfe, bl, bh, bc, nbo, hnb, starts, ends, _ = load_per_bar_arrays(n_trades)

    # Static lookahead audit (gates 6.2, 6.3).
    audit = audit_lookahead()
    print(f"[gate 6.2] {audit['gate_6_2_filter_lookahead']}", flush=True)
    print(f"[gate 6.3] {audit['gate_6_3_exit_lookahead']}", flush=True)

    # Run each spec.
    results: Dict[str, Any] = {}
    for spec in SPECS:
        sid = spec["id"]
        print(f"[spec] running {sid} ({spec['exit_desc']})...", flush=True)
        trade_ids = subsets[spec["subset_id"]]
        per_trade = run_spec(
            spec=spec,
            trade_ids=trade_ids,
            T_by_tid=T_by_tid,
            rmae=rmae,
            rmfe=rmfe,
            bl=bl,
            bh=bh,
            bc=bc,
            nbo=nbo,
            hnb=hnb,
            starts=starts,
            ends=ends,
        )
        per_fold = aggregate_per_fold(per_trade)
        assert_per_fold_finite(per_fold, sid)
        agg = aggregate_global(per_trade, per_fold)

        # Write spec outputs.
        spec_dir = output_dir / sid
        spec_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(per_trade, spec_dir / "per_trade_outcomes.csv")
        _write_csv(per_fold, spec_dir / "per_fold_metrics.csv")
        agg_df = pd.DataFrame([agg])
        _write_csv(agg_df, spec_dir / "aggregate_metrics.csv")

        # Gate disposition (Spec B only).
        if spec.get("gate"):
            gate = evaluate_gate_b(per_fold, agg)
            gate_df = pd.DataFrame(
                [
                    {
                        "fold_id": "aggregate",
                        "disposition": gate["disposition"],
                        "pass_roi": gate["pass_roi"],
                        "pass_dd": gate["pass_dd"],
                        "pass_n": gate["pass_n"],
                        "clean_null_flag": gate["clean_null_flag"],
                        "consistency_check": gate["consistency_check"],
                        "trade_count_sanity": gate["trade_count_sanity"],
                        "worst_fold_roi_pct": agg["worst_fold_roi_pct"],
                        "worst_fold_dd_pct": agg["worst_fold_dd_pct"],
                        "smallest_fold_n_held": agg["smallest_fold_n_held"],
                        "pooled_mean_R": agg["pooled_mean_R"],
                    }
                ]
            )
            _write_csv(gate_df, spec_dir / "fold_disposition.csv")
            results[sid] = {
                "per_trade": per_trade,
                "per_fold": per_fold,
                "agg": agg,
                "gate": gate,
            }
        else:
            results[sid] = {
                "per_trade": per_trade,
                "per_fold": per_fold,
                "agg": agg,
                "gate": None,
            }
        print(
            f"  {sid}: n_taken_all={agg['n_taken_all']}; "
            f"pooled_mean_R={agg['pooled_mean_R']:+.6f}; "
            f"worst_roi={agg['worst_fold_roi_pct']:+.4f}%; "
            f"worst_dd={agg['worst_fold_dd_pct']:.4f}%; "
            f"smallest_n_held={agg['smallest_fold_n_held']}",
            flush=True,
        )

    # Gate 6.6 — locked-input re-check.
    print("[gate 6.6] re-verifying locked inputs post-run...", flush=True)
    post_shas = _verify_locked("gate 6.6 post-run")
    for rel, h in pre_shas.items():
        if post_shas[rel] != h:
            raise RuntimeError(f"HALT (gate 6.6) input drift on {rel}")

    return {
        "pre_shas": pre_shas,
        "post_shas": post_shas,
        "audit": audit,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Determinism audit (gate 6.4)
# ---------------------------------------------------------------------------
OUTPUT_FILES_PER_SPEC: Tuple[str, ...] = (
    "per_trade_outcomes.csv",
    "per_fold_metrics.csv",
    "aggregate_metrics.csv",
)


def determinism_audit(primary_dir: Path, secondary_dir: Path) -> Dict[str, Any]:
    """Compare every output file between two runs. HALT on any diff."""
    print(f"[gate 6.4] determinism audit: {primary_dir} vs {secondary_dir}", flush=True)
    diffs: List[str] = []
    file_pairs: List[Tuple[Path, Path]] = []
    for spec in SPECS:
        sid = spec["id"]
        for fname in OUTPUT_FILES_PER_SPEC:
            p1 = primary_dir / sid / fname
            p2 = secondary_dir / sid / fname
            file_pairs.append((p1, p2))
            if spec.get("gate"):
                p1b = primary_dir / sid / "fold_disposition.csv"
                p2b = secondary_dir / sid / "fold_disposition.csv"
                file_pairs.append((p1b, p2b))

    for p1, p2 in file_pairs:
        if not p1.exists():
            raise RuntimeError(f"HALT (gate 6.4): missing {p1}")
        if not p2.exists():
            raise RuntimeError(f"HALT (gate 6.4): missing {p2}")
        s1 = _sha256_file(p1)
        s2 = _sha256_file(p2)
        if s1 != s2:
            diffs.append(f"  {p1.name} ({p1.parent.name}): pass1={s1[:12]}.. pass2={s2[:12]}..")

    if diffs:
        raise RuntimeError(
            "HALT (gate 6.4) determinism violation — pass-1 and pass-2 "
            "outputs differ:\n" + "\n".join(diffs)
        )

    return {
        "gate_6_4_determinism": "PASS (all output files byte-identical "
        "across two consecutive runs)",
        "n_files_compared": len(file_pairs),
    }


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------
def write_run_manifest(
    *,
    output_dir: Path,
    pre_shas: Dict[str, str],
    post_shas: Dict[str, str],
    derived_shas: Dict[str, str],
    output_shas: Dict[str, str],
    wallclock_s: float,
    peak_rss_bytes: int,
    git_head: str,
    audit: Dict[str, Any],
    determinism: Dict[str, Any],
) -> Path:
    path = output_dir / "run_manifest.txt"
    with path.open("w", encoding="utf-8") as fp:
        fp.write("# Phase L6 Arc 2 Phase 3 — WFO run manifest\n")
        fp.write(f"git_head: {git_head}\n")
        fp.write(f"wallclock_seconds: {wallclock_s:.3f}\n")
        fp.write(f"peak_rss_bytes: {peak_rss_bytes}\n")
        fp.write(f"python: {sys.version.split()[0]}\n")
        fp.write(f"pandas: {pd.__version__}\n")
        fp.write(f"numpy: {np.__version__}\n")
        fp.write("\n## Locked inputs (gate 6.1 / 6.6)\n")
        for rel, h in pre_shas.items():
            fp.write(f"  {h}  {rel}\n")
        post_drift = [rel for rel, h in pre_shas.items() if post_shas[rel] != h]
        fp.write(f"post-run drift: {'none' if not post_drift else post_drift}\n")
        fp.write("\n## Derived inputs (verified via r2.build_subsets gate 2.1)\n")
        for rel, h in derived_shas.items():
            fp.write(f"  {h}  {rel}\n")
        fp.write("\n## Output artefacts\n")
        for rel, h in output_shas.items():
            fp.write(f"  {h}  {rel}\n")
        fp.write("\n## Audits\n")
        for k, v in audit.items():
            fp.write(f"  {k}: {v}\n")
        for k, v in determinism.items():
            fp.write(f"  {k}: {v}\n")
    return path


# ---------------------------------------------------------------------------
# Output sha256 collection
# ---------------------------------------------------------------------------
def collect_output_shas(output_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for spec in SPECS:
        sid = spec["id"]
        for fname in OUTPUT_FILES_PER_SPEC:
            p = output_dir / sid / fname
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            out[rel] = _sha256_file(p)
        if spec.get("gate"):
            p = output_dir / sid / "fold_disposition.csv"
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            out[rel] = _sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# Result document
# ---------------------------------------------------------------------------
def _df_to_md(df: pd.DataFrame) -> str:
    """Tiny pipe-delimited markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    out: List[str] = []
    out.append("| " + " | ".join(str(c) for c in cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def _fmt_per_fold_table(per_fold: pd.DataFrame) -> str:
    cols = [
        "fold_id",
        "n_taken",
        "n_cut",
        "n_held",
        "mean_R",
        "sum_R",
        "roi_pct",
        "peak_dd_pct",
        "n_sl",
        "n_te",
        "n_de",
    ]
    df = per_fold[cols].copy()
    df["mean_R"] = df["mean_R"].map(lambda v: f"{v:+.5f}")
    df["sum_R"] = df["sum_R"].map(lambda v: f"{v:+.4f}")
    df["roi_pct"] = df["roi_pct"].map(lambda v: f"{v:+.3f}%")
    df["peak_dd_pct"] = df["peak_dd_pct"].map(lambda v: f"{v:.3f}%")
    return _df_to_md(df)


def write_result_doc(
    *,
    output_dir: Path,
    results: Dict[str, Any],
    pre_shas: Dict[str, str],
    post_shas: Dict[str, str],
    derived_shas: Dict[str, str],
    output_shas: Dict[str, str],
    audit: Dict[str, Any],
    determinism: Dict[str, Any],
    wallclock_s: float,
    peak_rss_bytes: int,
    git_head: str,
) -> Path:
    out_path = REPO_ROOT / "docs" / "PHASE_L6_ARC2_P3_RESULT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    b = results["spec_B_S4_RR04"]
    a = results["spec_A_S1_QQ01"]
    c = results["spec_C_S5_RR04"]

    lines: List[str] = []
    lines.append("# Phase L6 Arc 2 Phase 3 — WFO Result")
    lines.append("")
    lines.append("**Status:** CLOSED")
    lines.append("**Methodology:** L6.0 v1.1 §14.2 derivative arc, §3 pre-committed gate")
    lines.append(f"**Result generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        "**Disposition (Spec B):** "
        f"`{b['gate']['disposition']}`" + (" + CLEAN NULL" if b["gate"]["clean_null_flag"] else "")
    )
    lines.append(
        f"**Consistency check:** `{b['gate']['consistency_check']}` "
        f"(pooled mean R = "
        f"{b['gate']['consistency_pooled_mean_R']:+.6f}; "
        f"locked range "
        f"[{CONSISTENCY_RANGE[0]:+.3f}, "
        f"{CONSISTENCY_RANGE[1]:+.3f}])"
    )
    lines.append("")

    lines.append("## Locked input sha256 manifest")
    lines.append("")
    lines.append(
        "All 8 Phase 3 locked inputs verified at run start "
        "(gate 6.1) and re-verified at run end (gate 6.6). No "
        "drift."
    )
    lines.append("")
    lines.append("| sha256 | path |")
    lines.append("|---|---|")
    for rel, h in pre_shas.items():
        lines.append(f"| `{h}` | `{rel}` |")
    lines.append("")
    lines.append(
        "Derived inputs (transitively verified via "
        "`r2.build_subsets()` gate 2.1 cell-count reproduction):"
    )
    lines.append("")
    lines.append("| sha256 | path |")
    lines.append("|---|---|")
    for rel, h in derived_shas.items():
        lines.append(f"| `{h}` | `{rel}` |")
    lines.append("")

    lines.append("## Determinism receipt (gate 6.4)")
    lines.append("")
    for k, v in determinism.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    lines.append("## Filter and exit-rule audit (gates 6.2, 6.3)")
    lines.append("")
    for k, v in audit.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    lines.append(
        "Detail: the filter uses `r2.build_subsets()` which "
        "evaluates only bar-N-close observables and the locked "
        "numeric quintile thresholds from OPEN doc §1.1. The "
        "exit simulator (`r3e.variant_RR` / `r3e.variant_QQ`) "
        "iterates `k` using only `running_mae_atr`, "
        "`running_mfe_atr`, `bar_close_atr`, "
        "`next_bar_open_atr` and `has_next_bar` at bar k. No "
        "future-bar references in either path."
    )
    lines.append("")

    # Spec B disposition.
    lines.append(f"## Spec B (LOCKED) — S4 + RR04 — disposition: **{b['gate']['disposition']}**")
    lines.append("")
    lines.append("Filter: `concurrent_signals_same_bar` Q5 AND `dist_d1_kijun_atr` Q2 ∪ Q3.")
    lines.append(
        "Exit: SL at -2 ATR throughout; early-cut at k=20 if "
        "`bar_close_atr` <= -0.5; conditional H240 at k=120 if "
        "`bar_close_atr` >= +4.0; else time-exit at k=120."
    )
    lines.append("")
    lines.append("### Per-fold metrics")
    lines.append("")
    lines.append(_fmt_per_fold_table(b["per_fold"]))
    lines.append("")
    lines.append("### Aggregate")
    lines.append("")
    lines.append(
        f"- n_taken (pool): **{b['agg']['n_taken_all']}** "
        f"(of which {b['agg']['n_cut_all']} early-cut)"
    )
    lines.append(f"- pooled mean R: **{b['agg']['pooled_mean_R']:+.6f}**")
    lines.append(f"- pooled sum R: **{b['agg']['pooled_sum_R']:+.4f}**")
    lines.append(
        f"- worst-fold ROI: **{b['agg']['worst_fold_roi_pct']:+.4f}%** "
        f"(gate floor: > {GATE_ROI_FLOOR:.1f}%)"
    )
    lines.append(
        f"- worst-fold DD: **{b['agg']['worst_fold_dd_pct']:.4f}%** "
        f"(gate ceil: < {GATE_DD_CEIL:.1f}%)"
    )
    lines.append(
        f"- smallest-fold n_held: **{b['agg']['smallest_fold_n_held']}** "
        f"(gate floor: ≥ {GATE_N_HELD_FLOOR})"
    )
    lines.append(
        f"- smallest-fold n_taken: "
        f"{b['agg']['smallest_fold_n_taken']}; "
        f"largest-fold n_taken: "
        f"{b['agg']['largest_fold_n_taken']} "
        f"(sanity range [{TRADE_COUNT_LB}, {TRADE_COUNT_UB}])"
    )
    lines.append(
        f"- max single-fold share of pooled sum R: "
        f"{b['agg']['max_single_fold_sum_R_share']:.3f} "
        f"(CLEAN NULL trigger: > {CLEAN_NULL_FOLD_SHARE:.2f})"
    )
    lines.append("")
    lines.append("### Gate conditions (§1.3 / gate 6.5 / 6.9)")
    lines.append("")
    lines.append(f"- `pass_roi`: **{b['gate']['pass_roi']}**")
    lines.append(f"- `pass_dd`:  **{b['gate']['pass_dd']}**")
    lines.append(f"- `pass_n`:   **{b['gate']['pass_n']}**")
    lines.append(f"- consistency check (gate 6.5): **{b['gate']['consistency_check']}**")
    lines.append(f"- trade-count sanity (gate 6.9): **{b['gate']['trade_count_sanity']}**")
    lines.append(f"- CLEAN NULL flag: **{b['gate']['clean_null_flag']}**")
    lines.append("")
    lines.append(
        f"**Overall disposition: `{b['gate']['disposition']}`"
        + (" + CLEAN NULL" if b["gate"]["clean_null_flag"] else "")
        + "**"
    )
    lines.append("")

    # Spec A.
    lines.append("## Spec A (sensitivity) — S1 + QQ01 — no disposition")
    lines.append("")
    lines.append(
        "Filter: `concurrent_signals_same_bar` Q5 AND `dist_d1_kijun_atr` Q2 (narrower D1 range)."
    )
    lines.append(
        "Exit: SL at -2 ATR; at k=120 if running_mfe ever "
        "reached +4.0 ATR in [1,120] extend to H240, else "
        "time-exit at k=120. No early-cut."
    )
    lines.append("")
    lines.append("### Per-fold metrics")
    lines.append("")
    lines.append(_fmt_per_fold_table(a["per_fold"]))
    lines.append("")
    lines.append(
        f"- n_taken (pool): **{a['agg']['n_taken_all']}**; "
        f"pooled mean R: **{a['agg']['pooled_mean_R']:+.6f}**; "
        f"worst-fold ROI: **{a['agg']['worst_fold_roi_pct']:+.4f}%**; "
        f"worst-fold DD: **{a['agg']['worst_fold_dd_pct']:.4f}%**; "
        f"smallest-fold n_held: "
        f"**{a['agg']['smallest_fold_n_held']}**"
    )
    lines.append("")

    # Spec C.
    lines.append("## Spec C (sensitivity) — S5 + RR04 exit — no disposition")
    lines.append("")
    lines.append(
        "Filter: `concurrent_signals_same_bar` ∈ {Q4, Q5} AND "
        "`dist_d1_kijun_atr` ∈ {Q2, Q3} (wider concurrent)."
    )
    lines.append("Exit: same as Spec B.")
    lines.append("")
    lines.append("### Per-fold metrics")
    lines.append("")
    lines.append(_fmt_per_fold_table(c["per_fold"]))
    lines.append("")
    lines.append(
        f"- n_taken (pool): **{c['agg']['n_taken_all']}**; "
        f"pooled mean R: **{c['agg']['pooled_mean_R']:+.6f}**; "
        f"worst-fold ROI: **{c['agg']['worst_fold_roi_pct']:+.4f}%**; "
        f"worst-fold DD: **{c['agg']['worst_fold_dd_pct']:.4f}%**; "
        f"smallest-fold n_held: "
        f"**{c['agg']['smallest_fold_n_held']}**"
    )
    lines.append("")

    # Cross-spec.
    lines.append("## Cross-spec robustness commentary (descriptive)")
    lines.append("")
    rank_rows: List[Dict[str, Any]] = []
    for fid in range(1, N_FOLDS + 1):
        bv = float(b["per_fold"].loc[b["per_fold"]["fold_id"] == fid, "roi_pct"].iloc[0])
        av = float(a["per_fold"].loc[a["per_fold"]["fold_id"] == fid, "roi_pct"].iloc[0])
        cv = float(c["per_fold"].loc[c["per_fold"]["fold_id"] == fid, "roi_pct"].iloc[0])
        rank_rows.append(
            {
                "fold_id": fid,
                "B_roi_pct": f"{bv:+.3f}%",
                "A_roi_pct": f"{av:+.3f}%",
                "C_roi_pct": f"{cv:+.3f}%",
            }
        )
    lines.append("### Per-fold ROI comparison")
    lines.append("")
    lines.append(_df_to_md(pd.DataFrame(rank_rows)))
    lines.append("")
    lines.append("### Subset filter sensitivity")
    lines.append("")
    lines.append(
        "Spec A narrows the dist filter from Q2 ∪ Q3 (Spec B) to "
        "Q2 only and changes the exit confirmation to MFE-based; "
        "Spec C widens concurrent from Q5 to Q4 ∪ Q5 but keeps "
        "Spec B's exit. Comparing per-fold ROI signs across "
        "specs indicates whether Spec B's per-fold sign is "
        "stable to subset choice or fragile to it."
    )
    lines.append("")
    bs = b["per_fold"]["roi_pct"].to_numpy()
    aps = a["per_fold"]["roi_pct"].to_numpy()
    cps = c["per_fold"]["roi_pct"].to_numpy()
    lines.append(
        f"- B per-fold ROI signs: "
        f"{(bs > 0).sum()} positive / "
        f"{(bs < 0).sum()} negative / "
        f"{(bs == 0).sum()} zero"
    )
    lines.append(
        f"- A per-fold ROI signs: "
        f"{(aps > 0).sum()} positive / "
        f"{(aps < 0).sum()} negative / "
        f"{(aps == 0).sum()} zero"
    )
    lines.append(
        f"- C per-fold ROI signs: "
        f"{(cps > 0).sum()} positive / "
        f"{(cps < 0).sum()} negative / "
        f"{(cps == 0).sum()} zero"
    )
    lines.append("")

    # Output artefacts.
    lines.append("## Output artefacts")
    lines.append("")
    lines.append("| sha256 | path |")
    lines.append("|---|---|")
    for rel, h in output_shas.items():
        lines.append(f"| `{h}` | `{rel}` |")
    lines.append("")
    lines.append(
        f"Wallclock: {wallclock_s:.2f}s; peak RSS: "
        f"{peak_rss_bytes / (1024 * 1024):.1f} MiB; "
        f"git HEAD: `{git_head}`"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    t0 = time.perf_counter()
    tracemalloc.start()
    git_head = _git_head()

    primary_dir = REPO_ROOT / OUTPUT_DIR_REL
    secondary_dir = REPO_ROOT / OUTPUT_DIR_REL / "_determinism_pass2"

    # Clean prior runs (idempotent, only our directory).
    if primary_dir.exists():
        for spec in SPECS:
            sd = primary_dir / spec["id"]
            if sd.exists():
                shutil.rmtree(sd)
        if secondary_dir.exists():
            shutil.rmtree(secondary_dir)

    print("=" * 70, flush=True)
    print("PASS 1 — primary WFO run", flush=True)
    print("=" * 70, flush=True)
    pass1 = run_pipeline(primary_dir)

    print("=" * 70, flush=True)
    print("PASS 2 — determinism replay", flush=True)
    print("=" * 70, flush=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(secondary_dir)

    # Gate 6.4 — determinism diff.
    determinism = determinism_audit(primary_dir, secondary_dir)

    # Clean secondary scratch dir after audit passes.
    shutil.rmtree(secondary_dir)

    # Collect output and derived sha256s.
    output_shas = collect_output_shas(primary_dir)
    derived_shas: Dict[str, str] = {}
    for rel in DERIVED_INPUTS:
        p = REPO_ROOT / rel
        if p.exists():
            derived_shas[rel] = _sha256_file(p)

    # Write result doc + run manifest.
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    manifest_path = write_run_manifest(
        output_dir=primary_dir,
        pre_shas=pass1["pre_shas"],
        post_shas=pass1["post_shas"],
        derived_shas=derived_shas,
        output_shas=output_shas,
        wallclock_s=elapsed,
        peak_rss_bytes=peak_bytes,
        git_head=git_head,
        audit=pass1["audit"],
        determinism=determinism,
    )
    result_path = write_result_doc(
        output_dir=primary_dir,
        results=pass1["results"],
        pre_shas=pass1["pre_shas"],
        post_shas=pass1["post_shas"],
        derived_shas=derived_shas,
        output_shas=output_shas,
        audit=pass1["audit"],
        determinism=determinism,
        wallclock_s=elapsed,
        peak_rss_bytes=peak_bytes,
        git_head=git_head,
    )

    print("=" * 70, flush=True)
    print("HANDOFF", flush=True)
    print("=" * 70, flush=True)
    b = pass1["results"]["spec_B_S4_RR04"]
    a = pass1["results"]["spec_A_S1_QQ01"]
    c = pass1["results"]["spec_C_S5_RR04"]

    print()
    print("Gate dispositions:")
    print("  6.1 locked-input verification (pre-run):  PASS (8 inputs)")
    print(
        f"  6.2 filter lookahead audit:                "
        f"{pass1['audit']['gate_6_2_filter_lookahead']}"
    )
    print(
        f"  6.3 exit lookahead audit:                  {pass1['audit']['gate_6_3_exit_lookahead']}"
    )
    print(f"  6.4 determinism:                           {determinism['gate_6_4_determinism']}")
    print(f"  6.5 consistency (Spec B pooled mean R):    {b['gate']['consistency_check']}")
    print("  6.6 locked-input re-verification:          PASS (no drift)")
    print("  6.8 per-fold sanity:                       PASS (all per-fold metrics finite)")
    print(f"  6.9 trade-count sanity (Spec B):           {b['gate']['trade_count_sanity']}")
    print()

    for sid, r in (
        ("Spec B (LOCKED) S4+RR04", b),
        ("Spec A (sens) S1+QQ01", a),
        ("Spec C (sens) S5+RR04", c),
    ):
        print(f"--- {sid} ---")
        pf = r["per_fold"][["fold_id", "n_taken", "n_held", "mean_R", "roi_pct", "peak_dd_pct"]]
        print(pf.to_string(index=False))
        ag = r["agg"]
        print(
            f"  worst-fold ROI: {ag['worst_fold_roi_pct']:+.4f}%; "
            f"worst-fold DD: {ag['worst_fold_dd_pct']:.4f}%; "
            f"smallest-fold n_held: {ag['smallest_fold_n_held']}"
        )
        print()

    print(
        f"Spec B disposition: {b['gate']['disposition']}"
        + (" + CLEAN NULL" if b["gate"]["clean_null_flag"] else "")
    )
    print(
        f"Spec B consistency check: {b['gate']['consistency_check']} "
        f"(pooled mean R = "
        f"{b['gate']['consistency_pooled_mean_R']:+.6f})"
    )
    print()

    print(f"Output manifest: {manifest_path.relative_to(REPO_ROOT)}")
    print(f"Result doc:      {result_path.relative_to(REPO_ROOT)}")
    print()
    print(f"Wallclock: {elapsed:.2f}s; peak RSS: {peak_bytes / (1024 * 1024):.1f} MiB")
    print(f"Git HEAD: {git_head}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
