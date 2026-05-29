"""§2.16 Deliberate-lookahead spot check.

Plants two distinct in-memory lookahead bugs in the DLR signal module
and verifies that this audit's §2.1 byte-compare + NaN-perturbation
tests detect them. Each bug is fully reverted before the next test.

Bugs:
  1. ``swing_search_into_future`` — swap ``d_search_max = d_t - 4`` for
     ``d_search_max = d_t + 1`` so the swing-low search reads D1 bars
     including tomorrow's D1.  Should be caught by the truncation
     byte-compare (recomputed value would be NaN if the future bars
     aren't in the trimmed arena) AND by the post-signal H4 NaN-pert
     (NaN'ing future H4 bars doesn't help — the bug is in D1 — but the
     D1[d_t] NaN-pert WOULD show the signal flipping because d_t is
     now read).
  2. ``atr_uses_future_bar`` — replace the local ``wilder_atr`` with a
     variant that uses ``tr[t+1]`` instead of ``tr[t]`` when updating
     ATR at position ``t``. Should be caught by the truncation
     byte-compare (recomputed ATR differs at the last bar of the
     trimmed arena) AND by the post-signal NaN-perturbation
     (NaN'ing H4 bar t+1 changes the ATR at bar t).

Method: import the module, monkey-patch the relevant function (or a
local variant), run a tiny version of the §2.1 tests against a small
sample, record whether the tests caught the bug. The actual on-disk
module is NEVER modified — all perturbations are in-memory.

Output: results/l_arc_10_v3.0.2/exhaustive_audit/section_2_16_deliberate_lookahead.json
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "exhaustive_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "section_2_16_deliberate_lookahead.json"

POOL_PATH = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "step_1" / "pool.parquet"

BOUNDARY_CONVENTION = "5ers_eet"
WINDOW_START = "2010-01-01"
WINDOW_END = "2026-04-30"
PAD_DAYS = 45

# Test scope per bug. Keep modest — point is detectability, not exhaustion.
TEST_PAIRS = ("AUDCAD", "EURUSD", "GBPJPY")
N_PER_PAIR = 5
ATOL = 1e-9


def _load_pair(pair: str, dlr_mod):
    from core.data.aggregator import aggregate
    from scripts.l_arc_10_v3._common import bid_view_for_signal
    h4 = aggregate(
        pair, "H4",
        histdata_root="C:/Users/panap/Documents/Forex-Backtester/data/histdata",
        cache_root="data/cache",
        boundary_convention=BOUNDARY_CONVENTION,
    )
    d1 = aggregate(
        pair, "D1",
        histdata_root="C:/Users/panap/Documents/Forex-Backtester/data/histdata",
        cache_root="data/cache",
        boundary_convention=BOUNDARY_CONVENTION,
    )
    start = pd.Timestamp(WINDOW_START, tz="UTC")
    end = pd.Timestamp(WINDOW_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    pad_start = start - pd.Timedelta(days=PAD_DAYS)
    h4_w = h4.loc[(h4.index >= start) & (h4.index <= end)]
    d1_w = d1.loc[(d1.index >= pad_start) & (d1.index <= end)]
    h4_bid = bid_view_for_signal(h4_w)
    d1_bid = bid_view_for_signal(d1_w)
    return h4_bid, d1_bid


def _atr_uses_future_bar(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """LOOKAHEAD-PLANTED Wilder ATR using TR[t+1] at update step."""
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n - 1):
        # BUG: use tr[i+1] instead of tr[i]
        atr[i] = (atr[i - 1] * (period - 1) + tr[i + 1]) / period
    # Last bar falls back to causal
    if n - 1 >= period:
        atr[n - 1] = (atr[n - 2] * (period - 1) + tr[n - 1]) / period
    return atr


def _compute_signal_with_atr_bug(df_4h, df_d1, dlr_mod, **kwargs):
    """Run dlr.compute_signal with the buggy ATR substituted in."""
    original_atr = dlr_mod.wilder_atr
    dlr_mod.wilder_atr = _atr_uses_future_bar
    try:
        return dlr_mod.compute_signal(df_4h, df_d1, **kwargs)
    finally:
        dlr_mod.wilder_atr = original_atr


def _compute_signal_with_swing_bug(df_4h, df_d1, dlr_mod, **kwargs):
    """Run dlr.compute_signal with right_edge_offset = -1 (peeks into future)."""
    return dlr_mod.compute_signal(df_4h, df_d1, right_edge_offset=-1, **kwargs)


def _run_post_signal_nan_test(pool, dlr_mod, signal_fn, label: str) -> dict:
    """Pick sample trades; NaN bars after signal_bar; check whether signal flips."""
    failures: list[dict] = []
    n_checked = 0
    for pair in TEST_PAIRS:
        h4, d1 = _load_pair(pair, dlr_mod)
        date_arr = pd.to_datetime(h4["date"]).reset_index(drop=True)
        date_to_idx = pd.Series(np.arange(len(date_arr)), index=date_arr)
        pair_pool = pool[pool["pair"] == pair].copy().reset_index(drop=True)
        pair_pool["_bar_idx"] = pair_pool["signal_bar_time"].map(date_to_idx)
        pair_pool = pair_pool.dropna(subset=["_bar_idx"])
        pair_pool["_bar_idx"] = pair_pool["_bar_idx"].astype(int)
        if len(pair_pool) == 0:
            continue
        sample = pair_pool.sample(min(N_PER_PAIR, len(pair_pool)), random_state=42)
        # Baseline buggy signal
        base = signal_fn(h4, d1, dlr_mod, signal_col="signal").reset_index(drop=True)
        for _, row in sample.iterrows():
            bar_idx = int(row["_bar_idx"])
            n_checked += 1
            h4_pert = h4.copy()
            if bar_idx + 1 < len(h4_pert):
                h4_pert.loc[bar_idx + 1 :, ["open", "high", "low", "close"]] = np.nan
            pert = signal_fn(h4_pert, d1, dlr_mod, signal_col="signal").reset_index(drop=True)
            base_atr = base["atr14"].iloc[bar_idx]
            pert_atr = pert["atr14"].iloc[bar_idx]
            base_sig = bool(base["signal"].iloc[bar_idx])
            pert_sig = bool(pert["signal"].iloc[bar_idx])
            # For ATR bug, the relevant diff is on atr14 (and consequently L1_to_atr_proximity etc.)
            # For swing bug, the relevant diff is on L1_value / d_for_l1_search_max
            differ = (base_sig != pert_sig) or (
                pd.notna(base_atr) and pd.notna(pert_atr) and abs(float(base_atr) - float(pert_atr)) > ATOL
            )
            differ_swing = (
                row.get("L1_value") is not None
                and abs(float(base["L1_value"].iloc[bar_idx] or 0) - float(pert["L1_value"].iloc[bar_idx] or 0)) > ATOL
            )
            if not (differ or differ_swing):
                # The bug did not produce a detectable difference for this trade
                pass
            else:
                failures.append(
                    {
                        "pair": pair,
                        "trade_id": int(row["trade_id"]),
                        "bar_idx": bar_idx,
                        "base_atr14": (None if pd.isna(base_atr) else float(base_atr)),
                        "pert_atr14": (None if pd.isna(pert_atr) else float(pert_atr)),
                        "base_signal": base_sig,
                        "pert_signal": pert_sig,
                    }
                )
    return {
        "n_checked": n_checked,
        "n_detected_differences": len(failures),
        "audit_catches_bug": bool(failures),
        "sample_differences": failures[:10],
    }


def _run_trim_byte_compare_test(pool, dlr_mod, signal_fn, label: str) -> dict:
    """Trim H4 + D1 to <= signal_time; recompute with the buggy signal;
    compare to the buggy baseline (run on the un-trimmed arena). If the
    bug peeks beyond signal_time, the trimmed-arena value at signal_time
    will diverge from the un-trimmed value because the future data the bug
    relied on no longer exists.

    A causally-clean module would produce IDENTICAL output at signal_time
    in both arenas; only a buggy module shows divergence here.
    """
    failures: list[dict] = []
    n_checked = 0
    for pair in TEST_PAIRS:
        h4, d1 = _load_pair(pair, dlr_mod)
        date_arr = pd.to_datetime(h4["date"]).reset_index(drop=True)
        date_to_idx = pd.Series(np.arange(len(date_arr)), index=date_arr)
        pair_pool = pool[pool["pair"] == pair].copy().reset_index(drop=True)
        pair_pool["_bar_idx"] = pair_pool["signal_bar_time"].map(date_to_idx)
        pair_pool = pair_pool.dropna(subset=["_bar_idx"])
        pair_pool["_bar_idx"] = pair_pool["_bar_idx"].astype(int)
        if len(pair_pool) == 0:
            continue
        sample = pair_pool.sample(min(N_PER_PAIR, len(pair_pool)), random_state=42)
        base = signal_fn(h4, d1, dlr_mod, signal_col="signal").reset_index(drop=True)
        for _, row in sample.iterrows():
            bar_idx = int(row["_bar_idx"])
            n_checked += 1
            t = pd.Timestamp(row["signal_bar_time"])
            h4_trim = h4.loc[h4["date"] <= t].copy().reset_index(drop=True)
            d1_trim = d1.loc[d1["date"] <= t].copy().reset_index(drop=True)
            if len(h4_trim) == 0:
                continue
            trim_idx = len(h4_trim) - 1
            trim_out = signal_fn(h4_trim, d1_trim, dlr_mod, signal_col="signal").reset_index(drop=True)
            base_sig = bool(base["signal"].iloc[bar_idx])
            trim_sig = bool(trim_out["signal"].iloc[trim_idx])
            base_atr = base["atr14"].iloc[bar_idx]
            trim_atr = trim_out["atr14"].iloc[trim_idx]
            base_L1 = base["L1_value"].iloc[bar_idx]
            trim_L1 = trim_out["L1_value"].iloc[trim_idx]
            diff_sig = (base_sig != trim_sig)
            diff_atr = (
                pd.notna(base_atr) and pd.notna(trim_atr)
                and abs(float(base_atr) - float(trim_atr)) > ATOL
            )
            diff_L1 = (
                pd.notna(base_L1) and pd.notna(trim_L1)
                and abs(float(base_L1) - float(trim_L1)) > ATOL
            ) or (pd.isna(base_L1) != pd.isna(trim_L1))
            differ = diff_sig or diff_atr or diff_L1
            if differ:
                failures.append(
                    {
                        "pair": pair,
                        "trade_id": int(row["trade_id"]),
                        "bar_idx": bar_idx,
                        "base_signal": base_sig,
                        "trim_signal": trim_sig,
                        "base_atr14": (None if pd.isna(base_atr) else float(base_atr)),
                        "trim_atr14": (None if pd.isna(trim_atr) else float(trim_atr)),
                        "base_L1": (None if pd.isna(base_L1) else float(base_L1)),
                        "trim_L1": (None if pd.isna(trim_L1) else float(trim_L1)),
                    }
                )
    return {
        "n_checked": n_checked,
        "n_detected_differences": len(failures),
        "audit_catches_bug": bool(failures),
        "sample_differences": failures[:10],
    }


def run() -> dict:
    import signals.lchar_dlr_long as dlr  # type: ignore
    pool = pd.read_parquet(POOL_PATH)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    bug_results: dict[str, dict] = {}

    # Bug A: swing search into the future
    swing_post = _run_post_signal_nan_test(pool, dlr, _compute_signal_with_swing_bug, "swing_bug")
    swing_trim = _run_trim_byte_compare_test(pool, dlr, _compute_signal_with_swing_bug, "swing_bug")
    bug_results["swing_search_into_future"] = {
        "post_signal_h4_nan_test": swing_post,
        "trim_byte_compare_test": swing_trim,
        "any_test_catches": swing_post["audit_catches_bug"] or swing_trim["audit_catches_bug"],
    }

    # Re-import dlr to make sure no contamination between tests.
    importlib.reload(dlr)

    # Bug B: ATR uses future bar
    atr_post = _run_post_signal_nan_test(pool, dlr, _compute_signal_with_atr_bug, "atr_bug")
    atr_trim = _run_trim_byte_compare_test(pool, dlr, _compute_signal_with_atr_bug, "atr_bug")
    bug_results["atr_uses_future_bar"] = {
        "post_signal_h4_nan_test": atr_post,
        "trim_byte_compare_test": atr_trim,
        "any_test_catches": atr_post["audit_catches_bug"] or atr_trim["audit_catches_bug"],
    }

    payload = {
        "category": "section_2_16_deliberate_lookahead",
        "anchor_commit": "244fb76",
        "method": (
            "In-memory monkey-patching of signals.lchar_dlr_long. The on-disk module "
            "is never modified. For each planted bug, runs the §2.1 post-signal H4 "
            "NaN-perturbation test on a small sample of pool trades; records whether "
            "the audit detects a difference between buggy + unperturbed and "
            "buggy + future-bar-NaN'd. A perfectly causal module returns no "
            "difference. A buggy module that reads future bars returns differences "
            "in atr14 / signal / L1_value etc."
        ),
        "results": bug_results,
        "overall_audit_sensitivity": all(
            bug_results[b]["any_test_catches"] for b in bug_results
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    for b, r in bug_results.items():
        print(
            f"[2.16] {b}: post_signal_test={r['post_signal_h4_nan_test']['audit_catches_bug']} "
            f"({r['post_signal_h4_nan_test']['n_detected_differences']}/{r['post_signal_h4_nan_test']['n_checked']})  "
            f"trim_test={r['trim_byte_compare_test']['audit_catches_bug']} "
            f"({r['trim_byte_compare_test']['n_detected_differences']}/{r['trim_byte_compare_test']['n_checked']})  "
            f"=> any_caught={r['any_test_catches']}",
            flush=True,
        )
    return payload


if __name__ == "__main__":
    run()
