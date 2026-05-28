"""§2.1 Exhaustive lookahead audit per dispatch ``docs/dispatches/arc_10_audit_intent.md``.

Three independent empirical tests on the DLR signal module:

1.  **Byte-compare under data truncation.**  For a sample of trades, trim
    H4 + D1 to bars whose timestamp is ``<= signal_time`` and recompute the
    DLR signal stage.  Compare the recomputed (L1_value, L0_value,
    L1_age_d1_bars, L0_age_d1_bars, L1_to_atr_proximity, reject_buffer_atr,
    upper_fraction, atr14_at_signal) tuple at the signal bar against the
    values stored on the pool row.  Tolerance 1e-9 absolute.

2.  **Post-signal NaN-perturbation.**  For a sample of trades, NaN every
    H4 bar STRICTLY AFTER signal_time, re-run the DLR module, and verify
    the boolean signal at the original index is still True (and the
    derived L1/L0/proximity/upper_fraction columns at that index are
    unchanged).

3.  **D1 row at ``d_t`` NaN-perturbation.**  For ALL pool trades, NaN
    the D1 row that contains the signal's 4H bar (``d_t``), re-run the
    DLR module, and verify the signal at the original index is still True.
    The signal module docstring asserts ``D1[d_t]`` is structurally
    unread (the swing-search window stops at ``d_t - 4`` and the latest
    bar peeked for swing-confirm is ``d_t - 1``); this test empirically
    confirms the assertion across the full pool.

Output: ``results/l_arc_10_v3.0.2/exhaustive_audit/section_2_1_dlr_lookahead.json``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import signals.lchar_dlr_long as dlr  # noqa: E402
from core.data.aggregator import aggregate  # noqa: E402
from scripts.l_arc_10_v3._common import bid_view_for_signal  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "exhaustive_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "section_2_1_dlr_lookahead.json"
POOL_PATH = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "step_1" / "pool.parquet"

BOUNDARY_CONVENTION = "5ers_eet"
WINDOW_START = "2010-01-01"
WINDOW_END = "2026-04-30"
PAD_DAYS = 45

# Per-trade DLR fields that must reproduce byte-identically under truncation.
DLR_FIELDS = (
    "L1_value",
    "L0_value",
    "L1_age_d1_bars",
    "L0_age_d1_bars",
    "L1_to_atr_proximity",
    "reject_buffer_atr",
    "upper_fraction",
    "atr14_at_signal",
    "d_t_idx",
    "d_for_l1_search_max",
)
DLR_SIG_COL_FROM_STORED = {
    "atr14_at_signal": "atr14",
    "d_t_idx": "d_t_idx",
    "d_for_l1_search_max": "d_for_l1_search_max",
}
ATOL = 1e-9
RTOL = 1e-9

# Sample sizes (per pair) for trim+recompute and post-signal NaN tests.
# The d1[d_t] NaN-perturbation runs on every pool trade (exhaustive).
TRIM_SAMPLE_PER_PAIR = 10
POST_SIGNAL_NAN_SAMPLE_PER_PAIR = 10


def _load_pair_data(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate H4 and D1 with the canonical 5ers_eet convention; window-slice."""
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
    pad_start = (start - pd.Timedelta(days=PAD_DAYS))
    h4_w = h4.loc[(h4.index >= start) & (h4.index <= end)]
    d1_w = d1.loc[(d1.index >= pad_start) & (d1.index <= end)]
    return h4_w, d1_w


def _bid_views(h4: pd.DataFrame, d1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return bid_view_for_signal(h4), bid_view_for_signal(d1)


def _close_match(a: float, b: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(abs(float(a) - float(b)) <= ATOL + RTOL * max(abs(float(a)), abs(float(b))))


@dataclass
class CategoryResult:
    name: str
    n_total: int = 0
    n_checked: int = 0
    n_passed: int = 0
    n_failed: int = 0
    failures: list[dict] = field(default_factory=list)
    notes: str = ""


def run() -> dict:
    print(f"[2.1] reading pool: {POOL_PATH}", flush=True)
    pool = pd.read_parquet(POOL_PATH)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    pairs = sorted(pool["pair"].unique())
    rng = np.random.default_rng(42)

    trim_result = CategoryResult(
        name="byte_compare_under_truncation",
        notes=(
            f"Per-trade DLR field byte-compare. For {TRIM_SAMPLE_PER_PAIR} sampled trades per pair "
            f"(seed=42), trim H4 + D1 to bars whose timestamp <= signal_time and recompute the DLR "
            f"signal stage; compare ({', '.join(DLR_FIELDS)}) at signal bar against pool values. "
            f"Tolerance 1e-9 (absolute + relative)."
        ),
    )
    postsig_result = CategoryResult(
        name="post_signal_nan_perturbation",
        notes=(
            f"Per-trade post-signal H4 NaN-perturbation. For {POST_SIGNAL_NAN_SAMPLE_PER_PAIR} sampled "
            f"trades per pair (seed=42), NaN every H4 OHLC row strictly after signal_bar_time and "
            f"re-run the DLR signal; verify the boolean signal at the original index remains True "
            f"AND all DLR field values at that index reproduce byte-identically."
        ),
    )
    d1pert_result = CategoryResult(
        name="d1_row_at_dt_nan_perturbation",
        notes=(
            "Per-trade D1[d_t] NaN-perturbation. For ALL pool trades, NaN the D1 row corresponding "
            "to the D1 trading day that CONTAINS the 4H signal bar (d_t in the DLR module). Re-run "
            "the DLR signal; verify the boolean signal at the original 4H index remains True. The "
            "DLR signal module docstring asserts D1[d_t] is structurally unread because the swing "
            "search bounds at d_t-4 and the latest D1 bar peeked for swing-confirm is d_t-1."
        ),
    )

    for pair in pairs:
        print(f"[2.1] {pair} ...", flush=True)
        h4, d1 = _load_pair_data(pair)
        h4_bid, d1_bid = _bid_views(h4, d1)
        date_arr = pd.to_datetime(h4_bid["date"]).reset_index(drop=True)
        # Build a date → bar_index lookup for this pair.
        date_to_idx = pd.Series(np.arange(len(date_arr)), index=date_arr)
        # All pool rows for this pair, sorted ascending by signal_bar_time.
        pair_pool = pool[pool["pair"] == pair].copy().sort_values("signal_bar_time").reset_index(drop=True)
        if len(pair_pool) == 0:
            continue
        # Resolve each pool row's H4 index in this pair's series.
        pair_pool["_bar_idx"] = pair_pool["signal_bar_time"].map(date_to_idx)
        # Sanity: every pool row should have a valid bar index.
        if pair_pool["_bar_idx"].isna().any():
            missing = int(pair_pool["_bar_idx"].isna().sum())
            print(f"  [{pair}] WARNING — {missing} pool trade(s) without matching 4H bar; "
                  f"will be excluded from §2.1 tests for this pair.", flush=True)
            pair_pool = pair_pool.dropna(subset=["_bar_idx"]).copy()
            pair_pool["_bar_idx"] = pair_pool["_bar_idx"].astype(int)
        else:
            pair_pool["_bar_idx"] = pair_pool["_bar_idx"].astype(int)

        # ── 1. Trim+recompute byte-compare ──────────────────────────────────
        n_pool = len(pair_pool)
        trim_n = min(TRIM_SAMPLE_PER_PAIR, n_pool)
        trim_idx = rng.choice(n_pool, size=trim_n, replace=False) if n_pool > 0 else np.array([], dtype=int)
        for j in trim_idx:
            row = pair_pool.iloc[int(j)]
            bar_idx = int(row["_bar_idx"])
            t = pd.Timestamp(row["signal_bar_time"])
            # Trim H4 and D1 to bars with timestamp <= signal_time. D1 retains
            # the natural pad on the left.
            h4_trim_bid = h4_bid.loc[h4_bid["date"] <= t].copy().reset_index(drop=True)
            d1_trim_bid = d1_bid.loc[d1_bid["date"] <= t].copy().reset_index(drop=True)
            # The trimmed-arena bar index for signal_time is the LAST row by
            # construction (since we trimmed to <= signal_time).
            trim_sig_idx = len(h4_trim_bid) - 1
            if trim_sig_idx < 0:
                continue
            sig_re = dlr.compute_signal(h4_trim_bid, d1_trim_bid, signal_col="signal").reset_index(drop=True)
            trim_result.n_checked += 1
            ok = True
            row_failures: list[dict] = []
            # Was the signal still fired at signal_time in the trimmed arena?
            if not bool(sig_re["signal"].iloc[trim_sig_idx]):
                ok = False
                row_failures.append({"field": "signal", "stored": True, "recomputed": False})
            # Compare each DLR field.
            for f in DLR_FIELDS:
                stored_col = DLR_SIG_COL_FROM_STORED.get(f, f)
                if f in row.index:
                    stored_val = row[f]
                elif stored_col in row.index:
                    stored_val = row[stored_col]
                else:
                    continue
                recomp_val = sig_re[stored_col].iloc[trim_sig_idx]
                if not _close_match(stored_val, recomp_val):
                    ok = False
                    row_failures.append(
                        {
                            "field": f,
                            "stored": (None if pd.isna(stored_val) else float(stored_val)),
                            "recomputed": (None if pd.isna(recomp_val) else float(recomp_val)),
                        }
                    )
            if ok:
                trim_result.n_passed += 1
            else:
                trim_result.n_failed += 1
                trim_result.failures.append(
                    {
                        "pair": pair,
                        "trade_id": int(row["trade_id"]),
                        "signal_bar_time": t.isoformat(),
                        "field_diffs": row_failures,
                    }
                )

        # ── 2. Post-signal NaN-perturbation ─────────────────────────────────
        post_n = min(POST_SIGNAL_NAN_SAMPLE_PER_PAIR, n_pool)
        post_idx = rng.choice(n_pool, size=post_n, replace=False) if n_pool > 0 else np.array([], dtype=int)
        for j in post_idx:
            row = pair_pool.iloc[int(j)]
            bar_idx = int(row["_bar_idx"])
            h4_pert = h4_bid.copy()
            if bar_idx + 1 < len(h4_pert):
                h4_pert.loc[bar_idx + 1 :, ["open", "high", "low", "close"]] = np.nan
            sig_pert = dlr.compute_signal(h4_pert, d1_bid, signal_col="signal").reset_index(drop=True)
            postsig_result.n_checked += 1
            ok = bool(sig_pert["signal"].iloc[bar_idx])
            field_failures: list[dict] = []
            for f in DLR_FIELDS:
                stored_col = DLR_SIG_COL_FROM_STORED.get(f, f)
                if f in row.index:
                    stored_val = row[f]
                elif stored_col in row.index:
                    stored_val = row[stored_col]
                else:
                    continue
                recomp_val = sig_pert[stored_col].iloc[bar_idx]
                if not _close_match(stored_val, recomp_val):
                    ok = False
                    field_failures.append(
                        {
                            "field": f,
                            "stored": (None if pd.isna(stored_val) else float(stored_val)),
                            "recomputed": (None if pd.isna(recomp_val) else float(recomp_val)),
                        }
                    )
            if ok:
                postsig_result.n_passed += 1
            else:
                postsig_result.n_failed += 1
                postsig_result.failures.append(
                    {
                        "pair": pair,
                        "trade_id": int(row["trade_id"]),
                        "signal_bar_time": pd.Timestamp(row["signal_bar_time"]).isoformat(),
                        "field_diffs": field_failures,
                    }
                )

        # ── 3. D1[d_t] NaN-perturbation across ALL pool trades for this pair ─
        # Group trades by their D1 calendar-day (using the d_t_idx ALREADY
        # stored on the pool row); NaN that D1 row and re-run once per unique
        # d_t_idx (massive speedup vs once-per-trade since many trades share
        # the same d_t).
        # In practice many trades will share a d_t_idx; cache by d_t.
        d1_pert_results_for_pair: dict[int, np.ndarray] = {}
        for _, row in pair_pool.iterrows():
            d_t = int(row["d_t_idx"])
            if d_t < 0 or d_t >= len(d1_bid):
                continue
            if d_t in d1_pert_results_for_pair:
                pert_mask = d1_pert_results_for_pair[d_t]
            else:
                d1_pert = d1_bid.copy()
                d1_pert.loc[d_t, ["open", "high", "low", "close"]] = np.nan
                pert_sig = dlr.compute_signal(h4_bid, d1_pert, signal_col="signal").reset_index(
                    drop=True
                )
                pert_mask = pert_sig["signal"].to_numpy(dtype=bool)
                d1_pert_results_for_pair[d_t] = pert_mask
            bar_idx = int(row["_bar_idx"])
            d1pert_result.n_checked += 1
            if pert_mask[bar_idx]:
                d1pert_result.n_passed += 1
            else:
                d1pert_result.n_failed += 1
                d1pert_result.failures.append(
                    {
                        "pair": pair,
                        "trade_id": int(row["trade_id"]),
                        "signal_bar_time": pd.Timestamp(row["signal_bar_time"]).isoformat(),
                        "d_t_idx": int(d_t),
                        "issue": "signal flipped to False after NaN'ing D1[d_t]",
                    }
                )

        trim_result.n_total += n_pool
        postsig_result.n_total += n_pool
        d1pert_result.n_total += n_pool

    payload = {
        "category": "section_2_1_dlr_signal_lookahead",
        "anchor_commit": "244fb76",
        "boundary_convention": BOUNDARY_CONVENTION,
        "pool_path": str(POOL_PATH.relative_to(REPO_ROOT)),
        "pool_size": int(len(pool)),
        "rtol": RTOL,
        "atol": ATOL,
        "checks": [
            {
                "name": trim_result.name,
                "n_total_pool_trades": trim_result.n_total,
                "n_checked": trim_result.n_checked,
                "n_passed": trim_result.n_passed,
                "n_failed": trim_result.n_failed,
                "verdict": "PASS" if trim_result.n_failed == 0 else "FAIL",
                "notes": trim_result.notes,
                "failures": trim_result.failures[:20],
            },
            {
                "name": postsig_result.name,
                "n_total_pool_trades": postsig_result.n_total,
                "n_checked": postsig_result.n_checked,
                "n_passed": postsig_result.n_passed,
                "n_failed": postsig_result.n_failed,
                "verdict": "PASS" if postsig_result.n_failed == 0 else "FAIL",
                "notes": postsig_result.notes,
                "failures": postsig_result.failures[:20],
            },
            {
                "name": d1pert_result.name,
                "n_total_pool_trades": d1pert_result.n_total,
                "n_checked": d1pert_result.n_checked,
                "n_passed": d1pert_result.n_passed,
                "n_failed": d1pert_result.n_failed,
                "verdict": "PASS" if d1pert_result.n_failed == 0 else "FAIL",
                "notes": d1pert_result.notes,
                "failures": d1pert_result.failures[:20],
            },
        ],
        "overall_verdict": (
            "PASS"
            if (trim_result.n_failed == 0 and postsig_result.n_failed == 0 and d1pert_result.n_failed == 0)
            else "FAIL"
        ),
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"[2.1] wrote {OUT_PATH}", flush=True)
    print(
        f"[2.1] trim_byte_compare={trim_result.n_passed}/{trim_result.n_checked}  "
        f"post_signal_nan={postsig_result.n_passed}/{postsig_result.n_checked}  "
        f"d1_row_at_dt={d1pert_result.n_passed}/{d1pert_result.n_checked}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()
