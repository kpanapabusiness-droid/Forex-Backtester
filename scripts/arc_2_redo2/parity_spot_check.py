"""Arc 2 redo2 — Step 1 parity spot check (Phase B).

Compares the v2.1.1-schema fork output against the prior v2.0 arc_2_redo Step 1
output for the same signal config. Picks 3 trades from the prior file (one
short SL exit, one h-cap exit, one mid-trade exit), then for each trade
verifies:

  1. Held-portion semantic match: for bar_offset in [0, exit_offset]:
       - Fork's `is_held=1` rows must overlap the prior `still_open=1` rows
         AND the exit-bar row (where prior had still_open=0 + close_r=final_r).
       - close_r and OHLC must be identical to the prior file on all those bars.
     Per the fix-2 spec, is_held=1 covers entry to actual exit (inclusive),
     which matches prior still_open=1 plus the single exit bar.

  2. Forward-obs semantic match: for bar_offset in (exit_offset, 240]:
       - Fork must have is_held=0 with close_r computed from real-market
         (close_price - entry_price) / sl_distance_price, varying bar-to-bar.
       - Prior had close_r FROZEN at final_r for these same bars.
       - close_r values in fork must NOT all equal the frozen value
         (this is the regression the fix is meant to fix).
       - Raw OHLC must match the prior file (real-market data was preserved
         in the prior file too; only R-fields were frozen).

Reads:
  results/l_arc_2_redo/step1/trades_all.csv   (prior, reference)
  results/l_arc_2_redo/step1/trades_paths.csv (prior, frozen R-fields)
  results/l_arc_2_redo2/step1/trades_all.csv   (fork)
  results/l_arc_2_redo2/step1/trades_paths.csv (fork)

Writes:
  results/l_arc_2_redo2/step1/parity_spot_check.txt

Match key: (pair, signal_time, entry_time) — trade_id may differ between runs
if signal-order changes, but signal/entry time per pair is the natural identity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


PRIOR_DIR = _REPO_ROOT / "results" / "l_arc_2_redo" / "step1"
FORK_DIR = _REPO_ROOT / "results" / "l_arc_2_redo2" / "step1"


def _load_pair(prior_trades: pd.DataFrame, fork_trades: pd.DataFrame) -> pd.DataFrame:
    """Merge prior and fork trades on (pair, signal_time) and keep matched."""
    key = ["pair", "signal_time"]
    merged = prior_trades.merge(
        fork_trades, on=key, how="inner", suffixes=("_prior", "_fork")
    )
    return merged


def _pick_three_trades(merged: pd.DataFrame) -> List[pd.Series]:
    """Pick 3 trades with diverse exit profiles:
       (a) early SL hit (smallest bars_held among stop_loss),
       (b) mid SL hit (mid-range bars_held among stop_loss),
       (c) h-cap (time_exit, bars_held=120).
    """
    picks: List[pd.Series] = []
    sl_trades = merged[merged["exit_reason_prior"] == "stop_loss"].copy()
    te_trades = merged[merged["exit_reason_prior"] == "time_exit"].copy()

    if len(sl_trades) > 0:
        # Earliest SL.
        sl_sorted = sl_trades.sort_values(
            ["bars_held_prior", "pair", "signal_time"]
        ).reset_index(drop=True)
        picks.append(sl_sorted.iloc[0])
        # Mid SL (median bars_held).
        mid_idx = len(sl_sorted) // 2
        picks.append(sl_sorted.iloc[mid_idx])
    if len(te_trades) > 0:
        # First h-cap exit (deterministic).
        te_sorted = te_trades.sort_values(["pair", "signal_time"]).reset_index(drop=True)
        picks.append(te_sorted.iloc[0])

    return picks[:3]


def _trade_paths(
    paths: pd.DataFrame, trade_id: int
) -> pd.DataFrame:
    return paths[paths["trade_id"] == trade_id].sort_values("bar_offset").reset_index(drop=True)


def _compare_one(
    prior_trade: pd.Series,
    fork_trade: pd.Series,
    prior_paths_for_trade: pd.DataFrame,
    fork_paths_for_trade: pd.DataFrame,
) -> Tuple[List[str], bool, bool]:
    lines: List[str] = []
    held_ok = True
    fwd_ok = True

    pair = prior_trade["pair"]
    sig_t = prior_trade["signal_time"]
    exit_off_prior = int(prior_trade["bars_held"] if pd.notna(prior_trade["bars_held"]) else 0)
    # bars_held = sl_hit_idx - entry_idx + 1 (SL) or horizon (time_exit, 120).
    # In paths, exit_offset = exit_idx - entry_idx, which is (bars_held - 1) for SL
    # and 120 for time_exit per build logic.
    # We won't rely on this; we'll find the exit row from the path itself.
    final_r_prior = float(prior_trade["final_r"])
    exit_reason = str(prior_trade["exit_reason"])

    # Identify the exit_offset from prior path: the last row with close_r == final_r
    # adjacent to a still_open transition. More robust: find row where close_r equals
    # final_r AND still_open=0 (the freeze starts there).
    prior_path = prior_paths_for_trade
    fork_path = fork_paths_for_trade

    # In the prior file: still_open=1 for off < exit_offset; exit_offset row has still_open=0
    # AND close_r=final_r; then it remains frozen.
    if "still_open" in prior_path.columns:
        still_open_zero = prior_path[prior_path["still_open"] == 0]
        if len(still_open_zero) == 0:
            return [f"  trade {pair} {sig_t}: prior has no still_open=0 rows"], False, False
        exit_offset = int(still_open_zero["bar_offset"].iloc[0])
    else:
        return [f"  trade {pair} {sig_t}: prior file missing still_open column"], False, False

    lines.append(
        f"Trade: pair={pair} signal_time={sig_t} "
        f"exit_reason={exit_reason} bars_held={int(prior_trade['bars_held'])} "
        f"exit_offset_in_path={exit_offset} final_r={final_r_prior:.6f}"
    )

    # ---- Held-portion check: bars [0, exit_offset] ----
    # In fork: is_held=1 for off in [0, exit_offset] (inclusive of exit bar per the fix).
    # In prior: still_open=1 for [0, exit_offset-1]; still_open=0 at exit_offset with close_r=final_r.
    # We compare close_r and OHLC for off in [0, exit_offset] across files.
    prior_held = prior_path[prior_path["bar_offset"] <= exit_offset]
    fork_held = fork_path[fork_path["bar_offset"] <= exit_offset]
    if len(prior_held) != len(fork_held):
        held_ok = False
        lines.append(
            f"  HELD ROW-COUNT MISMATCH prior={len(prior_held)} fork={len(fork_held)}"
        )
    else:
        # is_held column check
        if "is_held" not in fork_held.columns:
            held_ok = False
            lines.append("  is_held column missing from fork file")
        else:
            held_ones = int((fork_held["is_held"] == 1).sum())
            held_zeros = int((fork_held["is_held"] == 0).sum())
            if held_zeros != 0:
                held_ok = False
                lines.append(
                    f"  is_held=0 should not appear in held-portion; saw {held_zeros}"
                )
            else:
                lines.append(
                    f"  is_held=1 count in [0..exit_offset]: {held_ones}/{len(fork_held)} OK"
                )

        # close_r equality.
        for col in ("close_r", "open", "high", "low", "close"):
            diff = (prior_held[col].astype(float).values - fork_held[col].astype(float).values)
            max_abs = abs(diff).max() if len(diff) > 0 else 0.0
            if max_abs > 1e-9:
                held_ok = False
                lines.append(
                    f"  HELD {col} mismatch: max_abs_diff={max_abs:.2e}"
                )
            else:
                lines.append(f"  HELD {col} byte-equiv (max_abs_diff < 1e-9) OK")

        # mfe/mae equality (running mins/maxes should match through held bars).
        for col in ("mfe_so_far_r", "mae_so_far_r"):
            diff = (prior_held[col].astype(float).values - fork_held[col].astype(float).values)
            max_abs = abs(diff).max() if len(diff) > 0 else 0.0
            if max_abs > 1e-9:
                held_ok = False
                lines.append(f"  HELD {col} mismatch: max_abs_diff={max_abs:.2e}")
            else:
                lines.append(f"  HELD {col} byte-equiv OK")

    # ---- Forward-obs check: bars (exit_offset, 240] ----
    prior_fwd = prior_path[prior_path["bar_offset"] > exit_offset]
    fork_fwd = fork_path[fork_path["bar_offset"] > exit_offset]

    if len(prior_fwd) != len(fork_fwd):
        fwd_ok = False
        lines.append(
            f"  FWD-OBS ROW-COUNT MISMATCH prior={len(prior_fwd)} fork={len(fork_fwd)}"
        )

    if len(fork_fwd) == 0:
        fwd_ok = False
        lines.append("  FWD-OBS no forward bars in fork (trade at data tail?)")
    else:
        # is_held=0 throughout.
        is_held_set = set(fork_fwd["is_held"].astype(int).unique())
        if is_held_set != {0}:
            fwd_ok = False
            lines.append(f"  FWD-OBS is_held should be all 0; saw values {is_held_set}")
        else:
            lines.append(f"  FWD-OBS is_held=0 for all {len(fork_fwd)} forward bars OK")

        # Real-market R-fields: close_r must vary, NOT be frozen at final_r.
        unique_close_r = fork_fwd["close_r"].astype(float).round(8).nunique()
        first_close_r = float(fork_fwd["close_r"].iloc[0])
        last_close_r = float(fork_fwd["close_r"].iloc[-1])
        diff_first_last = abs(last_close_r - first_close_r)
        lines.append(
            f"  FWD-OBS close_r: first={first_close_r:.6f} last={last_close_r:.6f} "
            f"unique_values={unique_close_r}"
        )
        if unique_close_r <= 1:
            fwd_ok = False
            lines.append(
                "  FWD-OBS close_r FROZEN (only 1 unique value across forward bars) — REGRESSION"
            )
        else:
            lines.append(
                f"  FWD-OBS close_r varies across {unique_close_r} unique values "
                f"(|first-last|={diff_first_last:.6f}) — not frozen OK"
            )

        # Confirm fork close_r matches the v1.3-style formula on real OHLC.
        entry_price_prior = float(prior_trade["entry_price"])
        sl_dist_prior = float(prior_trade["entry_price"]) - float(prior_trade["sl_price"])
        # In fork, recompute close_r from raw close in fork file directly.
        recomputed = (
            (fork_fwd["close"].astype(float) - entry_price_prior) / sl_dist_prior
        )
        max_dev = abs(recomputed.values - fork_fwd["close_r"].astype(float).values).max()
        # Tolerance set to 1e-5: well above expected CSV `%.10g` round-trip noise
        # (~3e-7 worst-case from subtract-and-divide on three independently-rounded
        # values: close, entry_price, sl_distance) and well below any meaningful
        # logic divergence.
        lines.append(
            f"  FWD-OBS close_r formula check (close-entry)/sl_dist: max|dev|={max_dev:.2e}"
        )
        if max_dev > 1e-5:
            fwd_ok = False
            lines.append("  FWD-OBS close_r formula DIVERGES from v1.3 reference — REGRESSION")
        else:
            lines.append(
                "  FWD-OBS close_r formula matches (close-entry)/sl_dist within CSV float tol OK"
            )

        # Verify raw OHLC in fork forward bars match prior file (sanity).
        if len(prior_fwd) == len(fork_fwd):
            for col in ("open", "high", "low", "close"):
                diff = (
                    prior_fwd[col].astype(float).values
                    - fork_fwd[col].astype(float).values
                )
                max_abs = abs(diff).max() if len(diff) > 0 else 0.0
                if max_abs > 1e-9:
                    fwd_ok = False
                    lines.append(f"  FWD-OBS raw {col} mismatch: max_abs_diff={max_abs:.2e}")
                else:
                    lines.append(f"  FWD-OBS raw {col} matches prior OK")

            # Also confirm prior was frozen (sanity on the regression test logic).
            prior_unique_cr = prior_fwd["close_r"].astype(float).round(8).nunique()
            lines.append(
                f"  FWD-OBS sanity: prior close_r unique values across forward bars = "
                f"{prior_unique_cr} (expected 1 — frozen at final_r={final_r_prior:.6f})"
            )

    lines.append("")
    return lines, held_ok, fwd_ok


def main() -> int:
    print(f"[parity] loading prior trades_all.csv from {PRIOR_DIR}", file=sys.stderr)
    prior_trades = pd.read_csv(PRIOR_DIR / "trades_all.csv")
    print(f"[parity] loading fork trades_all.csv from {FORK_DIR}", file=sys.stderr)
    fork_trades = pd.read_csv(FORK_DIR / "trades_all.csv")

    print(f"[parity] prior: {len(prior_trades)} trades; fork: {len(fork_trades)} trades", file=sys.stderr)

    merged = _load_pair(prior_trades, fork_trades)
    print(f"[parity] inner-join on (pair, signal_time): {len(merged)} matched trades", file=sys.stderr)

    if len(merged) < 3:
        print("[parity] fewer than 3 matched trades; cannot continue", file=sys.stderr)
        return 1

    picks = _pick_three_trades(merged)
    if len(picks) < 3:
        print(f"[parity] only {len(picks)} eligible picks (need 3)", file=sys.stderr)

    # Load paths once.
    print("[parity] loading prior trades_paths.csv (this may be a few GB)", file=sys.stderr)
    prior_paths = pd.read_csv(PRIOR_DIR / "trades_paths.csv")
    print("[parity] loading fork trades_paths.csv", file=sys.stderr)
    fork_paths = pd.read_csv(FORK_DIR / "trades_paths.csv")

    report_lines: List[str] = []
    report_lines.append("# Arc 2 redo2 — Step 1 Phase B parity spot check")
    report_lines.append("")
    report_lines.append(
        "Compares v2.1.1-schema fork (`results/l_arc_2_redo2/step1/`) against prior "
        "v2.0 arc_2_redo run (`results/l_arc_2_redo/step1/`)."
    )
    report_lines.append("")
    report_lines.append("Per-trade contract:")
    report_lines.append(
        "  HELD: for bar_offset in [0, exit_offset], close_r/OHLC/mfe/mae MUST match prior."
    )
    report_lines.append(
        "  FWD-OBS: for bar_offset in (exit_offset, 240], is_held=0; close_r MUST vary "
        "(not frozen) and equal (close-entry)/sl_dist; OHLC MUST match prior."
    )
    report_lines.append("")
    report_lines.append(f"Prior total trades: {len(prior_trades)}")
    report_lines.append(f"Fork  total trades: {len(fork_trades)}")
    report_lines.append(f"Matched on (pair, signal_time): {len(merged)}")
    report_lines.append("")

    overall_held_ok = True
    overall_fwd_ok = True

    for i, row in enumerate(picks):
        report_lines.append(f"## Pick {i + 1}")
        report_lines.append("")
        tid_prior = int(row["trade_id_prior"])
        tid_fork = int(row["trade_id_fork"])
        # Wrap series of relevant fields into a single Series for compare.
        prior_trade = pd.Series({
            "pair": row["pair"],
            "signal_time": row["signal_time"],
            "entry_price": row["entry_price_prior"],
            "sl_price": row["sl_price_prior"],
            "bars_held": row["bars_held_prior"],
            "final_r": row["final_r_prior"],
            "exit_reason": row["exit_reason_prior"],
        })
        fork_trade = pd.Series({
            "pair": row["pair"],
            "signal_time": row["signal_time"],
            "entry_price": row["entry_price_fork"],
            "sl_price": row["sl_price_fork"],
            "bars_held": row["bars_held_fork"],
            "final_r": row["final_r_fork"],
            "exit_reason": row["exit_reason_fork"],
        })
        prior_path_t = _trade_paths(prior_paths, tid_prior)
        fork_path_t = _trade_paths(fork_paths, tid_fork)

        lines, held_ok, fwd_ok = _compare_one(
            prior_trade, fork_trade, prior_path_t, fork_path_t
        )
        report_lines.extend("  " + ln for ln in lines)
        overall_held_ok = overall_held_ok and held_ok
        overall_fwd_ok = overall_fwd_ok and fwd_ok
        report_lines.append("")

    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(
        f"Held-portion parity: {'PASS' if overall_held_ok else 'FAIL'}"
    )
    report_lines.append(
        f"Forward-obs not frozen: {'PASS' if overall_fwd_ok else 'FAIL'}"
    )
    report_lines.append("")

    out_path = FORK_DIR / "parity_spot_check.txt"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[parity] report written: {out_path}", file=sys.stderr)
    print(f"[parity] held_ok={overall_held_ok} fwd_ok={overall_fwd_ok}", file=sys.stderr)

    return 0 if (overall_held_ok and overall_fwd_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
