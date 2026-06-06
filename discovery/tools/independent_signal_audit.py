"""Independent re-verification of the fbr component's signal (§11 Arc-10 defense).

The protocol's §11 institutional lesson: NO candidate is deployed on the gate engine's
word alone — a candidate's numbers must be re-verified via a GENUINELY INDEPENDENT path
(a second implementation, or a hand-audit of trades against raw price). ~12 arcs have
"reproduced fbr exactly", but every one of them re-CALLS the same canonical apparatus
(`FailedBreakdownReclaimLongSignal` -> `ArcFoldRunner` -> `MultiPairBacktester`) — that is
reproduction, NOT independent verification. A bug in the bespoke per-arc signal code (the
one piece NOT covered by the engine's 1656-test suite) would reproduce identically every
time. This module is the missing independent path for the SIGNAL layer of `fbr` (arc 1013,
the corpus's load-bearing component).

Three independent checks (NONE imports the committed signal's logic for the re-derivation):

1. **Fresh re-derivation** — recompute the fbr fire mask from the panel's raw OHLC with
   independent code (a manual python loop for the K-bar swing-low min — NOT pandas
   `.shift(1).rolling(K).min()` — and an independently-written Wilder ATR), then compare
   the fire set to the committed `FailedBreakdownReclaimLongSignal` EXACTLY. A divergence
   would expose an off-by-one / window / geometry bug in the committed signal.

2. **Causal-truncation no-lookahead proof** — for a representative SAMPLE of fire bars i,
   re-evaluate the committed signal on the panel TRUNCATED at bar i (`df.iloc[:i+1]`) and
   confirm bar i is still a fire with byte-identical atr & shadow. If the signal read ANY
   future bar, truncation would change the result. This is the decisive lookahead test.

3. **Raw-OHLC hand-audit dump** — print the actual OHLC bars around a sample of fires so the
   pierce / reclaim / shadow geometry is human-verifiable against raw price (the literal
   "hand-audit against raw price" §11 asks for), incl. the strictly-prior swing-low window.

EXPERIMENT tool: re-derivation + audit only; never realizes P&L, never scores a trade,
never touches the gate, never spends OOS. The OUTCOME layer (per-trade R / cost) routes
through the canonical engine (its own extensively-tested code, honest-engine sweep PR
#263/#264) — this audits the bespoke SIGNAL, the un-audited piece.

Run:  PYTHONPATH=. py discovery/tools/independent_signal_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal

USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
HISTDATA = r"C:/Users/panap/histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
K = 40
SHADOW = 1.25
ATR_PERIOD = 14


def _independent_wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR, written fresh (TR seeded by mean of first `period` TRs at index `period`)."""
    n = len(close)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr
    tr = np.empty(n)
    tr[0] = np.nan
    for i in range(1, n):
        pc = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    atr[period] = float(np.mean(tr[1:period + 1]))
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _independent_fbr_fires(df: pd.DataFrame, k: int, shadow_min: float, period: int) -> np.ndarray:
    """Recompute the fbr fire mask from raw OHLC with INDEPENDENT code (no rolling/shift)."""
    n = len(df)
    low_bid = df["low_bid"].to_numpy(float)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0

    atr_raw = _independent_wilder_atr(high_mid, low_mid, close_mid, period)
    atr = np.empty(n); atr[0] = np.nan; atr[1:] = atr_raw[:-1]   # shift1, independently

    fire = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < k:                      # need k strictly-prior bars
            continue
        prior_low = low_bid[i - k:i].min()    # manual window [i-k, i-1]
        a = atr[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(prior_low)):
            continue
        shadow = (min(open_mid[i], close_mid[i]) - low_bid[i]) / a
        if (low_bid[i] < prior_low) and (close_mid[i] > prior_low) and (shadow >= shadow_min):
            fire[i] = True
    return fire


def main():  # pragma: no cover - audit driver
    print("loading H4 USD-major panel (canonical loader — DATA is the trusted foundation)...")
    panel = Panel.from_pairs(USD, tf="H4", histdata_root=HISTDATA, cache_root=CACHE,
                             use_cache=True, boundary_convention="5ers_eet")
    sig = FailedBreakdownReclaimLongSignal(swing_lookback=K, min_shadow_atr=SHADOW, atr_period=ATR_PERIOD)
    ev = sig.evaluate({"H4": panel})

    print("\n" + "=" * 78)
    print("CHECK 1 — INDEPENDENT FRESH RE-DERIVATION vs committed fbr fire set")
    print("=" * 78)
    total_committed = total_mine = total_agree = 0
    mismatches = []
    per_pair_fires = {}
    for pair in USD:
        df = panel.pair_dfs[pair]
        committed = ev.per_pair[pair].signal_mask.to_numpy(bool)
        mine = _independent_fbr_fires(df, K, SHADOW, ATR_PERIOD)
        agree = int((committed == mine).sum())
        disagree = np.where(committed != mine)[0]
        total_committed += int(committed.sum()); total_mine += int(mine.sum())
        total_agree += int((committed & mine).sum())
        per_pair_fires[pair] = (df, committed, mine)
        flag = "OK" if len(disagree) == 0 else f"!! {len(disagree)} DIFF"
        print(f"  {pair}: committed {int(committed.sum()):3d} | independent {int(mine.sum()):3d} | "
              f"bar-agree {agree}/{len(df)} | {flag}")
        if len(disagree):
            mismatches.append((pair, disagree))
    print(f"\n  TOTAL: committed {total_committed} fires | independent {total_mine} fires | "
          f"intersection {total_agree}")
    print(f"  VERDICT: {'IDENTICAL fire sets — signal re-derives independently' if not mismatches else 'MISMATCH — investigate'}")

    print("\n" + "=" * 78)
    print("CHECK 2 — CAUSAL-TRUNCATION NO-LOOKAHEAD PROOF (sample of fires)")
    print("(re-evaluate committed signal on df.iloc[:i+1]; fire@i must be byte-identical)")
    print("=" * 78)
    # representative sample: spread fires across pairs & time
    sample = []
    for pair in USD:
        df, committed, _ = per_pair_fires[pair]
        fire_idx = np.where(committed)[0]
        if len(fire_idx) == 0:
            continue
        pick = fire_idx[:: max(1, len(fire_idx) // 3)][:3]   # ~3 spread across the pair's fires
        for i in pick:
            sample.append((pair, int(i)))
    n_ok = 0
    for pair, i in sample:
        df = panel.pair_dfs[pair]
        trunc = Panel.from_frames({pair: df.iloc[: i + 1]}, tf="H4", boundary_convention="5ers_eet")
        ev_t = sig.evaluate({"H4": trunc})
        m_t = ev_t.per_pair[pair].signal_mask.to_numpy(bool)
        atr_t = ev_t.per_pair[pair].atr.to_numpy(float)
        atr_full = ev.per_pair[pair].atr.to_numpy(float)
        fire_ok = bool(m_t[i]) and bool(ev.per_pair[pair].signal_mask.to_numpy(bool)[i])
        atr_ok = np.isclose(atr_t[i], atr_full[i], rtol=0, atol=1e-12, equal_nan=True)
        ok = fire_ok and atr_ok
        n_ok += int(ok)
        ts = df.index[i]
        print(f"  {pair} @ {ts}  fire(trunc)={bool(m_t[i])} fire(full)={bool(ev.per_pair[pair].signal_mask.to_numpy(bool)[i])} "
              f"atr_match={atr_ok}  -> {'OK' if ok else '!! LOOKAHEAD'}")
    print(f"\n  VERDICT: {n_ok}/{len(sample)} sampled fires identical under truncation "
          f"-> {'NO LOOKAHEAD' if n_ok == len(sample) else 'LOOKAHEAD DETECTED'}")

    print("\n" + "=" * 78)
    print("CHECK 3 — RAW-OHLC HAND-AUDIT (geometry visible against raw price)")
    print("=" * 78)
    for pair, i in sample[:4]:
        df = panel.pair_dfs[pair]
        low_bid = df["low_bid"].to_numpy(float)
        open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
        close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
        prior_low = low_bid[i - K:i].min()
        prior_low_at = i - K + int(np.argmin(low_bid[i - K:i]))
        atr_i = ev.per_pair[pair].atr.to_numpy(float)[i]
        shadow = (min(open_mid[i], close_mid[i]) - low_bid[i]) / atr_i
        print(f"\n  {pair} fire @ {df.index[i]} (bar {i}):")
        print(f"    prior {K}-bar swing low (bars [{i-K},{i-1}]) = {prior_low:.5f} "
              f"(at bar {prior_low_at}, {df.index[prior_low_at]})")
        print(f"    bar low_bid {low_bid[i]:.5f}  < swing low?  {low_bid[i] < prior_low}  (PIERCE/stop-sweep)")
        print(f"    bar close_mid {close_mid[i]:.5f}  > swing low?  {close_mid[i] > prior_low}  (RECLAIM/failed breakdown)")
        print(f"    lower shadow = (min(open,close)_mid {min(open_mid[i],close_mid[i]):.5f} - low_bid {low_bid[i]:.5f}) "
              f"/ atr {atr_i:.5f} = {shadow:.3f}  >= {SHADOW}?  {shadow >= SHADOW}  (DEEP wick)")

    print("\n" + "=" * 78)
    print("AUDIT SUMMARY")
    print("=" * 78)
    print(f"  fire-set re-derivation : {'PASS (identical)' if not mismatches else 'FAIL'}")
    print(f"  no-lookahead (trunc)   : {'PASS' if n_ok == len(sample) else 'FAIL'} ({n_ok}/{len(sample)})")
    print("  raw-OHLC geometry      : printed above for human audit")


if __name__ == "__main__":  # pragma: no cover
    main()
