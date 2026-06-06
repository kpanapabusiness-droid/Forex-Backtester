"""Independent §11 signal verification — the OTHER 3 book components (gap, me_long, me_short).

Arc 2034 independently verified `fbr` (1013); this extends the §11 Arc-10 defense to the rest of
the deployable book so EVERY component's signal is independently confirmed honest, not just fbr.
Reuses the independent Wilder ATR from `independent_signal_audit` (arc 2034 proved it reproduces the
committed `_atr_shift1_mid` byte-identically — so the shared ATR/mid construction all 4 components use
is already validated; this arc validates each component's bespoke fire LOGIC + no-lookahead).

Two checks per component, NONE importing the committed signal's fire logic for the re-derivation:

1. **Fresh re-derivation** — recompute the fire mask from raw OHLC with independent code, compare to
   the committed signal EXACTLY.
2. **Causal-truncation no-lookahead proof** — re-evaluate the committed signal on data truncated just
   past the fire bar; the fire + ATR must be byte-identical. A CALENDAR signal (month-end) legitimately
   reads the NEXT bar's TIMESTAMP (not price) to know bar i is the last trading day, so it is truncated
   at i+1 AND given a **price-isolation** check (corrupt bar i+1's PRICE, keep its timestamp → the fire
   must be unchanged, proving only the calendar position of i+1 is used, never its price). The weekend-gap
   signal reads only bars <= i (the inter-bar time-gap from i-1 to i), so it truncates cleanly at i.

EXPERIMENT tool: re-derivation + audit only; never realizes P&L / scores / touches the gate / spends OOS.
Only the trusted DATA loader (`Panel.from_pairs`) is canonical.

Run:  PYTHONPATH=. py discovery/tools/independent_signal_audit_book.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.independent_signal_audit import _independent_wilder_atr
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    MonthEndReversionShortSignal,
)

HISTDATA = r"C:/Users/panap/histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
BOUND = "5ers_eet"
JPY = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
PERIOD = 14


def _atr_shift1_independent(df: pd.DataFrame, period: int) -> np.ndarray:
    high = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    low = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    close = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    raw = _independent_wilder_atr(high, low, close, period)
    out = np.empty(len(df)); out[0] = np.nan; out[1:] = raw[:-1]
    return out


def _indep_gap_fires(df, thr, gap_hours, period):
    n = len(df)
    open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_independent(df, period)
    fire = np.zeros(n, dtype=bool)
    ts = df.index
    for i in range(1, n):
        dt_h = (ts[i] - ts[i - 1]).total_seconds() / 3600.0
        a = atr[i]
        if dt_h <= gap_hours or not (np.isfinite(a) and a > 0):
            continue
        gap_atr = (open_mid[i] - close_mid[i - 1]) / a
        if gap_atr <= -thr:
            fire[i] = True
    return fire


def _indep_me_fires(df, thr, into_bars, period, short):
    n = len(df)
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_independent(df, period)
    ts = df.index
    months = np.array([t.year * 12 + t.month for t in ts])   # independent month index
    fire = np.zeros(n, dtype=bool)
    for i in range(into_bars, n):
        if i + 1 >= n:                       # last trading day = next bar in a new month
            continue
        is_last = months[i + 1] != months[i]
        a = atr[i]
        if not (is_last and np.isfinite(a) and a > 0):
            continue
        into = (close_mid[i] - close_mid[i - into_bars]) / a
        if (into >= thr) if short else (into <= -thr):
            fire[i] = True
    return fire


def _sample(fire_idx, k=3):
    if len(fire_idx) == 0:
        return []
    return list(fire_idx[:: max(1, len(fire_idx) // k)][:k])


def _truncation_check(sig, pair, df, full_mask, full_atr, tf, fire_idx, calendar_bars, isolate_price):
    """Return (n_ok, n) over sampled fires; optional price-isolation of the calendar bars."""
    price_cols = [c for c in df.columns if any(k in c for k in ("open", "high", "low", "close"))]
    sample = _sample(fire_idx)
    n_ok = 0
    detail = []
    for i in sample:
        end = i + 1 + calendar_bars
        sub = df.iloc[:end].copy()
        if isolate_price:
            for j in range(i + 1, end):       # corrupt the future bar's PRICE, keep its timestamp
                for c in price_cols:
                    sub.iloc[j, sub.columns.get_loc(c)] = sub.iloc[j, sub.columns.get_loc(c)] * 3.0 + 1.0
        trunc = Panel.from_frames({pair: sub}, tf=tf, boundary_convention=BOUND)
        ev = sig.evaluate({tf: trunc})
        m = ev.per_pair[pair].signal_mask.to_numpy(bool)
        a = ev.per_pair[pair].atr.to_numpy(float)
        ok = bool(m[i]) and bool(full_mask[i]) and np.isclose(a[i], full_atr[i], atol=1e-12, equal_nan=True)
        n_ok += int(ok)
        detail.append((pair, df.index[i], bool(m[i]), ok))
    return n_ok, len(sample), detail


def _audit(label, sig, pairs, tf, indep_fn, calendar_bars, isolate_price):
    print("\n" + "=" * 78)
    print(f"COMPONENT: {label}  (tf={tf}, calendar_lookahead_bars={calendar_bars}, "
          f"price_isolation={'yes' if isolate_price else 'n/a'})")
    print("=" * 78)
    panel = Panel.from_pairs(pairs, tf=tf, histdata_root=HISTDATA, cache_root=CACHE,
                             use_cache=True, boundary_convention=BOUND)
    ev = sig.evaluate({tf: panel})
    tot_c = tot_m = mismatches = 0
    trunc_ok = trunc_n = 0
    iso_ok = iso_n = 0
    for pair in pairs:
        df = panel.pair_dfs[pair]
        committed = ev.per_pair[pair].signal_mask.to_numpy(bool)
        mine = indep_fn(df)
        diff = int((committed != mine).sum())
        tot_c += int(committed.sum()); tot_m += int(mine.sum()); mismatches += diff
        full_atr = ev.per_pair[pair].atr.to_numpy(float)
        fire_idx = np.where(committed)[0]
        ok, ntest, _ = _truncation_check(sig, pair, df, committed, full_atr, tf, fire_idx, calendar_bars, False)
        trunc_ok += ok; trunc_n += ntest
        if isolate_price:
            iok, intest, _ = _truncation_check(sig, pair, df, committed, full_atr, tf, fire_idx, calendar_bars, True)
            iso_ok += iok; iso_n += intest
        print(f"  {pair}: committed {int(committed.sum()):3d} | independent {int(mine.sum()):3d} | "
              f"re-derive {'OK' if diff == 0 else f'!! {diff} DIFF'} | trunc {ok}/{ntest}"
              + (f" | price-iso {iok}/{intest}" if isolate_price else ""))
    print(f"  TOTAL: committed {tot_c} | independent {tot_m} | re-derive "
          f"{'IDENTICAL' if mismatches == 0 else f'{mismatches} MISMATCH'} | "
          f"no-lookahead {trunc_ok}/{trunc_n}"
          + (f" | price-isolation {iso_ok}/{iso_n}" if isolate_price else ""))
    return mismatches == 0 and trunc_ok == trunc_n and (not isolate_price or iso_ok == iso_n)


def main():  # pragma: no cover - audit driver
    results = {}
    results["gap (1006)"] = _audit(
        "gap-fill LONG (1006)", WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36),
        JPY, "H4", lambda df: _indep_gap_fires(df, 0.5, 36, PERIOD), calendar_bars=0, isolate_price=False)
    results["me_long (1011)"] = _audit(
        "month-end reversion LONG (1011)", MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2),
        USD, "D1", lambda df: _indep_me_fires(df, 1.0, 2, PERIOD, short=False), calendar_bars=1, isolate_price=True)
    results["me_short (1019)"] = _audit(
        "month-end reversion SHORT (1019)", MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2),
        USD, "D1", lambda df: _indep_me_fires(df, 1.0, 2, PERIOD, short=True), calendar_bars=1, isolate_price=True)

    print("\n" + "=" * 78)
    print("BOOK SIGNAL-AUDIT SUMMARY (with fbr from arc 2034 = PASS)")
    print("=" * 78)
    for k, ok in results.items():
        print(f"  {k:18s}: {'PASS (re-derive identical + no-lookahead)' if ok else 'FAIL — investigate'}")
    print(f"  {'fbr (1013)':18s}: PASS (arc 2034: 356/356 fires, 21/21 no-lookahead)")


if __name__ == "__main__":  # pragma: no cover
    main()
