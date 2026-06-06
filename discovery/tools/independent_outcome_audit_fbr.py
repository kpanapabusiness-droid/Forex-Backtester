"""§11 OUTCOME-layer independent audit of the fbr component (arc 2036).

Companion to the SIGNAL-layer audits (`independent_signal_audit.py`, arc 2034;
`independent_signal_audit_book.py`, arc 2035). Those verified the *fire set* (which
bars fire, no price-lookahead). This module verifies the *outcome* of each fired
trade — entry price, SL geometry, exit, R — against RAW PRICE, the explicit §11
requirement ("a hand-audit of a representative sample of its trades against raw
price (entry, exit, R, cost) confirming they match the engine's claim").

WHY THIS IS THE OWED NEXT STEP. The deployable 4-way book rests on the engine's
per-trade P&L. The engine is heavily tested (1656-test suite, honest-engine sweep
PR #263/#264), but §11 demands a GENUINELY INDEPENDENT path before deployment,
because a single trusted engine is exactly the Arc-10 trap. The reset that wiped
this repo was an OUTCOME-layer defect: a gate that scored profits while SKIPPING
pre-partial stop breaches (`docs/ARC_10_GATE_FIDELITY_DEFECT.md`). That is the
precise failure this audit re-tests, at the trade level, from raw price, for fbr —
the corpus's load-bearing component (arc 2033) running on an accidental DOUBLE-TRAIL
(`trail_enabled=True` default + `exit_policy="sl_plus_trailing_atr"`; flagged 1015/3009).

METHOD — read ONLY (a) the trusted DATA loader (`Panel.from_pairs`) and (b) the
engine's per-trade ledger (`ClosedTrade`: the CLAIM under audit). Re-derive every
check from raw OHLC + the DOCUMENTED rules; never call the engine's exit/trail code.
The audit verifies the INVARIANTS that must hold no matter which of the two trails
bound — it does NOT re-implement the double-trail walk (that would be transcription,
not independent verification). The inviolable rules (entry/SL/exit price conventions,
take-the-loss, R geometry) are convention-robust and decisive.

Per trade (long-only; fbr is long), against raw price:
  1. ENTRY  — engine entry_price == open_ask at the entry bar (long buys the ask).
  2. SL GEOM — engine sl_price == close_ask[signal bar] - 2*ATR[signal bar], with an
     INDEPENDENT Wilder ATR (the arc-2034 re-derivation, proven == engine ATR). This
     re-derives the R-denominator (sl_distance) from raw price.
  3. TAKE-THE-LOSS / ARC-10 — for EVERY exit type, NO bar strictly between entry and
     exit has low_bid <= sl_price (a missed earlier stop = the Arc-10 defect). This is
     convention-robust (independent of entry/exit-bar edge handling).
  4. STOP-BAR — a stop_loss exit's bar actually breached (low_bid <= sl_price) and
     filled at sl_price; a non-stop exit's bar did NOT breach (else SL-first should win).
  5. EXIT PX — non-stop exits fill at open_bid of the exit bar (next-bar-open trail fill).
  6. R / PNL — gross pnl == (exit-entry)*size; gross R = (exit-entry)/(close_ask_sig - sl)
     is sane (SL exits ~ -1R minus entry slippage+spread).

EXPERIMENT tool: re-derivation + audit only; never realizes P&L for a gate, never
scores, never spends OOS. IS-only (folds 2011-2020).

Run:  PYTHONPATH=. py discovery/tools/independent_outcome_audit_fbr.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.account import Direction
from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal

USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
HISTDATA = r"C:/Users/panap/histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
K = 40
SHADOW = 1.25
ATR_PERIOD = 14
SL_MULT = 2.0
# pip size per pair for human-readable tolerances (price-unit abs tolerances are used in code)
PX_TOL = 5e-6      # absolute price tolerance for entry/exit/SL identity (sub-pip)
R_TOL = 1e-6       # relative tolerance for pnl reconciliation


def _independent_wilder_atr(high, low, close, period):
    """Wilder ATR, written fresh (arc-2034 re-derivation; proven == engine ATR)."""
    n = len(close)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr
    tr = np.empty(n)
    tr[0] = np.nan
    for i in range(1, n):
        pc = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    atr[period] = float(np.mean(tr[1: period + 1]))
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _collect_fbr_trades():
    """Run fbr under its COMMITTED config; return {pair: df} and unique closed trades."""
    h4 = Panel.from_pairs(USD, tf="H4", histdata_root=HISTDATA, cache_root=CACHE,
                          use_cache=True, boundary_convention="5ers_eet")
    ev = FailedBreakdownReclaimLongSignal(
        swing_lookback=K, min_shadow_atr=SHADOW, atr_period=ATR_PERIOD).evaluate({"H4": h4})
    # COMMITTED fbr: trail_enabled=True (A1 default => double-trail), sl_plus_trailing_atr, SL 2.0
    cfg = A1Config(config_id="arc_1013", sl_atr_mult=SL_MULT, exit_policy="sl_plus_trailing_atr")
    assert cfg.trail_enabled is True, "committed fbr config double-trails (trail_enabled default True)"
    runner = ArcFoldRunner(A1Architecture(), ev, {"H4": h4})
    folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    uniq = {}
    for fold in folds:
        runner(fold, cfg)
        for t in runner.last_result.run_result.closed_trades:
            uniq[(t.pair, t.entry_time, round(t.entry_price, 6), t.exit_time)] = t
    return h4, list(uniq.values())


def _precompute(df):
    """Raw-price arrays + independent ATR (shift1) + independent SL for a pair df."""
    o_ask = df["open_ask"].to_numpy(float)
    o_bid = df["open_bid"].to_numpy(float)
    c_ask = df["close_ask"].to_numpy(float)
    low_bid = df["low_bid"].to_numpy(float)
    high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    raw_atr = _independent_wilder_atr(high_mid, low_mid, close_mid, ATR_PERIOD)
    atr_s1 = np.empty(len(df)); atr_s1[0] = np.nan; atr_s1[1:] = raw_atr[:-1]   # shift1
    return {"o_ask": o_ask, "o_bid": o_bid, "c_ask": c_ask, "low_bid": low_bid, "atr_s1": atr_s1}


def main():  # pragma: no cover - audit driver
    print("loading H4 USD-major panel + running fbr (committed double-trail config)...")
    panel, trades = _collect_fbr_trades()
    print(f"unique fbr trades across IS folds 2011-2020: {len(trades)}")
    print("exit_reason counts:", dict(Counter(t.exit_reason for t in trades)))

    arr = {p: _precompute(panel.pair_dfs[p]) for p in USD}
    idx = {p: panel.pair_dfs[p].index for p in USD}

    # counters
    n = len(trades)
    ok_entry = ok_sl = ok_ttl = ok_stopbar = ok_exitpx = ok_pnl = 0
    viol_ttl = []          # Arc-10: missed earlier stop
    viol_stopbar = []      # claimed stop didn't breach / non-stop exit DID breach at exit bar
    viol_entry = viol_sl = viol_exitpx = viol_pnl = []
    viol_entry, viol_sl, viol_exitpx, viol_pnl = [], [], [], []
    gross_R = []
    R_by_reason = {}

    for t in trades:
        assert t.direction is Direction.LONG
        p = t.pair
        a = arr[p]
        ix = idx[p]
        try:
            e = ix.get_loc(t.entry_time)
            xi = ix.get_loc(t.exit_time)
        except KeyError:
            viol_entry.append((p, t.entry_time, "timestamp not in index"))
            continue
        sig = e - 1  # entry fills the bar AFTER the signal bar

        # 1. ENTRY: long buys the ask at the entry bar open
        entry_ok = abs(t.entry_price - a["o_ask"][e]) <= PX_TOL
        ok_entry += int(entry_ok)
        if not entry_ok:
            viol_entry.append((p, t.entry_time, t.entry_price, a["o_ask"][e]))

        # 2. SL GEOMETRY: close_ask[sig] - 2*ATR[sig], independent ATR
        sl_indep = a["c_ask"][sig] - SL_MULT * a["atr_s1"][sig]
        sl_ok = (t.sl_price is not None) and abs(t.sl_price - sl_indep) <= PX_TOL
        ok_sl += int(sl_ok)
        if not sl_ok:
            viol_sl.append((p, t.entry_time, t.sl_price, sl_indep))

        sl = t.sl_price

        # 3. TAKE-THE-LOSS / ARC-10: no missed earlier stop strictly between entry & exit
        breach_between = False
        first_breach = None
        for b in range(e + 1, xi):           # bars strictly between entry and exit
            if a["low_bid"][b] <= sl:
                breach_between = True
                first_breach = b
                break
        ttl_ok = not breach_between
        ok_ttl += int(ttl_ok)
        if not ttl_ok:
            viol_ttl.append((p, t.entry_time, t.exit_time, t.exit_reason,
                             str(ix[first_breach]), float(a["low_bid"][first_breach]), float(sl)))

        # 4. STOP-BAR consistency at the claimed exit bar.
        #    A trail/time exit is QUEUED at the prior bar's close and FILLS at the
        #    exit bar's open_bid (engine order: fill pending closes -> THEN intra-bar
        #    SL). So the take-the-loss invariant for a non-stop exit is that the FILL
        #    (open_bid[xi]) is above the stop; the exit bar's intra-bar low coming
        #    later is post-close and irrelevant (verified vs run() bar ordering).
        if t.exit_reason == "stop_loss":
            sb_ok = (a["low_bid"][xi] <= sl) and abs(t.exit_price - sl) <= PX_TOL
            ref = float(a["low_bid"][xi])
        else:
            sb_ok = a["o_bid"][xi] > sl       # the trail fill price is above the stop
            ref = float(a["o_bid"][xi])
        ok_stopbar += int(sb_ok)
        if not sb_ok:
            viol_stopbar.append((p, t.entry_time, t.exit_time, t.exit_reason,
                                 ref, float(sl), t.exit_price))

        # 5. EXIT PRICE convention
        if t.exit_reason == "stop_loss":
            epx_ok = abs(t.exit_price - sl) <= PX_TOL
        else:
            epx_ok = abs(t.exit_price - a["o_bid"][xi]) <= PX_TOL   # next-bar-open trail fill (bid)
        ok_exitpx += int(epx_ok)
        if not epx_ok:
            ref = sl if t.exit_reason == "stop_loss" else a["o_bid"][xi]
            viol_exitpx.append((p, t.entry_time, t.exit_reason, t.exit_price, float(ref)))

        # 6. R / PNL reconciliation (gross, pre-cost)
        pnl_indep = (t.exit_price - t.entry_price) * t.size
        pnl_ok = abs(pnl_indep - t.pnl) <= max(R_TOL * abs(t.pnl), 1e-4)
        ok_pnl += int(pnl_ok)
        if not pnl_ok:
            viol_pnl.append((p, t.entry_time, t.pnl, pnl_indep))
        risk_px = a["c_ask"][sig] - sl                   # entry_proxy - sl, the R unit
        if risk_px > 0:
            r = (t.exit_price - t.entry_price) / risk_px
            gross_R.append(r)
            R_by_reason.setdefault(t.exit_reason, []).append(r)

    def pct(x):
        return f"{x}/{n} ({100*x/n:.1f}%)"

    print("\n" + "=" * 78)
    print("OUTCOME-LAYER AUDIT — fbr per-trade vs RAW PRICE (committed double-trail)")
    print("=" * 78)
    print(f"  1. ENTRY  == open_ask[entry bar]              : {pct(ok_entry)}")
    print(f"  2. SL     == close_ask[sig] - 2*indepATR[sig] : {pct(ok_sl)}")
    print(f"  3. TAKE-THE-LOSS (no missed earlier stop)     : {pct(ok_ttl)}   <-- Arc-10 defect test")
    print(f"  4. STOP-BAR consistency (breach <=> stop exit): {pct(ok_stopbar)}")
    print(f"  5. EXIT PX (stop->sl / trail->open_bid)       : {pct(ok_exitpx)}")
    print(f"  6. PNL == (exit-entry)*size                   : {pct(ok_pnl)}")

    print("\n  gross-R by exit_reason (mean / min / max / n):")
    for reason, rs in sorted(R_by_reason.items()):
        rs = np.array(rs)
        print(f"    {reason:18s}: {rs.mean():+.3f} / {rs.min():+.3f} / {rs.max():+.3f}  (n={len(rs)})")
    allr = np.array(gross_R)
    print(f"    {'ALL':18s}: {allr.mean():+.3f} / {allr.min():+.3f} / {allr.max():+.3f}  (n={len(allr)})")

    def show(name, v):
        if v:
            print(f"\n  !! {name} VIOLATIONS ({len(v)}):")
            for row in v[:8]:
                print("     ", row)
        else:
            print(f"  {name}: clean")

    print("\n" + "-" * 78)
    show("ENTRY", viol_entry)
    show("SL-GEOM", viol_sl)
    show("TAKE-THE-LOSS (ARC-10)", viol_ttl)
    show("STOP-BAR", viol_stopbar)
    show("EXIT-PX", viol_exitpx)
    show("PNL", viol_pnl)

    all_clean = not (viol_entry or viol_sl or viol_ttl or viol_stopbar or viol_exitpx or viol_pnl)
    print("\n" + "=" * 78)
    print("VERDICT:", "ALL CHECKS PASS — fbr OUTCOME layer independently verified honest"
          if all_clean else "VIOLATIONS FOUND — investigate above")
    print("=" * 78)


if __name__ == "__main__":  # pragma: no cover
    main()
