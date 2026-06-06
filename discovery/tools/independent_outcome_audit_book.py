"""§11 OUTCOME-layer independent audit of the LONG simple-exit book legs (arc 2037).

Extends arc 2036 (`independent_outcome_audit_fbr.py`, which audited fbr) to the two
other LONG components whose exits are simple ({stop_loss, time_exit} — no trail, no
partial close): the weekend gap-fill (1006, H4 JPY crosses) and month-end reversion
LONG (me_long, 1011, D1 USD majors). With fbr (2036) this leaves only me_short
(1019, SHORT + partial-close multi-leg) for a future arc — the remaining OUTCOME-layer
§11 slice, plus per-trade cost re-derivation for all legs.

Same §11 discipline as 2036: read ONLY (a) the trusted DATA loader and (b) the engine's
`ClosedTrade` ledger (the CLAIM); re-derive every check from raw OHLC + the documented
engine conventions; never call the engine's exit code. Both legs here are LONG with
only stop_loss + time_exit exits, so the convention set is exactly fbr's non-trail subset:

  - ENTRY  : engine entry_price == open_ask[entry bar]            (long buys the ask)
  - SL GEOM: engine sl_price == close_ask[signal bar] - 2*ATR[signal bar]
             (ATR = the arc-2034 INDEPENDENT Wilder(14) mid, shift1)
  - TAKE-THE-LOSS / ARC-10: no bar strictly between entry & exit has low_bid <= sl
  - STOP-BAR: stop_loss bar breached (low_bid<=sl) & filled at sl; time_exit FILL
              (open_bid[exit bar]) is above the stop
  - EXIT PX: stop_loss -> sl ; time_exit -> open_bid[exit bar] (next-bar-open fill —
             the predicate fires at bar close and queues a close at the next open;
             its returned close_bid is discarded, verified vs _check_exits/_fill_pending_closes)
  - R / PNL: gross pnl == (exit-entry)*size ; gross R = (exit-entry)/(close_ask_sig - sl)

EXPERIMENT tool: re-derivation + audit only; never realizes P&L for a gate, never scores,
never spends OOS. IS-only (folds 2011-2020).

Run:  PYTHONPATH=. py discovery/tools/independent_outcome_audit_book.py
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.account import Direction
from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import MonthEndReversionLongSignal
from discovery.tools.time_exit_predicate import make_time_exit_predicate

HISTDATA = r"C:/Users/panap/histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
JPY = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
ATR_PERIOD = 14
SL_MULT = 2.0
PX_TOL = 5e-6
R_TOL = 1e-6


def _independent_wilder_atr(high, low, close, period):
    n = len(close)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr
    tr = np.empty(n); tr[0] = np.nan
    for i in range(1, n):
        pc = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    atr[period] = float(np.mean(tr[1: period + 1]))
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _precompute(df):
    o_ask = df["open_ask"].to_numpy(float)
    o_bid = df["open_bid"].to_numpy(float)
    c_ask = df["close_ask"].to_numpy(float)
    low_bid = df["low_bid"].to_numpy(float)
    high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    raw_atr = _independent_wilder_atr(high_mid, low_mid, close_mid, ATR_PERIOD)
    atr_s1 = np.empty(len(df)); atr_s1[0] = np.nan; atr_s1[1:] = raw_atr[:-1]
    return {"o_ask": o_ask, "o_bid": o_bid, "c_ask": c_ask, "low_bid": low_bid, "atr_s1": atr_s1}


def _attach_time_exit(sig, panel, n_bars):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n_bars)
    return dataclasses.replace(sig, per_pair={
        pair: dataclasses.replace(st, exit_predicate=pred) for pair, st in sig.per_pair.items()})


def _collect(eval_, panels, cfg):
    runner = ArcFoldRunner(A1Architecture(), eval_, panels)
    folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    uniq = {}
    for fold in folds:
        runner(fold, cfg)
        for t in runner.last_result.run_result.closed_trades:
            uniq[(t.pair, t.entry_time, round(t.entry_price, 6), t.exit_time)] = t
    return list(uniq.values())


def _audit(name, trades, panel, pairs):
    arr = {p: _precompute(panel.pair_dfs[p]) for p in pairs}
    idx = {p: panel.pair_dfs[p].index for p in pairs}
    n = len(trades)
    c = dict(entry=0, sl=0, ttl=0, stopbar=0, exitpx=0, pnl=0)
    viol = {k: [] for k in ["entry", "sl", "ttl", "stopbar", "exitpx", "pnl"]}
    R_by_reason = {}
    for t in trades:
        assert t.direction is Direction.LONG, f"{name} expected long"
        a, ix = arr[t.pair], idx[t.pair]
        e = ix.get_loc(t.entry_time); xi = ix.get_loc(t.exit_time); sig = e - 1
        if abs(t.entry_price - a["o_ask"][e]) <= PX_TOL: c["entry"] += 1
        else: viol["entry"].append((t.pair, t.entry_time, t.entry_price, a["o_ask"][e]))
        sl_indep = a["c_ask"][sig] - SL_MULT * a["atr_s1"][sig]
        if t.sl_price is not None and abs(t.sl_price - sl_indep) <= PX_TOL: c["sl"] += 1
        else: viol["sl"].append((t.pair, t.entry_time, t.sl_price, sl_indep))
        sl = t.sl_price
        breach = next((b for b in range(e + 1, xi) if a["low_bid"][b] <= sl), None)
        if breach is None: c["ttl"] += 1
        else: viol["ttl"].append((t.pair, t.entry_time, t.exit_time, t.exit_reason,
                                  str(ix[breach]), float(a["low_bid"][breach]), float(sl)))
        if t.exit_reason == "stop_loss":
            sb = (a["low_bid"][xi] <= sl) and abs(t.exit_price - sl) <= PX_TOL; ref = float(a["low_bid"][xi])
        else:
            sb = a["o_bid"][xi] > sl; ref = float(a["o_bid"][xi])
        if sb: c["stopbar"] += 1
        else: viol["stopbar"].append((t.pair, t.entry_time, t.exit_time, t.exit_reason, ref, float(sl), t.exit_price))
        if t.exit_reason == "stop_loss": epx = abs(t.exit_price - sl) <= PX_TOL; eref = sl
        else: epx = abs(t.exit_price - a["o_bid"][xi]) <= PX_TOL; eref = a["o_bid"][xi]
        if epx: c["exitpx"] += 1
        else: viol["exitpx"].append((t.pair, t.entry_time, t.exit_reason, t.exit_price, float(eref)))
        pnl_indep = (t.exit_price - t.entry_price) * t.size
        if abs(pnl_indep - t.pnl) <= max(R_TOL * abs(t.pnl), 1e-4): c["pnl"] += 1
        else: viol["pnl"].append((t.pair, t.entry_time, t.pnl, pnl_indep))
        risk_px = a["c_ask"][sig] - sl
        if risk_px > 0:
            R_by_reason.setdefault(t.exit_reason, []).append((t.exit_price - t.entry_price) / risk_px)

    def pc(x): return f"{x}/{n} ({100*x/n:.1f}%)"
    print(f"\n{'='*78}\nOUTCOME-LAYER AUDIT — {name} ({n} trades)  exit mix: {dict(Counter(t.exit_reason for t in trades))}\n{'='*78}")
    print(f"  1. ENTRY  == open_ask[entry]                  : {pc(c['entry'])}")
    print(f"  2. SL     == close_ask[sig] - 2*indepATR[sig] : {pc(c['sl'])}")
    print(f"  3. TAKE-THE-LOSS (no missed earlier stop)     : {pc(c['ttl'])}   <-- Arc-10 defect test")
    print(f"  4. STOP-BAR consistency                       : {pc(c['stopbar'])}")
    print(f"  5. EXIT PX (stop->sl / time->open_bid)        : {pc(c['exitpx'])}")
    print(f"  6. PNL == (exit-entry)*size                   : {pc(c['pnl'])}")
    for reason, rs in sorted(R_by_reason.items()):
        rs = np.array(rs)
        print(f"    R[{reason:10s}]: {rs.mean():+.3f} / {rs.min():+.3f} / {rs.max():+.3f}  (n={len(rs)})")
    clean = all(not v for v in viol.values())
    for k, v in viol.items():
        if v:
            print(f"  !! {k} VIOLATIONS ({len(v)}): {v[:5]}")
    print(f"  -> {name}: {'ALL CHECKS PASS' if clean else 'VIOLATIONS — investigate'}")
    return clean


def main():  # pragma: no cover
    print("loading panels (H4 JPY crosses + D1 USD majors)...")
    h4 = Panel.from_pairs(JPY, tf="H4", histdata_root=HISTDATA, cache_root=CACHE, use_cache=True, boundary_convention="5ers_eet")
    d1 = Panel.from_pairs(USD, tf="D1", histdata_root=HISTDATA, cache_root=CACHE, use_cache=True, boundary_convention="5ers_eet")

    gap_eval = _attach_time_exit(WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4}), h4, n_bars=24)
    gap_cfg = A1Config(config_id="arc_1006", sl_atr_mult=SL_MULT, trail_enabled=False, exit_policy=None)
    me_eval = _attach_time_exit(MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1}), d1, n_bars=2)
    me_cfg = A1Config(config_id="arc_1011", sl_atr_mult=SL_MULT, trail_enabled=False, exit_policy="sl_only")

    gap_trades = _collect(gap_eval, {"H4": h4}, gap_cfg)
    me_trades = _collect(me_eval, {"D1": d1}, me_cfg)

    ok_gap = _audit("gap (1006, H4 JPY)", gap_trades, h4, JPY)
    ok_me = _audit("me_long (1011, D1 USD)", me_trades, d1, USD)

    print(f"\n{'='*78}\nVERDICT: {'BOTH legs OUTCOME-verified honest' if (ok_gap and ok_me) else 'VIOLATIONS FOUND'}\n{'='*78}")


if __name__ == "__main__":  # pragma: no cover
    main()
