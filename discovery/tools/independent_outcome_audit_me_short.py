"""§11 OUTCOME-layer independent audit of me_short (1019) — the LAST book leg (arc 2038).

Completes the gross-OUTCOME §11 verification of the deployable 4-component reversion book.
Companions: `independent_outcome_audit_fbr.py` (2036, fbr) and `independent_outcome_audit_book.py`
(2037, gap 1006 + me_long 1011). Those covered the three LONG legs; this covers the one
remaining leg, which is the hardest / most Arc-10-prone outcome config in the book:

  me_short (1019): SHORT (entry `open_bid`, non-stop exit `open_ask`, SL trigger `high_ask >= sl`,
  SL = `close_bid[sig] + 2*ATR`) AND `sl_partial_close_1r_runner_trail` — a +1R PARTIAL close
  (50%) → a runner that trails 1R above its favorable extreme → MULTI-LEG `ClosedTrade`s grouped
  by `position_id`. Committed config (validate_4way_book.py): MonthEndReversionShortSignal(
  threshold_atr=1.0, into_bars=2), D1 USD majors, trail_enabled=False (native A1 TrailManager is
  short-deferred; only the exit-policy partial/trail binds).

Same §11 discipline as 2036/2037: read ONLY (a) the trusted DATA loader (`Panel.from_pairs`) and
(b) the engine's per-trade `ClosedTrade` ledger (the CLAIM under audit); re-derive every check
from raw OHLC + the DOCUMENTED engine conventions; NEVER call the engine's exit/partial code.
Verify the convention-robust INVARIANTS, not a re-coded partial/trail walk (that = transcription).

SHORT conventions, read from `core/sim/fill.py` + `multipair_backtester.py` + the partial-close
policy (to know WHAT to assert, not copied into the re-derivation):
  - ENTRY  : engine entry_price == open_bid[entry bar]            (short sells the bid)
  - SL GEOM: engine sl_price == close_bid[signal bar] + 2*ATR[signal bar]  (ATR = arc-2034
             INDEPENDENT Wilder(14) mid, shift1; sig = entry-1). R unit = sl - close_bid[sig]
             = 2*ATR[sig]; r_atr (partial/trail distance) = sl_atr_mult * atr_at_entry = same.
  - TAKE-THE-LOSS / ARC-10: NO bar strictly between entry & a leg's exit has high_ask >= sl.
  - PARTIAL leg (`partial_close_1r`): intra-bar fill at tp1 = entry - r_atr = open_bid[e] -
             2*ATR[sig]; fires only if the bar did NOT first breach the stop (SL-first: the
             engine checks the intra-bar stop BEFORE the partial hook, so a partial leg EXISTING
             implies high_ask[partial bar] < sl — the take-the-loss precedence for the partial).
  - RUNNER stop_loss leg : intra-bar high_ask >= sl, fill at sl.
  - RUNNER runner_trail_stop leg : queued at prior bar close, fills NEXT-bar open_ask
             (`_fill_pending_closes` runs BEFORE the intra-bar SL check → the take-the-loss
             reference is the FILL price: open_ask[xi] < sl, mirror of fbr's 2036 long resolution).
  - R / PNL: gross pnl == direction.sign * (exit-entry) * size (short sign = -1, i.e.
             (entry-exit)*size); gross R = sign*(exit-entry)/(sl - close_bid[sig]).

Plus a MULTI-LEG STRUCTURAL check (within each fold, where position_id is unique): every
partial-bearing position has exactly one `partial_close_1r` leg + one runner leg sharing the same
`parent_position_id`, with leg sizes summing to the original full position size (no size leakage).

EXPERIMENT tool: re-derivation + audit only; never realizes P&L for a gate, never scores, never
spends OOS. IS-only (folds 2011-2020).

Run:  PYTHONPATH=. py discovery/tools/independent_outcome_audit_me_short.py
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.account import Direction
from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from discovery.tools.month_end_signals import MonthEndReversionShortSignal

USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
HISTDATA = r"C:/Users/panap/histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
ATR_PERIOD = 14
SL_MULT = 2.0
PX_TOL = 5e-6
R_TOL = 1e-6


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


def _precompute(df):
    """Raw-price arrays + independent ATR (shift1) for a pair df (short conventions)."""
    o_bid = df["open_bid"].to_numpy(float)
    o_ask = df["open_ask"].to_numpy(float)
    c_bid = df["close_bid"].to_numpy(float)
    c_ask = df["close_ask"].to_numpy(float)
    high_ask = df["high_ask"].to_numpy(float)
    low_ask = df["low_ask"].to_numpy(float)
    high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    raw_atr = _independent_wilder_atr(high_mid, low_mid, close_mid, ATR_PERIOD)
    atr_s1 = np.empty(len(df)); atr_s1[0] = np.nan; atr_s1[1:] = raw_atr[:-1]  # shift1
    return {"o_bid": o_bid, "o_ask": o_ask, "c_bid": c_bid, "c_ask": c_ask,
            "high_ask": high_ask, "low_ask": low_ask, "atr_s1": atr_s1}


def _collect(eval_, panels, cfg):
    """Run me_short under committed config; dedup legs across overlapping folds.

    Returns (unique_legs, structural_check_dict). Multi-leg structural reconstruction
    is done WITHIN each fold (position_id is unique only within a fold's account).
    """
    runner = ArcFoldRunner(A1Architecture(), eval_, panels)
    folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    uniq = {}
    struct = {"positions": 0, "with_partial": 0, "closed_two_leg": 0, "ok_size_sum": 0,
              "runner_open_at_end": 0, "open_at_end_total": 0, "bad": []}
    for fold in folds:
        runner(fold, cfg)
        rr = runner.last_result.run_result
        trades = rr.closed_trades
        struct["open_at_end_total"] += rr.n_open_at_end
        # within-fold leg dedup into the global unique set (audit each leg once)
        for t in trades:
            key = (t.pair, t.entry_time, round(t.entry_price, 6), t.exit_time,
                   t.exit_reason, round(t.size, 6))
            uniq[key] = t
        # within-fold structural reconstruction (position_id unique here)
        by_pos = defaultdict(list)
        for t in trades:
            by_pos[t.position_id].append(t)
        for pos_id, legs in by_pos.items():
            struct["positions"] += 1
            partials = [lg for lg in legs if lg.exit_reason == "partial_close_1r"]
            runners = [lg for lg in legs if lg.exit_reason != "partial_close_1r"]
            if not partials:
                continue
            struct["with_partial"] += 1
            linked = all(lg.parent_position_id == pos_id for lg in legs)
            if len(partials) == 1 and len(runners) == 1:
                # normal CLOSED partial+runner pair → no size leakage (equal halves)
                struct["closed_two_leg"] += 1
                half_ok = abs(partials[0].size - runners[0].size) <= 1e-6 * max(partials[0].size, 1.0)
                struct["ok_size_sum"] += int(half_ok and linked)
                if not (half_ok and linked):
                    struct["bad"].append((fold.oos_start.year, pos_id, "size/link",
                                          [(lg.exit_reason, lg.size) for lg in legs]))
            elif len(partials) == 1 and not runners and rr.n_open_at_end > 0:
                # partial fired, runner STILL OPEN at fold end (engine does not
                # force-close; reported via n_open_at_end). Legitimate boundary
                # truncation, NOT size leakage — the partial leg itself is audited.
                struct["runner_open_at_end"] += int(linked)
                if not linked:
                    struct["bad"].append((fold.oos_start.year, pos_id, "open-runner-unlinked",
                                          [(lg.exit_reason, lg.size) for lg in legs]))
            else:
                struct["bad"].append((fold.oos_start.year, pos_id, "unexpected-leg-shape",
                                      [(lg.exit_reason, lg.size) for lg in legs]))
    return list(uniq.values()), struct


def _audit(name, trades, panel, pairs):
    arr = {p: _precompute(panel.pair_dfs[p]) for p in pairs}
    idx = {p: panel.pair_dfs[p].index for p in pairs}
    n = len(trades)
    c = dict(entry=0, sl=0, ttl=0, stopbar=0, exitpx=0, pnl=0)
    viol = {k: [] for k in ["entry", "sl", "ttl", "stopbar", "exitpx", "pnl"]}
    R_by_reason = {}
    for t in trades:
        assert t.direction is Direction.SHORT, f"{name} expected short"
        a, ix = arr[t.pair], idx[t.pair]
        e = ix.get_loc(t.entry_time); xi = ix.get_loc(t.exit_time); sig = e - 1
        r_atr = SL_MULT * a["atr_s1"][sig]            # = sl_atr_mult * atr_at_entry

        # 1. ENTRY == open_bid[entry] (short sells the bid)
        if abs(t.entry_price - a["o_bid"][e]) <= PX_TOL: c["entry"] += 1
        else: viol["entry"].append((t.pair, t.entry_time, t.entry_price, a["o_bid"][e]))

        # 2. SL == close_bid[sig] + 2*indepATR[sig]
        sl_indep = a["c_bid"][sig] + SL_MULT * a["atr_s1"][sig]
        if t.sl_price is not None and abs(t.sl_price - sl_indep) <= PX_TOL: c["sl"] += 1
        else: viol["sl"].append((t.pair, t.entry_time, t.sl_price, sl_indep))
        sl = t.sl_price

        # 3. TAKE-THE-LOSS (Arc-10): no bar strictly between entry & exit breaches (high_ask>=sl)
        breach = next((b for b in range(e + 1, xi) if a["high_ask"][b] >= sl), None)
        if breach is None: c["ttl"] += 1
        else: viol["ttl"].append((t.pair, t.entry_time, t.exit_time, t.exit_reason,
                                  str(ix[breach]), float(a["high_ask"][breach]), float(sl)))

        # 4. STOP-BAR consistency per exit reason
        if t.exit_reason == "stop_loss":
            sb = (a["high_ask"][xi] >= sl) and abs(t.exit_price - sl) <= PX_TOL
            ref = float(a["high_ask"][xi])
        elif t.exit_reason == "partial_close_1r":
            # intra-bar trigger low_ask<=tp1 AND SL-first precedence (high_ask<sl this bar)
            tp1 = t.entry_price - r_atr
            sb = (a["low_ask"][xi] <= tp1 + PX_TOL) and (a["high_ask"][xi] < sl)
            ref = float(a["low_ask"][xi])
        else:  # runner_trail_stop / market: next-bar-open buy-back fill above-stop-safe
            sb = a["o_ask"][xi] < sl
            ref = float(a["o_ask"][xi])
        if sb: c["stopbar"] += 1
        else: viol["stopbar"].append((t.pair, t.entry_time, t.exit_time, t.exit_reason,
                                      ref, float(sl), t.exit_price))

        # 5. EXIT PX per reason
        if t.exit_reason == "stop_loss":
            eref = sl
        elif t.exit_reason == "partial_close_1r":
            eref = t.entry_price - r_atr            # tp1 = entry - r_atr
        else:
            eref = a["o_ask"][xi]                   # runner_trail / market → next-bar open_ask
        if abs(t.exit_price - eref) <= PX_TOL: c["exitpx"] += 1
        else: viol["exitpx"].append((t.pair, t.entry_time, t.exit_reason, t.exit_price, float(eref)))

        # 6. PNL == sign*(exit-entry)*size  (short sign = -1)
        pnl_indep = t.direction.sign * (t.exit_price - t.entry_price) * t.size
        if abs(pnl_indep - t.pnl) <= max(R_TOL * abs(t.pnl), 1e-4): c["pnl"] += 1
        else: viol["pnl"].append((t.pair, t.entry_time, t.pnl, pnl_indep))

        risk_px = sl - a["c_bid"][sig]              # = 2*ATR[sig], the short R unit
        if risk_px > 0:
            R_by_reason.setdefault(t.exit_reason, []).append(
                t.direction.sign * (t.exit_price - t.entry_price) / risk_px)

    def pc(x): return f"{x}/{n} ({100*x/n:.1f}%)"
    print(f"\n{'='*82}\nOUTCOME-LAYER AUDIT — {name} ({n} legs)  "
          f"exit mix: {dict(Counter(t.exit_reason for t in trades))}\n{'='*82}")
    print(f"  1. ENTRY  == open_bid[entry]                       : {pc(c['entry'])}")
    print(f"  2. SL     == close_bid[sig] + 2*indepATR[sig]      : {pc(c['sl'])}")
    print(f"  3. TAKE-THE-LOSS (no missed earlier stop)          : {pc(c['ttl'])}   <-- Arc-10 test")
    print(f"  4. STOP-BAR consistency (per-reason)               : {pc(c['stopbar'])}")
    print(f"  5. EXIT PX (stop->sl / partial->tp1 / trail->o_ask): {pc(c['exitpx'])}")
    print(f"  6. PNL == sign*(exit-entry)*size                   : {pc(c['pnl'])}")
    for reason, rs in sorted(R_by_reason.items()):
        rs = np.array(rs)
        print(f"    R[{reason:18s}]: {rs.mean():+.3f} / {rs.min():+.3f} / {rs.max():+.3f}  (n={len(rs)})")
    clean = all(not v for v in viol.values())
    for k, v in viol.items():
        if v:
            print(f"  !! {k} VIOLATIONS ({len(v)}): {v[:6]}")
    print(f"  -> {name}: {'ALL CHECKS PASS' if clean else 'VIOLATIONS — investigate'}")
    return clean


def main():  # pragma: no cover - audit driver
    print("loading D1 USD-major panel + running me_short (committed partial-runner-trail config)...")
    d1 = Panel.from_pairs(USD, tf="D1", histdata_root=HISTDATA, cache_root=CACHE,
                          use_cache=True, boundary_convention="5ers_eet")
    ev = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    cfg = A1Config(config_id="arc_1019", sl_atr_mult=SL_MULT, trail_enabled=False,
                   exit_policy="sl_partial_close_1r_runner_trail")
    assert cfg.trail_enabled is False, "committed me_short config: native trail OFF (short-deferred)"

    legs, struct = _collect(ev, {"D1": d1}, cfg)
    print(f"unique legs across IS folds 2011-2020: {len(legs)}")

    ok = _audit("me_short (1019, D1 USD, partial-runner-trail)", legs, d1, USD)

    print(f"\n{'='*82}\nMULTI-LEG STRUCTURAL CHECK (partial-close reconstruction, within-fold)\n{'='*82}")
    wp = struct["with_partial"]
    closed = struct["closed_two_leg"]
    openrun = struct["runner_open_at_end"]
    print(f"  positions (within-fold): {struct['positions']}  |  with a +1R partial: {wp}")
    print(f"  CLOSED partial+runner pairs (1 partial + 1 runner, linked, equal halves): "
          f"{struct['ok_size_sum']}/{closed}")
    print(f"  partial fired, runner OPEN at fold end (n_open_at_end-backed, linked): "
          f"{openrun}  (total open-at-end across folds: {struct['open_at_end_total']})")
    print(f"  accounted: {struct['ok_size_sum'] + openrun}/{wp} partial-bearing positions")
    struct_ok = (wp > 0 and (struct["ok_size_sum"] + openrun) == wp and not struct["bad"])
    if struct["bad"]:
        print(f"  !! STRUCTURAL ANOMALIES ({len(struct['bad'])}): {struct['bad'][:6]}")
    print(f"  -> structural: {'CLEAN' if struct_ok else 'ANOMALIES — investigate'}")

    print(f"\n{'='*82}")
    print("VERDICT:", "me_short OUTCOME layer independently verified honest (6/6 + structural)"
          if (ok and struct_ok) else "VIOLATIONS FOUND — investigate above")
    print("=" * 82)


if __name__ == "__main__":  # pragma: no cover
    main()
