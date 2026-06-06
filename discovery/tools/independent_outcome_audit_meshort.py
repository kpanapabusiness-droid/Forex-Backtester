"""§11 INDEPENDENT OUTCOME-LAYER verification of the `me_short` component (arc 1039).

Completes the book's outcome-layer §11 audit: fbr (arc 2036 headline-config +
arc 1038 cosim-config + cost), gap + me_long (arc 2037) are done; this is the
remaining leg — `me_short` 1019, the month-end reversion SHORT under the
**`sl_partial_close_1r_runner_trail`** exit. That partial-runner is the MOST
Arc-10-relevant exit in the book: the retired fast-replay defect that reset the
repo was precisely a SAME-BAR partial suppression that let the runner survive a
stop touch and flattered realised R (RESET_MANIFEST). So independently
re-deriving `me_short`'s two-leg outcome from raw price — partial @+1R, runner
SL/trail, take-the-loss precedence, SHORT-side geometry, n_fills=3 cost — is the
sharpest possible outcome check, and it exercises the short side (entry open_bid,
exit open_ask, SL high_ask≥sl) the long-only audits never touched.

What is re-derived INDEPENDENTLY (no import of the exit policy or backtester),
for EVERY committed `me_short` POSITION in the engine ledger (legs grouped by
position_id):

  * entry  — short fills next bar open_bid; confirm == engine.
  * SL     — sl_price = (signal-bar close_bid) + 2·ATR (ABOVE, short); ATR via
             arc-2034's independent Wilder loop; confirm == engine sl_price.
  * legs   — a fresh bar walk replicating the engine per-bar order: intra-bar SL
             (high_ask≥sl → −1R, take-the-loss) checked BEFORE the +1R partial
             (low_ask≤entry−R → close 50% @ tp1 price); the runner then trails
             (trough+R, strictly AFTER the tp1 bar) or stops. Confirm each leg's
             exit_time / exit_price / exit_reason / size (50/50 split) == engine.
  * R      — per-leg final_r = −(exit − entry)/(sl − entry); confirm == engine,
             and engine.pnl == −(exit − entry)·size.
  * cost   — FundedNext per-POSITION commission + slippage (n_fills=3 when the
             partial fired, else 2) + spread; confirm == apply_cost_model.

SCOPE BOUNDARY (as 2034/2035/2036/2037/1038): audits the OUTCOME of executed
trades; per-trade R is size-invariant → fully independent; trade identities and
the per-leg `size` come from the engine ledger (signal-verified 2034/2035 +
heavily-tested sizing). Verification only; never realises P&L for a gate, never
scores a fold, never spends OOS.

Run:  PYTHONPATH=. py discovery/tools/independent_outcome_audit_meshort.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.sim.account import Direction
from core.sim.costs.model import CostModel, apply_cost_model
from core.sim.panel import Panel
from core.wfo.folds import Fold
from discovery.tools.independent_signal_audit import _independent_wilder_atr
from discovery.tools.month_end_signals import MonthEndReversionShortSignal

USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
HISTDATA = r"C:/Users/panap/histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
SL_ATR_MULT = 2.0
ATR_PERIOD = 14
RISK_PCT = 0.005
FN_SPREAD_MULT = 1.5
FN_COMMISSION_PER_LOT_RT = 5.0
FN_SLIP_PIPS_PER_FILL = 0.5
LOT = 100_000.0
PRICE_TOL = 1e-7
R_TOL = 1e-6


def _pip_size(pair: str) -> float:
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def _indep_meshort_legs(arr, entry_u, entry_price, sl_price, r_atr):
    """Re-derive a SHORT partial-runner trade's legs over the MULTI-PAIR UNION index.

    `arr` holds the position's pair OHLC reindexed onto the union of all audited
    pairs' timestamps (NaN where the pair has no bar) + a `present` mask. The book
    runs me_short on the 7-pair D1 panel, so the engine iterates the UNION index;
    on a union timestamp where this pair has NO bar the driver (`_check_exits` /
    `evaluate_at_close` / `_fill_pending_closes`) skips it — and crucially a queued
    close on the prior bar is DROPPED there (`_pending_closes={}` clears regardless
    of fill), so the runner re-trails to a later (worse) bar. Modelling the union
    iteration is what makes a multi-pair run re-derive exactly.

    Returns [(union_idx, exit_price, exit_reason, size_fraction)] — 1.0 (single
    stop) or 0.5 (partial) + 0.5 (runner). Engine per-bar order replicated: 1a
    fill/DROP a pending close, then (present bars only) 2a intra-bar SL FIRST,
    2b +1R partial, 3b at-close ratchet + runner trail (strictly after tp1).
    """
    ha = arr["high_ask"]; la = arr["low_ask"]; ca = arr["close_ask"]; oa = arr["open_ask"]
    present = arr["present"]
    n = len(present)
    tp1_price = entry_price - r_atr

    legs = []
    tp1_fired = False
    tp1_ord = None
    trough = np.nan
    pending = None     # queued exit reason -> fills at next PRESENT union bar's open_ask
    ordinal = 0        # increments ONLY on present bars (engine: evaluate_at_close skips bar=None)
    for u in range(entry_u, n):
        # 1a — fill a queued close at this bar's open, or DROP it if no bar (line 221)
        if pending is not None:
            if present[u]:
                legs.append((u, oa[u], pending, 0.5))
                return legs
            pending = None  # union ts with no bar for this pair -> close dropped, runner continues
        if not present[u]:
            continue
        # 2a — intra-bar SL (take-the-loss; full pre-tp1, runner post-tp1)
        if ha[u] >= sl_price:
            legs.append((u, sl_price, "stop_loss", 0.5 if tp1_fired else 1.0))
            return legs
        # 2b — intra-bar +1R partial (pre-tp1 only)
        if not tp1_fired and la[u] <= tp1_price:
            tp1_fired = True
            tp1_ord = ordinal
            legs.append((u, tp1_price, "partial_close_1r", 0.5))
        # 3b — at close: ratchet trough (from entry), runner trail strictly after tp1
        trough = la[u] if np.isnan(trough) else min(trough, la[u])
        if tp1_fired and tp1_ord is not None and ordinal > tp1_ord and not np.isnan(trough):
            if ca[u] >= trough + r_atr:
                pending = "runner_trail_stop"  # fills next present union bar
        ordinal += 1
    return legs  # open at end of data


def main():  # pragma: no cover - audit driver
    print("loading D1 USD-major panel (canonical loader — DATA is the trusted foundation)...")
    panel = Panel.from_pairs(USD, tf="D1", histdata_root=HISTDATA, cache_root=CACHE,
                             use_cache=True, boundary_convention="5ers_eet")
    print("running the CANONICAL me_short gate (A1 -> MultiPairBacktester) full-span IS ledger...")
    sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": panel})
    cfg = A1Config(config_id="arc_1019", sl_atr_mult=SL_ATR_MULT, trail_enabled=False,
                   exit_policy="sl_partial_close_1r_runner_trail", risk_pct=RISK_PCT)
    full_span = Fold(fold_id=1039, is_start=pd.Timestamp("2010-01-01").date(),
                     is_end=pd.Timestamp("2010-12-31").date(),
                     oos_start=pd.Timestamp("2011-01-01").date(),
                     oos_end=pd.Timestamp("2020-12-31").date())
    result = A1Architecture().run(signal_evaluation=sig, panels={"D1": panel}, fold=full_span,
                                  arch_config=cfg, config_id="arc_1019")
    trades = result.run_result.closed_trades
    costed = apply_cost_model(result.run_result, CostModel.fundednext())
    cost_by_pos = {int(r.position_id): r for _, r in costed.breakdown.iterrows()}

    # group legs by position_id
    by_pos = defaultdict(list)
    for tr in trades:
        by_pos[int(tr.position_id)].append(tr)
    n_legs_total = len(trades)
    n_pos = len(by_pos)
    print(f"  engine ledger: {n_legs_total} legs across {n_pos} positions")

    # independent Wilder ATR (shift1) per pair (NATIVE index — for SL re-derivation)
    atr_indep = {}
    for p in USD:
        df = panel.pair_dfs[p]
        hm = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
        lm = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
        cm = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
        raw = _independent_wilder_atr(hm, lm, cm, ATR_PERIOD)
        a = np.empty(len(df)); a[0] = np.nan; a[1:] = raw[:-1]
        atr_indep[p] = a

    # MULTI-PAIR UNION index (the engine iterates the union; the walk must too —
    # a union ts where a pair has no bar drops that pair's queued close, arc-1039
    # AUDUSD finding). Per-pair OHLC reindexed onto the union + a present mask.
    union_index = pd.DatetimeIndex(sorted(set().union(*[set(panel.pair_dfs[p].index) for p in USD])))
    union_pos = {ts: i for i, ts in enumerate(union_index)}
    union_arr = {}
    for p in USD:
        r = panel.pair_dfs[p].reindex(union_index)
        ca = r["close_ask"].to_numpy(float)
        union_arr[p] = {
            "high_ask": r["high_ask"].to_numpy(float),
            "low_ask": r["low_ask"].to_numpy(float),
            "close_ask": ca,
            "open_ask": r["open_ask"].to_numpy(float),
            "present": ~np.isnan(ca),
        }

    print("\n" + "=" * 80)
    print("INDEPENDENT OUTCOME RE-DERIVATION vs engine ledger (SHORT partial-runner)")
    print("=" * 80)
    ok_entry = ok_sl = ok_legcount = ok_legs = ok_R = ok_pnl = ok_cost = 0
    n = 0
    flags = []
    hand = []
    n_partial = n_single = 0
    for pos_id, legs in by_pos.items():
        legs = sorted(legs, key=lambda x: (pd.Timestamp(x.exit_time), float(x.size)))
        head = legs[0]
        if head.direction is not Direction.SHORT:
            flags.append(f"pos {pos_id}: non-SHORT me_short leg?!")
            continue
        n += 1
        pair = head.pair
        df = panel.pair_dfs[pair]
        idx = df.index.get_indexer([pd.Timestamp(head.entry_time)])[0]
        if idx < 1:
            flags.append(f"pos {pos_id} {head.entry_time}: entry not located")
            continue
        sig_i = idx - 1
        entry_price = head.entry_price
        sl_price = float(head.sl_price)

        # (a) entry == next-bar open_bid (short)
        indep_entry = float(df["open_bid"].to_numpy(float)[idx])
        entry_ok = abs(indep_entry - entry_price) <= PRICE_TOL
        ok_entry += int(entry_ok)
        # (b) SL == close_bid[sig] + 2*ATR_indep[sig]
        close_bid_sig = float(df["close_bid"].to_numpy(float)[sig_i])
        a_sig = atr_indep[pair][sig_i]
        indep_sl = close_bid_sig + SL_ATR_MULT * a_sig
        sl_ok = abs(indep_sl - sl_price) <= PRICE_TOL
        ok_sl += int(sl_ok)
        r_atr = SL_ATR_MULT * a_sig

        # (c) independent leg sequence — walked over the MULTI-PAIR UNION index
        entry_u = union_pos[pd.Timestamp(head.entry_time)]
        indep_legs = _indep_meshort_legs(union_arr[pair], entry_u, entry_price, sl_price, r_atr)
        total_size = float(sum(float(l.size) for l in legs))

        legcount_ok = (len(indep_legs) == len(legs))
        ok_legcount += int(legcount_ok)
        if len(legs) >= 2:
            n_partial += 1
        else:
            n_single += 1

        legs_match = legcount_ok
        R_match = True
        pnl_match = True
        if legcount_ok:
            for eng_leg, (xi, xpx, xreason, xfrac) in zip(legs, indep_legs):
                exit_t_ok = pd.Timestamp(union_index[xi]) == pd.Timestamp(eng_leg.exit_time)
                exit_px_ok = abs(xpx - eng_leg.exit_price) <= PRICE_TOL
                reason_ok = (xreason == eng_leg.exit_reason)
                size_ok = abs(xfrac * total_size - float(eng_leg.size)) <= max(
                    1e-3, total_size * 1e-6)
                if not (exit_t_ok and exit_px_ok and reason_ok and size_ok):
                    legs_match = False
                eng_R = eng_leg.direction.sign * (eng_leg.exit_price - entry_price) / abs(
                    entry_price - sl_price)
                indep_R = -1.0 * (xpx - entry_price) / (sl_price - entry_price)
                if abs(indep_R - eng_R) > R_TOL:
                    R_match = False
                eng_pnl_check = eng_leg.direction.sign * (eng_leg.exit_price - entry_price) * eng_leg.size
                if abs(eng_pnl_check - eng_leg.pnl) > max(1e-6, abs(eng_leg.pnl) * 1e-9):
                    pnl_match = False
        ok_legs += int(legs_match)
        ok_R += int(R_match)
        ok_pnl += int(pnl_match)

        # (d) FundedNext position cost (n_fills=3 if partial else 2)
        partial_fired = len(legs) >= 2
        n_fills = 3 if partial_fired else 2
        commission = (total_size / LOT) * FN_COMMISSION_PER_LOT_RT
        slippage = _pip_size(pair) * (FN_SLIP_PIPS_PER_FILL * n_fills) * total_size
        oa = df["open_ask"].to_numpy(float); ob = df["open_bid"].to_numpy(float)
        entry_spread = max(0.0, oa[idx] - ob[idx])
        spread = 0.0
        for eng_leg in legs:
            xi = df.index.get_indexer([pd.Timestamp(eng_leg.exit_time)])[0]
            exit_spread = max(0.0, oa[xi] - ob[xi])
            tot = entry_spread + exit_spread
            spread += (tot * (FN_SPREAD_MULT - 1.0) if tot > 0 else 0.0) * float(eng_leg.size)
        cb = cost_by_pos.get(pos_id)
        cost_ok = cb is not None and abs(commission - cb["commission"]) <= 1e-6 \
            and abs(slippage - cb["slippage"]) <= 1e-6 and abs(spread - cb["spread"]) <= 1e-6
        ok_cost += int(cost_ok)

        if not (entry_ok and sl_ok and legcount_ok and legs_match and R_match and pnl_match
                and cost_ok):
            flags.append(f"pos {pos_id} {pair} {head.entry_time}: entry={entry_ok} sl={sl_ok} "
                         f"legcount={legcount_ok}({len(indep_legs)}v{len(legs)}) legs={legs_match} "
                         f"R={R_match} pnl={pnl_match} cost={cost_ok}")
        if len(hand) < 6 and len(legs) >= 2:
            hand.append((pos_id, pair, head, legs, indep_legs, indep_sl))

    print(f"\n  positions audited: {n}  ({n_partial} two-leg partial+runner, {n_single} single stop)")
    print(f"  entry == next-bar open_bid (short)    : {ok_entry}/{n}")
    print(f"  SL == close_bid[sig] + 2*ATR_indep    : {ok_sl}/{n}")
    print(f"  leg COUNT matches                     : {ok_legcount}/{n}")
    print(f"  every leg (time/px/reason/50-50 size) : {ok_legs}/{n}")
    print(f"  per-leg final_r match                 : {ok_R}/{n}")
    print(f"  engine pnl == sign*(exit-entry)*size  : {ok_pnl}/{n}")
    print(f"  FundedNext cost (n_fills=3 partial)   : {ok_cost}/{n}")

    print("\n" + "=" * 80)
    print("HAND-AUDIT — sample two-leg (partial + runner) positions vs engine")
    print("=" * 80)
    for pos_id, pair, head, legs, indep_legs, indep_sl in hand:
        print(f"\n  {pair} pos {pos_id} entry {head.entry_time} @ {head.entry_price:.5f} SHORT "
              f"(SL {head.sl_price:.5f}; indep {indep_sl:.5f})")
        for eng_leg in legs:
            eng_R = eng_leg.direction.sign * (eng_leg.exit_price - head.entry_price) / abs(
                head.entry_price - head.sl_price)
            print(f"    engine leg: exit {eng_leg.exit_time} @ {eng_leg.exit_price:.5f} "
                  f"[{eng_leg.exit_reason}] size={eng_leg.size:.0f} R={eng_R:+.4f}")
        for xi, xpx, xr, xf in indep_legs:
            iR = -1.0 * (xpx - head.entry_price) / (head.sl_price - head.entry_price)
            print(f"    indep  leg: exit @ {xpx:.5f} [{xr}] frac={xf} R={iR:+.4f}")

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    all_ok = (ok_entry == ok_sl == ok_legcount == ok_legs == ok_R == ok_pnl == ok_cost == n)
    print(f"  VERDICT: {'PASS — me_short partial-runner OUTCOME layer independently re-derives from raw price' if all_ok and not flags else 'MISMATCH — investigate (see flags)'}")
    if flags:
        print(f"  FLAGS ({len(flags)}):")
        for f in flags[:30]:
            print(f"    !! {f}")


if __name__ == "__main__":  # pragma: no cover
    main()
