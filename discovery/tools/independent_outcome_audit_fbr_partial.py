"""§11 INDEPENDENT OUTCOME-LAYER verification of the *honest-DEPLOY* `fbr` exit (arc 2046).

The committed `fbr` (1013) outcome layer was independently verified by arc 2036 /
1038 — but ONLY for its COMMITTED exit (`sl_plus_trailing_atr` / SL2.0 /
`trail_enabled=True` double-trail). The §5f exit-honesty thread (arcs
2040/2042/2044/2045, +1042/1043/1044/1045) then re-identified the actual deployable
object as the 2-leg {me_long + fbr}, where **fbr's honest §5f-frozen deploy exit is a
DIFFERENT config: `sl_partial_close_1r_runner_trail` / SL1.5** (arc 2045: +1.477% IS,
the freeze_best_over_folds afp/worst pick — a ~25-30% haircut on the committed
trailing-atr headline). arc-2040 FLAG F2 and arc-2045 both note this explicitly: the
§11 outcome verification audited a NON-deploy fbr variant; the deployable
partial-runner/SL1.5 fbr's outcome layer is a small **owed** §11 re-derivation.

This module discharges that owed item. It is the highest-value remaining §11 gap
because (a) the partial-runner is the EXACT mechanism the retired Arc-10 fast-replay
flattered (a same-bar +1R-partial suppression that let a runner survive a stop touch
— RESET_MANIFEST), and (b) arc 2045's deployment-geometry profile of the 2-leg deploy
object RESTS on this un-audited exit. fbr is a LONG H4 component; the only prior
partial-runner outcome audits (2038/1039) were the SHORT D1 `me_short` — so a LONG
partial-runner outcome layer has never been independently re-derived.

What is re-derived INDEPENDENTLY (no import of the exit policy or backtester loop),
for EVERY committed fbr POSITION in the engine ledger (legs grouped by position_id):

  * entry  — long fills next bar open_ask; confirm == engine.
  * SL     — sl_price = (signal-bar close_ask) − 1.5·ATR (BELOW, long); ATR via
             arc-2034's independent Wilder loop; confirm == engine sl_price.
  * legs   — a fresh bar walk over the MULTI-PAIR UNION index replicating the engine
             per-bar order: intra-bar SL (low_bid≤sl → take-the-loss) checked BEFORE
             the +1R partial (high_bid≥entry+R → close 50% @ tp1 price); the runner
             then trails (peak−R, strictly AFTER the tp1 bar, close_bid≤peak−R →
             queue full-close at next present-bar open_bid) or stops. Confirm each
             leg's exit_time / exit_price / exit_reason / size (50/50) == engine.
  * R      — per-leg final_r = +(exit − entry)/(entry − sl); confirm == engine, and
             engine.pnl == +(exit − entry)·size.
  * cost   — FundedNext per-POSITION commission + slippage (n_fills=3 when the
             partial fired, else 2) + spread; confirm == apply_cost_model.

SCOPE BOUNDARY (as 2034/2035/2036/2037/2038/1038/1039): audits the OUTCOME of executed
trades; per-trade R is size-invariant → fully independent; trade identities and the
per-leg `size` come from the engine ledger (signal-verified 2034/2035 + heavily-tested
sizing). Verification only; never realises P&L for a gate, never scores a fold, never
spends OOS. Only the DATA loader and the engine LEDGER it audits are canonical.

Run:  PYTHONPATH=. py discovery/tools/independent_outcome_audit_fbr_partial.py
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
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.independent_signal_audit import _independent_wilder_atr

USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
HISTDATA = r"C:/Users/panap/histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
K = 40
SHADOW = 1.25
ATR_PERIOD = 14
SL_ATR_MULT = 1.5   # the honest-DEPLOY fbr exit (arc 2040/2045 freeze), NOT the committed 2.0
RISK_PCT = 0.005
FN_SPREAD_MULT = 1.5
FN_COMMISSION_PER_LOT_RT = 5.0
FN_SLIP_PIPS_PER_FILL = 0.5
LOT = 100_000.0
PRICE_TOL = 1e-7
R_TOL = 1e-6


def _pip_size(pair: str) -> float:
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def _indep_fbr_partial_legs(arr, entry_u, entry_price, sl_price, r_atr):
    """Re-derive a LONG partial-runner trade's legs over the MULTI-PAIR UNION index.

    `arr` holds the position's pair OHLC reindexed onto the union of all audited
    pairs' timestamps (NaN where the pair has no bar) + a `present` mask. The book
    runs fbr on the 7-pair H4 panel, so the engine iterates the UNION index; on a
    union timestamp where this pair has NO bar the driver skips it — and a queued
    close on the prior bar is DROPPED there (`_pending_closes={}` clears regardless
    of fill), so the runner re-trails to a later bar (arc-1039 union finding).

    Returns [(union_idx, exit_price, exit_reason, size_fraction)] — 1.0 (single
    stop) or 0.5 (partial) + 0.5 (runner). Engine per-bar order replicated (LONG
    mirror of arc-1039's short walk): 1a fill/DROP a pending close at open_bid, then
    (present bars only) 2a intra-bar SL FIRST (low_bid≤sl), 2b +1R partial
    (high_bid≥entry+R), 3b at-close peak ratchet (max high_bid) + runner trail
    (close_bid≤peak−R, strictly after tp1).
    """
    hb = arr["high_bid"]; lb = arr["low_bid"]; cb = arr["close_bid"]; ob = arr["open_bid"]
    present = arr["present"]
    n = len(present)
    tp1_price = entry_price + r_atr

    legs = []
    tp1_fired = False
    tp1_ord = None
    peak = np.nan
    pending = None     # queued exit reason -> fills at next PRESENT union bar's open_bid
    ordinal = 0        # increments ONLY on present bars (engine: evaluate_at_close skips bar=None)
    for u in range(entry_u, n):
        # 1a — fill a queued close at this bar's open_bid, or DROP it if no bar (arc-1039)
        if pending is not None:
            if present[u]:
                legs.append((u, ob[u], pending, 0.5))
                return legs
            pending = None  # union ts with no bar for this pair -> close dropped, runner continues
        if not present[u]:
            continue
        # 2a — intra-bar SL FIRST (take-the-loss; full pre-tp1, runner half post-tp1)
        if lb[u] <= sl_price:
            legs.append((u, sl_price, "stop_loss", 0.5 if tp1_fired else 1.0))
            return legs
        # 2b — intra-bar +1R partial (pre-tp1 only)
        if not tp1_fired and hb[u] >= tp1_price:
            tp1_fired = True
            tp1_ord = ordinal
            legs.append((u, tp1_price, "partial_close_1r", 0.5))
        # 3b — at close: ratchet peak (from entry), runner trail strictly after tp1
        peak = hb[u] if np.isnan(peak) else max(peak, hb[u])
        if tp1_fired and tp1_ord is not None and ordinal > tp1_ord and not np.isnan(peak):
            if cb[u] <= peak - r_atr:
                pending = "runner_trail_stop"  # fills next present union bar
        ordinal += 1
    return legs  # open at end of data


def main():  # pragma: no cover - audit driver
    print("loading H4 USD-major panel (canonical loader — DATA is the trusted foundation)...")
    panel = Panel.from_pairs(USD, tf="H4", histdata_root=HISTDATA, cache_root=CACHE,
                             use_cache=True, boundary_convention="5ers_eet")
    print("running the CANONICAL honest-deploy fbr gate (A1 -> MultiPairBacktester) "
          "full-span IS ledger: sl_partial_close_1r_runner_trail / SL1.5 ...")
    sig = FailedBreakdownReclaimLongSignal(
        swing_lookback=K, min_shadow_atr=SHADOW, atr_period=ATR_PERIOD).evaluate({"H4": panel})
    cfg = A1Config(config_id="arc_2046", sl_atr_mult=SL_ATR_MULT, trail_enabled=False,
                   exit_policy="sl_partial_close_1r_runner_trail", risk_pct=RISK_PCT)
    full_span = Fold(fold_id=2046, is_start=pd.Timestamp("2010-01-01").date(),
                     is_end=pd.Timestamp("2010-12-31").date(),
                     oos_start=pd.Timestamp("2011-01-01").date(),
                     oos_end=pd.Timestamp("2020-12-31").date())
    result = A1Architecture().run(signal_evaluation=sig, panels={"H4": panel}, fold=full_span,
                                  arch_config=cfg, config_id="arc_2046")
    trades = result.run_result.closed_trades
    costed = apply_cost_model(result.run_result, CostModel.fundednext())
    cost_by_pos = {int(r.position_id): r for _, r in costed.breakdown.iterrows()}

    by_pos = defaultdict(list)
    for tr in trades:
        by_pos[int(tr.position_id)].append(tr)
    print(f"  engine ledger: {len(trades)} legs across {len(by_pos)} positions")

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

    # MULTI-PAIR UNION index (engine iterates the union; the walk must too — arc 1039)
    union_index = pd.DatetimeIndex(sorted(set().union(*[set(panel.pair_dfs[p].index) for p in USD])))
    union_pos = {ts: i for i, ts in enumerate(union_index)}
    union_arr = {}
    for p in USD:
        r = panel.pair_dfs[p].reindex(union_index)
        cbid = r["close_bid"].to_numpy(float)
        union_arr[p] = {
            "high_bid": r["high_bid"].to_numpy(float),
            "low_bid": r["low_bid"].to_numpy(float),
            "close_bid": cbid,
            "open_bid": r["open_bid"].to_numpy(float),
            "present": ~np.isnan(cbid),
        }

    print("\n" + "=" * 80)
    print("INDEPENDENT OUTCOME RE-DERIVATION vs engine ledger (LONG partial-runner, SL1.5)")
    print("=" * 80)
    ok_entry = ok_sl = ok_legcount = ok_legs = ok_R = ok_pnl = ok_cost = 0
    n = 0
    flags = []
    hand = []
    n_partial = n_single = 0
    for pos_id, legs in by_pos.items():
        legs = sorted(legs, key=lambda x: (pd.Timestamp(x.exit_time), float(x.size)))
        head = legs[0]
        if head.direction is not Direction.LONG:
            flags.append(f"pos {pos_id}: non-LONG fbr leg?!")
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

        # (a) entry == next-bar open_ask (long)
        indep_entry = float(df["open_ask"].to_numpy(float)[idx])
        entry_ok = abs(indep_entry - entry_price) <= PRICE_TOL
        ok_entry += int(entry_ok)
        # (b) SL == close_ask[sig] - 1.5*ATR_indep[sig]
        close_ask_sig = float(df["close_ask"].to_numpy(float)[sig_i])
        a_sig = atr_indep[pair][sig_i]
        indep_sl = close_ask_sig - SL_ATR_MULT * a_sig
        sl_ok = abs(indep_sl - sl_price) <= PRICE_TOL
        ok_sl += int(sl_ok)
        r_atr = SL_ATR_MULT * a_sig

        # (c) independent leg sequence — walked over the MULTI-PAIR UNION index
        entry_u = union_pos[pd.Timestamp(head.entry_time)]
        indep_legs = _indep_fbr_partial_legs(union_arr[pair], entry_u, entry_price, sl_price, r_atr)
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
                indep_R = 1.0 * (xpx - entry_price) / (entry_price - sl_price)
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
    print(f"  entry == next-bar open_ask (long)     : {ok_entry}/{n}")
    print(f"  SL == close_ask[sig] - 1.5*ATR_indep  : {ok_sl}/{n}")
    print(f"  leg COUNT matches                     : {ok_legcount}/{n}")
    print(f"  every leg (time/px/reason/50-50 size) : {ok_legs}/{n}")
    print(f"  per-leg final_r match                 : {ok_R}/{n}")
    print(f"  engine pnl == sign*(exit-entry)*size  : {ok_pnl}/{n}")
    print(f"  FundedNext cost (n_fills=3 partial)   : {ok_cost}/{n}")

    print("\n" + "=" * 80)
    print("HAND-AUDIT — sample two-leg (partial + runner) positions vs engine")
    print("=" * 80)
    for pos_id, pair, head, legs, indep_legs, indep_sl in hand:
        print(f"\n  {pair} pos {pos_id} entry {head.entry_time} @ {head.entry_price:.5f} LONG "
              f"(SL {head.sl_price:.5f}; indep {indep_sl:.5f})")
        for eng_leg in legs:
            eng_R = eng_leg.direction.sign * (eng_leg.exit_price - head.entry_price) / abs(
                head.entry_price - head.sl_price)
            print(f"    engine leg: exit {eng_leg.exit_time} @ {eng_leg.exit_price:.5f} "
                  f"[{eng_leg.exit_reason}] size={eng_leg.size:.0f} R={eng_R:+.4f}")
        for xi, xpx, xr, xf in indep_legs:
            iR = 1.0 * (xpx - head.entry_price) / (head.entry_price - head.sl_price)
            print(f"    indep  leg: exit @ {xpx:.5f} [{xr}] frac={xf} R={iR:+.4f}")

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    all_ok = (ok_entry == ok_sl == ok_legcount == ok_legs == ok_R == ok_pnl == ok_cost == n)
    print(f"  VERDICT: {'PASS — honest-deploy fbr (partial-runner/SL1.5) OUTCOME layer independently re-derives from raw price' if all_ok and not flags else 'MISMATCH — investigate (see flags)'}")
    if flags:
        print(f"  FLAGS ({len(flags)}):")
        for f in flags[:30]:
            print(f"    !! {f}")


if __name__ == "__main__":  # pragma: no cover
    main()
