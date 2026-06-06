"""§11 INDEPENDENT OUTCOME-LAYER verification of the `fbr` component (arc 1038).

Arc 2034 independently re-derived `fbr`'s SIGNAL (the fire set + no-lookahead).
It explicitly deferred the OUTCOME layer — "the per-trade R / cost / SL-honest
trailing exit is engine-trusted (heavily-tested) but not yet INDEPENDENTLY
re-derived → flagged as the next §11 step (re-derive sample trade-R from raw
price under the exit policy)". Arc 2035 repeated the deferral for the other
three legs. This module is that next step for the LOAD-BEARING component
(`fbr` 1013; arc 2033 names it the book's heaviest leg).

WHY this is the highest-value §11 target: the outcome layer (SL-first
take-the-loss + the `sl_plus_trailing_atr` trailing exit + FundedNext cost
netting) is the EXACT code class that produced Arc 10 — a gate that scored on a
shortcut that skipped pre-+1R-partial stops. "Reproduces exactly via the
canonical apparatus" is NOT independent verification (the same engine code
agreeing with itself). A genuine §11 check re-derives the realised outcome from
RAW PRICE with independent code and confirms it matches the engine's claim.

What is re-derived here, INDEPENDENTLY (no import of the exit policy or the
backtester loop), for EVERY committed fbr trade in the engine's ledger:

  * entry  — entry fills at the next bar's open_ask (long); confirm == engine.
  * SL     — sl_price = (fire-bar close_ask) − sl_atr_mult·ATR, ATR via arc
             2034's independent Wilder loop; confirm == engine's sl_price.
  * exit   — a fresh bar-by-bar walk applying SL-first (low_bid ≤ sl → fill at
             sl_price, take-the-loss) THEN the +1R-activated 1R-below-peak
             trailing close (queued at bar close, fills next bar open_bid);
             confirm exit_time / exit_price / exit_reason == engine.
  * R      — final_r = (exit − entry)/(entry − sl); confirm == engine, and that
             engine.pnl == sign·(exit − entry)·size (gross P&L consistency).
  * cost   — FundedNext per-trade commission + slippage + spread re-derived from
             raw price; confirm == apply_cost_model's per-position breakdown.

SCOPE BOUNDARY (stated honestly): this audits the OUTCOME of each EXECUTED
trade. The entry/sizing/exposure-cap layer (which fires become trades) rides on
the canonical pool + the already-independently-verified signal (arc 2034) — the
ground-truth trade IDENTITIES come from the engine ledger; this module re-derives
their realised numbers. The ledger itself is produced by CALLING the canonical
apparatus (A1Architecture → MultiPairBacktester), exactly as §11 intends (audit
the engine's claim against raw price).

EXPERIMENT tool: re-derivation + comparison only; never realises P&L for a gate,
never scores a fold, never spends OOS. Only the DATA loader (`Panel.from_pairs`)
and the engine LEDGER it audits are canonical.

Run:  PYTHONPATH=. py discovery/tools/independent_outcome_audit.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.arc.signal_protocol import SignalEvaluation
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
SL_ATR_MULT = 2.0
RISK_PCT = 0.005

# FundedNext gate cost constants (re-derived independently, NOT imported)
FN_SPREAD_MULT = 1.5
FN_COMMISSION_PER_LOT_RT = 5.0
FN_SLIP_PIPS_PER_FILL = 0.5
LOT = 100_000.0
PRICE_TOL = 1e-7   # absolute price tolerance (atr/pandas-vs-loop float noise)
R_TOL = 1e-6       # final_r tolerance


def _pip_size(pair: str) -> float:
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


@dataclass
class IndepOutcome:
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    final_r: float
    commission: float
    slippage: float
    spread: float


def _indep_walk_outcome(
    df: pd.DataFrame, entry_idx: int, entry_price: float, sl_price: float, r_atr: float,
) -> IndepOutcome | None:
    """Re-derive a LONG trade's outcome from raw OHLC with INDEPENDENT code.

    Mirrors the engine per-bar order: at each bar from the entry bar onward,
    intra-bar SL is checked FIRST (take-the-loss); only a survivor is offered to
    the trailing policy at the bar CLOSE; a queued trail close fills at the NEXT
    bar's open_bid (so the next bar's SL never pre-empts it — closes fill before
    exit checks). No import of SlPlusTrailingAtrPolicy / MultiPairBacktester.
    """
    low_bid = df["low_bid"].to_numpy(float)
    high_bid = df["high_bid"].to_numpy(float)
    close_bid = df["close_bid"].to_numpy(float)
    open_bid = df["open_bid"].to_numpy(float)
    n = len(df)

    activated = False
    peak = np.nan
    for t in range(entry_idx, n):
        # 1) intra-bar SL (SL-first take-the-loss): bid touches the hard stop
        if low_bid[t] <= sl_price:
            return _finish(df, entry_idx, entry_price, sl_price, t, sl_price, "stop_loss")
        # 2) at-close trailing-atr evaluation
        if not activated:
            if high_bid[t] - entry_price >= r_atr:   # +1R MFE crossed
                activated = True
                peak = high_bid[t]
            else:
                continue
        else:
            if high_bid[t] > peak:
                peak = high_bid[t]
        trail_level = peak - r_atr
        if close_bid[t] <= trail_level:
            # queue full close -> fills at NEXT bar open_bid
            x = t + 1
            if x >= n:
                return None  # trade still open at end of data (not in ledger)
            return _finish(df, entry_idx, entry_price, sl_price, x, open_bid[x],
                           "trailing_stop_atr")
    return None  # open at end of data


def _finish(df, entry_idx, entry_price, sl_price, exit_idx, exit_price, reason) -> IndepOutcome:
    pair = str(df.attrs.get("pair", "")) or _PAIR_HINT[0]
    sl_dist = entry_price - sl_price
    final_r = (exit_price - entry_price) / sl_dist
    # FundedNext cost, single leg (fbr never partials) -> n_fills = 2
    size = _SIZE_HINT[0]
    commission = (size / LOT) * FN_COMMISSION_PER_LOT_RT
    slippage = _pip_size(pair) * (FN_SLIP_PIPS_PER_FILL * 2) * size
    entry_spread = max(0.0, df["open_ask"].to_numpy(float)[entry_idx]
                       - df["open_bid"].to_numpy(float)[entry_idx])
    exit_spread = max(0.0, df["open_ask"].to_numpy(float)[exit_idx]
                      - df["open_bid"].to_numpy(float)[exit_idx])
    spread_total = entry_spread + exit_spread
    extra_spread = spread_total * (FN_SPREAD_MULT - 1.0) if spread_total > 0 else 0.0
    spread = extra_spread * size
    return IndepOutcome(df.index[exit_idx], float(exit_price), reason, float(final_r),
                        commission, slippage, spread)


# tiny mutable hints so _finish can see the current trade's pair/size without
# threading them through every call (single-threaded audit driver)
_PAIR_HINT = [""]
_SIZE_HINT = [0.0]


def main():  # pragma: no cover - audit driver
    print("loading H4 USD-major panel (canonical loader — DATA is the trusted foundation)...")
    panel = Panel.from_pairs(USD, tf="H4", histdata_root=HISTDATA, cache_root=CACHE,
                             use_cache=True, boundary_convention="5ers_eet")
    for p in USD:
        panel.pair_dfs[p].attrs["pair"] = p

    print("running the CANONICAL fbr gate (A1Architecture -> MultiPairBacktester) over the "
          "full IS span to harvest the engine's ground-truth trade ledger...")
    sig: SignalEvaluation = FailedBreakdownReclaimLongSignal(
        swing_lookback=K, min_shadow_atr=SHADOW, atr_period=ATR_PERIOD).evaluate({"H4": panel})
    cfg = A1Config(config_id="arc_1013", sl_atr_mult=SL_ATR_MULT, trail_enabled=False,
                   exit_policy="sl_plus_trailing_atr", risk_pct=RISK_PCT)
    full_span = Fold(fold_id=1038, is_start=pd.Timestamp("2010-01-01").date(),
                     is_end=pd.Timestamp("2010-12-31").date(),
                     oos_start=pd.Timestamp("2011-01-01").date(),
                     oos_end=pd.Timestamp("2020-12-31").date())
    result = A1Architecture().run(signal_evaluation=sig, panels={"H4": panel}, fold=full_span,
                                  arch_config=cfg, config_id="arc_1013")
    engine_trades = result.run_result.closed_trades
    costed = apply_cost_model(result.run_result, CostModel.fundednext())
    cost_by_pos = {int(r.position_id): r for _, r in costed.breakdown.iterrows()}
    print(f"  engine ledger: {len(engine_trades)} closed fbr trades")

    # independent per-trade Wilder ATR (shift1) per pair — proven == engine (arc 2034)
    atr_indep = {}
    for p in USD:
        df = panel.pair_dfs[p]
        high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
        low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
        close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
        raw = _independent_wilder_atr(high_mid, low_mid, close_mid, ATR_PERIOD)
        a = np.empty(len(df)); a[0] = np.nan; a[1:] = raw[:-1]  # shift1, independent
        atr_indep[p] = a

    print("\n" + "=" * 80)
    print("INDEPENDENT OUTCOME RE-DERIVATION vs engine ledger (entry/SL/exit/R/cost)")
    print("=" * 80)
    n = ok_entry = ok_sl = ok_exit_t = ok_exit_px = ok_R = ok_pnl = ok_cost = 0
    flags = []
    hand_audit = []
    for tr in engine_trades:
        if tr.direction is not Direction.LONG:
            flags.append(f"{tr.pair} {tr.entry_time}: non-LONG fbr trade?!")
            continue
        df = panel.pair_dfs[tr.pair]
        idx = df.index.get_indexer([pd.Timestamp(tr.entry_time)])[0]
        if idx < 1:
            flags.append(f"{tr.pair} {tr.entry_time}: entry_time not located")
            continue
        n += 1
        fire = idx - 1  # signal fired the bar before the fill bar

        # (a) entry — re-derive next-bar open_ask
        indep_entry = float(df["open_ask"].to_numpy(float)[idx])
        entry_ok = abs(indep_entry - tr.entry_price) <= PRICE_TOL
        ok_entry += int(entry_ok)

        # (b) SL — re-derive close_ask[fire] - 2*ATR_indep[fire]
        close_ask_fire = float(df["close_ask"].to_numpy(float)[fire])
        a_fire = atr_indep[tr.pair][fire]
        indep_sl = close_ask_fire - SL_ATR_MULT * a_fire
        sl_ok = (tr.sl_price is not None) and abs(indep_sl - tr.sl_price) <= PRICE_TOL
        ok_sl += int(sl_ok)
        r_atr = SL_ATR_MULT * a_fire

        # (c)-(e) exit / R / cost — independent forward walk (engine entry/sl as identity)
        _PAIR_HINT[0] = tr.pair
        _SIZE_HINT[0] = float(tr.size)
        out = _indep_walk_outcome(df, idx, tr.entry_price, float(tr.sl_price), r_atr)
        if out is None:
            flags.append(f"{tr.pair} {tr.entry_time}: independent walk found no close "
                         f"(engine reason={tr.exit_reason})")
            continue

        exit_t_ok = pd.Timestamp(out.exit_time) == pd.Timestamp(tr.exit_time)
        exit_px_ok = abs(out.exit_price - tr.exit_price) <= PRICE_TOL
        reason_ok = (out.exit_reason == tr.exit_reason)
        ok_exit_t += int(exit_t_ok and reason_ok)
        ok_exit_px += int(exit_px_ok)

        eng_final_r = tr.direction.sign * (tr.exit_price - tr.entry_price) / abs(
            tr.entry_price - tr.sl_price)
        R_ok = abs(out.final_r - eng_final_r) <= R_TOL
        ok_R += int(R_ok)

        eng_pnl_check = tr.direction.sign * (tr.exit_price - tr.entry_price) * tr.size
        pnl_ok = abs(eng_pnl_check - tr.pnl) <= max(1e-6, abs(tr.pnl) * 1e-9)
        ok_pnl += int(pnl_ok)

        # cost vs engine breakdown for this position
        cb = cost_by_pos.get(int(tr.position_id))
        if cb is not None:
            cost_ok = (abs(out.commission - cb["commission"]) <= 1e-6
                       and abs(out.slippage - cb["slippage"]) <= 1e-6
                       and abs(out.spread - cb["spread"]) <= 1e-6)
        else:
            cost_ok = False
        ok_cost += int(cost_ok)

        if not (entry_ok and sl_ok and exit_t_ok and exit_px_ok and reason_ok and R_ok
                and pnl_ok and cost_ok):
            flags.append(
                f"{tr.pair} {tr.entry_time}: entry={entry_ok} sl={sl_ok} "
                f"exit_t/reason={exit_t_ok and reason_ok} exit_px={exit_px_ok} R={R_ok} "
                f"pnl={pnl_ok} cost={cost_ok} | indep exit {out.exit_time} {out.exit_price:.5f} "
                f"{out.exit_reason} R={out.final_r:+.4f} | eng {tr.exit_time} {tr.exit_price:.5f} "
                f"{tr.exit_reason} R={eng_final_r:+.4f}")
        if len(hand_audit) < 5:
            hand_audit.append((tr, out, indep_sl, eng_final_r))

    print(f"\n  trades audited: {n}")
    print(f"  entry == next-bar open_ask            : {ok_entry}/{n}")
    print(f"  SL == close_ask[fire] - 2*ATR_indep   : {ok_sl}/{n}")
    print(f"  exit_time + exit_reason match         : {ok_exit_t}/{n}")
    print(f"  exit_price match                      : {ok_exit_px}/{n}")
    print(f"  final_r match                         : {ok_R}/{n}")
    print(f"  engine pnl == sign*(exit-entry)*size  : {ok_pnl}/{n}")
    print(f"  FundedNext cost (comm+slip+spread)    : {ok_cost}/{n}")

    print("\n" + "=" * 80)
    print("HAND-AUDIT — sample trades re-derived vs engine (geometry visible)")
    print("=" * 80)
    for tr, out, indep_sl, eng_R in hand_audit:
        print(f"\n  {tr.pair} entry {tr.entry_time} @ {tr.entry_price:.5f} "
              f"(SL {tr.sl_price:.5f}; indep SL {indep_sl:.5f})")
        print(f"    engine : exit {tr.exit_time} @ {tr.exit_price:.5f} [{tr.exit_reason}] "
              f"R={eng_R:+.4f} pnl={tr.pnl:+.2f}")
        print(f"    indep  : exit {out.exit_time} @ {out.exit_price:.5f} [{out.exit_reason}] "
              f"R={out.final_r:+.4f}  cost(c/s/sp)={out.commission:.3f}/{out.slippage:.3f}/{out.spread:.3f}")

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    all_ok = (ok_entry == ok_sl == ok_exit_t == ok_exit_px == ok_R == ok_pnl == ok_cost == n)
    print(f"  VERDICT: {'PASS — fbr OUTCOME layer independently re-derives from raw price' if all_ok and not flags else 'MISMATCH — investigate (see flags)'}")
    if flags:
        print(f"  FLAGS ({len(flags)}):")
        for f in flags[:30]:
            print(f"    !! {f}")


if __name__ == "__main__":  # pragma: no cover
    main()
