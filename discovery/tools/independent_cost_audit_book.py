"""§11 COST-layer independent audit of the 4-component book (arc 2039) — the LAST §11 slice.

Completes the book's independent verification. Prior §11 work covered the SIGNAL layer (2034 fbr,
2035 the other three) and the gross OUTCOME layer (2036 fbr, 2037 gap+me_long, 2038 me_short). The
only remaining layer the gate depends on is the per-trade COST netting — the gate scores
`net = gross - cost`, with `cost` computed at the canonical chokepoint
`core.runners._fold_stats_helpers.build_fold_stats_from_run` → `core.sim.costs.model.apply_cost_model`
(FundedNext profile). That cost code is test-covered (honest-sweep Part C / PR #264 RESOLVED), but §11
demands a GENUINELY INDEPENDENT re-derivation before deployment — "a hand-audit of a representative
sample of its trades against raw price (entry, exit, R, **cost**) confirming they match the engine's
claim" (DISCOVERY_PROTOCOL §11). A single trusted cost path is exactly the Arc-10 trap.

METHOD (§11 discipline). The CLAIM under audit = `apply_cost_model(run_result).breakdown` (the engine's
per-position cost ledger). The INDEPENDENT check = a fresh re-implementation of the documented FundedNext
formula, computed from RAW PRICE (the trusted `Panel.from_pairs` open bid/ask at each leg's entry/exit
bar) + the closed-trade leg sizes — NEVER importing the cost primitives (`compute_commission_usd`,
`compute_slippage_pips`, `compute_extra_spread_price`) or `apply_cost_model`. The documented FundedNext
profile (read from `costs/model.py` + the three primitives' docstrings, to know WHAT to assert, not
copied):

  - commission = $5/lot round-turn × (original_size / 100_000), on the FULL original size (not per-leg).
  - slippage   = (0.5 pip/fill × n_fills) × pip_size(pair) × original_size, where n_fills = 3 if a +1R
                 partial fired (≥2 legs) else 2 — adverse, over-applied on full size (conservative).
  - spread     = Σ_legs (entry_spread + exit_spread) × (1.5 − 1) × leg.size, with entry_spread =
                 open_ask[entry bar] − open_bid[entry bar] and exit_spread = open_ask[exit bar] −
                 open_bid[exit bar] (both floored at 0 for data gaps), re-derived from RAW PRICE. The
                 engine records exit_bid/exit_ask as the exit bar's OPEN quotes for every exit type
                 (intra-bar SL/partial via `_check_exits`/`_apply_intrabar_policy_decisions`, queued
                 trail/time via `_fill_pending_closes`), so raw open bid/ask is the correct reference —
                 re-deriving from the Panel also cross-validates the ledger's recorded bid/ask.
  - total_cost = commission + slippage + spread.

Per position (grouped by position_id within a fold, where the id is unique) the four quantities are
compared to the engine's breakdown row. pip_size = 0.01 (JPY-quoted) else 0.0001, re-implemented fresh.

EXPERIMENT tool: re-derivation + audit only; never realizes P&L for a gate, never scores, never spends
OOS. IS-only (folds 2011-2020). Covers all four book legs: gap 1006, me_long 1011, fbr 1013, me_short 1019.

Run:  PYTHONPATH=. py discovery/tools/independent_cost_audit_book.py
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.costs.model import apply_cost_model  # the CLAIM under audit (engine cost ledger)
from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    MonthEndReversionShortSignal,
)
from discovery.tools.time_exit_predicate import make_time_exit_predicate

HISTDATA = r"C:/Users/panap/histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
JPY = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
LOT = 100_000.0
FN_COMMISSION_PER_LOT_RT = 5.0     # documented FundedNext gate default
FN_SLIP_PIPS_PER_FILL = 0.5
FN_SPREAD_MULT = 1.5
COST_TOL = 1e-6                     # absolute USD-equivalent tolerance per quantity


def _pip_size(pair: str) -> float:
    """Fresh re-implementation: 0.01 for JPY-quoted, else 0.0001."""
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def _attach_time_exit(sig, panel, n_bars):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n_bars)
    return dataclasses.replace(sig, per_pair={
        pair: dataclasses.replace(st, exit_predicate=pred) for pair, st in sig.per_pair.items()})


def _open_quotes(panel, pairs):
    """Raw open_bid/open_ask arrays + index per pair (the independent spread reference)."""
    out = {}
    for p in pairs:
        df = panel.pair_dfs[p]
        out[p] = {
            "ob": df["open_bid"].to_numpy(float),
            "oa": df["open_ask"].to_numpy(float),
            "ix": df.index,
        }
    return out


def _safe_spread(oa, ob):
    s = oa - ob
    return s if (np.isfinite(s) and s > 0.0) else 0.0


def _indep_position_cost(legs, q):
    """Independently re-derive (commission, slippage, spread, total) for one position
    from RAW PRICE + the documented FundedNext formula. Never calls the cost primitives."""
    pair = legs[0].pair
    original_size = float(sum(float(lg.size) for lg in legs))
    # commission — full original lots, $5/lot RT
    commission = FN_COMMISSION_PER_LOT_RT * (original_size / LOT)
    # slippage — n_fills = 3 if a partial fired (>=2 legs) else 2
    n_fills = 3 if len(legs) >= 2 else 2
    slip_pips = FN_SLIP_PIPS_PER_FILL * n_fills
    slippage = slip_pips * _pip_size(pair) * original_size
    # spread — per-leg (entry_spread + exit_spread) * (mult-1) * leg.size, raw open quotes
    qp = q[pair]
    e = qp["ix"].get_loc(legs[0].entry_time)
    entry_spread = _safe_spread(qp["oa"][e], qp["ob"][e])
    spread = 0.0
    for lg in legs:
        xi = qp["ix"].get_loc(lg.exit_time)
        exit_spread = _safe_spread(qp["oa"][xi], qp["ob"][xi])
        spread += (entry_spread + exit_spread) * (FN_SPREAD_MULT - 1.0) * float(lg.size)
    return commission, slippage, spread, commission + slippage + spread


def _collect_runs(eval_, panels, cfg):
    """Run the committed config per fold; return [(fold_year, run_result)] over IS 2011-2020."""
    runner = ArcFoldRunner(A1Architecture(), eval_, panels)
    folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    out = []
    for fold in folds:
        runner(fold, cfg)
        out.append((fold.oos_start.year, runner.last_result.run_result))
    return out


def _audit(name, runs, panel, pairs):
    q = _open_quotes(panel, pairs)
    n_pos = 0
    c = dict(comm=0, slip=0, spr=0, tot=0)
    viol = {k: [] for k in ["comm", "slip", "spr", "tot", "bidask"]}
    eng_tot = indep_tot = 0.0
    n_partial = 0
    for yr, rr in runs:
        breakdown = apply_cost_model(rr).breakdown   # the engine CLAIM (per-position cost ledger)
        groups = {}
        for t in rr.closed_trades:
            groups.setdefault(int(t.position_id), []).append(t)
        eng_by_pos = {int(r.position_id): r for _, r in breakdown.iterrows()}
        for pos_id in sorted(groups):
            legs = sorted(groups[pos_id], key=lambda x: (x.exit_time, float(x.size)))
            if pos_id not in eng_by_pos:
                viol["tot"].append((name, yr, pos_id, "no engine breakdown row"))
                continue
            er = eng_by_pos[pos_id]
            n_pos += 1
            if len(legs) >= 2:
                n_partial += 1
            ic, isl, isp, itot = _indep_position_cost(legs, q)
            eng_tot += float(er.total_cost); indep_tot += itot
            # cross-validate the ledger's recorded entry/exit bid/ask vs RAW open quotes
            qp = q[legs[0].pair]
            e = qp["ix"].get_loc(legs[0].entry_time)
            ba_ok = (abs(float(legs[0].entry_ask) - qp["oa"][e]) <= 1e-9
                     and abs(float(legs[0].entry_bid) - qp["ob"][e]) <= 1e-9)
            for lg in legs:
                xi = qp["ix"].get_loc(lg.exit_time)
                ba_ok = ba_ok and (abs(float(lg.exit_ask) - qp["oa"][xi]) <= 1e-9
                                   and abs(float(lg.exit_bid) - qp["ob"][xi]) <= 1e-9)
            if ba_ok: pass
            else: viol["bidask"].append((name, yr, pos_id, "ledger bid/ask != raw open"))
            # per-quantity match (USD-equivalent absolute tolerance)
            def chk(key, indep, eng):
                if abs(indep - float(eng)) <= max(COST_TOL, 1e-9 * abs(float(eng))):
                    c[key] += 1
                else:
                    viol[key].append((name, yr, pos_id, float(eng), indep))
            chk("comm", ic, er.commission)
            chk("slip", isl, er.slippage)
            chk("spr", isp, er.spread)
            chk("tot", itot, er.total_cost)

    def pc(x): return f"{x}/{n_pos} ({100*x/max(n_pos,1):.1f}%)"
    print(f"\n{'='*82}\nCOST-LAYER AUDIT — {name} ({n_pos} positions, {n_partial} with +1R partial)\n{'='*82}")
    print(f"  commission == $5/lot_RT * (orig_size/100k)        : {pc(c['comm'])}")
    print(f"  slippage   == 0.5*n_fills * pip_size * orig_size  : {pc(c['slip'])}")
    print(f"  spread     == sum_legs(entry+exit spr)*0.5*size   : {pc(c['spr'])}")
    print(f"  TOTAL COST == comm + slip + spread                : {pc(c['tot'])}")
    bidask_ok = not viol["bidask"]
    print(f"  ledger entry/exit bid/ask == RAW open quotes      : "
          f"{'CLEAN' if bidask_ok else f'{len(viol[chr(98)+chr(105)+chr(100)+chr(97)+chr(115)+chr(107)])} MISMATCH'}")
    print(f"  sum engine cost = {eng_tot:,.2f}  |  sum independent cost = {indep_tot:,.2f}  "
          f"(delta = {eng_tot - indep_tot:+.6f})")
    clean = all(not v for v in viol.values())
    for k, v in viol.items():
        if v:
            print(f"  !! {k} VIOLATIONS ({len(v)}): {v[:5]}")
    print(f"  -> {name}: {'ALL COST CHECKS PASS' if clean else 'VIOLATIONS — investigate'}")
    return clean


def main():  # pragma: no cover - audit driver
    print("loading panels (H4 JPY crosses + H4/D1 USD majors)...")
    h4_jpy = Panel.from_pairs(JPY, tf="H4", histdata_root=HISTDATA, cache_root=CACHE,
                              use_cache=True, boundary_convention="5ers_eet")
    h4_usd = Panel.from_pairs(USD, tf="H4", histdata_root=HISTDATA, cache_root=CACHE,
                              use_cache=True, boundary_convention="5ers_eet")
    d1 = Panel.from_pairs(USD, tf="D1", histdata_root=HISTDATA, cache_root=CACHE,
                          use_cache=True, boundary_convention="5ers_eet")

    # committed configs (validate_4way_book.py)
    gap_eval = _attach_time_exit(
        WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy}), h4_jpy, 24)
    gap_cfg = A1Config(config_id="arc_1006", sl_atr_mult=2.0, trail_enabled=False, exit_policy=None)
    me_long_eval = _attach_time_exit(
        MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1}), d1, 2)
    me_long_cfg = A1Config(config_id="arc_1011", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_only")
    fbr_eval = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd})
    fbr_cfg = A1Config(config_id="arc_1013", sl_atr_mult=2.0, exit_policy="sl_plus_trailing_atr")  # double-trail (committed)
    me_short_eval = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    me_short_cfg = A1Config(config_id="arc_1019", sl_atr_mult=2.0, trail_enabled=False,
                            exit_policy="sl_partial_close_1r_runner_trail")

    runs = {
        "gap (1006, H4 JPY)":       (_collect_runs(gap_eval, {"H4": h4_jpy}, gap_cfg), h4_jpy, JPY),
        "me_long (1011, D1 USD)":   (_collect_runs(me_long_eval, {"D1": d1}, me_long_cfg), d1, USD),
        "fbr (1013, H4 USD)":       (_collect_runs(fbr_eval, {"H4": h4_usd}, fbr_cfg), h4_usd, USD),
        "me_short (1019, D1 USD)":  (_collect_runs(me_short_eval, {"D1": d1}, me_short_cfg), d1, USD),
    }
    results = {name: _audit(name, r, panel, pairs) for name, (r, panel, pairs) in runs.items()}

    print(f"\n{'='*82}")
    allok = all(results.values())
    print("VERDICT:", "ALL 4 book components' COST layer independently verified honest"
          if allok else "VIOLATIONS FOUND — investigate above")
    print(f"  per-component: " + ", ".join(f"{n.split()[0]}={'OK' if ok else 'FAIL'}"
                                            for n, ok in results.items()))
    print("=" * 82)


if __name__ == "__main__":  # pragma: no cover
    main()
