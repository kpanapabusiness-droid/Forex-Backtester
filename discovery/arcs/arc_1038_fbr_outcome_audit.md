# arc 1038 — §11 INDEPENDENT OUTCOME-LAYER verification of the `fbr` component

**Chat:** 1000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; `fbr` UNCHANGED, PORTFOLIO)
**Disposition:** KILL · **passed:** N · **Component touched:** none (§11 Arc-10-defense verification of `fbr`'s OUTCOME layer)

> Arc 2034 independently re-derived `fbr`'s SIGNAL (fire set + no-lookahead) and explicitly DEFERRED the
> OUTCOME layer — "the per-trade R / cost / SL-honest trailing exit is engine-trusted but not yet
> INDEPENDENTLY re-derived → flagged as the next §11 step (re-derive sample trade-R from raw price under
> the exit policy)". Arc 2035 repeated the deferral for the other 3 legs. **This arc is that next step for
> the load-bearing component** (`fbr` 1013, the book's heaviest leg, arc 2033). The OUTCOME layer
> (SL-first take-the-loss + the `sl_plus_trailing_atr` trailing exit + FundedNext cost netting) is the
> EXACT code class that produced Arc 10. **Result: all 210 committed `fbr` trades re-derive BYTE-IDENTICAL
> from raw price on EVERY outcome dimension — entry, SL, exit_time/reason, exit_price, final_r, gross-P&L
> consistency, AND FundedNext cost — using independent code that imports neither the exit policy nor the
> backtester loop. `fbr`'s outcome layer is independently CONFIRMED HONEST.** The take-the-loss invariant
> is directly observed (stop fills land exactly at sl_price = −1.0000R). No canonical change, no FLAG.

---

## Log reading (step a — FRESH EYES, honest-era only)

Pulled main; no `discovery/STOP`. Resumed 1000s after arc 1037 (highest in-range → 1038). Read protocol,
full Tier-1 ledger, LESSONS, TOOL_REGISTRY.

Converged corpus state (~60 honest-era arcs across 3 chats): the shallow single-condition directional
space is CLOSED ground (momentum/breakout/mean-reversion/trend, long AND short, H1/H4/D1/W1, majors +
crosses, capture + drift lenses, stop-removed). **4 net-positive PORTFOLIO components survive:** gap-fill
1006 (JPY-cross H4), month-end-long `me_long` 1011 (USD-major D1), failed-breakdown-reclaim `fbr` 1013
(USD-major H4 — the crown jewel, +1.854% IS, 9/10), month-end-short `me_short` 1019 (USD-major D1, the
first robustly-+2018 leg). The 4-way book is a sound ~3-independent-bet (ENB 3.32, arc 2019), mean-positive
(t=2.66, P(mean<0)=0.004, arc 1023), temporally robust (2021) and cost-robust (3022) PORTFOLIO whose
all-folds-positive FAILURE is purely the per-year gate sitting **below the legs' noise floor** (2016/2017).

**The edge-hunt is closed on every documented lever** — path-B densification is provably closed (3021),
the explore-now MENU is exhausted (1027/2023/2028/1030/1031), every short construction for the 2018 leg is
dead (1014/2009/2011/3010/3012/1016/2013/1035), all `fbr` entry-quality refinements collapse (2014/2020/
1025/3013/3020/2029/2030). The sole deployability lever is the operator's **path-A (gate-governance call)**,
now fully quantified (gate-coarsening 1032, risk-geometry 1033, prop-firm feasibility 2033, vol-target
1036). The autonomous edge-hunt has nothing left to add to deployability.

**So the highest-value autonomous work has shifted to the §11 Arc-10 INSTITUTIONAL DEFENSE.** §11: "No
candidate is deployed on the gate engine's word alone — that single-engine trust is exactly what produced
Arc 10. Before real capital, a passer's numbers must be re-verified via a GENUINELY INDEPENDENT path: a
second implementation, OR a hand-audit of a representative sample of its trades against raw price (entry,
exit, R, cost) confirming they match the engine's claim." Arcs 2034/2035/1037 began this — but ONLY for
the SIGNAL layer (the fire set + no-lookahead), and BOTH explicitly deferred the OUTCOME layer as "the next
§11 step, owed for all 4 components before deployment." **That next step is open. It is the most
Arc-10-relevant verification possible** (Arc 10 was an OUTCOME-layer defect: a replay that skipped
pre-+1R-partial stops), so it is this arc.

## Idea + why (the named next §11 step — re-derive trade-R from raw price)

"Reproduces exactly via the canonical apparatus" — true of ~14 arcs for `fbr` — is NOT independent
verification: it is the same engine code (`FailedBreakdownReclaimLongSignal` → `ArcFoldRunner` →
`MultiPairBacktester`) agreeing with itself. A bug in the OUTCOME path (the SL-first take-the-loss, the
`sl_plus_trailing_atr` trailing geometry, the FundedNext cost netting) would reproduce identically forever —
the Arc-10 trap, now at the outcome level. The genuine §11 check: take each committed `fbr` trade and
RE-DERIVE its realised outcome from RAW OHLC with INDEPENDENT code, confirming entry/exit/R/cost match the
engine. `fbr` is the load-bearing leg (arc 2033 — the book's heaviest), so it is the highest-value target.

## Method (BUILT `independent_outcome_audit.py`; no canonical exit/engine code imported)

1. **Ground-truth ledger (canonical, as §11 intends — audit the engine's claim).** Ran the canonical
   `fbr` gate (`A1Architecture` → `MultiPairBacktester`, config exactly as `validate_4way_book.py`:
   `FailedBreakdownReclaimLongSignal(K=40, shadow=1.25)`, `A1Config(sl_atr_mult=2.0, trail_enabled=False,
   exit_policy="sl_plus_trailing_atr", risk_pct=0.005)`) over a full-span 2010-2020 fold → **210 closed
   `fbr` trades** (the engine's per-trade ledger: entry/exit time+price, size, pnl, sl_price, exit_reason),
   plus `apply_cost_model(FundedNext)`'s per-position cost breakdown.
2. **Independent re-derivation** (a fresh module — imports ONLY the trusted DATA loader `Panel.from_pairs`,
   the engine LEDGER it audits, and arc-2034's already-proven independent Wilder ATR; NO import of
   `SlPlusTrailingAtrPolicy` or the backtester loop). For EVERY one of the 210 trades:
   - **entry** — re-derive next-bar `open_ask`; confirm == engine `entry_price`.
   - **SL** — re-derive `close_ask[fire] − 2·ATR_indep[fire]` (ATR = arc-2034's independent loop, shift1);
     confirm == engine `sl_price` (independently re-confirms BOTH the SL geometry and the ATR).
   - **exit** — a fresh bar-by-bar walk replicating the engine's per-bar order: intra-bar SL FIRST
     (`low_bid ≤ sl` → fill at `sl_price`, take-the-loss), only a survivor offered to the +1R-activated
     1R-below-peak trailing close at bar CLOSE, a queued trail close fills at the NEXT bar `open_bid` (so
     the next bar's SL never pre-empts it — closes fill before exit checks). Confirm exit_time /
     exit_price / exit_reason == engine.
   - **R** — `final_r = (exit − entry)/(entry − sl)`; confirm == engine, AND `engine.pnl ==
     sign·(exit − entry)·size` (gross-P&L consistency).
   - **cost** — FundedNext commission (`size/lot × $5`) + slippage (`pip × 0.5 × n_fills=2 × size`, single
     leg never partials) + spread (`0.5·(entry_spread + exit_spread)·size`) re-derived from raw price;
     confirm == `apply_cost_model`'s per-position breakdown.

## Result — ALL 210 trades, ALL 7 outcome checks: BYTE-IDENTICAL

```
trades audited: 210
entry == next-bar open_ask            : 210/210
SL == close_ask[fire] - 2*ATR_indep   : 210/210
exit_time + exit_reason match         : 210/210
exit_price match                      : 210/210
final_r match                         : 210/210
engine pnl == sign*(exit-entry)*size  : 210/210
FundedNext cost (comm+slip+spread)    : 210/210
VERDICT: PASS — fbr OUTCOME layer independently re-derives from raw price
```

Hand-audit samples (geometry visible against raw price):
- **USDJPY 2011-01-03** entry @81.350 (SL 80.865) → trailing_stop_atr exit @83.050, **R=+3.5043** — a
  big trail-out winner; independent walk lands the exact next-bar `open_bid` fill bar + price.
- **USDJPY 2011-02-22** entry @83.090 (SL 82.69054) → **stop_loss exit @82.69054, R=−1.0000** — the
  **take-the-loss invariant directly observed**: the SL fills at exactly `sl_price` (independent SL
  re-derivation 82.69054 == engine), giving precisely −1R.
- **GBPUSD 2011-02-25** R=+1.1702, **NZDUSD 2011-03-09** R=+0.3731 — trailing exits across pairs, costs
  match (commission scales with size: USDJPY size→$0.05, GBPUSD→$2.45, NZDUSD→$4.40 RT).

## Interpretation + diagnosis (§2 — interrogate the clean pass)

A 210/210 clean pass is the EXPECTED outcome for an honest engine — but it was NOT guaranteed and is NOT
vacuous: the independent walk re-implements, from scratch, the precise sequence that Arc 10's retired
replay got WRONG (intra-bar SL evaluated BEFORE the trailing logic; the stop wins on a same-bar breach;
the hard SL frozen while the trail rides at close). Had the engine's outcome path carried an Arc-10-class
shortcut (e.g. surviving a same-bar stop touch, or trailing off the wrong reference side), the −1R stops
and the trail-out exit prices would have DIVERGED from this raw-price walk. They do not — on all 210, to
price tolerance 1e-7 and final_r tolerance 1e-6, with exit timestamps and reasons exact. The FundedNext
cost — the other half of the "net of costs" claim every PORTFOLIO disposition rests on — also re-derives
exactly per position (commission/slippage/spread), independently of `apply_cost_model`.

**Scope boundary (stated honestly, as 2034/2035 did for the signal layer):** this audits the OUTCOME of
each EXECUTED trade. The trade IDENTITIES (which fires become trades after the exposure cap, and each
trade's `size` from `LiveBalanceRisk`) come from the engine ledger — the entry/sizing/cap layer is the
canonical pool + the already-independently-verified signal (2034) + the simple, heavily-tested sizing
model, not the Arc-10-prone outcome geometry. The per-trade R is size-INVARIANT, so the R/exit
verification is fully independent of size; size enters only the (linear, formulaic) cost and the gross-P&L
consistency check, both of which pass. Re-deriving `size` itself would require replaying the compounding
account balance = re-rolling the engine, which is out of scope for "re-derive trade-R."

## Verdict + disposition

DIAGNOSTIC → **KILL** (no new component; not an edge arc). `fbr`'s OUTCOME layer is independently
CONFIRMED HONEST — the FIRST genuine §11 outcome-layer check in the programme, completing for `fbr` what
2034 did for its signal. `fbr` UNCHANGED (PORTFOLIO). No canonical change, no FLAG (the outcome layer is
correct), no council, no OOS spent. Lever unchanged = operator path-A.

**NEW lesson:** a §11 OUTCOME-layer check re-derives each committed trade's realised R AND cost from raw
price with independent code (not importing the exit policy / backtester), replicating the engine's
intra-bar SL-FIRST-then-trail order — and `fbr`'s 210 trades pass byte-identical, with the take-the-loss
−1R stops landing exactly at sl_price. Combined with arc 2034 (signal), `fbr` is now end-to-end
independently verified (signal + outcome) — the deepest §11 coverage of any component. The OUTCOME layer
for the OTHER 3 components (gap = time-exit, no trail; `me_long` = `sl_only` + time-exit; **`me_short` =
`sl_partial_close_1r_runner_trail`** — the partial-runner, the MOST Arc-10-relevant exit since the retired
shortcut was precisely a same-bar partial suppression) is the explicit next §11 step before deployment.

## Threads for the next arc

1. **§11 OUTCOME layer for the other 3 components** (the direct continuation; mirrors how 2035 followed
   2034 on the signal side). Highest-value sub-target = **`me_short`'s `sl_partial_close_1r_runner_trail`**
   — the +1R partial + runner is the exact mechanism the retired replay flattered; independently
   re-deriving its two-leg outcome (partial leg @+1R close, runner leg trailing/SL, n_fills=3 cost) is the
   sharpest possible Arc-10 outcome check. Then gap (24-bar time-exit predicate) + `me_long` (`sl_only` +
   2-bar time-exit). Reuse `independent_outcome_audit.py`'s walk skeleton; add the short-side mirror +
   the time-exit + the partial-leg accounting.
2. With signal (2034/2035/1037) + `fbr` outcome (this arc) + the other-3 outcome (thread 1) done, the
   WHOLE book is end-to-end independently §11-verified — the institutional pre-deployment gate the
   operator's path-A decision rests on.

---

## ADDENDUM (post-merge) — convergence with concurrent arc 2036 + a config-provenance fork it surfaced

Pulled main after committing: the concurrent **2000s chat independently did the SAME fbr outcome audit
(arc 2036)** — a cross-chat Arc-10 reproduction exactly like 2034/2035↔1037. **Both chats, different code,
converge: the `fbr` OUTCOME layer is HONEST** (every committed trade's realised R re-derives from raw
price; take-the-loss exact).

The convergence surfaced a real **config fork** I must state accurately (the merge showed 2036 used a
different fbr config than I did). I followed `scripts/cosim_validation/validate_4way_book.py` →
**`trail_enabled=False`**; arc 2036 used the committed arc-1013 headline → **`trail_enabled=True`** (the
"double-trail" flagged in arcs 1015/3009/1024: the KH-24 `TrailManager` AND the `sl_plus_trailing_atr`
exit policy BOTH active). I measured both directly (same 210 entries, different exits):

| fbr config | exits | per-fold mean | folds + | note |
|---|---|---|---|---|
| `trail_enabled=True` (committed headline; arc 2036) | 89 `stop_loss` / 24 native `trailing_stop` / 97 `trailing_stop_atr` | **+1.854%** | **9/10** (only 2018 −4.20) | the byte-exact +1.854%/9-of-10 headline (2034/1037/2036) |
| `trail_enabled=False` (`validate_4way_book.py`; this arc) | 89 `stop_loss` / 121 `trailing_stop_atr` | **+2.084%** | **8/10** (2018 −4.39 **and** 2019 −0.17) | the cosim-book + deployment-char variant (1033/2033) |

**So the two §11 outcome audits are COMPLEMENTARY, not redundant: arc 2036 verified the byte-exact headline
config; this arc independently verified the cosim-book `trail_enabled=False` variant — the fbr outcome
layer is now confirmed honest under BOTH trail configurations.** My "the exact `validate_4way_book` fbr
config" (Tier-1 row) is accurate; my framing as "the committed config" was loose — the +1.854%/9-of-10
HEADLINE is `trail_enabled=True`. Corrected here.

**FLAG (documentation/provenance, NOT an engine-honesty issue — both are honestly scored; code human-gated,
not patched):** the cosim book scripts (`validate_4way_book.py` → the deployment-characterization arcs
1033/2033/`equity_risk_profile`) run fbr at **`trail_enabled=False` = +2.084%/8-of-10**, which DIFFERS from
the committed arc-1013 component headline **`trail_enabled=True` = +1.854%/9-of-10** — the documented
double-trail / "fbr within ~1.49pp" fork (1015/3009/1024/1033), now **quantified at the per-fold level:
the only sign difference is the 2019 fold (+0.05 True → −0.17 False); the 2018 binding fold is −4.20 (True)
/ −4.39 (False) either way → the book's 2018 wall is config-ROBUST (mechanism-intrinsic, arc 2014).** The
operator's deployment numbers (1033/2033) sit on the False variant while the component's stated headline is
the True variant — worth awareness when the two are cited together. (No new component; both configs
PORTFOLIO-equivalent on the binding fold.)
