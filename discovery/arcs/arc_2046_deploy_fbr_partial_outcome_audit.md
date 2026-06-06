# arc 2046 — §11 INDEPENDENT OUTCOME audit of the honest-DEPLOY fbr exit (partial-runner/SL1.5)

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **DIAGNOSTIC → KILL** (no new
component; fbr stays PORTFOLIO; a verification, not an edge) · **Council:** none (a measurement /
Arc-10-defense check)

---

## (a) Log read / synthesis

Pulled main; STOP absent; highest 2000s arc = 2045 (mine), resume at 2046. State (honest-era, FRESH
EYES — pre-reset lists ignored):

- **Edge-hunt is structurally closed.** Closed ground = shallow single-condition directional (long
  AND short) on liquid FX, all TFs H1/H4/D1/W1 (3014); triangulation fully closed level+variance
  (3005/1027/2023/2028/1031); the explore-now MENU exhausted (M1/O1/L1/Q1/G1/S1); the 2018-leg hunt
  dead across ~18 routes (structure 1014/2009/2011/3011/1035, trend 3010, flow 1016/2013, vol 3012,
  rel-value 2010/2018, continuation 2012, carry 1017, shock 3019); fbr improvement closed on ≥6
  entry axes (2014/2017/2020/2029/2030/2031/1040/1041/3013/3020); option-B (thick fold-resolving
  standalone) closed on 4 constructions (1025/2018/1026/2024). Path-B (denser book) closed by
  portfolio math (3021); leg-hunt structurally closed (2019/2022).
- **4 PORTFOLIO components:** gap (1006), me_long (1011), fbr (1013, strongest/only fold-resolving),
  me_short (1019, first robust +2018). 4-way book mean-positive but never all-folds-positive
  (combination-invariant 2015 & 2018 wall).
- **The §5f exit-honesty thread (2040→2045, +1042/1043/1044/1045)** is the major recent development:
  the committed component headlines used FULL-SAMPLE-best exits (Arc-10-class exit-fishing). Under
  honest nested-WFO exit selection, **gap & me_short flip mean-NEGATIVE, fbr −40% (stays +), only
  me_long is exit-robust.** ⇒ the **honest deploy object is the 2-leg {me_long + fbr}** (the other 2
  net-subtract); ~2× Sharpe, recovers borderline significance (nested t=2.12, CI excludes 0), but
  STILL not AFP (2014 + 2015↔2018) and deployment geometry unchanged (Calmar ~0.245, ~99%
  underwater, prop-firm T_min ~4yr = vehicle-infeasible). Lever = operator path-A gate-call.
- **§11 independent verification is COMPLETE for the COMMITTED book** (signal 2034/2035/1037,
  outcome 2036/2037/2038/1038/1039, cost 2039).

**The owed item I act on.** arc-2040 **FLAG F2** + arc-2045 both flagged a live gap: the §11 outcome
audits (2036/1038) verified fbr's **committed** exit (`sl_plus_trailing_atr`/SL2.0/`trail_enabled=True`
double-trail), but the §5f-identified **DEPLOY** fbr exit is a DIFFERENT config —
`sl_partial_close_1r_runner_trail`/SL1.5 (arc 2045's `freeze_best_over_folds` afp/worst pick, IS
+1.477%). So **arc 2045's 2-leg deployment-geometry profile rests on an UN-audited fbr outcome
layer** — a live Arc-10-defense gap on the actual deploy object. (OOS book combination is OFF-LIMITS
— arc 2022/1045 classify it an operator §5g-firewall decision; conservative bias §8 governs.)

## (b) Idea (the *because*)

The partial-runner is the EXACT mechanism the retired Arc-10 fast-replay flattered (a same-bar
+1R-partial suppression that let a runner survive a stop touch — RESET_MANIFEST), so independently
re-deriving it from raw price is the sharpest possible §11 outcome check. fbr is a **LONG H4**
component; the only prior partial-runner outcome audits (2038/1039) were the **SHORT D1** me_short →
a **LONG partial-runner outcome layer has NEVER been independently re-derived**, and it is precisely
the layer the deploy object (2045) depends on. The highest-value in-scope contribution is therefore
to discharge the owed audit: re-derive every committed deploy-fbr POSITION's entry/SL/legs/R/cost
from raw OHLC with INDEPENDENT code (no import of the exit policy or backtester) and confirm it
matches the engine's claim.

## (c)/(f) Method

100% canonical ground-truth ledger (A1Architecture → MultiPairBacktester, FundedNext, risk 0.005)
over the full IS span, the EXACT deploy config: `FailedBreakdownReclaimLongSignal(K=40,
shadow=1.25)`, H4, 7 USD majors, `exit_policy="sl_partial_close_1r_runner_trail"`, **`sl_atr_mult=1.5`**,
`trail_enabled=False`. Built **BUILT** `independent_outcome_audit_fbr_partial.py` — the LONG mirror
of arc-1039's SHORT union-walk, with the partial-runner geometry re-coded fresh from the documented
conventions (read, not imported, from `core/sim/exit_policies/sl_partial_close_1r_runner_trail.py`):

- entry == next-bar `open_ask` (long); SL == `close_ask[sig] − 1.5·ATR_indep` (BELOW), ATR via
  arc-2034's independent Wilder loop; `r_atr = 1.5·ATR_indep`, `tp1 = entry + r_atr`.
- per-bar order over the **MULTI-PAIR UNION index** (arc-1039 lesson — the engine iterates the union;
  a queued close on a union ts where the pair has no bar is DROPPED, deferring the runner): 1a
  fill/DROP pending close at `open_bid` → 2a intra-bar SL FIRST `low_bid≤sl` (take-the-loss; full
  pre-tp1, 0.5 runner post-tp1) → 2b +1R partial `high_bid≥tp1` (0.5 @ tp1) → 3b at-close peak
  ratchet `max(high_bid)` + runner trail `close_bid ≤ peak−r_atr` strictly after the tp1 bar (queue
  `runner_trail_stop`, fills next present union bar `open_bid`).
- 6 checks per position + cost: entry, SL, leg-count, every leg (time/px/reason/50-50 size), per-leg
  `final_r = +(exit−entry)/(entry−sl)`, engine pnl == `+(exit−entry)·size`, FundedNext per-position
  cost (commission + slippage n_fills=3-on-partial-else-2 + size-weighted spread).

Reads ONLY the trusted `Panel.from_pairs` loader + the engine `ClosedTrade` ledger / `apply_cost_model`
breakdown (the CLAIMS); never the exit-policy or backtester code. Geometry/verification only — no
gate, no score, no OOS. Driver = the BUILT tool's `__main__`.

## Results

**Engine ledger (deploy config):** 331 legs across **210 positions** (121 two-leg partial+runner, 89
single-stop). (Note: the deploy partial-runner/SL1.5 produces a DIFFERENT trade ledger than the
committed trailing-atr/SL2.0 fbr — same fires, different exits/sizing — so this is genuinely the
un-audited variant, not a re-run of 2036/1038.)

**ALL CHECKS PASS — byte-identical, every position:**

| check | result |
|---|---|
| entry == next-bar `open_ask` (long) | **210/210** |
| SL == `close_ask[sig] − 1.5·ATR_indep` | **210/210** |
| leg COUNT matches | **210/210** |
| every leg (time / px / reason / 50-50 size) | **210/210** |
| per-leg `final_r` match | **210/210** |
| engine pnl == `+(exit−entry)·size` | **210/210** |
| FundedNext cost (n_fills=3 on partial) | **210/210** |

**Take-the-loss / Arc-10-defect signature directly observed.** Hand-audit USDJPY pos 3 (entry
2011-02-22 02:00 @ 83.090, SL 82.78791): the +1R partial fired @ 83.38209 (**+0.967R**) and then the
**runner closed at EXACTLY sl_price 82.78791 = −1.0000R** — the partial did NOT flatter the runner past
the stop touch; the runner took the full loss honestly. This is the precise mechanism the retired
replay suppressed, re-derived here from raw price and matching the engine.

**Honesty clusters** (hand-audit, 6 positions): `partial_close_1r` legs ~**+0.97 to +1.03R**
(slightly off exactly +1R because the policy's `r_atr = 1.5·ATR` reference differs from the realised
R-unit `entry − sl` by the reclaim-bar→next-open entry gap + half-spread — the honest entry-slippage
signature; unlike me_short's ~exact +1.000R because fbr has a meaningful reclaim→open gap);
`runner_trail_stop` winners convex right tail to **+5.38R** (AUDUSD pos 8); single-leg / runner stops
land at exactly **−1.0000R** at `sl_price`. Engine and independent walk agree on all of them.

## (h)/(g) Verdict — DIAGNOSTIC → KILL (no new component; fbr UNCHANGED)

**The honest-DEPLOY fbr exit (`sl_partial_close_1r_runner_trail`/SL1.5) OUTCOME layer is independently
CONFIRMED HONEST** — 210/210 positions / 331 legs re-derive byte-identical from raw price on all
seven dimensions; no geometry bug, no price-lookahead, no missed-stop (Arc-10) defect, honest
take-the-loss, honest LONG partial/runner multi-leg accounting, honest cost netting. **Discharges
arc-2040 FLAG F2 and arc-2045's owed re-derivation:** the actual deploy object (2-leg {me_long+fbr},
which uses THIS fbr exit, not the committed double-trail) now rests on a fully §11-verified fbr
outcome layer, closing the last bespoke-code Arc-10 exposure on the deployable book.

**Scope (honest, as 2036/1038/2038/1039):** audits the OUTCOME of executed trades (the Arc-10-prone
bespoke geometry); per-trade R is size-invariant → fully independent; trade identities + `size` come
from the engine ledger (signal-verified 2034/2035 + heavily-tested sizing). me_long's honest §5f exit
== its committed exit (`sl_only`/2-bar, arc 2042 — exit-robust, registry-inert at 2-bar) and is
already §11 outcome-audited (2037/2039), so **with this arc the 2-leg deploy object {me_long+fbr} is
§11-complete end-to-end** (signal + outcome + cost) at its DEPLOY configs, not just the committed ones.

**Components UNCHANGED** (fbr stays PORTFOLIO; me_long stays PORTFOLIO; the 2-leg book is a subset,
still not AFP → no survivor). Lever unchanged = operator path-A gate-governance call + the arc-2033
vehicle reality. No canonical change, no FLAG (the deploy exit's outcome layer is CORRECT), no
council, **no OOS**.

## (i) NEW lesson

"§11 outcome-verified" is **config-specific**: an exit-honesty correction (§5f) that re-selects a
thin book's deployable exit (here fbr committed `sl_plus_trailing_atr`/SL2.0 → deploy
`sl_partial_close_1r_runner_trail`/SL1.5) silently moves the deploy object OFF the variant the §11
audit covered — so the verification debt re-opens on the NEW config even though "fbr's outcome layer
was audited." Re-derive the outcome layer at the DEPLOY config, not the committed one, before leaning
on a deployment-geometry profile (arc 2045) that rests on it. Mechanically: a LONG partial-runner
outcome audit is the mirror of arc-1039's SHORT walk (swap entry `open_bid→open_ask`, SL
`+→−`/`close_bid→close_ask`, partial trigger `low_ask≤entry−R → high_bid≥entry+R`, trough→peak
ratchet, runner fill `open_ask→open_bid`) — and it must STILL iterate the multi-pair union index
(the dropped-queued-close behavior is direction-invariant). The deploy fbr's −1R stops land exactly
at `sl_price` even after a +1R partial fired (the Arc-10 defect would have flattered the runner past
that touch) — re-confirmed from raw price for the LONG partial-runner.

## (k) Re-orient

Detail persisted (this doc + DISCOVERY_LOG both-tier append, committed + pushed). Tool: **BUILT**
`discovery/tools/independent_outcome_audit_fbr_partial.py` (registered). Reused canonical
A1Architecture/MultiPairBacktester/`apply_cost_model` + `Panel.from_pairs` + arc-2034
`_independent_wilder_atr`; adapted arc-1039's union-walk to LONG. No canonical change, no council, no
OOS. The honest-exit thread + the §11 programme are now complete on the DEPLOY object; the edge-hunt
is structurally closed and the deployability lever is the operator's path-A call + the invariant
arc-2033 vehicle wall. Next: resume at arc 2047 (or graceful handoff if context low).
