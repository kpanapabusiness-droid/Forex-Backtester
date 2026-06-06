# arc 1039 — §11 INDEPENDENT OUTCOME-LAYER verification of `me_short` (completes the book)

**Chat:** 1000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; `me_short` UNCHANGED, PORTFOLIO)
**Disposition:** KILL · **passed:** N · **Component touched:** none (§11 Arc-10-defense verification of `me_short`'s OUTCOME layer)

> The remaining §11 outcome slice after fbr (arcs 2036 headline-config + 1038 cosim-config + cost) and
> gap + me_long (arc 2037): `me_short` 1019 under **`sl_partial_close_1r_runner_trail`** — the MOST
> Arc-10-relevant exit (the retired fast-replay defect that reset the repo was precisely a same-bar
> +1R-partial suppression that let the runner survive a stop touch), and the only SHORT leg. **Result:
> all 91 committed `me_short` positions (146 legs; 55 two-leg partial+runner, 36 single stops) re-derive
> BYTE-IDENTICAL from raw price on all 7 outcome dims — entry, SL, leg count, every leg
> (time/price/reason/50-50 size), per-leg final_r, gross P&L, AND FundedNext cost (n_fills=3 on partials)
> — with independent code importing neither the exit policy nor the backtester.** The +1R partial books
> ~+1R, single stops are −1R take-the-loss exactly, and the runner trails honestly. **With fbr + gap +
> me_long, the WHOLE 4-component book's OUTCOME layer is now independently §11-verified.** A genuine
> finding en route: a multi-pair book's per-position outcome is NOT a single-pair walk — the engine's
> UNION-index iteration can DROP a queued close on a timestamp where that pair has no bar, deferring the
> runner exit to a later (here WORSE / conservative) bar; modelling the union index was required to
> re-derive exactly. No canonical change; one soft documentation FLAG.

---

## Log reading (step a — continued from arc 1038, same chat)

State (post-1038 pull): the edge-hunt is closed; the deployability lever is operator path-A; the
highest-value autonomous work is the §11 Arc-10 institutional defense. Signal layer done (2034 fbr,
2035/1037 the other 3). Outcome layer: fbr done (arc 2036 committed `trail_enabled=True` headline + my
arc 1038 `trail_enabled=False` cosim variant + the FundedNext cost re-derivation 2036/2037 deferred);
gap + me_long done (arc 2037). **The one remaining outcome slice both 2036 and 2037 explicitly flagged:
`me_short` 1019's `sl_partial_close_1r_runner_trail`** — multi-leg (the +1R partial + runner), SHORT-side
(entry open_bid / exit open_ask / SL high_ask≥sl), n_fills=3 cost. It is the sharpest possible Arc-10
outcome check (the retired replay's exact flatter was a same-bar partial suppression) and completes the
book. STOP absent. This arc.

## Method (BUILT `independent_outcome_audit_meshort.py`; no exit-policy/engine code imported)

1. **Ground-truth ledger:** ran the canonical `me_short` gate (`MonthEndReversionShortSignal(1.0, into=2)`
   D1 USD majors; `A1Config(sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_partial_close_1r_runner_trail",
   risk_pct=0.005)` — exactly `validate_4way_book.py`'s me_short) over a full-span 2010-2020 fold →
   **91 positions / 146 legs** (the 2-per-USD exposure cap throttles the ~116-fire pool to 91 executed),
   plus `apply_cost_model(FundedNext)`'s per-position breakdown.
2. **Independent re-derivation** (fresh module; only the trusted loader + arc-2034's proven Wilder ATR +
   the engine ledger). For each POSITION (legs grouped by `position_id`): re-derive entry (next-bar
   open_bid), SL (close_bid[sig] + 2·ATR_indep, ABOVE for short), r_atr = 2·ATR; then a fresh bar walk
   replicating the engine per-bar order — **1a fill/drop any queued close → 2a intra-bar SL (high_ask≥sl,
   take-the-loss, FIRST) → 2b +1R partial (low_ask≤entry−R → close 50% at the +1R level) → 3b at-close
   ratchet the trough + runner trail (close_ask≥trough+R, strictly AFTER the tp1 bar; queued → next-bar
   open_ask)**. Compared each leg's exit_time / exit_price / exit_reason / size (the 50/50 split),
   per-leg final_r = −(exit−entry)/(sl−entry), gross pnl, and the FundedNext position cost (n_fills=3 when
   the partial fired).

## Result — ALL 91 positions / 146 legs, ALL 7 dims: BYTE-IDENTICAL

```
positions audited: 91  (55 two-leg partial+runner, 36 single stop)
entry == next-bar open_bid (short)    : 91/91
SL == close_bid[sig] + 2*ATR_indep    : 91/91
leg COUNT matches                     : 91/91
every leg (time/px/reason/50-50 size) : 91/91
per-leg final_r match                 : 91/91
engine pnl == sign*(exit-entry)*size  : 91/91
FundedNext cost (n_fills=3 partial)   : 91/91
VERDICT: PASS — me_short partial-runner OUTCOME layer independently re-derives from raw price
```

Hand-audit (the partial books ~+1R, runner trails; SHORT geometry visible):
- **EURUSD 2011-01-02** SHORT @1.33450 (SL 1.36556): partial 01-05 @1.30614 R=+0.9131 (50%), runner
  01-13 @1.33650 R=−0.0644 (50%) — a winner-then-giveback runner.
- **NZDUSD 2011-01-02** SHORT: partial @0.75762 R=+0.8888, runner trail @0.77280 R=+0.1547.
- **36 single-stop positions** = −1R exact (high_ask touches sl → fill at sl_price, take-the-loss), the
  short-side analog of fbr's −1.0000R stops (arc 1038).

## The multi-pair UNION-INDEX finding (§2 — the audit surfaced a real engine subtlety)

The first run flagged ONE position (AUDUSD entry 2011-02-01): my single-pair walk gave runner exit
**2011-03-20 @ 0.99610**, the engine ledger **2011-03-21 @ 1.00650**. Interrogated rather than waved away:

- The committed exit POLICY object, driven through AUDUSD's own bars, ALSO queues the trail on 03-17 to
  fill next bar 03-20 @ 0.99610 — i.e. my reimplementation matches the real policy.
- An **AUDUSD-only** engine run gives 03-20 @ 0.99610 — matching the single-pair walk exactly.
- The book runs me_short on the **7-pair D1 panel**, so `MultiPairBacktester` iterates the UNION of all
  pairs' timestamps. The union has a **2011-03-18 timestamp where ONLY USDJPY has a bar (AUDUSD absent)**.
  On that ts `snapshot[AUDUSD]=None`, so `_fill_pending_closes` skips AND `_pending_closes={}` **DROPS**
  AUDUSD's close queued on 03-17 (the dict is cleared regardless of fill); the runner stays open and
  **re-trails** on 03-20, filling 03-21 @ 1.00650.

⇒ The multi-pair engine's number (03-21 @ **1.00650**) is **WORSE** for the short than the single-pair
03-20 @ 0.99610 (R +0.197 vs +0.663) — the union-index drop is **conservative, non-flattering** (the
opposite of an Arc-10 flatter). Making the independent walk **union-index-aware** (iterate the union,
drop a pending close on a no-bar timestamp) re-derives the engine's 03-21 exit exactly → **91/91**. The
audit caught a subtle multi-pair behavior and confirmed it does not inflate realised R — exactly what §11
is for.

**Soft documentation FLAG (no code patched — code is human-gated):** `_fill_pending_closes`'s comment
"Untradable bar: drop the close silently; retry next bar" is imprecise — the `_pending_closes={}` clear
DROPS the queued close (it does not retry). For a re-firing trailing/SL exit this merely defers the exit
(conservative, self-correcting, as here); but a ONE-SHOT predicate close (e.g. a time/kijun exit) dropped
on a no-bar union timestamp would NOT re-queue and could only close later via SL or end-of-data. me_short
(trailing) is unaffected; worth a doc note for the predicate-exit components (gap/me_long use time-exit
predicates — arc 2037 verified them, but on whichever pairs share the D1 grid; the union-drop edge is
worth their attention).

## Verdict + disposition

DIAGNOSTIC → **KILL** (no new component). `me_short`'s OUTCOME layer — including the +1R partial, the
runner trail, the short-side geometry, and the n_fills=3 cost — is independently CONFIRMED HONEST. **With
fbr (2036/1038), gap + me_long (2037), and `me_short` (this arc), the ENTIRE 4-component book's OUTCOME
layer is now independently §11-verified**, complementing the signal-layer verification (2034/2035/1037).
The book is end-to-end §11-covered — the institutional pre-deployment gate the operator's path-A decision
rests on. `me_short` UNCHANGED (PORTFOLIO). No canonical change; one soft documentation FLAG; no council;
no OOS spent.

**NEW lesson:** a §11 OUTCOME audit of a MULTI-PAIR book leg must iterate the UNION index — a position's
realised exit is NOT a single-pair walk, because the engine drops a queued close on any union timestamp
where that pair has no bar (`_pending_closes={}`), deferring a trailing exit to a later bar (here a WORSE,
conservative price). A single-pair re-derivation false-flags those positions; the union-aware walk
re-derives them exactly. The drop is conservative (non-flattering) — the §11 check confirms the multi-pair
machinery does not inflate R. (Generalizes the no-lookahead/honesty invariants to the multi-pair
execution layer.)

## Threads for the next arc

1. The §11 OUTCOME layer is now complete for all 4 components (fbr 2036/1038, gap+me_long 2037, me_short
   here). Combined with the signal layer (2034/2035/1037), the whole book is end-to-end independently
   verified — the deepest Arc-10 defense the autonomous programme can build. Remaining pre-deployment §11
   work is for the operator (independent re-verification on a genuinely separate implementation / live
   reconciliation, §11) — not an autonomous-chat action.
2. The union-index drop note (above) could be cheaply checked for the predicate-exit legs (gap/me_long)
   on the multi-pair book panel — confirm no time-exit close is silently dropped on a no-bar union ts
   (arc 2037 audited them, plausibly per-pair). Low priority (those legs' arc-2037 audit passed), but the
   union-drop edge is the kind of subtlety that only the multi-pair grid exposes.
