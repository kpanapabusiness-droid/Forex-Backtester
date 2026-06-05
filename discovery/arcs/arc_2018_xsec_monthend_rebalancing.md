# arc 2018 — Cross-sectional month-end rebalancing reversion (USD-neutral)

**Chat:** 2000s | **Date:** 2026-06-05 | **Disposition:** KILL (cheap-kill at observation)

## Idea + because

Resuming behind arcs 2016/2017 (the two diagnostics that closed the thin-leg-hunt route): the 4-way
reversion book is noise-floor-capped, the AFP-calendar-year gate is operator-flagged as mismatched to
thin components, and arc 2017 named the ONLY surviving productive spec — **option (B): seek a component
thick enough that its folds RESOLVE and clear zero** (acknowledged genuinely hard, since closed ground
makes high-trade-count *directional* edges coin-flips). Leg-hunting a 5th thin component to flip a
sub-noise fold is explicitly flagged as chasing noise (arc 2016/2017), so I did NOT do that.

Instead I attacked option (B) head-on with the one mechanism in the corpus that is both *proven* and
*extendable to thickness*: **cross-sectional month-end rebalancing reversion, USD-neutral.**

- **because:** the proven `me` edge (1011 long / 1019 short, the WMR-4pm-fix forced inelastic
  rebalancing) is a per-pair **absolute** move into month-end → **USD-beta-exposed** (precisely why
  2015/2018 are its binding folds — in strong-USD years the absolute move IS the trend). A
  **cross-sectional** construction — rank currencies by their *idiosyncratic* month return vs USD, long
  the laggard / short the leader at month-end — is (a) **USD-neutral by construction** (the common USD
  move cancels across the long+short legs) → should strip the regime exposure that creates the binding
  folds, and (b) **thicker** (fires on the extremes EVERY month, both sides → ~50/yr vs me's ~11/yr) →
  targets arc-2017's "thick enough to resolve" spec.
- **genuinely untested:** arcs 2003/2010 killed relative-value but were NOT month-end-timed (the generic
  relative move is momentum/coin-flip); `me` 1011/1019 are per-pair absolute, not cross-sectional. The
  open question 2010 left implicit: *does the forced month-end flow create cross-sectional reversion that
  doesn't exist generically?* (the arc-1011 control logic: timing is load-bearing for `me`.)

## What happened — FALSIFIED at observation

Drift-lens observation (gross, no engine/cost realized), IS 2010-2020, 7 USD majors as 7 currencies vs
USD (`EURUSD/GBPUSD/AUDUSD/NZDUSD` = +ret; `USDJPY/USDCHF/USDCAD` = −ret). At each month-end: rank
currencies by month-to-date vs-USD return; market-neutral spread = forward-return(bottom-k) −
forward-return(top-k). Control: same rank on every non-month-end day. Driver
`_disco2000_work/arc2018_xsec_monthend.py`.

| cell | frac+ | spread median | ME-excess vs random | net of cost | 2018 |
|---|---|---|---|---|---|
| FWD1 top1 (2-leg) | 0.508 | +1.36bp | +0.14bp | −4.6bp (6bp) | **−14.8bp** |
| **FWD2 top1 (2-leg, me horizon)** | **0.546** | **+9.82bp** | **+14.60bp** | +3.8bp (opt 6bp) / **neg (real ~10bp)** | **−25.1bp** |
| FWD3 top1 (2-leg) | 0.500 | +0.26bp | −2.41bp | −5.7bp | +1.1bp |
| FWD2 top2 (4-leg) | 0.523 | +1.84bp (mean +11.98 = thin-tail) | +11.01bp | **−10.2bp** | **−24.8bp** |

Three decisive reads:

1. **The thesis FAILS on the binding fold.** USD-neutrality did NOT relieve 2018: it is robustly NEGATIVE
   cross-sectionally at every horizon (−14.8 / −25.1 / +1.1 / −20.5 bp). Reason (ties to arc 2017): the
   book's real 2018 obstacle is `fbr`-2018, and `me_long` actually carries a **real POSITIVE** 2018
   (+0.90%, the WMR directional rebalancing). Stripping USD beta REMOVES `me_long`'s helpful +2018
   directional contribution rather than adding one — the cross-sectional reframe is a strictly *worse*
   expression of the `me` edge for the book.
2. **Coin-flip + sub-cost (arc-2010 re-confirmed WITH month-end timing).** The month-end timing IS
   load-bearing (FWD2 ME-excess +14.6bp vs random-day ≈0 → the WMR reversion is real cross-sectionally,
   consistent with `me`), but the gross spread is ~coin-flip (frac+ 0.51-0.55) and small (~2-10bp
   median). The market-neutral construction needs ≥2 legs; the 4-leg (top2) version is net-NEGATIVE at
   every horizon (−10bp), and the 2-leg (top1) version clears only an *optimistic* 6bp cost on a single
   **knife-edge horizon** (FWD2 only; FWD1/FWD3 net-negative) — net-negative at a realistic ~10bp 2-leg
   FundedNext round-turn. This is exactly arc 2010's "doubled-cost vs a coin-flip," now shown to persist
   even when the relative move is timed to the forced-flow event.
3. **No new component either way.** The only net-marginal cell collapses to the existing edge: a
   *single-leg* "short the most-appreciated currency" (top fwd −9bp FWD2) re-introduces USD beta and is
   just `me_short` with cross-sectional selection — already a PORTFOLIO component (1019). The novel,
   decorrelated contribution requires the 2-leg USD-neutral form, which is sub-cost.

## Verdict: KILL (cheap-kill at observation)

No pool / engine / null / council spent — §5d applies (coin-flip capture-frequency entry + a structural
multi-leg cost problem already mapped by arc 2010; §5f exit-sweep does not bite on a coin-flip,
cost-dominated entry). OOS never touched. Components UNCHANGED.

## Threads / lessons

1. **The month-end WMR reversion is real cross-sectionally** (FWD2 ME-excess +14.6bp over random-day ≈0)
   — re-confirms the `me` mechanism (1011/1019) from a new angle — but it is the SAME edge, not a new
   decorrelated component; expressing it cross-sectionally adds doubled cost and removes the directional
   2018 benefit.
2. **USD-neutrality is NOT a lever to relieve the route's 2018 obstacle.** The book's binding 2018
   negativity is `fbr`-2018 (arc 2017, mechanism-intrinsic), and the `me` family is 2018-*helped* by its
   directional (USD-beta) leg; removing the beta removes the help. Stripping a regime exposure also
   strips the edge that lives on it (generalizes arc 1018's "edge & tail are the same exposure" from the
   universe axis to the cross-sectional/market-neutral axis).
3. **arc 2010 extended:** relative-value on liquid FX majors is doubled-cost-vs-coin-flip *even when
   timed to the month-end forced-flow event* — the one remaining "but what if it were event-timed?"
   sub-question is now closed. Cross-sectional / market-neutral on USD majors does not beat the
   multi-leg cost hurdle.
4. **arc-2017 option (B) attacked once, dead via this route:** the obvious way to thicken a proven thin
   edge (cross-sectional expansion of `me`) is killed by the multi-leg cost. A genuinely thick component
   that resolves folds remains unfound; the route stays operator-gated (the AFP/gate-resolution call).

## Tooling

No new BUILT tool — drift-lens arithmetic on the canonical `Panel` (vs-USD log returns + month-to-date
cumsum + forward sum), the standard month-end mask convention (arc 1005/1011). No canonical core touched.
Driver: `_disco2000_work/arc2018_xsec_monthend.py`.

**FLAGS (code not merged):** none.
