# NEEDS_ENABLEMENT — operator-gated structural queue

> **What this is.** A RANKED, ACCUMULATING queue of structures that require apparatus changes the
> discovery fleet **cannot build itself** (canonical-core changes are human-gated, protocol §9; data feeds
> and execution venues are operator/business decisions). Each entry: **what specifically must be enabled**
> (the change surface, at the granularity the [shorts probe](./SHORTS_ENABLEMENT_PROBE.md) produced),
> **why it unlocks that class**, and **what it would let the search reach**. **Nothing here is built
> automatically** — this is a proposal set for the operator to weigh.
>
> **APPEND / UPDATE across Strategist runs — do not overwrite.** Companion to the `explore-now` menu
> [`DISCOVERY_DIRECTION.md`](./DISCOVERY_DIRECTION.md) and the standing
> [`ESCALATION_apparatus_capability.md`](./ESCALATION_apparatus_capability.md).
>
> **Run log:** 2026-06-05 — first population (Strategist dispatch; generative council).
> **2026-06-06 — run 2 (Strategist re-run).** Re-ranked around ONE criterion the corpus forced into view:
> *does the unlock supply a CONTINUOUS NON-PRICE STATE (the only thing that breaks the in-charter
> conservation law) AND plausibly reach FUNDABLE size (mean fold >8%, worst >5%)?* Adds items **U**
> (different-instrument-universe), **X** (CME signed order-flow), **K** (carry on a swap-on venue), and a
> structural **short-vol note**; advances **M** and **O** with explicit fundable-plausibility calls; demotes
> **J** (COT) to a rare-event conditioner. Existing items E/D/F+V/J/C/O/M/W are NOT overwritten — see the
> run-2 block appended at the end. Read [`DISCOVERY_DIRECTION.md`](./DISCOVERY_DIRECTION.md) §0–§1 first for
> the conservation law and the short-vol unification that drive this re-rank.

---

## ⚠️ The cost-arithmetic correction that re-ranks this queue (read first)

The standing [`ESCALATION`](./ESCALATION_apparatus_capability.md) ranks **"a second simultaneous leg /
netted book"** as the #1 unlock, on the premise that internal netting lets the book "pay the spread once
per capital round-trip instead of per-trade." **The council (quant + capability-gap lenses, independently,
upheld by the adversarial cross-examiner) finds this premise OVERSTATED → largely WRONG on FundedNext:**

- Per-leg round-turn cost ≈ 3–5 bp = spread(1.5×) + slippage(0.5 pip/fill × 2) + commission($5/lot).
  **Commission and slippage are billed per leg, per lot, REGARDLESS of internal netting** — netting is a
  *book* concept, not a broker-billing concept. Internal netting saves only the **spread overlap** (~1–2 bp).
- Cost only *truly* halves when two legs collapse into **one quoted instrument** (trade EURGBP directly,
  not long-EURUSD/short-GBPUSD). **But that single cross is already in the closed directional ground** — a
  coin-flip (arc 2003/2010/2018; relative drift 2–10 bp with `corr(relstr,fwd_rel) ≈ 0`).
- Modeling a cross-instrument net that FundedNext bills separately would be a **textbook Arc-10 defect**
  (a gate that doesn't match real execution).

**Conclusion:** a netted book makes a coin-flip *cheaper, not winning*. It does **not** rescue
relative-value on liquid majors, and it re-lights **zero** of the 4 surviving forced-flow components
(they have disjoint event-timing — me_long/me_short fire on different instruments/months — so there is no
simultaneous opposing leg to net). The genuinely cheap, high-value unlock hiding behind the escalation is
**item E below (the single co-simulated equity curve)**. The queue is ranked accordingly.

---

## Ranked queue

### E — Single co-simulated portfolio EQUITY CURVE (build this FIRST) · small build · highest ROI

> **STATUS 2026-06-06 — BUILT, IN HUMAN REVIEW (gated-core PR, not yet merged).** `core/wfo/cosim_book.py`
> (`cosim_book_fold`) + CI gate tests (`tests/wfo/test_cosim_book.py`) + the honest-engine-sweep co-sim
> re-read (SAFE). Validated against the existing 4-way book (gap+me_long+fbr+me_short, arcs
> 1006/1011/1013/1019) — see [`COSIM_ITEM_E_VALIDATION.md`](./COSIM_ITEM_E_VALIDATION.md). **Result: every
> measurement (linear, co-sim cap-off, co-sim cap-on) FAILS all-folds-positive on IS → the book's failure is
> FUNDAMENTAL, not a linear-combiner artifact. The portfolio route is closed cleanly; IS did not clear so the
> 2021+ OOS was NOT touched.** Deployable-system count stays 0. Do not mark this item DONE until the PR merges.

- **What must be enabled.** A driver/gate path that marks **all components to ONE equity line** under the
  **shared 5% daily-DD cap** and the 2-per-currency exposure cap applied to the *book's* net exposure —
  replacing (or sitting beside) the per-fold-**LINEAR** combiner (`discovery/tools/combine_fold_roi.py`),
  which sums component fold-ROIs and is faithful only for disjoint-event-timing components. Reuses the
  `Account`/`Position` sign-based `parent_position_id` primitive (shorts probe Q2.6) and the canonical
  daily-DD logic; the new piece is a co-simulation loop that advances all open component positions on a
  shared clock and one equity line. All-folds-positive is then judged on that single curve. Re-pass
  determinism + the honest-engine sweep.
- **Why it unlocks.** Every portfolio diagnostic to date (arcs 2016/2017/2019/1023/2021) used the LINEAR
  combiner. It cannot see whether **shared-daily-cap interaction and real cross-component drawdown**
  change per-fold resolution — the book's *actual* risk geometry is untested. This is a **correctness fix
  to the measurement**, not a strategy, with **zero fabrication surface**: a shared cap + real DD
  interaction is strictly *harder* than linear summation, so it can only make the verdict equal or worse —
  it can never manufacture optimism.
- **What it lets the search reach.** A definitive, honest answer to the open question arc 2019 left for the
  operator: *is the 4-way book's all-folds-positive failure a linear-combiner artifact, or real?* If
  co-simulation flips ≥1 fold, the book's status changes; if it changes nothing, the thinness is
  fundamental and the route is closed cleanly. Either outcome is decision-grade for the operator's
  gate-governance call. **Cheapest information-per-build-dollar item in the entire programme.**

### D — Passive LIMIT-fill order type WITH an adverse-selection model · high leverage · ⚠️ Arc-10 trap if naive

- **What must be enabled.** A `limit_entry` order type in the engine with: (i) fill-iff-touched against
  bid/ask, (ii) an explicit **non-fill** branch (the move continues, you don't get in), (iii)
  earn-half-spread accounting, and **(iv) a conservative adverse-selection model** — limit orders are
  filled *preferentially when the trade goes against you* (you fill on the losing tail, miss on the
  winning tail).
- **Why it unlocks.** This is the only lever that attacks the COST term **on the already-real surviving
  components** rather than hunting a new edge. The cost wall is ~0.05–0.10R and the best edges live just
  inside it; the forced-flow reversions are "revert-to-a-level" entries that are *naturally* limit fills.
  If honest, flipping from spread-payer to spread-earner could move per-trade edge by ~the full spread and
  push the thin-but-mean-positive book toward fold resolution **with no new signal**.
- **⚠️ The hard precondition.** Item (iv) is the whole ballgame and the **canonical Arc-10 surface**. A
  naive limit-fill model that assumes you're filled at the level and the reversion always follows fabricates
  the exact falsely-optimistic PASS this repo was reset to purge. **Do not build (i)–(iii) without a
  conservative (iv).** If the adverse-selection model is honest, it may erase the half-spread it was meant
  to harvest — that erasure *is* the correct answer, not a failure of the build.
- **What it lets the search reach.** An honest re-score of the 4 survivors as passive entries — potentially
  the single most leveraged outcome available (resolve the existing book's folds), or a clean confirmation
  that the edge was never in the aggressive fill.

### F + V — Netted multi-leg / co-simulated BOOK engine, PAIRED with a real netting execution venue · ⚠️ downgraded; conditional

- **What must be enabled.** **(F)** a `LegGroup`/`SpreadOrder` construct (N≥2 legs with hedge ratios + a
  group_id, additive to the signal contract, default-empty so existing signals are byte-identical), a
  netted-cost path, and a co-simulated driver that opens/closes a group atomically on one equity line.
  **(V)** a **real, obtainable** execution venue with **portfolio margining / internal netting / raw
  spread** (prime-broker or netting ECN) — an operator/business decision.
- **Why it is DOWNGRADED (see the cost-arithmetic box above).** On FundedNext, F alone delivers (a)
  same-instrument internal netting (which the current edges **do not use**) and (b) the co-simulated equity
  curve (which is just item E, cheaper to build standalone) — **plus an Arc-10 trap** if cross-instrument
  netting is modeled. Its relative-value promise is **fictional without V**: the cost saving on a
  cross-pair spread requires a venue that actually nets the shared-currency leg, which FundedNext is not.
- **What it would let the search reach (only with V, and only if the edge term survives).** Cointegrated
  **basket** stat-arb and triangular **basis** trades under genuinely netted cost. **Caveat the council
  stresses:** even with V, the *edge* term on liquid majors is a coin-flip (arc 2010 point 1) — cheaper
  cost rescues a *real-but-cost-buried* edge, not a coin-flip. **Recommendation:** do **not** build F to
  rescue relative-value. Build E (its safe, valuable subset) first; consider F+V only if (1) E proves
  cross-component capital interaction is materially favorable, AND (2) a real netting venue is secured, AND
  (3) a *cost-model-swap* pre-test (re-cost an existing marginal RV signal at true netted cost) shows the
  edge clears — that pre-test is free and de-risks the entire build.

### J — CFTC-COT positioning data feed · cheap data unlock · low prior

- **What must be enabled.** Ingest the **free, weekly, public** CFTC Commitments-of-Traders report
  (leveraged-fund net positioning per currency) into the feature pipeline as an exogenous series
  (3-day-lagged as published; no lookahead).
- **Why it unlocks.** The only positioning proxy the search ever used was price (volume = magnitude,
  dead). COT is the *measured* crowded-trade extreme — when leveraged-fund net positioning hits a
  multi-year extreme, the marginal speculative buyer is exhausted and a forced unwind follows (the
  measured version of what fbr captures only as a price footprint).
- **What it lets the search reach.** Conditioning the existing fbr / reversion legs on COT-extreme weeks.
  **Low prior** (the cross-examiner's caveat): this is another *regime-conditioning* axis on a base edge
  that is already ~cost, and regime conditioning has failed across dispersion/vol/efficiency-ratio.
  Cheapest data unlock on the list, but weigh it below E/D.

### C — Economic-calendar timestamps · cheap data unlock · H1-cost-wall risk

- **What must be enabled.** A high-impact scheduled-release calendar (NFP, CPI, FOMC, ECB; free/cheap) as
  event timestamps in the panel.
- **Why it unlocks.** The post-release initial spike is driven by headline-reaction / stop-cascade flow
  that systematically over-shoots — a forced-flow reversion (the surviving family) on an event class the
  search never isolated (it tested gotobi/month-end calendar masks, never scheduled macro events).
- **What it lets the search reach.** Post-release "fade the knee-jerk" reversion, conditioned on
  top-decile-volatility releases, controlled against non-event same-clock-time bars. **Risk:** shares the
  H1 cost wall (~2× H4) that killed the microstructure cluster — but post-event moves are *multi-ATR*
  (unlike gotobi's tiny moves), which is the one thing that can beat a cost wall. Worth it only if the
  calendar feed is added cheaply.

### O — Options / strike / implied-vol surface data · permanent ceiling note (not a build proposal)

- **What is missing.** Strike-level / IV-surface data. The apparatus is OHLC bid+ask only.
- **Why it matters.** The **volatility risk premium** (implied > realized; harvested by *selling*
  options) is the single largest durable paid edge in FX that this programme structurally cannot touch.
  Gamma-pin / risk-reversal-skew structures are likewise unreachable (option-expiry collapses to
  round-number, dead — arc 2019). Recorded so the council does not relitigate it: this is a **permanent
  ceiling** under the current data charter, not a near-term build.

### M — Macro / rates feed · highest theoretical edge · largest cost · out of current charter

- **What must be enabled.** Rate-differential / central-bank-surprise / sovereign-positioning series in
  the feature pipeline.
- **Why it unlocks.** These are the **actual drivers of FX direction** that the apparatus forbids; their
  absence is the *mechanistic root* of the 14-arc directional-coin-flip convergence
  ([`ESCALATION`](./ESCALATION_apparatus_capability.md) "Why this is mechanistic"). This is the only
  unlock that could re-light the **EDGE** term rather than the cost term.
- **What it lets the search reach (and the warning).** Genuinely untested ground (not closed) — but the
  largest build+data cost on the list and it changes the project's nature (price-structure → macro). The
  council's recommendation: **defer**; if pursued, scope a **minimal single-series probe first** (e.g.
  the rate-differential sign as an ex-ante regime gate on the surviving fbr leg) before committing to a
  full feed.

### W — Dynamic-weight combiner · small · enabling-only (not a route to PASS)

- **What must be enabled.** Time-varying (ex-ante, causal) component weights in the combiner, for a
  neutral vol-target / risk-parity sizing default.
- **Why it is here only as a note.** A vol-target overlay is **not a route to a PASS** (it finds no edge;
  contemporaneous reweighting = lookahead = fabricated worst-fold lift; ex-ante it only scales a fold).
  Listed so it is on record as a small enabling piece *if* the operator wants an honest ex-ante sizing
  default applied after an edge exists — explicitly subordinate to the gate-governance call (arc 2019).

---

## Cross-reference

- Standing structural escalation + Arc-10 cost-regime warning: [`ESCALATION_apparatus_capability.md`](./ESCALATION_apparatus_capability.md)
- Change-surface style + the long-only→short PR spec (the model for F's scoping): [`SHORTS_ENABLEMENT_PROBE.md`](./SHORTS_ENABLEMENT_PROBE.md)
- The `explore-now` menu that runs WITHOUT any of these: [`DISCOVERY_DIRECTION.md`](./DISCOVERY_DIRECTION.md)
- The portfolio-route closure that makes E the priority: `DISCOVERY_LOG.md` arcs 2016 / 2017 / 2019 / 1023 / 2021

---
---

## RUN 2 (2026-06-06) — the fundable re-rank (advances the queue above; does NOT overwrite it)

> **Why re-rank.** Run 1 ranked by *cheapest-honest-build* and (correctly) put item E first to close the
> portfolio question. That question is now closed (E: the book fails all-folds-positive *fundamentally*).
> The live question is no longer "settle the book" but "**what can reach FUNDABLE size at all**" — the book
> is ~14–20× short and, by the in-charter conservation law ([`DISCOVERY_DIRECTION.md`](./DISCOVERY_DIRECTION.md)
> §0), nothing in-charter can close that gap. So this run ranks the operator-gated unlocks by **fundable-
> ROI-per-cost**, using a single discriminator established by the corpus:
>
> **A lever is fundable-relevant only if it supplies a CONTINUOUS NON-PRICE state** (one that is *not* a
> near-martingale at the actionable sampling interval). That is the only thing that breaks the
> `frequency × edge ≈ const` wall. A lever that only adds *another rare-event conditioner* moves *along* the
> wall and is sub-fundable by frequency before it is even tested. Items below are tagged
> **[continuous-state]** or **[rare-conditioner]** accordingly.

### ⚠️ The structural fact that orders this whole re-rank: every fundable-MAGNITUDE FX edge is SHORT-VOL

The reversion book (sells dislocation = short gamma), **carry** (short-vol by construction), and
**vol-selling** are the *same risk factor*. Short-vol books all co-fail on the *same* risk-off years (2015
CHF de-peg; 2018 strong-USD trend). Three corpus facts are one fact: the book fails all-folds on 2015 **and**
2018 under every weighting (item E); path-B densification is impossible because the legs share that one
risk-off factor (arc 3021, ρ≈0.12 floor); and carry would **co-crash, not diversify**. **Therefore the
only structural diversifier of the FX book is a LONG-VOL / positive-skew sleeve** — which in FX is
unfundable-as-steady (bleeds in calm; convexity dead, arc 1062), but in **cross-asset trend-following IS
documented-fundable**. This is why item **U** below outranks everything: it is simultaneously the cheapest
fundable shot AND the long-vol leg the short-vol book has always lacked (they fail in *opposite* regimes —
the all-folds route path-B never could reach).

### U — DIFFERENT INSTRUMENT UNIVERSE (cross-asset trend basket) · [continuous-state via universe] · run-2 RANK 1 fundable-ROI-per-cost

- **What must be enabled.** A data feed + symbol universe beyond liquid major FX: less-arbitraged classes
  where directional structure is *not* arbitraged to a coin-flip — EM/exotic FX (USDZAR, USDMXN, USDTRY,
  USDINR), commodities (XAUUSD, WTI, NATGAS, grains), equity-index futures, crypto (BTC/ETH). The
  *apparatus needs almost nothing new*: `Panel.from_pairs` / `MultiPairBacktester` / the WFO folds /
  take-the-loss / exit-policy registry are all instrument-agnostic OHLC — this is a **data + universe swap**,
  not an engine change. Two real preconditions: (i) **confirm the live vehicle can actually trade these**
  (FundedNext lists metals/energies/indices/crypto/some exotic FX on some account types, against the *same*
  daily-DD pool — VERIFY, do not assume; an Arc-10 defect is asserting tradability the funded account
  lacks), and (ii) a **gap-tail control** (commodities/crypto gap over weekends/inventory/news → worse than
  −1R, which directly threatens the 5% daily-DD cap; needs explicit modeling, not a naive stop assumption).
- **Why it unlocks (fundable-relevant).** The ~0.49 directional coin-flip is an established result *only on
  the most-arbitraged market on earth (liquid major FX)*. The corpus never tested whether it is
  instrument-universal — and the practitioner record says it is not: diversified trend-following is a
  capacity-large, **positive-skew** (cuts losers at −1R, rides winners — exactly what take-the-loss +
  trailing already do), **documented-fundable** edge (CTAs ~0.4–0.7 diversified Sharpe). It is also the
  **long-vol diversifier** the short-vol FX book structurally lacks (see the short-vol note above).
- **Falsifiable prediction.** Run the *existing* trend/breakout signals (already built, already dead on
  liquid FX) on the new universe under the honest engine: per-instrument they are whippy/thin, but a
  **diversified basket** co-simulated through item E should show positive-skew fold ROI with mean >8% and
  worst-fold supported by *cross-regime* diversification (the basket pays in the trend/crisis years 2015/2018
  where the FX reversion book dies). **Falsifier:** if trend on the less-efficient universe is *also* a
  coin-flip after that universe's (wider) costs, the conservation law is instrument-universal and this closes.
- **Fundable-plausibility.** **PLAUSIBLE — the run-2 best shot.** Mean >8% is realistic for diversified
  cross-asset trend; **worst-fold >5% is the live risk**, driven by gap-tail and basket-construction (the FX
  4-way book failed the basket combiner — but it failed *because FX is efficient*; a genuinely-trending,
  less-correlated universe is a different test). Cheapest fundable-ROI-per-cost because it reuses the entire
  validated apparatus and needs no new edge-term theory. **Recommendation: scope a minimal probe** — one
  trend signal, a 5–8 instrument cross-asset basket, honest engine, item-E co-sim — *after* confirming
  vehicle tradability. This is the council's recommended next *fundable* direction, ahead of any new-data build.

### X — CME FX FUTURES SIGNED ORDER-FLOW (continuous lead variable) · [continuous-state] · run-2 RANK 2

- **What must be enabled.** Ingest CME FX-futures **signed volume / cumulative delta** (and open-interest
  change) as an exogenous, sub-bar-aligned continuous series in the feature pipeline — a *true exchange tape*
  (signed), distinct from the directionless spot **tick** volume already killed (arc 3002). Needs the CME
  data + a time-alignment step + an exogenous-series hook in the pool builder (additive, default-empty so
  existing signals stay byte-identical, per the shorts-probe change-surface style).
- **Why it unlocks (fundable-relevant).** This is the **only lever that supplies a continuous non-price
  *lead* variable** — the exact thing the conservation law says is required to break the frequency wall.
  Signed flow is persistent (autocorrelated over hours–days), so unlike price it is *not* a martingale at the
  actionable bar; if it leads spot, it is tradeable on *every* bar, not just rare events.
- **Falsifiable prediction (and a FREE pre-test).** Before building anything: measure
  `corr(signed_flow_t, spot_return_{t+1..k})` on obtainable historical CME data. If it is a coin-flip like
  tick volume → KILL for the price of a correlation, no apparatus change. If `corr > 0` and survives one
  actionable bar of decay at FundedNext cost → build the ingest.
- **Fundable-plausibility.** **PLAUSIBLE-IF-LEAD-PERSISTS, but knife-edged.** Two real risks: (i)
  **representativeness** — CME FX futures are ~5–10% of total FX volume (spot OTC dominates); the futures
  tape may *mirror* spot with lag rather than *lead* it; (ii) the **cost wall** that killed the
  microstructure cluster — a minutes-to-hours lead captured at retail bar resolution may not clear cost. The
  free pre-test de-risks the entire item. **Mean >8% reachable only if the lead is both real and
  multi-bar-persistent.** Note: the **OI/COT positioning** half of "CME data" is a separate, weaker thing —
  it is a **[rare-conditioner]**, not continuous-state; see the demotion of item **J** below.

### M (ADVANCED from the queue above) — MACRO / RATES feed · [continuous-state] · run-2 RANK 3 · highest ceiling, biggest build

- **Run-2 update.** Re-affirmed as the **only lever that attacks the EDGE term directly** (rate differentials
  are the actual driver of FX *direction*), and it is genuinely [continuous-state] (the rate-differential
  level/momentum is a persistent non-martingale). But it is a **slow** state — it supplies a directional
  *tilt* (changes over months), not high frequency. The fundable mechanism: a tilt strong enough to flip the
  continuous directional base *off* the 0.49 coin-flip would make the **entire continuous apparatus**
  tradeable — that is the one path to fundable *frequency* via the edge term. New corpus caveat: the macro
  **event-reaction** form is already dead (NFP fade, arc 1048: priced efficiently, corr≈0) — pursue the
  **rate-differential regime** form, not event-fades.
- **Fundable-plausibility.** **HIGH CEILING, UNPROVEN, project-redefining.** Biggest build+data cost; turns
  a price-structure programme into a macro programme. **Minimal single-series probe first** (run 1's
  recommendation, sharpened): does `sign(rate_differential)` or its momentum, used as an ex-ante regime gate,
  flip the directional base's capture above 0.50 on the existing apparatus? That probe is cheap and gates the
  entire feed.

### K — CARRY on a SWAP-ON venue · [continuous-state, but fails the gate identically] · run-2 RANK 4

- **What must be enabled.** A **swap-paying execution venue** (NOT FundedNext, which zeroes carry by the
  swap-free add-on — this is a *venue/business* decision, the largest charter break here) + carry accrual in
  the cost/P&L model (currently swaps are off by protocol §1).
- **Why it is here.** Carry is the **largest historical FX return stream** (~0.5–0.7 Sharpe on a G10 carry
  basket pre-2008) and is genuinely [continuous-state] (earned every bar held, scales with holding time not
  event-count) — it sidesteps the directional coin-flip entirely.
- **Why it is RANK 4 despite real magnitude (the honest call).** Three corpus-grounded objections, in order:
  (1) it requires **leaving the target broker** — it is not an unlock of the current charter but a different
  business; (2) it is **short-vol and CO-CRASHES with the existing reversion book** on exactly 2015/2018
  (CHF un-peg gapped EURCHF ~30% through any stop — the canonical carry-crash), so it does **not** diversify
  the book and **fails all-folds-positive on the same crash years** the book already fails; (3) carry's left
  tail is structurally incompatible with a **DD-gated funded vehicle** (one gap-through-stop = instant fail).
- **Fundable-plausibility.** **Fundable in MAGNITUDE (mean >8% plausible at leverage), NOT fundable on the
  GATE (worst-fold >5% fails on the carry-crash years) and venue-gated.** Pursue only if the operator both
  (a) accepts a swap-on venue *and* (b) has a long-vol/crisis hedge for the tail — which is itself
  out-of-charter (options). Recorded as real-but-blocked, not recommended.

### O (ADVANCED) — OPTIONS / IV surface · [continuous-state but triple-gated] · run-2 RANK 5 · ceiling CONFIRMED

- **Run-2 update.** Confirmed as a **permanent ceiling**, with the short-vol note added: the **volatility
  risk premium** (IV > RV, harvested by *selling* options) is the single largest durable paid FX edge and is
  genuinely [continuous-state] — but it is **triple-gated** (no strike/IV data in-charter; needs an options
  *venue*; and selling vol is **short-gamma → the same risk-off crash tail** as carry and the book). Even the
  *defensive* use the divergent lens raised (risk-reversal skew as a **carry-crowdedness / long-vol crisis
  gauge**, the genre that *could* diversify the short-vol book) needs the same IV-surface data. Not a
  near-term build; recorded so it is not relitigated.

### J (DEMOTED) — CFTC-COT positioning · [rare-conditioner] · run-2 RANK 6

- **Run-2 update.** Explicitly reclassified as a **[rare-conditioner]**, not continuous-state: COT is
  weekly, 3-day-lagged, and positioning *extremes* are rare (a few per currency per year). It therefore
  moves *along* the conservation law (another rare-event conditioner on a base edge already ≈ cost), not
  across it. Same prior as the in-charter conditioners that all thinned folds or removed the help (arcs
  1059/1029/1055). **Sub-fundable by frequency** before it is tested. Cheapest data unlock on the list, but
  it cannot close the magnitude gap; weigh below U/X/M.

### Speculative long-vol / crisis-alpha data (NEW, low-prior, recorded for completeness)

The short-vol note implies the book's missing complement is a **long-vol crisis signal**. Two non-price
modalities could supply one cheaply, both [rare-conditioner] / convex-tail (unfundable-as-steady, fundable
only as a convex add-on): **(a) stablecoin redemption / secondary-market discount** as a real-time offshore
USD-funding-stress lead (on-chain mint-redeem + stablecoin price APIs — cheap); **(b) FX risk-reversal skew**
as a carry-crowdedness gauge (needs the item-O IV data). Both are crisis-alpha sleeves, not steady earners;
listed so the long-vol diversifier idea is on record, ranked below the fundable items.

### Run-2 ranked summary (fundable-ROI-per-cost)

1. **U — different instrument universe (cross-asset trend)** — [continuous-state via universe]; reuses the
   whole apparatus; fundable + the long-vol diversifier. *Best shot.* Precondition: verify vehicle tradability + gap-tail control.
2. **X — CME signed order-flow** — [continuous-state]; the only continuous non-price *lead*; **free
   corr pre-test** gates the build; representativeness + cost-wall risks.
3. **M — macro/rates regime** — [continuous-state, slow]; highest ceiling, biggest build, project-redefining; minimal single-series probe first.
4. **K — carry (swap-on venue)** — fundable magnitude but venue-gated + short-vol co-crash → fails the gate identically.
5. **O — options/IV vol-premium** — largest durable edge but triple-gated; permanent ceiling confirmed.
6. **J — COT positioning** — [rare-conditioner]; sub-fundable by frequency.
- (Unchanged from run 1, still valid as *measurement-honesty* builds, not fundable routes: **E** done/in-review, **D** passive-fill with adverse-selection, **F+V** downgraded netted-book, **C** calendar timestamps, **W** dynamic-weight combiner.)
