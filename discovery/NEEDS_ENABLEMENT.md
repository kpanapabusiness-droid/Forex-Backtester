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
