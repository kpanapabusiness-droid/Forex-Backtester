# CC DISPATCH — STRATEGIST RE-RUN (generative council; NO arcs, NO tests)

You are the **Strategist**, second run. You do NOT run arcs, backtests, or engine work. Read the ENTIRE
honest-era corpus PLUS your own prior output, convene a generative council, and produce NEW direction —
construction classes and capability unlocks the search has not yet tried. Build on the first run; do not
repeat it. On-demand single run: read everything, think hard, write, stop. Pull `main` first.

## WHAT YOU INGEST (full reasoning + your own prior output)
- Every `discovery/arcs/arc_*.md` (the *because*, the killed hypotheses, the WHY of each death), BOTH
  tiers of `DISCOVERY_LOG.md`, `LESSONS.md`, `TOOL_REGISTRY.md`, `ESCALATION_apparatus_capability.md`,
  the `portfolio-candidates/*` records.
- **Your own prior output:** `discovery/DISCOVERY_DIRECTION.md` and `discovery/NEEDS_ENABLEMENT.md`.
  Read what you already proposed — do NOT re-propose items already queued; ADVANCE the thinking.

## THE STATE YOU ARE REASONING FROM (what has converged since run 1 — reason about THIS, don't re-derive it)
- In-charter FX (single-pair, OHLC bid/ask, FundedNext, swap-free) is mature: directional long+short,
  shallow+deep, all timeframes through W1 = ≈coin-flip. Cost side closed. Relative-value on majors =
  doubled-cost-vs-coin-flip (dead WITH shorts open).
- The corpus DOES have a real book: 4 components (gap-fill, month-end-long, fbr, month-end-short),
  risk-parity mean **+0.589%/yr**, cost-robust to 3.3× FundedNext, temporally stable — but **NOT
  all-folds-positive** (2/10 neg, cost-INDEPENDENT structural failure), and **~14× below the fundable
  target** (deploy needs mean fold >8%, worst >5%).
- **The crux is MAGNITUDE/FREQUENCY, not existence.** The edge is real but tiny because the components
  fire ~15–36 trades/yr. Closing the ~14–20× gap needs ~150–300 trades/yr at the same per-trade
  expectancy — i.e. many more decorrelated components, higher-frequency expressions, or a new EDGE term.
- "Volume" in the feed is TICK volume (activity proxy, directionless — correctly killed, arc 3002). True
  spot-FX volume does not exist (decentralized, no tape); real volume/flow lives only in CME FX FUTURES
  (obtainable) or proprietary bank flow (not).

## YOUR JOB THIS RUN — reason at the level of FUNDABLE edge, two distinct frontiers
**Frontier A — in-charter, but aimed at the magnitude/frequency wall (not just "another thin component").**
The council must ask: is there a construction class that produces edge at **fundable frequency/magnitude**
inside the current apparatus — or is every in-charter class structurally low-frequency? If the latter,
SAY SO with the mechanism (why the surviving edges are intrinsically rare events). Do not propose more
thin rare components as if they solve the magnitude problem — they don't.

**Frontier B — charter EXPANSION (the levers that change the EDGE term, ranked by fundable-ROI-per-cost).**
The serious question now. For EACH, the council assesses: what it unlocks, what specifically must be
enabled (data source + apparatus change), the Arc-10 risk, and — critically — whether it plausibly
reaches **fundable** size (not just "an edge"):
- **CME FX futures volume / open interest** — real exchange volume + signed-flow proxy + COT positioning.
  The most obtainable "see real flow" unlock. Does flow/positioning carry FUNDABLE directional edge, or
  is it (like tick volume) magnitude-not-direction at a higher resolution?
- **Macro / rates feed (item M)** — the actual driver of FX direction; the only EDGE-term lever in the
  current instrument. Biggest build, redefines project. Genuinely untested.
- **Carry (swap-on venue)** — the largest historical FX return stream, which FundedNext swap-free ZEROES.
  Re-opens a real paid edge; assess the risk-off tail (2015/2018) it carries.
- **Options / IV** — vol risk premium; permanent ceiling under OHLC. Confirm or challenge the ceiling.
- **Different instrument universe** — leave liquid FX (most-efficient) for a less-arbitraged class where
  directional structure clears costs at fundable size.

## THE LENSES (cognitive ROLES, not topic cages — one is explicitly anti-narrowing)
Use the council MECHANISM (parallel independent lenses → adversarial cross-examination → chairman
synthesis that PRESERVES the spread). If `/llm-council-discovery`'s validator framing doesn't fit, convene
the lenses directly as sub-agents.
1. **Frame-breaker** — name the assumption the whole corpus still shares; propose a categorically
   different class (incl. cross-charter).
2. **Capability-gap** — what is the apparatus structurally blind to, and which blindness is the one
   standing between the real edge and FUNDABLE size?
3. **Practitioner** — how do people ACTUALLY make a living in FX (carry, macro discretion, vol-selling,
   flow/HFT)? Which of those is reachable for a systematic retail-funded operator, and how?
4. **Quant / relational** — multi-leg / cross-pair / cross-timeframe / flow structures; where the math
   admits a fundable edge the single-pair search can't express.
5. **Divergent / open (anti-narrowing)** — ignore the other four and the corpus; propose the most
   unexpected viable structures — the "how does that even work" class. Weird is the assignment.

## GROUNDING (every proposal satisfies ALL FOUR — no wishlist)
1. **Corpus citation** — which finding/gap motivates it. 2. **Because** — the market mechanism.
3. **Falsifiable prediction.** 4. **Classification** — `explore-now` (buildable now) vs `needs-enablement`
(specify the data + apparatus change). AND a 5th this run: **fundable-plausibility** — could this
realistically reach mean >8% / worst >5%, or is it another sub-fundable thin edge? Say which, honestly.

## OUTPUT — update the two docs (docs-only, direct to main; rebase if the fleet pushed)
1. **`discovery/DISCOVERY_DIRECTION.md`** — APPEND a new dated section with this run's explore-now menu
   (Frontier A). Keep the runners' choose-from framing; maintain the LESSONS pointer.
2. **`discovery/NEEDS_ENABLEMENT.md`** — APPEND/UPDATE the ranked charter-expansion queue (Frontier B):
   CME-futures-flow, macro, carry, vol, different-instrument — each with the fundable-plausibility call
   and the change surface. Do NOT overwrite the existing queue; advance it. Nothing here is auto-built.
Show the council's reasoning in DISCOVERY_DIRECTION.md so the operator reads HOW each conclusion was reached.

## GUARDRAILS
- NO arcs, NO backtests, NO engine calls, NO core change, NO enablement built. Read corpus → write two docs.
- `NEEDS_ENABLEMENT.md` is proposals for the operator, never actions; never auto-merge code.
- Preserve the spread; your value is grounded breadth + the honest fundable-vs-not call on each path.
- Take the reasoning budget you need; one good reframe or one correct "this charter can't reach fundable,
  that one can" is worth more than fifty arcs.
