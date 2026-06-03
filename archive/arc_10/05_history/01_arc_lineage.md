# Arc Lineage

> **Purpose:** Brief history of how we got to Arc 10. Helps future you avoid repeating dead ends.
> **For full chronology with rationale:** see `02_decisions_log.md` and `03_eliminated_approaches.md`.

## The story in 5 lines

1. NNFX indicator-sweep approach failed (GPT-4 hallucinated MQL4→Python conversions)
2. Bounded-event "Phase JL" approach invalidated by forward bias in pool construction
3. KGL/KH arc produced the KH-24 live system (now retired in favour of Arc 10)
4. L_ARC_PROTOCOL (the current research methodology) governed Arcs 1–10
5. Arc 10 v3.0.2 — DLR signal + three-stage exit — validated and deployed

## Pre-Arc work (~6 months)

**Stonehill / NNFX indicator sweep:** classical NNFX framework with indicator-based entries. Used GPT-4 + Aider to translate MQL4 indicators to Python. **Failed because GPT-4 produced hallucinated conversions** that didn't replicate MQL4 behaviour. Both GPT-4 and Aider permanently excluded from the toolchain after this.

**Phase JL — Bounded events:** new approach. Define "bounded events" (e.g. swing-low rejections) and build a pool from them. **Failed because pool construction had forward bias** — the criteria selecting events used future information. All Phase JL results invalidated. Permanent lesson: ex-ante population construction is non-negotiable.

## KGL/KH arc

Refined the bounded-event idea with strict ex-ante population. Produced the **KH-24 EA** (`kb_exhaustion_bar` 4H trend-pullback, long-only, 28 FX pairs). Deployed live; now being retired in favour of Arc 10.

KGL/KH proved:
- Ex-ante pool construction works
- Worst-fold WFO is the right gate
- One-day D1 lag (no same-day D1 close) is correct and must never be reverted

## L_ARC_PROTOCOL (current methodology)

Six-step pipeline for signal discovery:

1. **Plumbing** — pool construction, feature engineering
2. **Path-shape clustering** — group trades by forward trajectory archetype (NOT by magnitude)
3. **Capturability** — does an exit policy exist that extracts the trajectory's edge?
4. **Extractability** — does the signal exit policy survive realistic cost modelling?
5. **Cross-fold stability** — does the signal hold across folds with worst-fold gate?
6. **WFO** — final walk-forward optimisation under realistic costs

Arcs 1–9 explored various signal candidates. Arc 10 was the keeper.

## Arc 10 versions

| Version | Date | What changed |
|---|---|---|
| Arc 10 v1.x | early development | Initial DLR signal definition |
| Arc 10 v2.x | mid development | Three-stage exit policy refinement |
| **Arc 10 v3.0.2** | current | Locked deployment version |
| (future) | TBD | post-live-data refinements (deferred until 4+ weeks live) |

## What survived from earlier arcs

Despite the dead ends, several methodology pieces survived:

- **Worst-fold WFO gating** — from KGL/KH
- **Long-only constraint** — empirical asymmetry found in KGL
- **28-pair universe** — established in NNFX phase, kept throughout
- **Ex-ante population construction** — hard lesson from Phase JL
- **No future information in features** — same lesson, generalised
- **Per-bar audit fingerprints** — methodology refinement during Arc 8
- **Strategy Tester scenario validation** — from KH-24 live deployment work
- **D1 one-day lag** — established in KGL, never reverted

## What's NOT in Arc 10 from earlier work

- **Indicator-based entries** — abandoned in NNFX phase
- **Volume features** — failed when switched brokers (microstructure doesn't port)
- **Magnitude-based clustering** — replaced with path-shape clustering
- **GPT-4 or Aider for code** — permanently excluded
- **KH-24 strategy code** — different system, kept in repo for historical reference only

## Wave 1 of arcs (current cleanup)

Arcs 1-10 form "wave 1" of L_ARC_PROTOCOL research. Arc 10 succeeded; arcs 1-9 either failed or were superseded. Closing wave 1 means writing closure documents for each so the lineage is preserved.

After wave 1 closure: any further research becomes "wave 2" — built on top of the proven Arc 10 / KGL methodology.

## What's next: Lω (Lomega) discovery engine

**Lω (Lomega)** is a supervised feature discovery engine. Clusters the full dataset by forward price geometry, works backward to find predictive conditions. Designed as fallback if the L_ARC signal pipeline exhausts candidates without a deployable system.

**Status:** plan document drafted (`LOMEGA_DISCOVERY_PLAN.md`). Execution deferred — Arc 10 is the current deployed system; Lω only runs if we need new signal candidates after Arc 10 fully matures.

## Lessons that compounded across arcs

1. **One change per phase, pre-committed gate.** Bundling changes destroys interpretability.
2. **Worst-fold metric is the only deployment-relevant number.** Mean-fold is bait.
3. **Path-shape clustering before magnitude features.** Recurring failure: "found a signal, can't extract R" → because we clustered on outcome magnitudes, not on trajectory shapes.
4. **Broker data is not portable.** Volume + microstructure metrics fail when broker changes. Tested empirically (FTMO → 5ers).
5. **The break instinct is reliable signal.** When the body says stop, listen. Forced productivity through fatigue produces bugs.

## Where to find raw data

- Arc closure docs: `results/l_arc_<N>/ARC_CLOSURE.md` (and `__archive_for_research_reference__/` for failed arcs)
- KGL/KH arc history: scattered across earlier results folders
- Phase JL (invalidated): preserved in archive for "what NOT to do" reference
- NNFX phase: mostly deleted; methodology documented in journal
