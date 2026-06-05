# Arc 1008 — Gotobi-Day Tokyo-Morning USDJPY Long (Microstructure Settlement Flow)

> **Arc id:** 1008 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL (cheap-kill at observation).** The documented "gotobi" USD-settlement
> bias is ABSENT in the 2010–2026 corpus at both H4 and H1. The gotobi *premium* (gotobi minus
> non-gotobi Tokyo-morning USDJPY return) is **≤ +0.25 bp gross** in the single best fix-timed window
> (frac+ 0.501 — a coin flip), and **negative-to-flat everywhere else**, against a ~2 bp FundedNext
> round-trip cost. No capturable edge to take to the engine.
> **Lever tested:** MECHANISM CHANGE — execution/microstructure settlement flow (frontier item 3,
> available pre-shorts), a genuinely novel calendar-microstructure mechanism with a documented *because*.

Scored only descriptively (gross bar return + spread) — falsified at observation before pool/engine,
exactly as arcs 1001 / 2000 / 2003 / 3005. No council (no worthwhile-ceiling fork). No null baseline
(no edge to compare). The honest engine (costs + SL-first) can only make a no-edge gross signal worse
(arc 3003), so a ≤+0.25 bp coin-flip premium is guaranteed sub-cost — no engine compute spent.

## (a) Log read — FRESH EYES (honest-era only)

Pulled main. Honest-era corpus = 16 arcs / 3 chats. **Closed comprehensively:** single-condition shallow
directional prediction (momentum / breakout / mean-reversion / trend) across H1/H4/D1, majors + crosses
(28 pairs), under both the +1R-capture and the forward-drift lens, every exit/SL, and stop-removed (arc
3004); regime conditioning failed 3 ways (dispersion / vol / Kaufman-ER, strong trends INVERT); volume =
magnitude not direction; turn-of-month USD drift real but sub-cost (arc 1005); triangulation residual ≈ 0
(arc 3005). The binding wall is **EDGE < COST** for directional bets; the arc-3004 escalation says durable
FX edges need a structural unlock (relative-value / shorts) the operator hasn't flipped to yet.

**The one positive lead:** weekend gap-DOWN-fill long on JPY crosses (arc 1006) — first mean-positive IS
(+0.69%), beats random null, low DD, fold-fragile → **PORTFOLIO** disposition. It is an overshoot, not a
fill-to-target (1007); majors negative (2001); TF-invariant (2002). The mechanism that WORKED is a
**discrete-liquidity / flow EVENT reversion**, not continuous prediction.

**My lane (dispatch, pre-shorts):** hunt a **2nd net-positive long-only component** (decorrelated, for the
PORTFOLIO route) OR a novel structural mechanism with a *because*. Shorts capability is merged to main but
the operator has NOT flipped the lane → I stay long-only, no short arc. Refining the gap-fill again yields a
*correlated* component (already mapped 4 ways) → low value. I need a **decorrelated** mechanism. STOP absent.

## (b) Idea + observation — gotobi (documented *because*)

The validated-positive template is an event-driven FLOW dislocation. The single most-documented, untested,
decorrelated long-only FLOW in FX is the **gotobi effect**: on days whose date is divisible by 5 (5/10/15/
20/25) plus month-end ("gotobi" = 5-10 days), Japanese corporates settle invoices and banks buy USD/JPY to
cover, creating a documented upward USDJPY drift through the Tokyo morning, peaking at the **9:55 JST fixing
(= 00:55 UTC)**. Long-only-tradeable (long USDJPY, a major), event-timed, decorrelated from weekend gaps.
Arc 1000 tested *generic* time-of-day (dry) but never gotobi-day-conditioned Tokyo-morning USDJPY. The
effect is documented to have decayed post-2015 — so **observe before believing.**

**Data alignment.** Panel is UTC-indexed, EET-aligned H4 bars (EET-00:00 bar starts 22 UTC winter / 21 UTC
summer, ends 01/02 UTC) — this bar fully contains the Tokyo run-up and the 00:55 UTC fix. Gotobi days tagged
as 5/10/15/20/25 + last-trading-day of month, shifted to the nearest PRIOR trading day when the target isn't
a trading day (the standard settlement-shift rule). EURUSD carried as a placebo (no JP settlement flow).

**Observation 1 — H4 EET-00:00 (Tokyo-morning) bar return, gross mid bp:**

| pair | window | GOTOBI mean | non-goto mean | gotobi frac+ | premium |
|---|---|---|---|---|---|
| USDJPY | all years | **−0.061 bp** (n=1172) | +0.130 bp (n=3570) | 0.490 | **−0.19 bp** |
| USDJPY | 2010–2014 | +2.004 bp | +1.984 bp | 0.522 | +0.02 bp |
| USDJPY | 2015–2020 | −0.983 bp | −0.875 bp | 0.463 | −0.11 bp |
| USDJPY | 2021+ (OOS) | −0.969 bp | −0.487 bp | 0.489 | −0.48 bp |
| EURUSD (placebo) | all years | +0.625 bp | +0.136 bp | 0.518 | — |

The gotobi bias is **absent-to-negative**: gotobi ≤ non-gotobi in every era. The only positive era
(2010–2014, +2 bp) is NOT gotobi-specific — non-gotobi is identical (+1.98 bp), i.e. it is generic
Abenomics-era Tokyo-morning USDJPY drift, gone by 2015. Typical close spread on the gotobi Tokyo bar 0.66 bp
→ ~2 bp FundedNext round-trip. The gotobi *premium* ≈ −0.2 bp gross ≪ cost.

**Observation 2 — H1 fix-window (does a run INTO the 00:55 fix get netted away by the H4 bar that also
contains the post-fix hour?), gross bp, gotobi premium = gotobi − non-gotobi:**

| window (UTC) | GOTOBI mean | frac+ | premium |
|---|---|---|---|
| Tokyo run 22→00 (winter) | +0.447 bp (n=949) | 0.501 | +0.121 bp |
| Tokyo run 21→23 (summer) | −0.600 bp (n=841) | 0.480 | −0.396 bp |
| fix hour 00→00 | +0.037 bp (n=958) | 0.501 | **+0.246 bp** (best) |
| post-fix 01→02 | −0.536 bp | 0.484 | −0.068 bp |
| full morning 22→03 | −0.917 bp | 0.480 | −0.458 bp |

No run-into-the-fix exists. The largest gotobi premium anywhere is **+0.25 bp** (fix-hour-only, frac+ 0.501 =
pure coin flip), and it is negative in the other windows — confirming the H4 finding at finer resolution
(arc 2002's resolution-test discipline applied). The H4 bar is not masking a fix run-up; there simply is no
gotobi premium.

## (c)–(g) Characterize / cheap-kill / diagnose / validate — NOT REACHED

Falsified at observation (§5b). The gotobi entry shows **no gross drift** (premium ≤ +0.25 bp, coin-flip
frac+ 0.501) → it is a coin-flip entry, so the §5f "fail-the-best-version exit sweep" (reserved for entries
that beat the null OR show gross forward drift) does not bite. Best resolution already tested (H1 fix-window).
A gross premium ≪ cost is guaranteed SL-honest-negative on `MultiPairBacktester` (arc 3003: SL-honest
expectancy ≤ raw drift; take-the-loss only subtracts) — building the pool / running the engine would only
confirm a foregone sub-cost result. Cheap-killed pre-pool, as arcs 1001/2000/2003/3005.

## Council — NOT convened

No worthwhile-ceiling or diagnosis fork: the mechanism is cleanly absent in the data (gotobi ≈ non-gotobi),
not a real-but-hard-to-capture edge. The remaining moves (sweep the gotobi-day definition, finer intraday
windows on a single pair) are rescue-sweeps the council would call "ritual" (cf. arc 1005).

## Final verdict — FAIL (cheap-kill at observation) → KILL

The **gotobi USDJPY settlement-flow long is not deployable** and shows no edge at all. The documented
mechanism does NOT produce a capturable Tokyo-morning USDJPY bias in 2010–2026 — either fully arbitraged
(it is one of the most-published FX seasonalities; the literature notes its post-2015 decay) or never of a
magnitude that survives at H4/H1. The gotobi premium (≤ +0.25 bp gross, coin-flip) is an order of magnitude
below the ~2 bp FundedNext round-trip cost. EDGE ≈ 0 < COST.

## Lessons (candidate for LESSONS.md)

1. **The gotobi effect is absent/arbitraged in 2010–2026 H4/H1 USDJPY** — gotobi Tokyo-morning return ≤
   non-gotobi in every era; the only +era (2010–2014 Abenomics) is a generic regime drift, not gotobi-
   specific (non-gotobi identical). A famous, well-documented microstructure seasonality dissolves on honest
   data — the *because* being real (it genuinely was a flow) does NOT mean the edge is still capturable.
2. **A documented flow is NOT a free pass past observation.** Arc 1005 (turn-of-month USD) found a real but
   sub-cost calendar flow; arc 1008 finds a famous one that is ~zero. Both confirm: calendar/flow mechanisms
   are observation-first, and the realizable magnitude (not the existence of the flow) is what matters vs the
   FundedNext cost floor. The microstructure-settlement-flow frontier item is now tested (and dry for gotobi).
3. **Mean is additive across sub-bars** — a containing H4 bar's mean ≤ a sub-window's mean only if a later
   sub-window offsets it; the H1 fix-window check (run-into-fix vs post-fix) is the right, cheap way to rule
   out a netted-away intraday run (arc 2002's resolution discipline, made quantitative).

## Threads

- **Closed:** gotobi USDJPY settlement-flow long (absent/arbitraged at H4 and H1). Generic Tokyo-morning
  time-of-day on USDJPY (already dry, arc 1000; re-confirmed — the +2bp 2010–2014 drift is regime, gone by
  2015).
- **Open / next (still pre-shorts long-only):** the decorrelated-2nd-component hunt continues. Calendar/flow
  is now weakly mapped (turn-of-month sub-cost 1005; gotobi absent 1008) → the remaining pre-shorts lanes
  are (a) multi-timeframe STRUCTURE (frontier item 3, genuinely untested as a setup, not a regime filter),
  and (b) a frank standing note that after 17 arcs the only net-positive long-only edge is the weekend
  gap-fill (PORTFOLIO), and a 2nd decorrelated net-positive long-only component has not been found — the
  high-leverage move remains the operator's shorts/second-leg unlock (arc 3004 escalation, reinforced by
  arcs 2001/2003 FLAG-1).

## Flags (code NOT merged)

None. No canonical-core change required; no reusable BUILT tool added (gotobi-date tagging is a one-off
calendar helper, kept in scratch — like arc 3005's triangulation observer). Reproducible from this doc.

## Reproduction

Observation drivers scratch (`_disco_work/arc1008_observe.py` H4 gotobi vs non-gotobi Tokyo-morning bar
return + placebo + era split; `arc1008_h1_fixwindow.py` H1 fix-window run-into-fix check). Data:
`histdata_root=C:\Users\panap\histdata_backup`, tf H4 + H1, `boundary_convention="5ers_eet"`. Gotobi tag:
day-of-month ∈ {5,10,15,20,25} + last-trading-day, shifted to nearest prior trading day. EET-00:00 bar =
first bar of each EET trading day (Tokyo morning, contains the 00:55 UTC fix). Pairs: USDJPY (+ EURUSD
placebo). Falsified at observation — no pool, engine, council, or null baseline invoked.
