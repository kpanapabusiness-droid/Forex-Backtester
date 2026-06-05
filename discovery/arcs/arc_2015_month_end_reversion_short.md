# arc 2015 — Month-end reversion SHORT: independent reproduction + the 4-way combination

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **KILL** (4-way combined book, like arcs 2008/1015) — independently confirms the **me_short PORTFOLIO** component (co-discovered with arc 1019) · **Council:** none (combination arc / PORTFOLIO component, not a PASS survivor)

> **CONVERGES with independent concurrent arc 1019 (1000s) — same component, same construction, same
> verdict (PORTFOLIO), discovered in parallel = a strong Arc-10 independent reproduction.** Arc 1019
> deferred the 4-way combination to "arc 1020"; **this arc's primary additive contribution is that 4-way
> portfolio combination + convex search** — the deployable-gate test. The component record lives at
> `discovery/portfolio-candidates/arc_1019_month_end_reversion_short/` (1019's; not duplicated here).
>
> The portfolio route (the corpus's live deployable path) is blocked by a 4th-leg spec — net-positive on
> BOTH 2015 & 2018 (strong-USD/risk-off). ~18 arcs hunted it as a SEPARATE short/trend/vol/flow/rel-value
> component; all dead. Both 1019 and this arc find it via a different principle: the **short side of the ONE
> proven 2018-positive mechanism** (month-end forced rebalancing, arc 1011) — untested because shorts were
> disabled when 1011 ran. Shorts merged today (PR #273). My §3 numbers (null +0.92pp, n=145) match 1019's
> (null +0.80pp, n=116) up to universe/seed nuance — independent confirmation.

---

## 1. READ + SYNTHESIZE THE LOG (step a)

Pulled main (arcs 1018, 3015 landed mid-session — both 2018-leg KILLs); no STOP. Read protocol, full
Tier-1 ledger (0–3015), LESSONS, registry, the 3 portfolio-candidate configs. State: 3 PORTFOLIO edges
(gap 1006 JPY-cross H4; me-long 1011 USD-major D1, **the only demonstrably +2018 mechanism**; fbr 1013
USD-major H4, strongest, +2015 −2018). The 3-way book is provably blocked (2008/3009/1015: 0/5151 convex
weightings all-folds-positive; binding 2015-wants-fbr vs 2018-wants-me_long, mutually exclusive). The
2018 leg was unfound across every SEPARATE-component route.

**Key synthesis (the opening):** arc 3015's own diagnosis — *"me (1011) survives 2018 via the hard
inelastic WMR/index-rebalancing mandate unique to month-end."* That mandate reverts the month's move
**regardless of sign**, but 1011/1012/3008 only ever tested the LONG side (buy DOWN-moves) because shorts
were disabled. The SHORT side (sell UP-moves into month-end) is genuinely untested — and in strong-USD
years the dominant month-moves are USD-UP, so the short fires EXACTLY in the 2015/2018 months 1011
under-covers. A clean, mechanistic +2015&+2018 thesis using a PROVEN mechanism + the new short capability.

## 2. OBSERVATION (step b/c) — the short side reverts in the binding folds

Direction-aware observe (`observe_long_capture(direction="short")`), USD majors D1, big UP-move into
month-end (into≥+1 ATR), IS. Short cap **0.551** (n=118, >0.50), and the binding-fold acceptance test is
clean: **2015 cap 0.545 / drift +0.437** and **2018 cap 0.818 (9/11) / drift +0.293** — POSITIVE in both,
where the long side is negative-drift (2015 −0.076, 2018 −0.060). The signature is right: short lights up
in strong-USD/risk-off years (2011/2015/2018/2019), quiet in calm 2014/16/17. Non-coin-flip → §5f requires
the honest engine.

## 3. HONEST ENGINE §5f + soundness (step f/g)

Built `MonthEndReversionShortSignal` (registered; `Direction.SHORT`, mirror of 1011). §5f exit menu, IS WFO:

| exit | mean | neg/10 |
|---|---|---|
| sl_only | −0.498% | 5 |
| sl_plus_tp_3r | +0.629% | 4 |
| sl_plus_trailing_atr | +0.520% | 4 |
| **sl_partial_close_1r_runner_trail** | **+0.683%** | 3 |

The edge is in the OVERSHOOT (sl_only negative; the reversion RUNS in strong-USD years) — so a runner exit,
unlike 1011's tight 2-bar. Best exit per-fold (fold→year, anchor fold 6=2015): `2011:+3.39 2012:+1.69
2013:−0.90 2014:+0.98 2015:+0.40 2016:−0.91 2017:−0.68 2018:+0.86 2019:+1.29 2020:+0.71` — **POSITIVE in
both binding folds (2015 +0.40, 2018 +0.86)**; negatives are calm years (regime-complementary to the
existing components).

**Fair same-side NULL (decisive):** real +0.683% vs random-entry SHORT null **−0.234%** (3 seeds, matched
fire-rate, same exit) → **excess +0.92pp**. A real edge — NOT exit geometry or regime drift (unlike arc
2013's up-gap short, which lost to its null).

**Robustness (the arc-2013 / Arc-10 defense — mandatory):**
- **2018 is ROBUST:** survives every leave-one-pair-out (+0.59 to +1.99, incl. −GBPUSD +1.14) and every
  threshold (+0.68 to +1.72); EURUSD(+1.68,n3)/AUDUSD-carried, broad. **The corpus's first scalable +2018 leg.**
- **2015 is THIN REGIME-LUCK:** the +0.40 fold rests on a **single GBPUSD trade that ran +8.3R**; all other
  2015 trades are −1R/negative. Drop GBPUSD → 2015 −0.16; raise threshold to 1.25 → 2015 −0.76. **Do NOT
  rely on this for 2015.**
- Overall mean leans on GBPUSD (−GBPUSD → −0.016%, ~flat) but 6/7 LOO clearly positive; excluding the
  lucky 2015 fold the mean STAYS +0.71% (passes the arc-2013 "net-negative-without-the-lucky-fold" test).

## 4. PORTFOLIO CONTRIBUTION — the 4-way book (the deployable gate)

Reproduced all 3 existing components at their EXACT committed configs and **VERIFIED the headlines**
(gap +0.685%, me_long +0.232%, fbr +1.854%, with matching 2015/2018 signs) before trusting any combination
(Arc-10 discipline). Per-fold ROI correlation: me_short vs gap +0.188, me_long +0.157, fbr +0.406.

**4-way convex search (step 0.05, 1771 weightings — mirrors arc 2008):**
- **0/1771 all-folds-positive** (still blocked) — BUT best achievable worst-fold improves from the 3-way's
  **−0.222%** (arc 2008's hard wall) to **−0.124%** (w={gap:0, me_long:0.65, fbr:0.2, me_short:0.15});
  binding folds now MARGINAL: **2015 −0.05, 2018 −0.12**.
- me_short's robust +2018 nearly closed the 2018 wall (−0.77 → −0.12). **First new component to MOVE the
  convex-search wall** — 2006/2008/3009/1015 all hit the same −0.77/−0.222.
- Risk-parity 4-way: worst −0.422% (vs 3-way −0.770%).

## 5. VERDICT — PORTFOLIO (me_short); 4-way combination KILL (components advance)

The month-end reversion SHORT is a **real, null-beating (+0.92pp), mechanism-grounded** (the short side of
the proven 1011 forced-flow), **robustly +2018** PORTFOLIO component — the corpus's first scalable
+2018 leg, found via the new short capability where 18 separate-component arcs failed. Its 2015 is thin
regime-luck and its headline mean leans on GBPUSD (documented), so it is not all-folds-positive (3/10) and
not a solo survivor → **PORTFOLIO** (§11). The 4-way combination is **still KILL** (0/1771 all-folds-positive,
like 2006/2008/1015) — but it is the **closest the corpus has come to deployable**: best worst-fold
−0.222% → −0.124%, both blockers now within 0.12pp of zero.

**Threads / lessons.**
1. **The 2018 wall is breached (robustly), the 2015 wall is not.** me_short fixes 2018 (its robust leg);
   the residual block is 2015, where the only robust contributor is fbr (1013) and me_short's 2015 is luck.
   **Precise 5th-component / refinement spec: a robust +2015 leg — the MIRROR of me_short (robust +2018).**
   Strong-USD 2015 (SNB unpeg, EUR collapse, China deval) is acute-event-driven (vs 2018's grind), so the
   +2015 leg is likely a different mechanism than month-end reversion.
2. **The short side of a proven LONG flow-mechanism is a productive, untested lane** (shorts were disabled
   for all the 2018-era flow work). me_short is the short mirror of 1011; the gap-fill short mirror (1006)
   was dead (1016/2013, up-gaps don't fill at i+1) — so the lane is mechanism-specific, not universal: it
   works where the forced-flow reverts regardless of sign (month-end), not where the long edge was
   direction-specific (gap-fill's down-gap carry/JPY drift).
3. **me_short's value is +2018, NOT its headline mean** (GBPUSD-leaning) or 2015 (single-trade luck) — the
   honest portfolio read. Recorded with full robustness caveats so a later combination arc weights it for
   what is real (the 2018 leg), not the luck.

**Disposition: PORTFOLIO.** Recorded at `discovery/portfolio-candidates/arc_2015_month_end_reversion_short/`.

**FLAGS (code not merged):** none requiring the canonical core. Built + registered (BUILT, EXPERIMENT)
`MonthEndReversionShortSignal` (direction-mirror of `MonthEndReversionLongSignal`). Drivers scratch
`_disco2000_work/arc2015_*.py` (reproducible from this doc; the 4-way reproduces the 3 committed component
headlines EXACTLY before combining).
