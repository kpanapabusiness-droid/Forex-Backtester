> # ✅ COMPLETED 2026-05-13
>
> This document was the **plan** for the L characterization arc. The arc is now COMPLETE.
>
> **Deliverables from this arc (all extant):**
> - L0 through L5 phases executed
> - Four-layer atlas built (`LCHAR_ATLAS.md`)
> - Top-N registry produced (`LCHAR_TOPN_REGISTRY.md`) — 5 candidate signals ranked by DSR
> - Closure record: `L_ARC_CLOSURE.md`
> - Deferred candidates: `L_ARC_DEFERRED_CANDIDATES.md`
>
> **What happens next:** signal testing of the 5 registry candidates under `L_ARC_PROTOCOL.md` v1.0. Arcs 1–5 run as full separate arcs per the six-step extractability protocol. See `L_ARC_PROTOCOL.md` and `L_ARC_OPERATIONAL_SPEC.md`.
>
> **Status of this doc:** historical reference for how the registry was produced. Not authoritative for current or future signal-testing work.
>
> ---

# L Characterization Arc — Plan

**Status:** Planned, not yet started
**Created:** 2026-05-09
**Predecessor:** KH arc (closed by KH-27/28/29 diagnostic phases)
**Live system in production:** KH-24 (unchanged, out of scope for this arc)

---

## 1. Purpose

The L arc is a from-scratch exploratory data analysis effort to build a quantitative atlas of patterns and regularities in the FX data before any new signal building. The output is descriptive, not predictive. The goal is to characterize what is actually in the data so that future signal hypotheses come from evidence rather than guess.

This arc is independent of KH-24. No assumption that any pattern found will or should complement KH-24. No assumption that KH-24's choices (4H, long-only, exhaustion bar, 28-pair universe) carry forward as defaults. The L arc evaluates everything fresh.

---

## 2. Context

The KH arc closed in May 2026 after three post-lock diagnostic phases (KH-27, KH-28, KH-29) collectively established that fold 7's +1.92% ROI is the structural worst-fold ceiling for KH-24. Further KH-arc tuning will not move the needle. KH-24 stays live on the VPS, gate-passing on its own merits, ~$130k gross profit over 6 years, ~2.5% net annual ROI.

The L arc is a deliberate methodology shift. KH was top-down: hypothesize a signal class, build the population, test it. L is bottom-up: characterize the data first, let observed regularities suggest hypotheses through a pre-registered ranking rule, then test those hypotheses with the same KH-arc methodology that worked.

---

## 3. What This Arc Is

- Pure characterization of price, volatility, multi-timeframe, and cross-pair structure
- Multi-layer descriptive analysis producing an atlas of statistics with confidence intervals
- A pre-registered mechanical bridge from atlas findings to candidate signals
- Methodology validation: testing whether bottom-up characterization can produce candidate signals as well as top-down hypothesis-testing did

---

## 4. What This Arc Is Not

- Signal building (that comes after the atlas, in a separate arc)
- Backtesting or WFO running
- Strategy decisions or position sizing
- Building anything around, complementary to, or against KH-24
- An open-ended "explore until something interesting comes up" project — there is a structure, deliverables, and a budget

---

## 5. The Four-Layer Atlas Structure

The atlas is organized by analytical scope. Each layer is a separate phase with its own deliverables and gates.

### Layer 1 — Univariate price structure per pair

For each pair and each timeframe (1H, 4H, D1, plus W1 if pulled in time):

- Distribution of bar returns (mean, std, skew, kurtosis, percentiles)
- Autocorrelation of returns at multiple lags
- ATR distribution and its autocorrelation (volatility regime persistence)
- Run-length statistics: distribution of consecutive same-direction closes
- Distance-from-baseline statistics: distribution of (close − Kijun) / ATR, (close − SMA) / ATR
- Time-of-day and time-of-week effects: mean absolute return by hour and weekday

### Layer 2 — Multi-timeframe coherence

For each pair across all timeframe pairings (1H↔4H, 4H↔D1, D1↔W1):

- Correlation of same-bar returns at adjacent timeframes
- Lead-lag tests: does direction at the lower timeframe predict the next bar at the higher timeframe?
- Conditional ATR: when D1 ATR is in its 80th percentile, what is 4H ATR doing?
- Multi-timeframe alignment frequency: how often are multiple timeframes' trends co-aligned, and how does pair behavior differ in those regimes?

### Layer 3 — Cross-pair structure

Across the 28-pair universe:

- Rolling correlation matrices at each timeframe (60-bar window)
- Correlation matrix stability: how often does it shift, by how much, what triggers shifts
- Currency-strength index: rank the 8 currencies daily by strength; quantify persistence and reversals
- Lead-lag between pairs: does EUR/USD lead GBP/USD by N bars at what correlation
- Risk-on/risk-off clustering: do JPY pairs and AUD pairs anti-correlate, when, with what structure

### Layer 4 — Conditional structure (the bridge to signals)

Given specific structural conditions, what is the conditional return distribution?

- Conditional on extreme bars (top-decile range, top-decile distance-from-baseline): distribution of next 1, 4, 12, 24 bars
- Conditional on multi-timeframe alignment: next-N-bar distribution
- Conditional on specific structural setups (D1 closes through Kijun, range break, exhaustion patterns): conditional return profile
- Conditional on cross-pair structure (e.g., when one pair breaks recent high while a correlated pair does not): next-N-bar distribution

Layer 4 outputs are ranked by deviation from the unconditional baseline distribution, with proper statistical treatment of multiple comparisons. This is where the atlas hands off to signal candidacy.

---

## 6. Methodology Principles

### Descriptive, not predictive

Every output is a measured statistic with a confidence interval. The atlas does not say "if X happens, Y follows." It says "X happens with frequency F; conditional on X, the next-N-bar return distribution has mean M, standard deviation S, with N observations." The reader can decide whether the deviation from baseline is meaningful. The atlas does not.

This rule is what makes EDA on the full dataset safe. Descriptive statistics about the data are properties of the trading universe, not predictions. They cannot leak forward bias on their own. Forward bias only enters when descriptive findings are used to *select* which patterns to build signals from — that is what the pre-registered bridge guards against.

### The bridge from atlas to signals is pre-registered

Before the atlas is built, L0 locks in:

- The metric used to rank conditional patterns (e.g., absolute t-statistic with Bonferroni correction)
- The number N of top patterns that proceed to signal testing
- The signal-testing methodology that follows (which inherits KH-arc rules)

Once the atlas is complete, the top-N patterns by the locked metric proceed mechanically. No human "this one looks interesting" selection. No additions. No re-ranking after seeing results.

### Top-N candidate selection is mechanical

The bridge rule is the entire selection process. If the atlas surfaces a pattern that ranks #6 by the locked metric and looks compelling for narrative reasons, it does not enter signal testing this arc. It can be saved for a future arc with its own pre-registered bridge. Methodological discipline is preserved by mechanical execution.

### KH-arc non-negotiables carry over

All locked rules from the KH arc apply unchanged to any signal testing that follows the L atlas:

- No lookahead anywhere in signal, indicator, alignment, or stop logic
- `build_ex_ante_bounded_population` mandatory on every test
- WFO worst-fold is the only judge of success
- One change per phase, pre-committed gate, accept the result
- Per-bar MT5 spread data, never hardcoded defaults
- Risk = 1% of reset floor balance
- Clean labels are evaluation-only, never used for population selection
- GPT-4 and Aider permanently excluded from implementation
- Every phase gets a result document regardless of pass or fail

These rules are not L-arc-specific. They are project-permanent.

---

## 7. Phase Structure

Phases are sequential. Each produces specific deliverables and is gated before the next can begin.

### L0 — Methodology lock and bridge pre-registration

Output: `docs/L0_METHODOLOGY_LOCK.md` containing:
- The four-layer atlas structure (verbatim from this plan)
- The descriptive-not-predictive rule
- The pre-registered ranking metric and N for the bridge to signal testing
- The discipline rules and risk register
- The success criteria for the arc
- A pre-commit on time budget for the full arc

No code. No data. Pure planning lock. Cannot be modified after the arc starts.

### L1 — Layer 1 univariate analysis

Output: `results/lchar/atlas/layer1_univariate.csv` plus `results/lchar/PHASE_L1_RESULT.md`. Per-pair, per-timeframe descriptive statistics. Estimated time: 1–2 days.

### L2 — Layer 2 multi-timeframe analysis

Output: `results/lchar/atlas/layer2_multitimeframe.csv` plus `results/lchar/PHASE_L2_RESULT.md`. Coherence and lead-lag tables. Estimated time: 2–3 days.

### L3 — Layer 3 cross-pair analysis

Output: `results/lchar/atlas/layer3_crosspair.csv` plus `results/lchar/PHASE_L3_RESULT.md`. Correlation structures, currency strength, lead-lag between pairs. Estimated time: 3–5 days. This is the most complex layer.

### L4 — Layer 4 conditional analysis

Output: `results/lchar/atlas/layer4_conditional.csv` plus `results/lchar/PHASE_L4_RESULT.md`. Conditional distributions ranked by deviation from baseline, with multiple-comparisons correction. Estimated time: 3–5 days. Depends on L1–L3 to define candidate conditioning sets.

### L5 — Atlas synthesis and top-N candidate registry

Output: `LCHAR_ATLAS.md` (human-readable summary, organized by finding strength) and `LCHAR_TOPN_REGISTRY.md` (the mechanically-selected top-N candidates from L4 ranked by the L0 pre-registered metric). Estimated time: 1 session.

### L6 onward — Future signal testing (separate arcs)

The top-N candidates from L5 become the input to subsequent signal-testing arcs. Each candidate gets its own arc with KH-style methodology: ex-ante population, t-stat gate, lift harness, WFO. These arcs are not part of L. The L arc ends at L5.

---

## 8. Success Criteria

This is the most easily-misjudged part of the arc and worth being explicit about.

**Success is not "we found tradeable signals."** Whether the atlas surfaces signals worth testing is a property of the data, not of the arc's quality. The arc cannot guarantee tradeable findings exist.

**Success is the integrity of the characterization process.** The arc succeeds if:

1. The atlas is complete across all four layers without lookahead, contamination, or methodological breaches.
2. The bridge from atlas to candidate signals operates mechanically per the L0 pre-registration.
3. The top-N registry is produced from the atlas without human override.
4. Either signal candidates are surfaced for L6+ testing, OR the atlas conclusively shows that no significant patterns exist beyond the unconditional baseline (also a valid finding).

The arc is judged on process, not on whether process produces a winning system. This framing matters because it removes the temptation to massage the atlas toward "interesting" findings when the data does not support them.

A null result from the L arc is a real result. It tells the project that the data does not contain easily-extractable structure beyond what KH-24 already exploits, which constrains future research direction toward either deeper data sources (volume, sentiment, news) or different markets entirely.

---

## 9. Risk Register

Risks specific to this arc and their countermeasures.

**Scope creep.** Atlas-building is the kind of project that expands forever — every finding suggests another analysis. Countermeasure: L0 pre-commits a total time budget (suggested: 3–4 weeks of focused work). Whatever is in the atlas at the end of the budget is what proceeds to L5.

**The "interesting findings" trap.** When the atlas surfaces something that looks great and isn't in the pre-registered top-N, the temptation to add it is strong. Countermeasure: the bridge metric and N are locked in L0 and cannot be modified. Compelling-but-unranked findings are saved to a "future arcs" file and may be re-ranked in a future L-style arc with its own pre-registration.

**Statistical underpower.** Some Layer 4 analyses (especially cross-pair conditions) will be on small samples. Countermeasure: report confidence intervals honestly. Do not inflate confidence on underpowered findings. If the atlas's top-N candidates are all underpowered, that is the finding to report.

**Meta-lookahead via human pattern selection.** The classic JL-style failure: the human looks at outputs, notices what would have worked, builds populations around it. Countermeasure: the bridge is mechanical. Layers 1–4 are descriptive. Selection happens by metric, not by intuition.

**Confusion with KH-24 work.** Future Claude or future Keanu reads the project knowledge, sees KH-24 active, defaults to thinking in KH-arc terms. Countermeasure: this plan is the source of truth for the L arc. The handover prompt explicitly instructs the next chat to operate L independently. KH-24 is acknowledged but out of scope.

**Premature signal building.** Mid-atlas, the temptation to "just quickly test this one thing" appears. Countermeasure: the arc structure forbids signal testing inside L. Signal testing happens in L6+ arcs after L5 closes.

---

## 10. What Carries Over From the KH Arc

The methodology rules listed in section 6, unchanged.

The risk parameters: prop firm DD limits (10% max, 5% daily), 1% per-trade risk for any future signal that reaches live deployment.

The data: 1H, 4H, D1 across 28 pairs from Oct 2020 onward. Weekly may be added if a candidate signal in L6+ requires W1 alignment as a filter, but is not pulled for the L arc itself.

The infrastructure: Python backtester as source of truth, YAML config, deterministic outputs, CI-enforced. All future signal-testing arcs use this infrastructure.

The KH-24 live system: continues to run on the VPS, unchanged. Not a constraint on L; not a target for L; not affected by L outcomes.

---

## 11. What Does Not Carry Over

The L arc evaluates these fresh and may reach different conclusions:

- Timeframe choice (4H is one option among several; the atlas considers 1H, 4H, D1, and conditional W1)
- Long-only directionality (atlas characterizes both directions; signal candidates may be long, short, or both)
- Exhaustion bar pattern (one structural pattern among many)
- Pair set (28 pairs is the data available; the atlas may surface that some subset is more structurally interesting)
- WFO fold structure (KH-24 used 7 anchored expanding folds; L6+ signal testing may use a different fold structure if pair-split holdout discipline is adopted, depending on signal class)

---

## 12. First Step — L0

Before any data work, L0 produces a methodology lock document. The L0 prompt and execution plan are written when the next chat begins. The L0 deliverable must include, at minimum:

1. The four-layer atlas structure (this plan, verbatim)
2. The descriptive-not-predictive rule with examples
3. The pre-registered ranking metric for the bridge to signal testing
4. The pre-registered N (number of top candidates that proceed)
5. The total time budget for the L arc
6. The success criteria from section 8
7. The risk register from section 9
8. Sign-off: once L0 is locked, no modifications without an explicit re-planning phase

L0 is documentation only. No code. No data analysis. The lock is what makes everything that follows safe.

---

## 13. Tool Usage for the L Arc

Per project-permanent conventions:

- This chat: planning, methodology decisions, phase result interpretation, gate evaluation
- Cursor: single-file documentation patches, YAML config edits, small fixes
- Claude Code: multi-file analysis scripts, atlas computation, layer-by-layer execution
- GPT-4 and Aider: permanently excluded

Recommended models for L-arc work:

- Claude Code: Opus 4.7, xhigh effort, 1M context where available. The atlas computations involve multi-pair, multi-timeframe data alignment with strict no-lookahead requirements; correctness is non-negotiable and silent bugs (the SIGNAL_DIRECTION class) are the primary risk.
- This chat: Opus 4.7 with extended thinking. Strategic decisions, gate evaluation, methodology discussions.

---

## 14. Out of Scope for the L Arc

To keep the arc clean, the following are explicitly outside its scope:

- KH-24 modifications of any kind
- Signal backtesting or WFO
- Position sizing or risk-parameter changes
- Live system changes
- Broker, VPS, or EA changes
- Cross-arc comparisons (L vs KH performance, etc.)
- Portfolio construction questions (run KH-24 alongside L outputs, etc.)

These are real questions, but they belong to future work, not the L arc.

---

## 15. Phase Result Document Template

Every L-arc phase ends with a result document at `results/lchar/PHASE_LN_RESULT.md` containing:

1. Phase goal (verbatim from L0 or this plan)
2. Pre-commit gate or deliverable specification (verbatim)
3. Method summary, including any data-handling conventions and no-lookahead checks
4. Results, organized by metric or finding
5. Sanity checks performed
6. Caveats and known limitations
7. Final line, exact format: `STATUS: COMPLETE | INCOMPLETE | BLOCKED`

The format is identical to KH-arc phase results so the project's documentation conventions stay consistent across arcs.

---

## 16. Closure Conditions

The L arc closes when one of:

- L5 produces a top-N registry with at least one candidate that passes the L0 ranking threshold. Subsequent work moves to L6+ signal testing arcs (separate from L).
- L5 produces a registry with no candidates above the threshold. The arc closes with a null result. Subsequent research direction is a separate planning phase.
- The pre-committed time budget is exceeded. The arc closes with whatever atlas is complete; partial findings are documented; subsequent direction is a separate planning phase.

In all cases, an `L_ARC_CLOSURE.md` document is produced summarizing what was found, what was not, and what direction the project takes next.

---

*End of L Arc Plan. This document is the source of truth for the L characterization arc until L0 is signed off, after which `docs/L0_METHODOLOGY_LOCK.md` becomes the operational reference.*
