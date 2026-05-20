# L0 — Methodology Lock

**Arc:** L characterization arc
**Status:** LOCKED (pending sign-off)
**Locked date:** 2026-05-10
**Predecessor plan:** `L_ARC_PLAN.md`
**Modification rule:** No modifications to this document after sign-off without an explicit re-planning phase. Re-planning requires a new phase document, justification for the change, and acknowledgement that the change resets the relevant downstream work.

---

## 1. Purpose of L0

L0 is the methodology lock for the L characterization arc. It commits, in advance of any data work, to the structure of the atlas, the rule that keeps it descriptive, the mechanical bridge from atlas to candidate signals, the time budget, and the success criteria. Once L0 is signed off, the arc's degrees of freedom are constrained to *what the data shows under this methodology* — no post-hoc metric changes, no top-N expansion, no threshold relaxation.

L0 contains no code and no data analysis. It is documentation only. The lock is what makes everything that follows safe.

---

## 2. The Four-Layer Atlas Structure (verbatim from L_ARC_PLAN.md §5)

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

Layer 4 outputs are ranked by the L0 pre-registered metric (see §4) with proper statistical treatment of multiple comparisons. This is where the atlas hands off to signal candidacy.

---

## 3. Descriptive-Not-Predictive Rule

Every output of the atlas is a measured statistic with a confidence interval, attached to the data, expressed as a property of the historical trading universe. The atlas does not say *"if X happens, Y follows."* It says *"X happened with frequency F; conditional on X, the next-N-bar return distribution had mean M, standard deviation S, with N observations, 95% CI [...]."* Whether the deviation from baseline is meaningful is a question for the bridge in §4 to answer mechanically; the atlas does not interpret on its own behalf.

This rule is what makes EDA on the full dataset safe. Descriptive statistics about the data are properties of the trading universe, not predictions. They cannot leak forward bias on their own. Forward bias only enters when descriptive findings are used to *select* which patterns to build signals from — that is what the pre-registered bridge guards against.

### Examples

**Compliant (descriptive):**

- "On 4H across 28 pairs (Oct 2020–present), the unconditional next-1-bar log-return distribution has mean +0.00012, std 0.0035, skew 0.04, kurtosis 5.2. Conditional on top-decile bar range, next-1-bar mean is −0.00018, std 0.0042, skew −0.12, with 12,400 observations."
- "On D1, the autocorrelation of log returns at lag 1 is −0.04, 95% CI [−0.06, −0.02]."
- "Across the 28-pair correlation matrix, the median pairwise correlation in 2024 is 0.31; in 2022 it was 0.42."

**Non-compliant (predictive / interpretive):**

- "After top-decile range bars, mean reversion is likely, suggesting a fade signal."
- "D1 shows mean-reverting behavior, so a contrarian strategy could work."
- "These conditions look like good entry candidates."

The atlas reports. It does not recommend.

---

## 4. Pre-Registered Ranking Metric

### Metric

**Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado (2014). Probability that the observed conditional Sharpe ratio is real after correcting for selection bias from the family of trials considered.

DSR is computed for each (condition × horizon × timeframe) trial as:

```
DSR = Φ(  (SR_observed − SR_expected) × sqrt(T − 1)
        / sqrt(1 − γ_3 × SR_observed + ((γ_4 − 1)/4) × SR_observed²)  )
```

Where:
- `SR_observed` = observed Sharpe ratio of the conditional next-N-bar return series for the trial
- `SR_expected` = expected maximum Sharpe under null across the family size (see below)
- `T` = number of conditional observations for the trial
- `γ_3`, `γ_4` = skewness and kurtosis of the conditional return series
- `Φ` = standard normal CDF

`SR_expected` uses the standard approximation `SR_expected ≈ sqrt(2 × ln(N_trials)) / sqrt(T)` with the Euler-Mascheroni correction for the expected maximum of N_trials standard normals.

### Horizons (locked)

For each timeframe, three horizons in natural-time units {1 bar, 1 day, 1 week}:

| Timeframe | Horizon 1 | Horizon 2 | Horizon 3 |
|-----------|-----------|-----------|-----------|
| 1H        | 1 bar     | 24 bars   | 120 bars  |
| 4H        | 1 bar     | 6 bars    | 30 bars   |
| D1        | 1 bar     | 5 bars    | 20 bars   |

Each (condition × horizon × timeframe) is a separate trial in the DSR correction.

### Family size (locked)

**Total L4 trial cap: 300.** This is the `N_trials` value used in DSR computation for the entire arc. Specific conditions emerge from L1–L3 findings and cannot be enumerated at L0; the cap bounds selection bias regardless of which conditions are ultimately tested.

If fewer than 300 trials are run, DSR is over-corrected (conservative direction — acceptable). If more than 300 trials are needed, the arc must reopen L0 in an explicit re-planning phase. No exceptions.

### Minimum sample floor (locked)

**Trials with fewer than 100 conditional observations are excluded from L5 ranking.** They are recorded in the L4 results document for the atlas but do not contribute to the top-N registry. This floor is independent of the DSR; it prevents pathological small-sample effects from being mechanically promoted by a metric that handles them only asymptotically.

### Tie-breaking

Trials are ranked by descending DSR. In the event of exact DSR tie (extremely unlikely with continuous data), the secondary rank is by raw Sharpe ratio. Tertiary by larger T. No fourth tie-breaker should be needed; if one is, document it and treat the case as a reopening.

---

## 5. Pre-Registered N

**N = 5.** The L5 top-N registry contains exactly the top 5 trials by DSR rank from L4, subject to the §4 minimum sample floor.

### Registry semantics

- The registry always contains 5 entries (assuming ≥5 trials clear the min sample floor; if fewer clear, the registry contains all that do).
- The **closure threshold** for "candidate-producing" outcome is **DSR > 0.95** for at least one of the top 5.
- Trials in the top 5 that clear DSR > 0.95 proceed to L6+ signal-testing arcs.
- Trials in the top 5 that do not clear DSR > 0.95 are documented in the registry but do not proceed.
- If zero of the top 5 clear DSR > 0.95, the arc closes as null result per L_ARC_PLAN.md §16.

This separation (top-5 always populated, threshold gates which proceed) keeps the registry descriptive while keeping the bridge to L6+ honest.

---

## 6. Time Budget (locked)

- **Active Claude Code work:** 4 weeks
- **Calendar hard cap:** 6 weeks
- **Per-phase soft estimates** (from L_ARC_PLAN.md §7): L0 = 1 session, L1 = 1–2 days, L2 = 2–3 days, L3 = 3–5 days, L4 = 3–5 days, L5 = 1 session

If the calendar cap is hit before L5 closes, the arc closes per L_ARC_PLAN.md §16 closure condition 3 (whatever atlas is complete is documented; subsequent direction is a separate planning phase). The cap is not extensible without re-opening L0.

---

## 7. Success Criteria

Per L_ARC_PLAN.md §8: success is the integrity of the characterization process, not the production of tradeable signals. The arc succeeds if:

1. The atlas is complete across all four layers without lookahead, contamination, or methodological breaches.
2. The bridge from atlas to candidate signals operates mechanically per this L0 pre-registration.
3. The top-N registry is produced from the atlas without human override.
4. Every phase produces a result document regardless of pass or fail.
5. The arc closes with a documented disposition (candidates produced, null result, or budget-exhausted) and a clear handoff to whatever follows.

Whether the atlas surfaces tradeable patterns is a property of the data. The arc cannot guarantee tradeable findings exist, and a null result is a legitimate outcome.

---

## 8. Risk Register

Per L_ARC_PLAN.md §9, with countermeasures:

| Risk | Countermeasure |
|------|----------------|
| Forward bias re-entering via post-hoc condition selection | Family size capped at 300 trials, locked at L0. Conditions tested in L4 are documented in the L4 result doc; expansion requires reopening L0. |
| Researcher degrees of freedom in metric/threshold | Metric, horizons, threshold, min N floor, and N all locked at L0. No tuning after data is seen. |
| Confusion with KH-arc reasoning | This document is the source of truth for L. KH-24 is acknowledged but out of scope. Any defaulting to 4H/long-only/exhaustion-bar framing is a methodology error and triggers re-orientation. |
| Premature signal building inside L | Forbidden. Signal testing happens in L6+ arcs after L5 closes. No backtests, no WFO, no entry/exit logic in L. |
| Silent bugs in atlas computation (the SIGNAL_DIRECTION class) | Claude Code on Opus 4.7 xhigh effort, sanity checks on every layer's computation, deterministic outputs, every phase result document includes a sanity-check section. |
| Time budget overrun without closure | 6-week calendar hard cap. If hit, arc closes with whatever is complete; null partial result is documented. |

---

## 9. KH-Arc Non-Negotiables That Carry Over

The following project-permanent rules apply unchanged to all L-arc work and to any L6+ signal testing that follows:

- No lookahead anywhere in any layer's computation
- `build_ex_ante_bounded_population` mandatory on any future signal test
- WFO worst-fold is the only judge of signal success in L6+
- One change per phase, pre-committed gate, accept the result
- Per-bar MT5 spread data, never hardcoded defaults (relevant only in L6+)
- Risk = 1% of reset floor balance (relevant only in L6+)
- Clean labels are evaluation-only, never used for population selection
- GPT-4 and Aider permanently excluded from implementation
- Every phase produces a result document regardless of pass or fail
- Config-driven via YAML, no hardcoding, deterministic outputs, CI-enforced

---

## 10. Out of Scope for the L Arc

Per L_ARC_PLAN.md §14:

- KH-24 modifications of any kind
- Signal backtesting or WFO running
- Position sizing or risk-parameter changes
- Live system changes
- Broker, VPS, or EA changes
- Cross-arc comparisons (L vs KH performance)
- Portfolio construction questions (run KH-24 alongside L outputs)

These are real questions, but they belong to future work, not the L arc.

---

## 11. Tool Usage

- This chat (Opus 4.7, extended thinking): planning, methodology decisions, phase result interpretation, gate evaluation
- Cursor: documentation patches, YAML config edits
- Claude Code (Opus 4.7, xhigh effort, 1M context where available): atlas computation scripts, layer-by-layer execution
- GPT-4 and Aider: permanently excluded

---

## 12. Sign-Off

Once this document is committed, the L arc methodology is locked. L1 begins immediately on sign-off.

**Signed off by:** _______________
**Date:** _______________

---

*End of L0 Methodology Lock. This document is the operational reference for the L characterization arc until L5 closes.*
